import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

# Magic number to replace -inf similar to what Tensorizer uses
NEG_INF = -9984.0

@nki.jit
def transpose_p_local(p_local_transposed, p_local, Q_TILE_SIZE, LARGE_KV_TILE_SIZE):
    B_P_SIZE = nl.tile_size.pmax
    REDUCTION_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    B_F_SIZE = nl.tile_size.gemm_moving_fmax
    for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
        p_local_t_tmp = nl.ndarray(
            (REDUCTION_SIZE, B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
            buffer=nl.psum, dtype=nl.float32,
        )
        for j in nl.affine_range(B_F_SIZE // REDUCTION_SIZE):
            j_128_start = j * Q_TILE_SIZE
            ij_start = i * B_F_SIZE + j * REDUCTION_SIZE
            nisa.nc_transpose(
                dst=p_local_t_tmp[0:REDUCTION_SIZE, j_128_start:j_128_start+Q_TILE_SIZE],
                data=p_local[0:Q_TILE_SIZE, ij_start:ij_start+REDUCTION_SIZE]
            )
        out_start = i * (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE)
        out_size = B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE
        nisa.tensor_copy(
            dst=p_local_transposed[0:B_P_SIZE, out_start:out_start+out_size],
            src=p_local_t_tmp,
        )


@nki.jit
def _flash_attention_core(
    q, k, v, olm_prev,
    kernel_dtype, acc_type,
    tile_mask,
    q_tile_idx=None,
    Q_TILE_SIZE=128,
    LARGE_KV_TILE_SIZE=16384,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    qk_res_buf = nl.ndarray((Q_TILE_SIZE, LARGE_KV_TILE_SIZE), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.zeros((Q_TILE_SIZE, num_k_tile_per_large_tile), dtype=acc_type, name='max_local', buffer=nl.sbuf)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_start = k_i * B_F_SIZE
        multiplication_required = q_tile_idx * Q_TILE_SIZE >= k_i * B_F_SIZE

        if multiplication_required:
            q_local_tile = nl.ndarray((B_D_SIZE, Q_TILE_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_local_tile, src=q[0:B_D_SIZE, q_tile_idx*Q_TILE_SIZE:(q_tile_idx+1)*Q_TILE_SIZE])
            k_local_tile = nl.ndarray((B_D_SIZE, B_F_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_local_tile, src=k[0:B_D_SIZE, k_i_start:k_i_start+B_F_SIZE])

            qk_psum = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_local_tile, moving=k_local_tile)

            qk_sbuf = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

            tile_mask_tile = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile_mask_tile, src=tile_mask[0:Q_TILE_SIZE, k_i_start:k_i_start+B_F_SIZE])

            # masked = (qk - NEG_INF) * mask + NEG_INF
            qk_shifted = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
            masked_shifted = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=tile_mask_tile, op=nl.multiply)
            masked = nl.ndarray((Q_TILE_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=masked, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
            nisa.tensor_copy(dst=qk_res_buf[0:Q_TILE_SIZE, k_i_start:k_i_start+B_F_SIZE], src=masked)
            nisa.tensor_reduce(
                dst=max_local[0:Q_TILE_SIZE, k_i:k_i+1],
                op=nl.maximum, data=qk_res_buf[0:Q_TILE_SIZE, k_i_start:k_i_start+B_F_SIZE],
                axis=1, keepdims=True,
            )
        else:
            nisa.memset(dst=qk_res_buf[0:Q_TILE_SIZE, k_i_start:k_i_start+B_F_SIZE], value=NEG_INF)
            nisa.memset(dst=max_local[0:Q_TILE_SIZE, k_i:k_i+1], value=NEG_INF)

    max_ = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=max_, op=nl.maximum, data=max_local, axis=1, keepdims=True)

    olm_buffer = nl.ndarray((Q_TILE_SIZE, B_D_SIZE + 2), dtype=kernel_dtype, buffer=nl.sbuf)
    o_previous_scaled = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=acc_type, buffer=nl.sbuf)

    m_previous = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_previous, src=olm_prev[0:Q_TILE_SIZE, B_D_SIZE+1:B_D_SIZE+2])

    # m_current_neg = -max(max_, m_previous)
    m_max = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m_max, data1=max_, data2=m_previous, op=nl.maximum)
    m_current_neg = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m_current_neg, data=m_max, op0=nl.multiply, operand0=-1.0)

    REDUCTION_TILE = min(2048, LARGE_KV_TILE_SIZE // 2)
    p_local = nl.ndarray((Q_TILE_SIZE, LARGE_KV_TILE_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
    p_partial_sum = nl.ndarray((Q_TILE_SIZE, LARGE_KV_TILE_SIZE // REDUCTION_TILE), dtype=acc_type, buffer=nl.sbuf)

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        kr_start = k_r_i * REDUCTION_TILE
        nisa.activation(
            dst=p_local[0:Q_TILE_SIZE, kr_start:kr_start+REDUCTION_TILE],
            op=nl.exp,
            data=qk_res_buf[0:Q_TILE_SIZE, kr_start:kr_start+REDUCTION_TILE],
            bias=m_current_neg, scale=1.0,
        )
        nisa.tensor_reduce(
            dst=p_partial_sum[0:Q_TILE_SIZE, k_r_i:k_r_i+1],
            op=nl.add,
            data=p_local[0:Q_TILE_SIZE, kr_start:kr_start+REDUCTION_TILE],
            axis=1, keepdims=True,
        )

    ps = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ps, op=nl.add, data=p_partial_sum, axis=1, keepdims=True)

    p_local_transposed = nl.ndarray(
        (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
        dtype=kernel_dtype, buffer=nl.sbuf,
    )
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        Q_TILE_SIZE=Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
    )

    pv_psum = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=nl.float32, buffer=nl.sbuf, name='pv_psum')
    nisa.memset(dst=pv_psum, value=0.0)
    v_local = nl.ndarray((B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_local, src=v[0:B_P_SIZE, 0:LARGE_KV_TILE_SIZE//B_P_SIZE, 0:B_D_SIZE])

    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        p_col_start = k_i * Q_TILE_SIZE
        # v_local[:, k_i, :] is a (B_P_SIZE, B_D_SIZE) slice - use as moving
        v_slice = nl.ndarray((B_P_SIZE, B_D_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_slice, src=v_local[0:B_P_SIZE, k_i, 0:B_D_SIZE])
        pv_chunk = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=pv_chunk,
            stationary=p_local_transposed[0:B_P_SIZE, p_col_start:p_col_start+Q_TILE_SIZE],
            moving=v_slice,
        )
        pv_tmp = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pv_tmp, src=pv_chunk)
        nisa.tensor_tensor(dst=pv_psum, data1=pv_psum, data2=pv_tmp, op=nl.add)

    # alpha = exp(m_previous + m_current_neg)
    alpha = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
    nisa.activation(dst=alpha, op=nl.exp, data=m_previous, bias=m_current_neg, scale=1.0)

    # m_current = -m_current_neg
    m_current = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m_current, data=m_current_neg, op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(dst=olm_buffer[0:Q_TILE_SIZE, B_D_SIZE+1:B_D_SIZE+2], src=m_current)

    # o_previous_scaled = o_prev * alpha
    # alpha is (Q_TILE_SIZE, 1); broadcast to (Q_TILE_SIZE, B_D_SIZE) for tensor_tensor
    o_prev = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=o_prev, src=olm_prev[0:Q_TILE_SIZE, 0:B_D_SIZE])
    o_prev_f32 = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=acc_type, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_prev_f32, src=o_prev)
    alpha_broad = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=acc_type, buffer=nl.sbuf)
    for d in nl.affine_range(B_D_SIZE):
        nisa.tensor_copy(dst=alpha_broad[0:Q_TILE_SIZE, d:d+1], src=alpha)
    nisa.tensor_tensor(dst=o_previous_scaled, data1=o_prev_f32, data2=alpha_broad, op=nl.multiply)

    # olm_buffer[:, 0:B_D_SIZE] = o_previous_scaled + pv
    pv_final = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pv_final, src=pv_psum)
    o_new = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_new, data1=o_previous_scaled, data2=pv_final, op=nl.add)
    o_new_cast = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_new_cast, src=o_new)
    nisa.tensor_copy(dst=olm_buffer[0:Q_TILE_SIZE, 0:B_D_SIZE], src=o_new_cast)

    # l update: l_prev * alpha + ps
    l_prev = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=l_prev, src=olm_prev[0:Q_TILE_SIZE, B_D_SIZE:B_D_SIZE+1])
    l_prev_scaled = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=l_prev_scaled, data1=l_prev, data2=alpha, op=nl.multiply)
    ps_cast = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ps_cast, src=ps)
    l_new = nl.ndarray((Q_TILE_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=l_new, data1=l_prev_scaled, data2=ps_cast, op=nl.add)
    nisa.tensor_copy(dst=olm_buffer[0:Q_TILE_SIZE, B_D_SIZE:B_D_SIZE+1], src=l_new)

    olm = nl.ndarray((Q_TILE_SIZE, B_D_SIZE + 2), dtype=kernel_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=olm, src=olm_buffer)
    return olm

def test_nki(ref_func, test_func):
  """Test the kernel with different q_tile_idx values to cover all masking scenarios"""
  B_P_SIZE = 128
  B_D_SIZE = 128
  B_F_SIZE = 512
  Q_TILE_SIZE = 128
  seq_tile_size = 16384
  num_q_tiles = seq_tile_size // B_P_SIZE  # 16 tiles
  device = xm.xla_device()

  # Test multiple q_tile_idx values to cover different masking scenarios:
  # - q_tile_idx=0: Only diagonal_and_right_selection path
  # - q_tile_idx=1,2: Mix of left_diagonal and diagonal paths
  # - Later tiles: Test multiplication_required_selection optimization
  test_indices = [num_q_tiles // 2, num_q_tiles - 1]

  for q_tile_idx in test_indices:
    print(f"Testing q_tile_idx={q_tile_idx}...")

    # Create fresh random inputs for each test
    q_np = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
    k_np = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
    v_np = np.random.rand(B_P_SIZE, seq_tile_size // B_P_SIZE, B_D_SIZE).astype(np.float32)
    olm_prev_np = np.random.rand(Q_TILE_SIZE, B_D_SIZE + 2).astype(np.float32)

    # Generate causal tile mask for this query tile
    tile_mask_np = np.zeros((B_P_SIZE, seq_tile_size), dtype=bool)
    for i in range(B_P_SIZE):
      q_pos = q_tile_idx * B_P_SIZE + i
      tile_mask_np[i, :q_pos + 1] = True

    q = torch.from_numpy(q_np).to(device=device)
    k = torch.from_numpy(k_np).to(device=device)
    v = torch.from_numpy(v_np).to(device=device)
    olm_prev = torch.from_numpy(olm_prev_np).to(device=device)
    tile_mask = torch.from_numpy(tile_mask_np).to(device=device)

    # Run the kernel
    olm_ref = ref_func(
      q, k, v, olm_prev,
      kernel_dtype=nl.bfloat16,
      acc_type=nl.float32,
      tile_mask=tile_mask,
      q_tile_idx=q_tile_idx,
      Q_TILE_SIZE=B_P_SIZE,
      LARGE_KV_TILE_SIZE=seq_tile_size,
      B_P_SIZE=B_P_SIZE,
      B_F_SIZE=B_F_SIZE,
      B_D_SIZE=B_D_SIZE
    )
    olm_test = test_func(
      q, k, v, olm_prev,
      kernel_dtype=nl.bfloat16,
      acc_type=nl.float32,
      tile_mask=tile_mask,
      q_tile_idx=q_tile_idx,
      Q_TILE_SIZE=B_P_SIZE,
      LARGE_KV_TILE_SIZE=seq_tile_size,
      B_P_SIZE=B_P_SIZE,
      B_F_SIZE=B_F_SIZE,
      B_D_SIZE=B_D_SIZE
    )

    # Extract o, l, m from olm
    olm_ref_np = olm_ref.detach().cpu().to(torch.float32).numpy()
    olm_test_np = olm_test.detach().cpu().to(torch.float32).numpy()
    o_ref = olm_ref_np[:, :B_D_SIZE]
    l_ref = olm_ref_np[:, B_D_SIZE]
    m_ref = olm_ref_np[:, B_D_SIZE + 1]
    o_test = olm_test_np[:, :B_D_SIZE]
    l_test = olm_test_np[:, B_D_SIZE]
    m_test = olm_test_np[:, B_D_SIZE + 1]

    fail = False
    if not np.allclose(o_ref, o_test, atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: o_ref != o_test")
      print("o_ref", o_ref[:5,:5])
      print("o_test", o_test[:5,:5])
      fail = True
    if not np.allclose(l_ref, l_test, atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: l_ref != l_test")
      print("l_ref", l_ref[:5])
      print("l_test", l_test[:5])
      fail = True
    if not np.allclose(m_ref, m_test, atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: m_ref != m_test")
      print("m_ref", m_ref[:5])
      print("m_test", m_test[:5])
      fail = True
    if fail:
      return False

    print(f"  ✓ q_tile_idx={q_tile_idx} passed")

  return True

def benchmark_nki(nki_func):
  """Benchmark the flash attention kernel"""
  B_P_SIZE = 128
  B_D_SIZE = 128
  B_F_SIZE = 512
  Q_TILE_SIZE = 128
  seq_tile_size = 16384
  num_q_tiles = seq_tile_size // B_P_SIZE
  device = xm.xla_device()

  # Generate random data for benchmarking
  q_np = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
  k_np = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
  v_np = np.random.rand(B_P_SIZE, seq_tile_size // B_P_SIZE, B_D_SIZE).astype(np.float32)
  olm_prev_np = np.random.rand(Q_TILE_SIZE, B_D_SIZE + 2).astype(np.float32)
  q = torch.from_numpy(q_np).to(device=device)
  k = torch.from_numpy(k_np).to(device=device)
  v = torch.from_numpy(v_np).to(device=device)
  olm_prev = torch.from_numpy(olm_prev_np).to(device=device)
  test_indices = [0, num_q_tiles // 2, num_q_tiles - 1]
  p99_list = []
  for q_tile_idx in test_indices:
    # Generate causal tile mask for this query tile
    tile_mask_np = np.zeros((B_P_SIZE, seq_tile_size), dtype=bool)
    for i in range(B_P_SIZE):
      q_pos = q_tile_idx * B_P_SIZE + i
      tile_mask_np[i, :q_pos + 1] = True
    tile_mask = torch.from_numpy(tile_mask_np).to(device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q, k, v, olm_prev,
               kernel_dtype=nl.bfloat16,
               acc_type=nl.float32,
               tile_mask=tile_mask,
               q_tile_idx=q_tile_idx,
               Q_TILE_SIZE=B_P_SIZE,
               LARGE_KV_TILE_SIZE=seq_tile_size,
               B_P_SIZE=B_P_SIZE,
               B_F_SIZE=B_F_SIZE,
               B_D_SIZE=B_D_SIZE)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    p99_list.append(p99)
    print(f"{p99 / 1000.0} ms (P99) for q_tile_idx={q_tile_idx}")

  print("Latency: {:.3f} ms (P99)".format(np.mean(p99_list) / 1000.0))

if __name__ == "__main__":
  os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1" 
  test_result = test_nki(_flash_attention_core, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    print("Running benchmark...")
    benchmark_nki(test)
