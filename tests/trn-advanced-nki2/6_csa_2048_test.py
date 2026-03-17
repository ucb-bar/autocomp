import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

NEG_INF = -9984.0

@nki.jit
def flash_attention_core(q, k, v, causal_mask,
                          kernel_dtype, acc_type,
                          seq_len=2048,
                          d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    REDUCTION_TILE = 1024  # match beta1's two-level reduction structure
    num_k_tiles = seq_len // B_F_SIZE   # 4
    num_q_tiles = seq_len // B_P_SIZE   # 16

    o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
    l = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    m = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_start = q_tile_idx * B_P_SIZE
        q_end = q_start + B_P_SIZE

        # Load and transpose Q
        q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_end, 0:d_head])
        q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
        q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

        # qk_res_buf stores float32 masked QK for all k_tiles (matching beta1's affine_select behavior).
        # Always apply causal mask unconditionally (no two-variable affine_range conditionals).
        qk_res_buf = nl.ndarray((B_P_SIZE, seq_len), buffer=nl.sbuf, dtype=acc_type)

        k_tile = nl.ndarray((d_head, B_F_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        causal_tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)

        for k_i in nl.affine_range(num_k_tiles):
            k_start = k_i * B_F_SIZE
            k_end = k_start + B_F_SIZE
            nisa.dma_copy(dst=k_tile, src=k[0:d_head, k_start:k_end])
            qk_psum = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_tile_T, moving=k_tile)
            qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)
            nisa.dma_copy(dst=causal_tile, src=causal_mask[q_start:q_end, k_start:k_end])
            qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
            masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
            masked_qk = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=masked_qk, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
            # Store float32 masked QK directly (no bfloat16 quantization — matches beta1's affine_select)
            nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=masked_qk)

        # Global max over full qk_res_buf (static offset — no dynamic-offset bug)
        max_ = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)
        nisa.dma_copy(dst=m[q_start:q_end, 0:1], src=max_)

        neg_max = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max, data=max_, op0=nl.multiply, operand0=-1.0)

        # Two-pass exp+sum with STATIC offsets matching beta1's REDUCTION_TILE=1024 structure.
        # Static slices avoid the dynamic-offset activation read bug.
        p_local_0 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.activation(dst=p_local_0, op=nl.exp,
                        data=qk_res_buf[0:B_P_SIZE, 0:1024], bias=neg_max, scale=1.0)
        ps_0 = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ps_0, op=nl.add, data=p_local_0, axis=1, keepdims=True)

        p_local_1 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.activation(dst=p_local_1, op=nl.exp,
                        data=qk_res_buf[0:B_P_SIZE, 1024:2048], bias=neg_max, scale=1.0)
        ps_1 = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ps_1, op=nl.add, data=p_local_1, axis=1, keepdims=True)

        ps = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ps, data1=ps_0, data2=ps_1, op=nl.add)

        # PV matmul using two separate loops (one per p_local half) to avoid
        # dynamic-offset nc_transpose issues and two-variable affine_range conditionals.
        pv_accum = nl.ndarray((B_P_SIZE, d_head), dtype=acc_type, buffer=nl.sbuf)
        nisa.memset(dst=pv_accum, value=0.0)

        p_slice_local = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        p_slice_T = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        v_tile_pv = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)

        # First 1024 cols (p_local_0): v rows 0..1023
        for k_i in nl.affine_range(REDUCTION_TILE // B_P_SIZE):
            k_start = k_i * B_P_SIZE
            nisa.tensor_copy(dst=p_slice_local,
                             src=p_local_0[0:B_P_SIZE, k_start:k_start + B_P_SIZE])
            p_slice_T_psum = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=p_slice_T_psum, data=p_slice_local)
            nisa.tensor_copy(dst=p_slice_T, src=p_slice_T_psum)
            nisa.dma_copy(dst=v_tile_pv, src=v[k_start:k_start + B_P_SIZE, 0:d_head])
            pv_psum = nl.ndarray((B_P_SIZE, d_head), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv_psum, stationary=p_slice_T, moving=v_tile_pv)
            tmp = nl.ndarray((B_P_SIZE, d_head), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp, src=pv_psum)
            nisa.tensor_tensor(dst=pv_accum, data1=pv_accum, data2=tmp, op=nl.add)

        # Second 1024 cols (p_local_1): v rows 1024..2047
        for k_i in nl.affine_range(REDUCTION_TILE // B_P_SIZE):
            k_start = k_i * B_P_SIZE
            v_start = REDUCTION_TILE + k_start
            nisa.tensor_copy(dst=p_slice_local,
                             src=p_local_1[0:B_P_SIZE, k_start:k_start + B_P_SIZE])
            p_slice_T_psum = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=p_slice_T_psum, data=p_slice_local)
            nisa.tensor_copy(dst=p_slice_T, src=p_slice_T_psum)
            nisa.dma_copy(dst=v_tile_pv, src=v[v_start:v_start + B_P_SIZE, 0:d_head])
            pv_psum = nl.ndarray((B_P_SIZE, d_head), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv_psum, stationary=p_slice_T, moving=v_tile_pv)
            tmp = nl.ndarray((B_P_SIZE, d_head), dtype=acc_type, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp, src=pv_psum)
            nisa.tensor_tensor(dst=pv_accum, data1=pv_accum, data2=tmp, op=nl.add)

        # Store o
        o_cast = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_cast, src=pv_accum)
        nisa.dma_copy(dst=o[q_start:q_end, 0:d_head], src=o_cast)

        # Store l = log(ps) + max (float32 — no bfloat16 quantization so exp(l-m)=ps exactly)
        log_ps = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.activation(dst=log_ps, op=nl.log, data=ps)
        l_val = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=l_val, data1=log_ps, data2=max_, op=nl.add)
        nisa.dma_copy(dst=l[q_start:q_end, 0:1], src=l_val)

    return o, l, m

def test_nki(ref_func, test_func):
  """Test the kernel that processes all q tiles"""
  d_head = 128
  seq_len = 2048
  device = xm.xla_device()

  print(f"Testing flash attention with d_head={d_head} and seq_len={seq_len}...")

  # Create random inputs
  q_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  k_np = np.random.rand(d_head, seq_len).astype(nl.float32)
  v_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  causal_mask_np = np.zeros((seq_len, seq_len), dtype=np.float32)
  for i in range(seq_len):
    causal_mask_np[i, :i+1] = 1.0

  q = torch.from_numpy(q_np).to(device=device)
  k = torch.from_numpy(k_np).to(device=device)
  v = torch.from_numpy(v_np).to(device=device)
  causal_mask = torch.from_numpy(causal_mask_np).to(device=device)

  # Run the kernel
  o_ref, l_ref, m_ref = ref_func(
    q, k, v, causal_mask,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )
  o_test, l_test, m_test = test_func(
    q, k, v, causal_mask,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )

  fail = False
  o_ref_np = o_ref.detach().cpu().to(torch.float32).numpy()
  o_test_np = o_test.detach().cpu().to(torch.float32).numpy()
  l_ref_np = l_ref.detach().cpu().to(torch.float32).numpy()
  l_test_np = l_test.detach().cpu().to(torch.float32).numpy()
  m_ref_np = m_ref.detach().cpu().to(torch.float32).numpy()
  m_test_np = m_test.detach().cpu().to(torch.float32).numpy()
  if not np.allclose(o_ref_np, o_test_np, atol=0.01, rtol=0.001):
    print(f"FAIL: o_ref != o_test")
    print("o_ref shape:", o_ref_np.shape)
    print("o_test shape:", o_test_np.shape)
    print("o_ref", o_ref_np[:5,:5])
    print("o_test", o_test_np[:5,:5])
    fail = True
  if not np.allclose(l_ref_np, l_test_np, atol=0.01, rtol=0.001):
    print(f"FAIL: l_ref != l_test")
    print("l_ref shape:", l_ref_np.shape)
    print("l_test shape:", l_test_np.shape)
    print("l_ref", l_ref_np[:5])
    print("l_test", l_test_np[:5])
    fail = True
  if not np.allclose(m_ref_np, m_test_np, atol=0.01, rtol=0.001):
    print(f"FAIL: m_ref != m_test")
    print("m_ref shape:", m_ref_np.shape)
    print("m_test shape:", m_test_np.shape)
    print("m_ref", m_ref_np[:5])
    print("m_test", m_test_np[:5])
    fail = True
  if fail:
    return False

  print(f"  ✓ All tests passed")

  return True

def benchmark_nki(nki_func):
  """Benchmark the flash attention kernel"""
  d_head = 128
  seq_len = 2048
  device = xm.xla_device()

  q_np = np.random.rand(seq_len, d_head).astype(np.float32)
  k_np = np.random.rand(d_head, seq_len).astype(np.float32)
  v_np = np.random.rand(seq_len, d_head).astype(np.float32)
  causal_mask_np = np.zeros((seq_len, seq_len), dtype=np.float32)
  for i in range(seq_len):
    causal_mask_np[i, :i+1] = 1.0
  q = torch.from_numpy(q_np).to(device=device)
  k = torch.from_numpy(k_np).to(device=device)
  v = torch.from_numpy(v_np).to(device=device)
  causal_mask = torch.from_numpy(causal_mask_np).to(device=device)

  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(q, k, v, causal_mask,
             kernel_dtype=nl.bfloat16,
             acc_type=nl.float32,
             seq_len=seq_len,
             d_head=d_head)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(flash_attention_core, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    print("Running benchmark...")
    benchmark_nki(test)
