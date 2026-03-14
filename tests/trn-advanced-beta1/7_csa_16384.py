import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

# Magic number to replace -inf similar to what Tensorizer uses
NEG_INF = -9984.0

@nki.jit
def transpose_p_local_reference_version(
    p_local_transposed,
    p_local,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
):
    assert p_local.shape == (Q_TILE_SIZE, LARGE_KV_TILE_SIZE)
    B_P_SIZE = nl.tile_size.pmax
    REDUCTION_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    B_F_SIZE = nl.tile_size.gemm_moving_fmax
    for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
        p_local_t_tmp = nl.ndarray(
            (
                nl.par_dim(REDUCTION_SIZE),
                B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE,
            ),
            buffer=nl.psum,
            dtype=np.float32,
        )
        for j in nl.affine_range(B_F_SIZE // REDUCTION_SIZE):

            j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
            i_j_128_slice = nl.ds(
                i * B_F_SIZE + j * REDUCTION_SIZE, REDUCTION_SIZE
            )
            p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                p_local[:, i_j_128_slice],
                engine=nisa.tensor_engine,
            )
        p_local_transposed[
            :,
            nl.ds(
                i * (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
            ),
        ] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)

@nki.jit
def _flash_attention_core(
    q,
    k,
    v,
    olm_prev,
    kernel_dtype,
    acc_type,
    tile_mask,
    q_tile_idx=None,
    Q_TILE_SIZE=128,
    LARGE_KV_TILE_SIZE=2048,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q: (B_D_SIZE, LARGE_KV_TILE_SIZE)
    k: (B_D_SIZE, LARGE_KV_TILE_SIZE)
    v: (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE)
    The results are returned in olm
    olm: (Q_TILE_SIZE, B_D_SIZE + 2)
    """
    assert (
        LARGE_KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    qk_res_buf = nl.ndarray(
        (nl.par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    max_local = nl.zeros(
        (nl.par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile),
        dtype=acc_type,
    )
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        # Apply causal masking: only compute when q_tile_idx * Q_TILE_SIZE >= k_i * B_F_SIZE
        multiplication_required_selection = (
            q_tile_idx * Q_TILE_SIZE >= k_i * B_F_SIZE
        )

        if multiplication_required_selection:
            qk_psum = nl.ndarray(
                (nl.par_dim(Q_TILE_SIZE), B_F_SIZE),
                dtype=np.float32,
                buffer=nl.psum,
            )  # (128, 512)
            q_local_tile = nl.load(q[:, q_tile_idx * Q_TILE_SIZE:(q_tile_idx + 1) * Q_TILE_SIZE], dtype=kernel_dtype)
            k_local_tile = nl.load(k[:, k_i_b_f_slice], dtype=kernel_dtype)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k_local_tile, transpose_x=True
            )  # (p(128), 512)
            tile_mask_local_tile = nl.load(tile_mask[:, k_i_b_f_slice])
            qk_res_buf[:, k_i_b_f_slice] = nl.where(
                tile_mask_local_tile,
                qk_psum[:, nl.ds(0, B_F_SIZE)],
                NEG_INF,
                dtype=acc_type,
            )
            # Calculate max of the current tile
            max_local[:, k_i] = nisa.tensor_reduce(
                np.max,
                qk_res_buf[:, k_i_b_f_slice],
                axis=(1,),
                dtype=acc_type,
                negate=False,
            )
        else:
            qk_res_buf[:, k_i_b_f_slice] = NEG_INF
            max_local[:, k_i] = NEG_INF

    # Calculate max of the current tile
    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    olm_buffer = nl.ndarray((Q_TILE_SIZE, B_D_SIZE + 2), dtype=kernel_dtype, buffer=nl.sbuf)
    o_previous_scaled = nl.ndarray(
        (nl.par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=kernel_dtype,
    )

    m_previous = nl.load(olm_prev[:, B_D_SIZE + 1], dtype=kernel_dtype)
    m_current_neg = nisa.tensor_scalar(
        max_,
        nl.maximum,
        m_previous,
        op1=nl.multiply,
        operand1=-1,
    )

    p_local = nl.ndarray(
        (nl.par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    REDUCTION_TILE = min(2048, LARGE_KV_TILE_SIZE // 2)

    p_partial_sum = nl.ndarray(
        (nl.par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE // REDUCTION_TILE),
        dtype=acc_type,
    )

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=m_current_neg,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray(
        (nl.par_dim(B_P_SIZE), LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transpose_p_local_reference_version(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        Q_TILE_SIZE=Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
    )

    pv_psum = nl.zeros(
        (nl.par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    v_local = nl.load(v[:, :, :], dtype=kernel_dtype)
    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
            v_local[:, k_i, :],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    # Compute scaling factor
    alpha = nisa.activation(
        np.exp,
        m_previous,
        bias=m_current_neg,
        scale=1.0,
    )

    olm_buffer[:, B_D_SIZE + 1] = nisa.activation(
        nl.copy,
        m_current_neg,
        scale=-1.0,
    )
    o_previous_scaled[...] = nl.multiply(
        nl.load(olm_prev[:, nl.ds(0, B_D_SIZE)], dtype=kernel_dtype),
        alpha,
    )
    olm_buffer[:, nl.ds(0, B_D_SIZE)] = nl.add(o_previous_scaled, pv_psum)

    l_prev = nl.load(olm_prev[:, B_D_SIZE], dtype=kernel_dtype) * alpha
    olm_buffer[:, B_D_SIZE] = l_prev + ps
    olm = nl.ndarray((Q_TILE_SIZE, B_D_SIZE + 2), dtype=kernel_dtype, buffer=nl.shared_hbm)
    nl.store(olm, olm_buffer)
    return olm

def test_nki(ref_func, test_func):
  """Test the kernel with different q_tile_idx values to cover all masking scenarios"""
  B_P_SIZE = 128
  B_D_SIZE = 128
  B_F_SIZE = 512
  Q_TILE_SIZE = 128
  seq_tile_size = 16384
  num_q_tiles = seq_tile_size // B_P_SIZE  # 16 tiles
  
  # Test multiple q_tile_idx values to cover different masking scenarios:
  # - q_tile_idx=0: Only diagonal_and_right_selection path
  # - q_tile_idx=1,2: Mix of left_diagonal and diagonal paths
  # - Later tiles: Test multiplication_required_selection optimization
  test_indices = [num_q_tiles // 2, num_q_tiles - 1]
  
  for q_tile_idx in test_indices:
    print(f"Testing q_tile_idx={q_tile_idx}...")
    
    # Create fresh random inputs for each test
    q = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
    k = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
    v = np.random.rand(B_P_SIZE, seq_tile_size // B_P_SIZE, B_D_SIZE).astype(np.float32)
    olm_prev = np.random.rand(Q_TILE_SIZE, B_D_SIZE + 2).astype(np.float32)

    # Generate causal tile mask for this query tile
    # tile_mask[i, j] = True if query at position (q_tile_idx * B_P_SIZE + i) can attend to key at position j
    tile_mask = np.zeros((B_P_SIZE, seq_tile_size), dtype=bool)
    for i in range(B_P_SIZE):
      q_pos = q_tile_idx * B_P_SIZE + i
      tile_mask[i, :q_pos + 1] = True
    
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
    o_ref = olm_ref[:, :B_D_SIZE]
    l_ref = olm_ref[:, B_D_SIZE]
    m_ref = olm_ref[:, B_D_SIZE + 1]
    o_test = olm_test[:, :B_D_SIZE]
    l_test = olm_test[:, B_D_SIZE]
    m_test = olm_test[:, B_D_SIZE + 1]

    fail = False
    if not np.allclose(o_ref.astype(nl.float32), o_test.astype(nl.float32), atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: o_ref != o_test")
      print("o_ref", o_ref.astype(nl.float32)[:5,:5])
      print("o_test", o_test.astype(nl.float32)[:5,:5])
      fail = True
    if not np.allclose(l_ref.astype(nl.float32), l_test.astype(nl.float32), atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: l_ref != l_test")
      print("l_ref", l_ref.astype(nl.float32)[:5])
      print("l_test", l_test.astype(nl.float32)[:5])
      fail = True
    if not np.allclose(m_ref.astype(nl.float32), m_test.astype(nl.float32), atol=0.01, rtol=0.001):
      print(f"FAIL at q_tile_idx={q_tile_idx}: m_ref != m_test")
      print("m_ref", m_ref.astype(nl.float32)[:5])
      print("m_test", m_test.astype(nl.float32)[:5])
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

  # Generate random data for benchmarking
  q = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
  k = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
  v = np.random.rand(B_P_SIZE, seq_tile_size // B_P_SIZE, B_D_SIZE).astype(np.float32)
  olm_prev = np.random.rand(Q_TILE_SIZE, B_D_SIZE + 2).astype(np.float32)
  test_indices = [0, num_q_tiles // 2, num_q_tiles - 1]
  p99_list = []
  for q_tile_idx in test_indices:
    # Generate causal tile mask for this query tile
    tile_mask = np.zeros((B_P_SIZE, seq_tile_size), dtype=bool)
    for i in range(B_P_SIZE):
      q_pos = q_tile_idx * B_P_SIZE + i
      tile_mask[i, :q_pos + 1] = True
    
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
  test_result = test_nki(_flash_attention_core, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    print("Running benchmark...")
    benchmark_nki(test)
