import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def flash_attention_core(q, k, v,
                          kernel_dtype, acc_type,
                          seq_len=2048,
                          d_head=128
                          ):
  """
  The flash attention core function to calcualte self attention for all q tiles with K and V.
  The q, k, and v start in HBM and will be loaded into SBUF as tiles. The block size of K and V
  is defined in the seq_len parameter. Returns o, l, m:
  o: (seq_len, d_head) - output attention values for all q tiles
  l: (seq_len, 1) - log-sum-exp normalizer for all q tiles
  m: (seq_len, 1) - max values for all q tiles
  """
  B_P_SIZE = 128
  B_F_SIZE = 512
  LARGE_TILE_SZ = seq_len
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  num_q_tiles = seq_len // B_P_SIZE

  i_q_p = nl.arange(B_P_SIZE)[:, None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_d_p = nl.arange(d_head)[:, None]
  i_d_f = nl.arange(d_head)[None, :]
  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  # Create local storage for output, l, and m for all q tiles
  o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
  l = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
  m = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

  # Loop over all q tiles
  for q_tile_idx in nl.affine_range(num_q_tiles):
    # Load q tile from HBM into SBUF
    q_local_tile = nl.load(q[q_tile_idx * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)

    # mask are used to only apply computation to the lower half of the matrix,
    # which reduce the arthimetic intensity by half
    forward_mask = q_tile_idx * B_P_SIZE >= 0

    qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
      # Load k tile from HBM into SBUF
      k_tile = nl.load(k[i_d_p, k_i * B_F_SIZE + i_q_f], dtype=kernel_dtype)
      
      qk_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_F_SIZE),
                          dtype=np.float32, buffer=nl.psum)  # (128, 512)
      multiplication_required_selection = k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE
      qk_psum[i_q_p, i_q_f] += nl.matmul(q_local_tile, k_tile, transpose_x=True,
                                         mask=multiplication_required_selection) # (p(128), 512)

      left_diagonal_selection = q_tile_idx * B_P_SIZE >= (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < (k_i + 1) * B_F_SIZE) & forward_mask

      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = k_i * B_F_SIZE + i_q_f
      pred = q_pos >= k_pos
      # For tiles on and to the right of the diagonal, need to do affine_select.
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
        pred=pred,
        on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
        mask=diagonal_and_right_selection)

      # For tiles on the left of the diagonal, direct copy, no select required.
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)

      # Calculate max of the current tile
      max_local[i_q_p, k_i] = nisa.tensor_reduce(np.max, qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f], axis=(1,),
                                          dtype=acc_type, negate=False, mask=forward_mask)

    max_ = nisa.tensor_reduce(np.max, max_local[i_q_p, i_f_k_tiles], axis=(1, ),
                      dtype=acc_type, negate=False, mask=forward_mask)
    nl.store(m[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(max_, dtype=kernel_dtype))
    m_current = max_

    p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
    p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
      # compute exp(qk-max)
      p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f] = \
        nisa.activation(np.exp,
                        qk_res_buf[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                        bias=-1 * m_current,
                        scale=1.0,
                        dtype=kernel_dtype,
                        mask=forward_mask)

      # Compute partial row-tile sum of exp(qk-max))
      p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f], axis=1, dtype=acc_type, mask=forward_mask)

    p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
      p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
      for i_p_t_local in nl.affine_range(512//128):
        p_local_t_tmp[i_q_p, i_p_t_local*128 + i_f_128] = nisa.nc_transpose(p_local[i_q_p, i_p_t*512+i_p_t_local * B_P_SIZE + i_f_128], mask=forward_mask)
      i_f_512 = nl.arange(512)[None, :]
      p_local_transposed[i_q_p, i_p_t * 512 + i_f_512 ] = nl.copy(p_local_t_tmp[i_q_p, i_f_512], dtype=kernel_dtype, mask=forward_mask)

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
    pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
      # Load v tile from HBM into SBUF
      v_tile = nl.load(v[k_i * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)
      
      pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                         v_tile,
                                         transpose_x=True,
                                         mask=forward_mask) # (128, 128) (p(Br), d)

    nl.store(o[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(pv_psum[i_q_p, i_d_f], dtype=kernel_dtype))
    nl.store(l[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.add(nl.log(ps), max_))
  
  return o, l, m

def test_nki(ref_func, test_func):
  """Test the kernel that processes all q tiles"""
  d_head = 128
  seq_len = 2048
  
  print(f"Testing flash attention with d_head={d_head} and seq_len={seq_len}...")
  
  # Create random inputs
  q = np.random.rand(seq_len, d_head).astype(nl.float32)
  k = np.random.rand(d_head, seq_len).astype(nl.float32)
  v = np.random.rand(seq_len, d_head).astype(nl.float32)
  
  # Run the kernel
  o_ref, l_ref, m_ref = ref_func(
    q, k, v,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )
  o_test, l_test, m_test = test_func(
    q, k, v,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )

  fail = False
  if not np.allclose(o_ref.astype(nl.float32), o_test.astype(nl.float32), atol=0.01, rtol=0.001):
    print(f"FAIL: o_ref != o_test")
    print("o_ref shape:", o_ref.shape)
    print("o_test shape:", o_test.shape)
    print("o_ref", o_ref.astype(nl.float32)[:5,:5])
    print("o_test", o_test.astype(nl.float32)[:5,:5])
    fail = True
  if not np.allclose(l_ref.astype(nl.float32), l_test.astype(nl.float32), atol=0.01, rtol=0.001):
    print(f"FAIL: l_ref != l_test")
    print("l_ref shape:", l_ref.shape)
    print("l_test shape:", l_test.shape)
    print("l_ref", l_ref.astype(nl.float32)[:5])
    print("l_test", l_test.astype(nl.float32)[:5])
    fail = True
  if not np.allclose(m_ref.astype(nl.float32), m_test.astype(nl.float32), atol=0.01, rtol=0.001):
    print(f"FAIL: m_ref != m_test")
    print("m_ref shape:", m_ref.shape)
    print("m_test shape:", m_test.shape)
    print("m_ref", m_ref.astype(nl.float32)[:5])
    print("m_test", m_test.astype(nl.float32)[:5])
    fail = True
  if fail:
    return False
  
  print(f"  ✓ All tests passed")
  
  return True

def benchmark_nki(nki_func):
  """Benchmark the flash attention kernel"""
  d_head = 128
  seq_len = 2048
  
  q = nt.tensor[[seq_len, d_head], nl.bfloat16]
  k = nt.tensor[[d_head, seq_len], nl.bfloat16]
  v = nt.tensor[[seq_len, d_head], nl.bfloat16]
  
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(q, k, v,
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
