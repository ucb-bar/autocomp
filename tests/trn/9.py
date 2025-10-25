import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def flash_attention_core(q, k, v,
                          seqlen_q, q_tile_idx,
                          kernel_dtype, acc_type,
                          seq_tile_size=2048,
                          B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128
                          ):
  """
  The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
  The q, k, and v start in HBM and will be loaded into SBUF as tiles. The block size of K and V
  is defined in the seq_tile_size parameter. Returns o, l, m:
  o: (B_P_SIZE, B_D_SIZE) - output attention values
  l: (B_P_SIZE, 1) - log-sum-exp normalizer
  m: (B_P_SIZE, 1) - max values
  """
  LARGE_TILE_SZ = seq_tile_size
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

  i_q_p = nl.arange(B_P_SIZE)[:, None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_d_p = nl.arange(B_D_SIZE)[:, None]
  i_d_f = nl.arange(B_D_SIZE)[None, :]
  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  # Load q tile from HBM into SBUF
  q_local_tile = nl.load(q[q_tile_idx * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)

  # Create local storage for output, l, and m
  o = nl.ndarray((nl.par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype, buffer=nl.shared_hbm)
  l = nl.ndarray((nl.par_dim(B_P_SIZE), 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
  m = nl.ndarray((nl.par_dim(B_P_SIZE), 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

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
  nl.store(m[:], value=nl.copy(max_, dtype=kernel_dtype))
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
  pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum)
  for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
    # Load v tile from HBM into SBUF
    v_tile = nl.load(v[k_i, i_q_p, i_d_f], dtype=kernel_dtype)
    
    pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                       v_tile,
                                       transpose_x=True,
                                       mask=forward_mask) # (128, 128) (p(Br), d)

  nl.store(o[:], value=nl.copy(pv_psum[i_q_p, i_d_f], dtype=kernel_dtype))
  nl.store(l[:], value=nl.add(nl.log(ps), max_))
  
  return o, l, m

def test_nki(ref_func, test_func):
  """Test the kernel with different q_tile_idx values to cover all masking scenarios"""
  B_P_SIZE = 128
  B_D_SIZE = 128
  B_F_SIZE = 512
  seq_tile_size = 2048
  num_q_tiles = seq_tile_size // B_P_SIZE  # 16 tiles
  
  # Test multiple q_tile_idx values to cover different masking scenarios:
  # - q_tile_idx=0: Only diagonal_and_right_selection path
  # - q_tile_idx=1,2: Mix of left_diagonal and diagonal paths
  # - Later tiles: Test multiplication_required_selection optimization
  test_indices = [num_q_tiles // 2, num_q_tiles - 1]
  
  for q_tile_idx in test_indices:
    print(f"Testing q_tile_idx={q_tile_idx}...")
    
    # Create fresh random inputs for each test
    q = np.random.rand(seq_tile_size, B_D_SIZE).astype(np.float32)
    k = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
    v = np.random.rand(seq_tile_size // B_P_SIZE, B_P_SIZE, B_D_SIZE).astype(np.float32)
    
    # Run the kernel
    o_ref, l_ref, m_ref = ref_func(
      q, k, v,
      seqlen_q=B_P_SIZE,
      q_tile_idx=q_tile_idx,
      kernel_dtype=nl.bfloat16,
      acc_type=nl.float32,
      seq_tile_size=seq_tile_size,
      B_P_SIZE=B_P_SIZE,
      B_F_SIZE=B_F_SIZE,
      B_D_SIZE=B_D_SIZE
    )
    o_test, l_test, m_test = test_func(
      q, k, v,
      seqlen_q=B_P_SIZE,
      q_tile_idx=q_tile_idx,
      kernel_dtype=nl.bfloat16,
      acc_type=nl.float32,
      seq_tile_size=seq_tile_size,
      B_P_SIZE=B_P_SIZE,
      B_F_SIZE=B_F_SIZE,
      B_D_SIZE=B_D_SIZE
    )

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
  seq_tile_size = 2048
  num_q_tiles = seq_tile_size // B_P_SIZE

  q = nt.tensor[[seq_tile_size, B_D_SIZE], nl.bfloat16]
  k = nt.tensor[[B_D_SIZE, seq_tile_size], nl.bfloat16]
  v = nt.tensor[[seq_tile_size // B_P_SIZE, B_P_SIZE, B_D_SIZE], nl.bfloat16]
  
  test_indices = [0, num_q_tiles // 2, num_q_tiles - 1]
  p99_list = []
  for q_tile_idx in test_indices:
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q, k, v,
               seqlen_q=B_P_SIZE,
               q_tile_idx=q_tile_idx,
               kernel_dtype=nl.bfloat16,
               acc_type=nl.float32,
               seq_tile_size=seq_tile_size,
               B_P_SIZE=B_P_SIZE,
               B_F_SIZE=B_F_SIZE,
               B_D_SIZE=B_D_SIZE)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    p99_list.append(p99)
    print(f"{p99 / 1000.0} ms (P99) for q_tile_idx={q_tile_idx}")

  print("Latency: {:.3f} ms (P99)".format(np.mean(p99_list) / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(flash_attention_core, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    print("Running benchmark...")
    benchmark_nki(test)
