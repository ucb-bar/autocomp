import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import math
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

@nki.jit
def ref(q, k, v):
  """Beta 2 migrated attention kernel (matches `4_attention_test.py` semantics).

  Inputs are shaped (d_head=128, seqlen). Output is shaped (seqlen, d_head).
  """
  d_head, seqlen = q.shape
  assert q.shape == k.shape == v.shape
  assert d_head == nl.tile_size.pmax  # 128
  assert seqlen % 128 == 0
  assert seqlen >= 512

  PMAX = nl.tile_size.pmax                # 128
  Q_TILE = PMAX                           # 128 queries per tile
  K_TILE_SOFTMAX = nl.tile_size.gemm_moving_fmax  # 512 keys per tile for QK
  K_TILE_OUT = PMAX                       # 128 keys per tile for output matmul

  out = nl.ndarray((seqlen, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

  n_q_tiles = seqlen // Q_TILE
  n_k_tiles_softmax = seqlen // K_TILE_SOFTMAX
  n_k_tiles_out = seqlen // K_TILE_OUT

  # Temporary tiles reused per q-tile.
  q_tile = nl.ndarray((PMAX, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
  k_tile = nl.ndarray((PMAX, K_TILE_SOFTMAX), dtype=k.dtype, buffer=nl.sbuf)

  qk_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  norm_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  exp_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  sum_exp = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  scores = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)

  scores_tile = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.sbuf)
  v_tile = nl.ndarray((PMAX, K_TILE_OUT), dtype=v.dtype, buffer=nl.sbuf)
  attn_out = nl.ndarray((Q_TILE, d_head), dtype=q.dtype, buffer=nl.sbuf)

  for i_q in nl.affine_range(n_q_tiles):
    q_start = i_q * Q_TILE
    q_end = q_start + Q_TILE

    # Load q tile once.
    nisa.dma_copy(dst=q_tile, src=q[0:PMAX, q_start:q_end])

    # Build qk_buf for this q tile in 512-key chunks.
    for i_k in nl.affine_range(n_k_tiles_softmax):
      k_start = i_k * K_TILE_SOFTMAX
      k_end = k_start + K_TILE_SOFTMAX
      nisa.dma_copy(dst=k_tile, src=k[0:PMAX, k_start:k_end])
      # Declare qk_psum inside the loop so each nc_matmul starts fresh (no cross-iteration accumulation).
      qk_psum = nl.ndarray((Q_TILE, K_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=qk_psum, stationary=q_tile, moving=k_tile)
      nisa.tensor_copy(dst=qk_buf[0:PMAX, k_start:k_end], src=qk_psum)

    # Softmax across seqlen dimension for each query row.
    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=qk_buf, axis=1, keepdims=True)
    nisa.tensor_scalar(dst=norm_buf, data=qk_buf, op0=nl.subtract, operand0=row_max)
    nisa.activation(dst=exp_buf, op=nl.exp, data=norm_buf)
    nisa.tensor_reduce(dst=sum_exp, op=nl.add, data=exp_buf, axis=1, keepdims=True)
    nisa.reciprocal(dst=inv_sum, data=sum_exp)
    nisa.tensor_scalar(dst=scores, data=exp_buf, op0=nl.multiply, operand0=inv_sum)

    # Attention output: accumulate in sbuf, not psum
    attn_accum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=attn_accum, value=0.0)

    scores_t = nl.ndarray((PMAX, Q_TILE),    dtype=nl.float32, buffer=nl.sbuf)
    v_t      = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.sbuf)
    tmp_sbuf = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)

    for i_k in nl.affine_range(n_k_tiles_out):
      k_start = i_k * K_TILE_OUT
      k_end   = k_start + K_TILE_OUT

      nisa.tensor_copy(dst=scores_tile, src=scores[0:PMAX, k_start:k_end])
      nisa.dma_copy(dst=v_tile, src=v[0:PMAX, k_start:k_end])

      # Tensor Engine nc_transpose: SBUF -> PSUM.  Declare psum INSIDE the loop
      # so the compiler allocates a fresh slot each iteration (overwrite, not accumulate).
      # Then tensor_copy brings the transposed tile to SBUF for nc_matmul.
      scores_t_psum = nl.ndarray((PMAX, Q_TILE),    dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=scores_t_psum, data=scores_tile)
      nisa.tensor_copy(dst=scores_t, src=scores_t_psum)

      v_t_psum = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=v_t_psum, data=v_tile)
      nisa.tensor_copy(dst=v_t, src=v_t_psum)

      # Declare tmp_psum inside the loop so each nc_matmul starts fresh.
      tmp_psum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=tmp_psum, stationary=scores_t, moving=v_t)
      nisa.tensor_copy(dst=tmp_sbuf, src=tmp_psum)
      nisa.tensor_tensor(dst=attn_accum, data1=attn_accum, data2=tmp_sbuf, op=nl.add)

    # Store result
    nisa.tensor_copy(dst=attn_out, src=attn_accum)
    nisa.dma_copy(dst=out[q_start:q_end, 0:d_head], src=attn_out)

  return out



def construct_args(d_head, seq_len, dtype):
    device = xm.xla_device()
    q_np = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    k_np = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    v_np = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    q = torch.from_numpy(q_np).to(device=device)
    k = torch.from_numpy(k_np).to(device=device)
    v = torch.from_numpy(v_np).to(device=device)
    return q, k, v

def test_nki(ref_func, test_func):
  for _ in range(2):
    args = construct_args(128, 4096, dtype=nl.bfloat16)
    result_1 = ref_func(*args)
    result_2 = test_func(*args)
    r1 = result_1.detach().cpu().numpy().astype(np.float32)
    r2 = result_2.detach().cpu().numpy().astype(np.float32)
    print("result_1", r1[:5, :5])
    print("result_2", r2[:5, :5])
    if not np.allclose(r1, r2, atol=1e-2):
      return False
  return True

def benchmark_nki(nki_func):
  # Benchmarking with large matrices to show the differences more clearly
  args = construct_args(128, 4096, dtype=nl.bfloat16)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(*args)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(ref, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    benchmark_nki(test)