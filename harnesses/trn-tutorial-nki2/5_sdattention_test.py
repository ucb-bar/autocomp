import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import math
import argparse
from scipy.special import softmax
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

# NKI_EXAMPLE_31_BEGIN
@nki.jit
def ref(q, k, v, use_causal_mask=False):
  """Beta 2 SDPA kernel for tutorial SD-attention shapes.

  Computes softmax(q @ k.T * scale) @ v.
  q, k, v are shaped (seqlen, d_head). Output is (seqlen, d_head).
  """
  seqlen, d_head = q.shape
  assert q.shape == k.shape == v.shape
  assert d_head <= 128
  assert seqlen % 128 == 0

  # This migrated kernel currently targets the common non-causal path used by the tutorial.
  assert use_causal_mask == False  # use == not 'is' in NKI

  Q_TILE = 128
  K_TILE = 128

  scale = 1.0 / math.sqrt(d_head)

  out = nl.ndarray((seqlen, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

  n_q_tiles = seqlen // Q_TILE
  n_k_tiles = seqlen // K_TILE

  # Reusable SBUF tiles — SBUF does not auto-accumulate, safe to declare once and overwrite.
  q_blk = nl.ndarray((Q_TILE, d_head), dtype=q.dtype, buffer=nl.sbuf)
  k_blk = nl.ndarray((K_TILE, d_head), dtype=k.dtype, buffer=nl.sbuf)
  v_blk = nl.ndarray((K_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf)

  # Transposed q/k tiles in (d_head, tile) layout for nc_matmul stationary/moving operands.
  qT        = nl.ndarray((d_head, Q_TILE), dtype=nl.float32, buffer=nl.sbuf)
  qT_scaled = nl.ndarray((d_head, Q_TILE), dtype=nl.float32, buffer=nl.sbuf)
  kT        = nl.ndarray((d_head, K_TILE), dtype=nl.float32, buffer=nl.sbuf)

  # Full-seqlen logit / softmax buffers for one q-tile.
  logits  = nl.ndarray((Q_TILE, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  row_max = nl.ndarray((Q_TILE, 1),      dtype=nl.float32, buffer=nl.sbuf)
  norm    = nl.ndarray((Q_TILE, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  exp_buf = nl.ndarray((Q_TILE, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  sum_exp = nl.ndarray((Q_TILE, 1),      dtype=nl.float32, buffer=nl.sbuf)
  inv_sum = nl.ndarray((Q_TILE, 1),      dtype=nl.float32, buffer=nl.sbuf)
  prob    = nl.ndarray((Q_TILE, seqlen), dtype=nl.float32, buffer=nl.sbuf)

  # Intermediate SBUF tiles for the output matmul.
  prob_tile = nl.ndarray((Q_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
  prob_t    = nl.ndarray((K_TILE, Q_TILE), dtype=nl.float32, buffer=nl.sbuf)
  tmp_sbuf  = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
  attn_out  = nl.ndarray((Q_TILE, d_head), dtype=q.dtype,    buffer=nl.sbuf)

  for i_q in nl.affine_range(n_q_tiles):
    q_start = i_q * Q_TILE
    q_end   = q_start + Q_TILE

    # Load q block (Q_TILE x d_head) and transpose to (d_head x Q_TILE) for nc_matmul.
    # Tensor Engine nc_transpose: SBUF → PSUM (required). Declare PSUM inside i_q loop
    # so each q-tile gets a fresh PSUM slot (no cross-tile accumulation).
    nisa.dma_copy(dst=q_blk, src=q[q_start:q_end, 0:d_head])
    qT_psum = nl.ndarray((d_head, Q_TILE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_psum, data=q_blk)
    nisa.tensor_copy(dst=qT, src=qT_psum)
    nisa.tensor_scalar(dst=qT_scaled, data=qT, op0=nl.multiply, operand0=scale)

    # Build logits (Q_TILE x seqlen) in K_TILE chunks.
    # nc_matmul: result[j,k] = sum_d qT_scaled[d,j] * kT[d,k] = (Q_scaled @ K^T)[j,k]
    for i_k in nl.affine_range(n_k_tiles):
      k_start = i_k * K_TILE
      k_end   = k_start + K_TILE

      nisa.dma_copy(dst=k_blk, src=k[k_start:k_end, 0:d_head])
      # Declare PSUM inside the loop so each nc_transpose/nc_matmul starts fresh
      # (no cross-iteration accumulation in PSUM).
      kT_psum = nl.ndarray((d_head, K_TILE), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=kT_psum, data=k_blk)
      nisa.tensor_copy(dst=kT, src=kT_psum)

      logits_psum = nl.ndarray((Q_TILE, K_TILE), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=logits_psum, stationary=qT_scaled, moving=kT)
      nisa.tensor_copy(dst=logits[0:Q_TILE, k_start:k_end], src=logits_psum)

    # Softmax across seqlen.
    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=logits, axis=1, keepdims=True)
    nisa.tensor_scalar(dst=norm, data=logits, op0=nl.subtract, operand0=row_max)
    nisa.activation(dst=exp_buf, op=nl.exp, data=norm)
    nisa.tensor_reduce(dst=sum_exp, op=nl.add, data=exp_buf, axis=1, keepdims=True)
    nisa.reciprocal(dst=inv_sum, data=sum_exp)
    nisa.tensor_scalar(dst=prob, data=exp_buf, op0=nl.multiply, operand0=inv_sum)

    # Compute output = prob @ v, accumulated in SBUF (not PSUM) to avoid
    # cross-iteration auto-accumulation issues.
    # nc_matmul: result[j,k] = sum_key prob_t[key,j] * v[key,k] = (prob @ V)[j,k]
    attn_accum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=attn_accum, value=0.0)

    for i_k in nl.affine_range(n_k_tiles):
      k_start = i_k * K_TILE
      k_end   = k_start + K_TILE

      # Extract prob slice for this k-chunk and transpose: (Q_TILE,K_TILE) → (K_TILE,Q_TILE).
      nisa.tensor_copy(dst=prob_tile, src=prob[0:Q_TILE, k_start:k_end])
      # Declare all PSUM buffers inside the loop for fresh slots each iteration.
      prob_t_psum = nl.ndarray((K_TILE, Q_TILE), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=prob_t_psum, data=prob_tile)
      nisa.tensor_copy(dst=prob_t, src=prob_t_psum)

      nisa.dma_copy(dst=v_blk, src=v[k_start:k_end, 0:d_head])
      tmp_psum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=tmp_psum, stationary=prob_t, moving=v_blk)
      nisa.tensor_copy(dst=tmp_sbuf, src=tmp_psum)
      nisa.tensor_tensor(dst=attn_accum, data1=attn_accum, data2=tmp_sbuf, op=nl.add)

    nisa.tensor_copy(dst=attn_out, src=attn_accum)
    nisa.dma_copy(dst=out[q_start:q_end, 0:d_head], src=attn_out)

  return out



def benchmark_nki(nki_func):
  device = xm.xla_device()
  seqlen, d_head = 4096, 64
  q_np = np.random.rand(seqlen, d_head).astype(np.float32)
  k_np = np.random.rand(seqlen, d_head).astype(np.float32)
  v_np = np.random.rand(seqlen, d_head).astype(np.float32)
  q_tensor = torch.from_numpy(q_np).to(device=device)
  k_tensor = torch.from_numpy(k_np).to(device=device)
  v_tensor = torch.from_numpy(v_np).to(device=device)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(q_tensor, k_tensor, v_tensor)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))



def test_nki(ref_func, test_func):
    device = xm.xla_device()
    for _ in range(2):
        seqlen, d_head = 4096, 64
        q_np = np.random.rand(seqlen, d_head).astype(np.float32)
        k_np = np.random.rand(seqlen, d_head).astype(np.float32)
        v_np = np.random.rand(seqlen, d_head).astype(np.float32)
        q_tensor = torch.from_numpy(q_np).to(device=device)
        k_tensor = torch.from_numpy(k_np).to(device=device)
        v_tensor = torch.from_numpy(v_np).to(device=device)
        ref_out = ref_func(q_tensor, k_tensor, v_tensor)
        test_out = test_func(q_tensor, k_tensor, v_tensor)
        if not np.allclose(ref_out.detach().cpu().numpy(), test_out.detach().cpu().numpy(), atol=1e-4, rtol=1e-3):
            return False
    return True

if __name__ == "__main__":
    test_result = test_nki(ref, solution)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(solution)
  
