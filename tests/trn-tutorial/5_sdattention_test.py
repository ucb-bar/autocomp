import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
from scipy.special import softmax

# SUBSTITUTE HERE

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

# NKI_EXAMPLE_31_BEGIN
'''DEPRECATED
@nki.jit
def ref(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen,d_head), \
  f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
      v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
      q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]
      ],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                              stationary=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

  return out_ref
'''

def benchmark_nki(nki_func):
  seqlen, d_head = 4096, 64
  q_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
  k_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
  v_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(q_tensor, k_tensor, v_tensor)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))



def test_nki(ref_func, test_func):
    for _ in range(2):
        seqlen, d_head = 4096, 64
        q_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
        k_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
        v_tensor = np.random.rand(seqlen, d_head).astype(np.float32)
        
        ref_out = ref_func(q_tensor, k_tensor, v_tensor)
        test_out = test_func(q_tensor, k_tensor, v_tensor)
        if not np.allclose(ref_out, test_out, atol=1e-4, rtol=1e-3):
            return False
    return True

if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
  
