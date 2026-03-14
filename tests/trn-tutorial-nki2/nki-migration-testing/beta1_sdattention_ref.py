import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim


@nki.jit
def ref(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
  # Beta 1 SDPA reference kernel (from `5_sdattention_test.py`).
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  seqlen, d_head = q_ref.shape
  assert d_head <= 128
  assert tuple(k_ref.shape) == (seqlen, d_head)
  assert tuple(v_ref.shape) == (seqlen, d_head)
  out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q (tuned for d_head=64 in tutorial).
  softmax_scale = 0.125

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  # Step 1. transpose(tensor_v)
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
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
        moving=k_local[i_k_seq_tile, ip_k, if_k],
        stationary=q_local[i_q_seq_tile, ip_q, if_q])

      if use_causal_mask:
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      else:
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(
          qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

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

    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,), dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]
      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
        moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
        stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)
    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])
    nl.store(out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
             value=attn_res_div)

  return out_ref


if __name__ == "__main__":
  q = np.load("sdattn_q.npy")
  k = np.load("sdattn_k.npy")
  v = np.load("sdattn_v.npy")
  out = ref(q, k, v)
  np.save("out_beta1_sdattention.npy", out)

