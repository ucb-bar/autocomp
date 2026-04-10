CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv(ref, start, step):
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return [ref[start::step, :]], [ref[start + 1 :: step, :]]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    k_list, v_list = [], []
    b_start = start // packing
    b_step = step // packing
    b_ref = ref.bitcast(jnp.uint32)
    b = b_ref[b_start::b_step, :]

    # TODO(chengjiyao): use the general strided loading logic for bf16 after
    # fixing the issue in mosaic\'s infer vector layout pass
    if ref.dtype == jnp.bfloat16:
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      k_list.append(k)
      v_list.append(v)
    else:
      bitwidth = 32 // packing
      bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
      for i in range(0, packing, 2):
        bk = b >> (i * bitwidth)
        k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
        k_list.append(k)
        bv = b >> ((i + 1) * bitwidth)
        v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
        v_list.append(v)

    return k_list, v_list

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      row_ids = (
          (kv_len - q_len)
          + q_len_start
          - q_start
          + jax.lax.broadcasted_iota(
              jnp.int32,
              (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
              0,
          )
          // num_q_heads_per_kv_head
      )
      col_ids = kv_len_start + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
          1,
      )
      causal_mask = row_ids < col_ids
      if sliding_window is not None:
        causal_mask = jnp.logical_or(causal_mask,
                                     row_ids - sliding_window >= col_ids)
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
      out = lax.div(
          l_alpha * o_curr + beta * qkv,
          l_next_safe,
      )
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      kv_packing = get_dtype_packing(kv_ref.dtype)
      # NOTE: kv_packing is divided by 2 because k and v are packed together.
      kv_load_step = max(1, kv_packing // 2)
      for kv_head_chunk_idx in range(0, num_kv_heads_per_blk, kv_load_step):
        k_list, v_list = strided_load_kv(
            kv_ref, kv_head_chunk_idx * 2, num_combined_kv_heads_per_blk
        )
        for step_idx in range(kv_load_step):
          k = k_list[step_idx]
          v = v_list[step_idx]
          if k_scale is not None:
            # NOTE: Conversion between arbitrary data types is not supported.
            # That\'s why it is converted to float32 first.
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q_ref.dtype)
          if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q_ref.dtype)
          kv_head_idx = kv_head_chunk_idx + step_idx
          q_head_idx = kv_head_idx * num_q_heads_per_kv_head
          # TODO(jevinjiang): extra handling for packed type that can start at
          # unaligned position!
          q = fold_on_2nd_minor(
              q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
          )
          flash_attention(
              q,
              k,
              v,
              l_ref.at[kv_head_idx],
              m_ref.at[kv_head_idx],
              acc_ref.at[
                  :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
              ],
              kv_blk_idx=kv_blk_idx,
          )
      return kv_blk_idx + 1, next_buf_idx

    _, next_buf_idx = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx),  # (kv_blk_idx, buf_idx)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.644,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 0.644 ms\n{"correct": true, "latency": 0.644, "error": "", "all_times_ms": [0.635, 0.636, 0.636, 0.636, 0.636, 0.636, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.64, 0.64, 0.64, 0.64, 0.64, 0.641, 0.641, 0.641, 0.641, 0.641, 0.641, 0.641, 0.641, 0.641, 0.642, 0.642, 0.642, 0.643, 0.643, 0.644, 0.644, 0.644, 0.644, 0.644, 0.644, 0.644, 0.644, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.648, 0.648, 0.648, 0.648, 0.649, 0.649, 0.649, 0.649, 0.65, 0.65, 0.65, 0.651, 0.652, 0.652, 0.652, 0.653, 0.653, 0.655, 0.656, 0.658, 0.661, 0.682], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Looking at this ragged paged attention kernel, I\'ll analyze the code for optimization opportunities.

## Analysis

The kernel performs flash attention with paged KV cache. Key observations:

1. **Current grid**: `(num_heads_blks, num_q_blks)` with both dimensions marked as `"arbitrary"`
2. **Main computation loop**: Uses nested `while_loop` to iterate over sequences and KV blocks
3. **Memory access pattern**: Double-buffered KV page prefetching with semaphores
4. **Scratch buffers**: `l_ref`, `m_ref`, `acc_ref` for flash attention online softmax

## Selected Strategy: Strategy 10 - Fuse the l/m/acc update computations

**Rationale**: In the `flash_attention` function, the current implementation performs multiple separate operations:
1. Loads `m_prev` and `l_prev` from scratch refs
2. Computes `m_next`, `l_next`, `l_alpha`, `beta`
3. Broadcasts these values to match output shape
4. Performs the weighted accumulation `l_alpha * o_curr + beta * qkv`
5. Stores results back to three separate scratch refs

Each of these operations materializes intermediate arrays and causes register pressure. The `broadcast_to_shape` function with concatenation is particularly wasteful.

## Optimization Plan

**Fuse l/m/acc updates into a single pass with vectorized broadcast**:

1. **Eliminate `broadcast_to_shape` function**: Instead of using `jnp.concatenate` to broadcast scaling factors, compute the scaled values directly using einsum or reshape operations that the compiler can optimize better.

2. **Fuse scaling computations**: Currently:
   ```python
   l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
   beta = broadcast_to_shape(beta, qkv.shape)
   l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
   out = lax.div(l_alpha * o_curr + beta * qkv, l_next_safe)
   ```
   
   Replace with a single fused computation that:
   - Keeps `l_alpha`, `beta`, `l_next_safe` in their natural shapes `(num_q_per_blk * num_q_heads_per_kv_head, 1)`
   - Uses broadcasting semantics directly without explicit concatenation
   - Computes `out = (l_alpha * o_curr + beta * qkv) / l_next_safe` in one expression where broadcasting happens implicitly

3. **Combine load_with_init pattern**: The repeated pattern of:
   ```python
   m_prev = load_with_init(head_m_ref, -jnp.inf)
   l_prev = load_with_init(head_l_ref, 0.0)
   o_curr = load_with_init(head_acc_ref, 0.0)
   ```
   
   Can be fused into a single conditional check at the start, loading all three values together when `kv_blk_idx != 0`.

4. **Restructure masked_store calls**: Currently there are 3 separate `masked_store` calls for `m`, `l`, and `acc`. Group these stores together to reduce control flow overhead.

**Expected Performance Improvement**:
- Reduces intermediate array materialization from 6+ temporary arrays to 2-3
- Eliminates the expensive `jnp.concatenate` operations in `broadcast_to_shape`
- Reduces register spilling by keeping intermediate values in registers longer
- Better instruction-level parallelism by allowing compiler to schedule fused operations

**Implementation Changes**:
1. Remove `broadcast_to_shape` function entirely
2. Reshape scaling factors to `(num_q_per_blk, num_q_heads_per_kv_head, 1)` to naturally broadcast with `head_dim`
3. Compute the output update as: `out = (l_alpha[..., None] * o_curr + beta[..., None] * qkv_reshaped) / l_next_safe[..., None]`
4. This avoids creating full-sized copies of the scaling factors while achieving the same broadcast semantics''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv(ref, start, step):
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return [ref[start::step, :]], [ref[start + 1 :: step, :]]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    k_list, v_list = [], []
    b_start = start // packing
    b_step = step // packing
    b_ref = ref.bitcast(jnp.uint32)
    b = b_ref[b_start::b_step, :]

    # TODO(chengjiyao): use the general strided loading logic for bf16 after
    # fixing the issue in mosaic\'s infer vector layout pass
    if ref.dtype == jnp.bfloat16:
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      k_list.append(k)
      v_list.append(v)
    else:
      bitwidth = 32 // packing
      bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
      for i in range(0, packing, 2):
        bk = b >> (i * bitwidth)
        k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
        k_list.append(k)
        bv = b >> ((i + 1) * bitwidth)
        v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
        v_list.append(v)

    return k_list, v_list

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      row_ids = (
          (kv_len - q_len)
          + q_len_start
          - q_start
          + jax.lax.broadcasted_iota(
              jnp.int32,
              (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
              0,
          )
          // num_q_heads_per_kv_head
      )
      col_ids = kv_len_start + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
          1,
      )
      causal_mask = row_ids < col_ids
      if sliding_window is not None:
        causal_mask = jnp.logical_or(causal_mask,
                                     row_ids - sliding_window >= col_ids)
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      # Reshape scaling factors to broadcast naturally with head_dim
      # l_alpha, beta, l_next_safe have shape (num_q_per_blk * num_q_heads_per_kv_head, 1)
      # o_curr, qkv have shape (num_q_per_blk * num_q_heads_per_kv_head, head_dim)
      # Broadcasting happens implicitly without concatenation
      out = lax.div(
          l_alpha * o_curr + beta * qkv,
          l_next_safe,
      )
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      kv_packing = get_dtype_packing(kv_ref.dtype)
      # NOTE: kv_packing is divided by 2 because k and v are packed together.
      kv_load_step = max(1, kv_packing // 2)
      for kv_head_chunk_idx in range(0, num_kv_heads_per_blk, kv_load_step):
        k_list, v_list = strided_load_kv(
            kv_ref, kv_head_chunk_idx * 2, num_combined_kv_heads_per_blk
        )
        for step_idx in range(kv_load_step):
          k = k_list[step_idx]
          v = v_list[step_idx]
          if k_scale is not None:
            # NOTE: Conversion between arbitrary data types is not supported.
            # That\'s why it is converted to float32 first.
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q_ref.dtype)
          if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q_ref.dtype)
          kv_head_idx = kv_head_chunk_idx + step_idx
          q_head_idx = kv_head_idx * num_q_heads_per_kv_head
          # TODO(jevinjiang): extra handling for packed type that can start at
          # unaligned position!
          q = fold_on_2nd_minor(
              q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
          )
          flash_attention(
              q,
              k,
              v,
              l_ref.at[kv_head_idx],
              m_ref.at[kv_head_idx],
              acc_ref.at[
                  :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
              ],
              kv_blk_idx=kv_blk_idx,
          )
      return kv_blk_idx + 1, next_buf_idx

    _, next_buf_idx = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx),  # (kv_blk_idx, buf_idx)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.64,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.640 ms\n{"correct": true, "latency": 0.64, "error": "", "all_times_ms": [0.633, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.635, 0.635, 0.635, 0.635, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.641, 0.641, 0.641, 0.641, 0.641, 0.642, 0.642, 0.642, 0.642, 0.643, 0.644, 0.644, 0.644, 0.644, 0.644, 0.644, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.646, 0.646, 0.646, 0.646, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.648, 0.648, 0.648, 0.649, 0.649, 0.649, 0.649, 0.65, 0.65, 0.651, 0.679], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Based on the analysis of the provided code and the TPU v6e architecture, I will apply **Strategy 11: Switch to `jax.lax.dot_general` with explicit dimension numbers for the QK and QKV matrix multiplications to enable fusion of the implicit Right-Hand-Side (RSC) transpose**.

### Rationale
The original code uses `jnp.einsum` for the matrix multiplications in the flash attention kernel:
1. `qk = jnp.einsum("nd,md->nm", q, k, ...)`
2. `qkv = jnp.dot(s_curr, v, ...)`

On the TPU VPU/MXU, the `dot_general` operation allows fusing a transpose of the Right-Hand Side (RHS) matrix for "free". The standard matrix multiplication $D = A \cdot B$ is computed as a reduction over the contracting dimension (typically the last axis of A and the second-to-last of B). For the QK calculation, the contracting dimension is the last axis of both Q and K. However, for the QKV calculation ($S \cdot V$), the contracting dimension is the last axis of S and the last axis of V (since we are summing over the "KV length" or "columns of S").

Standard `jnp.dot` or `einsum` handling might introduce separate transpose operations or suboptimal dimension shuffling, which consumes cycles on the XLU unit. By explicitly specifying `dimension_numbers` in `jax.lax.dot_general`, we can ensure the hardware utilizes its fused transpose capability, minimizing data movement instructions.

In the QKV step (`s_curr @ v`), the reduction is over the `num_kv_per_blk` dimension, which is the last dimension of `s_curr` (shape `(num_q, num_kv)`) and the first dimension of `v` (shape `(num_kv, head_dim)`). This is the standard contracting behavior (RHS last dim is non-contracting). We will rewrite this to use `jax.lax.dot_general`.

### Modification Plan

We will modify the `flash_attention` inner function within `ragged_paged_attention_kernel`.

1.  **Replace QK einsum**:
    Change:
    ```python
    qk = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32) * sm_scale
    ```
    To:
    ```python
    qk = jax.lax.dot_general(
        q, k, dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32
    ) * sm_scale
    ```
    *Reasoning*: `q` shape `(N, D)`, `k` shape `(M, D)`. We contract dimension 1 of `q` with dimension 1 of `k`. Batch dimensions are empty. This computes $Q K^T$.

2.  **Replace QKV dot**:
    Change:
    ```python
    qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
    ```
    To:
    ```python
    qkv = jax.lax.dot_general(
        s_curr, v, dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32
    )
    ```
    *Reasoning*: `s_curr` shape `(N, M)`, `v` shape `(M, D)`. We contract dimension 1 of `s_curr` with dimension 0 of `v`. This ensures the V matrix is read with dimension 0 as the contracting dimension, utilizing the native dot hardware efficiently.

This change localizes the optimization to the arithmetic intensity bottleneck of the kernel without altering the block scheduling or memory allocation strategy.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv(ref, start, step):
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return [ref[start::step, :]], [ref[start + 1 :: step, :]]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    k_list, v_list = [], []
    b_start = start // packing
    b_step = step // packing
    b_ref = ref.bitcast(jnp.uint32)
    b = b_ref[b_start::b_step, :]

    # TODO(chengjiyao): use the general strided loading logic for bf16 after
    # fixing the issue in mosaic\'s infer vector layout pass
    if ref.dtype == jnp.bfloat16:
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      k_list.append(k)
      v_list.append(v)
    else:
      bitwidth = 32 // packing
      bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
      for i in range(0, packing, 2):
        bk = b >> (i * bitwidth)
        k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
        k_list.append(k)
        bv = b >> ((i + 1) * bitwidth)
        v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
        v_list.append(v)

    return k_list, v_list

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jax.lax.dot_general(
              q,
              k,
              dimension_numbers=(((1,), (1,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      row_ids = (
          (kv_len - q_len)
          + q_len_start
          - q_start
          + jax.lax.broadcasted_iota(
              jnp.int32,
              (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
              0,
          )
          // num_q_heads_per_kv_head
      )
      col_ids = kv_len_start + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
          1,
      )
      causal_mask = row_ids < col_ids
      if sliding_window is not None:
        causal_mask = jnp.logical_or(causal_mask,
                                     row_ids - sliding_window >= col_ids)
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jax.lax.dot_general(
          s_curr,
          v,
          dimension_numbers=(((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      # Reshape scaling factors to broadcast naturally with head_dim
      # l_alpha, beta, l_next_safe have shape (num_q_per_blk * num_q_heads_per_kv_head, 1)
      # o_curr, qkv have shape (num_q_per_blk * num_q_heads_per_kv_head, head_dim)
      # Broadcasting happens implicitly without concatenation
      out = lax.div(
          l_alpha * o_curr + beta * qkv,
          l_next_safe,
      )
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      kv_packing = get_dtype_packing(kv_ref.dtype)
      # NOTE: kv_packing is divided by 2 because k and v are packed together.
      kv_load_step = max(1, kv_packing // 2)
      for kv_head_chunk_idx in range(0, num_kv_heads_per_blk, kv_load_step):
        k_list, v_list = strided_load_kv(
            kv_ref, kv_head_chunk_idx * 2, num_combined_kv_heads_per_blk
        )
        for step_idx in range(kv_load_step):
          k = k_list[step_idx]
          v = v_list[step_idx]
          if k_scale is not None:
            # NOTE: Conversion between arbitrary data types is not supported.
            # That\'s why it is converted to float32 first.
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q_ref.dtype)
          if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q_ref.dtype)
          kv_head_idx = kv_head_chunk_idx + step_idx
          q_head_idx = kv_head_idx * num_q_heads_per_kv_head
          # TODO(jevinjiang): extra handling for packed type that can start at
          # unaligned position!
          q = fold_on_2nd_minor(
              q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
          )
          flash_attention(
              q,
              k,
              v,
              l_ref.at[kv_head_idx],
              m_ref.at[kv_head_idx],
              acc_ref.at[
                  :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
              ],
              kv_blk_idx=kv_blk_idx,
          )
      return kv_blk_idx + 1, next_buf_idx

    _, next_buf_idx = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx),  # (kv_blk_idx, buf_idx)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.638,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='gpt-5.4',
stdout='Latency: 0.638 ms\n{"correct": true, "latency": 0.638, "error": "", "all_times_ms": [0.632, 0.633, 0.633, 0.633, 0.633, 0.633, 0.633, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.634, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.639, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.641, 0.641, 0.642, 0.642, 0.643, 0.643, 0.643, 0.643, 0.643, 0.644, 0.645, 0.645, 0.645, 0.646, 0.646, 0.646, 0.647, 0.647, 0.647, 0.648, 0.648, 0.649, 0.649, 0.649, 0.65, 0.651, 0.652, 0.653, 0.655, 0.656, 0.671], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 11. Precompute causal mask `row_ids` / `col_ids` base values outside the inner loop and use incremental updates within the loop.**

### Why this is the best single change here

In `ragged_paged_attention_kernel`, the hottest path is:

- per **Q block**
- per **KV block**
- per **KV head**

Inside that path, `flash_attention(...)` rebuilds these tensors every time:

- `row_ids`
- `col_ids`
- `causal_mask`

But for a fixed `(cur_seq_idx, q_blk_idx, kv_blk_idx)`, those values do **not** depend on `kv_head_idx`.  
So today the kernel repeats the same iota/broadcast/compare work `num_kv_heads_per_blk` times.

On a **TPU v6e-1**, these elementwise integer/bool tensor constructions are not free; they burn VPU cycles and VMEM bandwidth in a kernel that already moves a lot of data. This is especially wasteful because the repeated mask generation happens entirely on-chip and adds no useful math.

---

## Plan

### 1. Hoist query-row position construction out of `flash_attention`
In `compute_with_cur_q_blk`, once `q_start`, `q_end`, `q_len`, and `kv_len` are known, build a **1D int32 row-position vector once** for the whole current Q block:

- shape: `[num_q_per_blk * num_q_heads_per_kv_head]`
- value equivalent to the current first-axis formula:
  ```python
  row_base = (
      (kv_len - q_len)
      + q_len_start
      - q_start
      + iota_q // num_q_heads_per_kv_head
  )
  ```

This is sequence/Q-block specific, so it should be computed once per `compute_with_cur_q_blk` iteration, not once per KV head.

---

### 2. Hoist KV-column position construction out of the per-head call
In `compute_with_kv_blk_in_cur_seq`, before the `for kv_head_idx in range(num_kv_heads_per_blk):` loop, build a **1D int32 column-position vector once** for the current KV block:

- shape: `[num_kv_per_blk]`
- value equivalent to the current second-axis formula:
  ```python
  col_base = kv_len_start + iota_k
  ```

This is KV-block specific, so it should be computed once per KV block, then reused across all KV heads.

---

### 3. Update `flash_attention` to accept precomputed bases
Change the helper so it no longer creates `row_ids` and `col_ids` internally.

Instead, pass in:

- `row_base`
- `col_base`

and form the mask from those:

```python
causal_mask = row_base[:, None] < col_base[None, :]
if sliding_window is not None:
  causal_mask = jnp.logical_or(
      causal_mask,
      row_base[:, None] - sliding_window >= col_base[None, :]
  )
```

That preserves semantics exactly, but removes repeated iota/arithmetic generation from the inner-most hot loop.

---

### 4. Make the KV block offset incremental
Right now `kv_len_start` is recomputed as:

```python
kv_len_start = kv_blk_idx * num_kv_per_blk
```

Instead, thread an `int32 kv_col_start` through the KV-block `lax.while_loop` state and update it with:

```python
next_kv_col_start = kv_col_start + num_kv_per_blk
```

Then build:

```python
col_base = kv_col_start + iota_k
```

This matches the selected strategy exactly: precompute bases and update them incrementally rather than rebuilding from scratch each iteration.

---

## Where the edits go

### In `compute_with_cur_q_blk(...)`
Add precompute of `row_base` once after:
```python
q_start = cu_q_lens_ref[cur_seq_idx]
q_end = cu_q_lens_ref[cur_seq_idx + 1]
q_len = q_end - q_start
kv_len = kv_lens_ref[cur_seq_idx]
```

### In `compute_with_kv_blk_in_cur_seq(...)`
Add precompute of `col_base` once before the KV-head loop, and pass it into `flash_attention`.

### In `flash_attention(...)`
Remove internal construction of:
- `row_ids`
- `col_ids`

and replace with mask construction from the passed-in bases.

---

## Why this should help this kernel specifically

For the provided workload:

- `num_q_heads = 32`
- `num_kv_heads = 8`
- so `num_q_heads_per_kv_head = 4`
- and `num_kv_heads_per_blk` is likely 8 for this model layout

That means the kernel currently regenerates the same causal position tensors **8 times per KV block**. This is pure overhead.

The change should:

- reduce integer/bool elementwise work in the hottest loop
- reduce repeated broadcast/iota generation
- reduce pressure on VREG/VMEM traffic for mask creation
- keep the same numerical behavior, because only the way mask indices are formed changes

---

## Semantics / correctness

This is semantically safe because:

- `row_ids` depend only on sequence/Q-block metadata, not on KV head
- `col_ids` depend only on KV-block position, not on KV head
- the attention math, softmax accumulation, masking value, and final cast all remain unchanged

So the rewritten kernel should remain equivalent to the original within the same small numerical tolerance.

---

## Important implementation notes for TPU v6e-1

- Keep `row_base` / `col_base` as **int32**.
- Keep attention score, softmax, and accumulation in **float32** exactly as today.
- Do **not** change the public function name or signature.
- Do **not** add large new VMEM scratch buffers; this optimization should be done by reusing small temporary arrays and loop-carried scalars, which is safer under the v6e VMEM budget.

If you want, I can next turn this plan into the actual rewritten code.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
      out = lax.div(
          l_alpha * o_curr + beta * qkv,
          l_next_safe,
      )
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(q_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = fold_on_2nd_minor(
            q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
        )
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.634,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.634 ms\n{"correct": true, "latency": 0.634, "error": "", "all_times_ms": [0.624, 0.624, 0.625, 0.625, 0.625, 0.625, 0.625, 0.626, 0.626, 0.626, 0.626, 0.626, 0.627, 0.627, 0.627, 0.627, 0.628, 0.629, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.633, 0.633, 0.633, 0.633, 0.633, 0.633, 0.634, 0.634, 0.634, 0.634, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.637, 0.637, 0.637, 0.637, 0.637, 0.637, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.638, 0.639, 0.639, 0.64, 0.64, 0.641, 0.641, 0.641, 0.641, 0.642, 0.642, 0.643, 0.645, 0.646, 0.666], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr='F0403 06:21:56.195845 1438894 lower_to_llo.cc:1002] Check failed: lhs_cst >> 32 == 0 (-1 vs. 0) \n*** Check failure stack trace: ***\n    @     0x7d4ca462ff84  (unknown)\n    @     0x7d4ca462ff38  (unknown)\n    @     0x7d4c9d370612  (unknown)\n    @     0x7d4c9d39d4fb  (unknown)\n    @     0x7d4c9d39ce62  (unknown)\n    @     0x7d4c9d39ca69  (unknown)\n    @     0x7d4c9d39cb28  (unknown)\n    @     0x7d4c9d39c9ef  (unknown)\n    @     0x7d4ca2a258c7  (unknown)\n    @     0x7d4ca2a64baa  (unknown)\n    @     0x7d4ca2a61e19  (unknown)\n    @     0x7d4ca2a261d1  (unknown)\n    @     0x7d4ca2a259a9  (unknown)\n    @     0x7d4ca2a262f0  (unknown)\n    @     0x7d4ca2a2b51a  (unknown)\n    @     0x7d4c9d36c0b3  (unknown)\n    @     0x7d4ca2c06acd  (unknown)\n    @     0x7d4ca2c07189  (unknown)\n    @     0x7d4ca2c084d4  (unknown)\n    @     0x7d4ca2c06d01  (unknown)\n    @     0x7d4ca2c09321  (unknown)\n    @     0x7d4c9d3e2c99  (unknown)\n    @     0x7d4c9d3e2925  (unknown)\n    @     0x7d4c9bea9fc5  (unknown)\n    @     0x7d4c9bea4732  (unknown)\n    @     0x7d4c9bea3be8  (unknown)\n    @     0x7d4c9be94c13  (unknown)\n    @     0x7d4c9be743e9  (unknown)\n    @     0x7d4c9be94cd9  (unknown)\n    @     0x7d4c9be993bd  (unknown)\n    @     0x7d4c9be9cf5c  (unknown)\n    @     0x7d4ca4210445  (unknown)\n    @     0x7d4ca4215776  (unknown)\n    @     0x7d4ca421e632  (unknown)\n    @     0x7d4ca448b863  (unknown)\n    @     0x7d4d2a894ac3  (unknown)\nhttps://symbolize.stripped_domain/r/?trace=7d4ca462ff84,7d4ca462ff37,7d4c9d370611,7d4c9d39d4fa,7d4c9d39ce61,7d4c9d39ca68,7d4c9d39cb27,7d4c9d39c9ee,7d4ca2a258c6,7d4ca2a64ba9,7d4ca2a61e18,7d4ca2a261d0,7d4ca2a259a8,7d4ca2a262ef,7d4ca2a2b519,7d4c9d36c0b2,7d4ca2c06acc,7d4ca2c07188,7d4ca2c084d3,7d4ca2c06d00,7d4ca2c09320,7d4c9d3e2c98,7d4c9d3e2924,7d4c9bea9fc4,7d4c9bea4731,7d4c9bea3be7,7d4c9be94c12,7d4c9be743e8,7d4c9be94cd8,7d4c9be993bc,7d4c9be9cf5b,7d4ca4210444,7d4ca4215775,7d4ca421e631,7d4ca448b862,7d4d2a894ac2&map= \nhttps://symbolize.stripped_domain/r/?trace=7d4d2a8969fc,7d4d2a84251f&map= \n*** SIGABRT received by PID 1438218 (TID 1438894) on cpu 27 from PID 1438218; ***\nE0403 06:21:56.205713 1438894 coredump_hook.cc:301] RAW: Remote crash data gathering hook invoked.\nE0403 06:21:56.205722 1438894 coredump_hook.cc:340] RAW: Skipping coredump since rlimit was 0 at process start.\nE0403 06:21:56.205725 1438894 client.cc:270] RAW: Coroner client retries enabled, will retry for up to 30 sec.\nE0403 06:21:56.205727 1438894 coredump_hook.cc:396] RAW: Sending fingerprint to remote end.\nE0403 06:21:56.205740 1438894 coredump_hook.cc:405] RAW: Cannot send fingerprint to Coroner: [NOT_FOUND] stat failed on crash reporting socket /var/google/services/logmanagerd/remote_coredump.socket (Is the listener running?): No such file or directory\nE0403 06:21:56.205743 1438894 coredump_hook.cc:457] RAW: Dumping core locally.\nF0403 06:21:56.195845 1438894 lower_to_llo.cc:1002] Check failed: lhs_cst >> 32 == 0 (-1 vs. 0) \nE0403 06:21:56.391008 1438894 process_state.cc:808] RAW: Raising signal 6 with default behavior\n'),
plan='''To optimize the performance of the ragged paged attention kernel on TPU v6e, we will implement **Strategy 10: Cache the softmax denominator (l_next_safe) reciprocal to convert division to multiplication in the output accumulation.**

In the current implementation of the `flash_attention` sub-kernel, the output normalization step calculates the updated attention output by dividing the weighted sum of the previous and current values by the new softmax denominator (`l_next_safe`). On TPU v6e hardware, floating-point division is significantly more expensive than multiplication. Since the normalization is performed for every element along the `head_dim` axis (which is typically 128 or more), we can improve efficiency by computing the reciprocal of the denominator once and then performing a cheaper elementwise multiplication.

### Optimization Plan

1.  **Identify the normalization step**: Locate the `flash_attention` function inside the `ragged_paged_attention_kernel`. Specifically, find where `l_next_safe` is calculated and used to normalize the accumulation.
2.  **Compute the reciprocal**: Immediately after calculating `l_next_safe` (which is the updated sum of exponentials), compute `inv_l_next = 1.0 / l_next_safe`. This operation is performed once per row of the attention tile.
3.  **Broadcast and Multiply**:
    *   Modify the `broadcast_to_shape` logic to broadcast `inv_l_next` to the shape of the query-key-value product tile (`qkv`).
    *   Replace the `lax.div` operation, which divides the partial output by `l_next_safe`, with a multiplication of the partial output by the broadcasted `inv_l_next`.
4.  **Maintain consistency**: Ensure `l_next_safe` is still stored in `head_l_ref` to be used as `l_prev` in the next block iteration of the FlashAttention algorithm.

This change reduces the arithmetic intensity of the kernel\'s most frequently executed section by replacing elementwise divisions with elementwise multiplications, leading to lower latency.

```python
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

class MultiPageAsyncCopyDescriptor:
  def __init__(self, pages_hbm_ref, vmem_buf, sem, page_indices_ref, metadata):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf

def ragged_paged_attention_kernel(
    kv_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref,
    num_seqs_ref, q_ref, kv_pages_hbm_ref, o_ref,
    kv_bufs, sems, l_ref, m_ref, acc_ref,
    *, sm_scale: float, sliding_window: int | None = None,
    soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None, v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = kv_bufs.shape
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = pl.program_id(0), pl.program_id(1)
  num_heads_blks = pl.num_programs(0)
  init_seq_idx, init_buf_idx = seq_buf_idx_ref[0], seq_buf_idx_ref[1]
  q_len_start, q_len_end = q_blk_idx * num_q_per_blk, q_blk_idx * num_q_per_blk + num_q_per_blk

  def create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size))
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    return MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[:, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :],
        kv_bufs.at[buf_idx], sems.at[buf_idx], page_indices_ref, metadata)

  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1: return ref[start::step, :], ref[start+1::step, :]
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      k = pltpu.bitcast(b << 16, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(b & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth, bitcast_dst_dtype = 32 // packing, jnp.dtype(f"uint{32 // packing}")
    b = b_ref[(start // packing) :: (step // packing), :]
    k = pltpu.bitcast((b >> ((start % packing) * bitwidth)).astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast((b >> (((start % packing) + 1) * bitwidth)).astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  def fold_on_2nd_minor(vec):
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0: vec = vec.astype(jnp.float32)
    return vec.reshape(-1, vec.shape[-1])

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    create_kv_async_copy_descriptors(heads_blk_idx, init_seq_idx, 0, init_buf_idx).start()

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start, q_end = cu_q_lens_ref[cur_seq_idx], cu_q_lens_ref[cur_seq_idx + 1]
    q_len, kv_len = q_end - q_start, kv_lens_ref[cur_seq_idx]

    def flash_attention(q, k, v, head_l_ref, head_m_ref, head_acc_ref, *, kv_blk_idx, row_base, col_base):
      kv_len_start = kv_blk_idx * num_kv_per_blk
      kv_mask = lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      k, v = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype), jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)
      qk = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32) * sm_scale
      store_start, store_end = jnp.maximum(q_start - q_len_start, 0), jnp.minimum(q_end - q_len_start, num_q_per_blk)
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None: causal_mask = jnp.logical_or(causal_mask, row_base[:, None] - sliding_window >= col_base[None, :])
      if soft_cap is not None: qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr, l_curr = jnp.broadcast_to(m_curr, lm_store_shape), jnp.broadcast_to(s_curr.sum(axis=1, keepdims=True), lm_store_shape)
      m_prev = jnp.where(kv_blk_idx == 0, jnp.full_like(head_m_ref, -jnp.inf), head_m_ref[...])
      l_prev = jnp.where(kv_blk_idx == 0, jnp.zeros_like(head_l_ref), head_l_ref[...])
      m_next = jnp.maximum(m_prev, m_curr)
      iota_m = lax.broadcasted_iota(jnp.int32, head_m_ref.shape, 0) // num_q_heads_per_kv_head
      pl.store(head_m_ref, idx=tuple(slice(None) for _ in head_m_ref.shape), val=m_next, mask=jnp.logical_and(iota_m >= store_start, iota_m < store_end))
      alpha, beta = jnp.exp(m_prev - m_next), jnp.exp(m_curr - m_next)
      l_alpha, l_next = alpha * l_prev, alpha * l_prev + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # OPTIMIZATION: Cache reciprocal of softmax denominator
      inv_l_next = 1.0 / l_next_safe
      pl.store(head_l_ref, idx=tuple(slice(None) for _ in head_l_ref.shape), val=l_next_safe, mask=jnp.logical_and(iota_m >= store_start, iota_m < store_end))

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape: return arr
        return jnp.concatenate([arr for _ in range(shape[1] // arr.shape[1])], axis=1)

      o_curr = jnp.where(kv_blk_idx == 0, jnp.zeros_like(head_acc_ref), head_acc_ref[...]).reshape(-1, head_dim)
      l_alpha, beta = broadcast_to_shape(l_alpha, qkv.shape), broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      # OPTIMIZATION: Multiply by reciprocal instead of lax.div
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      iota_o = lax.broadcasted_iota(jnp.int32, head_acc_ref.shape, 0)
      pl.store(head_acc_ref, idx=tuple(slice(None) for _ in head_acc_ref.shape), val=out.reshape(head_acc_ref.shape), mask=jnp.logical_and(iota_o >= store_start, iota_o < store_end))

    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (kv_len - q_len) + q_len_start - q_start + iota_q // num_q_heads_per_kv_head

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv, is_seq_end = next_kv_blk_idx * num_kv_per_blk >= kv_len, q_end <= q_len_end
      next_seq = lax.select(is_last_kv, lax.select(is_seq_end, cur_seq_idx + 1, cur_seq_idx), cur_seq_idx)
      next_heads_blk = lax.select(next_seq == num_seqs, heads_blk_idx + 1, heads_blk_idx)
      next_buf, next_seq_bounded = lax.select(cur_buf_idx == 0, 1, 0), lax.select(next_seq == num_seqs, 0, next_seq)
      @pl.when(next_heads_blk < num_heads_blks)
      def prefetch_next(): create_kv_async_copy_descriptors(next_heads_blk, next_seq_bounded, lax.select(is_last_kv, 0, next_kv_blk_idx), next_buf).start()
      kv_ref = create_kv_async_copy_descriptors(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx).wait().reshape(-1, head_dim)
      col_base = kv_col_start + jax.lax.iota(jnp.int32, num_kv_per_blk)
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(kv_ref, kv_head_idx, num_combined_kv_heads_per_blk)
        if k_scale is not None: k = (k.astype(jnp.float32) * k_scale).astype(q_ref.dtype)
        if v_scale is not None: v = (v.astype(jnp.float32) * v_scale).astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        flash_attention(fold_on_2nd_minor(q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]), k, v, l_ref.at[kv_head_idx], m_ref.at[kv_head_idx], acc_ref.at[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :], kv_blk_idx=kv_blk_idx, row_base=row_base, col_base=col_base)
      return kv_blk_idx + 1, next_buf, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(lambda s: s[0] * num_kv_per_blk < kv_len, compute_with_kv_blk_in_cur_seq, (0, cur_buf_idx, jnp.int32(0)))
    return lax.select(q_end < q_len_end, done, 1), lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx), next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(lambda s: jnp.logical_and(s[0] == 0, jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs], s[1] < num_seqs)), compute_with_cur_q_blk, (0, init_seq_idx, init_buf_idx))
  seq_buf_idx_ref[0], seq_buf_idx_ref[1] = lax.select(seq_idx < num_seqs, seq_idx, 0), buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)

def cdiv(a, b): return (a + b - 1) // b
def get_dtype_packing(dtype): return 32 // dtypes.bit_width(dtype)
def get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype):
  kv_p = get_dtype_packing(kv_dtype)
  max_kv_t = 8 * kv_p
  min_kv = max_kv_t if num_combined_kv_heads % max_kv_t == 0 else num_combined_kv_heads
  return min_kv // 2 * (num_q_heads // (num_combined_kv_heads // 2)), min_kv

@functools.partial(jax.jit, static_argnames=["sm_scale", "mask_value", "num_kv_pages_per_block", "num_queries_per_block", "vmem_limit_bytes", "sliding_window", "soft_cap", "k_scale", "v_scale"])
def ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale: float = 1.0, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None):
  if mask_value is None: mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype)
  num_kv_pages_per_blk, num_q_per_blk = (num_kv_pages_per_block, num_queries_per_block) if num_kv_pages_per_block else get_tuned_block_sizes(q.dtype, kv_pages.dtype, num_q_heads_per_blk, num_combined_kv_heads_per_blk // 2, head_dim, page_size, num_q_tokens, page_indices.shape[1])
  num_q_heads_per_kv_head = num_q_heads // (num_combined_kv_heads // 2)
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  grid = (num_q_heads // num_q_heads_per_blk, cdiv(num_q_tokens, num_q_per_blk))
  q_spec = pl.BlockSpec((num_q_per_blk, num_q_heads_per_blk, head_dim), lambda h, q: (q, h, 0))
  lm_scratch = pltpu.VMEM((num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128), jnp.float32)
  scalar_prefetches = (kv_lens, page_indices, cu_q_lens, jnp.array((0, 0), jnp.int32), num_seqs)
  return pl.pallas_call(functools.partial(ragged_paged_attention_kernel, sm_scale=sm_scale, sliding_window=sliding_window, soft_cap=soft_cap, mask_value=mask_value, k_scale=k_scale, v_scale=v_scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=len(scalar_prefetches), in_specs=[q_spec, pl.BlockSpec(memory_space=pltpu.ANY)], out_specs=q_spec, grid=grid, scratch_shapes=[pltpu.VMEM((2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim), kv_pages.dtype), pltpu.SemaphoreType.DMA((2,)), lm_scratch, lm_scratch, pltpu.VMEM((num_q_per_blk, num_q_heads_per_blk, head_dim), jnp.float32)]),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes), out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype), name="ragged_paged_attention_kernel")(*scalar_prefetches, q, kv_pages)
```''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(q_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = fold_on_2nd_minor(
            q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
        )
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.63,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.630 ms\n{"correct": true, "latency": 0.63, "error": "", "all_times_ms": [0.624, 0.625, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.633, 0.633, 0.633, 0.633, 0.634, 0.634, 0.635, 0.635, 0.635, 0.636, 0.636, 0.636, 0.637, 0.637, 0.637, 0.637, 0.638, 0.638, 0.638, 0.638, 0.638, 0.639, 0.64, 0.64, 0.64, 0.641, 0.642, 0.648, 0.662], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 3. Hoist redundant operations out of loops**

The clearest redundancy in this kernel is that the **same folded Q submatrices are rebuilt inside the KV-block loop**, even though `q_ref` is invariant for the entire kernel invocation.

### What is redundant now
Inside `compute_with_kv_blk_in_cur_seq`, for every KV block and every KV head, the kernel does:
```python
q = fold_on_2nd_minor(
    q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
)
```
But `q_ref` depends only on `(heads_blk_idx, q_blk_idx)`, not on:
- `kv_blk_idx`
- current KV page buffer
- current sequence within the same Q block

So this fold/slice/reshape is repeated unnecessarily.

### Planned change
Add one extra VMEM scratch buffer to cache the folded Q once per kernel invocation, then reuse it for all KV blocks.

#### 1) Add a Q-cache scratch buffer in `ragged_paged_attention`
Allocate:
- shape:  
  `(num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim)`
- dtype:
  - `q.dtype` in the common aligned case
  - `jnp.float32` if `fold_on_2nd_minor` would upcast because `num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0`

For the current tuned workload on v6e-1:
- shape is `(8, 256, 128)`
- size is only **512 KiB** in bf16, or **1 MiB** in f32  
So this comfortably fits in VMEM alongside the existing ~8 MiB KV double buffer and the other scratch buffers.

#### 2) Extend `ragged_paged_attention_kernel` to take that scratch ref
Keep the public `ragged_paged_attention(...)` signature unchanged. Only the internal kernel scratch list changes.

#### 3) Materialize folded Q once, before entering the sequence/KV loops
At the top of `ragged_paged_attention_kernel`, add a small loop:
- for each `kv_head_idx`
- read the corresponding `q_ref[:, q_head_idx:q_head_idx + num_q_heads_per_kv_head, :]`
- run the existing `fold_on_2nd_minor(...)`
- write the result into `q_folded_ref.at[kv_head_idx][...]`

This must use explicit Ref reads/writes, e.g.:
- read from `q_ref[...]`
- write to `q_folded_ref.at[kv_head_idx][...] = folded_q`

#### 4) Replace the repeated fold inside the KV loop
In `compute_with_kv_blk_in_cur_seq`, replace:
```python
q = fold_on_2nd_minor(...)
```
with:
```python
q = q_folded_ref.at[kv_head_idx][...]
```

### Why this should help on v6e-1
This removes repeated:
- VMEM reads of the same Q slices
- reshape/fold work
- extra register traffic inside the hot KV loop

That is especially useful here because the kernel is already doing substantial KV DMA and softmax math; reducing per-KV-block Q-side overhead lowers pressure in the innermost path without changing scheduling, tiling, or semantics.

### Semantics
This is numerically equivalent to the current code:
- same folded Q values
- same KV loads
- same float32 attention accumulation path
- same final cast to `o_ref.dtype`

Only the location where Q folding happens changes.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  def fold_on_2nd_minor(vec):
    assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
    assert len(vec.shape) >= 2
    last_dim = vec.shape[-1]
    packing = get_dtype_packing(vec.dtype)
    if vec.shape[-2] % packing != 0:
      vec = vec.astype(jnp.float32)
    return vec.reshape(-1, last_dim)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  for kv_head_idx in range(num_kv_heads_per_blk):
    q_head_idx = kv_head_idx * num_q_heads_per_kv_head
    q_folded_ref.at[kv_head_idx][...] = fold_on_2nd_minor(
        q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
    )

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(q_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = q_folded_ref.at[kv_head_idx][...]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )
  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  q_folded_scratch = pltpu.VMEM(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_dtype,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
      q_folded_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.628,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='gpt-5.4',
stdout='Latency: 0.628 ms\n{"correct": true, "latency": 0.628, "error": "", "all_times_ms": [0.621, 0.623, 0.623, 0.623, 0.623, 0.623, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.63, 0.63, 0.63, 0.631, 0.631, 0.631, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.634, 0.634, 0.635, 0.635, 0.635, 0.635, 0.637, 0.638, 0.638, 0.638, 0.639, 0.639, 0.639, 0.639, 0.64, 0.64, 0.64, 0.64, 0.641, 0.644, 0.674], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Looking at this code, I need to analyze the strategies and identify the best one to apply.

Let me examine the code structure:

1. The kernel has a `q_folded_ref` scratch buffer and a loop that folds queries at the start of every kernel invocation:
```python
for kv_head_idx in range(num_kv_heads_per_blk):
    q_head_idx = kv_head_idx * num_q_heads_per_kv_head
    q_folded_ref.at[kv_head_idx][...] = fold_on_2nd_minor(
        q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
    )
```

This folding operation reshapes queries from `[num_q_per_blk, num_q_heads_per_blk, head_dim]` to `[num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]` for GQA (Grouped Query Attention).

**Strategy 7** is directly applicable here: "Emit or keep queries upstream in the folded GQA layout the kernel actually consumes, so `q_folded_ref` and the per-block folding loop disappear entirely."

This is a clear optimization opportunity because:
1. The folding loop runs every kernel invocation, wasting compute
2. The `q_folded_scratch` VMEM allocation can be eliminated, freeing memory
3. The `fold_on_2nd_minor` function and its reshaping logic can be removed from the kernel

## Plan: Apply Strategy 7 - Pre-fold queries to GQA layout

### Changes Required:

1. **Modify the input query layout before kernel invocation:**
   - Before calling `pallas_call`, reshape `q` from `[max_num_batched_tokens, num_q_heads, head_dim]` to the folded GQA layout: `[max_num_batched_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]`
   - Then further reshape to `[max_num_batched_tokens * num_q_heads_per_kv_head, num_kv_heads, head_dim]` or equivalently a layout that matches what the kernel consumes

2. **Update the BlockSpec for queries:**
   - Change `q_block_spec` to load directly in the folded layout `[num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]`
   - Update `q_index_map` accordingly to index into the pre-folded array

3. **Remove from kernel:**
   - Delete the `q_folded_scratch` from `scratch_shapes`
   - Delete the `q_folded_ref` parameter from the kernel
   - Remove the folding loop at the start of the kernel
   - Remove the `fold_on_2nd_minor` helper function
   - Change `q = q_folded_ref.at[kv_head_idx][...]` to directly read from the input `q_ref` which is now already in the folded layout

4. **Update output handling:**
   - The output `o_ref` and `acc_ref` remain in the original layout `[num_q_per_blk, num_q_heads_per_blk, head_dim]`
   - After computation, reshape the output back to original layout if needed, or adjust the output spec similarly

5. **Handle dtype casting:**
   - The `fold_on_2nd_minor` function casts to float32 when packing doesn\'t align. Move this dtype handling to the pre-processing step outside the kernel
   - Use `q_folded_dtype` determined by the same logic when creating the pre-folded input

This optimization eliminates:
- The per-invocation folding loop over `num_kv_heads_per_blk` iterations
- The `q_folded_scratch` VMEM allocation
- Memory bandwidth for reading q and writing to scratch each iteration

The pre-folding happens once outside the kernel rather than `num_heads_blks * num_q_blks` times inside.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )
      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      masked_store(
          head_acc_ref,
          out.reshape(head_acc_ref.shape),
          store_start,
          store_end,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(q_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = q_folded_ref.at[kv_head_idx][...]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_block_spec,
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.603,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.603 ms\n{"correct": true, "latency": 0.603, "error": "", "all_times_ms": [0.597, 0.597, 0.598, 0.598, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.604, 0.604, 0.604, 0.604, 0.605, 0.605, 0.605, 0.605, 0.606, 0.606, 0.606, 0.606, 0.606, 0.607, 0.607, 0.608, 0.608, 0.608, 0.608, 0.608, 0.608, 0.608, 0.609, 0.609, 0.609, 0.609, 0.61, 0.61, 0.61, 0.61, 0.611, 0.611, 0.611, 0.612, 0.612, 0.612, 0.612, 0.613, 0.613, 0.615, 0.616, 0.617, 0.635, 0.674], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Looking at this complex ragged paged attention kernel, I need to identify a key inefficiency and apply one strategy to improve performance.

## Analysis

After reviewing the code, I notice several inefficiencies, but I\'ll focus on **Strategy 12: Hoist the mask computation in `masked_store`**.

The `masked_store` function inside `flash_attention` is called multiple times per KV head iteration:
```python
def masked_store(ref, val, start, end, group=1):
    iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
    mask = jnp.logical_and(iota >= start, iota < end)
    pl.store(ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask)
```

This function is called 4 times per KV head iteration (for `head_m_ref`, `head_l_ref`, `head_l_ref` again via `l_next_safe`, and `head_acc_ref`). Each call recomputes `lax.broadcasted_iota` which is expensive. Since:
1. `store_start` and `store_end` are invariant within the KV head loop (they depend only on `q_start`, `q_end`, `q_len_start`, `num_q_per_blk`)
2. The shapes for `lm_scratch` and `acc_scratch` are known upfront
3. The masks can be precomputed once before entering the KV head loop

## Plan

**Strategy: Hoist the mask computation in `masked_store`**

1. **Precompute the iota arrays once** before the `for kv_head_idx in range(num_kv_heads_per_blk)` loop:
   - Create `lm_iota` for the L/M scratch shape `(num_q_per_blk * num_q_heads_per_kv_head, 128)`
   - Create `acc_iota` for the accumulator shape `(num_q_per_blk, num_q_heads_per_kv_head, head_dim)`

2. **Precompute the masks** using `store_start` and `store_end` which are already computed before the loop:
   - `lm_mask = (lm_iota // num_q_heads_per_kv_head >= store_start) & (lm_iota // num_q_heads_per_kv_head < store_end)`
   - `acc_mask = (acc_iota >= store_start) & (acc_iota < store_end)`

3. **Modify `masked_store`** to accept a precomputed mask instead of computing it each time, or replace the `masked_store` calls with direct `pl.store` calls using the precomputed masks.

4. **Move the mask computation outside `flash_attention`** into `compute_with_kv_blk_in_cur_seq`, right after computing `store_start` and `store_end`, and pass the precomputed masks into `flash_attention`.

5. **Update `flash_attention` signature** to accept `lm_mask` and `acc_mask` as parameters, eliminating the need for the `masked_store` helper function entirely.

This optimization eliminates:
- 4 × `num_kv_heads_per_blk` calls to `lax.broadcasted_iota` per KV block iteration
- The associated division and comparison operations for mask generation

Since `num_kv_heads_per_blk` can be 4-8 typically, and this runs in a tight inner loop over all KV blocks across all sequences, hoisting these computations should provide measurable speedup by reducing redundant computation in the critical path.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
        lm_mask,  # precomputed mask for L/M stores
        acc_mask,  # precomputed mask for accumulator stores
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      # Use precomputed lm_mask for masked store
      pl.store(head_m_ref, idx=(slice(None), slice(None)), val=m_next, mask=lm_mask)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      # Use precomputed lm_mask for masked store
      pl.store(head_l_ref, idx=(slice(None), slice(None)), val=l_next_safe, mask=lm_mask)

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      # Use precomputed acc_mask for masked store
      pl.store(
          head_acc_ref,
          idx=(slice(None), slice(None), slice(None)),
          val=out.reshape(head_acc_ref.shape),
          mask=acc_mask,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    # Precompute store_start and store_end once per sequence
    store_start = jnp.maximum(q_start - q_len_start, 0)
    store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

    # Precompute masks for L/M and accumulator stores (hoisted out of inner loops)
    lm_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk * num_q_heads_per_kv_head, 128),
        0,
    ) // num_q_heads_per_kv_head
    lm_mask = jnp.logical_and(lm_iota >= store_start, lm_iota < store_end)

    acc_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk, num_q_heads_per_kv_head, head_dim),
        0,
    )
    acc_mask = jnp.logical_and(acc_iota >= store_start, acc_iota < store_end)

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(q_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(q_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = q_folded_ref.at[kv_head_idx][...]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
            lm_mask=lm_mask,
            acc_mask=acc_mask,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_block_spec,
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.601,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.601 ms\n{"correct": true, "latency": 0.601, "error": "", "all_times_ms": [0.597, 0.597, 0.597, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.601, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.602, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.603, 0.604, 0.604, 0.604, 0.605, 0.605, 0.605, 0.605, 0.606, 0.606, 0.606, 0.608, 0.608, 0.609, 0.609, 0.609, 0.609, 0.609, 0.611, 0.612, 0.612, 0.613, 0.613, 0.614, 0.616, 0.642], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 4. Rewrite the algorithm to reduce total work**

On **v6e-1** I would use this phase to remove a **redundant query data path** from the kernel.

### Why this code is wasting work
Inside `ragged_paged_attention_kernel`, the kernel receives both:

- `q_ref` with shape `[num_q_per_blk, num_q_heads_per_blk, head_dim]`
- `q_folded_ref` with shape `[num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]`

But the actual attention math uses **only `q_folded_ref`**:
```python
q = q_folded_ref.at[kv_head_idx][...]
```

`q_ref` is not used for the dot-product path at all. It is only serving as a source of:
- shape info
- dtype info for casts / final store

That means every grid invocation is paying an unnecessary **HBM→VMEM transfer** and consuming extra VMEM for a tile that does no useful math.

For the current tuned config:
- `num_q_per_blk = 64`
- `num_q_heads_per_blk = 32`
- `head_dim = 128`
- bf16 tile size for `q_ref` = `64 * 32 * 128 * 2 bytes = 524,288 bytes`

So this is an extra **512 KiB per program** of pure overhead. With `num_q_blks = 32` and `num_heads_blks = 1`, that is about **16 MiB of unnecessary query traffic per call**, plus extra VMEM pressure.

---

## Concrete change to make

### 1) Remove `q_ref` from the kernel inputs
Change the kernel signature so it no longer takes the original query tile:

Current:
```python
def ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    seq_buf_idx_ref,
    num_seqs_ref,
    q_ref,
    q_folded_ref,
    kv_pages_hbm_ref,
    o_ref,
    kv_bufs,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    ...
):
```

New:
```python
def ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    seq_buf_idx_ref,
    num_seqs_ref,
    q_folded_ref,
    kv_pages_hbm_ref,
    o_ref,
    kv_bufs,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    ...
):
```

---

### 2) Derive shape/dtype from `o_ref` instead of `q_ref`
Replace:
```python
num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
```

with:
```python
num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
```

Replace casts like:
```python
k = k.astype(q_ref.dtype)
v = v.astype(q_ref.dtype)
o_ref[...] = acc_ref[...].astype(q_ref.dtype)
```

with:
```python
k = k.astype(o_ref.dtype)
v = v.astype(o_ref.dtype)
o_ref[...] = acc_ref[...].astype(o_ref.dtype)
```

This preserves semantics because `o_ref.dtype == q.dtype`.

---

### 3) Keep pre-folding outside the kernel, but pass only the folded queries
Do **not** change the public function signature of `ragged_paged_attention`. It should still accept:
```python
(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, ...)
```

Keep the existing host-side folding:
```python
q_reshaped = q.reshape(...)
q_transposed = q_reshaped.transpose(...)
q_folded = q_transposed.reshape(...)
```

But when launching the kernel, pass only `q_folded` and `kv_pages` as tensor inputs.

Current:
```python
in_specs = [
    q_block_spec,
    q_folded_block_spec,
    pl.BlockSpec(memory_space=pltpu.ANY),
]
...
return kernel(*scalar_prefetches, q, q_folded, kv_pages)
```

New:
```python
in_specs = [
    q_folded_block_spec,
    pl.BlockSpec(memory_space=pltpu.ANY),
]
...
return kernel(*scalar_prefetches, q_folded, kv_pages)
```

`out_specs` should remain the original `q_block_spec`, since output layout is still `[num_q_tokens, num_q_heads, head_dim]`.

---

## Why this is a good fit for v6e-1
- **v6e-1 has one TensorCore per chip**, so this phase should focus on per-core efficiency rather than Megacore parallelization.
- Ragged paged attention is typically **HBM-sensitive** because it streams KV pages. Removing a dead query tile reduces memory traffic and VMEM residency without changing the math.
- Lower VMEM pressure also reduces the risk of compiler spills and can make the existing async K/V buffering more effective.

---

## What stays unchanged
To keep this strictly within **one strategy**, I would **not** change:
- grid order
- `dimension_semantics`
- buffer count
- tuned block sizes
- software pipeline structure
- numerical algorithm

The only change is: **stop moving and staging the unused `q_ref` tile**.

---

## Expected result
This should give a measurable latency reduction by:
1. eliminating one unnecessary HBM read per program,
2. reducing VMEM footprint per invocation,
3. slightly reducing pipeline overhead around block setup.

It is semantically equivalent to the original kernel within the same tolerance, because the attention computation already depends on `q_folded_ref`, not on `q_ref`.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
        lm_mask,  # precomputed mask for L/M stores
        acc_mask,  # precomputed mask for accumulator stores
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      # Use precomputed lm_mask for masked store
      pl.store(head_m_ref, idx=(slice(None), slice(None)), val=m_next, mask=lm_mask)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      # Use precomputed lm_mask for masked store
      pl.store(head_l_ref, idx=(slice(None), slice(None)), val=l_next_safe, mask=lm_mask)

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      # Use precomputed acc_mask for masked store
      pl.store(
          head_acc_ref,
          idx=(slice(None), slice(None), slice(None)),
          val=out.reshape(head_acc_ref.shape),
          mask=acc_mask,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    # Precompute store_start and store_end once per sequence
    store_start = jnp.maximum(q_start - q_len_start, 0)
    store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

    # Precompute masks for L/M and accumulator stores (hoisted out of inner loops)
    lm_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk * num_q_heads_per_kv_head, 128),
        0,
    ) // num_q_heads_per_kv_head
    lm_mask = jnp.logical_and(lm_iota >= store_start, lm_iota < store_end)

    acc_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk, num_q_heads_per_kv_head, head_dim),
        0,
    )
    acc_mask = jnp.logical_and(acc_iota >= store_start, acc_iota < store_end)

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(o_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(o_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = q_folded_ref.at[kv_head_idx][...]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
            lm_mask=lm_mask,
            acc_mask=acc_mask,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.592,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.592 ms\n{"correct": true, "latency": 0.592, "error": "", "all_times_ms": [0.585, 0.587, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.592, 0.592, 0.592, 0.592, 0.592, 0.592, 0.592, 0.593, 0.593, 0.593, 0.593, 0.594, 0.594, 0.594, 0.594, 0.594, 0.595, 0.595, 0.595, 0.596, 0.596, 0.596, 0.597, 0.597, 0.597, 0.597, 0.597, 0.597, 0.597, 0.598, 0.598, 0.598, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.599, 0.6, 0.6, 0.601, 0.601, 0.602, 0.602, 0.603, 0.603, 0.603, 0.603, 0.604, 0.605, 0.61, 0.613, 0.629], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''The provided code implements a Ragged Paged Attention kernel on TPU. A significant bottleneck is identified in the `strided_load_kv_single` helper function, which uses bit-shifting, masking, and bitcasting to unpack interleaved Key (K) and Value (V) heads from the `kv_pages` HBM buffer. These scalar-like bitwise operations on vector registers are computationally expensive on the TPU\'s Vector Processing Unit (VPU).

To optimize this, we apply **Strategy 7**: pre-transposing the HBM `kv_pages` layout to `[total_num_pages, 2, num_kv_heads, page_size, head_dim]`. This restructuring allows the kernel to perform native, contiguous vector loads for K and V pages independently, entirely eliminating the need for expensive bit-manipulation logic.

### Optimization Plan:
1.  **Restructure HBM Layout**: Change the input `kv_pages` format from an interleaved `[total_num_pages, page_size, 2 * num_kv_heads, head_dim]` to a structured `[total_num_pages, 2, num_kv_heads, page_size, head_dim]`.
2.  **Update Async Copy Logic**: Modify `MultiPageAsyncCopyDescriptor` and `create_kv_async_copy_descriptors` to fetch slices along the new head dimension (axis 2) and the K/V dimension (axis 1).
3.  **Optimize Kernel Buffer Access**:
    *   Update the `kv_bufs` scratch buffer shape to `[2, num_kv_pages_per_blk, 2, num_kv_heads_per_blk, page_size, head_dim]`.
    *   Remove `strided_load_kv_single`.
    *   In the compute loop, directly slice K and V from the VMEM buffer: `k = kv_ref[:, 0, head_idx, :, :]` and `v = kv_ref[:, 1, head_idx, :, :]`.
4.  **Reshape for Computation**: Flatten the leading `num_kv_pages_per_blk` and `page_size` dimensions into a single `kv_len` dimension using TPU-supported reshapes (flattening leading dims onto the second-to-last dim) to prepare inputs for the attention matrix multiplication.
5.  **Alignment**: Ensure all block shapes and slices maintain alignment with (8, 128) tile boundaries where required.

```python
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

class MultiPageAsyncCopyDescriptor:
  def __init__(self, pages_hbm_ref, vmem_buf, sem, page_indices_ref, metadata):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf

def ref_ragged_paged_attention(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
                               *, sm_scale=1.0, sliding_window=None, soft_cap=None, mask_value=DEFAULT_MASK_VALUE,
                               k_scale=None, v_scale=None):
  if mask_value is None: mask_value = DEFAULT_MASK_VALUE
  _, _, num_kv_heads, page_size, head_dim = kv_pages.shape
  num_q_heads = queries.shape[1]
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start, q_end, kv_len, indices = cu_q_lens[i], cu_q_lens[i + 1], kv_lens[i], page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, 0].transpose(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:kv_len]
    v = kv_pages[indices, 1].transpose(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:kv_len]
    if k_scale is not None: k = (k.astype(jnp.float32) * k_scale).astype(q.dtype)
    if v_scale is not None: v = (v.astype(jnp.float32) * v_scale).astype(q.dtype)
    k, v = jnp.repeat(k, num_query_per_kv, axis=1), jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32) * sm_scale
    q_span, kv_span = (kv_len - (q_end - q_start)) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1), jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None: mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None: attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn = jax.nn.softmax(attn + jnp.where(mask, mask_value, 0.0), axis=-1).astype(v.dtype)
    outputs.append(jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype))
  return jnp.concatenate(outputs, axis=0)

def ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref, num_seqs_ref,
                                  q_folded_ref, kv_pages_hbm_ref, o_ref,
                                  kv_bufs, sems, l_ref, m_ref, acc_ref,
                                  *, sm_scale, sliding_window=None, soft_cap=None, mask_value=DEFAULT_MASK_VALUE,
                                  k_scale=None, v_scale=None):
  if mask_value is None: mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
  pages_per_seq, num_seqs = page_indices_ref.shape[-1], num_seqs_ref[0]
  _, num_kv_pages_per_blk, _, num_kv_heads_per_blk, page_size, _ = kv_bufs.shape
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = pl.program_id(0), pl.program_id(1)
  num_heads_blks = pl.num_programs(0)
  init_seq_idx, init_buf_idx = seq_buf_idx_ref[0], seq_buf_idx_ref[1]
  q_len_start, q_len_end = q_blk_idx * num_q_per_blk, (q_blk_idx + 1) * num_q_per_blk

  def create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx):
    start_page, end_page = kv_blk_idx * num_kv_pages_per_blk, jnp.minimum(pages_per_seq, (kv_lens_ref[seq_idx] + page_size - 1) // page_size)
    heads_start = heads_blk_idx * num_kv_heads_per_blk
    return MultiPageAsyncCopyDescriptor(kv_pages_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk)],
                                        kv_bufs.at[buf_idx], sems.at[buf_idx], page_indices_ref, (seq_idx, start_page, end_page))

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first(): create_kv_async_copy_descriptors(heads_blk_idx, init_seq_idx, 0, init_buf_idx).start()

  def flash_attention(q, k, v, head_l_ref, head_m_ref, head_acc_ref, kv_blk_idx, row_base, col_base, lm_mask, acc_mask, kv_len):
    kv_len_start = kv_blk_idx * num_kv_per_blk
    kv_mask = lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
    k, v = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype), jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)
    qk = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32) * sm_scale
    causal_mask = row_base[:, None] < col_base[None, :]
    if sliding_window is not None: causal_mask = jnp.logical_or(causal_mask, row_base[:, None] - sliding_window >= col_base[None, :])
    if soft_cap is not None: qk = soft_cap * jnp.tanh(qk / soft_cap)
    qk += jnp.where(causal_mask, mask_value, 0.0)
    m_curr, m_prev = jnp.max(qk, axis=1, keepdims=True), jnp.where(kv_blk_idx == 0, -jnp.inf, head_m_ref[...])
    s_curr = jnp.exp(qk - m_curr)
    qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
    m_next = jnp.maximum(m_prev, jnp.broadcast_to(m_curr, head_m_ref.shape))
    pl.store(head_m_ref, (slice(None), slice(None)), m_next, mask=lm_mask)
    l_prev = jnp.where(kv_blk_idx == 0, 0.0, head_l_ref[...])
    alpha, beta = jnp.exp(m_prev - m_next), jnp.exp(m_curr - m_next)
    l_next = (alpha * l_prev + beta * jnp.broadcast_to(s_curr.sum(axis=1, keepdims=True), head_l_ref.shape))
    l_safe = jnp.where(l_next == 0.0, 1.0, l_next)
    pl.store(head_l_ref, (slice(None), slice(None)), l_safe, mask=lm_mask)
    o_prev = jnp.where(kv_blk_idx == 0, 0.0, head_acc_ref[...]).reshape(-1, head_dim)
    inv_l = 1.0 / broadcast_to_match(l_safe, qkv.shape)
    out = (broadcast_to_match(alpha * l_prev, qkv.shape) * o_prev + broadcast_to_match(beta, qkv.shape) * qkv) * inv_l
    pl.store(head_acc_ref, (slice(None), slice(None), slice(None)), out.reshape(head_acc_ref.shape), mask=acc_mask)

  def broadcast_to_match(arr, shape): return jnp.concatenate([arr] * (shape[1] // arr.shape[1]), axis=1) if arr.shape[1] != shape[1] else arr

  def is_needed(state): return jnp.logical_and(state[0] == 0, jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs], state[1] < num_seqs))

  def compute_q_blk(state):
    done, cur_seq, cur_buf = state
    q_start, q_end, kv_len = cu_q_lens_ref[cur_seq], cu_q_lens_ref[cur_seq + 1], kv_lens_ref[cur_seq]
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (kv_len - (q_end - q_start)) + q_len_start - q_start + iota_q // num_q_heads_per_kv_head
    s_start, s_end = jnp.maximum(q_start - q_len_start, 0), jnp.minimum(q_end - q_len_start, num_q_per_blk)
    lm_mask = (lax.broadcasted_iota(jnp.int32, (num_q_per_blk * num_q_heads_per_kv_head, 128), 0) // num_q_heads_per_kv_head).between(s_start, s_end - 1)
    acc_mask = lax.broadcasted_iota(jnp.int32, (num_q_per_blk, num_q_heads_per_kv_head, head_dim), 0).between(s_start, s_end - 1)

    def compute_kv_blk(kv_state):
      kv_idx, buf_idx, col_start = kv_state
      next_kv, next_buf = kv_idx + 1, (0 if buf_idx == 1 else 1)
      is_last_kv = next_kv * num_kv_per_blk >= kv_len
      next_seq = cur_seq + (1 if (is_last_kv and q_end <= q_len_end) else 0)
      next_heads = heads_blk_idx + (1 if (is_last_kv and next_seq == num_seqs) else 0)
      @pl.when(next_heads < num_heads_blks)
      def prefetch(): create_kv_async_copy_descriptors(next_heads, (0 if next_seq == num_seqs else next_seq), (0 if is_last_kv else next_kv), next_buf).start()
      kv_ref = create_kv_async_copy_descriptors(heads_blk_idx, cur_seq, kv_idx, buf_idx).wait()
      col_base = col_start + jax.lax.iota(jnp.int32, num_kv_per_blk)
      for h_idx in range(num_kv_heads_per_blk):
        k, v = kv_ref.at[:, 0, h_idx, :, :][...].reshape(-1, head_dim), kv_ref.at[:, 1, h_idx, :, :][...].reshape(-1, head_dim)
        if k_scale: k = (k.astype(jnp.float32) * k_scale).astype(o_ref.dtype)
        if v_scale: v = (v.astype(jnp.float32) * v_scale).astype(o_ref.dtype)
        flash_attention(q_folded_ref.at[h_idx][...], k, v, l_ref.at[h_idx], m_ref.at[h_idx], acc_ref.at[:, h_idx*num_q_heads_per_kv_head:(h_idx+1)*num_q_heads_per_kv_head, :], kv_idx, row_base, col_base, lm_mask, acc_mask, kv_len)
      return next_kv, next_buf, col_start + num_kv_per_blk

    _, next_buf, _ = lax.while_loop(lambda s: s[0] * num_kv_per_blk < kv_len, compute_kv_blk, (0, cur_buf, jnp.int32(0)))
    return lax.select(q_end < q_len_end, done, 1), (cur_seq + (1 if q_end <= q_len_end else 0)), next_buf

  _, seq_idx, buf_idx = lax.while_loop(is_needed, compute_q_blk, (0, init_seq_idx, init_buf_idx))
  seq_buf_idx_ref[0], seq_buf_idx_ref[1] = lax.select(seq_idx < num_seqs, seq_idx, 0), buf_idx
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale=1.0, sliding_window=None, soft_cap=None, mask_value=DEFAULT_MASK_VALUE, k_scale=None, v_scale=None, num_kv_pages_per_block=None, num_queries_per_block=None, vmem_limit_bytes=None):
  if mask_value is None: mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  total_pages, page_size, _, _ = kv_pages.shape
  num_kv_heads = kv_pages.shape[2] // 2
  pages_per_seq = page_indices.shape[1]
  # Transpose kv_pages for Strategy 7: [pages, 2, kv_heads, page_size, d]
  kv_pages = kv_pages.reshape(total_pages, page_size, 2, num_kv_heads, head_dim).transpose(0, 2, 3, 1, 4)
  num_q_heads_per_blk, num_kv_heads_per_blk = get_min_heads_per_blk(num_q_heads, num_kv_heads * 2, q.dtype, kv_pages.dtype)
  num_kv_heads_per_blk //= 2
  if num_queries_per_block is None: _, num_queries_per_block = get_tuned_block_sizes(q.dtype, kv_pages.dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, num_q_tokens, pages_per_seq)
  if num_kv_pages_per_block is None: num_kv_pages_per_block, _ = get_tuned_block_sizes(q.dtype, kv_pages.dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, num_q_tokens, pages_per_seq)
  num_q_heads_per_kv = num_q_heads // num_kv_heads
  q_folded = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv, head_dim).transpose(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
  grid = (num_q_heads // num_q_heads_per_blk, (num_q_tokens + num_queries_per_block - 1) // num_queries_per_block)
  lm_scratch = pltpu.VMEM((num_kv_heads_per_blk, num_queries_per_block * num_q_heads_per_kv, 128), jnp.float32)
  kernel = pl.pallas_call(functools.partial(ragged_paged_attention_kernel, sm_scale=sm_scale, sliding_window=sliding_window, soft_cap=soft_cap, mask_value=mask_value, k_scale=k_scale, v_scale=v_scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=5, grid=grid, scratch_shapes=[pltpu.VMEM((2, num_kv_pages_per_block, 2, num_kv_heads_per_blk, page_size, head_dim), kv_pages.dtype), pltpu.SemaphoreType.DMA((2,)), lm_scratch, lm_scratch, pltpu.VMEM((num_queries_per_block, num_q_heads_per_blk, head_dim), jnp.float32)],
      in_specs=[pl.BlockSpec((num_kv_heads_per_blk, num_queries_per_block * num_q_heads_per_kv, head_dim), lambda h, q: (h, q, 0)), pl.BlockSpec(memory_space=pltpu.ANY)],
      out_specs=pl.BlockSpec((num_queries_per_block, num_q_heads_per_blk, head_dim), lambda h, q: (q, h, 0))),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem_limit_bytes),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype), name="ragged_paged_attention_kernel")
  return kernel(kv_lens, page_indices, cu_q_lens, jnp.array((0, 0), jnp.int32), num_seqs, q_folded, kv_pages)

def get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype):
  q_pack, kv_pack = 32 // dtypes.bit_width(q_dtype), 32 // dtypes.bit_width(kv_dtype)
  max_kv_tiling = 8 * kv_pack
  min_kv = max_kv_tiling if num_combined_kv_heads % max_kv_tiling == 0 else num_combined_kv_heads
  min_q = (min_kv // 2) * (num_q_heads // (num_combined_kv_heads // 2))
  return (min_q, min_kv) if (min_q // q_pack) in (1, 2, 4, 8) or (min_q // q_pack) % 8 == 0 else (num_q_heads, num_combined_kv_heads)
```''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
        lm_mask,  # precomputed mask for L/M stores
        acc_mask,  # precomputed mask for accumulator stores
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
      v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      # Use precomputed lm_mask for masked store
      pl.store(head_m_ref, idx=(slice(None), slice(None)), val=m_next, mask=lm_mask)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      # Use precomputed lm_mask for masked store
      pl.store(head_l_ref, idx=(slice(None), slice(None)), val=l_next_safe, mask=lm_mask)

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      # Use precomputed acc_mask for masked store
      pl.store(
          head_acc_ref,
          idx=(slice(None), slice(None), slice(None)),
          val=out.reshape(head_acc_ref.shape),
          mask=acc_mask,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    # Precompute store_start and store_end once per sequence
    store_start = jnp.maximum(q_start - q_len_start, 0)
    store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

    # Precompute masks for L/M and accumulator stores (hoisted out of inner loops)
    lm_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk * num_q_heads_per_kv_head, 128),
        0,
    ) // num_q_heads_per_kv_head
    lm_mask = jnp.logical_and(lm_iota >= store_start, lm_iota < store_end)

    acc_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk, num_q_heads_per_kv_head, head_dim),
        0,
    )
    acc_mask = jnp.logical_and(acc_iota >= store_start, acc_iota < store_end)

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(o_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(o_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handling for packed type that can start at
        # unaligned position!
        q = q_folded_ref.at[kv_head_idx][...]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
            lm_mask=lm_mask,
            acc_mask=acc_mask,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.591,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.591 ms\n{"correct": true, "latency": 0.591, "error": "", "all_times_ms": [0.585, 0.586, 0.586, 0.586, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.591, 0.591, 0.591, 0.591, 0.591, 0.592, 0.592, 0.592, 0.592, 0.592, 0.592, 0.593, 0.593, 0.593, 0.593, 0.594, 0.594, 0.594, 0.594, 0.594, 0.594, 0.594, 0.594, 0.595, 0.595, 0.595, 0.596, 0.596, 0.596, 0.597, 0.597, 0.597, 0.597, 0.597, 0.598, 0.598, 0.598, 0.599, 0.599, 0.599, 0.599, 0.599, 0.6, 0.6, 0.601, 0.601, 0.602, 0.602, 0.603, 0.603, 0.604, 0.613, 0.643], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''

Looking at the code, I can see a significant inefficiency in the inner loop that iterates over KV heads. Let me analyze this and create a specific optimization plan.

## Analysis of the Code

The main inefficiency is in the `compute_with_kv_blk_in_cur_seq` function, specifically in this loop:

```python
for kv_head_idx in range(num_kv_heads_per_blk):
    k, v = strided_load_kv_single(...)
    if k_scale is not None:
        k = k.astype(jnp.float32) * k_scale
        k = k.astype(o_ref.dtype)
    if v_scale is not None:
        v = v.astype(jnp.float32) * v_scale
        v = v.astype(o_ref.dtype)
    q = q_folded_ref.at[kv_head_idx][...]
    flash_attention(...)
```

The problem is that `q_folded_ref.at[kv_head_idx][...]` is computed **inside** the loop for each KV block iteration. Since this operation fetches data from `q_folded_ref` based on `kv_head_idx` and doesn\'t depend on the current KV block being processed, it can be hoisted outside the loop.

## Optimization Plan (Strategy 6 & 17)

I will apply **strategy 6** (hoist redundant operations) and **strategy 17** (use `lax.select` instead of `jnp.where` for scalar conditions) together since they\'re closely related:

**Changes to make:**
1. Pre-compute `q_folded_ref.at[kv_head_idx][...]` for all KV heads **once** before the inner KV block loop, caching these in a list/dictionary
2. Use `jax.lax.select` instead of `jnp.where` for `kv_mask` since it\'s a scalar condition (the loop iterates through sequential indices, so the condition is essentially scalar per iteration)

Here\'s the specific change:

**Current code (inside `compute_with_kv_blk_in_cur_seq`):**
```python
for kv_head_idx in range(num_kv_heads_per_blk):
    k, v = strided_load_kv_single(
        kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
    )
    if k_scale is not None:
        k = k.astype(jnp.float32) * k_scale
        k = k.astype(o_ref.dtype)
    if v_scale is not None:
        v = v.astype(jnp.float32) * v_scale
        v = v.astype(o_ref.dtype)
    q_head_idx = kv_head_idx * num_q_heads_per_kv_head
    # This is computed inside the loop - redundant!
    q = q_folded_ref.at[kv_head_idx][...]
    flash_attention(...)
```

**Optimized code:**
```python
# Pre-cache all Q slices for all KV heads BEFORE the KV block loop
q_slices = [
    q_folded_ref.at[kv_head_idx][...]
    for kv_head_idx in range(num_kv_heads_per_blk)
]

for kv_head_idx in range(num_kv_heads_per_blk):
    k, v = strided_load_kv_single(
        kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
    )
    if k_scale is not None:
        k = k.astype(jnp.float32) * k_scale
        k = k.astype(o_ref.dtype)
    if v_scale is not None:
        v = v.astype(jnp.float32) * v_scale
        v = v.astype(o_ref.dtype)
    q_head_idx = kv_head_idx * num_q_heads_per_kv_head
    # Use cached Q slice instead of computing inside loop
    q = q_slices[kv_head_idx]
    flash_attention(...)
```

Additionally, in the `flash_attention` function, replace `jnp.where` with `lax.select` for `kv_mask` application since `kv_blk_idx` is a loop variable (effectively scalar-like in this context):

**Current:**
```python
kv_mask = (
    lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
)
k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)
```

**Optimized:**
```python
kv_mask = (
    lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
)
# Use lax.select for scalar condition (lower overhead)
k = lax.select(kv_mask, k.astype(jnp.float32), jnp.float32(0)).astype(k.dtype)
v = lax.select(kv_mask, v.astype(jnp.float32), jnp.float32(0)).astype(v.dtype)
```

This optimization reduces redundant memory reads from `q_folded_ref` since the Q data for each KV head doesn\'t change between KV block iterations.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
        lm_mask,  # precomputed mask for L/M stores
        acc_mask,  # precomputed mask for accumulator stores
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      # Use lax.select for better performance with broadcasted mask
      k = lax.select(kv_mask, k.astype(jnp.float32), jnp.zeros_like(k, dtype=jnp.float32)).astype(k.dtype)
      v = lax.select(kv_mask, v.astype(jnp.float32), jnp.zeros_like(v, dtype=jnp.float32)).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      # Use precomputed lm_mask for masked store
      pl.store(head_m_ref, idx=(slice(None), slice(None)), val=m_next, mask=lm_mask)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      # Use precomputed lm_mask for masked store
      pl.store(head_l_ref, idx=(slice(None), slice(None)), val=l_next_safe, mask=lm_mask)

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      # Use precomputed acc_mask for masked store
      pl.store(
          head_acc_ref,
          idx=(slice(None), slice(None), slice(None)),
          val=out.reshape(head_acc_ref.shape),
          mask=acc_mask,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    # Precompute store_start and store_end once per sequence
    store_start = jnp.maximum(q_start - q_len_start, 0)
    store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

    # Precompute masks for L/M and accumulator stores (hoisted out of inner loops)
    lm_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk * num_q_heads_per_kv_head, 128),
        0,
    ) // num_q_heads_per_kv_head
    lm_mask = jnp.logical_and(lm_iota >= store_start, lm_iota < store_end)

    acc_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk, num_q_heads_per_kv_head, head_dim),
        0,
    )
    acc_mask = jnp.logical_and(acc_iota >= store_start, acc_iota < store_end)

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      # Pre-cache all Q slices for all KV heads BEFORE the KV head loop
      # to avoid redundant reads from q_folded_ref inside the loop.
      q_slices = [
          q_folded_ref.at[kv_head_idx][...]
          for kv_head_idx in range(num_kv_heads_per_blk)
      ]
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(o_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(o_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # Use cached Q slice instead of computing inside loop
        q = q_slices[kv_head_idx]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
            lm_mask=lm_mask,
            acc_mask=acc_mask,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.59,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.590 ms\n{"correct": true, "latency": 0.59, "error": "", "all_times_ms": [0.584, 0.585, 0.585, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.592, 0.593, 0.593, 0.593, 0.593, 0.594, 0.594, 0.594, 0.594, 0.594, 0.594, 0.595, 0.595, 0.596, 0.596, 0.596, 0.597, 0.598, 0.599, 0.599, 0.599, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.601, 0.602, 0.605, 0.614, 0.631], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''

Looking at this ragged paged attention kernel, I can see several potential optimizations. The most impactful strategy for this specific code is to **make the grid dimensions parallel instead of arbitrary**, which will allow the TPU to execute independent Q block computations concurrently.

Currently, the kernel uses:
```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=(
        "arbitrary",
        "arbitrary",
    ),
    ...
)
```

This forces sequential execution of the grid iterations. However, looking at the kernel structure:
- Each `(heads_blk_idx, q_blk_idx)` combination processes independent Q blocks
- There are no cross-iteration dependencies
- The sequence iteration is handled internally via `lax.while_loop`

By changing both dimensions to `"parallel"`, the TPU can execute multiple Q blocks concurrently, significantly improving throughput.

Here\'s the specific change to make:

```python
# In the ragged_paged_attention function, find this section:
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",  # Changed from "arbitrary" 
              "parallel",  # Changed from "arbitrary"
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )
```

This change enables the TPU\'s Megacore (on v5p) or multiple parallel execution units (on v6e) to process different Q blocks simultaneously. The kernel maintains correctness because:
1. Each Q block operates on different sequence indices (handled by `seq_buf_idx_ref`)
2. Each head block operates on different head indices
3. Output writes are to differentmemory locations (`o_ref` slices)

This is phase 15 out of 15 optimization phases, and making grid dimensions parallel is a fundamental optimization that unlocks hardware parallelism that was previously being left on the table due to the conservative "arbitrary" semantics.''',
code='''# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      metadata,  # [seq_idx, start_page_idx, end_page_idx]
  ):
    self._vmem_buf = vmem_buf
    seq_id, start_page_idx, end_page_idx = metadata
    self._async_copies = []
    # TODO(jevinjiang): Only fetch dynamic shape in need! This will insert
    # a bunch of if-ops. Check the performance when we have benchmarking setup.
    for i in range(vmem_buf.shape[0]):
      page_idx = start_page_idx + i
      page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
      self._async_copies.append(
          pltpu.make_async_copy(
              pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
              vmem_buf.at[i],
              sem,
          )
      )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  static_validate_inputs(
      queries,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0
  num_query_per_kv = num_q_heads // num_kv_heads
  outputs = []
  for i in range(num_seqs[0]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices = page_indices[i]
    q = queries[q_start:q_end]
    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[
        :kv_len
    ]
    if k_scale is not None:
      k = k.astype(jnp.float32) * k_scale
      k = k.astype(q.dtype)
    if v_scale is not None:
      v = v.astype(jnp.float32) * v_scale
      v = v.astype(q.dtype)
    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


# Expect to run these checks during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_batched_tokens = q.shape[0]
  page_size = kv_pages.shape[1]
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs[0] > max_num_seqs:
    raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
  max_kv_len = jnp.max(kv_lens)
  min_pages_per_seq = cdiv(max_kv_len, page_size)
  if pages_per_seq < min_pages_per_seq:
    raise ValueError(
        f"{pages_per_seq=} must be greater or equal to"
        f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
    )
  if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
        f" {max_num_batched_tokens=}."
    )
  for i in range(num_seqs[0]):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if q_len > kv_len:
      raise ValueError(
          f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
      )


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  _, num_q_heads, head_dim = q.shape
  _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  assert isinstance(k_scale, float) or k_scale is None
  assert isinstance(v_scale, float) or v_scale is None
  num_kv_heads = num_combined_kv_heads // 2
  max_num_seqs, pages_per_seq = page_indices.shape
  if num_seqs.shape != (1,):
    raise ValueError(f"{num_seqs.shape=} must be (1,)")
  if head_dim_k != head_dim:
    raise ValueError(
        f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
    )
  if kv_lens.shape != (max_num_seqs,):
    raise ValueError(
        f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
        " `max_num_seqs` is `page_indices.shape[0]`."
    )
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
        f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if (
      num_kv_pages_per_block is not None
      and not 0 < num_kv_pages_per_block <= pages_per_seq
  ):
    raise ValueError(
        f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
    )
  if num_queries_per_block is not None and num_queries_per_block <= 0:
    raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")
  del sm_scale  # No constraints on sm_scale.
  del mask_value  # No consstraints on mask_value.


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_folded_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
    kv_pages_hbm_ref,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_per_blk, num_q_heads_per_blk, head_dim = o_ref.shape
  pages_per_seq = page_indices_ref.shape[-1]
  num_seqs = num_seqs_ref[0]
  _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = (
      kv_bufs.shape
  )
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]
  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
    end_kv_page_idx = jnp.minimum(
        pages_per_seq, cdiv(kv_lens_ref[seq_idx], page_size)
    )
    metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
    heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
    async_copy_kv = MultiPageAsyncCopyDescriptor(
        kv_pages_hbm_ref.at[
            :, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :
        ],
        kv_bufs.at[buf_idx],
        sems.at[buf_idx],
        page_indices_ref,
        metadata,
    )
    return async_copy_kv

  # TODO(jevinjiang): Add these to Mosaic:
  # 1. Support arbitrary strided load/store for int4 and int8 dtype.
  # 2. Support arbitrary strided load/store for any last dimension.
  def strided_load_kv_single(ref, kv_head_idx, step):
    start = kv_head_idx * 2
    packing = get_dtype_packing(ref.dtype)
    if packing == 1:
      return ref[start::step, :], ref[start + 1 :: step, :]
    assert packing in (2, 4, 8)
    assert step % packing == 0
    b_ref = ref.bitcast(jnp.uint32)
    if ref.dtype == jnp.bfloat16:
      b = b_ref[(start // packing) :: (step // packing), :]
      bk = b << 16
      bv = b & jnp.uint32(0xFFFF0000)
      k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
      v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
      return k, v
    bitwidth = 32 // packing
    bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
    packed_lane = start % packing
    b = b_ref[(start // packing) :: (step // packing), :]
    bk = b >> (packed_lane * bitwidth)
    bv = b >> ((packed_lane + 1) * bitwidth)
    k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
    v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
    return k, v

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_kv = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_kv.start()

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs],
                                 cur_seq_idx < num_seqs)
    return jnp.logical_and(done == 0, should_run)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
        row_base,  # [num_q_per_blk * num_q_heads_per_kv_head]
        col_base,  # [num_kv_per_blk]
        lm_mask,  # precomputed mask for L/M stores
        acc_mask,  # precomputed mask for accumulator stores
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert (
          k.shape
          == v.shape
          == (
              num_kv_per_blk,
              head_dim,
          )
      )
      assert k.dtype == v.dtype
      assert (
          head_m_ref.shape
          == head_l_ref.shape
          == (
              num_q_per_blk * num_q_heads_per_kv_head,
              128,
          )
      )
      assert head_acc_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def load_with_init(ref, init_val):
        return jnp.where(
            kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
        )

      # kv lens will be contracting dim, we should mask out the NaNs.
      kv_mask = (
          lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
      )
      # Use lax.select for better performance with broadcasted mask
      k = lax.select(kv_mask, k.astype(jnp.float32), jnp.zeros_like(k, dtype=jnp.float32)).astype(k.dtype)
      v = lax.select(kv_mask, v.astype(jnp.float32), jnp.zeros_like(v, dtype=jnp.float32)).astype(v.dtype)

      qk = (
          jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Use precomputed row_base and col_base to form causal mask.
      causal_mask = row_base[:, None] < col_base[None, :]
      if sliding_window is not None:
        causal_mask = jnp.logical_or(
            causal_mask,
            row_base[:, None] - sliding_window >= col_base[None, :]
        )
      if soft_cap is not None:
        qk = soft_cap * jnp.tanh(qk / soft_cap)
      qk += jnp.where(causal_mask, mask_value, 0.0)
      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = load_with_init(head_m_ref, -jnp.inf)
      l_prev = load_with_init(head_l_ref, 0.0)
      m_next = jnp.maximum(m_prev, m_curr)
      # Use precomputed lm_mask for masked store
      pl.store(head_m_ref, idx=(slice(None), slice(None)), val=m_next, mask=lm_mask)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      # Cache reciprocal of softmax denominator to convert division to multiplication
      inv_l_next = 1.0 / l_next_safe
      # Use precomputed lm_mask for masked store
      pl.store(head_l_ref, idx=(slice(None), slice(None)), val=l_next_safe, mask=lm_mask)

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      inv_l_next = broadcast_to_shape(inv_l_next, qkv.shape)
      out = (l_alpha * o_curr + beta * qkv) * inv_l_next
      # Use precomputed acc_mask for masked store
      pl.store(
          head_acc_ref,
          idx=(slice(None), slice(None), slice(None)),
          val=out.reshape(head_acc_ref.shape),
          mask=acc_mask,
      )

    # Precompute row_base once per sequence/Q-block (shape [num_q_per_blk * num_q_heads_per_kv_head]).
    iota_q = jax.lax.iota(jnp.int32, num_q_per_blk * num_q_heads_per_kv_head)
    row_base = (
        (kv_len - q_len)
        + q_len_start
        - q_start
        + iota_q // num_q_heads_per_kv_head
    )

    # Precompute store_start and store_end once per sequence
    store_start = jnp.maximum(q_start - q_len_start, 0)
    store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

    # Precompute masks for L/M and accumulator stores (hoisted out of inner loops)
    lm_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk * num_q_heads_per_kv_head, 128),
        0,
    ) // num_q_heads_per_kv_head
    lm_mask = jnp.logical_and(lm_iota >= store_start, lm_iota < store_end)

    acc_iota = lax.broadcasted_iota(
        jnp.int32,
        (num_q_per_blk, num_q_heads_per_kv_head, head_dim),
        0,
    )
    acc_mask = jnp.logical_and(acc_iota >= store_start, acc_iota < store_end)

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx, kv_col_start = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
        # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
        # DMA to fixed size buffer!
        next_async_copy_kv = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        next_async_copy_kv.start()

      cur_async_copy_kv = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )
      kv_ref = cur_async_copy_kv.wait().reshape(
          num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
          head_dim,
      )
      # Precompute col_base once per KV block (shape [num_kv_per_blk]).
      iota_k = jax.lax.iota(jnp.int32, num_kv_per_blk)
      col_base = kv_col_start + iota_k
      # Pre-cache all Q slices for all KV heads BEFORE the KV head loop
      # to avoid redundant reads from q_folded_ref inside the loop.
      q_slices = [
          q_folded_ref.at[kv_head_idx][...]
          for kv_head_idx in range(num_kv_heads_per_blk)
      ]
      for kv_head_idx in range(num_kv_heads_per_blk):
        k, v = strided_load_kv_single(
            kv_ref, kv_head_idx, num_combined_kv_heads_per_blk
        )
        if k_scale is not None:
          # NOTE: Conversion between arbitrary data types is not supported.
          # That\'s why it is converted to float32 first.
          k = k.astype(jnp.float32) * k_scale
          k = k.astype(o_ref.dtype)
        if v_scale is not None:
          v = v.astype(jnp.float32) * v_scale
          v = v.astype(o_ref.dtype)
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # Use cached Q slice instead of computing inside loop
        q = q_slices[kv_head_idx]
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            acc_ref.at[
                :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
            ],
            kv_blk_idx=kv_blk_idx,
            row_base=row_base,
            col_base=col_base,
            lm_mask=lm_mask,
            acc_mask=acc_mask,
        )
      return kv_blk_idx + 1, next_buf_idx, kv_col_start + num_kv_per_blk

    _, next_buf_idx, _ = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx, jnp.int32(0)),  # (kv_blk_idx, buf_idx, kv_col_start)
    )
    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
  )
  # Reset seq_idx for next kv_heads_blk if run out of seqs!
  seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
  seq_buf_idx_ref[1] = buf_idx
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def get_min_heads_per_blk(
    num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)

  def can_be_xla_fully_tiled(x, packing):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
    raise ValueError(
        f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled."
    )
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_combined_kv_tiling = 8 * kv_packing
  min_combined_kv_heads = (
      max_combined_kv_tiling
      if num_combined_kv_heads % max_combined_kv_tiling == 0
      else num_combined_kv_heads
  )
  min_q_heads = min_combined_kv_heads // 2 * ratio
  if can_be_xla_fully_tiled(min_q_heads, q_packing):
    return min_q_heads, min_combined_kv_heads
  return num_q_heads, num_combined_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    kv_pages: jax.Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences\' queries.
    kv_pages: paged KV cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_pages,
      kv_lens,
      page_indices,
      cu_q_lens,
      num_seqs,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  num_q_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_combined_kv_heads, _ = kv_pages.shape
  assert num_combined_kv_heads % 2 == 0
  num_kv_heads = num_combined_kv_heads // 2
  _, pages_per_seq = page_indices.shape
  num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_combined_kv_heads, q.dtype, kv_pages.dtype
  )
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  if num_q_per_blk is None or num_kv_pages_per_blk is None:
    num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
        q.dtype,
        kv_pages.dtype,
        num_q_heads_per_blk,
        num_combined_kv_heads_per_blk // 2,
        head_dim,
        page_size,
        num_q_tokens,
        pages_per_seq,
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
  assert num_combined_kv_heads_per_blk % 2 == 0
  num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk
  grid = (num_heads_blks, num_q_blks)

  # Pre-fold queries to GQA layout outside the kernel
  # Original: [num_q_tokens, num_q_heads, head_dim]
  # Folded: [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  q_folded_dtype = (
      jnp.float32
      if num_q_heads_per_kv_head % get_dtype_packing(q.dtype) != 0
      else q.dtype
  )
  # Reshape to [num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim]
  q_reshaped = q.reshape(num_q_tokens, num_kv_heads, num_q_heads_per_kv_head, head_dim)
  # Transpose to [num_kv_heads, num_q_tokens, num_q_heads_per_kv_head, head_dim]
  q_transposed = q_reshaped.transpose(1, 0, 2, 3)
  # Cast if needed and reshape to [num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim]
  if q_folded_dtype != q.dtype:
    q_transposed = q_transposed.astype(q_folded_dtype)
  q_folded = q_transposed.reshape(num_kv_heads, num_q_tokens * num_q_heads_per_kv_head, head_dim)

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  def q_folded_index_map(heads_blk_idx, q_blk_idx, *_):
    return (heads_blk_idx, q_blk_idx, 0)

  q_folded_block_spec = pl.BlockSpec(
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim),
      q_folded_index_map,
  )
  in_specs = [
      q_folded_block_spec,
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_specs = q_block_spec
  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  acc_scratch = pltpu.VMEM(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      jnp.float32,
  )
  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_combined_kv_heads_per_blk,
          head_dim,
      ),
      kv_pages.dtype,
  )
  scratch_shapes = [
      double_buf_scratch,  # kv_bufs
      pltpu.SemaphoreType.DMA((2,)),  # Semaphores for double buffers.
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
      acc_scratch,
  ]
  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
      num_seqs,
  )
  kernel = pl.pallas_call(
      functools.partial(
          ragged_paged_attention_kernel,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          k_scale=k_scale,
          v_scale=v_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
          ),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      name="ragged_paged_attention_kernel",
  )

  return kernel(*scalar_prefetches, q_folded, kv_pages)


import math


CONFIG = {
    \'name\': \'pallas_ragged_paged_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_ragged_paged_attention\',
    \'max_num_batched_tokens\': 2048,
    \'max_num_seqs\': 32,
    \'num_q_heads\': 32,
    \'num_kv_heads\': 8,
    \'head_dim\': 128,
    \'page_size\': 16,
    \'pages_per_seq\': 128,
    \'atol\': 0.2,
    \'rtol\': 0.2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'num_kv_pages_per_block\': 64, \'num_queries_per_block\': 64, \'vmem_limit_bytes\': 33554432}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    max_tokens = CONFIG[\'max_num_batched_tokens\']
    max_seqs = CONFIG[\'max_num_seqs\']
    H_q = CONFIG[\'num_q_heads\']
    H_kv = CONFIG[\'num_kv_heads\']
    D = CONFIG[\'head_dim\']
    page_size = CONFIG[\'page_size\']
    pages_per_seq = CONFIG[\'pages_per_seq\']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )

    # Vary kv_lens and shuffle page indices so correctness checks exercise
    # partial-block and non-contiguous paths.
    max_kv_len = pages_per_seq * page_size
    kv_lens = jax.random.randint(
        k3, (max_seqs,), max_kv_len // 2, max_kv_len + 1
    )
    kv_lens = jnp.clip(
        (kv_lens // page_size) * page_size, page_size, max_kv_len
    ).astype(jnp.int32)

    all_pages = jax.random.permutation(
        k4, jnp.arange(total_num_pages, dtype=jnp.int32)
    )
    page_indices = all_pages.reshape(max_seqs, pages_per_seq)

    tokens_per_seq = max_tokens // max_seqs
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=TUNED_PARAMS[\'num_kv_pages_per_block\'],
        num_queries_per_block=TUNED_PARAMS[\'num_queries_per_block\'],
        vmem_limit_bytes=TUNED_PARAMS[\'vmem_limit_bytes\'],
    )
''',
score=0.587,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.587 ms\n{"correct": true, "latency": 0.587, "error": "", "all_times_ms": [0.583, 0.583, 0.584, 0.584, 0.584, 0.584, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.585, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.586, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.588, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.589, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.591, 0.591, 0.591, 0.591, 0.591, 0.591, 0.592, 0.592, 0.593, 0.593, 0.596, 0.596, 0.597, 0.597, 0.597, 0.599, 0.6, 0.6, 0.601, 0.603, 0.622], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr='')