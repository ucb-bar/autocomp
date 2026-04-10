CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''# Copyright 2023 The JAX Authors.
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

"""Flash Attention TPU kernel."""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

  q: jax.Array  # [batch_size, q_seq_len]
  kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tile sizes parameterizing FlashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.
  """
  block_q: int
  block_k_major: int
  block_k: int
  block_b: int

  block_q_major_dkv: int | None = None
  block_k_major_dkv: int | None = None
  block_k_dkv: int | None = None
  block_q_dkv: int | None = None

  block_k_major_dq: int | None = None
  block_k_dq: int | None = None
  block_q_dq: int | None = None

  def __post_init__(self):
    def verify_major_minor(prefix, suffix, major, minor):
      if minor > major:
        raise ValueError(
            f"{prefix}{suffix}={minor} should be smaller than"
            f" {prefix}_major{suffix}={major}"
        )
      if major % minor != 0:
        raise ValueError(
            f"{prefix}{suffix}={minor} should divide"
            f" {prefix}_major{suffix}={major}"
        )

    verify_major_minor("block_k", "", self.block_k_major, self.block_k)
    if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
      verify_major_minor(
          "block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv
      )
    if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
      verify_major_minor(
          "block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv
      )
    if self.block_k_major_dq is not None and self.block_k_dq is not None:
      verify_major_minor(
          "block_k", "_dq", self.block_k_major_dq, self.block_k_dq
      )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_major_dkv,
        self.block_k_major_dkv,
        self.block_q_dkv,
        self.block_k_dkv,
        self.block_k_major_dq,
        self.block_k_dq,
        self.block_q_dq,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
    return BlockSizes(
        block_q=128,
        block_k_major=128,
        block_k=128,
        block_b=1,
        block_q_major_dkv=128,
        block_k_major_dkv=128,
        block_k_dkv=128,
        block_q_dkv=128,
        block_k_major_dq=128,
        block_k_dq=128,
        block_q_dq=128,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, d_model = q.shape
  batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
  batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
  if batch_size != batch_size_k or batch_size != batch_size_v:
    raise ValueError(
        f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
        f" {batch_size_v} (for q, k, v respectively)"
    )
  if num_heads != num_heads_k or num_heads != num_heads_v:
    raise ValueError(
        f"Head count mismatch: got {num_heads}, {num_heads_k},"
        f" {num_heads_v} (for q, k, v respectively)"
    )
  if d_model != d_model_k:
    raise ValueError(
        f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k"
        " respectively)"
    )
  if d_model != d_model_v:
    raise NotImplementedError(
        "V model dimension unequal to KV model dimension unsupported"
    )
  if kv_seq_len != kv_seq_len_v:
    raise ValueError(
        f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}"
    )
  if ab is not None:
    if ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
      raise ValueError(
          f"Attention bias shape mismatch: expected ({batch_size=},"
          f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
      )
  if segment_ids is not None:
    if segment_ids.q.shape != (batch_size, q_seq_len):
      raise ValueError(
          f"Q segment ids shape mismatch: expected ({batch_size=},"
          f" {q_seq_len=},), got {segment_ids.q.shape}"
      )
    if segment_ids.kv.shape != (batch_size, kv_seq_len):
      raise ValueError(
          f"KV segment ids shape mismatch: expected ({batch_size=},"
          f" {kv_seq_len=},), got {segment_ids.kv.shape}"
      )
  if block_sizes is None:
    block_sizes = BlockSizes.get_default(
        batch_size, num_heads, q_seq_len, kv_seq_len, d_model
    )
  return _flash_attention(
      q, k, v, ab, segment_ids, False, causal, sm_scale, block_sizes, debug
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  return _flash_attention_impl(
      q,
      k,
      v,
      ab,
      segment_ids,
      save_residuals,
      causal,
      sm_scale,
      block_sizes.block_b,
      block_sizes.block_q,
      block_sizes.block_k_major,
      block_sizes.block_k,
      debug,
  )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  o, l, m = _flash_attention(
      q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug
  )
  return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    residuals,
    do,
):
  """VJP rule for FlashAttention."""
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  (q, k, v, ab, segment_ids, o, l, m) = residuals
  if not block_sizes.has_backward_blocks:
    raise ValueError(
        "Program is being differentiated, but not all backward blocks are"
        " specified"
    )

  di = jnp.sum(
      o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
  )  # [batch_size, num_heads, q_seq_len]

  dk, dv = _flash_attention_bwd_dkv(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_major_dkv,
      block_k_major=block_sizes.block_k_major_dkv,
      block_k=block_sizes.block_k_dkv,
      block_q=block_sizes.block_q_dkv,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )

  dq, ds = _flash_attention_bwd_dq(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_dq,
      block_k_major=block_sizes.block_k_major_dq,
      block_k=block_sizes.block_k_dq,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )
  return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
  # A block is considered below or on diagonal as long as the bottom left
  # corner of the block is below or on diagonal.
  return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
  block_b = q_tile_ref.shape[0]
  # If we\'re not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[-1]

  kv_seq_idx = pl.program_id(3)
  @pl.when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[batch_idx] = jnp.full(
        m_scratch_ref.shape[2:], -jnp.inf, jnp.float32
    )
    l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
    acc_scratch_ref[batch_idx] = jnp.zeros(
        acc_scratch_ref.shape[2:], jnp.float32
    )

  q_seq_idx = pl.program_id(2)
  if causal:
    should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    @pl.loop(0, block_k_major // block_k, unroll=True)
    def _body(i):
      m_prev = m_scratch_ref[batch_idx]
      l_prev = l_scratch_ref[batch_idx]
      q = q_tile_ref[batch_idx]  # [block_q, head_dim]
      start_k = i * block_k
      k = k_tile_ref[
          (*batch_idx, pl.dslice(start_k, block_k), slice(None))
      ]  # [block_k, head_dim]

      s = jax.lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q, block_k]

      # Add attention bias if needed.
      # TODO(tanburn) Should the attention bias be added before or after
      # multiplication by sm_scale?
      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
        ].astype(jnp.float32)
        s += ab

      if sm_scale != 1.0:
        s *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
              f"kv block size must be a multiple of {NUM_LANES}"
          )
        q_segment_ids = pltpu.repeat(
            q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            batch_idx[0], :1, pl.dslice(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_idx * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

      m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
      m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

      block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
      if rem:
        raise NotImplementedError(
            f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
        )
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

      alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

      l_corr = alpha * l_prev

      l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

      l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
      acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
    if l_ref is not None:
      l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale != 1.0:
    s *= sm_scale

  mask = None
  if q_segment_ids_tile_ref is not None:
    repeats, rem = divmod(block_k, NUM_LANES)
    if rem:
      raise NotImplementedError(
          f"kv block size must be a multiple of {NUM_LANES}"
      )
    q_segment_ids = q_segment_ids_tile_ref[
        batch_idx[0]
    ]  # [block_q, NUM_LANES].
    q_segment_ids = pltpu.repeat(
        q_segment_ids, repeats, axis=1
    )  # [block_q, block_k].
    kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
    mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

  if causal:
    q_seq_idx = pl.program_id(2)
    mask_shape = (block_q, block_k)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    row_ids += q_seq_idx * block_q
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = col_ids <= row_ids
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

  m = jnp.max(s, axis=1)[:, None]
  p = jnp.exp(s - m)
  l = jnp.sum(p, axis=1)[:, None]
  p /= l

  if m_ref is not None:
    m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
  if l_ref is not None:
    l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

  v = v_tile_ref[batch_idx]
  o_tile_ref[batch_idx] = jax.lax.dot(
      p.astype(v.dtype), v, preferred_element_type=jnp.float32
  ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
  body_cost = pl.estimate_cost(
    mha_reference,
    q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
  )
  input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
  output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
  return pl.CostEstimate(
      flops=body_cost.flops,
      transcendentals=body_cost.transcendentals,
      bytes_accessed=input_bytes + output_bytes,
  )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
  _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
  _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

  # TODO(apaszke): Tile over heads as well.
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      should_run = below_or_on_diag(
          q_seq_index, block_q, kv_seq_index, block_k_major
      )
      # If the ab block is skipped, prefetch the next valid ab block, i.e. the
      # 0th kv to be used for the next block_q rows.
      next_q_index = lax.select(
          should_run,
          q_seq_index,
          lax.select(
              q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
          ),
      )
      next_kv_index = lax.select(should_run, kv_seq_index, 0)
    else:
      next_q_index = q_seq_index
      next_kv_index = kv_seq_index

    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        next_kv_index = lax.select(
            below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (block_b, block_q, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      cost_estimate=_fwd_cost_estimate(
          q,
          k,
          v,
          ab,
          segment_ids,
          causal=causal,
          sm_scale=sm_scale,
          kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
          kernel_outputs_specs=out_shape,
      ),
  )(q, k, v, ab, q_segment_ids, kv_segment_ids)
  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
  _, _, block_q_major, _ = q_tile_ref.shape
  _, _, block_k_major, _ = k_tile_ref.shape

  q_seq_index = pl.program_id(axis=3)
  kv_seq_index = pl.program_id(axis=2)

  @pl.when(q_seq_index == 0)
  def start_new_sequence():
    dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
    dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

  def q_body(j, _):
    start_q = j * block_q
    def k_body(i, _):
      start_k = i * block_k
      k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
      l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(
          jnp.float32
      )  # [block_q, 128]

      capped_logits = lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q_major, block_k]

      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            0,
            0,
            pl.dslice(j * block_q, block_q),
            pl.dslice(i * block_k, block_k),
        ].astype(jnp.float32)
        capped_logits += ab

      if sm_scale != 1.0:
        capped_logits *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
          )
        q_segment_ids = q_segment_ids_tile_ref[
            0, pl.ds(start_q, block_q), :
        ]  # [block_q, NUM_LANES].
        q_segment_ids = pltpu.repeat(
            q_segment_ids, repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            :, 0, pl.ds(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_index * block_q_major + start_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_index * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      capped_logits = (
          capped_logits
          if mask is None
          else capped_logits + jnp.where(mask, 0.0, mask_value)
      )

      p = jnp.exp(
          capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
      )
      p = p * pltpu.repeat(
          1 / l, block_k // MIN_BLOCK_SIZE, axis=1
      )  # [block_q_major, block_k_major]
      dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
      dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(
          dv_scratch_ref.dtype
      )

      # di: [block_q, 128]
      # do: [block_q, head_dim]
      # v: [block_k_major, head_dim]
      dp = lax.dot_general(
          do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )
      ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

      if sm_scale != 1.0:
        ds = ds * sm_scale

      # ds: [block_q_major, block_k_major]
      # q: [block_q_major, head_dim]
      dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
      dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(
          dk_scratch_ref.dtype
      )
    lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

  @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
  def end_of_q_sequence():
    dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
    dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
  _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

  # kv index needs to be before q index since q index is the contractng
  # dimension.
  grid = (
      batch_size,
      num_heads,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
  )

  def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    if causal:
      # If the q block is skipped, stay at the 0th q block.
      next_q_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          q_seq_index,
          0,
      )
    else:
      next_q_index = q_seq_index

    return (batch_index, head_index, next_q_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  assert qo_spec.block_shape is not None
  assert q.ndim == len(qo_spec.block_shape)
  do_spec = qo_spec
  assert do.ndim == len(qo_spec.block_shape)

  def kv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, _, q_seq_index):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index
    ):
      del head_index
      if causal:
        next_q_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            q_seq_index,
            0,
        )
      else:
        next_q_index = q_seq_index
      return (batch_index, next_q_index, 0)

    def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
      del head_index
      return (batch_index, 0, kv_seq_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           k.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           v.dtype),
  ]
  def dkv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
  out_specs = [dkv_spec, dkv_spec]
  scratch_shapes = [
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      block_q=block_q,  # type: ignore
      block_k=block_k,  # type: ignore
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      q_seq_len=q_seq_len,
  )
  name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
    assert dk.shape == k.shape
    assert dv.shape == v.shape
  return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
  _, _, block_k_major, _ = k_tile_ref.shape
  _, _, block_q_major, _ = q_tile_ref.shape

  kv_seq_index = pl.program_id(axis=3)
  q_seq_index = pl.program_id(axis=2)

  @pl.when(kv_seq_index == 0)
  def start_new_sequence():
    dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

  def body(i, _):
    k_slice = pl.ds(i * block_k, block_k)
    q = q_tile_ref[0, 0, :, :]
    k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
    di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

    capped_logits = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )

    if ab_tile_ref is not None:
      ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(
          jnp.float32
      )
      capped_logits += ab

    if sm_scale != 1.0:
      capped_logits *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
      repeats, rem = divmod(block_k, NUM_LANES)
      if rem:
        raise NotImplementedError(
            f"kv block size must be a multiple of {NUM_LANES}"
        )
      q_segment_ids = pltpu.repeat(
          q_segment_ids_tile_ref[0], repeats, axis=1
      )  # [block_q, block_k].
      kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]  # [1, block_k].
      mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
      mask_shape = (block_q_major, block_k)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      row_ids += q_seq_index * block_q_major
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      col_ids += kv_seq_index * block_k_major + i * block_k
      causal_mask = col_ids <= row_ids
      mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    capped_logits = (
        capped_logits
        if mask is None
        else capped_logits + jnp.where(mask, 0.0, mask_value)
    )

    p = jnp.exp(
        capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
    )
    p = p * pltpu.repeat(
        1 / l, block_k // MIN_BLOCK_SIZE, axis=1
    )  # [block_q_major, block_k]

    # di: [block_q_major, 128]
    # do: [block_q_major, head_dim]
    # v: [block_k_major, head_dim]
    dp = jax.lax.dot_general(
        do,
        v,
        TRANS_B_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p
    # dp = jnp.dot(do, v.T)
    # ds = (dp - (dp * p).sum(axis=1)[:, None]) * p

    if sm_scale != 1.0:
      ds = ds * sm_scale

    if ds_tile_ref is not None:
      ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(
          ds_tile_ref.dtype
      )

    # dp: [block_q_major, block_k]
    # k: [block_k, head_dim]
    dq_scratch_ref[:, :] += lax.dot(
        ds.astype(k.dtype),
        k,
        preferred_element_type=jnp.float32,
    ).astype(dq_scratch_ref.dtype)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
    should_not_run = lax.select(should_run, False, True)
  else:
    should_run = True
    should_not_run = False  # type: ignore

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

  @pl.when(should_not_run)
  def zero_out_ds():
    if ds_tile_ref is not None:
      ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

  @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
  def end_of_kv_sequence():
    dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

  grid = (
      batch_size,
      num_heads,
      q_seq_len // block_q_major,
      kv_seq_len // block_k_major,
  )

  def qo_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  do_spec = qo_spec

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        # If the kv block is skipped, prefetch the next valid kv block, i.e. the
        # 0th one to be used for the next block_q rows.
        next_kv_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
  ]
  dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  out_specs = [
      dq_spec,
      dab_spec,
  ]
  scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      block_k=block_k,  # type: ignore
      kv_seq_len=kv_seq_len,
  )
  name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dq, ds = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

  # dab is just ds
  return dq, ds


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out


@functools.partial(
    jax.jit, static_argnames=["causal", "mask_value", "sm_scale"]
)
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
  return _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=False,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  return mha_reference_no_custom_vjp(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=save_residuals,
  )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  if save_residuals:
    raise NotImplementedError
  res = _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=True,
  )
  assert isinstance(res, tuple)
  out, l, m = res
  return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "sm_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
  if sm_scale != 1.0:
    raise NotImplementedError

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  if ab is not None:
    logits += ab

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  unnormalized = jnp.exp(logits - m[..., None])
  p = unnormalized / l[..., None]
  dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

  dp = jnp.einsum(
      "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
      ..., None
  ]  # [batch_size, num_heads, q_seq_len]

  ds = (dp - di) * p
  dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

  # dab is just ds
  dab = ds if ab is not None else None
  return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
  del save_residuals
  q, k, v, ab, segment_ids, o, l, m = residuals
  dq, dk, dv, dab = mha_reference_bwd(
      q,
      k,
      v,
      ab,
      segment_ids,
      o,
      l,
      m,
      do,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
  )
  return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
  if block > dim:
    raise ValueError(
        f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
    )
  if should_divide and dim % block != 0:
    raise ValueError(
        f"{dim_name}={dim} should be divisible by {block_name}={block}"
    )


CONFIG = {
    \'name\': \'pallas_flash_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 32,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'block_q\': 2048, \'block_k_major\': 2048, \'block_k\': 512, \'block_b\': 1, \'block_q_major_dkv\': 128, \'block_k_major_dkv\': 128, \'block_k_dkv\': 128, \'block_q_dkv\': 128, \'block_k_major_dq\': 128, \'block_k_dq\': 128, \'block_q_dq\': 128}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG[\'batch\']
    H = CONFIG[\'num_heads\']
    S = CONFIG[\'seq_len\']
    D = CONFIG[\'head_dim\']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS[\'block_q\'],
        block_k_major=TUNED_PARAMS[\'block_k_major\'],
        block_k=TUNED_PARAMS[\'block_k\'],
        block_b=TUNED_PARAMS[\'block_b\'],
        block_q_major_dkv=TUNED_PARAMS[\'block_q_major_dkv\'],
        block_k_major_dkv=TUNED_PARAMS[\'block_k_major_dkv\'],
        block_k_dkv=TUNED_PARAMS[\'block_k_dkv\'],
        block_q_dkv=TUNED_PARAMS[\'block_q_dkv\'],
        block_k_major_dq=TUNED_PARAMS[\'block_k_major_dq\'],
        block_k_dq=TUNED_PARAMS[\'block_k_dq\'],
        block_q_dq=TUNED_PARAMS[\'block_q_dq\'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.371,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 0.371 ms\n{"correct": true, "latency": 0.371, "error": "", "all_times_ms": [0.365, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.367, 0.367, 0.367, 0.367, 0.367, 0.367, 0.368, 0.368, 0.368, 0.368, 0.368, 0.368, 0.368, 0.368, 0.368, 0.368, 0.369, 0.369, 0.369, 0.369, 0.369, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.371, 0.371, 0.371, 0.371, 0.372, 0.372, 0.372, 0.372, 0.372, 0.373, 0.373, 0.373, 0.373, 0.373, 0.373, 0.373, 0.374, 0.374, 0.374, 0.374, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.376, 0.376, 0.376, 0.377, 0.377, 0.377, 0.377, 0.377, 0.378, 0.378, 0.378, 0.378, 0.378, 0.379, 0.379, 0.379, 0.38, 0.38, 0.381, 0.384, 0.387, 0.405, 0.425, 0.44], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''This optimization applies the **Unnormalized Online Softmax Accumulation** strategy to the Flash Attention forward kernel.

### Optimization Plan
The original Flash Attention kernel performs normalization (division by the running sum of exponentials $L$) and rescaling ($L_{prev}/L_{curr}$) inside every iteration of the K/V block loop. On TPU, divisions and repeated rescaling are expensive VPU operations.

1.  **Mathematical Restructuring**: We transition from tracking normalized attention to unnormalized attention. Instead of maintaining $O = \frac{\sum e^{s_i-m} v_i}{\sum e^{s_i-m}}$, we maintain the numerator $Acc = \sum e^{s_i-m} v_i$ and the denominator $L = \sum e^{s_i-m}$ separately.
2.  **Algorithm Details**:
    *   Initialize $m = -\infty, L = 0, Acc = 0$.
    *   For each K/V block:
        *   Find $m_{curr} = \max(S)$.
        *   Find $m_{next} = \max(m, m_{curr})$.
        *   Compute rescaling factor $\alpha = e^{m - m_{next}}$.
        *   Compute unnormalized probabilities $P = e^{S - m_{next}}$.
        *   Update $Acc = Acc \cdot \alpha + P \cdot V$.
        *   Update $L = L \cdot \alpha + \text{row\_sum}(P)$.
    *   After the loop: $O = Acc / L$.
3.  **Performance Impact**: This eliminates one `1/L` reciprocal and two large matrix-vector multiplications ($Acc \cdot \text{scale}$) per loop iteration, replacing them with a single multiplication by a scalar $\alpha$. The final division is performed once per Q-block sequence.
4.  **Hardware Alignment**: By maintaining all accumulators in `float32` within VMEM, we preserve numerical precision while minimizing the conversion and VPU overhead.

```python
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import functools
import math
from typing import Any, NamedTuple

# ... [SegmentIds, BlockSizes, and flash_attention remain unchanged] ...

def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[-1]
  
  # Constants for tiling
  block_k_repeats = block_k // 128
  head_dim_repeats = head_dim // 128

  kv_seq_idx = pl.program_id(3)
  @pl.when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[batch_idx] = jnp.full((block_q, 128), -jnp.inf, jnp.float32)
    l_scratch_ref[batch_idx] = jnp.zeros((block_q, 128), jnp.float32)
    acc_scratch_ref[batch_idx] = jnp.zeros((block_q, head_dim), jnp.float32)

  q_seq_idx = pl.program_id(2)
  if causal:
    should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    @pl.loop(0, block_k_major // block_k, unroll=True)
    def _body(i):
      m_prev = m_scratch_ref[batch_idx]
      l_prev = l_scratch_ref[batch_idx]
      acc_prev = acc_scratch_ref[batch_idx]
      
      q = q_tile_ref[batch_idx]
      start_k = i * block_k
      k = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]

      s = jax.lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )

      if ab_tile_ref is not None:
        ab = ab_tile_ref[(*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))].astype(jnp.float32)
        s += ab

      if sm_scale != 1.0:
        s *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[batch_idx[0]], block_k_repeats, axis=1)
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1, pl.dslice(start_k, block_k)]
        mask = jnp.equal(q_segment_ids, kv_segment_ids)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + (kv_seq_idx * block_k_major + start_k)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

      s = jnp.where(mask, s, mask_value) if mask is not None else s

      m_curr = jnp.max(s, axis=1)[:, None]
      m_next = jnp.maximum(m_prev, m_curr)

      # Alpha rescaling for the running statistics
      alpha_prev = jnp.exp(m_prev - m_next)
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

      l_curr = jnp.sum(p, axis=1)[:, None]
      l_next = l_prev * alpha_prev + l_curr
      
      # Accumulate unnormalized results: Acc = Acc * alpha + P @ V
      acc_rescaled = acc_prev * (pltpu.repeat(alpha_prev, head_dim_repeats, 1) if head_dim_repeats > 1 else alpha_prev)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      acc_next = acc_rescaled + jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next
      acc_scratch_ref[batch_idx] = acc_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Final normalization step: O = Acc / L
    l_final = l_scratch_ref[batch_idx]
    acc_final = acc_scratch_ref[batch_idx]
    # Division by zero safety: though L should be >= 1.0 if any token was seen
    o_tile_ref[batch_idx] = (acc_final / (pltpu.repeat(l_final, head_dim_repeats, 1) if head_dim_repeats > 1 else l_final)).astype(o_tile_ref.dtype)
    
    if l_ref is not None:
      l_ref[batch_idx] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)

# ... [Remaining functions _flash_attention_kernel_single_batch_single_step, _flash_attention_impl, etc. remain unchanged] ...
```''',
code='''# Copyright 2023 The JAX Authors.
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

"""Flash Attention TPU kernel."""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

  q: jax.Array  # [batch_size, q_seq_len]
  kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tile sizes parameterizing FlashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.
  """
  block_q: int
  block_k_major: int
  block_k: int
  block_b: int

  block_q_major_dkv: int | None = None
  block_k_major_dkv: int | None = None
  block_k_dkv: int | None = None
  block_q_dkv: int | None = None

  block_k_major_dq: int | None = None
  block_k_dq: int | None = None
  block_q_dq: int | None = None

  def __post_init__(self):
    def verify_major_minor(prefix, suffix, major, minor):
      if minor > major:
        raise ValueError(
            f"{prefix}{suffix}={minor} should be smaller than"
            f" {prefix}_major{suffix}={major}"
        )
      if major % minor != 0:
        raise ValueError(
            f"{prefix}{suffix}={minor} should divide"
            f" {prefix}_major{suffix}={major}"
        )

    verify_major_minor("block_k", "", self.block_k_major, self.block_k)
    if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
      verify_major_minor(
          "block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv
      )
    if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
      verify_major_minor(
          "block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv
      )
    if self.block_k_major_dq is not None and self.block_k_dq is not None:
      verify_major_minor(
          "block_k", "_dq", self.block_k_major_dq, self.block_k_dq
      )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_major_dkv,
        self.block_k_major_dkv,
        self.block_q_dkv,
        self.block_k_dkv,
        self.block_k_major_dq,
        self.block_k_dq,
        self.block_q_dq,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
    return BlockSizes(
        block_q=128,
        block_k_major=128,
        block_k=128,
        block_b=1,
        block_q_major_dkv=128,
        block_k_major_dkv=128,
        block_k_dkv=128,
        block_q_dkv=128,
        block_k_major_dq=128,
        block_k_dq=128,
        block_q_dq=128,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, d_model = q.shape
  batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
  batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
  if batch_size != batch_size_k or batch_size != batch_size_v:
    raise ValueError(
        f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
        f" {batch_size_v} (for q, k, v respectively)"
    )
  if num_heads != num_heads_k or num_heads != num_heads_v:
    raise ValueError(
        f"Head count mismatch: got {num_heads}, {num_heads_k},"
        f" {num_heads_v} (for q, k, v respectively)"
    )
  if d_model != d_model_k:
    raise ValueError(
        f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k"
        " respectively)"
    )
  if d_model != d_model_v:
    raise NotImplementedError(
        "V model dimension unequal to KV model dimension unsupported"
    )
  if kv_seq_len != kv_seq_len_v:
    raise ValueError(
        f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}"
    )
  if ab is not None:
    if ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
      raise ValueError(
          f"Attention bias shape mismatch: expected ({batch_size=},"
          f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
      )
  if segment_ids is not None:
    if segment_ids.q.shape != (batch_size, q_seq_len):
      raise ValueError(
          f"Q segment ids shape mismatch: expected ({batch_size=},"
          f" {q_seq_len=},), got {segment_ids.q.shape}"
      )
    if segment_ids.kv.shape != (batch_size, kv_seq_len):
      raise ValueError(
          f"KV segment ids shape mismatch: expected ({batch_size=},"
          f" {kv_seq_len=},), got {segment_ids.kv.shape}"
      )
  if block_sizes is None:
    block_sizes = BlockSizes.get_default(
        batch_size, num_heads, q_seq_len, kv_seq_len, d_model
    )
  return _flash_attention(
      q, k, v, ab, segment_ids, False, causal, sm_scale, block_sizes, debug
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  return _flash_attention_impl(
      q,
      k,
      v,
      ab,
      segment_ids,
      save_residuals,
      causal,
      sm_scale,
      block_sizes.block_b,
      block_sizes.block_q,
      block_sizes.block_k_major,
      block_sizes.block_k,
      debug,
  )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  o, l, m = _flash_attention(
      q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug
  )
  return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    residuals,
    do,
):
  """VJP rule for FlashAttention."""
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  (q, k, v, ab, segment_ids, o, l, m) = residuals
  if not block_sizes.has_backward_blocks:
    raise ValueError(
        "Program is being differentiated, but not all backward blocks are"
        " specified"
    )

  di = jnp.sum(
      o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
  )  # [batch_size, num_heads, q_seq_len]

  dk, dv = _flash_attention_bwd_dkv(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_major_dkv,
      block_k_major=block_sizes.block_k_major_dkv,
      block_k=block_sizes.block_k_dkv,
      block_q=block_sizes.block_q_dkv,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )

  dq, ds = _flash_attention_bwd_dq(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_dq,
      block_k_major=block_sizes.block_k_major_dq,
      block_k=block_sizes.block_k_dq,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )
  return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
  # A block is considered below or on diagonal as long as the bottom left
  # corner of the block is below or on diagonal.
  return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
  block_b = q_tile_ref.shape[0]
  # If we\'re not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[-1]

  kv_seq_idx = pl.program_id(3)
  @pl.when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[batch_idx] = jnp.full(
        m_scratch_ref.shape[2:], -jnp.inf, jnp.float32
    )
    l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
    acc_scratch_ref[batch_idx] = jnp.zeros(
        acc_scratch_ref.shape[2:], jnp.float32
    )

  q_seq_idx = pl.program_id(2)
  if causal:
    should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    @pl.loop(0, block_k_major // block_k, unroll=True)
    def _body(i):
      m_prev = m_scratch_ref[batch_idx]
      l_prev = l_scratch_ref[batch_idx]
      acc_prev = acc_scratch_ref[batch_idx]
      q = q_tile_ref[batch_idx]  # [block_q, head_dim]
      start_k = i * block_k
      k = k_tile_ref[
          (*batch_idx, pl.dslice(start_k, block_k), slice(None))
      ]  # [block_k, head_dim]

      s = jax.lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q, block_k]

      # Add attention bias if needed.
      # TODO(tanburn) Should the attention bias be added before or after
      # multiplication by sm_scale?
      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
        ].astype(jnp.float32)
        s += ab

      if sm_scale != 1.0:
        s *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
              f"kv block size must be a multiple of {NUM_LANES}"
          )
        q_segment_ids = pltpu.repeat(
            q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            batch_idx[0], :1, pl.dslice(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_idx * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

      m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
      m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

      block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
      if rem:
        raise NotImplementedError(
            f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
        )
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

      # Unnormalized online softmax: track Acc (unnormalized) and L separately
      alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

      l_curr = jnp.sum(p, axis=1)[:, None]  # Shape [block_q, 1]
      l_next = l_prev * alpha + l_curr  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      alpha_broadcast = lambda a: pltpu.repeat(a, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          alpha_broadcast = lambda a: a[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      # Rescale previous accumulator and add new contribution
      acc_rescaled = acc_prev * alpha_broadcast(alpha)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      acc_next = acc_rescaled + o_curr

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next
      acc_scratch_ref[batch_idx] = acc_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Final normalization: O = Acc / L
    l_final = l_scratch_ref[batch_idx]
    acc_final = acc_scratch_ref[batch_idx]
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda l: l[:, :head_dim]
      else:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )
    l_final_safe = jnp.where(l_final == 0.0, 1.0, l_final)
    o_tile_ref[batch_idx] = (acc_final / l_broadcast(l_final_safe)).astype(o_tile_ref.dtype)
    if l_ref is not None:
      l_ref[batch_idx] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale != 1.0:
    s *= sm_scale

  mask = None
  if q_segment_ids_tile_ref is not None:
    repeats, rem = divmod(block_k, NUM_LANES)
    if rem:
      raise NotImplementedError(
          f"kv block size must be a multiple of {NUM_LANES}"
      )
    q_segment_ids = q_segment_ids_tile_ref[
        batch_idx[0]
    ]  # [block_q, NUM_LANES].
    q_segment_ids = pltpu.repeat(
        q_segment_ids, repeats, axis=1
    )  # [block_q, block_k].
    kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
    mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

  if causal:
    q_seq_idx = pl.program_id(2)
    mask_shape = (block_q, block_k)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    row_ids += q_seq_idx * block_q
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = col_ids <= row_ids
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

  m = jnp.max(s, axis=1)[:, None]
  p = jnp.exp(s - m)
  l = jnp.sum(p, axis=1)[:, None]
  p /= l

  if m_ref is not None:
    m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
  if l_ref is not None:
    l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

  v = v_tile_ref[batch_idx]
  o_tile_ref[batch_idx] = jax.lax.dot(
      p.astype(v.dtype), v, preferred_element_type=jnp.float32
  ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
  body_cost = pl.estimate_cost(
    mha_reference,
    q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
  )
  input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
  output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
  return pl.CostEstimate(
      flops=body_cost.flops,
      transcendentals=body_cost.transcendentals,
      bytes_accessed=input_bytes + output_bytes,
  )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
  _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
  _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

  # TODO(apaszke): Tile over heads as well.
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      should_run = below_or_on_diag(
          q_seq_index, block_q, kv_seq_index, block_k_major
      )
      # If the ab block is skipped, prefetch the next valid ab block, i.e. the
      # 0th kv to be used for the next block_q rows.
      next_q_index = lax.select(
          should_run,
          q_seq_index,
          lax.select(
              q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
          ),
      )
      next_kv_index = lax.select(should_run, kv_seq_index, 0)
    else:
      next_q_index = q_seq_index
      next_kv_index = kv_seq_index

    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        next_kv_index = lax.select(
            below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (block_b, block_q, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      cost_estimate=_fwd_cost_estimate(
          q,
          k,
          v,
          ab,
          segment_ids,
          causal=causal,
          sm_scale=sm_scale,
          kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
          kernel_outputs_specs=out_shape,
      ),
  )(q, k, v, ab, q_segment_ids, kv_segment_ids)
  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
  _, _, block_q_major, _ = q_tile_ref.shape
  _, _, block_k_major, _ = k_tile_ref.shape

  q_seq_index = pl.program_id(axis=3)
  kv_seq_index = pl.program_id(axis=2)

  @pl.when(q_seq_index == 0)
  def start_new_sequence():
    dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
    dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

  def q_body(j, _):
    start_q = j * block_q
    def k_body(i, _):
      start_k = i * block_k
      k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
      l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(
          jnp.float32
      )  # [block_q, 128]

      capped_logits = lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q_major, block_k]

      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            0,
            0,
            pl.dslice(j * block_q, block_q),
            pl.dslice(i * block_k, block_k),
        ].astype(jnp.float32)
        capped_logits += ab

      if sm_scale != 1.0:
        capped_logits *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
          )
        q_segment_ids = q_segment_ids_tile_ref[
            0, pl.ds(start_q, block_q), :
        ]  # [block_q, NUM_LANES].
        q_segment_ids = pltpu.repeat(
            q_segment_ids, repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            :, 0, pl.ds(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_index * block_q_major + start_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_index * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      capped_logits = (
          capped_logits
          if mask is None
          else capped_logits + jnp.where(mask, 0.0, mask_value)
      )

      p = jnp.exp(
          capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
      )
      p = p * pltpu.repeat(
          1 / l, block_k // MIN_BLOCK_SIZE, axis=1
      )  # [block_q_major, block_k_major]
      dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
      dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(
          dv_scratch_ref.dtype
      )

      # di: [block_q, 128]
      # do: [block_q, head_dim]
      # v: [block_k_major, head_dim]
      dp = lax.dot_general(
          do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )
      ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

      if sm_scale != 1.0:
        ds = ds * sm_scale

      # ds: [block_q_major, block_k_major]
      # q: [block_q_major, head_dim]
      dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
      dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(
          dk_scratch_ref.dtype
      )
    lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

  @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
  def end_of_q_sequence():
    dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
    dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
  _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

  # kv index needs to be before q index since q index is the contractng
  # dimension.
  grid = (
      batch_size,
      num_heads,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
  )

  def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    if causal:
      # If the q block is skipped, stay at the 0th q block.
      next_q_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          q_seq_index,
          0,
      )
    else:
      next_q_index = q_seq_index

    return (batch_index, head_index, next_q_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  assert qo_spec.block_shape is not None
  assert q.ndim == len(qo_spec.block_shape)
  do_spec = qo_spec
  assert do.ndim == len(qo_spec.block_shape)

  def kv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, _, q_seq_index):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index
    ):
      del head_index
      if causal:
        next_q_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            q_seq_index,
            0,
        )
      else:
        next_q_index = q_seq_index
      return (batch_index, next_q_index, 0)

    def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
      del head_index
      return (batch_index, 0, kv_seq_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           k.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           v.dtype),
  ]
  def dkv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
  out_specs = [dkv_spec, dkv_spec]
  scratch_shapes = [
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      block_q=block_q,  # type: ignore
      block_k=block_k,  # type: ignore
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      q_seq_len=q_seq_len,
  )
  name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
    assert dk.shape == k.shape
    assert dv.shape == v.shape
  return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
  _, _, block_k_major, _ = k_tile_ref.shape
  _, _, block_q_major, _ = q_tile_ref.shape

  kv_seq_index = pl.program_id(axis=3)
  q_seq_index = pl.program_id(axis=2)

  @pl.when(kv_seq_index == 0)
  def start_new_sequence():
    dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

  def body(i, _):
    k_slice = pl.ds(i * block_k, block_k)
    q = q_tile_ref[0, 0, :, :]
    k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
    di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

    capped_logits = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )

    if ab_tile_ref is not None:
      ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(
          jnp.float32
      )
      capped_logits += ab

    if sm_scale != 1.0:
      capped_logits *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
      repeats, rem = divmod(block_k, NUM_LANES)
      if rem:
        raise NotImplementedError(
            f"kv block size must be a multiple of {NUM_LANES}"
        )
      q_segment_ids = pltpu.repeat(
          q_segment_ids_tile_ref[0], repeats, axis=1
      )  # [block_q, block_k].
      kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]  # [1, block_k].
      mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
      mask_shape = (block_q_major, block_k)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      row_ids += q_seq_index * block_q_major
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      col_ids += kv_seq_index * block_k_major + i * block_k
      causal_mask = col_ids <= row_ids
      mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    capped_logits = (
        capped_logits
        if mask is None
        else capped_logits + jnp.where(mask, 0.0, mask_value)
    )

    p = jnp.exp(
        capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
    )
    p = p * pltpu.repeat(
        1 / l, block_k // MIN_BLOCK_SIZE, axis=1
    )  # [block_q_major, block_k]

    # di: [block_q_major, 128]
    # do: [block_q_major, head_dim]
    # v: [block_k_major, head_dim]
    dp = jax.lax.dot_general(
        do,
        v,
        TRANS_B_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p
    # dp = jnp.dot(do, v.T)
    # ds = (dp - (dp * p).sum(axis=1)[:, None]) * p

    if sm_scale != 1.0:
      ds = ds * sm_scale

    if ds_tile_ref is not None:
      ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(
          ds_tile_ref.dtype
      )

    # dp: [block_q_major, block_k]
    # k: [block_k, head_dim]
    dq_scratch_ref[:, :] += lax.dot(
        ds.astype(k.dtype),
        k,
        preferred_element_type=jnp.float32,
    ).astype(dq_scratch_ref.dtype)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
    should_not_run = lax.select(should_run, False, True)
  else:
    should_run = True
    should_not_run = False  # type: ignore

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

  @pl.when(should_not_run)
  def zero_out_ds():
    if ds_tile_ref is not None:
      ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

  @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
  def end_of_kv_sequence():
    dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

  grid = (
      batch_size,
      num_heads,
      q_seq_len // block_q_major,
      kv_seq_len // block_k_major,
  )

  def qo_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  do_spec = qo_spec

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        # If the kv block is skipped, prefetch the next valid kv block, i.e. the
        # 0th one to be used for the next block_q rows.
        next_kv_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
  ]
  dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  out_specs = [
      dq_spec,
      dab_spec,
  ]
  scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      block_k=block_k,  # type: ignore
      kv_seq_len=kv_seq_len,
  )
  name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dq, ds = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

  # dab is just ds
  return dq, ds


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out


@functools.partial(
    jax.jit, static_argnames=["causal", "mask_value", "sm_scale"]
)
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
  return _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=False,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  return mha_reference_no_custom_vjp(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=save_residuals,
  )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  if save_residuals:
    raise NotImplementedError
  res = _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=True,
  )
  assert isinstance(res, tuple)
  out, l, m = res
  return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "sm_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
  if sm_scale != 1.0:
    raise NotImplementedError

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  if ab is not None:
    logits += ab

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  unnormalized = jnp.exp(logits - m[..., None])
  p = unnormalized / l[..., None]
  dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

  dp = jnp.einsum(
      "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
      ..., None
  ]  # [batch_size, num_heads, q_seq_len]

  ds = (dp - di) * p
  dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

  # dab is just ds
  dab = ds if ab is not None else None
  return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
  del save_residuals
  q, k, v, ab, segment_ids, o, l, m = residuals
  dq, dk, dv, dab = mha_reference_bwd(
      q,
      k,
      v,
      ab,
      segment_ids,
      o,
      l,
      m,
      do,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
  )
  return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
  if block > dim:
    raise ValueError(
        f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
    )
  if should_divide and dim % block != 0:
    raise ValueError(
        f"{dim_name}={dim} should be divisible by {block_name}={block}"
    )


CONFIG = {
    \'name\': \'pallas_flash_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 32,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'block_q\': 2048, \'block_k_major\': 2048, \'block_k\': 512, \'block_b\': 1, \'block_q_major_dkv\': 128, \'block_k_major_dkv\': 128, \'block_k_dkv\': 128, \'block_q_dkv\': 128, \'block_k_major_dq\': 128, \'block_k_dq\': 128, \'block_q_dq\': 128}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG[\'batch\']
    H = CONFIG[\'num_heads\']
    S = CONFIG[\'seq_len\']
    D = CONFIG[\'head_dim\']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS[\'block_q\'],
        block_k_major=TUNED_PARAMS[\'block_k_major\'],
        block_k=TUNED_PARAMS[\'block_k\'],
        block_b=TUNED_PARAMS[\'block_b\'],
        block_q_major_dkv=TUNED_PARAMS[\'block_q_major_dkv\'],
        block_k_major_dkv=TUNED_PARAMS[\'block_k_major_dkv\'],
        block_k_dkv=TUNED_PARAMS[\'block_k_dkv\'],
        block_q_dkv=TUNED_PARAMS[\'block_q_dkv\'],
        block_k_major_dq=TUNED_PARAMS[\'block_k_major_dq\'],
        block_k_dq=TUNED_PARAMS[\'block_k_dq\'],
        block_q_dq=TUNED_PARAMS[\'block_q_dq\'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.332,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.332 ms\n{"correct": true, "latency": 0.332, "error": "", "all_times_ms": [0.327, 0.328, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.332, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.334, 0.334, 0.334, 0.334, 0.334, 0.334, 0.334, 0.335, 0.335, 0.335, 0.336, 0.336, 0.336, 0.336, 0.337, 0.337, 0.337, 0.337, 0.338, 0.338, 0.338, 0.338, 0.338, 0.338, 0.338, 0.339, 0.339, 0.339, 0.339, 0.34, 0.34, 0.34, 0.34, 0.343, 0.364], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr=''),
plan='''**New strategy:** **causal wavefront microtiling** for the forward full-sequence path.

### Why this fits this code on v6e-1
Your tuned case is:

- `B=1`
- `H=32`
- `S=2048`
- `D=128`
- `causal=True`
- `block_q=2048`
- `block_k_major=2048`
- `block_k=512`

So each head runs as **one big full-sequence Q tile** against **4 K/V subtiles** of width 512.

That is good for reducing grid overhead, but it hides a big inefficiency in the **causal** case:

- with `S=2048` and `block_k=512`, the logical causal attention matrix is 4×4 in K/Q tiles
- only the **lower-triangular** tiles are needed
- total useful Q×K subtiles = `1 + 2 + 3 + 4 = 10`
- current kernel effectively does work equivalent to **16** subtiles

So about **37.5% of the subtile work is structurally useless** in the causal benchmark shape.  
The mask zeros it out after the fact, but the kernel still pays for:
- QK matmul
- softmax update
- PV matmul
- causal mask generation

## What to change

Add a **specialized internal fast path** inside `_flash_attention_impl` for the common case:

- `causal=True`
- `block_b == 1`
- `block_q == q_seq_len`
- `block_k_major == kv_seq_len`
- `ab is None`
- `segment_ids is None`

Keep the public API exactly the same: `flash_attention(...)`.

### Core rewrite
Instead of processing the whole `block_q=2048` tile at once, process it as **Q microtiles** aligned to `block_k`:

- choose `q_micro = block_k = 512`
- loop over 4 query microtiles: rows `[0:512]`, `[512:1024]`, `[1024:1536]`, `[1536:2048]`
- for query microtile `qt`, only run K tiles `0..qt`

That gives a wavefront schedule:

- Q tile 0 → K tiles 0
- Q tile 1 → K tiles 0,1
- Q tile 2 → K tiles 0,1,2
- Q tile 3 → K tiles 0,1,2,3

### Important extra optimization
Only the **diagonal** subtile needs a triangular mask.

For `kt < qt`, the whole subtile is already strictly below the diagonal, so **no causal mask computation is needed at all**.

That means in the 10 executed subtiles:

- 6 are mask-free
- 4 are diagonal masked
- 6 upper-triangular subtiles are skipped entirely

## Concretely

Add a helper like:

- `_flash_attention_impl_fullseq_causal_wavefront(...)`
- `_flash_attention_kernel_fullseq_causal_wavefront(...)`

and dispatch to it from `_flash_attention_impl(...)` under the condition above.

### Kernel structure
Keep `q/k/v` as full-sequence refs, but inside the kernel:

1. Slice one `q_micro` tile from `q_ref`
2. Initialize local online-softmax state for that tile only:
   - `m_scratch`: `(q_micro, 128)` fp32
   - `l_scratch`: `(q_micro, 128)` fp32
   - `acc_scratch`: `(q_micro, head_dim)` fp32
3. Iterate only over needed K tiles for that query tile
4. For strict-lower K tiles, run unmasked QK / PV
5. For the diagonal K tile, apply the causal mask
6. Normalize and store the output slice into `o_ref`
7. If `save_residuals=True`, store corresponding `l` and `m` slices too

## Why it should be faster

### 1. Less total math
For the benchmark shape:

- old effective subtile count: **16**
- new subtile count: **10**

So QK / softmax / PV work drops by about **37.5%** in the causal path.

### 2. Much less mask work
Current code builds causal masking logic for every K chunk.  
With wavefront microtiling, only the diagonal tile of each Q microtile needs masking.

### 3. Smaller live temporaries
Current forward path operates on very large score tiles like `2048 x 512`.  
The microtiled path uses `512 x 512`, which is much friendlier to v6e VMEM and reduces spill risk.

### 4. Smaller scratch state
Current scratch spans the full query tile:
- `m/l`: `(2048, 128)`
- `acc`: `(2048, 128)`

New scratch only spans one microtile:
- `m/l`: `(512, 128)`
- `acc`: `(512, 128)`

That cuts scratch footprint by **4x** for the forward state.

## Why this is safe numerically
You are not changing the math, only the traversal order.

Use the same online-softmax recurrence as today, still in `float32`:

- QK with `preferred_element_type=jnp.float32`
- softmax stats in fp32
- accumulation in fp32
- cast back to output dtype only on final store

That preserves semantic equivalence within the same small tolerance as the original kernel.

## TPU-specific details to preserve
When you implement this specialized path:

- keep `q_micro` a multiple of 8; `512` is good
- keep `block_k=512` and `head_dim=128`, so the last two dims remain TPU-friendly
- read refs explicitly (`q_ref[...]`, `k_ref[...]`, etc.)
- store with explicit dtype cast:
  - `o_ref[...] = acc.astype(o_ref.dtype)`
- keep reductions on verified tile shapes only
- use `int32` for any `jnp.arange` used in the diagonal-mask tile

## Minimal dispatch logic
Inside `_flash_attention_impl`, do something like:

- if full-sequence causal benchmark case → call new wavefront kernel
- else → keep existing generic kernel unchanged

That keeps the code semantically identical while targeting the hot path that dominates your measured latency.

---

## Expected impact
This is a strong phase-3 change because it reduces **actual work**, not just overhead:

- skips fully-masked upper-triangular causal blocks
- skips unnecessary mask construction on strict-lower blocks
- shrinks live state to better fit v6e VMEM

For this exact workload, this should be one of the highest-leverage forward-path optimizations.''',
code='''# Copyright 2023 The JAX Authors.
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

"""Flash Attention TPU kernel."""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

  q: jax.Array  # [batch_size, q_seq_len]
  kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tile sizes parameterizing FlashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.
  """
  block_q: int
  block_k_major: int
  block_k: int
  block_b: int

  block_q_major_dkv: int | None = None
  block_k_major_dkv: int | None = None
  block_k_dkv: int | None = None
  block_q_dkv: int | None = None

  block_k_major_dq: int | None = None
  block_k_dq: int | None = None
  block_q_dq: int | None = None

  def __post_init__(self):
    def verify_major_minor(prefix, suffix, major, minor):
      if minor > major:
        raise ValueError(
            f"{prefix}{suffix}={minor} should be smaller than"
            f" {prefix}_major{suffix}={major}"
        )
      if major % minor != 0:
        raise ValueError(
            f"{prefix}{suffix}={minor} should divide"
            f" {prefix}_major{suffix}={major}"
        )

    verify_major_minor("block_k", "", self.block_k_major, self.block_k)
    if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
      verify_major_minor(
          "block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv
      )
    if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
      verify_major_minor(
          "block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv
      )
    if self.block_k_major_dq is not None and self.block_k_dq is not None:
      verify_major_minor(
          "block_k", "_dq", self.block_k_major_dq, self.block_k_dq
      )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_major_dkv,
        self.block_k_major_dkv,
        self.block_q_dkv,
        self.block_k_dkv,
        self.block_k_major_dq,
        self.block_k_dq,
        self.block_q_dq,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
    return BlockSizes(
        block_q=128,
        block_k_major=128,
        block_k=128,
        block_b=1,
        block_q_major_dkv=128,
        block_k_major_dkv=128,
        block_k_dkv=128,
        block_q_dkv=128,
        block_k_major_dq=128,
        block_k_dq=128,
        block_q_dq=128,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, d_model = q.shape
  batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
  batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
  if batch_size != batch_size_k or batch_size != batch_size_v:
    raise ValueError(
        f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
        f" {batch_size_v} (for q, k, v respectively)"
    )
  if num_heads != num_heads_k or num_heads != num_heads_v:
    raise ValueError(
        f"Head count mismatch: got {num_heads}, {num_heads_k},"
        f" {num_heads_v} (for q, k, v respectively)"
    )
  if d_model != d_model_k:
    raise ValueError(
        f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k"
        " respectively)"
    )
  if d_model != d_model_v:
    raise NotImplementedError(
        "V model dimension unequal to KV model dimension unsupported"
    )
  if kv_seq_len != kv_seq_len_v:
    raise ValueError(
        f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}"
    )
  if ab is not None:
    if ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
      raise ValueError(
          f"Attention bias shape mismatch: expected ({batch_size=},"
          f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
      )
  if segment_ids is not None:
    if segment_ids.q.shape != (batch_size, q_seq_len):
      raise ValueError(
          f"Q segment ids shape mismatch: expected ({batch_size=},"
          f" {q_seq_len=},), got {segment_ids.q.shape}"
      )
    if segment_ids.kv.shape != (batch_size, kv_seq_len):
      raise ValueError(
          f"KV segment ids shape mismatch: expected ({batch_size=},"
          f" {kv_seq_len=},), got {segment_ids.kv.shape}"
      )
  if block_sizes is None:
    block_sizes = BlockSizes.get_default(
        batch_size, num_heads, q_seq_len, kv_seq_len, d_model
    )
  return _flash_attention(
      q, k, v, ab, segment_ids, False, causal, sm_scale, block_sizes, debug
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  return _flash_attention_impl(
      q,
      k,
      v,
      ab,
      segment_ids,
      save_residuals,
      causal,
      sm_scale,
      block_sizes.block_b,
      block_sizes.block_q,
      block_sizes.block_k_major,
      block_sizes.block_k,
      debug,
  )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  o, l, m = _flash_attention(
      q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug
  )
  return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    residuals,
    do,
):
  """VJP rule for FlashAttention."""
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  (q, k, v, ab, segment_ids, o, l, m) = residuals
  if not block_sizes.has_backward_blocks:
    raise ValueError(
        "Program is being differentiated, but not all backward blocks are"
        " specified"
    )

  di = jnp.sum(
      o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
  )  # [batch_size, num_heads, q_seq_len]

  dk, dv = _flash_attention_bwd_dkv(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_major_dkv,
      block_k_major=block_sizes.block_k_major_dkv,
      block_k=block_sizes.block_k_dkv,
      block_q=block_sizes.block_q_dkv,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )

  dq, ds = _flash_attention_bwd_dq(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_dq,
      block_k_major=block_sizes.block_k_major_dq,
      block_k=block_sizes.block_k_dq,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )
  return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
  # A block is considered below or on diagonal as long as the bottom left
  # corner of the block is below or on diagonal.
  return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
  block_b = q_tile_ref.shape[0]
  # If we\'re not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[-1]

  kv_seq_idx = pl.program_id(3)
  @pl.when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[batch_idx] = jnp.full(
        m_scratch_ref.shape[2:], -jnp.inf, jnp.float32
    )
    l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
    acc_scratch_ref[batch_idx] = jnp.zeros(
        acc_scratch_ref.shape[2:], jnp.float32
    )

  q_seq_idx = pl.program_id(2)
  if causal:
    should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    @pl.loop(0, block_k_major // block_k, unroll=True)
    def _body(i):
      m_prev = m_scratch_ref[batch_idx]
      l_prev = l_scratch_ref[batch_idx]
      acc_prev = acc_scratch_ref[batch_idx]
      q = q_tile_ref[batch_idx]  # [block_q, head_dim]
      start_k = i * block_k
      k = k_tile_ref[
          (*batch_idx, pl.dslice(start_k, block_k), slice(None))
      ]  # [block_k, head_dim]

      s = jax.lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q, block_k]

      # Add attention bias if needed.
      # TODO(tanburn) Should the attention bias be added before or after
      # multiplication by sm_scale?
      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
        ].astype(jnp.float32)
        s += ab

      if sm_scale != 1.0:
        s *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
              f"kv block size must be a multiple of {NUM_LANES}"
          )
        q_segment_ids = pltpu.repeat(
            q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            batch_idx[0], :1, pl.dslice(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_idx * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

      m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
      m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

      block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
      if rem:
        raise NotImplementedError(
            f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
        )
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

      # Unnormalized online softmax: track Acc (unnormalized) and L separately
      alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

      l_curr = jnp.sum(p, axis=1)[:, None]  # Shape [block_q, 1]
      l_next = l_prev * alpha + l_curr  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      alpha_broadcast = lambda a: pltpu.repeat(a, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          alpha_broadcast = lambda a: a[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      # Rescale previous accumulator and add new contribution
      acc_rescaled = acc_prev * alpha_broadcast(alpha)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      acc_next = acc_rescaled + o_curr

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next
      acc_scratch_ref[batch_idx] = acc_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Final normalization: O = Acc / L
    l_final = l_scratch_ref[batch_idx]
    acc_final = acc_scratch_ref[batch_idx]
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda l: l[:, :head_dim]
      else:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )
    l_final_safe = jnp.where(l_final == 0.0, 1.0, l_final)
    o_tile_ref[batch_idx] = (acc_final / l_broadcast(l_final_safe)).astype(o_tile_ref.dtype)
    if l_ref is not None:
      l_ref[batch_idx] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale != 1.0:
    s *= sm_scale

  mask = None
  if q_segment_ids_tile_ref is not None:
    repeats, rem = divmod(block_k, NUM_LANES)
    if rem:
      raise NotImplementedError(
          f"kv block size must be a multiple of {NUM_LANES}"
      )
    q_segment_ids = q_segment_ids_tile_ref[
        batch_idx[0]
    ]  # [block_q, NUM_LANES].
    q_segment_ids = pltpu.repeat(
        q_segment_ids, repeats, axis=1
    )  # [block_q, block_k].
    kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
    mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

  if causal:
    q_seq_idx = pl.program_id(2)
    mask_shape = (block_q, block_k)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    row_ids += q_seq_idx * block_q
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = col_ids <= row_ids
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

  m = jnp.max(s, axis=1)[:, None]
  p = jnp.exp(s - m)
  l = jnp.sum(p, axis=1)[:, None]
  p /= l

  if m_ref is not None:
    m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
  if l_ref is not None:
    l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

  v = v_tile_ref[batch_idx]
  o_tile_ref[batch_idx] = jax.lax.dot(
      p.astype(v.dtype), v, preferred_element_type=jnp.float32
  ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
  body_cost = pl.estimate_cost(
    mha_reference,
    q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
  )
  input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
  output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
  return pl.CostEstimate(
      flops=body_cost.flops,
      transcendentals=body_cost.transcendentals,
      bytes_accessed=input_bytes + output_bytes,
  )


def _flash_attention_kernel_fullseq_causal_wavefront(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    *,
    sm_scale: float,
    block_k: int,
    q_seq_len: int,
    head_dim: int,
    mask_value: float,
):
  """Specialized kernel for full-sequence causal attention with wavefront microtiling."""
  num_q_tiles = q_seq_len // block_k
  
  for qt in range(num_q_tiles):
    q_start = qt * block_k
    q_tile = q_ref[0, 0, q_start:q_start + block_k, :]  # [block_k, head_dim]
    
    m_scratch = jnp.full((block_k, MIN_BLOCK_SIZE), -jnp.inf, jnp.float32)
    l_scratch = jnp.zeros((block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = jnp.zeros((block_k, head_dim), jnp.float32)
    
    for kt in range(qt + 1):
      k_start = kt * block_k
      k_tile = k_ref[0, 0, k_start:k_start + block_k, :]  # [block_k, head_dim]
      v_tile = v_ref[0, 0, k_start:k_start + block_k, :]  # [block_k, head_dim]
      
      s = jax.lax.dot_general(
          q_tile, k_tile, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_k, block_k]
      
      if sm_scale != 1.0:
        s = s * sm_scale
      
      if kt == qt:
        mask_shape = (block_k, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids = row_ids + q_start
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids = col_ids + k_start
        causal_mask = col_ids <= row_ids
        s = s + jnp.where(causal_mask, 0.0, mask_value)
      
      m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
      m_next = jnp.maximum(m_scratch, m_curr)  # [block_k, MIN_BLOCK_SIZE]
      
      block_k_repeats = block_k // MIN_BLOCK_SIZE
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))  # [block_k, block_k]
      
      alpha = jnp.exp(m_scratch - m_next)  # [block_k, MIN_BLOCK_SIZE]
      
      l_curr = jnp.sum(p, axis=1)[:, None]  # [block_k, 1]
      l_next = l_scratch * alpha + l_curr  # [block_k, MIN_BLOCK_SIZE]
      
      head_dim_repeats = head_dim // MIN_BLOCK_SIZE
      acc_rescaled = acc_scratch * pltpu.repeat(alpha, head_dim_repeats, 1)
      o_curr = jax.lax.dot(
          p.astype(v_tile.dtype), v_tile, preferred_element_type=jnp.float32
      )
      acc_scratch = acc_rescaled + o_curr
      
      l_scratch = l_next
      m_scratch = m_next
    
    l_final_safe = jnp.where(l_scratch == 0.0, 1.0, l_scratch)
    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    o_normalized = acc_scratch / pltpu.repeat(l_final_safe, head_dim_repeats, 1)
    o_ref[0, 0, q_start:q_start + block_k, :] = o_normalized.astype(o_ref.dtype)
    
    if l_ref is not None:
      l_ref[0, 0, q_start:q_start + block_k, :] = l_scratch.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, q_start:q_start + block_k, :] = m_scratch.astype(m_ref.dtype)


def _flash_attention_impl_fullseq_causal_wavefront(
    q,
    k,
    v,
    save_residuals,
    sm_scale,
    block_k,
    debug,
):
  """Specialized implementation for full-sequence causal attention with wavefront microtiling."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  
  grid = (batch_size, num_heads)
  
  def index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)
  
  full_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), index_map)
  
  kernel = functools.partial(
      _flash_attention_kernel_fullseq_causal_wavefront,
      sm_scale=sm_scale,
      block_k=block_k,
      q_seq_len=q_seq_len,
      head_dim=head_dim,
      mask_value=DEFAULT_MASK_VALUE,
  )
  
  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [full_spec]
  
  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), index_map)
    out_specs = [full_spec, lm_spec, lm_spec]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l, m]
  else:
    out_specs = [full_spec, None, None]
    out_shape = [out_shape[0], None, None]
  
  in_specs = [full_spec, full_spec, full_spec]
  
  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[],
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)
  
  if save_residuals:
    l, m = aux[-2:]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
  _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
  _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

  use_wavefront = (
      causal
      and block_b == 1
      and block_q == q_seq_len
      and block_k_major == kv_seq_len
      and ab is None
      and segment_ids is None
      and q_seq_len == kv_seq_len
      and q_seq_len % block_k == 0
      and head_dim % MIN_BLOCK_SIZE == 0
      and block_k % MIN_BLOCK_SIZE == 0
  )
  
  if use_wavefront:
    return _flash_attention_impl_fullseq_causal_wavefront(
        q, k, v, save_residuals, sm_scale, block_k, debug
    )

  # TODO(apaszke): Tile over heads as well.
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      should_run = below_or_on_diag(
          q_seq_index, block_q, kv_seq_index, block_k_major
      )
      # If the ab block is skipped, prefetch the next valid ab block, i.e. the
      # 0th kv to be used for the next block_q rows.
      next_q_index = lax.select(
          should_run,
          q_seq_index,
          lax.select(
              q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
          ),
      )
      next_kv_index = lax.select(should_run, kv_seq_index, 0)
    else:
      next_q_index = q_seq_index
      next_kv_index = kv_seq_index

    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        next_kv_index = lax.select(
            below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (block_b, block_q, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      cost_estimate=_fwd_cost_estimate(
          q,
          k,
          v,
          ab,
          segment_ids,
          causal=causal,
          sm_scale=sm_scale,
          kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
          kernel_outputs_specs=out_shape,
      ),
  )(q, k, v, ab, q_segment_ids, kv_segment_ids)
  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
  _, _, block_q_major, _ = q_tile_ref.shape
  _, _, block_k_major, _ = k_tile_ref.shape

  q_seq_index = pl.program_id(axis=3)
  kv_seq_index = pl.program_id(axis=2)

  @pl.when(q_seq_index == 0)
  def start_new_sequence():
    dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
    dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

  def q_body(j, _):
    start_q = j * block_q
    def k_body(i, _):
      start_k = i * block_k
      k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
      l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(
          jnp.float32
      )  # [block_q, 128]

      capped_logits = lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q_major, block_k]

      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            0,
            0,
            pl.dslice(j * block_q, block_q),
            pl.dslice(i * block_k, block_k),
        ].astype(jnp.float32)
        capped_logits += ab

      if sm_scale != 1.0:
        capped_logits *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
          )
        q_segment_ids = q_segment_ids_tile_ref[
            0, pl.ds(start_q, block_q), :
        ]  # [block_q, NUM_LANES].
        q_segment_ids = pltpu.repeat(
            q_segment_ids, repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            :, 0, pl.ds(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_index * block_q_major + start_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_index * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      capped_logits = (
          capped_logits
          if mask is None
          else capped_logits + jnp.where(mask, 0.0, mask_value)
      )

      p = jnp.exp(
          capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
      )
      p = p * pltpu.repeat(
          1 / l, block_k // MIN_BLOCK_SIZE, axis=1
      )  # [block_q_major, block_k_major]
      dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
      dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(
          dv_scratch_ref.dtype
      )

      # di: [block_q, 128]
      # do: [block_q, head_dim]
      # v: [block_k_major, head_dim]
      dp = lax.dot_general(
          do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )
      ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

      if sm_scale != 1.0:
        ds = ds * sm_scale

      # ds: [block_q_major, block_k_major]
      # q: [block_q_major, head_dim]
      dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
      dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(
          dk_scratch_ref.dtype
      )
    lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

  @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
  def end_of_q_sequence():
    dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
    dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
  _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

  # kv index needs to be before q index since q index is the contractng
  # dimension.
  grid = (
      batch_size,
      num_heads,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
  )

  def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    if causal:
      # If the q block is skipped, stay at the 0th q block.
      next_q_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          q_seq_index,
          0,
      )
    else:
      next_q_index = q_seq_index

    return (batch_index, head_index, next_q_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  assert qo_spec.block_shape is not None
  assert q.ndim == len(qo_spec.block_shape)
  do_spec = qo_spec
  assert do.ndim == len(qo_spec.block_shape)

  def kv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, _, q_seq_index):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index
    ):
      del head_index
      if causal:
        next_q_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            q_seq_index,
            0,
        )
      else:
        next_q_index = q_seq_index
      return (batch_index, next_q_index, 0)

    def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
      del head_index
      return (batch_index, 0, kv_seq_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           k.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           v.dtype),
  ]
  def dkv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
  out_specs = [dkv_spec, dkv_spec]
  scratch_shapes = [
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      block_q=block_q,  # type: ignore
      block_k=block_k,  # type: ignore
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      q_seq_len=q_seq_len,
  )
  name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
    assert dk.shape == k.shape
    assert dv.shape == v.shape
  return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
  _, _, block_k_major, _ = k_tile_ref.shape
  _, _, block_q_major, _ = q_tile_ref.shape

  kv_seq_index = pl.program_id(axis=3)
  q_seq_index = pl.program_id(axis=2)

  @pl.when(kv_seq_index == 0)
  def start_new_sequence():
    dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

  def body(i, _):
    k_slice = pl.ds(i * block_k, block_k)
    q = q_tile_ref[0, 0, :, :]
    k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
    di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

    capped_logits = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )

    if ab_tile_ref is not None:
      ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(
          jnp.float32
      )
      capped_logits += ab

    if sm_scale != 1.0:
      capped_logits *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
      repeats, rem = divmod(block_k, NUM_LANES)
      if rem:
        raise NotImplementedError(
            f"kv block size must be a multiple of {NUM_LANES}"
        )
      q_segment_ids = pltpu.repeat(
          q_segment_ids_tile_ref[0], repeats, axis=1
      )  # [block_q, block_k].
      kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]  # [1, block_k].
      mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
      mask_shape = (block_q_major, block_k)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      row_ids += q_seq_index * block_q_major
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      col_ids += kv_seq_index * block_k_major + i * block_k
      causal_mask = col_ids <= row_ids
      mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    capped_logits = (
        capped_logits
        if mask is None
        else capped_logits + jnp.where(mask, 0.0, mask_value)
    )

    p = jnp.exp(
        capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
    )
    p = p * pltpu.repeat(
        1 / l, block_k // MIN_BLOCK_SIZE, axis=1
    )  # [block_q_major, block_k]

    # di: [block_q_major, 128]
    # do: [block_q_major, head_dim]
    # v: [block_k_major, head_dim]
    dp = jax.lax.dot_general(
        do,
        v,
        TRANS_B_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p
    # dp = jnp.dot(do, v.T)
    # ds = (dp - (dp * p).sum(axis=1)[:, None]) * p

    if sm_scale != 1.0:
      ds = ds * sm_scale

    if ds_tile_ref is not None:
      ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(
          ds_tile_ref.dtype
      )

    # dp: [block_q_major, block_k]
    # k: [block_k, head_dim]
    dq_scratch_ref[:, :] += lax.dot(
        ds.astype(k.dtype),
        k,
        preferred_element_type=jnp.float32,
    ).astype(dq_scratch_ref.dtype)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
    should_not_run = lax.select(should_run, False, True)
  else:
    should_run = True
    should_not_run = False  # type: ignore

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

  @pl.when(should_not_run)
  def zero_out_ds():
    if ds_tile_ref is not None:
      ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

  @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
  def end_of_kv_sequence():
    dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

  grid = (
      batch_size,
      num_heads,
      q_seq_len // block_q_major,
      kv_seq_len // block_k_major,
  )

  def qo_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  do_spec = qo_spec

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        # If the kv block is skipped, prefetch the next valid kv block, i.e. the
        # 0th one to be used for the next block_q rows.
        next_kv_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
  ]
  dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  out_specs = [
      dq_spec,
      dab_spec,
  ]
  scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      block_k=block_k,  # type: ignore
      kv_seq_len=kv_seq_len,
  )
  name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dq, ds = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

  # dab is just ds
  return dq, ds


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out


@functools.partial(
    jax.jit, static_argnames=["causal", "mask_value", "sm_scale"]
)
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
  return _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=False,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  return mha_reference_no_custom_vjp(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=save_residuals,
  )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  if save_residuals:
    raise NotImplementedError
  res = _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=True,
  )
  assert isinstance(res, tuple)
  out, l, m = res
  return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "sm_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
  if sm_scale != 1.0:
    raise NotImplementedError

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  if ab is not None:
    logits += ab

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  unnormalized = jnp.exp(logits - m[..., None])
  p = unnormalized / l[..., None]
  dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

  dp = jnp.einsum(
      "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
      ..., None
  ]  # [batch_size, num_heads, q_seq_len]

  ds = (dp - di) * p
  dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

  # dab is just ds
  dab = ds if ab is not None else None
  return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
  del save_residuals
  q, k, v, ab, segment_ids, o, l, m = residuals
  dq, dk, dv, dab = mha_reference_bwd(
      q,
      k,
      v,
      ab,
      segment_ids,
      o,
      l,
      m,
      do,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
  )
  return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
  if block > dim:
    raise ValueError(
        f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
    )
  if should_divide and dim % block != 0:
    raise ValueError(
        f"{dim_name}={dim} should be divisible by {block_name}={block}"
    )


CONFIG = {
    \'name\': \'pallas_flash_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 32,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'block_q\': 2048, \'block_k_major\': 2048, \'block_k\': 512, \'block_b\': 1, \'block_q_major_dkv\': 128, \'block_k_major_dkv\': 128, \'block_k_dkv\': 128, \'block_q_dkv\': 128, \'block_k_major_dq\': 128, \'block_k_dq\': 128, \'block_q_dq\': 128}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG[\'batch\']
    H = CONFIG[\'num_heads\']
    S = CONFIG[\'seq_len\']
    D = CONFIG[\'head_dim\']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS[\'block_q\'],
        block_k_major=TUNED_PARAMS[\'block_k_major\'],
        block_k=TUNED_PARAMS[\'block_k\'],
        block_b=TUNED_PARAMS[\'block_b\'],
        block_q_major_dkv=TUNED_PARAMS[\'block_q_major_dkv\'],
        block_k_major_dkv=TUNED_PARAMS[\'block_k_major_dkv\'],
        block_k_dkv=TUNED_PARAMS[\'block_k_dkv\'],
        block_q_dkv=TUNED_PARAMS[\'block_q_dkv\'],
        block_k_major_dq=TUNED_PARAMS[\'block_k_major_dq\'],
        block_k_dq=TUNED_PARAMS[\'block_k_dq\'],
        block_q_dq=TUNED_PARAMS[\'block_q_dq\'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.271,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.271 ms\n{"correct": true, "latency": 0.271, "error": "", "all_times_ms": [0.265, 0.265, 0.265, 0.266, 0.266, 0.266, 0.267, 0.267, 0.267, 0.267, 0.267, 0.267, 0.268, 0.268, 0.268, 0.268, 0.268, 0.268, 0.268, 0.268, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.269, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.271, 0.272, 0.272, 0.272, 0.273, 0.273, 0.273, 0.273, 0.273, 0.273, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.274, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.276, 0.276, 0.276, 0.276, 0.276, 0.276, 0.279, 0.279, 0.279, 0.28, 0.28, 0.283, 0.283, 0.285, 0.298], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr=''),
plan='''A good next strategy for this code on **TPU v6e-1** is:

## Strategy: coarsen the fast path over the **head axis** (`block_h > 1`)

This is different from the listed strategies. The idea is to make each kernel invocation process **multiple heads at once** in the specialized full-sequence causal wavefront path, instead of launching one kernel program per head.

### Why this helps here

For your workload:

- `batch = 1`
- `num_heads = 32`
- `q_seq_len = kv_seq_len = 2048`
- `head_dim = 128`
- `causal = True`
- tuned forward path hits `use_wavefront == True`

So today `_flash_attention_impl_fullseq_causal_wavefront` runs with:

- `grid = (batch_size, num_heads) = (1, 32)`
- one program handles exactly one head

On **v6e-1**, there is only **one TensorCore per chip**, so those 32 programs are not buying you intra-chip head parallelism. They mostly add:

- more kernel prologue/epilogue overhead
- more DMA/block-setup overhead
- more repeated scratch initialization
- more grid/control overhead

Since heads are completely independent, you can safely batch a few of them together per program.

---

## What to change

Only change the **full-sequence causal wavefront fast path** used by this workload.

### 1. Add a small head tile, e.g. `block_h = 2`

Use `block_h = 2` on v6e as a safe starting point.

Why 2:

- one head of `[2048, 128]` bf16 is `2048 * 128 * 2 = 512 KiB`
- for `q/k/v/o`, two heads is about `4 MiB` total live tensor data
- even with TPU buffering and scratch, this stays comfortably under v6e VMEM limits

I would **not** start with 4 here unless profiling confirms no VMEM spill.

---

### 2. Change the fast-path grid/spec from one-head-per-program to two-heads-per-program

Today:

```python
grid = (batch_size, num_heads)

def index_map(batch_index, head_index):
  return (batch_index, head_index, 0, 0)

full_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), index_map)
```

Change to:

```python
block_h = 2
grid = (batch_size, num_heads // block_h)

def index_map(batch_index, head_block_index):
  return (batch_index, head_block_index, 0, 0)

full_spec = pl.BlockSpec((1, block_h, q_seq_len, head_dim), index_map)
```

This keeps the same public API and output shape. It just makes each program own a block of 2 heads.

---

### 3. Rewrite the wavefront kernel to loop over the heads inside the tile

Instead of assuming `shape[1] == 1`, process `q_ref.shape[1]` heads.

Sketch:

```python
def _flash_attention_kernel_fullseq_causal_wavefront_multihead(
    q_ref, k_ref, v_ref, o_ref, l_ref, m_ref, *,
    sm_scale, block_k, q_seq_len, head_dim, mask_value,
):
  num_heads_in_tile = q_ref.shape[1]
  num_q_tiles = q_seq_len // block_k

  for h in range(num_heads_in_tile):
    for qt in range(num_q_tiles):
      q_start = qt * block_k
      q_tile = q_ref[0, h, q_start:q_start + block_k, :]

      m_scratch = jnp.full((block_k, MIN_BLOCK_SIZE), -jnp.inf, jnp.float32)
      l_scratch = jnp.zeros((block_k, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = jnp.zeros((block_k, head_dim), jnp.float32)

      for kt in range(qt + 1):
        k_start = kt * block_k
        k_tile = k_ref[0, h, k_start:k_start + block_k, :]
        v_tile = v_ref[0, h, k_start:k_start + block_k, :]

        s = jax.lax.dot_general(
            q_tile, k_tile, TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        if sm_scale != 1.0:
          s = s * sm_scale

        if kt == qt:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)

        # same online-softmax update as today, in float32
        ...
        o_ref[0, h, q_start:q_start + block_k, :] = out.astype(o_ref.dtype)
```

Important points:

- still read Refs explicitly with indexing
- still accumulate scores/softmax in `float32`
- still write output with `astype(o_ref.dtype)`
- do **not** change the math order within a head

So numerics stay effectively the same.

---

### 4. Use this only in the inference fast path

Because your current workload does **not** save residuals, keep this optimization gated:

```python
if use_wavefront and not save_residuals and num_heads % 2 == 0:
  return _flash_attention_impl_fullseq_causal_wavefront_multihead(...)
```

For `save_residuals=True`, keep the current one-head path for now. That avoids growing `l/m` output tiles and keeps VMEM pressure low.

---

## Why this should reduce latency

For this exact config, it cuts the number of wavefront programs from:

- **32 → 16** with `block_h = 2`

That means less repeated:

- program setup
- HBM window/block mapping work
- scratch init/control work
- output commit overhead

And because `[B, H, S, D]` is laid out with `S, D` contiguous, a `(1, 2, 2048, 128)` block is still a clean, large contiguous HBM slice, which TPUs handle efficiently.

Since v6e-1 has a single TensorCore, this is a better granularity choice than one-program-per-head.

---

## Expected impact

This won’t change total FLOPs, but it should lower overhead around the FLOPs. For your tuned full-sequence causal workload, this is one of the cleaner ways to improve latency without changing the algorithm.

If I were implementing phase 5, I’d do exactly this:

- add `block_h = 2`
- specialize only the wavefront forward inference path
- leave backward and residual-saving paths unchanged for now

If you want, I can write the exact patched version of `_flash_attention_impl_fullseq_causal_wavefront` and its kernel.''',
code='''# Copyright 2023 The JAX Authors.
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

"""Flash Attention TPU kernel."""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

  q: jax.Array  # [batch_size, q_seq_len]
  kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tile sizes parameterizing FlashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.
  """
  block_q: int
  block_k_major: int
  block_k: int
  block_b: int

  block_q_major_dkv: int | None = None
  block_k_major_dkv: int | None = None
  block_k_dkv: int | None = None
  block_q_dkv: int | None = None

  block_k_major_dq: int | None = None
  block_k_dq: int | None = None
  block_q_dq: int | None = None

  def __post_init__(self):
    def verify_major_minor(prefix, suffix, major, minor):
      if minor > major:
        raise ValueError(
            f"{prefix}{suffix}={minor} should be smaller than"
            f" {prefix}_major{suffix}={major}"
        )
      if major % minor != 0:
        raise ValueError(
            f"{prefix}{suffix}={minor} should divide"
            f" {prefix}_major{suffix}={major}"
        )

    verify_major_minor("block_k", "", self.block_k_major, self.block_k)
    if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
      verify_major_minor(
          "block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv
      )
    if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
      verify_major_minor(
          "block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv
      )
    if self.block_k_major_dq is not None and self.block_k_dq is not None:
      verify_major_minor(
          "block_k", "_dq", self.block_k_major_dq, self.block_k_dq
      )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_major_dkv,
        self.block_k_major_dkv,
        self.block_q_dkv,
        self.block_k_dkv,
        self.block_k_major_dq,
        self.block_k_dq,
        self.block_q_dq,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
    return BlockSizes(
        block_q=128,
        block_k_major=128,
        block_k=128,
        block_b=1,
        block_q_major_dkv=128,
        block_k_major_dkv=128,
        block_k_dkv=128,
        block_q_dkv=128,
        block_k_major_dq=128,
        block_k_dq=128,
        block_q_dq=128,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, d_model = q.shape
  batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
  batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
  if batch_size != batch_size_k or batch_size != batch_size_v:
    raise ValueError(
        f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
        f" {batch_size_v} (for q, k, v respectively)"
    )
  if num_heads != num_heads_k or num_heads != num_heads_v:
    raise ValueError(
        f"Head count mismatch: got {num_heads}, {num_heads_k},"
        f" {num_heads_v} (for q, k, v respectively)"
    )
  if d_model != d_model_k:
    raise ValueError(
        f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k"
        " respectively)"
    )
  if d_model != d_model_v:
    raise NotImplementedError(
        "V model dimension unequal to KV model dimension unsupported"
    )
  if kv_seq_len != kv_seq_len_v:
    raise ValueError(
        f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}"
    )
  if ab is not None:
    if ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
      raise ValueError(
          f"Attention bias shape mismatch: expected ({batch_size=},"
          f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
      )
  if segment_ids is not None:
    if segment_ids.q.shape != (batch_size, q_seq_len):
      raise ValueError(
          f"Q segment ids shape mismatch: expected ({batch_size=},"
          f" {q_seq_len=},), got {segment_ids.q.shape}"
      )
    if segment_ids.kv.shape != (batch_size, kv_seq_len):
      raise ValueError(
          f"KV segment ids shape mismatch: expected ({batch_size=},"
          f" {kv_seq_len=},), got {segment_ids.kv.shape}"
      )
  if block_sizes is None:
    block_sizes = BlockSizes.get_default(
        batch_size, num_heads, q_seq_len, kv_seq_len, d_model
    )
  return _flash_attention(
      q, k, v, ab, segment_ids, False, causal, sm_scale, block_sizes, debug
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  return _flash_attention_impl(
      q,
      k,
      v,
      ab,
      segment_ids,
      save_residuals,
      causal,
      sm_scale,
      block_sizes.block_b,
      block_sizes.block_q,
      block_sizes.block_k_major,
      block_sizes.block_k,
      debug,
  )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  o, l, m = _flash_attention(
      q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug
  )
  return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    residuals,
    do,
):
  """VJP rule for FlashAttention."""
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")
  (q, k, v, ab, segment_ids, o, l, m) = residuals
  if not block_sizes.has_backward_blocks:
    raise ValueError(
        "Program is being differentiated, but not all backward blocks are"
        " specified"
    )

  di = jnp.sum(
      o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
  )  # [batch_size, num_heads, q_seq_len]

  dk, dv = _flash_attention_bwd_dkv(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_major_dkv,
      block_k_major=block_sizes.block_k_major_dkv,
      block_k=block_sizes.block_k_dkv,
      block_q=block_sizes.block_q_dkv,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )

  dq, ds = _flash_attention_bwd_dq(
      q,
      k,
      v,
      ab,
      segment_ids,
      l,
      m,
      do,
      di,
      block_q_major=block_sizes.block_q_dq,
      block_k_major=block_sizes.block_k_major_dq,
      block_k=block_sizes.block_k_dq,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      debug=debug,
  )
  return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
  # A block is considered below or on diagonal as long as the bottom left
  # corner of the block is below or on diagonal.
  return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
  block_b = q_tile_ref.shape[0]
  # If we\'re not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[-1]

  kv_seq_idx = pl.program_id(3)
  @pl.when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[batch_idx] = jnp.full(
        m_scratch_ref.shape[2:], -jnp.inf, jnp.float32
    )
    l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
    acc_scratch_ref[batch_idx] = jnp.zeros(
        acc_scratch_ref.shape[2:], jnp.float32
    )

  q_seq_idx = pl.program_id(2)
  if causal:
    should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    @pl.loop(0, block_k_major // block_k, unroll=True)
    def _body(i):
      m_prev = m_scratch_ref[batch_idx]
      l_prev = l_scratch_ref[batch_idx]
      acc_prev = acc_scratch_ref[batch_idx]
      q = q_tile_ref[batch_idx]  # [block_q, head_dim]
      start_k = i * block_k
      k = k_tile_ref[
          (*batch_idx, pl.dslice(start_k, block_k), slice(None))
      ]  # [block_k, head_dim]

      s = jax.lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q, block_k]

      # Add attention bias if needed.
      # TODO(tanburn) Should the attention bias be added before or after
      # multiplication by sm_scale?
      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
        ].astype(jnp.float32)
        s += ab

      if sm_scale != 1.0:
        s *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
              f"kv block size must be a multiple of {NUM_LANES}"
          )
        q_segment_ids = pltpu.repeat(
            q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            batch_idx[0], :1, pl.dslice(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_idx * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

      m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
      m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

      block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
      if rem:
        raise NotImplementedError(
            f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
        )
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

      # Unnormalized online softmax: track Acc (unnormalized) and L separately
      alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

      l_curr = jnp.sum(p, axis=1)[:, None]  # Shape [block_q, 1]
      l_next = l_prev * alpha + l_curr  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      alpha_broadcast = lambda a: pltpu.repeat(a, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          alpha_broadcast = lambda a: a[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      # Rescale previous accumulator and add new contribution
      acc_rescaled = acc_prev * alpha_broadcast(alpha)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      acc_next = acc_rescaled + o_curr

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next
      acc_scratch_ref[batch_idx] = acc_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Final normalization: O = Acc / L
    l_final = l_scratch_ref[batch_idx]
    acc_final = acc_scratch_ref[batch_idx]
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda l: l[:, :head_dim]
      else:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )
    l_final_safe = jnp.where(l_final == 0.0, 1.0, l_final)
    o_tile_ref[batch_idx] = (acc_final / l_broadcast(l_final_safe)).astype(o_tile_ref.dtype)
    if l_ref is not None:
      l_ref[batch_idx] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  block_k_major = k_tile_ref.shape[2]
  block_q = q_tile_ref.shape[2]

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale != 1.0:
    s *= sm_scale

  mask = None
  if q_segment_ids_tile_ref is not None:
    repeats, rem = divmod(block_k, NUM_LANES)
    if rem:
      raise NotImplementedError(
          f"kv block size must be a multiple of {NUM_LANES}"
      )
    q_segment_ids = q_segment_ids_tile_ref[
        batch_idx[0]
    ]  # [block_q, NUM_LANES].
    q_segment_ids = pltpu.repeat(
        q_segment_ids, repeats, axis=1
    )  # [block_q, block_k].
    kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
    mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

  if causal:
    q_seq_idx = pl.program_id(2)
    mask_shape = (block_q, block_k)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    row_ids += q_seq_idx * block_q
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = col_ids <= row_ids
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

  m = jnp.max(s, axis=1)[:, None]
  p = jnp.exp(s - m)
  l = jnp.sum(p, axis=1)[:, None]
  p /= l

  if m_ref is not None:
    m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
  if l_ref is not None:
    l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

  v = v_tile_ref[batch_idx]
  o_tile_ref[batch_idx] = jax.lax.dot(
      p.astype(v.dtype), v, preferred_element_type=jnp.float32
  ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
  body_cost = pl.estimate_cost(
    mha_reference,
    q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
  )
  input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
  output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
  return pl.CostEstimate(
      flops=body_cost.flops,
      transcendentals=body_cost.transcendentals,
      bytes_accessed=input_bytes + output_bytes,
  )


def _flash_attention_kernel_fullseq_causal_wavefront(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    *,
    sm_scale: float,
    block_k: int,
    q_seq_len: int,
    head_dim: int,
    mask_value: float,
):
  """Specialized kernel for full-sequence causal attention with wavefront microtiling.
  
  This kernel processes multiple heads per program invocation (block_h heads).
  """
  num_q_tiles = q_seq_len // block_k
  num_heads_in_tile = q_ref.shape[1]  # block_h heads per program
  
  for h in range(num_heads_in_tile):
    for qt in range(num_q_tiles):
      q_start = qt * block_k
      q_tile = q_ref[0, h, q_start:q_start + block_k, :]  # [block_k, head_dim]
      
      m_scratch = jnp.full((block_k, MIN_BLOCK_SIZE), -jnp.inf, jnp.float32)
      l_scratch = jnp.zeros((block_k, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = jnp.zeros((block_k, head_dim), jnp.float32)
      
      for kt in range(qt + 1):
        k_start = kt * block_k
        k_tile = k_ref[0, h, k_start:k_start + block_k, :]  # [block_k, head_dim]
        v_tile = v_ref[0, h, k_start:k_start + block_k, :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q_tile, k_tile, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s = s * sm_scale
        
        if kt == qt:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids = row_ids + q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids = col_ids + k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_scratch, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats = block_k // MIN_BLOCK_SIZE
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))  # [block_k, block_k]
        
        alpha = jnp.exp(m_scratch - m_next)  # [block_k, MIN_BLOCK_SIZE]
        
        l_curr = jnp.sum(p, axis=1)[:, None]  # [block_k, 1]
        l_next = l_scratch * alpha + l_curr  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats = head_dim // MIN_BLOCK_SIZE
        acc_rescaled = acc_scratch * pltpu.repeat(alpha, head_dim_repeats, 1)
        o_curr = jax.lax.dot(
            p.astype(v_tile.dtype), v_tile, preferred_element_type=jnp.float32
        )
        acc_scratch = acc_rescaled + o_curr
        
        l_scratch = l_next
        m_scratch = m_next
      
      l_final_safe = jnp.where(l_scratch == 0.0, 1.0, l_scratch)
      head_dim_repeats = head_dim // MIN_BLOCK_SIZE
      o_normalized = acc_scratch / pltpu.repeat(l_final_safe, head_dim_repeats, 1)
      o_ref[0, h, q_start:q_start + block_k, :] = o_normalized.astype(o_ref.dtype)
      
      if l_ref is not None:
        l_ref[0, h, q_start:q_start + block_k, :] = l_scratch.astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[0, h, q_start:q_start + block_k, :] = m_scratch.astype(m_ref.dtype)


def _flash_attention_impl_fullseq_causal_wavefront(
    q,
    k,
    v,
    save_residuals,
    sm_scale,
    block_k,
    debug,
):
  """Specialized implementation for full-sequence causal attention with wavefront microtiling.
  
  This implementation processes multiple heads per kernel program (block_h=2) to reduce
  kernel launch overhead on single-TensorCore TPUs like v6e.
  """
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  
  # Use block_h=2 to reduce grid size and kernel overhead on v6e-1
  # Fall back to block_h=1 if num_heads is odd
  block_h = 2 if num_heads % 2 == 0 else 1
  
  grid = (batch_size, num_heads // block_h)
  
  def index_map(batch_index, head_block_index):
    return (batch_index, head_block_index, 0, 0)
  
  full_spec = pl.BlockSpec((1, block_h, q_seq_len, head_dim), index_map)
  
  kernel = functools.partial(
      _flash_attention_kernel_fullseq_causal_wavefront,
      sm_scale=sm_scale,
      block_k=block_k,
      q_seq_len=q_seq_len,
      head_dim=head_dim,
      mask_value=DEFAULT_MASK_VALUE,
  )
  
  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [full_spec]
  
  if save_residuals:
    lm_spec = pl.BlockSpec((1, block_h, q_seq_len, MIN_BLOCK_SIZE), index_map)
    out_specs = [full_spec, lm_spec, lm_spec]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l, m]
  else:
    out_specs = [full_spec, None, None]
    out_shape = [out_shape[0], None, None]
  
  in_specs = [full_spec, full_spec, full_spec]
  
  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[],
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)
  
  if save_residuals:
    l, m = aux[-2:]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
  _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
  _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

  use_wavefront = (
      causal
      and block_b == 1
      and block_q == q_seq_len
      and block_k_major == kv_seq_len
      and ab is None
      and segment_ids is None
      and q_seq_len == kv_seq_len
      and q_seq_len % block_k == 0
      and head_dim % MIN_BLOCK_SIZE == 0
      and block_k % MIN_BLOCK_SIZE == 0
  )
  
  if use_wavefront:
    return _flash_attention_impl_fullseq_causal_wavefront(
        q, k, v, save_residuals, sm_scale, block_k, debug
    )

  # TODO(apaszke): Tile over heads as well.
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      should_run = below_or_on_diag(
          q_seq_index, block_q, kv_seq_index, block_k_major
      )
      # If the ab block is skipped, prefetch the next valid ab block, i.e. the
      # 0th kv to be used for the next block_q rows.
      next_q_index = lax.select(
          should_run,
          q_seq_index,
          lax.select(
              q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
          ),
      )
      next_kv_index = lax.select(should_run, kv_seq_index, 0)
    else:
      next_q_index = q_seq_index
      next_kv_index = kv_seq_index

    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      mask_value=DEFAULT_MASK_VALUE,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        next_kv_index = lax.select(
            below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (block_b, block_q, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      cost_estimate=_fwd_cost_estimate(
          q,
          k,
          v,
          ab,
          segment_ids,
          causal=causal,
          sm_scale=sm_scale,
          kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
          kernel_outputs_specs=out_shape,
      ),
  )(q, k, v, ab, q_segment_ids, kv_segment_ids)
  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
  _, _, block_q_major, _ = q_tile_ref.shape
  _, _, block_k_major, _ = k_tile_ref.shape

  q_seq_index = pl.program_id(axis=3)
  kv_seq_index = pl.program_id(axis=2)

  @pl.when(q_seq_index == 0)
  def start_new_sequence():
    dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
    dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

  def q_body(j, _):
    start_q = j * block_q
    def k_body(i, _):
      start_k = i * block_k
      k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
      l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(
          jnp.float32
      )  # [block_q, 128]

      capped_logits = lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q_major, block_k]

      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            0,
            0,
            pl.dslice(j * block_q, block_q),
            pl.dslice(i * block_k, block_k),
        ].astype(jnp.float32)
        capped_logits += ab

      if sm_scale != 1.0:
        capped_logits *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
          )
        q_segment_ids = q_segment_ids_tile_ref[
            0, pl.ds(start_q, block_q), :
        ]  # [block_q, NUM_LANES].
        q_segment_ids = pltpu.repeat(
            q_segment_ids, repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            :, 0, pl.ds(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_index * block_q_major + start_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_index * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      capped_logits = (
          capped_logits
          if mask is None
          else capped_logits + jnp.where(mask, 0.0, mask_value)
      )

      p = jnp.exp(
          capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
      )
      p = p * pltpu.repeat(
          1 / l, block_k // MIN_BLOCK_SIZE, axis=1
      )  # [block_q_major, block_k_major]
      dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
      dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(
          dv_scratch_ref.dtype
      )

      # di: [block_q, 128]
      # do: [block_q, head_dim]
      # v: [block_k_major, head_dim]
      dp = lax.dot_general(
          do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )
      ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

      if sm_scale != 1.0:
        ds = ds * sm_scale

      # ds: [block_q_major, block_k_major]
      # q: [block_q_major, head_dim]
      dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
      dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(
          dk_scratch_ref.dtype
      )
    lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

  @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
  def end_of_q_sequence():
    dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
    dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
  _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

  # kv index needs to be before q index since q index is the contractng
  # dimension.
  grid = (
      batch_size,
      num_heads,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
  )

  def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    if causal:
      # If the q block is skipped, stay at the 0th q block.
      next_q_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          q_seq_index,
          0,
      )
    else:
      next_q_index = q_seq_index

    return (batch_index, head_index, next_q_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  assert qo_spec.block_shape is not None
  assert q.ndim == len(qo_spec.block_shape)
  do_spec = qo_spec
  assert do.ndim == len(qo_spec.block_shape)

  def kv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, _, q_seq_index):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index
    ):
      del head_index
      if causal:
        next_q_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            q_seq_index,
            0,
        )
      else:
        next_q_index = q_seq_index
      return (batch_index, next_q_index, 0)

    def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
      del head_index
      return (batch_index, 0, kv_seq_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           k.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           v.dtype),
  ]
  def dkv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
  out_specs = [dkv_spec, dkv_spec]
  scratch_shapes = [
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      block_q=block_q,  # type: ignore
      block_k=block_k,  # type: ignore
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      q_seq_len=q_seq_len,
  )
  name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
    assert dk.shape == k.shape
    assert dv.shape == v.shape
  return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
  _, _, block_k_major, _ = k_tile_ref.shape
  _, _, block_q_major, _ = q_tile_ref.shape

  kv_seq_index = pl.program_id(axis=3)
  q_seq_index = pl.program_id(axis=2)

  @pl.when(kv_seq_index == 0)
  def start_new_sequence():
    dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

  def body(i, _):
    k_slice = pl.ds(i * block_k, block_k)
    q = q_tile_ref[0, 0, :, :]
    k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
    di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

    capped_logits = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )

    if ab_tile_ref is not None:
      ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(
          jnp.float32
      )
      capped_logits += ab

    if sm_scale != 1.0:
      capped_logits *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
      repeats, rem = divmod(block_k, NUM_LANES)
      if rem:
        raise NotImplementedError(
            f"kv block size must be a multiple of {NUM_LANES}"
        )
      q_segment_ids = pltpu.repeat(
          q_segment_ids_tile_ref[0], repeats, axis=1
      )  # [block_q, block_k].
      kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]  # [1, block_k].
      mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
      mask_shape = (block_q_major, block_k)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      row_ids += q_seq_index * block_q_major
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      col_ids += kv_seq_index * block_k_major + i * block_k
      causal_mask = col_ids <= row_ids
      mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    capped_logits = (
        capped_logits
        if mask is None
        else capped_logits + jnp.where(mask, 0.0, mask_value)
    )

    p = jnp.exp(
        capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
    )
    p = p * pltpu.repeat(
        1 / l, block_k // MIN_BLOCK_SIZE, axis=1
    )  # [block_q_major, block_k]

    # di: [block_q_major, 128]
    # do: [block_q_major, head_dim]
    # v: [block_k_major, head_dim]
    dp = jax.lax.dot_general(
        do,
        v,
        TRANS_B_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p
    # dp = jnp.dot(do, v.T)
    # ds = (dp - (dp * p).sum(axis=1)[:, None]) * p

    if sm_scale != 1.0:
      ds = ds * sm_scale

    if ds_tile_ref is not None:
      ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(
          ds_tile_ref.dtype
      )

    # dp: [block_q_major, block_k]
    # k: [block_k, head_dim]
    dq_scratch_ref[:, :] += lax.dot(
        ds.astype(k.dtype),
        k,
        preferred_element_type=jnp.float32,
    ).astype(dq_scratch_ref.dtype)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
    should_not_run = lax.select(should_run, False, True)
  else:
    should_run = True
    should_not_run = False  # type: ignore

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

  @pl.when(should_not_run)
  def zero_out_ds():
    if ds_tile_ref is not None:
      ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

  @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
  def end_of_kv_sequence():
    dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
  _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

  grid = (
      batch_size,
      num_heads,
      q_seq_len // block_q_major,
      kv_seq_len // block_k_major,
  )

  def qo_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  do_spec = qo_spec

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
      del head_index
      return (batch_index, q_seq_index, 0)

    def kv_segment_ids_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index
    ):
      del head_index
      if causal:
        # If the kv block is skipped, prefetch the next valid kv block, i.e. the
        # 0th one to be used for the next block_q rows.
        next_kv_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, 0, next_kv_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
  ]
  dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  out_specs = [
      dq_spec,
      dab_spec,
  ]
  scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      sm_scale=sm_scale,
      causal=causal,
      mask_value=mask_value,
      block_k=block_k,  # type: ignore
      kv_seq_len=kv_seq_len,
  )
  name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dq, ds = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

  # dab is just ds
  return dq, ds


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out


@functools.partial(
    jax.jit, static_argnames=["causal", "mask_value", "sm_scale"]
)
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
  return _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=False,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  return mha_reference_no_custom_vjp(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=save_residuals,
  )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
  if save_residuals:
    raise NotImplementedError
  res = _mha_reference(
      q,
      k,
      v,
      ab,
      segment_ids,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
      save_residuals=True,
  )
  assert isinstance(res, tuple)
  out, l, m = res
  return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "sm_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
  if sm_scale != 1.0:
    raise NotImplementedError

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  if ab is not None:
    logits += ab

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

  unnormalized = jnp.exp(logits - m[..., None])
  p = unnormalized / l[..., None]
  dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

  dp = jnp.einsum(
      "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
      ..., None
  ]  # [batch_size, num_heads, q_seq_len]

  ds = (dp - di) * p
  dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

  # dab is just ds
  dab = ds if ab is not None else None
  return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
  del save_residuals
  q, k, v, ab, segment_ids, o, l, m = residuals
  dq, dk, dv, dab = mha_reference_bwd(
      q,
      k,
      v,
      ab,
      segment_ids,
      o,
      l,
      m,
      do,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
  )
  return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
  if block > dim:
    raise ValueError(
        f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
    )
  if should_divide and dim % block != 0:
    raise ValueError(
        f"{dim_name}={dim} should be divisible by {block_name}={block}"
    )


CONFIG = {
    \'name\': \'pallas_flash_attention_llama8b\',
    \'model\': \'Llama-3.1-8B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 32,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {\'block_q\': 2048, \'block_k_major\': 2048, \'block_k\': 512, \'block_b\': 1, \'block_q_major_dkv\': 128, \'block_k_major_dkv\': 128, \'block_k_dkv\': 128, \'block_q_dkv\': 128, \'block_k_major_dq\': 128, \'block_k_dq\': 128, \'block_q_dq\': 128}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG[\'batch\']
    H = CONFIG[\'num_heads\']
    S = CONFIG[\'seq_len\']
    D = CONFIG[\'head_dim\']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS[\'block_q\'],
        block_k_major=TUNED_PARAMS[\'block_k_major\'],
        block_k=TUNED_PARAMS[\'block_k\'],
        block_b=TUNED_PARAMS[\'block_b\'],
        block_q_major_dkv=TUNED_PARAMS[\'block_q_major_dkv\'],
        block_k_major_dkv=TUNED_PARAMS[\'block_k_major_dkv\'],
        block_k_dkv=TUNED_PARAMS[\'block_k_dkv\'],
        block_q_dkv=TUNED_PARAMS[\'block_q_dkv\'],
        block_k_major_dq=TUNED_PARAMS[\'block_k_major_dq\'],
        block_k_dq=TUNED_PARAMS[\'block_k_dq\'],
        block_q_dq=TUNED_PARAMS[\'block_q_dq\'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.264,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.264 ms\n{"correct": true, "latency": 0.264, "error": "", "all_times_ms": [0.255, 0.256, 0.256, 0.257, 0.257, 0.257, 0.258, 0.258, 0.259, 0.259, 0.259, 0.259, 0.259, 0.26, 0.26, 0.26, 0.26, 0.26, 0.261, 0.261, 0.261, 0.261, 0.261, 0.261, 0.261, 0.261, 0.261, 0.261, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.262, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.263, 0.264, 0.264, 0.264, 0.264, 0.264, 0.264, 0.264, 0.264, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.265, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.267, 0.267, 0.267, 0.268, 0.268, 0.268, 0.268, 0.268, 0.269, 0.269, 0.269, 0.272, 0.272, 0.274, 0.292], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr='')