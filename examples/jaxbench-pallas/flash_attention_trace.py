CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
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
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.897,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 0.897 ms\n{"correct": true, "latency": 0.897, "error": "", "all_times_ms": [0.889, 0.89, 0.89, 0.891, 0.891, 0.891, 0.891, 0.892, 0.892, 0.892, 0.892, 0.892, 0.893, 0.893, 0.893, 0.893, 0.893, 0.893, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.897, 0.897, 0.897, 0.897, 0.897, 0.897, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.899, 0.899, 0.899, 0.9, 0.9, 0.9, 0.9, 0.901, 0.901, 0.901, 0.901, 0.902, 0.902, 0.902, 0.902, 0.902, 0.903, 0.903, 0.903, 0.904, 0.904, 0.905, 0.905, 0.905, 0.905, 0.905, 0.906, 0.906, 0.906, 0.906, 0.907, 0.908, 0.91, 0.913, 0.918, 0.918, 0.919, 0.922, 0.959], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''The current Flash Attention implementation for TPU v6e uses a "normalized" version of the Online Softmax algorithm, where the output accumulator is normalized by the running sum $L$ at every step of the KV-sequence reduction. This approach is computationally expensive on the TPU\'s Vector Processing Unit (VPU), as it requires performing a reciprocal and multiple elementwise multiplications on the large accumulator matrix ($block\_q 	imes head\_dim$) for every block in the KV sequence.

Following **Strategy 5 (rewrite the algorithm to reduce total work)**, we can optimize the kernel by switching to the standard "unnormalized" Online Softmax algorithm. In this version, we maintain an unnormalized accumulator $O_{unnorm} = \sum \exp(S_i - M_{final}) V_i$. During each step, we rescale the existing accumulator by $\exp(M_{prev} - M_{next})$ to adjust for the updated running maximum and add the contribution of the new block. The division by the final normalization factor $L_{final}$ is deferred until the very end of the KV-sequence. This significantly reduces the total number of VPU cycles by eliminating redundant divisions and scaling operations from the innermost loops.

### Optimization Plan

1.  **Modify `_flash_attention_kernel_single_batch`**:
    *   Initialize `acc_scratch_ref` to zero (already done).
    *   Inside the `_body` loop (reduction over $block\_k$):
        *   Compute the local max $m_{curr}$ and updated max $m_{next}$.
        *   Compute the rescaling factor $lpha = \exp(m_{prev} - m_{next})$.
        *   Update the unnormalized accumulator: `acc_scratch_ref[...] = acc_scratch_ref[...] * alpha + o_curr`, where `o_curr` is the unnormalized product $\exp(s - m_{next}) V$.
        *   Update the running sum $l$: `l_scratch_ref[...] = l_prev * alpha + sum(exp(s - m_{next}))`.
    *   In the `store_output` block (after the grid KV reduction completes):
        *   Compute the final normalization factor `l_inv = 1.0 / l_scratch_ref[...]`.
        *   Compute the final normalized output: `o_tile_ref[...] = acc_scratch_ref[...] * l_inv`.
2.  **Maintain Residuals**: Ensure `l_ref` and `m_ref` still store the final sum-of-exponentials and maximum values required for the backward pass.

```python
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
        repeats, _ = divmod(block_k, NUM_LANES)
        q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1)
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1, pl.dslice(start_k, block_k)]
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + kv_seq_idx * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

      s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

      m_curr = jnp.max(s, axis=1)[:, None]
      m_next = jnp.maximum(m_prev, m_curr)

      block_k_repeats = block_k // MIN_BLOCK_SIZE
      p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
      alpha = jnp.exp(m_prev - m_next)
      
      head_dim_repeats = head_dim // MIN_BLOCK_SIZE
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)

      # Algorithmic change: Accumulate unnormalized to reduce VPU cycles
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = alpha * l_prev + jnp.sum(p, axis=1)[:, None]
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
    if l_ref is not None:
      l_ref[batch_idx] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)
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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.824,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.824 ms\n{"correct": true, "latency": 0.824, "error": "", "all_times_ms": [0.817, 0.817, 0.817, 0.817, 0.818, 0.818, 0.818, 0.818, 0.818, 0.818, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.82, 0.82, 0.82, 0.82, 0.821, 0.821, 0.821, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.823, 0.823, 0.823, 0.823, 0.823, 0.823, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.824, 0.825, 0.825, 0.825, 0.825, 0.825, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.827, 0.827, 0.827, 0.827, 0.827, 0.828, 0.828, 0.828, 0.828, 0.828, 0.829, 0.83, 0.83, 0.83, 0.831, 0.831, 0.831, 0.832, 0.832, 0.833, 0.834, 0.837, 0.837, 0.84, 0.843, 0.843, 0.846, 0.949], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''## Analysis

Looking at the code, I need to identify inefficiencies that can be addressed by one of the strategies.

Let me analyze the forward pass kernel `_flash_attention_kernel_single_batch`:

1. **Scalar `sm_scale` broadcast inside loop**: The `sm_scale` multiplication happens inside the inner loop:
```python
if sm_scale != 1.0:
    s *= sm_scale
```
This scalar-to-vector broadcast occurs on every iteration of the `block_k_major // block_k` loop.

2. **Looking at Strategy 13**: "Pre-broadcast `sm_scale` into a VREG at kernel entry to avoid scalar-to-vector broadcast latency inside the loop."

This is a clear optimization opportunity. Currently, `sm_scale` is a Python float that gets broadcast to match the shape of `s` (which is `[block_q, block_k]`) on every inner loop iteration. On TPU, scalar-to-vector broadcasts have latency that can be avoided by pre-computing the broadcast once at kernel entry.

## Plan: Apply Strategy 13 - Pre-broadcast `sm_scale` into a VREG at kernel entry

### Specific Changes:

1. **In `_flash_attention_kernel_single_batch`**:
   - At the beginning of the kernel (before the inner loop), pre-broadcast `sm_scale` into a VREG with shape `[block_q, block_k]` (or `[1, 1]` that will be broadcast):
   ```python
   # Pre-broadcast sm_scale to avoid repeated scalar broadcasts in the loop
   sm_scale_broadcast = jnp.full((block_q, block_k), sm_scale, dtype=jnp.float32)
   ```
   - Inside the loop, replace `s *= sm_scale` with `s *= sm_scale_broadcast` (or use `s = s * sm_scale_broadcast`)

2. **In `_flash_attention_kernel_single_batch_single_step`**:
   - Apply the same optimization: pre-broadcast `sm_scale` before use

3. **In backward kernels `_flash_attention_dkv_kernel` and `_flash_attention_dq_kernel`**:
   - Pre-broadcast `sm_scale` at kernel entry before the inner loops where it\'s used:
   ```python
   # For dkv kernel:
   sm_scale_broadcast = jnp.full((block_q, block_k), sm_scale, dtype=jnp.float32)
   # Then use sm_scale_broadcast instead of sm_scale in the inner body
   ```

### Implementation Details:

For the forward kernel `_flash_attention_kernel_single_batch`:
- Move the `sm_scale` broadcast outside the `@pl.loop` decorated function
- Create a local variable `sm_scale_vec` at the start of the `run()` block that will be captured by the inner loop

For the backward kernels:
- In `_flash_attention_dkv_kernel`, the `sm_scale` is used twice in the inner `k_body`: once for `capped_logits *= sm_scale` and once for `ds = ds * sm_scale`. Pre-broadcast once and reuse.
- In `_flash_attention_dq_kernel`, similar pattern in the `body` function.

The key insight is that the broadcast from scalar to `(block_q, block_k)` shape is happening on every inner loop iteration. By hoisting this to kernel entry, we:
1. Reduce the number of broadcast operations from `O(iterations)` to `O(1)`
2. Potentially allow the compiler to keep the broadcast value in VREGs across iterations
3. Eliminate repeated scalar-to-vector conversion latency

This is a low-risk optimization that maintains numerical equivalence while reducing redundant computation inside hot loops.''',
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.823,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='gpt-5.4',
stdout='Latency: 0.823 ms\n{"correct": true, "latency": 0.823, "error": "", "all_times_ms": [0.815, 0.816, 0.817, 0.817, 0.817, 0.818, 0.818, 0.818, 0.818, 0.818, 0.818, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.819, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.821, 0.821, 0.821, 0.821, 0.821, 0.821, 0.821, 0.821, 0.821, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822, 0.823, 0.823, 0.823, 0.823, 0.823, 0.823, 0.823, 0.823, 0.824, 0.824, 0.824, 0.825, 0.825, 0.825, 0.825, 0.825, 0.825, 0.825, 0.825, 0.825, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.827, 0.827, 0.827, 0.827, 0.827, 0.828, 0.828, 0.828, 0.829, 0.829, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.831, 0.831, 0.832, 0.833, 0.833, 0.833, 0.834, 0.854], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''**Selected strategy: 15. Use a KV-stationary persistent schedule**

### Why this code is leaving performance on the table
For the forward pass, the current launch is:

- `grid = (batch, head, q_block, kv_block)`
- reduction over `kv_block` is the last grid axis

That is correct for the online-softmax accumulation, but on **v6e-1** it causes the same `K/V` major tile to be fetched again for every `Q` block.

For your workload:

- `B=1, H=64, S=2048, D=128`
- `block_q=1024` → **2 Q blocks**
- `block_k_major=1024` → **2 KV major blocks**

Per head, the current schedule touches:

- `Q`: 2 tiles total
- `K`: 2 KV-major tiles × 2 Q-blocks = **4 loads**
- `V`: same = **4 loads**

So each `K/V` major tile is reloaded once for each `Q` block. On v6e this is a bad trade because all accumulators for both Q blocks fit comfortably in VMEM.

---

## Plan

### 1) Add a v6e fast path in `_flash_attention_impl`
Keep the public API exactly the same, but inside `_flash_attention_impl` dispatch to a new forward kernel when these are true:

- `block_b == 1`
- `ab is None`
- `segment_ids is None`
- `q_seq_len % block_q == 0`
- `kv_seq_len % block_k_major == 0`
- `block_k_major % block_k == 0`
- estimated VMEM use fits comfortably on v6e

Otherwise, fall back to the existing implementation unchanged.

This keeps semantics the same for all cases while optimizing the hot path your benchmark actually uses.

---

### 2) Change the forward grid from `(B, H, Qblk, KVblk)` to `(B, H)`
Use one Pallas program per `(batch, head)`.

#### New `in_specs`
Map each program to one full head slice:

- `q`: `BlockSpec((1, 1, q_seq_len, head_dim), lambda b, h: (b, h, 0, 0))`
- `k`: `BlockSpec((1, 1, kv_seq_len, head_dim), lambda b, h: (b, h, 0, 0))`
- `v`: same as `k`

For outputs:

- `o`: `BlockSpec((1, 1, q_seq_len, head_dim), lambda b, h: (b, h, 0, 0))`

If `save_residuals=True`, keep the existing residual storage convention:

- `l`: `BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), lambda b, h: (b, h, 0, 0))`
- `m`: same

Use:
```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=("parallel", "parallel")
)
```

This is valid because batch/head programs are independent, and the reduction over KV is now handled inside the kernel rather than across grid iterations.

---

### 3) Keep all Q-block accumulators live in VMEM for the whole head
Inside the new kernel, allocate persistent VMEM scratch for **all Q blocks of the head**:

- `m_scratch`: `(num_q_blocks, block_q, MIN_BLOCK_SIZE)`, `f32`
- `l_scratch`: `(num_q_blocks, block_q, MIN_BLOCK_SIZE)`, `f32`
- `acc_scratch`: `(num_q_blocks, block_q, head_dim)`, `f32`

For your config (`num_q_blocks=2`, `block_q=1024`, `D=128`), this is only a few MB and fits well on v6e.

Initialize once at kernel start:

- `m = -inf`
- `l = 0`
- `acc = 0`

Because there is only one program per `(batch, head)`, no `program_id`-based accumulator reset is needed.

---

### 4) Reorder the inner work to make KV stationary
Refactor the current `_flash_attention_kernel_single_batch` math into a helper that processes **one `(q_block, kv_major_block)` pair** using explicit indices instead of `pl.program_id(2/3)`.

Then structure the new kernel like:

1. **Outer loop over `kv_major`**
   - Slice `k_ref[..., kv_start:kv_start+block_k_major, :]`
   - Slice `v_ref[..., kv_start:kv_start+block_k_major, :]`

2. **Inner loop over all `q_block`s**
   - Slice `q_ref[..., q_start:q_start+block_q, :]`
   - Slice corresponding `m/l/acc` scratch
   - Run the same online-softmax update over the `block_k` subtiles inside that KV-major tile

3. After all KV-major tiles, normalize and store each Q block to `o_ref`

Important: inside the helper, read from `Ref`s explicitly, e.g.
```python
q = q_ref[0, 0, pl.ds(q_start, block_q), :]
k = k_ref[0, 0, pl.ds(k_start, block_k), :]
```
and write back explicitly:
```python
o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)
```

The math itself stays the same as the current online-softmax recurrence, so numerics remain equivalent up to normal tolerance.

---

### 5) Preserve the exact softmax accumulation order per Q block
Do **not** change the order of KV accumulation for a given Q block. Each Q block should still see KV-major blocks in increasing order:

- `kv_major = 0, 1, ...`

That preserves the same online-softmax recurrence and keeps output differences minimal.

Only the schedule changes:
- before: process one Q block through all KV blocks, then move to next Q block
- after: keep one KV block resident and update all Q blocks before evicting it

Same math, less HBM traffic.

---

### 6) Keep loops small and explicit
Since Pallas loops are unrolled, only use this fast path when trip counts are small enough. For this workload they are:

- `num_q_blocks = 2`
- `num_kv_major_blocks = 2`
- `block_k_major // block_k = 8`

Those are safe.

If these grow too large, fall back to the original kernel.

---

## What this should save
For the benchmarked configuration, per head:

- current input traffic:
  - `Q`: 0.5 MB
  - `K`: 1.0 MB
  - `V`: 1.0 MB
  - total inputs ≈ **2.5 MB**

- KV-stationary persistent version:
  - `Q`: 0.5 MB
  - `K`: 0.5 MB
  - `V`: 0.5 MB
  - total inputs ≈ **1.5 MB**

So this removes about:

- **40% of input HBM traffic**
- **33% of total HBM traffic** if you include the unchanged output write

That is the right direction for a v6e forward FlashAttention kernel, especially since this implementation is already using large tiles and the remaining waste is repeated `K/V` reloads.

---

## Concrete code changes
1. Keep `flash_attention(...)` signature unchanged.
2. In `_flash_attention_impl`, add a fast-path branch for the v6e causal/no-bias/no-segment case.
3. Add a new kernel, e.g. `_flash_attention_kernel_head_kv_stationary`.
4. Factor the existing per-`(q_block, kv_block)` update math out of `_flash_attention_kernel_single_batch` into a reusable helper taking explicit block indices.
5. Launch the new kernel with:
   - `grid=(batch_size, num_heads)`
   - full-head `BlockSpec`s
   - VMEM scratch for all Q-block accumulators
   - `dimension_semantics=("parallel", "parallel")`
6. Keep the existing implementation as fallback.

That applies exactly one optimization strategy, is specific to this code, and targets the main avoidable cost on **TPU v6e-1**.''',
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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


# ---------------------------------------------------------------------------
# KV-stationary persistent forward kernel for v6e
# ---------------------------------------------------------------------------


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  # Initialize accumulators for all Q blocks
  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  # Outer loop: KV-major blocks (stream K/V once)
  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    # Inner loop: all Q blocks (reuse K/V tile)
    @pl.loop(0, num_q_blocks, unroll=True)
    def _q_loop(q_idx):
      q_start = q_idx * block_q

      # Causal skip: if entire Q block is above the diagonal for this KV block
      if causal:
        # bottom-left corner of Q block vs top-right corner of KV-major block
        should_run = below_or_on_diag(q_idx, block_q, kv_major_idx, block_k_major)
      else:
        should_run = True

      @pl.when(should_run)
      def _run_qk():
        # Load accumulators for this Q block
        m_prev = m_scratch_ref[q_idx, :, :]
        l_prev = l_scratch_ref[q_idx, :, :]
        acc_prev = acc_scratch_ref[q_idx, :, :]

        q = q_ref[0, 0, pl.ds(q_start, block_q), :]  # [block_q, head_dim]

        # Loop over k sub-tiles within the KV-major block
        @pl.loop(0, num_k_per_major, unroll=True)
        def _k_loop(k_sub_idx):
          k_start = kv_major_start + k_sub_idx * block_k
          k = k_ref[0, 0, pl.ds(k_start, block_k), :]  # [block_k, head_dim]

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )  # [block_q, block_k]

          if sm_scale_block is not None:
            s = s * sm_scale_block

          if causal:
            mask_shape = (block_q, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids = row_ids + q_start
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids = col_ids + k_start
            causal_mask = col_ids <= row_ids
            s = s + jnp.where(causal_mask, 0.0, mask_value)

          # Online softmax update
          m_curr = jnp.max(s, axis=1)[:, None]  # [block_q, 1]
          m_next = jnp.maximum(m_scratch_ref[q_idx, :, :], m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

          alpha = jnp.exp(m_scratch_ref[q_idx, :, :] - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_scratch_ref[q_idx, :, :]

          v = v_ref[0, 0, pl.ds(k_start, block_k), :]  # [block_k, head_dim]
          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = (
              acc_scratch_ref[q_idx, :, :] * alpha_broad + o_curr
          )
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

  # Finalize: normalize and store output for each Q block
  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


CONFIG = {
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.731,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.731 ms\n{"correct": true, "latency": 0.731, "error": "", "all_times_ms": [0.717, 0.718, 0.719, 0.719, 0.719, 0.72, 0.72, 0.72, 0.72, 0.721, 0.721, 0.721, 0.722, 0.723, 0.723, 0.724, 0.724, 0.724, 0.724, 0.724, 0.724, 0.725, 0.725, 0.725, 0.726, 0.726, 0.726, 0.726, 0.727, 0.727, 0.727, 0.727, 0.728, 0.728, 0.728, 0.729, 0.729, 0.729, 0.729, 0.729, 0.729, 0.729, 0.729, 0.729, 0.729, 0.73, 0.73, 0.73, 0.73, 0.73, 0.731, 0.731, 0.731, 0.731, 0.731, 0.731, 0.731, 0.732, 0.732, 0.732, 0.732, 0.732, 0.732, 0.733, 0.733, 0.733, 0.733, 0.733, 0.733, 0.733, 0.733, 0.733, 0.733, 0.734, 0.734, 0.734, 0.734, 0.734, 0.734, 0.735, 0.735, 0.735, 0.736, 0.737, 0.737, 0.738, 0.738, 0.74, 0.74, 0.74, 0.74, 0.741, 0.742, 0.743, 0.745, 0.746, 0.748, 0.757, 0.779, 0.823], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''To optimize the Flash Attention kernel for TPU v6e, we will refine the **KV-stationary path**. This path is particularly effective for v6e\'s 16 MiB VMEM as it allows holding a significant portion of the sequence on-chip, minimizing HBM reloads.

### Optimization Plan: Loop Reordering for VMEM-to-VREG Reuse

The current `kv_stationary` implementation iterates through Q-blocks in an outer loop and K/V sub-tiles in an inner loop. This causes each K and V sub-tile to be re-read from VMEM into the Vector Processing Unit (VREGs) for every Q-block. For a sequence length of 2048 with `block_q=1024`, K/V tiles are loaded twice. 

**Our strategy is to reorder the loops** so that the K and V sub-tiles are loaded once into VREGs and then reused across all Q-blocks. This reduces internal data movement (VMEM-to-VREG) by a factor equal to the number of Q-blocks. Given the "single-threaded" nature of Pallas on TPU, minimizing these register loads and maximizing reuse within unrolled loops significantly improves MXU throughput.

1.  **Reorder Nesting**: Move the K/V sub-tile loop (`_k_loop`) outside the Q-block loop (`_q_loop`).
2.  **Load Once**: Inside the `_k_loop`, load the K and V sub-tiles into VREGs.
3.  **Update All Q**: The inner `_q_loop` (which is unrolled) will then apply these K/V tiles to update the online softmax accumulators (`m`, `l`, `acc`) for all Q-blocks resident in VMEM.
4.  **Preserve Online Softmax**: Ensure the online softmax logic correctly reads from and writes to the VMEM scratch buffers for each Q-block.

### Optimized Flash Attention Code

```python
import functools
import math
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))

def below_or_on_diag(r, r_blk_size, c, c_blk_size):
  return ((r + 1) * r_blk_size - 1) >= (c * c_blk_size)

def _flash_attention_kernel_kv_stationary(
    q_ref, k_ref, v_ref, o_ref, l_ref, m_ref,
    m_scratch_ref, l_scratch_ref, acc_scratch_ref,
    *, causal: bool, sm_scale: float, block_q: int, block_k: int,
    block_k_major: int, q_seq_len: int, kv_seq_len: int, mask_value: float,
):
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  # Initialize accumulators in VMEM scratch space
  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = jnp.full((block_q, block_k), sm_scale, jnp.float32) if sm_scale != 1.0 else None

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      # Load K and V sub-tiles into VREGs once for use by all Q blocks
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        # Causal optimization: check if this Q-block needs to interact with this K-tile
        if causal:
          should_run = (q_idx + 1) * block_q > k_start
        else:
          should_run = True

        @pl.when(should_run)
        def _run():
          m_prev = m_scratch_ref[q_idx]
          l_prev = l_scratch_ref[q_idx]
          acc_prev = acc_scratch_ref[q_idx]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
          if sm_scale_block is not None:
            s *= sm_scale_block

          if causal:
            mask_shape = (block_q, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_start
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + k_start
            s += jnp.where(col_ids <= row_ids, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)
          
          # Compute softmax components in f32
          p = jnp.exp(s - pltpu.repeat(m_next, block_k // MIN_BLOCK_SIZE, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
          
          # Update online softmax accumulators in VMEM
          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1) if head_dim_repeats > 0 else alpha[:, :head_dim]
          
          acc_scratch_ref[q_idx] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx] = l_next
          m_scratch_ref[q_idx] = m_next

  # Normalize and write back to HBM
  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_inv = 1.0 / jnp.where(l_scratch_ref[q_idx] == 0.0, 1.0, l_scratch_ref[q_idx])
    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1) if head_dim_repeats > 0 else l_inv[:, :head_dim]
    
    o_ref[0, 0, pl.ds(q_start, block_q), :] = (acc_scratch_ref[q_idx] * l_inv_broad).astype(o_ref.dtype)
    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_scratch_ref[q_idx].astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx].astype(m_ref.dtype)

def _flash_attention_impl_kv_stationary(q, k, v, save_residuals, causal, sm_scale, block_q, block_k_major, block_k, debug):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)
  idx_map = lambda b, h: (b, h, 0, 0)

  # Scratch shapes for online softmax (kept in f32 for precision)
  scratch_shapes = [
      pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32),
      pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32),
      pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32),
  ]

  out_shape = [jax.ShapeDtypeStruct(q.shape, q.dtype)]
  out_specs = [pl.BlockSpec((1, 1, q_seq_len, head_dim), idx_map)]

  if save_residuals:
    res_shape = (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE)
    out_shape.extend([jax.ShapeDtypeStruct(res_shape, jnp.float32)] * 2)
    out_specs.extend([pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), idx_map)] * 2)
  else:
    out_shape.extend([None, None])
    out_specs.extend([None, None])

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary, causal=causal, sm_scale=sm_scale,
      block_q=block_q, block_k=block_k, block_k_major=block_k_major,
      q_seq_len=q_seq_len, kv_seq_len=kv_seq_len, mask_value=DEFAULT_MASK_VALUE)

  res = pl.pallas_call(
      kernel, grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0, grid=grid, in_specs=[
              pl.BlockSpec((1, 1, q_seq_len, head_dim), idx_map),
              pl.BlockSpec((1, 1, kv_seq_len, head_dim), idx_map),
              pl.BlockSpec((1, 1, kv_seq_len, head_dim), idx_map)],
          out_specs=out_specs, scratch_shapes=scratch_shapes),
      out_shape=out_shape, debug=debug,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  )(q, k, v)
  
  if save_residuals:
    return res[0], res[1][..., 0], res[2][..., 0]
  return res
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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


# ---------------------------------------------------------------------------
# KV-stationary persistent forward kernel for v6e
# ---------------------------------------------------------------------------


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  Reorders loops to reuse each K/V sub-tile across all resident Q blocks
  before loading the next K/V sub-tile.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        if causal:
          should_run = (q_idx + 1) * block_q > k_start
        else:
          should_run = True

        @pl.when(should_run)
        def _run_qk():
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          if causal:
            mask_shape = (block_q, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_start
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + k_start
            s += jnp.where(col_ids <= row_ids, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


CONFIG = {
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.673,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='gpt-5.4',
stdout='Latency: 0.673 ms\n{"correct": true, "latency": 0.673, "error": "", "all_times_ms": [0.664, 0.665, 0.665, 0.665, 0.666, 0.666, 0.666, 0.667, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.669, 0.669, 0.669, 0.669, 0.669, 0.669, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.671, 0.671, 0.671, 0.671, 0.671, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.673, 0.673, 0.673, 0.673, 0.673, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.674, 0.675, 0.675, 0.675, 0.675, 0.675, 0.675, 0.676, 0.676, 0.676, 0.676, 0.677, 0.677, 0.677, 0.677, 0.677, 0.677, 0.677, 0.677, 0.678, 0.678, 0.678, 0.678, 0.679, 0.679, 0.679, 0.679, 0.679, 0.679, 0.68, 0.682, 0.683, 0.683, 0.687, 0.697], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''The analysis of the current implementation reveals a significant inefficiency in the handling of the causal mask during the forward pass. The KV-stationary kernel is selected for the forward pass on TPU v6e.

### Inefficiency
In the `_flash_attention_kernel_kv_stationary` function, the causal mask is regenerated inside the innermost loop (`_q_loop`). In every iteration, the code performs:
1.  `jax.lax.broadcasted_iota` to create row and column indices.
2.  Comparison operations to generate the mask.
3.  A `jnp.where` op to apply the mask.

This happens `num_kv_major_blocks * num_k_per_major * num_q_blocks` times (potentially thousands of times). On the TPU VPU, generating and applying this mask in every inner loop iteration involves significant instruction overhead for the broadcasts and comparisons, which can be entirely avoided.

### Optimization Strategy
**Strategy 9: Pre-compute causal masks statically.**

The causal mask structure is entirely determined by the block indices (`q_idx`, `k_sub_idx`) and the compile-time constant block sizes (`block_q`, `block_k`). Instead of computing element-wise indices and comparisons in the inner loop, we can determine the status of the block statically.

There are three states for a block given a causal diagonal:
1.  **Fully Masked (Above Diagonal):** The entire block is invalid (Attention score should be `-inf`). No dot-product or softmax logic is needed (scores are `-inf`, `exp` is `0`, contribution is `0`).
2.  **Fully Unmasked (Below Diagonal):** The entire block is valid. No masking logic is needed at all.
3.  **Partially Masked (On Diagonal):** Only part of the block is valid. This requires the element-wise mask application.

By inspecting the corner indices of the blocks, we can determine the state before entering the `_run_qk` body.
- **Fully Masked**: Skip the computation entirely (output contribution is zero). This saves significant compute for the upper-triangular region.
- **Fully Unmasked**: Skip the `broadcasted_iota` and `jnp.where` steps. This saves instruction issue overhead and register pressure for the majority of the matrix (lower triangle).

### Plan
Modify the `_flash_attention_kernel_kv_stationary` function to check the block\'s status relative to the causal diagonal before executing the attention logic.

1.  Inside `_q_loop`, before loading `q` or computing `s`:
    - Check if the bottom-left corner of the Q-block `(q_idx + 1) * block_q - 1` is less than the top-left corner of the K-block `k_start`. If so, the block is **Fully Masked**. The execution should skip (do nothing), preserving the original `m_scratch` (-inf) and `l_scratch` (0).
    - Check if the top-left corner of the Q-block `q_idx * block_q` is greater than or equal to the bottom-right corner of the K-block `(k_start + block_k - 1)`. If so, the block is **Fully Unmasked**. Execute the QK matmul and softmax update *without* the mask generation and application.
    - If neither, the block is **Partially Masked**. Execute the existing logic with the `broadcasted_iota` mask.

2.  This change eliminates expensive VPU operations for the bulk of the matrix and skips compute entirely for the irrelevant upper triangular blocks. This directly targets the computation bottleneck inside the inner loop of the KV-stationary kernel.''',
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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


# ---------------------------------------------------------------------------
# KV-stationary persistent forward kernel for v6e
# ---------------------------------------------------------------------------


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  Reorders loops to reuse each K/V sub-tile across all resident Q blocks
  before loading the next K/V sub-tile.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        if causal:
          # Check if block is fully masked (above diagonal): skip entirely
          # Bottom-left of Q-block row < top-left of K-block col means all masked
          q_block_bottom_row = (q_idx + 1) * block_q - 1
          k_block_top_col = k_start
          fully_masked = q_block_bottom_row < k_block_top_col

          # Check if block is fully unmasked (below diagonal):
          # Top-left of Q-block row >= bottom-right of K-block col means all valid
          q_block_top_row = q_start
          k_block_bottom_col = k_start + block_k - 1
          fully_unmasked = q_block_top_row >= k_block_bottom_col
        else:
          fully_masked = False
          fully_unmasked = True  # No mask needed at all

        @pl.when(~fully_masked & fully_unmasked)
        def _run_qk_no_mask():
          # Block is fully below diagonal: no mask needed
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        @pl.when(~fully_masked & ~fully_unmasked)
        def _run_qk_with_mask():
          # Block is on the diagonal: apply partial mask
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          mask_shape = (block_q, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + k_start
          s += jnp.where(col_ids <= row_ids, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        # When fully_masked is True, we skip entirely (do nothing).
        # m_scratch stays at -inf, l_scratch stays at 0, acc_scratch stays at 0.

  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


CONFIG = {
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.668,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.668 ms\n{"correct": true, "latency": 0.668, "error": "", "all_times_ms": [0.661, 0.662, 0.663, 0.663, 0.663, 0.663, 0.663, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.669, 0.669, 0.669, 0.669, 0.669, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.671, 0.671, 0.671, 0.671, 0.672, 0.672, 0.672, 0.673, 0.673, 0.673, 0.674, 0.675, 0.675, 0.675, 0.675, 0.676, 0.676, 0.676, 0.676, 0.676, 0.677, 0.678, 0.679, 0.679, 0.68, 0.68, 0.687, 0.687], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''

Looking at this Flash Attention implementation for TPU v6e, I can identify a significant inefficiency in the KV-stationary kernel (`_flash_attention_kernel_kv_stationary`).

## Inefficiency Analysis

In the `_run_qk_with_mask` function inside the innermost loop, the causal mask is generated using `jax.lax.broadcasted_iota`:

```python
mask_shape = (block_q, block_k)
row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_start
col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1) + k_start
s += jnp.where(col_ids <= row_ids, 0.0, mask_value)
```

This mask computation happens inside the innermost loop for every (q_block, k_block) combination. The key observation is that the relative offset between Q and K block positions (`k_start - q_start`) is constant for all elements within a specific (q_idx, k_sub_idx) pair. Instead of constructing full row/column index matrices and comparing them, we can compute this offset once and use it directly.

## Optimization Plan: Strategy #11

Apply **vectorized mask generation by pre-computing relative offsets** to eliminate redundant broadcast operations.

### Changes to `_flash_attention_kernel_kv_stationary`:

1. Compute the constant relative offset `k_start - q_start` once per inner loop iteration
2. Replace the full `broadcasted_iota` + comparison with a more efficient offset-based mask computation

The modified mask computation will use the fact that if the offset is ≥ `block_q`, the block is fully masked (skip); if offset < 0, fully unmasked; otherwise apply the partial mask using just the column offset from the start of the K block.

### Semantic Equivalence

This optimization produces identical results: instead of comparing `(col + k_start) <= (row + q_start)`, we compare `col <= row + (q_start - k_start)`. Since `q_start - k_start` is the relative offset, the logical result is the same, just computed more efficiently by avoiding the iota construction.''',
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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


# ---------------------------------------------------------------------------
# KV-stationary persistent forward kernel for v6e
# ---------------------------------------------------------------------------


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  Reorders loops to reuse each K/V sub-tile across all resident Q blocks
  before loading the next K/V sub-tile.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        if causal:
          # Check if block is fully masked (above diagonal): skip entirely
          # Bottom-left of Q-block row < top-left of K-block col means all masked
          q_block_bottom_row = (q_idx + 1) * block_q - 1
          k_block_top_col = k_start
          fully_masked = q_block_bottom_row < k_block_top_col

          # Check if block is fully unmasked (below diagonal):
          # Top-left of Q-block row >= bottom-right of K-block col means all valid
          q_block_top_row = q_start
          k_block_bottom_col = k_start + block_k - 1
          fully_unmasked = q_block_top_row >= k_block_bottom_col
        else:
          fully_masked = False
          fully_unmasked = True  # No mask needed at all

        @pl.when(~fully_masked & fully_unmasked)
        def _run_qk_no_mask():
          # Block is fully below diagonal: no mask needed
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        @pl.when(~fully_masked & ~fully_unmasked)
        def _run_qk_with_mask():
          # Block is on the diagonal: apply partial mask
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          # Use relative offset to avoid constructing full row/col index matrices
          # Condition: col + k_start <= row + q_start  <=>  col <= row + (q_start - k_start)
          rel_offset = q_start - k_start
          mask_shape = (block_q, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          s += jnp.where(col_ids <= row_ids + rel_offset, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        # When fully_masked is True, we skip entirely (do nothing).
        # m_scratch stays at -inf, l_scratch stays at 0, acc_scratch stays at 0.

  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


CONFIG = {
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.667,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.667 ms\n{"correct": true, "latency": 0.667, "error": "", "all_times_ms": [0.658, 0.658, 0.66, 0.661, 0.661, 0.661, 0.661, 0.661, 0.662, 0.662, 0.662, 0.662, 0.662, 0.663, 0.663, 0.663, 0.663, 0.663, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.664, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.668, 0.67, 0.67, 0.671, 0.671, 0.671, 0.671, 0.671, 0.671, 0.671, 0.671, 0.672, 0.672, 0.672, 0.673, 0.673, 0.673, 0.673, 0.674, 0.674, 0.675, 0.676, 0.678, 0.678, 0.678, 0.68, 0.682, 0.684, 0.687, 0.688, 0.694, 0.696], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr=''),
plan='''Based on the analysis of the code and the provided strategies, here is the optimization plan.

## Analysis of Current Implementation

The current forward pass implementation (`_flash_attention_impl`) uses a grid where one dimension iterates over the KV sequence length (`kv_seq_len // block_k_major`). This forces the hardware to reload the same Q blocks from High Bandwidth Memory (HBM) into Vector Memory (VMEM) for every iteration of the KV loop.

Specifically, for a sequence length of 2048 and `block_q=1024`, the output is computed in two strips. For each strip, the kernel loops over KV blocks. This leads to redundant HBM loads of Q.
- **HBM Accesses (Current):** ~$(\text{num\_q\_blocks} \times \text{num\_kv\_blocks})$ loads of Q blocks.
- **HBM Accesses (Optimized):** ~$\text{num\_kv\_blocks}$ loads of Q blocks.

## Strategy: Q-Stationary Loop Tiling (KV-Major Hoisting)

This plan implements a **Q-Stationary** persistent kernel strategy, which corresponds to **Strategy 2 (Loop Tiling)** and **Strategy 1 (Caching Reused Data)**.

### Optimization Plan

1.  **Change the Grid**: Transform the grid from `(batch, heads, q_blocks, kv_blocks)` to `(batch, heads)`.
2.  **Persistent Kernel**: Implement a kernel that runs entirely in VMEM for a single (batch, head) pair, keeping the reduction accumulators (`m`, `l`, `o`) in VMEM ("stationary") while streaming K and V blocks.
3.  **Loop Reordering**: Hoist the Q-block loop *outside* the KV-block loop. This ensures that each Q tile is loaded from HBM exactly once and reused against all K/V tiles.
4.  **Refactor Indexing**: Replace `BlockSpec` dynamic indexing with manual slicing using `pl.ds` (dynamic slice) inside the kernel loops. This ensures the compiler groups HBM accesses correctly.
5.  **Eliminate Control Flow Overhead**: Replace `@pl.when` blocks for causal masking with `jax.lax.cond` (Strategy 9) and relative offset calculations to decide whether to skip or process a block. This allows the kernel to handle the causal diagonal efficiently without expensive predication inside the innermost loop.

### Performance Benefit

- **Reduced HBM Traffic**: This reduces the number of Q-tile loads from HBM by a factor of `num_kv_blocks` (roughly 16x for the provided Llama configuration).
- **Better VMEM Utilization**: The kernel retains accumulators for all Q blocks in VMEM, amortizing load costs.
- **Instruction Level Parallelism**: Using `jax.lax.cond` for block masking reduces the overhead of Pallas\'s `when`, which compiles to separate program IDs and more complex control flow.

### Proposed Changes

The proposed code refactors `_flash_attention_impl` to use a new kernel `_flash_attention_kernel_kv_stationary` and updates the `BlockSizes` to better match the hardware constraints (keeping `block_q` large enough to maximize reuse but small enough to fit accumulators).

We will introduce a new implementation path `_flash_attention_impl_kv_stationary` and select it when the problem shape fits in VMEM (which is the case for the target configuration).

```python
# ---------------------------------------------------------------------------

def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  Reorders loops to reuse each K/V sub-tile across all resident Q blocks
  before loading the next K/V sub-tile.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        if causal:
          # Check if block is fully masked (above diagonal): skip entirely
          # Bottom-left of Q-block row < top-left of K-block col means all masked
          q_block_bottom_row = (q_idx + 1) * block_q - 1
          k_block_top_col = k_start
          fully_masked = q_block_bottom_row < k_block_top_col

          # Check if block is fully unmasked (below diagonal):
          # Top-left of Q-block row >= bottom-right of K-block col means all valid
          q_block_top_row = q_start
          k_block_bottom_col = k_start + block_k - 1
          fully_unmasked = q_block_top_row >= k_block_bottom_col
        else:
          fully_masked = False
          fully_unmasked = True # No mask needed at all

        @pl.when(~fully_masked & fully_unmasked)
        def _run_qk_no_mask():
          # Block is fully below diagonal: no mask needed
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        @pl.when(~fully_masked & ~fully_unmasked)
        def _run_qk_with_mask():
          # Block is on the diagonal: apply partial mask
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          # Use relative offset to avoid constructing full row/col index matrices
          # Condition: col + k_start <= row + q_start  <=>  col <= row + (q_start - k_start)
          rel_offset = q_start - k_start
          mask_shape = (block_q, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          s += jnp.where(col_ids <= row_ids + rel_offset, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        # When fully_masked is True, we skip entirely (do nothing).
        # m_scratch stays at -inf, l_scratch stays at 0, acc_scratch stays at 0.

  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


```

And an update to `_flash_attention_impl` to select this path:

```python
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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
    )

  # ... existing non-stationary path ...
```
(Relevant parts of the existing `_flash_attention_impl` remain for cases not covered by the optimization).

### Integration with `workload`

The provided `workload` function uses 1024x1024 blocks. With the new path, this configuration perfectly matches the v6e optimization criteria. The $Q$ tensor will be loaded into VMEM once per head, and K/V will be streamed, reducing memory bandwidth pressure significantly.

```python
def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG[\'head_dim\'])
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all 
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V 
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may 
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024, 
        block_k_major=1024, 
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes, 
    )
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        s *= sm_scale_block

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

      l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # Shape [block_q, 128]

      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda x: x[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # Unnormalized accumulation: defer division by l to the end
      acc_scratch_ref[batch_idx] = acc_scratch_ref[batch_idx] * l_broadcast(alpha) + o_curr
      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    # Deferred normalization: only perform division once per sequence
    l_final = l_scratch_ref[batch_idx]
    l_inv = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda x: pltpu.repeat(x, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda x: x[:, :head_dim]

    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(l_inv)).astype(o_tile_ref.dtype)
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  assert kv_seq_len == block_k_major == block_k

  q = q_tile_ref[batch_idx]  # [block_q, head_dim]
  k = k_tile_ref[batch_idx]  # [block_k, head_dim]
  s = jax.lax.dot_general(
      q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
  )  # [block_q, block_k]

  if ab_tile_ref is not None:
    s += ab_tile_ref[batch_idx].astype(jnp.float32)
  if sm_scale_block is not None:
    s *= sm_scale_block

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

  # Fast path for v6e: KV-stationary persistent schedule when all Q-block
  # accumulators fit in VMEM and there are no masks/biases.
  num_q_blocks = pl.cdiv(q_seq_len, block_q)
  num_kv_major_blocks = kv_seq_len // block_k_major
  use_kv_stationary = (
      block_b == 1
      and ab is None
      and segment_ids is None
      and q_seq_len % block_q == 0
      and kv_seq_len % block_k_major == 0
      and block_k_major % block_k == 0
      # Keep loops small (unrolled at compile time)
      and num_q_blocks <= 8
      and num_kv_major_blocks <= 16
      and (block_k_major // block_k) <= 16
  )
  if use_kv_stationary:
    return _flash_attention_impl_kv_stationary(
        q, k, v, save_residuals, causal, sm_scale,
        block_q, block_k_major, block_k, debug,
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
  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

      if sm_scale_block is not None:
        capped_logits *= sm_scale_block

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

      if sm_scale_block is not None:
        ds = ds * sm_scale_block

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
  sm_scale_block = (
      jnp.full((block_q_major, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

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

    if sm_scale_block is not None:
      capped_logits *= sm_scale_block

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

    if sm_scale_block is not None:
      ds = ds * sm_scale_block

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


# ---------------------------------------------------------------------------
# KV-stationary persistent forward kernel for v6e
# ---------------------------------------------------------------------------


def _flash_attention_kernel_kv_stationary(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    block_k_major: int,
    q_seq_len: int,
    kv_seq_len: int,
    mask_value: float,
):
  """KV-stationary kernel: one program per (batch, head).

  Keeps all Q-block accumulators live in VMEM and streams K/V once.
  Reorders loops to reuse each K/V sub-tile across all resident Q blocks
  before loading the next K/V sub-tile.
  """
  head_dim = q_ref.shape[-1]
  num_q_blocks = q_seq_len // block_q
  num_kv_major_blocks = kv_seq_len // block_k_major
  num_k_per_major = block_k_major // block_k

  m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
  l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
  acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  sm_scale_block = (
      jnp.full((block_q, block_k), sm_scale, jnp.float32)
      if sm_scale != 1.0
      else None
  )

  @pl.loop(0, num_kv_major_blocks, unroll=True)
  def _kv_major_loop(kv_major_idx):
    kv_major_start = kv_major_idx * block_k_major

    @pl.loop(0, num_k_per_major, unroll=True)
    def _k_loop(k_sub_idx):
      k_start = kv_major_start + k_sub_idx * block_k
      k = k_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)
      v = v_ref[0, 0, pl.ds(k_start, block_k), :].astype(jnp.float32)

      @pl.loop(0, num_q_blocks, unroll=True)
      def _q_loop(q_idx):
        q_start = q_idx * block_q
        if causal:
          # Check if block is fully masked (above diagonal): skip entirely
          # Bottom-left of Q-block row < top-left of K-block col means all masked
          q_block_bottom_row = (q_idx + 1) * block_q - 1
          k_block_top_col = k_start
          fully_masked = q_block_bottom_row < k_block_top_col

          # Check if block is fully unmasked (below diagonal):
          # Top-left of Q-block row >= bottom-right of K-block col means all valid
          q_block_top_row = q_start
          k_block_bottom_col = k_start + block_k - 1
          fully_unmasked = q_block_top_row >= k_block_bottom_col
        else:
          fully_masked = False
          fully_unmasked = True  # No mask needed at all

        @pl.when(~fully_masked & fully_unmasked)
        def _run_qk_no_mask():
          # Block is fully below diagonal: no mask needed
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        @pl.when(~fully_masked & ~fully_unmasked)
        def _run_qk_with_mask():
          # Block is on the diagonal: apply partial mask
          m_prev = m_scratch_ref[q_idx, :, :]
          l_prev = l_scratch_ref[q_idx, :, :]
          acc_prev = acc_scratch_ref[q_idx, :, :]
          q = q_ref[0, 0, pl.ds(q_start, block_q), :].astype(jnp.float32)

          s = jax.lax.dot_general(
              q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
          )

          if sm_scale_block is not None:
            s *= sm_scale_block

          # Use relative offset to avoid constructing full row/col index matrices
          # Condition: col + k_start <= row + q_start  <=>  col <= row + (q_start - k_start)
          rel_offset = q_start - k_start
          mask_shape = (block_q, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          s += jnp.where(col_ids <= row_ids + rel_offset, 0.0, mask_value)

          m_curr = jnp.max(s, axis=1)[:, None]
          m_next = jnp.maximum(m_prev, m_curr)

          block_k_repeats = block_k // MIN_BLOCK_SIZE
          p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
          alpha = jnp.exp(m_prev - m_next)
          l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

          o_curr = jax.lax.dot(
              p.astype(v.dtype), v, preferred_element_type=jnp.float32
          )

          head_dim_repeats = head_dim // MIN_BLOCK_SIZE
          if head_dim_repeats == 0:
            alpha_broad = alpha[:, :head_dim]
          else:
            alpha_broad = pltpu.repeat(alpha, head_dim_repeats, 1)

          acc_scratch_ref[q_idx, :, :] = acc_prev * alpha_broad + o_curr
          l_scratch_ref[q_idx, :, :] = l_next
          m_scratch_ref[q_idx, :, :] = m_next

        # When fully_masked is True, we skip entirely (do nothing).
        # m_scratch stays at -inf, l_scratch stays at 0, acc_scratch stays at 0.

  @pl.loop(0, num_q_blocks, unroll=True)
  def _store_loop(q_idx):
    q_start = q_idx * block_q
    l_final = l_scratch_ref[q_idx, :, :]
    l_inv = 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)

    head_dim_repeats = head_dim // MIN_BLOCK_SIZE
    if head_dim_repeats == 0:
      l_inv_broad = l_inv[:, :head_dim]
    else:
      l_inv_broad = pltpu.repeat(l_inv, head_dim_repeats, 1)

    out = acc_scratch_ref[q_idx, :, :] * l_inv_broad
    o_ref[0, 0, pl.ds(q_start, block_q), :] = out.astype(o_ref.dtype)

    if l_ref is not None:
      l_ref[0, 0, pl.ds(q_start, block_q), :] = l_final.astype(l_ref.dtype)
    if m_ref is not None:
      m_ref[0, 0, pl.ds(q_start, block_q), :] = m_scratch_ref[q_idx, :, :].astype(m_ref.dtype)


def _flash_attention_impl_kv_stationary(
    q,
    k,
    v,
    save_residuals,
    causal,
    sm_scale,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  """KV-stationary implementation: grid = (batch, heads)."""
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  num_q_blocks = q_seq_len // block_q

  grid = (batch_size, num_heads)

  def qkv_index_map(batch_index, head_index):
    return (batch_index, head_index, 0, 0)

  q_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)
  k_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  v_spec = pl.BlockSpec((1, 1, kv_seq_len, head_dim), qkv_index_map)
  o_spec = pl.BlockSpec((1, 1, q_seq_len, head_dim), qkv_index_map)

  in_specs = [q_spec, k_spec, v_spec]

  out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
  out_specs = [o_spec]

  # Scratch for all Q-block accumulators
  m_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  l_scratch = pltpu.VMEM((num_q_blocks, block_q, MIN_BLOCK_SIZE), jnp.float32)
  acc_scratch = pltpu.VMEM((num_q_blocks, block_q, head_dim), jnp.float32)
  scratch_shapes = [m_scratch, l_scratch, acc_scratch]

  if save_residuals:
    lm_spec = pl.BlockSpec((1, 1, q_seq_len, MIN_BLOCK_SIZE), qkv_index_map)
    out_specs = [o_spec, lm_spec, lm_spec]
    l_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m_out = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = [out_shape[0], l_out, m_out]
  else:
    out_specs = [o_spec, None, None]
    out_shape = [out_shape[0], None, None]

  kernel = functools.partial(
      _flash_attention_kernel_kv_stationary,
      causal=causal,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      block_k_major=block_k_major,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      mask_value=DEFAULT_MASK_VALUE,
  )

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
          dimension_semantics=("parallel", "parallel")
      ),
  )(q, k, v)

  if save_residuals:
    l, m = aux[-2], aux[-1]
    l = l[..., 0]
    m = m[..., 0]
    return (o, l, m)
  else:
    return o


CONFIG = {
    \'name\': \'pallas_flash_attention_llama70b\',
    \'model\': \'Llama-3.1-70B\',
    \'operator\': \'pallas_flash_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 128,
    \'atol\': 2e-3,
    \'rtol\': 2e-3,
}


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
    # v6e-tuned block sizes: the upstream get_default() hardcodes 128 for all
    # TPUs (see the TODO in BlockSizes.get_default). Larger tiles reduce K/V
    # HBM reloads from 16 to 2 for seq_len=2048. Other TPU generations may
    # need different values.
    block_sizes = BlockSizes(
        block_q=1024,
        block_k_major=1024,
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
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.663,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.663 ms\n{"correct": true, "latency": 0.663, "error": "", "all_times_ms": [0.652, 0.653, 0.654, 0.654, 0.654, 0.655, 0.655, 0.655, 0.656, 0.656, 0.656, 0.657, 0.657, 0.657, 0.657, 0.657, 0.658, 0.658, 0.658, 0.658, 0.659, 0.659, 0.659, 0.659, 0.659, 0.659, 0.66, 0.66, 0.66, 0.66, 0.661, 0.661, 0.661, 0.661, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.662, 0.663, 0.663, 0.663, 0.663, 0.663, 0.663, 0.664, 0.664, 0.664, 0.664, 0.664, 0.665, 0.665, 0.665, 0.665, 0.665, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.666, 0.667, 0.667, 0.667, 0.667, 0.667, 0.667, 0.668, 0.668, 0.668, 0.669, 0.669, 0.669, 0.67, 0.67, 0.671, 0.671, 0.672, 0.673, 0.674, 0.674, 0.675, 0.676, 0.677, 0.678, 0.679, 0.679, 0.68, 0.685, 0.694, 0.717, 0.749], "max_diff": 0.001953, "max_rel_diff": 0.000675}',
stderr='')