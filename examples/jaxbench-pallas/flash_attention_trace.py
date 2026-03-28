CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
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
  # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
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
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'atol': 2e-3,
    'rtol': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {
    # Autotuned (forward pass).
    'block_q': 2048,
    'block_k_major': 2048,
    'block_k': 512,
    # Not autotuned (batch=1, backward-only).
    'block_b': 1,
    'block_q_major_dkv': 128,
    'block_k_major_dkv': 128,
    'block_k_dkv': 128,
    'block_q_dkv': 128,
    'block_k_major_dq': 128,
    'block_k_dq': 128,
    'block_q_dq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS['block_q'],
        block_k_major=TUNED_PARAMS['block_k_major'],
        block_k=TUNED_PARAMS['block_k'],
        block_b=TUNED_PARAMS['block_b'],
        block_q_major_dkv=TUNED_PARAMS['block_q_major_dkv'],
        block_k_major_dkv=TUNED_PARAMS['block_k_major_dkv'],
        block_k_dkv=TUNED_PARAMS['block_k_dkv'],
        block_q_dkv=TUNED_PARAMS['block_q_dkv'],
        block_k_major_dq=TUNED_PARAMS['block_k_major_dq'],
        block_k_dq=TUNED_PARAMS['block_k_dq'],
        block_q_dq=TUNED_PARAMS['block_q_dq'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.625,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 0.625 ms\n{"correct": true, "latency": 0.625, "error": "", "all_times_ms": [0.614, 0.616, 0.617, 0.617, 0.617, 0.617, 0.618, 0.618, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.619, 0.62, 0.62, 0.62, 0.62, 0.621, 0.621, 0.621, 0.621, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.626, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.628, 0.628, 0.628, 0.628, 0.629, 0.629, 0.629, 0.629, 0.63, 0.63, 0.63, 0.63, 0.63, 0.631, 0.632, 0.632, 0.633, 0.633, 0.633, 0.633, 0.634, 0.634, 0.636, 0.637, 0.675], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 3) rewrite the algorithm to reduce total work**

### Why this is the best first change on v6e-1
The hot path in `workload()` is **causal forward attention** with:

- `block_q = 2048`
- `block_k_major = 2048`
- `block_k = 512`
- `q_seq_len = kv_seq_len = 2048`
- `ab is None`
- `segment_ids is None`

So each head executes one forward kernel that loads the full `Q/K/V` tiles, then does **4** `Q @ K_i^T` micro-steps of shape `[2048,128] x [512,128]^T`.

For `causal=True`, a large fraction of those products are guaranteed to be masked away:

- microtile 0: all 2048 rows can use it
- microtile 1: only rows `512:2048` can use it
- microtile 2: only rows `1024:2048` can use it
- microtile 3: only rows `1536:2048` can use it

But the current kernel still multiplies **all 2048 rows** against all 4 K tiles, then applies the causal mask afterward. That means a lot of MXU work is spent producing values that are immediately discarded.

On a **single-core v6e-1**, this forward path is primarily compute-heavy, so cutting matmul work is a better first move than trying to hide memory latency.

---

## Concrete plan

### 1) Add a specialized causal forward kernel for the current hot configuration
Keep the public API and function signatures unchanged. Add a fast path inside `_flash_attention_kernel` / `_flash_attention_impl` that is taken only when all of the following hold:

- `causal is True`
- `ab_tile_ref is None`
- `q_segment_ids_tile_ref is None`
- `kv_segment_ids_tile_ref is None`
- `block_q == block_k_major == kv_seq_len`
- `block_q % block_k == 0`

For this workload, that means a specialized kernel for the existing tuned case:
`2048 x 2048` causal attention with `block_k=512`.

If any condition fails, fall back to the current kernel unchanged.

---

### 2) Process the loaded `Q` tile in causal **q-subblocks** of size `block_k`
Instead of treating the full `Q[2048,128]` tile as one unit, split it **inside the kernel** into 4 subblocks of shape `[512,128]`.

So within the kernel body, do:

- `q_sub = 0`: rows `[0:512]`
- `q_sub = 1`: rows `[512:1024]`
- `q_sub = 2`: rows `[1024:1536]`
- `q_sub = 3`: rows `[1536:2048]`

For each `q_sub`, maintain its own local online-softmax state:

- `m_scratch_ref`
- `l_scratch_ref`
- `acc_scratch_ref`

These should be reinitialized for each subblock and written back to the correct slice of `o_tile_ref` (and `l_ref` / `m_ref` if residuals are requested).

---

### 3) Only visit K/V microtiles that can actually contribute
For subblock `q_sub`, only iterate `k_sub = 0 .. q_sub`.

That gives:

- `q_sub=0` → 1 K/V microtile
- `q_sub=1` → 2 K/V microtiles
- `q_sub=2` → 3 K/V microtiles
- `q_sub=3` → 4 K/V microtiles

Total active microtiles per head:
- **current:** 4 q-rows-groups are all implicitly multiplied against 4 K tiles = 16 equivalent subproblems
- **new:** `1 + 2 + 3 + 4 = 10`

So the plan removes about **37.5%** of the causal forward matmul work, while keeping the same math.

---

### 4) Apply the causal mask only on the **diagonal** subproblem
For `k_sub < q_sub`, every column is strictly to the left of the current query block, so those tiles are fully valid and need **no causal mask generation**.

Only when `k_sub == q_sub` do we need the triangular mask within the `[512,512]` subproblem.

That means:

- less `broadcasted_iota`
- less `where`
- less wasted VPU work

while preserving exact causal semantics.

---

### 5) Shrink scratch buffers to the q-subblock size
Because the fast path processes one `512`-row query chunk at a time, change the forward scratch allocations in `_flash_attention_impl` for this path from:

- `(block_b, 1, block_q, MIN_BLOCK_SIZE)`
- `(block_b, 1, block_q, MIN_BLOCK_SIZE)`
- `(block_b, 1, block_q, head_dim)`

to:

- `(block_b, 1, block_k, MIN_BLOCK_SIZE)`
- `(block_b, 1, block_k, MIN_BLOCK_SIZE)`
- `(block_b, 1, block_k, head_dim)`

For the current tuned case, that changes scratch from 2048-row to 512-row buffers.

This is still part of the same algorithm rewrite: we are changing the unit of computation from “full q tile” to “causal q subblock”.

It also helps VMEM pressure on v6e-1.

---

## What to change in code

### In `_flash_attention_kernel`
Dispatch to a new helper, e.g. `_flash_attention_kernel_single_batch_causal_triangular`, when the fast-path conditions hold. Keep the existing helpers for fallback.

### In the new helper
For each batch slice:

1. Loop over `q_sub` in `range(block_q // block_k)`  
2. Read the q slice from the `Ref`:
   - use `q_tile_ref[...]` with `pl.dslice(...)`
3. Zero/init the local scratch refs for that q subblock
4. Loop over `k_sub in range(q_sub + 1)`
5. Read `k` and `v` slices via `Ref` indexing
6. Compute:
   - `s = dot_general(q, k, ..., preferred_element_type=jnp.float32)`
   - apply `sm_scale`
   - apply causal mask only if `k_sub == q_sub`
   - do the same online softmax update in float32
   - accumulate `p @ v` into `acc_scratch_ref`
7. Write the finished slice back:
   - `o_tile_ref[...] = acc.astype(o_tile_ref.dtype)`
   - `l_ref[...] = ...` and `m_ref[...] = ...` when requested

All arithmetic still happens on arrays loaded from refs, and stores go back through refs.

---

## Why this stays correct
This is semantically equivalent because it does **not** change the softmax recurrence or output definition. It only avoids evaluating score blocks that are known to be masked out.

The optimized path still:

- accumulates scores in `float32`
- computes softmax normalization in `float32`
- accumulates outputs in `float32`
- casts only on final output store

So it respects TPU numerical requirements and should match the original within the same small tolerance.

---

## Why this is better than changing the grid first
A grid-level retile of `block_q` to 512 would also expose the triangular structure, but it would cause the same K/V tiles to be reloaded from HBM for multiple q blocks.

This plan avoids that by:

- keeping the existing full-tile HBM→VMEM load pattern
- exploiting causality **inside** the loaded tile
- cutting compute without increasing K/V traffic

That makes it a strong phase-1 change for v6e-1.

---

## Expected impact
For the exact hot configuration in `workload()`:

- same public function signatures
- same outputs
- roughly **37.5% less causal forward matmul work**
- less mask-generation overhead
- smaller scratch footprint in VMEM

That should reduce the measured forward latency meaningfully on v6e-1 before touching buffering or prefetching in later phases.''',
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
  # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_causal_triangular(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Always None in this path
    q_segment_ids_tile_ref,  # Always None in this path
    kv_segment_ids_tile_ref,  # Always None in this path
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  """Optimized causal forward kernel that skips fully-masked subproblems.
  
  This kernel is used when:
  - causal=True
  - ab_tile_ref is None
  - segment_ids are None
  - block_q == block_k_major == kv_seq_len (single grid cell in kv dimension)
  - block_q % block_k == 0
  
  Instead of computing all Q rows against all K/V tiles and masking afterward,
  we split Q into subblocks of size block_k and only compute the lower-triangular
  subproblems that contribute to the causal attention.
  """
  del ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref  # Unused
  
  block_b = q_tile_ref.shape[0]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[3]
  num_q_subs = block_q // block_k
  
  for batch_idx in range(block_b):
    bidx = (batch_idx, 0)
    
    # Process each q subblock
    for q_sub in range(num_q_subs):
      q_start = q_sub * block_k
      q = q_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :]  # [block_k, head_dim]
      
      # Initialize scratch for this q subblock
      m_scratch_ref[bidx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[bidx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
      acc_scratch_ref[bidx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)
      
      # Only iterate over k subblocks that can contribute (k_sub <= q_sub for causal)
      for k_sub in range(q_sub + 1):
        k_start = k_sub * block_k
        
        m_prev = m_scratch_ref[bidx]
        l_prev = l_scratch_ref[bidx]
        
        k = k_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s *= sm_scale
        
        # Apply causal mask only on diagonal subproblem (k_sub == q_sub)
        if k_sub == q_sub:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids += q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids += k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
        
        alpha = jnp.exp(m_prev - m_next)
        l_corr = alpha * l_prev
        l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if hd_rem:
          if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
          else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )
        else:
          l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        
        l_scratch_ref[bidx] = l_next
        m_scratch_ref[bidx] = m_next
        
        l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        acc_scratch_ref[bidx] *= l_broadcast(l_corr * l_next_inv_safe)
        
        v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        )
        acc_scratch_ref[bidx] += o_curr * l_broadcast(l_next_inv_safe)
      
      # Write output for this q subblock
      o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = acc_scratch_ref[bidx].astype(o_tile_ref.dtype)
      if l_ref is not None:
        l_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = l_scratch_ref[bidx].astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = m_scratch_ref[bidx].astype(m_ref.dtype)


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

  # Check if we can use the optimized causal triangular kernel
  use_causal_triangular = (
      causal
      and ab is None
      and segment_ids is None
      and block_q == block_k_major
      and block_k_major == kv_seq_len
      and block_q % block_k == 0
      and block_k != kv_seq_len  # Not the single-step case
  )
  
  if use_causal_triangular:
    kernel = functools.partial(
        _flash_attention_kernel_causal_triangular,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        mask_value=DEFAULT_MASK_VALUE,
    )
    # Use smaller scratch buffers sized for q subblocks (block_k rows)
    m_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_k, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    if block_k != kv_seq_len:
      m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
      scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
      scratch_shapes = []
  
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

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
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'atol': 2e-3,
    'rtol': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {
    # Autotuned (forward pass).
    'block_q': 2048,
    'block_k_major': 2048,
    'block_k': 512,
    # Not autotuned (batch=1, backward-only).
    'block_b': 1,
    'block_q_major_dkv': 128,
    'block_k_major_dkv': 128,
    'block_k_dkv': 128,
    'block_q_dkv': 128,
    'block_k_major_dq': 128,
    'block_k_dq': 128,
    'block_q_dq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS['block_q'],
        block_k_major=TUNED_PARAMS['block_k_major'],
        block_k=TUNED_PARAMS['block_k'],
        block_b=TUNED_PARAMS['block_b'],
        block_q_major_dkv=TUNED_PARAMS['block_q_major_dkv'],
        block_k_major_dkv=TUNED_PARAMS['block_k_major_dkv'],
        block_k_dkv=TUNED_PARAMS['block_k_dkv'],
        block_q_dkv=TUNED_PARAMS['block_q_dkv'],
        block_k_major_dq=TUNED_PARAMS['block_k_major_dq'],
        block_k_dq=TUNED_PARAMS['block_k_dq'],
        block_q_dq=TUNED_PARAMS['block_q_dq'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.445,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.445 ms\n{"correct": true, "latency": 0.445, "error": "", "all_times_ms": [0.439, 0.44, 0.44, 0.44, 0.441, 0.441, 0.441, 0.441, 0.441, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.442, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.443, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.446, 0.446, 0.446, 0.446, 0.446, 0.446, 0.446, 0.446, 0.446, 0.446, 0.447, 0.447, 0.447, 0.447, 0.447, 0.447, 0.447, 0.447, 0.447, 0.448, 0.448, 0.448, 0.448, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.45, 0.45, 0.45, 0.45, 0.451, 0.451, 0.452, 0.452, 0.455, 0.456, 0.467, 0.469, 0.47], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr=''),
plan='''

Looking at this Flash Attention implementation, I can see there's an optimization opportunity around mask computation. In the inner KV-block loop within the forward kernels (`_flash_attention_kernel_single_batch` and `_flash_attention_kernel_causal_triangular`), the code recomputes `broadcasted_iota` shapes and creates mask indices on every iteration, even though many of these computations are invariant across iterations.

However, the most impactful change I can make is to **hoist the invariant mask shape computations** (the `broadcasted_iota` tensors for row/col IDs) **outside the inner k-block loop**. Currently, these are recomputed for every KV block iteration, but the shape is constant and only the offsets change.

Let me create a plan:

## Optimization Plan: Hoist Invariant Mask Computations

**Strategy**: 14. Hoist invariant mask shape computations and broadcast patterns outside the inner k-block loop

**Target**: The `_flash_attention_kernel_single_batch` function, specifically the inner loop that iterates over KV blocks.

**Current inefficiency**: Inside the `@pl.loop(0, block_k_major // block_k, unroll=True)` loop, the code creates:
- `row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)` 
- `col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)`

These are recreated in every iteration even though they have the same shape. Only the offsets added (`row_ids += q_seq_idx * block_q` and `col_ids += kv_seq_idx * block_k_major + start_k`) change per iteration.

**Solution**: Move the `broadcasted_iota` creation outside the loop and reuse the base tensors, adding the iteration-specific offsets inside the loop.

**Expected benefit**: Reduces overhead from repeated tensor creation in the inner loop, which runs `block_k_major // block_k` times (e.g., 4 times for block_k_major=2048, block_k=512).

This is a straightforward optimization that should reduce instruction overhead without changing the algorithm or affecting numerical accuracy.


## Optimization Plan

**Selected Strategy**: 14. Hoist invariant mask shape computations and broadcast patterns outside the inner k-block loop

**Target Function**: `_flash_attention_kernel_single_batch`

### Current Inefficiency

In the inner KV-block loop (`@pl.loop(0, block_k_major // block_k, unroll=True)`), the code computes:

```python
row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
row_ids += q_seq_idx * block_q
col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
col_ids += kv_seq_idx * block_k_major + start_k
causal_mask = col_ids <= row_ids
```

The `broadcasted_iota` calls create new tensors in **every iteration** of the inner loop, even though the shapes are constant. Only the offsets (`+ start_k`, `+ kv_seq_idx * block_k_major`, etc.) vary per iteration.

For a typical configuration with `block_k_major=2048` and `block_k=512`, this loop runs 4 times per Q block. For larger sequences, this repeated tensor creation adds significant overhead.

### Solution

Move the `broadcasted_iota` creation **before** the inner loop, storing the base tensors. Inside the loop, only compute the offset-adjusted versions needed for that iteration.

This involves:
1. Before the loop: Create base `row_ids` and `col_ids` using `broadcasted_iota` with shape `(block_q, block_k)`
2. Inside the loop: Clone the base tensors and add the iteration-specific offset (`start_k`)

Since the offsets are simple additions, we can use `jnp.add(base, offset, ...)` which is cheap, rather than recomputing `broadcasted_iota` from scratch each time.''',
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
  # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_causal_triangular(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Always None in this path
    q_segment_ids_tile_ref,  # Always None in this path
    kv_segment_ids_tile_ref,  # Always None in this path
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  """Optimized causal forward kernel that skips fully-masked subproblems.
  
  This kernel is used when:
  - causal=True
  - ab_tile_ref is None
  - segment_ids are None
  - block_q == block_k_major == kv_seq_len (single grid cell in kv dimension)
  - block_q % block_k == 0
  
  Instead of computing all Q rows against all K/V tiles and masking afterward,
  we split Q into subblocks of size block_k and only compute the lower-triangular
  subproblems that contribute to the causal attention.
  """
  del ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref  # Unused
  
  block_b = q_tile_ref.shape[0]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[3]
  num_q_subs = block_q // block_k
  
  for batch_idx in range(block_b):
    bidx = (batch_idx, 0)
    
    # Process each q subblock
    for q_sub in range(num_q_subs):
      q_start = q_sub * block_k
      q = q_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :]  # [block_k, head_dim]
      
      # Initialize scratch for this q subblock
      m_scratch_ref[bidx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[bidx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
      acc_scratch_ref[bidx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)
      
      # Only iterate over k subblocks that can contribute (k_sub <= q_sub for causal)
      for k_sub in range(q_sub + 1):
        k_start = k_sub * block_k
        
        m_prev = m_scratch_ref[bidx]
        l_prev = l_scratch_ref[bidx]
        
        k = k_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s *= sm_scale
        
        # Apply causal mask only on diagonal subproblem (k_sub == q_sub)
        if k_sub == q_sub:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids += q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids += k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
        
        alpha = jnp.exp(m_prev - m_next)
        l_corr = alpha * l_prev
        l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if hd_rem:
          if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
          else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )
        else:
          l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        
        l_scratch_ref[bidx] = l_next
        m_scratch_ref[bidx] = m_next
        
        l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        acc_scratch_ref[bidx] *= l_broadcast(l_corr * l_next_inv_safe)
        
        v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        )
        acc_scratch_ref[bidx] += o_curr * l_broadcast(l_next_inv_safe)
      
      # Write output for this q subblock
      o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = acc_scratch_ref[bidx].astype(o_tile_ref.dtype)
      if l_ref is not None:
        l_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = l_scratch_ref[bidx].astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = m_scratch_ref[bidx].astype(m_ref.dtype)


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
    mask_shape = (block_q, block_k)
    base_row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    base_col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    row_ids = base_row_ids + q_seq_idx * block_q
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
        col_ids = base_col_ids + kv_seq_idx * block_k_major + start_k
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

  # Check if we can use the optimized causal triangular kernel
  use_causal_triangular = (
      causal
      and ab is None
      and segment_ids is None
      and block_q == block_k_major
      and block_k_major == kv_seq_len
      and block_q % block_k == 0
      and block_k != kv_seq_len  # Not the single-step case
  )
  
  if use_causal_triangular:
    kernel = functools.partial(
        _flash_attention_kernel_causal_triangular,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        mask_value=DEFAULT_MASK_VALUE,
    )
    # Use smaller scratch buffers sized for q subblocks (block_k rows)
    m_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_k, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    if block_k != kv_seq_len:
      m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
      scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
      scratch_shapes = []
  
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

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
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'atol': 2e-3,
    'rtol': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {
    # Autotuned (forward pass).
    'block_q': 2048,
    'block_k_major': 2048,
    'block_k': 512,
    # Not autotuned (batch=1, backward-only).
    'block_b': 1,
    'block_q_major_dkv': 128,
    'block_k_major_dkv': 128,
    'block_k_dkv': 128,
    'block_q_dkv': 128,
    'block_k_major_dq': 128,
    'block_k_dq': 128,
    'block_q_dq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS['block_q'],
        block_k_major=TUNED_PARAMS['block_k_major'],
        block_k=TUNED_PARAMS['block_k'],
        block_b=TUNED_PARAMS['block_b'],
        block_q_major_dkv=TUNED_PARAMS['block_q_major_dkv'],
        block_k_major_dkv=TUNED_PARAMS['block_k_major_dkv'],
        block_k_dkv=TUNED_PARAMS['block_k_dkv'],
        block_q_dkv=TUNED_PARAMS['block_q_dkv'],
        block_k_major_dq=TUNED_PARAMS['block_k_major_dq'],
        block_k_dq=TUNED_PARAMS['block_k_dq'],
        block_q_dq=TUNED_PARAMS['block_q_dq'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.443,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='gpt-5.4',
stdout='Latency: 0.443 ms\n{"correct": true, "latency": 0.443, "error": "", "all_times_ms": [0.431, 0.434, 0.434, 0.435, 0.435, 0.435, 0.435, 0.436, 0.436, 0.437, 0.437, 0.437, 0.437, 0.437, 0.438, 0.438, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.441, 0.442, 0.442, 0.442, 0.442, 0.442, 0.443, 0.443, 0.443, 0.443, 0.443, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.444, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.445, 0.446, 0.446, 0.446, 0.446, 0.447, 0.447, 0.447, 0.447, 0.447, 0.447, 0.448, 0.448, 0.448, 0.448, 0.448, 0.45, 0.45, 0.451, 0.452, 0.452, 0.454, 0.457, 0.457, 0.457, 0.462, 0.467, 0.47], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr=''),
plan='''A good phase-4 strategy here is:

## **Reformulate the online softmax state so `acc_scratch_ref` stores the unnormalized numerator, not the normalized partial output**

This is **different from the listed strategies**. It does not change pipelining, buffering, block sizes, or memory placement. It changes only the algebra inside the forward kernel.

---

## Why this code is currently inefficient

In both multi-step forward kernels:

- `_flash_attention_kernel_single_batch`
- `_flash_attention_kernel_causal_triangular`

the code maintains `acc_scratch_ref` as a **normalized** running output.

That forces this work on **every K subblock**:

```python
l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
acc_scratch_ref[...] *= l_broadcast(l_corr * l_next_inv_safe)
acc_scratch_ref[...] += o_curr * l_broadcast(l_next_inv_safe)
```

So per K tile, you pay for:

- one full `block_q x head_dim` scale of the old accumulator
- one full `block_q x head_dim` scale of the new `o_curr`
- repeated broadcasting of `1 / l_next` across `head_dim`

On TPU v6e, those are VPU-side elementwise ops on large tiles. The MXU matmul is still the main cost, but this extra VPU work sits in the inner loop and adds noticeable latency.

For your tuned forward path:

- `block_q = 2048`
- `block_k_major = 2048`
- `block_k = 512`
- `head_dim = 128`
- `causal=True`

the optimized causal triangular kernel runs multiple K substeps per Q subblock. That means this normalization work is repeated many times.

---

## Better recurrence

Instead of storing normalized output, store the **unnormalized numerator**:

\[
\text{acc\_num} = \sum_j \exp(s_j - m)\, v_j
\]

Then the online update becomes:

- `m_next = max(m_prev, m_curr)`
- `alpha = exp(m_prev - m_next)`
- `p = exp(s - m_next)`
- `l_next = alpha * l_prev + sum(p)`
- `acc_next = alpha * acc_prev + p @ v`

Only at the **end** do you normalize once:

\[
o = \frac{\text{acc\_num}}{l}
\]

This is mathematically equivalent to the current algorithm, just a cheaper state representation.

---

## What to change

### 1. `_flash_attention_kernel_single_batch`

Replace the per-step normalized update:

```python
l_corr = alpha * l_prev
l_next = jnp.sum(p, axis=1)[:, None] + l_corr

l_scratch_ref[batch_idx] = l_next
m_scratch_ref[batch_idx] = m_next

l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)

v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
o_curr = jax.lax.dot(
    p.astype(v.dtype), v, preferred_element_type=jnp.float32
)
acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)
```

with:

```python
l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
o_curr = jax.lax.dot(
    p.astype(v.dtype), v, preferred_element_type=jnp.float32
)

acc_prev = acc_scratch_ref[batch_idx]
acc_scratch_ref[batch_idx] = acc_prev * l_broadcast(alpha) + o_curr

l_scratch_ref[batch_idx] = l_next
m_scratch_ref[batch_idx] = m_next
```

Then in `store_output()` normalize once:

```python
@pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
def store_output():
  l_final = l_scratch_ref[batch_idx]
  inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
  o_tile_ref[batch_idx] = (
      acc_scratch_ref[batch_idx] * l_broadcast(inv_l)
  ).astype(o_tile_ref.dtype)
  if l_ref is not None:
    l_ref[batch_idx] = l_final.astype(l_ref.dtype)
  if m_ref is not None:
    m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)
```

---

### 2. `_flash_attention_kernel_causal_triangular`

Do the same transformation inside the `k_sub` loop.

Replace:

```python
l_corr = alpha * l_prev
l_next = jnp.sum(p, axis=1)[:, None] + l_corr

l_scratch_ref[bidx] = l_next
m_scratch_ref[bidx] = m_next

l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
acc_scratch_ref[bidx] *= l_broadcast(l_corr * l_next_inv_safe)

v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
o_curr = jax.lax.dot(
    p.astype(v.dtype), v, preferred_element_type=jnp.float32
)
acc_scratch_ref[bidx] += o_curr * l_broadcast(l_next_inv_safe)
```

with:

```python
l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev

v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
o_curr = jax.lax.dot(
    p.astype(v.dtype), v, preferred_element_type=jnp.float32
)

acc_prev = acc_scratch_ref[bidx]
acc_scratch_ref[bidx] = acc_prev * l_broadcast(alpha) + o_curr

l_scratch_ref[bidx] = l_next
m_scratch_ref[bidx] = m_next
```

Then when writing each Q subblock, normalize once:

```python
l_final = l_scratch_ref[bidx]
inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)

o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = (
    acc_scratch_ref[bidx] * l_broadcast(inv_l)
).astype(o_tile_ref.dtype)
```

---

## Why this should help on TPU v6e

This change is attractive on v6e because it:

- **reduces inner-loop VPU work**
- **reduces broadcast pressure across `head_dim`**
- **shortens the dependency chain inside each K step**
- **does not increase VMEM usage**

The scratch shapes stay the same:

- `m_scratch_ref`
- `l_scratch_ref`
- `acc_scratch_ref`

So it respects the ~16 MB VMEM budget. You are only changing what `acc_scratch_ref` means.

For the current tuned forward path, this removes one large `head_dim`-wide scaling per K substep and pushes normalization to the epilogue. That is exactly the kind of improvement that helps when the kernel is already fairly optimized structurally but still has too much elementwise work in the hot loop.

---

## Why it is still semantically correct

This is the same online softmax algorithm, just expressed with a different state variable.

You still preserve:

- exact causal masking behavior
- the same `m` recurrence
- the same `l` recurrence
- the same output `o`
- the same public function names and signatures

Numerically, results may differ slightly because the order of floating-point operations changes, but it should remain within the same small tolerance regime as the existing kernel.

---

## Scope recommendation

For this phase, I would apply this **only to the forward multi-step kernels**:

- `_flash_attention_kernel_single_batch`
- `_flash_attention_kernel_causal_triangular`

and leave:

- `_flash_attention_kernel_single_batch_single_step`
- backward kernels

unchanged for now.

That keeps the change focused, low-risk, and directly aligned with the measured forward latency.''',
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
  # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_causal_triangular(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Always None in this path
    q_segment_ids_tile_ref,  # Always None in this path
    kv_segment_ids_tile_ref,  # Always None in this path
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  """Optimized causal forward kernel that skips fully-masked subproblems.
  
  This kernel is used when:
  - causal=True
  - ab_tile_ref is None
  - segment_ids are None
  - block_q == block_k_major == kv_seq_len (single grid cell in kv dimension)
  - block_q % block_k == 0
  
  Instead of computing all Q rows against all K/V tiles and masking afterward,
  we split Q into subblocks of size block_k and only compute the lower-triangular
  subproblems that contribute to the causal attention.
  """
  del ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref  # Unused
  
  block_b = q_tile_ref.shape[0]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[3]
  num_q_subs = block_q // block_k
  
  for batch_idx in range(block_b):
    bidx = (batch_idx, 0)
    
    # Process each q subblock
    for q_sub in range(num_q_subs):
      q_start = q_sub * block_k
      q = q_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :]  # [block_k, head_dim]
      
      # Initialize scratch for this q subblock
      m_scratch_ref[bidx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[bidx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
      acc_scratch_ref[bidx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)
      
      # Only iterate over k subblocks that can contribute (k_sub <= q_sub for causal)
      for k_sub in range(q_sub + 1):
        k_start = k_sub * block_k
        
        m_prev = m_scratch_ref[bidx]
        l_prev = l_scratch_ref[bidx]
        
        k = k_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s *= sm_scale
        
        # Apply causal mask only on diagonal subproblem (k_sub == q_sub)
        if k_sub == q_sub:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids += q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids += k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
        
        alpha = jnp.exp(m_prev - m_next)
        l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if hd_rem:
          if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
          else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )
        else:
          l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        
        v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        )
        
        acc_prev = acc_scratch_ref[bidx]
        acc_scratch_ref[bidx] = acc_prev * l_broadcast(alpha) + o_curr
        
        l_scratch_ref[bidx] = l_next
        m_scratch_ref[bidx] = m_next
      
      # Normalize and write output for this q subblock
      l_final = l_scratch_ref[bidx]
      head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
      if hd_rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )
      else:
        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
      o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = (acc_scratch_ref[bidx] * l_broadcast(inv_l)).astype(o_tile_ref.dtype)
      if l_ref is not None:
        l_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = l_scratch_ref[bidx].astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = m_scratch_ref[bidx].astype(m_ref.dtype)


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
    mask_shape = (block_q, block_k)
    base_row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    base_col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    row_ids = base_row_ids + q_seq_idx * block_q
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
        col_ids = base_col_ids + kv_seq_idx * block_k_major + start_k
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
      l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )

      acc_prev = acc_scratch_ref[batch_idx]
      acc_scratch_ref[batch_idx] = acc_prev * l_broadcast(alpha) + o_curr

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    l_final = l_scratch_ref[batch_idx]
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda l: l[:, :head_dim]
      else:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )
    inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(inv_l)).astype(o_tile_ref.dtype)
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

  # Check if we can use the optimized causal triangular kernel
  use_causal_triangular = (
      causal
      and ab is None
      and segment_ids is None
      and block_q == block_k_major
      and block_k_major == kv_seq_len
      and block_q % block_k == 0
      and block_k != kv_seq_len  # Not the single-step case
  )
  
  if use_causal_triangular:
    kernel = functools.partial(
        _flash_attention_kernel_causal_triangular,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        mask_value=DEFAULT_MASK_VALUE,
    )
    # Use smaller scratch buffers sized for q subblocks (block_k rows)
    m_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_k, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    if block_k != kv_seq_len:
      m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
      scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
      scratch_shapes = []
  
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

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
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'atol': 2e-3,
    'rtol': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {
    # Autotuned (forward pass).
    'block_q': 2048,
    'block_k_major': 2048,
    'block_k': 512,
    # Not autotuned (batch=1, backward-only).
    'block_b': 1,
    'block_q_major_dkv': 128,
    'block_k_major_dkv': 128,
    'block_k_dkv': 128,
    'block_q_dkv': 128,
    'block_k_major_dq': 128,
    'block_k_dq': 128,
    'block_q_dq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS['block_q'],
        block_k_major=TUNED_PARAMS['block_k_major'],
        block_k=TUNED_PARAMS['block_k'],
        block_b=TUNED_PARAMS['block_b'],
        block_q_major_dkv=TUNED_PARAMS['block_q_major_dkv'],
        block_k_major_dkv=TUNED_PARAMS['block_k_major_dkv'],
        block_k_dkv=TUNED_PARAMS['block_k_dkv'],
        block_q_dkv=TUNED_PARAMS['block_q_dkv'],
        block_k_major_dq=TUNED_PARAMS['block_k_major_dq'],
        block_k_dq=TUNED_PARAMS['block_k_dq'],
        block_q_dq=TUNED_PARAMS['block_q_dq'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.432,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.432 ms\n{"correct": true, "latency": 0.432, "error": "", "all_times_ms": [0.421, 0.422, 0.423, 0.424, 0.424, 0.424, 0.424, 0.425, 0.425, 0.426, 0.426, 0.427, 0.427, 0.427, 0.427, 0.427, 0.427, 0.428, 0.428, 0.428, 0.428, 0.428, 0.429, 0.429, 0.429, 0.429, 0.429, 0.429, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.436, 0.436, 0.436, 0.436, 0.437, 0.438, 0.438, 0.438, 0.438, 0.438, 0.439, 0.44, 0.441, 0.443, 0.445, 0.448, 0.454, 0.456], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr=''),
plan='''## Performance Optimization Plan

### Analysis of Inefficiency

The current forward pass implementation for the causal attention kernel uses a generic "masked" approach. Even though the attention is causal (lower-triangular), the kernel iterates over the full grid of Q blocks x KV blocks. For iterations where the KV block is entirely "above" the diagonal (and thus fully masked), the code executes a significant amount of "dead" computation:
1. It loads full tiles of `k_tile_ref`, `v_tile_ref`, and `q_tile_ref`.
2. It performs matrix multiplications to compute attention logits `s = q @ k.T`.
3. It applies large masking values.
4. It performs expensive exponential operations on masked values (`jnp.exp(s - m_next)`).
5. It performs reduction operations (`jnp.max`, `jnp.sum`) and updates accumulator states — all of which are ultimately discarded or yield identity transformations (multiply by ~0, add ~0).

This results in wasted memory bandwidth and compute cycles on the MXU and VPU.

### Selected Strategy

**Strategy 3: Eliminate redundant computation.**

We will implement an optimized causal triangular kernel that skips fully-masked subproblems entirely, only computing attention for the lower-triangular blocks that contribute to the output.

### Implementation Plan

1.  **Identify the optimization condition**: We can apply this when `causal=True`, `ab=None`, `segment_ids=None`, `block_q == block_k_major`, `block_q % block_k == 0`, and the sequence fits in a single grid cell (`block_q == kv_seq_len` or close to it such that inefficiency is high).

2.  **Create a specialized kernel**: Implement `_flash_attention_kernel_causal_triangular`.
    -   Instead of iterating over K/V blocks with a `should_run` mask that computes then discards, this kernel will directly iterate only over necessary K/V blocks.
    -   It splits the Q block into "subblocks" of size `block_k`.
    -   For each Q subblock starting at row `q_start`, it iterates K/V blocks `k_start` only up to `q_start` (the lower triangular boundary).
    -   This eliminates the loading and processing of K/V tiles that are entirely above the diagonal.

3.  **Adjust scratch memory**: The optimized kernel accumulates into smaller scratch buffers sized to the Q subblock size (`block_k` rows) rather than the full `block_q`, reducing VMEM pressure from `(block_b, 1, block_q, ...)` to `(block_b, 1, block_k, ...)`.

4.  **Integrate into dispatch logic**: Update `_flash_attention_impl` to select this optimized path when conditions are met.

### Application to Code

```python
def _flash_attention_kernel_causal_triangular(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Always None in this path
    q_segment_ids_tile_ref,  # Always None in this path
    kv_segment_ids_tile_ref,  # Always None in this path
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  """Optimized causal forward kernel that skips fully-masked subproblems.
  
  This kernel is used when:
  - causal=True
  - ab_tile_ref is None
  - segment_ids are None
  - block_q == block_k_major == kv_seq_len (single grid cell in kv dimension)
  - block_q % block_k == 0
  
  Instead of computing all Q rows against all K/V tiles and masking afterward,
  we split Q into subblocks of size block_k and only compute the lower-triangular
  subproblems that contribute to the causal attention.
  """
  del ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref  # Unused
  
  block_b = q_tile_ref.shape[0]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[3]
  num_q_subs = block_q // block_k
  
  for batch_idx in range(block_b):
    bidx = (batch_idx, 0)
    
    # Process each q subblock
    for q_sub in range(num_q_subs):
      q_start = q_sub * block_k
      q = q_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :]  # [block_k, head_dim]
      
      # Initialize scratch for this q subblock
      m_scratch_ref[bidx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[bidx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
      acc_scratch_ref[bidx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)
      
      # Only iterate over k subblocks that can contribute (k_sub <= q_sub for causal)
      for k_sub in range(q_sub + 1):
        k_start = k_sub * block_k
        
        m_prev = m_scratch_ref[bidx]
        l_prev = l_scratch_ref[bidx]
        
        k = k_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s *= sm_scale
        
        # Apply causal mask only on diagonal subproblem (k_sub == q_sub)
        if k_sub == q_sub:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids += q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids += k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
        
        alpha = jnp.exp(m_prev - m_next)
        l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if hd_rem:
          if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
          else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )
        else:
          l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        
        v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        )
        
        acc_prev = acc_scratch_ref[bidx]
        acc_scratch_ref[bidx] = acc_prev * l_broadcast(alpha) + o_curr
        
        l_scratch_ref[bidx] = l_next
        m_scratch_ref[bidx] = m_next
      
      # Normalize and write output for this q subblock
      l_final = l_scratch_ref[bidx]
      head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
      if hd_rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )
      else:
        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
      o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = (acc_scratch_ref[bidx] * l_broadcast(inv_l)).astype(o_tile_ref.dtype)
      if l_ref is not None:
        l_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = l_scratch_ref[bidx].astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = m_scratch_ref[bidx].astype(m_ref.dtype)
```

Finally, update `_flash_attention_impl` to check the conditions and dispatch to this kernel:

```python
  # Check if we can use the optimized causal triangular kernel
  use_causal_triangular = (
      causal
      and ab is None
      and segment_ids is None
      and block_q == block_k_major
      and block_k_major == kv_seq_len
      and block_q % block_k == 0
      and block_k != kv_seq_len  # Not the single-step case
  )
  
  if use_causal_triangular:
    kernel = functools.partial(
        _flash_attention_kernel_causal_triangular,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        mask_value=DEFAULT_MASK_VALUE,
    )
    # Use smaller scratch buffers sized for q subblocks (block_k rows)
    m_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_k, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    if block_k != kv_seq_len:
      m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
      scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
      scratch_shapes = []
```

This change specifically addresses the redundant computation in the causal forward pass by restructuring the loop order to skip obviously-masked work, directly applying Strategy 3.''',
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
  # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
  if kwargs["block_k"] == kwargs["kv_seq_len"]:
    kernel = _flash_attention_kernel_single_batch_single_step
  else:
    kernel = _flash_attention_kernel_single_batch
  for batch_idx in range(block_b):
    kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_causal_triangular(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Always None in this path
    q_segment_ids_tile_ref,  # Always None in this path
    kv_segment_ids_tile_ref,  # Always None in this path
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
  """Optimized causal forward kernel that skips fully-masked subproblems.
  
  This kernel is used when:
  - causal=True
  - ab_tile_ref is None
  - segment_ids are None
  - block_q == block_k_major == kv_seq_len (single grid cell in kv dimension)
  - block_q % block_k == 0
  
  Instead of computing all Q rows against all K/V tiles and masking afterward,
  we split Q into subblocks of size block_k and only compute the lower-triangular
  subproblems that contribute to the causal attention.
  """
  del ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref  # Unused
  
  block_b = q_tile_ref.shape[0]
  block_q = q_tile_ref.shape[2]
  head_dim = q_tile_ref.shape[3]
  num_q_subs = block_q // block_k
  
  for batch_idx in range(block_b):
    bidx = (batch_idx, 0)
    
    # Process each q subblock
    for q_sub in range(num_q_subs):
      q_start = q_sub * block_k
      q = q_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :]  # [block_k, head_dim]
      
      # Initialize scratch for this q subblock
      m_scratch_ref[bidx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[bidx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
      acc_scratch_ref[bidx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)
      
      # Only iterate over k subblocks that can contribute (k_sub <= q_sub for causal)
      for k_sub in range(q_sub + 1):
        k_start = k_sub * block_k
        
        m_prev = m_scratch_ref[bidx]
        l_prev = l_scratch_ref[bidx]
        
        k = k_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]  # [block_k, head_dim]
        
        s = jax.lax.dot_general(
            q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )  # [block_k, block_k]
        
        if sm_scale != 1.0:
          s *= sm_scale
        
        # Apply causal mask only on diagonal subproblem (k_sub == q_sub)
        if k_sub == q_sub:
          mask_shape = (block_k, block_k)
          row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          row_ids += q_start
          col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
          col_ids += k_start
          causal_mask = col_ids <= row_ids
          s = s + jnp.where(causal_mask, 0.0, mask_value)
        
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_k, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_k, MIN_BLOCK_SIZE]
        
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))
        
        alpha = jnp.exp(m_prev - m_next)
        l_next = jnp.sum(p, axis=1)[:, None] + alpha * l_prev  # [block_k, MIN_BLOCK_SIZE]
        
        head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if hd_rem:
          if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
          else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )
        else:
          l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        
        v = v_tile_ref[batch_idx, 0, pl.dslice(k_start, block_k), :]
        o_curr = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        )
        
        acc_prev = acc_scratch_ref[bidx]
        acc_scratch_ref[bidx] = acc_prev * l_broadcast(alpha) + o_curr
        
        l_scratch_ref[bidx] = l_next
        m_scratch_ref[bidx] = m_next
      
      # Normalize and write output for this q subblock
      l_final = l_scratch_ref[bidx]
      head_dim_repeats, hd_rem = divmod(head_dim, MIN_BLOCK_SIZE)
      if hd_rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )
      else:
        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
      o_tile_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = (acc_scratch_ref[bidx] * l_broadcast(inv_l)).astype(o_tile_ref.dtype)
      if l_ref is not None:
        l_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = l_scratch_ref[bidx].astype(l_ref.dtype)
      if m_ref is not None:
        m_ref[batch_idx, 0, pl.dslice(q_start, block_k), :] = m_scratch_ref[bidx].astype(m_ref.dtype)


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
    mask_shape = (block_q, block_k)
    base_row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    base_col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    row_ids = base_row_ids + q_seq_idx * block_q
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
        col_ids = base_col_ids + kv_seq_idx * block_k_major + start_k
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
      l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      if rem:
        if head_dim_repeats == 0:
          l_broadcast = lambda l: l[:, :head_dim]
        else:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          p.astype(v.dtype), v, preferred_element_type=jnp.float32
      )

      acc_prev = acc_scratch_ref[batch_idx]
      acc_scratch_ref[batch_idx] = acc_prev * l_broadcast(alpha) + o_curr

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

  @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
  def store_output():
    l_final = l_scratch_ref[batch_idx]
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
    if rem:
      if head_dim_repeats == 0:
        l_broadcast = lambda l: l[:, :head_dim]
      else:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )
    inv_l = jnp.where(l_final == 0.0, 1.0, 1.0 / l_final)
    o_tile_ref[batch_idx] = (acc_scratch_ref[batch_idx] * l_broadcast(inv_l)).astype(o_tile_ref.dtype)
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

  # Check if we can use the optimized causal triangular kernel
  use_causal_triangular = (
      causal
      and ab is None
      and segment_ids is None
      and block_q == block_k_major
      and block_k_major == kv_seq_len
      and block_q % block_k == 0
      and block_k != kv_seq_len  # Not the single-step case
  )
  
  if use_causal_triangular:
    kernel = functools.partial(
        _flash_attention_kernel_causal_triangular,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        mask_value=DEFAULT_MASK_VALUE,
    )
    # Use smaller scratch buffers sized for q subblocks (block_k rows)
    m_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_k, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_k, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    if block_k != kv_seq_len:
      m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
      acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
      scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
      scratch_shapes = []
  
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

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
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'atol': 2e-3,
    'rtol': 2e-3,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {
    # Autotuned (forward pass).
    'block_q': 2048,
    'block_k_major': 2048,
    'block_k': 512,
    # Not autotuned (batch=1, backward-only).
    'block_b': 1,
    'block_q_major_dkv': 128,
    'block_k_major_dkv': 128,
    'block_k_dkv': 128,
    'block_q_dkv': 128,
    'block_k_major_dq': 128,
    'block_k_dq': 128,
    'block_q_dq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    block_sizes = BlockSizes(
        block_q=TUNED_PARAMS['block_q'],
        block_k_major=TUNED_PARAMS['block_k_major'],
        block_k=TUNED_PARAMS['block_k'],
        block_b=TUNED_PARAMS['block_b'],
        block_q_major_dkv=TUNED_PARAMS['block_q_major_dkv'],
        block_k_major_dkv=TUNED_PARAMS['block_k_major_dkv'],
        block_k_dkv=TUNED_PARAMS['block_k_dkv'],
        block_q_dkv=TUNED_PARAMS['block_q_dkv'],
        block_k_major_dq=TUNED_PARAMS['block_k_major_dq'],
        block_k_dq=TUNED_PARAMS['block_k_dq'],
        block_q_dq=TUNED_PARAMS['block_q_dq'],
    )
    return flash_attention(
        q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
    )
''',
score=0.431,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='gpt-5.4',
stdout='Latency: 0.431 ms\n{"correct": true, "latency": 0.431, "error": "", "all_times_ms": [0.423, 0.423, 0.423, 0.423, 0.424, 0.424, 0.424, 0.425, 0.425, 0.425, 0.426, 0.426, 0.426, 0.426, 0.427, 0.427, 0.428, 0.428, 0.428, 0.428, 0.428, 0.428, 0.428, 0.428, 0.428, 0.428, 0.429, 0.429, 0.429, 0.429, 0.429, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.431, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.432, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.433, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.434, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.436, 0.436, 0.436, 0.436, 0.436, 0.436, 0.437, 0.437, 0.437, 0.438, 0.443, 0.444, 0.445, 0.446, 0.458], "max_diff": 0.000977, "max_rel_diff": 0.000338}',
stderr='')