import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


# Conservative per-kernel VMEM working-set target for v6e-1.
# This leaves headroom for compiler temporaries and pipeline buffering.
_V6E_GEMM_WORKING_SET_BUDGET = 2 * 1024 * 1024  # 2 MiB
_FLASH_BLOCK_Q = 128
# Larger Q grouping amortizes full-sequence K/V loads across multiple Q tiles.
_FLASH_Q_GROUP = 4
# INCREASED from 128 to 256 for better compute-to-memory ratio
# This reduces KV block loads from 16 to 8, improving memory bandwidth utilization
_FLASH_BLOCK_KV = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    """Choose a TPU-valid block size for one axis.

    Returns either:
      * a divisor of `dim` aligned to `multiple`, or
      * the full dimension `dim` (which is also legal on TPU block axes).
    """
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    """Approximate live VMEM footprint under TPU double-buffered pipelining."""
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    # Inputs and outputs are typically double buffered by the TPU pipeline.
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b

    # Accumulator scratch is float32 and must stay live across K-reduction steps.
    acc_bytes = bm * bn * 4

    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    """Generate legal block candidates, largest first."""
    candidates = {dim}  # full-dimension fallback is always legal on TPU
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    """Pick TPU-valid tiles that fit comfortably in v6e VMEM."""
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue

                # Maximize useful work first, then favor larger output tiles.
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    # Guaranteed-valid fallback.
    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _flash_attention_kernel(
    q_nope_ref, q_rope_ref, k_latent_ref, k_rope_ref, kv_up_ref, o_ref
):
    """Persistent FlashAttention-style streaming kernel with MLA latent KV.

    OPTIMIZATION: Fused k_up and v_up expansions into single grouped GEMM.
    
    Grid: (batch, head)
    Block shapes:
      q_nope_ref: [1, 1, S, nope]
      q_rope_ref: [1, 1, S, rope]
      k_latent_ref: [1, S, kvl]      (shared across heads)
      k_rope_ref: [1, 1, S, rope]    (shared across heads)
      kv_up_ref: [1, kvl, nope + dv]  (fused k_up + v_up concatenated)
      o_ref: [1, 1, S, DV]
    """
    k_latent_all = k_latent_ref[0, :, :].astype(jnp.float32)
    k_rope_all = k_rope_ref[0, 0, :, :].astype(jnp.float32)
    kv_up = kv_up_ref[0, :, :].astype(jnp.float32)
    
    nope = q_nope_ref.shape[-1]
    dv = o_ref.shape[-1]

    seq_len = k_latent_all.shape[0]
    kv_block = _FLASH_BLOCK_KV
    q_group = _FLASH_BLOCK_Q * _FLASH_Q_GROUP

    num_q_groups = seq_len // q_group

    for qg in range(num_q_groups):
        q_start = qg * q_group
        q_end = q_start + q_group
        q_nope_tile = q_nope_ref[0, 0, q_start:q_end, :].astype(jnp.float32)
        q_rope_tile = q_rope_ref[0, 0, q_start:q_end, :].astype(jnp.float32)
        q_pos = q_start + jnp.arange(q_group, dtype=jnp.int32)
        active_kv_blocks = (q_end + kv_block - 1) // kv_block

        m = jnp.full((q_group,), -jnp.inf, dtype=jnp.float32)
        l = jnp.zeros((q_group,), dtype=jnp.float32)
        acc = jnp.zeros((q_group, dv), dtype=jnp.float32)

        for kv_idx in range(active_kv_blocks):
            kv_start = kv_idx * kv_block
            k_latent_tile = k_latent_all[kv_start:kv_start + kv_block, :]
            k_rope_tile = k_rope_all[kv_start:kv_start + kv_block, :]
            
            # OPTIMIZATION: Single grouped GEMM for both k_nope and v
            # Instead of two separate GEMMs:
            #   k_nope_tile = k_latent_tile @ k_up
            #   v_tile = k_latent_tile @ v_up
            # We do one matmul against concatenated [k_up | v_up]
            fused_kv = jax.lax.dot_general(
                k_latent_tile,
                kv_up,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            # Split the result: first `nope` dims are k_nope, last `dv` dims are v
            k_nope_tile = fused_kv[:, :nope]
            v_tile = fused_kv[:, nope:]

            # Split score computation: q_nope @ k_nope.T + q_rope @ k_rope.T
            scores_nope = jax.lax.dot_general(
                q_nope_tile,
                k_nope_tile,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            scores_rope = jax.lax.dot_general(
                q_rope_tile,
                k_rope_tile,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            scores = scores_nope + scores_rope

            if kv_start + kv_block > q_start:
                k_pos = kv_start + jnp.arange(kv_block, dtype=jnp.int32)[None, :]
                causal_mask = q_pos[:, None] >= k_pos
                scores = jnp.where(
                    causal_mask, scores, jnp.array(-1e9, dtype=jnp.float32)
                )

            block_max = jnp.max(scores, axis=1)
            new_m = jnp.maximum(m, block_max)
            exp_m_scale = jnp.exp(m - new_m)
            exp_scores = jnp.exp(scores - new_m[:, None])
            l = l * exp_m_scale + jnp.sum(exp_scores, axis=1)
            acc = acc * exp_m_scale[:, None] + jax.lax.dot_general(
                exp_scores,
                v_tile,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            m = new_m

        o_ref[0, 0, q_start:q_end, :] = (acc / l[:, None]).astype(o_ref.dtype)


def _flash_attention(
    q_nope, q_rope, k_latent, k_rope, k_up_heads, v_up_heads, block_q=_FLASH_BLOCK_Q
):
    """Blocked causal attention with MLA latent KV kept unexpanded until on-chip use.

    OPTIMIZATION: k_up and v_up are concatenated before kernel call to enable
    single grouped GEMM in the kernel.

    Args:
        q_nope: [B, H, S, nope] - query nope component
        q_rope: [B, H, S, rope] - query rope component
        k_latent: [B, S, kvl] - latent key/value state shared across heads
        k_rope: [B, 1, S, rope] - key rope component (shared across heads)
        k_up_heads: [H, kvl, nope] - per-head key expansion weights
        v_up_heads: [H, kvl, DV] - per-head value expansion weights
    """
    B, H, S, nope = q_nope.shape
    rope = q_rope.shape[-1]
    DV = v_up_heads.shape[-1]
    kvl = k_latent.shape[-1]

    if S % block_q != 0:
        raise ValueError(f"Sequence length must be divisible by block_q: S={S}, block_q={block_q}")
    if S % _FLASH_BLOCK_KV != 0:
        raise ValueError(
            f"Sequence length must be divisible by KV block: S={S}, kv_block={_FLASH_BLOCK_KV}"
        )

    q_group = block_q * _FLASH_Q_GROUP
    if S % q_group != 0:
        raise ValueError(
            f"Sequence length must be divisible by grouped Q block: S={S}, q_group={q_group}"
        )

    # OPTIMIZATION: Concatenate k_up and v_up along the feature dimension
    # This enables a single grouped GEMM in the kernel instead of two separate ones
    # Original: k_up_heads [H, kvl, nope], v_up_heads [H, kvl, DV]
    # Concat: kv_up_heads [H, kvl, nope + DV]
    kv_up_heads = jnp.concatenate([k_up_heads, v_up_heads], axis=-1)

    outs = pl.pallas_call(
        _flash_attention_kernel,
        out_shape=[jax.ShapeDtypeStruct((B, H, S, DV), q_nope.dtype)],
        grid=(B, H),
        in_specs=[
            pl.BlockSpec((1, 1, S, nope), lambda b0, h0: (b0, h0, 0, 0)),
            pl.BlockSpec((1, 1, S, rope), lambda b0, h0: (b0, h0, 0, 0)),
            pl.BlockSpec((1, S, kvl), lambda b0, h0: (b0, 0, 0)),
            pl.BlockSpec((1, 1, S, rope), lambda b0, h0: (b0, 0, 0, 0)),
            # Changed from two separate specs to one fused spec
            pl.BlockSpec((1, kvl, nope + DV), lambda b0, h0: (h0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, S, DV), lambda b0, h0: (b0, h0, 0, 0)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
    )(q_nope, q_rope, k_latent, k_rope, kv_up_heads)
    return outs[0]


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # Flatten batch and sequence for dense projections.
    x2d = x.reshape(B * S, E)

    # Use XLA-optimized dense matmuls for straightforward projections.
    q_low2d = jax.lax.dot_general(
        x2d,
        q_down_proj,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    q2d = jax.lax.dot_general(
        q_low2d,
        q_up_proj,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv2d = jax.lax.dot_general(
        x2d,
        kv_down_proj,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    k_up_heads = k_up_proj.reshape(kvl, H, nope).transpose(1, 0, 2)
    v_up_heads = v_up_proj.reshape(kvl, H, vd).transpose(1, 0, 2)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    # Apply RoPE to q_rope [B, S, H, rope]
    q_rope = _apply_rope(q_rope, cos, sin)
    # Apply RoPE to k_rope_raw [B, S, rope] - keep it head-shared
    k_rope_raw_expanded = k_rope_raw[:, :, None, :]  # [B, S, 1, rope]
    k_rope = _apply_rope(k_rope_raw_expanded, cos, sin)  # [B, S, 1, rope]

    # Attention inputs - keep nope and rope separate and retain latent KV shared across heads.
    q_nope = q_nope.transpose(0, 2, 1, 3)  # [B, H, S, nope]
    q_rope = q_rope.transpose(0, 2, 1, 3)  # [B, H, S, rope]
    k_rope = k_rope.transpose(0, 2, 1, 3)  # [B, 1, S, rope] (shared across heads)

    # Apply scaling to query components
    hd = nope + rope
    scale = hd ** -0.5
    q_nope = (q_nope.astype(jnp.float32) * scale).astype(x.dtype)
    q_rope = (q_rope.astype(jnp.float32) * scale).astype(x.dtype)

    # FlashAttention-style fused causal attention with split nope/rope and latent KV.
    # This avoids materializing dense per-head K/V in HBM before attention.
    out = _flash_attention(q_nope, q_rope, k_latent, k_rope, k_up_heads, v_up_heads)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = jax.lax.dot_general(
        out2d,
        o_proj,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)

    return final2d.reshape(B, S, E)
