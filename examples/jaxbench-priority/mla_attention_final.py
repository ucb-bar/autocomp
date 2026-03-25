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


_FLASH_BLOCK_Q = 128
_FLASH_Q_GROUP = 4
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


def _flash_attention_kernel(
    q_combined_ref, k_latent_ref, k_rope_ref, k_up_ref, v_up_ref,
    o_ref, kv_expanded_scratch
):
    """FlashAttention kernel optimized with fused score matmul over concatenated Q/K."""
    seq_len = k_latent_ref.shape[1]
    nope = k_up_ref.shape[2]
    rope = k_rope_ref.shape[3]
    dv = v_up_ref.shape[2]

    k_up = k_up_ref[0, :, :].astype(jnp.float32)
    v_up = v_up_ref[0, :, :].astype(jnp.float32)
    kv_up_f32 = jnp.concatenate([k_up, v_up], axis=1)

    k_lat_all_f32 = k_latent_ref[0, :, :].astype(jnp.float32)
    kv_expanded_all = jax.lax.dot_general(
        k_lat_all_f32,
        kv_up_f32,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    kv_expanded_scratch[...] = kv_expanded_all

    k_rope_all = k_rope_ref[0, 0, :, :].astype(jnp.float32)
    kv_block = _FLASH_BLOCK_KV
    q_group = _FLASH_BLOCK_Q * _FLASH_Q_GROUP
    num_q_groups = seq_len // q_group

    for qg in range(num_q_groups):
        q_start = qg * q_group
        q_end = q_start + q_group

        q_combined_f32 = q_combined_ref[0, 0, q_start:q_end, :].astype(jnp.float32)
        q_pos = q_start + jnp.arange(q_group, dtype=jnp.int32)

        m = jnp.full((q_group,), -1e9, dtype=jnp.float32)
        l = jnp.zeros((q_group,), dtype=jnp.float32)
        acc = jnp.zeros((q_group, dv), dtype=jnp.float32)

        active_kv_blocks = (q_end + kv_block - 1) // kv_block
        for kv_idx in range(active_kv_blocks):
            kv_start = kv_idx * kv_block
            kv_tile = kv_expanded_scratch[kv_start:kv_start + kv_block, :]
            k_nope_tile = kv_tile[:, :nope]
            v_tile = kv_tile[:, nope:]
            k_rope_tile = k_rope_all[kv_start:kv_start + kv_block, :]
            k_combined_tile = jnp.concatenate([k_nope_tile, k_rope_tile], axis=1)

            scores = jax.lax.dot_general(
                q_combined_f32,
                k_combined_tile,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

            if kv_start + kv_block > q_start:
                k_pos = kv_start + jnp.arange(kv_block, dtype=jnp.int32)[None, :]
                scores = jnp.where(
                    q_pos[:, None] >= k_pos,
                    scores,
                    jnp.array(-1e9, dtype=jnp.float32),
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


def _flash_attention(q_nope, q_rope, k_latent, k_rope, k_up_heads, v_up_heads):
    B, H, S, nope = q_nope.shape
    rope = q_rope.shape[-1]
    DV = v_up_heads.shape[-1]
    kvl = k_latent.shape[-1]
    q_combined = jnp.concatenate([q_nope, q_rope], axis=-1)

    q_group = _FLASH_BLOCK_Q * _FLASH_Q_GROUP
    if S % _FLASH_BLOCK_Q != 0:
        raise ValueError(f"Sequence length must be divisible by block_q: S={S}, block_q={_FLASH_BLOCK_Q}")
    if S % _FLASH_BLOCK_KV != 0:
        raise ValueError(
            f"Sequence length must be divisible by KV block: S={S}, kv_block={_FLASH_BLOCK_KV}"
        )
    if S % q_group != 0:
        raise ValueError(
            f"Sequence length must be divisible by grouped Q block: S={S}, q_group={q_group}"
        )

    kv_expanded_scratch_shape = pltpu.VMEM((S, nope + DV), jnp.float32)

    return pl.pallas_call(
        _flash_attention_kernel,
        out_shape=[jax.ShapeDtypeStruct((B, H, S, DV), q_nope.dtype)],
        grid=(B, H),
        in_specs=[
            pl.BlockSpec((1, 1, S, nope + rope), lambda b, h: (b, h, 0, 0)),
            pl.BlockSpec((1, S, kvl), lambda b, h: (b, 0, 0)),
            pl.BlockSpec((1, 1, S, rope), lambda b, h: (b, 0, 0, 0)),
            pl.BlockSpec((1, kvl, nope), lambda b, h: (h, 0, 0)),
            pl.BlockSpec((1, kvl, DV), lambda b, h: (h, 0, 0)),
        ],
        out_specs=[pl.BlockSpec((1, 1, S, DV), lambda b, h: (b, h, 0, 0))],
        scratch_shapes=[kv_expanded_scratch_shape],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
    )(q_combined, k_latent, k_rope, k_up_heads, v_up_heads)[0]


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H, nope, rope, vd, kvl = (
        C["num_heads"],
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
        C["kv_lora_rank"],
    )
    x2d = x.reshape(B * S, E)

    q_low = (x2d @ q_down_proj).astype(x.dtype)
    q = (q_low @ q_up_proj).reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = (x2d @ kv_down_proj).reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    k_up_heads = k_up_proj.reshape(kvl, H, nope).transpose(1, 0, 2)
    v_up_heads = v_up_proj.reshape(kvl, H, vd).transpose(1, 0, 2)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    q_rope = _apply_rope(q_rope, cos, sin).transpose(0, 2, 1, 3)
    k_rope = _apply_rope(k_rope_raw[:, :, None, :], cos, sin).transpose(0, 2, 1, 3)
    q_nope = q_nope.transpose(0, 2, 1, 3)

    scale = (nope + rope) ** -0.5
    q_nope = (q_nope.astype(jnp.float32) * scale).astype(x.dtype)
    q_rope = (q_rope.astype(jnp.float32) * scale).astype(x.dtype)

    attn_out = _flash_attention(q_nope, q_rope, k_latent, k_rope, k_up_heads, v_up_heads)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B * S, H * vd)
    return (attn_out @ o_proj).reshape(B, S, E)
