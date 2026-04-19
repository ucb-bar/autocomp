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

_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def _apply_rope_2d(x, cos, sin):
    """Apply RoPE to a 2D tensor (S, rope_dim)."""
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(s[0], s[1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)
    sin_f32 = sin.astype(jnp.float32)
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    if multiple == 8:
        common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)
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


def _estimate_gemm_working_set_bytes(bm, bn, bk, in_dtype):
    in_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * in_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _pick_gemm_tiles(M, N, K, dtype, bm, bn, bk):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                ws = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype)
                if ws > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (min(M, bm), min(N, bn), min(K, bk))


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile, b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False),
        ),
    )(a, b)


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b0.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype), jax.ShapeDtypeStruct((M, N), a.dtype)],
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32), pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    _, N0 = b0.shape
    _, N1 = b1.shape
    bm_a = min(M, bm)
    bk_a = min(K, bk)
    while M % bm_a != 0 and bm_a > 8:
        bm_a -= 8
    while K % bk_a != 0 and bk_a > 128:
        bk_a -= 128
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N0), a.dtype), jax.ShapeDtypeStruct((M, N1), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, N0), jnp.float32), pltpu.VMEM((bm_a, N1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    """Update attention state given combined scores."""
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask):
    """Update attention state with causal mask."""
    scores_f32 = jnp.where(causal_mask, scores_f32, jnp.float32(-1.0e30))
    
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_kernel(
    q_nope_ref, q_rope_ref, k_nope_ref, k_rope_ref, v_ref, out_ref, acc_ref, m_ref, l_ref
):
    """Attention kernel where k_rope is already broadcast to have head dimension."""
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    S = k_nope_ref.shape[2]
    num_k_blocks = S // _ATTN_KEY_BLOCK
    nope_dim = q_nope_ref.shape[-1]
    rope_dim = q_rope_ref.shape[-1]
    
    scale = jnp.float32(nope_dim + rope_dim) ** jnp.float32(-0.5)

    # Load Q blocks: (H_block, Q_block, dim)
    q_nope_f32 = q_nope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, nope)
    q_rope_f32 = q_rope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, rope)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            # Compute nope scores: (H_block, Q_block, nope) @ (H_block, K_block, nope).T
            # -> (H_block, Q_block, K_block)
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            # Compute rope scores: (H_block, Q_block, rope) @ (H_block, K_block, rope).T
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_f32 = (scores_nope + scores_rope) * scale
            _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_f32 = (scores_nope + scores_rope) * scale
            
            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _causal_attention_pallas(q_nope, q_rope, k_nope, k_rope, v):
    """
    Attention where k_rope is broadcast to have head dimension.
    
    Args:
        q_nope: (B, H, S, nope_dim)
        q_rope: (B, H, S, rope_dim)
        k_nope: (B, H, S, nope_dim)
        k_rope: (B, H, S, rope_dim) - broadcast across heads
        v: (B, H, S, v_dim)
    
    Returns:
        output: (B, H, S, v_dim)
    """
    B, H, S, nope_dim = q_nope.shape
    rope_dim = q_rope.shape[-1]
    Dv = v.shape[-1]
    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            # q_nope: (B, H, S, nope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, nope_dim), 
                        lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            # q_rope: (B, H, S, rope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, rope_dim), 
                        lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            # k_nope: (B, H, S, nope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, nope_dim), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            # k_rope: (B, H, S, rope_dim) - broadcast across heads
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, rope_dim), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            # v: (B, H, S, v_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
        ],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), 
                              lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, True, True, True, True),
        ),
    )(q_nope, q_rope, k_nope, k_rope, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    x2d = x.reshape(B * S, E)

    # Down projections (fused)
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)

    # Q up projection
    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV processing
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]  # (B, S, rope)

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent2d, k_up_proj, v_up_proj)

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # Apply RoPE
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    
    # Apply RoPE to Q (per-head)
    q_rope = _apply_rope(q_rope, cos, sin)
    
    # Apply RoPE to K_rope and broadcast to head dimension
    # k_rope_raw is (B, S, rope)
    k_rope_with_rope = _apply_rope_2d(k_rope_raw[0], cos, sin)  # (S, rope)
    # Broadcast to (B, H, S, rope) for attention
    k_rope_broadcast = jnp.broadcast_to(
        k_rope_with_rope[None, None, :, :], 
        (B, H, S, rope)
    )

    # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
    q_nope_t = q_nope.transpose(0, 2, 1, 3)
    q_rope_t = q_rope.transpose(0, 2, 1, 3)
    k_nope_t = k_nope.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Attention with k_rope broadcast across heads
    out = _causal_attention_pallas(q_nope_t, q_rope_t, k_nope_t, k_rope_broadcast, v_t)

    # Output projection
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj)
    return final2d.reshape(B, S, E)
