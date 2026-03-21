import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "retnet_6_7b_retention",
    "model": "RetNet-6.7B",
    "operator": "multi_scale_retention",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 16,
    "head_dim": 256,
    "d_model": 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """
    Multi-scale retention optimized for TPU v6e-1.

    Exact optimization:
      * Precompute K_scaled[n] = K_n * c_n once per KV block.
      * Precompute S_n = K_scaled[n]^T @ V_n once per KV block.
      * Reuse K_scaled[n] for exact lower-block normalization:
            sum(abs((Q * r) @ K_scaled[n]^T), axis=1)
        instead of re-scaling K on every lower-triangular iteration.

    This preserves the original semantics while removing redundant
    per-lower-block K scaling work.
    """
    B, H, S, D = query.shape
    assert key.shape == (B, H, S, D)
    assert value.shape == (B, H, S, D)

    BLOCK_M = 512
    BLOCK_N = 512

    assert S % BLOCK_M == 0, f"S={S} must be divisible by BLOCK_M={BLOCK_M}"
    assert S % BLOCK_N == 0, f"S={S} must be divisible by BLOCK_N={BLOCK_N}"
    assert BLOCK_M == BLOCK_N, "This version assumes square causal blocks."
    assert D % 128 == 0, f"D={D} must satisfy TPU block constraints."

    num_m = S // BLOCK_M
    num_n = S // BLOCK_N
    assert num_m == num_n

    # (BM, D) @ (D, D) -> (BM, D)
    qS_dims = (((1,), (0,)), ((), ()))
    # (BM, D) @ (BN, D)^T -> (BM, BN)
    qk_dims = (((1,), (1,)), ((), ()))
    # (BM, BN) @ (BN, D) -> (BM, D)
    ov_dims = (((1,), (0,)), ((), ()))
    # (D, BN) @ (BN, D) -> (D, D)
    kv_dims = (((1,), (0,)), ((), ()))

    def retention_kernel(
        q_ref,
        k_ref,
        v_ref,
        out_ref,
        k_scaled_blocks_ref,
        S_precomputed_ref,
        qk_scratch_ref,
    ):
        # One program handles one (batch, head) slice; refs are shaped (S, D).
        h_idx = pl.program_id(1)
        h_f32 = h_idx.astype(jnp.float32)

        gamma = jnp.float32(1.0) - jnp.exp2(jnp.float32(-5.0) - h_f32)
        log_gamma = jnp.log(gamma)

        # Static causal mask for the diagonal block.
        local_idx = jnp.arange(BLOCK_M, dtype=jnp.int32)
        causal_mask = (local_idx[:, None] >= local_idx[None, :]).astype(jnp.float32)

        # ------------------------------------------------------------------
        # Pass 1: precompute per-block scaled K and S_n = K_scaled^T @ V
        # ------------------------------------------------------------------
        for nj in range(num_n):
            n0 = nj * BLOCK_N
            n1 = n0 + BLOCK_N

            k_tile = k_ref[n0:n1, :].astype(jnp.float32)  # (BLOCK_N, D)
            v_tile = v_ref[n0:n1, :].astype(jnp.float32)  # (BLOCK_N, D)

            c_idx = (n0 + jnp.arange(BLOCK_N)).astype(jnp.float32)
            c_vec = jnp.exp(-log_gamma * c_idx)  # (BLOCK_N,)

            k_scaled = k_tile * c_vec[:, None]  # (BLOCK_N, D)
            k_scaled_blocks_ref[nj, :, :] = k_scaled

            S_n = jax.lax.dot_general(
                jnp.transpose(k_scaled),  # (D, BLOCK_N)
                v_tile,                   # (BLOCK_N, D)
                kv_dims,
                preferred_element_type=jnp.float32,
            )
            S_precomputed_ref[nj, :, :] = S_n

        # ------------------------------------------------------------------
        # Pass 2: compute output block by block
        # ------------------------------------------------------------------
        for mi in range(num_m):
            m0 = mi * BLOCK_M
            m1 = m0 + BLOCK_M

            q_tile = q_ref[m0:m1, :].astype(jnp.float32)  # (BLOCK_M, D)

            r_idx = (m0 + jnp.arange(BLOCK_M)).astype(jnp.float32)
            r_vec = jnp.exp(log_gamma * r_idx)  # (BLOCK_M,)

            q_scaled = q_tile * r_vec[:, None]  # (BLOCK_M, D)

            acc = jnp.zeros((BLOCK_M, D), dtype=jnp.float32)
            norm_acc = jnp.zeros((BLOCK_M, 1), dtype=jnp.float32)

            # Strictly lower-triangular blocks use precomputed S_n and K_scaled_n.
            for nj in range(mi):
                S_n = S_precomputed_ref[nj, :, :]          # (D, D)
                k_scaled = k_scaled_blocks_ref[nj, :, :]   # (BLOCK_N, D)

                contrib = jax.lax.dot_general(
                    q_scaled,  # (BLOCK_M, D)
                    S_n,       # (D, D)
                    qS_dims,
                    preferred_element_type=jnp.float32,
                )
                acc = acc + contrib

                qk_scratch_ref[:, :] = jax.lax.dot_general(
                    q_scaled,  # (BLOCK_M, D)
                    k_scaled,  # (BLOCK_N, D), transposed via contraction on axis 1
                    qk_dims,
                    preferred_element_type=jnp.float32,
                )
                qk_tile = qk_scratch_ref[:, :]
                norm_acc = norm_acc + jnp.sum(
                    jnp.abs(qk_tile), axis=1, keepdims=True
                )

            # Diagonal block: exact causal computation.
            k_scaled_diag = k_scaled_blocks_ref[mi, :, :]      # (BLOCK_N, D)
            v_tile = v_ref[m0:m1, :].astype(jnp.float32)       # (BLOCK_N, D)

            qk_scratch_ref[:, :] = jax.lax.dot_general(
                q_scaled,
                k_scaled_diag,
                qk_dims,
                preferred_element_type=jnp.float32,
            )
            qk_diag = qk_scratch_ref[:, :] * causal_mask

            norm_acc = norm_acc + jnp.sum(
                jnp.abs(qk_diag), axis=1, keepdims=True
            )

            diag_contrib = jax.lax.dot_general(
                qk_diag.astype(jnp.float32),
                v_tile,
                ov_dims,
                preferred_element_type=jnp.float32,
            )
            acc = acc + diag_contrib

            norm_tile = jnp.maximum(norm_acc, jnp.float32(1.0))
            out_ref[m0:m1, :] = acc / norm_tile

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(B, H),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, S, D),
                index_map=lambda b, h: (b, h, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, S, D),
                index_map=lambda b, h: (b, h, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, S, D),
                index_map=lambda b, h: (b, h, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, S, D),
            index_map=lambda b, h: (b, h, 0, 0),
        ),
        scratch_shapes=[
            pltpu.VMEM((num_n, BLOCK_N, D), jnp.float32),  # precomputed K_scaled blocks
            pltpu.VMEM((num_n, D, D), jnp.float32),        # precomputed S_n blocks
            pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32),   # reusable QK scratch
        ],
    )

    out_f32 = pl.pallas_call(
        retention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), jnp.float32),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        name="retnet_v6e_precomputed_kscaled_and_S",
    )(query, key, value)

    return out_f32.astype(query.dtype)
