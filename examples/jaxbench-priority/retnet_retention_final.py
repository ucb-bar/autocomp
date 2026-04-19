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

    Optimization: Precompute S_n = (K_n * c_n)^T @ V_n for each key/value block n,
    then reuse this (D x D) matrix for all query blocks m > n, avoiding redundant
    K-V matmul recomputation.
    """
    B, H, S, D = query.shape
    assert key.shape == (B, H, S, D)
    assert value.shape == (B, H, S, D)

    BLOCK_M = 512
    BLOCK_N = 512

    assert S % BLOCK_M == 0, f"S={S} must be divisible by BLOCK_M={BLOCK_M}"
    assert S % BLOCK_N == 0, f"S={S} must be divisible by BLOCK_N={BLOCK_N}"
    assert BLOCK_M == BLOCK_N, "This optimized causal-skip version assumes square blocks."

    num_m = S // BLOCK_M
    num_n = S // BLOCK_N
    assert num_m == num_n

    # (BM, D) @ (D, D) -> (BM, D)
    qS_dims = (((1,), (0,)), ((), ()))
    # (BM, D) @ (BN, D)^T -> (BM, BN)
    qk_dims = (((1,), (1,)), ((), ()))
    # (BM, BN) @ (BN, D) -> (BM, D)
    ov_dims = (((1,), (0,)), ((), ()))
    # (D, BN) @ (BN, D) -> (D, D) for S_n precomputation
    kv_dims = (((1,), (0,)), ((), ()))

    def retention_kernel(q_ref, k_ref, v_ref, out_ref, S_precomputed_ref, qk_scratch_ref):
        # Per-program refs have shape (S, D) for inputs/output
        # S_precomputed_ref: (num_n, D, D) for precomputed K^T @ V blocks
        # qk_scratch_ref: (BLOCK_M, BLOCK_N) for diagonal block computation
        h_idx = pl.program_id(1)
        h_f32 = h_idx.astype(jnp.float32)

        gamma = jnp.float32(1.0) - jnp.exp2(jnp.float32(-5.0) - h_f32)
        log_gamma = jnp.log(gamma)

        # Pass 1: Precompute S_n = (K_n * c_n[:, None])^T @ V_n for each block n
        for nj in range(num_n):
            n0 = nj * BLOCK_N
            n1 = (nj + 1) * BLOCK_N

            k_tile = k_ref[n0:n1, :].astype(jnp.float32)  # (BLOCK_N, D)
            v_tile = v_ref[n0:n1, :].astype(jnp.float32)  # (BLOCK_N, D)

            # Column decay: c_j = exp(-log_gamma * j)
            c_idx = (n0 + jnp.arange(BLOCK_N)).astype(jnp.float32)
            c_vec = jnp.exp(-log_gamma * c_idx)  # (BLOCK_N,)

            # Scale K by column decay
            k_scaled = k_tile * c_vec[:, None]  # (BLOCK_N, D)

            # Compute S_n = K_scaled^T @ V = (D, BLOCK_N) @ (BLOCK_N, D) -> (D, D)
            S_n = jax.lax.dot_general(
                jnp.transpose(k_scaled),  # (D, BLOCK_N)
                v_tile,  # (BLOCK_N, D)
                kv_dims,
                preferred_element_type=jnp.float32,
            )
            S_precomputed_ref[nj, :, :] = S_n

        # Pass 2: Compute output for each query block
        for mi in range(num_m):
            m0 = mi * BLOCK_M
            m1 = (mi + 1) * BLOCK_M

            # Load query tile
            q_tile = q_ref[m0:m1, :].astype(jnp.float32)  # (BLOCK_M, D)

            # Row decay: r_i = exp(log_gamma * i)
            r_idx = (m0 + jnp.arange(BLOCK_M)).astype(jnp.float32)
            r_vec = jnp.exp(log_gamma * r_idx)  # (BLOCK_M,)

            # Scale Q by row decay
            q_scaled = q_tile * r_vec[:, None]  # (BLOCK_M, D)

            # Accumulator for output
            acc = jnp.zeros((BLOCK_M, D), dtype=jnp.float32)
            
            # Accumulator for normalization (sum of absolute weights)
            norm_acc = jnp.zeros((BLOCK_M, 1), dtype=jnp.float32)

            # Process strictly lower-triangular blocks (nj < mi) using precomputed S_n
            for nj in range(num_m):
                # Only process if nj < mi
                is_lower = nj < mi
                
                if is_lower:
                    n0 = nj * BLOCK_N
                    
                    # Load precomputed S_n
                    S_n = S_precomputed_ref[nj, :, :]  # (D, D)
                    
                    # Compute contribution: Q_scaled @ S_n
                    # This gives us the unnormalized weighted sum for this block
                    contrib = jax.lax.dot_general(
                        q_scaled,  # (BLOCK_M, D)
                        S_n,  # (D, D)
                        qS_dims,
                        preferred_element_type=jnp.float32,
                    )  # (BLOCK_M, D)
                    
                    acc = acc + contrib
                    
                    # For normalization, we need sum of |weights| = sum of |Q @ K^T * decay|
                    # Compute Q @ K^T * decay for this block to get absolute weight sum
                    k_tile = k_ref[n0:n0 + BLOCK_N, :].astype(jnp.float32)
                    c_idx = (n0 + jnp.arange(BLOCK_N)).astype(jnp.float32)
                    c_vec = jnp.exp(-log_gamma * c_idx)
                    
                    qk_tile = jax.lax.dot_general(
                        q_tile,
                        k_tile,
                        qk_dims,
                        preferred_element_type=jnp.float32,
                    )
                    decay_tile = r_vec[:, None] * c_vec[None, :]
                    weights = qk_tile * decay_tile
                    norm_acc = norm_acc + jnp.sum(jnp.abs(weights), axis=1, keepdims=True)

            # Process diagonal block (nj == mi) with causal mask
            n0 = mi * BLOCK_N
            n1 = (mi + 1) * BLOCK_N

            k_tile = k_ref[n0:n1, :].astype(jnp.float32)
            v_tile = v_ref[n0:n1, :].astype(jnp.float32)

            c_idx = (n0 + jnp.arange(BLOCK_N)).astype(jnp.float32)
            c_vec = jnp.exp(-log_gamma * c_idx)

            qk_tile = jax.lax.dot_general(
                q_tile,
                k_tile,
                qk_dims,
                preferred_element_type=jnp.float32,
            )
            decay_tile = r_vec[:, None] * c_vec[None, :]

            # Causal mask for diagonal block
            mask = (r_idx[:, None] >= c_idx[None, :]).astype(jnp.float32)
            qk_masked = qk_tile * decay_tile * mask

            # Add to normalization
            norm_acc = norm_acc + jnp.sum(jnp.abs(qk_masked), axis=1, keepdims=True)

            # Compute diagonal contribution
            diag_contrib = jax.lax.dot_general(
                qk_masked.astype(v_ref.dtype),
                v_tile,
                ov_dims,
                preferred_element_type=jnp.float32,
            )
            acc = acc + diag_contrib

            # Normalize
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
            pltpu.VMEM((num_n, D, D), jnp.float32),  # S_precomputed: precomputed K^T @ V blocks
            pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32),  # qk_scratch for diagonal
        ],
    )

    out_f32 = pl.pallas_call(
        retention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), jnp.float32),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        name="retnet_v6e_precomputed_S",
    )(query, key, value)

    return out_f32.astype(query.dtype)
