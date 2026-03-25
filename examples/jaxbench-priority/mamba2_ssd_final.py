import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Row-owned SSD kernel for TPU v6e.

    This version explicitly splits strictly-lower-triangular tiles from diagonal
    tiles. Off-diagonal tiles avoid the causal-mask path entirely, while the
    diagonal tile applies the mask directly.
    """
    B, H, S, D = query.shape
    out_dtype = query.dtype

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This SSD kernel assumes bm == bn, so row/col tile counts must match.")

    # TPU-friendly layout: (B, H, D, S)
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    q_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    k_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    v_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    prefix_full_spec = pl.BlockSpec(
        block_shape=(None, bh, S),
        index_map=lambda b, hg: (b, hg, 0),
    )
    out_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )

    def kernel(
        q_full_ref,       # (bh, D, S)
        k_full_ref,       # (bh, D, S)
        v_full_ref,       # (bh, D, S)
        prefix_full_ref,  # (bh, S)
        o_full_ref,       # (bh, D, S)
        row_sum_acc_ref,  # scratch: (bh, bm) f32
        out_acc_ref,      # scratch: (bh, D, bm) f32
    ):
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        causal_mask = local_rows[:, None] >= local_cols[None, :]  # (bm, bn)

        def accumulate_block(q_block, row_prefix, col_blk, apply_causal_mask):
            col_start = col_blk * bn
            col_stop = col_start + bn

            k_block = k_full_ref[:, :, col_start:col_stop].astype(jnp.float32)      # (bh, D, bn)
            v_block = v_full_ref[:, :, col_start:col_stop].astype(jnp.float32)      # (bh, D, bn)
            col_prefix = prefix_full_ref[:, col_start:col_stop].astype(jnp.float32) # (bh, bn)

            score_tile = jax.lax.dot_general(
                q_block,
                k_block,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            base_scores = score_tile * jnp.exp(diff)

            if apply_causal_mask:
                decayed_scores = jnp.where(
                    causal_mask[None, :, :],
                    base_scores,
                    jnp.zeros_like(base_scores),
                )
            else:
                decayed_scores = base_scores

            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            partial_out = jax.lax.dot_general(
                v_block,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        for row_blk in range(n_row_tiles):
            row_start = row_blk * bm
            row_stop = row_start + bm

            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

            q_block = q_full_ref[:, :, row_start:row_stop].astype(jnp.float32)      # (bh, D, bm)
            row_prefix = prefix_full_ref[:, row_start:row_stop].astype(jnp.float32) # (bh, bm)

            # Strictly lower-triangular blocks: no causal mask needed.
            for col_blk in range(row_blk):
                accumulate_block(q_block, row_prefix, col_blk, False)

            # Diagonal block: causal mask required.
            accumulate_block(q_block, row_prefix, row_blk, True)

            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)  # (bh, bm)

            normalized = out_acc_ref[...] / den[:, None, :]
            o_full_ref[:, :, row_start:row_stop] = normalized.astype(out_dtype)

    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), out_dtype),
        grid=(B, H // bh),
        in_specs=[
            q_full_spec,
            k_full_spec,
            v_full_spec,
            prefix_full_spec,
        ],
        out_specs=out_full_spec,
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum)

    return jnp.transpose(out_t, (0, 1, 3, 2))


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a row-owned fused TPU Pallas kernel."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
