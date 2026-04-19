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


def _triangular_row_col(tri_idx):
    tri_idx = jnp.asarray(tri_idx, dtype=jnp.int32)
    tri_f = tri_idx.astype(jnp.float32)
    row_blk = jnp.floor((jnp.sqrt(8.0 * tri_f + 1.0) - 1.0) * 0.5).astype(jnp.int32)
    row_start = (row_blk * (row_blk + 1)) // 2
    col_blk = tri_idx - row_start
    return row_blk, col_blk


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Fused SSD kernel with triangular iteration over valid (row_blk, col_blk) pairs only.
    Uses transposed layout (B, H, D, S) for better memory alignment on TPU.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This triangular SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This triangular SSD kernel requires S//bm == S//bn.")

    n_tri_tiles = n_row_tiles * (n_row_tiles + 1) // 2

    # Transpose inputs from (B, H, S, D) to (B, H, D, S) for better alignment
    # This makes the last two dims (D, S) where D=64 (div by 8) and S blocks are 256 (div by 128)
    query_t = jnp.transpose(query, (0, 1, 3, 2))  # (B, H, D, S)
    key_t = jnp.transpose(key, (0, 1, 3, 2))      # (B, H, D, S)
    value_t = jnp.transpose(value, (0, 1, 3, 2))  # (B, H, D, S)

    def kernel(
        q_ref,           # (bh, D, bm)
        k_ref,           # (bh, D, bn)
        v_ref,           # (bh, D, bn)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, D, bm)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, D, bm)
    ):
        tri_blk = pl.program_id(2)
        row_blk, col_blk = _triangular_row_col(tri_blk)

        def _load_common():
            # Load and convert to float32 for computation
            q = q_ref[...].astype(jnp.float32)  # (bh, D, bm)
            k = k_ref[...].astype(jnp.float32)  # (bh, D, bn)
            v = v_ref[...].astype(jnp.float32)  # (bh, D, bn)
            row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

            # Compute score: q^T @ k -> (bh, bm, bn)
            # q is (bh, D, bm), k is (bh, D, bn)
            # We want (bh, bm, bn) = (bh, bm, D) @ (bh, D, bn) in original layout
            # With transposed layout: q^T is (bh, bm, D), so we need:
            # score[h, m, n] = sum_d q[h, d, m] * k[h, d, n]
            # This is a batched matmul: q.transpose(0,2,1) @ k for each h
            # Using dot_general: contract over D (axis 1 of q and k), batch over h (axis 0)
            score_tile = jax.lax.dot_general(
                q,
                k,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            return score_tile, diff, v

        def _accumulate(decayed_scores, v):
            # decayed_scores: (bh, bm, bn)
            # v: (bh, D, bn)
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Compute partial_out: (bh, D, bm)
            # partial_out[h, d, m] = sum_n decayed_scores[h, m, n] * v[h, d, n]
            # This is: decayed_scores.transpose(0,2,1) @ v.transpose(0,2,1) then transpose back
            # Or: v @ decayed_scores^T for each h
            # Using dot_general: contract over n (axis 2 of decayed_scores, axis 2 of v), batch over h
            partial_out = jax.lax.dot_general(
                v,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        @pl.when(col_blk == 0)
        def _init():
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        @pl.when(col_blk < row_blk)
        def _lower():
            score_tile, diff, v = _load_common()
            decayed_scores = score_tile * jnp.exp(diff)
            _accumulate(decayed_scores, v)

        @pl.when(col_blk == row_blk)
        def _diag_and_store():
            score_tile, diff, v = _load_common()

            local_rows = jnp.arange(bm, dtype=jnp.int32)
            local_cols = jnp.arange(bn, dtype=jnp.int32)
            causal_mask = local_rows[:, None] >= local_cols[None, :]

            decayed_scores = jnp.where(
                causal_mask[None, :, :],
                score_tile * jnp.exp(diff),
                jnp.zeros_like(score_tile),
            )
            _accumulate(decayed_scores, v)

            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)
            # out_acc_ref: (bh, D, bm), den: (bh, bm)
            normalized = out_acc_ref[...] / den[:, None, :]
            o_ref[...] = normalized.astype(query.dtype)

    # Output shape in transposed layout
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_tri_tiles),
        in_specs=[
            # q_ref: (bh, D, bm) from query_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[0]),
            ),
            # k_ref: (bh, D, bn) from key_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bn),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[1]),
            ),
            # v_ref: (bh, D, bn) from value_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bn),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[1]),
            ),
            # row_prefix_ref: (bh, bm) from log_a_cumsum (B, H, S)
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[0]),
            ),
            # col_prefix_ref: (bh, bn) from log_a_cumsum (B, H, S)
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[1]),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, D, bm),
            index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[0]),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)

    # Transpose output back from (B, H, D, S) to (B, H, S, D)
    output = jnp.transpose(out_t, (0, 1, 3, 2))
    return output


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a single fused TPU Pallas kernel to minimize HBM traffic."""
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
