"""
TPU (JAX Pallas) solution snippet.

Expected format: define a `test(x, y)` entrypoint that the harness calls.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]


def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, f"Inner dimensions must match: {k} != {k2}"
    assert m % bm == 0 and k % bk == 0 and n % bn == 0, (
        f"Shapes must be divisible by block sizes: "
        f"(m,k,n)=({m},{k},{n}) vs (bm,bk,bn)=({bm},{bk},{bn})"
    )

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        grid=(m // bm, n // bn, k // bk),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)


def test(x, y):
    out = matmul(x, y)
    jax.block_until_ready(out)
    return out