"""
NKI Triangular Multiply -- Single-Channel GEMM Kernel (Beta 2)

Single-channel [N,N]x[N,N] GEMM from OpenFold3 triangular multiplicative update.

nc_matmul: dst += lhsT^T @ rhs
  lhsT[P=128, free<=128] x rhs[P=128, free<=512] -> dst[128, free<=512]

N must be a multiple of 128.
"""

import nki
import nki.language as nl
import nki.isa as nisa

TILE_P = 128  # partition = K contraction
TILE_S = 128  # stationary free = I
TILE_M = 512  # moving free = J


@nki.jit
def solution(a_ki, b_kj):
    """Single-channel GEMM: result[i,j] = sum_k a[k,i]^T * b[k,j]

    Args:
        a_ki: [N, N] -- lhsT layout (K=dim0, I=dim1)
        b_kj: [N, N] -- rhs layout (K=dim0, J=dim1)
    Returns:
        result: [N, N] -- (I=dim0, J=dim1)
    """
    N = a_ki.shape[0]

    n_k = N // TILE_P
    n_i = N // TILE_S
    # For j, handle N < 512 case
    n_j = N // TILE_M if N >= TILE_M else 1
    jtile = TILE_M if N >= TILE_M else N

    result = nl.ndarray((N, N), dtype=a_ki.dtype, buffer=nl.shared_hbm)

    for i_t in nl.affine_range(n_i):
        i0 = i_t * TILE_S

        for j_t in nl.affine_range(n_j):
            j0 = j_t * jtile

            # Fresh PSUM accumulator
            acc = nl.ndarray((TILE_S, jtile), dtype=nl.float32, buffer=nl.psum)

            for k_t in nl.sequential_range(n_k):
                k0 = k_t * TILE_P

                # Load lhsT tile
                lhsT_tile = nl.ndarray(
                    (TILE_P, TILE_S), dtype=a_ki.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=lhsT_tile, src=a_ki[k0 : k0 + TILE_P, i0 : i0 + TILE_S]
                )

                # Load rhs tile
                rhs_tile = nl.ndarray((TILE_P, jtile), dtype=b_kj.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile, src=b_kj[k0 : k0 + TILE_P, j0 : j0 + jtile])

                # Accumulate
                nisa.nc_matmul(dst=acc, stationary=lhsT_tile, moving=rhs_tile)

            # PSUM -> SBUF -> HBM
            out = nl.ndarray((TILE_S, jtile), dtype=a_ki.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[i0 : i0 + TILE_S, j0 : j0 + jtile], src=out)

    return result
