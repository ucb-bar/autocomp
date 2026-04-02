"""NKI GEMM kernel for xpu-perf Neuron backend.

Implements a tiled matrix multiplication C = A @ B using NKI Beta 2 APIs,
targeting NeuronCore-v3 (trn2). Designed to handle all shapes from the
xpu-perf GEMM workload:
  - M: 1 to 32768 (including non-tile-aligned values like 1, 2, 4, 8, 16, 32, 64)
  - K: 1024, 4096, 8192 (always multiples of 128)
  - N: 1024, 4096, 8192 (always multiples of 512)

Layout mapping to nc_matmul:
  nc_matmul computes: dst = stationary.T @ moving
  For C = A @ B:
    stationary = A^T  shape [K_tile, M_tile]  (partition=K, free=M)
    moving     = B    shape [K_tile, N_tile]  (partition=K, free=N)
    dst        = C    shape [M_tile, N_tile]  (partition=M, free=N)

  Input A [M, K] must be transposed to [K, M] before loading as stationary.
  We accept A^T [K, M] as a kernel argument (transposed on the host side).
"""

import nki
import nki.language as nl
import nki.isa as nisa


# Hardware tile limits for NeuronCore-v3
TILE_K = 128  # nl.tile_size.pmax — partition/contraction dimension
TILE_M = 128  # nl.tile_size.gemm_stationary_fmax — stationary free dim
TILE_N = 512  # nl.tile_size.gemm_moving_fmax — moving free dim


@nki.jit
def test(a_t_hbm, b_hbm):
    """Tiled GEMM: C[M,N] = A[M,K] @ B[K,N].

    Args:
        a_t_hbm: A transposed, shape [K, M], in HBM.
        b_hbm:   B, shape [K, N], in HBM.

    Returns:
        c_hbm: C (output), shape [M, N], allocated in HBM.
    """
    K = a_t_hbm.shape[0]
    M = a_t_hbm.shape[1]
    N = b_hbm.shape[1]

    # Allocate output in HBM — this is returned to the XLA runtime
    c_hbm = nl.ndarray((M, N), dtype=a_t_hbm.dtype, buffer=nl.hbm)

    # Number of tiles along each dimension
    num_k_tiles = K // TILE_K
    num_m_tiles = (M + TILE_M - 1) // TILE_M
    num_n_tiles = (N + TILE_N - 1) // TILE_N

    # Outer loop over M tiles
    for m_tile_idx in nl.affine_range(num_m_tiles):
        m_start = m_tile_idx * TILE_M
        m_size = min(M - m_start, TILE_M)

        # Inner loop over N tiles
        for n_tile_idx in nl.affine_range(num_n_tiles):
            n_start = n_tile_idx * TILE_N
            n_size = min(N - n_start, TILE_N)

            # Allocate PSUM for accumulation across K tiles.
            result_psum = nl.ndarray((m_size, n_size), dtype=nl.float32, buffer=nl.psum)

            # K-dimension blocking with sequential_range to preserve
            # accumulation order in PSUM
            for k_idx in nl.sequential_range(num_k_tiles):
                # Load A^T tile: [TILE_K, m_size]
                a_tile = nl.ndarray(
                    (TILE_K, m_size), dtype=a_t_hbm.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=a_tile,
                    src=a_t_hbm[
                        k_idx * TILE_K : (k_idx + 1) * TILE_K,
                        m_start : m_start + m_size,
                    ],
                )

                # Load B tile: [TILE_K, n_size]
                b_tile = nl.ndarray((TILE_K, n_size), dtype=b_hbm.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=b_tile,
                    src=b_hbm[
                        k_idx * TILE_K : (k_idx + 1) * TILE_K,
                        n_start : n_start + n_size,
                    ],
                )

                # nc_matmul: result_psum += a_tile.T @ b_tile
                nisa.nc_matmul(
                    dst=result_psum,
                    stationary=a_tile,
                    moving=b_tile,
                )

            # Move PSUM -> SBUF -> HBM
            result_sbuf = nl.ndarray(
                (m_size, n_size), dtype=a_t_hbm.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=result_sbuf, src=result_psum)
            nisa.dma_copy(
                dst=c_hbm[m_start : m_start + m_size, n_start : n_start + n_size],
                src=result_sbuf,
            )

    return c_hbm
