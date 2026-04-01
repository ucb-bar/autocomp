@nki.jit
def test(
    a,
    b,
):
    """Triangular multiplicative update (outgoing) for Boltz-2 pairformer.

    Computes result[i,j,d] = sum_k a[i,k,d] * b[j,k,d] for all (i,j,d).
    Equivalent to D independent NxN matmuls: result[:,:,d] = a[:,:,d] @ b[:,:,d].T

    Args:
        a: (N, N, D) bfloat16 — first gated projection.
        b: (N, N, D) bfloat16 — second gated projection.

    Returns:
        output: (N, N, D) bfloat16 — contraction result.
    """
    N, N2, D = a.shape
    N_b, N2_b, D_b = b.shape
    P_MAX = 128

    assert N == N2
    assert N == N_b and N2 == N2_b and D == D_b
    assert N % P_MAX == 0
    assert D <= P_MAX

    output = nl.ndarray((N, N, D), dtype=a.dtype, buffer=nl.shared_hbm)

    stride_i = N * D
    stride_k = D

    n_tiles = N // P_MAX

    for d in nl.affine_range(D):
        for i_tile in nl.affine_range(n_tiles):
            i_start = i_tile * P_MAX

            for j_tile in nl.affine_range(n_tiles):
                j_start = j_tile * P_MAX

                acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc, value=0.0)

                for k_tile_idx in nl.sequential_range(n_tiles):
                    k_start = k_tile_idx * P_MAX

                    # Load a_tile: a[i_start:i_end, k_start:k_end, d]
                    a_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=a_tile,
                        src=a.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=i_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    # Load b_tile: b[j_start:j_end, k_start:k_end, d]
                    b_tile = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=b_tile,
                        src=b.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=j_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    # Transpose both for nc_matmul: a^T @ b^T = a @ b^T
                    a_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=a_t_psum, data=a_tile)
                    a_t = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=a_t, src=a_t_psum)

                    b_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=b_t_psum, data=b_tile)
                    b_t = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=b_t, src=b_t_psum)

                    partial_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=partial_psum, stationary=a_t, moving=b_t)
                    partial = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=partial, src=partial_psum)

                    nisa.tensor_tensor(dst=acc, data1=acc, data2=partial, op=nl.add)

                out_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile, src=acc)

                nisa.dma_copy(
                    dst=output.ap(
                        pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                        offset=i_start * stride_i + j_start * stride_k + d,
                    ),
                    src=out_tile,
                )

    return output
