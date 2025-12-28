@nki.jit
def nki_matmul_tiled_batched_(lhsT, rhs):
    """Batched NKI matmul kernel using the same tiling scheme as nki_matmul_tiled_.

    Args:
        lhsT: input tensor of shape [B, K, M], where K and M are multiples of 128.
             It is the left-hand-side argument of the matrix multiplication,
             delivered transposed for optimal performance, with a leading batch dim.
        rhs:  input tensor of shape [K, N], where K is a multiple of 128 and N
             is a multiple of 512. It is the right-hand-side argument.
    Returns:
        result: output tensor of shape [B, M, N]
    """

    B, K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((B, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    if M < TILE_M:
        TILE_M = M
    
    if N < TILE_N:
        TILE_N = N

    if K < TILE_K:
        TILE_K = K

    # Pad N for alignment
    N_padded = (N + TILE_N - 1) // TILE_N * TILE_N

    # Outer batch loop reusing the same tiling for each batch element.
    for b in nl.affine_range(B):
        # Use affine_range to loop over tiles in M and N for this batch element
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N_padded // TILE_N):
                # Allocate a tensor in PSUM
                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Declare the tiles on SBUF
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                    # Create index tiles (matching test 2 pattern)
                    i_k = k * TILE_K + nl.arange(TILE_K)[:, None]  # (TILE_K, 1)
                    i_m = m * TILE_M + nl.arange(TILE_M)[None, :]  # (1, TILE_M)
                    i_n = n * TILE_N + nl.arange(TILE_N)[None, :]   # (1, TILE_N)
                    
                    # Masks for bounds checking
                    mask_k = (i_k < K)
                    mask_m = (i_m < M)
                    mask_n = (i_n < N)
                    mask_lhsT = mask_k & mask_m
                    mask_rhs = mask_k & mask_n

                    # Load tiles from lhsT and rhs with masks
                    lhsT_tile[...] = nl.load(lhsT[b, i_k, i_m], mask=mask_lhsT)
                    rhs_tile[...] = nl.load(rhs[i_k, i_n], mask=mask_rhs)

                    # Accumulate partial-sums into PSUM
                    res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                res_sb = nl.copy(res_psum, dtype=result.dtype)
                
                # Create output indices and mask (matching test 2 pattern)
                i_m_out = m * TILE_M + nl.arange(TILE_M)[None, :]  # (1, TILE_M)
                i_n_out = n * TILE_N + nl.arange(TILE_N)[None, :]  # (1, TILE_N)
                mask_store = (i_m_out < M) & (i_n_out < N)
                
                nl.store(result[b, i_m_out, i_n_out], value=res_sb, mask=mask_store)

    return result

import torch

def test(hidden_states, lm_head_weight):
    # hidden_states: (32, 1, 2048)
    # lm_head_weight: (2048, 64128)
    # output: (32, 1, 64128)
    return nki_matmul_tiled_batched_(hidden_states.transpose(1, 2), lm_head_weight)

