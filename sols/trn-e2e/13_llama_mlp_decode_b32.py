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

    # Outer batch loop reusing the same tiling for each batch element.
    for b in nl.affine_range(B):
        # Use affine_range to loop over tiles in M and N for this batch element
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                # Allocate a tensor in PSUM
                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Declare the tiles on SBUF
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                    # Load tiles from lhsT and rhs
                    lhsT_tile[...] = nl.load(
                        lhsT[b, k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M]
                    )
                    rhs_tile[...] = nl.load(
                        rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N]
                    )

                    # Accumulate partial-sums into PSUM
                    res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                res_sb = nl.copy(res_psum, dtype=result.dtype)
                nl.store(
                    result[
                        b,
                        m * TILE_M:(m + 1) * TILE_M,
                        n * TILE_N:(n + 1) * TILE_N,
                    ],
                    value=res_sb,
                )

    return result

import torch

def forward_reference(x, up_proj_weight, gate_proj_weight, down_proj_weight):
    b, s, h = x.shape
    up = nki_matmul_tiled_batched_(x.t(), up_proj_weight)
    gate = nki_matmul_tiled_batched_(x.t(), gate_proj_weight)
    act = torch.nn.SiLU()(gate) * up
    output = nki_matmul_tiled_batched_(act.t() , down_proj_weight)

    return output
