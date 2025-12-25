import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def nki_matmul_tiled_reference_(lhsT, rhs):
    """NKI kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples for
            128.  It is the left-hand-side argument of the matrix multiplication,
            delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
            is a multiple of 512.  It is the right-hand-side argument of the matrix
            multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

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

    # Use affine_range to loop over tiles
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
                lhsT_tile[...] = nl.load(lhsT[i_k, i_m], mask=mask_lhsT)
                rhs_tile[...] = nl.load(rhs[i_k, i_n], mask=mask_rhs)

                # Accumulate partial-sums into PSUM
                res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            
            # Create output indices and mask (matching test 2 pattern)
            i_m_out = m * TILE_M + nl.arange(TILE_M)[None, :]  # (1, TILE_M)
            i_n_out = n * TILE_N + nl.arange(TILE_N)[None, :]  # (1, TILE_N)
            mask_store = (i_m_out < M) & (i_n_out < N)
            
            nl.store(result[i_m_out, i_n_out], value=res_sb, mask=mask_store)

    return result

def get_test_data(dtype):
    """Create test data for logits kernel."""
    # hidden_states: (1, 1, 2048)
    hidden_states = np.random.randn(1, 1, 2048).astype(dtype)
    # lm_head_weight: (64128, 2048)
    lm_head_weight = np.random.randn(2048, 64128).astype(dtype)
    return (hidden_states, lm_head_weight)

def compare_outputs(reference_out, test_out, atol=1e-3, rtol=1e-3):
    """Compare test output against reference output."""
    ref_f32 = reference_out.astype(nl.float32)
    test_f32 = test_out.astype(nl.float32)
    if not np.allclose(ref_f32, test_f32, atol=atol, rtol=rtol):
        print("reference_out[:8]: %s", ref_f32.flatten()[:8])
        print("test_out[:8]: %s", test_f32.flatten()[:8])
        diff = np.abs(ref_f32 - test_f32)
        print("max_diff: %s", np.max(diff))
        print("mean_diff: %s", np.mean(diff))
        print("FAIL: test output does not match reference")
        return False
    return True

def test_nki(ref_func, test_func):
    np.random.seed(0)
    dtype = nl.bfloat16
    
    for _ in range(2):
        hidden_states, lm_head_weight = get_test_data(dtype)
        ref_out = ref_func(hidden_states.reshape(1, 2048).T, lm_head_weight).reshape(1, 1, 64128)
        test_out = test_func(hidden_states, lm_head_weight)
        if not compare_outputs(ref_out, test_out):
            return False
    return True

def benchmark_nki(nki_func):
    dtype = nl.bfloat16
    test_data = get_test_data(dtype)
    
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(*test_data)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    test_result = test_nki(nki_matmul_tiled_reference_, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)

