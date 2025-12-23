import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

@nki.jit
def test(hidden_states, lm_head_weight):
    """
    Optimized for constant shapes:
      hidden_states: [1, 1, 2048]
      lm_head_weight: [2048, 64128]
    Returns:
      out: [1, 1, 64128]

    Key optimizations:
      1) Precompute x^T tiles once: hidden_states[1,128] -> transpose -> xT[128,1] in SBUF.
      2) Load RHS in larger contiguous N-blocks (2048 columns) and slice into 4x512 for matmul.
      3) Load full hidden_states (1, 2048) at once to avoid small inefficient DMAs.
    """
    # ---- Constants (fixed shapes) ----
    K = 2048
    N = 64128

    TILE_K = 128
    TILE_N = 512

    NUM_K = K // TILE_K              # 16
    FULL_CHUNK_TILES = 4             # 4 * 512 = 2048 columns at a time
    FULL_CHUNK_N = FULL_CHUNK_TILES * TILE_N
    FULL_CHUNKS = N // FULL_CHUNK_N  # 31
    TAIL_COLS = N - FULL_CHUNKS * FULL_CHUNK_N  # 640
    TAIL_TILES = (TAIL_COLS + TILE_N - 1) // TILE_N  # 2
    TAIL_CHUNK_N = TAIL_TILES * TILE_N  # 1024

    # Partition dimension for M is 1 (single row / single token)
    P_M = 1

    # Output on HBM: keep final shape [1,1,N]
    out = nl.ndarray((1, 1, N), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Load all hidden_states at once: (1, 1, 2048) -> (1, 2048) tile in SBUF
    # Using pure basic indexing [0, 0:1, :] avoids mixed indexing errors.
    x_full = nl.load(hidden_states[0, 0:1, :]) 

    # ---- Precompute xT tiles once (SBUF residency) ----
    # xT_all[k] is a tile of shape [128, 1] with partition dim = 128 (K tile)
    xT_all = nl.ndarray((NUM_K, nl.par_dim(TILE_K), P_M),
                        dtype=hidden_states.dtype, buffer=nl.sbuf)

    for k in nl.affine_range(NUM_K):
        k0 = k * TILE_K

        # Slice x block from the full loaded tile in SBUF
        # Uses basic indexing on SBUF tile: [:, slice]
        x_blk = x_full[:, nl.ds(k0, TILE_K)]  # (1, 128) on SBUF

        # Transpose P/F => (128,1) in PSUM, then copy to SBUF
        xT_psum = nisa.nc_transpose(x_blk, engine=nisa.tensor_engine)  # (128, 1) on PSUM
        xT_all[k] = nisa.tensor_copy(xT_psum, dtype=hidden_states.dtype)  # (128, 1) on SBUF

    # ---- Main compute: full chunks (no masks needed) ----
    for chunk in nl.affine_range(FULL_CHUNKS):
        n_base = chunk * FULL_CHUNK_N  # start column in N

        # 4 independent accumulators for 4x512 tiles in this chunk
        acc = nl.zeros((FULL_CHUNK_TILES, nl.par_dim(P_M), TILE_N),
                       dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(NUM_K):
            k0 = k * TILE_K

            # Load a large RHS block [128,2048] once, slice into 4x [128,512]
            # Uses pure basic indexing: [nl.ds, nl.ds]
            w_big = nl.load(lm_head_weight[nl.ds(k0, TILE_K), nl.ds(n_base, FULL_CHUNK_N)])  # (128,2048)

            xT = xT_all[k]  # (128,1) stationary for this k

            # 4 matmuls into 4 accumulators
            for sub in range(FULL_CHUNK_TILES):
                w_tile = w_big[:, nl.ds(sub * TILE_N, TILE_N)]  # (128,512)
                acc[sub] += nisa.nc_matmul(stationary=xT, moving=w_tile)  # (1,512) on PSUM

        # Store 4 output tiles back to HBM
        for sub in range(FULL_CHUNK_TILES):
            n0 = n_base + sub * TILE_N
            acc_sb = nisa.tensor_copy(acc[sub], dtype=hidden_states.dtype)  # PSUM->SBUF cast
            
            # Use pure basic indexing for store: [0, 0:1, nl.ds]
            # Since n0 is aligned and fully in-bounds in main loop, this is safe.
            nl.store(out[0, 0:1, nl.ds(n0, TILE_N)], value=acc_sb)

    # ---- Tail chunk (masked load/store) ----
    if TAIL_COLS != 0:
        tail_base = FULL_CHUNKS * FULL_CHUNK_N  # 63488

        acc_tail = nl.zeros((TAIL_TILES, nl.par_dim(P_M), TILE_N),
                            dtype=nl.float32, buffer=nl.psum)

        # Indices for masked RHS load (avoid OOB on N)
        i_n_tail_big = tail_base + nl.arange(TAIL_CHUNK_N)[None, :]  # (1,1024)

        for k in nl.affine_range(NUM_K):
            k0 = k * TILE_K
            i_k = k0 + nl.arange(TILE_K)[:, None]  # (128,1)

            # Mask only on N dimension (K is exact)
            mask_w = (i_n_tail_big < N)
            
            # Pure advanced indexing: [i_k, i_n_tail_big]
            w_tail_big = nl.load(lm_head_weight[i_k, i_n_tail_big], mask=mask_w)  # (128,1024)

            xT = xT_all[k]

            for sub in range(TAIL_TILES):  # 2 tiles
                w_tile = w_tail_big[:, nl.ds(sub * TILE_N, TILE_N)]  # (128,512)
                acc_tail[sub] += nisa.nc_matmul(stationary=xT, moving=w_tile)

        # Masked stores for the last partial 512-tile
        # We need purely advanced indexing here to support masking/OOB handling properly.
        # We create an advanced index for the singleton dimensions 0 and 1.
        i_m_p_adv = nl.arange(P_M)[:, None]  # Shape (1, 1), Value 0

        for sub in range(TAIL_TILES):
            n0 = tail_base + sub * TILE_N
            i_n = n0 + nl.arange(TILE_N)[None, :]  # (1,512)
            mask_store = (i_n < N)

            acc_sb = nisa.tensor_copy(acc_tail[sub], dtype=hidden_states.dtype)
            
            # Pure advanced indexing: out[i_m_p_adv, i_m_p_adv, i_n]
            # Broadcasting: (1,1), (1,1), (1,512) -> (1,512) tile
            nl.store(out[i_m_p_adv, i_m_p_adv, i_n], value=acc_sb, mask=mask_store)

    return out

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

