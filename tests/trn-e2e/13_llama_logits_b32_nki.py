import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(hidden_states, lm_head_weight):
    """
    Optimized for constant shapes (batch 32 version):
      hidden_states: [32, 1, 2048]
      lm_head_weight: [2048, 64128]
    Returns:
      out: [32, 1, 64128]

    Key optimizations:
      1) Precompute x^T tiles once per batch: hidden_states[b,1,128] -> transpose -> xT[128,1] in SBUF.
      2) Load RHS in larger contiguous N-blocks (2048 columns) and slice into 4x512 for matmul.
      3) Load full hidden_states per batch (1, 2048) at once to avoid small inefficient DMAs.
    """
    # ---- Constants (fixed shapes) ----
    B = 32
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

    # Output on HBM: keep final shape [32,1,N]
    out = nl.ndarray((B, 1, N), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # ---- Process each batch element ----
    for b in nl.affine_range(B):
        # Load all hidden_states for this batch at once: (32, 1, 2048) -> (1, 2048) tile in SBUF
        # Using pure basic indexing [b, 0:1, :] avoids mixed indexing errors.
        x_full = nl.load(hidden_states[b, 0:1, :]) 

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
                
                # Use pure basic indexing for store: [b, 0:1, nl.ds]
                # Since n0 is aligned and fully in-bounds in main loop, this is safe.
                nl.store(out[b, 0:1, nl.ds(n0, TILE_N)], value=acc_sb)

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
            # Create indices for all dimensions
            for sub in range(TAIL_TILES):
                n0 = tail_base + sub * TILE_N
                i_n = n0 + nl.arange(TILE_N)[None, :]  # (1,512)
                mask_store = (i_n < N)

                acc_sb = nisa.tensor_copy(acc_tail[sub], dtype=hidden_states.dtype)
                
                # Create batch and M indices for advanced indexing
                # For batch dimension: create a (1,1) array with value b
                # Use arange to create base, then adjust to get b
                i_b_base = nl.arange(1)[:, None]  # (1, 1) with value 0
                i_b_adv = i_b_base * 0 + b  # (1, 1) with value b
                i_m_p_adv = nl.arange(P_M)[:, None]  # (1, 1), Value 0
                
                # Pure advanced indexing: out[i_b_adv, i_m_p_adv, i_n]
                # Broadcasting: (1,1), (1,1), (1,512) -> (1,512) tile
                nl.store(out[i_b_adv, i_m_p_adv, i_n], value=acc_sb, mask=mask_store)

    return out

def get_test_data(dtype):
    """Create test data for logits kernel."""
    # hidden_states: (32, 1, 2048)
    hidden_states = np.random.randn(32, 1, 2048).astype(dtype)
    # lm_head_weight: (2048, 64128)
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
        # Transpose hidden_states from (32, 1, 2048) to (32, 2048, 1) for batched matmul
        # The reference expects [B, K, M] where B=32, K=2048, M=1
        # hidden_states_T = hidden_states.swapaxes(1, 2)  # (32, 2048, 1)
        ref_out = ref_func(hidden_states, lm_head_weight)  # (32, 1, 64128)
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
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)

