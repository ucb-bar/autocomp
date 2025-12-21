import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(
    x_tensor: nt.tensor,                         # [1, 1, 2048]
    rms_w_tensor: nt.tensor,                     # [1, 2048]
    W_packedT: nt.tensor,                        # [2048, 8192]
    W_downT: nt.tensor,                          # [4096, 2048]
):
    """
    Optimized fused kernel for single-token MLP block.
    
    Optimizations applied:
    1. **Streaming Weight Loads**: Instead of buffering large chunks of weights (which increases latency 
       before compute starts), weights are loaded in 512-column tiles inside the pipeline. This allows 
       better overlap of DMA loads and Compute (pipelining).
    2. **Minimized Transposes**: Input vectors are transposed once and stored in SBUF for reuse across 
       all weight columns.
    3. **Efficient Memory Usage**: Tiling strategies ensure we stay well within SBUF limits while 
       maximizing tile sizes (128x512) for DMA efficiency.
    4. **Fused RMSNorm Ops**: Correctly split tensor_scalar (for broadcasting) and element-wise multiply.
    """
    # ----------------------------
    # Constants
    # ----------------------------
    K_RMS = 2048
    N_UPGATE = 8192
    N_SPLIT = 4096
    K_DOWN = 4096
    N_DOWN = 2048
    eps = 1.0e-5

    # Tile sizes
    TILE_K = nl.tile_size.pmax              # 128 (Partition Dim)
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512 (Free Dim)

    # ----------------------------
    # Output Buffer (HBM)
    # ----------------------------
    out_tensor = nl.ndarray((1, N_DOWN), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # ----------------------------
    # 1. RMSNorm (Compute in SBUF)
    # ----------------------------
    # Load input x and RMS weights
    x_sb = nl.load(x_tensor[0, 0:1, 0:K_RMS])            # (1, 2048)
    w_sb = nl.load(rms_w_tensor[0:1, 0:K_RMS])           # (1, 2048)

    # Standard RMS Norm calculation
    x2 = nl.square(x_sb)
    ms = nl.mean(x2, axis=[1], keepdims=True)            # (1, 1)
    inv_rms = nl.rsqrt(ms + eps)                         # (1, 1)

    # Fix: split the fused op because w_sb is a tensor (1, 2048), not a vector (1, 1).
    # tensor_scalar requires operand to be scalar or vector (free dim = 1).
    
    # 1. Multiply x_sb by inv_rms (1, 1). This is a broadcast, so tensor_scalar works perfectly.
    x_scaled = nisa.tensor_scalar(x_sb, op0=np.multiply, operand0=inv_rms)

    # 2. Multiply by w_sb (1, 2048). This is an element-wise multiply of two tensors of same shape.
    x_norm = nl.multiply(x_scaled, w_sb)                 # (1, 2048)

    # ----------------------------
    # 2. Transpose x_norm for Vector-Matrix Multiply
    #    We need tiles of (128, 1) to act as Stationary operand in nc_matmul.
    #    x_norm is (1, 2048). We process it in (1, 128) chunks.
    # ----------------------------
    NUM_K_RMS = K_RMS // TILE_K
    # Allocate buffer for transposed tiles
    xT_tiles = nl.ndarray((NUM_K_RMS, nl.par_dim(TILE_K), 1), dtype=nl.bfloat16, buffer=nl.sbuf)

    for k in nl.affine_range(NUM_K_RMS):
        k0 = k * TILE_K
        # Load chunk (1, 128)
        chunk = x_norm[:, nl.ds(k0, TILE_K)]
        # Transpose to (128, 1) and store. Input (1, 128) -> Output (128, 1)
        xT_tiles[k] = nisa.nc_transpose(chunk)

    # ----------------------------
    # 3. Up/Gate Projection
    #    x (1, 2048) @ W (2048, 8192) -> (1, 8192)
    #    Strategy: Iterate over N in chunks of 512 (TILE_N).
    #              Inner loop iterates K (accumulating partial sums).
    # ----------------------------
    NUM_N_UPGATE = N_UPGATE // TILE_N
    
    # Buffer for the full intermediate result in SBUF
    res_sb = nl.ndarray((nl.par_dim(1), N_UPGATE), dtype=nl.bfloat16, buffer=nl.sbuf)

    for n in nl.affine_range(NUM_N_UPGATE):
        n0 = n * TILE_N
        
        # Initialize accumulator for this output tile (1, 512)
        psum = nl.zeros((nl.par_dim(1), TILE_N), dtype=nl.float32, buffer=nl.psum)
        
        for k in nl.affine_range(NUM_K_RMS):
            k0 = k * TILE_K
            
            # Load Weight Tile (128, 512) from HBM
            # W_packedT layout is (K, N)
            w_tile = nl.load(W_packedT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_N)])
            
            # Retrieve pre-transposed x tile (128, 1)
            x_tile = xT_tiles[k]
            
            # Compute: x_tile.T @ w_tile -> (1, 128) @ (128, 512) -> (1, 512)
            psum += nisa.nc_matmul(x_tile, w_tile)
            
        # Copy result from PSUM to SBUF
        res_sb[:, nl.ds(n0, TILE_N)] = nisa.tensor_copy(psum, dtype=nl.bfloat16)

    # ----------------------------
    # 4. Activation (SiLU * Gate)
    #    Split res_sb into Up (first 4096) and Gate (next 4096)
    # ----------------------------
    # Slices are views into res_sb
    up_part = res_sb[:, 0:N_SPLIT]           # (1, 4096)
    gate_part = res_sb[:, N_SPLIT:N_UPGATE]  # (1, 4096)

    # Apply SiLU to Gate
    gate_act = nisa.activation(op=nl.silu, data=gate_part)
    # Element-wise multiply: Up * SiLU(Gate)
    act_sb = nl.multiply(up_part, gate_act)  # (1, 4096)

    # ----------------------------
    # 5. Transpose Activation for Down Proj
    #    Similar to Step 2, transpose (1, 4096) -> tiles of (128, 1)
    # ----------------------------
    NUM_K_DOWN = K_DOWN // TILE_K
    actT_tiles = nl.ndarray((NUM_K_DOWN, nl.par_dim(TILE_K), 1), dtype=nl.bfloat16, buffer=nl.sbuf)

    for k in nl.affine_range(NUM_K_DOWN):
        k0 = k * TILE_K
        chunk = act_sb[:, nl.ds(k0, TILE_K)]
        actT_tiles[k] = nisa.nc_transpose(chunk)

    # ----------------------------
    # 6. Down Projection
    #    act (1, 4096) @ W (4096, 2048) -> (1, 2048)
    #    Same streaming strategy as Up/Gate projection.
    # ----------------------------
    NUM_N_DOWN = N_DOWN // TILE_N

    for n in nl.affine_range(NUM_N_DOWN):
        n0 = n * TILE_N
        
        # Initialize accumulator
        psum = nl.zeros((nl.par_dim(1), TILE_N), dtype=nl.float32, buffer=nl.psum)
        
        for k in nl.affine_range(NUM_K_DOWN):
            k0 = k * TILE_K
            
            # Load Weight Tile (128, 512)
            w_tile = nl.load(W_downT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_N)])
            
            # Retrieve pre-transposed activation tile (128, 1)
            act_tile = actT_tiles[k]
            
            # Compute matmul
            psum += nisa.nc_matmul(act_tile, w_tile)
            
        # Store result directly to HBM (casting to bfloat16 via tensor_copy)
        out_tile = nisa.tensor_copy(psum, dtype=nl.bfloat16)
        nl.store(out_tensor[:, nl.ds(n0, TILE_N)], value=out_tile)

    return out_tensor


def forward_reference(x, post_attention_layernorm_weight, up_proj_weight, gate_proj_weight, down_proj_weight, kernel):
    # Prepare RMS weight as [1, 2048]
    rms_w_2d = post_attention_layernorm_weight.reshape(1, 2048)

    # Pack weights into [K, N] layout on host
    # Up/Gate: [2048, 8192]
    W_packedT = np.concatenate((up_proj_weight.T, gate_proj_weight.T), axis=1)

    # Down: [4096, 2048]
    W_downT = down_proj_weight.T

    # Run optimized NKI kernel
    out = kernel(x, rms_w_2d, W_packedT, W_downT)
    return out

def get_test_weights(hidden_size, intermediate_size, dtype):
    """Create test weights for MLP."""
    post_attention_layernorm_weight = np.random.randn(hidden_size).astype(dtype)
    up_proj_weight = np.random.randn(intermediate_size, hidden_size).astype(dtype)
    gate_proj_weight = np.random.randn(intermediate_size, hidden_size).astype(dtype)
    down_proj_weight = np.random.randn(hidden_size, intermediate_size).astype(dtype)
    return (
        post_attention_layernorm_weight,
        up_proj_weight,
        gate_proj_weight,
        down_proj_weight,
    )


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
    hidden_size = 2048
    intermediate_size = 4096
    weights = get_test_weights(hidden_size, intermediate_size, dtype)
    
    for _ in range(2):
        batch, seq = 1, 1
        x = np.random.randn(batch, seq, hidden_size).astype(dtype)
        ref_out = forward_reference(x, *weights, kernel=ref_func)
        test_out = forward_reference(x, *weights, kernel=test_func)
        if not compare_outputs(ref_out, test_out):
            return False
    return True

def benchmark_nki(nki_func):
    hidden_size = 2048
    intermediate_size = 4096
    H = hidden_size
    U = intermediate_size
    D = hidden_size
    
    x_tensor = nt.tensor[[1, 1, H], nl.bfloat16]
    gamma = nt.tensor[[1, H], nl.bfloat16]
    ug_wT = nt.tensor[[H, 2 * U], nl.bfloat16]
    down_wT = nt.tensor[[U, D], nl.bfloat16]
    
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(x_tensor, gamma, ug_wT, down_wT)
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
        # benchmark_nki(nki_fused_mlp_kernel_reference)
        benchmark_nki(test)
