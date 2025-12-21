import math
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

@nki.jit
def test(
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
