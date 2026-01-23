def PROMPT():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
@nki.jit
def nki_fused_all_proj_kernel(x_tensor, g_tensor, up_wT, gate_wT, down_wT):
    '''
    Fused RMSNorm + up/gate projections + SiLU(gate)*up + down projection.

    Inputs:
      x_tensor : [R, H]
      g_tensor : [H]
      up_wT    : [H, U]   (transposed weight; K-major)
      gate_wT  : [H, U]   (transposed weight; K-major)
      down_wT  : [U, D]   (transposed weight; K-major)

    Output:
      out      : [R, D]
    '''
    R, H = x_tensor.shape
    H2, U = up_wT.shape
    H3, U2 = gate_wT.shape
    U3, D = down_wT.shape
    assert H2 == H and H3 == H and U2 == U and U3 == U

    out = nl.ndarray((R, D), dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    # Indices used for loads
    i_h = nl.arange(H)[None, :]          # [1, H]
    i_one = nl.arange(1)[:, None]        # [1, 1]

    # Load RMSNorm scale once into SBUF as [1, H]
    g_tile = nl.load(g_tensor.reshape((1, H))[i_one, i_h])  # [1, H]

    # Row tiling
    P_TILE = min(nl.tile_size.pmax, R)   # compile-time constant per specialization
    i_p = nl.arange(P_TILE)[:, None]     # [P_TILE, 1]
    trip = (R + P_TILE - 1) // P_TILE

    # Common tile sizes
    TILE_K = nl.tile_size.pmax  # 128 (contraction for H and for down\'s K-subtiles)

    # Down-proj tiling (assume common LLM shape: D multiple of 512)
    TILE_D = nl.tile_size.gemm_moving_fmax  # typically 512
    assert (D % TILE_D) == 0
    num_d = D // TILE_D
    # Keep on-chip accumulator count bounded (covers D up to 2048 with TILE_D=512)
    assert num_d <= 4

    # Up/gate output tiling
    do_swap = (P_TILE < nl.tile_size.pmax) and (U >= nl.tile_size.pmax)

    # H tiling (assume common LLM shape: H multiple of 128)
    assert (H % TILE_K) == 0
    num_k = H // TILE_K

    for p in nl.affine_range(trip):
        row_idx = p * P_TILE + i_p                # [P_TILE, 1]
        row_mask = (row_idx < R)                  # [P_TILE, 1]

        # Load one [P_TILE, H] tile
        x_tile = nl.load(x_tensor[row_idx, i_h], mask=row_mask)

        # RMSNorm: y = x / rms(x) * g
        sq = nl.square(x_tile, mask=row_mask)
        sq_sum = nl.sum(sq, axis=[1], mask=row_mask)            # [P_TILE, 1]
        mean = sq_sum / float(H)                                # [P_TILE, 1]
        inv_rms = nl.rsqrt(mean + 1.0e-5, mask=row_mask)         # [P_TILE, 1]
        y_tile = nl.multiply(x_tile, inv_rms, mask=row_mask)
        y_tile = nl.multiply(y_tile, g_tile, mask=row_mask)      # g_tile broadcasts on partition

        # Down-proj accumulators (keep in PSUM across all U tiles)
        acc_down0 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down1 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down2 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down3 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)

        if do_swap:
            # Swap path for skinny M: produce [M, 128] act tiles
            TILE_U = nl.tile_size.pmax  # 128
            assert (U % TILE_U) == 0
            num_n = U // TILE_U

            # Precompute y^T blocks in SBUF:
            # yT_cat: [Ktile=128, num_k * M], slice out [128, M] per k
            yT_cat = nl.ndarray((nl.par_dim(TILE_K), num_k * P_TILE),
                                dtype=x_tensor.dtype, buffer=nl.sbuf)

            for k in nl.affine_range(num_k):
                k0 = k * TILE_K
                y_block = y_tile[:, nl.ds(k0, TILE_K)]  # [M, 128]
                yT_psum = nisa.nc_transpose(y_block, engine=nisa.tensor_engine)  # PSUM [128, M]
                yT_sb = nisa.tensor_copy(yT_psum, dtype=x_tensor.dtype)          # SBUF [128, M]
                yT_cat[:, nl.ds(k * P_TILE, P_TILE)] = yT_sb

            for n in nl.affine_range(num_n):
                n0 = n * TILE_U

                # up/gate accumulators in swapped layout: [128, M] on PSUM
                acc_up = nl.zeros((nl.par_dim(TILE_U), P_TILE), dtype=nl.float32, buffer=nl.psum)
                acc_gate = nl.zeros((nl.par_dim(TILE_U), P_TILE), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    lhsT = yT_cat[:, nl.ds(k * P_TILE, P_TILE)]  # [K=128, M]
                    up_w_tile = nl.load(up_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])        # [K=128, N=128]
                    gate_w_tile = nl.load(gate_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])    # [K=128, N=128]
                    acc_up += nl.matmul(up_w_tile, lhsT, transpose_x=True)                  # [N, M]
                    acc_gate += nl.matmul(gate_w_tile, lhsT, transpose_x=True)              # [N, M]

                # Convert to [M, 128] in SBUF for SiLU/multiply
                up_nm_sb = nisa.tensor_copy(acc_up, dtype=x_tensor.dtype)       # SBUF [128, M]
                gate_nm_sb = nisa.tensor_copy(acc_gate, dtype=x_tensor.dtype)   # SBUF [128, M]

                up_mn_psum = nisa.nc_transpose(up_nm_sb, engine=nisa.tensor_engine)         # PSUM [M, 128]
                gate_mn_psum = nisa.nc_transpose(gate_nm_sb, engine=nisa.tensor_engine)     # PSUM [M, 128]
                up_mn = nisa.tensor_copy(up_mn_psum, dtype=x_tensor.dtype)                  # SBUF [M, 128]
                gate_mn = nisa.tensor_copy(gate_mn_psum, dtype=x_tensor.dtype)              # SBUF [M, 128]

                gate_silu = nisa.activation(op=nl.silu, data=gate_mn, dtype=gate_mn.dtype)  # SBUF [M, 128]
                act_mn = nl.multiply(up_mn, gate_silu)                                      # SBUF [M, 128]

                # Down-proj update: act_mn [M,128] @ down_wT[n0:n0+128, d0:d0+512]
                down0 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(0 * TILE_D, TILE_D)])
                acc_down0 += nl.matmul(act_mn, down0)

                if num_d > 1:
                    down1 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(1 * TILE_D, TILE_D)])
                    acc_down1 += nl.matmul(act_mn, down1)
                if num_d > 2:
                    down2 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(2 * TILE_D, TILE_D)])
                    acc_down2 += nl.matmul(act_mn, down2)
                if num_d > 3:
                    down3 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(3 * TILE_D, TILE_D)])
                    acc_down3 += nl.matmul(act_mn, down3)

        else:
            # Standard path: compute up/gate in [M, 512] tiles, then split into 128-K chunks for down
            TILE_U = nl.tile_size.gemm_moving_fmax  # typically 512
            assert (U % TILE_U) == 0
            num_n = U // TILE_U

            for n in nl.affine_range(num_n):
                n0 = n * TILE_U

                acc_up = nl.zeros((nl.par_dim(P_TILE), TILE_U), dtype=nl.float32, buffer=nl.psum)
                acc_gate = nl.zeros((nl.par_dim(P_TILE), TILE_U), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    x_block = y_tile[:, nl.ds(k0, TILE_K)]                                     # [M, 128]
                    up_w_tile = nl.load(up_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])           # [128, 512]
                    gate_w_tile = nl.load(gate_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])       # [128, 512]
                    acc_up += nl.matmul(x_block, up_w_tile)                                    # [M, 512]
                    acc_gate += nl.matmul(x_block, gate_w_tile)                                # [M, 512]

                up_sb = nisa.tensor_copy(acc_up, dtype=x_tensor.dtype)                          # SBUF [M, 512]
                gate_sb = nisa.tensor_copy(acc_gate, dtype=x_tensor.dtype)                      # SBUF [M, 512]
                gate_silu = nisa.activation(op=nl.silu, data=gate_sb, dtype=gate_sb.dtype)      # SBUF [M, 512]
                act_sb = nl.multiply(up_sb, gate_silu)                                          # SBUF [M, 512]

                # Down-proj in 4x128 chunks to satisfy matmul K<=128
                # act_blk: [M,128], down_blk: [128,512] => [M,512]
                for u_sub in range(0, TILE_U, TILE_K):
                    u_base = n0 + u_sub
                    act_blk = act_sb[:, nl.ds(u_sub, TILE_K)]                                   # [M, 128]

                    down0 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(0 * TILE_D, TILE_D)])
                    acc_down0 += nl.matmul(act_blk, down0)

                    if num_d > 1:
                        down1 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(1 * TILE_D, TILE_D)])
                        acc_down1 += nl.matmul(act_blk, down1)
                    if num_d > 2:
                        down2 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(2 * TILE_D, TILE_D)])
                        acc_down2 += nl.matmul(act_blk, down2)
                    if num_d > 3:
                        down3 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(3 * TILE_D, TILE_D)])
                        acc_down3 += nl.matmul(act_blk, down3)

        # Store down-proj output tiles
        i_d = nl.arange(TILE_D)[None, :]

        out0_sb = nisa.tensor_copy(acc_down0, dtype=x_tensor.dtype)
        nl.store(out[row_idx, 0 * TILE_D + i_d], value=out0_sb, mask=row_mask)

        if num_d > 1:
            out1_sb = nisa.tensor_copy(acc_down1, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 1 * TILE_D + i_d], value=out1_sb, mask=row_mask)
        if num_d > 2:
            out2_sb = nisa.tensor_copy(acc_down2, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 2 * TILE_D + i_d], value=out2_sb, mask=row_mask)
        if num_d > 3:
            out3_sb = nisa.tensor_copy(acc_down3, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 3 * TILE_D + i_d], value=out3_sb, mask=row_mask)

    return out


def test(x, post_attention_layernorm_weight, up_proj_weight, gate_proj_weight, down_proj_weight):
    # Shapes (prefill example):
    # x: (1, 32, 2048)
    # weights: up/gate (8192, 2048), down (2048, 8192)
    b, s, h = x.shape

    # Flatten batch*seq to match kernels (R, H)
    x2d = x.view(-1, h)

    # One kernel: RMSNorm + up/gate + SiLU*up + down
    output = nki_fused_all_proj_kernel(
        x2d,
        post_attention_layernorm_weight,
        up_proj_weight.t(),
        gate_proj_weight.t(),
        down_proj_weight.t(),
    )  # (R, 2048)

    return output
"""

def PROMPT_2():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
@nki.jit
def test(x_tensor, gamma, ug_wT, down_wT):
    '''
    Optimized NKI kernel using larger load blocks (Optimization 8).
    Processing R=1, H=2048, U=8192, D=2048.
    '''
    # --- Shapes ---
    R, H = x_tensor.shape          # (1, 2048)
    H2, ug_cols = ug_wT.shape      # (2048, 16384)
    U, D = down_wT.shape           # (8192, 2048)

    # Basic constants
    TILE_K = 128
    TILE_D = 512
    num_k = 16          # 2048/128
    num_d = 4           # 2048/512

    PACK_TILE_U = 256   
    pack_blocks = 32    # 8192/256 total blocks

    # New blocking for Optimization 8: Process 4 blocks at once
    CHUNK_SIZE = 4 
    num_chunks = pack_blocks // CHUNK_SIZE # 32 // 4 = 8

    P_TILE = 1
    i_p = nl.arange(P_TILE)[:, None]      
    i_h = nl.arange(H)[None, :]           
    i_d = nl.arange(TILE_D)[None, :]      

    # Initialize Output
    out = nl.ndarray((R, D), dtype=x_tensor.dtype, buffer=nl.shared_hbm)
    
    # Output accumulator in SBUF
    out_acc = nl.zeros((nl.par_dim(P_TILE), D), dtype=nl.float32, buffer=nl.sbuf)

    # Load inputs and perform RMSNorm
    g_tile = nl.load(gamma.reshape((1, H))[0:1, nl.ds(0, H)])
    row_idx = i_p
    row_mask = (row_idx < R)
    x_tile = nl.load(x_tensor[row_idx, i_h], mask=row_mask)

    sq = nl.square(x_tile, mask=row_mask)
    sq_sum = nl.sum(sq, axis=[1], mask=row_mask)             
    mean = sq_sum / float(H)
    inv_rms = nl.rsqrt(mean + 1.0e-5, mask=row_mask)         
    y_tile = nl.multiply(x_tile, inv_rms, mask=row_mask)     
    y_tile = nl.multiply(y_tile, g_tile, mask=row_mask)      

    # Precompute transposed y tiles
    yT_all = nl.ndarray((num_k, nl.par_dim(TILE_K), P_TILE),
                        dtype=x_tensor.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(num_k):
        k0 = k * TILE_K
        y_blk = y_tile[:, nl.ds(k0, TILE_K)]  
        yT_psum = nisa.nc_transpose(y_blk, engine=nisa.tensor_engine)  
        yT_all[k, :, :] = nisa.tensor_copy(yT_psum, dtype=x_tensor.dtype)  

    # Main Loop: Iterate over chunks of blocks
    for chunk in nl.affine_range(num_chunks):
        chunk_idx = chunk * CHUNK_SIZE 
        
        # Allocate accumulators for the 4 sub-blocks in this chunk.
        # Shape: (4, 128, 1). Using CHUNK_SIZE as outer dim.
        # Note: nl.par_dim(TILE_K) marks the 128 dim as partition dim for the slices.
        acc_up0_all = nl.zeros((CHUNK_SIZE, nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)
        acc_gate0_all = nl.zeros((CHUNK_SIZE, nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)
        acc_up1_all = nl.zeros((CHUNK_SIZE, nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)
        acc_gate1_all = nl.zeros((CHUNK_SIZE, nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)

        # Base column index for ug_wT for this chunk
        base_ug_col = chunk * 2048

        # 1. Accumulate Up/Gate projections for the entire chunk
        for k in nl.affine_range(num_k):
            k0 = k * TILE_K
            yT_sb = yT_all[k]

            # Optimization: Load LARGE block [128, 2048] from ug_wT
            ug_big = nl.load(ug_wT[nl.ds(k0, TILE_K), nl.ds(base_ug_col, 2048)]) 
            
            # Distribute computation to sub-blocks
            for sub in nl.affine_range(CHUNK_SIZE):
                sub_col_offset = sub * 512
                
                # Slice the large loaded tile
                # ug_big is [128, 2048]. Slicing on dim 1.
                up_w0 = ug_big[:, nl.ds(sub_col_offset + 0 * TILE_K, TILE_K)]
                up_w1 = ug_big[:, nl.ds(sub_col_offset + 1 * TILE_K, TILE_K)]
                gate_w0 = ug_big[:, nl.ds(sub_col_offset + 2 * TILE_K, TILE_K)]
                gate_w1 = ug_big[:, nl.ds(sub_col_offset + 3 * TILE_K, TILE_K)]

                acc_up0_all[sub] += nl.matmul(up_w0, yT_sb, transpose_x=True)
                acc_gate0_all[sub] += nl.matmul(gate_w0, yT_sb, transpose_x=True)
                acc_up1_all[sub] += nl.matmul(up_w1, yT_sb, transpose_x=True)
                acc_gate1_all[sub] += nl.matmul(gate_w1, yT_sb, transpose_x=True)

        # 2. Finalize and Down Projection for each sub-block in the chunk
        for sub in nl.affine_range(CHUNK_SIZE):
            nb = chunk_idx + sub
            pack_u0 = nb * PACK_TILE_U
            
            # --- Half 0 ---
            # Copy from PSUM accumulator to SBUF
            up_nm_sb0 = nisa.tensor_copy(acc_up0_all[sub], dtype=x_tensor.dtype)
            gate_nm_sb0 = nisa.tensor_copy(acc_gate0_all[sub], dtype=x_tensor.dtype)
            
            # Transpose to get [1, 128]
            up_mn0 = nisa.tensor_copy(nisa.nc_transpose(up_nm_sb0, engine=nisa.tensor_engine), dtype=x_tensor.dtype)
            gate_mn0 = nisa.tensor_copy(nisa.nc_transpose(gate_nm_sb0, engine=nisa.tensor_engine), dtype=x_tensor.dtype)
            
            # SiLU and element-wise mul
            gate_silu0 = nisa.activation(op=nl.silu, data=gate_mn0, dtype=gate_mn0.dtype)
            act_mn0 = nl.multiply(up_mn0, gate_silu0)

            n0 = pack_u0
            
            # Optimization: Load LARGE down_wT row strip [128, 2048]
            # Covers all di blocks (4 * 512 = 2048 columns) in one load.
            down_big0 = nl.load(down_wT[nl.ds(n0, TILE_K), nl.ds(0, 4 * TILE_D)]) 

            for di in nl.affine_range(num_d):
                d0 = di * TILE_D
                # Slice from SBUF tile for matmul
                down_blk = down_big0[:, nl.ds(d0, TILE_D)]
                res = nl.matmul(act_mn0, down_blk)
                out_acc[:, nl.ds(d0, TILE_D)] += res

            # --- Half 1 ---
            up_nm_sb1 = nisa.tensor_copy(acc_up1_all[sub], dtype=x_tensor.dtype)
            gate_nm_sb1 = nisa.tensor_copy(acc_gate1_all[sub], dtype=x_tensor.dtype)
            
            up_mn1 = nisa.tensor_copy(nisa.nc_transpose(up_nm_sb1, engine=nisa.tensor_engine), dtype=x_tensor.dtype)
            gate_mn1 = nisa.tensor_copy(nisa.nc_transpose(gate_nm_sb1, engine=nisa.tensor_engine), dtype=x_tensor.dtype)
            
            gate_silu1 = nisa.activation(op=nl.silu, data=gate_mn1, dtype=gate_mn1.dtype)
            act_mn1 = nl.multiply(up_mn1, gate_silu1)

            n1 = pack_u0 + TILE_K
            
            # Optimization: Load LARGE down_wT row strip [128, 2048]
            down_big1 = nl.load(down_wT[nl.ds(n1, TILE_K), nl.ds(0, 4 * TILE_D)])

            for di in nl.affine_range(num_d):
                d0 = di * TILE_D
                # Slice from SBUF tile for matmul
                down_blk = down_big1[:, nl.ds(d0, TILE_D)]
                res = nl.matmul(act_mn1, down_blk)
                out_acc[:, nl.ds(d0, TILE_D)] += res

    # Store result
    for di in nl.affine_range(num_d):
        d0 = di * TILE_D
        out_tile = nisa.tensor_copy(out_acc[:, nl.ds(d0, TILE_D)], dtype=x_tensor.dtype)
        nl.store(out[row_idx, d0 + i_d], value=out_tile, mask=row_mask)

    return out
"""

def PROMPT_3():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
```
@nki.jit
def nki_fused_layer_(lhs, up_w, gate_w, down_w):
    '''
    Fused NKI kernel for MLP layer:
    1. up = lhs @ up_w
    2. gate = lhs @ gate_w
    3. act = silu(gate) * up
    4. out = act @ down_w
    
    Optimized for x.shape = (1, 1, 2048).
    Accepts 3D input (B, S, K) and produces 3D output (B, S, N_out) directly,
    eliminating external reshape overhead.
    '''
    
    B, S, K_in = lhs.shape
    _, N_inter = up_w.shape
    _, N_out = down_w.shape
    
    # Output tensor with 3D shape (B, S, N_out)
    out_hbm = nl.ndarray((B, S, N_out), dtype=lhs.dtype, buffer=nl.shared_hbm)
    
    # Tile sizes
    # Optimized for M=1 inference case (B=1, S=1)
    TILE_M = 1 
    TILE_K_IN = 128
    TILE_N_INTER = 128  
    TILE_N_OUT = 512    
    
    # Pad N_out for SBUF accumulator alignment
    N_out_padded = (N_out + TILE_N_OUT - 1) // TILE_N_OUT * TILE_N_OUT

    # Indices for Batch and Sequence dimensions
    # For B=1, S=1, these are fixed to 0. 
    # Use nl.arange to create index tiles compatible with advanced indexing.
    # Avoiding nl.zeros ensures we don\'t trigger "type: tensor is not supported" error in nl.load.
    i_p0 = nl.arange(TILE_M)[:, None] 
    i_b = i_p0 * 0
    i_s = i_p0 * 0

    # Main loop over batch/sequence (Runs once for B=1, S=1)
    for _ in nl.affine_range(1):
        
        # Allocate Accumulator in SBUF [TILE_M, N_out_padded] -> [1, N_out_padded]
        acc_sbuf = nl.zeros((TILE_M, N_out_padded), dtype=nl.float32, buffer=nl.sbuf)
        
        # Loop over Intermediate Dimension (N_inter) - Sequential for accumulation on acc_sbuf
        for n_i in nl.sequential_range((N_inter + TILE_N_INTER - 1) // TILE_N_INTER):
            
            # PSUM Accumulators for [TILE_M, TILE_N_INTER] -> [1, 128]
            psum_up = nl.zeros((TILE_M, TILE_N_INTER), dtype=nl.float32, buffer=nl.psum)
            psum_gate = nl.zeros((TILE_M, TILE_N_INTER), dtype=nl.float32, buffer=nl.psum)
            
            i_ni = n_i * TILE_N_INTER + nl.arange(TILE_N_INTER)[None, :]
            mask_ni = (i_ni < N_inter) 

            # Reduce over Input Dimension (K_in)
            for k_in in nl.affine_range((K_in + TILE_K_IN - 1) // TILE_K_IN):
                
                # --- Efficient Load and Transpose of LHS ---
                # Load lhs[0, 0, k_slice] -> Tile [1, 128]
                # Using 3D indexing: lhs[i_b, i_s, i_ki_lhs] with broadcasted indices
                i_ki_lhs = k_in * TILE_K_IN + nl.arange(TILE_K_IN)[None, :]
                mask_k = (i_ki_lhs < K_in)
                
                lhs_tile = nl.load(lhs[i_b, i_s, i_ki_lhs], mask=mask_k)
                
                # Transpose [1, 128] -> [128, 1] using Tensor Engine to allow contiguous K load
                lhs_tile_T_psum = nisa.nc_transpose(lhs_tile, engine=nisa.tensor_engine)
                lhs_tile_T = nisa.tensor_copy(lhs_tile_T_psum, dtype=lhs.dtype)
                
                # --- Load Weights ---
                i_ki_w = k_in * TILE_K_IN + nl.arange(TILE_K_IN)[:, None]
                mask_w = (i_ki_w < K_in) & mask_ni
                
                up_w_tile = nl.load(up_w[i_ki_w, i_ni], mask=mask_w)
                gate_w_tile = nl.load(gate_w[i_ki_w, i_ni], mask=mask_w)
                
                # --- Matmul ---
                # lhs_tile_T [128, 1] (Stat) @ Weight [128, 128] (Mov) -> [1, 128]
                psum_up += nl.matmul(lhs_tile_T, up_w_tile, transpose_x=True)
                psum_gate += nl.matmul(lhs_tile_T, gate_w_tile, transpose_x=True)
            
            # Copy to SBUF
            sb_up = nisa.tensor_copy(psum_up, dtype=lhs.dtype)
            sb_gate = nisa.tensor_copy(psum_gate, dtype=lhs.dtype)
            
            # ---------------------------------------------------------
            # Step 2: Activation (SiLU * Up)
            # ---------------------------------------------------------
            
            # Transpose to [128, 1] for next matmul stationary
            sb_up_T_psum = nisa.nc_transpose(sb_up, engine=nisa.tensor_engine)
            sb_gate_T_psum = nisa.nc_transpose(sb_gate, engine=nisa.tensor_engine)
            
            sb_up_T = nisa.tensor_copy(sb_up_T_psum, engine=nisa.vector_engine)
            sb_gate_T = nisa.tensor_copy(sb_gate_T_psum, engine=nisa.vector_engine)
            
            # Act_T [128, 1]
            act_T = nl.multiply(nl.silu(sb_gate_T), sb_up_T)
            
            # ---------------------------------------------------------
            # Step 3: Down Projection and Accumulate
            # ---------------------------------------------------------
            
            i_ni_col = n_i * TILE_N_INTER + nl.arange(TILE_N_INTER)[:, None]
            mask_ni_col = (i_ni_col < N_inter) 
            
            for n_o in nl.affine_range(N_out_padded // TILE_N_OUT):
                idx_no = n_o * TILE_N_OUT
                i_no = idx_no + nl.arange(TILE_N_OUT)[None, :]
                
                mask_no = (i_no < N_out)
                mask_dw = mask_ni_col & mask_no
                
                down_w_tile = nl.load(down_w[i_ni_col, i_no], mask=mask_dw)
                
                # Matmul: act_T [128, 1] @ down_w [128, 512] -> [1, 512]
                res_psum = nisa.nc_matmul(act_T, down_w_tile)
                
                # Accumulate result into SBUF
                curr_acc = acc_sbuf[:, nl.ds(idx_no, TILE_N_OUT)]
                new_acc = nl.add(curr_acc, res_psum)
                acc_sbuf[:, nl.ds(idx_no, TILE_N_OUT)] = new_acc

        # ---------------------------------------------------------
        # Step 4: Store Final Output to HBM
        # ---------------------------------------------------------
        for n_o in nl.affine_range(N_out_padded // TILE_N_OUT):
            idx_no = n_o * TILE_N_OUT
            i_no_out = idx_no + nl.arange(TILE_N_OUT)[None, :]
            
            mask_out = (i_no_out < N_out)
            
            tile_out = acc_sbuf[:, nl.ds(idx_no, TILE_N_OUT)]
            
            # Store to 3D output using 3D indexing
            nl.store(out_hbm[i_b, i_s, i_no_out], value=tile_out, mask=mask_out)
            
    return out_hbm

def test(x, up_proj_weight, gate_proj_weight, down_proj_weight):
    # Direct NKI kernel call with 3D input; view operations are removed.
    return nki_fused_layer_(x, up_proj_weight, gate_proj_weight, down_proj_weight)
"""