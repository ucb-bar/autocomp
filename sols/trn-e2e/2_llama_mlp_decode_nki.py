@nki.jit
def test(lhs, up_w, gate_w, down_w):
    """
    Fused NKI kernel for MLP layer:
    1. up = lhs @ up_w
    2. gate = lhs @ gate_w
    3. act = silu(gate) * up
    4. out = act @ down_w
    
    Optimized for x.shape = (1, 1, 2048). up_w.shape = (2048, 4096), gate_w.shape = (2048, 4096), down_w.shape = (4096, 2048).
    Accepts 3D input (B, S, K) and produces 3D output (B, S, N_out) directly,
    eliminating external reshape overhead.
    """
    
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
