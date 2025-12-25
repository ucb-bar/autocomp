@nki.jit
def test(Q, K, V, past_k, past_v, attention_mask):
    """
    Token-generation attention kernel in NKI.
    Optimized for Trainium/Inferentia.
    Fixes the partition broadcast error by explicitly broadcasting the mask.
    Hoists Q transpose out of the group loop for performance (Opt #2).
    """
    out_hbm = nl.ndarray(Q.shape, dtype=Q.dtype, buffer=nl.shared_hbm)

    # Common index tiles
    i_p1   = nl.arange(1)[:, None]      # (1,1)
    i_p4   = nl.arange(4)[:, None]      # (4,1)
    i_p16  = nl.arange(16)[:, None]     # (16,1)
    i_p64  = nl.arange(64)[:, None]     # (64,1)
    i_p128 = nl.arange(128)[:, None]    # (128,1)

    i_f4   = nl.arange(4)[None, :]      # (1,4)
    i_f64  = nl.arange(64)[None, :]     # (1,64)
    i_f128 = nl.arange(128)[None, :]    # (1,128)
    i_f512 = nl.arange(512)[None, :]    # (1,512)

    # Load mask once: shape (1,512)
    mask_1x512 = nl.load(attention_mask[0, 0, i_p1, i_f512])
    # Explicitly broadcast mask to (4, 512) to match head groups. 
    # This prevents the "Unexpected partition broadcast" assertion error in nl.where.
    mask_4x512 = nl.broadcast_to(mask_1x512, shape=(4, 512))

    # -- Hoist Q-transpose out of the group loop --
    # Load all 16 heads of Q: (16, 64)
    Q_all = nl.load(Q[0, i_p16, 0, i_f64])
    # Transpose Q to (64, 16) stationary for nc_matmul. Result in PSUM.
    Q_all_T = nisa.nc_transpose(Q_all)
    # Copy back to SBUF for slicing. qT_all is (64, 16)
    qT_all = nisa.tensor_copy(Q_all_T, dtype=Q.dtype)

    inv_sqrt_d = 0.125
    neg_inf = -3.4028235e38

    # Process 4 KV groups
    for g in range(4):
        h0 = g * 4

        # Extract the 4-wide slice of qT_all for this group
        # qT_all is (64, 16), first dim is partition.
        # We want the columns corresponding to heads [h0 : h0+4].
        # Result qT is (64, 4), which acts as stationary for nc_matmul (K=64, M=4).
        qT = qT_all[i_p64, (g * 4) + i_f4]

        # ---- Load K active: (64,1) using transpose load pattern (indices on dim 3)
        k_act = nl.load(K[0, g, 0, i_p64])
        
        # Active score: qT.T (4,64) @ k_act (64,1) -> (4,1)
        score_act_psum = nisa.nc_matmul(qT, k_act)
        score_act = nisa.tensor_scalar(score_act_psum, op0=np.multiply, operand0=inv_sqrt_d, dtype=nl.float32)

        # ---- Construct K_prior^T (64, 512) in SBUF
        k_prior_T = nl.ndarray((64, 512), dtype=K.dtype, buffer=nl.sbuf)

        for j in range(4):
            seq0 = j * 128
            # Load chunk (128,64)
            k_chunk = nl.load(past_k[0, g, seq0 + i_p128, i_f64])
            # Transpose to (64,128)
            k_chunk_T = nisa.tensor_copy(nisa.nc_transpose(k_chunk), dtype=K.dtype)
            # Copy to larger buffer
            k_prior_T[i_p64, seq0 + i_f128] = k_chunk_T[i_p64, i_f128]

        # Prior scores: qT.T (4,64) @ k_prior_T (64,512) -> (4,512)
        score_prior_psum = nisa.nc_matmul(qT, k_prior_T)
        score_prior = nisa.tensor_scalar(score_prior_psum, op0=np.multiply, operand0=inv_sqrt_d, dtype=nl.float32)

        # Apply mask using explicit broadcast tile (4, 512)
        score_prior_masked = nl.where(mask_4x512, score_prior, neg_inf)

        # ---- Softmax
        # Max over prior (4,512) -> (4,1)
        max_prior = nl.max(score_prior_masked, axis=1, keepdims=True)
        # Combine with active score max
        max_val = nl.maximum(max_prior, score_act)

        score_prior_shifted = nl.subtract(score_prior_masked, max_val)
        score_act_shifted = nl.subtract(score_act, max_val)

        # Exp + Sum Prior
        sum_exp = nl.zeros((4, 1), dtype=nl.float32, buffer=nl.sbuf)
        exp_prior = nisa.activation(op=nl.exp, data=score_prior_shifted, 
                                    reduce_op=nl.add, reduce_res=sum_exp, reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                    dtype=nl.float32)
        
        exp_act = nl.exp(score_act_shifted)
        denom = nl.add(sum_exp, exp_act)

        # Calculate probabilities
        probs_prior = nisa.tensor_copy(nl.divide(exp_prior, denom), dtype=Q.dtype)
        probs_act = nisa.tensor_copy(nl.divide(exp_act, denom), dtype=Q.dtype)

        # ---- Attn Prior: probs_prior (4,512) @ V_prior (512,64) -> (4,64)
        attn_prior_psum = nl.zeros((4, 64), dtype=nl.float32, buffer=nl.psum)

        for j in range(4):
            seq0 = j * 128
            # Take prob chunk (4,128)
            p_chunk = probs_prior[i_p4, seq0 + i_f128]
            # Transpose to (128,4) stationary
            p_chunk_T = nisa.tensor_copy(nisa.nc_transpose(p_chunk), dtype=Q.dtype)
            
            # Load V chunk (128,64)
            v_chunk = nl.load(past_v[0, g, seq0 + i_p128, i_f64])
            
            # Matmul
            attn_prior_psum += nisa.nc_matmul(p_chunk_T, v_chunk)

        attn_prior = nisa.tensor_copy(attn_prior_psum, dtype=Q.dtype)

        # ---- Attn Active
        v_act = nl.load(V[0, g, i_p1, i_f64]) # (1,64)
        # Explicit broadcast to (4, 64) to safely match partition dimension
        v_act_4x64 = nl.broadcast_to(v_act, shape=(4, 64))
        # probs_act is (4,1), v_act_4x64 is (4,64) -> result (4,64)
        attn_act = nl.multiply(probs_act, v_act_4x64)

        # Final accumulation and store
        out_tile = nl.add(attn_prior, attn_act)
        nl.store(out_hbm[0, h0 + i_p4, 0, i_f64], out_tile)

    return out_hbm
