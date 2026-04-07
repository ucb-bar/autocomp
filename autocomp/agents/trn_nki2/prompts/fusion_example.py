def PROMPT():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
@nki.jit
def nki_fused_all_proj_kernel(x_tensor, g_tensor, up_wT, gate_wT, down_wT):
    '''
    Fused RMSNorm + up/gate projections + SiLU(gate)*up + down projection.
    Beta 2, standard prefill layout (TILE_M = 128).

    Inputs:
      x_tensor : [R, H]   input activations
      g_tensor : [H]      RMSNorm scale weights
      up_wT    : [H, U]   up-projection weight (K-major in HBM)
      gate_wT  : [H, U]   gate-projection weight
      down_wT  : [U, D]   down-projection weight

    Output:
      out      : [R, D]
    '''
    R, H = x_tensor.shape
    H2, U = up_wT.shape
    H3, U2 = gate_wT.shape
    U3, D = down_wT.shape
    assert H2 == H and H3 == H and U2 == U and U3 == U

    out = nl.ndarray((R, D), dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                  # 128
    TILE_U = nl.tile_size.gemm_moving_fmax      # 512
    TILE_D = nl.tile_size.gemm_moving_fmax      # 512
    G_BCAST_COLS = TILE_U                       # broadcast g in 512-column chunks

    assert R % TILE_M == 0
    assert H % TILE_K == 0
    assert U % TILE_U == 0
    assert D % TILE_D == 0

    # Load RMSNorm scale g once into SBUF as [1, H].
    # par_dim=1 is fine here because g is loaded via a 2D reshape and
    # only used as the moving operand in nc_matmul (is_stationary_onezero).
    g_tile = nl.ndarray((1, H), dtype=g_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_tile, src=g_tensor.reshape((1, H))[0:1, 0:H])

    # Scratch tiles reused across p-tiles
    xT_psum = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.psum)
    xT      = nl.ndarray((TILE_K, TILE_M), dtype=x_tensor.dtype, buffer=nl.sbuf)

    for p in nl.affine_range(R // TILE_M):
        p0 = p * TILE_M

        # --- Load x tile [TILE_M, H] ---
        x_tile = nl.ndarray((TILE_M, H), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_tensor[p0:p0+TILE_M, 0:H])

        # --- RMSNorm ---
        # 1. element-wise square
        sq = nl.ndarray((TILE_M, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x_tile, data2=x_tile, op=nl.multiply)
        # 2. sum across H → [TILE_M, 1]
        sq_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sq_sum, op=nl.add, data=sq, axis=1, keepdims=True)
        # 3. mean = sum / H → sqrt → reciprocal
        mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean, data=sq_sum, op0=nl.multiply, operand0=1.0/H)
        sqrt_m = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sqrt_m, op=nl.sqrt, data=mean)
        inv_rms = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_rms, data=sqrt_m)
        # 4. y = x * inv_rms
        y_tile = nl.ndarray((TILE_M, H), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=inv_rms)
        # 5. y = y * g  (broadcast g [1,H] to [TILE_M, H] via nc_matmul with ones)
        ones = nl.ndarray((1, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        for h_b in nl.affine_range(H // G_BCAST_COLS):
            h0 = h_b * G_BCAST_COLS
            g_chunk = nl.ndarray((1, G_BCAST_COLS), dtype=g_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_chunk, src=g_tile[0:1, h0:h0+G_BCAST_COLS])
            g_bcast = nl.ndarray((TILE_M, G_BCAST_COLS), dtype=x_tensor.dtype, buffer=nl.psum)
            nisa.nc_matmul(dst=g_bcast, stationary=ones, moving=g_chunk,
                           is_stationary_onezero=True)
            nisa.tensor_tensor(dst=y_tile[0:TILE_M, h0:h0+G_BCAST_COLS],
                               data1=y_tile[0:TILE_M, h0:h0+G_BCAST_COLS],
                               data2=g_bcast, op=nl.multiply)

        # --- Up/gate projections: y [TILE_M, H] @ wT [H, U] → [TILE_M, U] ---
        # Accumulate in SBUF (one TILE_U column-block at a time)
        for u in nl.affine_range(U // TILE_U):
            u0 = u * TILE_U
            acc_up   = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.sbuf)
            acc_gate = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc_up,   value=0.0)
            nisa.memset(dst=acc_gate, value=0.0)

            for k in nl.affine_range(H // TILE_K):
                k0 = k * TILE_K
                # Transpose y tile chunk [TILE_M, TILE_K] → xT [TILE_K, TILE_M]
                nisa.nc_transpose(dst=xT_psum, data=y_tile[0:TILE_M, k0:k0+TILE_K])
                nisa.tensor_copy(dst=xT, src=xT_psum)

                # Load weight tiles [TILE_K, TILE_U]
                up_w   = nl.ndarray((TILE_K, TILE_U), dtype=up_wT.dtype,   buffer=nl.sbuf)
                gate_w = nl.ndarray((TILE_K, TILE_U), dtype=gate_wT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=up_w,   src=up_wT  [k0:k0+TILE_K, u0:u0+TILE_U])
                nisa.dma_copy(dst=gate_w, src=gate_wT[k0:k0+TILE_K, u0:u0+TILE_U])

                # nc_matmul: xT[K,M].T @ W[K,U] → [M, U], par_dim=M=128 ✓
                psum_up   = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.psum)
                psum_gate = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum_up,   stationary=xT, moving=up_w)
                nisa.nc_matmul(dst=psum_gate, stationary=xT, moving=gate_w)

                tmp_up   = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.sbuf)
                tmp_gate = nl.ndarray((TILE_M, TILE_U), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tmp_up,   src=psum_up)
                nisa.tensor_copy(dst=tmp_gate, src=psum_gate)
                nisa.tensor_tensor(dst=acc_up,   data1=acc_up,   data2=tmp_up,   op=nl.add)
                nisa.tensor_tensor(dst=acc_gate, data1=acc_gate, data2=tmp_gate, op=nl.add)

            # Cast and apply SiLU → act [TILE_M, TILE_U]
            up_f   = nl.ndarray((TILE_M, TILE_U), dtype=x_tensor.dtype, buffer=nl.sbuf)
            gate_f = nl.ndarray((TILE_M, TILE_U), dtype=x_tensor.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=up_f,   src=acc_up,   engine=nisa.vector_engine)
            nisa.tensor_copy(dst=gate_f, src=acc_gate, engine=nisa.vector_engine)
            gate_silu = nl.ndarray((TILE_M, TILE_U), dtype=x_tensor.dtype, buffer=nl.sbuf)
            nisa.activation(dst=gate_silu, op=nl.silu, data=gate_f)
            act = nl.ndarray((TILE_M, TILE_U), dtype=x_tensor.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=act, data1=gate_silu, data2=up_f, op=nl.multiply)

            # --- Down projection: act [TILE_M, TILE_U] @ down_wT [TILE_U, D] ---
            for d in nl.affine_range(D // TILE_D):
                d0 = d * TILE_D
                acc_down = nl.ndarray((TILE_M, TILE_D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc_down, value=0.0)

                for k in nl.affine_range(TILE_U // TILE_K):
                    k_sub = k * TILE_K
                    # Transpose act sub-block [TILE_M, TILE_K] → actT [TILE_K, TILE_M]
                    actT_psum = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.psum)
                    actT      = nl.ndarray((TILE_K, TILE_M), dtype=x_tensor.dtype, buffer=nl.sbuf)
                    nisa.nc_transpose(dst=actT_psum, data=act[0:TILE_M, k_sub:k_sub+TILE_K])
                    nisa.tensor_copy(dst=actT, src=actT_psum)

                    down_w = nl.ndarray((TILE_K, TILE_D), dtype=down_wT.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=down_w,
                                  src=down_wT[u0+k_sub:u0+k_sub+TILE_K, d0:d0+TILE_D])

                    psum_d = nl.ndarray((TILE_M, TILE_D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=psum_d, stationary=actT, moving=down_w)
                    tmp_d = nl.ndarray((TILE_M, TILE_D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=tmp_d, src=psum_d)
                    nisa.tensor_tensor(dst=acc_down, data1=acc_down, data2=tmp_d, op=nl.add)

                out_f = nl.ndarray((TILE_M, TILE_D), dtype=x_tensor.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_f, src=acc_down, engine=nisa.vector_engine)
                nisa.dma_copy(dst=out[p0:p0+TILE_M, d0:d0+TILE_D], src=out_f)

    return out


def solution(x, post_attention_layernorm_weight, up_proj_weight, gate_proj_weight, down_proj_weight):
    b, s, h = x.shape
    x2d = x.view(-1, h)  # [R, H]
    output = nki_fused_all_proj_kernel(
        x2d,
        post_attention_layernorm_weight,
        up_proj_weight.t(),
        gate_proj_weight.t(),
        down_proj_weight.t(),
    )
    return output
"""

def PROMPT_2():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
@nki.jit
def solution(x_tensor, gamma, ug_wT, down_wT):
    '''
    Single-token (M=1) fused MLP: RMSNorm + up/gate + SiLU*up + down.
    Beta 2, K-first layout (par_dim = TILE_K = 128 throughout).
    Processing R=1, H=2048, U=8192, D=2048.

    Key idea: to avoid par_dim=1, keep the K-dimension (H=128-sized chunk) as
    the partition dim by transposing input/output relative to the standard GEMM
    layout.  nc_matmul(stationary=W[K,M], moving=x[K,N]) = W.T @ x → [M, N]
    with par_dim=M=K=128 ✓.

    Inputs:
      x_tensor : [1, H]       input activations (single token)
      gamma    : [H]          RMSNorm scale
      ug_wT    : [H, 2*U]     combined up+gate weight (up in [:U], gate in [U:])
      down_wT  : [U, D]       down-projection weight
    '''
    _, H    = x_tensor.shape
    _, ugU  = ug_wT.shape
    U = ugU // 2
    _, D    = down_wT.shape

    TILE_K = nl.tile_size.pmax  # 128

    assert H % TILE_K == 0
    assert U % TILE_K == 0
    assert D % TILE_K == 0

    # Output [D, 1] in transposed K-first layout; reshape to [1, D] outside
    out_T = nl.ndarray((D, 1), dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    # --- RMSNorm ---
    # Load x as [H, 1] (K-first) to avoid par_dim=1 in DMA copy
    x_T = nl.ndarray((H, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_T, src=x_tensor.reshape((H, 1))[0:H, 0:1])
    # Compute sum-of-squares: reduce across H
    sq = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x_T, data2=x_T, op=nl.multiply)
    sq_sum = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sq_sum, value=0.0)
    for k in nl.static_range(H // TILE_K):
        k0 = k * TILE_K
        tmp = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tmp, op=nl.add,
                           data=sq[k0:k0+TILE_K, 0:1], axis=1, keepdims=True)
        nisa.tensor_tensor(dst=sq_sum, data1=sq_sum, data2=tmp, op=nl.add)
    # mean, sqrt, reciprocal → scalar inv_rms stored as [TILE_K, 1] broadcast
    mean = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mean, data=sq_sum, op0=nl.multiply, operand0=1.0/H)
    sqrt_m = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sqrt_m, op=nl.sqrt, data=mean)
    inv_rms_scalar = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_rms_scalar, data=sqrt_m)
    # Scale x_T by inv_rms and gamma in TILE_K chunks → y_T [H, 1]
    y_T = nl.ndarray((H, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
    for k in nl.static_range(H // TILE_K):
        k0 = k * TILE_K
        g_chunk = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_chunk,
                      src=gamma.reshape((H, 1))[k0:k0+TILE_K, 0:1])
        scaled = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_T[k0:k0+TILE_K, 0:1],
                           op0=nl.multiply, operand0=inv_rms_scalar[0:TILE_K, 0:1])
        nisa.tensor_tensor(dst=y_T[k0:k0+TILE_K, 0:1],
                           data1=scaled, data2=g_chunk, op=nl.multiply)

    # --- Up/gate projections: y [H] @ ug_wT [H, 2U] → [2U] in K-first layout ---
    # For each chunk of U: acc_up/gate [TILE_K, 1], par_dim=128 ✓
    D_blocks = D // TILE_K
    acc_out = nl.ndarray((TILE_K, D_blocks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc_out, value=0.0)

    for n_i in nl.static_range(U // TILE_K):
        ni0 = n_i * TILE_K
        acc_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc_up,   value=0.0)
        nisa.memset(dst=acc_gate, value=0.0)

        for k in nl.static_range(H // TILE_K):
            k0 = k * TILE_K
            y_chunk = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=y_chunk, src=y_T[k0:k0+TILE_K, 0:1])

            up_tile   = nl.ndarray((TILE_K, TILE_K), dtype=ug_wT.dtype, buffer=nl.sbuf)
            gate_tile = nl.ndarray((TILE_K, TILE_K), dtype=ug_wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_tile,
                          src=ug_wT[k0:k0+TILE_K, ni0:ni0+TILE_K])
            nisa.dma_copy(dst=gate_tile,
                          src=ug_wT[k0:k0+TILE_K, U+ni0:U+ni0+TILE_K])

            psum_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            psum_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_up,   stationary=up_tile,   moving=y_chunk)
            nisa.nc_matmul(dst=psum_gate, stationary=gate_tile, moving=y_chunk)

            tmp_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            tmp_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp_up,   src=psum_up)
            nisa.tensor_copy(dst=tmp_gate, src=psum_gate)
            nisa.tensor_tensor(dst=acc_up,   data1=acc_up,   data2=tmp_up,   op=nl.add)
            nisa.tensor_tensor(dst=acc_gate, data1=acc_gate, data2=tmp_gate, op=nl.add)

        up_f   = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        gate_f = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=up_f,   src=acc_up,   engine=nisa.vector_engine)
        nisa.tensor_copy(dst=gate_f, src=acc_gate, engine=nisa.vector_engine)
        gate_silu = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.activation(dst=gate_silu, op=nl.silu, data=gate_f)
        act_chunk = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act_chunk, data1=gate_silu, data2=up_f, op=nl.multiply)

        for j in nl.static_range(D_blocks):
            no0 = j * TILE_K
            down_tile = nl.ndarray((TILE_K, TILE_K), dtype=down_wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=down_tile, src=down_wT[ni0:ni0+TILE_K, no0:no0+TILE_K])
            psum_d = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_d, stationary=down_tile, moving=act_chunk)
            tmp_d = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp_d, src=psum_d)
            nisa.tensor_tensor(dst=acc_out[0:TILE_K, j:j+1],
                               data1=acc_out[0:TILE_K, j:j+1],
                               data2=tmp_d, op=nl.add, engine=nisa.vector_engine)

    for j in nl.static_range(D_blocks):
        no0 = j * TILE_K
        out_chunk = nl.ndarray((TILE_K, 1), dtype=x_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_chunk, src=acc_out[0:TILE_K, j:j+1],
                         engine=nisa.vector_engine)
        nisa.dma_copy(dst=out_T[no0:no0+TILE_K, 0:1], src=out_chunk)

    return out_T
"""

def PROMPT_3():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
```
@nki.jit
def nki_fused_layer_(lhs_T, up_w, gate_w, down_w):
    '''
    Fused MLP for single-token inference: out = silu(lhs @ gate_w) * (lhs @ up_w) @ down_w
    Beta 2, K-first layout (par_dim = TILE_K = 128 everywhere).

    To avoid par_dim=1 (which causes DMA shape mismatches), inputs are passed
    in transposed layout so the contraction dimension K=128 is always the
    partition dimension.

    nc_matmul(stationary=W[K,M], moving=x[K,N]) = W.T @ x → result [M, N]
      with par_dim = M = K = 128 ✓

    lhs_T : [K_in, 1]      transposed input (lhs = lhs_T.T, shape [1, K_in])
    up_w  : [K_in, N_inter]
    gate_w: [K_in, N_inter]
    down_w: [N_inter, N_out]
    Returns: out_T [N_out, 1]  (reshape to [1, N_out] or [B, S, N_out] outside)
    '''
    K_in, _    = lhs_T.shape
    _, N_inter = up_w.shape
    _, N_out   = down_w.shape

    TILE_K = nl.tile_size.pmax  # 128

    assert K_in    % TILE_K == 0
    assert N_inter % TILE_K == 0
    assert N_out   % TILE_K == 0

    # Output [N_out, 1] in K-first transposed layout
    out_T = nl.ndarray((N_out, 1), dtype=lhs_T.dtype, buffer=nl.shared_hbm)

    # Output accumulator: [TILE_K, N_out // TILE_K]
    # Interpretation: acc_out[i, j] = output[j * TILE_K + i]
    N_out_blocks = N_out // TILE_K
    acc_out = nl.ndarray((TILE_K, N_out_blocks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc_out, value=0.0)

    for n_i in nl.static_range(N_inter // TILE_K):
        ni0 = n_i * TILE_K

        # Accumulate up/gate for this N_inter chunk across all K_in tiles
        acc_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc_up,   value=0.0)
        nisa.memset(dst=acc_gate, value=0.0)

        for k in nl.static_range(K_in // TILE_K):
            k0 = k * TILE_K

            # Load lhsT_tile [TILE_K, 1]  par_dim = TILE_K = 128 ✓
            lhsT_tile = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile, src=lhs_T[k0:k0+TILE_K, 0:1])

            # Load weight tiles [TILE_K, TILE_K]
            up_tile   = nl.ndarray((TILE_K, TILE_K), dtype=up_w.dtype,   buffer=nl.sbuf)
            gate_tile = nl.ndarray((TILE_K, TILE_K), dtype=gate_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_tile,   src=up_w  [k0:k0+TILE_K, ni0:ni0+TILE_K])
            nisa.dma_copy(dst=gate_tile, src=gate_w[k0:k0+TILE_K, ni0:ni0+TILE_K])

            # nc_matmul: W[K,M=128].T @ x[K,1] → [M=128, 1], par_dim=128 ✓
            psum_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            psum_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_up,   stationary=up_tile,   moving=lhsT_tile)
            nisa.nc_matmul(dst=psum_gate, stationary=gate_tile, moving=lhsT_tile)

            tmp_up   = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            tmp_gate = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp_up,   src=psum_up)
            nisa.tensor_copy(dst=tmp_gate, src=psum_gate)
            nisa.tensor_tensor(dst=acc_up,   data1=acc_up,   data2=tmp_up,   op=nl.add)
            nisa.tensor_tensor(dst=acc_gate, data1=acc_gate, data2=tmp_gate, op=nl.add)

        # Cast to input dtype; apply SiLU; multiply → act_chunk [TILE_K, 1]
        up_f   = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
        gate_f = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=up_f,   src=acc_up,   engine=nisa.vector_engine)
        nisa.tensor_copy(dst=gate_f, src=acc_gate, engine=nisa.vector_engine)
        gate_silu = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
        nisa.activation(dst=gate_silu, op=nl.silu, data=gate_f)
        act_chunk = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act_chunk, data1=gate_silu, data2=up_f, op=nl.multiply)

        # Down projection for each N_out block
        for j in nl.static_range(N_out_blocks):
            no0 = j * TILE_K
            down_tile = nl.ndarray((TILE_K, TILE_K), dtype=down_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=down_tile, src=down_w[ni0:ni0+TILE_K, no0:no0+TILE_K])

            psum_down = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_down, stationary=down_tile, moving=act_chunk)
            tmp_down = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp_down, src=psum_down)
            nisa.tensor_tensor(
                dst=acc_out[0:TILE_K, j:j+1],
                data1=acc_out[0:TILE_K, j:j+1],
                data2=tmp_down, op=nl.add, engine=nisa.vector_engine)

    # Store: each column j of acc_out → out_T[j*TILE_K:(j+1)*TILE_K, 0:1]
    for j in nl.static_range(N_out_blocks):
        no0 = j * TILE_K
        out_chunk = nl.ndarray((TILE_K, 1), dtype=lhs_T.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_chunk, src=acc_out[0:TILE_K, j:j+1],
                         engine=nisa.vector_engine)
        nisa.dma_copy(dst=out_T[no0:no0+TILE_K, 0:1], src=out_chunk)

    return out_T


def solution(x, up_proj_weight, gate_proj_weight, down_proj_weight):
    # x: [B, S, K_in] with B=1, S=1 (single token inference)
    b, s, k_in = x.shape
    n_out = down_proj_weight.shape[-1]
    # Transpose to K-first layout [K_in, 1] for kernel
    lhs_T = x.reshape(k_in, 1)
    out_T = nki_fused_layer_(lhs_T, up_proj_weight, gate_proj_weight, down_proj_weight)
    return out_T.reshape(b, s, n_out)
```
"""
