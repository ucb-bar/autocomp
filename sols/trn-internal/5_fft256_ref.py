@nki.jit
def test(
    X_real_hbm,
    X_imag_hbm,
    W_128_real_hbm,
    W_128_imag_hbm,
    twiddle_real_hbm,
    twiddle_imag_hbm,
):
    """256-point radix-2 Cooley-Tukey FFT using NKI Tensor Engine.

    Single-kernel implementation: deinterleave -> 128-pt DFT matmul -> twiddle -> butterfly.

    Uses nc_n_gather for even/odd split, nc_matmul for DFT base case with PSUM
    accumulation for complex arithmetic, and element-wise ops for twiddle + butterfly.

    Args:
        X_real_hbm: Input real part [128, 256] float32, pre-padded to 128 rows.
        X_imag_hbm: Input imag part [128, 256] float32.
        W_128_real_hbm: 128-pt DFT matrix real part [128, 128] float32.
        W_128_imag_hbm: 128-pt DFT matrix imag part [128, 128] float32.
        twiddle_real_hbm: Twiddle factors real part [128, 128] float32.
        twiddle_imag_hbm: Twiddle factors imag part [128, 128] float32.

    Returns:
        (Y_real_hbm, Y_imag_hbm): Output [128, 256] float32.
    """
    TILE_H = 128
    N = 256
    N_half = 128

    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load input
    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=X_real, src=X_real_hbm[0:TILE_H, 0:N])
    nisa.dma_copy(dst=X_imag, src=X_imag_hbm[0:TILE_H, 0:N])

    # Load DFT matrix and twiddle factors
    W_128_real = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_imag = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=W_128_real, src=W_128_real_hbm[0:N_half, 0:N_half])
    nisa.dma_copy(dst=W_128_imag, src=W_128_imag_hbm[0:N_half, 0:N_half])

    twiddle_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=twiddle_real, src=twiddle_real_hbm[0:TILE_H, 0:N_half])
    nisa.dma_copy(dst=twiddle_imag, src=twiddle_imag_hbm[0:TILE_H, 0:N_half])

    # Even/odd column deinterleave via nc_n_gather
    even_idx = nl.ndarray((TILE_H, N_half), dtype=nl.uint32, buffer=nl.sbuf)
    odd_idx = nl.ndarray((TILE_H, N_half), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=even_idx, pattern=[[2, N_half]], offset=0, channel_multiplier=0)
    nisa.iota(dst=odd_idx, pattern=[[2, N_half]], offset=1, channel_multiplier=0)

    X_even_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_even_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=X_even_real, data=X_real, indices=even_idx)
    nisa.nc_n_gather(dst=X_even_imag, data=X_imag, indices=even_idx)
    nisa.nc_n_gather(dst=X_odd_real, data=X_real, indices=odd_idx)
    nisa.nc_n_gather(dst=X_odd_imag, data=X_imag, indices=odd_idx)

    # --- 128-pt DFT on even half (inlined _fft1d_matmul_isa) ---
    # Transpose DFT matrix for Tensor Engine
    W_real_T_psum = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=W_real_T_psum, data=W_128_real)
    W_real_T = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=W_real_T, src=W_real_T_psum)

    W_imag_T_psum = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=W_imag_T_psum, data=W_128_imag)
    W_imag_T = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=W_imag_T, src=W_imag_T_psum)

    neg_W_imag_T = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_W_imag_T, data=W_imag_T, op0=nl.multiply, operand0=-1.0)

    # Even half: transpose input
    Xe_real_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xe_real_T_psum, data=X_even_real)
    Xe_real_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xe_real_T, src=Xe_real_T_psum)

    Xe_imag_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xe_imag_T_psum, data=X_even_imag)
    Xe_imag_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xe_imag_T, src=Xe_imag_T_psum)

    # Even half: Y_real = W_real @ X_real - W_imag @ X_imag (via PSUM accumulation)
    X_even_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_e_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_e_real, stationary=Xe_real_T, moving=W_real_T)
    nisa.nc_matmul(dst=psum_e_real, stationary=Xe_imag_T, moving=neg_W_imag_T)
    nisa.tensor_copy(dst=X_even_fft_real, src=psum_e_real)

    # Even half: Y_imag = W_imag @ X_real + W_real @ X_imag
    X_even_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_e_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_e_imag, stationary=Xe_real_T, moving=W_imag_T)
    nisa.nc_matmul(dst=psum_e_imag, stationary=Xe_imag_T, moving=W_real_T)
    nisa.tensor_copy(dst=X_even_fft_imag, src=psum_e_imag)

    # --- 128-pt DFT on odd half (inlined _fft1d_matmul_isa) ---
    Xo_real_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xo_real_T_psum, data=X_odd_real)
    Xo_real_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xo_real_T, src=Xo_real_T_psum)

    Xo_imag_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xo_imag_T_psum, data=X_odd_imag)
    Xo_imag_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xo_imag_T, src=Xo_imag_T_psum)

    X_odd_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_o_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_o_real, stationary=Xo_real_T, moving=W_real_T)
    nisa.nc_matmul(dst=psum_o_real, stationary=Xo_imag_T, moving=neg_W_imag_T)
    nisa.tensor_copy(dst=X_odd_fft_real, src=psum_o_real)

    X_odd_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_o_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_o_imag, stationary=Xo_real_T, moving=W_imag_T)
    nisa.nc_matmul(dst=psum_o_imag, stationary=Xo_imag_T, moving=W_real_T)
    nisa.tensor_copy(dst=X_odd_fft_imag, src=psum_o_imag)

    # --- Twiddle factor application ---
    X_odd_tw_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_tw_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)

    ac = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ac, data1=X_odd_fft_real, data2=twiddle_real, op=nl.multiply)
    bd = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=bd, data1=X_odd_fft_imag, data2=twiddle_imag, op=nl.multiply)
    nisa.tensor_tensor(dst=X_odd_tw_real, data1=ac, data2=bd, op=nl.subtract)

    ad = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ad, data1=X_odd_fft_real, data2=twiddle_imag, op=nl.multiply)
    bc = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=bc, data1=X_odd_fft_imag, data2=twiddle_real, op=nl.multiply)
    nisa.tensor_tensor(dst=X_odd_tw_imag, data1=ad, data2=bc, op=nl.add)

    # --- Butterfly combination ---
    Y_first_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    Y_first_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    Y_second_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    Y_second_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_tensor(
        dst=Y_first_real, data1=X_even_fft_real, data2=X_odd_tw_real, op=nl.add
    )
    nisa.tensor_tensor(
        dst=Y_first_imag, data1=X_even_fft_imag, data2=X_odd_tw_imag, op=nl.add
    )
    nisa.tensor_tensor(
        dst=Y_second_real, data1=X_even_fft_real, data2=X_odd_tw_real, op=nl.subtract
    )
    nisa.tensor_tensor(
        dst=Y_second_imag, data1=X_even_fft_imag, data2=X_odd_tw_imag, op=nl.subtract
    )

    # Assemble output
    Y_combined_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_combined_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_copy(dst=Y_combined_real[0:TILE_H, 0:N_half], src=Y_first_real)
    nisa.tensor_copy(dst=Y_combined_imag[0:TILE_H, 0:N_half], src=Y_first_imag)
    nisa.tensor_copy(dst=Y_combined_real[0:TILE_H, N_half:N], src=Y_second_real)
    nisa.tensor_copy(dst=Y_combined_imag[0:TILE_H, N_half:N], src=Y_second_imag)

    nisa.dma_copy(dst=Y_real_hbm[0:TILE_H, 0:N], src=Y_combined_real)
    nisa.dma_copy(dst=Y_imag_hbm[0:TILE_H, 0:N], src=Y_combined_imag)

    return Y_real_hbm, Y_imag_hbm
