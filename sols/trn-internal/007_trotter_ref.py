@nki.jit
def test(
    psi_real_in,
    psi_imag_in,
    zz_phase_real,
    zz_phase_imag,
    cos_val_in,
    sin_val_in,
    n_steps_dummy,
    rx_mat_rr_T_in,
    rx_mat_ri_T_in,
    rx_mat_ir_T_in,
    rx_mat_ii_T_in,
    free_xor_indices_in,
):
    """Multi-step Trotter-Suzuki Hamiltonian evolution on Trainium2.

    Runs n_steps symmetric Trotter substeps entirely on-chip, keeping the
    quantum state vector in SBUF across all steps via nl.sequential_range.
    Each substep: R_x(half, all qubits) -> ZZ(full) -> R_x(half, all qubits).

    Free-dim R_x uses nc_n_gather (GpSimd) for XOR-partner permutation.
    Cross-partition R_x uses nc_matmul (Tensor Engine) with fused 128x128
    block matrix for all cross-partition qubits.

    For 3x3 lattice: 9 qubits, dim=512, F=4, 7 cross-partition + 2 free-dim.
    For 4x4 lattice: 16 qubits, dim=65536, F=512, 7 cross-partition + 9 free-dim.

    Args:
        psi_real_in: (128, F) float32 — state vector real part.
        psi_imag_in: (128, F) float32 — state vector imaginary part.
        zz_phase_real: (128, F) float32 — combined ZZ diagonal phase (real).
        zz_phase_imag: (128, F) float32 — combined ZZ diagonal phase (imag).
        cos_val_in: (128, 1) float32 — cos(h*dt/2) broadcast scalar.
        sin_val_in: (128, 1) float32 — sin(h*dt/2) broadcast scalar.
        n_steps_dummy: (n_steps,) float32 — dummy tensor encoding step count.
        rx_mat_rr_T_in: (128, 128) float32 — combined cross-partition M_rr^T.
        rx_mat_ri_T_in: (128, 128) float32 — combined cross-partition M_ri^T.
        rx_mat_ir_T_in: (128, 128) float32 — combined cross-partition M_ir^T.
        rx_mat_ii_T_in: (128, 128) float32 — combined cross-partition M_ii^T.
        free_xor_indices_in: (N_FREE, 128, F) uint32 — XOR permutation tables.

    Returns:
        (out_real, out_imag): both (128, F) float32 — evolved state.
    """
    F_K = psi_real_in.shape[1]
    N_STEPS = n_steps_dummy.shape[0]

    out_real = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.shared_hbm)
    out_imag = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.shared_hbm)

    pr = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    pi = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pr[0:128, 0:F_K], src=psi_real_in[0:128, 0:F_K])
    nisa.dma_copy(dst=pi[0:128, 0:F_K], src=psi_imag_in[0:128, 0:F_K])

    zz_r = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    zz_i = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=zz_r[0:128, 0:F_K], src=zz_phase_real[0:128, 0:F_K])
    nisa.dma_copy(dst=zz_i[0:128, 0:F_K], src=zz_phase_imag[0:128, 0:F_K])

    cos_r = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sin_r = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_r[0:128, 0:1], src=cos_val_in[0:128, 0:1])
    nisa.dma_copy(dst=sin_r[0:128, 0:1], src=sin_val_in[0:128, 0:1])

    tmp_r = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    tmp_i = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    scratch = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)

    rx_partner_r = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    rx_partner_i = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    rx_tmp1 = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)
    rx_tmp2 = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.sbuf)

    M_rr_T = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    M_ri_T = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    M_ir_T = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    M_ii_T = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=M_rr_T[0:128, 0:128], src=rx_mat_rr_T_in[0:128, 0:128])
    nisa.dma_copy(dst=M_ri_T[0:128, 0:128], src=rx_mat_ri_T_in[0:128, 0:128])
    nisa.dma_copy(dst=M_ir_T[0:128, 0:128], src=rx_mat_ir_T_in[0:128, 0:128])
    nisa.dma_copy(dst=M_ii_T[0:128, 0:128], src=rx_mat_ii_T_in[0:128, 0:128])

    N_FREE_K = free_xor_indices_in.shape[0]
    free_xor_idx_list = []
    for fi in range(N_FREE_K):
        idx_sbuf = nl.ndarray((128, F_K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_sbuf[0:128, 0:F_K],
            src=free_xor_indices_in[fi, 0:128, 0:F_K],
        )
        free_xor_idx_list.append(idx_sbuf)

    for step in nl.sequential_range(N_STEPS):
        # R_x half-step: free-dim qubits (nc_n_gather)
        for fi in range(len(free_xor_idx_list)):
            xor_idx = free_xor_idx_list[fi]
            nisa.nc_n_gather(
                dst=rx_partner_r[0:128, 0:F_K],
                data=pr[0:128, 0:F_K],
                indices=xor_idx[0:128, 0:F_K],
            )
            nisa.nc_n_gather(
                dst=rx_partner_i[0:128, 0:F_K],
                data=pi[0:128, 0:F_K],
                indices=xor_idx[0:128, 0:F_K],
            )
            nisa.tensor_scalar(
                dst=rx_tmp1[0:128, 0:F_K],
                data=pr[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=cos_r,
            )
            nisa.tensor_scalar(
                dst=rx_tmp2[0:128, 0:F_K],
                data=rx_partner_i[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=sin_r,
            )
            nisa.tensor_tensor(
                dst=rx_tmp1[0:128, 0:F_K],
                data1=rx_tmp1[0:128, 0:F_K],
                data2=rx_tmp2[0:128, 0:F_K],
                op=nl.subtract,
            )
            nisa.tensor_scalar(
                dst=rx_tmp2[0:128, 0:F_K],
                data=pi[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=cos_r,
            )
            nisa.tensor_scalar(
                dst=rx_partner_r[0:128, 0:F_K],
                data=rx_partner_r[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=sin_r,
            )
            nisa.tensor_tensor(
                dst=rx_tmp2[0:128, 0:F_K],
                data1=rx_tmp2[0:128, 0:F_K],
                data2=rx_partner_r[0:128, 0:F_K],
                op=nl.add,
            )
            nisa.tensor_copy(dst=pr[0:128, 0:F_K], src=rx_tmp1[0:128, 0:F_K])
            nisa.tensor_copy(dst=pi[0:128, 0:F_K], src=rx_tmp2[0:128, 0:F_K])

        # R_x half-step: cross-partition qubits (Tensor Engine matmul)
        psum_r = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(psum_r, stationary=M_rr_T[0:128, 0:128], moving=pr[0:128, 0:F_K])
        nisa.nc_matmul(psum_r, stationary=M_ri_T[0:128, 0:128], moving=pi[0:128, 0:F_K])
        nisa.tensor_copy(dst=rx_tmp1[0:128, 0:F_K], src=psum_r[0:128, 0:F_K])
        psum_i = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(psum_i, stationary=M_ir_T[0:128, 0:128], moving=pr[0:128, 0:F_K])
        nisa.nc_matmul(psum_i, stationary=M_ii_T[0:128, 0:128], moving=pi[0:128, 0:F_K])
        nisa.tensor_copy(dst=rx_tmp2[0:128, 0:F_K], src=psum_i[0:128, 0:F_K])
        nisa.tensor_copy(dst=pr[0:128, 0:F_K], src=rx_tmp1[0:128, 0:F_K])
        nisa.tensor_copy(dst=pi[0:128, 0:F_K], src=rx_tmp2[0:128, 0:F_K])

        # ZZ full step
        nisa.tensor_tensor(
            dst=tmp_r[0:128, 0:F_K],
            data1=pr[0:128, 0:F_K],
            data2=zz_r[0:128, 0:F_K],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=scratch[0:128, 0:F_K],
            data1=pi[0:128, 0:F_K],
            data2=zz_i[0:128, 0:F_K],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=tmp_r[0:128, 0:F_K],
            data1=tmp_r[0:128, 0:F_K],
            data2=scratch[0:128, 0:F_K],
            op=nl.subtract,
        )
        nisa.tensor_tensor(
            dst=tmp_i[0:128, 0:F_K],
            data1=pi[0:128, 0:F_K],
            data2=zz_r[0:128, 0:F_K],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=scratch[0:128, 0:F_K],
            data1=pr[0:128, 0:F_K],
            data2=zz_i[0:128, 0:F_K],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=tmp_i[0:128, 0:F_K],
            data1=tmp_i[0:128, 0:F_K],
            data2=scratch[0:128, 0:F_K],
            op=nl.add,
        )
        nisa.tensor_copy(dst=pr[0:128, 0:F_K], src=tmp_r[0:128, 0:F_K])
        nisa.tensor_copy(dst=pi[0:128, 0:F_K], src=tmp_i[0:128, 0:F_K])

        # R_x half-step: free-dim qubits (nc_n_gather)
        for fi in range(len(free_xor_idx_list)):
            xor_idx = free_xor_idx_list[fi]
            nisa.nc_n_gather(
                dst=rx_partner_r[0:128, 0:F_K],
                data=pr[0:128, 0:F_K],
                indices=xor_idx[0:128, 0:F_K],
            )
            nisa.nc_n_gather(
                dst=rx_partner_i[0:128, 0:F_K],
                data=pi[0:128, 0:F_K],
                indices=xor_idx[0:128, 0:F_K],
            )
            nisa.tensor_scalar(
                dst=rx_tmp1[0:128, 0:F_K],
                data=pr[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=cos_r,
            )
            nisa.tensor_scalar(
                dst=rx_tmp2[0:128, 0:F_K],
                data=rx_partner_i[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=sin_r,
            )
            nisa.tensor_tensor(
                dst=rx_tmp1[0:128, 0:F_K],
                data1=rx_tmp1[0:128, 0:F_K],
                data2=rx_tmp2[0:128, 0:F_K],
                op=nl.subtract,
            )
            nisa.tensor_scalar(
                dst=rx_tmp2[0:128, 0:F_K],
                data=pi[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=cos_r,
            )
            nisa.tensor_scalar(
                dst=rx_partner_r[0:128, 0:F_K],
                data=rx_partner_r[0:128, 0:F_K],
                op0=nl.multiply,
                operand0=sin_r,
            )
            nisa.tensor_tensor(
                dst=rx_tmp2[0:128, 0:F_K],
                data1=rx_tmp2[0:128, 0:F_K],
                data2=rx_partner_r[0:128, 0:F_K],
                op=nl.add,
            )
            nisa.tensor_copy(dst=pr[0:128, 0:F_K], src=rx_tmp1[0:128, 0:F_K])
            nisa.tensor_copy(dst=pi[0:128, 0:F_K], src=rx_tmp2[0:128, 0:F_K])

        # R_x half-step: cross-partition qubits (Tensor Engine matmul)
        psum_r2 = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            psum_r2, stationary=M_rr_T[0:128, 0:128], moving=pr[0:128, 0:F_K]
        )
        nisa.nc_matmul(
            psum_r2, stationary=M_ri_T[0:128, 0:128], moving=pi[0:128, 0:F_K]
        )
        nisa.tensor_copy(dst=rx_tmp1[0:128, 0:F_K], src=psum_r2[0:128, 0:F_K])
        psum_i2 = nl.ndarray((128, F_K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            psum_i2, stationary=M_ir_T[0:128, 0:128], moving=pr[0:128, 0:F_K]
        )
        nisa.nc_matmul(
            psum_i2, stationary=M_ii_T[0:128, 0:128], moving=pi[0:128, 0:F_K]
        )
        nisa.tensor_copy(dst=rx_tmp2[0:128, 0:F_K], src=psum_i2[0:128, 0:F_K])
        nisa.tensor_copy(dst=pr[0:128, 0:F_K], src=rx_tmp1[0:128, 0:F_K])
        nisa.tensor_copy(dst=pi[0:128, 0:F_K], src=rx_tmp2[0:128, 0:F_K])

    nisa.dma_copy(dst=out_real[0:128, 0:F_K], src=pr[0:128, 0:F_K])
    nisa.dma_copy(dst=out_imag[0:128, 0:F_K], src=pi[0:128, 0:F_K])

    return out_real, out_imag
