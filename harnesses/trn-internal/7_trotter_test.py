import math
import numpy as np

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# ==============================================================================
# Self-contained host-side helpers for 2D TFIM Trotter evolution
# ==============================================================================


def get_2d_lattice_bonds(L):
    """Nearest-neighbor bonds for LxL lattice (open BC)."""
    bonds = []
    for row in range(L):
        for col in range(L):
            i = row * L + col
            if col + 1 < L:
                bonds.append((i, row * L + (col + 1)))
            if row + 1 < L:
                bonds.append((i, (row + 1) * L + col))
    return bonds


def precompute_zz_phases(L, J, dt):
    """Combined ZZ diagonal phase factors for all bonds."""
    N = L * L
    dim = 1 << N
    bonds = get_2d_lattice_bonds(L)
    indices = np.arange(dim)
    total_angle = np.zeros(dim, dtype=np.float64)
    for q0, q1 in bonds:
        bit_q0 = (indices >> (N - 1 - q0)) & 1
        bit_q1 = (indices >> (N - 1 - q1)) & 1
        z0 = 1 - 2 * bit_q0
        z1 = 1 - 2 * bit_q1
        total_angle += J * dt * z0 * z1
    return np.cos(total_angle).astype(np.float32), np.sin(total_angle).astype(
        np.float32
    )


def classify_qubits(N, F):
    """Split qubits into cross-partition and free-dim lists."""
    cross_partition = []
    free_dim = []
    for q in range(N):
        stride = 1 << (N - 1 - q)
        if stride >= F:
            cross_partition.append((q, stride // F))
        else:
            free_dim.append((q, stride))
    return cross_partition, free_dim


def build_combined_rx_matrix(cross_partition_qubits, rx_cos, rx_sin):
    """Build fused 128x128 block matrix for cross-partition R_x gates."""
    c, s = rx_cos, rx_sin
    M_rr = np.eye(128, dtype=np.float64)
    M_ri = np.zeros((128, 128), dtype=np.float64)
    M_ir = np.zeros((128, 128), dtype=np.float64)
    M_ii = np.eye(128, dtype=np.float64)
    for q, p_stride in cross_partition_qubits:
        P = np.zeros((128, 128), dtype=np.float64)
        for p in range(128):
            P[p, p ^ p_stride] = 1.0
        G_rr = c * np.eye(128, dtype=np.float64)
        G_ri = -s * P
        G_ir = s * P
        G_ii = c * np.eye(128, dtype=np.float64)
        new_rr = G_rr @ M_rr + G_ri @ M_ir
        new_ri = G_rr @ M_ri + G_ri @ M_ii
        new_ir = G_ir @ M_rr + G_ii @ M_ir
        new_ii = G_ir @ M_ri + G_ii @ M_ii
        M_rr, M_ri, M_ir, M_ii = new_rr, new_ri, new_ir, new_ii
    return (
        M_rr.T.astype(np.float32),
        M_ri.T.astype(np.float32),
        M_ir.T.astype(np.float32),
        M_ii.T.astype(np.float32),
    )


def build_free_xor_index_tables(free_dim_qubits, F):
    """Build XOR permutation index tables for nc_n_gather."""
    N_FREE = len(free_dim_qubits)
    if N_FREE == 0:
        return np.zeros((0, 128, F), dtype=np.uint32)
    table = np.zeros((N_FREE, 128, F), dtype=np.uint32)
    for fi, (q, stride) in enumerate(free_dim_qubits):
        row = np.arange(F, dtype=np.uint32) ^ np.uint32(stride)
        table[fi, :, :] = row[np.newaxis, :]
    return table


def full_trotter_step_cpu(psi_real, psi_imag, zz_r, zz_i, rx_cos, rx_sin, N, F):
    """One symmetric Trotter step on CPU matching NKI kernel logic."""

    def apply_rx_one_qubit(pr, pi, c, s, q):
        stride = 1 << (N - 1 - q)
        if stride < F:
            block_size = 2 * stride
            n_blocks = F // block_size
            for blk in range(n_blocks):
                f0 = blk * block_size
                f1 = f0 + stride
                r0 = pr[:, f0 : f0 + stride].copy()
                i0 = pi[:, f0 : f0 + stride].copy()
                r1 = pr[:, f1 : f1 + stride].copy()
                i1 = pi[:, f1 : f1 + stride].copy()
                pr[:, f0 : f0 + stride] = c * r0 - s * i1
                pi[:, f0 : f0 + stride] = c * i0 + s * r1
                pr[:, f1 : f1 + stride] = c * r1 - s * i0
                pi[:, f1 : f1 + stride] = c * i1 + s * r0
        else:
            p_stride = stride // F
            group_size = 2 * p_stride
            n_groups = 128 // group_size
            for g in range(n_groups):
                p0 = g * group_size
                p1 = p0 + p_stride
                r0 = pr[p0 : p0 + p_stride, :].copy()
                i0 = pi[p0 : p0 + p_stride, :].copy()
                r1 = pr[p1 : p1 + p_stride, :].copy()
                i1 = pi[p1 : p1 + p_stride, :].copy()
                pr[p0 : p0 + p_stride, :] = c * r0 - s * i1
                pi[p0 : p0 + p_stride, :] = c * i0 + s * r1
                pr[p1 : p1 + p_stride, :] = c * r1 - s * i0
                pi[p1 : p1 + p_stride, :] = c * i1 + s * r0
        return pr, pi

    for q in range(N):
        psi_real, psi_imag = apply_rx_one_qubit(psi_real, psi_imag, rx_cos, rx_sin, q)
    new_r = psi_real * zz_r - psi_imag * zz_i
    new_i = psi_imag * zz_r + psi_real * zz_i
    psi_real, psi_imag = new_r, new_i
    for q in range(N):
        psi_real, psi_imag = apply_rx_one_qubit(psi_real, psi_imag, rx_cos, rx_sin, q)
    return psi_real, psi_imag


def prepare_trotter_inputs(L=3, J=1.0, h=1.0, dt=0.01, n_steps=10):
    """Prepare all inputs for the Trotter NKI kernel."""
    N = L * L
    dim = 1 << N
    F = dim // 128

    zz_r_flat, zz_i_flat = precompute_zz_phases(L, J, dt)
    zz_r = zz_r_flat.reshape(128, F)
    zz_i = zz_i_flat.reshape(128, F)

    rx_cos = math.cos(h * dt / 2)
    rx_sin = math.sin(h * dt / 2)
    cos_broadcast = np.full((128, 1), rx_cos, dtype=np.float32)
    sin_broadcast = np.full((128, 1), rx_sin, dtype=np.float32)

    cross_qubits, free_qubits = classify_qubits(N, F)
    M_rr_T, M_ri_T, M_ir_T, M_ii_T = build_combined_rx_matrix(
        cross_qubits, rx_cos, rx_sin
    )
    free_xor_table = build_free_xor_index_tables(free_qubits, F)

    # Initial state: |00...0>
    psi_real = np.zeros((128, F), dtype=np.float32)
    psi_imag = np.zeros((128, F), dtype=np.float32)
    psi_real[0, 0] = 1.0

    return {
        "psi_real": psi_real,
        "psi_imag": psi_imag,
        "zz_r": zz_r,
        "zz_i": zz_i,
        "cos_broadcast": cos_broadcast,
        "sin_broadcast": sin_broadcast,
        "n_steps": n_steps,
        "M_rr_T": M_rr_T,
        "M_ri_T": M_ri_T,
        "M_ir_T": M_ir_T,
        "M_ii_T": M_ii_T,
        "free_xor_table": free_xor_table,
        "rx_cos": rx_cos,
        "rx_sin": rx_sin,
        "N": N,
        "F": F,
    }


# SUBSTITUTE HERE


@nki.jit
def ref(
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
    """Reference: identical to test (multi-step Trotter kernel)."""
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


def test_nki(ref_func, test_func):
    """Correctness check: 3x3 lattice, 10 Trotter steps, compare to CPU reference."""
    device = xm.xla_device()
    L, n_steps = 3, 10

    for seed in range(2):
        np.random.seed(42 + seed)
        dt = 0.01 + seed * 0.005  # slightly different dt per seed

        inputs = prepare_trotter_inputs(L=L, J=1.0, h=1.0, dt=dt, n_steps=n_steps)
        N, F = inputs["N"], inputs["F"]

        # CPU reference evolution
        cpu_r = inputs["psi_real"].copy()
        cpu_i = inputs["psi_imag"].copy()
        for _ in range(n_steps):
            cpu_r, cpu_i = full_trotter_step_cpu(
                cpu_r,
                cpu_i,
                inputs["zz_r"],
                inputs["zz_i"],
                inputs["rx_cos"],
                inputs["rx_sin"],
                N,
                F,
            )

        # Transfer to device
        psi_r = torch.tensor(inputs["psi_real"], dtype=torch.float32, device=device)
        psi_i = torch.tensor(inputs["psi_imag"], dtype=torch.float32, device=device)
        zz_r_d = torch.tensor(inputs["zz_r"], dtype=torch.float32, device=device)
        zz_i_d = torch.tensor(inputs["zz_i"], dtype=torch.float32, device=device)
        cos_d = torch.tensor(
            inputs["cos_broadcast"], dtype=torch.float32, device=device
        )
        sin_d = torch.tensor(
            inputs["sin_broadcast"], dtype=torch.float32, device=device
        )
        steps_d = torch.zeros(n_steps, dtype=torch.float32, device=device)
        rr_d = torch.tensor(inputs["M_rr_T"], dtype=torch.float32, device=device)
        ri_d = torch.tensor(inputs["M_ri_T"], dtype=torch.float32, device=device)
        ir_d = torch.tensor(inputs["M_ir_T"], dtype=torch.float32, device=device)
        ii_d = torch.tensor(inputs["M_ii_T"], dtype=torch.float32, device=device)
        xor_d = torch.tensor(inputs["free_xor_table"], dtype=torch.int32, device=device)

        args = (
            psi_r,
            psi_i,
            zz_r_d,
            zz_i_d,
            cos_d,
            sin_d,
            steps_d,
            rr_d,
            ri_d,
            ir_d,
            ii_d,
            xor_d,
        )

        # Run ref
        out_r_ref, out_i_ref = ref_func(*args)
        ref_r_np = out_r_ref.cpu().numpy().flatten()
        ref_i_np = out_i_ref.cpu().numpy().flatten()

        # Run test
        out_r_test, out_i_test = test_func(*args)
        test_r_np = out_r_test.cpu().numpy().flatten()
        test_i_np = out_i_test.cpu().numpy().flatten()

        # Compare NKI ref vs test (NKI-to-NKI match)
        ref_vec = np.concatenate([ref_r_np, ref_i_np])
        test_vec = np.concatenate([test_r_np, test_i_np])
        cos_sim = np.dot(ref_vec, test_vec) / (
            np.linalg.norm(ref_vec) * np.linalg.norm(test_vec) + 1e-12
        )
        if cos_sim < 0.999:
            print(f"  FAIL: ref vs test cosine={cos_sim:.6f} (seed={seed})")
            return False

        # Compare NKI vs CPU (fidelity)
        nki_flat = ref_r_np + 1j * ref_i_np
        cpu_flat = cpu_r.flatten() + 1j * cpu_i.flatten()
        nki_n = nki_flat / (np.linalg.norm(nki_flat) + 1e-12)
        cpu_n = cpu_flat / (np.linalg.norm(cpu_flat) + 1e-12)
        fidelity = np.abs(np.conj(cpu_n) @ nki_n) ** 2
        if fidelity < 0.999:
            print(f"  FAIL: NKI vs CPU fidelity={fidelity:.6f} (seed={seed})")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    inputs = prepare_trotter_inputs(L=3, J=1.0, h=1.0, dt=0.01, n_steps=10)

    psi_r = torch.tensor(inputs["psi_real"], dtype=torch.float32, device=device)
    psi_i = torch.tensor(inputs["psi_imag"], dtype=torch.float32, device=device)
    zz_r_d = torch.tensor(inputs["zz_r"], dtype=torch.float32, device=device)
    zz_i_d = torch.tensor(inputs["zz_i"], dtype=torch.float32, device=device)
    cos_d = torch.tensor(inputs["cos_broadcast"], dtype=torch.float32, device=device)
    sin_d = torch.tensor(inputs["sin_broadcast"], dtype=torch.float32, device=device)
    steps_d = torch.zeros(10, dtype=torch.float32, device=device)
    rr_d = torch.tensor(inputs["M_rr_T"], dtype=torch.float32, device=device)
    ri_d = torch.tensor(inputs["M_ri_T"], dtype=torch.float32, device=device)
    ir_d = torch.tensor(inputs["M_ir_T"], dtype=torch.float32, device=device)
    ii_d = torch.tensor(inputs["M_ii_T"], dtype=torch.float32, device=device)
    xor_d = torch.tensor(inputs["free_xor_table"], dtype=torch.int32, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(
        psi_r,
        psi_i,
        zz_r_d,
        zz_i_d,
        cos_d,
        sin_d,
        steps_d,
        rr_d,
        ri_d,
        ir_d,
        ii_d,
        xor_d,
    )
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))


if __name__ == "__main__":
    test_result = test_nki(ref, solution)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(solution)
