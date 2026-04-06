import numpy as np
import math
import os

os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none --enable-mixed-precision-accumulation"

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


def _compute_dft_matrix(N):
    """Compute N-point DFT matrix split into real/imag."""
    k = np.arange(N, dtype=np.float32).reshape(N, 1)
    n = np.arange(N, dtype=np.float32).reshape(1, N)
    angles = -2.0 * np.pi * k * n / N
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _compute_twiddle_factors(N, H):
    """Compute twiddle factors for radix-2 FFT."""
    k = np.arange(N // 2, dtype=np.float32)
    angles = -2.0 * np.pi * k / N
    tr = np.tile(np.cos(angles).astype(np.float32), (H, 1))
    ti = np.tile(np.sin(angles).astype(np.float32), (H, 1))
    return tr, ti


# SUBSTITUTE HERE


@nki.jit
def ref(
    X_real_hbm,
    X_imag_hbm,
    W_128_real_hbm,
    W_128_imag_hbm,
    twiddle_real_hbm,
    twiddle_imag_hbm,
):
    """Reference: identical to test (256-pt radix-2 FFT)."""
    TILE_H = 128
    N = 256
    N_half = 128

    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)

    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=X_real, src=X_real_hbm[0:TILE_H, 0:N])
    nisa.dma_copy(dst=X_imag, src=X_imag_hbm[0:TILE_H, 0:N])

    W_128_real = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_imag = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=W_128_real, src=W_128_real_hbm[0:N_half, 0:N_half])
    nisa.dma_copy(dst=W_128_imag, src=W_128_imag_hbm[0:N_half, 0:N_half])

    twiddle_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=twiddle_real, src=twiddle_real_hbm[0:TILE_H, 0:N_half])
    nisa.dma_copy(dst=twiddle_imag, src=twiddle_imag_hbm[0:TILE_H, 0:N_half])

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

    Xe_real_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xe_real_T_psum, data=X_even_real)
    Xe_real_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xe_real_T, src=Xe_real_T_psum)

    Xe_imag_T_psum = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=Xe_imag_T_psum, data=X_even_imag)
    Xe_imag_T = nl.ndarray((N_half, TILE_H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xe_imag_T, src=Xe_imag_T_psum)

    X_even_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_e_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_e_real, stationary=Xe_real_T, moving=W_real_T)
    nisa.nc_matmul(dst=psum_e_real, stationary=Xe_imag_T, moving=neg_W_imag_T)
    nisa.tensor_copy(dst=X_even_fft_real, src=psum_e_real)

    X_even_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    psum_e_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=psum_e_imag, stationary=Xe_real_T, moving=W_imag_T)
    nisa.nc_matmul(dst=psum_e_imag, stationary=Xe_imag_T, moving=W_real_T)
    nisa.tensor_copy(dst=X_even_fft_imag, src=psum_e_imag)

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

    Y_combined_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_combined_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_copy(dst=Y_combined_real[0:TILE_H, 0:N_half], src=Y_first_real)
    nisa.tensor_copy(dst=Y_combined_imag[0:TILE_H, 0:N_half], src=Y_first_imag)
    nisa.tensor_copy(dst=Y_combined_real[0:TILE_H, N_half:N], src=Y_second_real)
    nisa.tensor_copy(dst=Y_combined_imag[0:TILE_H, N_half:N], src=Y_second_imag)

    nisa.dma_copy(dst=Y_real_hbm[0:TILE_H, 0:N], src=Y_combined_real)
    nisa.dma_copy(dst=Y_imag_hbm[0:TILE_H, 0:N], src=Y_combined_imag)

    return Y_real_hbm, Y_imag_hbm


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test against numpy.fft.fft."""
    device = xm.xla_device()
    H = 64
    W = 256
    TILE_H = 128

    W_128_real_np, W_128_imag_np = _compute_dft_matrix(128)
    twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, TILE_H)

    for seed in range(2):
        np.random.seed(42 + seed)
        x_np = np.random.randn(H, W).astype(np.float32)

        # NumPy reference
        ref_fft = np.fft.fft(x_np, axis=-1)
        ref_real_np = ref_fft.real.astype(np.float32)
        ref_imag_np = ref_fft.imag.astype(np.float32)

        # Pad height to 128
        x_real_padded = np.zeros((TILE_H, W), dtype=np.float32)
        x_imag_padded = np.zeros((TILE_H, W), dtype=np.float32)
        x_real_padded[:H, :] = x_np

        # Transfer to device
        xr = torch.tensor(x_real_padded, dtype=torch.float32, device=device)
        xi = torch.tensor(x_imag_padded, dtype=torch.float32, device=device)
        wr = torch.tensor(W_128_real_np, dtype=torch.float32, device=device)
        wi = torch.tensor(W_128_imag_np, dtype=torch.float32, device=device)
        tr = torch.tensor(twiddle_real_np, dtype=torch.float32, device=device)
        ti = torch.tensor(twiddle_imag_np, dtype=torch.float32, device=device)

        # Run ref
        yr_ref, yi_ref = ref_func(xr, xi, wr, wi, tr, ti)
        yr_ref_np = yr_ref.detach().cpu().numpy()[:H, :]
        yi_ref_np = yi_ref.detach().cpu().numpy()[:H, :]

        # Run test
        yr_test, yi_test = test_func(xr, xi, wr, wi, tr, ti)
        yr_test_np = yr_test.detach().cpu().numpy()[:H, :]
        yi_test_np = yi_test.detach().cpu().numpy()[:H, :]

        # Compare test vs ref (NKI-to-NKI match)
        ref_vec = np.concatenate([yr_ref_np.flatten(), yi_ref_np.flatten()])
        test_vec = np.concatenate([yr_test_np.flatten(), yi_test_np.flatten()])
        cos_sim = np.dot(ref_vec, test_vec) / (
            np.linalg.norm(ref_vec) * np.linalg.norm(test_vec) + 1e-12
        )
        if cos_sim < 0.999:
            return False

        # Also verify against numpy (sanity check)
        numpy_vec = np.concatenate([ref_real_np.flatten(), ref_imag_np.flatten()])
        cos_numpy = np.dot(ref_vec, numpy_vec) / (
            np.linalg.norm(ref_vec) * np.linalg.norm(numpy_vec) + 1e-12
        )
        if cos_numpy < 0.999:
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    H = 128
    W = 256
    TILE_H = 128

    W_128_real_np, W_128_imag_np = _compute_dft_matrix(128)
    twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, TILE_H)

    np.random.seed(42)
    x_real_np = np.random.randn(TILE_H, W).astype(np.float32)
    x_imag_np = np.zeros((TILE_H, W), dtype=np.float32)

    xr = torch.tensor(x_real_np, dtype=torch.float32, device=device)
    xi = torch.tensor(x_imag_np, dtype=torch.float32, device=device)
    wr = torch.tensor(W_128_real_np, dtype=torch.float32, device=device)
    wi = torch.tensor(W_128_imag_np, dtype=torch.float32, device=device)
    tr = torch.tensor(twiddle_real_np, dtype=torch.float32, device=device)
    ti = torch.tensor(twiddle_imag_np, dtype=torch.float32, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(xr, xi, wr, wi, tr, ti)
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
