import numpy as np
import torch

import nki
import nki.language as nl
import nki.isa as nisa
from torch_xla.core import xla_model as xm


# ==============================================================================
# Host-side helpers (data layout transformation)
# ==============================================================================


def selective_scan_reference(x, dt, A, B, C, D):
    """Sequential O(L) reference selective scan (CPU)."""
    batch, seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    state = torch.zeros(batch, num_heads, head_dim, ssm_state_size, dtype=x.dtype)
    y = torch.zeros_like(x)

    for t in range(seq_len):
        dA = torch.exp(dt[:, t, :] * A)
        dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :, :]
        dBx = dB.unsqueeze(2) * x[:, t, :, :].unsqueeze(-1)

        state = dA.unsqueeze(-1).unsqueeze(-1) * state + dBx
        y[:, t, :, :] = torch.einsum("bhds,bhs->bhd", state, C[:, t, :, :])
        y[:, t, :, :] += D.view(1, -1, 1) * x[:, t, :, :]

    return y, state


def prepare_scan_inputs(x, dt, A, B, C, D):
    """Pre-compute and transpose inputs for the NKI kernel."""
    batch, seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    dA_exp = torch.exp(dt * A.view(1, 1, -1))
    dB = dt.unsqueeze(-1) * B
    dBx = dB.unsqueeze(3) * x.unsqueeze(-1)

    dA_exp_t = dA_exp[0].transpose(0, 1).contiguous()

    dBx_0 = dBx[0]
    dBx_reshaped = dBx_0.permute(0, 2, 3, 1).reshape(
        seq_len, head_dim * ssm_state_size * num_heads
    )
    dBx_t = dBx_reshaped.transpose(0, 1).contiguous()

    C_0 = C[0]
    C_reshaped = C_0.permute(0, 2, 1).reshape(seq_len, ssm_state_size * num_heads)
    C_t = C_reshaped.transpose(0, 1).contiguous()

    x_0 = x[0]
    x_reshaped = x_0.permute(0, 2, 1).reshape(seq_len, head_dim * num_heads)
    x_t = x_reshaped.transpose(0, 1).contiguous()

    Dx_0 = D.view(1, -1, 1) * x_0
    Dx_reshaped = Dx_0.permute(0, 2, 1).reshape(seq_len, head_dim * num_heads)
    Dx_t = Dx_reshaped.transpose(0, 1).contiguous()

    return {
        "dA_exp_t": dA_exp_t.float(),
        "dBx_t": dBx_t.float(),
        "C_t": C_t.float(),
        "Dx_t": Dx_t.float(),
        "x_t": x_t.float(),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "ssm_state_size": ssm_state_size,
    }


def unpack_scan_outputs(
    y_flat, state_flat, num_heads, head_dim, ssm_state_size, seq_len
):
    """Unpack NKI kernel outputs back to standard shapes."""
    y_reshaped = y_flat.reshape(head_dim, num_heads, seq_len)
    y = y_reshaped.permute(2, 1, 0).unsqueeze(0).contiguous()

    state_reshaped = state_flat.reshape(head_dim, ssm_state_size, num_heads)
    final_state = state_reshaped.permute(2, 0, 1).unsqueeze(0).contiguous()

    return y, final_state


# SUBSTITUTE HERE


P_MAX = 128


@nki.jit
def ref(
    dA_exp_t,
    dBx_t,
    C_t,
    Dx_t,
    x_t,
    hd_range,
    ss_range,
):
    """Reference: identical to test (Mamba2 selective scan)."""
    NH = dA_exp_t.shape[0]
    SL = dA_exp_t.shape[1]
    HD = hd_range.shape[0]
    SS = ss_range.shape[0]

    y_out = nl.ndarray((NH * HD, SL), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (NH * HD * SS, 1), dtype=nl.float32, buffer=nl.shared_hbm
    )

    dA_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dA_sb, value=0.0)
    nisa.dma_copy(
        dst=dA_sb[0:NH, 0:SL],
        src=dA_exp_t[0:NH, 0:SL],
    )

    for d in nl.affine_range(HD):
        y_acc_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=y_acc_sb, value=0.0)
        Dx_row_start = d * NH
        nisa.dma_copy(
            dst=y_acc_sb[0:NH, 0:SL],
            src=Dx_t[Dx_row_start : Dx_row_start + NH, 0:SL],
        )

        for s in nl.affine_range(SS):
            dBx_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=dBx_sb, value=0.0)
            dBx_row = (d * SS + s) * NH
            nisa.dma_copy(
                dst=dBx_sb[0:NH, 0:SL],
                src=dBx_t[dBx_row : dBx_row + NH, 0:SL],
            )

            init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=init_sb, value=0.0)

            state_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=state_sb[0:NH, 0:SL],
                data0=dA_sb[0:NH, 0:SL],
                data1=dBx_sb[0:NH, 0:SL],
                initial=init_sb[0:NH, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=final_sb[0:NH, 0:1],
                src=state_sb[0:NH, SL - 1 : SL],
            )
            fs_row = (d * SS + s) * NH
            nisa.dma_copy(
                dst=final_state_out[fs_row : fs_row + NH, 0:1],
                src=final_sb[0:NH, 0:1],
            )

            C_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=C_sb, value=0.0)
            C_row = s * NH
            nisa.dma_copy(
                dst=C_sb[0:NH, 0:SL],
                src=C_t[C_row : C_row + NH, 0:SL],
            )

            Cs_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=Cs_sb[0:NH, 0:SL],
                data1=C_sb[0:NH, 0:SL],
                data2=state_sb[0:NH, 0:SL],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=y_acc_sb[0:NH, 0:SL],
                data1=y_acc_sb[0:NH, 0:SL],
                data2=Cs_sb[0:NH, 0:SL],
                op=nl.add,
            )

        y_row = d * NH
        nisa.dma_copy(
            dst=y_out[y_row : y_row + NH, 0:SL],
            src=y_acc_sb[0:NH, 0:SL],
        )

    return y_out, final_state_out


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test NKI kernels against CPU reference.

    Uses small dimensions (NH=32, HD=4, SS=8, SL=16) for fast compilation.
    """
    device = xm.xla_device()

    for seed in range(2):
        batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 32, 4, 8
        torch.manual_seed(42 + seed)

        x = torch.randn(batch, seq_len, num_heads, head_dim)
        dt = torch.rand(batch, seq_len, num_heads) * 0.1
        A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
        B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
        C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
        D = torch.ones(num_heads)

        # CPU reference
        y_cpu, state_cpu = selective_scan_reference(x, dt, A, B, C, D)

        # Prepare inputs for NKI
        inputs = prepare_scan_inputs(x, dt, A, B, C, D)

        dA_dev = inputs["dA_exp_t"].to(device)
        dBx_dev = inputs["dBx_t"].to(device)
        C_dev = inputs["C_t"].to(device)
        Dx_dev = inputs["Dx_t"].to(device)
        x_dev = inputs["x_t"].to(device)
        hd_dev = torch.zeros(head_dim, dtype=torch.float32, device=device)
        ss_dev = torch.zeros(ssm_state_size, dtype=torch.float32, device=device)

        # Run ref kernel
        y_flat_ref, state_flat_ref = ref_func(
            dA_dev, dBx_dev, C_dev, Dx_dev, x_dev, hd_dev, ss_dev
        )
        y_ref_nki, state_ref_nki = unpack_scan_outputs(
            y_flat_ref.cpu(),
            state_flat_ref.cpu(),
            num_heads,
            head_dim,
            ssm_state_size,
            seq_len,
        )

        # Run test kernel
        y_flat_test, state_flat_test = test_func(
            dA_dev, dBx_dev, C_dev, Dx_dev, x_dev, hd_dev, ss_dev
        )
        y_test_nki, state_test_nki = unpack_scan_outputs(
            y_flat_test.cpu(),
            state_flat_test.cpu(),
            num_heads,
            head_dim,
            ssm_state_size,
            seq_len,
        )

        # Compare test vs ref (NKI-to-NKI match)
        ref_vec = y_ref_nki.flatten().numpy()
        test_vec = y_test_nki.flatten().numpy()
        cos_sim = np.dot(ref_vec, test_vec) / (
            np.linalg.norm(ref_vec) * np.linalg.norm(test_vec) + 1e-12
        )
        if cos_sim < 0.999:
            print(f"  FAIL: ref vs test cosine={cos_sim:.6f} (seed={seed})")
            return False

        # Also verify against CPU reference (sanity check)
        cpu_vec = y_cpu.flatten().numpy()
        cos_cpu = np.dot(ref_vec, cpu_vec) / (
            np.linalg.norm(ref_vec) * np.linalg.norm(cpu_vec) + 1e-12
        )
        if cos_cpu < 0.999:
            print(f"  FAIL: NKI vs CPU cosine={cos_cpu:.6f} (seed={seed})")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()

    batch, seq_len, num_heads, head_dim, ssm_state_size = 1, 16, 32, 4, 8
    torch.manual_seed(42)

    x = torch.randn(batch, seq_len, num_heads, head_dim)
    dt = torch.rand(batch, seq_len, num_heads) * 0.1
    A = -torch.arange(1, num_heads + 1, dtype=torch.float32)
    B = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    C = torch.randn(batch, seq_len, num_heads, ssm_state_size)
    D = torch.ones(num_heads)

    inputs = prepare_scan_inputs(x, dt, A, B, C, D)

    dA_dev = inputs["dA_exp_t"].to(device)
    dBx_dev = inputs["dBx_t"].to(device)
    C_dev = inputs["C_t"].to(device)
    Dx_dev = inputs["Dx_t"].to(device)
    x_dev = inputs["x_t"].to(device)
    hd_dev = torch.zeros(head_dim, dtype=torch.float32, device=device)
    ss_dev = torch.zeros(ssm_state_size, dtype=torch.float32, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(dA_dev, dBx_dev, C_dev, Dx_dev, x_dev, hd_dev, ss_dev)
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
        benchmark_nki(test)
