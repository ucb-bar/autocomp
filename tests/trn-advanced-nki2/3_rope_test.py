import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE


@nki.jit
def ref(x_in, cos, sin, lnc_shard=False, first_second_half_impl=True):
    d_head, S = x_in.shape
    d_half = d_head // 2

    x_out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    x_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    x_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_upper, src=x_in[0:d_half, 0:S])
    nisa.dma_copy(dst=x_lower, src=x_in[d_half:d_head, 0:S])

    sb_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    sb_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb_cos, src=cos[0:d_half, 0:S])
    nisa.dma_copy(dst=sb_sin, src=sin[0:d_half, 0:S])

    e_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_cos, data1=x_upper, data2=sb_cos, op=nl.multiply)

    o_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_sin, data1=x_lower, data2=sb_sin, op=nl.multiply)

    o_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_cos, data1=x_lower, data2=sb_cos, op=nl.multiply)

    e_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_sin, data1=x_upper, data2=sb_sin, op=nl.multiply)

    out_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_upper, data1=e_cos, data2=o_sin, op=nl.subtract)

    out_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_lower, data1=o_cos, data2=e_sin, op=nl.add)

    nisa.dma_copy(dst=x_out[0:d_half, 0:S], src=out_upper)
    nisa.dma_copy(dst=x_out[d_half:d_head, 0:S], src=out_lower)

    return x_out



def test_nki(ref_func, test_func):
    """
    Test function to compare reference and test implementations of RoPE.

    Args:
        ref_func: Reference implementation function (pure NumPy)
        test_func: Test implementation function to validate (NKI kernel)

    Returns:
        bool: True if test passes, False otherwise
    """
    device = xm.xla_device()
    for _ in range(2):
        x_in_np = np.random.rand(128, 4096).astype(np.float32)
        cos_np  = np.random.rand(64, 4096).astype(np.float32)
        sin_np  = np.random.rand(64, 4096).astype(np.float32)
        x_in = torch.from_numpy(x_in_np).to(device=device)
        cos  = torch.from_numpy(cos_np).to(device=device)
        sin  = torch.from_numpy(sin_np).to(device=device)
        ref_out  = ref_func(x_in, cos, sin)
        test_out = test_func(x_in, cos, sin)
        if not np.allclose(test_out.detach().cpu().numpy(), ref_out.detach().cpu().numpy(), atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    device = xm.xla_device()
    x_in_np = np.random.rand(128, 4096).astype(np.float32)
    cos_np  = np.random.rand(64, 4096).astype(np.float32)
    sin_np  = np.random.rand(64, 4096).astype(np.float32)
    x_in = torch.from_numpy(x_in_np).to(device=device)
    cos  = torch.from_numpy(cos_np).to(device=device)
    sin  = torch.from_numpy(sin_np).to(device=device)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(x_in, cos, sin)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1" 
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)