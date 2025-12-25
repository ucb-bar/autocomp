import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np
import torch_xla.core.xla_model as xm
import math
import time

# SUBSTITUTE HERE

@nki.jit
def nki_matmul_tiled_batched_(lhsT, rhs):
    """Batched NKI matmul kernel using the same tiling scheme as nki_matmul_tiled_.

    Args:
        lhsT: input tensor of shape [B, K, M], where K and M are multiples of 128.
             It is the left-hand-side argument of the matrix multiplication,
             delivered transposed for optimal performance, with a leading batch dim.
        rhs:  input tensor of shape [K, N], where K is a multiple of 128 and N
             is a multiple of 512. It is the right-hand-side argument.
    Returns:
        result: output tensor of shape [B, M, N]
    """

    B, K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((B, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    if M < TILE_M:
        TILE_M = M
    
    if N < TILE_N:
        TILE_N = N

    if K < TILE_K:
        TILE_K = K

    # Outer batch loop reusing the same tiling for each batch element.
    for b in nl.affine_range(B):
        # Use affine_range to loop over tiles in M and N for this batch element
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                # Allocate a tensor in PSUM
                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Declare the tiles on SBUF
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                    # Load tiles from lhsT and rhs
                    lhsT_tile[...] = nl.load(
                        lhsT[b, k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M]
                    )
                    rhs_tile[...] = nl.load(
                        rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N]
                    )

                    # Accumulate partial-sums into PSUM
                    res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                res_sb = nl.copy(res_psum, dtype=result.dtype)
                nl.store(
                    result[
                        b,
                        m * TILE_M:(m + 1) * TILE_M,
                        n * TILE_N:(n + 1) * TILE_N,
                    ],
                    value=res_sb,
                )

    return result

def forward_reference(x, up_proj_weight, gate_proj_weight, down_proj_weight):
    b, s, h = x.shape
    # x = RmsNorm.apply(x, post_attention_layernorm_weight, 1e-5, len(x.shape) - 1)
    # x = nki_rmsnorm_kernel_reference(x, post_attention_layernorm_weight)
    up = nki_matmul_tiled_batched_(x.transpose(1, 2), up_proj_weight)
    gate = nki_matmul_tiled_batched_(x.transpose(1, 2), gate_proj_weight)
    act = torch.nn.SiLU()(gate) * up
    output = nki_matmul_tiled_batched_(act.transpose(1, 2), down_proj_weight)

    return output

def get_test_weights(hidden_size, intermediate_size, dtype, device):
    """Create test weights for MLP."""
    up_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    gate_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    down_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    return (
        up_proj_weight,
        gate_proj_weight,
        down_proj_weight,
    )


def compare_outputs(reference_out, test_out, atol=1e-3, rtol=1e-3):
    """Compare test output against reference output."""
    if not torch.allclose(reference_out, test_out, atol=atol, rtol=rtol):
        ref_cpu = reference_out.cpu()
        test_cpu = test_out.cpu()
        print("reference_out[:8]: %s", ref_cpu.flatten()[:8])
        print("test_out[:8]: %s", test_cpu.flatten()[:8])
        diff = (ref_cpu - test_cpu).abs()
        print("max_diff: %s", diff.max())
        print("mean_diff: %s", diff.mean())
        print("FAIL: test output does not match reference")
        return False
    return True

def benchmark_forward(forward_fn, x, weights, perf_iters=50):
    """Benchmark a forward function."""
    # Warmup
    _ = forward_fn(x, *weights)
    
    t0 = time.time()
    for _ in range(perf_iters):
        _ = forward_fn(x, *weights)
    t1 = time.time()
    ms_per_iter = (t1 - t0) * 1000.0 / perf_iters
    return ms_per_iter

def test_nki(ref_func, test_func):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = xm.xla_device()
    hidden_size = 2048
    intermediate_size = 4096
    weights = get_test_weights(hidden_size, intermediate_size, dtype, device)
    
    for _ in range(2):
        batch, seq = 32, 1
        x = torch.randn(batch, seq, hidden_size, dtype=dtype, device=device)
        ref_out = ref_func(x, *weights)
        test_out = test_func(x, *weights)
        if not compare_outputs(ref_out, test_out):
            return False
    return True

def benchmark_nki(nki_func):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = xm.xla_device()
    hidden_size = 2048
    intermediate_size = 4096
    weights = get_test_weights(hidden_size, intermediate_size, dtype, device)
    batch, seq = 32, 1
    x = torch.randn(batch, seq, hidden_size, dtype=dtype, device=device)
    
    test_ms = benchmark_forward(nki_func, x, weights, 50)
    print("Latency: {:.3f} ms/iter".format(test_ms))

if __name__ == "__main__":
    test_result = test_nki(forward_reference, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
