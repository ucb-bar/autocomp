import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np
import torch_xla.core.xla_model as xm
import math
import time

@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
  # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
  # Where RMS(a_tensor) = sqrt(eps + (1/N) * sum(a_tensor * a_tensor))
  # and N = a_tensor.shape[1], eps is 1e-5
  # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                            buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[1] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[1])[None, :]

    num_rows = a_tensor.shape[0]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently
    for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

        # Load input data from external memory to on-chip memory
        a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                         mask=(i * 128 + ix < num_rows))

        # Compute element-wise square of a_tensor
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements, along last dimension
        square_sum = nl.sum(in_square, axis=[1])

        # Scale and get a reciprocal
        mean = square_sum / a_tensor.shape[1]

        # Take square root of mean and then reciprocal with
        # rsqrt API (one ISA instruction)
        rms_reciprocal = nl.rsqrt(mean + 1e-5)

        # Scale the input tensor
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        # num_rows_active = min(num_rows - i * 128, 128)
        g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast,
                            mask=(i * 128 + ix < num_rows))

        # store the addition results back to external memory (out_tensor)
        nl.store(out_tensor[i * 128 + ix, iy], value=out_tile,
                 mask=(i * 128 + ix < num_rows))

    return out_tensor

# SUBSTITUTE HERE

@nki.jit
def nki_rmsnorm_kernel_reference(a_tensor, g_tensor):
  # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
  # Where RMS(a_tensor) = sqrt(eps + (1/N) * sum(a_tensor * a_tensor))
  # and N = a_tensor.shape[1], eps is 1e-5
  # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                            buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[1] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[1])[None, :]

    num_rows = a_tensor.shape[0]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently
    for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

        # Load input data from external memory to on-chip memory
        a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                         mask=(i * 128 + ix < num_rows))

        # Compute element-wise square of a_tensor
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements, along last dimension
        square_sum = nl.sum(in_square, axis=[1])

        # Scale and get a reciprocal
        mean = square_sum / a_tensor.shape[1]

        # Take square root of mean and then reciprocal with
        # rsqrt API (one ISA instruction)
        rms_reciprocal = nl.rsqrt(mean + 1e-5)

        # Scale the input tensor
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        # num_rows_active = min(num_rows - i * 128, 128)
        g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast,
                            mask=(i * 128 + ix < num_rows))

        # store the addition results back to external memory (out_tensor)
        nl.store(out_tensor[i * 128 + ix, iy], value=out_tile,
                 mask=(i * 128 + ix < num_rows))

    return out_tensor

@nki.jit
def nki_matmul_tiled_reference_(lhsT, rhs):
    """NKI kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples for
            128.  It is the left-hand-side argument of the matrix multiplication,
            delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
            is a multiple of 512.  It is the right-hand-side argument of the matrix
            multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    if M < TILE_M:
        TILE_M = M
    
    if N < TILE_N:
        TILE_N = N

    if K < TILE_K:
        TILE_K = K

    # Use affine_range to loop over tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhsT and rhs
                lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                            m * TILE_M:(m + 1) * TILE_M])
                rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                            n * TILE_N:(n + 1) * TILE_N])

                # Accumulate partial-sums into PSUM
                res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                    value=res_sb)

    return result

def forward_reference(x, post_attention_layernorm_weight, up_proj_weight, gate_proj_weight, down_proj_weight):
    b, s, h = x.shape
    x = x.view(-1, h)
    # x = RmsNorm.apply(x, post_attention_layernorm_weight, 1e-5, len(x.shape) - 1)
    x = nki_rmsnorm_kernel_reference(x, post_attention_layernorm_weight)
    up = nki_matmul_tiled_reference_(x.t(), up_proj_weight.t())
    gate = nki_matmul_tiled_reference_(x.t(), gate_proj_weight.t())
    act = torch.nn.SiLU()(gate) * up
    output = nki_matmul_tiled_reference_(act.t() , down_proj_weight.t())

    return output

def get_test_weights(hidden_size, intermediate_size, dtype, device):
    """Create test weights for MLP."""
    post_attention_layernorm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    up_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    gate_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    down_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    return (
        post_attention_layernorm_weight,
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
    intermediate_size = 8192
    weights = get_test_weights(hidden_size, intermediate_size, dtype, device)
    
    for _ in range(2):
        batch, seq = 1, 32
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
    intermediate_size = 8192
    weights = get_test_weights(hidden_size, intermediate_size, dtype, device)
    batch, seq = 1, 32
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
