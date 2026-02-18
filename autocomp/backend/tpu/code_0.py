import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import time
import numpy as np

# naive matmul kernel
def matmul_kernel(x_ref, y_ref, o_ref):
    """
    The 'naive' version: Explicitly iterating through the indices.
    This gives AutoComp a clear scalar/vector loop pattern to optimize
    into a tiled MXU (Matrix Unit) operation.
    """
    M, K = x_ref.shape
    K_alt, N = y_ref.shape
    
    # Initialize output
    o_ref[...] = jnp.zeros((M, N), dtype=o_ref.dtype)

    # We use a single loop over rows. 
    # This keeps it naive (HBM bound) but uses Vector-Matrix ops
    # which the TPU compiler actually understands.
    for i in range(M):
        # x_ref[i:i+1, :] stays a 2D 'row' tensor (1, K)
        # y_ref[...] is a 2D 'matrix' tensor (K, N)
        # Result is (1, N)
        row_x = x_ref[i:i+1, :] 
        o_ref[i:i+1, :] = jnp.matmul(row_x, y_ref[...])

# Wrap the kernel with pallas_call
@jax.jit
def pallas_matmul(x, y):
    M, K = x.shape
    _, N = y.shape
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid=(1,) 
    )(x, y)

# baseline matmul impl
def jax_matmul_baseline(x, y):
    return jnp.matmul(x, y)

def get_input_tensors(M, K, N, dtype):
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)
    return x, y

def run_compare(dtype=jnp.float32):
    M, K, N = 512, 128, 512
    
    args = get_input_tensors(M, K, N, dtype)
    
    print(f"Running Naive Pallas Matmul ({M}x{K}x{N})...")
    t0 = time.perf_counter()
    pallas_out = jax.block_until_ready(pallas_matmul(*args))
    t1 = time.perf_counter()
    print(f"Pallas Latency: {(t1 - t0) * 1000:.3f} ms")

    baseline_out = jax_matmul_baseline(*args)
    
    # Validation
    if jnp.allclose(baseline_out, pallas_out, atol=1e-2, rtol=1e-2):
        print("SUCCESS: Naive Pallas matches JAX baseline.")
        return True
    else:
        print("FAIL: Verification failed.")
        return False

if __name__ == "__main__":
    run_compare()

# import jax
# import jax.numpy as jnp
# from jax.experimental import pallas as pl
# from jax.experimental.pallas import tpu as pltpu
# import time
# import numpy as np

# # Define the matmul kernel
# def matmul_kernel(x_ref, y_ref, o_ref):
#     """
#     Simple matmul kernel using Pallas.
#     Computes: o = x @ y
#     """
#     # Read from VMEM
#     x = x_ref[...]
#     y = y_ref[...]
#     # Compute matmul and write to VMEM
#     o_ref[...] = jnp.dot(x, y)

# # Wrap the kernel with pallas_call
# @jax.jit
# def pallas_matmul(x, y):
#     """
#     Pallas-based matrix multiplication.
    
#     Args:
#         x: array of shape (M, K)
#         y: array of shape (K, N)
#     Returns:
#         result: x @ y of shape (M, N)
#     """
#     M, K = x.shape
#     K2, N = y.shape
#     assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    
#     return pl.pallas_call(
#         matmul_kernel,
#         out_shape=jax.ShapeDtypeStruct((M, N), x.dtype)
#     )(x, y)



# def jax_matmul_baseline(x, y):
#     # """
#     # JAX baseline for matrix multiplication.
    
#     # Args:
#     #     x: array of shape (M, K)
#     #     y: array of shape (K, N)
#     # Returns:
#     #     result: x @ y of shape (M, N)
#     # """
#     return jnp.matmul(x, y)
#     # #slow matmul
#     # rows_A = len(x)
#     # cols_A = len(x[0])
#     # rows_B = len(y)
#     # cols_B = len(y[0])

#     # if cols_A != rows_B:
#     #     raise ValueError("Matrices cannot be multiplied!")

#     # # Create a result matrix filled with zeros
#     # C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

#     # for i in range(rows_A):
#     #     for j in range(cols_B):
#     #         for k in range(cols_A):
#     #             C[i][j] += x[i][k] * y[k][j]
#     # return C
    

# def get_input_tensors(M, K, N, dtype):
#     """Create test input tensors."""
#     x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
#     y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)
#     return x, y

# def run_compare(dtype=jnp.float32):
#     # Matrix dimensions (M, K) @ (K, N) = (M, N)
#     M, K, N = 512, 512, 512
    
#     def run_baseline(x, y):
#         out = jax_matmul_baseline(x, y)
#         jax.block_until_ready(out)
#         return out
    
#     def run_pallas(x, y):
#         out = pallas_matmul(x, y)
#         jax.block_until_ready(out)
#         return out
    
#     args = get_input_tensors(M, K, N, dtype)
#     baseline_out = run_baseline(*args)
#     pallas_out = run_pallas(*args)
    
#     if not jnp.allclose(baseline_out, pallas_out, atol=1e-3, rtol=1e-3):
#         b0 = np.array(baseline_out)
#         p0 = np.array(pallas_out)
#         diff_l2_norm = np.linalg.norm(b0 - p0)
#         b0_l2_norm = np.linalg.norm(b0)
#         print(f"diff_l2_norm / b0_l2_norm: {diff_l2_norm / b0_l2_norm}")
#         if diff_l2_norm / b0_l2_norm < 1e-3:
#             print("Failed allclose, but L2 norm of difference is less than 1e-3 of baseline L2 norm")
#         else:
#             print(f"baseline_out.shape: {b0.shape}")
#             print(f"pallas_out.shape: {p0.shape}")
#             print(f"baseline_out[0, :8]: {b0[0, :8]}")
#             print(f"pallas_out[0, :8]: {p0[0, :8]}")
#             diff = np.abs(b0 - p0)
#             print(f"max_diff: {diff.max()}")
#             print(f"mean_diff: {diff.mean()}")
#             print("FAIL: test does not match baseline")
#             return False
    
#     # Lightweight perf check
#     perf_iters = 50
#     if perf_iters > 0:
#         args = get_input_tensors(M, K, N, dtype)
#         # Warmup to ensure compilation happens before timing
#         _ = run_baseline(*args)
#         jax.block_until_ready(_)
#         t0 = time.perf_counter()
#         for _ in range(perf_iters):
#             baseline_out = run_baseline(*args)
#         jax.block_until_ready(baseline_out)
#         t1 = time.perf_counter()
#         baseline_ms = (t1 - t0) * 1000.0 / perf_iters
#         print("Latency: {:.3f} ms".format(baseline_ms))
#     return True

# if __name__ == "__main__":
#     success = run_compare()
#     if not success:
#         exit(1)
#     print("Test passed")
