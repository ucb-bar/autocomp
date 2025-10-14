import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [64,512]
  """
  result = nl.ndarray((64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Defining indexes for input LHS.T
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_lhsT_p, i_lhsT_f = nl.mgrid[0:128, 0:64]

  # Defining indexes for input RHS
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_rhs_p, i_rhs_f = nl.mgrid[0:128, 0:512]

  # Defining indexes for the output ([64,128]@[128,512] -> [64,512])
  i_out_p, i_out_f = nl.mgrid[0:64, 0:512]

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
  rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

  # Perform the matrix-multiplication
  # Note1: We set transpose_x to True, to indicate that the LHS input is transposed
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # The result of a [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nl.store(result[i_out_p, i_out_f], value=result_sbuf)

  return result

def test_nki(ref_func, test_func):
    for _ in range(2):
        # a = input_matrix = np.random.rand(8192, 4096).astype(nl.bfloat16)
        # b = input_matrix = np.random.rand(8192, 8192).astype(nl.bfloat16)
        a = input_matrix = np.random.rand(128, 64).astype(nl.bfloat16)
        b = input_matrix = np.random.rand(128, 512).astype(nl.bfloat16)
        result_1 = ref_func(a, b)
        result_2 = test_func(a, b)
        # print("result_1", result_1[:5, :5])
        # print("result_2", result_2[:5, :5])
        if not np.allclose(result_1.astype(nl.float32), result_2.astype(nl.float32), atol=1, rtol=0.001):
            return False
    return True

def benchmark_nki(nki_func):
    # Benchmarking with large matrices to show the differences more clearly
    lhsT = nt.tensor[[128, 64], nl.bfloat16]
    rhs = nt.tensor[[128, 512], nl.bfloat16]

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(lhsT, rhs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        benchmark_nki(test)