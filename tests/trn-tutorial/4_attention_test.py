import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import math

# SUBSTITUTE HERE

def numpy_attention(q, k, v):
    """NumPy reference implementation"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    
    # Not doing Q @ K.T due to NKI layout constraints which require
    # Q transposed for matmul since contraction dimension 
    # has to be mapped to the partition dimension
    # Shape: (seqlen_q, seqlen_kv)
    qk = np.matmul(q.T, k)  
    
    # Softmax
    # Shape: (seqlen_q, 1)
    row_max = np.max(qk, axis=1, keepdims=True) 
    
    # Shape: (seqlen_q, seqlen_kv)
    norm_row = qk - row_max
    exp_row = np.exp(norm_row)
    
    # Shape: (seqlen_q, 1)
    sum_row = np.sum(exp_row, axis=1, keepdims=True)  
    
    # Shape: (seqlen_q, seqlen_kv)
    scores = exp_row / sum_row  
    
    # V transpose
    v_t = v.T  # Shape: (seqlen_kv, d_head)
    
    # scores @ V
    attn_out = np.matmul(scores, v_t)  # Shape: (seqlen_q, d_head)
    
    return attn_out

def construct_args(d_head, seq_len, dtype):
    # Set up input tensors
    q = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    k = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    v = (np.random.random_sample([d_head, seq_len]).astype(dtype) - 0.5) * 2
    return q, k, v

def test_nki(ref_func, test_func):
  for _ in range(2):
    args = construct_args(128, 4096, dtype=nl.float32)
    result_1 = ref_func(*args)
    result_2 = test_func(*args)
    print("result_1", result_1[:5, :5])
    print("result_2", result_2[:5, :5])
    if not np.allclose(result_1.astype(nl.float32), result_2.astype(nl.float32), atol=1e-2):
      return False
  return True

def benchmark_nki(nki_func):
  # Benchmarking with large matrices to show the differences more clearly
  args = construct_args(128, 4096, dtype=nl.bfloat16)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(*args)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(numpy_attention, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    benchmark_nki(test)