import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import math

# SUBSTITUTE HERE

@nki.jit
def mamba_v3(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Magic number, decided through empiracal profiling data
    seq_len_fsize = 512
    n_seq_len_tile = seq_len // seq_len_fsize

    # Fix this later with mask
    assert channels % channel_psize == 0
    assert seq_len % seq_len_fsize == 0

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Last scan result
                scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)
                # FIXME: sequential_range gives incorrect answer and also much worse perf than static_range
                # for i_seq_len_tile in nl.sequential_range(n_seq_len_tile):
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_len_start = i_seq_len_tile * seq_len_fsize

                    # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                    deltaA = nisa.activation(op=nl.exp,
                            data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            scale=A_i)

                    # Load the relevant tile from B
                    B_i = nl.load(B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                    deltaU = nisa.tensor_tensor(delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            op=nl.multiply)
                    B_i_bcast = B_i.broadcast_to((channel_psize, seq_len_fsize))
                    deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                    # Step 4: Associative scan between deltaA and deltaBu
                    scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                            op0=np.multiply, op1=np.add)
                    scan_init[...] = scan_res[0:channel_psize, seq_len_fsize-1]

                    # Load the relevant tile from C
                    C_i = nl.load(C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 5: Element-wise multiplication of scan_res and C_i
                    C_i_bcast = C_i.broadcast_to((channel_psize, seq_len_fsize))
                    scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                    # Step 6: Accumulation of scanC along state_size dimension
                    scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize] += scanC

            # Store scanC_accum for a single batch to output
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    return output

def construct_args(batch, seq_len, channels, state_size):
    # Set up input tensors
    dtype = np.float32
    delta = np.random.rand(batch, channels, seq_len).astype(dtype)
    u = np.random.rand(batch, channels, seq_len).astype(dtype)
    A = -np.random.rand(channels, state_size).astype(dtype)
    B = np.random.rand(batch, state_size, seq_len).astype(dtype)
    C = np.random.rand(batch, state_size, seq_len).astype(dtype)
    return delta, u, A, B, C

def test_nki(ref_func, test_func):
  for _ in range(2):
    args = construct_args(1, 2048, 256, 16)
    result_1 = ref_func(*args)
    result_2 = test_func(*args)
    print("result_1", result_1[:5, :5])
    print("result_2", result_2[:5, :5])
    if not np.allclose(result_1.astype(nl.float32), result_2.astype(nl.float32), atol=1e-3, rtol=1e-3):
      return False
  return True

def benchmark_nki(nki_func):
  # Benchmarking with large matrices to show the differences more clearly
  args = construct_args(1, 2048, 256, 16)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(*args)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(mamba_v3, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    benchmark_nki(test)