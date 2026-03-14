import math

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def ref(delta, u, A, B, C):
  """Beta 1 Mamba reference kernel (from `3_mamba_test.py`)."""
  batch_size, channels, seq_len = delta.shape
  output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                      buffer=nl.shared_hbm)
  _, state_size = A.shape

  # Map channels to the partition dimension and tile it.
  channel_psize = nl.tile_size.pmax
  n_channel_tile = channels // channel_psize

  # Magic number, decided through empirical profiling data.
  seq_len_fsize = 512
  n_seq_len_tile = seq_len // seq_len_fsize

  # Fix later with mask
  assert channels % channel_psize == 0
  assert seq_len % seq_len_fsize == 0

  for i_batch in nl.affine_range(batch_size):
    for i_channel_tile in nl.affine_range(n_channel_tile):
      channel_start = i_channel_tile * channel_psize

      # partial accumulated scanC result with processed states
      scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

      # Load delta/u once to be reused across states
      delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
      u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

      for i_state in nl.affine_range(state_size):
        A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

        # Last scan result
        scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)
        for i_seq_len_tile in nl.static_range(n_seq_len_tile):
          seq_len_start = i_seq_len_tile * seq_len_fsize

          # Step 1&2: exp(delta_i * A_i)
          deltaA = nisa.activation(op=nl.exp,
                                   data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                                   scale=A_i)

          B_i = nl.load(B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

          # Step 3: delta_i * B_i * u_i
          deltaU = nisa.tensor_tensor(delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                                      u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                                      op=nl.multiply)
          B_i_bcast = B_i.broadcast_to((channel_psize, seq_len_fsize))
          deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

          # Step 4: Associative scan
          scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                                                op0=np.multiply, op1=np.add)
          scan_init[...] = scan_res[0:channel_psize, seq_len_fsize-1]

          C_i = nl.load(C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])
          C_i_bcast = C_i.broadcast_to((channel_psize, seq_len_fsize))
          scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

          # Step 6: accumulate across state_size
          scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize] += scanC

      nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
               scanC_accum[0:channel_psize, 0:seq_len])

  return output


if __name__ == "__main__":
  delta = np.load("mamba_delta.npy")
  u = np.load("mamba_u.npy")
  A = np.load("mamba_A.npy")
  B = np.load("mamba_B.npy")
  C = np.load("mamba_C.npy")

  out = ref(delta, u, A, B, C)
  np.save("out_beta1_mamba.npy", out)

