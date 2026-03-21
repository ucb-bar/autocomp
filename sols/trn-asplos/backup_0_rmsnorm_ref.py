import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

def row_indices(width):
  """Build canonical 2D indices for a single-row tile."""
  ip = nl.arange(1)[:, None]
  iy = nl.arange(width)[None, :]
  return ip, iy


def row_dst(out_tensor, row_idx):
  """Return the destination view for one output row."""
  ip, iy = row_indices(out_tensor.shape[1])
  return out_tensor[row_idx + ip, iy]


def square(a_tensor, out_tensor):
  """(1, N) -> out_tensor(1, N): element-wise square."""
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices(a_tensor.shape[1])

  a_tile = nl.load(a_tensor[ip, iy])
  result = nisa.tensor_tensor(a_tile, a_tile, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)


def mean(a_tensor, out_tensor):
  """(1, N) -> out_tensor(1, 1): mean along free dimension."""
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices(a_tensor.shape[1])
  iw = nl.arange(1)[None, :]

  a_tile = nl.load(a_tensor[ip, iy])
  result = nisa.tensor_reduce(op=np.add, data=a_tile,
                              axis=1, keepdims=True) / a_tensor.shape[1]
  nl.store(out_tensor[ip, iw], value=result)


def rsqrt(a_tensor, out_tensor):
  """(1, 1) -> out_tensor(1, 1): reciprocal square root."""
  assert a_tensor.shape == (1, 1)
  ip, _ = row_indices(1)
  iw = nl.arange(1)[None, :]

  a_tile = nl.load(a_tensor[ip, iw])
  result = nisa.activation(op=nl.rsqrt, data=a_tile)
  nl.store(out_tensor[ip, iw], value=result)


def col_multiply(a_tensor, b_tensor, out_tensor):
  """(1, N) * (1, 1) -> out_tensor(1, N): scale row by scalar."""
  assert a_tensor.shape[0] == 1
  assert b_tensor.shape == (1, 1)
  ip, iy = row_indices(a_tensor.shape[1])
  iw = nl.arange(1)[None, :]

  a_tile = nl.load(a_tensor[ip, iy])
  b_tile = nl.load(b_tensor[ip, iw])
  b_bcast = b_tile.broadcast_to(a_tensor.shape)
  result = nisa.tensor_tensor(a_tile, b_bcast, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)


def row_multiply(a_tensor, g_tensor, out_tensor):
  """(1, N) * (N,) -> out_tensor(1, N): apply RMSNorm weight."""
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices(a_tensor.shape[1])

  a_tile = nl.load(a_tensor[ip, iy])
  g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[ip, iy])
  result = nisa.tensor_tensor(a_tile, g_tile, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)


def write_row(out_tensor, row_idx, row_tensor):
  """Write one (1, N) row into the full output tensor."""
  ip, iy = row_indices(row_tensor.shape[1])
  nl.store(row_dst(out_tensor, row_idx), value=nl.load(row_tensor[ip, iy]))


@nki.jit
def test(a_tensor, g_tensor):
  """RMSNorm with per-row 1D helper functions."""
  N = a_tensor.shape[1]

  out_tensor   = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  squared_buf  = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  mean_buf     = nl.ndarray((1, 1), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  rsqrt_buf    = nl.ndarray((1, 1), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  scaled_buf   = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  out_row_buf  = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  for i in nl.sequential_range(a_tensor.shape[0]):
    a_row = a_tensor[i:i + 1, :]
    square(a_row, squared_buf)
    mean(squared_buf, mean_buf)
    rsqrt(mean_buf, rsqrt_buf)
    col_multiply(a_row, rsqrt_buf, scaled_buf)
    row_multiply(scaled_buf, g_tensor, out_row_buf)
    write_row(out_tensor, i, out_row_buf)

  return out_tensor
