## neuron_profile_for_nki.html

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
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

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler to change this loop by,
    # for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x])

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype=lhsT.dtype,
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nisa.nc_matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

            # Accumulate on corresponding SBUF tile
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                   dtype=result_tiles.dtype,
                                   buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])

  return result


class NKILinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NKILinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        weight_T = self.weight.t()
        x_T = x.t()
        output = nki_matmul_fully_optimized_(x_T, weight_T)
        return output + self.bias


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = NKILinear(2048, 2048)
        self.fc2 = NKILinear(2048, 1024)
        self.fc3 = NKILinear(1024, 1024)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
```

## neuron_profile_for_nki.html

```python
import torch
from neuronxcc import nki
import neuronxcc.nki.language as nl
import math
import os

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  sz_p, sz_f = in_tensor.shape

  i_f = nl.arange(sz_f)[None, :]

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

    return out_tensor
```

```python
from neuronxcc import nki
from neuronxcc.nki.typing import tensor
import neuronxcc.nki.language as nl
import math
from pathlib import Path

WORKING_DIRECTORY = Path.home() / 'reports'

@nki.profile(working_directory=WORKING_DIRECTORY, save_neff_name='file.neff', save_trace_name='profile.ntff', profile_nth=2)
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor
  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  sz_p, sz_f = in_tensor.shape
  i_f = nl.arange(sz_f)[None, :]
  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))
    # perform the computation
    out_tile = nl.exp(in_tile)
    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

  return out_tensor
```

## getting_started.html

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors"""
    
    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape
    
    # Check size of the first dimension does not exceed on-chip memory tile size limit
    assert a_input.shape[0] <= nl.tile_size.pmax
    
    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    
    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)
    
    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    
    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)
    
    # Return kernel output as function output
    return c_output
```

```python
import numpy as np

a = np.ones((4, 3), dtype=np.float16)
b = np.ones((4, 3), dtype=np.float16)

c = nki_tensor_add_kernel(a, b)
```

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.ones((4, 3), dtype=torch.float16).to(device=device)
b = torch.ones((4, 3), dtype=torch.float16).to(device=device)

c = nki_tensor_add_kernel(a, b)
```

```python
import jax.numpy as jnp

a = jnp.ones((4, 3), dtype=jnp.float16)
b = jnp.ones((4, 3), dtype=jnp.float16)

c = nki_tensor_add_kernel(a, b)
```

## nki.isa.nc_matmul.html

```python
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

##################################################################
# Example 1:
# multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
# to get matrix c in PSUM of shape (128, 512)
##################################################################
a_mgrid = nl.mgrid[0:128, 0:128]
b_mgrid = nl.mgrid[0:128, 0:512]
c_mgrid = nl.mgrid[0:128, 0:512]

a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])

c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])

nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)

##################################################################
# Example 2:
# multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
# to get matrix f in PSUM of shape (128, 512) using psum accumulation
##################################################################
d_mgrid = nl.mgrid[0:128, 0:128]
e_mgrid = nl.mgrid[0:128, 0:512]
f_mgrid = nl.mgrid[0:128, 0:512]

f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

for i_contract in nl.affine_range(2):
  d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
  e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
  f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
  
nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)

##################################################################
# Example 3:
# perform batched matrix multiplication on matrix g of shape (16, 64, 64) 
# and matrix h of shape (16, 64, 512) to get matrix i of (16, 64, 512) 
# using Tensor Engine PE tiling mode. 
##################################################################
g_mgrid = nl.mgrid[0:64, 0:64]
h_mgrid = nl.mgrid[0:64, 0:512]
i_mgrid = nl.mgrid[0:64, 0:512]

for i in nl.affine_range(4):
  for j in nl.affine_range(4):
    g = nl.load(g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
    h = nl.load(h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
    i_psum = nisa.nc_matmul(g, h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
    nl.store(i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], i_psum)
```

## nki.baremetal.html

```python
from neuronxcc.nki import baremetal
import neuronxcc.nki.language as nl
import numpy as np

@baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = a + b

  nl.store(c_tensor, c)

  return c_tensor
```

## nki.simulate_kernel.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit
def print_kernel(a_tensor):
  b = nl.empty_like(a_tensor, buffer=nl.hbm)

  # Load tensor into sbuf
  a = nl.load(a_tensor)

  # Print tensor y
  nl.device_print("value of a:", a)

  # Directly store a into hbm
  nl.store(b, value=a)

  return b
```

## nki.jit.html

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = a + b

  nl.store(c_tensor, c)

  return c_tensor
```