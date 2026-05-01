## Language: Loop Constructs

NKI loop constructs control parallelization and data dependencies in kernel loops.

### affine_range

`nl.affine_range(num_iterations: int)` creates parallel loop iterators for loops without loop-carried dependencies.

This is the preferred loop construct when each iteration operates on independent data. Associative reductions into a buffer outside the loop (e.g., multiple nc_matmul accumulating into the same PSUM tile) are allowed. Use when overlapping loads and computes is desired; the compiler can vectorize and pipeline affine_range loops efficiently.

Examples:
```python
for i_input in nl.affine_range(input.shape[1] // 512):
  offset = i_input * 512
  input_sb = nl.ndarray((input.shape[0], 512), dtype=input.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=input_sb, src=input[0:input.shape[0], offset:offset+512])
  result = nl.multiply(input_sb, input_sb)
  nisa.dma_copy(dst=output[0:input.shape[0], offset:offset+512], src=result)
```

### sequential_range

`nl.sequential_range(num_iterations: int)` creates sequential loop iterators for loops with loop-carried dependencies.

Use when the result of one iteration is needed by the next. The Neuron compiler executes iterations in order without pipelining.

Examples:
```python
init = nl.zeros((128, 1), dtype=input0.dtype)
for i_input in nl.sequential_range(input0.shape[1] // 512):
  offset = i_input * 512
  result = nl.ndarray((128, 512), dtype=input0.dtype, buffer=nl.sbuf)
  nisa.tensor_tensor_scan(dst=result, data0=input0[:, offset:offset+512], data1=input1[:, offset:offset+512], initial=init, op0=nl.multiply, op1=nl.add)
  nisa.dma_copy(dst=output[0:input0.shape[0], offset:offset+512], src=result)
  init[:, :] = result[:, 511]
```

## Language: Allocation and Tensor Creation

### ndarray

`nl.ndarray(shape: tuple, dtype: nki_dtype, buffer=nl.sbuf|nl.psum|nl.shared_hbm)` creates a new tensor on the specified buffer.

### zeros

`nl.zeros(shape: tuple, dtype: nki_dtype, buffer=nl.sbuf|nl.psum|nl.shared_hbm)` creates a new zeroed tensor.

### par_dim

`nl.par_dim(dim: int)` marks a dimension as the partition dimension in Tensor layout. Beta 2: block dimensions removed; partition dimension must be the first dimension.

### tile_size

Class with attributes for hardware tile constraints:
- `pmax`: Maximum partition dimension (128)
- `sbuf_min_align`: Minimum SBUF free-dimension alignment (bytes)
- `gemm_stationary_fmax`: Max free dim for stationary matmul operand (128)
- `gemm_moving_fmax`: Max free dim for moving matmul operand (512)
- `psum_fmax`: Max free dim in PSUM (512)
- `bn_stats_fmax`: Max free dim for bn_stats input (512)

### mgrid

`nl.mgrid[start:end, start:end, ...]` creates dense mesh-grid arrays (NumPy-compatible).

## Memory and Data Movement

### dma_copy

`nisa.dma_copy(dst: tile, src: tile, dst_rmw_op=None, dge_mode=unknown, oob_mode=error, dtype=src.dtype)` copies data between HBM and SBUF using DMA, with optional read-modify-write.

Supports static and dynamic addressing. With swdge/hwdge modes, src/dst can have dynamic start addresses and can perform gather/scatter. Only nl.add is supported for dst_rmw_op.

Examples:
```python
x = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=x, src=in_tensor)
# ... compute on x ...
nisa.dma_copy(dst=out_tensor, src=x)
```

### dma_transpose

`nisa.dma_transpose(dst: tile, src: tile, axes=None, dge_mode=unknown, oob_mode=error)` transposes via DMA.

Supports axes (1,0), (2,1,0), (3,1,2,0). hwdge mode requires src.shape[0]==16, src.shape[-1]%128==0, and 2-byte dtype.

### tensor_copy

`nisa.tensor_copy(dst: tile, src: tile, engine=unknown, dtype=src.dtype)` copies within NeuronCore SBUF/PSUM.

Can use Vector, Scalar (v3+), or GpSimd engines. GpSimd cannot access PSUM. Faster than dma_copy for SBUF-to-SBUF or PSUM-to-SBUF movement.

## Compute: Matrix Multiplication

### nc_matmul

`nisa.nc_matmul(dst: tile[PSUM], stationary: tile[SBUF], moving: tile[SBUF], is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), perf_mode=none, dtype=dst.dtype)` multiplies stationary.T @ moving on Tensor Engine.

Stationary: [K, M] (K<=128 partition, M<=128 free); Moving: [K, N] (K<=128 partition, N<=512 free); Output: [M, N] in PSUM.

Key performance tips:
- Make K as close to 128 as possible.
- For large K, accumulate multiple nc_matmul outputs into the same PSUM tile.
- Prefer to put large matrix as stationary (Fast LoadStationary benefit).
- Use float8/bfloat16/float16/tfloat32 over float32 for ~4x better throughput.

Examples:
```python
a = nl.ndarray((128, 128), dtype=a_tensor.dtype, buffer=nl.sbuf)
b = nl.ndarray((128, 512), dtype=b_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor)
nisa.dma_copy(dst=b, src=b_tensor)
c_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
nisa.nc_matmul(dst=c_psum, stationary=a, moving=b)
nisa.dma_copy(dst=c_tensor, src=c_psum)
```

### nc_transpose

`nisa.nc_transpose(dst: tile, data: tile, dtype=data.dtype, engine=unknown)` transposes partition and free axes using Tensor or Vector Engine.

Tensor Engine: up to (128, 128); Vector Engine: up to (32, 32). If not specified, compiler chooses engine. Output goes to PSUM (Tensor) or SBUF/PSUM (Vector).

## Compute: Elementwise and Activation (Scalar Engine)

### activation

`nisa.activation(dst: tile, op: activation_fn, data: tile, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=idle, dtype=data.dtype)` applies activation with optional multiply/add and reduction on Scalar Engine.

Supports exp, log, sqrt, tanh, sigmoid, relu, silu, gelu, etc. Can multiply by scalar or vector scale and add vector bias before activation. Optional reduction after activation without extra cost.

Examples:
```python
a = nl.ndarray((128, 1024), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor)
activated_a = nl.ndarray((128, 1024), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.activation(dst=activated_a, op=nl.exp, data=a)
nisa.dma_copy(dst=a_act_tensor, src=activated_a)
```

### tensor_scalar

`nisa.tensor_scalar(dst: tile, data: tile, op0: op, operand0: scalar|tile[vec], reverse0=False, op1=None, operand1=None, reverse1=False, dtype=data.dtype, engine=unknown)` applies one or two math operators (data op0 operand0) op1 operand1 on Vector/Scalar/GpSimd Engine.

operand0 and operand1 are broadcast in free dimension. Two operators have same cost as one. Scalar Engine on Trainium2 only supports specific combos: (mul,add), (mul,none), (add,none).

## Compute: Reductions (Vector Engine)

### tensor_reduce

`nisa.tensor_reduce(dst: tile, op: reduce_op, data: tile, axis: int|tuple, negate=False, keepdims=False, dtype=data.dtype)` reduces along free dimensions.

axis must start at 1 and be consecutive: [1], [1,2], [1,2,3], [1,2,3,4]. Supports add, max, min, multiply, bitwise_and, bitwise_or, etc. Result has same partition size as input.

Examples:
```python
b = nl.ndarray((128, 1), dtype=a.dtype, buffer=nl.sbuf)
nisa.tensor_reduce(dst=b, op=np.add, data=a[0:128, 0:512], axis=[1])
```

### tensor_tensor

`nisa.tensor_tensor(dst: tile, data1: tile, data2: tile, op: binary_op, dtype=None, engine=unknown)` element-wise binary operation on Vector or GpSimd Engine.

Both input tiles must have same partition and free sizes. Supports arithmetic and bitwise ops. GpSimd cannot access PSUM; cannot use both PSUM inputs.

## Compute: Batch Normalization (Vector Engine)

### bn_stats

`nisa.bn_stats(dst: tile, data: tile, dtype=data.dtype)` computes mean/variance statistics per partition.

Output is 6 elements per partition (count_even, mean_even, var*count_even, count_odd, mean_odd, var*count_odd). Input free dim must be <=512. Pass to bn_aggr for final mean/variance.

### bn_aggr

`nisa.bn_aggr(dst: tile, data: tile, dtype=data.dtype)` aggregates bn_stats outputs to mean and variance.

Input is bn_stats output (multiple of 3 elements per partition); output is 2 elements per partition (mean, variance).

## Compute: Specialized Operations

### memset

`nisa.memset(dst: tile[SBUF], value: scalar, engine=unknown)` fills tile with constant value.

### iota

`nisa.iota(dst: tile[SBUF], pattern: list, offset: int32, channel_multiplier: int32=0, dtype=dst.dtype)` generates affine index pattern on GpSimd Engine.

For each channel, w, z, y, x: value = offset + (channel_id * channel_multiplier) + (w*step_w + z*step_z + y*step_y + x*step_x).

Pattern is [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]. Fewer than 4D patterns are auto-padded with size 1.

### affine_select

`nisa.affine_select(dst: tile[SBUF], pattern: list, offset: int32, channel_multiplier: int32, on_true_tile: tile[SBUF], on_false_value: scalar, cmp_op: cmp_op, dtype=on_true_tile.dtype)` selects based on affine predicate on GpSimd Engine.

For each element, computes affine_value = offset + (channel_id * channel_multiplier) + (...); if cmp_op(affine_value, 0) then on_true_tile else on_false_value.

Common use: causal mask in attention.

### dropout

`nisa.dropout(dst: tile, data: tile, prob: scalar|tile[vector], dtype=data.dtype)` randomly replaces elements with zero on Vector Engine.

prob is drop probability: 1.0 drops all, 0.0 keeps all. Can be scalar or tile (shape[0], 1).

## Compute: Advanced (Vector Engine)

### reciprocal

`nisa.reciprocal(dst: tile, data: tile, dtype=data.dtype)` computes element-wise 1.0/x.

### local_gather

`nisa.local_gather(dst: tile[SBUF], src_buffer: tile[SBUF], index: tile[SBUF], num_elem_per_idx: int=1, num_valid_indices=None, dtype=src_buffer.dtype)` gathers from SBUF using indices on GpSimd Engine.

Each of 8 GpSimd cores connects to 16 partitions and gathers independently. Indices are uint16. num_elem_per_idx: 1, 2, 4, 8, 16, 32 elements per index.

### max8

`nisa.max8(dst: tile[par_dim, 8], src: tile, dtype=src.dtype)` finds 8 largest values per partition on Vector Engine.

Input can be up to 5D; elements per partition must be 8–16,384. Output always 2D: [par_dim, 8] in descending order.

## Architecture

The Trainium NeuronCore has:
- **SBUF**: 192 KB, 128 partitions, local memory for loads and temporary storage
- **PSUM**: 384 KB, for Tensor Engine matmul accumulation
- **Tensor Engine**: Optimized for nc_matmul, 128x128 max input tiles
- **Vector Engine**: Element-wise ops, reductions, data movement within chip
- **Scalar Engine**: Activation functions, multiply-add-activate pipeline
- **GpSimd Engine**: Index generation, specialized gather/scatter, affine patterns

Key layout rules:
- Partition dimension (P) ≤ 128, free dimension (F) ≤ 512 in PSUM or unbounded in SBUF
- For nc_matmul: contraction axis K must be P dimension
- Tiles must have P dimension first; non-matmul ops require parallel axis as P
- Partition dimension maps to 128 SBUF partitions

Data flow: HBM → SBUF (dma_copy) → Compute → SBUF/PSUM → HBM (dma_copy)
