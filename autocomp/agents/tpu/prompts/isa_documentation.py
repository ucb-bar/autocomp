def get_pallas_isa_documentation() -> str:
    return """Pallas is JAX's kernel programming interface for TPUs. It allows writing custom kernels that run directly on TPU hardware.

Key concepts:
- VMEM (Vector Memory): On-chip memory for storing data during kernel execution
- HBM (High Bandwidth Memory): Off-chip memory accessed via loads/stores
- Kernel structure: Each Pallas kernel function receives memory references (x_ref, y_ref, o_ref) and operates on them
- Use pl.pallas_call() to wrap kernel functions and make them callable from JAX

Basic kernel pattern:
```python
def my_kernel(x_ref, y_ref, o_ref):
    # Read from VMEM
    x = x_ref[...]
    y = y_ref[...]
    # Compute
    result = x + y  # or other operations
    # Write to VMEM
    o_ref[...] = result

@jax.jit
def pallas_function(x, y):
    return pl.pallas_call(
        my_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)
```

Memory operations:
- x_ref[...] reads entire tensor from VMEM
- o_ref[...] = value writes entire tensor to VMEM
- Can use slicing: x_ref[0:128, 0:64] for partial reads/writes

Important constraints:
- VMEM size is 16MB+ per core, which is large for on-chip memory
- Operations should be tiled to fit in VMEM
- Use jax.block_until_ready() after kernel calls to ensure completion before timing
- Kernel functions should be pure (no side effects except through refs)

Performance tips:
- Minimize HBM↔VMEM transfers
- Maximize data reuse in VMEM
- Use appropriate tile sizes for your operation
- Consider precision (float32 vs bfloat16) for better performance

Supported data types:
At the moment Pallas TPU supports the following data types:
- jnp.float32
- jnp.bfloat16
- jnp.int* (all precisions, except for jnp.int4)
- jnp.uint* (all precisions)
- jnp.bool_

BlockSpecs and grid iteration
BlockSpecs (see BlockSpec, a.k.a. how to chunk up inputs) generally behave as expected in Pallas — every invocation of the kernel body gets access to slices of the inputs and is meant to initialize a slice of the output.

Note

Not all block shapes are supported. On TPU, only blocks with rank at least 1
are supported. Furthermore, the last two dimensions of your block shape must be divisible by 8 and 128 respectively, or be equal to the respective dimensions of the overall array.

One interesting aspect of Pallas TPU kernels is the way they handle memory spaces: While the inputs to pallas_call will often reside in HBM (the main TPU memory), the references passed in to the kernel body will point to buffers in lower levels of memory hierarchy (VMEM or SMEM). This enables the kernel body to write and read them at very high speeds, while all the communication with HBM (which has very high latency) is handled by the compiler and overlapped with compute.

What’s more, compared to GPUs, TPUs are actually highly sequential machines. Ergo, the grid is generally not processed in parallel, but sequentially, in lexicographic order (though see the Multicore TPU configurations section for exceptions). This unlocks some interesting capabilities:

When two (lexicographically) consecutive grid indices use the same slice of an input, the HBM transfer for the second iteration is skipped, as the data is already available.

Multiple invocations of the kernel body can write to the same slice of the output, without any risk of race conditions. However, we do require that all invocations that write to a particular slice are consecutive.

The “consecutive” restriction on the output usually means that some prefix of the grid dimensions always varies the slice of the output an invocation needs to access, while the output window remains constant for the remaining suffix.

For example, when implementing a Pallas TPU kernel for matrix multiplication, one would generally use a 3 dimensional grid: the first two dimensions would correspond to slicing along the first axis of the left operand and the second axis of the second operand. The third and last grid axis would tile the reduction dimension. The grid axis corresponding to the reduction dimension has to be the last one, since the output window does not vary along this axis. The output reference can be then used as an accumulator for partial results.

Note

VMEM is fairly large for such a low-level memory hierarchy (16MB+), making it possible to use large window sizes. And, oftentimes, the larger the window size, the better the eventual hardware utilization will be. However, it is possible to specify a window size that (together with space necessary to hold spilled vector registers) exceeds the size of VMEM. In this case, you will likely see a low-level compiler error message complaining about an out-of-memory error.

Array Layouts
Dimension ordering of arrays is meaningful in Pallas. In JAX programs, the ordering of intermediate arrays inside jax.jit usually has no impact on performance, as the compiler is free to rearrange them. However, as Pallas is meant to expose lower-level capabilities, the dimension order can have great impact on the quality of generated code.

TPUs perform the bulk of the computation on 2D vector registers, which are typically of size 8x128 for 32-bit values (as of TPU v6). When a vector value is loaded from VMEM into registers (e.g. x = x_ref[...]), the last two dimensions of the array will be tiled into the registers. Pallas will only ever consider mapping the last two dimensions of intermediate arrays to the 8x128 vector register dimensions (sublanes and lanes respectively).

Here is a graphical example of how a 12x320 array can be tiled using 6 8x128 tiles:

../../_images/vector_layout_example.svg
Tiled layouts have several import ramifications for kernel writers:

The last two axes of an array are treated differently than other axes. For example, reductions, reshapes, and transposes are generally more expensive when involving the last two axes. Some reshapes involving the last two dimensions are not supported and will result in a compiler error, but are “free” and performed at compile time for other dimensions.

While sometimes unavoidable, it is generally wasteful to have singleton dimensions in the last two axes, since they will occupy 1 element out of the entire tile dimension. Consuming too many registers can also potentially cause register spills into VMEM which degrades kernel performance.

Related to the above point, all vector computation is padded up to the tile size. Adding a two 1x1 arrays costs as much as adding two 8x128 arrays, and adding two 8x128x1x1 arrays will be 1024 times as expensive as adding two 8x128 arrays, since the 8x128x1x1 array will be padded to 8x128x8x128.

Multicore TPU configurations
In newer TPU generations, the two cores on a chip are often abstracted as a single device. To take advantage of multiple cores, Pallas has to break the sequential grid execution guarantees, and will need to parallelize one of the grid axes over cores. This is an opt-in procedure. To allow that, pallas_call requires an extra parameter named dimension_semantics:

```python
pallas_call(
    ...,
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=["parallel", "parallel", "arbitrary"]
    ),
  )
```

That parameter is a list, with as many entries as many axes there are in the grid. Only parallel dimensions can be partitioned over cores. As a rule of thumb, the dimensions are parallel, unless the output window does not vary. As such, dimension_semantics is always a number of parallel axes followed by a number of arbitrary axes.

While partitioning a kernel over a 2-core TPU device often leads to a 2x speedup, it can be in fact significantly smaller. This is especially true if different instances of the body have highly varying cost. If all of the expensive steps get mapped to one core, but all cheap steps are assigned to the other, the second core will be sitting idle until the first one completes its tasks.

Pallas TPU generally favors partitioning axes of a size that is a multiple of the number of TPU cores, and prefers to partition leading grid axes.

Placing operands in SMEM
Most of the compute on the TPU will happen on the vector unit. Still, there are many cases where it is useful to perform a number of scalar operations, e.g., to carry out control-flow. For that reason, TPUs come with a separate scalar unit, and a separate scalar memory (SMEM) attached to it. As a rule of thumb, any data used to perform control-flow decisions should be placed in SMEM.

SMEM is a low-latency memory that supports random access, but lets you only read and write 32-bit values with a single instruction (very small compared to the 4KBi granularity of VMEM transactions, but much more flexible due to lack of alignment requirements!).

The scalar memory is also very useful when implementing kernels that do not access the tiles of inputs in a regular pattern, such as when writing block-sparse kernels. In Pallas, this can be achieved by replacing the grid argument to pallas_call with a grid_spec of PrefetchScalarGridSpec with a non-zero num_scalar_prefetch argument. If num_scalar_prefetch is n, then the first n arguments to pallas_call will be placed in SMEM. No BlockSpecs should be specified for those arguments. But, the BlockSpecs for all subsequent arguments will receive not only the grid indices, but also the SMEM references to the leading operands.

Matrix multiplication
Matrix multiplication always produces results in the float32 format. If your inputs are not float32, we recommend using lax.dot with preferred_element_type set to jnp.float32.

When using lax.dot_general, it is possible to fuse transpositions of the last two dimensions of matrix multiplication operands into the operation, which can improve overall kernel performance.

Precision control
Pallas TPU lowering is aware of jax.default_matmul_precision. For best performance (and lowest precision), use bfloat16. If you care about numerical accuracy, you might want to set the precision to float32.

Warning

Even if you pass in 32-bit operands to a matrix multiplication, they will be rounded to bfloat16 unless float32 precision is requested.

Matrix multiplication block size tuning
For matmul kernels using pallas_call with a 3D grid (m_tiles, n_tiles, k_tiles), the block sizes bm, bk, bn critically determine performance. Matmul FLOPs are cubic (2*m*k*n) while memory bandwidth is quadratic, so bigger blocks push the kernel toward being compute-bound rather than memory-bound.

The block sizes bm, bk, bn control how much work is done per pipeline iteration:
- bk (reduction dimension block): should be as large as possible. Larger bk means fewer sequential reduction steps (k // bk iterations). For a 4096x4096 matmul, bk=1024 means only 4 reduction steps vs 32 with bk=128.
- bm and bn (output tile dimensions): determine MXU utilization and output tile size. Larger tiles amortize pipeline overhead.
- VMEM budget: each iteration needs bm*bk + bk*bn elements for inputs plus a bm*bn accumulator. All must fit in VMEM (~16MB). For bf16 inputs: (bm*bk + bk*bn) * 2 bytes + bm*bn * 4 bytes (fp32 accumulator).
- Asymmetric block sizes often outperform symmetric ones. For example, bm=512, bk=1024, bn=1024 achieves 78-91% MXU utilization vs 3-6% with bm=bk=bn=128 on TPU v5e.
- When tuning, vary bm, bk, bn independently. Try the largest bk that divides k first, then maximize bn, then bm within the VMEM budget.
- Two constraints limit block sizes: (1) VMEM capacity -- too-large blocks cause out-of-memory errors, and (2) pipeline bubbles -- too-large blocks relative to matrix size means fewer pipeline iterations, increasing bubble overhead.

Recommended bf16 matmul pattern using PrefetchScalarGridSpec
PrefetchScalarGridSpec with num_scalar_prefetch=0 enables automatic HBM-to-VMEM pipelining (prefetches next tiles while current tiles are being computed). This is simpler and more reliable than emit_pipeline for standard matmul. The scratch_shapes parameter allocates a persistent VMEM accumulator buffer. Here is the recommended high-performance matmul pattern:

```python
import functools

def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)

@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(x, y, *, bm=512, bk=1024, bn=1024):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

Key elements of this pattern:
- grid_spec=pltpu.PrefetchScalarGridSpec enables automatic HBM-VMEM pipelining.
- scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)] provides a persistent VMEM accumulator that avoids repeated HBM read-modify-write on the output.
- The kernel initializes the accumulator to zeros on the first K step (pl.program_id(2) == 0) and writes back to the output on the last K step (pl.program_id(2) == nsteps - 1).
- jnp.dot with preferred_element_type=jnp.float32 performs bf16 compute with fp32 accumulation.
- dimension_semantics=("parallel", "parallel", "arbitrary") marks the M and N grid axes as parallelizable and the K reduction axis as arbitrary (sequential).

Transposition
If the value has at least 4 dimensions, arbitrary transpositions of all but the last two axes are free. Otherwise, only the transposition of the last two axes is implemented. Note that some transpositions of the last two dimensions can be fused into matrix multiplication.

Fused RHS transpose in matmul: The MXU supports computing x @ y.T natively. Use jax.lax.dot_general to invoke this:
- Normal matmul: dims = ((1,), (0,)), ((), ())
- Fused RHS transpose: dims = ((1,), (1,)), ((), ())
This fuses the transpose into the MXU operation at zero additional cost.

Fused activation in matmul: An activation function (e.g. jax.nn.relu) can be fused into the matmul kernel by applying it on the last reduction step when writing back the accumulator:
  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)
This avoids a separate memory-bound activation kernel after the compute-bound matmul.

Accessing memory
Arbitrary slices of references can be read or updated, subject to implementation constraints. Currently, no restrictions are placed on inputs that are 32-bit wide, but only some slicing patterns are supported for narrower types. Reads and writes that are aligned to multiples of, and have a length that is a multiple of 8 and 128 respectively in the last two dimensions are always supported.

Reads and writes to vector memory generally happen on tiles of shape (8, 128). As such, when reading or writing to references that have at least two dimensions, the best performance is achieved when the base offset of the memory access has indices divisible by the tiling, and the size of the read region is a multiple of the tile size.

Elementwise operations
Many elementwise operations are supported. It is worth noting that the hardware generally only supports elementwise computation using 32-bit types. When loading operands that use lower-precision types, they should generally be upcast to a 32-bit type before applying elementwise ops.

It is worth noting that they can vary significantly in their cost. As such, we outline three categories of supported operations: cheap (🟢), medium (🌕) and expensive (🔴).

| Operation              | Cost |
|------------------------|------|
| jnp.add, +             | 🟢   |
| jnp.sub, -             | 🟢   |
| jnp.mul, *             | 🟢   |
| /, //, %               | 🌕   |
| jnp.max, jnp.min       | 🟢   |
| jnp.where (select)     | 🟢   |
| jnp.abs                | 🟢   |
| |, ^, &, ~             | 🟢   |
| <<, >>                 | 🟢   |
| Comparisons (==, …)    | 🟢   |
| Type casts (.astype)   | 🟢   |
| jnp.exp                | 🌕   |
| jnp.tanh               | 🌕   |
| jnp.pow                | 🌕   |
| jnp.sin                | 🔴   |
| jnp.cos                | 🔴   |


Many JAX functions are implemented in terms of other JAX primitives, so this list might not be comprehensive. For example, jax.nn.relu is implemented in terms of comparisons and jnp.where will work in Pallas kernels too.

Array constructors
All constant array constructors are supported (jnp.ones, jnp.zeros, jnp.full).

Reductions
sum, max, min (for floating point values) reductions are supported, as well as any and all for boolean values. Integer reductions are not supported.

Reductions over the last array dimension are generally the slowest. Reductions over the second last dimension are faster, but still slower than over the leading dimensions.

Reduction and accumulation in pipelined kernels
When accumulating partial results (e.g. in matmul), the reduction dimension must be the last (innermost) axis of the grid. This is because the Pallas pipeline reuses the same SRAM buffer when consecutive iterations map to the same output slice, enabling in-place accumulation. Once the output slice changes, the buffer is written back to HBM.
Rules for correct accumulation:
1. The output buffer starts with uninitialized (garbage) values. You must explicitly initialize it on the first reduction iteration using @pl.when(pl.program_id(reduction_axis) == 0).
2. Accumulate into the output (or a scratch buffer) across iterations of the reduction axis.
3. If using a scratch VMEM accumulator, write back the final result on the last reduction iteration using @pl.when(pl.program_id(reduction_axis) == nsteps - 1).

Broadcasting
The performance characteristics of broadcasting are very similar to those of reductions. Broadcasting along all but the two trailing dimensions is always supported and free. Broadcasting along the second to last dimension is slower, while broadcasting along the last dimension is the slowest.

Reshapes
As usual, reshapes in all dimensions but the last two dimensions are supported and free.

The only two supported cases when a reshape can modify the last two dimensions of an array is when (1) some leading dimensions are flattened onto the second to last dimension, or (2) it adds a dimension that was just removed by a reduction.

Random Number Generation
Pallas supports the most commonly used functions from the jax.random module, such as uniform, normal, and bernoulli. The key should be a threefry2x32 key, which is the default setting in JAX. Keys can be directly passed into a kernel, or generated inside of a kernel.

Control flow
The TPU backend features limited support for control flow at the moment. The currently supported functions are cond, fori_loop and for_loop. However, loop primitives get fully unrolled during the compilation at the moment, so try to keep the loop trip count reasonably small.

Overusing control flow can lead to significant regressions in low-level code generation, and it is recommended to try to squeeze as many computationally expensive operations into a single basic block as possible.

TPU Pipelining
This guide serves as a reference for TPU-specific pipelining concerns. We’ll review the memory hierarchy and compute units on TPUs, and TPU-specific features of the pipelining API. For a more general-purpose overview of pipelining, see the Software Pipelining.

```python
#@title Imports

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

TPU and its memory spaces
A TPU and its TensorCore consist of memory spaces (where arrays can reside), registers (which temporarily store scalar and array values) and compute units (that do computation with values in registers). Below is a diagram of a TPU in which x and y are arrays that live in high-bandwidth memory (HBM):

TPU Memory Space Cartoon.png

Let’s talk about the components of this diagram in more detail:

Memory spaces: A TPU has high-bandwidth memory (HBM) which is what we often think of as “device memory”. There is also vector memory (VMEM), a cache meant for storing vector and array values, and scalar memory (SMEM), a cache designed to store scalar values.

Registers: A TensorCore has two main types of registers: vector registers (VREGs) store array values, and scalar registers (SREGs) store scalar values. Values can be loaded into memory from their respective caches (VMEM for VREGs and SMEM for SREGs).

Compute units: A TensorCore has a scalar unit, vector unit (VPU) and matrix unit (MXU) that can do numerical computation. Each of these compute units can operate asynchronously, but this is managed by the TPU compiler and thus from the programmer’s perspective a TPU program is single-threaded. Compute units operate on values that live in SREGs and VREGs and output values into those registers as well.

TPU-specific Pipelining Features
Pallas TPU supports the following platform-specific features.

TPU Memory Spaces
Pallas exposes all levels of the TPU memory hierarchy to users. The following table maps from Pallas TPU memory spaces to their standard memory types (DRAM/SRAM):

| Pallas Enum       | TPU Memory Space       | Type (DRAM/SRAM) |
|-------------------|------------------------|------------------|
| pl.ANY            | HBM (usually) or VMEM  | DRAM             |
| pltpu.VMEM        | VMEM                   | SRAM             |
| pltpu.SMEM        | SMEM                   | SRAM             |
| pltpu.SEMAPHORE   | Semaphore              | SRAM             |

MemorySpace.VMEM denotes vector SRAM. It is the default memory space if nothing is specified.

MemorySpace.SMEM denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

MemorySpace.ANY is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the ANY memory space cannot be dereferenced normally using array indexing syntax (e.g. x[...]). Instead, we must first copy the values into a VMEM or SMEM buffer using pltpu.sync_copy or pltpu.async_copy.

MemorySpace.SEMAPHORE is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The default behavior for pallas_call on TPU is that arguments to pallas_call are assumed to live in HBM, and inputs to the user kernel body are stored in VMEM.

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the memory_space argument on a BlockSpec. Note that pipelining is not allowed unless the memory_space is marked as VMEM. Memory spaces can also be used to specify scratch arguments to a kernel via the scratch_shapes argument on pallas_call. Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in VMEM, SMEM, or SEMAPHORE.

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer x_hbm_ref into a scratch VMEM buffer scratch_vmem_ref before using it for arithmetic and storing the result into an output VMEM buffer:

```python
def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
  pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
  out_vmem_ref[...] = scratch_vmem_ref[...] + 1

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
out = pl.pallas_call(hbm_vmem_kernel,
  in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
  out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
  scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),)
)(x)

np.testing.assert_allclose(out, x[0:1] + 1)
```

Multiple Buffering
Multiple buffering can be specified on a per-argument basis to the pipeline via the pipeline_mode option on pl.BlockSpec. To do so, pass a pl.Buffered object to pl.BlockSpec specifying the number of buffers to allocate for this particular argument:

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count)
)
```

The default buffer count is 2 for all inputs and outputs.

pltpu.emit_pipeline
pltpu.emit_pipeline is a pipelining API implemented in Pallas that allows you to construct pipelines inside of a kernel rather than only on kernel entry. This several use-cases over using pl.pallas_call, such as:

For constructing nested pipelines. For example, an outer pipeline that communicates between chips, and an inner pipeline that performs HBM-VMEM pipelining.

For using emit_pipeline specific features such as lookahead prefetch and dynamic block shapes (covered below).

pltpu.emit_pipeline follows a similar signature to pl.pallas_call and requires you to specify a body kernel, a grid, and block specs for inputs and outputs:

```python
def emit_pipeline(
    kernel: Callable,
    grid: tuple[int],
    in_specs: PyTree[BlockSpec] = None,
    out_specs: PyTree[BlockSpec] = None,
    dimension_semantics: tuple[GridDimensionSemantics] = None,
    core_axis: int | None = None,
) -> Callable:
  ... # Returns a custom pipeline given an inner kernel and BlockSpecs.
```

The dimension_semantics and core_axis arguments are used for partitioning the kernel grid over Megacore (see below).

Lookahead Prefetch
Lookahead prefetch is a pipelining feature where the pipeline will attempt to prefetch the next input block as soon as a buffering slot is available, rather than the iteration directly before it would be used. For example, if the kernel had a grid of (8,) and the block indices to fetch on each iteration were 0, 0, 0, 0, 1, 1, 1, 1, then lookahead prefetch will begin fetching both blocks 0 and 1 on iteration 0, whereas the standard pipeline schedule would fetch block 0 on iteration 0 but not begin fetching block 1 until iteration 3. There is a small amount of control flow overhead in performing lookahead so it is disabled by default.

Lookahead is primarily useful when there is a variable amount of compute work in each block, such as when some blocks contain skipped or a reduced amount of work. In these cases, there may not be enough compute work in the iteration immediately preceding the step when the block is needed to fully overlap with the memory transfer. Therefore, we would like to begin fetching blocks earlier in the pipeline.

Lookahead prefetch can be used in conjunction with multiple buffering and can likewise be enabled by passing pl.Buffered into the pipeline_mode argument:

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
)
```

Dynamic Block Shapes
pltpu.emit_pipeline supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with pl.BoundedSlice(max_size) rather than a static integer size, where max_size is the maximum size of the block. In addition, the corresponding index returned by index_map should be a dynamic slice constructed via pl.ds(start, size) where both start and size are element indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:

```python
pl.BlockSpec(
   block_shape=(pl.BoundedSlice(32), 256),
   index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
)
```

```python
# The following kernel copies `x` to the output in dynamic-sized chunks
# passed in via `slices`.

def dynamic_block_example_kernel(x_hbm, slices_hbm, o_hbm, slices_smem):
    pltpu.sync_copy(slices_hbm, slices_smem)  # Copy slices into SMEM.
    def pipeline_body(x_vmem, o_vmem):
        o_vmem[...] = x_vmem[...]
    def index_map(i):
        start = slices_smem[i, 0]
        size = slices_smem[i, 1] - slices_smem[i, 0]
        return (pl.ds(start, size), 0)
    block_spec = pl.BlockSpec(block_shape=(pl.BoundedSlice(8), 128),
                              index_map=index_map)
    pltpu.emit_pipeline(
        pipeline_body,
        grid=(slices.shape[0],),
        in_specs=[block_spec],
        out_specs=block_spec
    )(x_hbm, o_hbm)

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
slices = jnp.array([[0, 2], [2, 3], [3, 5], [5, 8]], dtype=jnp.int32)

hbm_block_spec = pl.BlockSpec(memory_space=pl.ANY)
out = pl.pallas_call(dynamic_block_example_kernel,
               in_specs=[hbm_block_spec, hbm_block_spec],
               out_specs=hbm_block_spec,
               out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
               scratch_shapes=(pltpu.SMEM(slices.shape, jnp.int32),)
              )(x, slices)

np.testing.assert_allclose(x, out)
```

TPUs in Megacore configuration
Some TPU chips have two TensorCores but appear as one device to JAX users. This is called “megacore”. The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs and compute units but share HBM.

TPU Memory Space Cartoon (Megacore).png

Conceptually, TPUs in Megacore behave like very simple GPUs, i.e. they have only two threads. How do we modify our kernels to utilize both TensorCores simultaneously?

The basic idea is that if we have embarrassingly parallel dimensions in our computation, we can split up those dimensions across the TensorCores. We can indicate which dimensions are parallelizable by providing an annotation to pallas_call called dimension_semantics.

```python
def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  z_vregs = x_vregs + y_vregs
  z_vmem_ref[:, :] = z_vregs

def add_matrices_pipelined_megacore(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(2,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )(x, y)

x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices_pipelined_megacore(x, y)
# Array([[2., 2., 2., ..., 2., 2., 2.],
#        ...
#        [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
```

dimension_semantics should be a tuple of same length as grid where each entry is either "parallel" or "arbitrary". "parallel" indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. "arbitrary" indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

By specifying dimension_semantics, we now execute the kernel simultaneously on each TensorCore. Pallas will handle splitting up the grid automatically.

Note that Megacore is only currently available on TPU v4 and TPU v5p. Supplying dimension_semantics annotations is a no-op on other platforms, but not specifying it will result in only one TensorCore being used (even if there are more than one available).

When using pltpu.emit_pipeline, core_axis should be passed into emit_pipeline. core_axis should be the index of a parallel grid axis to partition the grid on. For example, the following template can be used to partition the kernel over a leading parallel grid dimension:

```python
def kernel_body(...):
  def inner_pipeline_body(...):
    ...
  pltpu.emit_pipeline(inner_pipeline_body,
                      grid=(4, 4), 
                      core_axis=0,
                      dimension_semantics=("parallel", "sequential"))

pl.pallas_call(
      kernel_body,
      grid=(num_cores,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )
```

Information about Jax:

1. Functional Essentials
JAX operates on Array objects (immutable). You cannot do x[0] = 1.0.

| Feature | Syntax | Notes |
|---|---|---|
| Randomness | key = jax.random.PRNGKey(seed) | JAX is stateless; you must pass/split keys. |
| Splitting Keys | key, subkey = jax.random.split(key) | Use subkey for the op, key for the next split. |
| Math Ops | jnp.matmul(a, b), jnp.exp(x) | Mirrors NumPy, but returns DeviceArrays. |
| Conditionals | jax.lax.cond(pred, true_fn, false_fn, operand) | Avoid Python if/else inside JIT. |

2. The "Big Four" Transformations
These are the core tools for mapping code to hardware.

| Transformation | Syntax | Purpose |
|---|---|---|
| JIT | @jax.jit | Compiles Python to XLA (High-performance). |
| VMAP | jax.vmap(fn, in_axes=(0, None)) | Auto-vectorization (removes manual loops). |
| PMAP | jax.pmap(fn, axis_name='p') | Parallelizes across multiple TPU/GPU cores. |
| GRAD | jax.grad(loss_fn) | Automatic differentiation (backprop). |

3. High-Performance Looping (jax.lax)
Python for loops are unrolled at compile time, leading to slow compilation. Use these instead:

```python
# Scan: Carrying state through a loop (preferred for RNNs/Optimization)
# carry: the state that changes, x: the array to iterate over
final_state, outputs = jax.lax.scan(loop_body, init_carry, xs)

# Fori_loop: Simple index-based loop
# (lower_bound, upper_bound, body_fun, init_val)
result = jax.lax.fori_loop(0, 100, lambda i, val: val + i, 0)
```

4. Pallas & TPU Memory Syntax
This is the "dangerous" side of JAX where you manage memory explicitly.

Memory References
Inside a pallas_call, you don't use arrays; you use Refs.
- Read: data = x_ref[...] or slice = x_ref[i:i+8, :]
- Write: o_ref[...] = data
- In-place Add: o_ref[...] += data

Pallas Call Structure:

```python
pl.pallas_call(
    kernel_func,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    grid=(M // 128, N // 128), # How many blocks to launch
    in_specs=[
        pl.BlockSpec(lambda i, j: (i, 0), (128, K)), # Map grid to HBM
        pl.BlockSpec(lambda i, j: (0, j), (K, 128))
    ],
    out_specs=pl.BlockSpec(lambda i, j: (i, j), (128, 128))
)(x, y)
```

5. Performance Debugging
Use these to see what the compiler (and your AutoComp agent) is actually doing.
- Inspect IR: print(jax.make_jaxpr(your_func)(*args)) (See the "Jaxpr" graph).
- Force Execution: result.block_until_ready() (Required for accurate timing).
- Shape/Type: jax.eval_shape(fn, *args) (Dry-run to see output shapes without computing).

6. TPU-Specific Constraints
- Padding: Always try to keep dimensions as multiples of 128.
- Dtypes: Use jnp.bfloat16 for maximum speed on TPU Matrix Units (MXU).
- Scalar vs Vector: If you see jax.lax.convert_element_type or heavy scalar indexing in your Jaxpr, your code is likely falling back to the slow Scalar Unit.
"""
