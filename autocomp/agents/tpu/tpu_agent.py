import pathlib
import random
import re

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.backend.eval_backend import EvalBackend
from autocomp.hw_config.hardware_config import HardwareConfig


class TpuLLMAgent(LLMAgent):
    def __init__(self, model: str, hw_config: HardwareConfig | None = None, eval_backend: EvalBackend | None = None):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend

    def __repr__(self):
        return f"TpuLLMAgent({self.llm_client.model})"

    def _get_convert_to_pallas_menu_options(self) -> list[str]:
        return [
            "convert non-Pallas code into Pallas kernel code",
            "move non-Pallas code into a Pallas kernel",
            "fuse multiple Pallas kernels into a single kernel",
        ]

    def get_opt_menu_options(self, prob: Prob):
        """Get optimization menu options for Pallas/TPU kernels"""
        base = [
            "Fuse the reduction loop into a single kernel invocation so partial outputs stay in VMEM/registers and are written once at the end.",
            "Use smaller autotuned block sizes for M, N, and especially K to improve parallelism, pipelining, and hardware utilization.",
            "Use multi-level tiling so outer tiles maximize reuse and inner tiles match TPU vector/matmul execution better.",
            "Keep accumulator tiles resident in VMEM/registers across reduction steps instead of repeatedly reading and writing output tiles.",
            "Minimize HBM↔VMEM transfers by reusing staged tiles as much as possible before loading new ones.",
            "Pipeline data movement and compute so the next tiles are loaded while the current tiles are being processed.",
            "Use asynchronous copies instead of synchronous copies when possible to overlap memory traffic with compute.",
            "Use double buffering or deeper buffering for input tiles so one buffer can be filled while another is used for compute.",
            "Choose between compiler-managed staging and manual VMEM scratch staging based on which gives better overlap and lower overhead.",
            "Reorder loops so the stationary operand is reused longer and the moving operand is streamed efficiently.",
            "Hoist invariant loads, address calculations, and shape/layout logic out of inner loops.",
            "Improve VMEM layout and access patterns so tile loads/stores are contiguous and aligned with TPU-friendly access patterns.",
            "Preserve or transform operand layouts to avoid unnecessary transpose-like movement inside the kernel.",
            "Use larger contiguous compute blocks only when they improve reuse without causing excessive VMEM or register pressure.",
            "Use lower precision for inputs and temporary buffers when allowed, while accumulating in higher precision if needed.",
            "Reduce intermediate tensor materialization by fusing dependent operations into the same kernel.",
            "Use in-place accumulation/update patterns when safe to avoid unnecessary temporaries.",
            "Increase parallel work across output tiles so more compute resources stay busy.",
            "Select a different TPU primitive or matmul form if it maps better to the hardware than the current implementation.",
            "Target tile sizes, layouts, and scheduling choices to the exact input/output shapes being benchmarked.",
            "Simplify or remove unnecessary control flow, masking, and bookkeeping inside hot inner loops.",
            "Other TPU-specific optimizations not listed here."
        ]
        if prob is not None and getattr(prob, 'prob_type', None) == 'tpu' and getattr(prob, 'prob_id', None) == 0:
            base = [
                "Cast inputs to bfloat16 before the matmul while keeping the accumulator in float32. The MXU processes bf16 at 2x the throughput of fp32. Use jax.lax.dot with preferred_element_type=jnp.float32 inside the kernel, or cast x_ref/y_ref to bfloat16 before the @ operator.",
                "Use jax.lax.dot or jax.lax.dot_general instead of the @ operator inside the kernel to enable explicit transpose fusion and precision control via preferred_element_type.",
                "Use pltpu.emit_pipeline for the inner K-reduction loop inside an outer pallas_call that partitions M/N tiles across megacore TensorCores. The outer kernel receives full HBM refs (memory_space=pl.ANY) and emit_pipeline manages all HBM->VMEM pipelining automatically.",
                "Tune tile sizes (bm, bn, bk) jointly for the exact benchmark shapes. Maximize bm*bn for MXU reuse while ensuring m_tiles*n_tiles >= 2*num_cores for megacore utilization. Keep bk small (128-256) for pipelining depth.",
                "Use pl.Buffered with use_lookahead=True on input BlockSpecs so the compiler begins prefetching tiles as soon as a buffer slot is free, rather than waiting until the iteration before they are needed.",
                "On the first K iteration (kk==0) write the matmul result directly to z_ref instead of zeroing then accumulating. This eliminates a full-tile zero-store and avoids an unnecessary output-tile prefetch on the first iteration.",
                "Reorder the grid axes or swap which operand is stationary vs streaming to improve data reuse. For example, making the LHS (x) stationary across consecutive j iterations can reduce redundant HBM->VMEM transfers.",
                "Use scratch_shapes to allocate explicit VMEM scratch buffers for manual double-buffering or for staging intermediate results, giving more control than compiler-managed pipelining alone.",
                "Increase parallel work by ensuring the grid has enough output tiles for both TensorCores. If bm and bn are too large, the grid may have too few tiles to keep both cores busy.",
                "Simplify or remove unnecessary control flow, helper functions, assertions, and Python-level overhead inside the kernel and the pallas_call setup. Minimize compilation complexity.",
            ]
        return base

    def _get_pallas_isa_documentation(self, prob: Prob = None) -> str:
        """Generate Pallas/TPU ISA documentation for prompts"""
        doc = """Pallas is JAX's kernel programming interface for TPUs. It allows writing custom kernels that run directly on TPU hardware.

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
- VMEM size is limited (For reference, the VMEM of a TPU (for generations v4/v5) is typically on the order of 10-100MB, whereas HBM ranges from 10-100GB.)
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

pallas_call(
    ...,
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=["parallel", "parallel", "arbitrary"]
    ),
  )
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

Transposition
If the value has at least 4 dimensions, arbitrary transpositions of all but the last two axes are free. Otherwise, only the transposition of the last two axes is implemented. Note that some transpositions of the last two dimensions can be fused into matrix multiplication.

Accessing memory
Arbitrary slices of references can be read or updated, subject to implementation constraints. Currently, no restrictions are placed on inputs that are 32-bit wide, but only some slicing patterns are supported for narrower types. Reads and writes that are aligned to multiples of, and have a length that is a multiple of 8 and 128 respectively in the last two dimensions are always supported.

Reads and writes to vector memory generally happen on tiles of shape (8, 128). As such, when reading or writing to references that have at least two dimensions, the best performance is achieved when the base offset of the memory access has indices divisible by the tiling, and the size of the read region is a multiple of the tile size.

Elementwise operations
Many elementwise operations are supported. It is worth noting that the hardware generally only supports elementwise computation using 32-bit types. When loading operands that use lower-precision types, they should generally be upcast to a 32-bit type before applying elementwise ops.

It is worth noting that they can vary significantly in their cost. As such, we outline three categories of supported operations: cheap (🟢), medium (🌕) and expensive (🔴).

Operation
Cost:
jnp.add, +: 🟢
jnp.sub, -: 🟢
jnp.mul, *: 🟢
/, //, %: 🌕
jnp.max, jnp.min: 🟢
jnp.where (select): 🟢
jnp.abs: 🟢
|, ^, &, ~: 🟢
<<, >>: 🟢
Comparisons (==, …): 🟢
Type casts (.astype): 🟢
jnp.exp: 🌕     
jnp.tanh: 🌕
jnp.pow: 🌕
jnp.sin: 🔴
jnp.cos: 🔴 


Many JAX functions are implemented in terms of other JAX primitives, so this list might not be comprehensive. For example, jax.nn.relu is implemented in terms of comparisons and jnp.where will work in Pallas kernels too.
Array constructors
All constant array constructors are supported (jnp.ones, jnp.zeros, jnp.full).

Reductions
sum, max, min (for floating point values) reductions are supported, as well as any and all for boolean values. Integer reductions are not supported.

Reductions over the last array dimension are generally the slowest. Reductions over the second last dimension are faster, but still slower than over the leading dimensions.

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

#@title Imports

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
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

Pallas Enum

TPU Memory Space

Type (DRAM/SRAM)

pl.ANY

HBM (usually) or VMEM

DRAM

pltpu.VMEM

VMEM

SRAM

pltpu.SMEM

SMEM

SRAM

pltpu.SEMAPHORE

Semaphore

SRAM

MemorySpace.VMEM denotes vector SRAM. It is the default memory space if nothing is specified.

MemorySpace.SMEM denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

MemorySpace.ANY is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the ANY memory space cannot be dereferenced normally using array indexing syntax (e.g. x[...]). Instead, we must first copy the values into a VMEM or SMEM buffer using pltpu.sync_copy or pltpu.async_copy.

MemorySpace.SEMAPHORE is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The default behavior for pallas_call on TPU is that arguments to pallas_call are assumed to live in HBM, and inputs to the user kernel body are stored in VMEM.

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the memory_space argument on a BlockSpec. Note that pipelining is not allowed unless the memory_space is marked as VMEM. Memory spaces can also be used to specify scratch arguments to a kernel via the scratch_shapes argument on pallas_call. Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in VMEM, SMEM, or SEMAPHORE.

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer x_hbm_ref into a scratch VMEM buffer scratch_vmem_ref before using it for arithmetic and storing the result into an output VMEM buffer:

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
Multiple Buffering
Multiple buffering can be specified on a per-argument basis to the pipeline via the pipeline_mode option on pl.BlockSpec. To do so, pass a pl.Buffered object to pl.BlockSpec specifying the number of buffers to allocate for this particular argument:

pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count)
)
The default buffer count is 2 for all inputs and outputs.

pltpu.emit_pipeline
pltpu.emit_pipeline is a pipelining API implemented in Pallas that allows you to construct pipelines inside of a kernel rather than only on kernel entry. This several use-cases over using pl.pallas_call, such as:

For constructing nested pipelines. For example, an outer pipeline that communicates between chips, and an inner pipeline that performs HBM-VMEM pipelining.

For using emit_pipeline specific features such as lookahead prefetch and dynamic block shapes (covered below).

pltpu.emit_pipeline follows a similar signature to pl.pallas_call and requires you to specify a body kernel, a grid, and block specs for inputs and outputs:

def emit_pipeline(
    kernel: Callable,
    grid: tuple[int],
    in_specs: PyTree[BlockSpec] = None,
    out_specs: PyTree[BlockSpec] = None,
    dimension_semantics: tuple[GridDimensionSemantics] = None,
    core_axis: int | None = None,
) -> Callable:
  ... # Returns a custom pipeline given an inner kernel and BlockSpecs.
The dimension_semantics and core_axis arguments are used for partitioning the kernel grid over Megacore (see below).

jax.experimental.pallas.tpu.emit_pipeline
jax.experimental.pallas.tpu.emit_pipeline(body, *, grid, in_specs=(), out_specs=(), tiling=None, should_accumulate_out=False, core_axis=None, core_axis_name=None, dimension_semantics=None, trace_scopes=True, no_pipelining=False, _explicit_indices=False)[source]
Creates a function to emit a manual pallas pipeline.

This has the same semantics as pallas_call but is meant to be called inside pallas_call for nesting grids. This is useful when you need to have separate windowing strategies for communication and computation.

The new argument should_accumulate_out can be used to specify which outputs we should accumulate into automatically within and across pipeline invocations.

Parameters
:
body – pallas kernel to set up pipeline for.

grid (tuple[int | jax.Array, ...]) – a pallas grid definition.

in_specs – input pallas block specs

out_specs – output pallas block specs

tiling (tpu_info.Tiling | None) – optional tiling to assume for the refs.

should_accumulate_out (bool) – booleans to indicate which outputs should be treated as accumulators.

core_axis (tuple[int, ...] | int | None) – optional int or tuple of int, indicates whether or not to partition the grid along the core axis.

core_axis_name (tuple[str, ...] | str | None) – optional str or tuple of str, indicates whether or not to partition the grid along the core axis.

dimension_semantics (tuple[GridDimensionSemantics, ...] | None) – optional tuple of GridDimensionSemantics (e.g. PARALLEL or ARBITRARY).

trace_scopes (bool) – optional bool, indicates whether to annotate each region in the pipeline using named_scope.

no_pipelining (bool) – If True, turns off pipelining and all copies will be made synchronous. This is useful for debugging multiple-buffering related bugs.

_explicit_indices (bool) – If True, the body will receive the iteration indices as its first argument. This parameter is meant for internal use only.

"For matmul, emit_pipeline can be used for the inner K-reduction loop inside an outer pallas_call that handles M/N tiling and megacore parallelism. Example pattern:\n"
"  outer pallas_call with grid=(m_tiles, n_tiles) and dimension_semantics=('parallel','parallel')\n"
"  inner emit_pipeline with grid=(k_tiles,) handling HBM->VMEM pipelining for K reduction\n"


Lookahead Prefetch
Lookahead prefetch is a pipelining feature where the pipeline will attempt to prefetch the next input block as soon as a buffering slot is available, rather than the iteration directly before it would be used. For example, if the kernel had a grid of (8,) and the block indices to fetch on each iteration were 0, 0, 0, 0, 1, 1, 1, 1, then lookahead prefetch will begin fetching both blocks 0 and 1 on iteration 0, whereas the standard pipeline schedule would fetch block 0 on iteration 0 but not begin fetching block 1 until iteration 3. There is a small amount of control flow overhead in performing lookahead so it is disabled by default.

Lookahead is primarily useful when there is a variable amount of compute work in each block, such as when some blocks contain skipped or a reduced amount of work. In these cases, there may not be enough compute work in the iteration immediately preceding the step when the block is needed to fully overlap with the memory transfer. Therefore, we would like to begin fetching blocks earlier in the pipeline.

Lookahead prefetch can be used in conjunction with multiple buffering and can likewise be enabled by passing pl.Buffered into the pipeline_mode argument:

pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
)
Dynamic Block Shapes
pltpu.emit_pipeline supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with pl.BoundedSlice(max_size) rather than a static integer size, where max_size is the maximum size of the block. In addition, the corresponding index returned by index_map should be a dynamic slice constructed via pl.ds(start, size) where both start and size are element indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:

pl.BlockSpec(
   block_shape=(pl.BoundedSlice(32), 256),
   index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
)
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
TPUs in Megacore configuration
Some TPU chips have two TensorCores but appear as one device to JAX users. This is called “megacore”. The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs and compute units but share HBM.

TPU Memory Space Cartoon (Megacore).png

Conceptually, TPUs in Megacore behave like very simple GPUs, i.e. they have only two threads. How do we modify our kernels to utilize both TensorCores simultaneously?

The basic idea is that if we have embarrassingly parallel dimensions in our computation, we can split up those dimensions across the TensorCores. We can indicate which dimensions are parallelizable by providing an annotation to pallas_call called dimension_semantics.

def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
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
Array([[2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       ...,
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
dimension_semantics should be a tuple of same length as grid where each entry is either "parallel" or "arbitrary". "parallel" indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. "arbitrary" indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

By specifying dimension_semantics, we now execute the kernel simultaneously on each TensorCore. Pallas will handle splitting up the grid automatically.

Note that Megacore is only currently available on TPU v4 and TPU v5p. Supplying dimension_semantics annotations is a no-op on other platforms, but not specifying it will result in only one TensorCore being used (even if there are more than one available).

When using pltpu.emit_pipeline, core_axis should be passed into emit_pipeline. core_axis should be the index of a parallel grid axis to partition the grid on. For example, the following template can be used to partition the kernel over a leading parallel grid dimension:

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

Information about Jax:
1. Functional EssentialsJAX operates on Array objects (immutable). You cannot do x[0] = 1.0.FeatureSyntaxNotesRandomnesskey = jax.random.PRNGKey(seed)JAX is stateless; you must pass/split keys.Splitting Keyskey, subkey = jax.random.split(key)Use subkey for the op, key for the next split.Math Opsjnp.matmul(a, b), jnp.exp(x)Mirrors NumPy, but returns DeviceArrays.Conditionalsjax.lax.cond(pred, true_fn, false_fn, operand)Avoid Python if/else inside JIT.2. The "Big Four" TransformationsThese are the core tools for mapping code to hardware.TransformationSyntaxPurposeJIT@jax.jitCompiles Python to XLA (High-performance).VMAPjax.vmap(fn, in_axes=(0, None))Auto-vectorization (removes manual loops).PMAPjax.pmap(fn, axis_name='p')Parallelizes across multiple TPU/GPU cores.GRADjax.grad(loss_fn)Automatic differentiation (backprop).3. High-Performance Looping (jax.lax)Python for loops are unrolled at compile time, leading to slow compilation. Use these instead:Python# Scan: Carrying state through a loop (preferred for RNNs/Optimization)
# carry: the state that changes, x: the array to iterate over
final_state, outputs = jax.lax.scan(loop_body, init_carry, xs)

# Fori_loop: Simple index-based loop
# (lower_bound, upper_bound, body_fun, init_val)
result = jax.lax.fori_loop(0, 100, lambda i, val: val + i, 0)
4. Pallas & TPU Memory SyntaxThis is the "dangerous" side of JAX where you manage memory explicitly.Memory ReferencesInside a pallas_call, you don't use arrays; you use Refs.Read: data = x_ref[...] or slice = x_ref[i:i+8, :]Write: o_ref[...] = dataIn-place Add: o_ref[...] += dataPallas Call StructurePythonpl.pallas_call(
    kernel_func,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    grid=(M // 128, N // 128), # How many blocks to launch
    in_specs=[
        pl.BlockSpec(lambda i, j: (i, 0), (128, K)), # Map grid to HBM
        pl.BlockSpec(lambda i, j: (0, j), (K, 128))
    ],
    out_specs=pl.BlockSpec(lambda i, j: (i, j), (128, 128))
)(x, y)
5. Performance DebuggingUse these to see what the compiler (and your AutoComp agent) is actually doing.Inspect IR: print(jax.make_jaxpr(your_func)(*args)) (See the "Jaxpr" graph).Force Execution: result.block_until_ready() (Required for accurate timing).Shape/Type: jax.eval_shape(fn, *args) (Dry-run to see output shapes without computing).6. TPU-Specific ConstraintsPadding: Always try to keep dimensions as multiples of 128.Dtypes: Use jnp.bfloat16 for maximum speed on TPU Matrix Units (MXU).Scalar vs Vector: If you see jax.lax.convert_element_type or heavy scalar indexing in your Jaxpr, your code is likely falling back to the slow Scalar Unit.
"""
        if prob is not None and getattr(prob, 'prob_type', None) == 'tpu' and getattr(prob, 'prob_id', None) == 0:
            doc += """

High-performance pipelined matmul on TPU
=========================================
To match or beat the XLA matmul kernel (jnp.matmul), a Pallas matmul MUST overlap HBM->VMEM data movement with MXU compute. A single-tile or full-K approach with synchronous copies leaves the MXU idle during transfers and achieves very low utilization.

Why pipelining matters for matmul:
- The MXU can compute a tile matmul in far less time than it takes to load the next tiles from HBM.
- Without pipelining, the kernel alternates between "loading" and "computing" phases with the MXU idle during loads.
- With pipelining (double/triple buffering), the DMA engine loads the NEXT tile while the MXU computes the CURRENT tile, keeping both busy simultaneously.
- This is exactly what XLA's optimized matmul does internally, and what a Pallas kernel must replicate.

Approach 1: Standard 3D grid with compiler-managed pipelining (recommended starting point)
Use grid=(m_tiles, n_tiles, k_tiles) with K as the LAST axis. The Pallas compiler automatically double-buffers input tiles: while the MXU computes x[i,kk] @ y[kk,j], the DMA prefetches x[i,kk+1] and y[kk+1,j] into a second VMEM buffer. This gives pipelining for free.

```python
def matmul_kernel(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)
    z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(x, y, bm=128, bk=128, bn=128):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        grid=(m // bm, n // bn, k // bk),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)
```
Key: bk should be SMALL (128-256) so that k_tiles > 1, giving the compiler multiple iterations to pipeline over. Using bk=K (full-K, single iteration) disables pipelining entirely.

Deeper buffering with pl.Buffered:
By default the compiler uses 2 buffers (double buffering). For matmul with small K tiles, triple buffering can hide more latency:
```python
pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk),
             pipeline_mode=pl.Buffered(buffer_count=3))
```

Approach 2: emit_pipeline with megacore (highest performance)
For maximum performance, use an outer pallas_call that partitions over cores, and an inner emit_pipeline that handles tiling and pipelining. The outer kernel receives full HBM refs (memory_space=pl.ANY), and emit_pipeline manages all HBM->VMEM transfers with automatic pipelining.

```python
def matmul_kernel_body(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)
    z_ref[...] += x_ref[...] @ y_ref[...]

def outer_kernel(x_hbm_ref, y_hbm_ref, z_hbm_ref):
    pltpu.emit_pipeline(
        matmul_kernel_body,
        grid=(m_tiles, n_tiles, k_tiles),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        core_axis=0,
        dimension_semantics=("parallel", "parallel", "arbitrary"),
    )(x_hbm_ref, y_hbm_ref, z_hbm_ref)

result = pl.pallas_call(
    outer_kernel,
    out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pl.ANY)],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    grid=(num_cores,),
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=("parallel",)),
)(x, y)
```

Tile size guidance for matmul:
- bk (K tile): Use 128 or 256. This is the most important parameter. Small bk = more K iterations = more pipelining opportunities. bk=K (full reduction in one step) means NO pipelining at all.
- bm, bn (output tile): Use 128-512. Larger tiles improve MXU reuse but reduce grid parallelism. Balance so that m_tiles * n_tiles >= 2 * num_cores for megacore utilization.
- Grid ordering: The K axis MUST be last (it is the reduction/"arbitrary" axis). For the M and N axes, order them so the operand that benefits most from reuse is stationary across consecutive iterations.
- All tile dimensions in the last two axes must be divisible by 8 and 128 respectively.

Common anti-patterns that prevent matching XLA matmul performance:
- Using bk=K (full K dimension) which creates only 1 K iteration and disables pipelining.
- Using bm=M, bn=N (full output) which creates only 1 output tile and prevents megacore parallelism.
- Using manual sync_copy in a fori_loop instead of letting BlockSpec + compiler handle pipelining automatically.
- Not setting dimension_semantics, which leaves the second TensorCore idle.
"""
        return doc

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        is_matmul = (prob is not None
                     and getattr(prob, 'prob_type', None) == 'tpu'
                     and getattr(prob, 'prob_id', None) == 0)

        rules = [
            "The rewritten program must remain semantically equivalent to the original program, within a small numerical tolerance.",
            "Maintain correct tensor shapes, dtypes, indexing patterns, and output layout.",
            "The following imports have already been run: import jax; import jax.numpy as jnp; from jax.experimental import pallas as pl; from jax.experimental.pallas import tpu as pltpu; import numpy as np;",
            "Use pl.pallas_call() to wrap TPU kernel functions and make them callable from JAX.",
            "Kernel functions should receive memory references (x_ref, y_ref, o_ref, etc.) and operate on them directly.",
            "Use jax.block_until_ready() on the final result before timing or returning from the benchmark harness.",
        ]

        if is_matmul:
            rules += [
                "For matmul, the most important optimization is overlapping HBM->VMEM data movement with MXU compute via pipelining. Use a 3D grid (m_tiles, n_tiles, k_tiles) with SMALL K tiles (bk=128 or 256) so the Pallas compiler can automatically double-buffer input tiles across K iterations.",
                "With the standard 3D grid approach, the output tile (z_ref) is kept in VMEM by the compiler across all K iterations for a given (i,j). The pattern z_ref[...] += x_ref[...] @ y_ref[...] does NOT cause HBM read-modify-write; it is VMEM-resident accumulation. This is safe and efficient.",
                "Do NOT use bk=K (full K dimension in one tile) because it creates only 1 K iteration and completely disables compiler pipelining. Small bk with many K iterations is essential.",
                "Do NOT use bm=M and bn=N (full output in one tile) because it creates only 1 output tile and prevents megacore parallelism. Use bm=128-512 and bn=128-512.",
                "Do NOT use manual pltpu.sync_copy in a fori_loop for the K reduction. Let BlockSpec + the Pallas compiler handle all HBM->VMEM data movement and pipelining automatically.",
                "Use dimension_semantics=('parallel', 'parallel', 'arbitrary') so both TensorCores are utilized. The K axis must be 'arbitrary' (last).",
                "Consider pltpu.emit_pipeline for the inner tiling loop with an outer pallas_call that partitions over megacore TensorCores for maximum performance.",
                "Consider pl.Buffered(buffer_count=3) on input BlockSpecs for deeper prefetching beyond the default double buffering.",
                "Prefer hardware-friendly tile sizes: last two dims divisible by 8 and 128 respectively. Tune bm, bn, bk for the specific input shapes.",
            ]
        else:
            rules += [
                "Prefer kernel structures that reduce HBM↔VMEM traffic, especially by keeping accumulator tiles resident and writing outputs once when possible.",
                "Prefer tiling strategies that create enough tile-level parallelism and enough K-tiles to support reuse and pipelining; do not default to a single giant tile unless it is clearly best.",
                "When optimizing contractions or matmuls, strongly consider fusing the reduction loop inside one kernel invocation so partial outputs are not repeatedly read and written.",
                "Prefer hardware-friendly tile sizes for M, N, and especially K; treat tile sizes as tunable rather than fixed.",
                "Prefer smaller K tiles when they enable better overlap of memory movement and compute.",
                "Consider both compiler-managed staging via BlockSpec/memory spaces and manual staging into VMEM scratch buffers; choose the simpler option unless manual staging clearly improves performance.",
                "When staging tiles manually, prefer asynchronous copies and pipelined execution over purely synchronous copy-then-compute structure.",
                "Consider double buffering when there is repeated tile loading and enough work per step to hide memory latency.",
                "Do not introduce extra output-tile read-modify-write traffic across reduction steps unless there is a clear reason.",
            ]

        rules += [
            "Prefer loop orders and grid orders that maximize reuse of stationary operands across adjacent iterations.",
            "Preserve or improve operand layout and access patterns so reads/writes are contiguous, aligned, and TPU-friendly.",
            "Avoid unnecessary transpose-like data movement or layout conversions inside hot loops.",
            "Use lower precision inputs or temporary buffers such as bfloat16 when allowed by the problem, while preserving required output accuracy.",
            "Hoist invariant computations, indexing math, and shape-dependent logic out of hot inner loops whenever possible.",
            "Avoid rematerializing intermediate tensors or allocating unnecessary temporaries inside inner loops.",
            "Do not make optimization changes that only affect code style or boilerplate without improving the generated kernel structure.",
        ]
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
            if random.random() < 0.4:
                rules.append("Limit the scope of the plan so that the rewritten program is still correct.")
            elif random.random() < 0.3:
                rules.append("Plans can be highly targeted to one particular part of the code.")
            rules.append("Do not count out any of the <optimizations> unless they are clearly irrelevant to the code.")
        if coding:
            rules.append("Optimize the test() function and do not change its name.")
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")

        # Problem-specific rules can be added here if needed
        # if prob and prob.prob_type == "tpu-tutorial" and prob.prob_id == 0:
        #     rules.append("You are optimizing for constant shapes: x.shape = (M, K), y.shape = (K, N). Make sure to take advantage of these shapes.")

        prompt_text = ""
        for i, rule in enumerate(rules):
            prompt_text += f"{i+1}. {rule}\n"
        return prompt_text

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_spad_acc_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                          translate: bool,
                                         ) -> str:
        # Select which menu options will appear
        menu_options_text = ""
        if translate:
            opt_lst = self._get_convert_to_pallas_menu_options()
        else:
            opt_lst = self.get_opt_menu_options(prob)
            if dropout_menu_options < 1 and not force_opt_menu:
                opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
            if shuffle_opts:
                random.shuffle(opt_lst)
        include_score_feedback = random.random() < give_score_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} ms.\n" + parents_prompt

            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with Pallas context
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += self._get_pallas_isa_documentation(prob)
        
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        
        prompt_text += "Please carefully review the Pallas code to identify any inefficiencies. "
        if prob is not None and getattr(prob, 'prob_type', None) == 'tpu' and getattr(prob, 'prob_id', None) == 0:
            prompt_text += "The goal is to match or beat XLA's optimized jnp.matmul, which achieves ~0.08ms for 1024x1024x1024 float32 on TPU v6e. "
            prompt_text += "The key to reaching that level is overlapping HBM->VMEM data movement with MXU compute via pipelining, using the right precision, and maximizing MXU throughput. "
            prompt_text += "Do NOT repeat optimizations that have already been applied to the current code. Choose a fundamentally different approach from what is already there. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "

            choose_or_invent = random.random()
            if choose_or_invent < 0.1 and not translate:
                # Prompt to invent a new optimization inspired by the <optimizations>
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2 and not translate:
                # Prompt to invent a new optimization different from the <optimizations>
                prompt_text += "Think of a new optimization different from the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time."

        prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        if prob is None:
            raise ValueError("TpuLLMAgent requires prob parameter to be provided")
        prompt_text += self._get_pallas_isa_documentation(prob)

        prompt_text += "The original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized Pallas code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += "You are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
        prompt_text += "Let's combine the following optimized Pallas code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized combined Pallas code:"
        return prompt_text

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Generate a prompt to reimplement failed code based on stdout/stderr feedback.
        """
        if prob is None:
            raise ValueError("TpuLLMAgent requires prob parameter to be provided")

        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += self._get_pallas_isa_documentation(prob)

        prompt_text += "\n\nYou are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
        prompt_text += "\nThe code was:\n"
        prompt_text += candidate.code
        
        # Add error information
        prompt_text += "\n\nHowever, the code failed with the following output:\n"
        if candidate.stderr:
            prompt_text += "=== STDERR ===\n"
            # Limit the length of each line to 400 characters
            stderr_lines = candidate.stderr.split("\n")
            stderr_lines = [line[:400] for line in stderr_lines]
            stderr_lines = "\n".join(stderr_lines)
            prompt_text += stderr_lines
            prompt_text += "\n"
        if candidate.stdout:
            prompt_text += "=== STDOUT ===\n"
            stdout_lines = candidate.stdout.split("\n")
            stdout_lines = [line[:400] for line in stdout_lines]
            stdout_lines = "\n".join(stdout_lines)
            prompt_text += stdout_lines
            prompt_text += "\n"
        
        prompt_text += "\nPlease fix the code to address the errors while still applying the optimization plan. "
        prompt_text += "Make sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nFixed and optimized Pallas code:"

        return prompt_text
