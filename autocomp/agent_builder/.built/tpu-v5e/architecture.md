## TPU v5e Hardware Architecture Summary

The TPU v5e is a Google-designed ML accelerator optimized for matrix-heavy workloads. Each chip contains a single TensorCore — there is no Megacore (multi-TensorCore) and no SparseCore on this generation. Programming is done via JAX's `jax.experimental.pallas` API, which lowers through the Mosaic compiler to TPU machine code. The TPU executes as a sequential, single-threaded processor with a very wide vector datapath — not a massively parallel GPU-style architecture. Grid iterations in Pallas execute in lexicographic order, enabling data reuse across consecutive iterations and deterministic write ordering.

### Memory Hierarchy

The TPU v5e has a three-level memory hierarchy:

1. **HBM (High-Bandwidth Memory):** 16 GB per chip, ~800 GiB/s bandwidth. This is the main off-chip memory where all `pallas_call` inputs and outputs reside by default. HBM cannot be accessed directly by compute units; data must be transferred to on-chip memory via DMA. These transfers are managed by the Pallas runtime/compiler and can be overlapped with computation through software pipelining.

2. **VMEM (Vector Memory):** On-chip SRAM serving as the primary scratchpad for array data. This is the default memory space for Pallas kernel operands. VMEM has ~10× lower latency than HBM. Transaction granularity is 4 KiB with alignment requirements. Reads and writes aligned to multiples of 8 (second-to-last dimension) and 128 (last dimension) are always efficient. Pallas uses double-buffering by default to overlap HBM↔VMEM transfers with compute.

3. **SMEM (Scalar Memory):** On-chip SRAM for scalar values, supporting random access at 32-bit granularity with no alignment constraints. Used for control-flow data, loop indices, and indirect/sparse access patterns (e.g., scalar prefetch for block-sparse kernels).

4. **Registers:** VREGs (vector registers) hold array data loaded from VMEM; SREGs (scalar registers) hold scalar data loaded from SMEM. All computation operates on register values. Vector registers are 2D tiles, nominally 8×128 for 32-bit values. The last two array dimensions are tiled onto these registers (8 sublanes × 128 lanes). Computation on arrays smaller than a full tile is padded to tile size — a 1×1 array costs the same as an 8×128 array. Excessive register pressure causes spills to VMEM, degrading performance.

### Compute Units

Each TensorCore contains three types of compute units that can operate asynchronously with respect to each other:

1. **Matrix Multiply Units (MXUs):** Four 128×128 systolic arrays per chip. Each MXU performs 16K multiply-accumulate operations per cycle. Inputs are bfloat16; accumulation is in float32. Even float32 operands are rounded to bfloat16 unless higher precision is explicitly requested. Peak throughput: **197 TFLOPS (bf16)** or **393 TOPS (int8)**. Transpositions of the last two dimensions of matmul operands can be fused into the MXU operation at no cost.

2. **Vector Processing Unit (VPU):** Handles elementwise operations (activations, reductions, etc.). Significantly lower throughput than MXUs. Cheap operations: add, sub, mul, max, min, abs, bitwise ops, comparisons, casts. Medium cost: division, exp, tanh, pow. Expensive: sin, cos. Reductions over leading dimensions are fastest; reductions over the last dimension are slowest. Broadcasting along all but the last two dimensions is free.

3. **Scalar Unit:** Handles control flow, memory address computation, and maintenance operations. Operates on 0D scalar values stored in SREGs.

### Key Constraints and Optimization Characteristics

**Arithmetic intensity threshold:** With 197 TFLOPS bf16 and ~800 GiB/s HBM bandwidth, the crossover is ~246 ops/byte. Operations below this ratio (elementwise, reductions, small matmuls) are memory-bandwidth-bound; operations above it (large matmuls) are compute-bound. Fusing operations to avoid HBM round-trips is critical for bandwidth-bound kernels.

**Tiling and block shape constraints:** Block dimensions in the last two axes must be divisible by 8 and 128 respectively, or equal to the full array dimension. Matrix tiles should align to 128×128 to fully utilize MXU systolic arrays. Sub-128 dimensions waste compute due to zero-padding. Blocks must have rank ≥ 1 (no scalar blocks). Larger blocks improve utilization but risk exceeding VMEM capacity.

**Pipelining:** Pallas on TPU uses double-buffered software pipelining (2 buffer slots) to overlap HBM↔VMEM transfers with computation. Configurable via `pl.Buffered(buffer_count=N)`. Lookahead prefetch is available for variable-work iterations. Consecutive grid iterations accessing the same output slice skip redundant transfers, enabling efficient accumulation patterns. Reduction dimensions must be the innermost (last) grid dimensions.

**Data types:** Native support for float32, bfloat16, all int/uint precisions (except int4), and bool. Elementwise operations natively execute at 32-bit width; narrower types should be upcast. bfloat16 halves memory footprint and transfer volume versus float32, directly benefiting bandwidth-bound kernels. Subnormals are flushed to zero.

**Layout sensitivity:** Arrays with singleton dimensions in the last two axes are extremely wasteful (each element occupies an entire 8×128 tile). Reshapes involving the last two dimensions are restricted and often memory-bound. Transpositions of all but the last two dimensions are free for arrays with ≥4 dimensions; only last-two-axis transposition is otherwise supported.

**Control flow:** `cond`, `fori_loop`, and `for_loop` are supported but loops are fully unrolled at compile time — keep trip counts small. Excessive control flow degrades code generation quality.

**Ref access semantics:** Input SRAM buffers are read-only (writes do not propagate to HBM). Output SRAM buffers are write-only (reads see uninitialized data). Accumulation into outputs requires consecutive grid iterations writing to the same slice, with explicit initialization via `pl.when(pl.program_id(axis) == 0)`.