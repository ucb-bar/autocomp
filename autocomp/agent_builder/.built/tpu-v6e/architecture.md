## TPU v6e Hardware Architecture Summary

### Overview

The TPU v6e (Trillium) is a Google "lite" class TPU with **1 TensorCore per chip** and no Megacore support. It is programmed through JAX Pallas, which lowers to the Mosaic TPU compiler. The execution model is **single-threaded from the programmer's perspective**: a kernel body executes sequentially over a grid in lexicographic order, with the compiler and runtime managing asynchronous overlap of compute and memory transfers behind the scenes. Each grid invocation operates on tiles of data staged into on-chip memory.

### Memory Hierarchy

The TPU v6e exposes a multi-level memory hierarchy. All capacities and bandwidths from `get_tpu_info()` are **per-TensorCore**, which on v6e equals per-chip since there is only one TensorCore.

| Memory | Pallas Enum | Role | Key Characteristics |
|--------|-------------|------|---------------------|
| **HBM** | `pl.ANY` | Main off-chip DRAM | 32 GiB capacity. ~819 GB/s bandwidth (v5e reference; v6e similar class). Cannot be dereferenced directly in kernel body; data must be copied to VMEM/SMEM first. `pallas_call` BlockSpecs automate HBM↔VMEM transfers. |
| **VMEM** | `pltpu.VMEM` | Vector on-chip SRAM | 16+ MB capacity. Default memory space for kernel refs. Transaction granularity: 4 KiB. Best performance when access offsets are aligned to (8, 128) tile boundaries and region sizes are multiples of tile size. This is the primary working memory for all tensor computation. |
| **SMEM** | `pltpu.SMEM` | Scalar on-chip SRAM | Small capacity. Supports random access at single-32-bit-value granularity. Used for scalar values, loop indices, control-flow data, and irregular access patterns (e.g., block-sparse index tables). |
| **CMEM** | `pltpu.CMEM` | Communication memory | Used for inter-core communication. |
| **Registers** | (implicit) | Fastest storage | **VREGs**: vector registers shaped (8, 128) for 32-bit values — 8 sublanes × 128 lanes. All VPU/MXU computation operates on VREGs loaded from VMEM. **SREGs**: scalar registers loaded from SMEM. |

**Data movement path**: HBM → VMEM (DMA, managed by pipelining) → VREGs (explicit load via `ref[...]`) → Compute → VREGs → VMEM (explicit store) → HBM (DMA, managed by pipelining).

**Critical capacity constraint**: All live tiles (input buffers, output buffers, scratch/accumulator buffers, plus compiler-spilled registers) must fit simultaneously in VMEM. A single f32[2048, 2048] array is 16 MiB, which can exhaust VMEM on its own.

### Compute Units

The TensorCore contains several units that operate **asynchronously** with respect to each other (the compiler schedules overlap):

- **VPU (Vector Processing Unit)**: Operates on VREGs. Handles elementwise operations, reductions, and broadcasting. All computation is implicitly padded to the native (8, 128) tile for 32-bit types — a 1×1 operation costs the same as an 8×128 operation.

- **MXU (Matrix Multiply Unit)**: Executes matrix multiplications asynchronously. Native operation is on small matrices (up to ~256 per dimension); Pallas automatically tiles larger blocks. Always accumulates in float32. BF16 inputs are native; f32 inputs are **rounded to bf16** unless `jax.default_matmul_precision` is set. Supports fused RHS transpose. Key throughput numbers (v5e reference): **197 TFLOP/s bf16**, with int8 at ~2× and fp8 at ~2× bf16 throughput.

- **XLU**: Executes matrix transpositions and permutations asynchronously.

- **Scalar Unit**: Operates on SREGs for control flow and scalar computation.

- **DMA Subunits**: Handle asynchronous HBM↔VMEM transfers, overlapped with compute via pipelining.

**Arithmetic intensity crossover** (v5e reference): ~240 FLOPs/byte. Below this, kernels are memory-bound; above, compute-bound. Elementwise operations are typically memory-bound. Matrix multiplications become compute-bound at sufficient size (roughly m > 1440 for f32 square matmul on v5e).

### Tile and Block Constraints

- The native vector register shape is **(8 sublanes, 128 lanes)** for 32-bit values. For bf16, this holds 2048 elements; for int8, 4096.
- **Block shape rule**: the last two dimensions must be divisible by **8 and 128** respectively, OR span the full array dimension on that axis.
- Minimum block rank is 1 (rank-0 not supported).
- **Singleton trailing dimensions are extremely wasteful**: an 8×128×1×1 array pads to 8×128×8×128, costing 1024× more than necessary.
- `get_sublane_tiling(dtype)` returns the native tiling for a given dtype, which should guide BlockSpec choices.

### Operation Cost Characteristics

**Elementwise ops**:
- Cheap: add, sub, mul, max, min, where, abs, bitwise ops, shifts, comparisons, casts
- Medium: division, exp, tanh, pow
- Expensive: sin, cos

**Reductions** (sum, max, min for float; any/all for bool; **no integer reductions**):
- Cheapest over leading dimensions; most expensive over the last dimension.

**Broadcasting**: Free for all but last two dimensions; progressively more expensive for second-to-last and last dimensions.

**Reshapes**: Free for all but last two dimensions. Only specific reshape patterns on last two dims are supported.

**Transpositions**: Arbitrary permutations of all but last two dimensions are free (with ≥4 dims). Last-two-dim transposes can be fused into matmul.

**Control flow**: `cond`, `fori_loop`, `for_loop`, and `while_loop` are supported. Loops with **statically known (Python-int) bounds** are fully unrolled at compile time — each iteration contributes to code size and register pressure. Loops with **data-dependent bounds** (e.g. `fori_loop(0, traced_value, body)`) are lowered as real loops and do not unroll. Python `for`/`range` over traced values will fail to compile — use `jax.lax.fori_loop` / `jax.lax.while_loop` instead.

### Pipelining Model

Pallas uses **double-buffered pipelining** by default on TPU (2 buffer slots per input/output). The pipeline has three phases: prologue (partial utilization), steady state (compute overlapped with copy-in of next block and copy-out of previous block), and epilogue (partial utilization).

- Buffer count is configurable per-argument via `pl.Buffered(buffer_count=N)`.
- **Lookahead prefetch** (`use_lookahead=True`) prefetches the next block as soon as a buffer slot is free, useful when per-block compute time varies.
- Consecutive grid iterations accessing the same input slice skip the HBM transfer (data already in VMEM).
- **Reductions must be over the innermost (last) grid dimensions** so that the accumulator buffer persists in VMEM across iterations writing to the same output slice.
- Output buffers contain garbage initially and must be explicitly initialized (e.g., zeroed on first iteration with `pl.when(pl.program_id(axis) == 0)`).

### SparseCore

TPU v6e has **2 SparseCores per chip**, each with 16 vector subcores and 8 lanes (f32) or 16 lanes (bf16). SparseCores are optimized for irregular access patterns (gather/scatter), sparse embeddings, and low-compute-intensity parallel work. They operate concurrently with the TensorCore, enabling overlap of sparse and dense computation. SparseCore gather is ~4.5× faster than TensorCore `jnp.take` for indexed lookups.

### Key Optimization Principles

1. **Maximize block sizes** to improve compute-to-memory ratio, but ensure all buffers fit in VMEM.
2. **Align to (8, 128) tile boundaries** for memory access efficiency.
3. **Avoid singleton trailing dimensions** — they cause massive padding waste.
4. **Prefer reductions and broadcasts over leading dimensions** (last dimension is most expensive).
5. **Fuse transposes into matmuls** when possible (free).
6. **Order grid axes** so reduction axes are last (enables accumulation and HBM transfer skipping).
7. **Use SMEM for scalar/index data** via `PrefetchScalarGridSpec`.
8. **Block sizes are the most important tuning parameter** — sweep candidates and profile.
9. **Supported dtypes**: f32, bf16, all int/uint precisions except int4, bool. Hardware natively computes in 32-bit; narrow types should be upcast for elementwise ops.
10. **Kernels cannot close over constants** — all data must be passed as explicit inputs with BlockSpecs.