## TPU v6e (Trillium) Hardware Architecture Summary

### Overview

The TPU v6e is Google's Trillium-generation tensor processing unit, programmed via JAX Pallas kernels that lower through the Mosaic compiler to TPU machine code. The TPU executes as a **sequential, pipelined processor** — unlike GPUs, grid iterations run serially in lexicographic order, not in parallel. The programming model uses `pallas_call` with a grid of iterations, where `BlockSpec` definitions tile inputs/outputs into blocks that are automatically pipelined between memory levels. The compiler overlaps memory transfers with compute via double-buffering.

### Memory Hierarchy

| Level | Name | Capacity | Access Characteristics |
|-------|------|----------|----------------------|
| **Off-chip** | HBM | 32 GiB | High latency, ~819 GB/s bandwidth (v5e reference). Cannot be accessed directly by compute — must be prefetched to VMEM/SMEM via DMA. Transfers are asynchronous and overlapped with compute by the compiler. |
| **On-chip SRAM** | VMEM (Vector Memory) | 16+ MB (96 MB on v5p) | ~10× lower latency than HBM. **4 KiB transaction granularity** with alignment requirements. Default memory space for kernel `Ref` arguments. Spilled vector registers also consume VMEM. Each TensorCore has its own private VMEM. |
| **On-chip SRAM** | SMEM (Scalar Memory) | Small | Supports **random access** at single-32-bit-value granularity. No alignment requirements. Used for control-flow data, dynamic indices, and irregular access patterns (e.g., block-sparse indexing). |
| **Registers** | VREGs (Vector) | — | **2D tiles of 8×128 elements** (for 32-bit types on v6e). All vector/matrix computation operates on data in VREGs. Reading a VMEM `Ref` loads into VREGs. |
| **Registers** | SREGs (Scalar) | — | Scalar values for control flow and address computation. |

**Key data flow:** HBM → VMEM (automatic prefetch before kernel body) → VREGs (explicit `ref[...]` load) → Compute → VREGs → VMEM → HBM (automatic writeback after kernel body). The kernel author manages VMEM↔VREG transfers; the compiler manages HBM↔VMEM pipelining.

### Compute Units (per TensorCore)

1. **MXU (Matrix Multiply Unit):** Systolic array of **256×256 multiply-accumulators** (v6e; prior generations were 128×128). Executes matrix multiplications asynchronously in the background. Inputs must be **bfloat16**; accumulation is always **float32**. Even float32 operands are rounded to bfloat16 unless explicit precision is requested. Last-two-dimension transposes of operands fuse into matmul for free. Produces 16K multiply-accumulate operations per cycle.

2. **Vector Processing Unit (VPU):** Operates on 2D vector registers (8×128 for 32-bit). Handles elementwise operations, reductions, and non-matmul computation. Hardware supports only **32-bit elementwise compute** — narrower types should be upcast. Cost tiers: cheap (add, mul, bitwise, comparisons, casts), medium (div, exp, tanh, pow), expensive (sin, cos).

3. **Scalar Unit:** Handles control flow, address computation, and maintenance operations. Loops are **fully unrolled** at compile time — keep trip counts small.

4. **XLU:** Handles matrix transpositions and permutations, asynchronously scheduled.

All three main compute units (Scalar, VPU, MXU) can operate **asynchronously** with respect to each other, managed by the compiler. From the programmer's perspective, execution is single-threaded.

**Reference throughput (v5e):** 197 TFLOP/s for bf16/f32 operations, yielding an arithmetic intensity crossover of ~240 FLOPs/byte. The v6e MXU is 4× larger (256×256 vs 128×128), so peak throughput is substantially higher.

### Tiling and Layout Constraints

- Vector registers are **8×128 tiles** (sublanes × lanes) for 32-bit types. The last two dimensions of all arrays are tiled onto this shape.
- **Block shape requirements:** Rank ≥ 1. The last two dimensions must be divisible by **8 and 128** respectively, OR equal to the full array dimension along that axis.
- Reads/writes aligned to multiples of **8 and 128** in the last two dimensions are always supported for 32-bit types.
- **All computation is padded to tile size:** a 1×1 array costs the same as an 8×128 array. Singleton dimensions in the last two axes are extremely wasteful — an 8×128×1×1 array is 1024× more expensive than 8×128 due to per-element tile padding.

### Dimension Sensitivity

- **Last two axes are special** and map to the physical tile layout.
- Reductions: last dim = slowest, second-to-last = medium, leading dims = fastest.
- Reshapes: free for all but last two dims. Only limited last-two-dim reshapes are supported.
- Transpositions: arbitrary permutations of all but last two axes are **free** for ≥4D arrays. Last-two-dim transposes can fuse into matmul.

### Pipelining Model

- Default pipelining is **double-buffered** (2 VMEM buffers per input/output) for HBM↔VMEM transfers.
- Three pipeline stages per iteration: **copy_in** (HBM→VMEM), **compute** (VMEM↔VREGs), **copy_out** (VMEM→HBM).
- Consecutive grid iterations that access the same input slice skip the HBM transfer — the existing VMEM buffer is reused. This enables accumulation patterns for reductions.
- Reduction axes must be the **last (innermost) grid dimensions**. Output buffers contain garbage initially and must be explicitly initialized.
- Block size is the **most important tuning parameter**: it controls pipeline granularity, VMEM consumption, and per-block arithmetic intensity.

### Multicore (Megacore)

- **Not available on v6e.** Megacore (2 TensorCores per chip) exists only on TPU v4 and v5p. Specifying `dimension_semantics` is a no-op on v6e.

### SparseCore

- v6e has **2 SparseCores per chip**, each with 16 vector subcores (SIMD width 8 for f32, 16 for bf16).
- Optimized for irregular/random memory access patterns: gather, scatter, sorting, histograms, ragged operations.
- SparseCore and TensorCore can execute **concurrently** when placed in the same `jax.jit`.

### Inter-Chip Interconnect (ICI)

- TPUs are connected in pod topologies via ICI (ND torus). Communication uses a **push-only RDMA model** — a device can push data to any other device but can only read local data.
- **v5e restriction:** devices can only route to power-of-2 offsets. Best practice: transfer only to neighboring devices.
- DMA semaphores track async transfer progress. Barrier semaphores are a limited resource (~20-30 per TPU).

### Supported Data Types

`float32`, `bfloat16`, all `int*` except `int4`, all `uint*`, `bool_`. bfloat16 halves memory bandwidth requirements and is the native MXU input format. Subnormals are flushed to zero in bfloat16.

### Key Optimization Principles

1. **Maximize arithmetic intensity per block** — larger blocks amortize memory transfer costs and approach compute-bound regime.
2. **Use bfloat16** for operands to halve memory bandwidth and match native MXU input format.
3. **Avoid small trailing dimensions** — padding waste to 8×128 tiles is the dominant source of inefficiency for undersized blocks.
4. **Place reduction axes last in the grid** to enable accumulation without extra HBM round-trips.
5. **Fuse operations** (activations, transposes) into matmul kernels — these are essentially free when compute-bound.
6. **Sweep block sizes** empirically — the optimal block size depends on the tradeoff between VMEM capacity, pipeline bubble overhead, and per-block arithmetic intensity.