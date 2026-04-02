## Hardware Architecture Summary: AWS Trainium2 / NeuronCore-v3

### Overview

AWS Trainium2 contains 8 NeuronCore-v3 units per device. NKI (Neuron Kernel Interface) kernels execute on a single NeuronCore (or up to two NeuronCore-v3s via SPMD). NKI is a tile-level programming language that provides near-ISA control over compute, memory allocation, and data movement. There are no hardware caches — all on-chip memory is software-managed, and the programmer explicitly orchestrates data movement between memory tiers.

### Memory Hierarchy

**HBM (Device Memory):** 96 GiB total across the device, 2.9 TB/s aggregate bandwidth (~362 GB/s per NeuronCore). HBM is a flat, one-dimensional address space. All kernel inputs and outputs reside in HBM. Sequential access patterns yield best DMA performance. Achieved memory bandwidth utilization (MBU) of 60%+ is considered good in practice.

**SBUF (State Buffer) — Main On-Chip SRAM:** 24 MiB per NeuronCore-v2 (28 MiB per NeuronCore-v3). Organized as 128 partitions, each 192 KiB on v2 (16 KiB reserved for compiler use, leaving 176 KiB usable). SBUF bandwidth is ~20× higher than HBM. All compute engines can access SBUF. Tensors in SBUF are two-dimensional: partition dimension (P) × free dimension (F), with the same start offset across all partitions. Free dimension supports up to 4D tensorized strided access patterns. If the most-minor F-dimension stride exceeds 16 bytes, SBUF bandwidth drops to 50% of peak. Each tensor access has ~60 cycles of static overhead on NeuronCore-v2.

**PSUM (Partial Sum Buffer) — Accumulation Buffer:** 2 MiB per NeuronCore. 128 partitions, each 16 KiB, divided into 8 banks of 512 FP32 values each. Maximum free dimension per tile: 512 FP32 elements. PSUM supports near-memory read-accumulate-write controlled by the Tensor Engine — critical for efficient matmul tiling where partial products accumulate without extra data movement. A PSUM tile cannot cross bank boundaries. Up to 8 independent accumulation groups (one per bank). The Tensor Engine, Vector Engine, and Scalar Engine can all read/write PSUM, but only TensorE controls accumulation. Results must be copied from PSUM to SBUF before storing to HBM.

**GpSimd TCM (Tightly-Coupled Memory):** 64 KB per processor × 8 processors = 512 KB total. 3-cycle access latency, 512-bit data width.

### DMA Engines

16 DMA engines per NeuronCore on v2 (128 total across a Trainium2 device). Each engine handles one transfer at a time at up to 27 GiB/s peak. All engines operate in parallel. Each transfer is scatter-gather (lists of source/destination buffers). An SBUF tensor spanning P partitions requires at least P DMA buffers. For ideal bandwidth, each DMA transfer should be ≥32 KiB (e.g., 8 partitions × 1024 elements × 4 bytes). Maximize both F-dimension (≥4 KiB) and P-dimension (ideally 128) for loads/stores. `nl.load_transpose2d` has significantly lower DMA bandwidth than `nl.load` — avoid in memory-bound kernels. DMA engines run in parallel with compute engines, enabling overlap of data movement and computation.

### Compute Engines

Each NeuronCore has four heterogeneous compute engines that execute asynchronously in parallel with independent instruction streams. Synchronization is via hardware semaphores (compiler-inserted).

**Tensor Engine (TensorE):** 128×128 systolic array running at 2.8 GHz (v2). Peak throughput: 79 TFLOPS at BF16/FP16/TF32 on v3 (92 TFLOPS on v2), 158 TFLOPS at cFP8 on v3. FP32 inputs are 4× slower. Structured sparsity on v3 doubles throughput (up to 316 TFLOPS). Computes `stationary.T @ moving` — the contraction axis must be mapped to the partition dimension for both operands. Reads from SBUF, writes exclusively to PSUM. Output is always FP32. Tile constraints: stationary free axis ≤ 128, moving free axis ≤ 512, contraction (partition) axis ≤ 128. For contraction dimensions > 128, accumulate multiple nc_matmul outputs into the same PSUM tile. MultiplyMoving initiation interval: ~max(N, 64) TensorE cycles for non-FP32 types. LoadStationary can be up to 4× faster than MultiplyMoving — prefer mapping the larger free-axis matrix as stationary. PE tiling allows packing multiple small matmuls onto the 128×128 array when tiles are smaller (minimum granularity: 32, with v2 restriction that both row and column cannot simultaneously be 32). Best throughput: back-to-back nc_matmul with stationary [128,128] and moving [128,512]. The saturation arithmetic intensity threshold is ~222 Flops/Byte for BF16 on v2.

**Vector Engine (VectorE):** 128 parallel vector lanes at 1.12 GHz (v2). ~2.3 TFLOPS FP32 on v2, ~1 TFLOPS FP32 on v3. Handles operations where each output depends on multiple inputs: reductions, layer normalization, pooling, tensor-tensor operations. Supports all NKI data types with zero-overhead auto-casting to FP32 arithmetic. Cost model: ~N cycles for single input tile, ~2N cycles for two input tiles (N = free dimension elements), plus ~100 cycles static overhead. Supports 32×32 cross-partition transpose via Reshape Bank and 32-partition arbitrary shuffle.

**Scalar Engine (ScalarE):** 128 parallel lanes at 1.4 GHz (v2). ~2.9 TFLOPS FP32 on v2, ~1.2 TFLOPS FP32 on v3. Handles element-wise operations: activations (GELU, exp, sqrt, etc.), scale/bias. Supports pipelined multiply-add: `func(in_tile * scale + bias)` at the same cost as a plain activation — always combine when possible. Supports pipelined activation-reduce (activation + reduction in one pass). Cost model similar to VectorE. Reads one input tensor per instruction.

**GpSimd Engine:** 8 fully-programmable 512-bit SIMD processors. Each connects to 16 SBUF partitions (processor i → partitions i×16 to i×16+15). Effective parallelism: 128 lanes at 32-bit. Used for custom operations not supported by other engines (e.g., triangular masking, iota).

### Engine Parallelism Constraints

All engines execute asynchronously, but certain SBUF/PSUM access conflicts cause serialization:
- VectorE and GpSimdE **cannot** access SBUF in parallel (serialized).
- VectorE and ScalarE **cannot** access PSUM in parallel (serialized).
- All other engine combinations can access SBUF or PSUM simultaneously at peak bandwidth.

### Tile and Layout Constraints

- **Partition dimension (P):** Maximum 128 elements. Maps to parallel compute lanes.
- **PSUM free dimension:** Maximum 512 elements (one bank).
- **Matmul contraction axis:** Must be on P-dimension for both operands. For `[M,K] @ [K,N]`, provide shapes `[K,M]` and `[K,N]` to nc_matmul.
- **Non-matmul operations:** Parallel axis should be on P-dimension.
- **Partition start alignment:** Depends on partition count — 128 partitions must start at 0; ≤32 partitions can start at 0/32/64/96.
- **PSUM accumulation pattern:** Requires `psum_buf = nl.zeros(..., buffer=nl.psum)` + `nl.affine_range` loop + `psum_buf += nl.matmul(...)`. Using `psum_buf = psum_buf + nc_matmul(...)` does NOT reliably trigger hardware accumulation.

### Key Optimization Principles

1. **Maximize TensorE utilization** — it provides 30-80× more FLOPS than Vector/Scalar engines. Use non-FP32 input types for 4× throughput. Maximize tile sizes toward [128,128] stationary and [128,512] moving.
2. **Minimize HBM traffic** — SBUF is ~20× higher bandwidth. Block/tile computations to maximize data reuse in SBUF. Arithmetic intensity must exceed ~222 Flops/Byte (BF16, v2) to be compute-bound.
3. **Overlap compute and data movement** — use double/multi-buffering so DMA and compute engines operate in parallel on different tiles. Single-buffering completely serializes execution.
4. **Avoid SBUF spilling** — exceeding SBUF capacity causes spills to HBM. Monitor spill ratios (>30% warrants optimization). Declare buffers inside inner loops if needed to reduce live memory.
5. **Maximize partition utilization** — using fewer than 128 partitions underutilizes engines proportionally. Pack small operations to use all 128 lanes ("partition vectorization").
6. **Minimize instruction overhead** — Vector/Scalar engines have ~100 cycles static overhead per instruction. Use large free dimensions (≥128 elements) and combine operations (e.g., fused multiply-add-activation on ScalarE).
7. **Use `nl.affine_range`** for loops without true loop-carried dependencies (enables compiler optimization). Use `nl.sequential_range` only when loop ordering must be preserved. Associative reductions (matmul accumulation) are not loop-carried dependencies.
8. **DMA efficiency** — maximize transfer sizes (≥32 KiB per transfer, F-dim ≥1024 elements). Use all 128 partitions. Avoid `nl.load_transpose2d` in memory-bound kernels. Keep most-minor F-dim stride ≤16 bytes for full SBUF bandwidth.
9. **Pre-transpose matmul inputs** — use `nisa.nc_matmul` directly with pre-transposed LHS to avoid layout shuffling overhead from `nl.matmul`. Avoid using TensorE for reshaping when it could be doing useful matmul work.
10. **Fuse operations** — combine softmax, attention, and other multi-step computations to avoid materializing large intermediates in HBM. ScalarE's pipelined multiply-add-activation and activation-reduce are particularly valuable for fusion.