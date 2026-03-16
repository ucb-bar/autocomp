## Hardware Architecture Summary: AWS Trainium1 (NeuronCore-v2)

### Device Overview

AWS Trainium1 (trn1 instances) contains NeuronDevices, each with 2 NeuronCore-v2 units. Each NeuronCore-v2 is a fully independent heterogeneous compute unit with four parallel engines and software-managed on-chip memory. NKI kernels execute on a single NeuronCore-v2. There is no hardware cache anywhere in the memory hierarchy — all data movement and placement is explicitly managed by the programmer and compiler.

### Memory Hierarchy

**HBM (Device Memory):** 32 GiB total per device (shared across 2 NeuronCores), 820 GB/s bandwidth. Linear address space; most performant when accessed sequentially. All kernel inputs and outputs must reside in HBM. Accessed via `nl.load` (HBM→SBUF) and `nl.store` (SBUF→HBM).

**SBUF (State Buffer) — Primary On-Chip SRAM:** 24 MiB total per NeuronCore, organized as 128 partitions × 192 KiB each (16 KiB reserved per partition, leaving 176 KiB usable). 2D memory structure with a partition dimension (P, up to 128) and a free dimension (F, up to 64K elements per partition). SBUF bandwidth is approximately 20× higher than HBM and sufficient to sustain peak compute throughput. Peak bandwidth per read/write interface: 128 elements/cycle at 1.4 GHz when the most-minor free-dimension stride is ≤16 bytes; half bandwidth if stride exceeds 16 bytes. All four compute engines can read from SBUF. Static overhead per tensor access: ~60 cycles.

**PSUM (Partial Sum Buffer) — Accumulator Memory:** 2 MiB total per NeuronCore, organized as 128 partitions × 16 KiB each. Each partition has 8 banks, each holding up to 512 FP32 values (max free dimension: 4K elements per partition). Supports near-memory read-accumulate-write controlled exclusively by the Tensor Engine. Accumulation is always in FP32. Up to 8 outstanding matmul accumulation groups (one per bank). Vector and Scalar engines can also read/write PSUM. PSUM capacity is limited; results should be evicted to SBUF as soon as possible.

**DMA Engines:** 16 DMA engines per NeuronCore, each capable of one transfer at a time at up to 27 GiB/s. All 16 operate in parallel and run concurrently with compute. Each `nl.load`/`nl.store` of 128 partitions maps to 16 parallel transfers (8 partitions per engine). Minimum transfer size for ideal bandwidth: 32 KiB per engine (e.g., 8 partitions × 1024 elements × 4 bytes). Free dimension of ~1024 elements is ideal; beyond 1024 has diminishing returns. `nl.load_transpose2d` has significantly lower bandwidth than `nl.load`.

**Spilling:** When live data exceeds SBUF capacity, the compiler automatically inserts spill/reload transfers to HBM. Spill traffic exceeding 30% of total SBUF↔HBM traffic indicates a significant optimization opportunity.

### Compute Engines

All four engines execute asynchronously in parallel, synchronized via compiler-inserted semaphores. Instructions on different engines with no data dependencies run concurrently.

**Tensor Engine (TensorE):** 128×128 systolic array operating at 2.8 GHz. Peak throughput: 92 TFLOPS for BF16/FP16/TF32/cFP8; 23 TFLOPS for FP32 (4× slower). Reads from SBUF, writes to PSUM. Accounts for >90% of NeuronCore FLOPS. Executes `nc_matmul(stationary[K,M], moving[K,N])` → `output[M,N]` via two internal instructions: LoadStationary (LS) then MultiplyMoving (MM). LS can run up to 4× faster than MM with the same free axis size. MM initiation interval: max(N, 64) TensorE cycles for BF16/FP16/TF32/cFP8. FP32 matmul costs ~4× more. Mixed-precision is native: inputs in BF16/FP16/TF32/cFP8, accumulation always in FP32. Also used for on-chip transpose (matmul with identity matrix), broadcast, and partition shuffle, though these consume matmul cycles.

**Vector Engine (VectorE):** 128 parallel vector lanes at 1.12 GHz, one per SBUF partition. 2.3 TFLOPS FP32. Handles operations where each output depends on multiple inputs: reductions, layer normalization, pooling, element-wise ops between two tensors. Supports all NKI data types with automatic FP32 casting. Can read/write both SBUF and PSUM. Supports 32×32 cross-partition transpose and arbitrary shuffle within groups of 32 partitions.

**Scalar Engine (ScalarE):** 128 parallel lanes at 1.4 GHz. 2.9 TFLOPS FP32. Handles element-wise operations where each output depends on one input: activations (GELU, exp, sqrt, etc.), scale, bias. Reads only one input tensor per instruction. Has deeply pipelined stages: `activation(func, data, bias, scale)` computes `func(data * scale + bias)` at the same cost as a plain activation — always fuse multiply-add with nonlinear functions when possible. Also supports pipelined activation+reduction (`activation_reduce`).

**GpSimd Engine (GpSimdE):** 8 fully programmable 512-bit SIMD processors at 1.4 GHz, each with 64 KB tightly-coupled memory (3-cycle access). Each processor connects to 16 SBUF partitions. Effective 128 FP32 lanes. Used for custom operations not mappable to other engines (e.g., triangular masking).

**Access Serialization Rules:** VectorE and GpSimdE cannot access SBUF simultaneously. VectorE and ScalarE cannot access PSUM simultaneously. All other engine combinations can access SBUF/PSUM concurrently without interference.

### Tile Size Constraints

| Constraint | Limit |
|---|---|
| Partition dimension (P) for SBUF and PSUM | ≤ 128 (`pmax`) |
| PSUM free dimension (F) | ≤ 512 FP32 elements (`psum_fmax`) |
| Matmul stationary (LHS) free dimension | ≤ 128 (`gemm_stationary_fmax`) |
| Matmul moving (RHS) free dimension | ≤ 512 (`gemm_moving_fmax`) |
| Matmul contraction axis (K) | ≤ 128 (must map to P dimension) |
| SBUF free dimension | ≤ 64K elements per partition |

### Layout Constraints

For matrix multiplication, the contraction axis K of both input tiles must be mapped to the partition dimension (P). For `[M,K] × [K,N]`, inputs are passed as `stationary[K,M]` and `moving[K,N]`. The free axis of the stationary tensor becomes the partition (first) axis of the output; the free axis of the moving tensor becomes the free axis of the output.

For non-matmul operations, the parallel axis should be mapped to the partition dimension to maximize lane utilization.

### Key Optimization Principles

The Tensor Engine is ~30-40× more powerful than Vector/Scalar engines in equivalent precision. Maximizing TensorE utilization and keeping it fed with data is the primary optimization target. The arithmetic intensity threshold for compute-bound behavior on NeuronCore-v2 is 222 Flops/Byte for BF16.

All compute engines require at least 128 elements per partition in the free dimension to be efficient. Static instruction overhead is ~100 cycles, making small tiles extremely inefficient. Using fewer than 128 partitions directly under-utilizes compute (e.g., 64 partitions = 50% utilization).

PSUM accumulation for tiled matmul requires a specific code pattern: a `nl.zeros` initialized PSUM buffer, an `nl.affine_range` loop, and `+=` accumulation with `nl.matmul`. Using `=` assignment instead of `+=` does not reliably trigger hardware accumulation.

Prefer mapping the tensor with the smaller free axis to the moving operand of matmul for better throughput (LS is up to 4× faster than MM). Downcast FP32 inputs to BF16/FP16 before matmul to avoid the 4× FP32 penalty.