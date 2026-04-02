## Hardware Architecture Summary: AWS Trainium2 (NeuronCore-v3)

### Overview

Trainium2 is a custom ML accelerator from AWS. Each Trainium2 device contains 8 NeuronCores (v3). NKI (Neuron Kernel Interface) kernels target a single NeuronCore (or a pair via LNC=2 mode, where two physical cores share HBM). The programming model is tile-based and bare-metal: the programmer explicitly manages data placement across a software-managed memory hierarchy (no hardware caches) and targets heterogeneous compute engines. NKI kernels compile to device instructions that the hardware schedules with instruction-level parallelism across engines. Most NKI Python code evaluates at compile time; only `nki.isa.*` calls generate runtime device instructions.

### Memory Hierarchy

**HBM (Off-chip Device Memory)**
- 96 GiB total per device (shared across 8 NeuronCores), 3 TB/s aggregate bandwidth.
- Kernel inputs and outputs must reside in HBM. Data is moved to/from on-chip memory via explicit DMA transfers (`nki.isa.dma_copy`).
- 16 DMA engines per NeuronCore, each with 27.2 GB/s theoretical bandwidth (total ~435 GB/s per core). Optimal transfer performance requires ≥4 KiB per partition in the free dimension and all 128 partitions active. Minimum 32 KiB per transfer for ideal bandwidth utilization.

**SBUF (State Buffer — On-chip SRAM)**
- 28 MiB per NeuronCore, organized as 128 partitions × 224 KiB each.
- ~20× higher bandwidth than HBM. Sufficient bandwidth to keep all compute engines busy.
- Two-dimensional memory: first dimension is the partition dimension (P), second is the free dimension (F). Tensors always map their first axis to the partition dimension.
- Free dimension supports up to 64K elements per partition and flexible strided/gathered access. Partition dimension does not support flexible indexing.
- Software-managed: all data placement is explicit. If live data exceeds capacity, the compiler inserts spill/refill traffic to HBM (a performance penalty).

**PSUM (Partial Sum Buffer — On-chip Accumulator)**
- 2 MiB per NeuronCore, organized as 128 partitions × 16 KiB each.
- Each partition has 8 banks, each holding up to 512 FP32 values. Free dimension up to 4K elements per partition.
- Supports near-memory read-accumulate-write controlled by the Tensor Engine — matmul results accumulate directly into PSUM without consuming compute cycles.
- Up to 8 outstanding matmul accumulation groups (one per bank). Multi-buffering across banks enables pipelined accumulation.
- PSUM should be reserved for Tensor Engine outputs; results should be evicted to SBUF promptly via Vector/Scalar engine copy to free capacity.

**Partition Alignment Rules (SBUF/PSUM)**
- 64 < num_partitions ≤ 128: start at partition 0.
- 32 < num_partitions ≤ 64: start at 0 or 64.
- 0 < num_partitions ≤ 32: start at 0, 32, 64, or 96.
- Tensors occupy contiguous partitions with uniform free-dimension access patterns.

### Compute Engines

Each NeuronCore-v3 has four heterogeneous compute engines with independent sequencers that execute asynchronously in parallel. Synchronization is handled by the compiler via hardware semaphores.

**Tensor Engine (TensorE)** — 2.4 GHz
- 128×128 systolic array of processing elements.
- Operations: GEMM, convolution, transpose.
- Peak throughput: 158 TFLOPS FP8, 79 TFLOPS BF16/FP16/TF32, 20 TFLOPS FP32. With structured sparsity: up to 316 TFLOPS.
- Inputs read from SBUF; output written to PSUM (always FP32 accumulation on v3).
- `nc_matmul(stationary[K,M], moving[K,N])` computes `stationary.T @ moving → output[M,N]`. Contraction axis K must be on the partition dimension.
- Tile limits: stationary free dim (M) ≤ 128, partition dim (K) ≤ 128, moving free dim (N) ≤ 512.
- Double FP8 mode: 2× throughput for FP8 by doubling the effective contraction dimension to 256. Requires size-2 second dimension in both inputs. Cannot combine with column tiling, sparse matmul, or transpose mode.
- Built-in transpose mode: bit-accurate, handles NaN/Inf correctly. 2× speedup for FP32 transpose. FP16/BF16 transpose produces 16-bit PSUM output for faster eviction.
- MM initiation interval: ~max(N, 64) TensorE cycles for BF16/FP16/FP8. LoadStationary is up to 4× faster than MultiplyMoving — when one operand has a small free axis, prefer mapping it as the moving operand.
- Best throughput: back-to-back nc_matmul with 128×128 stationary and 128×512 moving tiles.

**Vector Engine (VectorE)** — 0.96 GHz
- 128 parallel vector lanes (one per SBUF partition for standard types; 512 elements/cycle for BF16/FP16).
- 1.0 TFLOPS FP32. Handles vector reductions, element-wise two-tensor operations, cross-partition data movement (32×32 transpose, 32-element shuffle within partition groups).
- Reads/writes both SBUF and PSUM.
- Performance mode (automatic on v3): up to 4× throughput for tensor_copy/tensor_scalar when both tensors are in SBUF, BF16/FP16, and contiguous in the innermost free dimension. 2× for tensor_tensor with both inputs in SBUF and BF16/FP16.
- Cost: ~N cycles for 1 input tile, ~2N for 2 input tiles (N = free axis size, when N > 128). ~100 cycle static overhead per instruction.

**Scalar Engine (ScalarE)** — 1.2 GHz
- 128 parallel lanes. 1.2 TFLOPS FP32.
- Handles element-wise operations and hardware-accelerated non-linear functions (GELU, sqrt, exp, etc.). Function set is software-updatable.
- Pipelined multiply-add-activate: `activation(in, scale, bias)` computes `func(in * scale + bias)` at the same cost as a bare activation — always fuse when possible.
- Pipelined reduction: `activation_reduce` combines activation + reduction (addition) in one instruction.
- Reads/writes both SBUF and PSUM. Supports bit-accurate tensor copies without FP32 casting (new in v3).

**GpSimd Engine (GpSimdE)** — 1.2 GHz
- 8 fully programmable 512-bit SIMD processors executing C/C++ code.
- Each processor: 64 KB tightly-coupled memory (TCM), 3-cycle access, 512-bit width. Connects to 16 SBUF partitions.
- Effective parallelism: 128 FP32 lanes across all 8 processors.
- Reads/writes SBUF only (no PSUM access).
- Each processor has an integrated DMA engine (new in v3): can move data in parallel with GpSimdE computation and main DMA engines. Total integrated DMA bandwidth: 307 GB/s.

### Engine Parallelism Rules (v3 — relaxed from v2)

- VectorE and GpSimdE **can** access SBUF in parallel (hardware arbitrates shared bus).
- VectorE and ScalarE **can** access PSUM in parallel (no bank collisions required).
- SBUF can simultaneously drive peak bandwidth for VectorE+ScalarE+TensorE or GpSimdE+ScalarE+TensorE.
- All compute engines run in parallel with DMA engines, enabling overlap of data movement and computation.

### Data Types

Supported: float32, tfloat32 (1S,8E,10M), bfloat16, float16, float8_e4m3, float8_e5m2, int8/16/32, uint8/16/32. TensorE inputs can be FP8/BF16/FP16/TF32/FP32 with FP32 accumulation. Mixed precision allowed except FP32/TF32 inputs cannot mix with smaller types.

### DMA Transpose (New in Trainium2)

- Transposes data during DMA transfer (swaps most-minor dimension to partition dimension).
- HBM→SBUF: up to 90% DMA throughput. Best when output partition dim is multiple of 128 and innermost free dim is multiple of 16.
- SBUF→SBUF: up to 50% DMA throughput. Useful alternative to TensorE transpose, especially in ScalarE/VectorE-bound kernels.

### Descriptor Generation Engine (DGE) — New in v3

- 2 hardware DGE instances per NeuronCore. Generates DMA descriptors on demand without storing them in memory.
- Each DGE-based DMA instruction takes ~600 ns. Does not support indirect DMA (gather/scatter).
- When triggered from ScalarE, DGE execution can be hidden behind earlier compute instructions.

### Key Optimization Principles

1. **Keep TensorE fed**: TensorE provides >90% of NeuronCore FLOPS. Maximize its utilization by ensuring data is prefetched into SBUF and PSUM is promptly evicted.
2. **Maximize arithmetic intensity**: Target ≥222 Flops/Byte (BF16) to saturate TensorE. Use blocking/tiling to increase data reuse in SBUF.
3. **Tile to hardware limits**: P-dimension ≤ 128 (pmax), PSUM free dimension ≤ 512 (psum_fmax), matmul stationary free dim ≤ 128, matmul moving free dim ≤ 512.
4. **Minimize instruction overhead**: ~100 cycles per instruction overhead. Combine operations using fused ISA instructions (e.g., ScalarE multiply-add-activate). Use ≥128 elements per partition in the free dimension.
5. **Use all 128 partitions**: Fewer partitions waste compute lanes proportionally.
6. **Pipeline across engines**: All four engines plus DMA run concurrently. Structure computation so different engines work on different tiles simultaneously.
7. **Manage SBUF capacity (28 MiB)**: Balance tile sizes to avoid spilling. Declare buffers at appropriate loop scope. Target ≤24 MiB working set for safety margin.
8. **Optimize DMA transfers**: Maximize free dimension (≥4 KiB/partition) and use all 128 partitions. Batch small transfers into larger ones.
9. **Use `affine_range` for parallel loops**: When no true loop-carried dependencies exist (associative accumulation into PSUM is not a dependency). Use `sequential_range` only when hand-tuned ordering must be preserved.
10. **Leverage v3 features**: Double FP8 mode for 2× matmul throughput, DMA transpose for layout transformations, VectorE performance mode for BF16/FP16 copies, hardware DGE to free GpSimdE.