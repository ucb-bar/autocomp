## Hardware Architecture Summary: AWS Trainium 3 (NeuronCore-v4)

### Device Overview

Trainium 3 contains 8 NeuronCore-v4 cores per device, with 144 GiB HBM at 4.7 TB/s bandwidth, 128 DMA engines, 20 CC-Cores for collective communication, and 4 NeuronLink-v4 interconnects. NKI kernels execute on individual NeuronCores. Up to 2 physical NeuronCores can be grouped into a Logical NeuronCore (LNC=2) sharing HBM, where each core receives a distinct `program_id` (0 or 1) for work partitioning.

### Programming Model

NKI (Neuron Kernel Interface) is a tile-based kernel programming language with NumPy/Triton-like syntax. Kernels are compiled via `@nki.jit` and bypass high-level compiler stages — the developer explicitly controls tiling, memory placement, data movement, and instruction selection. The compiler performs minimal automatic optimization (resource allocation and instruction scheduling only), so performance is largely determined by source-level decisions. All on-chip memory is software-managed with no hardware caches. Operations map to specific compute engines via `nki.isa.*` APIs, and independent operations across engines execute in parallel.

### Memory Hierarchy

**HBM (Device Memory):** 144 GiB total, 4.7 TB/s bandwidth. Linear address space. Kernel inputs/outputs reside here. Sequential access patterns are critical for performance.

**SBUF (State Buffer):** 32 MiB per NeuronCore. Primary on-chip scratchpad. Organized as a 2D memory with **128 partitions** of 256 KiB each. The first tensor dimension maps to the partition dimension (P); remaining dimensions pack into the free dimension (F). Bandwidth is ~20× higher than HBM. All compute engines read from and write to SBUF. Supports row-major, col-major-2B, and col-major-4B layouts. New in v4: DMA engines can perform read-add-write accumulation directly into SBUF at full DMA throughput.

**PSUM (Partial Sum Buffer):** 2 MiB per NeuronCore. 128 partitions, each divided into 8 banks of 512 float32 values (or 1024 bfloat16 values on v4). Dedicated accumulator for Tensor Engine matmul outputs. Supports near-memory FP32 and BF16 read-accumulate-write controlled by TensorE. PSUM should be evicted to SBUF promptly due to limited capacity. Up to 8 outstanding accumulation groups enable pipelining TensorE with other engines.

**Compiler Spilling:** If live data exceeds SBUF or PSUM capacity, the compiler automatically inserts spill/refill traffic to HBM, degrading performance. Spill traffic exceeding ~30% of total SBUF↔HBM traffic warrants optimization.

### Partition and Free Dimension Constraints

All on-chip memory (SBUF/PSUM) is 2D: partition dimension (P) × free dimension (F). The P dimension enables parallel access across 128 partitions; the F dimension is accessed sequentially. Flexible strided/gather/scatter indexing is supported along F but not along P. Tensors must occupy contiguous partitions with alignment rules: P>64 must start at 0; P>32 at 0 or 64; P≤32 at 0, 32, 64, or 96.

Key tile size constants:
- `pmax` = 128 (max partition dimension)
- `psum_fmax` = 512 (max free dim in PSUM for float32; 1024 for bfloat16 on v4)
- `gemm_stationary_fmax` = 128 (max stationary free dim)
- `gemm_moving_fmax` = 512 (max moving free dim for float32 dst; up to 4096/8192 on v4)

### Compute Engines (4 per NeuronCore)

All four engines have independent sequencers and execute asynchronously in parallel. Synchronization uses hardware semaphores managed by the compiler.

**Tensor Engine (TensorE)** — 2.4 GHz, 128×128 systolic array.
- Peak: **315 TFLOPS** MXFP8/MXFP4, **79 TFLOPS** BF16/FP16/TF32, **20 TFLOPS** FP32.
- Structured sparsity (patterns 1:2 through 4:16): up to **315 TFLOPS** BF16/FP16/TF32.
- Reads from SBUF, writes to PSUM. Internal accumulation in FP32; output can be FP32 or BF16 (v4).
- `nc_matmul(stationary[K,M], moving[K,N])` computes `stationary.T @ moving → dst[M,N]`. Contraction axis K must be in the partition dimension.
- Each call produces LoadStationary + MultiplyMoving instructions. LoadStationary can run up to 4× faster than MultiplyMoving, so place the larger free-axis tensor as stationary.
- MM initiation interval: `max(N, 64)` TensorE cycles for BF16/FP16. FP32 is ~4× slower.
- Row tiling: 4×32-row tiles on v3+. Column tiling also available (independently configurable).
- **MX matmul** (`nc_matmul_mx`): 4× throughput vs BF16. Logical 512×128 systolic array. Requires x4-packed data types (`float8_e5m2_x4`, `float8_e4m3fn_x4`, `float4_e2m1fn_x4`) with uint8 scale tensors. Scaling groups of 32 elements (8 partitions × 4 elements). Row tiling only.
- **BF16 PSUM output** (new in v4): doubles effective PSUM capacity for matmul accumulation. Supports BF16 near-memory accumulation with RNE or stochastic rounding.
- **Background transpose** (new in v4): transpose runs in parallel with another matmul or transpose automatically.
- Also used for data reshapes: 128×128 transpose, partition broadcast/shuffle, cross-partition summation (via matmul with constant matrices).

**Vector Engine (VectorE)** — 1.2 GHz, 512 elements/cycle (BF16/FP16/FP8) or 256 elements/cycle (other types).
- **1.2 TFLOPS** FP32. 128 parallel lanes. All arithmetic internally FP32.
- Optimized for multi-input operations: reductions, LayerNorm, pooling, axpy.
- Performance mode (auto-detected): 4× throughput for `tensor_copy`/`tensor_scalar` when both tensors in SBUF and BF16/FP16 with contiguous inner free dim; 2× for `tensor_tensor` under similar conditions.
- **MX Quantization** (`nisa.quantize_mx`, new in v4): BF16/FP16 → MXFP8 at 4 elements/partition/cycle. Runs exclusively on VectorE.
- **Fast exponential** (`nisa.exponential`, new in v4): 4× throughput vs ScalarE exp. Fused `exp(src - max) + reduce_sum` in one instruction — directly accelerates softmax.
- 32×32 transpose and 32-partition shuffle via reshape/compute banks.
- Cost model: ~N VectorE cycles for 1-input ops, ~2N for 2-input ops (N = free dim size). ~100 cycle static overhead for small N or dependent instructions.

**Scalar Engine (ScalarE)** — 1.2 GHz, 256 elements/cycle (BF16/FP16/FP8) or 128 elements/cycle (other types).
- **1.2 TFLOPS** FP32. 128 parallel lanes. All arithmetic internally FP32.
- Optimized for element-wise operations: activations (GELU, sqrt, exp), scale/bias.
- Pipelined multiply-add-activate: `out = func(in * scale + bias)` in a single `nisa.activation` instruction — always combine when possible.
- Pipelined reduction: `nisa.activation_reduce` computes activation + accumulation simultaneously.
- New in v4: supports `tensor_scalar` and `tensor_copy` at 2× performance for BF16/FP16, enabling load-balancing with VectorE.
- No layout constraints for scalar ops (unlike VectorE which requires parallel axis in P dim).

**GpSimd Engine** — 1.2 GHz, 8 fully-programmable 512-bit vector processors.
- Executes arbitrary C/C++ code. Each processor has 64 KB TCM (3-cycle access).
- Each processor connects to 16 SBUF partitions (512 bits/cycle read and write).
- Effective parallelism: 128 FP32 lanes, 256 FP16 lanes, 512 INT8 lanes.
- Each processor has an integrated DMA engine (new in v3+): 307 GB/s total across all 8 processors, operates in parallel with main DMA and GpSimd compute.
- Used for custom operators not efficiently mapped to other engines.

### Engine Parallelism (v3/v4)

VectorE and GpSimdE can access SBUF in parallel (was serialized on v2). VectorE and ScalarE can access PSUM in parallel at full bandwidth if no bank collisions. SBUF can drive peak bandwidth for VectorE/ScalarE/TensorE or GpSimdE/ScalarE/TensorE simultaneously.

### Data Movement (DMA)

16 DMA engines per NeuronCore, each handling 8 SBUF partitions. Peak bandwidth per engine: **33 B/ns** (aggregate **528 GB/s** per NeuronCore on Trn3). Transfers are asynchronous to compute.

**Bandwidth saturation:** ≥4 KiB per partition for full throughput; ≥2 KiB minimum recommended. Per-transfer overhead is ~1300 ns. Small/frequent transfers are latency-bound.

Minimum free dimension for 2 KiB/partition: 512 (float32), 1024 (bf16/fp16), 2048 (float8).

**DMA capabilities:** HBM↔SBUF, HBM↔HBM, SBUF↔SBUF. Supports copy, transpose (HBM→SBUF at ~90% throughput, SBUF→SBUF at ~50%), scatter-gather, and datatype casting during transfer (source→FP32→dest; not supported for MXFP4/MXFP8).

**16 DMA queues** per engine. Same-queue transfers execute in order. Cross-queue scheduling uses QoS configuration on v4.

**Hardware Descriptor Generation Engine (DGE):** 2 per NeuronCore-v3+. Generates DMA descriptors on-demand (~600 ns per instruction) without consuming SBUF or GpSimdE resources. Preferred over software DGE. Does not support indirect DMA.

**SBUF Read-Add-Write (new in v4):** DMA performs `B += A` where B is in SBUF at full DMA copy throughput — near-memory accumulation without compute engine involvement.

### Data Types

TensorE inputs: MXFP8, MXFP4, FP8 (e4m3, e5m2), BF16, FP16, TF32, FP32. TensorE output: FP32 or BF16 (v4). VectorE/ScalarE: FP8, BF16, FP16, TF32, FP32, INT8, INT16, INT32. Programmable rounding: RNE and stochastic rounding. Adjustable exponent biasing for cFP8/MXFP8.

### Key Optimization Principles

1. **Maximize TensorE utilization** — it provides >99% of peak FLOPS. Use full 128×128 tiles. Prefer MXFP8 (4× over BF16) or FP8 double-row mode (2× over BF16) when precision allows.
2. **Minimize HBM access** — compute-to-bandwidth ratio is ~67 ops/byte (315 TFLOPS / 4.7 TB/s for MXFP8). Maximize data reuse in SBUF's 32 MiB.
3. **Tile for SBUF capacity** — avoid spilling. Budget SBUF usage across all live tiles.
4. **Evict PSUM promptly** — only 2 MiB; use BF16 PSUM output to double effective capacity.
5. **Maximize DMA payload size** — ≥4 KiB per partition, use all 128 partitions, coalesce transfers.
6. **Pipeline compute and data movement** — DMA, TensorE, VectorE, ScalarE, and GpSimdE all run concurrently. Use `affine_range` for compiler pipelining; `sequential_range` when loop-carried dependencies exist.
7. **Combine operations** — use `nisa.activation` for fused scale+bias+nonlinearity (3× faster than separate ops); use `nisa.exponential` for fused exp+reduce in softmax.
8. **Load-balance engines** — ScalarE and VectorE have equal FP32 throughput on v4; distribute work to avoid bottlenecking either.
9. **Use hardware DGE** for DMA descriptor generation to free GpSimdE for compute.
10. **Contraction axis in partition dimension** — matmul layout constraint LC#1. Non-matmul parallel axis should also be in partition dimension (LC#2).