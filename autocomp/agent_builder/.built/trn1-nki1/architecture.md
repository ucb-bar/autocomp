## Hardware Architecture Summary: AWS Trainium 1 (NeuronCore-v2)

### Device Overview

AWS Trainium 1 (trn1 instances) contains NeuronDevices, each with 2 NeuronCore-v2 compute units, 32 GiB HBM across 2 stacks at 820 GB/s aggregate bandwidth, and 32 DMA engines (16 per NeuronCore). NKI kernels execute on a single NeuronCore. The programming model is tile-based: the programmer explicitly manages data movement between HBM and on-chip SRAM, performs computation on tiles resident in on-chip memory, and stores results back to HBM. There is no hardware cache; all on-chip memory is software-managed. The Neuron compiler handles memory allocation, instruction scheduling, and instruction-level parallelism across engines, but the programmer is responsible for tiling strategy, data layout, and instruction selection.

### Memory Hierarchy

**SBUF (State Buffer)** — 24 MiB total, organized as 128 partitions × 192 KiB each (176 KiB usable per partition after 16 KiB reserved for compiler). This is the primary on-chip working memory accessible by all compute engines. SBUF bandwidth is approximately 20× higher than HBM, sufficient to sustain peak Tensor Engine throughput. Maximum free dimension per tensor is 64K elements. All computation operates on data in SBUF; inputs must be explicitly loaded from HBM via `nl.load` and outputs stored back via `nl.store`.

**PSUM (Partial Sum Buffer)** — 2 MiB total, organized as 128 partitions × 16 KiB each. Each partition is divided into 8 banks, each holding up to 512 FP32 values. PSUM is the dedicated accumulation buffer for Tensor Engine matmul results, supporting near-memory read-accumulate-write. Maximum free dimension per tensor is 4K elements (512 FP32 values per bank). Matmul results always land in PSUM in FP32. PSUM should be treated as scarce; results should be evicted to SBUF promptly.

**HBM (Device Memory)** — 32 GiB total per device, 820 GB/s bandwidth. Linear address space; most performant with sequential access patterns. All kernel input/output tensors must reside in HBM.

### On-Chip Memory Layout

SBUF and PSUM are 2D memories with a **partition dimension (P)** and a **free dimension (F)**. NKI tiles map `shape[0]` to the partition dimension. The partition dimension has a hard maximum of 128 elements (`pmax = 128`). All active partitions share the same free-dimension access pattern. Free-dimension access supports up to 4D tensorized strided access within each partition. When the most-minor free-dimension stride is less than 16 bytes, peak streaming bandwidth is 128 elements/cycle at 1.4 GHz per read/write interface; strides ≥ 16 bytes reduce bandwidth to approximately 50% of peak. Each tensor access request incurs roughly 60 cycles of static overhead, so large tile dimensions amortize this cost.

### Compute Engines

Each NeuronCore-v2 has four heterogeneous engines that execute asynchronously in parallel, synchronized by compiler-inserted semaphores.

**Tensor Engine (TensorE)** — 128×128 systolic array, 2.8 GHz. Peak throughput: 92 TFLOPS at BF16/FP16/TF32/cFP8, 23 TFLOPS at FP32. Executes `nc_matmul(stationary[K,M], moving[K,N])` computing `stationary.T @ moving → [M,N]`. Internally decomposes into LoadStationary (LS) and MultiplyMoving (MM) instructions. Tile size constraints: stationary free axis (M) ≤ 128, partition axis (K) ≤ 128, moving free axis (N) ≤ 512. Both inputs must have the contraction dimension K in the partition dimension. MM initiation interval is max(N, 64) TensorE cycles for BF16/FP16/TF32/cFP8; FP32 costs ~4× more. LS can execute up to 4× faster than MM with the same free axis size, so the matrix with the larger free axis should be stationary. Best throughput comes from back-to-back nc_matmul with 128×128 stationary and 128×512 moving tiles. PSUM supports up to 8 outstanding accumulation groups (one per bank). TensorE can also perform 128×128 transposes and partition broadcasts/shuffles using identity or ones matrices, but these block matmul work. The arithmetic intensity threshold to saturate TensorE at BF16 is approximately 222 Flops/Byte.

**Vector Engine (VectorE)** — 128 parallel vector lanes at 1.12 GHz, 2.3 TFLOPS FP32. Handles operations where each output depends on multiple inputs (reductions, tensor-tensor ops, layer normalization, pooling). Reads from and writes to SBUF and PSUM. Cost for free axis size N > 128: approximately N cycles for one-input ops, 2N cycles for two-input ops. Supports 32×32 transpose and 32-partition shuffle within groups of 32 partitions.

**Scalar Engine (ScalarE)** — 128 parallel lanes at 1.4 GHz, 2.9 TFLOPS FP32. Handles element-wise single-input operations and non-linear functions (Gelu, Sqrt, exp, etc.). Features a pipelined multiply-add-activate path: `out[i][k] = func(in[i][k] * scale[i] + bias[i])` in one instruction, providing 2× speedup over separate operations. Also supports pipelined activation-reduce (activation + reduction with no additional cost for the reduction, plus ~64 ScalarE cycles to write the reduction result). Reduction operator is addition only on NeuronCore-v2.

**GpSimd Engine** — 8 fully programmable 512-bit SIMD processors at 1.4 GHz, each with 64 KB tightly-coupled memory (3-cycle latency). Each processor connects to 16 SBUF partitions. Effective parallelism: 128 FP32 lanes. Used for custom operations not expressible through other engines.

All engines perform arithmetic in FP32 internally with zero-overhead casting for other data types.

### DMA Subsystem

16 DMA engines per NeuronCore, each capable of 27 GiB/s peak bandwidth, operating in parallel. A single `nl.load`/`nl.store` with 128 partitions maps to 16 parallel DMA transfers (one per engine, each handling 8 partitions). Minimum transfer size for ideal bandwidth utilization is ≥32 KiB per engine (e.g., 8 partitions × 1024 elements × 4 bytes). Free dimension size of ~1024 elements is ideal; beyond 1024 has diminishing returns. `nl.load_transpose2d` has significantly lower DMA bandwidth than `nl.load`; prefer `nl.load` followed by `nisa.nc_transpose` when TensorE is idle.

### Key Constraints and Optimization Principles

**Tile size constraints:** Partition dimension ≤ 128 (hard limit). PSUM free dimension ≤ 512. Matmul stationary free axis ≤ 128, moving free axis ≤ 512.

**Layout constraints:** Matmul contraction axis must be in the partition dimension for both inputs. Non-matmul operations should map the parallel axis to the partition dimension.

**Compute balance:** TensorE provides ~92 TFLOPS vs. VectorE at 2.3 TFLOPS and ScalarE at 2.9 TFLOPS — a ~30-40× gap. Kernels should maximize TensorE utilization and minimize vector/scalar work or overlap it with tensor compute via engine-level parallelism.

**Memory management:** SBUF capacity is 24 MiB; exceeding live data capacity causes compiler-inserted spills to HBM. Declare buffers in the innermost loop that uses them. Spill traffic exceeding ~30% of total SBUF read/write bytes warrants optimization. PSUM is only 2 MiB; evict matmul results to SBUF promptly.

**Engine parallelism:** All four engines can execute concurrently, but VectorE and GpSimdE cannot access SBUF in parallel, and VectorE and ScalarE cannot access PSUM in parallel (compiler serializes these). Double-buffering or multi-buffering of physical tiles enables overlap of DMA with compute.

**Loop constructs:** `nl.affine_range` enables compiler reordering and parallelism (use when no loop-carried dependencies; associative reductions like matmul accumulation via `+=` are permitted). `nl.sequential_range` enforces strict ordering for true loop-carried dependencies.

**Partition utilization:** Using fewer than 128 partitions underutilizes compute engines proportionally (e.g., 64 partitions = 50% utilization). Partition start alignment depends on partition count: >64 must start at 0; >32 at 0 or 64; ≤32 at 0, 32, 64, or 96.

**Performance targets:** Compute-bound kernels should achieve ≥90% engine utilization. Memory-bound kernels should target ≥60% memory bandwidth utilization. DMA bandwidth utilization below 60% indicates a data movement bottleneck.