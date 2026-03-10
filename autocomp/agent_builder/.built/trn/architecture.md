## Hardware Architecture Summary: AWS Trainium1 / NeuronCore-v2

### Overview

AWS Trainium1 (trn1 instances) contains NeuronDevices with 2 NeuronCores-v2 each. NKI (Neuron Kernel Interface) programs a single NeuronCore-v2, providing direct access to the ISA through a tile-based programming model with NumPy/Triton-like semantics. The NKI compiler handles memory allocation and instruction scheduling but the programmer controls tiling, layout, data movement, and compute mapping. An advanced "direct allocation" mode gives manual control over on-chip memory placement.

### Memory Hierarchy (All Software-Managed — No Hardware Caches)

**HBM (Device Memory):** 32 GiB total (2 stacks), 820 GB/s bandwidth. Linear (1D) address space; most performant with sequential access. All kernel inputs and outputs must reside in HBM. Data must be explicitly moved to/from on-chip memory via DMA (`nl.load`/`nl.store`).

**SBUF (State Buffer):** 24 MiB total on-chip SRAM. Organized as 128 partitions × 192 KiB each (176 KiB usable per partition; 16 KiB reserved for compiler). 2D memory: partition dimension (P) and free dimension (F). Supports up to 64K elements per tile along the free dimension. ~20× higher bandwidth than HBM. Accessible by all four compute engines. This is the main working memory — all computation operates on SBUF-resident data.

**PSUM (Partial Sum Buffer):** 2 MiB total on-chip SRAM. 128 partitions × 16 KiB each. Organized into 8 banks; tiles cannot cross bank boundaries. Maximum free dimension: 512 FP32 elements (2 KiB) per bank per partition. Dedicated accumulation buffer for Tensor Engine matmul outputs; supports atomic read-add-write. PSUM's partition dimension is rotated 90° relative to SBUF. Results must be copied from PSUM → SBUF (via Vector/Scalar engine) before storing to HBM. Reserve PSUM for matmul accumulation and evict promptly.

**Host Memory:** CPU DRAM, not directly accessible from NKI kernels.

### Compute Engines (4 Heterogeneous, Asynchronous, Parallel)

All four engines execute independent instruction streams in parallel, synchronized via hardware semaphores (compiler auto-inserts synchronization based on data dependencies).

**Tensor Engine (TensorE):** 128×128 systolic array at 2.8 GHz. Performs matrix multiplication: `nc_matmul(stationary[K,M], moving[K,N])` computes `stationary.T @ moving`. Peak throughput: 92 TFLOPS for BF16/FP16/TF32/cFP8; 23 TFLOPS for FP32 (4× slower). Accumulation always in FP32. Reads from SBUF, writes to PSUM. Instruction sequence: LoadStationary (LS) caches one operand, then MultiplyMoving (MM) streams the other. MM initiation interval: `max(N, 64)` TensorE cycles. Background LS can overlap with current MM. "Fast LoadStationary" is up to 4× faster than MM — put the larger matrix as stationary for vector-matrix multiplies. Tile constraints: stationary free axis ≤ 128, moving free axis ≤ 512, contraction (partition) axis ≤ 128. Best throughput with maximum tile sizes (128×128 stationary, 128×512 moving). FP32 inputs should be downcast to BF16/FP16/TF32/cFP8 before matmul.

**Vector Engine (VectorE):** 128 parallel lanes at 1.12 GHz. Handles reductions, tensor-tensor element-wise operations. Reads/writes SBUF and PSUM. Supports 32×32 on-chip transpose (within groups of 32 partitions via `nc_transpose`) and 32-partition shuffle. Cost for free axis size N > 128: ~N cycles (single input) or ~2N cycles (two inputs). Static overhead ~100 cycles per instruction. Partition dim ≤ 128; free dim ≤ 64K (SBUF) or 4K (PSUM).

**Scalar Engine (ScalarE):** 128 parallel lanes at 1.4 GHz. Element-wise single-input operations with hardware-accelerated nonlinear functions (exp, gelu, sqrt, rsqrt, sigmoid, etc.). Supports pipelined `func(input * scale + bias)` in a single instruction via `nisa.activation` — always fuse multiply-add with activation. Pipelined reduction (`activation_reduce`) adds ~64 ScalarE cycles overhead. Reads/writes SBUF and PSUM. Same tile size constraints as VectorE.

**GpSimd Engine (GpSimdE):** 8 fully-programmable 512-bit SIMD processors at 1.4 GHz. Each has 64 KB local TCM (3-cycle latency). Each processor connects to 16 SBUF partitions (processor 0 → partitions 0–15, etc.), yielding 128 total lanes for 32-bit operations. Used for miscellaneous operations (e.g., triangular masking).

### DMA Engines

Separate from compute engines; run in parallel with all compute. Handle HBM↔SBUF transfers and compiler-generated spill/reload traffic. Overlap data loading with computation for latency hiding.

### Key Constraints and Optimization Principles

**Tile Size Constraints:**
- Partition dimension (P): always ≤ 128 (`pmax`)
- PSUM free dimension: ≤ 512 (`psum_fmax`)
- Matmul stationary free dim: ≤ 128; moving free dim: ≤ 512
- Contraction axis of both matmul inputs must map to partition dimension

**Layout Constraints:**
- Matmul: contraction axis K must be the partition (first) dimension for both operands
- Non-matmul ops: parallel axis should map to partition dimension
- Free dimension supports strided/flexible indexing; partition dimension does not

**Engine Parallelism Restrictions:**
- VectorE and GpSimdE cannot access SBUF simultaneously (serialized)
- VectorE and ScalarE cannot access PSUM simultaneously (serialized)

**Performance Principles:**
- Maximize partition utilization: using fewer than 128 partitions wastes lanes (e.g., 64 partitions = 50% utilization)
- Minimum efficient free dimension: ≥128 elements to amortize ~100-cycle instruction overhead
- Arithmetic intensity threshold: 222 Flops/Byte (BF16) to be compute-bound on NeuronCore-v2
- SBUF capacity is the key tension: larger tiles improve instruction efficiency but increase spill pressure; smaller tiles enable better pipeline overlap
- Spill traffic >30% of total DMA traffic warrants optimization
- Use `nl.affine_range` for loops without carried dependencies (enables compiler parallelism); use `nl.sequential_range` only when ordering matters
- Matmul accumulation via `+=` into a zero-initialized PSUM buffer with `nl.affine_range` is the canonical pattern
- ≥90% engine active time is the compute-bound target; ≥60% memory bandwidth utilization is the memory-bound target