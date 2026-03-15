## Hardware Architecture Summary: AWS Trainium1 / NeuronCore-v2

### Overview

AWS Trainium1 (trn1 instances) contains NeuronDevices with 2 NeuronCores-v2 each. NKI (Neuron Kernel Interface) is a tile-level DSL that compiles kernels targeting a single NeuronCore-v2. NKI bypasses the compiler's graph-level, loop-level, and intrinsic-mapping passes, entering directly at the hardware-specific optimization stage (memory allocation and instruction scheduling). Developers can further override allocation decisions using direct allocation APIs.

### Memory Hierarchy (Software-Managed, No Hardware Caches)

**PSUM (Partial Sum Buffer)** — Top of hierarchy
- 2 MiB total, 128 partitions × 16 KiB each
- 2D organization: partition dimension (P) × free dimension (F)
- F max: 512 FP32 elements per partition (4K elements total across banks); 8 banks
- Dedicated accumulator for TensorE matmul output; supports atomic read-add-write
- PSUM partition dimension is rotated 90° relative to SBUF partition dimension
- Data must be copied to SBUF before storing to HBM

**SBUF (State Buffer)** — Main on-chip SRAM
- 24 MiB total, 128 partitions × 192 KiB each (176 KiB usable per partition after 16 KiB compiler reservation)
- 2D organization: partition dimension (P) × free dimension (F)
- F max: 64K elements per tile
- ~20× higher bandwidth than HBM
- All compute engines can read/write SBUF; all data must pass through SBUF for computation
- Supports flexible strided/gather/scatter access along the free dimension; partition dimension does NOT support flexible indexing
- When live data exceeds capacity, the compiler inserts spill/refill DMA transfers to HBM

**HBM (Device Memory)**
- 32 GiB total (2 HBM stacks), 820 GB/s bandwidth
- Linear (1D) memory; most performant with sequential access
- All kernel inputs and outputs must reside in HBM
- 32 DMA engines handle HBM↔SBUF transfers, running in parallel with compute

**Host Memory** — Not directly accessible from NKI kernels.

### Compute Engines (4 Heterogeneous, Asynchronous, Parallel)

All four engines execute independent instruction streams concurrently, synchronized via hardware semaphores. The compiler auto-inserts synchronization based on data dependencies.

**Tensor Engine (TensorE)** — Matrix Multiplication
- 128×128 systolic array, 2.8 GHz
- Data path: 2×128 elements/cycle input, 1×128 elements/cycle output
- Reads from SBUF, writes to PSUM
- Throughput: 92 TFLOPS for BF16/FP16/TF32/cFP8; 23 TFLOPS for FP32 (4× slower)
- Accumulation always in FP32; output always FP32
- Instruction: `nc_matmul(stationary, moving)` executes LoadStationary (LS) then MultiplyMoving (MM), computing `stationary.T @ moving`
- Tile size limits: stationary free axis (M) ≤ 128, moving free axis (N) ≤ 512, contraction/partition axis (K) ≤ 128
- MM initiation interval: `max(N, 64)` TensorE cycles; ideal at N=512
- Background LS: next LS overlaps with current MM; Fast LS is up to 4× faster than MM for same free axis size, so put the matrix with the larger free axis as stationary when dimensions differ
- Best throughput: back-to-back nc_matmul with max tiles (128×128 stationary, 128×512 moving)
- Also used for 128×128 transpose (matmul with identity), partition broadcast, and shuffle (low utilization)

**Vector Engine (VectorE)** — Reductions, Element-wise Tensor-Tensor Ops
- 128 parallel vector lanes, 1.12 GHz
- Reads/writes SBUF and PSUM
- Arithmetic in FP32 with zero-overhead casting
- Cost: ~N cycles for one input tile (N = free axis size, when N > 128), ~2N for two input tiles
- Static overhead: ~100 engine cycles per instruction when N is small or instructions are back-to-back dependent
- Cost is the same whether using all 128 lanes or fewer — always maximize partition usage ("partition vectorization")
- Cross-partition: 32×32 transpose and 32-partition shuffle via reshape/compute banks
- Cannot access SBUF in parallel with GpSimdE; cannot access PSUM in parallel with ScalarE

**Scalar Engine (ScalarE)** — Element-wise Scalar Ops, Non-linear Functions
- 128 parallel lanes, 1.4 GHz
- Reads/writes SBUF and PSUM; reads one input tensor per instruction
- Non-linear function set (exp, gelu, sqrt, rsqrt, etc.) is software-updatable
- `nki.isa.activation`: fused `func(in_tile * scale + bias)` in one instruction — 2× speedup vs. separate ops; all activation instructions have the same cost regardless of scale/bias enablement
- `nki.isa.activation_reduce`: fused activation + reduction in one instruction; readback costs ~64 ScalarE cycles
- Same partition utilization rules as VectorE

**GpSimd Engine (GpSimdE)** — General-Purpose SIMD
- 8 processors × 512-bit SIMD, 1.4 GHz
- Each processor: 64 KB TCM (3-cycle access), connects to 16 SBUF partitions
- 16 FP32 lanes per processor (128 total for FP32); 32 FP16 lanes per processor; 64 INT8 lanes per processor
- Reads/writes SBUF only (not PSUM)
- Cannot access SBUF in parallel with VectorE

### Key Constraints and Optimization Principles

**Tile Size Constraints (Hard Limits)**
- Partition dimension (P): always ≤ 128 for both SBUF and PSUM
- PSUM free dimension: ≤ 512 FP32 elements
- Matmul stationary free axis: ≤ 128; matmul moving free axis: ≤ 512
- Contraction axis of both matmul inputs must map to partition dimension

**Layout Constraints**
- Matmul: contraction axis K must be in partition dimension for both operands; for M×K @ K×N, pass tiles shaped [K,M] and [K,N]
- Non-matmul ops: parallel axis should map to partition dimension

**Performance Targets**
- Compute-bound kernels: ≥90% compute engine utilization; MFU approaching 100% for matmul-dominated workloads
- Memory-bound kernels: memory bandwidth utilization (MBU) ≥60%
- Arithmetic intensity saturation threshold: 222 Flops/Byte for BF16 on NeuronCore-v2
- Spill traffic exceeding ~30% of total SBUF↔HBM traffic warrants optimization

**Instruction Efficiency**
- Static instruction overhead: ~100 cycles; single-element-per-partition instructions waste nearly all time in overhead
- Minimum ~128 elements per partition in the free dimension for efficient compute engine utilization
- Instructions using fewer than 128 partitions under-utilize engines proportionally (e.g., 64 partitions = 50%)
- Fuse operations where possible: ScalarE multiply-add-activation in one instruction (3 ops → 1 instruction ≈ 3× latency reduction)

**Engine Parallelism**
- All four compute engines + DMA run concurrently; maximize overlap by ensuring independent operations target different engines
- Double/multi-buffering of SBUF tiles enables overlapping compute and data movement
- Serialization conflicts: VectorE+GpSimdE on SBUF; VectorE+ScalarE on PSUM

**Memory Management**
- Minimize HBM accesses; maximize SBUF data reuse (20× bandwidth advantage)
- Reserve PSUM for TensorE matmul outputs; evict to SBUF as soon as possible
- Tile size is a critical tuning parameter: too small → instruction overhead dominates; too large → SBUF pressure and poor pipelining
- Use `nl.affine_range` for loops without loop-carried dependencies (enables compiler reordering/parallelism); use `nl.sequential_range` only when loop-carried dependencies exist
- Prefer BF16/FP16/TF32/cFP8 over FP32 for 4× TensorE throughput