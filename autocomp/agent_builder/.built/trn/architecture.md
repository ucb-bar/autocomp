## Hardware Architecture Summary: AWS Trainium 1 (NeuronCore-v2)

### Device Overview

Each Trainium chip contains **2 NeuronCore-v2** cores. A Trn1 instance has 16 chips (32 NeuronCores). An NKI kernel executes on a **single NeuronCore-v2**, which is a fully-independent heterogeneous compute unit with four asynchronous engines and software-managed on-chip SRAM (no hardware caches).

### Memory Hierarchy

All memory is software-managed. Data must be explicitly moved between tiers.

| Memory | Capacity | Organization | Bandwidth | Notes |
|--------|----------|-------------|-----------|-------|
| **HBM** | 32 GiB per chip (shared by 2 cores) | Linear (flat address space) | 820 GB/s per chip | All kernel inputs/outputs reside here. Most performant with sequential access. |
| **SBUF** (State Buffer) | 24 MiB per core | 2D: **128 partitions × 192 KiB** each | ~20× HBM bandwidth | Main on-chip working memory. Accessible by all four engines. All computation requires data in SBUF. |
| **PSUM** (Partial Sum) | 2 MiB per core | 2D: **128 partitions × 16 KiB** each (512 FP32 elements per partition per bank) | — | Dedicated accumulation buffer for Tensor Engine matmul outputs. Supports read-add-write accumulation. Also accessible by Vector and Scalar engines. |

**DMA engines** (16 per core from 32 per device) move data between HBM and SBUF, running **in parallel** with computation. DMA bandwidth is 1 TB/s per chip and supports inline compression/decompression.

SBUF and PSUM are **2D memories**: the first dimension is the **partition dimension (P)**, with all 128 partitions read/written in parallel; the second is the **free dimension (F)**, read/written sequentially. Every tensor tile maps `shape[0]` to the partition dimension.

### Compute Engines

The four engines execute asynchronously with semaphore-based synchronization (compiler-inserted). Independent API calls targeting different engines can run in parallel.

**Serialization constraints:** VectorE and GpSimdE cannot access SBUF simultaneously; VectorE and ScalarE cannot access PSUM simultaneously.

#### Tensor Engine (TensorE)
- **128×128 systolic array**, 2.8 GHz
- **Peak throughput:** 92 TFLOPS at BF16/FP16/TF32/cFP8; 23 TFLOPS at FP32 (4× slower)
- Accelerates **matrix multiplication** and 2D convolution
- Reads from SBUF, writes to PSUM
- Accumulation always in FP32; output always FP32
- **`nc_matmul(stationary[K,M], moving[K,N])`** computes `stationary.T @ moving → output[M,N]`
- Both inputs must have contraction dimension K in the **partition dimension**
- **Tile size limits:** K ≤ 128 (partition dim), M ≤ 128 (stationary free axis), N ≤ 512 (moving free axis)
- Best throughput: stationary 128×128, moving 128×512, BF16/FP16/cFP8/TF32
- Each matmul executes two ISA instructions: **LoadStationary (LS)** then **MultiplyMoving (MM)**
- Back-to-back MM initiation interval: **max(N, 64)** TensorE cycles (BF16/FP16/TF32/cFP8)
- LS can overlap with current MM; LS is up to **4× faster** than MM for the same free axis size
- Can also perform transpose, broadcast, and partition shuffle via special matrix patterns (identity, ones, zero/one masks), but this blocks matmul throughput
- **Recommendation:** Downcast FP32 inputs to BF16/FP16/TF32/cFP8 before matmul. For vector-matrix multiply, place the matrix as stationary (large free axis benefits from fast LS).

#### Vector Engine (VectorE)
- **128 parallel lanes**, 1.12 GHz
- **Peak throughput:** 2.3 TFLOPS FP32
- Accelerates reductions, element-wise ops between two tensors (axpy, layer norm, pooling)
- Reads/writes SBUF and PSUM; arithmetic in FP32 with zero-overhead casting
- Parallel axis must map to **partition dimension**; P ≤ 128, F ≤ 64K (SBUF) or 4K (PSUM)
- Cost with all 128 lanes = cost with fewer lanes → **maximize partition axis usage**
- Cost model (F > 128): ~N cycles for 1-input ops, ~2N for 2-input ops (N = free axis size)
- Small N or fully dependent chains: add ~100 cycles static overhead per instruction
- Cross-partition data movement limited to **groups of 32 partitions** (32×32 transpose, 32-partition shuffle)

#### Scalar Engine (ScalarE)
- **128 parallel lanes**, 1.4 GHz
- **Peak throughput:** 2.9 TFLOPS FP32
- Accelerates element-wise operations where each output depends on one input (activation functions)
- Hardware-accelerated non-linear functions: GeLU, Sqrt, Sigmoid, Tanh, ReLU, SiLU, Softplus, Mish, Erf, etc.
- Arithmetic in FP32 with zero-overhead casting; P ≤ 128, F ≤ 64K (SBUF) or 4K (PSUM)
- **Pipelined multiply-add:** `nki.isa.activation` computes `func(in * scale + bias)` in a single instruction — **2× speedup** vs. separate multiply-add + activation
- **Pipelined reduction:** `nki.isa.activation_reduce` fuses activation + reduction; reduction is addition-only on NeuronCore-v2
- All `activation` instructions have the same cost regardless of scale/bias enablement → always fuse

#### GpSimd Engine (GpSimdE)
- **8 fully-programmable 512-bit vector processors**, 1.4 GHz
- Each processor: 64 KB local TCM (3-cycle access), processes 16×FP32 or 32×FP16 or 64×INT8 per cycle
- Each processor connects to **16 SBUF partitions** (512-bit/cycle read and write)
- 128 effective FP32 lanes across all 8 processors
- Used for operations that don't map to other engines (e.g., triangular masking, custom C/C++ operators)
- No simple cost model; refer to per-instruction latency estimates

### Tile Size Constraints Summary

| Constraint | Limit |
|-----------|-------|
| Partition dimension (all on-chip tiles) | ≤ **128** (`pmax`) |
| PSUM free dimension | ≤ **512** (`psum_fmax`) |
| Matmul stationary free axis (M) | ≤ **128** |
| Matmul moving free axis (N) | ≤ **512** |
| Matmul contraction axis (K) | ≤ **128** (must be in partition dim) |
| VectorE/ScalarE free dimension (SBUF) | ≤ **64K** elements |
| VectorE/ScalarE free dimension (PSUM) | ≤ **4K** elements |

### Data Types

Supported: FP32, BF16, FP16, TF32, cFP8 (float8_e4m3, float8_e5m2), INT8, INT16, INT32, UINT8, UINT16, UINT32. Tensor Engine accumulates in FP32 regardless of input type. Vector and Scalar engines compute internally in FP32 with zero-overhead casting. Programmable rounding modes include Round-to-Nearest-Even and Stochastic Rounding.

### Programming Model

NKI kernels follow an **SPMD** launch model (similar to Triton) with `program_id`/`num_programs`. The standard kernel pattern is: (1) DMA load from HBM to SBUF, (2) compute on SBUF/PSUM tiles, (3) DMA store from SBUF to HBM. Loop iterators include `affine_range` (parallel/pipelinable, default), `sequential_range` (enforces ordering for loop-carried dependencies), and `static_range` (fully unrolled). The compiler manages SBUF/PSUM allocation; when live data exceeds capacity, it inserts spills/refills. Careful tiling and loop fusion minimize spilling.

### Key Optimization Principles

1. **Maximize Tensor Engine utilization** with full-size tiles (128×128 stationary, 128×512 moving) in BF16/FP16/cFP8/TF32.
2. **Overlap DMA with compute** — DMA engines run in parallel with all compute engines.
3. **Maximize partition dimension usage** — all 128 lanes cost the same as fewer on Vector/Scalar engines.
4. **Fuse multiply-add with activation functions** into single ScalarE instructions.
5. **Use `affine_range`** for loops without true loop-carried dependencies to enable compiler optimizations.
6. **Minimize SBUF/PSUM live data** through tiling and loop fusion to avoid compiler-inserted spills.
7. **Prefer sequential HBM access patterns** for best DMA throughput.
8. **Be aware of engine serialization:** VectorE+GpSimdE share SBUF access; VectorE+ScalarE share PSUM access.
9. **Arithmetic intensity threshold:** ~232 ops/byte (BF16) to be compute-bound vs. HBM-bandwidth-bound per core.