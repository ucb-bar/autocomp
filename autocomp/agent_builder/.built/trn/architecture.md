# AWS Trainium/Inferentia2 Hardware Architecture Summary for NKI Kernel Optimization

## Overview

AWS Trainium and Inferentia2 chips are built around **NeuronCores** — fully independent heterogeneous compute units. NKI (Neuron Kernel Interface) kernels execute on a single NeuronCore. Each NeuronCore contains multiple specialized engines and software-managed on-chip memory. There are three generations relevant to NKI:

| Generation | NeuronCore Version | Chips |
|---|---|---|
| Trainium1 / Inferentia2 | NeuronCore-v2 | 2 cores/chip |
| Trainium2 | NeuronCore-v3 | 8 cores/chip |
| Trainium3 | NeuronCore-v4 | 8 cores/chip |

## Compute Engines (per NeuronCore)

Each NeuronCore contains four main engines that can execute concurrently:

### Tensor Engine (TensorE)
- **Power-optimized systolic array** for GEMM, convolution, and transpose operations.
- **Mixed-precision**: accepts lower-precision inputs and accumulates in higher precision.
- Outputs are always **FP32 or INT32** (v2/v3) or **FP32/BF16** (v4); explicit casting is needed for downstream lower-precision consumption.

| Version | FP8 TFLOPS | BF16/FP16/TF32 TFLOPS | FP32 TFLOPS | Notes |
|---|---|---|---|---|
| v2 (Trn1) | ~95 | ~95 | ~23.75 | Per core (chip: 190/47.5) |
| v3 (Trn2) | 158 | 79 | — | Structured sparsity up to 2× (316 TFLOPS) |
| v4 (Trn3) | 315 | 79 | 20 | MXFP4/MXFP8 at 315 TFLOPS; structured sparsity support |

- **Supported input types**: cFP8 (e5m2, e4m3, e3m4), FP16, BF16, TF32, FP32, INT8; v4 adds MXFP4/MXFP8.
- **Structured sparsity** (v3/v4): patterns 4:16, 4:12, 4:8, 2:8, 2:4, 1:4, 1:2 on one input tensor.
- **Key optimization rule**: Feed 16-bit or 8-bit data to the Tensor Engine for maximum throughput. FP32 matmul is 4× slower than BF16/FP16.

### Vector Engine (VectorE)
- Operations where each output depends on **multiple inputs**: LayerNorm, pooling, reductions, axpy (Z=aX+Y).
- v2: **2.3 TFLOPS FP32**; v3: **1 TFLOPS FP32**; v4: **1.2 TFLOPS FP32**.
- Supported types: cFP8, FP16, BF16, TF32, FP32, INT8, INT16, INT32.
- v4 adds fast exponential (4× faster than ScalarE) and MXFP8 quantization from BF16/FP16.

### Scalar Engine (ScalarE)
- **Element-wise** operations (each output depends on one input): GELU, sigmoid, exp, ReLU, reciprocal.
- v2: **2.9 TFLOPS FP32**; v3: **1.2 TFLOPS FP32**; v4: **1.2 TFLOPS FP32**.
- Same data type support as VectorE.

### GPSIMD Engine (GpSimdE)
- **Eight fully-programmable 512-bit wide vector processors** per NeuronCore.
- Can execute **general-purpose C/C++ code** with direct access to on-chip SRAM.
- This is the mechanism by which NKI kernels execute custom logic on-chip.

### Additional Engines
- **Sync Engine**: Synchronization and DMA triggering.
- **Collective Communication Engine (CCE)**: Dedicated hardware for collective operations (AllReduce, AllGather); executes on hardware **separate from compute engines**, enabling compute-communication overlap.

## Memory Hierarchy

### On-Chip Memory (Software-Managed, Not Hardware-Cached)

| Memory | Description | Key Properties |
|---|---|---|
| **SBUF (State Buffer)** | Main on-chip working memory | All engines read from SBUF; software must explicitly manage data movement |
| **PSUM (Partial Sum Buffer)** | Secondary on-chip memory | Has **near-memory accumulation** support — TensorE can write-accumulate directly into PSUM without separate read-add-write cycles |

SBUF is the primary staging area for data. The programmer (or compiler) must explicitly orchestrate DMA transfers between HBM and SBUF.

**On-chip SRAM sizes per NeuronCore:**

| Version | SRAM per Core | Total per Chip |
|---|---|---|
| v2 (Trn1) | Not publicly specified (estimated ~24 MiB) | ~48 MiB |
| v3 (Trn2) | **28 MiB** | 224 MiB |
| v4 (Trn3) | **32 MiB** | 256 MiB |

**SBUF layout support** (v3+): Row-major, Col-major-2B, Col-major-4B. Column-major layouts enable more efficient access patterns for certain operations without explicit transpose.

### Off-Chip Memory (HBM)

| Chip | HBM Capacity | HBM Bandwidth | DMA Bandwidth |
|---|---|---|---|
| Trainium1 (Trn1) | 32 GiB/chip (16 GiB/core) | 820 GiB/s | ~1 TB/s (with inline compression) |
| Trainium2 (Trn2) | 96 GiB/chip (24 GiB per 2-core bank) | 2.9 TB/s | 3.5 TB/s (with inline compression) |
| Trainium3 (Trn3) | 144 GiB/chip | 4.9 TB/s | 4.9 TB/s (with inline computation) |

DMA supports **inline memory compression/decompression**, which can effectively increase usable bandwidth beyond raw HBM throughput. v4 adds **near-memory accumulation in SRAM** via DMA (read-add-write in a single transfer).

## Key Performance Ratios and Roofline Characteristics

The arithmetic intensity (ops/byte) required to be compute-bound rather than memory-bandwidth-bound:

| Chip | BF16 Compute | HBM BW | AI Crossover (BF16) |
|---|---|---|---|
| Trn1 | 190 TFLOPS/chip | 820 GB/s | ~232 ops/byte |
| Trn2 | 667 TFLOPS/chip | 2.9 TB/s | ~230 ops/byte |
| Trn3 | 671 TFLOPS/chip | 4.9 TB/s | ~137 ops/byte |

These are very high arithmetic intensity requirements, meaning **many operations will be memory-bandwidth-bound**. Kernels must maximize data reuse through tiling, fusion, and keeping data in SBUF/PSUM to avoid HBM round-trips.

## Supported Data Types

| Type | Bits | Exponent | Mantissa | Notes |
|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | Full precision; 4× slower on TensorE |
| TF32 | 19 | 8 | 10 | FP32 range, reduced precision |
| BF16 | 16 | 8 | 7 | FP32 range, low precision; default auto-cast target |
| FP16 | 16 | 5 | 10 | Higher precision than BF16 but limited range (±65504) |
| FP8 (e5m2) | 8 | 5 | 2 | Configurable FP8 variant |
| FP8 (e4m3) | 8 | 4 | 3 | Configurable FP8 variant |
| FP8 (e3m4) | 8 | 3 | 4 | Configurable FP8 variant |
| MXFP8/MXFP4 | 8/4 | — | — | v4 only; OCP-compliant microscaling formats |
| INT8/INT16/INT32 | 8/16/32 | — | — | Integer types for Vector/Scalar engines |

**Default compiler behavior**: FP32 matmul operations are auto-cast to **BF16** for performance. The compiler preserves I/O tensor types.

**Rounding modes**: Round-to-Nearest-Even (RNE, default) and **Stochastic Rounding** (SR). SR is important for training convergence with reduced-precision accumulations — it prevents small values from being systematically rounded away during accumulation into large-magnitude numbers.

## Key Constraints for NKI Kernel Optimization

1. **Software-managed SRAM**: SBUF and PSUM are not hardware-cached. Kernels must explicitly manage all data movement (DMA from HBM → SBUF, compute, DMA from SBUF → HBM). Efficient tiling and prefetching are critical.

2. **TensorE accumulates in FP32**: Matrix multiply always produces FP32/INT32 output regardless of input precision. Explicit casting is needed before storing results in lower precision.

3. **Transpose cost**: Fast transpose uses the TensorE in FP16/BF16 (fast path). Byte-level transpose (lossless, full precision) is significantly slower.

4. **Precision-performance tradeoff**: Using BF16/FP16/FP8 for TensorE inputs provides up to 4× throughput over FP32. FP16 risks overflow beyond ±65504; BF16 is safer for unknown value ranges.

5. **PSUM near-memory accumulation**: The Partial Sum Buffer supports direct accumulation from TensorE output, avoiding separate read-modify-write cycles. This is critical for efficient matmul accumulation patterns.

6. **Compute-communication overlap**: Collective communication runs on dedicated CCE hardware, not shared with compute engines. NKI kernel execution is not blocked by concurrent collective operations.

7. **Single-NeuronCore execution**: Each NKI kernel executes on one NeuronCore. Cross-core parallelism is handled at the framework level, not within the kernel.

8. **Logical NeuronCore configuration** (Trn2/Trn3): Multiple physical NeuronCores can be combined into a single logical NeuronCore (e.g., LNC=2 on Trn2 combines 2 physical cores sharing a 24 GiB HBM bank). The compiler LNC setting must match the runtime configuration.

9. **Fixed compilation**: Models/kernels are compiled for specific shapes and configurations. Dynamic shapes are supported at the ISA level (v2+) but sequence bucketing is common in practice.

10. **Batching amortizes parameter reads**: The NeuronCore reads parameters once from HBM and computes across all batch samples before reading the next layer's parameters. Larger batches improve hardware utilization up to the compute-bound ceiling.