# Trainium 1 (NeuronCore-v2) Hardware Architecture Summary

## Overview

Trainium 1 (Trn1) is AWS's first-generation training accelerator built around the **NeuronCore-v2** microarchitecture. Each Trainium chip contains **2 NeuronCores**. The largest instance, **trn1.32xlarge**, has 16 Trainium chips providing **32 NeuronCores** total, interconnected via a **2D torus topology**. NeuronCores are individually addressable, and each NKI kernel executes on a **single NeuronCore** — there is no cross-core pipelining on NeuronCore-v2, so all kernel optimization must target single-core performance.

## Programming Model

Code is compiled by the **neuronx-cc** compiler (using XLA HLO as intermediate representation) into **NEFF** (Neuron Executable File Format) binaries that execute on NeuronCores. NeuronCore-v2 supports **control flow and dynamic shapes** (unlike v1). However, the execution model is fundamentally **graph-based**: computational graphs are compiled ahead of time, and static, known-at-compile-time tensor shapes are strongly preferred to avoid recompilation. NKI (Neuron Kernel Interface) provides direct kernel-level programming of a single NeuronCore's compute engines and memory hierarchy.

## Memory Hierarchy

- **HBM (High Bandwidth Memory)**: Off-chip device DRAM where model weights, activations, and compiled artifacts are stored. This is the largest but slowest tier of device memory. Compiled models are staged in device DRAM before execution. Device memory is a constrained resource — large models or disabled parameter aliasing can cause out-of-device-memory errors.

- **SBUF (State Buffer)**: On-chip SRAM serving as the primary working memory for compute engines. Data must be loaded from HBM into SBUF (via DMA) before computation. SBUF is the input source for both the tensor engine and vector engine.

- **PSUM (Partial Sum Buffer)**: On-chip accumulation buffer used by the matrix multiply (tensor) engine. Accumulations are performed in **FP32** regardless of input data type. Results in PSUM are typically moved to SBUF for further processing or written back to HBM.

- **Host Memory**: CPU-side memory. Data transfers between host and NeuronCores occur via a distinct transfer path. Minimizing host-device transfers and aggregating them improves performance. Pre-casting tensors to reduced precision (FP16/BF16) before transfer avoids FP32 transfer bottlenecks.

NeuronCore-v2 has **more memory per core and improved memory bandwidth** compared to NeuronCore-v1.

## Compute Engines

### Tensor Engine (Matrix Multiply Engine)
- Performs matrix multiplication operations.
- **Native input data types**: FP16, BF16, TF32, FP32 — but **FP16 and BF16 provide significantly higher throughput** than FP32/TF32.
- **Accumulation is always in FP32** (partial sums stored in PSUM).
- By default, the compiler auto-casts FP32 matmuls to BF16 for performance.
- Can also perform **fast transpose** via matmul in FP16/BF16 (much faster than byte-level data movement, but introduces casting precision loss).

### Vector Engine (Activation/Element-wise Operations)
- Handles activations, element-wise operations, and general vector computations.
- Supports **FP32, TF32, FP16, and BF16**.
- Operations like Softmax and SiLU have specialized compiler lowerings, indicating they are performance-critical paths.

### GPSIMD Engine (General Purpose SIMD)
- A **fully programmable** SIMD engine, new in NeuronCore-v2 (not present in v1).
- Provides flexibility for custom operations that don't map directly to the tensor or vector engines.

### Scalar Engine
- Handles scalar operations and control flow.

## Supported Data Types

| Type | Bits | Range | Notes |
|------|------|-------|-------|
| FP32 | 32 | ±3.40E+38 | Full precision; slower on matmul engine |
| TF32 | 19 effective | ±3.40E+38 | Compromise between FP32 range and reduced compute cost |
| BF16 | 16 | ±3.40E+38 | Preferred for training; covers full FP32 range |
| FP16 | 16 | ±65504 | Better precision than BF16 for mid-range values; limited range |
| FP8 | 8 | — | Supported by runtime for stochastic rounding targets |

**Stochastic rounding** is a hardware-level feature (runtime-controlled) that applies to all internal FP32 → reduced precision casts (FP16, BF16, FP8, TF32). It improves training convergence for reduced-precision workloads.

## Key Constraints and Optimization Considerations

1. **Prefer BF16/FP16 for matmul inputs**: The tensor engine achieves maximum throughput with 16-bit inputs. FP32 matmuls are significantly slower and are auto-cast to BF16 by default.

2. **Single NeuronCore execution**: Each NKI kernel runs on one NeuronCore. All optimization (tiling, data movement, compute scheduling) targets single-core resources.

3. **Data movement is critical**: Data must be explicitly moved between HBM → SBUF → compute engines → SBUF → HBM. Minimizing data movement and maximizing data reuse in SBUF is essential for performance.

4. **Fast transpose via matmul**: Transpose operations done through the tensor engine (FP16/BF16) are much faster than byte-level data movement. Use when precision loss is acceptable.

5. **Static shapes preferred**: The compiler and hardware are optimized for statically known tensor shapes. Dynamic shapes trigger recompilation.

6. **FP32 accumulation is free**: The tensor engine always accumulates in FP32 regardless of input precision, so there is no throughput penalty for FP32 partial sums.

7. **Pre-cast I/O tensors**: The compiler preserves input/output tensor types. Casting to FP16/BF16 in the framework before compilation avoids FP32 transfer overhead and enables faster execution.

8. **Multi-core allocation constraints**: For collective communication workloads, valid NeuronCore counts are **1, 2, 8, or 32**, and allocations must start at core IDs that are multiples of the count.

9. **No PagedAttention**: Memory management for attention/KV-cache must use alternative strategies.

10. **Sequence bucketing**: Input sequence lengths must fit pre-compiled buckets; padding to fixed lengths avoids recompilation.