## AWS NeuronCore Hardware Architecture Summary

### Overview

AWS Trainium and Inferentia are custom-built ML accelerators programmed via the Neuron Kernel Interface (NKI). NKI kernels execute on individual NeuronCores — fully independent heterogeneous compute units. The primary targets for NKI kernel optimization are **NeuronCore-v2** (Trainium/Inferentia2, Trn1/Inf2 instances) and **NeuronCore-v3** (Trainium2, Trn2 instances). Each NeuronCore contains four compute engines and a software-managed memory hierarchy with no hardware caches — all data movement is explicit in the program.

### Memory Hierarchy

The memory hierarchy is four levels deep, all software-managed (no hardware caches):

| Memory | Type | Capacity (per NeuronCore) | Bandwidth | Latency | Notes |
|--------|------|--------------------------|-----------|---------|-------|
| **PSUM** (Partial Sum Buffer) | On-chip, 2D (128 partitions) | Small (dedicated accumulation buffer) | Highest | Lowest | Reserved for Tensor Engine matmul outputs; supports read-add-write accumulation. Evict results to SBUF promptly. |
| **SBUF** (State Buffer) | On-chip, 2D (128 partitions) | NeuronCore-v2: ~24 MiB; NeuronCore-v3: 28 MiB | ~20× HBM bandwidth | Low | Main on-chip working memory. Accessible by all compute engines. All computation operands must reside here (or PSUM). |
| **HBM** (Device Memory) | Off-chip, linear | NeuronCore-v2: 16 GiB/core (32 GiB/chip); NeuronCore-v3: 12–24 GiB/core (96 GiB/chip) | NeuronCore-v2: 820 GiB/s/chip; NeuronCore-v3: 2.9 TB/s/chip | Medium | Kernel inputs/outputs reside here. Tensors stored in flattened row-major layout. |
| **Host Memory** | CPU DRAM | Instance-dependent (up to 2 TiB on Trn2) | Lowest | Highest | Not directly accessible from NKI kernels; framework handles host↔device transfers. |

**Key memory characteristics:**
- SBUF and PSUM are **2-dimensional**, organized into **128 partitions**. The first tensor dimension maps to partitions ("partition dimension"), the remaining dimensions are the "free dimension" within each partition.
- HBM is **linear** memory — multi-dimensional tensors are flattened.
- When on-chip data exceeds SBUF/PSUM capacity, the compiler inserts spills to HBM, which is costly. Careful tiling is essential.
- On Trn2 with LNC=1, two physical NeuronCores **share** a single 24 GiB HBM bank, creating potential noisy-neighbor memory pressure.
- HBM also holds scratchpad (compiler-managed spill space), model constants, DMA descriptors, and executable code.

### Compute Engines

Each NeuronCore has four specialized engines that execute concurrently on independent sequencers:

| Engine | Optimized For | Throughput (per NeuronCore) | Data Types | Key Operations |
|--------|--------------|---------------------------|------------|----------------|
| **Tensor Engine** | Matrix/tensor ops (systolic array) | v2: 95 FP16/BF16 TFLOPS; v3: 79 BF16/FP16 TFLOPS, 158 cFP8 TFLOPS | cFP8, FP16, BF16, TF32, FP32 inputs → FP32 outputs | GEMM, convolution, transpose. Writes to PSUM with accumulation. v3 adds structured sparsity (up to 2× throughput). |
| **Vector Engine** | Reduction/vector ops | v2: 2.3 FP32 TFLOPS; v3: 1 FP32 TFLOPS | cFP8, FP16, BF16, TF32, FP32, INT8/16/32 | LayerNorm, pooling, axpy (Z=aX+Y), reductions. Each output depends on multiple inputs. |
| **Scalar Engine** | Element-wise ops | v2: 2.9 FP32 TFLOPS; v3: 1.2 FP32 TFLOPS | cFP8, FP16, BF16, TF32, FP32, INT8/16/32 | Activation functions (GELU, sigmoid, exp), element-wise math. Each output depends on one input. |
| **GpSimd Engine** | General-purpose programmable | 8× 512-bit vector processors | Arbitrary (C code) | Custom operators, complex control flow, DMA triggers. Accesses SBUF directly. |

**Engine concurrency:** All four engines have independent instruction sequencers and can execute in parallel, synchronized via semaphores. Overlapping compute with DMA is a key optimization strategy.

### DMA Engines

Each NeuronCore has **16 DMA engines** for data movement:
- Bidirectional HBM↔SBUF, intra-HBM, and intra-SBUF transfers.
- Each DMA engine handles 8 of the 128 SBUF partitions (128/16 = 8).
- Per-engine bandwidth: ~27.2 GB/s (v2/v3), ~38.4 GB/s (v4).
- Support scatter-gather, inline data type casting, and transpose during transfer.
- **Optimal throughput** requires ≥4 KiB per partition in the free dimension and using all 128 partitions (all 16 engines active).
- Small, frequent transfers are latency-bound; batch into larger transfers when possible.
- DMA operates asynchronously from compute — overlap data loading with computation for best performance.

### Key Constraints and Optimization Considerations

1. **Partition dimension alignment:** SBUF/PSUM have exactly 128 partitions. The first dimension of on-chip tensors maps to partitions, so tile sizes in this dimension should be multiples of 128 (or at most 128) for full utilization.

2. **Tiling is critical:** On-chip SBUF is limited (24–28 MiB). Kernels must tile computations to fit working sets in SBUF, iterating over tiles with explicit DMA loads/stores. Poor tiling causes compiler-inserted spills to HBM.

3. **Compute-DMA overlap:** Since DMA engines and compute engines operate independently, double-buffering (loading the next tile while computing on the current one) is a primary optimization technique.

4. **Tensor Engine dominance:** The Tensor Engine has 10–40× the throughput of Vector/Scalar engines. Maximize time spent in matmul; minimize time in element-wise and reduction operations. Keep PSUM free by promptly moving matmul results to SBUF.

5. **DMA transfer sizing:** Maximize bytes per partition (≥4 KiB) and use all 128 partitions to saturate DMA bandwidth. Reshape/fold tensors to achieve this when natural shapes are suboptimal.

6. **Free dimension contiguity:** HBM tensors are linear/row-major. DMA transfers are most efficient when the free dimension (innermost) is large and contiguous in HBM.

7. **Data types matter:** Lower precision (cFP8, BF16) doubles or more the Tensor Engine throughput versus FP32. Use the lowest precision that maintains acceptable accuracy. DMA can cast between types inline.

8. **Structured sparsity (v3+):** The Tensor Engine supports M:N sparsity patterns (e.g., 2:4), delivering up to 2× effective TFLOPS when applicable.

9. **Software-managed memory:** There are no hardware caches. Every byte movement between memory levels must be explicitly programmed or will be inserted by the compiler. This gives full control but requires careful orchestration.

10. **Scratchpad/spill management:** Large intermediate tensors that don't fit in SBUF spill to HBM scratchpad. Scratchpad page size (default 512 MiB, configurable up to 3.5 GiB) affects memory efficiency — mismatched page sizes cause redundant private allocations.