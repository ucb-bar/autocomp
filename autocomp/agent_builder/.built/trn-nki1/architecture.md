## NeuronCore-v2 Hardware Architecture

AWS Trainium processors contain multiple NeuronCore-v2 accelerators, each capable of executing NKI (Neuron Kernel Interface) kernels independently. Understanding the hardware structure is essential for writing efficient NKI code.

### NeuronCore Components

Each NeuronCore-v2 contains:

1. **Instruction Fetch & Control**: Fetches and executes NKI instructions sequentially within a single kernel instance (no cross-core parallelism).

2. **On-Chip Memory Hierarchy**:
   - **SBUF (Scratchpad Buffer)**: 24 MiB total, organized as 128 partitions × ~192 KiB each (the per-partition free-dimension budget). Software-managed; stores input tiles, weight tiles, and intermediate results.
   - **PSUM (Partial Sum Buffer)**: 2 MiB total, organized as 128 partitions × 16 KiB each, divided into 8 banks per partition (each bank holds up to 512 FP32 values). Dedicated accumulation buffer for Tensor Engine matmul results.
   - **HBM (High Bandwidth Memory)**: Off-chip device memory (32 GiB per device). Accessed via DMA load/store.

3. **Four Compute Engines**: Execute different instruction types in parallel within the same NeuronCore.
   - **Tensor Engine**: Matrix multiplication (matmul, transpose) on tiled matrices. Highest throughput for FLOPS.
   - **Vector Engine**: Element-wise operations, reductions, statistics (bn_stats, bn_aggr). Efficient for per-partition parallel work.
   - **Scalar Engine**: Activation functions with optional scaling, bias, and reduction. Executes the same operation across all 128 partitions in parallel.
   - **GpSimd Engine**: Specialized for select operations, gathering, and affine predicates. Lower throughput but useful for conditionals and complex indexing.

### Memory Layout and Partitioning

NKI tensors follow a strict 2D physical layout model:

- **Partition Dimension (P)**: First logical dimension, maps to SBUF's 128 physical partitions. Each element in this dimension goes to a different partition.
- **Free Dimension (F)**: Remaining dimensions, laid out contiguously within each partition.

A **Tile** is a tensor whose first dimension is the partition dimension. All compute instructions require tiles as input.

Key constraints:
- Partition dimension ≤ 128 (`nl.tile_size.pmax`; SBUF/PSUM physical limit).
- SBUF per-partition free-dimension budget ≈ 192 KiB.
- PSUM combined free dimensions ≤ 512 FP32 elements per bank, 8 banks per partition.
- Matmul: LHS free dimension ≤ 128, RHS free dimension ≤ 512.

### Typical Kernel Flow

1. **Load phase**: Transfer data from HBM to SBUF using `nl.load()` (DMA).
2. **Compute phase**: Process tiles using Tensor, Vector, Scalar, or GpSimd engines. Results accumulate in PSUM (for matmul) or SBUF (for element-wise).
3. **Store phase**: Write results from SBUF/PSUM back to HBM using `nl.store()` (DMA).

### Performance Principles

1. **Minimize HBM traffic**: Reuse loaded data as much as possible. Keep data in SBUF across outer loop iterations.
2. **Maximize engine utilization**: Keep all four engines busy through pipelining and overlap of load/compute/store.
3. **Respect layout constraints**: Partition dimension should contain the parallel axis for non-matmul ops, and the contraction dimension for matmul.
4. **Use largest tiles**: Tensor Engine is most efficient with 128x128 (stationary) and 128x512 (moving) tiles.
5. **Avoid redundant transposes**: Use `nl.matmul(transpose_x=True)` or `nisa.nc_transpose()` judiciously.
