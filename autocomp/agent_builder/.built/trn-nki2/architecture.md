## AWS Trainium NeuronCore Architecture with NKI v2

Trainium chips contain NeuronCores optimized for high-performance tensor operations. The NKI v2 ISA provides a Python-based interface to program these cores.

### NeuronCore Hardware Structure

Each Trainium has 2 NeuronCores. Each NeuronCore contains:

- **SBUF (Scratchpad Buffer)**: 24 MiB total, organized as 128 partitions × ~192 KiB each (the per-partition free-dimension budget). Software-managed scratchpad used for tile loads from HBM and intermediate compute results. Partitions map to the first (partition) dimension of tiles.
- **PSUM (Partial Sum Buffer)**: 2 MiB total, organized as 128 partitions × 16 KiB each (8 banks per partition, each bank holds up to 512 FP32 values). Dedicated accumulation buffer for Tensor Engine matmul results.
- **Four Compute Engines**:
  - **Tensor Engine (nc_matmul)**: Optimized for matrix multiplication, supports stationary and moving operand dataflows.
  - **Vector Engine**: Element-wise operations and activation functions on tile data.
  - **Scalar Engine**: Scalar-tile operations and reductions.
  - **GpSimd Engine**: General-purpose SIMD for complex logic.

### Kernel Structure and Data Flow

Every NKI kernel follows a three-stage pipeline:

1. **Load**: Transfer data tiles from HBM to SBUF via `nisa.dma_copy()`.
2. **Compute**: Operate on tiles in SBUF/PSUM using ISA APIs.
3. **Store**: Transfer results from SBUF back to HBM via `nisa.dma_copy()`.

### Tensor and Tile Concepts

- **Tensor**: An NKI array in HBM, SBUF, or PSUM with a canonical 2D layout (partition dimension P + free dimension F).
- **Tile**: A tensor whose first dimension is the partition dimension; required for all ISA compute operations.
- **Partition Dimension**: P ≤ 128 (hardware limit). Maps to SBUF partitions.
- **Free Dimension**: F ≤ 512 in PSUM, unconstrained in SBUF.

### Matmul Layout Constraint

For `nki.isa.nc_matmul`, the contraction dimension (K) must be the partition dimension:
- **LHS layout**: [K, M] (K=partition, M=free)
- **RHS layout**: [K, N] (K=partition, N=free)
- **Output**: [M, N] in PSUM

This ensures hardware utilization across all compute lanes.

### Key Indexing Rules

- Use integer slicing (e.g., `t[0:128, 0:512]`) for contiguous access and DMA transfers.
- Affine_range loop variables cannot be used as list indices; use slicing instead.
- Avoid loop-carried dependencies when using affine_range (associative reductions allowed).
- Use `nl.sequential_range()` when iterations depend on prior results.

### Performance Optimization Principles

- **Minimize data movement**: Load each DRAM block once and reuse in SBUF across multiple compute iterations.
- **Maximize tile reuse**: Use outer loops to load larger tiles and inner loops to compute on them repeatedly.
- **Overlap engines**: Pipeline DMA loads, compute, and stores using affine_range and double-buffering.
- **Avoid partial tiles**: Prefer divisible loop bounds; handle remainder with `min()` slicing.
- **Keep data in SBUF/PSUM**: Avoid round-tripping to HBM unnecessarily.

### Beta 2 API Changes

NKI v2 (Beta 2) introduced breaking changes from Beta 1:

- Use `nki.isa` (not `nki.language`) for low-level operations.
- All ISA APIs require explicit `dst` parameter.
- `nisa.dma_copy()` replaces deprecated `nl.load()` and `nl.store()`.
- Masking is deprecated; use in-bounds slicing with `min()` instead.
- Dynamic slicing with `nl.ds()` for simple indexing patterns.
