## Hardware Architecture Summary: NVIDIA GPUs

This agent targets NVIDIA discrete GPUs running CUDA workloads. The exact SKU is provided via the `CudaHardwareConfig` (e.g. `NVIDIA L40S`, with a specific PyTorch version and CUDA toolkit version). Cross-architecture concepts that apply to virtually all current NVIDIA GPUs are summarized below.

### Execution Model

NVIDIA GPUs execute kernels as a 1D, 2D, or 3D grid of thread blocks. Each block contains up to 1024 threads, organized into 32-thread warps that execute in lockstep on a single Streaming Multiprocessor (SM). Warps on an SM share the SM's register file, shared memory (SMEM), L1 cache, and warp schedulers; warp scheduling hides memory latency by switching to other ready warps.

### Memory Hierarchy

- **Global memory (HBM/GDDR)** — large (tens of GB) but high latency. Coalesced, contiguous accesses by warps achieve peak bandwidth; scattered accesses degrade throughput sharply.
- **L2 cache** — device-wide, hundreds of MB on data-center SKUs. Persistent L2 windows can pin frequently-reused tensors.
- **Shared memory / L1** — per-SM, ~100-200 KB, configurable split. Software-managed cache; bank-conflict-free patterns are essential.
- **Registers** — per-thread, ~64K 32-bit registers per SM. Excessive register usage causes spills to local memory (slow) and reduces occupancy.
- **Constant and texture memory** — small, read-only, broadcast-friendly.

### Compute Throughput

Modern NVIDIA SMs combine general-purpose CUDA cores with **Tensor Cores** that execute mixed-precision matrix-multiply-accumulate (MMA) operations. Tensor Cores deliver an order of magnitude more FLOPS than CUDA cores for supported shapes and dtypes (FP16, BF16, TF32, FP8 on Hopper/Ada, INT8). They can be reached through:

- High-level libraries: cuBLAS / cuBLASLt for GEMMs, cuDNN for convolutions, PyTorch's tensor-core paths (e.g. `torch.backends.cuda.matmul.allow_tf32`, `torch.amp`).
- Low-level: the `nvcuda::wmma` and PTX `mma` instructions in CUDA C++.

To use Tensor Core paths in cuBLAS/cuDNN you generally must opt in (`CUBLAS_TENSOR_OP_MATH` / `CUDNN_TENSOR_OP_MATH`), use supported dtypes (FP16, BF16, etc.), and ensure shape constraints (e.g. K, lda, ldb, ldc multiples of 8; channel dims multiples of 8 for cuDNN).

### Asynchronous Execution

CUDA streams allow overlapping kernel execution with host↔device memory copies. Pinned (page-locked) host memory is required for true async copies. CUDA Graphs let you capture a sequence of CUDA work and replay it with much lower per-launch overhead — particularly useful for inference and small kernels.

### Common Bottlenecks

- **Memory bandwidth** for elementwise and small-arithmetic-intensity kernels.
- **Launch overhead** for many small kernels — fuse, batch, or capture into a CUDA Graph.
- **Warp divergence** when threads in a warp take different control-flow paths.
- **Bank conflicts** in shared memory.
- **Tensor Core under-utilization** caused by shape misalignment, wrong dtype, or layout mismatch.
- **Register pressure / occupancy** trade-offs.

### Optimization Principles

The agent's job is to produce code that runs faster on the target GPU while preserving the original solution's semantics within a small numerical tolerance. The most reliable optimizations are usually: route GEMMs and convolutions through cuBLAS/cuDNN/Tensor Cores; use lower-precision compute where the original allows it; fuse operations to reduce launch and memory traffic; convert hot Python loops to compiled CUDA / Triton kernels; and capture stable subgraphs into CUDA Graphs to amortize launch overhead.
