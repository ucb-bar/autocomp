## CUDA Programming Interfaces

The CUDA agent targets a wide and well-documented programming surface. Rather than reproduce reference material here, this section enumerates the API families the agent should consider when planning. The model is expected to know the specific function signatures from its training data; if it does not, the planning prompt should include a note to fall back to a more conservative API.

### CUDA C++ Runtime API

Memory: `cudaMalloc`, `cudaMallocHost` (pinned), `cudaMallocManaged`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaFree`. Streams and events: `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaEventRecord`, `cudaEventElapsedTime`. Graphs: `cudaGraphCreate`, `cudaStreamBeginCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`. Compilation flags `-O3` and `--use_fast_math` are common when generating kernel code.

### CUDA C++ Kernel APIs

Thread/block primitives: `__shared__`, `__constant__`, `__restrict__`, `#pragma unroll`. Synchronization: `__syncthreads`, `__syncwarp`. Warp-level: `__shfl_sync`, `__shfl_xor_sync`, `__ballot_sync`, `__any_sync`. Atomics: `atomicAdd`, `atomicCAS`, etc. Async copy: `cuda::memcpy_async`, `__pipeline_memcpy_async`.

### Tensor Core APIs

`nvcuda::wmma` — header `<mma.h>`. Fragment types: `wmma::fragment<wmma::matrix_a, M, N, K, T, layout>` and `matrix_b`, `accumulator`. Operations: `wmma::load_matrix_sync`, `wmma::store_matrix_sync`, `wmma::fill_fragment`, `wmma::mma_sync`. Supported tile shapes (`M x N x K`) include `16 x 16 x 16` for FP16/BF16 and `8 x 32 x 16`, `32 x 8 x 16`. The full GEMM-on-WMMA pattern is shown in the `cuda_wmma_example` code example.

For Hopper and newer, lower-level `mma.sync` PTX instructions and the cuTLASS library expose Tensor Cores at finer granularity.

### cuBLAS and cuBLASLt

cuBLAS handles dense BLAS (GEMM is `cublasSgemm`, `cublasGemmEx`, etc.). To use Tensor Cores: create a handle, call `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)`, ensure `k`, `lda`, `ldb`, `ldc` are multiples of 8, and `m` is a multiple of 4. Inputs/outputs must be FP16 or FP32. cuBLASLt (`cublasLtMatmul`) supports more layouts and is the recommended path for INT8 GEMMs and for fine-grained algorithm selection. See the `cublas_example` and `cublaslt_example` code examples.

### cuDNN

cuDNN provides convolutions, pooling, normalization, activations, RNNs. Tensor descriptors: `cudnnCreateTensorDescriptor`, `cudnnSetTensorNdDescriptor`. For Tensor Core convolutions: set the math type via `cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)`, use FP16 inputs/filters/outputs, ensure both input and output channel dimensions are multiples of 8, and pick `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`. cuDNN also exposes fused convolution + bias + ReLU operations. See the `cudnn_example` code example.

### PyTorch / ATen

`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.amp.autocast`, `torch.cuda.amp.GradScaler` for low-precision compute. `torch.compile` for kernel fusion. `torch.utils.cpp_extension` for inline CUDA kernels. ATen `at::` functions are often called directly from C++ extensions for lower overhead than the Python `torch::` API.

### Triton

Triton kernels are written in Python and JIT-compiled. They expose explicit block/program IDs (`tl.program_id`), shared-memory-style loads (`tl.load`, `tl.store`), and `tl.dot` for tensor-core matmuls. Useful when a custom layout or fusion is needed but writing CUDA C++ is too heavy.
