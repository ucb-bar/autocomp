## cublas_tensor_core_gemm

SUMMARY: Using Tensor Cores in cuBLAS for FP16/FP32 GEMM. Shows how to opt in via CUBLAS_TENSOR_OP_MATH and the dimension multiples-of-8 requirements that enable the Tensor Core path in cublasGemmEx.

```cpp
// First, create a cuBLAS handle:
cublasStatus_t cublasStat = cublasCreate(&handle);

// Set the math mode to allow cuBLAS to use Tensor Cores:
cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Allocate and initialize your matrices (only the A matrix is shown):
size_t matrixSizeA = (size_t)rowsA * colsA;
T_ELEM_IN **devPtrA = 0;

cudaMalloc((void**)&devPtrA[0], matrixSizeA * sizeof(devPtrA[0][0]));
T_ELEM_IN A  = (T_ELEM_IN *)malloc(matrixSizeA * sizeof(A[0]));

memset( A, 0xFF, matrixSizeA* sizeof(A[0]));
status1 = cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);

// ... allocate and initialize B and C matrices (not shown) ...

// Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8,
// and m is a multiple of 4:
cublasStat = cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                          A, CUDA_R_16F, lda,
                          B, CUDA_R_16F, ldb,
                          beta, C, CUDA_R_16F, ldc, CUDA_R_32F, algo);
```

A few simple rules:

- The routine must be a GEMM. Currently, only GEMMs support Tensor Core execution.
- The math mode must be set to `CUBLAS_TENSOR_OP_MATH`. Floating point math is not associative, so the results of the Tensor Core math routines are not bit-equivalent to the analogous non-Tensor Core routines. cuBLAS requires you to opt in.
- All of `k`, `lda`, `ldb`, and `ldc` must be a multiple of eight; `m` must be a multiple of four. The Tensor Core math routines stride through input data in steps of eight values.
- The input and output data types for the matrices must be either half-precision or single-precision (`CUDA_R_16F` or `CUDA_R_32F`).
- GEMMs that do not satisfy these rules fall back to a non-Tensor Core implementation.

## cublaslt_int8_matmul

SUMMARY: Tensor-op INT8 GEMM via cuBLASLt with the COL32 / COL4_4R2_8C memory transforms required by the IGEMM kernels. End-to-end pattern: create matmul/layout/transform descriptors, transform A and B, run cublasLtMatmul, transform C back.

```cpp
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed,
/// alpha assumed 1, beta assumed 0, stream assumed 0
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const int8_t *A, int lda,
                   const int8_t *B, int ldb,
                   int32_t *C, int ldc) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = NULL, *Btransform = NULL;
    int32_t *Ctransform = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform);
    cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform);
    cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform);

    cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    // tensor op igemm kernels only support NT gemm
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    // create descriptors for original matrices
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc);

    // create descriptors for transformed matrices (COL32 for A and C; COL4_4R2_8C for B on Turing)
    cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform);
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform);
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
    cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform);
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    // transforms and computation
    cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0);
    // B matrix is non-transposed, but transposed matrix is needed - add transpose op in matrix transform.
    cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose));
    cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0);

    // no need to transform C matrix as beta is assumed to be 0
    cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                   Atransform, AtransformDesc, Btransform, BtransformDesc,
                   &beta, Ctransform, CtransformDesc, Ctransform, CtransformDesc,
                   NULL, NULL, 0, 0);

    opTranspose = CUBLAS_OP_N;
    cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose));
    // transform outputs to COL order
    cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0);

    if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
    if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
    if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
    if (transformDesc) cublasLtMatrixTransformDescDestroy(transformDesc);

    cudaDeviceSynchronize();
    if (Ctransform) cudaFree(Ctransform);
    if (Btransform) cudaFree(Btransform);
    if (Atransform) cudaFree(Atransform);
}
```

## cudnn_tensor_core_conv

SUMMARY: Using Tensor Cores in cuDNN for convolution. Shows the descriptor setup, how to opt in via CUDNN_TENSOR_OP_MATH, the IMPLICIT_PRECOMP_GEMM algorithm choice, and the channel-multiple-of-8 requirement.

```cpp
// Create a cuDNN handle:
cudnnCreate(&handle_);

// Create your tensor descriptors:
cudnnCreateTensorDescriptor(&cudnnIdesc);
cudnnCreateFilterDescriptor(&cudnnFdesc);
cudnnCreateTensorDescriptor(&cudnnOdesc);
cudnnCreateConvolutionDescriptor(&cudnnConvDesc);

// Set tensor dimensions as multiples of eight (only the input tensor is shown here):
int dimA[]    = {1, 8, 32, 32};
int strideA[] = {8192, 1024, 32, 1};
cudnnSetTensorNdDescriptor(cudnnIdesc, getDataType(), convDim+2, dimA, strideA);

// Allocate and initialize tensors (again, only the input tensor is shown):
cudaMalloc((void**)&(devPtrI), insize * sizeof(devPtrI[0]));
hostI = (T_ELEM*)calloc(insize, sizeof(hostI[0]));
initImage(hostI, insize);
cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice);

// Set the compute data type (below as CUDNN_DATA_FLOAT):
cudnnSetConvolutionNdDescriptor(cudnnConvDesc, convDim, padA, convstrideA, dilationA,
                                CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

// Set the math type to allow cuDNN to use Tensor Cores:
cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH);

// Choose a supported algorithm:
cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

// Allocate your workspace:
cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc,
                                        cudnnOdesc, algo, &workSpaceSize);
if (workSpaceSize > 0) cudaMalloc(&workSpace, workSpaceSize);

// Invoke the convolution:
cudnnConvolutionForward(handle_, &alpha, cudnnIdesc, devPtrI,
                        cudnnFdesc, devPtrF, cudnnConvDesc, algo,
                        workSpace, workSpaceSize, &beta,
                        cudnnOdesc, devPtrO);
```

A few simple rules:

- The convolution algorithm must be `ALGO_1` (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` for forward).
- The math type must be set to `CUDNN_TENSOR_OP_MATH`.
- Both input and output channel dimensions must be a multiple of eight.
- The input, filter, and output data types for the convolutions must be half-precision.
- Convolutions that do not satisfy these rules fall back to a non-Tensor Core implementation.

## cuda_wmma_walkthrough

SUMMARY: Using Tensor Cores directly from CUDA C++ via the nvcuda::wmma API. Walks through fragment declarations, fill_fragment, the K-loop with load_matrix_sync / mma_sync, and the alpha*A*B + beta*C epilogue with store_matrix_sync.

```cpp
#include <mma.h>
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c,
                             int M, int N, int K,
                             float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}
```

Notes:

- The fragment template parameters are: which matrix (`matrix_a` / `matrix_b` / `accumulator`), the WMMA tile shape, the input dtype, and (for A and B only) row- vs. column-major layout. The example does no transposition so both A and B are column-major.
- The third argument to `load_matrix_sync` is the leading dimension in memory of the source matrix.
- `mma_sync` accumulates in place, so the first and last arguments are both the accumulator fragment.

## cuda_wmma_full_driver

SUMMARY: A complete standalone HMMA/IMMA driver: a templated WMMA kernel parameterised on dtypes, layouts, transposes; a launcher that picks the right specialization; CPU reference matmul; and a main() that benchmarks both FP16 and INT8 WMMA paths. Useful as a copy-and-modify template when the agent needs to write a self-contained Tensor Core kernel from scratch.

```cpp
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mma.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100, int num_warmups = 100) {
    cudaEvent_t start, stop;
    float time;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    for (int i = 0; i < num_warmups; ++i) bound_function(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < num_repeats; ++i) bound_function(stream);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return time / num_repeats;
}

template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
    T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose,
    bool is_B_transpose, float alpha, float beta) {
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_A> a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_B> b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> c_frag{};
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    for (uint32_t ki = 0; ki < k; ki += WMMA_K) {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M : ki};
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki : warpN * WMMA_N};

        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n)) {
            T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx + matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx + matrix_mma_b_col_idx * ldb};
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};
    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n) {
        T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx + matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc, nvcuda::wmma::mem_col_major);
        for (uint32_t i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}

template <typename T1, typename T2>
void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream) {
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};
    constexpr int WARP_SIZE{32};

    // Block size of 128x4 means 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    dim3 blockDim, gridDim;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    if ((!is_A_transpose) && (!is_B_transpose)) {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major, nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose, is_B_transpose, alpha, beta);
    } else if ((is_A_transpose) && (!is_B_transpose)) {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major, nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose, is_B_transpose, alpha, beta);
    } else if ((!is_A_transpose) && (is_B_transpose)) {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major, nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose, is_B_transpose, alpha, beta);
    } else {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major, nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose, is_B_transpose, alpha, beta);
    }
}
```
