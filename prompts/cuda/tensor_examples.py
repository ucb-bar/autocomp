def PROMPT():
    cublas_example = """How to use Tensor Cores in cuBLAS

You can take advantage of Tensor Cores by making a few changes to your existing cuBLAS code. The changes are small changes in your use of the cuBLAS API.

The following example code applies a few simple rules to indicate to cuBLAS that Tensor Cores should be used. These rules are enumerated explicitly after the code.
Example code

The following code example is largely the same as the common code used to invoke a GEMM in cuBLAS on previous architectures.
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
A few simple rules

cuBLAS users will notice a few changes from the existing cuBLAS GEMM code:

    The routine must be a GEMM. Currently, only GEMMs support Tensor Core execution.
    The math mode must be set to CUBLAS_TENSOR_OP_MATH. Floating point math is not associative, so the results of the Tensor Core math routines are not quite bit-equivalent to the results of the analogous non-Tensor Core math routines. cuBLAS requires you to opt-in to the use of Tensor Cores.
    All of k, lda, ldb, and ldc must be a multiple of eight; m must be a multiple of four. The Tensor Core math routines stride through input data in steps of eight values, so the dimensions of the matrices must be multiples of eight.
    The input and output data types for the matrices must be either half-precision or single-precision. (Only CUDA_R_16F is shown in the example, but CUDA_R_32F also is supported.)

GEMMs that do not satisfy these rules fall back to a non-Tensor Core implementation."""
    cudnn_example = """How to use Tensor Cores in cuDNN

Using Tensor Cores in cuDNN is also easy, and again involves only slight changes to existing code.
Example code

The example code for using Tensor Cores in cuDNN can be found in conv_sample.cpp in the cuDNN samples directory. We copied some excerpts in this post. The cuDNN samples directory is packaged with the documentation.
// Create a cuDNN handle:
checkCudnnErr(cudnnCreate(&handle_));
 
// Create your tensor descriptors:
checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));
 
// Set tensor dimensions as multiples of eight (only the input tensor is shown here):
int dimA[] = {1, 8, 32, 32};
int strideA[] = {8192, 1024, 32, 1};
 
checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, getDataType(), 
                                          convDim+2, dimA, strideA) );
 
// Allocate and initialize tensors (again, only the input tensor is shown):
checkCudaErr( cudaMalloc((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]) ));
hostI = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
 
initImage(hostI, insize);
 
checkCudaErr( cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice));
 
// Set the compute data type (below as CUDNN_DATA_FLOAT):
checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                               convDim,
                                               padA,
                                               convstrideA,
                                               dilationA,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_DATA_FLOAT) );
 
// Set the math type to allow cuDNN to use Tensor Cores:
checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
 
// Choose a supported algorithm:
cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
 
// Allocate your workspace:
checkCudnnErr( cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, 
                                                       cudnnFdesc, cudnnConvDesc,
                                                       cudnnOdesc, algo, &workSpaceSize) );
 
if (workSpaceSize > 0) {
   cudaMalloc(&workSpace, workSpaceSize);
}
 
// Invoke the convolution:
checkCudnnErr( cudnnConvolutionForward(handle_, (void*)(&alpha), cudnnIdesc, devPtrI,
                                       cudnnFdesc, devPtrF, cudnnConvDesc, algo,
                                       workSpace, workSpaceSize, (void*)(&beta),
                                       cudnnOdesc, devPtrO) );
A few simple rules

There are a few changes from the common cuDNN use:

    The convolution algorithm must be ALGO_1 (IMPLICIT_PRECOMP_GEMM for forward). Other convolution algorithms besides ALGO_1 may use Tensor Cores in future cuDNN releases.
    The math type must be set to CUDNN_TENSOR_OP_MATH. As in cuBLAS, the results of the Tensor Core math routines are not quite bit-equivalent to the results of the analogous non-tensor core math routines, so cuDNN requires you to opt-in to the use of Tensor Cores.
    Both input and output channel dimensions must be a multiple of eight.  Again as in cuBLAS, the Tensor Core math routines stride through input data in steps of eight values, so the dimensions of the input data must be multiples of eight.
    The input, filter, and output data types for the convolutions must be half-precision.

Convolutions that do not satisfy these rules fall back to a non-Tensor Core implementation.

The example code shows NCHW data format. For NHWC support as well, see the conv_sample.cpp sample."""
    cublaslt_example = """Example of using cuBLASLt for Tensor Core GEMM operations

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "sample_cublasLt_LtIgemmTensor.h"
#include "helpers.h"

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int lda,
                   const int8_t *B,
                   int ldb,
                   int32_t *C,
                   int ldc) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = NULL, *Btransform = NULL;
    int32_t *Ctransform                   = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));

    checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    // tensor op igemm kernels only support NT gemm
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));

    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    opTranspose = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order
    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));
}"""
    cuda_example = """While cuBLAS and cuDNN cover many of the potential uses for Tensor Cores, you can also program them directly in CUDA C++.

Tensor Cores are exposed in CUDA 9.0 through a set of functions and types in the nvcuda::wmma namespace. These enable you to load or initialize values into the special format required by the Tensor Cores, perform matrix multiply-accumulate (MMA) steps, and store values back out to memory.

During program execution, multiple Tensor Cores are used concurrently by a full warp. This enables the warp to perform a 16x16x16 MMA at very high throughput (Figure 5).
A warp performs D=A*B+C where A, B, C and D are 16x16 matrices.
Figure 5. A warp performs D=A*B+C where A, B, C and D are 16×16 matrices

There’s a change in numbering from Figure 1 to Figure 5, as multiple Tensor Core operations are combined by the WMMA API to perform 16×16 matrix-multiply and accumulate operations.

Here’s an example that shows how you can use the WMMA (Warp Matrix Multiply Accumulate) API to perform a matrix multiplication. This example is not tuned for high performance and mostly serves as a demonstration of the API. For an example of optimizations you might apply to this code to get better performance, see the cudaTensorCoreGemm sample in the CUDA Toolkit. For highest performance in production code, use cuBLAS, as described earlier.
Headers and namespaces

The WMMA API is contained within the mma.h header file. The complete namespace is nvcuda::wmma::*, but it’s useful to keep wmma explicit throughout the code, so we just be using the nvcuda namespace.
#include <mma.h>
using namespace nvcuda;
Declarations and initialization

The full GEMM specification enables the algorithm to work on transpositions of a or b, and for data strides to be larger than the strides in the matrix. For simplicity, assume that neither a nor b are transposed, and that the leading dimensions of the memory and the matrix are the same.

The strategy to employ is to have a single warp responsible for a single 16×16 section of the output matrix. By using a 2D grid and thread blocks, you can effectively tile the warps across the 2D output matrix.
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

Before the MMA operation is performed, the operand matrices must be represented in the registers of the GPU. As an MMA is a warp-wide operation these registers are distributed amongst the threads of a warp with each thread holding a fragment of the overall matrix. The mapping between individual matrix parameters to their fragments is opaque, so your program should not make assumptions about it.

In CUDA,, a fragment is a templated type with template parameters that describe the following:

    Which matrix the fragment holds (A, B, or accumulator)
    The shape of the overall WMMA operation
    The data type
    For A and B matrices, whether the data is row- or column-major

This last argument can be used to perform the transposition of either A or B matrices. This example does no transposition so both matrices are column-major, which is standard for GEMM.
// Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

The final part of the initialization step is to fill the accumulator fragment with zeros.
wmma::fill_fragment(acc_frag, 0.0f);
The inner loop

The strategy for the GEMM is to compute one tile of the output matrix per warp. To do this, loop over rows of the A matrix, and columns of the B matrix. This is along the K-dimension of both matrices and produces an MxN output tile. The load matrix function takes data from memory (global memory in this example, although it could be any memory space) and places it into a fragment.

The third argument to the load is the leading dimension in memory of the matrix. The 16×16 tile b being loaded is discontinuous in memory so the function has to know the stride between consecutive columns or rows, if these were row-major fragments.

The MMA call accumulates in place, so both the first and last arguments are the accumulator fragment previously initialized to zero.
// Loop over the K-dimension
for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;
     
    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
        wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}
Finishing up

acc_frag now holds the result for this warp’s output tile based on the multiplication of A and B. The complete GEMM specification enables this result to be scaled, and for it to be accumulated on top of a matrix in place.

One way to do this scaling is to perform element-wise operations on the fragment. Although the mapping from matrix coordinates to threads isn’t defined, element-wise operations don’t have to know this mapping so they can still be performed using fragments.

As such, it’s legal to perform scaling operations on fragments or to add the contents of one fragment to another, so long as those two fragments have the same template parameters. If the fragments have different template parameters, the results are undefined. Using this feature, load in the existing data in C and, with the correct scaling, accumulate the result of the computation so far with it.
// Load in current value of c, scale by beta, and add to result scaled by alpha
int cRow = warpM * WMMA_M;
int cCol = warpN * WMMA_N;
 
if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
     
    for(int i=0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

Finally, store the data out to memory. Again, the target pointer can be any memory space visible to the GPU and the leading dimension in memory must be specified. There is also an option to specify whether the output is to be written row or column major.
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}"""
    wmma_example = """#include <cassert>
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
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

// All the data in the matrices are stored in a column-major order,
// which is the consistent with most of the cuBLAS GEMM APIs.
// For matrix A of shape M x N, the leading dimension is M.
// For matrix A that is transposed and is of shape N x M,
// the leading dimension is N.
// Matrix A: M x K, or K x N (if transposed).
// Matrix B: K x M, or M x K (if transposed).
// Matrix C: M x N.
// WMMA_FRAG_LAYOUT_A: nvcuda::wmma::row_major if A is
// transposed, otherwise nvcuda::wmma::col_major.
// WMMA_FRAG_LAYOUT_B: nvcuda::wmma::row_major if B is
// transposed, otherwise nvcuda::wmma::col_major.
template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
    T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose,
    bool is_B_transpose, float alpha, float beta)
{
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_A>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_B>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        c_frag{};

    // Make sure the accumulator starts from 0.
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        // Determine the first element of the mma matrices on the linear memory.
        // Matrix A mma matrix
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
                                                           : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
                                                           : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
                                                           : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
                                                           : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            // Determine the memory address of the first element of the mma
            // matrices. Notice that all the matrices are assumed to be
            // column-major. Therefore, the indexing is different from the
            // row-major indexing that we commonly see.
            T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha.
    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
    {
        T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);
        // Let the compiler figure out how to do the elementwise operation.
        // Such elementwise operation can be scaling, accumulation,
        // quantization, etc.
        // https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#id40
        // Be careful when dealing with the integer types.
        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}

template <typename T1, typename T2>
void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // Block size of 128x4 means we have 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    // Each block has 4x4 warps, totalling 4x4x32 threads.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up.
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B^T
    else
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    CHECK_LAST_CUDA_ERROR();
}

// A and B are column-major matrices.
template <typename T1, typename T2>
void mm_a_col_major_b_col_major(T1 const* A, T1 const* B, T2* C, uint32_t m,
                                uint32_t n, uint32_t k, uint32_t lda,
                                uint32_t ldb, uint32_t ldc, bool is_A_transpose,
                                bool is_B_transpose)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            // Compute C[mi, ni]
            T2 accum{0};
            // C = A * B
            if ((!is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ki, ni]
                    accum += A[ki * lda + mi] * B[ni * ldb + ki];
                }
            }
            // C = A^T * B
            else if ((is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ki, ni]
                    accum += A[mi * lda + ki] * B[ni * ldb + ki];
                }
            }
            // C = A * B^T
            else if ((!is_A_transpose) && (is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ni, ki]
                    accum += A[ki * lda + mi] * B[ki * ldb + ni];
                }
            }
            // C = A^T * B^T
            else
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ni, ki]
                    accum += A[mi * lda + ki] * B[ki * ldb + ni];
                }
            }
            C[ni * ldc + mi] = accum;
        }
    }
}

template <typename T1, typename T2>
void launch_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
               uint32_t k, bool is_A_transpose, bool is_B_transpose)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    mm_a_col_major_b_col_major(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose,
                               is_B_transpose);
}

void fill_random_float_values(float* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int8_values(int8_t* arr, size_t n,
                             std::default_random_engine& e)
{
    std::uniform_int_distribution<int8_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int32_values(int32_t* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_int_distribution<int32_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void float2half(__half* half_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
    }
}

template <typename T>
float get_avg_abs_diff_ratio(T const* arr_1, T const* arr_2, size_t n)
{
    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < n; ++i)
    {
        sum_abs_diff_ratio += std::abs(static_cast<float>(arr_1[i]) -
                                       static_cast<float>(arr_2[i])) /
                              std::abs(static_cast<float>(arr_1[i]) +
                                       static_cast<float>(arr_2[i]));
    }
    return sum_abs_diff_ratio / n;
}

template <typename T>
bool array_equal(T const* arr_1, T const* arr_2, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (arr_1[i] != arr_2[i])
        {
            return false;
        }
    }
    return true;
}

void print_test_header(bool is_A_transpose, bool is_B_transpose)
{
    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A * B" << std::endl;
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A^T * B" << std::endl;
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        std::cout << "C = A * B^T" << std::endl;
    }
    // C = A^T * B^T
    else
    {
        std::cout << "C = A^T * B^T" << std::endl;
    }
}

int main()
{
    constexpr int num_repeats{10};
    constexpr int num_warmups{10};

    uint32_t const matrix_size_m{1024};
    uint32_t const matrix_size_n{1024};
    uint32_t const matrix_size_k{1024};
    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // HMMA
    std::cout << "FP16 HMMA" << std::endl;
    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c_float(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_float_reference(matrix_size_m * matrix_size_n);

    float* h_matrix_a_float{matrix_a_float.data()};
    float* h_matrix_b_float{matrix_b_float.data()};
    __half* h_matrix_a_half{matrix_a_half.data()};
    __half* h_matrix_b_half{matrix_b_half.data()};
    float* h_matrix_c_float{matrix_c_float.data()};
    float* h_matrix_c_float_reference{matrix_c_float_reference.data()};

    fill_random_float_values(h_matrix_a_float, matrix_a_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_b_float, matrix_b_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float, matrix_c_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float_reference,
                             matrix_c_float_reference.size(), random_engine);
    float2half(h_matrix_a_half, h_matrix_a_float, matrix_a_float.size());
    float2half(h_matrix_b_half, h_matrix_b_float, matrix_b_float.size());

    half *d_matrix_a_half, *d_matrix_b_half;
    float* d_matrix_c_float;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half,
                                matrix_size_m * matrix_size_k * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half,
                                matrix_size_k * matrix_size_n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float,
                                matrix_size_m * matrix_size_n * sizeof(float)));

    // Copy data from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half, h_matrix_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half, h_matrix_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CPU.
            launch_mm(h_matrix_a_float, h_matrix_b_float,
                      h_matrix_c_float_reference, matrix_size_m, matrix_size_n,
                      matrix_size_k, is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CUDA WMMA.
            launch_wmma_mm(d_matrix_a_half, d_matrix_b_half, d_matrix_c_float,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float, d_matrix_c_float,
                                        matrix_c_float.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));

            float const avg_abs_diff_ratio{get_avg_abs_diff_ratio(
                h_matrix_c_float, h_matrix_c_float_reference,
                matrix_c_float.size())};
            if (avg_abs_diff_ratio > 0.01)
            {
                std::cout << "Got high average absolute diff ratio: "
                          << avg_abs_diff_ratio << std::endl;
            }

            // Performance measurement.
            std::function<void(cudaStream_t)> const function_hmma{std::bind(
                launch_wmma_mm<__half, float>, d_matrix_a_half, d_matrix_b_half,
                d_matrix_c_float, matrix_size_m, matrix_size_n, matrix_size_k,
                is_A_transpose, is_B_transpose, std::placeholders::_1)};
            float const latency_hmma{measure_performance(
                function_hmma, stream, num_repeats, num_warmups)};
            std::cout << std::fixed << std::setprecision(3)
                      << "HMMA Latency: " << latency_hmma << " ms" << std::endl;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float));

    // IMMA
    std::cout << "INT8 IMMA" << std::endl;
    std::vector<int8_t> matrix_a_int8(matrix_size_m * matrix_size_k);
    std::vector<int8_t> matrix_b_int8(matrix_size_k * matrix_size_n);
    std::vector<int32_t> matrix_c_int32(matrix_size_m * matrix_size_n);
    std::vector<int32_t> matrix_c_int32_reference(matrix_size_m *
                                                  matrix_size_n);

    int8_t* h_matrix_a_int8{matrix_a_int8.data()};
    int8_t* h_matrix_b_int8{matrix_b_int8.data()};
    int32_t* h_matrix_c_int32{matrix_c_int32.data()};
    int32_t* h_matrix_c_int32_reference{matrix_c_int32_reference.data()};

    fill_random_int8_values(h_matrix_a_int8, matrix_a_int8.size(),
                            random_engine);
    fill_random_int8_values(h_matrix_b_int8, matrix_b_int8.size(),
                            random_engine);
    fill_random_int32_values(h_matrix_c_int32, matrix_c_int32.size(),
                             random_engine);
    fill_random_int32_values(h_matrix_c_int32_reference,
                             matrix_c_int32_reference.size(), random_engine);

    // Profile INT8 IMMA without verifying the correctness.
    int8_t *d_matrix_a_int8, *d_matrix_b_int8;
    int32_t* d_matrix_c_int32;

    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_a_int8, matrix_size_m * matrix_size_k * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_b_int8, matrix_size_k * matrix_size_n * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_c_int32, matrix_size_m * matrix_size_n * sizeof(int32_t)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_int8, h_matrix_a_int8,
                                matrix_a_int8.size() * sizeof(int8_t),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_int8, h_matrix_b_int8,
                                matrix_b_int8.size() * sizeof(int8_t),
                                cudaMemcpyHostToDevice));

    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CPU.
            launch_mm(h_matrix_a_int8, h_matrix_b_int8,
                      h_matrix_c_int32_reference, matrix_size_m, matrix_size_n,
                      matrix_size_k, is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CUDA WMMA.
            launch_wmma_mm(d_matrix_a_int8, d_matrix_b_int8, d_matrix_c_int32,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_int32, d_matrix_c_int32,
                                        matrix_c_int32.size() * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
            // Integer matrix multiplications from CPU and CUDA should be
            // bitwise identical.
            assert(array_equal(h_matrix_c_int32, h_matrix_c_int32_reference,
                               matrix_c_int32.size()));

            // Performance measurement.
            std::function<void(cudaStream_t)> const function_imma{
                std::bind(launch_wmma_mm<int8_t, int32_t>, d_matrix_a_int8,
                          d_matrix_b_int8, d_matrix_c_int32, matrix_size_m,
                          matrix_size_n, matrix_size_k, is_A_transpose,
                          is_B_transpose, std::placeholders::_1)};
            float const latency_imma{measure_performance(
                function_imma, stream, num_repeats, num_warmups)};
            std::cout << std::fixed << std::setprecision(3)
                      << "IMMA Latency: " << latency_imma << " ms" << std::endl;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_int8));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_int8));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_int32));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}"""

    return cublas_example + "\n" + cublaslt_example + "\n" + cudnn_example + "\n" + cuda_example + "\n" + wmma_example