## Extracted from page 193 of Metal Shading Language Specification

SUMMARY: This part of the document covers SIMD-group matrix operations in Metal, demonstrating how to load matrices from device memory, perform multiply-accumulate operations, and store results back to device memory for optimized compute kernels on Apple GPUs.

```metal
// Example of using SIMD-group matrices
kernel void float_matmad(device float *pMatA, device float *pMatB,
                         device float *pMatC, device float *pMatR)
{
    simdgroup_float8x8 sgMatA;
    simdgroup_float8x8 sgMatB;
    simdgroup_float8x8 sgMatC;
    simdgroup_float8x8 sgMatR;
    
    simdgroup_load(sgMatA, pMatA);
    simdgroup_load(sgMatB, pMatB);
    simdgroup_load(sgMatC, pMatC);
    
    simdgroup_multiply_accumulate(sgMatR, sgMatA, sgMatB, sgMatC);
}
```

```metal
// SIMD-group matrix load from device memory
void simdgroup_load(thread simdgroup_matrix<T,Cols,Rows> a,
                    const device T *src,
                    ulong elements_per_row = Cols,
                    ulong2 matrix_origin = 0,
                    bool transpose_matrix = false);

// SIMD-group matrix store to device memory
void simdgroup_store(thread simdgroup_matrix<T,Cols,Rows> a,
                     device T *dst,
                     ulong elements_per_row = Cols,
                     ulong2 matrix_origin = 0,
                     bool transpose_matrix = false);

// Matrix multiply-accumulate: d = a * b + c
void simdgroup_multiply_accumulate(thread simdgroup_matrix<T,Cols,Rows>& d,
                                   thread simdgroup_matrix<T,K,Rows>& a,
                                   thread simdgroup_matrix<T,Cols,K>& b,
                                   thread simdgroup_matrix<T,Cols,Rows>& c);

// Matrix multiply: d = a * b
void simdgroup_multiply(thread simdgroup_matrix<T,Cols,Rows>& d,
                        thread simdgroup_matrix<T,K,Rows>& a,
                        thread simdgroup_matrix<T,Cols,K>& b);
```