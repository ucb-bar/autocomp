## Vector Configuration

Vector configuration instructions set the vector length (vl) and type (vtype) for subsequent vector operations.

### vsetvl Intrinsics

```c
// Set vector length for element width and LMUL
size_t vl = __riscv_vsetvl_e8m1(avl);   // 8-bit elements, LMUL=1
size_t vl = __riscv_vsetvl_e16m2(avl);  // 16-bit elements, LMUL=2
size_t vl = __riscv_vsetvl_e32m4(avl);  // 32-bit elements, LMUL=4
size_t vl = __riscv_vsetvl_e64m8(avl);  // 64-bit elements, LMUL=8

// Set vector length with explicit max
size_t vl = __riscv_vsetvlmax_e32m1();  // Get max vl for e32m1
```

### Stripmining Pattern

The standard pattern for RVV loops processes a variable number of elements per iteration using vsetvl:

```c
size_t avl = n;  // Application vector length
for (size_t vl; avl > 0; avl -= vl) {
    vl = __riscv_vsetvl_e32m4(avl);
    // Vector operations with vl elements
    ptr += vl;
}
```

### Performance Notes

- vsetvl in inner loops can cause pipeline bubbles (especially with Rocket core)
- Shuttle core has vsetvl bypass network to reduce bubbles
- Use highest LMUL that avoids register spilling to reduce vsetvl frequency
- Mixed-precision code requires frequent vsetvl; consider cost versus benefit

## Vector Memory Operations

### Unit-Stride Loads/Stores

Unit-stride operations have the highest throughput and should be preferred when data layout permits.

```c
// Load vl elements contiguously
vfloat32m4_t vec = __riscv_vle32_v_f32m4(ptr, vl);
vint64m8_t vec = __riscv_vle64_v_i64m8(ptr, vl);

// Store vl elements contiguously
__riscv_vse32_v_f32m4(ptr, vec, vl);
__riscv_vse64_v_i64m8(ptr, vec, vl);
```

### Strided Loads/Stores

Strided operations load/store with a constant byte stride. Memory throughput is limited to one address per cycle.

```c
// Load with constant stride (in bytes)
vfloat32m4_t vec = __riscv_vlse32_v_f32m4(ptr, stride, vl);

// Store with constant stride
__riscv_vsse32_v_f32m4(ptr, stride, vec, vl);
```

### Indexed (Gather/Scatter) Loads/Stores

Gather and scatter operations use a vector of indices to access non-uniform addresses. Like strided, throughput is one address per cycle.

```c
// Gather: load from ptr + indices[i]
vuint32m4_t indices = ...;
vfloat32m4_t vec = __riscv_vluxei32_v_f32m4(ptr, indices, vl);

// Scatter: store to ptr + indices[i]
__riscv_vsuxei32_v_f32m4(ptr, indices, vec, vl);
```

### Segmented Loads/Stores

Segmented operations interleave multiple fields (e.g., RGB pixels) stored in memory into separate vector registers, or vice versa. This is efficient for array-of-structs to struct-of-arrays conversion.

```c
// Load NF fields interleaved in memory into NF vector registers
// e.g., RGB pixels: [R0,G0,B0,R1,G1,B1,...] -> vr, vg, vb
vfloat32m2x3_t rgb = __riscv_vlseg3e32_v_f32m2x3(ptr, vl);
vfloat32m2_t vr = __riscv_vget_v_f32m2x3_f32m2(rgb, 0);
vfloat32m2_t vg = __riscv_vget_v_f32m2x3_f32m2(rgb, 1);
vfloat32m2_t vb = __riscv_vget_v_f32m2x3_f32m2(rgb, 2);

// Store multiple fields back interleaved
__riscv_vsseg3e32_v_f32m2x3(ptr, rgb, vl);
```

### Performance Notes

- Unit-stride saturates memory bandwidth; prefer over strided/indexed when possible
- Strided and indexed: 1 element address per cycle (memory-bound)
- Segmented loads never worse than equivalent manual unpacking
- Masked unit-stride loads ignore mask (apply at VRF write); stores use mask

## Integer Arithmetic Operations

### Add/Subtract/Multiply

Basic integer operations support both vector-vector and vector-scalar variants.

```c
// Add/subtract
vint32m4_t c = __riscv_vadd_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vsub_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vadd_vx_i32m4(a, scalar, vl);  // vector + scalar

// Multiply
vint32m4_t c = __riscv_vmul_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vmacc_vv_i32m4(acc, a, b, vl);  // acc += a * b
vint32m4_t c = __riscv_vmadd_vv_i32m4(a, b, c, vl);   // a = a * b + c
```

### Shift Operations

Shift left, arithmetic right shift, and logical right shift are available.

```c
// Left shift
vint32m4_t c = __riscv_vsll_vv_i32m4(a, shift, vl);

// Arithmetic right shift
vint32m4_t c = __riscv_vsra_vv_i32m4(a, shift, vl);

// Logical right shift
vint32m4_t c = __riscv_vsrl_vv_i32m4(a, shift, vl);
```

### Min/Max

```c
vint32m4_t c = __riscv_vmin_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vmax_vv_i32m4(a, b, vl);
```

## Floating-Point Arithmetic Operations

### Basic Operations

Add, subtract, multiply, and divide with floating-point semantics.

```c
// Add/subtract/multiply/divide
vfloat32m4_t c = __riscv_vfadd_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfsub_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfmul_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfdiv_vv_f32m4(a, b, vl);  // iterative, slow
```

### Fused Multiply-Add

Fused multiply-add has a 4-cycle pipeline and is critical for GEMM and similar kernels. Both vector-vector and vector-scalar variants exist.

```c
// Fused multiply-add: acc += a * b
vfloat32m4_t c = __riscv_vfmacc_vv_f32m4(acc, a, b, vl);

// Fused multiply-add: a = a * b + c
vfloat32m4_t c = __riscv_vfmadd_vv_f32m4(a, b, c, vl);

// Negative fused multiply-add: acc -= a * b
vfloat32m4_t c = __riscv_vfnmacc_vv_f32m4(acc, a, b, vl);

// Scalar operand versions (broadcast scalar to all lanes)
vfloat32m4_t c = __riscv_vfmul_vf_f32m4(a, scalar, vl);
vfloat32m4_t c = __riscv_vfmacc_vf_f32m4(acc, scalar, b, vl);
```

### Min/Max and Comparison

```c
vfloat32m4_t c = __riscv_vfmin_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfmax_vv_f32m4(a, b, vl);
```

### Square Root and Reciprocal

These are iterative operations (element-wise) and should be avoided in tight inner loops.

```c
vfloat32m4_t c = __riscv_vfsqrt_v_f32m4(a, vl);
```

## Widening Operations

Widening operations double the output element width and are useful for avoiding intermediate overflow.

```c
// Widening multiply: i16 × i16 → i32
vint32m4_t c = __riscv_vwmul_vv_i32m4(a_i16, b_i16, vl);

// Widening multiply-add: acc += a_i16 * b_i16 (result i32)
vint32m4_t c = __riscv_vwmacc_vv_i32m4(acc, a_i16, b_i16, vl);

// Floating-point widening
vfloat64m4_t c = __riscv_vfwmacc_vv_f64m4(acc, a_f32, b_f32, vl);
```

## Reduction Operations

Reductions combine all vector elements into a single scalar result. The common pattern is to accumulate in a vector across loop iterations and perform a single reduction at the end.

### Sum Reduction

```c
// Initialize scalar accumulator
vfloat32m1_t scalar_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);

// Reduce vector to scalar (acc = acc + sum(vec))
scalar_acc = __riscv_vfredusum_vs_f32m4_f32m1(vec, scalar_acc, vl);

// Extract scalar result
float result = __riscv_vfmv_f_s_f32m1_f32(scalar_acc);
```

### Max and Min Reduction

```c
// Max reduction
vfloat32m1_t max_val = __riscv_vfredmax_vs_f32m4_f32m1(vec, init, vl);

// Min reduction
vfloat32m1_t min_val = __riscv_vfredmin_vs_f32m4_f32m1(vec, init, vl);
```

### Integer Reductions

```c
// Integer sum
vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(vec, init, vl);

// Integer max/min
vint32m1_t max_val = __riscv_vredmax_vs_i32m4_i32m1(vec, init, vl);
vint32m1_t min_val = __riscv_vredmin_vs_i32m4_i32m1(vec, init, vl);
```

### Optimized Reduction Pattern

Instead of reducing within each loop iteration, accumulate in a vector across iterations and reduce once at the end. This is much more efficient.

```c
// BAD: for each chunk: acc = vredsum(vmul(a, b), acc)

// GOOD: accumulate in vector, reduce once at end:
vfloat32m4_t vec_acc;
for (first chunk) {
    vec_acc = __riscv_vfmul_vv_f32m4(a, b, vl);
}
for (remaining chunks) {
    vec_acc = __riscv_vfmacc_vv_f32m4(vec_acc, a, b, vl);
}
// Single reduction at end
vfloat32m1_t result = __riscv_vfredusum_vs_f32m4_f32m1(vec_acc, zero, vl);
```

## Permutation Operations

### Slide Operations

Slide operations shift elements within a vector register at DLEN bits/cycle and are efficient for regular access patterns (e.g., convolution stencils).

```c
// Slide down: shift elements toward lower indices
vfloat32m4_t c = __riscv_vslidedown_vx_f32m4(src, offset, vl);

// Slide up: shift elements toward higher indices
vfloat32m4_t c = __riscv_vslideup_vx_f32m4(dst, src, offset, vl);

// Slide by immediate (more efficient for constant offsets)
vfloat32m4_t c = __riscv_vslidedown_vi_f32m4(src, 1, vl);  // slide by 1
```

### Register Gather

Register gather rearranges elements from a source register using indices and is element-wise (slower than slides).

```c
// Gather elements from src using indices in idx
vuint32m4_t idx = ...;
vfloat32m4_t c = __riscv_vrgather_vv_f32m4(src, idx, vl);
```

### Compress

Compress packs elements where a mask is true. This is element-wise and typically slower.

```c
// Compress: pack elements where mask is true
vbool8_t mask = ...;
vfloat32m4_t c = __riscv_vcompress_vm_f32m4(src, mask, vl);
```

### Move and Broadcast

```c
// Broadcast scalar to all elements
vfloat32m4_t c = __riscv_vfmv_v_f_f32m4(scalar, vl);

// Extract element 0 to scalar
float s = __riscv_vfmv_f_s_f32m4_f32(vec);

// Set element 0 from scalar
vfloat32m4_t c = __riscv_vfmv_s_f_f32m4(scalar, vl);
```

## Comparison and Mask Operations

### Creating Masks

Compare operations produce masks (one bit per element) that control conditional execution.

```c
// Floating-point comparisons
vbool8_t mask = __riscv_vmflt_vv_f32m4_b8(a, b, vl);  // a < b
vbool8_t mask = __riscv_vmfle_vv_f32m4_b8(a, b, vl);  // a <= b
vbool8_t mask = __riscv_vmfeq_vv_f32m4_b8(a, b, vl);  // a == b
vbool8_t mask = __riscv_vmfgt_vf_f32m4_b8(a, 0.0f, vl);  // a > 0

// Integer comparisons
vbool8_t mask = __riscv_vmslt_vv_i32m4_b8(a, b, vl);  // signed a < b
vbool8_t mask = __riscv_vmsltu_vv_u32m4_b8(a, b, vl); // unsigned a < b
```

### Masked Operations

Masked operations execute conditionally based on mask bits.

```c
// Masked load (inactive elements undefined)
vfloat32m4_t c = __riscv_vle32_v_f32m4_m(mask, ptr, vl);

// Masked store (only store where mask is true)
__riscv_vse32_v_f32m4_m(mask, ptr, vec, vl);

// Masked arithmetic
vfloat32m4_t c = __riscv_vfadd_vv_f32m4_m(mask, a, b, vl);

// Merge: select from two sources based on mask
vfloat32m4_t c = __riscv_vmerge_vvm_f32m4(false_val, true_val, mask, vl);
```

### Mask Manipulation

```c
// Mask AND/OR/XOR
vbool8_t c = __riscv_vmand_mm_b8(mask1, mask2, vl);
vbool8_t c = __riscv_vmor_mm_b8(mask1, mask2, vl);
vbool8_t c = __riscv_vmnot_m_b8(mask, vl);

// Count set bits in mask
unsigned count = __riscv_vcpop_m_b8(mask, vl);

// Find first set bit (-1 if none set)
long first = __riscv_vfirst_m_b8(mask, vl);
```

## Saturn Optimization Guide

### 1. Maximize LMUL

Use the highest LMUL that avoids register spilling. Higher LMUL means:
- Longer chimes (better amortization of loop overhead)
- Fewer dynamic instructions
- Better saturation of pipelines (especially FMA)

With VLEN=512 and DLEN=128, LMUL=8 (e32) uses all 32 registers as one logical group.

### 2. Enable Chaining

Interleave loads and arithmetic to enable chaining between sequencers. Chaining occurs when instructions in different sequencers (load, store, integer, floating-point) can begin before previous results are fully written back.

**Example pattern**: load A, load B, compute on A, load C, compute on B, store result, ...

### 3. Balance Across Sequencers

This instance's issue queue configuration (Unified, Shared, or Split) affects integer/FP parallelism:
- **Unified**: Single sequencer, no parallel int/FP (lowest area)
- **Shared**: Separate int/FP sequencers with shared issue queue (requires interleaving)
- **Split**: Separate queues for int and FP (most flexible)

### 4. Minimize vsetvl Overhead

- Keep vsetvl outside inner loops when possible
- Use consistent element width within loop nests to avoid reconfiguration
- Select high LMUL to reduce vsetvl call frequency

### 5. Memory Access Patterns

- Unit-stride loads/stores saturate bandwidth (preferred)
- Strided/indexed: 1 address per cycle (use only when necessary)
- Segmented loads for array-of-structs avoid manual unpacking overhead

### 6. Reduction Strategy

Accumulate in vector registers across loop iterations, then perform a single reduction at the end (not per-iteration). This is dramatically more efficient.

### 7. Avoid Pipeline Stalls

- FMA latency is typically 4 cycles; need LMUL ≥ 4 or independent FMAs to saturate
- Avoid division/sqrt in inner loops (iterative, element-wise)
- Schedule scalar bookkeeping to overlap with vector execution

### 8. Widening and Overflow Avoidance

Use widening operations (e.g., vwmul) instead of clamping or saturating arithmetic when precision allows. This simplifies code and is often faster.

### Example: Optimized Dot Product

Shows LMUL=8, vector accumulation, single reduction at end:

```c
int64_t dotp_v64b(int64_t *a, int64_t *b, uint64_t avl) {
    size_t orig_avl = avl;
    size_t vl = __riscv_vsetvl_e64m8(avl);

    vint64m8_t acc, buf_a, buf_b;
    vint64m1_t red = __riscv_vmv_s_x_i64m1(0, vl);

    for (; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e64m8(avl);
        buf_a = __riscv_vle64_v_i64m8(a, vl);
        buf_b = __riscv_vle64_v_i64m8(b, vl);
        if (avl == orig_avl) {
            acc = __riscv_vmul_vv_i64m8(buf_a, buf_b, vl);
        } else {
            acc = __riscv_vmacc_vv_i64m8(acc, buf_a, buf_b, vl);
        }
        a += vl;
        b += vl;
    }

    red = __riscv_vredsum_vs_i64m8_i64m1(acc, red, vl);
    return __riscv_vmv_x_s_i64m1_i64(red);
}
```

### Example: Register-Blocked GEMM

Shows register blocking, multiple accumulators, and scalar broadcast (vfmacc_vf):

```c
// C[M,N] = A[M,K] * B[K,N]
void sgemm_blocked(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc) {
    const size_t TILE_M = 4;  // Rows of C tile

    for (size_t i = 0; i < M; i += TILE_M) {
        size_t m_tile = (i + TILE_M <= M) ? TILE_M : (M - i);

        for (size_t j = 0; j < N; ) {
            size_t vl = __riscv_vsetvl_e32m4(N - j);

            // Initialize C accumulators
            vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (size_t k = 0; k < K; ++k) {
                vfloat32m4_t b_row = __riscv_vle32_v_f32m4(&B[k * ldb + j], vl);

                // vfmacc_vf broadcasts scalar A element
                if (m_tile > 0) c0 = __riscv_vfmacc_vf_f32m4(c0, A[(i+0)*lda+k], b_row, vl);
                if (m_tile > 1) c1 = __riscv_vfmacc_vf_f32m4(c1, A[(i+1)*lda+k], b_row, vl);
                if (m_tile > 2) c2 = __riscv_vfmacc_vf_f32m4(c2, A[(i+2)*lda+k], b_row, vl);
                if (m_tile > 3) c3 = __riscv_vfmacc_vf_f32m4(c3, A[(i+3)*lda+k], b_row, vl);
            }

            // Store C tile
            if (m_tile > 0) __riscv_vse32_v_f32m4(&C[(i+0)*ldc+j], c0, vl);
            if (m_tile > 1) __riscv_vse32_v_f32m4(&C[(i+1)*ldc+j], c1, vl);
            if (m_tile > 2) __riscv_vse32_v_f32m4(&C[(i+2)*ldc+j], c2, vl);
            if (m_tile > 3) __riscv_vse32_v_f32m4(&C[(i+3)*ldc+j], c3, vl);

            j += vl;
        }
    }
}
```

### Example: Convolution with vslidedown

Shows vslidedown for stencil access and scalar broadcast for filter weights:

```c
// 3x3 convolution - key pattern using vslidedown
// Load input row with padding, then slide for each filter column
vfloat64m2_t row = __riscv_vle64_v_f64m2(in_ptr, vl_padded);

// Column 0: no slide
vfloat64m2_t out = __riscv_vfmul_vf_f64m2(row, filter[0], vl);

// Column 1: slide by 1
vfloat64m2_t row_s1 = __riscv_vslidedown_vx_f64m2(row, 1, vl);
out = __riscv_vfmacc_vf_f64m2(out, filter[1], row_s1, vl);

// Column 2: slide by 2
vfloat64m2_t row_s2 = __riscv_vslidedown_vx_f64m2(row, 2, vl);
out = __riscv_vfmacc_vf_f64m2(out, filter[2], row_s2, vl);
```
