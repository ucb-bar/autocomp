## Reduce Data Movement

SUMMARY: Demonstrates keeping accumulators live in vector registers to avoid redundant memory traffic. Shows a bad pattern (store-reload between FMAs) vs good pattern (accumulate in registers, store once) for a two-term linear combination z = a*x + b*y.

```c
// BAD: reloads z from memory between terms — redundant memory traffic
void lincomb_bad(size_t n, float a, float b,
                 const float *x, const float *y, float *z) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
    vfloat32m4_t vz = __riscv_vfmul_vf_f32m4(vx, a, vl);
    __riscv_vse32_v_f32m4(z + i, vz, vl);

    vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
    vz = __riscv_vle32_v_f32m4(z + i, vl);  // Redundant reload!
    vz = __riscv_vfmacc_vf_f32m4(vz, b, vy, vl);
    __riscv_vse32_v_f32m4(z + i, vz, vl);
    i += vl;
  }
}

// GOOD: keep accumulator live in registers, store once
void lincomb_keep_live(size_t n, float a, float b,
                       const float *x, const float *y, float *z) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
    vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
    vfloat32m4_t vz = __riscv_vfmul_vf_f32m4(vx, a, vl);
    vz = __riscv_vfmacc_vf_f32m4(vz, b, vy, vl);
    __riscv_vse32_v_f32m4(z + i, vz, vl);
    i += vl;
  }
}
```

## Overlap Data Movement and Compute

SUMMARY: Software-pipelined AXPY that preloads the next chunk while computing the current chunk. Demonstrates prime-steady-drain pattern for overlapping load and compute across loop iterations using Saturn's decoupled memory and execute sequencers.

```c
void axpy_prefetch_style(size_t n, float a, const float *x, float *y) {
  size_t i = 0;
  if (n == 0) return;

  size_t vl = __riscv_vsetvl_e32m1(n);
  vfloat32m1_t vx = __riscv_vle32_v_f32m1(x, vl);
  vfloat32m1_t vy = __riscv_vle32_v_f32m1(y, vl);
  i += vl;

  while (i < n) {
    size_t next_vl = __riscv_vsetvl_e32m1(n - i);
    // Load next chunk first — overlaps with current compute via chaining
    vfloat32m1_t vx_next = __riscv_vle32_v_f32m1(x + i, next_vl);
    vfloat32m1_t vy_next = __riscv_vle32_v_f32m1(y + i, next_vl);

    vy = __riscv_vfmacc_vf_f32m1(vy, a, vx, vl);
    __riscv_vse32_v_f32m1(y + i - vl, vy, vl);

    vx = vx_next; vy = vy_next; vl = next_vl;
    i += vl;
  }
  // Drain final chunk
  vy = __riscv_vfmacc_vf_f32m1(vy, a, vx, vl);
  __riscv_vse32_v_f32m1(y + i - vl, vy, vl);
}
```

## Register Blocking for GEMM

SUMMARY: Register-blocked 1xN GEMM micro-kernel that keeps the C tile accumulator live in vector registers across the K loop. Broadcasts A scalars via vfmacc_vf and streams B vectors. Demonstrates caching reused data in registers to reduce memory traffic.

```c
void gemm_1xN_rvv(size_t M, size_t N, size_t K,
                  const float *A, size_t lda,
                  const float *B, size_t ldb,
                  float *C, size_t ldc) {
  for (size_t m = 0; m < M; ++m) {
    size_t n_left = N;
    const float *b_ptr = B;
    float *c_ptr = C + m * ldc;

    while (n_left > 0) {
      size_t vl = __riscv_vsetvl_e32m1(n_left);
      // Load C tile once, keep live through K loop
      vfloat32m1_t acc = __riscv_vle32_v_f32m1(c_ptr, vl);
      const float *a_ptr = A + m * lda;
      const float *bk_ptr = b_ptr;

      for (size_t k = 0; k < K; ++k) {
        vfloat32m1_t bvec = __riscv_vle32_v_f32m1(bk_ptr, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, *a_ptr, bvec, vl);
        ++a_ptr;
        bk_ptr += ldb;
      }

      __riscv_vse32_v_f32m1(c_ptr, acc, vl);
      c_ptr += vl; b_ptr += vl; n_left -= vl;
    }
  }
}
```

## Loop Tiling with Vectorization

SUMMARY: Cache-tiled GEMM with RVV strip-mining inside each tile. Tiles across M, K, N dimensions for cache locality, vectorizes the innermost N dimension. Shows how to combine loop tiling with VLA strip-mine loops.

```c
#define BM 64
#define BN 64
#define BK 64

void gemm_tiled_rvv(size_t M, size_t N, size_t K,
                    const float *A, size_t lda,
                    const float *B, size_t ldb,
                    float *C, size_t ldc) {
  for (size_t mm = 0; mm < M; mm += BM) {
    size_t m_end = (mm + BM < M) ? (mm + BM) : M;
    for (size_t kk = 0; kk < K; kk += BK) {
      size_t k_end = (kk + BK < K) ? (kk + BK) : K;
      for (size_t nn = 0; nn < N; nn += BN) {
        size_t n_end = (nn + BN < N) ? (nn + BN) : N;
        for (size_t m = mm; m < m_end; ++m) {
          for (size_t n = nn; n < n_end; ) {
            size_t vl = __riscv_vsetvl_e32m1(n_end - n);
            vfloat32m1_t acc = __riscv_vle32_v_f32m1(&C[m * ldc + n], vl);
            for (size_t k = kk; k < k_end; ++k) {
              vfloat32m1_t bvec = __riscv_vle32_v_f32m1(&B[k * ldb + n], vl);
              acc = __riscv_vfmacc_vf_f32m1(acc, A[m * lda + k], bvec, vl);
            }
            __riscv_vse32_v_f32m1(&C[m * ldc + n], acc, vl);
            n += vl;
          }
        }
      }
    }
  }
}
```

## Deferred Reduction with Tail-Undisturbed Accumulation

SUMMARY: Dual dot product that accumulates with element-wise vfmacc in the loop body and defers horizontal reduction to after the loop. Demonstrates the critical Saturn pattern of avoiding scalar extractions (vfmv_f_s) inside inner loops, and using tail-undisturbed (_tu) policy to preserve partial sums when vl shrinks on the final iteration.

```c
void dot2_unrolled_rvv(const float *a0, const float *a1,
                       const float *b, float *out0, float *out1, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e32m4();

  // Vector accumulators — element-wise, reduced once at the end
  vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

  size_t avl = n;
  while (avl > 0) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t va0 = __riscv_vle32_v_f32m4(a0, vl);
    vfloat32m4_t va1 = __riscv_vle32_v_f32m4(a1, vl);
    vfloat32m4_t vb  = __riscv_vle32_v_f32m4(b,  vl);

    // Element-wise multiply-accumulate — NO scalar extraction here
    // _tu preserves tail elements when vl < vlmax on final iteration
    acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, va0, vb, vl);
    acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, va1, vb, vl);

    a0 += vl; a1 += vl; b += vl;
    avl -= vl;
  }

  // Horizontal reduction OUTSIDE the loop
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  *out0 = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(acc0, zero, vlmax));
  *out1 = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(acc1, zero, vlmax));
}
```

## Fused Widen-Compute-Narrow Pipeline

SUMMARY: Fused operation pipeline that widens u8 data to u16 once, performs arithmetic in widened precision, then narrows back to u8 with saturating clip. Avoids multiple narrow/widen transitions by keeping data in the wider domain for all intermediate operations.

```c
void add_bias_saturate_u8(const uint8_t *x, uint8_t *y, uint8_t bias, size_t n) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e8m1(n - i);
    vuint8m1_t vx = __riscv_vle8_v_u8m1(x + i, vl);
    vuint16m2_t wx = __riscv_vwcvtu_x_x_v_u16m2(vx, vl);  // Widen once
    wx = __riscv_vadd_vx_u16m2(wx, bias, vl);               // Work in wider domain
    vuint8m1_t out = __riscv_vnclipu_wx_u8m1(wx, 0, __RISCV_VXRM_RDN, vl); // Narrow once
    __riscv_vse8_v_u8m1(y + i, out, vl);
    i += vl;
  }
}
```

## Lower Precision for Bandwidth

SUMMARY: Processing u8 data with u16 intermediate precision to trade accuracy for memory bandwidth. Demonstrates widening multiply, bias addition, and saturating narrowing clip for quantized integer pipelines.

```c
void scale_bias_clamp_u8(const uint8_t *x, uint8_t *y,
                         uint16_t scale, uint16_t bias, size_t n) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e8m2(n - i);
    vuint8m2_t vx = __riscv_vle8_v_u8m2(x + i, vl);
    vuint16m4_t wx = __riscv_vwcvtu_x_x_v_u16m4(vx, vl);
    wx = __riscv_vmul_vx_u16m4(wx, scale, vl);
    wx = __riscv_vadd_vx_u16m4(wx, bias, vl);
    vuint8m2_t out = __riscv_vnclipu_wx_u8m2(wx, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_u8m2(y + i, out, vl);
    i += vl;
  }
}
```

## Double Buffering

SUMMARY: Register-level double buffering that loads the next chunk of data while computing on the current chunk. Demonstrates ping-pong between two sets of vector registers to overlap load and compute on Saturn's independent load and execute sequencers.

```c
void add_double_buffered(const float *a, const float *b, float *c, size_t n) {
  if (n == 0) return;
  size_t i = 0;
  size_t vl0 = __riscv_vsetvl_e32m1(n);
  vfloat32m1_t a0 = __riscv_vle32_v_f32m1(a, vl0);
  vfloat32m1_t b0 = __riscv_vle32_v_f32m1(b, vl0);
  i += vl0;

  while (i < n) {
    size_t vl1 = __riscv_vsetvl_e32m1(n - i);
    vfloat32m1_t a1 = __riscv_vle32_v_f32m1(a + i, vl1);  // Next buffer
    vfloat32m1_t b1 = __riscv_vle32_v_f32m1(b + i, vl1);
    vfloat32m1_t c0 = __riscv_vfadd_vv_f32m1(a0, b0, vl0); // Consume current
    __riscv_vse32_v_f32m1(c + i - vl0, c0, vl0);
    a0 = a1; b0 = b1; vl0 = vl1;                            // Rotate
    i += vl1;
  }
  vfloat32m1_t c0 = __riscv_vfadd_vv_f32m1(a0, b0, vl0);   // Drain
  __riscv_vse32_v_f32m1(c + i - vl0, c0, vl0);
}
```

## Software-Pipelined Inner Loop

SUMMARY: Software-pipelined inner K loop for GEMM-like accumulation. Loads the next B vector and A scalar while computing the current FMA, using prime-steady-drain structure to overlap memory and compute across iterations.

```c
void gemm_inner_software_pipelined(size_t K, const float *a_row,
                                   const float *b_panel, size_t ldb,
                                   float *c_row, size_t n_cols) {
  for (size_t n = 0; n < n_cols; ) {
    size_t vl = __riscv_vsetvl_e32m1(n_cols - n);
    vfloat32m1_t acc = __riscv_vle32_v_f32m1(c_row + n, vl);
    if (K == 0) { __riscv_vse32_v_f32m1(c_row + n, acc, vl); n += vl; continue; }

    // Prime
    vfloat32m1_t bvec = __riscv_vle32_v_f32m1(b_panel + n, vl);
    float a_scalar = a_row[0];

    // Steady state: load next while computing current
    for (size_t k = 1; k < K; ++k) {
      vfloat32m1_t bnext = __riscv_vle32_v_f32m1(b_panel + k * ldb + n, vl);
      float anext = a_row[k];
      acc = __riscv_vfmacc_vf_f32m1(acc, a_scalar, bvec, vl);
      bvec = bnext; a_scalar = anext;
    }

    // Drain
    acc = __riscv_vfmacc_vf_f32m1(acc, a_scalar, bvec, vl);
    __riscv_vse32_v_f32m1(c_row + n, acc, vl);
    n += vl;
  }
}
```

## VL Predication for Tail Handling

SUMMARY: Standard VLA strip-mine loop using vl predication for natural tail handling. No separate tail code needed — the final iteration simply uses a smaller vl. Uses LMUL=4 to maximize chime length on Saturn.

```c
void add_vl_predicated(size_t n, const float *a, const float *b, float *c) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
    vfloat32m4_t vc = __riscv_vfadd_vv_f32m4(va, vb, vl);
    __riscv_vse32_v_f32m4(c + i, vc, vl);
    i += vl;
  }
}
```

## Segment Loads for Interleaved Data

SUMMARY: RGB-to-grayscale conversion using vlseg3 to deinterleave packed RGB pixels into separate R, G, B vectors. Demonstrates segment loads for AOS-to-SOA conversion, widening for weighted arithmetic, and narrowing at the end.

```c
void rgb_to_gray_seg(const uint8_t *src, uint8_t *dst, size_t pixels) {
  for (size_t i = 0; i < pixels; ) {
    size_t vl = __riscv_vsetvl_e8m1(pixels - i);
    vuint8m1x3_t rgb = __riscv_vlseg3e8_v_u8m1x3(src + 3 * i, vl);
    vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb, 0);
    vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb, 1);
    vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb, 2);

    vuint16m2_t acc = __riscv_vmul_vx_u16m2(__riscv_vwcvtu_x_x_v_u16m2(r, vl), 77, vl);
    acc = __riscv_vmacc_vx_u16m2(acc, 150, __riscv_vwcvtu_x_x_v_u16m2(g, vl), vl);
    acc = __riscv_vmacc_vx_u16m2(acc, 29, __riscv_vwcvtu_x_x_v_u16m2(b, vl), vl);

    vuint8m1_t gray = __riscv_vnsrl_wx_u8m1(acc, 8, vl);
    __riscv_vse8_v_u8m1(dst + i, gray, vl);
    i += vl;
  }
}
```

## Mixed-Width Conditional Select

SUMMARY: Per-element absolute difference |a-b| for u8 vectors with u16 output. Demonstrates building a mask via compare, conditional select with vmerge, and fractional LMUL (mf2) to pair with widened output (m1) for consistent VLMAX.

```c
void abs_diff_u8(const uint8_t *a, const uint8_t *b,
                 uint16_t *out, size_t n) {
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e8mf2(n - i);
    vuint8mf2_t va = __riscv_vle8_v_u8mf2(a + i, vl);
    vuint8mf2_t vb = __riscv_vle8_v_u8mf2(b + i, vl);

    vbool16_t ge = __riscv_vmsgeu_vv_u8mf2_b16(va, vb, vl);
    vuint8mf2_t d1 = __riscv_vsub_vv_u8mf2(va, vb, vl);
    vuint8mf2_t d2 = __riscv_vsub_vv_u8mf2(vb, va, vl);
    vuint8mf2_t absd = __riscv_vmerge_vvm_u8mf2(d2, d1, ge, vl);

    vuint16m1_t wide = __riscv_vwcvtu_x_x_v_u16m1(absd, vl);
    __riscv_vse16_v_u16m1(out + i, wide, vl);
    i += vl;
  }
}
```

## Fixed-VL Hot Loop with VLA Tail

SUMMARY: Split hot-loop/tail pattern using fixed vlmax for the main loop (no vsetvl overhead per iteration) and a VLA tail iteration for remaining elements. Uses LMUL=4 for maximum throughput on Saturn. Demonstrates an alternative to pure VLA strip-mining when the overhead of per-iteration vsetvl matters.

```c
void axpy_vls_fastpath(size_t n, double a, const double *x, double *y) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  size_t i = 0;

  // Hot loop at fixed vlmax
  for (; i + vlmax <= n; i += vlmax) {
    vfloat64m4_t vx = __riscv_vle64_v_f64m4(x + i, vlmax);
    vfloat64m4_t vy = __riscv_vle64_v_f64m4(y + i, vlmax);
    vy = __riscv_vfmacc_vf_f64m4(vy, a, vx, vlmax);
    __riscv_vse64_v_f64m4(y + i, vy, vlmax);
  }

  // VLA tail — still vectorized, just with smaller vl
  if (i < n) {
    size_t vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t vx = __riscv_vle64_v_f64m4(x + i, vl);
    vfloat64m4_t vy = __riscv_vle64_v_f64m4(y + i, vl);
    vy = __riscv_vfmacc_vf_f64m4(vy, a, vx, vl);
    __riscv_vse64_v_f64m4(y + i, vy, vl);
  }
}
```
