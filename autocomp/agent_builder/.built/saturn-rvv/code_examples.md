## rvv-intrinsic-examples.md

SUMMARY: This document provides a collection of C code examples demonstrating the use of RISC-V Vector (RVV) intrinsics to implement common computational kernels, including memory operations, arithmetic, matrix multiplication, string manipulation, and conditional control flow.

```c
#include <riscv_vector.h>

void *memcpy_rvv(void *restrict destination, const void *restrict source,
    size_t n) {
  unsigned char *dst = destination;
  const unsigned char *src = source;
  // copy data byte by byte
  for (size_t vl; n > 0; n -= vl, src += vl, dst += vl) {
    vl = __riscv_vsetvl_e8m8(n);
    // Load src[0..vl)
    vuint8m8_t vec_src = __riscv_vle8_v_u8m8(src, vl);
    // Store dst[0..vl)
    __riscv_vse8_v_u8m8(dst, vec_src, vl);
  }
  return destination;
}
```

```c
void saxpy_rvv(size_t n, const float a, const float *x, float *y) {
  for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m8(n);
    // Load x[i..i+vl)
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
    // Load y[i..i+vl)
    vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
    // Computes vy[0..vl) + a*vx[0..vl)
    // and stores it in y[i..i+vl)
    __riscv_vse32_v_f32m8(y, __riscv_vfmacc_vf_f32m8(vy, a, vx, vl), vl);
  }
}
```

```c
void matmul_rvv(double *a, double *b, double *c, int n, int m, int p) {
  size_t vlmax = __riscv_vsetvlmax_e64m1();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      double *ptr_a = &a[i * p];
      double *ptr_b = &b[j];
      int k = p;
      // Set accumulator to  zero.
      vfloat64m1_t vec_s = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
      vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
      for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl * m) {
        vl = __riscv_vsetvl_e64m1(k);

        // Load row a[i][k..k+vl)
        vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(ptr_a, vl);
        // Load column b[k..k+vl)[j]
        vfloat64m1_t vec_b =
          __riscv_vlse64_v_f64m1(ptr_b, sizeof(double) * m, vl);

        // Accumulate dot product of row and column. If vl < vlmax we need to
        // preserve the existing values of vec_s, hence the tu policy.
        vec_s = __riscv_vfmacc_vv_f64m1_tu(vec_s, vec_a, vec_b, vl);
      }

      // Final accumulation.
      vfloat64m1_t vec_sum =
        __riscv_vfredusum_vs_f64m1_f64m1(vec_s, vec_zero, vlmax);
      double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
      c[i * m + j] = sum;
    }
}
```

```c
char *strcpy_rvv(char *destination, const char *source) {
  unsigned char *dst = (unsigned char *)destination;
  unsigned char *src = (unsigned char *)source;
  size_t vlmax = __riscv_vsetvlmax_e8m8();
  long first_set_bit = -1;

  // This loop stops when among the loaded bytes we find the null byte
  // of the string i.e., when first_set_bit >= 0
  for (size_t vl; first_set_bit < 0; src += vl, dst += vl) {
    // Load up to vlmax elements if possible.
    vuint8m8_t vec_src = __riscv_vle8ff_v_u8m8(src, &vl, vlmax);

    // Mask that states where null bytes are in the loaded bytes.
    vbool1_t string_terminate = __riscv_vmseq_vx_u8m8_b1(vec_src, 0, vl);

    // If the null byte is not in the loaded bytes the resulting mask will
    // be all ones, otherwise only the elements up to and including the
    // first null byte of the resulting will be enabled.
    vbool1_t mask = __riscv_vmsif_m_b1(string_terminate, vl);

    // Store the enabled elements as determined by the mask above.
    __riscv_vse8_v_u8m8_m(mask, dst, vec_src, vl);

    // Determine if we found the null byte in the loaded bytes.
    first_set_bit = __riscv_vfirst_m_b1(string_terminate, vl);
  }
  return destination;
}
```

```c
void branch_rvv(double *a, double *b, double *c, int n, double constant) {
  size_t vlmax = __riscv_vsetvlmax_e64m1();
  vfloat64m1_t vec_constant = __riscv_vfmv_v_f_f64m1(constant, vlmax);
  for (size_t vl; n > 0; n -= vl, a += vl, b += vl, c += vl) {
    vl = __riscv_vsetvl_e64m1(n);

    // Load a[i..i+vl)
    vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(a, vl);
    // Load b[i..i+vl)
    vfloat64m1_t vec_b = __riscv_vle64_v_f64m1(b, vl);

    // Compute a mask whose enabled elements will correspond to the
    // elements of b that are not zero.
    vbool64_t mask = __riscv_vmfne_vf_f64m1_b64(vec_b, 0.0, vl);

    // Use mask undisturbed policy to compute the division for the
    // elements enabled in the mask, otherwise set them to the given
    // constant above (maskedoff).
    vfloat64m1_t vec_c = __riscv_vfdiv_vv_f64m1_mu(
        mask, /*maskedoff*/ vec_constant, vec_a, vec_b, vl);

    // Store into c[i..i+vl)
    __riscv_vse64_v_f64m1(c, vec_c, vl);
  }
}
```

```c
void reduce_rvv(double *a, double *b, double *result_sum, int *result_count,
    int n) {
  int count = 0;
  size_t vlmax = __riscv_vsetvlmax_e64m1();
  vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  vfloat64m1_t vec_s = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  for (size_t vl; n > 0; n -= vl, a += vl, b += vl) {
    vl = __riscv_vsetvl_e64m1(n);

    // Load a[i..i+vl)
    vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(a, vl);
    // Load b[i..i+vl)
    vfloat64m1_t vec_b = __riscv_vle64_v_f64m1(b, vl);

    // Compute a mask whose enabled elements will correspond to the
    // elements of a that are not 42.
    vbool64_t mask = __riscv_vmfne_vf_f64m1_b64(vec_a, 42.0, vl);

    // vec_s[e] ← vec_s[e] + vec_a[e] * vec_b[e], if mask[e] is enabled
    vec_s = __riscv_vfmacc_vv_f64m1_tumu(mask, vec_s, vec_a, vec_b, vl);

    // Adds to count the number of elements in mask that are enabled.
    count += __riscv_vcpop_m_b64(mask, vl);
  }

  vfloat64m1_t vec_sum;
  // Final accumulation.
  vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_s, vec_zero, vlmax);
  double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);

  // Return values.
  *result_sum = sum;
  *result_count = count;
}
```

## rvvop.pdf:page_2

SUMMARY: This document provides optimization guidelines for RISC-V Vector (RVV) intrinsics, focusing on LMUL selection, instruction variant preferences, and efficient memory access patterns for various data structures.

```c
// Adding 1.0 to each element of an array of 32-bit floats
// (Note: Example assumes standard RVV intrinsic naming conventions)
vfloat32m1_t vec = vle32_v_f32m1(ptr, vl);
vec = vfadd_vf_f32m1(vec, 1.0f, vl);
```

```c
// Broadcast 3 across all elements of the register group starting at v8
vint32m1_t v8 = vmv_v_x_i32m1(3, vl);
```

```c
// Splat alternating values of 0xaaaaaaaa and 0xbbbbbbbb into v2 using masked splat
vint32m1_t v2 = vmv_v_x_i32m1(0xaaaaaaaa, vl);
vbool32_t mask = vmsne_vx_i32m1_b32(vindex, 0, vl); // Assuming vindex defines the pattern
v2 = vfmerge_vxm_i32m1(v2, 0xbbbbbbbb, mask, vl);
```

```c
// Set the first element of a vector register to 2 and the remaining elements to 0
vint32m1_t v = vmv_v_i_i32m1(0, vl);
v = vmv_s_x_i32m1(v, 2, vl);
```

```c
// Copying an array of bytes whose size is a multiple of 64kb using whole register loads/stores
// a0: destination, a1: source, a2: number of bytes
for (; a2 > 0; a2 -= vl) {
    vl = vsetvlmax_e8m8();
    vint8m8_t data = vlse8_v_i8m8(a1, 1, vl);
    vsse8_v_i8m8(a0, 1, data, vl);
    a1 += vl;
    a0 += vl;
}
```

## rvvop.pdf:page_3

SUMMARY: This document demonstrates how to use RISC-V Vector (RVV) unit-stride segment load instructions to unpack interleaved RGB data into separate color channels for grayscale conversion. It highlights the performance benefits of using vector-vector (.vv) instructions over scalar-vector variants to minimize register transfer overhead.

```c
#include <riscv_vector.h>

void rgb_to_grayscale(const uint8_t *src, uint8_t *dst, size_t n) {
  for (size_t vl; n > 0; n -= vl) {
    vl = __riscv_vsetvl_e8m1(n);

    // Load interleaved RGB data into three separate vector registers
    vuint8m1x3_t rgb = __riscv_vlseg3e8_v_u8m1x3(src, vl);
    vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb, 0);
    vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb, 1);
    vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb, 2);

    // Compute grayscale: (R + G + B) / 3 (simplified example)
    vuint8m1_t sum = __riscv_vadd_vv_u8m1(r, g, vl);
    sum = __riscv_vadd_vv_u8m1(sum, b, vl);
    vuint8m1_t gray = __riscv_vdivu_vx_u8m1(sum, 3, vl);

    // Store the result using a unit-stride store
    __riscv_vse8_v_u8m1(dst, gray, vl);

    src += vl * 3;
    dst += vl;
  }
}
```