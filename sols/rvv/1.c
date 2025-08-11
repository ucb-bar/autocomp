#include <riscv_vector.h>

void gemm_f32(float *y, const float *x1, const float *x2) {
  // 128 x 128 x 128 matrix multiplication
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
  
  for (size_t i = 0; i < 128; i += 1) {
    for (size_t j = 0; j < 128; j += 1) {
      // Inline dot product computation
      size_t remaining = 128;
      const float *x1_ptr = x1 + i * 128;
      const float *x2_ptr = x2 + j * 128;
      
      vfloat32m1_t vec_r = __riscv_vfmv_v_f_f32m1(0, vlmax);
      
      while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(x1_ptr, sizeof(float), vl);
        vfloat32m1_t vec_y = __riscv_vlse32_v_f32m1(x2_ptr, sizeof(float), vl);
        vec_r = __riscv_vfmacc_vv_f32m1(vec_r, vec_x, vec_y, vl);
        
        x1_ptr += vl;
        x2_ptr += vl;
        remaining -= vl;
      }
      
      vec_r = __riscv_vfredusum_vs_f32m1_f32m1(vec_r, vec_zero, vlmax);
      y[i * 128 + j] = __riscv_vfmv_f_s_f32m1_f32(vec_r);
    }
  }
}
