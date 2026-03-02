static void test(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params)
{
  const float xmin = -0x1.5ebb82p6;
  const float r_ln2f = 0x1.715476p+0f;
  const float l2uf = 0x1.62E400p-1f;
  const float l2lf = 0x1.7F7D1Cp-20f;
  const float c6 = 0x1.6850e4p-10f;
  const float c5 = 0x1.123bccp-7;
  const float c4 = 0x1.555b98p-5f;
  const float c3 = 0x1.55548ep-3f;
  const float c2 = 0x1.fffff8p-2f;
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m4(n);

  vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m4(avl);
    avl -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, *max, vl);
    input += vl;

    vx = __riscv_vfmax_vf_f32m4(vx, xmin, vl);

    vfloat32m4_t v = __riscv_vfmul_vf_f32m4(vx, r_ln2f, vl);
    vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
    vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

    vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(vx, l2uf, z, vl);
    s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

    vfloat32m4_t poly_z;
    vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(c5, vl);
    y = __riscv_vfmacc_vf_f32m4(y, c6, s, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c4, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c3, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c2, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
    vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p, vl);
    vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
    vfloat32m4_t vexp = __riscv_vfmul_vv_f32m4(y, qf, vl);

    __riscv_vse32_v_f32m4(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vexp, vl);
  } while(avl > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  *sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsum, v0, n));
}