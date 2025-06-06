void test(int8_t A[32][2048], int8_t B[2048][2048], int8_t C[32][2048]) {
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
        A_MATRIX_NAME, B_MATRIX_NAME,
        NULL, C_MATRIX_NAME,
        A_TRANSPOSE ? MAT_DIM_I : MAT_DIM_K, A_TRANSPOSE ? MAT_DIM_K : MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        A_TRANSPOSE, B_TRANSPOSE,
        false, false,
        0,
        WEIGHT_STATIONARY);
    fence();
  }
  