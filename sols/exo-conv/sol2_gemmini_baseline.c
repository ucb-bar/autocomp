void test(int8_t inp[4][16][16][256], int8_t weights[3][3][256][256], int32_t bias[1][256], int8_t output[4][14][14][256]) {
    tiled_conv_auto(
            N, INP_ROWS, INP_COLS, C,
            K, P, Q,
            STRIDE, 1, 1, 0, R,
            false, false, false,
            false, false,
            INP_MATRIX_NAME,
            WEIGHT_MATRIX_NAME,
            BIAS_MATRIX_NAME,
            OUTPUT_MATRIX_NAME,
            NO_ACTIVATION, 1,
            0, 0, 0,
            WEIGHT_STATIONARY);
    fence();
}
