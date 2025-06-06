void test(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
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