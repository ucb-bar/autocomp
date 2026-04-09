// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION

// #define FULL_BIAS_WIDTH 0
// #if FULL_BIAS_WIDTH
// typedef acc_t ACC_T;
// #else
// typedef elem_t ACC_T;
// #endif

#define MAKESTR(NAME) #NAME
#define XMAKESTR(NAME) MAKESTR(NAME)

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define NO_BIAS 1
#define REPEATING_BIAS 0
#define SUB_BIAS 0

#define INP_MATRIX_NAME inp
#define WEIGHT_MATRIX_NAME weights
#define BIAS_MATRIX_NAME bias
#define OUTPUT_MATRIX_NAME output

#define R 3
#define S 3
#define P 14
#define Q 14
#define C 256
#define K 256
#define N 4
#define STRIDE 1
#define INP_ROWS (P+(R/2)*2)
#define INP_COLS (Q+(S/2)*2)

#define CHECK_RESULT 1
#undef FAST
// #define FAST

int full_is_equal(elem_t x[N][P][Q][K], elem_t y[N][P][Q][K]) {
  int num_prints = 0;
  for (size_t n = 0; n < N; ++n)
    for (size_t p = 0; p < P; ++p)
      for (size_t q = 0; q < Q; ++q)
        for (size_t k = 0; k < K; ++k) {
          // printf("n: %d, p: %d, q: %d, k: %d, output: %d, gold: %d\n", n, p, q, k, x[n][p][q][k], y[n][p][q][k]);
          if (x[n][p][q][k] != y[n][p][q][k]) {
            printf("n: %d, p: %d, q: %d, k: %d, output: %d, gold: %d\n", n, p, q, k, x[n][p][q][k], y[n][p][q][k]);
            // return 0;
            num_prints++;
            if (num_prints > 10) {
              return 0;
            }
          }
        }
  return 1;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

  // int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]
  // inputs

  static elem_t INP_MATRIX_NAME[N][INP_ROWS][INP_COLS][C] row_align(1);
  static elem_t WEIGHT_MATRIX_NAME[R][S][C][K] row_align(1);
  static acc_t BIAS_MATRIX_NAME[1][K] row_align_acc(1);

  // outputs
  static elem_t gold[N][P][Q][K] row_align(1);
  static elem_t OUTPUT_MATRIX_NAME[N][P][Q][K] row_align(1);

#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif
#define REPEAT_TEST_ITERS 3
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    // printf("Init A\n");
    for (size_t n = 0; n < N; n++) {
    for (size_t row = 0; row < INP_ROWS; row++) {
    for (size_t col = 0; col < INP_COLS; col++) {
    for (size_t c = 0; c < C; c++) {
        elem_t elem = (elem_t) (RAND % (1<<3) == 0);
        INP_MATRIX_NAME[n][row][col][c] = elem;
    }}}}

    for (size_t r = 0; r < R; r++) {
    for (size_t s = 0; s < S; s++) {
    for (size_t c = 0; c < C; c++) {
    for (size_t k = 0; k < K; k++) {
        elem_t elem = (elem_t) (RAND % (1<<3) == 0);
        WEIGHT_MATRIX_NAME[r][s][c][k] = elem;
    }}}}

    // printf("Init B\n");
    for (size_t k = 0; k < K; k++) {
        acc_t elem = (acc_t) (RAND % (1<<3) == 0);
        BIAS_MATRIX_NAME[0][k] = elem;
    }

#define RUN_BASELINE_CODE 1
#if RUN_BASELINE_CODE
    // Baseline implementation
    tiled_conv_auto(
            N, INP_ROWS, INP_COLS, C,
            K, P, Q,
            STRIDE, 1, 1, 0, R,
            false, false, false,
            false, false,
            INP_MATRIX_NAME,
            WEIGHT_MATRIX_NAME,
            BIAS_MATRIX_NAME,
            gold,
            NO_ACTIVATION, 1,
            0, 0, 0,
            WEIGHT_STATIONARY);
#endif

    // Generated implementation
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("Correct result\n");
}
