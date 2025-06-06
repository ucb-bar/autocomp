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

#define NO_BIAS 0
#define REPEATING_BIAS 0
#define SUB_BIAS 0

// float Kinf[4][12], float x_i[12][1], float d_i[4][1], float u_i[4][1]
#define A_MATRIX_NAME Kinf
#define B_MATRIX_NAME x_i
#define C_MATRIX_NAME u_i
#define OUTPUT_MATRIX_NAME d
#if NO_BIAS==0
  #define D_MATRIX_NAME d_i
#endif

#define MAT_DIM_I 4
#define MAT_DIM_K 12
#define MAT_DIM_J 1

#if A_TRANSPOSE==1
  #define A_ROWS MAT_DIM_K
  #define A_COLS MAT_DIM_I
#else
  #define A_ROWS MAT_DIM_I
  #define A_COLS MAT_DIM_K
#endif

#if B_TRANSPOSE==1
  #define B_ROWS MAT_DIM_J
  #define B_COLS MAT_DIM_K
#else
  #define B_ROWS MAT_DIM_K
  #define B_COLS MAT_DIM_J
#endif

#define CHECK_RESULT 1
#undef FAST
// #define FAST

#define NHORIZON 10
#define OUTPUT_DIM_1 NHORIZON
#define OUTPUT_DIM_2 4
#define OUTPUT_DIM_3 1

int full_is_equal(elem_t x[OUTPUT_DIM_1][OUTPUT_DIM_2][OUTPUT_DIM_3], elem_t y[OUTPUT_DIM_1][OUTPUT_DIM_2][OUTPUT_DIM_3]) {
  // bool pass = true;
  for (size_t i1 = 0; i1 < OUTPUT_DIM_1; ++i1)
    for (size_t i2 = 0; i2 < OUTPUT_DIM_2; ++i2)
      for (size_t i3 = 0; i3 < OUTPUT_DIM_3; ++i3) {
        // printf("Mismatch at (%d, %d, %d): %d != %d\n", i1, i2, i3, (int) x[i1][i2][i3], (int) y[i1][i2][i3]);
        if (x[i1][i2][i3] != y[i1][i2][i3]) {
          // pass = false;
          return 0;
        }
      }
  // if (pass) {
  //   return 1;
  // } else {
  //   return 0;
  // }
  return 1;
}

int print_B_p_r(elem_t x[4][1][1]) {
  bool pass = true;
  for (size_t i1 = 0; i1 < OUTPUT_DIM_1; ++i1)
    for (size_t i2 = 0; i2 < OUTPUT_DIM_2; ++i2)
      for (size_t i3 = 0; i3 < OUTPUT_DIM_3; ++i3)
        printf("B_p_r at (%d, %d, %d):  %d \n", i1, i2, i3, (int) x[i1][i2][i3]);
}

int is_equal_matrix(elem_t *x, elem_t *y, int rows, int cols) {
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      if (*(x + r*cols + c) != *(y + r*cols + c))
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

void negate_matrix(elem_t* in, elem_t* out, int rows, int cols) {
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      *(out + r*cols + c) = -*(in + r*cols + c);
    }
  }
}

void add_matrix(elem_t* A, elem_t* B, elem_t* out, int rows, int cols) {
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      *(out + r*cols + c) = *(A +r*cols + c) + *(B +r*cols + c);
    }
  }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

//     static elem_t A_MATRIX_NAME[A_ROWS][A_COLS] row_align(1);
//     static elem_t B_MATRIX_NAME[B_ROWS][B_COLS] row_align(1);
// #if NO_BIAS==0
//     static acc_t D_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
// #endif
//     static acc_t gold[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
//     static acc_t C_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif

    static elem_t Bdyn[12][4];
    static elem_t Quu_inv[4][4];
    static elem_t Kinf[4][12];
    static elem_t AmBKt[12][12];
    static elem_t p[NHORIZON][12][1];
    static elem_t p_gold[NHORIZON][12][1];
    static elem_t d[NHORIZON][4][1];
    static elem_t gold[NHORIZON][4][1]; // reference output
    static elem_t r[NHORIZON][4][1];
    static elem_t q[NHORIZON][12][1];

#define REPEAT_TEST_ITERS 1
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    for (size_t i = 0; i < 12; i++) {
      for (size_t j = 0; j < 4; j++) {
        Bdyn[i][j] = (elem_t) (RAND % (1<<2) == 0);
        Kinf[j][i] = (elem_t) (RAND % (1<<2) == 0);
      }
    }
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        Quu_inv[i][j] = (elem_t) (RAND % (1<<2) == 0);
      }
    }
    for (size_t i = 0; i < 12; i++) {
      for (size_t j = 0; j < 12; j++) {
        AmBKt[i][j] = (elem_t) (RAND % (1<<2) == 0);
      }
    }
    for (size_t i = 0; i < 12; i++) {
      elem_t elem = (elem_t) (RAND % (1<<2) == 0);
      p[NHORIZON-1][i][0] = elem;
      p_gold[NHORIZON-1][i][0] = elem;
    }
    for (size_t i = 0; i < NHORIZON; i++) {
      for (size_t j = 0; j < 4; j++) {
        r[i][j][0] = (elem_t) (RAND % (1<<2) == 0);
      }
    }
    for (size_t i = 0; i < NHORIZON; i++) {
      for (size_t j = 0; j < 12; j++) {
        q[i][j][0] = (elem_t) (RAND % (1<<2) == 0);
      }
    }

    {
      static elem_t B_p[4][1];
      static elem_t B_p_r[4][1];
      static elem_t K_r[12][1];
      static elem_t K_r_neg[12][1];
      static elem_t AmBKt_p[12][1];
      static elem_t q_AmBKt_p[12][1];

      for (int i = NHORIZON - 2; i >= 0; i--) {
        tiled_matmul_outer_eigen(Bdyn, p_gold[i+1], B_p, 4, 12, 1, true, false);
        add_matrix(B_p, r[i], B_p_r, 4, 1);
        tiled_matmul_outer_eigen(Quu_inv, B_p_r, gold[i], 4, 4, 1, true, false);
        tiled_matmul_outer_eigen(Kinf, r[i], K_r, 12, 4, 1, true, false);
        tiled_matmul_outer_eigen(AmBKt, p_gold[i+1], AmBKt_p, 12, 12, 1, false, false);
        negate_matrix(K_r, K_r_neg, 12, 1);
        add_matrix(q[i], AmBKt_p, q_AmBKt_p, 12, 1);
        add_matrix(q_AmBKt_p, K_r_neg, p_gold[i], 12, 1);
      }
    }

    printf("Finished baseline\n");

    // Generated implementation
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("Correct result\n");
}
