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
#define OUTPUT_MATRIX_NAME u
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

#define NHORIZON 5
#define OUTPUT_DIM_1 NHORIZON
#define OUTPUT_DIM_2 4
#define OUTPUT_DIM_3 1

int full_is_equal(elem_t x[OUTPUT_DIM_1][OUTPUT_DIM_2][OUTPUT_DIM_3], elem_t y[OUTPUT_DIM_1][OUTPUT_DIM_2][OUTPUT_DIM_3]) {
  for (size_t i1 = 0; i1 < OUTPUT_DIM_1; ++i1)
    for (size_t i2 = 0; i2 < OUTPUT_DIM_2; ++i2)
      for (size_t i3 = 0; i3 < OUTPUT_DIM_3; ++i3)
        if (x[i1][i2][i3] != y[i1][i2][i3]) {
          return 0;
        }
  return 1;
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

    static elem_t Adyn[12][12];
    static elem_t Bdyn[12][4];
    static elem_t Kinf[4][12];
    static elem_t x[NHORIZON + 1][12][1];
    static elem_t x_gold[NHORIZON + 1][12][1];
    static elem_t d[NHORIZON][4][1];
    static elem_t gold[NHORIZON][4][1]; // reference output
    static elem_t u[NHORIZON][4][1]; // output

#define REPEAT_TEST_ITERS 5
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    // printf("Init A\n");
    for (size_t i = 0; i < 12; i++) {
      for (size_t j = 0; j < 12; j++) {
        Adyn[i][j] = (elem_t) (RAND % (1<<3) == 0);
      }
    }
    for (size_t i = 0; i < 12; i++) {
      for (size_t j = 0; j < 4; j++) {
        Bdyn[i][j] = (elem_t) (RAND % (1<<3) == 0);
        Kinf[j][i] = (elem_t) (RAND % (1<<3) == 0);
      }
    }
    for (size_t i = 0; i < 12; i++) {
        elem_t elem = (elem_t) (RAND % (1<<3) == 0);
        x[0][i][0] = elem;
        x_gold[0][i][0] = elem;
    }
    for (size_t i = 0; i < NHORIZON; i++) {
        for (size_t j = 0; j < 4; j++) {
            d[i][j][0] = (elem_t) (RAND % (1<<3) == 0);
        }
    }

    {
      static acc_t Kinf_x[4][1] row_align_acc(1);
      static acc_t Kinf_x_negated[4][1] row_align_acc(1);
      static acc_t d_i_negated[4][1] row_align_acc(1);
      static acc_t A_x[12][1] row_align_acc(1);
      static acc_t B_u[12][1] row_align_acc(1);

      for (int i = 0; i < NHORIZON; i++) {
        tiled_matmul_outer_eigen(Kinf, x_gold[i], Kinf_x, 4, 12, 1, false, false);
        negate_matrix(Kinf_x, Kinf_x_negated, 4, 1);
        negate_matrix(d[i], d_i_negated, 4, 1);
        add_matrix(Kinf_x_negated, d_i_negated, gold[i], 4, 1);

        tiled_matmul_outer_eigen(Adyn, x_gold[i], A_x, 12, 12, 1, false, false);
        tiled_matmul_outer_eigen(Bdyn, gold[i], B_u, 12, 4, 1, false, false);
        
        add_matrix(A_x, B_u, x_gold[i+1], 12, 1);
      }
    }

    printf("Finished baseline\n");

    // Generated implementation
    // SUBSTITUTE HERE
    // SUBSTITUTE END

  }
  printf("Correct result\n");
}
 