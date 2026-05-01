## tiling_example

SUMMARY: Increasing scratchpad tile size for the Y dimension of a 512x512 (X x Z) matrix A and 512x512 (Z x Y) matrix B multiplication. Reduces redundant loads of B by reusing a larger loaded tile across more inner-loop iterations.

Original code:

```c
uint32_t b_offset = 16 * 16 * 4 * 8 * sizeof(int8_t); 

for (int_fast32_t y = 0; y < 8; y++) {
    uint32_t b_base_y = 64 * y; 

    // Load B matrix slice
    for (int_fast32_t zo = 0; zo < 8; zo++) {
        uint32_t b_zo_offset = 4 * 16 * zo; // Number of columns per zo iteration
        for (int_fast32_t z = 0; z < 4; z++) {
            uint32_t b_index = ((zo * 4 + z) * ((16 * 4) * 16)) / 16; // Divide number of elements by 16 since scratchpad is row-indexed
            mvin3(&B[b_zo_offset + 16 * z][b_base_y], b_offset + b_index, 16 * 4, 16);
        }
    }

    for (int_fast32_t x = 0; x < 32; x++) {
        uint32_t res = 1 << 31;
        uint32_t a_base_x = 16 * x; 

        // Load A matrix slice
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t a_index = (zo * (16 * 4) * 16) / 16;
            mvin2(&A[a_base_x][64 * zo], a_index, 16 * 4, 16);
        }

        // Computation
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t a_index = (zo * (16 * 4) * 16) / 16;
            for (int_fast32_t z = 0; z < 4; z++) {
                uint32_t preload_flag = (zo == 0 && z == 0) ? 0 : 0x40000000;
                for (int_fast32_t y_in_o = 0; y_in_o < 4; y_in_o++) {
                    uint32_t preload_index = ((zo * 4 + z) * ((16 * 4) * 16) + y_in_o * (16 * 16)) / 16; // Find correct scratchpad index to load B from
                    preload(b_offset + preload_index, res + (y_in_o * (16 * 16)) / 16 | preload_flag, 16, 16, 16, 16);
                    compute_preloaded(a_index + (z * (16 * 16)) / 16, ~((uint32_t)0), 16, 16, 16, 16);
                }
            }
        }

        // Store C matrix slice
        for (int_fast32_t y_in_o = 0; y_in_o < 4; y_in_o++) {
            mvout(&C[a_base_x][b_base_y + 16 * y_in_o], res + (y_in_o * (16 * 16)) / 16, 16, 16); // Divide number of elements by 16 since accumulator is row-indexed
        }
    }
}
```

Retiled code:

```c
uint32_t b_offset = 16 * 16 * 4 * 8 * sizeof(int8_t); 

for (int_fast32_t y = 0; y < 2; y++) { // Reduce number of y dimension outer loop iterations
    uint32_t b_base_y = 256 * y;

    // Load larger B matrix slice
    // Tiling reduces redundant loads of B matrix, reducing data movement and increasing data reuse
    for (int_fast32_t zo = 0; zo < 8; zo++) {
        uint32_t b_zo_offset = 4 * 16 * zo; // Number of columns per zo iteration
        for (int_fast32_t z = 0; z < 4; z++) {
            for (int_fast32_t y_in = 0; y_in < 4; y_in++) {
                uint32_t b_index = (((zo * 4 + z) * 4 + y_in) * ((16 * 4) * 16)) / 16; // Divide number of elements by 16 since scratchpad is row-indexed
                mvin3(&B[b_zo_offset + 16 * z][b_base_y + 64 * y_in], b_offset + b_index, 16 * 4, 16);
            }
        }
    }

    for (int_fast32_t x = 0; x < 32; x++) {
        uint32_t res = 1 << 31;
        uint32_t a_base_x = 16 * x;

        // Load A matrix slice
        // Tiling reduces redundant loads of A matrix, reducing data movement and increasing data reuse
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t a_index = (zo * (16 * 4) * 16) / 16;
            mvin2(&A[a_base_x][64 * zo], a_index, 16 * 4, 16);
        }

        // Computation
        for (int_fast32_t zo = 0; zo < 8; zo++) {
            uint32_t a_index = (zo * (16 * 4) * 16) / 16;
            for (int_fast32_t z = 0; z < 4; z++) {
                uint32_t preload_flag = (zo == 0 && z == 0) ? 0 : 0x40000000;
                for (int_fast32_t y_in_o = 0; y_in_o < 16; y_in_o++) { // Increase number of Y dimension inner loop iterations to increase tile size
                    uint32_t preload_index = (((zo * 4 + z) * 4) * ((16 * 4) * 16) + y_in_o * (16 * 16)) / 16; // Find correct scratchpad index to load B from
                    preload(b_offset + preload_index, res + (y_in_o * (16 * 16)) / 16 | preload_flag, 16, 16, 16, 16);
                    compute_preloaded(a_index + (z * (16 * 16)) / 16, ~((uint32_t)0), 16, 16, 16, 16);
                }
            }
        }

        // Store C matrix slice
        for (int_fast32_t y_in_o = 0; y_in_o < 16; y_in_o++) { // Move out a larger tile in the Y dimension
            mvout(&C[a_base_x][b_base_y + 16 * y_in_o], res + (y_in_o * 16 * 16) / 16, 16, 16); // Divide number of elements by 16 since accumulator is row-indexed
        }
    }
}
```

## if_example_conv

SUMMARY: Loading data to different scratchpad locations in a 2D convolution operation to increase data reuse. Splits an outer loop and uses if-gated loads so weights are only moved in once for the entire batch and inputs only when needed.

Original code:

```c
void solution(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
  ...
  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t ocol_o = 0; ocol_o < 3; ocol_o++) {
      for (int_fast32_t orow = 0; orow < 56; orow++) { // Split this loop
        uint32_t res = 1 << 31;
        for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
          mvin(&bias[0][16 * och_o], res + ((och_o) * (256))/16, 16, (16) );
        }
        for (int_fast32_t krow = 0; krow < 3; krow++) {
          uint32_t i_s = 0;
          for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
            uint32_t w_s = 16 * 16 * 4 * 3 * sizeof(int8_t) / 16;
            for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
              mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][0], w_s + ((kch_o) * (1024))/16, 16*(4), (16) );
            }
            config_ld(64, 1.0f, (16), 2);
            mvin3( &inp[(b)][(krow + orow)][(kcol + 16 * ocol_o)][0], i_s + ((kcol) * (1024))/16, 16*(4), (16) );
...
```

Optimized code:

```c
void solution(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
  ...
  uint32_t i_s = 0;
  uint32_t w_s = 16 * 16 * 4 * 3 * 30 * sizeof(int8_t) / 16;
  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t ocol_o = 0; ocol_o < 3; ocol_o++) {
      for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) { // First part of the split loop
        for (int_fast32_t orow_io = 0; orow_io < 4; orow_io++) { // Second part of the split loop
          for (int_fast32_t orow_ii = 0; orow_ii < 7; orow_ii++) { // Third part of the split loop
            uint32_t res = 1 << 31;
            for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
              mvin(&bias[0][16 * och_o], res + ((och_o) * (256))/16, 16, (16) );
            }
            for (int_fast32_t krow = 0; krow < 3; krow++) {
              for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
                // Only need to mvin weights once since we are loading the whole weight tensor
                if (b == 0) {
                  if (orow_o == 0) {
                    if (orow_io == 0) {
                      if (orow_ii == 0) {
                        if (ocol_o == 0) {
                          for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                            mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][0], w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16, 16*(4), (16) );
                          }
                        }
                      }
                    }
                  }
                }
                // Different values of krow, orow_ii, and orow_io can result in the same value for (krow + orow_ii + 7 * orow_io)
                // If statement prevents redundant loads of the same data due to nested loop structure
                if (orow_ii + 7 * orow_io == 0 || krow == 2) {
                  config_ld(64, 1.0f, (16), 2);
                  mvin3( &inp[(b)][(krow + orow_ii + 7 * orow_io + 28 * orow_o)][(kcol + 16 * ocol_o)][0], i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024))/16, 16*(4), (16) );
                }
...
```

## if_example_matmul

SUMMARY: Utilizing conditional execution in a matrix multiplication to gate redundant data movement. With outer-loop-aware if-gates around mvin2/mvin3, each piece of A and B is loaded only once instead of being reloaded each inner iteration.

Original code:

```c
void solution(int8_t A[12544][64], int8_t B[64][256], int8_t C[12544][256]) {
  // 3,435,455 cycles
  ...
  uint32_t a = 0;
  uint32_t b = 16 * 16 * 4 * 1;
  uint32_t res = 1 << 31;
  for (int_fast32_t i = 0; i < 784; i++) {
    for (int_fast32_t j = 0; j < 4; j++) {
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvin( 0, res + ((j_in_o) * (256))/16,(16 + 0), (16 + 0) );
      }
      for (int_fast32_t ko = 0; ko < 1; ko++) {
        mvin2( &A[(16 * i)][64 * ko], a + ((ko) * (1024))/16, 16*(4 + 0), (16 + 0) );
        for (int_fast32_t k = 0; k < 4; k++) {
          mvin3(&B[(16 * k + 64 * ko)][64 * j], b + ((ko) * (4096) + (k) * (1024))/16, 16*(4 + 0), (16 + 0) );
        }
        for (int_fast32_t k = 0; k < 4; k++) {
          for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
            if (ko == 0 && k == 0) {
              preload(b + ((ko) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, (res + ((j_in_o) * (256))/16), (16 + 0), (16 + 0), (16 + 0), (16 + 0));
            } else {
              preload(b + ((ko) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, (res + ((j_in_o) * (256))/16) | 0x40000000, (16 + 0), (16 + 0), (16 + 0), (16 + 0));
            }
            compute_preloaded(a + ((ko) * (1024) + (k) * (256))/16, ~((uint32_t)0), (16 + 0), (16 + 0), 16, 16);
          }
        }
      }
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvout( &C[(16 * i)][16 * j_in_o + 64 * j], res + ((j_in_o) * (256))/16, (16 + 0), (16 + 0) );
      }
    }
  }
  fence();
}
```

Optimized code:

```c
void solution(int8_t A[12544][64], int8_t B[64][256], int8_t C[12544][256]) {
  // 1,519,076 cycles
  ...
  uint32_t a = 0;
  uint32_t b = 16 * 16 * 4 * 1 * 196 * sizeof(int8_t) / 16;
  uint32_t res = 1 << 31;
  for (int_fast32_t i = 0; i < 784; i++) {
    for (int_fast32_t j = 0; j < 4; j++) {
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvin( 0, res + ((j) * (1024) + (j_in_o) * (256))/16,(16 + 0), (16 + 0) );
      }
      if (j == 0) { // Only need to load A once per i block
        mvin2( &A[(16 * i)][0], a + ((i / 4) * (1024))/16, 16*(4 + 0), (16 + 0) );
      }
      if (i == 0) { // Only need to load B once; we are loading the whole matrix
        for (int_fast32_t k = 0; k < 4; k++) {
          mvin3(&B[(16 * k)][64 * j], b + ((j) * (4096) + (k) * (1024))/16, 16*(4 + 0), (16 + 0) );
        }
      }
      for (int_fast32_t k = 0; k < 4; k++) {
        for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
          preload(b + ((j) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, res + ((j) * (1024) + (j_in_o) * (256))/16 | 0x40000000, (16 + 0), (16 + 0), (16 + 0), (16 + 0));
          compute_preloaded(a + ((i / 4) * (1024) + (k) * (256))/16, ~((uint32_t)0), (16 + 0), (16 + 0), 16, 16);
        }
      }
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvout( &C[(16 * i)][16 * j_in_o + 64 * j], res + ((j) * 1024 + (j_in_o) * (256))/16, (16 + 0), (16 + 0) );
      }
    }
  }
  fence();
}
```
