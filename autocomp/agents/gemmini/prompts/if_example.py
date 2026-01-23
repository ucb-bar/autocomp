def PROMPT():
    orig_code = """void test(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
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
..."""

    new_code = """void test(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
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
..."""
    prompt_text = "Here is an example of loading data to different scratchpad locations in a 2D convolution operation to increase data reuse. Original code:\n" + orig_code + "\Optimized code\n" + new_code
    return prompt_text
