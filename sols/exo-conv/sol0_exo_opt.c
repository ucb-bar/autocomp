void test(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
  config_st((64));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((64), 1.0f, 16, 1);
  config_ld(0, 1.0f, 0, 0);

  uint32_t i_s = 0;
  uint32_t w_s = 16 * 16 * 4 * 3 * 30 * sizeof(int8_t) / 16;
  uint32_t i_s_1 = 0;
  uint32_t w_s_1 = 16 * 16 * 4 * 3 * 30 * sizeof(int8_t) / 16;
  uint32_t res = 1 << 31;
  uint32_t res_1 = 1 << 31;

  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t ocol_o = 0; ocol_o < 3; ocol_o++) {
      for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) {
        for (int_fast32_t orow_io = 0; orow_io < 4; orow_io++) {
          for (int_fast32_t orow_ii = 0; orow_ii < 7; orow_ii++) {
            mvin(&bias[0][0], res + (0)/16, 16, (16) );
            mvin(&bias[0][16], res + (256)/16, 16, (16) );
            mvin(&bias[0][32], res + ((2) * (256))/16, 16, (16) );
            mvin(&bias[0][48], res + ((3) * (256))/16, 16, (16) );
            for (int_fast32_t krow = 0; krow < 3; krow++) {
              for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
                if (ocol_o == 0) {
                  if (b == 0) {
                    if (orow_o == 0) {
                      if (orow_ii + 7 * orow_io == 0) {
                        for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                          mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][0], w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16, 16*(4), (16) );
                        }
                      }
                    }
                  }
                }
                if (orow_ii + 7 * orow_io == 0 || krow == 2) {
                  config_ld(64, 1.0f, (16), 2);
                  mvin3( &inp[(b)][(krow + orow_ii + 7 * orow_io + 28 * orow_o)][(kcol + 16 * ocol_o)][0], i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024))/16, 16*(4), (16) );
                }
                for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                  preload(w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16, res + (0)/16 | 0x40000000, (16), (16), (16), (16));
                  compute_preloaded(i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
                  preload(w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + 256)/16, res + (256)/16 | 0x40000000, (16), (16), (16), (16));
                  compute_preloaded(i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
                  preload(w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (2) * (256))/16, res + ((2) * (256))/16 | 0x40000000, (16), (16), (16), (16));
                  compute_preloaded(i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
                  preload(w_s + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (3) * (256))/16, res + ((3) * (256))/16 | 0x40000000, (16), (16), (16), (16));
                  compute_preloaded(i_s + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
                }
              }
            }
            mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(16 * ocol_o)][0], res + (0)/16, (16), (16) );
            mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(16 * ocol_o)][16], res + (256)/16, (16), (16) );
            mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(16 * ocol_o)][32], res + ((2) * (256))/16, (16), (16) );
            mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(16 * ocol_o)][48], res + ((3) * (256))/16, (16), (16) );
          }
        }
      }
    }
    for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) {
      for (int_fast32_t orow_io = 0; orow_io < 4; orow_io++) {
        for (int_fast32_t orow_ii = 0; orow_ii < 7; orow_ii++) {
          mvin(&bias[0][0], res_1 + (0)/16, 16, (8) );
          mvin(&bias[0][16], res_1 + (144)/16, 16, (8) );
          mvin(&bias[0][32], res_1 + ((2) * (144))/16, 16, (8) );
          mvin(&bias[0][48], res_1 + ((3) * (144))/16, 16, (8) );
          for (int_fast32_t krow = 0; krow < 3; krow++) {
            for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
              if (b == 0) {
                if (orow_o == 0) {
                  if (orow_ii + 7 * orow_io == 0) {
                    for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                      mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][0], w_s_1 + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16, 16*(4), (16) );
                    }
                  }
                }
              }
              if (orow_ii + 7 * orow_io == 0 || krow == 2) {
                config_ld(64, 1.0f, (8), 2);
                mvin3( &inp[(b)][(krow + orow_ii + 7 * orow_io + 28 * orow_o)][(48 + kcol)][0], i_s_1 + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512))/16, 16*(4), (8) );
              }
              for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                preload(w_s_1 + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16, res_1 + (0)/16 | 0x40000000, (16), (16), (16), (8));
                compute_preloaded(i_s_1 + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16, ~((uint32_t)0), (16), (8), 16, 16);
                preload(w_s_1 + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + 256)/16, res_1 + (144)/16 | 0x40000000, (16), (16), (16), (8));
                compute_preloaded(i_s_1 + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16, ~((uint32_t)0), (16), (8), 16, 16);
                preload(w_s_1 + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (2) * (256))/16, res_1 + ((2) * (144))/16 | 0x40000000, (16), (16), (16), (8));
                compute_preloaded(i_s_1 + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16, ~((uint32_t)0), (16), (8), 16, 16);
                preload(w_s_1 + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (3) * (256))/16, res_1 + ((3) * (144))/16 | 0x40000000, (16), (16), (16), (8));
                compute_preloaded(i_s_1 + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16, ~((uint32_t)0), (16), (8), 16, 16);
              }
            }
          }
          mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(48)][0], res_1 + (0)/16, (16), (8) );
          mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(48)][16], res_1 + (144)/16, (16), (8) );
          mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(48)][32], res_1 + ((2) * (144))/16, (16), (8) );
          mvout(&output[(b)][(orow_ii + 7 * orow_io + 28 * orow_o)][(48)][48], res_1 + ((3) * (144))/16, (16), (8) );
        }
      }
    }
  }
  fence();
}