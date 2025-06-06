void test(int8_t inp[4][58][58][64], int8_t weights[3][3][64][64], int32_t bias[1][64], int8_t output[4][56][56][64]) {
  config_st((64));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((64), 1.0f, 16, 1);
  config_ld(0, 1.0f, 16, 0);

  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t orow = 0; orow < 56; orow++) {
      for (int_fast32_t ocol_o = 0; ocol_o < 3; ocol_o++) {
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
            for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
              for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
                preload(w_s + ((kch_o) * (1024) + (och_o) * (256))/16, res + ((och_o) * (256))/16 | 0x40000000, (16), (16), (16), (16));
                compute_preloaded(i_s + ((kcol) * (1024) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
              }
            }
          }
        }
        for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
          mvout(&output[(b)][(orow)][(16 * ocol_o)][16 * och_o], res + ((och_o) * (256))/16, (16), (16) );
        }
      }
      uint32_t res = 1 << 31;
      for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
        mvin(&bias[0][16 * och_o], res + ((och_o) * (144))/16, 16, (8) );
      }
      for (int_fast32_t krow = 0; krow < 3; krow++) {
        uint32_t i_s = 0;
        for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
          uint32_t w_s = 16 * 8 * 4 * 3 * sizeof(int8_t) / 16;
          for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
            mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][0], w_s + ((kch_o) * (1024))/16, 16*(4), (16) );
          }
          config_ld(64, 1.0f, (8), 2);
          mvin3( &inp[(b)][(krow + orow)][(48 + kcol)][0], i_s + ((kcol) * (512))/16, 16*(4), (8) );
          for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
            for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
              preload(w_s + ((kch_o) * (1024) + (och_o) * (256))/16, res + ((och_o) * (144))/16 | 0x40000000, (16), (16), (16), (8));
              compute_preloaded(i_s + ((kcol) * (512) + (kch_o) * (128))/16, ~((uint32_t)0), (16), (8), 16, 16);
            }
          }
        }
      }
      for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
        mvout(&output[(b)][(orow)][(48)][16 * och_o], res + ((och_o) * (144))/16, (16), (8) );
      }
    }
  }
  fence();
}
