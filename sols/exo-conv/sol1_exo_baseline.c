void test(int8_t inp[4][30][30][128], int8_t weights[3][3][128][128], int32_t bias[1][128], int8_t output[4][28][28][128]) {
  config_st((128));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((128), 1.0f, 16, 1);
  config_ld(0, 1.0f, 16, 0);

  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t orow = 0; orow < 28; orow++) {
      for (int_fast32_t ocol_o = 0; ocol_o < 1; ocol_o++) {
        uint32_t res = 1 << 31;
        for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
          mvin( &bias[0][16 * och_o], res + ((och_o) * (256))/16, 16, (16) );
        }
        for (int_fast32_t krow = 0; krow < 3; krow++) {
          uint32_t i_s = 0;
          for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
            uint32_t w_s = 16 * 16 * 8 * 3 * sizeof(int8_t) / 16;
            for (int_fast32_t kch_o = 0; kch_o < 8; kch_o++) {
              for (int_fast32_t och_o_o = 0; och_o_o < 2; och_o_o++) {
                mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][64 * och_o_o], w_s + ((kch_o) * (2048) + (4 * och_o_o) * (256))/16, 16*(4), (16) );
              }
            }
            for (int_fast32_t kch_o_o = 0; kch_o_o < 2; kch_o_o++) {
              config_ld(128, 1.0f, 16, 2);
              mvin3( &inp[(b)][(krow + orow)][(kcol + 16 * ocol_o)][64 * kch_o_o], i_s + ((kcol) * (2048) + (4 * kch_o_o) * (256))/16, 16*(4), (16) );
            }
            for (int_fast32_t kch_o = 0; kch_o < 8; kch_o++) {
              for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
                preload(w_s + ((kch_o) * (2048) + (och_o) * (256))/16, res + ((och_o) * (256))/16 | 0x40000000, (16), (16), (16), (16));
                compute_preloaded(i_s + ((kcol) * (2048) + (kch_o) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
              }
            }
          }
        }
        for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
          mvout( &output[(b)][(orow)][(16 * ocol_o)][16 * och_o], res + ((och_o) * (256))/16, (16), (16) );
        }
      }
      uint32_t res = 1 << 31;
      for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
        mvin( &bias[0][16 * och_o], res + ((och_o) * (208))/16, 16, (12) );
      }
      for (int_fast32_t krow = 0; krow < 3; krow++) {
        uint32_t i_s = 0;
        for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
          uint32_t w_s = 16 * 12 * 8 * 3 * sizeof(int8_t) / 16;
          for (int_fast32_t kch_o = 0; kch_o < 8; kch_o++) {
            for (int_fast32_t och_o_o = 0; och_o_o < 2; och_o_o++) {
              mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][64 * och_o_o], w_s + ((kch_o) * (2048) + (4 * och_o_o) * (256))/16, 16*(4), (16) );
            }
          }
          for (int_fast32_t kch_o_o = 0; kch_o_o < 2; kch_o_o++) {
            config_ld(128, 1.0f, 12, 2);
            mvin3( &inp[(b)][(krow + orow)][(16 + kcol)][64 * kch_o_o], i_s + ((kcol) * (1536) + (4 * kch_o_o) * (192))/16, 16*(4), (12) );
          }
          for (int_fast32_t kch_o = 0; kch_o < 8; kch_o++) {
            for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
              preload(w_s + ((kch_o) * (2048) + (och_o) * (256))/16, res + ((och_o) * (208))/16 | 0x40000000, (16), (16), (16), (12));
              compute_preloaded(i_s + ((kcol) * (1536) + (kch_o) * (192))/16, ~((uint32_t)0), (16), (12), 16, 16);
            }
          }
        }
      }
      for (int_fast32_t och_o = 0; och_o < 8; och_o++) {
        mvout( &output[(b)][(orow)][(16)][16 * och_o], res + ((och_o) * (208))/16, (16), (12) );
      }
    }
  }
  fence();
}
