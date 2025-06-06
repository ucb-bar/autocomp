void test(int8_t inp[4][16][16][256], int8_t weights[3][3][256][256], int32_t bias[1][256], int8_t output[4][14][14][256]) {
  config_st((256));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((256), 1.0f, 16, 1);
  config_ld(0, 1.0f, 16, 0);

  for (int_fast32_t och_out = 0; och_out < 4; och_out++) {
    for (int_fast32_t b = 0; b < 4; b++) {
      for (int_fast32_t orow = 0; orow < 14; orow++) {
        uint32_t res = 1 << 31;
        for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
          mvin( &bias[0][64 * och_out + 16 * och_o], res + ((och_o) * (240))/16, 16, (14 + 0) );
        }
        for (int_fast32_t krow = 0; krow < 3; krow++) {
          uint32_t i_s = 0;
          for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
            uint32_t w_s = 16 * 14 * 16 * 3 * sizeof(int8_t) / 16;
            for (int_fast32_t kch_o = 0; kch_o < 16; kch_o++) {
              mvin2( &weights[(krow)][(kcol)][(16 * kch_o)][64 * och_out], w_s + ((kch_o) * (1024))/16, 16*(4 + 0), (16 + 0) );
            }
            config_ld(256, 1.0f, (14), 2);
            for (int_fast32_t kch_o_o = 0; kch_o_o < 4; kch_o_o++) {
              mvin3( &inp[(b)][(orow + krow)][(kcol)][(64 * kch_o_o)], i_s + ((kcol) * (3584) + (kch_o_o) * (896))/16, 16*(4), (14) );
            }
            for (int_fast32_t kch_o = 0; kch_o < 16; kch_o++) {
              for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
                preload(w_s + ((kch_o) * (1024) + (och_o) * (256))/16, res + ((och_o) * (240))/16 | 0x40000000, (16 + 0), (16 + 0), (16 + 0), (14 + 0));
                compute_preloaded(i_s + ((kcol) * (3584) + (kch_o) * (224))/16, ~((uint32_t)0), (16 + 0), (14 + 0), 16, 16);
              }
            }
          }
        }
        for (int_fast32_t och_o = 0; och_o < 4; och_o++) {
          mvout( &output[(b)][(orow)][0][16 * och_o + 64 * och_out], res + ((och_o) * (240))/16, (16 + 0), (14 + 0) );
        }
      }
    }
  }
  fence();
}
