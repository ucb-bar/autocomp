void test(int8_t inp[4][16][16][256], int8_t weights[3][3][256][256], int32_t bias[1][256], int8_t output[4][14][14][256]) {
  config_st((256));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((256), 1.0f, 16, 1);
  config_ld(0, 1.0f, 0, 0);

  uint32_t i_s = 0;
  uint32_t w_s = 16 * 14 * 16 * 3 * 9 * sizeof(int8_t) / 16;
  uint32_t res = 1 << 31;
  for (int_fast32_t och_out = 0; och_out < 4; och_out++) {
    for (int_fast32_t b = 0; b < 4; b++) {
      for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) {
        for (int_fast32_t orow_i = 0; orow_i < 7; orow_i++) {
          mvin( &bias[0][64 * och_out], res + ((orow_i) * (960))/16, 16, (14) );
          mvin( &bias[0][16 + 64 * och_out], res + ((orow_i) * (960) + 240)/16, 16, (14) );
          mvin( &bias[0][32 + 64 * och_out], res + ((orow_i) * (960) + (2) * (240))/16, 16, (14) );
          mvin( &bias[0][48 + 64 * och_out], res + ((orow_i) * (960) + (3) * (240))/16, 16, (14) );
          for (int_fast32_t krow = 0; krow < 3; krow++) {
            for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
              if (b == 0) {
                if (orow_o == 0) {
                  if (orow_i == 0) {
                    for (int_fast32_t kch_o = 0; kch_o < 16; kch_o++) {
                      mvin2( &weights[krow][kcol][16 * kch_o][64 * och_out], w_s + ((krow) * (49152) + (kcol) * (16384) + (kch_o) * (1024))/16, 16*(4), (16) );
                    }
                  }
                }
              }
              if (orow_i == 0 || krow == 2) {
                config_ld(256, 1.0f, (14), 2);
                for (int_fast32_t kch_o_o = 0; kch_o_o < 4; kch_o_o++) {
                  mvin3( &inp[b][krow + orow_i + 7 * orow_o][kcol][(64 * kch_o_o)], i_s + ((krow + orow_i) * (10752) + (kcol) * (3584) + (kch_o_o) * (896))/16, 16*(4), (14) );
                }
              }
              for (int_fast32_t kch_o = 0; kch_o < 16; kch_o++) {
                preload(w_s + ((krow) * (49152) + (kcol) * (16384) + (kch_o) * (1024))/16, res + ((orow_i) * (960))/16 | 0x40000000, (16), (16), (16), (14));
                compute_preloaded(i_s + ((krow + orow_i) * (10752) + (kcol) * (3584) + (kch_o) * (224))/16, ~((uint32_t)0), (16), (14), 16, 16);
                preload(w_s + ((krow) * (49152) + (kcol) * (16384) + (kch_o) * (1024) + 256)/16, res + ((orow_i) * (960) + 240)/16 | 0x40000000, (16), (16), (16), (14));
                compute_preloaded(i_s + ((krow + orow_i) * (10752) + (kcol) * (3584) + (kch_o) * (224))/16, ~((uint32_t)0), (16), (14), 16, 16);
                preload(w_s + ((krow) * (49152) + (kcol) * (16384) + (kch_o) * (1024) + (2) * (256))/16, res + ((orow_i) * (960) + (2) * (240))/16 | 0x40000000, (16), (16), (16), (14));
                compute_preloaded(i_s + ((krow + orow_i) * (10752) + (kcol) * (3584) + (kch_o) * (224))/16, ~((uint32_t)0), (16), (14), 16, 16);
                preload(w_s + ((krow) * (49152) + (kcol) * (16384) + (kch_o) * (1024) + (3) * (256))/16, res + ((orow_i) * (960) + (3) * (240))/16 | 0x40000000, (16), (16), (16), (14));
                compute_preloaded(i_s + ((krow + orow_i) * (10752) + (kcol) * (3584) + (kch_o) * (224))/16, ~((uint32_t)0), (16), (14), 16, 16);
              }
            }
          }
          mvout( &output[b][orow_i + 7 * orow_o][0][64 * och_out], res + ((orow_i) * (960))/16, (16), (14) );
          mvout( &output[b][orow_i + 7 * orow_o][0][16 + 64 * och_out], res + ((orow_i) * (960) + 240)/16, (16), (14) );
          mvout( &output[b][orow_i + 7 * orow_o][0][32 + 64 * och_out], res + ((orow_i) * (960) + (2) * (240))/16, (16), (14) );
          mvout( &output[b][orow_i + 7 * orow_o][0][48 + 64 * och_out], res + ((orow_i) * (960) + (3) * (240))/16, (16), (14) );
        }
      }
    }
  }
  fence();
}