void test(int8_t A[512][512], int8_t B[512][512], int8_t C[512][512]) {
  config_st((512));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((512), 1.0f, 16, 2);
  config_ld((512), 1.0f, 16, 1);
  config_ld(0, 1.0f, 0, 0);

  for (int_fast32_t i = 0; i < 32; i++) {
    for (int_fast32_t j = 0; j < 8; j++) {
      uint32_t res = 1 << 31;
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvin( 0, res + ((j_in_o) * (256))/16, 16, 16 );
      }
      uint32_t a = 0;
      uint32_t b = 16 * 16 * 4 * 8 * sizeof(int8_t);
      for (int_fast32_t ko = 0; ko < 8; ko++) {
        mvin2( &A[(16 * i)][64 * ko], a + ((ko) * (1024))/16, 16*(4), 16 );
        for (int_fast32_t k = 0; k < 4; k++) {
          mvin3( &B[(64 * ko + 16 * k)][64 * j], b + ((ko) * (4096) + (k) * (1024))/16, 16*(4), 16 );
        }
        for (int_fast32_t k = 0; k < 4; k++) {
          for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
            preload(b + ((ko) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, (res + ((j_in_o) * (256))/16) | 0x40000000, 16, 16, 16, 16);
            compute_preloaded(a + ((ko) * (1024) + (k) * (256))/16, ~((uint32_t)0), 16, 16, 16, 16);
          }
        }
      }
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvout( &C[(16 * i)][64 * j + 16 * j_in_o], res + ((j_in_o) * (256))/16, 16, 16 );
      }
    }
  }
  fence();
}
