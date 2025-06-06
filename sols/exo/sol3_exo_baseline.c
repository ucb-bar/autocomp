void test(int8_t A[3136][128], int8_t B[128][512], int8_t C[3136][512]) {
  config_st((512));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((512), 1.0f, 16, 2);
  config_ld((128), 1.0f, 16, 1);
  config_ld(0, 1.0f, 0, 0);

  for (int_fast32_t i = 0; i < 196; i++) {
    for (int_fast32_t j = 0; j < 8; j++) {
      uint32_t res = 1 << 31;
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvin( 0, res + ((j_in_o) * (256))/16,(16 + 0), (16 + 0) );
      }
      uint32_t a = 0;
      uint32_t b = 16 * 16 * 4 * 2 * sizeof(int8_t) / 16;
      for (int_fast32_t ko = 0; ko < 2; ko++) {
        mvin2( &A[(16 * i)][64 * ko], a + ((ko) * (1024))/16, 16*(4 + 0), (16 + 0) );
        for (int_fast32_t k = 0; k < 4; k++) {
          mvin3( &B[(64 * ko + 16 * k)][64 * j], b + ((ko) * (4096) + (k) * (1024))/16, 16*(4 + 0), (16 + 0) );
        }
        for (int_fast32_t k = 0; k < 4; k++) {
          for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
            preload(b + ((ko) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, res + ((j_in_o) * (256))/16 | 0x40000000, (16 + 0), (16 + 0), (16 + 0), (16 + 0));
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
