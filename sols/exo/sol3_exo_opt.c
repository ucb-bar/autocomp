void test(int8_t A[3136][128], int8_t B[128][512], int8_t C[3136][512]) {
  config_st((512));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((512), 1.0f, 16, 2);
  config_ld((128), 1.0f, 16, 1);
  config_ld(0, 1.0f, 0, 0);

  uint32_t res = 1 << 31;
  uint32_t a = 0;
  uint32_t b = 16 * 16 * 4 * 2 * 49 * sizeof(int8_t) / 16;
  for (int_fast32_t io = 0; io < 4; io++) {
    for (int_fast32_t i = 0; i < 49; i++) {
      for (int_fast32_t j = 0; j < 8; j++) {
        mvin( 0, res + ((j) * (1024))/16,(16), (16) );
        mvin( 0, res + ((j) * (1024) + 256)/16,(16), (16) );
        mvin( 0, res + ((j) * (1024) + (2) * (256))/16,(16), (16) );
        mvin( 0, res + ((j) * (1024) + (3) * (256))/16,(16), (16) );
        for (int_fast32_t ko = 0; ko < 2; ko++) {
          if (j == 0) {
            mvin2( &A[(16 * i + 784 * io)][64 * ko], a + ((i) * (2048) + (ko) * (1024))/16, 16*(4), (16) );
          }
          if (io == 0) {
            if (i == 0) {
              mvin3( &B[(64 * ko)][64 * j], b + ((j) * (8192) + (ko) * (4096))/16, 16*(4), (16) );
            }
          }
          if (io == 0) {
            if (i == 0) {
              mvin3( &B[(16 + 64 * ko)][64 * j], b + ((j) * (8192) + (ko) * (4096) + 1024)/16, 16*(4), (16) );
            }
          }
          if (io == 0) {
            if (i == 0) {
              mvin3( &B[(32 + 64 * ko)][64 * j], b + ((j) * (8192) + (ko) * (4096) + (2) * (1024))/16, 16*(4), (16) );
            }
          }
          if (io == 0) {
            if (i == 0) {
              mvin3( &B[(48 + 64 * ko)][64 * j], b + ((j) * (8192) + (ko) * (4096) + (3) * (1024))/16, 16*(4), (16) );
            }
          }
          preload(b + ((j) * (8192) + (ko) * (4096))/16, res + ((j) * (1024))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + 256)/16, res + ((j) * (1024) + 256)/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (2) * (256))/16, res + ((j) * (1024) + (2) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (3) * (256))/16, res + ((j) * (1024) + (3) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + 1024)/16, res + ((j) * (1024))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + 256)/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + 1024 + 256)/16, res + ((j) * (1024) + 256)/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + 256)/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + 1024 + (2) * (256))/16, res + ((j) * (1024) + (2) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + 256)/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + 1024 + (3) * (256))/16, res + ((j) * (1024) + (3) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + 256)/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (2) * (1024))/16, res + ((j) * (1024))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (2) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (2) * (1024) + 256)/16, res + ((j) * (1024) + 256)/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (2) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (2) * (1024) + (2) * (256))/16, res + ((j) * (1024) + (2) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (2) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (2) * (1024) + (3) * (256))/16, res + ((j) * (1024) + (3) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (2) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (3) * (1024))/16, res + ((j) * (1024))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (3) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (3) * (1024) + 256)/16, res + ((j) * (1024) + 256)/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (3) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (3) * (1024) + (2) * (256))/16, res + ((j) * (1024) + (2) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (3) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
          preload(b + ((j) * (8192) + (ko) * (4096) + (3) * (1024) + (3) * (256))/16, res + ((j) * (1024) + (3) * (256))/16 | 0x40000000, (16), (16), (16), (16));
          compute_preloaded(a + ((i) * (2048) + (ko) * (1024) + (3) * (256))/16, ~((uint32_t)0), (16), (16), 16, 16);
        }
        mvout( &C[(16 * i + 784 * io)][64 * j], res + ((j) * (1024))/16, (16), (16) );
        mvout( &C[(16 * i + 784 * io)][16 + 64 * j], res + ((j) * (1024) + 256)/16, (16), (16) );
        mvout( &C[(16 * i + 784 * io)][32 + 64 * j], res + ((j) * (1024) + (2) * (256))/16, (16), (16) );
        mvout( &C[(16 * i + 784 * io)][48 + 64 * j], res + ((j) * (1024) + (3) * (256))/16, (16), (16) );
      }
    }
  }
  fence();
}