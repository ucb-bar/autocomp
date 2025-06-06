void test(float Adyn[12][12], float Bdyn[12][4] float Kinf[4][12], float x[NHORIZON + 1][12][1], float d[NHORIZON][4][1], float u[NHORIZON][4][1]) {
    static elem_t Kinf_x[4][1];
    static elem_t A_x[12][1];
    static elem_t B_u[12][1];

    for (int i = 0; i < NHORIZON; i++) {
        // define spad addresses for cached matrices
        // spad is row addressed and each row is 4 elements wide
        static uint32_t A_sp_addr = 0; // 144 elements, 0 to 35
        static uint32_t B_sp_addr = 36; // 48 elements, 36 to 47
        static uint32_t Kinf_sp_addr = 48; // 48 elements, 48 to 59
        static uint32_t C1_sp_addr = 60; // 16 elements, 60 to 63
        static uint32_t C2_sp_addr = 64; // 144 elements, 64 to 99
        static uint32_t x_sp_addr = 100; // 12 elements (at a time), 100 to 111
        static uint32_t u_sp_addr = 112; // 12 elements (at a time), 112 to 123
        static uint32_t acc_start_addr = 1 << 31;

        // tiled_matmul_spad_dram(Kinf, x[i], Kinf_x, NINPUTS, false, false);
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
        config_st(4, 1.0);
        config_ld(48, 1.000000, 4, 0);
        config_ld(4, 1.000000, 4, 1);
        config_ld(4, 1.000000, 4, 2);
        mvin(Kinf, Kinf_sp_addr, 12, 4);
        mvin2(x[i][0], x_sp_addr, 1, 4);
        preload(x_sp_addr, acc_start_addr, 1, 4, 1, 4);
        compute_preloaded(Kinf_sp_addr, 0xffffffff, 4, 4, 4, 4);
        mvin2(x[i][4], x_sp_addr + 4, 1, 4);
        preload(x_sp_addr + 4, acc_start_addr | (1 << 30), 1, 4, 1, 4);
        compute_preloaded(Kinf_sp_addr + 4, 0xffffffff, 4, 4, 4, 4);
        mvin2(x[i][8], x_sp_addr + 8, 1, 4);
        preload(x_sp_addr + 8, acc_start_addr | (1 << 30), 1, 4, 1, 4);
        compute_preloaded(Kinf_sp_addr + 8, 0xffffffff, 4, 4, 4, 4);
        mvout(Kinf_x[0], acc_start_addr | (1 << 30), 1, 4);
        fence();

        static acc_t Kinf_x_negated[4][1] row_align_acc(1);
        static acc_t d_i_negated[4][1] row_align_acc(1);
        negate_matrix(Kinf_x, Kinf_x_negated, 4, 1);
        negate_matrix(d[i], d_i_negated, 4, 1);
        add_matrix(Kinf_x_negated, d_i_negated, u[i], 4, 1);

        // tiled_matmul_spad_dram(Adyn, x[i], A_x, NSTATES, false, false);
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
        config_st(4, 1.0);
        config_ld(48, 1.000000, 4, 0);
        config_ld(4, 1.000000, 4, 1);
        config_ld(4, 1.000000, 4, 2);
        for (int chunk = 0; chunk < 3; chunk++) {
            mvin(Adyn[chunk*4], A_sp_addr + chunk*12, 12, 4);
        }
        mvin2(x[i][0], x_sp_addr, 1, 4);
        mvin2(x[i][4], x_sp_addr + 4, 1, 4);
        mvin2(x[i][8], x_sp_addr + 8, 1, 4);

        preload(x_sp_addr, acc_start_addr, 1, 4, 1, 4);
        compute_preloaded(A_sp_addr, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, acc_start_addr + 4, 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 12, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, acc_start_addr + 8, 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 24, 0xffffffff, 4, 4, 4, 4);

        preload(x_sp_addr + 4, acc_start_addr | (1 << 30), 1, 4, 1, 4);
        compute_preloaded(A_sp_addr + 4, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, (acc_start_addr + 4) | (1 << 30), 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 4 + 12, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, (acc_start_addr + 8) | (1 << 30), 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 4 + 24, 0xffffffff, 4, 4, 4, 4);

        preload(x_sp_addr + 8, acc_start_addr | (1 << 30), 1, 4, 1, 4);
        compute_preloaded(A_sp_addr + 8, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, (acc_start_addr + 4) | (1 << 30), 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 8 + 12, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, (acc_start_addr + 8) | (1 << 30), 1, 4, 1, 4);
        compute_accumulated(A_sp_addr + 8 + 24, 0xffffffff, 4, 4, 4, 4);

        mvout(A_x[0], acc_start_addr, 1, 4);
        mvout(A_x[4], acc_start_addr + 4, 1, 4);
        mvout(A_x[8], acc_start_addr + 8, 1, 4);
        fence();
        
        // tiled_matmul_spad_dram(Bdyn, u[i], B_u, NSTATES, false, false);
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
        config_st(4, 1.0);
        config_ld(16, 1.000000, 4, 0);
        config_ld(4, 1.000000, 4, 1);
        config_ld(4, 1.000000, 4, 2);
        for (int chunk = 0; chunk < 3; chunk++) {
            mvin(Bdyn[chunk*4], B_sp_addr + chunk*4, 4, 4);
        }
        mvin2(u[i][0], x_sp_addr, 1, 4);
        preload(x_sp_addr, acc_start_addr, 1, 4, 1, 4);
        compute_preloaded(B_sp_addr, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, acc_start_addr + 4, 1, 4, 1, 4);
        compute_accumulated(B_sp_addr + 4, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, acc_start_addr + 8, 1, 4, 1, 4);
        compute_accumulated(B_sp_addr + 8, 0xffffffff, 4, 4, 4, 4);
        mvout(B_u[0], acc_start_addr, 1, 4);
        mvout(B_u[4], acc_start_addr + 4, 1, 4);
        mvout(B_u[8], acc_start_addr + 8, 1, 4);
        fence();

        add_matrix(A_x, B_u, x[i+1], 12, 1);
    }
}
