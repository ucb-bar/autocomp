void test(float Bdyn[12][4], float Quu_inv[4][4], float Kinf[4][12], float AmBKt[12][12], float p[NHORIZON + 1][12][1], float d[NHORIZON + 1][4][1], float r[NHORIZON][4][1], float q[NHORIZON][12][1]) {
    static elem_t B_p[4][1];
    static elem_t B_p_r[4][1];
    static elem_t K_r[12][1];
    static elem_t K_r_neg[12][1];
    static elem_t AmBKt_p[12][1];
    static elem_t q_AmBKt_p[12][1];

    // void tiled_matmul_outer_eigen(float A[][], float B[][], float C[][], bool A_transpose, bool B_transpose);

    for (int i = NHORIZON - 2; i >= 0; i--) {
        // tiled_matmul_outer_eigen(solver->work->Bdyn, solver->work->p.col(i + 1), B_p, true, false);
        static uint32_t Bdyn_sp_addr = 0x24; // 48 elements, 36 to 47
        config_ld(16, 1.0, false, 0);
        for (int chunk = 0; chunk < 3; chunk++) {
            mvin(&Bdyn[chunk*4][0], Bdyn_sp_addr + chunk*4, 4, 4);
        }
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, true, false);
        config_st(4, 1.0);
        config_ld(4, 1.0, 4, 1);
        mvin2(p[i+1][0], 0x3ff4, 1, 4);
        preload(0x3ff4, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x24, 0xffffffff, 4, 4, 4, 4);
        mvin2(p[i+1][4], 0x3ff8, 1, 4);
        preload(0x3ff8, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x28, 0xffffffff, 4, 4, 4, 4);
        mvin2(p[i+1][8], 0x3ffc, 1, 4);
        preload(0x3ffc, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x2c, 0xffffffff, 4, 4, 4, 4);
        mvout(B_p, 0xc0000000, 1, 4);
        fence();

        // B_p_r = B_p + solver->work->r.col(i);
        add_matrix(B_p, r[i], B_p_r, 4, 1);

        // tiled_matmul_outer_eigen(solver->cache->Quu_inv, B_p, dcol, true, false);
        static uint32_t Quu_inv_sp_addr = 0x3c; // 16 elements, 60 to 63
        config_ld(16, 1.0, 4, 0);
        mvin(&Quu_inv[0][0], Quu_inv_sp_addr, 4, 4);
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, true, false);
        config_st(4, 1.0);
        config_ld(4, 1.0, 4, 1);
        mvin2(B_p_r, 0x3ffc, 1, 4);
        preload(0x3ffc, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x3c, 0xffffffff, 4, 4, 4, 4);
        mvout(d[i][0], 0xc0000000, 1, 4);
        fence();

        // tiled_matmul_outer_eigen(solver->cache->Kinf, solver->work->r.col(i), K_r, true, false);
        static uint32_t Kinf_sp_addr = 0x30; // 48 elements, 48 to 59
        config_ld(48, 1.0, 4, 0);
        mvin(&Kinf[0][0], Kinf_sp_addr, 12, 4);
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, true, false);
        config_st(4, 1.0);
        config_ld(4, 1.0, 4, 1);
        mvin2(r[i][0], 0x3ffc, 1, 4);
        preload(0x3ffc, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x30, 0xffffffff, 4, 4, 4, 4);
        mvout(K_r[0], 0xc0000000, 1, 4);
        preload(0xffffffff, 0x80000004, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x34, 0xffffffff, 4, 4, 4, 4);
        mvout(K_r[4], 0xc0000004, 1, 4);
        preload(0xffffffff, 0x80000008, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x38, 0xffffffff, 4, 4, 4, 4);
        mvout(K_r[8], 0xc0000008, 1, 4);
        fence();

        // tiled_matmul_outer_eigen(solver->cache->AmBKt, solver->work->p.col(i + 1), AmBKt_p, false, false);
        static uint32_t AmBKt_sp_addr = 0x40; // 144 elements, 64 to 99
        config_ld(48, 1.0, 4, 0);
        for (int chunk = 0; chunk < 3; chunk++) {
            mvin(&AmBKt[chunk*4][0], AmBKt_sp_addr + chunk*12, 12, 4);
        }
        config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
        config_st(4, 1.0);
        config_ld(4, 1.0, 4, 1);
        mvin2(p[i+1][0], 0x3ff4, 1, 4);
        preload(0x3ff4, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x40, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0x80000004, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x4c, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0x80000008, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x58, 0xffffffff, 4, 4, 4, 4);
        mvin2(p[i+1][4], 0x3ff8, 1, 4);
        preload(0x3ff8, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x44, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0xc0000004, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x50, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0xc0000008, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x5c, 0xffffffff, 4, 4, 4, 4);
        mvin2(p[i+1][8], 0x3ffc, 1, 4);
        preload(0x3ffc, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x48, 0xffffffff, 4, 4, 4, 4);
        mvout(AmBKt_p[0], 0xc0000000, 1, 4);
        preload(0xffffffff, 0xc0000004, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x54, 0xffffffff, 4, 4, 4, 4);
        mvout(AmBKt_p[4], 0xc0000004, 1, 4);
        preload(0xffffffff, 0xc0000008, 1, 4, 1, 4);
        gemmini_extended_compute_accumulated(0x60, 0xffffffff, 4, 4, 4, 4);
        mvout(AmBKt_p[8], 0xc0000008, 1, 4);
        fence();
        
        // (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        add_matrix(q[i], AmBKt_p, q_AmBKt_p, 12, 1);
        negate_matrix(K_r, K_r_neg, 12, 1);
        add_matrix(q_AmBKt_p, K_r_neg, p[i], 12, 1);  
    }
}