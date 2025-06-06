void test(float Adyn[12][12], float Bdyn[12][4] float Kinf[4][12], float x[NHORIZON + 1][12][1], float d[NHORIZON][4][1], float u[NHORIZON][4][1]) {
    /* Hand-written implementation from https://github.com/ucb-bar/Accelerated-TinyMPC/blob/09b86d4edf5b21730aff200c3f16fabd99dc4a4a/src/tinympc/admm_gemmini.cpp#L611
    */

    static elem_t Kinf_x[4][1];
    static elem_t A_x[12][1];
    static elem_t B_u[12][1];
    static acc_t Kinf_x_negated[4][1] row_align_acc(1);
    static acc_t d_i_negated[4][1] row_align_acc(1);

    static uint32_t A_sp_addr = 0; // 144 elements, 0 to 35
    static uint32_t B_sp_addr = 36; // 48 elements, 36 to 47
    static uint32_t Kinf_sp_addr = 48; // 48 elements, 48 to 59

    config_ld(48, 1.000000, 4, 0);
    for (int chunk = 0; chunk < 3; chunk++) {
        mvin(Adyn[chunk*4], A_sp_addr + chunk*12, 12, 4);
    }
    mvin(Kinf, Kinf_sp_addr, 12, 4);

    config_ld(16, 1.000000, 4, 0);
    for (int chunk = 0; chunk < 3; chunk++) {
        mvin(Bdyn[chunk*4], B_sp_addr + chunk*4, 4, 4);
    }

    config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
    config_st(4);
    config_ld(4, 1.000000, 4, 1);
    config_ld(4, 1.000000, 4, 2);
    for (int i = 0; i < NHORIZON; i++)
    {
        // tiled_matmul_spad_dram(Kinf_sp_addr, solver->work->x.col(i), Kinf_x, NINPUTS, false, false);
        config_ld(48, 1.000000, 4, 0);
        mvin2(&x[i][0][0], 0x3ff4, 1, 4);
        mvin2(&x[i][4][0], 0x3ff8, 1, 4);
        mvin2(&x[i][8][0], 0x3ffc, 1, 4);
        preload(0x3ff4, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x30, 0xffffffff, 4, 4, 4, 4);

        preload(0x3ff8, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x34, 0xffffffff, 4, 4, 4, 4);

        preload(0x3ffc, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x38, 0xffffffff, 4, 4, 4, 4);
        mvout(Kinf_x + 0x0, 0xc0000000, 1, 4);
        

        // tiled_matmul_spad_dram(A_sp_addr, solver->work->x.col(i), A_x, NSTATES, false, false);
        preload(0x3ff4, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x0, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0x80000004, 1, 4, 1, 4);
        compute_accumulated(0xc, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0x80000008, 1, 4, 1, 4);
        compute_accumulated(0x18, 0xffffffff, 4, 4, 4, 4);

        preload(0x3ff8, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x4, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0xc0000004, 1, 4, 1, 4);
        compute_accumulated(0x10, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0xc0000008, 1, 4, 1, 4);
        compute_accumulated(0x1c, 0xffffffff, 4, 4, 4, 4);

        preload(0x3ffc, 0xc0000000, 1, 4, 1, 4);
        compute_preloaded(0x8, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0xc0000004, 1, 4, 1, 4);
        compute_accumulated(0x14, 0xffffffff, 4, 4, 4, 4);

        preload(0xffffffff, 0xc0000008, 1, 4, 1, 4);
        compute_accumulated(0x20, 0xffffffff, 4, 4, 4, 4);
        mvout(&A_x[0][0], 0xc0000000, 1, 4);
        mvout(&A_x[4][0], 0xc0000004, 1, 4);
        mvout(&A_x[8][0], 0xc0000008, 1, 4);

        fence();

        // (solver->work->u.col(i)).noalias() = -Kinf_x - solver->work->d.col(i);
        negate_matrix(Kinf_x, Kinf_x_negated, 4, 1);
        negate_matrix(d[i], d_i_negated, 4, 1);
        add_matrix(Kinf_x_negated, d_i_negated, u[i], 4, 1);
        
        // tiled_matmul_spad_dram(B_sp_addr, solver->work->u.col(i), B_u, NSTATES, false, false);
        config_ld(16, 1.000000, 4, 0);
        mvin2(u[i] + 0x0, 0x3ffc, 1, 4);
        preload(0x3ffc, 0x80000000, 1, 4, 1, 4);
        compute_preloaded(0x24, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0x80000004, 1, 4, 1, 4);
        compute_accumulated(0x28, 0xffffffff, 4, 4, 4, 4);
        preload(0xffffffff, 0x80000008, 1, 4, 1, 4);
        compute_accumulated(0x2c, 0xffffffff, 4, 4, 4, 4);
        mvout(&B_u[0][0], 0xc0000000, 1, 4);
        mvout(&B_u[4][0], 0xc0000004, 1, 4);
        mvout(&B_u[8][0], 0xc0000008, 1, 4);
        fence();
        
        // (solver->work->x.col(i + 1)).noalias() = A_x + B_u;
        add_matrix(A_x, B_u, x[i+1], 12, 1);
    }
}