void test(float Adyn[12][12], float Bdyn[12][4] float Kinf[4][12], float x[NHORIZON + 1][12][1], float d[NHORIZON][4][1], float u[NHORIZON][4][1]) {
    static elem_t B_u[12][1];

    gemmini_extended_config_ex(1, 0, 0, 1, false, false);
    gemmini_extended3_config_ld(4, 1.0, false, 1);
    gemmini_extended3_config_ld(4, 1.0, false, 2);
    for (int i = 0; i < NHORIZON; i++)
    {
        gemmini_extended_config_st(4, 0, -1.0);
        gemmini_extended3_config_ld(48, 1.0, false, 0);
        gemmini_loop_ws(1, 1, 3, 0, 3, 0, Kinf, x[i], d[i], u[i], 12, 1, 1, 1, false, false, false, false, true, 0, 1, 1, false);
        gemmini_fence();
        // solver->work->u.col(i) << .001, .02, .3, 4;

        gemmini_extended_config_st(4, 0, 1.0);
        gemmini_extended3_config_ld(16, 1.0, false, 0);
        gemmini_loop_ws(3, 1, 1, 0, 3, 0, Bdyn, u[i], NULL, B_u, 4, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
        gemmini_fence();

        gemmini_extended3_config_ld(48, 1.0, false, 0);
        gemmini_loop_ws(3, 1, 3, 0, 3, 0, Adyn, x[i], B_u, x[i+1], 12, 1, 1, 1, false, false, false, false, true, 0, 1, 1, false);
        gemmini_fence();
        // (solver->work->u.col(i)).noalias() -= solver->work->d.col(i);

        // (solver->work->x.col(i + 1)).noalias() += B_u;
    }
}