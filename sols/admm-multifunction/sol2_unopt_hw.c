void test(float Bdyn[12][4], float Quu_inv[4][4], float Kinf[4][12], float AmBKt[12][12], float p[NHORIZON + 1][12][1], float d[NHORIZON + 1][4][1], float r[NHORIZON][4][1], float q[NHORIZON][12][1]) {
    static elem_t B_p[4][1];
    static elem_t B_p_r[4][1];
    static elem_t K_r[12][1];
    static elem_t K_r_neg[12][1];
    static elem_t AmBKt_p[12][1];
    static elem_t q_AmBKt_p[12][1];

    for (int i = NHORIZON - 2; i >= 0; i--)
    {
        gemmini_extended_config_ex(1, 0, 0, 1, true, false);
        gemmini_extended_config_st(4, 0, 1.000000);
        gemmini_extended3_config_ld(16, 1.000000, false, 0);
        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        gemmini_loop_ws(1, 1, 3, 0, 3, 0, Bdyn, p[i+1], NULL, B_p, 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
        gemmini_fence();

        // B_p += solver->work->r.col(i);
        add_matrix(B_p, r[i], B_p_r, 4, 1);

        gemmini_extended_config_ex(1, 0, 0, 1, true, false);
        gemmini_extended_config_st(4, 0, 1.000000);
        gemmini_extended3_config_ld(16, 1.000000, false, 0);
        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        gemmini_loop_ws(1, 1, 1, 0, 3, 0, Quu_inv, B_p_r, NULL, d[i], 4, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
        gemmini_fence();

        gemmini_extended_config_ex(1, 0, 0, 1, true, false);
        gemmini_extended_config_st(4, 0, 1.000000);
        gemmini_extended3_config_ld(48, 1.000000, false, 0);
        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        gemmini_loop_ws(3, 1, 1, 0, 3, 0, Kinf, r[i], NULL, K_r, 12, 1, 1, 1, true, false, false, false, false, 0, 1, 1, false);
        gemmini_fence();

        gemmini_extended_config_ex(1, 0, 0, 1, false, false);
        gemmini_extended_config_st(4, 0, 1.000000);
        gemmini_extended3_config_ld(48, 1.000000, false, 0);
        gemmini_extended3_config_ld(4, 1.000000, false, 1);
        gemmini_extended3_config_ld(4, 1.000000, false, 2);
        gemmini_loop_ws(3, 1, 3, 0, 3, 0, AmBKt, p[i+1], NULL, AmBKt_p, 12, 1, 1, 1, false, false, false, false, false, 0, 1, 1, false);
        gemmini_fence();

        // (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        add_matrix(q[i], AmBKt_p, q_AmBKt_p, 12, 1);
        negate_matrix(K_r, K_r_neg, 12, 1);
        add_matrix(q_AmBKt_p, K_r_neg, p[i], 12, 1);  
    }
}
