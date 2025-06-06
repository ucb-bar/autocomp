c
void test(float Kinf[12][4], float x_i[12][1], float d_i[4][1], float u_i[4][1]) {
    /* This function implements the operation u_i = -Kinf * x_i - d_i.
    */

    static acc_t Kinf_x_i_prod[4][1];

    // Configure the accelerator for matrix multiplication
    config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, true, false);
    config_ld(4 * sizeof(float), -1, 0); // Kinf matrix (4x12) transposed, so 4 columns, negated
    config_ld(1 * sizeof(float), 1, 1); // x_i vector (12x1), so 1 column
    config_ld(1 * sizeof(float), -1, 2); // d_i vector (4x1), so 1 column, negated
    config_st(1 * sizeof(float)); // Result u_i (4x1), so 1 column

    // Memory addresses
    static uint32_t Kinf_sp_addr = 0;  // Kinf (4x12), occupies 12 rows
    static uint32_t x_i_sp_addr = 12;  // x_i (12x1), occupies 12 rows
    static uint32_t d_i_sp_addr = 24;  // d_i (4x1), occupies 4 rows
    static uint32_t prod_acc_addr = 1 << 31; // u_i result (4x1), in accumulator

    // Move Kinf and x_i to scratchpad
    mvin(&Kinf[0][0], Kinf_sp_addr, 4, 4);
    mvin2(&x_i[0][0], x_i_sp_addr, 1, 4);
    mvin(&Kinf[4][0], Kinf_sp_addr + 4, 4, 4);
    mvin2(&x_i[4][0], x_i_sp_addr + 4, 1, 4);
    mvin(&Kinf[8][0], Kinf_sp_addr + 8, 4, 4);
    mvin2(&x_i[8][0], x_i_sp_addr + 8, 1, 4);

    // Move d_i to scratchpad
    mvin3(d_i, d_i_sp_addr, 1, 4);

    // Preload and compute, using d_i as bias for one compute operation
    preload(x_i_sp_addr, prod_acc_addr, 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr, d_i_sp_addr, 4, 4, 1, 4);
    preload(x_i_sp_addr + 4, prod_acc_addr | (1 << 30), 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr + 4, 0xffffffff, 4, 4, 0, 0);
    preload(x_i_sp_addr + 8, prod_acc_addr | (1 << 30), 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr + 8, 0xffffffff, 4, 4, 0, 0);

    // Move out the result from accumulator to DRAM
    mvout(u_i, prod_acc_addr, 1, 4);

    // Ensure all operations are completed before proceeding
    fence();
}
