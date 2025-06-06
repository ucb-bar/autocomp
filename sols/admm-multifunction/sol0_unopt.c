c
void test(float Kinf[12][4], float x_i[12][1], float d_i[4][1], float u_i[4][1]) {
    /* This function implements the operation u_i = -(Kinf * x_i) - d_i. However, it may not achieve optimal performance.
    */

    static acc_t Kinf_x_i_prod[4][1];

    // Configure the accelerator for matrix multiplication
    config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, true, false);
    config_ld(4 * sizeof(float), 1, 4, 0); // Kinf matrix (4x12) transposed, so 4 columns
    config_ld(1 * sizeof(float), 1, 4, 1); // x_i vector (12x1), so 1 column
    config_st(1 * sizeof(float)); // Result Kinf_x_i_prod (4x1), so 1 column

    // Memory addresses
    static uint32_t Kinf_sp_addr = 0;  // Kinf (4x12), occupies 12 rows
    static uint32_t x_i_sp_addr = 12;  // x_i (12x1), occupies 3 rows
    static uint32_t prod_acc_addr = 1 << 31; // Kinf_x_i_prod result (4x1), in accumulator

    // Move Kinf and x_i to scratchpad
    mvin(&Kinf[0][0], Kinf_sp_addr, 4, 4);
    mvin(&Kinf[4][0], Kinf_sp_addr + 4, 4, 4);
    mvin(&Kinf[8][0], Kinf_sp_addr + 8, 4, 4);

    mvin2(&x_i[0][0], x_i_sp_addr, 1, 4);
    mvin2(&x_i[4][0], x_i_sp_addr + 4, 1, 4);
    mvin2(&x_i[8][0], x_i_sp_addr + 8, 1, 4);

    // Preload and compute
    preload(x_i_sp_addr, prod_acc_addr, 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr, 0xffffffff, 4, 4, 0, 0);
    preload(x_i_sp_addr + 4, prod_acc_addr | (1 << 30), 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr + 4, 0xffffffff, 4, 4, 0, 0);
    preload(x_i_sp_addr + 8, prod_acc_addr | (1 << 30), 1, 4, 1, 4);
    compute_preloaded(Kinf_sp_addr + 8, 0xffffffff, 4, 4, 0, 0);

    // Move out the product result from accumulator to DRAM to be used for negation
    mvout(Kinf_x_i_prod, prod_acc_addr, 1, 4);

    // Negate Kinf_x_i_prod and d_i
    // Configure the negation with scale factor -1
    config_ld(1 * sizeof(float), -1, 4, 0); // Kinf_x_i_prod negation with mvin
    config_ld(1 * sizeof(float), -1, 4, 1); // d_i negation with mvin2
    config_ld(4 * sizeof(float), 1, 4, 2); // identity matrix with mvin3

    static uint32_t prod_negate_sp_addr = 16;  // Address for negated Kinf_x_i_prod
    static uint32_t di_negate_sp_addr = 20;    // Address for negated d_i
    static uint32_t I_sp_addr = 24;

    mvin(Kinf_x_i_prod, prod_negate_sp_addr, 1, 4);
    mvin2(d_i, di_negate_sp_addr, 1, 4);

    // Set up identity matrix for matrix addition
    static elem_t identity_matrix[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    mvin3(identity_matrix, I_sp_addr, 4, 4);

    // Adding the two negated results in the accumulator
    // Configure for addition
    config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
    preload(I_sp_addr, prod_acc_addr, 4, 4, 1, 4);
    compute_preloaded(prod_negate_sp_addr, di_negate_sp_addr, 1, 4, 1, 4);

    // Move out the result to u_i
    mvout(u_i, prod_acc_addr, 1, 4);

    // Ensure all operations are completed before proceeding
    fence();
}
