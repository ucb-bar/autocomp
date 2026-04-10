#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/quantization.h"
#include "gemm-microkernel-tester.h"

typedef void (*candidate_fn_t)(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* params,
    const struct xnn_qd8_quantization_params* quantization_params);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

static size_t next_prime(size_t n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;
    for (;;) {
        bool is_prime = true;
        for (size_t i = 3; i * i <= n; i += 2) {
            if (n % i == 0) { is_prime = false; break; }
        }
        if (is_prime) return n;
        n += 2;
    }
}

// Macro: run one Test() call, accumulate cycles, bail on failure.
#define RUN_TEST(tester, label) do {                                          \
    tester.Test(gemm_fn, xnn_init_f32_minmax_scalar_params,                   \
                xnn_pack_qs8_gemm_goi_w);                                     \
    total_cycles += tester.kernel_cycles();                                    \
    if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {     \
        printf("INCORRECT: candidate %d failed at %s "                        \
               "(m=%zu n=%zu k=%zu)\n",                                       \
               _cand_id, label, tester.m(), tester.n(), tester.k());          \
        if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);                \
        goto next_candidate;                                                  \
    }                                                                         \
} while (0)

int main() {
    const size_t nr = __riscv_vsetvlmax_e32m4();
    const size_t mr = 1;
    const size_t kr = 1;
    const size_t sr = 1;
    const size_t k_block = 1;
    const size_t adj_k_block = 1;

    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm_fn =
            (xnn_qd8_f32_qc8w_gemm_ukernel_fn)test_fn;

        // Base tester configuration — matches CreateTests2
        GemmMicrokernelTester base;
        base.mr(mr).nr(nr).kr(kr).sr(sr);
        unsigned long total_cycles = 0;

        // ==========================================
        // Follows XNNPACK CreateTests2 exactly
        // for k_block=1, adj_k_block=1, is_igemm=false
        // ==========================================

        // ----- k_eq_1 -----
        {
            GemmMicrokernelTester t = base;
            t.m(mr).n(nr).k(k_block);
            RUN_TEST(t, "k_eq_1");
        }

        // ----- k_eq_1_strided_a -----
        {
            GemmMicrokernelTester t = base;
            t.m(mr).n(nr).k(k_block).a_stride(next_prime(k_block + 1));
            RUN_TEST(t, "k_eq_1_strided_a");
        }

        // ----- k_eq_1_subtile: loop_n(1, nr), loop_m(1, mr) -----
        for (size_t n = 1; n <= nr; n++) {
            for (size_t m = 1; m <= mr; m++) {
                GemmMicrokernelTester t = base;
                t.m(m).n(n).k(k_block);
                RUN_TEST(t, "k_eq_1_subtile");
            }
        }

        // ----- k_eq_1_subtile_m: loop_m(1, mr) -----
        for (size_t m = 1; m <= mr; m++) {
            GemmMicrokernelTester t = base;
            t.m(m).n(nr).k(k_block);
            RUN_TEST(t, "k_eq_1_subtile_m");
        }

        // ----- k_eq_1_subtile_n: loop_n(1, nr) -----
        for (size_t n = 1; n <= nr; n++) {
            GemmMicrokernelTester t = base;
            t.m(mr).n(n).k(k_block);
            RUN_TEST(t, "k_eq_1_subtile_n");
        }

        // ----- k_lt tests: SKIPPED (k_block == 1) -----

        // ----- k_gt_1: loop_k(adj_k_block+1, adj_k_block*2-1, k_block) -----
        // adj_k_block=1 → loop_k(2, 1, 1) → empty range, no iterations
        for (size_t k = adj_k_block + 1; k <= adj_k_block * 2 - 1; k += k_block) {
            GemmMicrokernelTester t = base;
            t.m(mr).n(nr).k(k);
            RUN_TEST(t, "k_gt_1");
        }

        // ----- k_gt_1_subtile -----
        for (size_t k = adj_k_block + 1; k <= adj_k_block * 2 - 1; k += k_block) {
            for (size_t n = 1; n <= nr; n++) {
                for (size_t m = 1; m <= mr; m++) {
                    GemmMicrokernelTester t = base;
                    t.m(m).n(n).k(k);
                    RUN_TEST(t, "k_gt_1_subtile");
                }
            }
        }

        // ----- k_div tests: SKIPPED (k_block == 1) -----

        // ----- n_gt_nr: loop_n(nr+1, nr*2-1, 4), loop_k(1, k_block*3, k_block+1) -----
        for (size_t n = nr + 1; n <= nr * 2 - 1; n += 4) {
            for (size_t k = 1; k <= k_block * 3; k += k_block + 1) {
                GemmMicrokernelTester t = base;
                t.m(mr).n(n).k(k);
                RUN_TEST(t, "n_gt_nr");
            }
        }

        // ----- n_gt_nr_strided_a: loop_n(nr+1, nr*2-1, 4), loop_k(1, k_block*3, k_block) -----
        for (size_t n = nr + 1; n <= nr * 2 - 1; n += 4) {
            for (size_t k = 1; k <= k_block * 3; k += k_block) {
                GemmMicrokernelTester t = base;
                t.m(mr).n(n).k(k).a_stride(next_prime(k_block * 3 + 1));
                RUN_TEST(t, "n_gt_nr_strided_a");
            }
        }

        // ----- n_gt_nr_subtile: loop_n(nr+1, nr*2-1, 4), loop_k(1, k_block*3, k_block+1), loop_m(1, mr) -----
        for (size_t n = nr + 1; n <= nr * 2 - 1; n += 4) {
            for (size_t k = 1; k <= k_block * 3; k += k_block + 1) {
                for (size_t m = 1; m <= mr; m++) {
                    GemmMicrokernelTester t = base;
                    t.m(m).n(n).k(k);
                    RUN_TEST(t, "n_gt_nr_subtile");
                }
            }
        }

        // ----- n_div_nr: loop_n(nr*2, nr*3, nr), loop_k(1, k_block*3, k_block+1) -----
        for (size_t n = nr * 2; n <= nr * 3; n += nr) {
            for (size_t k = 1; k <= k_block * 3; k += k_block + 1) {
                GemmMicrokernelTester t = base;
                t.m(mr).n(n).k(k);
                RUN_TEST(t, "n_div_nr");
            }
        }

        // ----- n_div_nr_strided_a: loop_n(nr*2, nr*3, nr), loop_k(1, k_block*3, k_block) -----
        for (size_t n = nr * 2; n <= nr * 3; n += nr) {
            for (size_t k = 1; k <= k_block * 3; k += k_block) {
                GemmMicrokernelTester t = base;
                t.m(mr).n(n).k(k).a_stride(next_prime(k_block * 3 + 1));
                RUN_TEST(t, "n_div_nr_strided_a");
            }
        }

        // ----- n_div_nr_subtile: loop_n(nr*2, nr*3, nr), loop_k(1, k_block*3, k_block+1), loop_m(1, mr) -----
        for (size_t n = nr * 2; n <= nr * 3; n += nr) {
            for (size_t k = 1; k <= k_block * 3; k += k_block + 1) {
                for (size_t m = 1; m <= mr; m++) {
                    GemmMicrokernelTester t = base;
                    t.m(m).n(n).k(k);
                    RUN_TEST(t, "n_div_nr_subtile");
                }
            }
        }

        // ----- igemm-only tests: SKIPPED (not igemm) -----

        // ----- strided_cm_subtile: cm_stride=NextPrime(nr+1), loop_k(1, k_block*3, k_block+1), loop_n(1, nr), loop_m(1, mr) -----
        for (size_t k = 1; k <= k_block * 3; k += k_block + 1) {
            for (size_t n = 1; n <= nr; n++) {
                for (size_t m = 1; m <= mr; m++) {
                    GemmMicrokernelTester t = base;
                    t.m(m).n(n).k(k).cm_stride(next_prime(nr + 1));
                    RUN_TEST(t, "strided_cm_subtile");
                }
            }
        }

        // ----- strided_cm -----
        {
            GemmMicrokernelTester t = base;
            t.m(mr).n(nr).k(k_block).cm_stride(next_prime(nr + 1));
            RUN_TEST(t, "strided_cm");
        }

        printf("Correct result\n");
        printf("ID %d latency: %lu cycles\n", _cand_id, total_cycles);
        if (NUM_CANDIDATES == 1) {
            printf("Generated implementation latency: %lu cycles\n", total_cycles);
        }
        next_candidate:;
    }

    sys_reboot(SYS_REBOOT_COLD);
}
