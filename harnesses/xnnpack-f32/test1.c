#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "transposec-microkernel-tester.h"

#define BLOCK_SIZE 256

typedef void (*candidate_fn_t)(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

int main() {
    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        TransposecMicrokernelTester tester;
        tester.iterations(1);

        // Square cases near 256 (5 tests).
        for (size_t size = BLOCK_SIZE - 2; size <= BLOCK_SIZE + 2; size++) {
            tester.block_height(size)
                  .block_width(size)
                  .input_stride(size)
                  .output_stride(size)
                  .Test(test_fn);

            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (size=%zu)\n", _cand_id, size);
                if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);
                goto next_candidate;
            }
        }

        // Non-square: vary height, fix width (5 tests).
        for (size_t h = BLOCK_SIZE - 2; h <= BLOCK_SIZE + 2; h++) {
            tester.block_height(h)
                  .block_width(BLOCK_SIZE)
                  .input_stride(BLOCK_SIZE)
                  .output_stride(h)
                  .Test(test_fn);

            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (h=%zu w=%d)\n", _cand_id, h, BLOCK_SIZE);
                if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);
                goto next_candidate;
            }
        }

        // Non-square: fix height, vary width (5 tests).
        for (size_t w = BLOCK_SIZE - 2; w <= BLOCK_SIZE + 2; w++) {
            tester.block_height(BLOCK_SIZE)
                  .block_width(w)
                  .input_stride(w)
                  .output_stride(BLOCK_SIZE)
                  .Test(test_fn);

            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (h=%d w=%zu)\n", _cand_id, BLOCK_SIZE, w);
                if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);
                goto next_candidate;
            }
        }

        printf("Correct result\n");
        printf("ID %d latency: %lu cycles\n", _cand_id, tester.kernel_cycles());
        if (NUM_CANDIDATES == 1) {
            printf("Generated implementation latency: %lu cycles\n", tester.kernel_cycles());
        }
        next_candidate:;
    }

    sys_reboot(SYS_REBOOT_COLD);
}
