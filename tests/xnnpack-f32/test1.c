#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "transposec-microkernel-tester.h"

#define BLOCK_SIZE 256

// Candidate kernel — body injected by autocomp
static void candidate_kernel(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
// SUBSTITUTE HERE
// SUBSTITUTE END
}

int main() {
    TransposecMicrokernelTester tester;
    tester.iterations(1);

    // Square cases near 256 (5 tests).
    for (size_t size = BLOCK_SIZE - 2; size <= BLOCK_SIZE + 2; size++) {
        tester.block_height(size)
              .block_width(size)
              .input_stride(size)
              .output_stride(size)
              .Test(candidate_kernel);

        if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
            printf("INCORRECT: assertion failed (size=%zu)\n", size);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }

    // Non-square: vary height, fix width (5 tests).
    for (size_t h = BLOCK_SIZE - 2; h <= BLOCK_SIZE + 2; h++) {
        tester.block_height(h)
              .block_width(BLOCK_SIZE)
              .input_stride(BLOCK_SIZE)
              .output_stride(h)
              .Test(candidate_kernel);

        if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
            printf("INCORRECT: assertion failed (h=%zu w=%d)\n", h, BLOCK_SIZE);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }

    // Non-square: fix height, vary width (5 tests).
    for (size_t w = BLOCK_SIZE - 2; w <= BLOCK_SIZE + 2; w++) {
        tester.block_height(BLOCK_SIZE)
              .block_width(w)
              .input_stride(w)
              .output_stride(BLOCK_SIZE)
              .Test(candidate_kernel);

        if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
            printf("INCORRECT: assertion failed (h=%d w=%zu)\n", BLOCK_SIZE, w);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }

    printf("Correct result\n");
    printf("Generated implementation latency: %lu cycles\n", tester.kernel_cycles());

    sys_reboot(SYS_REBOOT_COLD);
}
