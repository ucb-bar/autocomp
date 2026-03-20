#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microfnptr.h"
#include "dwconv2d-microkernel-tester.h"

static inline unsigned long read_cycles() {
    unsigned long cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

static inline void fence() {
    __asm__ volatile("fence" ::: "memory");
    __asm__ volatile("fence.i");
}

// Candidate kernel — body injected by autocomp
static void candidate_kernel(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* params)
{
// SUBSTITUTE HERE
// SUBSTITUTE END
}

int main() {
    // Sizes from MobileNet stride-2 depthwise conv layers
    static const size_t test_sizes[][2] = {
        {7,7},{12, 8}, {14, 14}, {28, 28}, {256, 256}
    };

    // 3x3s2p1: 3x3 kernel, stride 2, padding 1 on all sides
    DWConv2DMicrokernelTester tester;
    tester.kernel_height(3)
          .kernel_width(3)
          .subsampling(1)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .iterations(1);

    for (size_t i = 0; i < sizeof(test_sizes) / sizeof(test_sizes[0]); i++) {
        size_t h = test_sizes[i][0];
        size_t w = test_sizes[i][1];

        tester.input_height(h)
              .input_width(w)
              .Test(candidate_kernel, xnn_init_f32_minmax_scalar_params);

        if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
            printf("INCORRECT: assertion failed (h=%zu w=%zu)\n", h, w);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }

    printf("Correct result\n");
    printf("Generated implementation latency: %lu cycles\n", tester.kernel_cycles());

    sys_reboot(SYS_REBOOT_COLD);
}
