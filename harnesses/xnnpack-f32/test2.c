#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microfnptr.h"
#include "dwconv2d-microkernel-tester.h"

typedef void (*candidate_fn_t)(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* params);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

int main() {
    // Sizes from MobileNet stride-2 depthwise conv layers
    static const size_t test_sizes[][2] = {
        {7,7},{12, 8}, {14, 14}, {28, 28}, {256, 256}
    };

    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        xnn_dwconv2d_chw_ukernel_fn kernel_fn =
            (xnn_dwconv2d_chw_ukernel_fn)test_fn;

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
                  .Test(kernel_fn, xnn_init_f32_minmax_scalar_params);

            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (h=%zu w=%zu)\n", _cand_id, h, w);
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
