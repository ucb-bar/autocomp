#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/raddstoreexpminusmax.h"
#include "raddstoreexpminusmax-microkernel-tester.h"

#define BATCH_SIZE 256

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

int main() {
    volatile double dummy = 1.0;
    dummy = dummy / 1.0000001;

    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        xnn_f32_raddstoreexpminusmax_ukernel_fn kernel_fn =
            (xnn_f32_raddstoreexpminusmax_ukernel_fn)test_fn;

        RAddStoreExpMinusMaxMicrokernelTester tester;
        for (size_t elems = BATCH_SIZE - 8; elems <= BATCH_SIZE + 8; elems++) {
            tester.elements(elems)
                  .iterations(1)
                  .Test(kernel_fn, nullptr);
            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (elems=%zu)\n", _cand_id, elems);
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
