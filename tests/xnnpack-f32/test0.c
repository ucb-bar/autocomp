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
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params)
{
// SUBSTITUTE HERE
// SUBSTITUTE END
}

int main() {

    volatile double dummy = 1.0;
    dummy = dummy / 1.0000001;

    RAddStoreExpMinusMaxMicrokernelTester tester;
    for (size_t elems = BATCH_SIZE - 8; elems <= BATCH_SIZE + 8; elems++) {
        tester.elements(elems)
              .iterations(1)
              .Test(candidate_kernel, nullptr);
        if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
            printf("INCORRECT: assertion failed\n");
            sys_reboot(SYS_REBOOT_COLD);
        }
    }

    printf("Correct result\n");
    printf("Generated implementation latency: %lu cycles\n", tester.kernel_cycles());

    sys_reboot(SYS_REBOOT_COLD);
}
