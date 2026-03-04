// Test harness for f32-raddstoreexpminusmax (float32 reduce-add-store-exp-minus-max)
// RVV implementation test

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#define BATCH_SIZE 256
#define OUTPUT_MATRIX_NAME output_arr

static float input_arr[BATCH_SIZE];
static float output_arr[BATCH_SIZE];
static double gold[BATCH_SIZE];  
static float max_val;
static float sum_val;
static double gold_sum;

#define REPEAT_TEST_ITERS 1

void reference_f32_raddstoreexpminusmax(
    size_t batch,
    const float* input,
    const float* max,
    double* output,
    double* sum,
    const void* params)
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(max != NULL);
    assert(output != NULL);
    assert(sum != NULL);

    const double x_max = (double)*max;
    double sum_acc = 0.0;

    for (; batch >= sizeof(float); batch -= sizeof(float)) {
        const double y_ref_value = exp((double)*input++ - x_max);
        *output++ = y_ref_value;
        sum_acc += y_ref_value;
    }

    *sum = sum_acc;
}


int full_is_equal(float* x, double* y) {
    size_t n = BATCH_SIZE;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)x[i] - y[i]);
        if (diff > fabs(y[i]) * 1.0e-6) {
            printf("Mismatch at element %zu / %d: got %f, expected %f, x_max %f\n",
                   i, BATCH_SIZE, (double)x[i], y[i], max_val);
            return 0;
        }
    }
    return 1;
}

// Saturn-specific cycle counter
unsigned long read_cycles() {
    unsigned long cc;
    __asm__ volatile("rdcycle %0" : "=r"(cc));
    return cc;
}

// Compatibility macros for code injected by get_test_code()
#define gemmini_flush(x) do {} while(0)  // no-op for Saturn

// Reset state macro for FireSim batched runs
#define RESET_STATE() do { \
    batch = BATCH_SIZE * sizeof(float); \
    input = input_arr; \
    output = output_arr; \
    sum_val = 0.0f; \
    sum = &sum_val; \
} while(0)

static inline void fence(void) {
    __asm__ volatile("fence" ::: "memory");
}


static float rand_float() {
    return 90.0f + ((float)rand() / (float)RAND_MAX) * 10.0f;
}

int main() {
    for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
        // Initialize input data
        max_val = -INFINITY;
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            input_arr[i] = rand_float();
            if (input_arr[i] > max_val) {
                max_val = input_arr[i];
            }
        }

        // Clear output
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            output_arr[i] = 0.0f;
        }

        // Compute gold reference
        gold_sum = 0.0;
        reference_f32_raddstoreexpminusmax(BATCH_SIZE * sizeof(float), input_arr, &max_val, gold, &gold_sum, NULL);

        // Clear output again for candidate
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            output_arr[i] = 0.0f;
        }

        // Set up variables for injected code
        size_t batch = BATCH_SIZE * sizeof(float);
        const float* input = input_arr;
        const float* max = &max_val;
        float* output = output_arr;
        sum_val = 0.0f;
        float* sum = &sum_val;
        const void* params = NULL;

        // SUBSTITUTE HERE
        // SUBSTITUTE END

        // Verify results
        if (!full_is_equal(output_arr, gold)) {
            printf("FAIL: Output mismatch\n");
            sys_reboot(SYS_REBOOT_COLD);
        }

        // Verify sum (|sum_ref - sum| < |sum_ref| * 1e-6)
        double sdiff = fabs(gold_sum - (double)sum_val);
        if (sdiff > fabs(gold_sum) * 1.0e-6) {
            printf("FAIL: Sum mismatch: got %f, expected %f\n", (double)sum_val, gold_sum);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }
    printf("Correct result\n");
    sys_reboot(SYS_REBOOT_COLD);
}
