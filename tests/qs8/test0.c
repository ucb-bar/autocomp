// Test harness for qs8-vaddc (quantized int8 vector add with constant)
// RVV implementation test

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

struct xnn_qs8_add_minmax_params {
    struct {
        int32_t bias;
        int32_t a_multiplier;
        int32_t b_multiplier;
        uint32_t shift;
        int32_t output_min;
        int32_t output_max;
        int32_t output_zero_point;
    } scalar;
};

#define BATCH_SIZE 64
#define OUTPUT_MATRIX_NAME output_arr

static int8_t input_a_arr[BATCH_SIZE];
static int8_t input_b_val;
static int8_t output_arr[BATCH_SIZE];
static int8_t gold[BATCH_SIZE];

#define REPEAT_TEST_ITERS 1

// Scalar reference implementation
void reference_qs8_vaddc(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const struct xnn_qs8_add_minmax_params* params)
{
    const int32_t bias = params->scalar.bias + (int32_t)(*input_b) * params->scalar.b_multiplier;
    const int32_t a_multiplier = params->scalar.a_multiplier;
    const uint32_t shift = params->scalar.shift;
    const int32_t output_min = params->scalar.output_min;
    const int32_t output_max = params->scalar.output_max;
    const int32_t output_zero_point = params->scalar.output_zero_point;

    for (size_t i = 0; i < batch; i++) {
        int32_t a_i32 = (int32_t)input_a[i];
        int32_t result = a_i32 * a_multiplier;
        result = result + bias;
        result = result >> shift;
        result = result + output_zero_point;

        if (result < output_min) result = output_min;
        if (result > output_max) result = output_max;

        output[i] = (int8_t)result;
    }
}

int full_is_equal(int8_t* x, int8_t* y) {
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        if (x[i] != y[i]) {
            printf("Mismatch at index %zu: got %d, expected %d\n", i, x[i], y[i]);
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
// Resets variables that get modified during candidate execution
#define RESET_STATE() do { \
    batch = BATCH_SIZE; \
    input_a = input_a_arr; \
    input_b = &input_b_val; \
    output = output_arr; \
} while(0)

static inline void fence(void) {
    __asm__ volatile("fence" ::: "memory");
}

int main() {
    static struct xnn_qs8_add_minmax_params params_storage;

    // Initialize parameters
    params_storage.scalar.bias = 0;
    params_storage.scalar.a_multiplier = 1;
    params_storage.scalar.b_multiplier = 1;
    params_storage.scalar.shift = 0;
    params_storage.scalar.output_min = -128;
    params_storage.scalar.output_max = 127;
    params_storage.scalar.output_zero_point = 0;

    for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
        // Initialize input data
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            input_a_arr[i] = (int8_t)(rand() % 256 - 128);
        }
        input_b_val = (int8_t)(rand() % 256 - 128);

        // Clear output
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            output_arr[i] = 0;
        }

        // Compute gold reference
        reference_qs8_vaddc(BATCH_SIZE, input_a_arr, &input_b_val, gold, &params_storage);

        // Set up variables for injected code
        size_t batch = BATCH_SIZE;
        const int8_t* input_a = input_a_arr;
        const int8_t* input_b = &input_b_val;
        int8_t* output = output_arr;
        const struct xnn_qs8_add_minmax_params* params = &params_storage;

        // SUBSTITUTE HERE
        // SUBSTITUTE END
    }
    printf("Correct result\n");
    sys_reboot(SYS_REBOOT_COLD);
}
