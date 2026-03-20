// Test harness for f32-dwconv2d-chw-3x3p1-minmax (float32 depthwise conv 3x3, padding 1, minmax)
// Structured after XNNPACK dwconv2d-microkernel-tester.h

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#define INPUT_HEIGHT 12
#define INPUT_WIDTH 16
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define KERNEL_SIZE (KERNEL_HEIGHT * KERNEL_WIDTH)
#define PADDING_TOP 1
#define PADDING_LEFT 1
#define PADDING_BOTTOM 1
#define PADDING_RIGHT 1
#define SUBSAMPLING 1
#define BATCH_SIZE (INPUT_HEIGHT * INPUT_WIDTH)
#define OUTPUT_HEIGHT INPUT_HEIGHT
#define OUTPUT_WIDTH INPUT_WIDTH

#define REPEAT_TEST_ITERS 1

static float input_arr[BATCH_SIZE + 2 * 16 /* extra bytes */];
static float output_arr[OUTPUT_HEIGHT * OUTPUT_WIDTH];
static float output_ref[OUTPUT_HEIGHT * OUTPUT_WIDTH];
static float packed_weights[KERNEL_SIZE + 1]; // bias + 3x3 kernel
static float zero_arr[INPUT_WIDTH + 2 * 16];

static float rand_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

unsigned long read_cycles() {
    unsigned long cc;
    __asm__ volatile("rdcycle %0" : "=r"(cc));
    return cc;
}

static inline void fence(void) {
    __asm__ volatile("fence" ::: "memory");
}

// SUBSTITUTE HERE
void test(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  size_t vlmax = __riscv_vsetvlmax_e32m1();

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk10 = weights[4];
  const float vk11 = weights[5];
  const float vk12 = weights[6];
  const float vk20 = weights[7];
  const float vk21 = weights[8];
  const float vk22 = weights[9];

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);
  float* o4 = (float*) ((uintptr_t) o3 + input_width);
  float* o5 = (float*) ((uintptr_t) o4 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i4 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i5 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i6 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(output_height < 7) {
      i7 = zero;
    }

    size_t w = input_width >> XNN_LOG2_SIZEOF_FLOAT;
    size_t vl =  __riscv_vsetvl_e32m1(w);
    vfloat32m1_t vi0x1 =  __riscv_vle32_v_f32m1(i0, vl);
    vfloat32m1_t vi0x0 =  __riscv_vfslide1up_vf_f32m1(vi0x1, 0.0f, vl);
    i0 += vl;
    vfloat32m1_t vi1x1 =  __riscv_vle32_v_f32m1(i1, vl);
    vfloat32m1_t vi1x0 =  __riscv_vfslide1up_vf_f32m1(vi1x1, 0.0f, vl);
    i1 += vl;
    vfloat32m1_t vi2x1 =  __riscv_vle32_v_f32m1(i2, vl);
    vfloat32m1_t vi2x0 =  __riscv_vfslide1up_vf_f32m1(vi2x1, 0.0f, vl);
    i2 += vl;
    vfloat32m1_t vi3x1 =  __riscv_vle32_v_f32m1(i3, vl);
    vfloat32m1_t vi3x0 =  __riscv_vfslide1up_vf_f32m1(vi3x1, 0.0f, vl);
    i3 += vl;
    vfloat32m1_t vi4x1 =  __riscv_vle32_v_f32m1(i4, vl);
    vfloat32m1_t vi4x0 =  __riscv_vfslide1up_vf_f32m1(vi4x1, 0.0f, vl);
    i4 += vl;
    vfloat32m1_t vi5x1 =  __riscv_vle32_v_f32m1(i5, vl);
    vfloat32m1_t vi5x0 =  __riscv_vfslide1up_vf_f32m1(vi5x1, 0.0f, vl);
    i5 += vl;
    vfloat32m1_t vi6x1 =  __riscv_vle32_v_f32m1(i6, vl);
    vfloat32m1_t vi6x0 =  __riscv_vfslide1up_vf_f32m1(vi6x1, 0.0f, vl);
    i6 += vl;
    vfloat32m1_t vi7x1 =  __riscv_vle32_v_f32m1(i7, vl);
    vfloat32m1_t vi7x0 =  __riscv_vfslide1up_vf_f32m1(vi7x1, 0.0f, vl);
    i7 += vl;

    while (w > vlmax) {
      vfloat32m1_t vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x1, *i0, vl);
      vfloat32m1_t vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x1, *i1, vl);
      vfloat32m1_t vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x1, *i2, vl);
      vfloat32m1_t vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x1, *i3, vl);
      vfloat32m1_t vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x1, *i4, vl);
      vfloat32m1_t vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x1, *i5, vl);
      vfloat32m1_t vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x1, *i6, vl);
      vfloat32m1_t vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x1, *i7, vl);

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi3x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi4x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi5x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi4x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi5x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi6x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi5x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi6x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi7x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi3x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi4x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi5x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi4x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi5x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi6x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi5x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi6x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi7x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi3x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi4x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi5x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi4x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi5x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi6x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi5x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi6x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi7x2, vl);

      vo0p0 = __riscv_vfmax_vf_f32m1(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m1(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m1(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m1(vo3p0, vmin, vl);
      vo4p0 = __riscv_vfmax_vf_f32m1(vo4p0, vmin, vl);
      vo5p0 = __riscv_vfmax_vf_f32m1(vo5p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m1(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m1(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m1(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m1(vo3p0, vmax, vl);
      vo4p0 = __riscv_vfmin_vf_f32m1(vo4p0, vmax, vl);
      vo5p0 = __riscv_vfmin_vf_f32m1(vo5p0, vmax, vl);

      __riscv_vse32_v_f32m1(o5, vo5p0, vl);
      o5 += vl;
      __riscv_vse32_v_f32m1(o4, vo4p0, vl);
      o4 += vl;
      __riscv_vse32_v_f32m1(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m1(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0p0, vl);
      o0 += vl;

      w -= vl;
      vl = __riscv_vsetvl_e32m1(w);
      vi0x1 =  __riscv_vle32_v_f32m1(i0, vl);
      vi0x0 =  __riscv_vfslide1up_vf_f32m1(vi0x1, *(i0-1), vl);
      i0 += vl;
      vi1x1 =  __riscv_vle32_v_f32m1(i1, vl);
      vi1x0 =  __riscv_vfslide1up_vf_f32m1(vi1x1, *(i1-1), vl);
      i1 += vl;
      vi2x1 =  __riscv_vle32_v_f32m1(i2, vl);
      vi2x0 =  __riscv_vfslide1up_vf_f32m1(vi2x1, *(i2-1), vl);
      i2 += vl;
      vi3x1 =  __riscv_vle32_v_f32m1(i3, vl);
      vi3x0 =  __riscv_vfslide1up_vf_f32m1(vi3x1, *(i3-1), vl);
      i3 += vl;
      vi4x1 =  __riscv_vle32_v_f32m1(i4, vl);
      vi4x0 =  __riscv_vfslide1up_vf_f32m1(vi4x1, *(i4-1), vl);
      i4 += vl;
      vi5x1 =  __riscv_vle32_v_f32m1(i5, vl);
      vi5x0 =  __riscv_vfslide1up_vf_f32m1(vi5x1, *(i5-1), vl);
      i5 += vl;
      vi6x1 =  __riscv_vle32_v_f32m1(i6, vl);
      vi6x0 =  __riscv_vfslide1up_vf_f32m1(vi6x1, *(i6-1), vl);
      i6 += vl;
      vi7x1 =  __riscv_vle32_v_f32m1(i7, vl);
      vi7x0 =  __riscv_vfslide1up_vf_f32m1(vi7x1, *(i7-1), vl);
      i7 += vl;
    }
    // Always process the last tile separately to account for right edge.
    assert(w <= vlmax);
    {
      vfloat32m1_t vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x1, 0.0f, vl);
      vfloat32m1_t vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x1, 0.0f, vl);
      vfloat32m1_t vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x1, 0.0f, vl);
      vfloat32m1_t vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x1, 0.0f, vl);
      vfloat32m1_t vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x1, 0.0f, vl);
      vfloat32m1_t vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x1, 0.0f, vl);
      vfloat32m1_t vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x1, 0.0f, vl);
      vfloat32m1_t vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x1, 0.0f, vl);

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi3x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi4x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi5x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi4x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi5x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi6x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi5x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi6x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi7x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi3x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi4x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi5x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi4x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi5x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi6x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi5x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi6x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi7x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi3x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi4x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi5x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi4x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi5x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi6x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi5x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi6x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi7x2, vl);

      vo0p0 = __riscv_vfmax_vf_f32m1(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m1(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m1(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m1(vo3p0, vmin, vl);
      vo4p0 = __riscv_vfmax_vf_f32m1(vo4p0, vmin, vl);
      vo5p0 = __riscv_vfmax_vf_f32m1(vo5p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m1(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m1(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m1(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m1(vo3p0, vmax, vl);
      vo4p0 = __riscv_vfmin_vf_f32m1(vo4p0, vmax, vl);
      vo5p0 = __riscv_vfmin_vf_f32m1(vo5p0, vmax, vl);

      __riscv_vse32_v_f32m1(o5, vo5p0, vl);
      o5 += vl;
      __riscv_vse32_v_f32m1(o4, vo4p0, vl);
      o4 += vl;
      __riscv_vse32_v_f32m1(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m1(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0p0, vl);
      o0 += vl;
    }

    i0 = (const float*) ((uintptr_t) i6 - input_width);
    i1 = (const float*) ((uintptr_t) i7 - input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o5;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);
    o4 = (float*) ((uintptr_t) o3 + input_width);
    o5 = (float*) ((uintptr_t) o4 + input_width);

    output_height = doz(output_height, 6);
  } while (output_height != 0);
}
// SUBSTITUTE END

int main() {
    unsigned long total_kernel_cycles = 0;

    for (int iteration = 0; iteration < REPEAT_TEST_ITERS; iteration++) {
        // Generate random input
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            input_arr[i] = rand_float();
        }

        // Generate random packed weights (bias + kernel)
        for (size_t i = 0; i < KERNEL_SIZE + 1; i++) {
            packed_weights[i] = rand_float();
        }

        // Initialize zero buffer
        for (size_t i = 0; i < INPUT_WIDTH; i++) {
            zero_arr[i] = 0.0f;
        }

        // Compute reference output using simple nested loops (matches tester.h pattern)
        for (size_t oy = 0; oy < OUTPUT_HEIGHT; oy++) {
            for (size_t ox = 0; ox < OUTPUT_WIDTH; ox++) {
                float acc = packed_weights[0]; // bias
                for (size_t ky = 0; ky < KERNEL_HEIGHT; ky++) {
                    const size_t iy = oy + ky - PADDING_TOP;
                    for (size_t kx = 0; kx < KERNEL_WIDTH; kx++) {
                        const size_t ix = ox + kx - PADDING_LEFT;
                        if (ix < INPUT_WIDTH && iy < INPUT_HEIGHT) {
                            const float input_val = input_arr[iy * INPUT_WIDTH + ix];
                            const float kernel_val = packed_weights[1 + ky * KERNEL_WIDTH + kx];
                            acc += input_val * kernel_val;
                        }
                    }
                }
                output_ref[oy * OUTPUT_WIDTH + ox] = acc;
            }
        }

        // Compute clamping parameters from reference output
        float accumulated_min = output_ref[0];
        float accumulated_max = output_ref[0];
        for (size_t i = 1; i < OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) {
            if (output_ref[i] < accumulated_min) accumulated_min = output_ref[i];
            if (output_ref[i] > accumulated_max) accumulated_max = output_ref[i];
        }
        const float accumulated_range = accumulated_max - accumulated_min;
        const float output_min = accumulated_min + accumulated_range / 255.0f * 0.0f;   // qmin=0
        const float output_max = accumulated_max - accumulated_range / 255.0f * 0.0f;   // qmax=255

        // Prepare params
        struct xnn_f32_minmax_params params;
        params.scalar.min = output_min;
        params.scalar.max = output_max;

        // Clamp reference results
        for (size_t i = 0; i < OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) {
            if (output_ref[i] < output_min) output_ref[i] = output_min;
            if (output_ref[i] > output_max) output_ref[i] = output_max;
        }

        // Clear output for candidate
        for (size_t i = 0; i < OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) {
            output_arr[i] = 0.0f;
        }

        // Call optimized micro-kernel with cycle counting
        unsigned long t0, t1;
        __asm__ volatile("rdcycle %0" : "=r"(t0));
        __asm__ volatile("fence" ::: "memory");
        __asm__ volatile("fence.i");
        test(INPUT_HEIGHT, INPUT_WIDTH * sizeof(float), input_arr,
             packed_weights, zero_arr, output_arr, PADDING_TOP, &params);
        __asm__ volatile("fence.i");
        __asm__ volatile("fence" ::: "memory");
        __asm__ volatile("rdcycle %0" : "=r"(t1));
        total_kernel_cycles += (t1 - t0);

        // Verify results
        for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
            for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
                float ref_val = output_ref[y * OUTPUT_WIDTH + x];
                float out_val = output_arr[y * OUTPUT_WIDTH + x];
                float diff = fabsf(ref_val - out_val);
                if (diff > fabsf(ref_val) * 1.0e-5f) {
                    printf("FAIL at x=%zu, y=%zu: got %f, expected %f\n",
                           x, y, out_val, ref_val);
                    sys_reboot(SYS_REBOOT_COLD);
                }
            }
        }
    }

    printf("Correct result (cycles: %lu)\n", total_kernel_cycles);
    sys_reboot(SYS_REBOOT_COLD);
}
