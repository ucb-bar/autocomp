#include <metal_stdlib>
using namespace metal;

struct block_q8_0 {
    half    d;
    int8_t  qs[32];
};

struct ggml_metal_kargs_mul_mm {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

// =============================================================================
// kernel_mul_mm_q8_0_f32  (matrix-matrix, tiled, Q8_0 weights x F32 activations)
//
// Targets: M2 (8-core GPU, no tensor API, has simdgroup_matrix)
// =============================================================================
//
// Computes: dst[M, N] = src0[M, K] @ src1[K, N]
//   where src0 is Q8_0 quantized, src1 is F32
//
// Each threadgroup computes a 64x32 output tile using 4 simdgroups.
// The K dimension is walked in blocks of NK=32. Each iteration:
//   1. 128 threads cooperatively load + dequantize a 64x32 tile of src0
//      and a 32x32 tile of src1 into threadgroup shared memory.
//   2. 4 simdgroups perform 8x8 matmuls using simdgroup_multiply_accumulate.
//
// Hardcoded for Q8_0 + F32 on M2 (no GGML_METAL_HAS_TENSOR):
//   S0=half, S1=float, block_q=block_q8_0, nl=2
// ---------------------------------------------------------------------------

// Dequantize Q8_0: reads 16 values from block, writes 4x4 half matrix.
// il selects which half of the 32-element block (0 or 1).
inline void dequantize_q8_0(device const block_q8_0 * xb, short il, thread half4x4 & reg) {
    device const int8_t * qs = xb->qs;
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = qs[i + 16*il] * d;
    }
    reg = (half4x4)reg_f;
}

[[host_name("kernel_mul_mm_q8_0_f32")]]
kernel void kernel_mul_mm_q8_0_f32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0       [[buffer(1)]],
        device const char * src1       [[buffer(2)]],
        device       char * dst        [[buffer(3)]],
        threadgroup  char * shmem      [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // -- tile dimensions --
    constexpr int NR0 = 64;   // output rows per threadgroup tile (M direction)
    constexpr int NR1 = 32;   // output cols per threadgroup tile (N direction)
    constexpr int NK  = 32;   // reduction block size (K direction)

    // loading geometry
    constexpr int NL0 = NK / 16;  // = 2: src0 loads per thread in K direction
    constexpr int NL1 = NK / 8;   // = 4: src1 loads per thread in K direction

    // Q8_0: each block is 32 values, nl=2 means dequantize reads 16 at a time
    constexpr short nl = 2;

    // -- shared memory: two tiles side by side --
    // sa: dequantized src0 halfs in 8x8-blocked layout
    // sb: src1 floats in 8x8-blocked layout
    threadgroup half  * sa = (threadgroup half  *)shmem;
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    // -- output tile position --
    const int im = tgpig.z;             // batch
    const int r0 = tgpig.y * NR0;      // starting output row (M)
    const int r1 = tgpig.x * NR1;      // starting output col (N)

    // boundary: actual tile size (may be smaller at edges)
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // clamp thread load indices to avoid OOB
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;

    const short il0 = tiitg % NL0;  // 0 or 1

    // -- batch offset for src0 --
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;
    const uint64_t offset0 = (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;

    // il tracks which 16-value sub-block of Q8_0 we're reading
    short il = il0;
    const short offset1 = il0 / nl;  // 0 for Q8_0 when il0 < 2

    // -- src0 pointer: points to one Q8_0 block in row (r0 + lr0) --
    device const block_q8_0 * x = (device const block_q8_0 *)(
        src0 + args.nb01 * (r0 + lr0) + offset0) + offset1;

    // -- src1 pointer: points to row (r1 + lr1), offset by iy columns --
    const short iy = 8 * (tiitg % NL1);
    device const float * y = (device const float *)(
        src1
        + args.nb13 * i13
        + args.nb12 * i12
        + args.nb11 * (r1 + lr1)
        + args.nb10 * iy);

    // -- 8 accumulator matrices: 4 simdgroups tile the 64x32 output --
    // Each simdgroup owns a 32x16 sub-tile:
    //   sgitg & 1 selects left/right 32 rows
    //   sgitg >> 1 selects top/bottom 16 cols
    // 8 matrices of 8x8 = covers 32 rows x 16 cols
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // -- main loop: walk K in steps of NK=32 --
    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {

        // === LOAD src0 tile: dequantize Q8_0 into shared memory ===
        //
        // dequantize_q8_0 reads 16 int8 values from one block and writes
        // a half4x4 (4x4 = 16 values). We then scatter those 16 values
        // into sa in an 8x8-blocked layout ready for simdgroup_load.
        half4x4 temp_a;
        dequantize_q8_0(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scatter 16 dequantized values into sa in the 8x8 blocked layout
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;

            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;

            const short ib = 8 * sx + sy;

            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i/4][i%4];
        }

        // === LOAD src1 tile into shared memory ===
        //
        // Each thread loads 8 f32 values from src1 into sb.
        // Layout: 8x8-blocked for simdgroup_load compatibility.
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib_s = 4 * sx + sy;

            // load 8 consecutive floats as a float2x4 (= 8 values)
            *(threadgroup float2x4 *)(sb + 64 * ib_s + 8 * ly) =
                *((device float2x4 *)y);
        }

        // advance src0 pointer through Q8_0 blocks
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1) / nl : x;

        // advance src1 pointer by NK columns
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === COMPUTE: 8x8 simdgroup matrix multiply-accumulate ===
        //
        // lsma -> this simdgroup's 32 rows (4 x 8x8 blocks) in sa
        // lsmb -> this simdgroup's 16 cols (2 x 8x8 blocks) in sb
        threadgroup const half  * lsma = sa + 4 * 64 * (sgitg & 1);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg >> 1);

        simdgroup_half8x8  ma[4];
        simdgroup_float8x8 mb[2];

        _Pragma("clang loop unroll(full)") for (short ik = 0; ik < NK / 8; ik++) {    // 4 iterations
            simdgroup_barrier(mem_flags::mem_none);

            // load 4 sub-tiles of A (32 rows x 8 cols)
            _Pragma("clang loop unroll(full)") for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // load 2 sub-tiles of B (8 rows x 16 cols)
            _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 8 multiply-accumulate ops cover the 32x16 sub-tile
            _Pragma("clang loop unroll(full)") for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8 * 64;  // advance by one 8-row slice
            lsmb += 4 * 64;
        }
    }

    // === WRITE OUTPUT ===
    //
    // If the full 64x32 tile fits within bounds, write directly to device memory.
    // Otherwise, stage through shared memory to avoid OOB writes.

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        // fast path: full tile, no bounds checking
        device float * C = (device float *)dst
            + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0
            + im * args.ne1 * args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        // slow path: partial tile at matrix edge, stage via shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // first simdgroup copies valid elements from shmem to device memory
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *)dst + r0 + (r1 + j) * args.ne0 + im * args.ne1 * args.ne0;
                device float4 * D4 = (device float4 *)D;

                threadgroup float  * C  = temp_str + j * NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}
