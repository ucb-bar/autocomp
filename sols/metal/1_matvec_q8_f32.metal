#include <metal_stdlib>
using namespace metal;

struct block_q8_0 {
    half    d;
    int8_t  qs[32];
};

struct ggml_metal_kargs_mul_mv {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
};

// =============================================================================
// kernel_mul_mv_q8_0_f32  (matrix-vector, Q8_0 weights x F32 activation)
//
// Targets: M2 (Apple8 family)
// =============================================================================
//
// Computes: dst[M] = src0[M, K] @ src1[K]   (one column of output per src1 row)
//   where src0 is Q8_0 quantized, src1 is F32.
//
// Each threadgroup handles NR0=2 rows of src0 and 1 row of src1.
// NSG=4 simdgroups of 32 threads cooperate on the K reduction.
// Each thread processes NQ=8 quants at a time within a Q8_0 block.
//
// The reduction is: for each row, dot(dequant(src0_row), src1_row).
// After the K-loop, partial sums are reduced across simdgroups via shared memory.
//
// Hardcoded constants (matching ggml dispatch for Q8_0):
//   NR0 = 2  (N_R0_Q8_0)
//   NSG = 4  (N_SG_Q8_0)
//   QK  = 32 (quants per Q8_0 block)
//   NQ  = 8  (quants per thread per iteration)
// ---------------------------------------------------------------------------

// Reduce partial sums across simdgroups and write result.
template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = 32;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0       [[buffer(1)]],
        device const char * src1       [[buffer(2)]],
        device       char * dst        [[buffer(3)]],
        threadgroup  char * shmem      [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 2;    // DO NOT CHANGE — rows of src0 per threadgroup
    constexpr short NSG = 4;    // DO NOT CHANGE — simdgroups per threadgroup
    constexpr short NW  = 32;   // DO NOT CHANGE — threads per simdgroup
    constexpr short NQ  = 8;    // quants processed per thread per iteration
    constexpr short QK  = 32;   // DO NOT CHANGE — quants per Q8_0 block

    const int nb = args.ne00 / QK;  // number of Q8_0 blocks per row

    const int r0 = tgpig.x * NR0;  // starting src0 row for this threadgroup
    const int r1 = tgpig.y;        // src1 row index
    const int im = tgpig.z;        // batch index

    // -- batch offsets --
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;

    device const float * y = (device const float *)(src1 + offset1);

    // -- pointers to src0 rows --
    device const block_q8_0 * ax[NR0];
    _Pragma("clang loop unroll(full)")
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q8_0 *)((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    // -- thread position within simdgroup --
    // Each thread handles NQ=8 quants. With NW=32 threads and NQ=8,
    // NW/NQ = 4 threads share a block, each reading 8 consecutive quants.
    const short ix = tiisg / (NW / NQ);   // which block within the simdgroup's set
    const short il = tiisg % (NW / NQ);   // which 8-quant chunk within the block

    const int ib0 = sgitg * NQ + ix;      // starting block index for this thread

    float yl[NQ];

    device const float * yb = y + ib0 * QK + il * NQ;

    // -- main K-loop: each iteration covers NSG*NQ blocks --
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        // load 8 activation values
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // dot product against each of the NR0 weight rows
        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.f;
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK;
    }

    // -- reduce and write --
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}
