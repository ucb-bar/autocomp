#include "kernel-runner.hpp"
#include "harness-utils.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// ── Kernel args struct (matches Metal kernel's ggml_metal_kargs_mul_mm) ──
// NO pack(1) — must match Metal's natural alignment (uint64_t is 8-byte aligned)
struct Q8MulMmArgs {
    int32_t  ne00;   // K
    int32_t  ne02;   // src0 batch dim (1)
    uint64_t nb01;   // src0 bytes per row = (K/32)*34
    uint64_t nb02;   // src0 batch stride
    uint64_t nb03;   // src0 batch stride
    int32_t  ne12;   // src1 batch dim (1)
    uint64_t nb10;   // src1 element stride = 4
    uint64_t nb11;   // src1 row stride = K*4
    uint64_t nb12;   // src1 batch stride
    uint64_t nb13;   // src1 batch stride
    int32_t  ne0;    // output rows (M)
    int32_t  ne1;    // output cols (N)
    int16_t  r2;     // batch repeat (1)
    int16_t  r3;     // batch repeat (1)
};

// ── Random data generation (deterministic, seeded) ─────────────
static std::vector<uint8_t> randomQ8_0(int M, int K) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> qs_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    int blocks_per_row = K / 32;
    size_t bytes_per_row = (size_t)blocks_per_row * 34;
    std::vector<uint8_t> data((size_t)M * bytes_per_row);

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t* block = data.data() + row * bytes_per_row + (size_t)b * 34;
            float scale = scale_dist(rng);
            __fp16 d = (__fp16)scale;
            memcpy(block, &d, 2);
            for (int i = 0; i < 32; i++) {
                block[2 + i] = (int8_t)qs_dist(rng);
            }
        }
    }
    return data;
}

static std::vector<float> randomVec(size_t n) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ── main ───────────────────────────────────────────────────────
// Usage: <harness> <candidate_metallib> <reference_metallib> [M] [N] [K]
//
// Correctness: run both reference and candidate kernels on the same inputs,
// compare GPU outputs (GPU vs GPU — no CPU reference needed).
// Latency: benchmark the candidate kernel.

static int arg(int argc, char* argv[], int i, int def) {
    return (i < argc) ? std::atoi(argv[i]) : def;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <candidate_metallib> <reference_metallib> [M=12288] [N=128] [K=1536]\n";
        printResult(false, 0, 0, "Missing metallib path arguments");
        return 1;
    }

    std::string candidatePath = argv[1];
    std::string referencePath = argv[2];
    int M = arg(argc, argv, 3, 12288);
    int N = arg(argc, argv, 4, 128);
    int K = arg(argc, argv, 5, 1536);

    if (K % 32 != 0) {
        printResult(false, 0, 0, "K must be a multiple of 32");
        return 1;
    }

    size_t bytes_per_row_q8 = (size_t)(K / 32) * 34;

    // Generate test data (deterministic)
    auto src0 = randomQ8_0(M, K);
    auto src1 = randomVec(N * K);

    // Kernel args
    Q8MulMmArgs args = {};
    args.ne00 = K;
    args.ne02 = 1;
    args.nb01 = bytes_per_row_q8;
    args.nb02 = (uint64_t)M * bytes_per_row_q8;
    args.nb03 = (uint64_t)M * bytes_per_row_q8;
    args.ne12 = 1;
    args.nb10 = 4;
    args.nb11 = (uint64_t)K * 4;
    args.nb12 = (uint64_t)N * K * 4;
    args.nb13 = (uint64_t)N * K * 4;
    args.ne0  = M;
    args.ne1  = N;
    args.r2   = 1;
    args.r3   = 1;

    // Dispatch params (shared between ref and candidate)
    DispatchParams dp;
    dp.grid[0] = (N + 31) / 32;
    dp.grid[1] = (M + 63) / 64;
    dp.grid[2] = 1;
    dp.threadgroup[0] = 128;
    dp.threadgroup[1] = 1;
    dp.threadgroup[2] = 1;
    dp.dispatchByThreadgroups = true;
    dp.threadgroupMemBytes = 6144;

    const std::string kernelName = "kernel_mul_mm_q8_0_f32";
    size_t outBytes = size_t(M * N) * 4;

    // --- Run reference kernel (1 run, copy back output) ---
    std::vector<float> refOut(M * N, 0.0f);
    {
        DispatchParams refDp = dp;
        refDp.warmupRuns = 1;
        refDp.benchRuns  = 1;

        auto refResult = runKernel(referencePath, kernelName,
            {{0, sizeof(args),                 &args},
             {1, (size_t)M * bytes_per_row_q8, src0.data()},
             {2, size_t(N * K) * 4,            src1.data()},
             {3, outBytes,                      refOut.data()}},
            refDp, true);

        if (!refResult.success) {
            printResult(false, 0, 0, "Reference kernel failed: " + refResult.error);
            return 1;
        }
    }

    // --- Run candidate kernel (benchmark) ---
    std::vector<float> candOut(M * N, 0.0f);
    {
        auto candResult = runKernel(candidatePath, kernelName,
            {{0, sizeof(args),                 &args},
             {1, (size_t)M * bytes_per_row_q8, src0.data()},
             {2, size_t(N * K) * 4,            src1.data()},
             {3, outBytes,                      candOut.data()}},
            dp, true);

        if (!candResult.success) {
            printResult(false, 0, 0, "Candidate kernel failed: " + candResult.error);
            return 1;
        }

        // --- Compare outputs ---
        bool correct = compareFloat(refOut.data(), candOut.data(), M * N, 1e-3f, 1e-3f);

        double median = candResult.gpuTimeMs;
        double stddev = computeStddev(candResult.gpuTimesMs);

        printResult(correct, median, stddev);
    }

    return 0;
}
