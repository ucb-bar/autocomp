#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <variant>

struct KernelRunResult {
    double gpuTimeMs;
    double wallTimeMs;
    std::vector<double> gpuTimesMs;
    bool   success;
    std::string error;
};

struct BufferSpec {
    uint32_t index;
    size_t   size;       // bytes
    void*    data;       // initial data (nullptr = zero-fill)
};

struct FuncConstant {
    uint32_t index;
    std::variant<bool, int32_t, float> value;
};

struct DispatchParams {
    uint32_t grid[3]        = {1, 1, 1};
    uint32_t threadgroup[3] = {256, 1, 1};

    // If true, grid[] is threadgroup counts (dispatchThreadgroups).
    // If false, grid[] is thread counts (dispatchThreads).
    bool dispatchByThreadgroups = false;

    uint32_t threadgroupMemBytes = 0;

    int warmupRuns = 5;
    int benchRuns  = 100;

    std::vector<FuncConstant> funcConstants;
};

// Run a Metal compute kernel and benchmark it.
// If copyBack is true, GPU buffer contents are copied back to BufferSpec.data
// pointers after the final benchmark run (enables correctness checking).
KernelRunResult runKernel(const std::string& metallibPath,
                          const std::string& kernelName,
                          const std::vector<BufferSpec>& buffers,
                          const DispatchParams& params,
                          bool copyBack = false);
