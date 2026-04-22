#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "kernel-runner.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>

KernelRunResult runKernel(const std::string& metallibPath,
                          const std::string& kernelName,
                          const std::vector<BufferSpec>& bufferSpecs,
                          const DispatchParams& params,
                          bool copyBack)
{
    KernelRunResult result{};
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        result.error = "No Metal device found";
        pool->release();
        return result;
    }

    NS::Error* error = nullptr;
    NS::String* path = NS::String::string(metallibPath.c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(path);
    MTL::Library* library = device->newLibrary(url, &error);
    if (!library) {
        result.error = "Failed to load " + metallibPath;
        if (error) { result.error += ": "; result.error += error->localizedDescription()->utf8String(); }
        pool->release();
        return result;
    }

    NS::String* fnName = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
    MTL::Function* function = nullptr;

    if (params.funcConstants.empty()) {
        function = library->newFunction(fnName);
    } else {
        MTL::FunctionConstantValues* fcv = MTL::FunctionConstantValues::alloc()->init();
        for (auto& fc : params.funcConstants) {
            if (auto* b = std::get_if<bool>(&fc.value)) {
                fcv->setConstantValue(b, MTL::DataTypeBool, fc.index);
            } else if (auto* i = std::get_if<int32_t>(&fc.value)) {
                fcv->setConstantValue(i, MTL::DataTypeInt, fc.index);
            } else if (auto* f = std::get_if<float>(&fc.value)) {
                fcv->setConstantValue(f, MTL::DataTypeFloat, fc.index);
            }
        }
        function = library->newFunction(fnName, fcv, &error);
        fcv->release();
    }

    if (!function) {
        result.error = "Kernel '" + kernelName + "' not found in " + metallibPath;
        if (error) { result.error += ": "; result.error += error->localizedDescription()->utf8String(); }
        library->release(); pool->release();
        return result;
    }

    MTL::ComputePipelineState* pso = device->newComputePipelineState(function, &error);
    if (!pso) {
        result.error = "Pipeline creation failed";
        if (error) { result.error += ": "; result.error += error->localizedDescription()->utf8String(); }
        function->release(); library->release(); pool->release();
        return result;
    }

    std::vector<MTL::Buffer*> gpuBuffers;
    for (auto& spec : bufferSpecs) {
        MTL::Buffer* buf = device->newBuffer(spec.size, MTL::ResourceStorageModeShared);
        if (spec.data) {
            memcpy(buf->contents(), spec.data, spec.size);
        } else {
            memset(buf->contents(), 0, spec.size);
        }
        gpuBuffers.push_back(buf);
    }

    MTL::CommandQueue* queue = device->newCommandQueue();

    MTL::Size tgSize(params.threadgroup[0], params.threadgroup[1], params.threadgroup[2]);
    MTL::Size gridSize(params.grid[0], params.grid[1], params.grid[2]);

    auto dispatch = [&]() {
        MTL::CommandBuffer* cmdBuf = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        for (size_t i = 0; i < gpuBuffers.size(); i++) {
            enc->setBuffer(gpuBuffers[i], 0, bufferSpecs[i].index);
        }
        if (params.threadgroupMemBytes > 0) {
            enc->setThreadgroupMemoryLength(params.threadgroupMemBytes, 0);
        }
        if (params.dispatchByThreadgroups) {
            enc->dispatchThreadgroups(gridSize, tgSize);
        } else {
            enc->dispatchThreads(gridSize, tgSize);
        }
        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        return cmdBuf;
    };

    // Warmup
    for (int i = 0; i < params.warmupRuns; i++) dispatch();

    // Reset buffers before benchmark
    for (size_t i = 0; i < bufferSpecs.size(); i++) {
        if (bufferSpecs[i].data) {
            memcpy(gpuBuffers[i]->contents(), bufferSpecs[i].data, bufferSpecs[i].size);
        } else {
            memset(gpuBuffers[i]->contents(), 0, bufferSpecs[i].size);
        }
    }

    // Benchmark
    std::vector<double> gpuTimes;
    gpuTimes.reserve(params.benchRuns);
    auto wallStart = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < params.benchRuns; run++) {
        MTL::CommandBuffer* cmdBuf = dispatch();
        gpuTimes.push_back((cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0);
    }

    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();

    // Copy back GPU buffer contents for correctness checking
    if (copyBack) {
        for (size_t i = 0; i < bufferSpecs.size(); i++) {
            if (bufferSpecs[i].data) {
                memcpy(bufferSpecs[i].data, gpuBuffers[i]->contents(), bufferSpecs[i].size);
            }
        }
    }

    std::sort(gpuTimes.begin(), gpuTimes.end());
    result.gpuTimesMs = gpuTimes;
    result.gpuTimeMs  = gpuTimes[gpuTimes.size() / 2];
    result.wallTimeMs = wallMs / params.benchRuns;
    result.success    = true;

    for (auto* buf : gpuBuffers) buf->release();
    queue->release();
    pso->release();
    function->release();
    library->release();
    pool->release();

    return result;
}
