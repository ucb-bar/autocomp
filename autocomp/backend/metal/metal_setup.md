# Metal Backend

Testing infrastructure for evaluating Metal compute kernels on Apple Silicon GPUs.

## Architecture

```
Agent generates .metal kernel
        |
        v
metal_eval.py
  1. extract_metal_code() — strip markdown fences / non-kernel text
  2. Write to tmp_files/candidate_N.metal
  3. Compile candidate → candidate.metallib  (xcrun metal/metallib via Makefile)
  4. Compile reference sol → ref.metallib     (cached)
  5. Run harness binary with both metallib paths
  6. Harness runs reference kernel, then candidate kernel on same inputs
  7. Compare GPU outputs (GPU vs GPU)
  8. Parse structured output → {correct, latency, stddev}
```

Kernels are standalone `.metal` files — no code injection. Correctness is verified by comparing the candidate's GPU output against the reference kernel's GPU output.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`

## Setup

Download Apple's metal-cpp C++ headers:

```bash
cd autocomp/backend/metal/runner
make setup
```

This downloads the headers from Apple's developer site into `runner/metal-cpp/`. If the automatic download fails, manually download from https://developer.apple.com/metal/cpp/ and extract into `runner/metal-cpp/`.

Verify the setup:

```bash
# Build the Q8 GEMM harness
make harness PROBLEM=0_gemm_q8_f32 HARNESS_DIR=../../../../harnesses/metal-m2

# Compile the reference solution kernel
make metallib KERNEL_SRC=../../../../sols/metal-m2/0_gemm_q8_f32.metal

# Run (candidate_metallib reference_metallib — use same kernel for both to verify)
./build/0_gemm_q8_f32 build/0_gemm_q8_f32.metallib build/0_gemm_q8_f32.metallib
```

Expected output:
```
CORRECT: true
MEDIAN_MS: <value>
STDDEV_MS: <value>
```

## File Layout

```
autocomp/backend/metal/
├── metal_eval.py         # Python eval backend (orchestrates compile → run → parse)
├── README.md
├── tmp_files/            # Temp directory for candidate .metal files (gitignored)
└── runner/               # Reusable C++ infrastructure
    ├── kernel-runner.hpp # Structs: KernelRunResult, BufferSpec, DispatchParams
    ├── kernel-runner.cpp # runKernel(): loads .metallib, dispatches, benchmarks
    ├── harness-utils.hpp # Utilities: compareFloat(), printResult(), computeStddev()
    ├── Makefile          # Build targets: harness, metallib, setup, clean
    └── metal-cpp/        # Apple metal-cpp headers (downloaded via make setup)

harnesses/metal-m2/
└── 0_gemm_q8_f32.cpp    # Per-problem harness: data gen, dispatch, CPU ref, correctness

sols/metal-m2/
└── 0_gemm_q8_f32.metal  # Reference kernel solution
```

## Configuration Guide

### Change tensor dimensions / problem sizes

Edit the defaults in the harness `.cpp` file, or pass them as CLI arguments:

```bash
# In 0_gemm_q8_f32.cpp, defaults are M=512, N=128, K=1536
# Override via CLI:
./build/0_gemm_q8_f32 build/kernel.metallib 1024 256 3072
```

### Change benchmark parameters (warmup runs, bench runs)

Edit `DispatchParams` in the harness `.cpp`:

```cpp
// In 0_gemm_q8_f32.cpp
dp.warmupRuns = 5;   // change warmup count
dp.benchRuns  = 100;  // change benchmark iterations
```

### Change correctness tolerance

Edit the `compareFloat()` call in the harness `.cpp`:

```cpp
// atol = absolute tolerance, rtol = relative tolerance
bool correct = compareFloat(ref.data(), gpuOut.data(), M * N, 1e-2f, 1e-2f);
```

### Change dispatch configuration (threadgroup size, shared memory)

Edit `DispatchParams` in the harness `.cpp`:

```cpp
dp.grid[0] = (N + 31) / 32;
dp.grid[1] = (M + 63) / 64;
dp.threadgroup[0] = 128;
dp.threadgroupMemBytes = 6144;
```

### Add a new kernel problem

1. Create a solution `.metal` file in `sols/metal-m2/` (e.g., `1_conv2d_f32.metal`)
2. Create a harness `.cpp` in `harnesses/metal-m2/` (e.g., `1_conv2d_f32.cpp`) containing:
   - `main()` accepting `<metallib_path>` as first arg
   - Deterministic test data generation (seeded RNG)
   - CPU reference implementation
   - Buffer layout and dispatch params matching the kernel
   - Hardcoded kernel function name matching the kernel's `[[host_name(...)]]`
   - Calls to `runKernel()`, `compareFloat()`, and `printResult()`
3. Build and test:
   ```bash
   cd autocomp/backend/metal/runner
   make harness PROBLEM=1_conv2d_f32 HARNESS_DIR=../../../../harnesses/metal-m2
   make metallib KERNEL_SRC=../../../../sols/metal-m2/1_conv2d_f32.metal
   ./build/1_conv2d_f32 build/1_conv2d_f32.metallib build/1_conv2d_f32.metallib
   ```

## Metrics

- **Median latency** (primary): Robust to outliers. Measured via Metal GPU timestamps (`GPUEndTime - GPUStartTime`), more accurate than wall-clock timing.
- **Stddev**: Standard deviation across benchmark runs, for stability assessment.
- Default: 5 warmup runs + 100 benchmark runs.

## Kernel Function Name Convention

Agents must keep the `[[host_name("...")]]` attribute as specified in the problem prompt. The harness hardcodes this name to load the correct kernel function from the compiled `.metallib`.
