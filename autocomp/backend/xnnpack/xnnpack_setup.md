# XNNPACK Eval Configuration Guide

## Overview

`xnnpack_eval.py` evaluates RVV code candidates for XNNPACK microkernels using:
1. **Spike** - Fast functional simulation for correctness checking
2. **FireSim** - FPGA simulation for cycle-accurate latency measurement

FireSim evaluation includes live uartlog polling with per-candidate hang detection. If a candidate hangs during simulation, it is automatically removed and the remaining candidates are re-run.

## Configuration

Add/Edit these constants at the top of `/autocomp/backend/xnnpack/xnnpack_eval.py`:

```python
# Paths — must be configured before use
XNNPACK_CHIPYARD_PATH = ""   # e.g., "/scratch/user/chipyard"
XNNPACK_ZEPHYR_BASE = ""     # e.g., "/scratch/user/zephyr-chipyard-sw"

# Timeouts (seconds)
XNNPACK_SPIKE_TIMEOUT = 60.0
XNNPACK_COMPILE_TIMEOUT = 300.0
XNNPACK_FIRESIM_PER_CANDIDATE_TIMEOUT = 60.0  # Per-candidate hang detection timeout
XNNPACK_FIRESIM_MIN_TIMEOUT = 45.0            # Base timeout before first result

# FireSim sim slot directory (where the live uartlog is written during simulation)
XNNPACK_FIRESIM_SIM_SLOT_DIR = ""  # e.g., "/scratch/user/FIRESIM_RUNS_DIR/sim_slot_0"
```

Optionally modify the following (more will be explained below):
```python
XNNPACK_TEMP_DIR = pathlib.Path(__file__).parent / "tmp_dir"
XNNPACK_ZEPHYR_APP_PATH = pathlib.Path(__file__).parent / "rvv_bench"
```

| Variable | Description |
|----------|-------------|
| `XNNPACK_CHIPYARD_PATH` | Root of Chipyard installation |
| `XNNPACK_ZEPHYR_BASE` | Zephyr root |
| `XNNPACK_TEMP_DIR` | Directory for build artifacts |
| `XNNPACK_ZEPHYR_APP_PATH` | Zephyr app template with `src/main.cpp`, `CMakeLists.txt`, and `prj.conf` |
| `XNNPACK_FIRESIM_SIM_SLOT_DIR` | Path to the FireSim sim slot where the live `uartlog` is written during simulation |

## Test Harness Structure

XNNPACK test harnesses differ from Saturn in that they use XNNPACK's own GTest-based microkernel testers for correctness checking (e.g., `GemmMicrokernelTester`, `RAddStoreExpMinusMaxMicrokernelTester`, `TransposecMicrokernelTester`). The test harness owns the correctness logic entirely — the Python eval backend only checks for `"Correct result"` in the output.

The evaluation pipeline:
1. `clean_code()` strips any function wrapper from LLM-generated code
2. The function signature is parsed from the solution file in `sols/` (not from the test harness)
3. `XnnpackTest.inject_candidates()` wraps each candidate body in a `__attribute__((noinline))` function and injects it between `// SUBSTITUTE CANDIDATES` markers

See `tests/xnnpack-f32/test0.c` through `test3.c` for working examples.

### Test harness requirements

- `typedef void (*candidate_fn_t)(...)` defining the kernel function signature
- `// SUBSTITUTE CANDIDATES` and `// SUBSTITUTE CANDIDATES END` markers where candidate functions, arrays, and `NUM_CANDIDATES` are injected
- A `candidate_fns[]` / `candidate_ids[]` loop pattern that iterates over injected candidates
- Cast from `candidate_fn_t` to the actual XNNPACK kernel type (e.g., `xnn_f32_raddstoreexpminusmax_ukernel_fn`, `xnn_dwconv2d_chw_ukernel_fn`)
- Use XNNPACK's microkernel tester `.Test()` method for correctness
- Must print `"Correct result\n"` on success
- Must print `"ID %d latency: %lu cycles\n"` per candidate (used by uartlog polling for FireSim hang detection)
- For spike single-candidate mode, must also print `"Generated implementation latency: %lu cycles\n"`
- On failure, print `"INCORRECT:"` and use `goto next_candidate` to skip to the next candidate
- Call `sys_reboot(SYS_REBOOT_COLD)` to terminate on single-candidate failure or after all candidates complete

### Solution files

Solution files in `sols/<prob_type>/` contain the reference kernel implementation with the full function signature. The eval backend parses the first `void` function definition from the sol file to extract the parameter signature for candidate injection. The function can have any name (not required to be `void test`).

### FireSim batching

For FireSim evaluation, all passing candidates are injected into a single test binary using `inject_candidates()`. The test harness loop runs each candidate through the GTest tester and reports per-candidate cycle counts. This is more efficient than saturn's approach since XNNPACK's testers already handle all the setup/teardown internally.

## XNNPACK Library Modifications for Cycle Counting

XNNPACK's GTest microkernel testers do not include cycle counting by default. To support latency measurement on spike/FireSim, the tester classes in the XNNPACK third-party source need to be patched:

1. Add a `kernel_cycles()` method to each microkernel tester class that returns accumulated cycle counts
2. Wrap the kernel invocation inside the tester's `Test()` method with `rdcycle` reads:
   ```cpp
   unsigned long start = read_cycles();
   // ... kernel call ...
   unsigned long end = read_cycles();
   total_kernel_cycles_ += (end - start);
   ```
3. These modifications go in the tester headers under `ZEPHYR_BASE/../../third-party/XNNPACK/test/` (e.g., `gemm-microkernel-tester.h`, `raddstoreexpminusmax-microkernel-tester.h`)

Without these patches, the test harnesses can still check correctness but cannot report cycle-accurate latency.

## Output Format

```python
{
    "correct": True,           # Passed all tests
    "test_results": {0: True}, # Per-test pass/fail
    "latency": 1234,           # Latency in cycles
    "firesim_latency": 5678    # FireSim latency (only if simulator="firesim")
}
```

## Zephyr Setup

### Installation

These installation instructions are from the [Zephyr repo](https://github.com/ucb-bar/zephyr-chipyard-sw) using SDK installation.

```bash
# Clone the repository
git clone git@github.com:ucb-bar/zephyr-chipyard-sw.git
cd zephyr-chipyard-sw

# Initialize submodules
git submodule update --init

# Install conda
source scripts/install_conda.sh

# Install submodules
bash scripts/install_submodules.sh

# Install toolchain SDK
bash scripts/install_toolchain_sdk.sh
```

Make sure `XNNPACK_ZEPHYR_BASE` in `xnnpack_eval.py` points to your installation.

### XNNPACK as a Zephyr Third-Party Dependency

The XNNPACK library must be available at `ZEPHYR_BASE/../../third-party/XNNPACK`. The `CMakeLists.txt` in `rvv_bench/` builds XNNPACK as a subproject and links it with GTest for microkernel testing.

Key build details:
- GTest is downloaded and built alongside XNNPACK (via `DownloadGoogleTest.cmake`)
- POSIX features unavailable on bare-metal Zephyr are disabled (`GTEST_HAS_FILE_SYSTEM=0`, `GTEST_HAS_STREAM_REDIRECTION=0`, etc.)
- A stub header (`gtest_zephyr_stubs.h`) provides `isatty()` which picolibc doesn't have
- XNNPACK tester `.cc` files needed by test harnesses must be listed in `CMakeLists.txt` under `target_sources` (e.g., `gemm-microkernel-tester.cc`, `next_prime.cc`). Add new tester sources here when adding new kernel types.

### Workload Project Structure

Inside of `/autocomp/backend/xnnpack`, we provide a template app `rvv_bench`:

```
rvv_bench/
├── CMakeLists.txt         # Zephyr + XNNPACK + GTest build configuration
├── prj.conf               # Project config (enables RVV, C++20, large heap)
├── riscv_vector.conf      # RVV config
├── gtest_zephyr_stubs.h   # isatty() stub for bare-metal GTest
└── src/
    └── main.cpp           # Placeholder (replaced by test harness during build)
```

Note: XNNPACK test harnesses are C++ (`.cpp`) because they use GTest. This differs from saturn which uses plain C (`.c`).

## Chipyard

Chipyard is required for both spike and FireSim evaluation. Clone [Chipyard](https://github.com/ucb-bar/chipyard) and check out commit `1c628f7f4fac8e7208ff088073559222b83b91f3`. Then run Chipyard's setup script as described in the [Chipyard docs](https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html), and `source` the created environment.

## FireSim Setup

FireSim is included as a Chipyard submodule but requires additional configuration. See the [FireSim docs](https://docs.fires.im/en/1.20.1/) for full details.

### Initial Setup

```bash
cd chipyard/sims/firesim
source sourceme-manager.sh
firesim managerinit --platform <your_platform> # e.g., xilinx_alveo_u250
```

### Adding a Target Configuration for Bitstream

Add your configuration to `chipyard/generators/firechip/chip/src/main/scala/TargetConfigs.scala`. An example configuration:

```scala
class FireSimREFV512D256RocketConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.REFV512D256RocketConfig)
```

Then update the FireSim build configuration files:
- `firesim/deploy/config_build_recipes.yaml`
- `firesim/deploy/config_build.yaml`

An example build recipe can be found in [config_build_recipes_example.yaml](config_build_recipes_example.yaml). In `config_build.yaml`, change `default_build_dir` to a directory of your choice and update the name of the build recipe under the list of `builds_to_run`.

Then run the bitstream build. Make sure that FPGA-related toolchains are available in your PATH.

```bash
firesim buildbitstream
```

Once the command successfully completes, it will print out a YAML entry you can paste into `deploy/config_hwdb.yaml` to provide FireSim a pointer to the generated bitstream.

### Runtime Configuration

In `deploy/config_runtime.yaml`, set `default_platform` to match your FPGA (according to the FireSim docs), `default_simulation_dir` to your choice of directory, `default_hw_config` to the name of your bitstream from the last step, and `workload_name: saturn.json`. See [config_runtime_example.yaml](config_runtime_example.yaml) for an example.

**Important:** `XNNPACK_FIRESIM_SIM_SLOT_DIR` must point to the sim slot directory where the live `uartlog` is written during simulation. This is typically `<default_simulation_dir>/sim_slot_0`. Check your `config_runtime.yaml` for the correct `default_simulation_dir`.

### Workload Setup

Create the directory `deploy/workloads/saturn` and the file `deploy/workloads/saturn.json`:

```json
{
    "benchmark_name": "saturn",
    "common_simulation_outputs": ["uartlog"],
    "workloads": [
        {
            "name": "saturn_test-baremetal",
            "bootbinary": "saturn_test-baremetal",
            "rootfs": "../../../../../software/firemarshal/boards/default/installers/firesim/dummy.rootfs"
        }
    ]
}
```

The evaluation script copies compiled binaries to `deploy/workloads/saturn/saturn_test-baremetal`.

## Autocomp
Finally, set up Autocomp and its Python dependencies: 
```
cd autocomp
pip install -e .
```
