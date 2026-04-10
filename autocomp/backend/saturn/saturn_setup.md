# Saturn / XNNPACK Eval Configuration Guide

Both the `saturn` and `xnnpack` backends evaluate RVV code on the Saturn vector core using the same infrastructure:
1. **Spike** - Fast functional simulation for correctness checking
2. **FireSim** - FPGA simulation for cycle-accurate latency measurement

FireSim evaluation includes live uartlog polling with per-candidate hang detection. If a candidate hangs during simulation, it is automatically removed and the remaining candidates are re-run.

The **`saturn`** backend is for general RVV kernels with custom test harnesses. The **`xnnpack`** backend targets XNNPACK microkernels using XNNPACK's GTest-based testers for correctness.

## Configuration

Both backends share the same configuration pattern. Edit the constants at the top of the respective `*_eval.py`:

| Variable | Description |
|----------|-------------|
| `*_CHIPYARD_PATH` | Root of Chipyard installation |
| `*_ZEPHYR_BASE` | Zephyr root |
| `*_TEMP_DIR` | Directory for build artifacts |
| `*_ZEPHYR_APP_PATH` | Zephyr app template |
| `*_FIRESIM_SIM_SLOT_DIR` | Path to the FireSim sim slot where the live `uartlog` is written during simulation |
| `*_FIRESIM_PER_CANDIDATE_TIMEOUT` | Per-candidate hang detection timeout (seconds) |
| `*_FIRESIM_MIN_TIMEOUT` | Base timeout before first result (seconds) |

## Output Format

```python
{
    "correct": True,           # Passed all tests
    "test_results": {0: True}, # Per-test pass/fail
    "latency": 1234,           # Latency in cycles
    "firesim_latency": 5678    # FireSim latency (only if simulator="firesim")
}
```

## Test Harness Structure

### Saturn backend (`backend_name = "saturn"`)

Saturn test harnesses use `// SUBSTITUTE HERE` / `// SUBSTITUTE END` markers for code injection. The evaluation pipeline:

1. `clean_code()` strips any function wrapper from LLM-generated code
2. `SaturnTest.get_test_code()` wraps the cleaned body with timing (`read_cycles()`/`fence()`) and correctness checking (`full_is_equal(OUTPUT_MATRIX_NAME, gold)`)
3. The wrapped code is injected between the markers in the test harness

See `harnesses/rvv-f32/test0.c` and `test1.c` for working examples.

**Test harness requirements:**

- `// SUBSTITUTE HERE` and `// SUBSTITUTE END` markers in `main()` where candidate code is injected
- Variables set up before the markers that the candidate code can use (e.g., `batch`, `input`, `output`, pointers to data arrays)
- `OUTPUT_MATRIX_NAME` macro defined to point to the output array
- `gold` array containing the reference output
- `full_is_equal(float*, double*)` function for element-wise comparison
- `read_cycles()` and `fence()` for cycle-accurate timing
- `REPEAT_TEST_ITERS` define for configurable iteration count
- `RESET_STATE()` macro that resets output state and re-initializes candidate code variables (used during FireSim batch mode)
- Must print `"Correct result\n"` on success

**FireSim batching:** `_build_firesim_combined_code()` reads the function signature from the sol file, wraps each candidate in a `__attribute__((noinline))` function, and generates a combined `main()` that calls `RESET_STATE()` between candidates and prints `"ID %d latency: %lu cycles\n"` per candidate. The test harness only needs to provide `RESET_STATE()` and the data arrays.

**Workload project (`rvv_bench/`):** Plain C, compiled as `main.c`.

### XNNPACK backend (`backend_name = "xnnpack"`)

XNNPACK test harnesses use XNNPACK's own GTest-based microkernel testers for correctness (e.g., `GemmMicrokernelTester`, `RAddStoreExpMinusMaxMicrokernelTester`). The test harness owns the correctness logic entirely.

The evaluation pipeline:

1. `clean_code()` strips any function wrapper from LLM-generated code
2. The function signature is parsed from the solution file in `sols/`
3. `XnnpackTest.inject_candidates()` wraps each candidate body in a `__attribute__((noinline))` function and injects it between `// SUBSTITUTE CANDIDATES` markers

See `harnesses/xnnpack-f32/test0.c` through `test3.c` for working examples.

**Test harness requirements:**

- `typedef void (*candidate_fn_t)(...)` defining the kernel function signature
- `// SUBSTITUTE CANDIDATES` and `// SUBSTITUTE CANDIDATES END` markers
- A `candidate_fns[]` / `candidate_ids[]` loop pattern that iterates over injected candidates
- Cast from `candidate_fn_t` to the actual XNNPACK kernel type (e.g., `xnn_f32_raddstoreexpminusmax_ukernel_fn`)
- Use XNNPACK's microkernel tester `.Test()` method for correctness
- Must print `"Correct result\n"` on success and `"ID %d latency: %lu cycles\n"` per candidate
- For spike single-candidate mode, must also print `"Generated implementation latency: %lu cycles\n"`
- On failure, print `"INCORRECT:"` and `goto next_candidate`

**Solution files:** The eval backend parses the first `void` function definition from the sol file to extract the parameter signature. The function can have any name.

**FireSim batching:** All passing candidates are injected into a single test binary using `inject_candidates()`. The test harness loop handles running each candidate through the GTest tester.

**Workload project (`rvv_bench/`):** C++ (`.cpp`) because of GTest. Includes `gtest_zephyr_stubs.h` for bare-metal compatibility.

### XNNPACK library modifications for cycle counting

XNNPACK's GTest microkernel testers do not include cycle counting by default. To support latency measurement, the tester classes need to be patched:

1. Add a `kernel_cycles()` method to each tester class
2. Wrap the kernel invocation in the tester's `Test()` method with `rdcycle` reads
3. These modifications go in the tester headers under `ZEPHYR_BASE/../../third-party/XNNPACK/test/`

The XNNPACK library must be available at `ZEPHYR_BASE/../../third-party/XNNPACK`. The `CMakeLists.txt` builds XNNPACK as a subproject and links it with GTest. Key build details:
- GTest is downloaded via `DownloadGoogleTest.cmake`
- POSIX features unavailable on bare-metal Zephyr are disabled
- A stub header (`gtest_zephyr_stubs.h`) provides `isatty()` which picolibc doesn't have
- XNNPACK tester `.cc` files must be listed in `CMakeLists.txt` under `target_sources`. Add new tester sources here when adding new kernel types.

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

Make sure `*_ZEPHYR_BASE` in the respective `*_eval.py` points to your installation.

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

### Adding a Saturn Target Configuration for Bitstream

Add your Saturn configuration to `chipyard/generators/firechip/chip/src/main/scala/TargetConfigs.scala`. An example Saturn Configuration to add to the file is:

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

In `deploy/config_runtime.yaml`, set `default_platform` to match your FPGA (according to the FireSim docs), `default_simulation_dir` to your choice of directory (you can point to the `firesim` directory), `default_hw_config` to the name of your bitstream from the last step, and `workload_name: saturn.json`. See [config_runtime_example.yaml](config_runtime_example.yaml) for an example.

**Important:** `*_FIRESIM_SIM_SLOT_DIR` must point to the sim slot directory where the live `uartlog` is written during simulation. This is typically `<default_simulation_dir>/sim_slot_0`. Check your `config_runtime.yaml` for the correct `default_simulation_dir`.

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
