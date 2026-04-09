# Saturn Eval Configuration Guide

## Overview

`saturn_eval.py` evaluates RVV code candidates using:
1. **Spike** - Fast functional simulation for correctness checking
2. **FireSim** - FPGA simulation for cycle-accurate latency measurement

FireSim evaluation includes live uartlog polling with per-candidate hang detection. If a candidate hangs during simulation, it is automatically removed and the remaining candidates are re-run.

## Configuration

Add/Edit these constants at the top of `/autocomp/backend/saturn/saturn_eval.py`:

```python
# Paths — must be configured before use
SATURN_CHIPYARD_PATH = ""   # e.g., "/scratch/user/chipyard"
SATURN_ZEPHYR_BASE = ""     # e.g., "/scratch/user/zephyr-chipyard-sw"

# Timeouts (seconds)
SATURN_SPIKE_TIMEOUT = 60.0
SATURN_COMPILE_TIMEOUT = 120.0
SATURN_FIRESIM_PER_CANDIDATE_TIMEOUT = 60.0  # Per-candidate hang detection timeout
SATURN_FIRESIM_MIN_TIMEOUT = 45.0            # Base timeout before first result

# FireSim sim slot directory (where the live uartlog is written during simulation)
SATURN_FIRESIM_SIM_SLOT_DIR = ""  # e.g., "/scratch/user/FIRESIM_RUNS_DIR/sim_slot_0"
```

Optionally modify the following (more will be explained below):
```python
SATURN_TEMP_DIR = pathlib.Path(__file__).parent / "tmp_dir"
SATURN_ZEPHYR_APP_PATH = pathlib.Path(__file__).parent / "rvv_bench"
```

| Variable | Description |
|----------|-------------|
| `SATURN_CHIPYARD_PATH` | Root of Chipyard installation |
| `SATURN_ZEPHYR_BASE` | Zephyr root |
| `SATURN_TEMP_DIR` | Directory for build artifacts |
| `SATURN_ZEPHYR_APP_PATH` | Zephyr app template with `src/main.c`, `CMakeLists.txt`, and `prj.conf` |
| `SATURN_FIRESIM_SIM_SLOT_DIR` | Path to the FireSim sim slot where the live `uartlog` is written during simulation |

## Test Harness Structure

Saturn test harnesses use `// SUBSTITUTE HERE` / `// SUBSTITUTE END` markers for code injection. The evaluation pipeline:

1. `clean_code()` strips any function wrapper from LLM-generated code (handles any function name, not just `void test`)
2. `SaturnTest.get_test_code()` wraps the cleaned body with timing (`read_cycles()`/`fence()`) and correctness checking (`full_is_equal(OUTPUT_MATRIX_NAME, gold)`)
3. The wrapped code is injected between the markers in the test harness

See `tests/f32/test0.c` and `test1.c` for working examples.

### Test harness requirements

The test harness must provide:

- `// SUBSTITUTE HERE` and `// SUBSTITUTE END` markers in `main()` where candidate code is injected
- Variables set up before the markers that the candidate code can use (e.g., `batch`, `input`, `output`, pointers to data arrays)
- `OUTPUT_MATRIX_NAME` macro defined to point to the output array (used by the injected correctness check `full_is_equal(OUTPUT_MATRIX_NAME, gold)`)
- `gold` array containing the reference output
- `full_is_equal(float*, double*)` function for element-wise comparison
- `read_cycles()` and `fence()` for cycle-accurate timing
- `REPEAT_TEST_ITERS` define for configurable iteration count
- `RESET_STATE()` macro that resets output state and re-initializes candidate code variables (used during FireSim batch mode)
- Must print `"Correct result\n"` on success (the eval backend checks for this string)

### FireSim batching

For FireSim evaluation, `_build_firesim_combined_code()` generates the batched test code automatically. It:

1. Reads the function signature from the solution file in `sols/`
2. Wraps each candidate body in a `__attribute__((noinline))` function
3. Generates a combined `main()` that calls `RESET_STATE()` between candidates, measures cycles with `fence()`/`read_cycles()`, and prints `"ID %d latency: %lu cycles\n"` per candidate (used by uartlog polling for hang detection)

The test harness itself does not need to handle multi-candidate logic — it only needs to provide the `RESET_STATE()` macro and the variables/arrays that candidates operate on.

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

As mentioned above, make sure `SATURN_ZEPHYR_BASE` in `saturn_eval.py` point to your installation.

### Workload Project Structure

Inside of `/autocomp/backend/saturn`, we provide a template app `rvv_bench` for Autocomp to use when compiling RVV code for Saturn:

```
rvv_bench/            # Name of your app
├── CMakeLists.txt    # Zephyr build configuration
├── prj.conf          # Project config (enables RVV, sets memory, etc.)
├── riscv_vector.conf # RVV config
└── src/
    └── main.c        # Placeholder (replaced by test template during build)
```

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Zephyr CMake build file |
| `prj.conf` | Config options (e.g., `CONFIG_RISCV_ISA_EXT_V=y`) |
| `main.c` | Placeholder (replaced by test template during build) |

See the [samples](https://github.com/ucb-bar/zephyr-chipyard-sw/tree/dev/samples/) directory in Zephyr for other example app templates. Make sure `SATURN_ZEPHYR_APP_PATH` in `saturn_eval.py` point to your benchmark app.

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

**Important:** `SATURN_FIRESIM_SIM_SLOT_DIR` must point to the sim slot directory where the live `uartlog` is written during simulation. This is typically `<default_simulation_dir>/sim_slot_0`. Check your `config_runtime.yaml` for the correct `default_simulation_dir`.

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
