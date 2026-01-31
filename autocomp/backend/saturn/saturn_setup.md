# Saturn Eval Configuration Guide

## Overview

`saturn_eval.py` evaluates RVV code candidates using:
1. **Spike** - Fast functional simulation for correctness checking
2. **FireSim** - FPGA simulation for cycle-accurate latency measurement

## Configuration

Edit these constants at the top of `saturn_eval.py`:

```python
# Paths
SATURN_CHIPYARD_PATH = "/scratch/kchern2/chipyard-again"
SATURN_ZEPHYR_BASE = "/scratch/kchern2/zephyr-chipyard-sw"
SATURN_ZEPHYR_APP_PATH = "/scratch/kchern2/zephyr-chipyard-sw/samples/rvv_bench"
SATURN_TEMP_DIR = "/scratch/kchern2/saturn_tmp"

# Timeouts (seconds)
SATURN_SPIKE_TIMEOUT = 60.0
SATURN_COMPILE_TIMEOUT = 120
SATURN_FIRESIM_TIMEOUT = 300.0
SATURN_FIRESIM_INDIVIDUAL_TIMEOUT = 500.0
```

| Variable | Description |
|----------|-------------|
| `SATURN_CHIPYARD_PATH` | Root of Chipyard installation |
| `SATURN_ZEPHYR_BASE` | Zephyr root |
| `SATURN_ZEPHYR_APP_PATH` | Zephyr app template with `src/main.c`, `CMakeLists.txt`, and `prj.conf` |
| `SATURN_TEMP_DIR` | Directory for build artifacts |

## Output Format

```python
{
    "correct": True,           # Passed all tests
    "test_results": {0: True}, # Per-test pass/fail
    "latency": 1234,           # Spike latency (cycles)
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

Update `SATURN_ZEPHYR_BASE` in `saturn_eval.py` to point to your installation.

### Workload Project Structure

The RVV benchmark app requires:

```
rvv_bench/            # Name of your app
├── CMakeLists.txt    # Zephyr build configuration
├── prj.conf          # Project config (enables RVV, sets memory, etc.)
└── src/
    └── main.c        # Placeholder (replaced by test template during build)
```

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Zephyr CMake build file |
| `prj.conf` | Config options (e.g., `CONFIG_RISCV_ISA_EXT_V=y`) |
| `main.c` | Placeholder (replaced by test template during build) |

See the [samples](https://github.com/ucb-bar/zephyr-chipyard-sw/tree/dev/samples/) directory in Zephyr for example app templates. Update `SATURN_ZEPHYR_APP_PATH` in `saturn_eval.py` to point to your benchmark app.

## Chipyard

Chipyard is required for both spike and FireSim evaluation. Clone [Chipyard](https://github.com/ucb-bar/chipyard) and check out commit `1c628f7f4fac8e7208ff088073559222b83b91f3`. Then run Chipyard's setup script as described in the [Chipyard docs](https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html), and `source` the created environment.

## FireSim Setup

FireSim is included as a Chipyard submodule but requires additional configuration. See the [FireSim docs](https://docs.fires.im/en/1.20.1/) for full details.

### Initial Setup

```bash
cd firesim
firesim managerinit --platform <your_platform>
source sourceme-manager.sh
```

### Adding a Saturn Target Configuration for Bitstream

Add your Saturn configuration to `chipyard/generators/firechip/chip/src/main/TargetConfigs.scala`. An example Saturn Configuration to add to the file is:

```scala
class FireSimREFV512D256RocketConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.REFV512D256RocketConfig)
```

Then update the FireSim build configuration files:
- `firesim/deploy/config_build.yaml`
- `firesim/deploy/config_build_recipes.yaml`

### Runtime Configuration

| File | Required Changes |
|------|------------------|
| `deploy/config_hwdb.yaml` | Set `bitstream_tar` to your Saturn FPGA bitstream |
| `deploy/config_runtime.yaml` | Set `default_platform`, `default_hw_config`, and `workload_name: saturn.json` |

### Workload Setup

Create `deploy/workloads/saturn/saturn.json`:

```json
{
    "benchmark_name": "saturn",
    "common_simulation_outputs": ["uartlog"],
    "workloads": [
        {
            "name": "saturn_test-baremetal",
            "bootbinary": "saturn_test-baremetal"
        }
    ]
}
```

The evaluation script copies compiled binaries to `deploy/workloads/saturn/saturn_test-baremetal`.

## Autocomp
Finally, set up Autocomp and its Python dependencies: ``pip install -e .``
