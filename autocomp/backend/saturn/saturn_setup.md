# Saturn Eval Configuration Guide

## Overview

`saturn_eval.py` evaluates RVV code candidates using:
1. **Spike** - Fast functional simulation for correctness checking
2. **FireSim** - FPGA simulation for accurate latency measurement

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
| `SATURN_ZEPHYR_BASE` | Zephyr SDK root |
| `SATURN_ZEPHYR_APP_PATH` | Zephyr app template with `src/main.c`, `CMakeLists.txt` |
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

These Installations instructions were taken from the [Zephyr repo](https://github.com/ucb-bar/zephyr-chipyard-sw) using the SDK installation.

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

Update `SATURN_ZEPHYR_BASE` with zephyr repo in `saturn_eval.py` to point to your installation.
### Workload Project Structure

The RVV benchmark app requires:

```
rvv_bench/            # Name of your app
├── CMakeLists.txt    # Zephyr build configuration
├── prj.conf          # Project config (enables RVV, sets memory, etc.)
└── src/
    └── main.c        # Test harness template (overwritten during build)
```

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Zephyr CMake build file |
| `prj.conf` | config options (e.g., `CONFIG_RISCV_ISA_EXT_V=y`) |
| `main.c` | Placeholder (replaced by test template during build) |

Update  `SATURN_ZEPHYR_APP_PATH` in `saturn_eval.py` to point to this Zephyr benchmark setup.


## FireSim Setup

FireSim requires additional configuration:

1. **Source environment** before running:
   ```bash
   source {SATURN_CHIPYARD_PATH}/sourceme-manager.sh
   ```

2. **Workload config** at `{firesim_path}/deploy/workloads/saturn/`:
   - Binary copied to: `saturn_test-baremetal`
   - May need `saturn.json` depending on your FireSim setup

3. **Runtime config** in `config_runtime.yaml` must reference the saturn workload

