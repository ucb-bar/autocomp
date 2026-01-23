# ⚙️ Gemmini Backend Setup

[Chipyard](https://chipyard.readthedocs.io/en/latest/) and [FireSim](https://docs.fires.im/en/1.20.1/) are needed to replicate experiments with Gemmini (you can also set `"simulator"` in `search.py` to `"spike"`, but this will only optimize instruction counts, not cycle counts).

#### ⚠️ Note for AWS F1 users
The instruction above have been confirmed to work on a machine with a local Xilinx Alveo U250 FPGA. Due to the upcoming deprecation of AWS F1 instances, FireSim support for AWS is spotty at the moment, but we have confirmed that some configurations work with FireSim-as-top with older versions (such as [this one](https://github.com/charleshong3/firesim-dosa)). However, there may be version mismatches (for example with Gemmini software) if you check out old versions of FireSim, so proceed with caution.

## Chipyard

First, clone [Chipyard](https://github.com/ucb-bar/chipyard) and check out commit `dbc082e2206f787c3aba12b9b171e1704e15b707`. Then, run Chipyard's setup script as described in the Chipyard docs, and `source` the created environment.

## FireSim

Next, make sure FireSim is set up and ready to run. FireSim has already been cloned as a submodule of Chipyard, but requires some additional setup as described in the [FireSim docs](https://docs.fires.im/en/1.20.1/). Within the `firesim` directory, you will need to run `firesim managerinit --platform <your_platform>` and configure files such as `firesim/deploy/config_hwdb.yaml` and `firesim/deploy/config_runtime.yaml`. Make sure to use a FireSim bitstream for your FPGA platform with the Gemmini configuration you want to use. You will also need to set up a FireSim workload `json` file with `"benchmark_name": "gemmini"`. An example of files we used is [here](https://github.com/charleshong3/auto-comp-firesim-files). 

Under `firesim/deploy/workloads`, create a directory called `gemmini`. This will be pointed to by `config_runtime.yaml` and `autocomp/backend/gemmini/gemmini_eval.py` within Autocomp.

### TinyMPC Kernel Optimization
TinyMPC kernels (stored under the name `admm-multifunction`) run on an FP32 4x4 Gemmini configuration, which requires building a new FireSim bitstream with Gemmini's FP32DefaultConfig.

## Gemmini
The last dependency is [Gemmini](https://github.com/ucb-bar/gemmini), which has already been cloned as a Chipyard submodule. Navigate to `chipyard/generators/gemmini/software/gemmini-rocc-tests` and check out branch `auto-comp-v2`.

In order to collect scratchpad/accumulator utilization stats, you will need to use our modifications to Spike (the RISC-V ISA simulator). Navigate to `chipyard/generators/gemmini/software/libgemmini` and check out branch `auto-comp`. Then, run the following:
```sh
make
make install
```

## Autocomp
Finally, set up Autocomp and its Python dependencies: ``pip install -e .``

In `autocomp/backend/gemmini/gemmini_eval.py`, you will need to update at least one of the paths at the top of the file. By default, you will have set up Gemmini's "default" int8, 16x16 systolic array configuration, in which case you can set `INT8_16PE_CHIPYARD_PATH` to point to your Chipyard directory. For TinyMPC kernels, you would set the `FP32_4PE_CHIPYARD_PATH` to point to your Chipyard directory.
