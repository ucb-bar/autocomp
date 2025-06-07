 <div style="display: flex; flex-wrap: wrap;">
    <img src="accelerator.png" alt="Snow" style="width:35%; margin-right: 10px;">
    <img src="code.png" alt="Forest" style="width:28%; margin-right: 10px;">
    <img src="2phase.svg" alt="Mountains" style="width:32%;">
</div> 

# Autocomp: LLM-driven Code Optimization for Tensor Accelerators

[![arXiv](https://img.shields.io/badge/arXiv-2505.18574-b31b1b.svg)](https://arxiv.org/abs/2505.18574)
[![Blog Post](https://img.shields.io/badge/Blog-github.io-blue)](https://charleshong3.github.io/blog/autocomp.html)

Welcome to the code repository of **Autocomp**. Check out our introductory [blog post](https://charleshong3.github.io/blog/autocomp.html)!

**Paper**: [**Autocomp: LLM-Driven Code Optimization for Tensor Accelerators**](https://arxiv.org/abs/2505.18574)

**Authors**: [Charles Hong](https://charleshong3.github.io/), [Sahil Bhatia](https://x.com/sahilb17), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), and [Yakun Sophia Shao](https://people.eecs.berkeley.edu/~ysshao/) (UC Berkeley)

Note that this repository is still under construction.

## Setup

[Chipyard](https://chipyard.readthedocs.io/en/latest/) and [FireSim](https://docs.fires.im/en/1.20.1/) are needed to replicate experiments with Gemmini (you can also set `"simulator"` in `search.py` to `"spike"`, but this will only optimize instruction counts, not cycle counts).

### Chipyard

First, clone [Chipyard](https://github.com/ucb-bar/chipyard) and check out commit `dbc082e2206f787c3aba12b9b171e1704e15b707`. Then, run Chipyard's setup script as described in the Chipyard docs, and `source` the created environment.

### FireSim

Next, make sure FireSim is set up and ready to run. You will probably need to run `firesim managerinit --platform <your_platform>` and configure files such as `firesim/deploy/config_hwdb.yaml` and `firesim/deploy/config_runtime.yaml`. Make sure to use a FireSim bitstream for your FPGA platform with the Gemmini configuration you want to use. You will also need to set up a FireSim workload `json` file with `"benchmark_name": "gemmini"`. An example of files we used is [here](https://github.com/charleshong3/auto-comp-firesim-files). 

Under `firesim/deploy/workloads`, create a directory called `gemmini`. This will be pointed to by `config_runtime.yaml` and `autocomp/search/hardware_eval.py` within Autocomp.

### Gemmini
The last dependency is [Gemmini](https://github.com/ucb-bar/gemmini), which has already been cloned as a Chipyard subrepository. Navigate to `chipyard/generators/gemmini/software/gemmini-rocc-tests` and check out branch `auto-comp-v2`.

In order to collect scratchpad/accumulator utilization stats, you will need to use our modifications to Spike (the RISC-V ISA simulator). Navigate to `chipyard/generators/gemmini/software/libgemmini` and check out branch `auto-comp`. Then, run the following:
```sh
make
make install
```

### Autocomp
Finally, set up Autocomp and its Python dependencies: ``pip install -e .``

In `autocomp/search/hardware_eval.py`, you will need to update at least one of the paths at the top of the file. By default, you will have set up Gemmini's "default" configuration, in which case you can set `INT8_16PE_CHIPYARD_PATH` to point to your Chipyard directory.

### Note for AWS F1 users
The instruction above have been confirmed to work on a machine with a local Xilinx Alveo U250 FPGA. Due to the upcoming deprecation of AWS F1 instances, FireSim support for AWS is spotty at the moment, but we have confirmed that some configurations work with FireSim-as-top with older versions (such as [this one](https://github.com/charleshong3/firesim-dosa)). However, there may be version mismatches (for example with Gemmini software) if you check out old versions of FireSim, so proceed with caution.

## Repository Structure

**`autocomp/`**
- 

**`prompts/`**

**`sols/`**

**`tests/`**

## Usage

`autocomp/search/search.py` can be used to run Autocomp for GEMM and convolution benchmarks, but TinyMPC kernels (stored under the name `admm-multifunction`) require manual changes to the code.

## Citations
```
@misc{hong2025autocomp,
      title={Autocomp: LLM-Driven Code Optimization for Tensor Accelerators}, 
      author={Charles Hong and Sahil Bhatia and Alvin Cheung and Yakun Sophia Shao},
      year={2025},
      eprint={2505.18574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18574}, 
}
```
