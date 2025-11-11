# ⚙️ KernelBench Backend Setup

## KernelBench

First, clone [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

```sh
git clone https://github.com/ScalingIntelligence/KernelBench
cd KernelBench
```

Set up KernelBench as described in its README.

```sh
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -r requirements.txt
pip install -e . 
```

## Autocomp

Navigate back to the `autocomp` directory and set up its Python dependencies: ``pip install -e .``

Then, point `KERNELBENCH_DIR` in `autocomp/backend/kb_eval.py` to the root of the KernelBench directory.

## Usage

`prob_type` in `autocomp/search/search.py` should be set to `kb-level{1,2,3,4}`. `prob_id` should be set to the ID of the problem to optimize.
Autocomp will directly pull the initial code to optimize from `KERNELBENCH_DIR`, and call  `KernelBench/scripts/run_and_check.py` to evaluate the generated code.