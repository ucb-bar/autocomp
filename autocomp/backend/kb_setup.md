# ⚙️ KernelBench Backend Setup

## KernelBench

First, clone [KernelBench](https://github.com/ScalingIntelligence/KernelBench). The baseline code in `sols/kb-level{1,2,3,4}` currently reflects the version of KernelBench we used, so you should either update these files (copy them over from KernelBench and run `sols/process_kb.py`) or use the same version of KernelBench.

```sh
git clone https://github.com/ScalingIntelligence/KernelBench
cd KernelBench
git checkout 6500bbc8cf102520d7a8f09be34ee6d5db1c29b0
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