# TPU Backend Setup

## TPU VM Setup

Autocomp evaluates generated Pallas kernels on a remote TPU VM via SSH.

To create a TPU VM via gcloud CLI:

```sh
gcloud alpha compute tpus tpu-vm create <tpu_name> \
    --zone=<zone> \
    --accelerator-type=v6e-1 \
    --version=v2-alpha-tpuv6e \
    --project=<project>
```

With **gcloud** transport (default), the backend auto-creates the VM using `AUTOCOMP_TPU_NAME`, `AUTOCOMP_TPU_ZONE`, `AUTOCOMP_TPU_PROJECT`, `AUTOCOMP_TPU_ACCELERATOR_TYPE`, and `AUTOCOMP_TPU_RUNTIME_VERSION`. For **direct SSH**:

```sh
export AUTOCOMP_TPU_TRANSPORT=ssh
export AUTOCOMP_TPU_SSH_HOST=10.0.0.42
export AUTOCOMP_TPU_SSH_USER=myuser
```

The backend uses Python 3.11 (override with `AUTOCOMP_TPU_PYTHON`) and automatically installs `jax[tpu]==0.9.2` on the first run if the correct version is not present. Force reinstall with `AUTOCOMP_TPU_FORCE_PIP=1`.

## How Evaluation Works

The evaluator builds a single batch script that:

1. Inlines the **test harness** (`harnesses/tpu/test{prob_id}.py`) with imports and `if __name__` stripped.
2. Base64-encodes each LLM-generated implementation and `exec`s it, which defines a `solution()` function.
3. Calls `_run_autocomp_harness()` (from the harness) which invokes `solution()` for warmup, correctness, and benchmarking.
4. Parses stdout for `Latency:` and `Utilization:` lines.

## Adding a New Problem

Each problem needs up to three files. See the matmul problem (`prob_id=0`) as a reference.

1. **Test harness** — `harnesses/tpu/test{prob_id}.py`: self-contained file defining inputs, reference, and `_run_autocomp_harness()`.
2. **Baseline solution** — `sols/tpu/{prob_id}_*.py`: defines `solution(...)` matching the harness signature. Starting point for the LLM.
3. **Context** (optional) — `harnesses/tpu/context{prob_id}.txt`: one-liner describing the problem for the LLM prompt.
4. **Wire up** `run_search.py`: set `prob_type="tpu"` and `prob_id="{prob_id}"`.

## CLI Usage

`tpu_eval.py` can be run directly for debugging:

```sh
python -m autocomp.backend.tpu.tpu_eval <file_path>           # run a single file
python -m autocomp.backend.tpu.tpu_eval --ssh                  # interactive SSH shell
python -m autocomp.backend.tpu.tpu_eval --bench <file1.py> ... # batch-evaluate
```
