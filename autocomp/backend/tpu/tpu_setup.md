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

When using **gcloud** transport (the default), the backend will auto-create the VM if it doesn't exist, using `AUTOCOMP_TPU_PROJECT`, `AUTOCOMP_TPU_ACCELERATOR_TYPE`, and `AUTOCOMP_TPU_RUNTIME_VERSION`.

For **direct SSH** (e.g. a pre-provisioned or tunneled VM):

```sh
export AUTOCOMP_TPU_TRANSPORT=ssh
export AUTOCOMP_TPU_SSH_HOST=10.0.0.42
export AUTOCOMP_TPU_SSH_USER=myuser
```

When `AUTOCOMP_TPU_TRANSPORT=auto` (the default), the backend picks `ssh` if `AUTOCOMP_TPU_SSH_HOST` is set, otherwise `gcloud`.

## How Evaluation Works

The evaluator builds a single batch script that:

1. Inlines the **test harness** (`tests/tpu/test{prob_id}.py`) with imports and `if __name__` stripped.
2. Base64-encodes each LLM-generated implementation and `exec`s it, which defines a `test()` function.
3. Calls `_run_autocomp_harness()` (from the harness) which invokes `test()` for warmup, correctness, and benchmarking.
4. Parses stdout for `Latency:` and `Utilization:` lines.

## Adding a New Problem

Each problem needs up to three files. See the matmul problem (`prob_id=0`) as a reference.

1. **Test harness** — `tests/tpu/test{prob_id}.py`: self-contained file defining inputs, reference, and `_run_autocomp_harness()`.
2. **Baseline solution** — `sols/tpu/{prob_id}_*.py`: defines `test(...)` matching the harness signature. Starting point for the LLM.
3. **Context** (optional) — `tests/tpu/context{prob_id}.txt`: one-liner describing the problem for the LLM prompt.
4. **Wire up** `run_search.py`: set `prob_type="tpu"` and `prob_id="{prob_id}"`.

## CLI Usage

`tpu_eval.py` can be run directly for debugging:

```sh
python -m autocomp.backend.tpu.tpu_eval <file_path>           # run a single file
python -m autocomp.backend.tpu.tpu_eval --ssh                  # interactive SSH shell
python -m autocomp.backend.tpu.tpu_eval --bench <file1.py> ... # batch-evaluate
```
