# JAXBench Backend Setup

## JAXBench

[JAXBench](https://github.com/aryatschand/JAXBench) is a benchmark suite for JAX and TPU kernel optimization. It contains 50 operator-level workloads from LLM architectures, with both JAX baseline and Pallas-optimized variants.

```sh
git clone https://github.com/aryatschand/JAXBench
```

## Autocomp

```sh
git clone https://github.com/ucb-bar/autocomp
cd autocomp
pip install -e .
```

By default, the backend looks for `JAXBench/` as a sibling directory to the Autocomp repo root. To override:

```sh
export JAXBENCH_DIR=/path/to/JAXBench
```

## Workload Structure

All workloads live under `JAXBench/benchmark/`, each in a numbered directory:

```
benchmark/
  1p_Flash_Attention/
    baseline.py      # vanilla JAX implementation
    optimized.py     # Pallas TPU kernel (where available)
  18k_Conv2D_ReLU_BiasAdd/
    baseline.py      # KernelBench fused ops (baseline only)
```

Each workload file defines `CONFIG`, `create_inputs()`, `workload()`, and `benchmark()`.

## Problem Types

| `prob_type` | Variant used | Description |
|---|---|---|
| `jaxbench-pallas` | `optimized.py` | Pallas-optimized kernel as starting point |
| `jaxbench-baseline` | `baseline.py` | Vanilla JAX baseline as starting point |
| `jaxbench` | `baseline.py` | Alias for `jaxbench-baseline` |

`prob_id` is the workload directory name (e.g., `7p_Ragged_Paged_Attention`, `18k_Conv2D_ReLU_BiasAdd`).

## Running

Set the following in `run_pallas.py` or `run_search.py`:

```python
backend_name = "jaxbench"
agent_name = "built:tpu-v6e"
hw_config = TpuHardwareConfig("v6e-1")
prob_type = "jaxbench-pallas"
prob_id = "7p_Ragged_Paged_Attention"
```

The backend reuses the TPU eval transport layer (gcloud or direct SSH) and automatically installs JAX 0.9.2 on the TPU VM if needed. See [tpu_setup.md](../tpu/tpu_setup.md) for TPU VM creation and configuration.

## How It Works

The backend extracts `create_inputs()` and `workload()` from the workload file, gives the code to the LLM, and uploads generated implementations to the TPU VM via `jaxbench_runner.py` for correctness checking (`jnp.allclose`) and benchmarking.

## Pallas Block Size Autotuning

For `jaxbench-pallas` workloads, each `optimized.py` includes a `TUNED_PARAMS` dict that controls
Checked-in `TUNED_PARAMS` are for TPU v6e-1 / JAX 0.9.2. **Re-run the autotuner for different TPU types** — wrong block sizes can cost 5-70x performance.

```sh
# On the TPU VM:
python3.11 benchmark/tune_pallas.py
```

## Translation Workflow

For vanilla JAX workloads:

1. **Translation**: Run with `translate_iters > 0` to convert to Pallas.
2. **Optimization**: Use `continue_from` to optimize from the translated code.
