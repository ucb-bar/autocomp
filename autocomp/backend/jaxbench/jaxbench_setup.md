# JAXBench Backend Setup

## JAXBench

[JAXBench](https://github.com/aryatschand/JAXBench) is a benchmark suite for JAX and TPU kernel optimization. It contains operator-level workloads from LLM architectures — vanilla JAX baselines that Pallas optimization agents try to beat. All benchmarks target **TPU v6e-1**.

Clone the repo:

```sh
git clone https://github.com/aryatschand/JAXBench
```

## Autocomp

Clone Autocomp and install:

```sh
git clone https://github.com/ucb-bar/autocomp
cd autocomp
pip install -e .
```

By default, the backend looks for `JAXBench/` as a sibling directory to the Autocomp repo root. To override, set the `JAXBENCH_DIR` environment variable:

```sh
export JAXBENCH_DIR=/path/to/JAXBench
```

## Available Problem Types

| `prob_type` | Workloads | Description |
|---|---|---|
| `jaxbench-real` | 36 | Hand-written ops from 7 LLM families + attention variants (from [MaxText](https://github.com/AI-Hypercomputer/maxtext)) |
| `jaxbench-priority` | ~10 | Priority kernels selected for Pallas optimization |
| `jaxbench-tokamax` | 12 | TPU kernel benchmarks from [openxla/tokamax](https://github.com/openxla/tokamax) |
| `jaxkernelbench` | 200 | LLM-translated PyTorch→JAX operators (from [KernelBench](https://github.com/ScalingIntelligence/KernelBench)) |

`prob_id` is the workload filename without `.py` (e.g., `llama3_8b_gqa`, `mixtral_8x7b_moe`).

## Running

Set the following in `search.py`:

```python
backend_name = "jaxbench"
agent_name = "built:tpu-v6e"
hw_config = TpuHardwareConfig("v6e-1")
prob_type = "jaxbench-real"  # or jaxbench-priority, jaxbench-tokamax, jaxkernelbench
prob_id = "llama3_8b_gqa"
```

The backend reuses the TPU eval transport layer (gcloud or direct SSH). See [tpu_setup.md](../tpu/tpu_setup.md) for TPU VM configuration.

## How It Works

Each JAXBench workload file defines `create_inputs()` and `workload()`. The backend:

1. Extracts the workload code and gives it to the LLM as the starting point.
2. The LLM generates optimized implementations that redefine `workload()`.
3. Implementations are uploaded to the TPU VM alongside `jaxbench_runner.py`, which handles correctness checking (`jnp.allclose`) and benchmarking.

## Recommended Workflow

JAXBench workloads are vanilla JAX code, not Pallas kernels. We recommend a two-phase approach:

1. **Translation**: Run with `translate_iters > 0` to first convert the JAX workload into a Pallas kernel. Inspect the outputs to verify the translation is complete.
2. **Optimization**: Use `resume_from` to load the translated candidates and optimize from there.

This avoids wasting optimization iterations on code that hasn't been fully translated to Pallas yet.
