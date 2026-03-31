# JAXBench Backend Setup

## JAXBench

[JAXBench](https://github.com/aryatschand/JAXBench) is a benchmark suite for JAX and TPU kernel optimization. It contains operator-level workloads from LLM architectures.

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
| `jaxbench-pallas` | 6 | Upstream Pallas TPU kernels from JAX 0.6.2 (flash attention, matmul, etc.) |
| `jaxbench-real` | 36 | Hand-written ops from 7 LLM families + attention variants (from [MaxText](https://github.com/AI-Hypercomputer/maxtext)) |
| `jaxbench-priority` | 10 | Priority kernels selected for optimization |
| `jaxbench-tokamax` | 12 | TPU kernel benchmarks from [openxla/tokamax](https://github.com/openxla/tokamax) |
| `jaxkernelbench` | 200 | LLM-translated PyTorch→JAX operators (from [KernelBench](https://github.com/ScalingIntelligence/KernelBench)) |

`prob_id` is the workload filename without `.py` (e.g., `llama3_8b_gqa`, `mixtral_8x7b_moe`).

## Running

Set the following in `run_search.py`:

```python
backend_name = "jaxbench"
agent_name = "built:tpu-v6e"
hw_config = TpuHardwareConfig("v6e-1")
prob_type = "jaxbench-pallas"  # or jaxbench-pallas, jaxbench-priority, jaxbench-tokamax, jaxkernelbench
prob_id = "flash_attention"
```

The backend reuses the TPU eval transport layer (gcloud or direct SSH). See [tpu_setup.md](../tpu/tpu_setup.md) for TPU VM configuration.

## How It Works

Each JAXBench workload file defines `create_inputs()` and `workload()`. The backend:

1. Extracts the workload code and gives it to the LLM as the starting point.
2. The LLM generates optimized implementations that redefine `workload()`.
3. Implementations are uploaded to the TPU VM alongside `jaxbench_runner.py`, which handles correctness checking (`jnp.allclose`) and benchmarking.

Correctness is checked using `jnp.allclose(ref_out, impl_out, atol=atol, rtol=rtol)`.
Tolerances default to `atol=3.125e-2` and `rtol=1e-2` (overridable via `AUTOCOMP_JAXBENCH_ATOL`
and `AUTOCOMP_JAXBENCH_RTOL` environment variables). The `pallas_kernels` workloads specify
tighter per-kernel tolerances in their `CONFIG` dicts, matching the upstream JAX tests.

## Recommended Workflow

### Pallas kernel block size autotuning

For `jaxbench-pallas` workloads, each kernel file includes a `TUNED_PARAMS` dict that controls
block sizes. The checked-in values are tuned for TPU v6e-1. **If you are using a different TPU
type, you must re-run the autotuner first** — optimal block sizes vary significantly across TPU
generations, and starting from the wrong sizes can leave 5-70x performance on the table.

```sh
# Upload kernel files + autotuner to TPU VM, then:
cd /tmp
python3 autotune_block_sizes.py --apply    # tunes all 6 kernels, writes results back to files
```

See the [pallas_kernels README](https://github.com/aryatschand/JAXBench/blob/main/pallas_kernels/README.md) for details.

### Translation before optimization

For workloads that are vanilla JAX code, if you would like to translate them to Pallas kernels, we recommend a two-phase approach:

1. **Translation**: Run with `translate_iters > 0` to first convert the JAX workload into a Pallas kernel. Inspect the outputs to verify the translation is complete.
2. **Optimization**: Use `resume_from` to load the translated candidates and optimize from there.

This avoids wasting optimization iterations on code that hasn't been fully translated to Pallas yet.
