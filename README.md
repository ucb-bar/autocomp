<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/autocomp_logo_dark.svg">
    <img alt="Autocomp Logo" src="img/autocomp_logo.svg" width=55%>
  </picture>
</p>

<h3 align="center">
AI-Driven Code Optimizer for Tensor Accelerators
</h3>

<p align="center">
| <a href="https://arxiv.org/abs/2505.18574"><b>arXiv</b></a> | <a href="https://charleshong3.github.io/blog/autocomp.html"><b>Blog</b></a> |
</p>

Welcome to the code repository of **Autocomp**. Recent updates:

**(3/25/2026)** Added support for structured-output code edits in the code implementation phase.

**(3/17/2026)** Added preliminary TPU support and enhanced Autocomp's code translation capabilities.

**(3/13/2026)** Added the **Agent Builder** for automatically creating hardware-specific LLM agents from documentation sources.

**📚 Paper**: [**Autocomp: A Powerful and Portable Code Optimizer for Tensor Accelerators**](https://arxiv.org/abs/2505.18574)

**✏️ Authors**: [Charles Hong](https://charleshong3.github.io/), [Sahil Bhatia](https://x.com/sahilb17), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), and [Yakun Sophia Shao](https://people.eecs.berkeley.edu/~ysshao/) (UC Berkeley)

### What is Autocomp?

Autocomp is an LLM-driven code optimizer for tensor accelerators. Autocomp is designed to be portable and easy to use across a variety of hardware targets, and has already demonstrated strong performance on an industry accelerator ([AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)), an academic accelerator ([Gemmini](https://github.com/ucb-bar/gemmini)), NVIDIA GPUs, and even the RISC-V Vector Extension.

### How does Autocomp work?

Autocomp decomposes the optimization problem into a beam search, where each iteration is further divided into a planning phase and an implementation phase. Autocomp applies the user's domain knowledge, along with a variety of techniques to successfully explore the search space, in order to iteratively improve the code. For more details, see our [paper](https://arxiv.org/abs/2505.18574).

# ⚙️ Setup

## Hardware Target Setup

Autocomp can currently optimize code for the following hardware targets:
- AWS Trainium ([trn_setup.md](autocomp/backend/trn/trn_setup.md)) — uses a [built agent](autocomp/agent_builder/README.md) (`built:trn-nki1`)
- Google TPU ([tpu_setup.md](autocomp/backend/tpu/tpu_setup.md)) — uses a [built agent](autocomp/agent_builder/README.md) (`built:tpu-v6e`)
- Gemmini ([gemmini_setup.md](autocomp/backend/gemmini/gemmini_setup.md))
- CUDA via KernelBench ([kb_setup.md](autocomp/backend/kernelbench/kb_setup.md))
- CUDA via GPU MODE ([gpumode_setup.md](autocomp/backend/gpumode/gpumode_setup.md))

> **Note:** Built agents are the recommended path for new hardware targets (e.g., `built:trn-nki1` for Trainium, `built:tpu-v6e` for TPU). These are created by the Agent Builder and stored in `autocomp/agent_builder/.built/` by default. Set `agent_name = "built:<name>"` in `run_search.py` to use them. Legacy handcrafted agents in `autocomp/agents/` are still available for some targets.

Partially supported hardware targets:
- RISC-V Vector (RVV) on Canaan Kendryte K230. See `k230` branch for code. As the implementation is very hacky, we do not currently recommend using this hardware target.

For instructions on adding a new hardware target, see [ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md).

### 🏗️ Agent Builder (Recommended for New Hardware Targets)

The recommended way to add a new agent is with the **Agent Builder**, which automatically creates a hardware-specific agent from documentation sources (local directories, PDFs, and webpages):

```bash
pip install "autocomp[agent-builder]"

python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --source-dir path/to/docs \
    --agent-scope "Optimizing kernels for MyAccelerator using the XYZ programming interface."
```

For detailed usage, CLI options, Python API, and output format, see [Agent Builder documentation](autocomp/agent_builder/README.md).

## LLM Setup

Autocomp supports both local and remote endpoint LLM inference. For local inference, we support vLLM's [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/). For endpoint inference, we support a variety of providers (see below).

### Local Inference with vLLM

1. **Install and launch vLLM:**
   ```bash
   pip install vllm
   vllm serve --model Qwen/Qwen3-8B --port 8000 -tp <number of GPUs>
   ```

2. **Configure Autocomp:**
   Set `models`/`code_models` in `run_search.py`:
   ```python
   models = ["vllm::Qwen/Qwen3-8B"]
   ```
   Optionally set `VLLM_API_BASE` if using a different host/port (default: `http://localhost:8000/v1`).

3. **Multiple models on different ports:**
   You can serve multiple vLLM models on separate ports and use them together by encoding the base URL in the provider string with the format `vllm@<base_url>::<model_name>`:
   ```bash
   # Terminal 1
   vllm serve --model Qwen/Qwen3-8B --port 8000 -tp 1
   # Terminal 2
   vllm serve --model meta-llama/Llama-3-70B --port 8001 -tp 4
   ```
   ```python
   models = [
       "vllm@http://localhost:8000/v1::Qwen/Qwen3-8B",
       "vllm@http://localhost:8001/v1::meta-llama/Llama-3-70B",
   ]
   ```

For more details, see the [vLLM documentation](https://docs.vllm.ai/).

### LLM Endpoint Setup

API keys can be configured via environment variables or in `autocomp/common/keys.py`. Environment variables take precedence over the keys file. The variable names in `keys.py` match the corresponding environment variable names.

**Supported keys:**

| Provider | Environment Variable / Key Name | Provider Name in `run_search.py`
|----------|--------------------------------|--------------------------------|
| OpenAI | `OPENAI_API_KEY` | `openai`
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic`
| Together | `TOGETHER_API_KEY` | `together`
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | `aws`
| Google Cloud (Vertex AI) | `GOOGLE_CLOUD_LOCATION`, `GOOGLE_CLOUD_PROJECT` | `gcp`
| Google AI Studio | `GOOGLE_API_KEY` | `gcp`

**Example `autocomp/common/keys.py`:**

```python
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
TOGETHER_API_KEY = "..."
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_CLOUD_PROJECT = "my-project"
GOOGLE_API_KEY = "AIza..."
```

Keys can be omitted if not needed. On startup, Autocomp logs which keys are available.

#### Gemini Setup

**Option 1: Google Cloud (Vertex AI).** Install the Google Cloud CLI as described at https://docs.cloud.google.com/sdk/docs/install-sdk#linux. Run `gcloud auth application-default login` and set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

**Option 2: Google AI Studio (easiest).** Get an API key from [Google AI Studio](https://aistudio.google.com/apikey) and set `GOOGLE_API_KEY`.

If both Vertex AI credentials and `GOOGLE_API_KEY` are set, Vertex AI is used.

#### AWS Bedrock

Anthropic (Claude) models on Bedrock use the native Anthropic SDK adapter. All other Bedrock models (e.g., GLM, DeepSeek, Kimi) are supported via the Bedrock Converse API. Any model available in your Bedrock region can be used by passing its Bedrock model ID:

```python
models = [
    "aws::us.anthropic.claude-opus-4-5-20251101-v1:0",  # Claude (Anthropic adapter)
    "aws::zai.glm-4.7",                                 # GLM 4.7
    "aws::deepseek.v3.2",                               # DeepSeek-V3.2
    "aws::moonshotai.kimi-k2.5",                        # Kimi K2.5
]
```

By default the `us-west-2` region is used. Set the `AWS_REGION` environment variable (or add it to `keys.py`) to override.

## 🚀 Usage

`autocomp/search/run_search.py` is the entry point for running Autocomp optimization. Various parameters such as hardware target, models used, beam size, number of plans, number of code implementations, dropout, etc. can be configured here.

```bash
python -m autocomp.search.run_search
```

Notable parameters:

**Target & Environment**
- `backend_name`: The hardware target to use. Currently supported values are `trn`, `tpu`, `gemmini`, `kernelbench`, and `gpumode`.
- `agent_name`: The LLM agent type to use. For most targets, use a built agent: `"built:<name>"` (e.g., `"built:trn-nki1"` for Trainium, `"built:tpu-v6e"` for TPU). See the [Agent Builder docs](autocomp/agent_builder/README.md). Legacy handcrafted agents are also available for some targets (`trn`, `gemmini`, `cuda`).
- `hw_config`: A hardware configuration object describing the target hardware. Examples:
  - `TrnHardwareConfig("trn1.2xlarge")`
  - `TpuHardwareConfig("v6e-1")`
  - `GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)`
  - `CudaHardwareConfig("NVIDIA L40S", "2.5.0", "12.4")`
- `simulator`: The evaluation method to use, if multiple are supported.
  - For Trainium, doesn't matter (put `None`)
  - For TPU, doesn't matter (put `None`)
  - For Gemmini, `spike` (only optimizes instruction counts, not cycle counts) or `firesim`
  - For CUDA/KernelBench, doesn't matter (put `None`)
  - For CUDA/GPU MODE, `gpumode-local` or `gpumode-cli`
- `prob_type`: The problem type to use.
  - For Trainium, `trn-tutorial` or `trn-advanced`.
  - For TPU, `tpu`, `jaxbench-pallas`, `jaxbench-real`, `jaxbench-priority`, `jaxbench-tokamax`, or `jaxkernelbench`.
  - For Gemmini, `gemm`, `conv`, or `admm-multifunction`.
  - For CUDA/KernelBench, `kb-level1`, `kb-level2`, `kb-level3`, or `kb-level4`.
  - For CUDA/GPU MODE, `gpumode`.
- `prob_id`: The problem ID to use.

**Models**
- `models`: The list of models to use. Models are specified `"<provider>::<model>"`, for example `"openai::gpt-5.2"` or `"gcp::gemini-3-pro-preview"`. Currently supported endpoint providers are OpenAI (`openai`), Google Vertex AI (`gcp`), Anthropic (`anthropic`), AWS Bedrock (`aws`), and Together (`together`). Use provider `vllm` for local serving.
- `code_models`: The list of models to use for the implementation phase, if you would like to use a distinct set of models from planning. Can be set to `None` to use the same set of models.

**Search**
- `iterations`: The number of iterations to run.
- `search_strategy`: The search strategy to use. Currently only `beam` is supported.
- `num_plan_candidates`: Number of plans (strategies) generated per parent candidate per iteration. Default `4`.
- `num_code_candidates`: Number of code implementations generated per plan. Default `2`.
- `beam_size`: Number of candidates kept in the beam after each iteration. Default `4`.
- `dropout_menu_options`: Probability of dropping each strategy menu option from the prompt, encouraging diversity. Default `0.25`.
- `early_stop_iters`: Stop after N iterations without improvement (0 = disabled).
- `resume_from`: Path to a previous run's output directory. Loads the final candidates from that run as the starting beam (e.g., to optimize after a translation-only run).

**Code Generation**
- `use_edits`: If `True`, the LLM outputs structured JSON edits (`old_str`/`new_str` pairs) instead of rewriting the entire file. Generally more effective when code size is large. Defaults to `False`.
- `reimplement_failed`: Re-generate code for candidates that failed evaluation (only works on supported agents).

**Translation**
- `translate_iters`: Number of initial iterations that use translation strategies (converting code to the target representation) instead of optimization strategies. Defaults to `0` (no translation). Only works on supported agents. Built agents load strategies from `translate_menu.yaml`; see [Agent Builder docs](autocomp/agent_builder/README.md#translation-support).
- `translate_perf_threshold`: During translation iterations, candidates are kept if their score is within this factor of the best score (e.g., `1.2` means up to 20% worse).
- `translate_score`: If `True`, score translation candidates by code similarity to the original (how complete the translation is), not just latency. Defaults to `True`.
- `translate_drop_original`: If `True`, drop the original (untranslated) candidate from the beam after the last translation iteration. Defaults to `True`.

**Built Agent Options**
- `menu_strategy`: Set to `"one-shot"` to dynamically generate new strategies per candidate via an LLM call, or `None` for static menu only.
- `fine_grained_isa`: Enables two-level ISA filtering (section then subsection) to include only relevant ISA documentation in the prompt.
- `example_rate`: Per-example probability of including an LLM-selected code example in the planning prompt.

## 📁 Repository Structure

**`autocomp/`** - Core Autocomp code.
- `search/` - Search algorithm (`search.py`) and optimization infrastructure. `run_search.py` is the entry point.
- `agents/` - LLM agents for planning and code generation. Each hardware target has its own subdirectory (e.g., `gemmini/`, `trn/`, `cuda/`) with agent code and prompts.
- `agent_builder/` - Agent Builder pipeline for creating new hardware-specific agents from documentation sources. See [Agent Builder documentation](autocomp/agent_builder/README.md) for details.
- `backend/` - Eval backends for code evaluation. Each eval backend has its own subdirectory (e.g., `gemmini/`, `trn/`, `tpu/`, `kernelbench/`, `gpumode/`) with evaluation code and setup instructions. One hardware target can have multiple eval backends.
- `hw_config/` - Hardware configuration classes. Each hardware target has a config file (e.g., `cuda_config.py`, `gemmini_config.py`, `trn_config.py`, `tpu_config.py`).
- `common/` - Shared utilities (LLM interface, logging, etc.).
  - `llm_utils.py` - LLM interface. Modify this file if you want to add a new LLM provider.

**`sols/`** - Baseline code for benchmarks (organized by problem type).

**`tests/`** - Test cases corresponding to `sols/`.

**`examples/`** - Example optimization traces from Autocomp.

## 📜 Citation
```
@misc{hong2025autocomp,
      title={Autocomp: A Powerful and Portable Code Optimizer for Tensor Accelerators}, 
      author={Charles Hong and Sahil Bhatia and Alvin Cheung and Yakun Sophia Shao},
      year={2025},
      eprint={2505.18574},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2505.18574}, 
}
```

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
WANDB_MODE=disabled pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to add tests and the CI workflow.

## 📝 Changelog

**(1/22/2026)** Reorganized repo structure to make it easier to add a new hardware target.

**(1/8/2026)** Check out our latest [📝 blog post](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) on optimizing attention on Trainium!

**(11/18/2025)** Added documentation for adding a new hardware target ([ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md)), added the `examples` directory for example optimization traces, and published [📝 blog post 4](https://charleshong3.github.io/blog/autocomp_trainium_conv1d.html) about how we optimized conv1d on Trainium.

**(11/3/2025)** Added code/documentation for setting up Trainium.
Check out [📝 blog post 3](https://charleshong3.github.io/blog/autocomp_trainium.html) for more details.

**(9/22/2025)** Added code/documentation for setting up CUDA/KernelBench, plus code for RVV optimization. Check out [📝 blog post 2](https://charleshong3.github.io/blog/autocomp_update.html) for more details.

**(6/6/2025)** Initial code + [📝 blog post 1](https://charleshong3.github.io/blog/autocomp.html) release!
