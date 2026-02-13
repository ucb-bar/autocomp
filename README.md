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

**(1/22/2026)** Reorganized repo structure to make it easier to add a new hardware target.

**(1/8/2026)** Check out our latest [📝 blog post](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) on optimizing attention on Trainium!

**📚 Paper**: [**Autocomp: A Powerful and Portable Code Optimizer for Tensor Accelerators**](https://arxiv.org/abs/2505.18574)

**✏️ Authors**: [Charles Hong](https://charleshong3.github.io/), [Sahil Bhatia](https://x.com/sahilb17), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), and [Yakun Sophia Shao](https://people.eecs.berkeley.edu/~ysshao/) (UC Berkeley)

### What is Autocomp?

Autocomp is an LLM-driven code optimizer for tensor accelerators. Autocomp is designed to be portable and easy to use across a variety of hardware targets, and has already demonstrated strong performance on an industry accelerator ([AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)), an academic accelerator ([Gemmini](https://github.com/ucb-bar/gemmini)), NVIDIA GPUs, and even the RISC-V Vector Extension.

### How does Autocomp work?

Autocomp decomposes the optimization problem into a beam search, where each iteration is further divided into a planning phase and an implementation phase. Autocomp applies the user's domain knowledge, along with a variety of techniques to successfully explore the search space, in order to iteratively improve the code. For more details, see our [paper](https://arxiv.org/abs/2505.18574).

# ⚙️ Setup

## Hardware Target Setup

Autocomp can currently optimize code for the following hardware targets:
- Trainium ([trn_setup.md](autocomp/backend/trn/trn_setup.md))
- Gemmini ([gemmini_setup.md](autocomp/backend/gemmini/gemmini_setup.md))
- CUDA via KernelBench ([kb_setup.md](autocomp/backend/kernelbench/kb_setup.md))
- CUDA via GPU MODE ([gpumode_setup.md](autocomp/backend/gpumode/gpumode_setup.md))

Partially supported hardware targets:
- RISC-V Vector (RVV) on Canaan Kendryte K230. See `k230` branch for code. As the implementation is very hacky, we do not currently recommend using this hardware target.

For instructions on adding a new hardware target, see [ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md).

## LLM Setup

Autocomp supports both local and remote endpoint LLM inference. For local inference, we support vLLM's [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/). For endpoint inference, we support a variety of providers (see below).

### Local Inference with vLLM

1. **Install and launch vLLM:**
   ```bash
   pip install vllm
   vllm serve --model Qwen/Qwen3-8B --port 8000 -tp <number of GPUs>
   ```

2. **Configure Autocomp:**
   Set `models`/`code_models` in `search.py`:
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

| Provider | Environment Variable / Key Name | Provider Name in `search.py`
|----------|--------------------------------|--------------------------------|
| OpenAI | `OPENAI_API_KEY` | `openai`
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic`
| Together | `TOGETHER_API_KEY` | `together`
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | `aws`
| Google Cloud | `GOOGLE_CLOUD_LOCATION`, `GOOGLE_CLOUD_PROJECT` | `gcp`

**Example `autocomp/common/keys.py`:**

```python
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
TOGETHER_API_KEY = "..."
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_CLOUD_PROJECT = "my-project"
```

Keys can be omitted if not needed. On startup, Autocomp logs which keys are available.

#### Gemini Setup

To use Gemini via Google Cloud, install the Google Cloud CLI as described at https://docs.cloud.google.com/sdk/docs/install-sdk#linux.

Run `gcloud auth application-default login` to enable the Google Cloud SDK.

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

`autocomp/search/search.py` is the entry point for running Autocomp optimization. Various parameters such as hardware target, models used, beam size, number of plans, number of code implementations, dropout, etc. can be configured here.

Notable parameters:
- `backend_name`: The hardware target to use. Currently supported values are `gemmini`, `trn`, `kernelbench`, and `gpumode`.
- `agent_name`: The LLM agent type to use. Defaults based on `backend_name`. Currently supported agents are `gemmini`, `trn`, and `cuda` (used for both `kernelbench` and `gpumode`).
- `hw_config`: A hardware configuration object describing the target hardware. Examples:
  - `TrnHardwareConfig("trn1.2xlarge")`
  - `GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)`
  - `CudaHardwareConfig("NVIDIA L40S", "2.5.0", "12.4")`
- `models`: The list of models to use. Models are specified `"<provider>::<model>"`, for example `"openai::gpt-5.2"` or `"gcp::gemini-3-pro-preview"`. Currently supported endpoint providers are OpenAI (`openai`), Google Vertex AI (`gcp`), Anthropic (`anthropic`), AWS Bedrock (`aws`), and Together (`together`). Use provider `vllm` for local serving.
- `code_models`: The list of models to use for the implementation phase of prompting, if you would like to use a distinct set of models from planning. Can be set to `None` to use the same set of models.
- `simulator`: The evaluation method to use, if multiple are supported.
  - For Trainium, doesn't matter (put `None`)
  - For Gemmini, `spike` (only optimizes instruction counts, not cycle counts) or `firesim`
  - For CUDA/KernelBench, doesn't matter (put `None`)
  - For CUDA/GPU MODE, `gpumode-local` or `gpumode-cli`
- `iterations`: The number of iterations to run.
- `search_strategy`: The search strategy to use. Currently only `beam` is supported.
- `prob_type`: The problem type to use.
  - For Trainium, `trn-tutorial` or `trn-advanced`.
  - For Gemmini, `gemm`, `conv`, or `admm-multifunction`.
  - For CUDA/KernelBench, `kb-level1`, `kb-level2`, `kb-level3`, or `kb-level4`.
  - For CUDA/GPU MODE, `gpumode`.
- `prob_id`: The problem ID to use.

## 📁 Repository Structure

**`autocomp/`** - Core Autocomp code.
- `search/` - Search algorithm (`search.py`) and optimization infrastructure.
- `agents/` - LLM agents for planning and code generation. Each hardware target has its own subdirectory (e.g., `gemmini/`, `trn/`, `cuda/`) with agent code and prompts.
- `backend/` - Eval backends for code evaluation. Each eval backend has its own subdirectory (e.g., `gemmini/`, `trn/`, `kernelbench/`, `gpumode/`) with evaluation code and setup instructions. One hardware target can have multiple eval backends.
- `hw_config/` - Hardware configuration classes. Each hardware target has a config file (e.g., `cuda_config.py`, `gemmini_config.py`, `trn_config.py`).
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

## 📝 Changelog

**(11/18/2025)** Added documentation for adding a new hardware target ([ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md)), added the `examples` directory for example optimization traces, and published [📝 blog post 4](https://charleshong3.github.io/blog/autocomp_trainium_conv1d.html) about how we optimized conv1d on Trainium.

**(11/3/2025)** Added code/documentation for setting up Trainium.
Check out [📝 blog post 3](https://charleshong3.github.io/blog/autocomp_trainium.html) for more details.

**(9/22/2025)** Added code/documentation for setting up CUDA/KernelBench, plus code for RVV optimization. Check out [📝 blog post 2](https://charleshong3.github.io/blog/autocomp_update.html) for more details.

**(6/6/2025)** Initial code + [📝 blog post 1](https://charleshong3.github.io/blog/autocomp.html) release!
