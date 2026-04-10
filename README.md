# Autocomp

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/autocomp_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="img/autocomp_logo.svg">
    <img alt="Autocomp" src="img/autocomp_logo.svg" width="420">
  </picture>
  <br>
  <strong>Optimize any AI kernel, anywhere.</strong>
</p>

<p align="center">
| <a href="https://arxiv.org/abs/2505.18574"><b>Paper</b></a> | <a href="https://charleshong3.github.io/blog/autocomp.html"><b>Blog</b></a> | <a href="https://marketplace.visualstudio.com/items?itemName=charleshong3.autocomp-visualizer"><b>VS Code Extension</b></a> |
</p>

**Autocomp** is an extensible, portable framework for LLM-driven kernel optimization across tensor accelerators. Point it at a kernel, pick your hardware target, and Autocomp speeds it up, automatically.

It already delivers strong results across **[AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)**, **[Google TPU](https://cloud.google.com/tpu)**, **[NVIDIA GPUs](https://charleshong3.github.io/blog/autocomp_update.html)**, **[Gemmini](https://github.com/ucb-bar/gemmini)**, and **[RISC-V Vector Processors](https://saturn-vectors.org/)**. Need a new target? The **[Agent Builder](autocomp/agent_builder/README.md)** can spin up a hardware-specific optimization agent from your docs in minutes.

<p align="center">
<a href="https://arxiv.org/abs/2505.18574"><b>📚 Read the paper</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;<b>✏️ Authors:</b> <a href="https://charleshong3.github.io/">Charles Hong</a>, <a href="https://x.com/sahilb17">Sahil Bhatia</a>, <a href="https://people.eecs.berkeley.edu/~akcheung/">Alvin Cheung</a>, <a href="https://people.eecs.berkeley.edu/~ysshao/">Yakun Sophia Shao</a> (UC Berkeley)
</p>

## 🚀 Quick Start

Autocomp's workflow is:

1. Pick your hardware target:
   - Choose an optimization agent (or build your own with the Agent Builder).
   - Set up an evaluation backend.
2. Configure one or more LLMs.
3. Edit `autocomp/search/run_search.py` with your settings.
4. Run search.

For example, a Trainium run might look like this:

```python
# autocomp/search/run_search.py
backend_name = "trn"
agent_name = "built:trn1-nki1"
hw_config = TrnHardwareConfig("trn1.2xlarge")
prob_type = "trn-tutorial-nki1"
prob_id = 2
models = ["openai::gpt-5.4"]
```

Then run:

```bash
python -m autocomp.search.run_search
```

Keep reading for more on picking your hardware target, setting up your backend, configuring LLM providers, and tuning the search.

# ⚙️ Setup

## Hardware Targets

Each hardware target requires two things: an **optimization agent** that knows how to optimize code for that target, and an **evaluation backend** — the toolchain that compiles and benchmarks code on it. You also provide a **hardware config** (`hw_config`) that describes your specific hardware instance (e.g., `TrnHardwareConfig("trn1.2xlarge")`). The table below shows the supported targets and the agents/backends available for each.

| Hardware target | Optimization agent(s) | Evaluation backend(s) |
|---|---|---|
| AWS Trainium | `built:trn1-nki1` (Trainium 1, NKI v1)<br>`built:trn2-nki1` (Trainium 2, NKI v1)<br>`built:trn2-nki2` (Trainium 2, NKI v2) | `trn` ([trn_setup.md](autocomp/backend/trn/trn_setup.md)) |
| Google TPU | `built:tpu-v6e` (TPU v6e) | `tpu` ([tpu_setup.md](autocomp/backend/tpu/tpu_setup.md)) |
| Gemmini | `gemmini` | `gemmini` ([gemmini_setup.md](autocomp/backend/gemmini/gemmini_setup.md)) |
| NVIDIA GPU | `cuda` | `kernelbench` ([kb_setup.md](autocomp/backend/kernelbench/kb_setup.md))<br>`gpumode` ([gpumode_setup.md](autocomp/backend/gpumode/gpumode_setup.md)) |
| Saturn (RVV) | `built:saturn-rvv` | `saturn` ([saturn_setup.md](autocomp/backend/saturn/saturn_setup.md))<br>`xnnpack` ([xnnpack_setup.md](autocomp/backend/xnnpack/xnnpack_setup.md)) |

Partially supported hardware targets:
- RISC-V Vector (RVV) on Canaan Kendryte K230. See `k230` branch for code. As the implementation is very hacky, we do not currently recommend using this hardware target.

For instructions on adding full codebase support for a new hardware target (eval backend, config class, etc.), see [ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md).

### 🧠 Optimization Agents

Optimization agents decide what transformations to try and how to implement them. In `run_search.py`, this is controlled by `agent_name`. Each agent is designed for a specific hardware target — see the table above for the right agent for each target. We recommend using the Agent Builder as the fastest way to set up a complete agent from your hardware's documentation.

#### 🏗️ Agent Builder

Want to create a new agent? The **[Agent Builder](autocomp/agent_builder/README.md)** automatically generates hardware-specific optimization agents from documentation sources such as local directories, PDFs, and webpages. Built agents are stored in `autocomp/agent_builder/.built/` and selected with `agent_name = "built:<name>"`. Legacy handcrafted agents in `autocomp/agents/` (e.g., `gemmini`, `cuda`) are also available for some targets.

```bash
pip install "autocomp[agent-builder]"

python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --source-dir path/to/docs \
    --agent-scope "Optimizing kernels for MyAccelerator using the XYZ programming interface."
```

For detailed usage, CLI options, Python API, and output format, see the [Agent Builder documentation](autocomp/agent_builder/README.md).

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

**Option 2: Google AI Studio.** Get an API key from [Google AI Studio](https://aistudio.google.com/apikey) and set `GOOGLE_API_KEY`.

If both Vertex AI credentials and `GOOGLE_API_KEY` are set, Vertex AI is used.

#### AWS Bedrock

Anthropic (Claude) models on Bedrock use the native Anthropic SDK adapter. All other Bedrock models (e.g., GLM, DeepSeek, Kimi) are supported via the Bedrock Converse API. Any model available in your Bedrock region can be used by passing its Bedrock model ID:

```python
models = [
    "aws::us.anthropic.claude-opus-4-5-20251101-v1:0",  # Claude (Anthropic adapter)
    "aws::zai.glm-5",                                   # GLM 5
]
```

By default the `us-west-2` region is used. Set the `AWS_REGION` environment variable (or add it to `keys.py`) to override.

## 🚀 Usage

`autocomp/search/run_search.py` is the entry point for running Autocomp optimization.

```bash
python -m autocomp.search.run_search
```

The most important parameters are:

**Hardware Target**
- `hw_config`: A hardware configuration object describing the target hardware. Examples:
  - `TrnHardwareConfig("trn1.2xlarge")`
  - `TpuHardwareConfig("v6e-1")`
  - `GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)`
  - `CudaHardwareConfig("NVIDIA L40S", "2.5.0", "12.4")`
  - `SaturnHardwareConfig(vlen=512, dlen=256)`

**Evaluation Backend**
- `backend_name`: The evaluation backend to use. Currently supported values are `trn`, `tpu`, `gemmini`, `kernelbench`, `gpumode`, `saturn`, and `xnnpack`.
- `simulator`: The evaluation method to use, if the backend supports multiple. For all others, put `None`.
  - For Gemmini, `spike` (only optimizes instruction counts, not cycle counts) or `firesim`
  - For Saturn/XNNPACK, `spike` or `firesim`
  - For CUDA/GPU MODE, `gpumode-local` or `gpumode-cli`

**Benchmark**
- `prob_type`: The problem type to use.
  - For Trainium, `trn-tutorial-nki1`, `trn-tutorial-nki2`, `trn-advanced-nki1`, or `trn-advanced-nki2`.
  - For TPU, `tpu`, `jaxbench-pallas`, `jaxbench-real`, `jaxbench-priority`, `jaxbench-tokamax`, or `jaxkernelbench`.
  - For Gemmini, `gemm`, `conv`, or `admm-multifunction`.
  - For CUDA/KernelBench, `kb-level1`, `kb-level2`, `kb-level3`, or `kb-level4`.
  - For CUDA/GPU MODE, `gpumode`.
  - For Saturn, `rvv-f32` or `rvv-qs8`.
  - For XNNPACK, `xnnpack-f32`.
- `prob_id`: The problem ID to use.

**Optimization Agent**
- `agent_name`: The optimization agent to use. See the [table above](#hardware-targets) for the right agent for each target.

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
- `skip_planning`: If `True`, skip the separate planning phase and generate optimized code in a single LLM call. The model is still prompted to reason about its approach before outputting code. Defaults to `False`.
- `continue_from`: Path to a previous run's output directory. Loads the final candidates from that run as the starting beam (e.g., to optimize after a translation-only run).

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

## 🔍 Trace Visualizer (VS Code Extension)

The [Autocomp Trace Visualizer](https://marketplace.visualstudio.com/items?itemName=charleshong3.autocomp-visualizer) is a VS Code extension for exploring optimization runs interactively. After a run completes, use it to understand what strategies worked, how scores improved, and where the search spent its time. See the [Trace Visualizer documentation](visualizer/README.md) for install instructions and features.

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

**`harnesses/`** - Test harnesses and context files corresponding to `sols/`.

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

**(4/9/2026)** Added [Saturn RVV](https://saturn-vectors.org/) as a new hardware target.

**(4/6/2026)** Renamed `tests/` to `harnesses/` and solution entry point from `test()` to `solution()` for clarity. Improved agent builder logging.

**(4/3/2026)** Added run metrics (runtime and tokens) and updated Trace Visualizer to be self-contained.

**(3/25/2026)** Added support for structured-output code edits in the code implementation phase.

**(3/17/2026)** Added preliminary TPU support and enhanced Autocomp's code translation capabilities.

**(3/13/2026)** Added the **Agent Builder** for automatically creating hardware-specific LLM agents from documentation sources.

**(1/22/2026)** Reorganized repo structure to make it easier to add a new hardware target.

**(1/8/2026)** Check out our latest [📝 blog post](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) on optimizing attention on Trainium!

**(11/18/2025)** Added documentation for adding a new hardware target ([ADDING_HARDWARE_SUPPORT.md](ADDING_HARDWARE_SUPPORT.md)), added the `examples` directory for example optimization traces, and published [📝 blog post 4](https://charleshong3.github.io/blog/autocomp_trainium_conv1d.html) about how we optimized conv1d on Trainium.

**(11/3/2025)** Added code/documentation for setting up Trainium.
Check out [📝 blog post 3](https://charleshong3.github.io/blog/autocomp_trainium.html) for more details.

**(9/22/2025)** Added code/documentation for setting up CUDA/KernelBench, plus code for RVV optimization. Check out [📝 blog post 2](https://charleshong3.github.io/blog/autocomp_update.html) for more details.

**(6/6/2025)** Initial code + [📝 blog post 1](https://charleshong3.github.io/blog/autocomp.html) release!
