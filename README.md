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

**(1/8/2026)** Welcome to the code repository of **Autocomp**. Check out our latest [📝 blog post](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) on optimizing attention on Trainium!

**📚 Paper**: [**Autocomp: A Powerful and Portable Code Optimizer for Tensor Accelerators**](https://arxiv.org/abs/2505.18574)

**✏️ Authors**: [Charles Hong](https://charleshong3.github.io/), [Sahil Bhatia](https://x.com/sahilb17), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), and [Yakun Sophia Shao](https://people.eecs.berkeley.edu/~ysshao/) (UC Berkeley)

### What is Autocomp?

Autocomp is an LLM-driven code optimizer for tensor accelerators. Autocomp is designed to be portable and easy to use across a variety of hardware backends, and has already demonstrated strong performance on an industry accelerator ([AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)), an academic accelerator ([Gemmini](https://github.com/ucb-bar/gemmini)), NVIDIA GPUs, and even the RISC-V Vector Extension.

### How does Autocomp work?

Autocomp decomposes the optimization problem into a beam search, where each iteration is further divided into a planning phase and an implementation phase. Autocomp applies the user's domain knowledge, along with a variety of techniques to successfully explore the search space, in order to iteratively improve the code. For more details, see our [paper](https://arxiv.org/abs/2505.18574).

# ⚙️ Setup

## Backend Setup

Autocomp can currently optimize code for the following backends:
- Trainium ([trn_setup.md](autocomp/backend/trn_setup.md))
- Gemmini ([gemmini_setup.md](autocomp/backend/gemmini_setup.md))
- CUDA via KernelBench ([kb_setup.md](autocomp/backend/kb_setup.md))

Partially supported backends:
- RISC-V Vector (RVV) on Canaan Kendryte K230. See `k230` branch for code. As the implementation is very hacky, we do not currently recommend using this backend.

For instructions on adding a new backend, see [ADDING_A_BACKEND.md](autocomp/backend/ADDING_A_BACKEND.md).

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

For more details, see the [vLLM documentation](https://docs.vllm.ai/).

### LLM Endpoint Setup

#### OpenAI, Anthropic, Together

For these providers, define the appropriate environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`), or create the file `autocomp/common/openai_key.py` (or `anthropic_key.py`, `together_key.py`). The file should define the variable `key` as follows:

```python
key = "YOUR_OPENAI_API_KEY"
```

#### AWS Bedrock

To use AWS Bedrock, set the environment variables 
  `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or create the file `autocomp/common/aws_key.py` with the variables `aws_access_key` and `aws_secret_key` as follows:

```python
aws_access_key = "YOUR_AWS_ACCESS_KEY_ID"
aws_secret_key = "YOUR_AWS_SECRET_ACCESS_KEY"
```

Note that we currently only support Anthropic models on AWS Bedrock.

#### Gemini Endpoint Setup

To use Gemini via Google cloud, install the Google Cloud CLI as described at https://docs.cloud.google.com/sdk/docs/install-sdk#linux.

Run `gcloud auth application-default login` to enable the Google Cloud SDK.

Configure the location and/or project using the environment variables `GOOGLE_CLOUD_LOCATION` and `GOOGLE_CLOUD_PROJECT`, or in `autocomp/common/gcp_key.py`, as follows:

```python
location = "YOUR_GOOGLE_CLOUD_LOCATION"
project = "YOUR_GOOGLE_CLOUD_PROJECT"
```

## 🚀 Usage

`autocomp/search/search.py` is the entry point for running Autocomp optimization. Various parameters such as backend, models used, beam size, number of plans, number of code implementations, dropout, etc. can be configured here.

Notable parameters:
- `backend`: The hardware backend to use. Currently supported backends are `gemmini`, `trn`, and `cuda`.
- `models`: The list of models to use. Models are specified `"<provider>::<model>"`, for example `"openai::gpt-5.2"` or `"gcp::gemini-3-pro-preview"`. Currently supported endpoint providers are OpenAI (`openai`), Google Vertex AI (`gcp`), Anthropic (`anthropic`), AWS Bedrock (`aws`), and Together (`together`). Use provider `vllm` for local serving.
- `code_models`: The list of models to use for the implementation phase of prompting, if you would like to use a distinct set of models from planning. Can be set to `None` to use the same set of models.
- `simulator`: The evaluation method to use.
  - For Trainium, `trn`
  - For Gemmini, `spike` (only optimizes instruction counts, not cycle counts) or `firesim`
  - For CUDA, `kernelbench`
- `iterations`: The number of iterations to run.
- `search_strategy`: The search strategy to use. Currently only `beam` is supported.
- `prob_type`: The problem type to use.
  - For Trainium, `trn-tutorial` or `trn-advanced`.
  - For Gemmini, `gemm`, `conv`, or `admm-multifunction`.
  - For CUDA, `kb-level1`, `kb-level2`, `kb-level3`, or `kb-level4`.
- `prob_id`: The problem ID to use.

## 📁 Repository Structure

**`autocomp/`** - Core Autocomp code.
- `search/` - Core search and optimization infrastructure
  - `search.py` - Main search algorithm implementation. Implements the beam search described in the paper. Change search parameters within this file.
  - `llm_agent.py` - LLM agents for planning and code optimization. Implements the two prompt phases described in the paper. The optimization menu is defined within this file.
  - `llm_ensemble.py` - Wrapper around LLM agents that enables calls to be split between multiple agents.
  - `prob.py` - Wrapper for tests (parsed from the `tests/` directory) that edits the test file and appends LLM-generated code in order to test it.
  - `code_repo.py` - Abstraction for managing code candidates generated during optimization.
- `backend/` - Hardware evaluation utilities for different backends.
  - `hardware_backend.py` - Base class for hardware backends.
  - `gemmini_eval.py` - Hardware evaluation utilities for Gemmini. Must configure paths to Chipyard/FireSim/Gemmini here.
  - `trn_eval.py` - Hardware evaluation utilities for Trainium.
  - `kb_eval.py` - Hardware evaluation utilities for KernelBench. Must configure path to KernelBench here.
- `common/` - Shared utilities and helper functions
  - `llm_utils.py` - LLM interaction utilities. Implements the interface to the LLM providers listed above.
  - `my_logging.py` - Custom logging functionality.
  - `utils.py` - General utility functions.

**`prompts/`** - Contains various prompts imported by `autocomp/search/llm_agent.py`.
- `trn/` - Prompts and examples used for NKI (Trainium) optimization
  - `nki_isa_generator.py` - Generates the ISA string for the NKI ISA. If optimizing a new workload, configure the set of instructions to use here.
- `gemmini/` - Prompts and examples used for Gemmini code optimization

**`sols/`** - Contains baseline code for the benchmarks in the paper.
- `trn-tutorial/` - NKI (Trainium) unoptimized and optimized baseline code for the tutorial benchmarks in the paper.
- `trn-advanced/` - NKI (Trainium) unoptimized and optimized baseline code for the advanced benchmarks in the paper.
- `exo/` - Exo unoptimized and optimized baseline code for the Gemmini GEMM benchmarks in the paper. `sol{id}_exo_baseline.c` is the unoptimized code and is used by `autocomp/search/search.py` as the starting code fro optimization.
- `gemm/` - Additional Gemmini GEMM benchmarks used for schedule reuse. No hand-optimized code available.
- `exo-conv/` - Exo unoptimized and optimized baseline code for the Gemmini convolution benchmarks in the paper.
- `admm-multifunction/` - TinyMPC unoptimized and optimized baseline code. Only problem IDs 1 and 2 are used in the paper. Run with FP32 4x4 Gemmini.

**`tests/`** - Contains test cases corresponding to `sols/` above.

**`examples/`** - Contains examples of code optimized by Autocomp. Note that the generated code is specific to the input/output shapes used and may not be correct for other shapes.

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

**Update (11/18/2025)**: Added documentation for adding a new backend ([ADDING_A_BACKEND.md](autocomp/backend/ADDING_A_BACKEND.md)), added the `examples` directory for example optimization traces, and published [📝 blog post 4](https://charleshong3.github.io/blog/autocomp_trainium_conv1d.html) about how we optimized conv1d on Trainium.

**Update (11/3/2025)**: Added code/documentation for setting up Trainium backend.
Check out [📝 blog post 3](https://charleshong3.github.io/blog/autocomp_trainium.html) for more details.

**Update (9/22/2025)**: Added code/documentation for setting up CUDA/KernelBench backend, plus code for RVV optimization. Check out [📝 blog post 2](https://charleshong3.github.io/blog/autocomp_update.html) for more details.
