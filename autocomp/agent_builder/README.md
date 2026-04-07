# Agent Builder

The Agent Builder creates hardware-specific LLM agents from documentation sources. Point it at your docs, code, PDFs, or API webpages, and it produces a set of human-editable config files that define a fully functional Autocomp agent.

## Quick Start

```bash
# Install optional dependencies
pip install "autocomp[agent-builder]"

# Build an agent from a local docs directory:
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --source-dir path/to/docs \
    --agent-scope "Optimizing kernels for MyAccelerator using the XYZ programming interface."

# Use the built agent in run_search.py:
#   agent_name = "built:my_accelerator"
```

You can mix multiple source types in a single build:

```bash
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --source-dir path/to/docs \
    --source-file path/to/manual.pdf \
    --source-url https://docs.example.com/api \
    --agent-scope "Optimizing kernels for MyAccelerator using the XYZ programming interface."
```

## CLI Reference

### Source inputs

| Flag | Description |
|------|-------------|
| `--source-dir` | Path to a local directory of code/docs. Text files are read directly; PDFs are automatically extracted. |
| `--source-file` | Path to a single file -- PDF or text. Can be repeated. |
| `--source-url` | URL to crawl. Only links under the same path prefix are followed (see tip below). Can be repeated. |
| `--max-depth` | Max link-following depth for `--source-url` (default: 2) |
| `--max-pages` | Max pages to fetch per `--source-url` (default: 250) |

> **Tip — File size limit:** Individual text files larger than 512 KB are skipped during `--source-dir` ingestion (a warning is logged). If you have a large reference file, either split it into smaller pieces or provide it directly via `--source-file` (which has no size limit, though very large files will be chunked by the synthesizer's context budget).

> **Tip — URL scoping:** The crawler only follows links whose path starts with the parent directory of the URL you provide. For example, `--source-url https://docs.example.com/en/v2.0/api/index.html` crawls pages under `/en/v2.0/api/` and won't follow links to `/en/latest/` or `/en/v3.0/`. If you need content from multiple subtrees, provide a separate `--source-url` for each:
>
> ```bash
> --source-url https://docs.example.com/en/v2.0/api/index.html \
> --source-url https://docs.example.com/en/v2.0/guides/index.html
> ```

### Key options

**`--agent-scope`** is the most important flag after the source inputs. It is prepended to every LLM prompt and strongly influences which documents are considered relevant, which APIs are extracted, and what optimization strategies are generated. Be specific about:

- What level of code the agent optimizes (kernels, operators, full models)
- The target hardware and programming interface (NKI, CUDA, HLO, etc.)
- What's out of scope (deployment, serving, distributed training)

Example: `--agent-scope "Optimizing NKI kernel code on AWS Trainium 1. The agent rewrites single-kernel source code for better performance. Model-level concerns like sharding, serving, and distributed training are out of scope."`

Without an agent scope, the pipeline processes all documents without scope filtering, which can dilute the output with irrelevant content.

**`--model`** and **`--light-model`** control which LLMs are used. The main model handles synthesis and reduce steps. The light model handles high-volume parallel work (content routing, pre-filtering, ISA boundary detection). Using a cheaper light model significantly reduces build cost with minimal quality loss.

**`--context-budget`** sets the max characters (not tokens) of source content per LLM call. The default of 150,000 (~37-50K tokens) is conservative for 200K-token context windows. Increase if your model supports a larger window; decrease if you're hitting rate limits.

| Flag | Default | Description |
|------|---------|-------------|
| `--agent-scope` | `""` | Agent scope definition (see above) |
| `--model` | `aws::us.anthropic.claude-opus-4-6-v1` | Main LLM for synthesis and reduce steps |
| `--light-model` | `aws::us.anthropic.claude-haiku-4-5-20251001-v1:0` | Cheaper LLM for routing, filtering, and extraction |
| `--context-budget` | `150000` | Max characters (not tokens) per LLM call. ~3-4 chars per token |
| `--output-dir` | `.built/` | Base output directory |

### Other commands

**Dry run** -- test ingestion without making any LLM calls, to verify your sources are loaded correctly:

```bash
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator --source-dir path/to/docs --dry-run
```

**Inspect** -- print a diagnostic summary of a built agent (strategy counts, rule counts, ISA entries, etc.):

```bash
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator --inspect autocomp/agent_builder/.built/my_accelerator
```

**Re-run components** -- re-synthesize one or more components without rebuilding everything:

```bash
# Re-run a single component
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --inspect autocomp/agent_builder/.built/my_accelerator \
    --rerun rules --source-dir path/to/docs

# Re-run multiple components (ingests sources once, synthesizes each in sequence)
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --inspect autocomp/agent_builder/.built/my_accelerator \
    --rerun rules optimization_menu --source-dir path/to/docs
```

Valid components: `rules`, `optimization_menu`, `translate_menu`, `isa`, `architecture`, `examples`.

## Output

A built agent produces the following files in `.built/<agent_name>/`:

| File | Description |
|------|-------------|
| `agent_config.yaml` | Build metadata (model, sources, timestamps) |
| `architecture.md` | Hardware architecture summary |
| `isa_docs.md` | API/instruction-set reference |
| `optimization_menu.yaml` | List of optimization strategies |
| `rules.yaml` | Programming constraints (general, planning, coding) |
| `code_examples.md` | Annotated code examples, stochastically included during planning as reference patterns |
| `translate_menu.yaml` | *(optional, user-created)* Translation strategies for `translate_iters` — see [Translation support](#translation-support) |

All output files are human-editable. After a build, you can manually refine any component and it will be used as-is by the runtime agent.

A reference example is available at `.built/trn1-nki1/` (auto-generated with Agent Builder from the AWS Trainium NKI [documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/index.html)). Additional pre-built agents:

- `.built/trn2-nki1/` — Trainium 2 with NKI v1 APIs (from [v2.26.1 docs](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/general/nki/index.html))
- `.built/trn2-nki2/` — Trainium 2 with NKI v2 APIs (from [latest docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html))
- `.built/tpu-v6e/` — Google TPU v6e with JAX Pallas (from [JAX Pallas docs](https://docs.jax.dev/en/latest/pallas/index.html) and [Cloud TPU docs](https://docs.cloud.google.com/tpu/docs/))

## How It Works

The build pipeline has three stages: **ingest** (load and index sources), **synthesize** (LLM-based extraction), and **assemble** (write config files).

The synthesizer first runs a **pre-filter** that removes clearly irrelevant documents via parallel yes/no LLM prompts (driven by `--agent-scope`), then **routes** each remaining document into component buckets (`isa`, `architecture`, `optimization`, `rules`, `examples`) using one LLM call per document.

Each component is then synthesized using a **map-reduce** pattern: documents are processed in parallel (map), then the results are merged and deduplicated (reduce). Documents larger than the context budget are automatically split on paragraph boundaries, so the pipeline scales to arbitrarily large documentation sets.

### ISA filtering

Built agents can have large ISA docs. At runtime, the agent selects only the relevant sections for the current problem using a two-level LLM filter with parallel per-item yes/no prompts:

1. **Level 1:** Each `##` section is independently evaluated for relevance to the problem and code context. The prompt includes the section preamble and a summary of its subsections.
2. **Level 2** (`fine_grained_isa=True`): Within selected sections that have `###` subsections, each subsection is independently evaluated. The prompt includes the API signature and a content summary. Sections without subsections are included in full after passing L1.

Both levels are cached per-problem so the filtering cost is paid once.

### Code examples

At runtime, the agent stochastically selects relevant code examples from `code_examples.md` using the same per-item yes/no prompt approach. Selected examples are included as reference patterns at the top of the planning prompt. The inclusion rate is controlled by `example_rate`.

### Optimization menu

The optimization menu is the list of strategies the agent considers when planning. It has two layers:

- **Static (build time):** Generic defaults (loop tiling, reduce data movement, etc.) plus hardware-specific strategies extracted from your docs. You can edit `optimization_menu.yaml` after building.
- **Dynamic (runtime):** When `menu_strategy="one-shot"`, the agent generates workload-specific strategies by analyzing the current code candidate. These are cached and appended to the static menu, so the agent adapts to the specific problem.

## Customizing a Built Agent

All output files are plain YAML or Markdown, so you can customize a built agent without writing any code:

- Edit `optimization_menu.yaml` to add domain-specific strategies or remove irrelevant ones.
- Edit `rules.yaml` to add constraints you've discovered through experimentation.
- Edit `isa_docs.md` to add missing API entries or remove noisy ones.
- Use `--rerun <component>` to re-synthesize a single component from updated sources.

You can also copy an existing built agent directory, modify the files, and use it as a new agent (`agent_name = "built:my_custom_agent"`).

For deeper changes to agent behavior (e.g., prompt structure, ISA selection logic, or how the optimization menu is used during planning), subclass `BuiltLLMAgent` and add a new `elif` branch in `create_backend_and_agents()` in `search.py` to wire it up.

### Translation support

Translation lets the agent convert code from one representation to another (e.g., PyTorch → target intrinsics). When `translate_iters > 0` in `run_search.py`, the first `translate_iters` iterations use a translation menu instead of the optimization menu, with a relaxed performance threshold (`translate_perf_threshold`, default 1.2×) for keeping candidates.

To configure:

1. Set `translate_iters` to a positive value in `run_search.py` (e.g., `translate_iters = 2`).
2. Optionally create `translate_menu.yaml` in the agent's config directory:

```yaml
strategies:
  - "convert high-level framework calls into TargetISA intrinsics"
  - "move non-TargetISA operations into a TargetISA kernel"
  - "fuse multiple TargetISA kernels into a single kernel"
```

If `translate_menu.yaml` is absent, a generic default (`"convert high-level code to target kernel code"`) is used.

## Python API

```python
from autocomp.agent_builder import AgentBuilder

builder = AgentBuilder(
    llm_model="aws::us.anthropic.claude-opus-4-6-v1",
    light_llm_model="aws::us.anthropic.claude-haiku-4-5-20251001-v1:0",
    agent_scope="Optimizing NKI kernels on AWS Trainium.",
)
builder.add_source("directory", path="/path/to/docs")
builder.add_source("webpage", url="https://docs.example.com", max_depth=2)

config_dir = builder.build(
    agent_name="my_accelerator",
    output_dir="autocomp/agent_builder/.built",
)
```
