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
    --description "Optimizing kernels for MyAccelerator using the XYZ programming interface."

# Use the built agent in search.py:
#   agent_name = "built:my_accelerator"
```

You can mix multiple source types in a single build:

```bash
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --source-dir path/to/docs \
    --source-file path/to/manual.pdf \
    --source-url https://docs.example.com/api \
    --description "Optimizing kernels for MyAccelerator using the XYZ programming interface."
```

## CLI Reference

### Source inputs

| Flag | Description |
|------|-------------|
| `--source-dir` | Path to a local directory of code/docs. Text files are read directly; PDFs are automatically extracted. |
| `--source-file` | Path to a single file -- PDF or text. Can be repeated. |
| `--source-url` | URL to crawl. Same-domain links are followed. Can be repeated. |
| `--max-depth` | Max link-following depth for `--source-url` (default: 2) |
| `--max-pages` | Max pages to fetch per `--source-url` (default: 250) |

### Key options

**`--description`** is the most important flag after the source inputs. It is prepended to every LLM prompt and strongly influences which documents are considered relevant, which APIs are extracted, and what optimization strategies are generated. Be specific about:

- What level of code the agent optimizes (kernels, operators, full models)
- The programming interface (NKI, CUDA, HLO, etc.)
- What's out of scope (deployment, serving, distributed training)

Example: `--description "Optimizing NKI kernel code on AWS Trainium 1. The agent rewrites single-kernel source code for better performance. Model-level concerns like sharding, serving, and distributed training are out of scope."`

Without a description, the pipeline processes all documents without scope filtering, which can dilute the output with irrelevant content.

**`--model`** and **`--light-model`** control which LLMs are used. The main model handles synthesis and reduce steps. The light model handles high-volume parallel work (content routing, pre-filtering, ISA boundary detection). Using a cheaper light model significantly reduces build cost with minimal quality loss.

**`--context-budget`** sets the max characters (not tokens) of source content per LLM call. The default of 150,000 (~37-50K tokens) is conservative for 200K-token context windows. Increase if your model supports a larger window; decrease if you're hitting rate limits.

| Flag | Default | Description |
|------|---------|-------------|
| `--description` | `""` | Agent scope context (see above) |
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

**Re-run a single component** -- re-synthesize one component without rebuilding everything:

```bash
python -m autocomp.agent_builder.run_agent_builder \
    --agent-name my_accelerator \
    --inspect autocomp/agent_builder/.built/my_accelerator \
    --rerun rules --source-dir path/to/docs
```

Valid components: `rules`, `optimization_menu`, `isa`, `architecture`, `examples`.

## Output

A built agent produces the following files in `.built/<agent_name>/`:

| File | Description |
|------|-------------|
| `agent_config.yaml` | Build metadata (model, sources, timestamps) |
| `architecture.md` | Hardware architecture summary |
| `isa_docs.md` | API/instruction-set reference |
| `optimization_menu.yaml` | List of optimization strategies |
| `rules.yaml` | Programming constraints (general, planning, coding) |
| `code_examples.md` | Annotated code examples (not yet used at runtime; reserved for future ICL support) |

All output files are human-editable. After a build, you can manually refine any component and it will be used as-is by the runtime agent.

A reference example is available at `.built/trn-nki1/` (built with Agent Builder from the AWS Trainium NKI [documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/index.html)).

## How It Works

The build pipeline has three stages: **ingest** (load and index sources), **synthesize** (LLM-based extraction), and **assemble** (write config files).

The synthesizer first **routes** each document into component buckets (`isa`, `architecture`, `optimization`, `rules`, `examples`) using one LLM call per document. The `--description` flag drives scope filtering at this stage.

Each component is then synthesized using a **map-reduce** pattern: documents are processed in parallel (map), then the results are merged and deduplicated (reduce). Documents larger than the context budget are automatically split on paragraph boundaries, so the pipeline scales to arbitrarily large documentation sets.

### Optimization menu

The optimization menu is the list of strategies the agent considers when planning. It has two layers:

- **Static (build time):** Generic defaults (loop tiling, reduce data movement, etc.) plus hardware-specific strategies extracted from your docs. You can edit `optimization_menu.yaml` after building.
- **Dynamic (runtime):** When `menu_strategy="one-shot"`, the agent generates workload-specific strategies by analyzing the current code candidate. These are cached and appended to the static menu, so the agent adapts to the specific problem.

## Python API

```python
from autocomp.agent_builder import AgentBuilder

builder = AgentBuilder(
    llm_model="aws::us.anthropic.claude-opus-4-6-v1",
    light_llm_model="aws::us.anthropic.claude-haiku-4-5-20251001-v1:0",
    description="Optimizing NKI kernels on AWS Trainium.",
)
builder.add_source("directory", path="/path/to/docs")
builder.add_source("webpage", url="https://docs.example.com", max_depth=2)

config_dir = builder.build(
    agent_name="my_accelerator",
    output_dir="autocomp/agent_builder/.built",
)
```
