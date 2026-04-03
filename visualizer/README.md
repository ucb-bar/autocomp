# Autocomp Trace Visualizer

A VS Code extension for visualizing [Autocomp](https://github.com/ucb-bar/autocomp) beam search optimization traces. Explore how LLM-driven code optimization progresses across iterations — from plan proposals to score improvements.

## Install

- **VS Code** — Search "Autocomp Trace Visualizer" in the Extensions sidebar, or install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=charleshong3.autocomp-visualizer).
- **Cursor** — Search "autocomp-visualizer" in the Extensions sidebar, or install from [Open VSX](https://open-vsx.org/extension/charleshong3/autocomp-visualizer).

## Features

- **Run List** — Browse all optimization runs in an output directory. Each card shows the raw directory name, problem badge, beam size, iteration count, model list, and speedup. Sorted by speedup.
- **Score Progression Chart** — Interactive SVG chart plotting individual candidate lineage traces across iterations. Click any point to inspect the candidate.
- **Beam Tree** — Visual tree showing beam search ancestry with score-based coloring. Selecting a node highlights its ancestors and descendants.
- **Code Diff** — Compare any candidate against its ancestors using VS Code's native diff viewer. Choose which ancestor to diff against from a dropdown.
- **Candidate Summary** — View optimization plans, model attributions (plan and code models), and score improvements for each candidate.
- **Metrics Panel** — Collapsible panel with three levels of progressive disclosure:
  - **Run summary** — Total wall-clock time, aggregate LLM time, evaluation time, and total input/output token counts.
  - **Iteration rows** — Collapsible per-iteration summaries showing total, planning, coding, and eval durations. Expand to see phase details grouped under "Planning" and "Coding" wall-clock headers.
  - **Phase tables** — Per-phase, per-model breakdowns of call counts, token usage, average time per call, and slowest individual call. Phases are displayed in execution order: Context Selection, Menu Generation, Plan Generation, Code Generation.
- **Run Config** — Expandable panel showing all run metadata (beam size, models, metric, etc.) — automatically populated from `run_metadata.json`.
- **Plan Summarization** — Summarize optimization plans using an LLM. Supports OpenAI, Anthropic, AWS Bedrock (uses instance credentials on EC2), and Google Gemini. Configure via the gear icon next to the Summarize button. API keys are stored securely in VS Code SecretStorage.

## Getting Started

### Prerequisites

- An Autocomp output directory from one or more completed optimization run(s)

### Usage

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **"Autocomp: Open Trace Visualizer"**
3. The extension opens a settings page where you can:
   - **Browse** for your Autocomp output directory
   - Configure the LLM provider for plan summarization
4. Click **Ingest & Open Traces** to process the data and view runs

The extension remembers your last output directory across sessions.

### Plan Summarization

Click **Summarize Plans** in the run detail view to generate short summaries for each optimization plan. Use the gear icon to open the settings page and configure the provider and model:

- **OpenAI** — `gpt-5.4-mini`
- **Anthropic** — `claude-haiku-4-5-20251001`
- **AWS Bedrock** — `us.anthropic.claude-haiku-4-5-20251001-v1:0` (uses EC2 instance credentials automatically)
- **Google Gemini** — `gemini-3-flash-preview`

API keys are stored securely in VS Code SecretStorage. Bedrock does not require a key when running on EC2.

## Development

### Quick Start

```bash
cd visualizer
npm install
npm run build        # Build webview + extension
npm run test         # Run tests
npm run watch        # Watch mode for both
```

Press **F5** in VS Code to launch the Extension Development Host.

### Packaging

```bash
npm run package      # Builds and creates .vsix
```

### Architecture

```
visualizer/
├── src/
│   ├── extension/
│   │   ├── extension.ts      # Extension entry point (commands, webview, settings, ingestion)
│   │   ├── ingest.ts         # Data ingestion
│   │   ├── ingest.test.ts    # Ingestion tests (uses real fixture data)
│   │   └── summarize.ts      # Multi-provider LLM plan summarization
│   └── webview/
│       ├── App.tsx            # React UI (charts, beam tree, metrics panel)
│       └── lib/
│           ├── types.ts       # Data interfaces
│           └── format.ts      # Formatting utilities
├── dist/                      # Build output
├── package.json               # Extension manifest and build scripts
└── vite.config.ts             # Webview bundler config
```

**`extension.ts`** — Registers commands that open the visualizer panel. The panel opens to a settings page where the user can browse for an output directory and configure summarization. Handles ingestion (via `ingestDir` messages), settings persistence (provider/model in globalState, API keys in SecretStorage, last output dir), and plan summarization orchestration.

**`summarize.ts`** — Multi-provider LLM completion for plan summarization. Supports OpenAI, Anthropic, AWS Bedrock (uses instance credentials when no key is provided), and Google Gemini. Runs requests sequentially with progress reporting.

**`ingest.ts`** — Reads Autocomp output directories and produces JSON for the visualizer:
- **Python `repr()` parser** — Candidate files are serialized as Python `repr()` of `CodeCandidate` objects. A recursive-descent parser handles `CodeCandidate(...)` calls with keyword arguments, triple-quoted strings, `None`/`True`/`False`, numbers, lists, tuples, and dicts.
- **`ingestRun(runDir)`** — Ingests a single run: loads candidates, eval results, metrics; assigns stable IDs; detects carry-forwards; computes scores and speedups.
- **`ingestOutputDir(outputDir, outDir)`** — Ingests all runs in an output directory; writes per-run JSON files and a `runs.json` index.

**`App.tsx`** — React app bundled with Vite. Opens to a settings page by default. Renders run list (with raw directory names), score progression chart, beam tree, collapsible metrics panel with progressive disclosure, and code diffs. Navigates between settings and trace viewer via a gear icon.

### Data Flow

```
Autocomp output/           ingest.ts              .visualizer-data/
┌──────────────────┐      ┌──────────────┐      ┌─────────────────┐
│ candidates-iter-N/│─────▶│              │─────▶│ run_xyz.json    │
│ eval-results-iter-N/│    │ ingestRun()  │      │ runs.json       │
│ metrics-iter-N.json│     │              │      └─────────────────┘
│ run_metrics.json  │      └──────────────┘              │
│ run_metadata.json │                              postMessage
└──────────────────┘                                     │
                                                         ▼
                                                ┌─────────────────┐
                                                │  App.tsx (React) │
                                                │  webview panel   │
                                                └─────────────────┘
```

### Candidate File Format

Candidate files (`candidate_N.txt`) contain Python `repr()` output of `CodeCandidate` objects:

```
CodeCandidate(parent=CodeCandidate(parent=None,
plan='''Use vectorized operations...''',
code='''import numpy as np
...''',
score=1.234,
translation_score=None,
hw_feedback=[],
plan_gen_model='openai::gpt-5.4-mini',
code_gen_model='openai::gpt-5.4-mini',
stdout=None,
stderr=None),
plan='''Apply loop tiling...''',
code='''import numpy as np
...''',
score=0.987,
translation_score=None,
hw_feedback=[],
plan_gen_model='aws::us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='aws::us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout=None,
stderr=None)
```

The `parent` field is recursive, forming the full ancestry chain.

### Metrics Data

Autocomp writes optional metrics files alongside candidates:

- **`metrics-iter-N.json`** — Per-iteration: wall-clock durations for planning and coding phases, LLM token counts and call durations by phase and model (including average and slowest individual call times), evaluation time, and total iteration time.
- **`run_metrics.json`** — Aggregate: total run time, total LLM time, total eval time, total input/output tokens.

Missing fields render as "—" in the UI, so older runs without metrics display correctly.

### Adding New Features

**New data fields**: Add to `types.ts` (use optional fields for backward compatibility), update `ingest.ts` to extract them, and render in `App.tsx`.

**New commands**: Register in `extension.ts` `activate()` and add to `package.json` `contributes.commands`.
