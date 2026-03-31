# Autocomp Trace Visualizer

A VS Code extension for visualizing [Autocomp](https://github.com/ucb-bar/autocomp) beam search optimization traces. Explore how LLM-driven code optimization progresses across iterations — from plan proposals to score improvements.

## Install

- **VS Code** — Search "Autocomp Trace Visualizer" in the Extensions sidebar, or install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=charleshong3.autocomp-visualizer), or .
- **Cursor** — Search "autocomp-visualizer" in the Extensions sidebar, or install from [Open VSX](https://open-vsx.org/extension/charleshong3/autocomp-visualizer).

## Features

- **Score Progression Chart** — Interactive SVG chart plotting individual candidate lineage traces across iterations. Click any point to inspect the candidate.
- **Beam Tree** — Visual tree showing beam search ancestry with score-based coloring. Selecting a node highlights its ancestors and descendants.
- **Code Diff** — Compare any candidate against its ancestors using VS Code's native diff viewer. Choose which ancestor to diff against from a dropdown.
- **Candidate Summary** — View optimization plans, model attributions (plan and code models), and score improvements for each candidate.
- **AI Plan Summaries** — Generate concise one-line summaries of optimization plans using an LLM (requires the `autocomp` Python package with LLM API credentials).
- **Run Config** — Expandable panel showing all run metadata (beam size, models, metric, etc.) — automatically populated from `run_metadata.json`.

## Getting Started

### Prerequisites

- Python 3.10+ with the [`autocomp`](https://github.com/ucb-bar/autocomp) package installed (for data ingestion and plan summarization)
- An Autocomp output directory from one or more completed optimization run(s)

### Usage

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **"Autocomp: Ingest & Open Traces"** and select your Autocomp `output/` directory
3. The extension ingests the run data and opens the visualizer

If you've already ingested data, use **"Autocomp: Open Traces"** and select the directory containing `runs.json`.

### Plan Summarization

Click the **"Summarize Plans with AI"** button in the run detail view to generate short summaries for each optimization plan. You'll be prompted for an LLM model in `provider::model` format:

- `openai::gpt-5.4-mini`
- `aws::us.anthropic.claude-haiku-4-5-20251001-v1:0`
- `gcp::gemini-3-flash-preview`

This requires the `autocomp` Python package and appropriate API credentials.

## Development

```bash
cd visualizer
npm install
npm run build        # Build webview + extension
npm run watch        # Watch mode for both
```

Press **F5** in VS Code to launch the Extension Development Host.

## Packaging

```bash
npm run package      # Builds and creates .vsix
```
