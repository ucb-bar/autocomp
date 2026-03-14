# Autocomp Trace Visualizer

VS Code extension for visualizing Autocomp beam search optimization traces.

## Features

- **Score Progression Chart**: Interactive SVG chart showing individual candidate lineage traces across iterations
- **Beam Tree**: Visual tree of the beam search with ancestor/descendant highlighting
- **Native Code Diff**: Click any candidate to open a VS Code diff tab comparing it to its parent
- **Plan Summaries**: View optimization plan details and model attributions

## Usage

1. Run your Autocomp optimization to produce an output directory
2. Ingest the data: use **"Autocomp: Ingest & Open Traces"** from the command palette (requires Python 3 and the autocomp repo)
3. Or if you already have ingested JSON data, use **"Autocomp: Open Traces"** and select the directory containing `runs.json`

## Development

```bash
npm install
npm run build        # Build webview + extension
npm run watch        # Watch mode for both
```

Press **F5** in VS Code to launch the Extension Development Host for testing.

## Packaging

```bash
npm run build
npx vsce package
```

This produces a `.vsix` file you can share or publish to the VS Code Marketplace.
