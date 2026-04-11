# Tapeout NPU Backend

This backend extends autocomp to target an NPU accelerator via the `npu_model` cycle-accurate simulator. Below is a summary of how the search and evaluation pipeline differs from the standard autocomp flow (e.g., Gemmini, TRN, CUDA).

## What's Different

### Evaluation Backend (`tapeout_eval.py`)

Instead of compiling C/CUDA or running on real hardware, the tapeout backend:

- **Executes NPU programs in a cycle-accurate simulator.** Each candidate is a `def test()` function that returns a list of ISA `Instruction` objects.
- **Uses a test-harness substitution pattern.** The problem defines a test harness (in `harnesses/tapeout/`) that owns inputs, memory layout, and golden output. The LLM only generates the `def test()` body, which gets inserted at the `# SUBSTITUTE HERE` / `# SUBSTITUTE END` markers.
- **Golden checking is problem-owned.** The harness, not the LLM, defines `golden_result`. The backend compares simulator DRAM output against this tensor (within rtol/atol of 1e-2).
- **Error feedback is captured at every stage.** Code cleaning failure, program load failure, simulation error, and golden mismatch all populate `result["stderr"]` so models can iterate on their mistakes.

### Search Modifications (`search.py`)

The core beam search loop has been extended with a **translation mode** controlled by `translate_iters`:

- **Translation vs. optimization phases.** The first `translate_iters` iterations run in translation mode; remaining iterations run normal optimization. Translation is for getting a *working* kernel from a seed (which may be empty or incorrect); optimization is for improving latency.
- **Incorrect seeds are allowed.** When `translate_iters > 0`, the search no longer raises an error if the initial code fails evaluation. This lets you start from an empty `def test()` stub.
- **Early exit on success.** During translation iterations, if *any* candidate passes simulation + golden check, the search stops immediately and saves the result. No need to exhaust all translation iterations.
- **Failed candidates carry forward with error feedback.** If all candidates fail in a translation iteration, the top `beam_size` failures (sorted by error length) become parents for the next iteration. The error message is injected into the prompt so the model can see what went wrong and iterate on its previous attempt, rather than starting from scratch.
- **Translation completeness scoring.** When `translate_score=True`, correct candidates are scored on how completely they translate the original code (vs. using shortcuts or partial implementations).

### Agent & Prompt Changes (`built_agent.py`)

- **Error feedback in prompts.** When a parent candidate has `score=inf` and a non-empty `stderr`, the prompt shows "This code failed with the following error: ..." instead of "The latency of this code was inf." This gives the model actionable feedback.
- **Test harness injection.** The test harness source is injected directly into both `translate` and `implement` prompts (inside `<test_harness>` tags) so the model can see the memory layout, input shapes, and expected output format.

## Agent Setup

The agent is pre-built and lives in `agent_builder/.built/tapeout/`. It follows the standard `BuiltLLMAgent` procedure:

- `agent_config.yaml` — model routing, ISA section selection config
- `isa_docs.md` — NPU ISA reference documentation
- `architecture.md` — NPU architecture overview
- `code_examples.md` — example NPU programs
- `optimization_menu.yaml` / `translate_menu.yaml` — optimization and translation strategy menus
- `rules.yaml` — generation rules and constraints

The ISA docs and rules include some **manually curated entries** based on evaluation feedback (e.g., common mistakes models make, clarifications on instruction semantics, memory layout conventions). These were added after observing failure patterns in early runs.

> **TBD:** We may be able to automate this curation in a post-run feedback loop — analyzing common failure modes across iterations and auto-updating the rules/ISA docs. This is not yet implemented.

## External Dependency

This backend requires the `npu_model` simulator repo at `third_party/npu_model`. It is not bundled in this repo — clone it separately:

```
git clone git@github.com:ucb-ee194-tapeout/npu_model.git third_party/npu_model
```

## Quick Start

```bash
# Make sure npu_model is in place
ls third_party/npu_model

# Run the search
uv run python autocomp/search/run_search.py
```

Configure problem, models, and search parameters in `run_search.py`.
