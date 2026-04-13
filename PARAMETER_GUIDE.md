# Parameter Selection Guide

Practical guidance for choosing Autocomp search parameters. See the [main README](README.md) for parameter definitions.

## `iterations` and `translate_iters`

`iterations` is the total number of iterations. The first `translate_iters` of those use translation strategies (converting code to the target representation, e.g., PyTorch → NKI); the remaining `iterations - translate_iters` iterations use optimization strategies. For simple kernels (single matmul, elementwise ops), use `translate_iters=2` with `iterations=6` (2 translation + 4 optimization). For complex kernels (fused multi-op pipelines), use `translate_iters=3` with `iterations=7` or `8`. Without translation, set `translate_iters=0` and `iterations=4`–`6`.

When `translate_score=True`, translation ends early if any candidate reaches a perfect translation score (10.0). The remaining iterations automatically switch to optimization mode, so setting a generous `translate_iters` (e.g., 4) is safe — simple kernels that translate in one iteration will not waste budget on redundant translation.

## `translate_perf_threshold`

This controls how much slower a translated candidate can be versus its parent and still survive. Use `15` (the default) for translation iterations where the initial translation is expected to be much slower than the source (e.g., an unoptimized NKI kernel vs. PyTorch). Do not set this too low — early translations are often significantly slower than compiler-generated code and would be incorrectly pruned. For non-translation runs this parameter has no effect.

## `beam_size`, `num_plan_candidates`, `num_code_candidates`

Each iteration generates `beam_size × num_plan_candidates × num_code_candidates` total candidates for evaluation. The default (`beam_size=4`, `num_plan_candidates=4`, `num_code_candidates=2`) produces 32 candidates per iteration, which balances exploration with evaluation cost. `num_plan_candidates` should be >= the number of `models`, because plans are divided evenly across models — if you have 4 models but only 3 plan candidates, one model gets no work. Similarly, `num_code_candidates` should be >= the number of `code_models` (if set) for the same reason. Increase `beam_size` to `6` if you want broader exploration, but this scales evaluation cost linearly.

## `models`

Use 3–4 diverse models for best results. Model diversity matters more than count — different models propose different optimization strategies. Using a single model tends to converge prematurely. Recommended models by provider (last updated April 13, 2026):

| Provider | Model string | Notes |
|----------|-------------|-------|
| OpenAI | `"openai::gpt-5.4"` | Flagship model, strong at complex reasoning and coding. 1M context. |
| OpenAI | `"openai::gpt-5.4-mini"` | Strongest mini model for coding and high-volume workloads. 400k context. |
| Anthropic (direct) | `"anthropic::claude-opus-4-6"` | Strongest Anthropic model (4.5 is similar). 1M context. |
| Anthropic (direct) | `"anthropic::claude-sonnet-4-6"` | Good balance of speed and intelligence. 1M context. |
| AWS Bedrock (Claude) | `"aws::us.anthropic.claude-opus-4-6-v1"` | Opus 4.6 on Bedrock; uses Anthropic SDK adapter. |
| AWS Bedrock (Claude) | `"aws::us.anthropic.claude-opus-4-5-20251101-v1:0"` | Opus 4.5 on Bedrock (still available, similar to 4.6). |
| AWS Bedrock (open) | `"aws::zai.glm-5"` | GLM-5; strong at code, available via Converse API. |
| AWS Bedrock (open) | `"aws::moonshotai.kimi-k2.5"` | Kimi K2.5; adds diversity. |
| AWS Bedrock (open) | `"aws::deepseek.v3.2"` | DeepSeek V3.2; adds diversity. |
| AWS Bedrock (open) | `"aws::minimax.minimax-m2.5"` | MiniMax M2.5; adds diversity. |
| Google | `"gcp::gemini-3.1-pro-preview"` | Frontier Gemini model, strong reasoning. 1M context. |
| Google | `"gcp::gemini-3-flash-preview"` | Cheaper model adds diversity. 1M context. |

An example 4-model mix: `"openai::gpt-5.4"`, `"aws::us.anthropic.claude-opus-4-6-v1"`, `"aws::zai.glm-5"`, `"aws::moonshotai.kimi-k2.5"`. This combines two frontier models with two capable open models for maximum strategy diversity.

## `dropout_menu_options`

The value is the probability of **keeping** each strategy menu option — `0.25` means roughly 75% of options are dropped per candidate. With the built agent, the optimization menu is relatively fixed (~40 strategies), so `dropout_menu_options` controls how many strategies each candidate sees. At `0.25`, each candidate sees ~10 of ~40 options, forcing diverse exploration. At `1.0`, all options are kept and the LLM tends to repeatedly pick the same obvious strategies. Values below `0.1` leave too few options and may starve the LLM of useful strategies. Adjust based on the menu size for your target — a smaller menu needs a higher keep rate.

## `use_edits`

When `False` (default), the LLM rewrites the entire code file from scratch each iteration. When `True`, it outputs structured JSON edits (`old_str`/`new_str` pairs) that are applied to the existing code. Use `True` when the code is large (> ~200 lines) and optimizations are localized — edits avoid the LLM accidentally dropping or mangling unrelated code. Use `False` (default) for smaller kernels or when the optimization requires restructuring the entire file (e.g., translation iterations, algorithm rewrites).

## `reimplement_failed`

When `True`, failed candidates (those that produced errors during evaluation) get a second chance: the LLM sees the error output and attempts to fix the code. This is most effective when error messages are informative (e.g., compilation errors with line numbers, shape mismatch details) — vague or truncated errors give the LLM little to work with. Each reimplementation adds one LLM call per failed candidate, so the cost scales with failure rate. Improving initial correctness rates (via better ISA docs, rules, or examples) is generally more efficient than relying on reimplementation.

## Troubleshooting: Low correctness rates

If most candidates fail evaluation, there are three paths in order of impact: (1) check whether the ISA documentation is missing coverage for the operations the LLM is trying to use — if the LLM doesn't have correct API docs, it will hallucinate function signatures and arguments, (2) inspect the common errors from failed candidates and add rules to the agent's strategy menu or system prompt to prevent them (e.g., adding a constraint like "do not use `nl.matmul` — use `nisa.nc_matmul` instead"), or (3) increase `num_plan_candidates` or `num_code_candidates` to generate more attempts per iteration. Path 1 is the most impactful fix when errors are consistently about wrong API usage; path 2 is most helpful for miscellaneous pitfalls not covered in the ISA docs; path 3 is a quick band-aid fix that increases cost linearly.
