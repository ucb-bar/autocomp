# Parameter Selection Guide

Practical guidance for choosing Autocomp search parameters. See the [main README](README.md) for parameter definitions.

## `iterations` and `translate_iters`

`iterations` is the total number of iterations. The first `translate_iters` of those use translation strategies; the remaining use optimization strategies. Translation may take multiple (3–5) iterations if the kernel is complex and the LLM translates it incrementally. For challenging kernels, 8–15 total iterations is typical.

When `translate_score=True`, translation ends early once **all** beam candidates reach a translation score >= 9.0. This ensures no partially-translated code enters the optimization phase. The remaining iterations automatically switch to optimization mode, so setting a generous `translate_iters` is safe — simple kernels that translate in one iteration will not waste budget on redundant translation.

**Convergence**: Set `early_stop_iters` to automatically stop the search after N iterations without improvement. If the search converges and you want to push further, change the search conditions (swap/add models, adjust `dropout_menu_options`, add documentation or rules) rather than just adding more iterations. Convergence speed depends on `beam_size` — at `beam_size=1` or `2`, the best score is typically locked in by iteration 3–5 and additional iterations rarely help; at `beam_size=6` or `8`, runs are often still improving at iteration 8, so a larger budget is warranted. For `beam_size` ≤ 2, 5–6 iterations is usually sufficient. For `beam_size` ≥ 6, use the full 8 or more.

## `translate_perf_threshold`

This controls how much slower a translated candidate can be versus its parent and still survive. Use `15` (the default) for translation iterations where the initial translation is expected to be much slower than the source (e.g., an unoptimized NKI kernel vs. PyTorch). Do not set this too low — early translations are often significantly slower than compiler-generated code and would be incorrectly pruned. For non-translation runs this parameter has no effect.

## `beam_size`, `num_plan_candidates`, `num_code_candidates`

Each iteration generates `beam_size × num_plan_candidates × num_code_candidates` total candidates for evaluation. The default (`beam_size=4`, `num_plan_candidates=4`, `num_code_candidates=2`) produces 32 candidates per iteration, which balances exploration with evaluation cost. `num_plan_candidates` should be >= the number of `models`, because plans are divided evenly across models — if you have 4 models but only 3 plan candidates, one model gets no work. Similarly, `num_code_candidates` should be >= the number of `code_models` (if set) for the same reason.

**`beam_size=4` is the recommended default for most kernels.** A controlled study across 6 NKI kernels (RMSNorm, LayerNorm, Large MatMul, Mamba SSM, Flash Attention, SD Attention) at beam sizes {1, 2, 4, 6, 8} showed that beam 4 captures the large majority of the improvement available at beam 8, at significantly lower cost. Notably, larger beam is not monotonically better — beam 8 occasionally regresses relative to beam 4–6, likely because the wider beam explores noisier strategies that fail to compile or produce incorrect results. `beam_size=1` still finds meaningful improvements over baseline for all kernels and is a reasonable first pass when cost is a concern.

The main exception is kernels where multiple fundamentally different algorithmic approaches exist. In the Large MatMul case, beam 1 plateaued at a 13% latency reduction while beam 2 discovered a PSUM-resident accumulation strategy and achieved a 2.5× additional improvement — a discrete algorithmic breakthrough that single-beam search could not escape to. If you suspect the kernel has multiple viable optimization paths (e.g., competing memory access patterns or loop orderings), prefer `beam_size` ≥ 2 to allow that diversity.

Increasing `beam_size` to `6` or `8` gives diminishing returns for most kernels but may help when the problem has large optimization headroom (> 2× gap between baseline and target) and you are willing to pay the evaluation cost, which scales roughly linearly with beam size.

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
