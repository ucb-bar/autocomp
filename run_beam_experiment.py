"""Beam-size sweep experiment for trn-tutorial-nki1 benchmarks.

Varies beam_size in {1, 2, 4, 6, 8} across all 6 trn-tutorial-nki1 problems,
using open-source LLMs only (GLM-5, Kimi K2.5, DeepSeek V3.2).

Constant parameters (per CLAUDE.md):
    iterations=8, num_plan_candidates=4, num_code_candidates=2,
    dropout_menu_options=0.25, random.seed(1111)

Usage:
    python run_beam_experiment.py
    python run_beam_experiment.py --beam-sizes 1 2 4  # subset of beam sizes
    python run_beam_experiment.py --prob-ids 0 1 2    # subset of problems
"""
import argparse
import json
import pathlib
import random
import time
import traceback

from autocomp.common import logger
from autocomp.search.search import (
    create_backend_and_agents,
    load_initial_code,
    BeamSearchStrategy,
)
from autocomp.search.prob import Prob
from autocomp.hw_config import TrnHardwareConfig

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
BEAM_SIZES = [1, 2, 4, 6, 8]
PROB_TYPE = "trn-tutorial-nki1"
PROB_IDS = [3,4]

# Open-source models (via AWS Bedrock Converse API)
MODELS = [
    "aws::zai.glm-5",               # GLM-5
    "aws::moonshotai.kimi-k2.5",    # Kimi K2.5
    "aws::deepseek.v3.2",           # DeepSeek V3.2
]

# Fixed hyperparameters
ITERATIONS = 8
NUM_PLAN_CANDIDATES = 4
NUM_CODE_CANDIDATES = 2
DROPOUT_MENU_OPTIONS = 0.25
RANDOM_SEED = 1111

# Hardware / backend
BACKEND_NAME = "trn"
AGENT_NAME = "built:trn1-nki1"
HW_CONFIG = TrnHardwareConfig("trn1.2xlarge")

# Built-agent options (same as run_search.py defaults)
MENU_STRATEGY = "one-shot"
FINE_GRAINED_ISA = True
EXAMPLE_RATE = 0.25

# Search settings (same defaults as run_search.py)
METRIC = "latency"
GIVE_SCORE_FEEDBACK = 1
GIVE_UTIL_FEEDBACK = 0
GIVE_HW_FEEDBACK = 0
INCLUDE_ANCESTORS = False
PLAN_ICL_EXAMPLES = False
CODE_ICL_EXAMPLES = False
NUM_ANALYSES = 0
NUM_PAIRS_TO_COMBINE = 0
NUM_GEN_PER_COMBINE = 0
TRIGGER_EXHAUSTIVE_THRESHOLD = 1
TRIGGER_EXHAUSTIVE_ITERS = 20
START_EXHAUSTIVE_ITERS = 0
PREVENT_DUPLICATE_LEVEL = 0
EARLY_STOP_ITERS = 0
EARLY_STOP_THRESHOLD = 1.0
USE_EDITS = False
SKIP_PLANNING = False
TRANSLATE_ITERS = 0
TRANSLATE_PERF_THRESHOLD = 15
TRANSLATE_DROP_ORIGINAL = True
TRANSLATE_SCORE = True

OUTPUT_ROOT = pathlib.Path("output") / "beam_experiment"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_models(models):
    return [m.replace("/", "_") for m in models]


def _build_output_dir(beam_size: int, prob_id: int) -> pathlib.Path:
    """Construct a deterministic, human-readable output directory name."""
    model_suffix = "_".join(m.split("::")[-1][-20:] for m in _sanitize_models(MODELS))
    hw_desc = HW_CONFIG.get_hw_description().replace(" ", "").replace("(", "_").replace(")", "").replace(",", "_")
    name = (
        f"{AGENT_NAME}_{PROB_TYPE}_{prob_id}_beam_iters{ITERATIONS}"
        f"_{hw_desc}_{model_suffix}"
        f"_do{DROPOUT_MENU_OPTIONS}"
        f"_p{NUM_PLAN_CANDIDATES}_c{NUM_CODE_CANDIDATES}_b{beam_size}"
        f"_score{GIVE_SCORE_FEEDBACK}"
        f"_ms1_fgisa1_ex{EXAMPLE_RATE}"
    )
    return OUTPUT_ROOT / name


def _load_run_metrics(output_dir: pathlib.Path) -> dict:
    """Load run_metrics.json if it exists."""
    p = output_dir / "run_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _load_iter_latencies(output_dir: pathlib.Path, iterations: int) -> list:
    """Return list of best_score per iteration from metrics-iter-*.json files."""
    latencies = []
    for i in range(1, iterations + 1):
        p = output_dir / f"metrics-iter-{i}.json"
        if p.exists():
            try:
                with open(p) as f:
                    d = json.load(f)
                latencies.append(d.get("best_score"))
            except Exception:
                latencies.append(None)
    return latencies


def _collect_plan_diversity(output_dir: pathlib.Path, iterations: int) -> dict:
    """Count unique plan texts generated, grouped by iteration, as a diversity proxy."""
    diversity = {}
    total_plans = 0
    unique_plans = set()
    for i in range(1, iterations + 1):
        plan_dir = output_dir / f"generated-plans-iter-{i}"
        if not plan_dir.exists():
            continue
        iter_plans = []
        for plan_file in sorted(plan_dir.glob("*.json")):
            try:
                with open(plan_file) as f:
                    d = json.load(f)
                plan_text = d.get("plan", d.get("content", ""))
                iter_plans.append(plan_text)
                unique_plans.add(plan_text)
                total_plans += 1
            except Exception:
                pass
        diversity[f"iter_{i}"] = {
            "num_plans_generated": len(iter_plans),
        }
    diversity["total_plans_generated"] = total_plans
    diversity["total_unique_plans"] = len(unique_plans)
    diversity["diversity_ratio"] = (
        round(len(unique_plans) / total_plans, 4) if total_plans > 0 else None
    )
    return diversity


def run_single(beam_size: int, prob_id: int) -> dict:
    """Run a single (beam_size, prob_id) optimization and return metrics dict."""
    random.seed(RANDOM_SEED)

    models = _sanitize_models(MODELS)
    output_dir = _build_output_dir(beam_size, prob_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    import autocomp.common.my_logging
    autocomp.common.my_logging.move_log(output_dir, tag="search")
    logger.info(
        "=== Beam experiment: beam_size=%d  prob_id=%d  output=%s ===",
        beam_size, prob_id, output_dir,
    )

    prob = Prob(PROB_TYPE, prob_id)
    initial_code = load_initial_code(BACKEND_NAME, prob)
    eval_backend, agent, code_agent = create_backend_and_agents(
        BACKEND_NAME, AGENT_NAME, HW_CONFIG, prob, models, None,
        menu_strategy=MENU_STRATEGY,
        fine_grained_isa=FINE_GRAINED_ISA,
        example_rate=EXAMPLE_RATE,
        cache_dir=output_dir,
    )

    optimizer = BeamSearchStrategy(
        output_dir=output_dir,
        eval_backend=eval_backend,
        agent=agent,
        orig_code=initial_code,
        prob=prob,
        metric=METRIC,
        simulator=None,
        give_score_feedback=GIVE_SCORE_FEEDBACK,
        give_util_feedback=GIVE_UTIL_FEEDBACK,
        give_hw_feedback=GIVE_HW_FEEDBACK,
        include_ancestors=INCLUDE_ANCESTORS,
        plan_icl_examples=PLAN_ICL_EXAMPLES,
        code_icl_examples=CODE_ICL_EXAMPLES,
        num_analyses=NUM_ANALYSES,
        num_plan_candidates=NUM_PLAN_CANDIDATES,
        num_code_candidates=NUM_CODE_CANDIDATES,
        beam_size=beam_size,
        num_pairs_to_combine=NUM_PAIRS_TO_COMBINE,
        num_gen_per_combine=NUM_GEN_PER_COMBINE,
        dropout_menu_options=DROPOUT_MENU_OPTIONS,
        trigger_exhaustive_threshold=TRIGGER_EXHAUSTIVE_THRESHOLD,
        trigger_exhaustive_iters=TRIGGER_EXHAUSTIVE_ITERS,
        start_exhaustive_iters=START_EXHAUSTIVE_ITERS,
        prevent_duplicate_level=PREVENT_DUPLICATE_LEVEL,
        reimplement_failed=False,
        translate_iters=TRANSLATE_ITERS,
        translate_perf_threshold=TRANSLATE_PERF_THRESHOLD,
        translate_drop_original=TRANSLATE_DROP_ORIGINAL,
        translate_score=TRANSLATE_SCORE,
        code_agent=code_agent,
        early_stop_iters=EARLY_STOP_ITERS,
        early_stop_threshold=EARLY_STOP_THRESHOLD,
        continue_from="",
        use_edits=USE_EDITS,
        skip_planning=SKIP_PLANNING,
    )

    wall_t0 = time.perf_counter()
    optimizer.optimize(ITERATIONS)
    wall_elapsed = round(time.perf_counter() - wall_t0, 3)

    run_metrics = _load_run_metrics(output_dir)
    iter_latencies = _load_iter_latencies(output_dir, ITERATIONS)
    diversity = _collect_plan_diversity(output_dir, ITERATIONS)

    result = {
        "beam_size": beam_size,
        "prob_id": prob_id,
        "prob_type": PROB_TYPE,
        "models": MODELS,
        "output_dir": str(output_dir),
        "wall_elapsed_s": wall_elapsed,
        "best_score": run_metrics.get("best_score"),
        "total_input_tokens": run_metrics.get("total_input_tokens"),
        "total_output_tokens": run_metrics.get("total_output_tokens"),
        "total_llm_duration_s": run_metrics.get("total_llm_duration_s"),
        "total_eval_duration_s": run_metrics.get("total_eval_duration_s"),
        "run_total_s": run_metrics.get("run_total_s"),
        "iter_latencies": iter_latencies,
        "diversity": diversity,
    }
    return result


def save_summary(all_results: list, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved experiment summary to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Beam-size sweep experiment")
    parser.add_argument(
        "--beam-sizes", type=int, nargs="+", default=BEAM_SIZES,
        help="Beam sizes to sweep (default: 1 2 4 6 8)",
    )
    parser.add_argument(
        "--prob-ids", type=int, nargs="+", default=PROB_IDS,
        help="Problem IDs to run (default: 0 1 2 3 4 5)",
    )
    args = parser.parse_args()

    beam_sizes = args.beam_sizes
    prob_ids = args.prob_ids

    total_runs = len(beam_sizes) * len(prob_ids)
    logger.info(
        "Starting beam experiment: %d beam_sizes × %d problems = %d total runs",
        len(beam_sizes), len(prob_ids), total_runs,
    )
    logger.info("Beam sizes: %s", beam_sizes)
    logger.info("Problem IDs: %s  (prob_type=%s)", prob_ids, PROB_TYPE)
    logger.info("Models: %s", MODELS)

    summary_path = OUTPUT_ROOT / "experiment_summary.json"
    all_results = []

    # Load any prior results (allows resuming interrupted runs)
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                all_results = json.load(f)
            logger.info("Loaded %d prior results from %s", len(all_results), summary_path)
        except Exception:
            pass

    models_key = tuple(sorted(MODELS))
    completed = {
        (r["beam_size"], r["prob_id"])
        for r in all_results
        if tuple(sorted(r.get("models", [r.get("models", MODELS[0])]))) == models_key
    }

    run_num = 0
    for beam_size in beam_sizes:
        for prob_id in prob_ids:
            run_num += 1
            if (beam_size, prob_id) in completed:
                logger.info(
                    "[%d/%d] Skipping beam_size=%d prob_id=%d (already completed with same models)",
                    run_num, total_runs, beam_size, prob_id,
                )
                continue

            logger.info(
                "[%d/%d] Running beam_size=%d  prob_id=%d",
                run_num, total_runs, beam_size, prob_id,
            )
            try:
                result = run_single(beam_size, prob_id)
                result["status"] = "success"
            except Exception as e:
                logger.error(
                    "Run failed (beam_size=%d, prob_id=%d): %s\n%s",
                    beam_size, prob_id, e, traceback.format_exc(),
                )
                result = {
                    "beam_size": beam_size,
                    "prob_id": prob_id,
                    "prob_type": PROB_TYPE,
                    "status": "error",
                    "error": str(e),
                }

            all_results.append(result)
            save_summary(all_results, summary_path)

    logger.info("Experiment complete. %d runs saved to %s", len(all_results), summary_path)
    _print_summary_table(all_results)


def _print_summary_table(results: list):
    """Print a compact table of best scores by (prob_id, beam_size)."""
    if not results:
        return
    prob_ids = sorted(set(r["prob_id"] for r in results))
    beam_sizes = sorted(set(r["beam_size"] for r in results))

    # Index results
    idx = {(r["beam_size"], r["prob_id"]): r for r in results}

    header = f"{'prob_id':>8} | " + " | ".join(f"beam={b:>2}" for b in beam_sizes)
    print("\n" + "=" * len(header))
    print("Best latency scores (lower = better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for pid in prob_ids:
        row = f"{pid:>8} | "
        cells = []
        for b in beam_sizes:
            r = idx.get((b, pid))
            if r is None:
                cells.append(f"{'N/A':>8}")
            elif r.get("status") == "error":
                cells.append(f"{'ERR':>8}")
            else:
                score = r.get("best_score")
                cells.append(f"{score:>8.4f}" if score is not None else f"{'None':>8}")
        row += " | ".join(cells)
        print(row)
    print("=" * len(header))

    # Token cost summary
    print("\nTotal LLM tokens (input+output) by beam_size:")
    for b in beam_sizes:
        runs = [r for r in results if r["beam_size"] == b and r.get("status") == "success"]
        tok_in = sum(r.get("total_input_tokens") or 0 for r in runs)
        tok_out = sum(r.get("total_output_tokens") or 0 for r in runs)
        print(f"  beam={b}: input={tok_in:,}  output={tok_out:,}  total={tok_in+tok_out:,}")


if __name__ == "__main__":
    main()
