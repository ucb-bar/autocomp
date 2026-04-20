"""Entry point for running Autocomp optimization.

Usage:
    python -m autocomp.search.run_search

Configure the parameters in the `main()` function below.
"""
import pathlib
import random

from autocomp.common import logger
from autocomp.search.search import (
    create_backend_and_agents,
    load_initial_code,
    BeamSearchStrategy,
    ExhaustiveSearchStrategy,
)
from autocomp.search.prob import Prob
from autocomp.hw_config import (
    CudaHardwareConfig,
    GemminiHardwareConfig,
    MetalHardwareConfig,
    SaturnHardwareConfig,
    TrnHardwareConfig,
    TpuHardwareConfig,
)


def main():
    # ------------------------------------------------------------------
    # Target & environment
    # ------------------------------------------------------------------
    backend_name = "metal"           # "gemmini", "trn", "tpu", "jaxbench", "kernelbench", "gpumode", "saturn", "xnnpack", "metal"
    agent_name = "built:metal-m2"    # "gemmini", "trn", "cuda", "saturn", "built:<name>", or path
    simulator = None                # "firesim"/"spike" for gemmini, saturn, and xnnpack; "gpumode-local"/"gpumode-cli" for gpumode
    hw_config = MetalHardwareConfig("M2", "4.0", "apple8", 8)
    # hw_config = TrnHardwareConfig("trn1.2xlarge")
    # hw_config = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
    # hw_config = CudaHardwareConfig("NVIDIA L40S", "2.5.0", "12.4")
    # hw_config = TpuHardwareConfig("v6e-1")

    prob_type = "metal-m2"               # see README.md or sols/ for available problems
    prob_id = 1

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    # Format: "provider::model" (openai, anthropic, together, aws, gcp, vllm)
    models = [
        "aws::us.anthropic.claude-opus-4-5-20251101-v1:0",
        "aws::zai.glm-5",
        "aws::minimax.minimax-m2.5",
        "aws::moonshotai.kimi-k2.5",
    ]
    code_models = None  # None = same as planning models

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    search_strategy = "beam"
    metric = "latency"
    iterations = 8
    num_plan_candidates = 4
    num_code_candidates = 2
    beam_size = 4
    dropout_menu_options = 0.25
    early_stop_iters = 0            # 0 = disabled
    early_stop_threshold = 1.0
    skip_planning = False           # True = bypass plan phase, generate code directly
    continue_from = ""

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    use_edits = False
    reimplement_failed = False

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    translate_iters = 0
    translate_perf_threshold = 15
    translate_drop_original = True
    translate_score = True

    # ------------------------------------------------------------------
    # Built-agent options
    # ------------------------------------------------------------------
    menu_strategy = "one-shot"      # None (static menu) or "one-shot"
    fine_grained_isa = True
    example_rate = 0.25

    # ------------------------------------------------------------------
    # Advanced / rarely changed
    # ------------------------------------------------------------------
    give_score_feedback = 1
    give_util_feedback = 0
    give_hw_feedback = 0
    include_ancestors = False
    plan_icl_examples = False
    code_icl_examples = False
    num_analyses = 0
    num_pairs_to_combine = 0
    num_gen_per_combine = 0
    trigger_exhaustive_threshold = 1
    trigger_exhaustive_iters = 20
    start_exhaustive_iters = 0
    prevent_duplicate_level = 0     # 0: same parent+plan, 1: same parent, 2: any shared ancestor
    random.seed(1111)

    # ------------------------------------------------------------------
    # Sanitize model names for filesystem
    # ------------------------------------------------------------------
    models = [m.replace("/", "_") for m in models]
    if code_models is not None:
        code_models = [m.replace("/", "_") for m in code_models]

    # ------------------------------------------------------------------
    # Build output directory & start logging
    # ------------------------------------------------------------------
    built_menu_strategy_enum = {None: 0, "one-shot": 1}
    clean_agent_name = pathlib.Path(agent_name).name if "/" in agent_name else agent_name
    output_str = f"{clean_agent_name}"
    output_str += f"_{prob_type}_{prob_id}_{search_strategy}_iters{iterations}"
    if simulator is not None:
        output_str += f"_{simulator}"
    hw_desc = hw_config.get_hw_description().replace(" ", "").replace("(", "_").replace(")", "").replace(",", "_")
    output_str += f"_{hw_desc}"
    for model in models:
        output_str += f"_{model[-20:]}"
    if code_models is not None:
        output_str += "_code"
        for model in code_models:
            output_str += f"_{model[-20:]}"
    if dropout_menu_options:
        output_str += f"_do{dropout_menu_options}"
    if search_strategy == "beam":
        if num_analyses:
            output_str += f"_an{num_analyses}"
        output_str += f"_p{num_plan_candidates}_c{num_code_candidates}_b{beam_size}"
    if translate_iters > 0:
        output_str += f"_tr{translate_iters}_{translate_perf_threshold}"
        if translate_drop_original:
            output_str += "_trdrop"
        if translate_score:
            output_str += "_tscore"
    if give_score_feedback:
        output_str += f"_score{give_score_feedback}"
    if give_util_feedback:
        output_str += f"_util{give_util_feedback}"
    if give_hw_feedback:
        output_str += f"_hwfb{give_hw_feedback}"
    if include_ancestors:
        output_str += "_anc1"
    if prevent_duplicate_level:
        output_str += f"_pd{prevent_duplicate_level}"
    if plan_icl_examples:
        output_str += "_picl1"
    if code_icl_examples:
        output_str += "_cicl1"
    if reimplement_failed:
        output_str += "_reimpl1"
    if early_stop_iters > 0:
        output_str += f"_es{early_stop_iters}_{early_stop_threshold}"
    if menu_strategy:
        output_str += f"_ms{built_menu_strategy_enum[menu_strategy]}"
    if fine_grained_isa:
        output_str += "_fgisa1"
    if example_rate > 0:
        output_str += f"_ex{example_rate}"
    if continue_from:
        output_str += "_continued"
    if use_edits:
        output_str += "_edits"
    if skip_planning:
        output_str += "_noplan"
    output_dir = pathlib.Path("output") / output_str
    output_dir.mkdir(parents=True, exist_ok=True)

    import autocomp.common.my_logging
    autocomp.common.my_logging.move_log(output_dir, tag="search")
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Initialize and run
    # ------------------------------------------------------------------
    prob = Prob(prob_type, prob_id)
    initial_code = load_initial_code(backend_name, prob)
    eval_backend, agent, code_agent = create_backend_and_agents(
        backend_name, agent_name, hw_config, prob, models, code_models,
        menu_strategy=menu_strategy, fine_grained_isa=fine_grained_isa,
        example_rate=example_rate, cache_dir=output_dir,
    )

    common_kwargs = dict(
        output_dir=output_dir, eval_backend=eval_backend, agent=agent,
        orig_code=initial_code, prob=prob, metric=metric, simulator=simulator,
        give_score_feedback=give_score_feedback,
        give_util_feedback=give_util_feedback,
        give_hw_feedback=give_hw_feedback,
        include_ancestors=include_ancestors,
        plan_icl_examples=plan_icl_examples,
        code_icl_examples=code_icl_examples,
        dropout_menu_options=dropout_menu_options,
        prevent_duplicate_level=prevent_duplicate_level,
        translate_iters=translate_iters,
        translate_perf_threshold=translate_perf_threshold,
        translate_drop_original=translate_drop_original,
        translate_score=translate_score,
        code_agent=code_agent,
        early_stop_iters=early_stop_iters,
        early_stop_threshold=early_stop_threshold,
        continue_from=continue_from,
        use_edits=use_edits,
    )

    if search_strategy == "exhaustive":
        optimizer = ExhaustiveSearchStrategy(**common_kwargs)
    elif search_strategy == "beam":
        optimizer = BeamSearchStrategy(
            **common_kwargs,
            num_analyses=num_analyses,
            num_plan_candidates=num_plan_candidates,
            num_code_candidates=num_code_candidates,
            beam_size=beam_size,
            num_pairs_to_combine=num_pairs_to_combine,
            num_gen_per_combine=num_gen_per_combine,
            trigger_exhaustive_threshold=trigger_exhaustive_threshold,
            trigger_exhaustive_iters=trigger_exhaustive_iters,
            start_exhaustive_iters=start_exhaustive_iters,
            reimplement_failed=reimplement_failed,
            skip_planning=skip_planning,
        )
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    optimizer.optimize(iterations)


if __name__ == "__main__":
    main()
