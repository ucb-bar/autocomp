"""Integration test: run a full BeamSearchStrategy.optimize() loop with dummy LLM + eval backend."""

import json
import os

os.environ.setdefault("WANDB_MODE", "disabled")

import pathlib

import pytest

from autocomp.agent_builder.built_agent import BuiltLLMAgent
from autocomp.agents.llm_ensemble import LLMEnsemble
from autocomp.common import REPO_ROOT
from autocomp.hw_config.trn_config import TrnHardwareConfig
from autocomp.search.search import BeamSearchStrategy


BUILT_AGENT_DIR = REPO_ROOT / "autocomp" / "agent_builder" / ".built" / "trn1-nki1"


@pytest.fixture
def built_agent(dummy_eval_backend):
    hw_config = TrnHardwareConfig("trn1.2xlarge")
    agent = BuiltLLMAgent("dummy::test-model", BUILT_AGENT_DIR, hw_config, dummy_eval_backend)
    return LLMEnsemble([agent])


def _make_strategy(output_dir, eval_backend, agent, prob, **overrides):
    defaults = dict(
        output_dir=output_dir,
        eval_backend=eval_backend,
        agent=agent,
        orig_code="def nki_kernel():\n    pass\n",
        prob=prob,
        metric="p99_latency",
        simulator="dummy",
        give_score_feedback=1.0,
        give_util_feedback=0.0,
        give_hw_feedback=0.0,
        include_ancestors=False,
        plan_icl_examples=False,
        code_icl_examples=False,
        num_analyses=0,
        num_plan_candidates=1,
        num_code_candidates=1,
        beam_size=1,
        num_pairs_to_combine=0,
        num_gen_per_combine=0,
        dropout_menu_options=1.0,
        trigger_exhaustive_threshold=1.0,
        trigger_exhaustive_iters=999,
        start_exhaustive_iters=0,
        prevent_duplicate_level=0,
        reimplement_failed=False,
        translate_iters=0,
        translate_perf_threshold=0.0,
        translate_drop_original=False,
        translate_score=False,
    )
    defaults.update(overrides)
    return BeamSearchStrategy(**defaults)


def test_beam_search_two_iterations(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Run two iterations and verify candidates, artifacts, and metrics."""
    strategy = _make_strategy(tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob)
    strategy.optimize(iterations=2)

    # --- Candidates are produced ---
    iter0 = strategy.repository.get_candidates(0)
    assert len(iter0) >= 1
    assert iter0[0].score == 1.0

    iter1 = strategy.repository.get_candidates(1)
    assert len(iter1) >= 1

    iter2 = strategy.repository.get_candidates(2)
    assert len(iter2) >= 1

    # --- Output directory artifacts ---
    for i in (1, 2):
        assert (tmp_output_dir / f"candidates-iter-{i}").is_dir()
        assert any((tmp_output_dir / f"candidates-iter-{i}").iterdir())
        assert (tmp_output_dir / f"generated-plans-iter-{i}").is_dir()
        assert (tmp_output_dir / f"generated-code-iter-{i}").is_dir()
        assert (tmp_output_dir / f"eval-results-iter-{i}").is_dir()
        assert any((tmp_output_dir / f"eval-results-iter-{i}").iterdir())

    # --- Candidate scores are valid (not inf) ---
    for i in (1, 2):
        for cand in strategy.repository.get_candidates(i):
            assert cand.score != float("inf"), f"Candidate in iter {i} should have a valid score"
            assert cand.score > 0

    # --- Candidate ancestry ---
    for cand in strategy.repository.get_candidates(1):
        assert cand.parent is not None, "Iter 1 candidates should have a parent"

    # --- Scores improve (DummyEvalBackend returns decreasing latencies) ---
    best_1 = min(c.score for c in strategy.repository.get_candidates(1))
    best_2 = min(c.score for c in strategy.repository.get_candidates(2))
    assert best_2 <= best_1, "Best score should not worsen (beam keeps best)"

    # --- Per-iteration metrics ---
    for i in (1, 2):
        metrics_path = tmp_output_dir / f"metrics-iter-{i}.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["iteration"] == i
        assert metrics["iteration_total_s"] > 0
        assert "plan_duration_s" in metrics
        assert "code_duration_s" in metrics
        assert "evaluation" in metrics
        assert metrics["evaluation"]["duration_s"] >= 0

    # --- Run-level aggregate metrics ---
    run_metrics_path = tmp_output_dir / "run_metrics.json"
    assert run_metrics_path.exists()
    rm = json.loads(run_metrics_path.read_text())
    assert rm["run_total_s"] > 0
    assert len(rm["iterations"]) == 2
    assert "total_input_tokens" in rm
    assert "total_output_tokens" in rm
    assert "total_llm_duration_s" in rm
    assert "total_eval_duration_s" in rm


def test_skip_planning(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Run two iterations in no-plan (direct) mode and verify artifacts."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        skip_planning=True,
    )
    strategy.optimize(iterations=2)

    for i in (1, 2):
        assert (tmp_output_dir / f"candidates-iter-{i}").is_dir()
        assert (tmp_output_dir / f"generated-code-iter-{i}").is_dir()
        assert not (tmp_output_dir / f"generated-plans-iter-{i}").exists()

    rm = json.loads((tmp_output_dir / "run_metrics.json").read_text())
    assert rm["run_total_s"] > 0
    assert len(rm["iterations"]) == 2
