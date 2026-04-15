"""Integration test: run a full BeamSearchStrategy.optimize() loop with dummy LLM + eval backend."""

import json
import os

os.environ.setdefault("WANDB_MODE", "disabled")

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


def _assert_basic_outputs(strategy, tmp_output_dir, iterations, expect_plans=True):
    """Shared assertions for candidate production, artifacts, and metrics."""
    for i in range(1, iterations + 1):
        cands = strategy.repository.get_candidates(i)
        assert len(cands) >= 1, f"Iter {i}: expected at least 1 candidate"
        for cand in cands:
            assert cand.score is not None
            assert cand.score != float("inf"), f"Iter {i}: candidate should be correct"
            assert cand.score > 0

        assert (tmp_output_dir / f"candidates-iter-{i}").is_dir()
        assert any((tmp_output_dir / f"candidates-iter-{i}").iterdir())
        assert (tmp_output_dir / f"generated-code-iter-{i}").is_dir()
        assert (tmp_output_dir / f"eval-results-iter-{i}").is_dir()

        if expect_plans:
            assert (tmp_output_dir / f"generated-plans-iter-{i}").is_dir()
        else:
            assert not (tmp_output_dir / f"generated-plans-iter-{i}").exists()

    rm = json.loads((tmp_output_dir / "run_metrics.json").read_text())
    assert rm["run_total_s"] > 0
    assert len(rm["iterations"]) == iterations


# ---------------------------------------------------------------------------
# Basic optimization (no translation)
# ---------------------------------------------------------------------------

def test_beam_search_two_iterations(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Run two iterations and verify candidates, artifacts, and metrics."""
    strategy = _make_strategy(tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob)
    strategy.optimize(iterations=2)

    _assert_basic_outputs(strategy, tmp_output_dir, 2)

    iter0 = strategy.repository.get_candidates(0)
    assert iter0[0].score == 1.0

    for cand in strategy.repository.get_candidates(1):
        assert cand.parent is not None

    best_1 = min(c.score for c in strategy.repository.get_candidates(1))
    best_2 = min(c.score for c in strategy.repository.get_candidates(2))
    assert best_2 <= best_1

    for i in (1, 2):
        metrics = json.loads((tmp_output_dir / f"metrics-iter-{i}.json").read_text())
        assert metrics["iteration"] == i
        assert metrics["iteration_total_s"] > 0
        assert "plan_duration_s" in metrics
        assert "code_duration_s" in metrics
        assert "evaluation" in metrics

    rm = json.loads((tmp_output_dir / "run_metrics.json").read_text())
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
    _assert_basic_outputs(strategy, tmp_output_dir, 2, expect_plans=False)


# ---------------------------------------------------------------------------
# Edit mode (use_edits=True)
# ---------------------------------------------------------------------------

def test_use_edits(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Plan-then-edit mode produces valid candidates."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        use_edits=True,
    )
    strategy.optimize(iterations=2)
    _assert_basic_outputs(strategy, tmp_output_dir, 2)


def test_use_edits_skip_planning(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Direct edit mode (skip_planning + use_edits) produces valid candidates."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        use_edits=True,
        skip_planning=True,
    )
    strategy.optimize(iterations=2)
    _assert_basic_outputs(strategy, tmp_output_dir, 2, expect_plans=False)


def test_use_edits_with_translation(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Edit mode with translation iterations."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        use_edits=True,
        translate_iters=2,
    )
    strategy.optimize(iterations=3)
    _assert_basic_outputs(strategy, tmp_output_dir, 3)


# ---------------------------------------------------------------------------
# Translation iterations
# ---------------------------------------------------------------------------

def test_translate_then_optimize(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """First 2 iters are translation, last 2 are optimization — all produce valid candidates."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        translate_iters=2,
    )
    strategy.optimize(iterations=4)
    _assert_basic_outputs(strategy, tmp_output_dir, 4)


def test_translate_with_scoring(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Translation scoring runs and produces cache files."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        translate_iters=2,
        translate_score=True,
    )
    strategy.optimize(iterations=3)
    _assert_basic_outputs(strategy, tmp_output_dir, 3)

    for i in (1, 2):
        ts_cache = tmp_output_dir / f"eval-results-iter-{i}" / "translation_scores.json"
        assert ts_cache.exists(), f"Expected translation score cache for iter {i}"

    # Non-translate iter should not have a cache file
    assert not (tmp_output_dir / "eval-results-iter-3" / "translation_scores.json").exists()


def test_translate_drop_original(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """translate_drop_original removes the initial candidate from the beam at the last translate iter."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        translate_iters=2,
        translate_drop_original=True,
        beam_size=2,
    )
    strategy.optimize(iterations=3)

    # After translate iter 2, all beam candidates should have parents (original dropped)
    cands_iter2 = strategy.repository.get_candidates(2)
    for cand in cands_iter2:
        assert cand.parent is not None, "Original (parentless) candidate should be dropped"


def test_translate_skip_planning(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Translation with skip_planning=True (direct implementation, no plan phase)."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
        translate_iters=2,
        skip_planning=True,
    )
    strategy.optimize(iterations=3)
    _assert_basic_outputs(strategy, tmp_output_dir, 3, expect_plans=False)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stop_fires(built_agent, flat_eval_backend, dummy_prob, tmp_output_dir):
    """With a backend that returns identical scores, early stopping should trigger."""
    strategy = _make_strategy(
        tmp_output_dir, flat_eval_backend, built_agent, dummy_prob,
        early_stop_iters=2,
        early_stop_threshold=0.98,
    )
    strategy.optimize(iterations=10)

    rm = json.loads((tmp_output_dir / "run_metrics.json").read_text())
    assert len(rm["iterations"]) < 10, "Early stopping should have fired before all 10 iterations"


def test_early_stop_skips_translate_iters(built_agent, flat_eval_backend, dummy_prob, tmp_output_dir):
    """Early stopping should not fire during translation iterations, even if scores are flat."""
    strategy = _make_strategy(
        tmp_output_dir, flat_eval_backend, built_agent, dummy_prob,
        translate_iters=3,
        early_stop_iters=2,
        early_stop_threshold=0.98,
    )
    strategy.optimize(iterations=10)

    rm = json.loads((tmp_output_dir / "run_metrics.json").read_text())
    # Must complete at least translate_iters + early_stop_iters = 5 before stopping
    assert len(rm["iterations"]) >= 5, (
        f"Expected at least 5 iterations (3 translate + 2 window), got {len(rm['iterations'])}"
    )


# ---------------------------------------------------------------------------
# Cache / resume
# ---------------------------------------------------------------------------

def test_resume_from_cache(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Running optimize twice with same output_dir resumes from cached candidates."""
    strategy = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
    )
    strategy.optimize(iterations=2)

    # Second run should load from cache
    strategy2 = _make_strategy(
        tmp_output_dir, dummy_eval_backend, built_agent, dummy_prob,
    )
    strategy2.optimize(iterations=3)

    # Should have 3 iterations of candidates
    for i in range(1, 4):
        cands = strategy2.repository.get_candidates(i)
        assert len(cands) >= 1

