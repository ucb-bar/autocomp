"""Integration test: run a full BeamSearchStrategy.optimize() loop with dummy LLM + eval backend."""

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


def test_beam_search_two_iterations(built_agent, dummy_eval_backend, dummy_prob, tmp_output_dir):
    """Run two iterations of beam search and verify candidates are produced."""
    strategy = BeamSearchStrategy(
        output_dir=tmp_output_dir,
        eval_backend=dummy_eval_backend,
        agent=built_agent,
        orig_code="def nki_kernel():\n    pass\n",
        prob=dummy_prob,
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

    strategy.optimize(iterations=2)

    iter0 = strategy.repository.get_candidates(0)
    assert len(iter0) >= 1, "Should have at least the initial candidate"
    assert iter0[0].score == 1.0

    iter1 = strategy.repository.get_candidates(1)
    assert len(iter1) >= 1, "Should have at least one candidate after iteration 1"

    iter2 = strategy.repository.get_candidates(2)
    assert len(iter2) >= 1, "Should have at least one candidate after iteration 2"
