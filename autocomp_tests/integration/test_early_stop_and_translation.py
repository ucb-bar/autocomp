"""Tests for early stopping, translation score caching, and translation early-stop replay."""

import json
import os

os.environ.setdefault("WANDB_MODE", "disabled")

import pathlib
from unittest.mock import MagicMock

from autocomp.search.code_repo import CodeCandidate, CodeRepository
from autocomp.search.search import SearchStrategy


# ---------------------------------------------------------------------------
# Helpers: build a minimal SearchStrategy without the heavy __init__
# ---------------------------------------------------------------------------

def _make_stub_strategy(
    tmp_path: pathlib.Path,
    translate_iters: int = 0,
    translate_score: bool = False,
    early_stop_iters: int = 0,
    early_stop_threshold: float = 1.0,
) -> SearchStrategy:
    """Create a SearchStrategy with only the fields needed for the methods under test."""
    strategy = object.__new__(SearchStrategy)
    strategy.translate_iters = translate_iters
    strategy.translate_score = translate_score
    strategy.early_stop_iters = early_stop_iters
    strategy.early_stop_threshold = early_stop_threshold
    strategy.output_dir = tmp_path
    strategy.repository = CodeRepository()
    strategy.agent = MagicMock()
    strategy.prob = MagicMock()
    return strategy


def _make_candidate(score=1.0, translation_score=None):
    cand = CodeCandidate(parent=None, plan="plan", code="code", score=score)
    cand.translation_score = translation_score
    return cand


# ===========================================================================
# should_early_stop
# ===========================================================================

class TestShouldEarlyStop:
    def test_disabled_when_early_stop_iters_zero(self, tmp_path):
        s = _make_stub_strategy(tmp_path, early_stop_iters=0)
        assert s.should_early_stop([1.0, 1.0, 1.0, 1.0], cur_iter=4) is False

    def test_not_enough_iters(self, tmp_path):
        s = _make_stub_strategy(tmp_path, early_stop_iters=3)
        # Need at least early_stop_iters + 1 = 4 iterations
        assert s.should_early_stop([1.0, 1.0, 1.0], cur_iter=3) is False

    def test_triggers_when_stalled(self, tmp_path):
        s = _make_stub_strategy(tmp_path, early_stop_iters=3, early_stop_threshold=0.98)
        # 4 iters, no improvement: ratio = 1.0 >= 0.98
        losses = [1.0, 1.0, 1.0, 1.0]
        assert s.should_early_stop(losses, cur_iter=4) is True

    def test_does_not_trigger_when_improving(self, tmp_path):
        s = _make_stub_strategy(tmp_path, early_stop_iters=3, early_stop_threshold=0.98)
        # loss dropped from 1.0 to 0.5: ratio = 0.5 < 0.98
        losses = [1.0, 0.9, 0.7, 0.5]
        assert s.should_early_stop(losses, cur_iter=4) is False

    def test_skips_translation_iters(self, tmp_path):
        """Early stopping should not fire during or right after translation iters."""
        s = _make_stub_strategy(
            tmp_path, translate_iters=4, early_stop_iters=3, early_stop_threshold=0.98,
        )
        # Iter 7 = translate_iters(4) + early_stop_iters(3): still blocked
        losses = [1.0] * 7
        assert s.should_early_stop(losses, cur_iter=7) is False

        # Iter 8 = first eligible iter
        losses = [1.0] * 8
        assert s.should_early_stop(losses, cur_iter=8) is True

    def test_skips_translation_iters_with_early_translate_stop(self, tmp_path):
        """If translate_iters was reduced (early translate stop), the window adjusts."""
        s = _make_stub_strategy(
            tmp_path, translate_iters=2, early_stop_iters=3, early_stop_threshold=0.98,
        )
        # translate_iters=2, so first eligible = 2 + 3 + 1 = 6
        losses = [1.0] * 5
        assert s.should_early_stop(losses, cur_iter=5) is False
        losses = [1.0] * 6
        assert s.should_early_stop(losses, cur_iter=6) is True

    def test_zero_old_loss_returns_false(self, tmp_path):
        s = _make_stub_strategy(tmp_path, early_stop_iters=2, early_stop_threshold=0.98)
        losses = [0.0, 0.5, 0.5]
        assert s.should_early_stop(losses, cur_iter=3) is False


# ===========================================================================
# _score_translation caching
# ===========================================================================

class TestScoreTranslationCaching:
    def test_computes_and_caches(self, tmp_path):
        s = _make_stub_strategy(tmp_path)
        s.repository.add_candidates([_make_candidate(score=1.0)], 0)
        s.agent.score_translation_completeness.return_value = [8.5, 7.0]

        cands = [_make_candidate(score=0.5), _make_candidate(score=0.6)]
        save_dir = tmp_path / "eval-results"
        save_dir.mkdir()

        s._score_translation(cands, save_dir)

        assert cands[0].translation_score == 8.5
        assert cands[1].translation_score == 7.0
        assert (save_dir / "translation_scores.json").exists()
        cached = json.loads((save_dir / "translation_scores.json").read_text())
        assert cached == [8.5, 7.0]
        s.agent.score_translation_completeness.assert_called_once()

    def test_loads_from_cache(self, tmp_path):
        s = _make_stub_strategy(tmp_path)
        s.repository.add_candidates([_make_candidate(score=1.0)], 0)

        save_dir = tmp_path / "eval-results"
        save_dir.mkdir()
        (save_dir / "translation_scores.json").write_text(json.dumps([9.0, 6.0]))

        cands = [_make_candidate(score=0.5), _make_candidate(score=0.6)]
        s._score_translation(cands, save_dir)

        assert cands[0].translation_score == 9.0
        assert cands[1].translation_score == 6.0
        s.agent.score_translation_completeness.assert_not_called()

    def test_recomputes_on_length_mismatch(self, tmp_path):
        s = _make_stub_strategy(tmp_path)
        s.repository.add_candidates([_make_candidate(score=1.0)], 0)
        s.agent.score_translation_completeness.return_value = [7.0, 8.0, 5.0]

        save_dir = tmp_path / "eval-results"
        save_dir.mkdir()
        # Stale cache with 2 entries, but we now have 3 candidates
        (save_dir / "translation_scores.json").write_text(json.dumps([9.0, 6.0]))

        cands = [_make_candidate(score=0.5), _make_candidate(score=0.6), _make_candidate(score=0.7)]
        s._score_translation(cands, save_dir)

        assert [c.translation_score for c in cands] == [7.0, 8.0, 5.0]
        s.agent.score_translation_completeness.assert_called_once()
        # Cache should be overwritten
        cached = json.loads((save_dir / "translation_scores.json").read_text())
        assert cached == [7.0, 8.0, 5.0]


# ===========================================================================
# Translation early-stop replay from cache
# ===========================================================================

class TestCheckTranslationComplete:
    """Test _check_translation_complete (used by both live and cache-resume paths)."""

    def test_reduces_translate_iters(self, tmp_path):
        s = _make_stub_strategy(tmp_path, translate_iters=4, translate_score=True)
        cands = [
            _make_candidate(score=0.5, translation_score=9.5),
            _make_candidate(score=0.6, translation_score=9.2),
        ]
        s._check_translation_complete(cands, cur_iter=2)
        assert s.translate_iters == 2

    def test_no_change_when_scores_below_threshold(self, tmp_path):
        s = _make_stub_strategy(tmp_path, translate_iters=4, translate_score=True)
        cands = [
            _make_candidate(score=0.5, translation_score=7.0),
            _make_candidate(score=0.6, translation_score=9.5),
        ]
        s._check_translation_complete(cands, cur_iter=2)
        assert s.translate_iters == 4

    def test_no_change_on_last_translate_iter(self, tmp_path):
        """cur_iter >= translate_iters should not trigger (already at the boundary)."""
        s = _make_stub_strategy(tmp_path, translate_iters=4, translate_score=True)
        cands = [_make_candidate(score=0.5, translation_score=10.0)]
        s._check_translation_complete(cands, cur_iter=4)
        assert s.translate_iters == 4

    def test_no_change_when_translate_score_disabled(self, tmp_path):
        s = _make_stub_strategy(tmp_path, translate_iters=4, translate_score=False)
        cands = [_make_candidate(score=0.5, translation_score=10.0)]
        s._check_translation_complete(cands, cur_iter=2)
        assert s.translate_iters == 4

    def test_no_change_with_no_translation_scores(self, tmp_path):
        """Candidates without translation_score should not trigger early stop."""
        s = _make_stub_strategy(tmp_path, translate_iters=4, translate_score=True)
        cands = [_make_candidate(score=0.5), _make_candidate(score=0.6)]
        s._check_translation_complete(cands, cur_iter=2)
        assert s.translate_iters == 4
