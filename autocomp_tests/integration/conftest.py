import pathlib

import pytest

from autocomp.backend.eval_backend import EvalBackend
from autocomp.search.prob import Prob


class DummyEvalBackend(EvalBackend):
    """Eval backend that returns decreasing latencies to simulate improvement."""

    def __init__(self):
        self._call_count = 0

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> list[dict]:
        results = []
        for _ in code_strs:
            self._call_count += 1
            latency = 1.0 / self._call_count
            results.append({"correct": True, "p99_latency": latency})
        return results


class FlatEvalBackend(EvalBackend):
    """Eval backend that always returns the same latency (for testing early stopping)."""

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> list[dict]:
        return [{"correct": True, "p99_latency": 1.0} for _ in code_strs]


class FailFirstEvalBackend(EvalBackend):
    """Eval backend that passes the initial-code check, then fails the next `num_failures`
    evaluations, then succeeds.

    Used to exercise the reimplement-failed path: ``BeamSearchStrategy.__init__`` evaluates
    the initial candidate and refuses to start if it's incorrect, so we let that first
    call through, then force failures on subsequent iteration candidates.
    """

    def __init__(self, num_failures: int = 1):
        self._initial_check_done = False
        self._remaining_failures = num_failures
        self._call_count = 0

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> list[dict]:
        results = []
        for _ in code_strs:
            self._call_count += 1
            if not self._initial_check_done:
                self._initial_check_done = True
                results.append({"correct": True, "p99_latency": 1.0})
            elif self._remaining_failures > 0:
                self._remaining_failures -= 1
                results.append({
                    "correct": False,
                    "stdout": "",
                    "stderr": "dummy failure: simulated crash",
                })
            else:
                results.append({"correct": True, "p99_latency": 1.0 / self._call_count})
        return results


@pytest.fixture
def dummy_eval_backend():
    return DummyEvalBackend()


@pytest.fixture
def flat_eval_backend():
    return FlatEvalBackend()


@pytest.fixture
def fail_first_eval_backend():
    return FailFirstEvalBackend(num_failures=1)


@pytest.fixture
def tmp_output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def dummy_prob(tmp_path):
    """A Prob with explicit test_file so it skips filesystem auto-glob."""
    test_file = tmp_path / "test0.py"
    test_file.write_text("# dummy test file")
    sol_file = tmp_path / "0_dummy.py"
    sol_file.write_text("def nki_kernel():\n    pass\n")
    return Prob("trn-tutorial-nki1", 0, test_file=test_file, sol_file=sol_file)
