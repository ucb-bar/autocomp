import pathlib

import pytest

from autocomp.backend.eval_backend import EvalBackend
from autocomp.search.prob import Prob


class DummyEvalBackend(EvalBackend):
    """Eval backend that always reports correct with a fixed latency."""

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> list[dict]:
        return [{"correct": True, "p99_latency": 1.0} for _ in code_strs]


@pytest.fixture
def dummy_eval_backend():
    return DummyEvalBackend()


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
