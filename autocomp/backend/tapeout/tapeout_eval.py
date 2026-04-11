import re
import sys
import tempfile
import types
from pathlib import Path
from typing import List

import torch

from autocomp.backend.eval_backend import EvalBackend
from autocomp.common import logger
from autocomp.search.prob import Prob

# Path to the npu_model repo inside third_party
NPU_MODEL_ROOT = (Path(__file__).resolve().parents[3] / "third_party" / "npu_model").resolve()


def _ensure_npu_model_importable() -> None:
    """Add the npu_model repo root to sys.path if not already present."""
    root_str = str(NPU_MODEL_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _register_npu_configs() -> None:
    """Import ISA definitions and hardware configs so the simulator works."""
    import npu_model.configs.isa_definition  # noqa: F401 — registers @instr ops
    import npu_model.configs.hardware  # noqa: F401 — registers hardware configs


# ── Code cleaning ────────────────────────────────────────────────────

def _clean_code(code_str: str) -> str:
    """Extract the ``def test()`` function from LLM output.

    The LLM is expected to produce a Python function named ``test`` that
    returns a list of Instructions.  Its response may include markdown
    fences, prose, or multiple code blocks.  This function extracts just
    the ``def test(...):`` body.
    """
    # Strip markdown fences first
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", code_str, re.DOTALL)
    if blocks:
        # Prefer the block containing def test
        for block in blocks:
            if re.search(r"^def\s+test\s*\(", block, re.MULTILINE):
                code_str = block.strip()
                break
        else:
            code_str = max(blocks, key=len).strip()

    # Trim to the def test() function and everything after it
    lines = code_str.split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^def\s+test\s*\(", line):
            start = i
            break

    if start is not None:
        code_str = "\n".join(lines[start:]).rstrip()

    return code_str


# ── Harness substitution ────────────────────────────────────────────

def _substitute_into_harness(harness_code: str, test_func_code: str) -> str:
    """Insert LLM-generated test function into the test harness.

    Replaces everything between ``# SUBSTITUTE HERE`` and
    ``# SUBSTITUTE END`` with the provided code.
    """
    lines = harness_code.split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        if "# SUBSTITUTE HERE" in line:
            out.append(line)          # keep the marker as a comment
            out.append(test_func_code)
            skipping = True
        elif "# SUBSTITUTE END" in line:
            skipping = False
            out.append(line)
        elif not skipping:
            out.append(line)
    return "\n".join(out)


# ── Program loading ─────────────────────────────────────────────────

def _load_program_class(combined_source: str):
    """Exec the combined harness+candidate source and return the Program subclass."""
    from npu_model.software import Program

    mod = types.ModuleType("_autocomp_tapeout_candidate")
    mod.__file__ = "<candidate>"
    exec(compile(combined_source, "<candidate>", "exec"), mod.__dict__)

    program_classes = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Program)
            and obj is not Program
        ):
            program_classes.append(obj)

    if not program_classes:
        raise ValueError("No Program subclass found after substitution")
    return program_classes[0]


# ── Simulation ───────────────────────────────────────────────────────

def _run_simulation(program, hardware_config, max_cycles: int = 10000, verbose: bool = False):
    """Run simulation, return (SimulationStatistics, Simulation)."""
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        trace_path = f.name

    try:
        sim = Simulation(
            hardware_config=hardware_config,
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
            verbose=verbose,
        )
        sim.run(max_cycles=max_cycles)
        stats = sim.get_stats()
    finally:
        try:
            Path(trace_path).unlink(missing_ok=True)
        except (PermissionError, OSError):
            pass

    return stats, sim


def _check_golden(program, sim) -> bool:
    """Compare simulation output against program.golden_result (problem-owned)."""
    if not hasattr(program, "golden_result") or not program.golden_result:
        return True

    output_base, golden_tensor = program.golden_result
    size = golden_tensor.numel() * golden_tensor.element_size()
    mem_data = sim.core.arch_state.read_dram(output_base, size)
    actual = (
        mem_data.view(golden_tensor.dtype)
        .reshape(golden_tensor.shape)
        .clone()
    )
    if not torch.allclose(actual.float(), golden_tensor.float(), rtol=1e-2, atol=1e-2):
        diff = (actual.float() - golden_tensor.float()).abs().max()
        logger.info("Golden check failed: max diff = %.6f", diff.item())
        return False
    return True


# ── Backend class ────────────────────────────────────────────────────

class TapeoutEvalBackend(EvalBackend):
    """Evaluate NPU programs via the npu_model cycle-accurate simulator.

    Follows the test-harness pattern used by other autocomp backends:
    the problem defines a test harness (``harnesses/tapeout/<id>_<name>_test.py``)
    that owns inputs, memory layout, and golden output.  The LLM generates
    only a ``def test()`` function returning a list of Instructions, which
    gets substituted into the harness at the ``# SUBSTITUTE HERE`` marker.
    """

    def __init__(self, hw_config=None, max_cycles: int = 10000):
        super().__init__()
        self.max_cycles = max_cycles

        _ensure_npu_model_importable()
        _register_npu_configs()

        import npu_model.configs.hardware as hw_mod
        if hw_config is not None and hasattr(hw_config, "npu_hardware_config_name"):
            cfg_name = hw_config.npu_hardware_config_name
        else:
            cfg_name = "DefaultHardwareConfig"
        self.npu_hw_config_cls = getattr(hw_mod, cfg_name)

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """Evaluate candidate NPU test functions against the problem harness.

        Each code_str should be a ``def test()`` function (or LLM response
        containing one) that returns a ``List[Instruction]``.

        Returns one dict per candidate:
            {"correct": bool, "latency": int (cycles)}
        """
        # Load the test harness for this problem
        if prob.test_file:
            test_file = prob.test_file
        else:
            from autocomp.common import HARNESSES_DIR
            test_dir = HARNESSES_DIR / prob.prob_type
            matches = list(test_dir.glob(f"{prob.prob_id}_*_test.py"))
            if not matches:
                raise FileNotFoundError(
                    f"No test harness found for {prob} in {test_dir}")
            test_file = matches[0]
        harness_code = test_file.read_text()
        results: List[dict] = []
        for i, code_str in enumerate(code_strs):
            result: dict = {"correct": False}

            # 1. Clean LLM output → extract def test()
            try:
                cleaned = _clean_code(code_str)
                combined = _substitute_into_harness(harness_code, cleaned)
            except Exception as e:
                logger.info("Candidate %d: cleaning/substitution failed: %s", i, e)
                result["stderr"] = f"Code cleaning/substitution failed: {e}"
                results.append(result)
                continue

            # 2. Load the Program class from combined source
            try:
                prog_cls = _load_program_class(combined)
                program = prog_cls()
            except Exception as e:
                logger.info("Candidate %d: failed to load program: %s", i, e)
                result["stderr"] = f"Failed to load program: {e}"
                results.append(result)
                continue

            # 3. Simulate
            try:
                stats, sim = _run_simulation(
                    program,
                    hardware_config=self.npu_hw_config_cls(),
                    max_cycles=self.max_cycles,
                )
            except Exception as e:
                logger.info("Candidate %d: simulation error: %s", i, e)
                result["stderr"] = f"Simulation error: {e}"
                results.append(result)
                continue

            # 4. Check golden (problem-owned, not LLM-owned)
            correct = _check_golden(program, sim)
            result["correct"] = correct
            result["latency"] = stats.cycles

            if correct:
                logger.info("Candidate %d: PASS — %d cycles", i, stats.cycles)
            else:
                logger.info("Candidate %d: FAIL (golden mismatch) — %d cycles", i, stats.cycles)
                result["stderr"] = f"Golden mismatch: output does not match expected result after {stats.cycles} cycles"

            results.append(result)

        return results
