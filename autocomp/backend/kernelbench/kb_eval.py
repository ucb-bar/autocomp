import pathlib
import glob
import subprocess
from typing import List
import os
import shutil
from datetime import datetime

from autocomp.common import logger, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

KERNELBENCH_DIR = pathlib.Path("/scratch/charleshong/kernelbench/KernelBench")

class KBEvalBackend(EvalBackend):
    def get_backend_specific_rules(self) -> list[str]:
        return [
            "All generated code should be contained in a single Python file (inline CUDA code is allowed).",
            "Only class ModelNew will be imported during evaluation. Feel free to define other variables, functions, or classes, but make sure they are used by ModelNew.",
        ]

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        level_str = prob.prob_type.split("-")[1]
        ref_file = glob.glob(f"{KERNELBENCH_DIR}/KernelBench/{level_str}/{prob.prob_id}_*.py")[0]
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = pathlib.Path(__file__).parent / "tmp_files" / f"kb_eval_{timestamp}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, code_str in enumerate(code_strs):
            test_file = tmp_dir / f"code_{i}.py"
            test_file.write_text(code_str)

            cmd = [
                "python", 
                "scripts/run_and_check.py", 
                "ref_origin=local",
                f"ref_arch_src_path={str(ref_file)}",
                f"kernel_src_path={str(test_file)}",
                f"level={level_str[-1]}",
                f"problem_id={prob.prob_id}",
                "timeout=10",
            ]
            logger.info(f"Running command: {' '.join(cmd)} from cwd {KERNELBENCH_DIR}")
            try:
                result = subprocess.run(cmd, cwd=KERNELBENCH_DIR, check=False, capture_output=True, text=True, timeout=240)
            except Exception as e:
                logger.info(f"Error running command: {e}")
                results.append({"correct": False})
                continue
            stdout = result.stdout
            output_file = tmp_dir / f"output_{i}.txt"
            output_file.write_text(stdout)
            if " runtime_stats={'mean':" not in stdout:
                logger.info(f"Kernel did not pass correctness for code {i}")
                results.append({"correct": False})
            else:
                latency = float(stdout.split(" runtime_stats={'mean': ")[-1].split(",")[0])
                logger.info(f"Kernel passed correctness for code {i}, latency: {latency}")
                results.append({"correct": True, "latency": latency})
        return results

def main():
    prob_type = "kb-level1"
    prob_id = 1
    prob = Prob(prob_type, prob_id)
    files = glob.glob(str(SOLS_DIR / prob_type / f"{prob_id}_*.py"))
    code_strs = [pathlib.Path(file).read_text() for file in files]
    stats = KBEvalBackend().evaluate_code(prob, code_strs, "kernelbench")
    print(stats)

if __name__ == "__main__":
    main()
