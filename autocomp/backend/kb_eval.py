import pathlib
import glob
import subprocess
from typing import List
import os
import shutil

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.hardware_backend import HardwareBackend

KERNELBENCH_DIR = pathlib.Path("/scratch/charleshong/kernelbench/KernelBench")

class KBHardwareBackend(HardwareBackend):
    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        level_str = prob.prob_type.split("-")[1]
        ref_file = glob.glob(f"{KERNELBENCH_DIR}/KernelBench/{level_str}/{prob.prob_id}_*.py")[0]
        results = []
        for i, code_str in enumerate(code_strs):
            test_dir = pathlib.Path(__file__).parent / "tmp_files" / f"test_{level_str}_{prob.prob_id}"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / f"code_{i}.py"
            test_file.write_text(code_str)

            with open(test_file, "w") as f:
                f.write(code_str)
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
                result = subprocess.run(cmd, cwd=KERNELBENCH_DIR, check=False, capture_output=True, text=True, timeout=120)
            except Exception as e:
                logger.info(f"Error running command: {e}")
                results.append({"correct": False})
                continue
            stdout = result.stdout
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
    files = glob.glob(str(pathlib.Path(__file__).parent.parent.parent / "sols" / prob_type / f"{prob_id}_*.py"))
    code_strs = [pathlib.Path(file).read_text() for file in files]
    stats = KBHardwareBackend().evaluate_code(prob, code_strs, "kernelbench")
    print(stats)

if __name__ == "__main__":
    main()
