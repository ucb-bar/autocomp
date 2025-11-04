import pathlib
import glob
import subprocess
from typing import List
import os
import shutil
from datetime import datetime

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.hardware_backend import HardwareBackend

class TrnHardwareBackend(HardwareBackend):
    def _extract_latency(self, stdout: str) -> float:
        """Extract latency from stdout using pattern 'Latency: <latency> ms'"""
        lines = stdout.split('\n')
        for line in lines:
            if 'Latency:' in line and 'ms' in line:
                # Split by 'Latency:' and then by 'ms' to get the number
                parts = line.split('Latency:')[1].split('ms')[0].strip()
                try:
                    return float(parts)
                except ValueError:
                    continue
        return None
    
    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate the code based on the provided optimization metric.
        Returns list of dicts
        [
            {
                "correct": True,
                "test_results": {0: True, 1: True, ...}
            },
            ...
        ]
        """
        results = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = pathlib.Path(__file__).parent / "tmp_files" / "trn_eval" / timestamp
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Load the test code
        test_dir = pathlib.Path(__file__).parent.parent.parent / "tests" / prob.prob_type
        test_file = list(test_dir.glob(f"{prob.prob_id}_*.py"))[0]
        if not test_file:
            raise FileNotFoundError(f"No test file found for {prob.prob_type} {prob.prob_id} in {test_dir}")
        test_code = test_file.read_text()

        for i, code_str in enumerate(code_strs):
            test_code_i = test_code.replace("# SUBSTITUTE HERE", code_str)
            with open(temp_dir / f"code_{i}.py", "w") as f:
                f.write(test_code_i)

            # Run the test code and capture stdout
            cmd = ["python", str(temp_dir.resolve() / f"code_{i}.py")]
            logger.info(f"Running command {' '.join(cmd)}")
            try:
                p = subprocess.run(cmd, 
                                 capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                logger.error(f"Code {i} timed out after 120 seconds")
                results.append({"correct": False})
                continue

            # Save stdout and stderr to temp_dir
            with open(temp_dir / f"code_{i}_output.txt", "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(p.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(p.stderr)

            result_dict = {
                "correct": False,
                "latency": None,
                "stdout": p.stdout,
                "stderr": p.stderr
            }

            if p.returncode != 0:
                logger.error(f"Code {i} failed to run")
                results.append(result_dict)
                continue

            # Extract latency from stdout
            latency = self._extract_latency(p.stdout)
            if latency is None:
                logger.error(f"Code {i} did not produce latency output")
                results.append(result_dict)
                continue
            
            logger.info(f"Code {i} latency: {latency}")
            result_dict["correct"] = True
            result_dict["latency"] = latency
            results.append(result_dict)

        return results
