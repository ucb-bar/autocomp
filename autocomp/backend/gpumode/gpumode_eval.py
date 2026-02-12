import pathlib
from typing import List
import subprocess
import os
import math
import time
from datetime import datetime
import sys

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

GPUMODE_DIR = pathlib.Path("/scratch/charleshong/cuda-opt/reference-kernels/problems")
prob_names = {
    0: "trimul",
}
paths_to_probs = {
    0: GPUMODE_DIR / "bioml" / "trimul",
}

class GpuModeEvalBackend(EvalBackend):
    def get_backend_specific_rules(self) -> list[str]:
        return [
            "All generated code should be contained in a single Python file (inline CUDA code is allowed).",
        ]

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str, benchmark_num: int = None) -> List[dict]:
        """
        simulator: "gpumode" or "gpumode-cli"
        """
        os.environ["POPCORN_FD"] = "1"
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_files_dir = pathlib.Path(__file__).parent / "tmp_files" / f"test_gpumode_{prob.prob_id}_{timestamp}"
        tmp_files_dir.mkdir(parents=True, exist_ok=True)

        for i, code_str in enumerate(code_strs):
            test_latencies = []
            stdout_output = ""
            if simulator == "gpumode-local":
                cmd = [
                    "python", 
                    "eval.py", 
                    "benchmark",
                    "benchmark.txt",
                ]
                # Write code to expected submission file
                prob_dir = paths_to_probs[prob.prob_id]
                submission_file_loc = prob_dir / f"submission.py"
                with open(submission_file_loc, "w") as f:
                    f.write(code_str)
                logger.info(f"Running command: {' '.join(cmd)} from cwd {prob_dir}")
                try:
                    result = subprocess.run(cmd, cwd=prob_dir, check=False, capture_output=True, text=True, timeout=120)
                except Exception as e:
                    logger.info(f"Error running command: {e}")
                    results.append({"correct": False})
                    continue
                stdout_output = result.stdout
                with open(tmp_files_dir / f"code_{i}_output.txt", "w") as f:
                    f.write("=== STDOUT ===\n")
                    f.write(stdout_output)
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
                if result.returncode != 0:
                    logger.info(f"Command returned non-zero exit code {result.returncode} for code {i}")
                    results.append({"correct": False, "stdout": stdout_output, "stderr": result.stderr})
                    continue
                if "status: fail" in stdout_output:
                    logger.info(f"Kernel did not pass correctness for code {i}")
                    results.append({"correct": False, "stdout": stdout_output, "stderr": result.stderr})
                    continue
                # If no failures
                # Extract the number of benchmarks from "benchmark-count: N"
                num_bmarks = 0
                for line in stdout_output.splitlines():
                    if line.startswith("benchmark-count: "):
                        try:
                            num_bmarks = int(line.split("benchmark-count: ")[1].strip())
                        except Exception as e:
                            logger.info(f"Could not parse benchmark-count: {e}")
                        break
                for test_idx in range(num_bmarks):
                    if f"benchmark.{test_idx}.mean: " in stdout_output:
                        test_latency = float(stdout_output.split(f"benchmark.{test_idx}.mean: ")[-1].split("\n")[0])
                        test_latencies.append(test_latency)
                # Make sure we have all the test means
                if len(test_latencies) != num_bmarks:
                    logger.info(f"Kernel did not pass correctness for code {i}")
                    results.append({"correct": False, "stdout": stdout_output, "stderr": result.stderr})
                    continue

            elif simulator == "gpumode-cli":
                # Store code and outputs in tmp_files directory
                submission_file_loc = tmp_files_dir / f"code_{i}.py"
                template_dir = pathlib.Path(__file__).parent.parent.parent / "tests" / prob.prob_type
                matches = list(template_dir.glob(f"{prob.prob_id}_*.py"))
                with open(matches[0], "r") as f:
                    template_str = f.read()
                code_str = template_str.replace("# SUBSTITUTE HERE", code_str)
                with open(submission_file_loc, "w") as f:
                    f.write(code_str)

                output_file_loc = tmp_files_dir / f"output_{i}.txt"
                mode = "leaderboard"
                cmd = [
                    "popcorn-cli",
                    "submit", 
                    "--gpu",
                    "NVIDIA",
                    "--leaderboard",
                    "nvfp4_gemv", # TODO change back to prob_names[prob.prob_id]
                    "--mode",
                    mode,
                    "-o",
                    str(output_file_loc.resolve()),
                    str(submission_file_loc.resolve()),
                ]
                logger.info(f"Running command: {' '.join(cmd)}")
                try:
                    result = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                except Exception as e:
                    logger.info(f"Error running command: {e}")
                    results.append({"correct": False, "stdout": stdout_output, "stderr": result.stderr})
                    continue
                
                # Wait for output file to be created, then kill the process
                timeout = 600  # seconds
                start_time = time.time()
                while not output_file_loc.exists():
                    if time.time() - start_time > timeout:
                        logger.info(f"Timeout waiting for output file for code {i}")
                        result.terminate()
                        result.wait()
                        break
                    if result.poll() is not None:
                        # Process has finished
                        break
                    time.sleep(1)
                else:
                    # Output file exists, kill the process
                    if result.poll() is None:  # Process is still running
                        result.terminate()
                        result.wait()
                    
                    # Read the output file
                    try:
                        with open(output_file_loc, "r") as f:
                            stdout_output = f.read()
                    except Exception as e:
                        logger.info(f"Error reading output file for code {i}: {e}")
                        results.append({"correct": False})
                        continue

                    str_to_check = "Benchmarking successful" if mode == "benchmark" else "Leaderboard run successful"
                    bad_strs = ["Running tests failed", "Benchmarking failed", "Leaderboard run failed"]
                    if str_to_check not in stdout_output or any(bad_str in stdout_output for bad_str in bad_strs):
                        logger.info(f"Kernel did not pass correctness for code {i}")
                        results.append({
                            "correct": False,
                            "stdout": stdout_output,
                            "stderr": result.stderr,
                        })
                        continue
                    # ⏱ 306 ± 17.0 µs
                    for line in stdout_output.split("\n"):
                        if "⏱ " in line:
                            test_latency = float(line.split("⏱ ")[-1].split(" µs")[0].split(" ms")[0].split(" ±")[0])
                            if "ms" in line:
                                test_latency *= 1000
                            test_latencies.append(test_latency)
                    if mode == "leaderboard":
                        # Only use the bottom half of the test latencies for leaderboard mode
                        test_latencies = test_latencies[len(test_latencies) // 2:]
                if not test_latencies:
                    logger.info(f"Kernel did not pass correctness for code {i}")
                    results.append({"correct": False, "stdout": stdout_output, "stderr": result.stderr})
                    continue

            latency = round(math.prod(test_latencies) ** (1/len(test_latencies)), 3)  # geomean
            results.append({"correct": True, "latency": latency, "stdout": stdout_output})
            logger.info(f"Kernel passed correctness for code {i}, latency: {latency}")
        return results

if __name__ == "__main__":
    args = sys.argv[1:]
    prob = Prob("gpumode", int(args[0]))
    if len(args) > 2:
        benchmark_num = int(args[2])
    else:
        benchmark_num = None
    # code_strs = [pathlib.Path("/scratch/charleshong/kernelbench/autocomp-kb/sols/gpumode/4_nvfp4_gemv.py").read_text()]
    code_strs = [pathlib.Path(args[1]).read_text()]
    results = GpuModeEvalBackend().evaluate_code(prob, code_strs, "gpumode-cli", benchmark_num)
    print(results)