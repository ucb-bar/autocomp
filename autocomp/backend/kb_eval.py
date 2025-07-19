import pathlib
import glob
import subprocess
from typing import List
import os
import shutil

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.hardware_backend import HardwareBackend

import torch
from src import eval as kernel_eval

KERNELBENCH_DIR = pathlib.Path("/scratch/charleshong/kernelbench/KernelBench")

class CudaHardwareBackend(HardwareBackend):
    def __init__(self):
        pass

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        level_str = prob.prob_type.split("-")[1]
        ref_file = glob.glob(f"{KERNELBENCH_DIR}/KernelBench/{level_str}/{prob.prob_id}_*.py")[0]
        results = []
        for i, code_str in enumerate(code_strs):
            test_file = pathlib.Path(__file__).parent / "tmp_files" / f"test_{level_str}_{prob.prob_id}_{i}.py"
            test_file.write_text(code_str)

            # # Attempt 3
            # ref_arch_src = pathlib.Path(ref_file).read_text()
            # kernel_src = code_str
            # configs = {
            #     "ref_origin": "local",
            #     "ref_arch_src_path": str(ref_file),
            #     "kernel_src_path": str(test_file),
            #     "level": int(level_str[-1]),
            #     "problem_id": prob.prob_id,
            #     "timeout": 10,
            #     "dataset_name": "ScalingIntelligence/KernelBench",
            #     "num_correct_trials": 5,
            #     "num_perf_trials": 100,
            #     "verbose": False,
            #     "measure_performance": True,
            #     "build_dir_prefix": "",
            #     "clear_cache": False,
            #     "gpu_arch": ["Ada"],
            # }
            # device = torch.device("cuda:0")

            # kernel_hash = str(hash(kernel_src))
            # build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
            
            # if configs["clear_cache"]: # fresh kernel build
            #     print(f"[INFO] Clearing cache for build directory: {build_dir}")
            #     shutil.rmtree(build_dir, ignore_errors=True)
            # try:
            #     eval_result = kernel_eval.eval_kernel_against_ref(
            #         original_model_src=ref_arch_src,
            #         custom_model_src=kernel_src,
            #         measure_performance=configs["measure_performance"],
            #         verbose=configs["verbose"],
            #         num_correct_trials=configs["num_correct_trials"],
            #         num_perf_trials=configs["num_perf_trials"],
            #         build_dir=build_dir,
            #         device=device
            #     )
            # except Exception as e:
            #     print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
            #     if "CUDA error" in str(e): 
            #         # NOTE: count this as compilation failure as it is not runnable code
            #         metadata = {"cuda_error": f"CUDA Error: {str(e)}",
            #                     "hardware": torch.cuda.get_device_name(device=device),
            #                     "device": str(device)
            #                     }
            #         eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
            #                                             metadata=metadata)
            #     else:
            #         metadata = {"other_error": f"error: {str(e)}",
            #                     "hardware": torch.cuda.get_device_name(device=device),
            #                     "device": str(device)
            #                     }
            #         eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
            #                                             metadata=metadata)

            # results_dict = {"correct": False}
            # if not eval_result:
            #     pass
            # elif eval_result.correctness:
            #     results_dict["correct"] = True
            #     results_dict["latency"] = eval_result.runtime

            # results.append(results_dict)
            
            # # Attempt 2
            # try:
            #     kernel_eval_result = evaluate_single_sample_src(ref_arch_src, kernel_src, configs, device)
            # except Exception as e:
            #     logger.info(f"Error evaluating code {i}: {e}")
            #     results.append({"correct": False})
            #     continue
            # if not kernel_eval_result or not kernel_eval_result.correctness:
            #     logger.info(f"Kernel did not pass correctness for code {i}")
            #     results.append({"correct": False})
            # else:
            #     kernel_exec_time = kernel_eval_result.runtime
            #     logger.info(f"Kernel passed correctness for code {i}, latency: {kernel_exec_time}")
            #     results.append({"correct": True, "latency": kernel_exec_time})

            # # Attempt 1
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
            # proc = subprocess.Popen(cmd, cwd=KERNELBENCH_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # try:
            #     result = proc.communicate(timeout=120)
            # except subprocess.TimeoutExpired:
            #     proc.kill()
            #     result = proc.communicate()
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
    stats = CudaHardwareBackend().evaluate_code(prob, code_strs, "cuda")
    print(stats)

if __name__ == "__main__":
    main()