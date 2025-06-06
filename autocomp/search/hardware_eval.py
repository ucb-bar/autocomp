import subprocess
import pathlib
import multiprocessing
from typing import List
import time
import glob
import shutil
import os
import signal

from autocomp.common import logger
from autocomp.search.prob import Prob

FP32_4PE_CHIPYARD_PATH = None
INT8_16PE_CHIPYARD_PATH = None
INT8_32PE_CHIPYARD_PATH = None

def clean_code(code_str: str) -> str:
    """
    Takes LLM-generated code, removes the "test" wrapper function, and cleans up anything that is not runnable C code

    for example:
    '''
    void test(Kinf, r, K_r) {
        config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
        config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
        ...
        mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
        fence();
    }
    '''

    becomes 

    '''
    config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
    config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
    ...
    mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
    fence();
    '''
    """
    # Remove the function wrapper and return only the body of the function
    if not code_str:
        return ""
    after_void_test_str = code_str[code_str.find("void test("):]
    start = after_void_test_str.find('{') + 1
    end = after_void_test_str.rfind('}')
    body = after_void_test_str[start:end]
    return body
    # body = code_str[start:end].split("\n")
    # new_body = []
    # for line in body:
    #     new_body.append(line.strip())
    # return "\n".join(new_body)

def compile_gemmini_code(code_contents: str, gemmini_path: pathlib.Path):
    """
    Compile the provided code to a baremetal binary.
    Returns the path to the compiled binary.
    """
    test_name = "auto_comp_test"
    # if test_name_str:
    #     test_name += "_" + str(test_name_str)
    gemmini_sw_path = gemmini_path / "software" / "gemmini-rocc-tests"
    # copy in file
    with open(gemmini_sw_path / "bareMetalC" / (test_name + ".c"), "w") as f:
        f.write(code_contents)

    # # edit Makefile
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'r') as file:
    #     filedata = file.read()
    # if test_name not in filedata:
    #     filedata = filedata.replace("tests = .*\n", "tests = "+test_name+"\n")
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'w') as file:
    #     filedata = file.write(filedata)
    
    # build software
    p = subprocess.run(["sh", "./build.sh"], cwd=gemmini_sw_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        return
    
    return gemmini_sw_path / "build" / "bareMetalC" / (test_name + "-baremetal")

def run_spike_mp(code_contents_lst: list[str], gemmini_path: pathlib.Path, timeout: float=5):
    manager = multiprocessing.Manager()

    return_dicts = []
    procs = []
    for code_i, code_contents in enumerate(code_contents_lst):
        return_dict = manager.dict()
        return_dicts.append(return_dict)
        p = multiprocessing.Process(target=run_spike, args=(code_contents,return_dict,gemmini_path,str(code_i),timeout))
        p.start()
        procs.append(p)
        time.sleep(2)

    # while any(p.is_alive() for p in procs):
    #         # All the processes are done, break now.
    #     time.sleep(1)  # Just to avoid hogging the CPU

    start = time.time()
    while time.time() - start <= timeout:
        if not any(p.is_alive() for p in procs):
            # All the processes are done, break now.
            break
        time.sleep(.1)  # Just to avoid hogging the CPU
    else:
        # We only enter this if we didn't 'break' above.
        logger.info("spike ran for more than %d seconds, terminating.", timeout)
        for i, p in enumerate(procs):
            if p.is_alive():
                p.terminate()
                p.join()
                return_dicts[i]["retval"] = "Timeout"

    return [return_dict["retval"] for return_dict in return_dicts]
    
def run_spike(code_contents: str, return_dict: dict, gemmini_path: pathlib.Path, test_name_str: str, timeout: float):
    test_name = "auto_comp_test"
    # if test_name_str:
    #     test_name += "_" + str(test_name_str)
    gemmini_sw_path = gemmini_path / "software" / "gemmini-rocc-tests"
    # copy in file
    with open(gemmini_sw_path / "bareMetalC" / (test_name + ".c"), "w") as f:
        f.write(code_contents)

    # # edit Makefile
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'r') as file:
    #     filedata = file.read()
    # if test_name not in filedata:
    #     filedata = filedata.replace("tests = .*\n", "tests = "+test_name+"\n")
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'w') as file:
    #     filedata = file.write(filedata)
    
    # build software
    p = subprocess.run(["sh", "./build.sh"], cwd=gemmini_sw_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        # return_dict["retval"] = p.stdout.decode()
        return_dict["retval"] = "Compile error"
        return

    p = subprocess.run(["stdbuf", "-oL", "sh", "./scripts/run-spike.sh", test_name], cwd=gemmini_path, 
                         capture_output=True, text=True, errors="ignore")
    spike_output = p.stdout

    # # run with timeout
    # with subprocess.Popen(["stdbuf", "-oL", "sh", "./scripts/run-spike.sh", test_name], cwd=gemmini_path, 
    #                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore") as process:
    #     try:
    #         spike_output = process.communicate(timeout=timeout)[0]
    #     except subprocess.TimeoutExpired:
    #         os.killpg(process.pid, signal.SIGINT) # send signal to the process group
    #         logger.info("spike ran for more than %d seconds, terminating.", timeout)
    #         return_dict["retval"] = "Timeout"
    #         return

    # start = time.time()
    # while time.time() - start <= timeout:
    #     if p.poll() is not None:
    #         # Process is done, break now.
    #         break
    #     time.sleep(1)  # Just to avoid hogging the CPU
    # else:
    #     # We only enter this if we didn't 'break' above.
    #     logger.info("spike ran for more than %d seconds, terminating.", timeout)
    #     p.kill()
    #     return_dict["retval"] = "Timeout"
    #     return
    # spike_output = p.stdout

    # try:
    #     p = subprocess.run(["sh", "./scripts/run-spike.sh", test_name], cwd=GEMMINI_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    # except subprocess.TimeoutExpired:
    #     return "Timeout"
    # if p.returncode != 0:
    #     return str(p.stdout)

    # return output
    return_dict["retval"] = spike_output

def run_firesim(code_contents_lst: list[str], gemmini_path: pathlib.Path, firesim_path: pathlib.Path, timeout: float=60):
    results_dir = firesim_path / "deploy" / "results-workload"
    results = []

    # # Check correctness first with spike
    # spike_results = run_spike_mp(code_contents_lst, gemmini_path, timeout=20)

    # Launch FPGA runfarm, if on AWS
    # subprocess.run(["firesim", "launchrunfarm"])
    # target_config = "firesim_rocket_singlecore_gemmini_no_nic_l2_llc4mb_ddr3_v2"
    # Run each code content in sequence
    for i in range(len(code_contents_lst)):
        # if "Correct result" not in spike_results[i]:
        #     logger.info("Skipping FireSim run for code %d due to incorrect spike result.", i)
        #     results.append("")
        #     continue

        # Compile the code
        orig_binary_path = compile_gemmini_code(code_contents_lst[i], gemmini_path)
        if not orig_binary_path:
            logger.info("Failed to compile code.")
            results.append("")
            continue
        firesim_gemmini_workload_path = firesim_path / "deploy" / "workloads" / "gemmini"
        firesim_gemmini_workload_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(orig_binary_path, firesim_gemmini_workload_path / "auto_comp_test-baremetal")

        # Get the current directories under results_dir
        orig_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

        # config_runtime_path = firesim_path / "deploy" / "config_runtime.yaml"
        # with open(config_runtime_path, "r") as f:
        #     config = yaml.safe_load(f)
        # config["workload"]["workload_name"] = "auto_comp_test-baremetal"
        # config["target_config"]["default_hw_config"] = target_config
        # with open(config_runtime_path, "w") as f:
        #     yaml.dump(config, f)
        logger.info("Running `firesim infrasetup`")
        p = subprocess.Popen([str(firesim_path / "deploy" / "firesim"), "infrasetup"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        with p.stdout:
            for line in p.stdout:
                logger.debug(line.decode().strip())
        p.wait()
        logger.info("Running `firesim runworkload`")
        p = subprocess.Popen([str(firesim_path / "deploy" / "firesim"), "runworkload"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # with p.stdout:
        #     for line in p.stdout:
        #         logger.debug(line.decode().strip())
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait()
            logger.warning("Firesim runworkload timed out.")
            logger.info("Running `firesim kill`")
            subprocess.run([str(firesim_path / "deploy" / "firesim"), "kill"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
            results.append("")
            continue
        logger.info("Firesim runworkload finished, terminate now if you need to")
        time.sleep(2)

        # Terminate FPGA runfarm, if on AWS
        # p = subprocess.Popen(["firesim", "terminaterunfarm"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # p.communicate("yes\n".encode())

        # See if there are any new directories under results_dir
        new_dirs = list()
        current_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for dir in current_results_dirs:
            if dir not in orig_results_dirs:
                new_dirs.append(dir)
        if not new_dirs:
            logger.error("No new results directory found after running FireSim.")
            results.append("")
            continue

        relevant_logs = []
        for dir in new_dirs:
            dir_logs = glob.glob(f"{dir}/*/uartlog", recursive=True)
            for log in dir_logs:
                relevant_logs.append(log)
        if len(relevant_logs) > 1:
            logger.warning("Multiple new logs found, using the first one. Logs found: %s", str(relevant_logs))
        if len(relevant_logs) == 0:
            logger.error("No relevant logs found after running FireSim.")
            results.append("")
            continue
        logger.info("Parsing FireSim results from: %s", str(relevant_logs[0]))

        result_string = pathlib.Path(relevant_logs[0]).read_text()
        results.append(result_string)

    return results

def parse_spad_acc_utilization(spike_output: str, pe_dim: int, spad_size_kb: int, acc_size_kb: int) -> int:
    """
    Parse the output of the spike simulator to extract the spad utilization.
    """
    # Find the line that contains the spad utilization
    spad_addresses_used = set()
    acc_addresses_used = set()
    for line in spike_output.split("\n"):
        if "GEMMINI: mvin - " in line or "GEMMINI: mvout - " in line:
            if "mvin" in line:
                first_split = line.split("GEMMINI: mvin - ")[1]
            elif "mvout" in line:
                first_split = line.split("GEMMINI: mvout - ")[1]
            col_split = first_split.split(" cols and ")
            cols = int(col_split[0].strip(), 0) # actually just ignore this
            row_split = col_split[1].split(" rows from ")
            try:
                rows = int(row_split[0].strip(), 0)
            except:
                import pdb
                pdb.set_trace()
            addr_split = row_split[1].split(" to addr ")
            first_addr = int(addr_split[0].strip(), 0)
            second_addr = int(addr_split[1].strip(), 0)
            if "mvin" in line:
                spad_acc_addr = second_addr
            elif "mvout" in line:
                spad_acc_addr = first_addr
            accumulator = bool(spad_acc_addr >> 31)
            actual_addr = spad_acc_addr & 0x3FFFFFF
            for j in range(cols):
                for i in range(rows):
                    row_addr = actual_addr + pe_dim * (j // pe_dim * pe_dim) + i
                    if accumulator:
                        acc_addresses_used.add(row_addr)
                    else:
                        spad_addresses_used.add(row_addr)
    num_spad_rows = (spad_size_kb * 1024 / (pe_dim * 1))
    num_acc_rows = (acc_size_kb * 1024 / (pe_dim * 4))
    return len(spad_addresses_used) / num_spad_rows, len(acc_addresses_used) / num_acc_rows

class GemminiHardwareBackend:
    def __init__(self, pe_dim: int, spad_size_kb: int=256, acc_size_kb: int=64):
        self.pe_dim = pe_dim
        if pe_dim == 4:
            if not FP32_4PE_CHIPYARD_PATH:
                raise ValueError("FP32_4PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = FP32_4PE_CHIPYARD_PATH / "generators" / "gemmini"
            self.firesim_path = FP32_4PE_CHIPYARD_PATH / "sims" / "firesim"
        elif pe_dim == 16:
            if not INT8_16PE_CHIPYARD_PATH:
                raise ValueError("INT8_16PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = INT8_16PE_CHIPYARD_PATH / "generators" / "gemmini"
            self.firesim_path = INT8_16PE_CHIPYARD_PATH / "sims" / "firesim"
        elif pe_dim == 32:
            if not INT8_32PE_CHIPYARD_PATH:
                raise ValueError("INT8_32PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = INT8_32PE_CHIPYARD_PATH / "generators" / "gemmini"
            self.firesim_path = INT8_32PE_CHIPYARD_PATH / "sims" / "firesim"
        else:
            raise ValueError("supported Gemmini pe_dims: {4, 16, 32}")
        self.spad_size_kb = spad_size_kb
        self.acc_size_kb = acc_size_kb
        self.gemmini_sw_path = self.gemmini_path / "software" / "gemmini-rocc-tests"

    def __repr__(self):
        return f"GemminiHardwareBackend({self.pe_dim})"

    def evaluate_code_parallel_spike(self, prob: Prob, code_strs: list[str]) -> float:
        return self.evaluate_code(prob, code_strs, "spike")

    def evaluate_code_parallel_firesim(self, prob: Prob, code_strs: list[str]) -> float:
        return self.evaluate_code(prob, code_strs, "firesim")

    def get_spad_acc_utilization(self, prob: Prob, code_strs: list[str]) -> int:
        stats = [{
            "spad_util": 0,
            "acc_util": 0
        } for _ in code_strs]
        clean_code_strs = [clean_code(code_str) for code_str in code_strs]
        for test_i, test in enumerate(prob.tests):
            test_output_per_code_str = run_spike_mp([test.get_test_code([code_str], check_correct=False, repeat_iters=1) for code_str in clean_code_strs], self.gemmini_path, timeout=60)
            for code_i, test_output in enumerate(test_output_per_code_str):
                spad_util, acc_util = parse_spad_acc_utilization(test_output, self.pe_dim, self.spad_size_kb, self.acc_size_kb)
                stats[code_i]["spad_util"] = spad_util
                stats[code_i]["acc_util"] = acc_util
        return stats

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
        # Save stats, all passing at the beginning (flip to False on test failure)
        stats = [{
            "correct": True,
            "test_results": {}
        } for _ in code_strs]
        clean_code_strs = [clean_code(code_str) for code_str in code_strs]
        for test_i, test in enumerate(prob.tests):
            logger.info("Running spike on %d code candidates", len(code_strs))
            test_output_per_code_str = run_spike_mp([test.get_test_code([code_str]) for code_str in clean_code_strs], self.gemmini_path, timeout=3000)
            for code_i, test_output in enumerate(test_output_per_code_str):
                # logger.debug(test_output)
                if "Correct" in test_output:
                    logger.debug("Test %d Correct result", test_i)
                    stats[code_i]["test_results"][test_i] = True
                    if simulator == "spike": # Get instruction count from spike
                        if "Generated implementation latency" in test_output:
                            sol_latency = int(test_output.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                            stats[code_i]["latency"] = sol_latency
                else:
                    logger.debug("Test %d Incorrect result", test_i)
                    stats[code_i]["test_results"][test_i] = False
                    stats[code_i]["correct"] = False

        if simulator == "firesim":
            # test = prob.tests[0]
            # binary_paths = [compile_gemmini_code(test.get_test_code([code_str]), self.gemmini_path) for code_str in clean_code_strs]
            # for code_i, binary_path in enumerate(binary_paths):
            #     if binary_path:
            #         logger.debug("Code %d Compile success", code_i)
            #     else:
            #         logger.debug("Code %d Compile error", code_i)
            #         stats[code_i]["correct"] = False
            # Use batched run to speed up evaluation
            working_code_idxs = []
            for code_i in range(len(code_strs)):
                if stats[code_i]["correct"]: # Get correctness from spike
                    working_code_idxs.append(code_i)
            logger.info("%d of %d candidates passed spike", len(working_code_idxs), len(code_strs))
            logger.debug("Working code indices: %s", str(working_code_idxs))
            # logger.info("%d of %d candidates compiled successfully", len(working_code_idxs), len(code_strs))
            if len(working_code_idxs) == 0:
                return stats
            working_code_strs = [clean_code_strs[i] for i in working_code_idxs]
            first_test = prob.tests[0]
            # Batch N at a time
            batch_size = 100
            batch_start_idx = 0
            while batch_start_idx < len(working_code_strs):
                batch_end_idx = min(batch_start_idx + batch_size, len(working_code_strs))
                if self.pe_dim == 4:
                    this_batch_firesim_output = None
                else:
                    this_batch_firesim_output = run_firesim([first_test.get_test_code(working_code_strs[batch_start_idx:batch_end_idx], error_on_incorrect=False, repeat_iters=1)], 
                                                            self.gemmini_path, self.firesim_path, timeout=600)[0]
                if not this_batch_firesim_output:
                    if self.pe_dim != 4:
                        logger.warning("Code hang on batch, running individually")
                    if self.pe_dim == 4:    
                        individual_firesim_outputs = run_firesim([first_test.get_test_code([working_code_strs[idx]], error_on_incorrect=True, repeat_iters=20) for idx in range(batch_start_idx, batch_end_idx)], 
                                                                 self.gemmini_path, self.firesim_path, timeout=180)
                    else:
                        individual_firesim_outputs = run_firesim([first_test.get_test_code([working_code_strs[idx]], error_on_incorrect=False, repeat_iters=1) for idx in range(batch_start_idx, batch_end_idx)], 
                                                                 self.gemmini_path, self.firesim_path, timeout=60)
                    for idx, output in enumerate(individual_firesim_outputs):
                        orig_idx = working_code_idxs[batch_start_idx+idx]
                        if not output:
                            stats[orig_idx]["correct"] = False
                            logger.warning("Code hang on index %s, batch index %s", orig_idx, idx)
                        else:
                            latency_found = False
                            if "Incorrect result" in output:
                                stats[orig_idx]["correct"] = False
                                stats[orig_idx]["test_results"][0] = False
                                continue
                            for line in output.split("\n"):
                                if "Generated implementation latency" in line:
                                    sol_latency = int(line.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                                    if sol_latency == 99999999999:
                                        stats[orig_idx]["correct"] = False
                                        stats[orig_idx]["test_results"][0] = False
                                    else:
                                        stats[orig_idx]["latency"] = sol_latency
                                        stats[orig_idx]["test_results"][0] = True
                                    latency_found = True
                                    break
                            if not latency_found:
                                logger.error("Firesim output did not contain latency for index %s, batch index %s", orig_idx, idx)
                else:
                    cur_working_code_i = 0
                    for line in this_batch_firesim_output.split("\n"):
                        if "Generated implementation latency" in line:
                            sol_latency = int(line.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                            orig_idx = working_code_idxs[batch_start_idx+cur_working_code_i]
                            if sol_latency == 99999999999:
                                stats[orig_idx]["correct"] = False
                                stats[orig_idx]["test_results"][0] = False
                            else:
                                stats[orig_idx]["latency"] = sol_latency
                                stats[orig_idx]["test_results"][0] = True
                            cur_working_code_i += 1
                    if cur_working_code_i != (batch_end_idx - batch_start_idx):
                        raise ValueError("Firesim output did not contain enough latencies.")

                batch_start_idx = batch_end_idx
        logger.debug(stats)
        return stats

if __name__ == "__main__":
    prob = Prob("admm-multifunction", 2)
    files = [pathlib.Path(__file__).parent.parent.parent / "sols" / "admm-multifunction" / "sol2_5249.c"]
    code_strs = [file.read_text() for file in files]
    stats = GemminiHardwareBackend(4).evaluate_code(prob, code_strs, "firesim")
    # stats = GemminiHardwareBackend(4).get_spad_acc_utilization(prob, code_strs)
    print(stats)
