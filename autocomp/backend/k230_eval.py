import pathlib
import glob
import subprocess
from typing import List
import os
import shutil

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.hardware_backend import HardwareBackend

class K230HardwareBackend(HardwareBackend):
    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        test_dir = pathlib.Path(__file__).parent / "tmp_files" / f"test_{prob.prob_type}_{prob.prob_id}"
        test_dir.mkdir(parents=True, exist_ok=True)
        rvv_assembly_build_dir = pathlib.Path(f"/home/charleshong/assembly_build_dir_{prob.prob_id}")
        result_dir = pathlib.Path("/media/charleshong/K230_APP/")
        # Remove previous files
        for file in test_dir.glob("code_*.c"):
            file.unlink()
        for file in test_dir.glob("code_*.s"):
            file.unlink()
        for file in rvv_assembly_build_dir.glob("code_*.s"):
            file.unlink()
        for file in rvv_assembly_build_dir.glob("code_*.elf"):
            file.unlink()
        # for file in result_dir.glob("code_*_result.txt"):
        #     file.unlink()
        # Write new files
        for i, code_str in enumerate(code_strs):
            test_file = test_dir / f"code_{i}.c"
            test_file.write_text(code_str)
        cmd = [
            "make",
            "assemble",
        ]
        logger.info(f"Running command: {' '.join(cmd)} from cwd {test_dir}")
        subprocess.run(cmd, cwd=test_dir, check=True)
        # Copy the .s files to the location where we will compile them
        for file in test_dir.glob("code_*.s"):
            shutil.copy(file, rvv_assembly_build_dir / file.name)
        
        # Wait for result files to be populated
        import pdb; pdb.set_trace()

        # Collect results
        results = []
        num_code_strs = len(code_strs)
        for i in range(num_code_strs):
            result_file = result_dir / f"code_{i}_result.txt"
            if not result_file.exists():
                results.append({"correct": False})
                continue
            with open(result_file, "r") as f:
                result = f.read()
            if "PASS" not in result:
                logger.info(f"Code {i} did not pass correctness")
                results.append({"correct": False})
            else:
                latency = int(result.split(" cycles")[0])
                logger.info(f"Code {i} passed correctness, latency: {latency}")
                results.append({"correct": True, "latency": latency})
        return results

def main():
    pass

if __name__ == "__main__":
    main()
