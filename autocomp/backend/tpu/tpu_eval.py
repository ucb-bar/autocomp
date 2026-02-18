from concurrent.futures import process
import os
import pathlib
import subprocess
import shutil
from datetime import datetime
from typing import List

from autocomp.common import logger, TESTS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend
import time

def _gcloud_bin() -> str:
	"""
	Return the gcloud executable name.
	On Windows it is often `gcloud.cmd`, elsewhere `gcloud`.
	"""
	if os.name == "nt":
		return "gcloud.cmd"
	return "gcloud"


def _tpu_vm_exists(tpu_name: str, zone: str, project: str) -> bool:
	"""
	Check if a TPU VM instance exists in the given zone.
	Uses `gcloud compute tpus tpu-vm describe ...` and returns True on success.
	"""
	gcloud = _gcloud_bin()
	if shutil.which(gcloud) is None:
		logger.warning("gcloud not found on PATH; cannot check TPU VM existence.")
		return False
	cmd = [gcloud, "compute", "tpus", "tpu-vm", "describe", tpu_name, "--zone", zone, "--project", project]
	proc = subprocess.run(cmd, capture_output=True, text=True)
	return proc.returncode == 0


def _ensure_tpu_vm_running(
	tpu_name: str,
	zone: str,
	accelerator_type: str = "v6e-1",
	version: str = "v2-alpha-tpuv6e",
	project: str = "auto-tpu",
) -> None:
	"""
	Ensure a TPU VM exists; if not, create it.

	Command (from user request):
	  gcloud alpha compute tpus tpu-vm create my-v6e-tpu \
	    --zone=us-east5-a \
	    --accelerator-type=v6e-1 \
	    --version=v2-alpha-tpuv6e \
	    --project=auto-tpu
	"""
	if _tpu_vm_exists(tpu_name, zone, project):
		logger.info("TPU VM '%s' already exists in zone '%s'; skipping create.", tpu_name, zone)
		return

	gcloud = _gcloud_bin()
	if shutil.which(gcloud) is None:
		raise FileNotFoundError("gcloud not found on PATH; cannot create TPU VM.")

	cmd = [
		gcloud, "alpha", "compute", "tpus", "tpu-vm", "create", tpu_name,
		f"--zone={zone}",
		f"--accelerator-type={accelerator_type}",
		f"--version={version}",
		f"--project={project}",
	]
	logger.info("Creating TPU VM since none exists. Running: %s", " ".join(cmd))
	proc = subprocess.run(cmd, text=True)
	if proc.returncode != 0:
		raise RuntimeError(f"Failed to create TPU VM '{tpu_name}' (exit {proc.returncode}).")


class TpuHardwareBackend(EvalBackend):
	def __init__(self, tpu_name: str | None = None, tpu_zone: str | None = None, tpu_project: str | None = None):
		self.tpu_name = tpu_name or "my-v6e-tpu"
		self.tpu_zone = tpu_zone or "us-east5-a"
		# Ensure gcloud commands target the same project the TPU was created in.
		# Default matches the create command in _ensure_tpu_vm_running().
		self.tpu_project = tpu_project or os.getenv("AUTOCOMP_TPU_PROJECT") or "auto-tpu"
		self._tpu_vm_checked = False


# Upstream-style naming for consistency with upstream `search.py` patterns.
class TpuEvalBackend(TpuHardwareBackend):
	pass

	def ensure_tpu_vm(self) -> None:
		"""
		Ensure the TPU VM exists (create if missing).

		This is called from scp/ssh/evaluation so callers don't have to invoke
		`tpu_eval.py` as a script first.
		"""
		if self._tpu_vm_checked:
			return
		_ensure_tpu_vm_running(
			tpu_name=self.tpu_name,
			zone=self.tpu_zone,
			accelerator_type=os.getenv("AUTOCOMP_TPU_ACCELERATOR_TYPE") or "v6e-1",
			version=os.getenv("AUTOCOMP_TPU_RUNTIME_VERSION") or "v2-alpha-tpuv6e",
			project=self.tpu_project,
		)
		self._tpu_vm_checked = True

	def _extract_latency(self, stdout: str) -> float | None:
		"""Extract latency from stdout using pattern 'Latency: <latency> ms'."""
		lines = stdout.split("\n")
		for line in lines:
			if "Latency:" in line and "ms" in line:
				parts = line.split("Latency:")[1].split("ms")[0].strip()
				try:
					return float(parts)
				except ValueError:
					continue
		return None

	def _run_scp(self, local_path: pathlib.Path, remote_path: str) -> subprocess.CompletedProcess:
		"""Upload file to TPU via gcloud scp."""
		self.ensure_tpu_vm()
		gcloud = _gcloud_bin()
		if not local_path.exists():
			raise FileNotFoundError(f"Local file to scp not found: {local_path}")
		local_arg = local_path.resolve().as_posix()
		destination = f"{self.tpu_name}:{remote_path}"
		cmd = [
			gcloud, "compute", "tpus", "tpu-vm", "scp",
			local_arg,
			destination,
			"--quiet",
			"--zone", self.tpu_zone,
			"--project", self.tpu_project,
		]
		logger.info(f"Running command {' '.join(cmd)}")
		proc = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=300,
			stdin=subprocess.DEVNULL,
		)
		if proc.returncode != 0:
			logger.error("gcloud scp failed (exit %s). stderr:\n%s", proc.returncode, proc.stderr)
		return proc

	def _run_ssh(self, remote_command: str) -> subprocess.CompletedProcess:
		"""Run command on TPU via gcloud ssh."""
		self.ensure_tpu_vm()
		gcloud = _gcloud_bin()
		cmd = [
			gcloud, "compute", "tpus", "tpu-vm", "ssh",
			self.tpu_name,
			"--quiet",
			"--zone", self.tpu_zone,
			"--project", self.tpu_project,
			"--command", remote_command,
			"--", "-T"
		]
		logger.info(f"Running command {' '.join(cmd)}")
		return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

	def _run_ssh_with_enter_presses(
		self,
		remote_command: str,
		*,
		initial_wait_s: float = 10.0,
		enter_presses: int = 6,
		enter_interval_s: float = 2.0,
		timeout_s: float = 600.0,
	) -> subprocess.CompletedProcess:
		"""
		Run a TPU SSH command but periodically send Enter.

		Some TPU VMs buffer output until a newline is received on stdin, so for
		the "run python on VM" path we press Enter a few times before collecting
		output.
		"""
		self.ensure_tpu_vm()
		gcloud = _gcloud_bin()
		cmd = [
			gcloud, "compute", "tpus", "tpu-vm", "ssh",
			self.tpu_name,
			"--quiet",
			"--zone", self.tpu_zone,
			"--project", self.tpu_project,
			"--command", remote_command,
			"--", "-T",
		]
		logger.info(f"Running command {' '.join(cmd)}")

		start = time.monotonic()
		proc = subprocess.Popen(
			cmd,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)

		# Wait for the SSH session to fully come up.
		if initial_wait_s > 0:
			time.sleep(initial_wait_s)

		# Periodically "press Enter" to coax output from the VM.
		for _ in range(max(0, enter_presses)):
			if proc.poll() is not None:
				break
			if proc.stdin is not None:
				try:
					proc.stdin.write("\n")
					proc.stdin.flush()
				except BrokenPipeError:
					break
			if enter_interval_s > 0:
				time.sleep(enter_interval_s)

		# Collect output, respecting the overall timeout.
		elapsed = time.monotonic() - start
		remaining = max(1.0, timeout_s - elapsed)
		try:
			stdout, stderr = proc.communicate(timeout=remaining)
		except subprocess.TimeoutExpired:
			proc.kill()
			stdout, stderr = proc.communicate()
			raise

		return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

	def _wrap_remote_command_with_pip_install(self, run_python_command: str) -> str:
		
		# Keep it simple + close to the exact commands you provided.
		# - Uninstall may fail if not installed; ignore failures.
		# - Install uses the libtpu wheels index.
		uninstall = "pip uninstall -y jax jaxlib -q >/dev/null 2>&1 || true"
		install = "pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q"
		return f"{uninstall}; {install}; {run_python_command}"

	def run_file_on_tpu(self, file_path: str | pathlib.Path, remote_filename: str = "remote_upload.py") -> dict:
		"""
		Run a single Python file on the TPU VM via SSH.
		
		Args:
			file_path: Local path to the Python file to run
			remote_filename: Name to use for the file on the remote TPU (default: remote_upload.py)
		
		Returns:
			dict with keys: correct (bool), latency (float or None), stdout (str), stderr (str)
		"""
		self.ensure_tpu_vm()
		local_file = pathlib.Path(file_path)
		if not local_file.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		result_dict = {
			"correct": False,
			"latency": None,
			"stdout": "",
			"stderr": "",
		}
		
		# Upload file to TPU
		try:
			print("LOCAL FIEL: ", local_file.resolve())
			scp_proc = self._run_scp(local_file.resolve(), remote_filename)
		except subprocess.TimeoutExpired:
			logger.error(f"File upload timed out after 300 seconds")
			result_dict["stderr"] = "Upload timeout"
			return result_dict
		
		if scp_proc.returncode != 0:
			logger.error(f"File upload failed")
			result_dict["stdout"] = scp_proc.stdout
			result_dict["stderr"] = scp_proc.stderr
			return result_dict
		
		# Run file on TPU
		try:
			# Use interactive mode (press Enter) only for actually running code.
			remote_cmd = self._wrap_remote_command_with_pip_install(f"python3 {remote_filename}")
			run_proc = self._run_ssh_with_enter_presses(remote_cmd)
		except subprocess.TimeoutExpired:
			logger.error(f"File execution timed out after 600 seconds")
			result_dict["stderr"] = "Execution timeout"
			return result_dict
		
		result_dict["stdout"] = run_proc.stdout
		result_dict["stderr"] = run_proc.stderr
		
		if run_proc.returncode != 0:
			logger.error("File failed to run on TPU. stderr:\n%s", (run_proc.stderr or "").strip())
			return result_dict
		
		# Extract latency
		latency = self._extract_latency(run_proc.stdout)
		if latency is None:
			logger.warning(f"File did not produce latency output")
		else:
			logger.info(f"Latency: {latency} ms")
			result_dict["correct"] = True
			result_dict["latency"] = latency
		
		return result_dict

	def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
		"""
		Evaluate the code by uploading to a TPU VM and running via SSH.
		Returns list of dicts with success, latency, stdout, stderr.
		"""
		self.ensure_tpu_vm()

		local_file = pathlib.Path(__file__).parent / "code_0.py"
		if not local_file.exists():
			raise FileNotFoundError(f"Fixed TPU eval file not found: {local_file}")

		if code_strs:
			logger.info(
				"TPU eval is configured to run the fixed file %s verbatim; ignoring %d candidate code strings.",
				local_file,
				len(code_strs),
			)

		# Run once and replicate the result for all candidates.
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		eval_dir = pathlib.Path(__file__).parent / "eval_outputs" / timestamp
		eval_dir.mkdir(parents=True, exist_ok=True)

		result_dict = {
			"correct": False,
			"latency": None,
			"stdout": "",
			"stderr": "",
		}

		try:
			scp_proc = self._run_scp(local_file.resolve(), "remote_upload.py")
		except subprocess.TimeoutExpired:
			logger.error("Fixed file upload timed out after 300 seconds")
			result_dict["stderr"] = "Upload timeout"
			return [result_dict for _ in range(len(code_strs))]

		if scp_proc.returncode != 0:
			logger.error("Fixed file upload failed")
			result_dict["stdout"] = scp_proc.stdout
			result_dict["stderr"] = scp_proc.stderr
			return [result_dict for _ in range(len(code_strs))]

		try:
			remote_cmd = self._wrap_remote_command_with_pip_install("python3 remote_upload.py")
			run_proc = self._run_ssh_with_enter_presses(remote_cmd)
		except subprocess.TimeoutExpired:
			logger.error("Fixed file execution timed out after 600 seconds")
			result_dict["stderr"] = "Execution timeout"
			return [result_dict for _ in range(len(code_strs))]

		result_dict["stdout"] = run_proc.stdout
		result_dict["stderr"] = run_proc.stderr

		output_path = eval_dir / "fixed_code_output.txt"
		with open(output_path, "w", encoding="utf-8", errors="replace") as f:
			f.write("=== STDOUT ===\n")
			f.write(run_proc.stdout)
			f.write("\n=== STDERR ===\n")
			f.write(run_proc.stderr)

		if run_proc.returncode != 0:
			logger.error("Fixed file failed to run on TPU. See %s. stderr:\n%s", output_path, (run_proc.stderr or "").strip())
			return [result_dict for _ in range(len(code_strs))]

		latency = self._extract_latency(run_proc.stdout)
		if latency is None:
			logger.error("Fixed file did not produce latency output")
			return [result_dict for _ in range(len(code_strs))]

		logger.info("Fixed file latency: %s", latency)
		result_dict["correct"] = True
		result_dict["latency"] = latency
		return [result_dict for _ in range(len(code_strs))]


# Example usage:
if __name__ == "__main__":
	import sys
	
	if len(sys.argv) < 2:
		print("Usage: python -m autocomp.backend.tpu.tpu_eval <file_path>")
		print("Example: python -m autocomp.backend.tpu.tpu_eval my_tpu_script.py")
		sys.exit(1)
	
	file_path = sys.argv[1]
	backend = TpuHardwareBackend(	
		tpu_name="my-v6e-tpu",
		tpu_zone="us-east5-a",
	)

	# Ensure TPU VM exists before attempting to scp/ssh.
	backend.ensure_tpu_vm()
	
	
	print(f"Running {file_path} on TPU...")
	result = backend.run_file_on_tpu(file_path)
	
	print(f"\n{'='*60}")
	print(f"Success: {result['correct']}")
	print(f"Latency: {result['latency']} ms" if result['latency'] else "Latency: N/A")
	print(f"{'='*60}")
	print(f"\nSTDOUT:\n{result['stdout']}")
	if result['stderr']:
		print(f"\nSTDERR:\n{result['stderr']}")

# $ python -m autocomp.backend.tpu.tpu_eval autocomp/backend/tpu/code_0.py