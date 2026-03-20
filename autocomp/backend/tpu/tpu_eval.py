import os
import pathlib
import subprocess
import shutil
from datetime import datetime
from typing import List, Literal
import secrets
import re
import shlex

from autocomp.common import logger, TESTS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

def _gcloud_bin() -> str:
	if os.name == "nt":
		return "gcloud.cmd"
	return "gcloud"


def _tpu_vm_exists(tpu_name: str, zone: str, project: str) -> bool:
	gcloud = _gcloud_bin()
	if shutil.which(gcloud) is None:
		logger.warning("gcloud not found on PATH; cannot check TPU VM existence.")
		return False
	cmd = [gcloud, "compute", "tpus", "tpu-vm", "describe", tpu_name, "--zone", zone, "--project", project]
	proc = subprocess.run(cmd, capture_output=True, text=True)
	return proc.returncode == 0


def _tpu_vm_state(tpu_name: str, zone: str, project: str) -> str | None:
	#returns if tpu vm is running
	gcloud = _gcloud_bin()
	if shutil.which(gcloud) is None:
		return None
	cmd = [
		gcloud, "compute", "tpus", "tpu-vm", "describe", tpu_name,
		"--zone", zone, "--project", project,
		"--format", "value(state)",
	]
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if proc.returncode != 0:
		return None
	return (proc.stdout or "").strip() or None


def _ensure_tpu_vm_running(
	tpu_name: str,
	zone: str,
	accelerator_type: str = "v6e-1",
	version: str = "v2-alpha-tpuv6e",
	project: str = "ch-llm",
) -> None:
	"creates tpu vm if not exists"
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
	def __init__(
		self,
		tpu_name: str | None = None,
		tpu_zone: str | None = None,
		tpu_project: str | None = None,
		*,
		transport: Literal["auto", "gcloud", "ssh"] | None = None,
		ssh_host: str | None = None,
		ssh_user: str | None = None,
		ssh_port: int | None = None,
		ssh_identity_file: str | None = None,
		ssh_extra_args: list[str] | None = None,
	):
		self.tpu_name = tpu_name or "tpu-node"
		self.tpu_zone = tpu_zone or "us-east5-a"
		# Ensure gcloud commands target the same project the TPU was created in.
		# Default matches the create command in _ensure_tpu_vm_running().
		self.tpu_project = tpu_project or os.getenv("AUTOCOMP_TPU_PROJECT") or "ch-llm"
		self.transport = transport or os.getenv("AUTOCOMP_TPU_TRANSPORT") or "auto"

		self.ssh_host = ssh_host or os.getenv("AUTOCOMP_TPU_SSH_HOST")
		self.ssh_user = ssh_user or os.getenv("AUTOCOMP_TPU_SSH_USER")
		port_env = os.getenv("AUTOCOMP_TPU_SSH_PORT")
		self.ssh_port = ssh_port or (int(port_env) if port_env and port_env.isdigit() else None)
		self.ssh_identity_file = ssh_identity_file or os.getenv("AUTOCOMP_TPU_SSH_IDENTITY_FILE")
		extra_env = os.getenv("AUTOCOMP_TPU_SSH_EXTRA_ARGS")
		self.ssh_extra_args = (
			ssh_extra_args
			or (shlex.split(extra_env) if extra_env else None)
			or []
		)
		self.ssh_strict_host_key_checking = os.getenv(
			"AUTOCOMP_TPU_SSH_STRICT_HOST_KEY_CHECKING", "accept-new"
		)
		self._tpu_vm_checked = False


	def _transport_mode(self) -> Literal["gcloud", "ssh"]:
		if self.transport in ("gcloud", "ssh"):
			return self.transport
		if self.transport not in ("auto", None):
			logger.warning("Unknown AUTOCOMP_TPU_TRANSPORT=%r; falling back to auto.", self.transport)
		return "ssh" if self.ssh_host else "gcloud"

	def _ssh_target(self) -> str:
		if not self.ssh_host:
			raise ValueError(
				"Direct SSH transport selected but no host is configured. \
				Set AUTOCOMP_TPU_SSH_HOST or pass ssh_host=..."
			)
		if self.ssh_user:
			return f"{self.ssh_user}@{self.ssh_host}"
		return self.ssh_host

	def _ssh_common_opts(self, *, batch_mode: bool) -> list[str]:
		opts: list[str] = []
		if batch_mode:
			opts += ["-o", "BatchMode=yes"]
		if self.ssh_strict_host_key_checking:
			opts += ["-o", f"StrictHostKeyChecking={self.ssh_strict_host_key_checking}"]
		opts += ["-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
		if self.ssh_identity_file:
			opts += ["-i", self.ssh_identity_file]
		return opts

	def _build_ssh_cmd(self, *, remote_command: str | None, allocate_tty: bool, batch_mode: bool) -> list[str]:
		if self._transport_mode() == "gcloud":
			self.ensure_tpu_vm()
			gcloud = _gcloud_bin()
			cmd = [
				gcloud, "compute", "tpus", "tpu-vm", "ssh",
				self.tpu_name,
				"--quiet",
				"--zone", self.tpu_zone,
				"--project", self.tpu_project,
			]
			if remote_command is not None:
				cmd += ["--command", remote_command]
			# Forward non-interactive/tty preference to underlying SSH.
			cmd += ["--"]
			cmd += ["-tt"] if allocate_tty else ["-T"]
			return cmd

		# Direct SSH.
		target = self._ssh_target()
		cmd = ["ssh"]
		if self.ssh_port:
			cmd += ["-p", str(self.ssh_port)]
		cmd += self._ssh_common_opts(batch_mode=batch_mode)
		cmd += self.ssh_extra_args
		cmd += ["-tt"] if allocate_tty else ["-T"]
		cmd += [target]
		if remote_command is not None:
			cmd += ["--", remote_command]
		return cmd

	def _build_scp_cmd(self, *, source: str, dest: str) -> list[str]:
		if self._transport_mode() == "gcloud":
			self.ensure_tpu_vm()
			gcloud = _gcloud_bin()
			return [
				gcloud,
				"compute",
				"tpus",
				"tpu-vm",
				"scp",
				source,
				dest,
				"--quiet",
				"--zone",
				self.tpu_zone,
				"--project",
				self.tpu_project,
			]

		# Direct SCP.
		cmd = ["scp", "-q"]
		if self.ssh_port:
			cmd += ["-P", str(self.ssh_port)]
		cmd += self._ssh_common_opts(batch_mode=True)
		cmd += self.ssh_extra_args
		cmd += [source, dest]
		return cmd

	def start_tpu_vm(self) -> None:
		state = _tpu_vm_state(self.tpu_name, self.tpu_zone, self.tpu_project)
		if state == "READY":
			logger.info("TPU VM '%s' is already running", self.tpu_name)
			return
		gcloud = _gcloud_bin()
		if shutil.which(gcloud) is None:
			raise FileNotFoundError("gcloud not found on PATH; cannot start TPU VM.")
		cmd = [
			gcloud, "compute", "tpus", "tpu-vm", "start", self.tpu_name,
			f"--zone={self.tpu_zone}",
			f"--project={self.tpu_project}",
		]
		logger.info("Starting TPU VM: %s", " ".join(cmd))
		proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
		if proc.returncode != 0:
			logger.error("Failed to start TPU VM (exit %s). stderr:\n%s", proc.returncode, proc.stderr)
			raise RuntimeError(f"Failed to start TPU VM '{self.tpu_name}' (exit {proc.returncode}).")
		logger.info("TPU VM '%s' started successfully.", self.tpu_name)

	def stop_tpu_vm(self) -> None:
		gcloud = _gcloud_bin()
		if shutil.which(gcloud) is None:
			raise FileNotFoundError("gcloud not found on PATH; cannot stop TPU VM.")
		cmd = [
			gcloud, "compute", "tpus", "tpu-vm", "stop", self.tpu_name,
			f"--zone={self.tpu_zone}",
			f"--project={self.tpu_project}",
		]
		logger.info("Stopping TPU VM: %s", " ".join(cmd))
		proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
		if proc.returncode != 0:
			logger.error("Failed to stop TPU VM (exit %s). stderr:\n%s", proc.returncode, proc.stderr)
			raise RuntimeError(f"Failed to stop TPU VM '{self.tpu_name}' (exit {proc.returncode}).")
		logger.info("TPU VM '%s' stopped successfully.", self.tpu_name)

	def open_tpu_shell(self) -> int:
		cmd = self._build_ssh_cmd(remote_command=None, allocate_tty=True, batch_mode=False)
		logger.debug("Opening TPU shell: %s", " ".join(cmd))
		return subprocess.call(cmd)


class TpuEvalBackend(TpuHardwareBackend):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._jax_setup_done = False

	def ensure_tpu_vm(self) -> None:

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

	def _extract_latency(self, output_text: str) -> float | None:
		lines = output_text.split("\n")
		for line in lines:
			if "Latency:" in line and "ms" in line:
				parts = line.split("Latency:")[-1].split("ms")[0].strip()
				try:
					return float(parts)
				except ValueError:
					continue
			if "Pallas Latency:" in line and "ms" in line:
				parts = line.split("Pallas Latency:")[-1].split("ms")[0].strip()
				try:
					return float(parts)
				except ValueError:
					continue
		return None

	def _extract_util_percent(self, output_text: str) -> float | None:
		for line in output_text.split("\n"):
			l = line.strip()
			if not l:
				continue
			m = re.search(r"(?i)\b(utilization|util)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(%)?", l)
			if not m:
				continue
			val = float(m.group(2))
			has_pct = m.group(3) is not None
			if not has_pct and 0.0 <= val <= 1.0:
				val *= 100.0
			if val < 0 or val > 1000:
				continue
			return round(val, 3)
		return None

	def _append_autocomp_hw_feedback_block(
		self, *, stdout_text: str, stderr_text: str
	) -> tuple[str, float | None]:
		util = self._extract_util_percent(stdout_text) or self._extract_util_percent(stderr_text)

		if util is None:
			return stdout_text, None

		block_lines: list[str] = ["", "AUTOCOMP TPU FEEDBACK:"]
		if util is not None:
			block_lines.append(f"util: {util}%")
		block_lines.append("END AUTOCOMP TPU FEEDBACK")
		block = "\n".join(block_lines) + "\n"
		return stdout_text + block, util

	def _run_scp(self, local_path: pathlib.Path, remote_path: str) -> subprocess.CompletedProcess:
		#upload file to tpu
		if not local_path.exists():
			raise FileNotFoundError(f"Local file to scp not found: {local_path}")
		local_arg = local_path.resolve().as_posix()
		destination = (
			f"{self.tpu_name}:{remote_path}"
			if self._transport_mode() == "gcloud"
			else f"{self._ssh_target()}:{remote_path}"
		)
		cmd = self._build_scp_cmd(source=local_arg, dest=destination)
		logger.debug("Running command %s", " ".join(cmd))
		proc = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=300,
			stdin=subprocess.DEVNULL,
		)
		if proc.returncode != 0:
			logger.error("scp failed (exit %s). stderr:\n%s", proc.returncode, proc.stderr)
		return proc

	def _run_ssh(self, remote_command: str) -> subprocess.CompletedProcess:
		cmd = self._build_ssh_cmd(remote_command=remote_command, allocate_tty=False, batch_mode=True)
		logger.debug("Running command %s", " ".join(cmd))
		return subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)

	def _remote_run_id(self) -> str:
		return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + secrets.token_hex(4)

	def _remote_eval_dir(self, run_id: str) -> str:
		return f"/tmp/autocomp_eval/{run_id}"

	def _remote_run_python_to_files_command(self, remote_filename: str, remote_dir: str, skip_setup: bool = False) -> str:
		prog_stdout = f"{remote_dir}/program_stdout.txt"
		prog_stderr = f"{remote_dir}/program_stderr.txt"
		exit_code = f"{remote_dir}/program_exit_code.txt"
		setup_log = f"{remote_dir}/setup_log.txt"
		setup_exit_code = f"{remote_dir}/setup_exit_code.txt"

		if skip_setup:
			return (
				f"mkdir -p {remote_dir}; "
				f": > {prog_stdout}; : > {prog_stderr}; "
				f"python3 {remote_filename} > {prog_stdout} 2> {prog_stderr}; prog_rc=$?; "
				f"echo $prog_rc > {exit_code}; "
				f"true"
			)

		check_jax = "python3 -c 'import jax; import jaxlib' >/dev/null 2>&1"
		uninstall = "pip uninstall -y jax jaxlib -q >/dev/null 2>&1 || true"
		install = "pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q"
		setup = f"if [ \"${{AUTOCOMP_TPU_FORCE_PIP:-0}}\" = \"1\" ]; then {uninstall}; ({install}); else ({check_jax}) || ({install}); fi"

		return (
			f"mkdir -p {remote_dir}; "
			f": > {prog_stdout}; : > {prog_stderr}; "
			f"({setup}) >> {setup_log} 2>&1; setup_rc=$?; echo $setup_rc > {setup_exit_code}; "
			f"prog_rc=1; "
			f"if [ $setup_rc -eq 0 ]; then "
			f"  python3 {remote_filename} > {prog_stdout} 2> {prog_stderr}; prog_rc=$?; "
			f"else "
			f"  echo \"pip install failed (exit $setup_rc)\" >> {prog_stderr}; "
			f"fi; "
			f"echo $prog_rc > {exit_code}; "
			f"true"
		)

	@staticmethod
	def _load_harness(prob: Prob) -> str:
		harness_path = TESTS_DIR / prob.prob_type / f"test{prob.prob_id}.py"
		if not harness_path.exists():
			raise FileNotFoundError(f"Harness file not found: {harness_path}")
		harness = harness_path.read_text(encoding="utf-8", errors="replace")
		harness_lines = []
		for line in harness.splitlines():
			stripped = line.strip()
			if stripped.startswith("import ") or stripped.startswith("from "):
				continue
			harness_lines.append(line)
		return "\n".join(harness_lines)

	def run_file_on_tpu(self, file_path: str | pathlib.Path, remote_filename: str = "remote_upload.py") -> dict:
		
		# self.ensure_tpu_vm()
		local_file = pathlib.Path(file_path)
		if not local_file.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		result_dict = {
			"correct": False,
			"latency": None,
			"stdout": "",
			"stderr": "",
			"util": None,
		}
		
		# Upload file to TPU
		try:
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
		run_id = self._remote_run_id()
		remote_dir = self._remote_eval_dir(run_id)
		remote_cmd = self._remote_run_python_to_files_command(remote_filename, remote_dir)
		try:
			run_proc = self._run_ssh(remote_cmd)
		except subprocess.TimeoutExpired:
			logger.error("File execution timed out after 600 seconds")
			result_dict["stderr"] = "Execution timeout"
			return result_dict

		# Pull program outputs back from TPU.
		prog_stdout = self._read_remote_file(f"{remote_dir}/program_stdout.txt")
		prog_stderr = self._read_remote_file(f"{remote_dir}/program_stderr.txt")
		exit_code_text = self._read_remote_file(f"{remote_dir}/program_exit_code.txt").strip()
		try:
			prog_rc = int(exit_code_text) if exit_code_text else 1
		except ValueError:
			prog_rc = 1

		enriched_stdout, util = self._append_autocomp_hw_feedback_block(
			stdout_text=prog_stdout, stderr_text=prog_stderr
		)
		result_dict["stdout"] = enriched_stdout
		result_dict["stderr"] = prog_stderr if prog_stderr else (run_proc.stderr or "")
		result_dict["util"] = util

		if prog_rc != 0:
			logger.error(
				"File failed to run on TPU (program exit %s). SSH stderr:\n%s",
				prog_rc,
				(run_proc.stderr or "").strip(),
			)
			return result_dict
		
		# Extract latency
		latency = self._extract_latency(prog_stdout)
		if latency is None:
			logger.warning(f"File did not produce latency output")
		else:
			logger.info(f"Latency: {latency} ms")
			result_dict["correct"] = True
			result_dict["latency"] = latency
		
		return result_dict

	def _read_remote_file(self, remote_path: str) -> str:
		"""Read a remote file via SSH cat (avoids a separate SCP roundtrip)."""
		cmd = self._build_ssh_cmd(
			remote_command=f"cat {remote_path} 2>/dev/null || true",
			allocate_tty=False, batch_mode=True,
		)
		proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
		return proc.stdout or ""

	_BATCH_DELIM_START = "===AUTOCOMP_CANDIDATE_START==="
	_BATCH_DELIM_END = "===AUTOCOMP_CANDIDATE_END==="

	def _build_batch_script(self, code_strs: list[str], harness_text: str) -> str:
		"""Build a single Python script that evaluates all candidates in one process."""
		import base64
		encoded = [base64.b64encode(code.encode()).decode() for code in code_strs]

		script_lines = [
			"import os, sys, time, traceback, base64",
			"import numpy as np",
			"import jax",
			"import jax.numpy as jnp",
			"from jax.experimental import pallas as pl",
			"from jax.experimental.pallas import tpu as pltpu",
			"",
			f"DELIM_START = {self._BATCH_DELIM_START!r}",
			f"DELIM_END = {self._BATCH_DELIM_END!r}",
			"",
			"CANDIDATE_SOURCES = [",
		]
		for b64 in encoded:
			script_lines.append(f'    "{b64}",')
		script_lines.append("]")
		script_lines.append("")
		harness_funcs = []
		skip_main = False
		for line in harness_text.splitlines():
			if line.strip().startswith('if __name__'):
				skip_main = True
				continue
			if skip_main:
				if line and not line[0].isspace():
					skip_main = False
				else:
					continue
			harness_funcs.append(line)
		script_lines.append("\n".join(harness_funcs))
		script_lines.append("")
		script_lines.append("""
_shared_globals = {
    k: v for k, v in globals().items()
    if not k.startswith("_") and k != "CANDIDATE_SOURCES"
}

for _idx, _b64 in enumerate(CANDIDATE_SOURCES):
    print(DELIM_START, flush=True)
    print(f"CANDIDATE_IDX={_idx}", flush=True)
    try:
        _src = base64.b64decode(_b64).decode()
        _ns = dict(_shared_globals)
        exec(_src, _ns)
        _test_fn = _ns.get("test")
        if _test_fn is None:
            print("ERROR: no test() function defined")
            print(DELIM_END, flush=True)
            continue
        test = _test_fn
        _run_autocomp_harness()
    except SystemExit:
        print("FAIL: SystemExit raised")
    except Exception:
        traceback.print_exc()
    print(DELIM_END, flush=True)
""")
		return "\n".join(script_lines)

	def _parse_batch_output(self, stdout: str, stderr: str, num_candidates: int) -> list[dict]:
		"""Parse the delimited output from a batch run into per-candidate results."""
		results = []
		sections = stdout.split(self._BATCH_DELIM_START)
		# First section is before any delimiter (preamble), skip it.
		candidate_outputs = []
		for section in sections[1:]:
			end_idx = section.find(self._BATCH_DELIM_END)
			if end_idx != -1:
				candidate_outputs.append(section[:end_idx])
			else:
				candidate_outputs.append(section)

		for idx in range(num_candidates):
			result_dict = {
				"correct": False,
				"latency": None,
				"stdout": "",
				"stderr": stderr,
				"util": None,
			}
			if idx < len(candidate_outputs):
				cand_stdout = candidate_outputs[idx].strip()
				enriched, util = self._append_autocomp_hw_feedback_block(
					stdout_text=cand_stdout, stderr_text=stderr
				)
				result_dict["stdout"] = enriched
				result_dict["util"] = util
				latency = self._extract_latency(cand_stdout)
				if latency is not None and "FAIL" not in cand_stdout and "ERROR" not in cand_stdout:
					result_dict["correct"] = True
					result_dict["latency"] = latency
			results.append(result_dict)
		return results

	def _evaluate_code_batch(self, prob: Prob, code_strs: list[str], eval_dir: pathlib.Path) -> list[dict] | None:
		"""Run all candidates in a single remote Python process.

		Returns parsed results on success, or None if the batch
		infrastructure itself failed (so the caller can fall back).
		"""
		harness_text = self._load_harness(prob)
		batch_script = self._build_batch_script(code_strs, harness_text)

		local_batch = eval_dir / "batch_run.py"
		local_batch.write_text(batch_script, encoding="utf-8", errors="replace")

		run_id = self._remote_run_id()
		remote_dir = self._remote_eval_dir(run_id)
		remote_name = f"autocomp_batch_{run_id}.py"

		try:
			scp_proc = self._run_scp(local_batch.resolve(), remote_name)
		except subprocess.TimeoutExpired:
			logger.warning("Batch upload timed out")
			return None

		if scp_proc.returncode != 0:
			logger.warning("Batch upload failed")
			return None

		remote_cmd = self._remote_run_python_to_files_command(
			remote_name, remote_dir, skip_setup=self._jax_setup_done,
		)
		try:
			run_proc = self._run_ssh(remote_cmd)
		except subprocess.TimeoutExpired:
			logger.warning("Batch execution timed out")
			return None

		prog_stdout = self._read_remote_file(f"{remote_dir}/program_stdout.txt")
		prog_stderr = self._read_remote_file(f"{remote_dir}/program_stderr.txt")

		output_path = eval_dir / "batch_output.txt"
		with open(output_path, "w", encoding="utf-8", errors="replace") as f:
			f.write("=== STDOUT ===\n")
			f.write(prog_stdout)
			f.write("\n=== STDERR ===\n")
			f.write(prog_stderr)

		if self._BATCH_DELIM_START not in prog_stdout:
			logger.warning("Batch produced no candidate output; will retry individually")
			return None

		self._jax_setup_done = True
		return self._parse_batch_output(prog_stdout, prog_stderr, len(code_strs))

	def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
		self.ensure_tpu_vm()
		logger.info("Evaluating %d candidate(s) on TPU for %s (prob %s/%s)",
					len(code_strs), self.tpu_name, prob.prob_type, prob.prob_id)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		eval_dir = pathlib.Path(__file__).parent / "eval_outputs" / timestamp
		eval_dir.mkdir(parents=True, exist_ok=True)

		results = self._evaluate_code_batch(prob, code_strs, eval_dir)

		if results is None and len(code_strs) > 1:
			logger.info("Falling back to individual evaluation for %d candidates", len(code_strs))
			results = []
			for idx, code in enumerate(code_strs):
				single = self._evaluate_code_batch(prob, [code], eval_dir)
				if single:
					results.extend(single)
				else:
					results.append({"correct": False, "latency": None,
									"stdout": "", "stderr": "Batch failed", "util": None})
		elif results is None:
			results = [{"correct": False, "latency": None,
						"stdout": "", "stderr": "Batch failed", "util": None}
					   for _ in code_strs]

		for idx, r in enumerate(results):
			if r["correct"]:
				logger.info("Candidate %d/%d: %.3f ms (util %.1f%%)",
							idx + 1, len(code_strs), r["latency"], r["util"] or 0.0)
			else:
				logger.info("Candidate %d/%d: FAIL", idx + 1, len(code_strs))

		num_correct = sum(1 for r in results if r["correct"])
		latencies = [r["latency"] for r in results if r["latency"] is not None]
		best = f"{min(latencies):.3f} ms" if latencies else "N/A"
		logger.info("Eval done: %d/%d correct, best latency: %s", num_correct, len(results), best)
		return results

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 2:
		print("Usage:")
		print("  python -m autocomp.backend.tpu.tpu_eval <file_path>")
		print("  python -m autocomp.backend.tpu.tpu_eval --ssh")
		print("  python -m autocomp.backend.tpu.tpu_eval --bench <file1> [file2 ...]")
		print("")
		print("Direct SSH mode (no gcloud) example:")
		print("  AUTOCOMP_TPU_TRANSPORT=ssh AUTOCOMP_TPU_SSH_HOST=<HOST> AUTOCOMP_TPU_SSH_USER=<USER> \\")
		print(" python -m autocomp.backend.tpu.tpu_eval --ssh")
		sys.exit(1)

	backend = TpuEvalBackend(
		tpu_name="tpu-node",
		tpu_zone="us-east5-a",
		tpu_project="ch-llm",
	)

	if sys.argv[1] == "--ssh":
		raise SystemExit(backend.open_tpu_shell())

	if sys.argv[1] == "--bench":
		import logging as _logging
		import time as _time
		_logging.basicConfig(level=_logging.INFO)
		files = sys.argv[2:]
		if not files:
			print("Usage: --bench <file1> [file2 ...]")
			sys.exit(1)
		code_strs = []
		for fp in files:
			code_strs.append(pathlib.Path(fp).read_text(encoding="utf-8"))
		prob = Prob(prob_type="tpu", prob_id="0")
		backend.ensure_tpu_vm()

		backend._jax_setup_done = False
		t0 = _time.perf_counter()
		results = backend.evaluate_code(prob, code_strs, simulator="")
		t1 = _time.perf_counter()

		print(f"\n{'='*60}")
		print(f"Evaluated {len(code_strs)} candidates in {t1 - t0:.1f}s")
		for i, r in enumerate(results):
			status = f"{r['latency']:.3f} ms (util {r['util'] or 0:.1f}%)" if r['correct'] else "FAIL"
			print(f"  candidate {i}: {status}")
		print(f"{'='*60}")
		sys.exit(0)

	file_path = sys.argv[1]

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

# $ python -m autocomp.backend.tpu.tpu_eval sols/tpu/0_matmul_baseline.py