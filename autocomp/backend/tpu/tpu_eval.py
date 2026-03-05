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
import time

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


def _ensure_tpu_vm_running(
	tpu_name: str,
	zone: str,
	accelerator_type: str = "v6e-1",
	version: str = "v2-alpha-tpuv6e",
	project: str = "auto-tpu",
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
		self.tpu_name = tpu_name or "my-v6e-tpu"
		self.tpu_zone = tpu_zone or "us-east5-a"
		# Ensure gcloud commands target the same project the TPU was created in.
		# Default matches the create command in _ensure_tpu_vm_running().
		self.tpu_project = tpu_project or os.getenv("AUTOCOMP_TPU_PROJECT") or "auto-tpu"
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

	def open_tpu_shell(self) -> int:
		cmd = self._build_ssh_cmd(remote_command=None, allocate_tty=True, batch_mode=False)
		logger.info("Opening TPU shell: %s", " ".join(cmd))
		return subprocess.call(cmd)


class TpuEvalBackend(TpuHardwareBackend):
	pass

	def ensure_tpu_vm(self) -> None:

		#if direct ssh
		if self._transport_mode() != "gcloud":
			return
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

	def _extract_spad_acc_feedback_lines(self, output_text: str) -> list[str]:
		lines: list[str] = []
		for raw in output_text.split("\n"):
			l = raw.strip()
			if not l:
				continue
			ll = l.lower()
			if any(k in ll for k in ("spad", "scratchpad", "vmem", "accumulator", "acc " , "acc=" , "acc:")) and any(
				k in ll for k in ("util", "usage", "used", "bytes", "kb", "mb", "capacity")
			):
				lines.append(l)
		seen = set()
		uniq: list[str] = []
		for l in lines:
			if l in seen:
				continue
			seen.add(l)
			uniq.append(l)
		return uniq

	def _append_autocomp_hw_feedback_block(
		self, *, stdout_text: str, stderr_text: str
	) -> tuple[str, float | None, list[str]]:
		util = self._extract_util_percent(stdout_text) or self._extract_util_percent(stderr_text)
		spad_acc_lines = self._extract_spad_acc_feedback_lines(stdout_text)
		if not spad_acc_lines:
			spad_acc_lines = self._extract_spad_acc_feedback_lines(stderr_text)

		if util is None and not spad_acc_lines:
			return stdout_text, None, []

		block_lines: list[str] = ["", "AUTOCOMP TPU FEEDBACK:"]
		if util is not None:
			block_lines.append(f"util: {util}%")
		for l in spad_acc_lines[:8]:
			block_lines.append(l)
		block_lines.append("END AUTOCOMP TPU FEEDBACK")
		block = "\n".join(block_lines) + "\n"
		return stdout_text + block, util, spad_acc_lines

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
		logger.info("Running command %s", " ".join(cmd))
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

	def _run_scp_from_remote(self, remote_path: str, local_path: pathlib.Path) -> subprocess.CompletedProcess:
		#download file from tpu
		local_path.parent.mkdir(parents=True, exist_ok=True)
		source = (
			f"{self.tpu_name}:{remote_path}"
			if self._transport_mode() == "gcloud"
			else f"{self._ssh_target()}:{remote_path}"
		)
		cmd = self._build_scp_cmd(source=source, dest=local_path.resolve().as_posix())
		logger.info("Running command %s", " ".join(cmd))
		proc = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=300,
			stdin=subprocess.DEVNULL,
		)
		if proc.returncode != 0:
			logger.error("scp (download) failed (exit %s). stderr:\n%s", proc.returncode, proc.stderr)
		return proc

	def _run_ssh(self, remote_command: str) -> subprocess.CompletedProcess:
		#run command on tpu
		cmd = self._build_ssh_cmd(remote_command=remote_command, allocate_tty=False, batch_mode=True)
		logger.info("Running command %s", " ".join(cmd))
		return subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)

	def _remote_run_id(self) -> str:
		return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + secrets.token_hex(4)

	def _remote_eval_dir(self, run_id: str) -> str:
		return f"/tmp/autocomp_eval/{run_id}"

	def _remote_run_python_to_files_command(self, remote_filename: str, remote_dir: str) -> str:
		prog_stdout = f"{remote_dir}/program_stdout.txt"
		prog_stderr = f"{remote_dir}/program_stderr.txt"
		exit_code = f"{remote_dir}/program_exit_code.txt"
		setup_log = f"{remote_dir}/setup_log.txt"
		setup_exit_code = f"{remote_dir}/setup_exit_code.txt"

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

	def _read_text_file(self, path: pathlib.Path) -> str:
		try:
			return path.read_text(encoding="utf-8", errors="replace")
		except FileNotFoundError:
			return ""

	def _wrap_candidate_code_for_eval(self, code: str) -> str:
		imports = """
		
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

"""
		harness = r"""
def _default_peak_tflops(dtype) -> float:
    override = os.getenv("AUTOCOMP_TPU_PEAK_TFLOPS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    dt = jnp.dtype(dtype)
    if dt in (jnp.bfloat16, jnp.float16):
        return float(os.getenv("AUTOCOMP_TPU_V6E_BF16_TFLOPS", "918"))
    if dt == jnp.float32:
        return float(os.getenv("AUTOCOMP_TPU_V6E_FP32_TFLOPS", "459"))
    return float(os.getenv("AUTOCOMP_TPU_V6E_BF16_TFLOPS", "918"))


def _print_hw_feedback(*, M: int, K: int, N: int, dtype, latency_ms: float) -> None:
    flops = 2.0 * M * K * N
    secs = max(1e-9, latency_ms / 1000.0)
    achieved_tflops = flops / secs / 1e12
    peak_tflops = max(1e-9, _default_peak_tflops(dtype))
    util_pct = max(0.0, min(100.0, achieved_tflops / peak_tflops * 100.0))

    print(f"Utilization: {util_pct:.2f}%")
    print(f"Achieved TFLOP/s: {achieved_tflops:.3f} (peak={peak_tflops:.3f})")

    itemsize = np.dtype(np.array(0, dtype=np.dtype(jnp.dtype(dtype))).dtype).itemsize
    scratchpad_bytes = (K + N) * itemsize
    acc_bytes = N * np.dtype(np.float32).itemsize
    print(f"Scratchpad utilization is {scratchpad_bytes/1024.0:.1f}KB (estimate).")
    print(f"Accumulator utilization is {acc_bytes/1024.0:.1f}KB (estimate).")


def _run_autocomp_harness():
    # Fixed problem size for consistent evaluation across all iterations/candidates.
    # Intentionally not configurable so search results are comparable run-to-run.
    M = 1024
    K = 1024
    N = 1024
    dtype_str = os.getenv("AUTOCOMP_TPU_DTYPE", "float32")
    dtype = getattr(jnp, dtype_str, jnp.float32)

    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)

    print(f"Running TPU test() ({M}x{K}x{N}, dtype={dtype_str})...")

    # Warmup: compile + run once.
    out = test(x, y)
    jax.block_until_ready(out)

    # Correctness check.
    baseline = jnp.matmul(x, y)
    if not jnp.allclose(baseline, out, atol=1e-2, rtol=1e-2):
        print("FAIL: Verification failed.")
        raise SystemExit(1)

    # Timed run.
    t0 = time.perf_counter()
    out2 = test(x, y)
    jax.block_until_ready(out2)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    print(f"Latency: {latency_ms:.3f} ms")
    _print_hw_feedback(M=M, K=K, N=N, dtype=dtype, latency_ms=latency_ms)
    print("SUCCESS: test() matches JAX baseline.")


if __name__ == "__main__":
    _run_autocomp_harness()
"""
		return imports + "\n" + code.rstrip() + "\n" + harness.lstrip()

	def run_file_on_tpu(self, file_path: str | pathlib.Path, remote_filename: str = "remote_upload.py") -> dict:
		
		self.ensure_tpu_vm()
		local_file = pathlib.Path(file_path)
		if not local_file.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		result_dict = {
			"correct": False,
			"latency": None,
			"stdout": "",
			"stderr": "",
			"util": None,
			"spad_acc_stats": [],
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
		local_out_dir = pathlib.Path(__file__).parent / "eval_outputs" / run_id
		local_stdout_path = local_out_dir / "program_stdout.txt"
		local_stderr_path = local_out_dir / "program_stderr.txt"
		local_exit_code_path = local_out_dir / "program_exit_code.txt"

		# Best-effort downloads (even if the SSH command had noise on stderr).
		self._run_scp_from_remote(f"{remote_dir}/program_stdout.txt", local_stdout_path)
		self._run_scp_from_remote(f"{remote_dir}/program_stderr.txt", local_stderr_path)
		self._run_scp_from_remote(f"{remote_dir}/program_exit_code.txt", local_exit_code_path)

		prog_stdout = self._read_text_file(local_stdout_path)
		prog_stderr = self._read_text_file(local_stderr_path)
		exit_code_text = self._read_text_file(local_exit_code_path).strip()
		try:
			prog_rc = int(exit_code_text) if exit_code_text else 1
		except ValueError:
			prog_rc = 1

		enriched_stdout, util, spad_acc_lines = self._append_autocomp_hw_feedback_block(
			stdout_text=prog_stdout, stderr_text=prog_stderr
		)
		result_dict["stdout"] = enriched_stdout
		result_dict["stderr"] = prog_stderr if prog_stderr else (run_proc.stderr or "")
		result_dict["util"] = util
		result_dict["spad_acc_stats"] = spad_acc_lines

		if prog_rc != 0:
			logger.error(
				"File failed to run on TPU (program exit %s). Output saved under %s. SSH stderr:\n%s",
				prog_rc,
				local_out_dir,
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

	def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
		self.ensure_tpu_vm()

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		eval_dir = pathlib.Path(__file__).parent / "eval_outputs" / timestamp
		eval_dir.mkdir(parents=True, exist_ok=True)
		# If the caller didn't pass code strings, fall back to the bundled test file.
		if not code_strs:
			local_file = pathlib.Path(__file__).parent / "code_0.py"
			if not local_file.exists():
				raise FileNotFoundError(f"Fixed TPU eval file not found: {local_file}")
			code_strs = [local_file.read_text(encoding="utf-8", errors="replace")]

		results: list[dict] = []
		for idx, code in enumerate(code_strs):
			code_to_run = self._wrap_candidate_code_for_eval(code)
			result_dict = {
				"correct": False,
				"latency": None,
				"stdout": "",
				"stderr": "",
				"util": None,
				"spad_acc_stats": [],
			}

			# Write candidate code locally, upload, run, then scp back outputs.
			local_candidate = eval_dir / f"candidate_{idx}.py"
			local_candidate.write_text(code_to_run, encoding="utf-8", errors="replace")

			run_id = self._remote_run_id()
			remote_dir = self._remote_eval_dir(run_id)
			remote_name = f"autocomp_candidate_{run_id}_{idx}.py"

			try:
				scp_proc = self._run_scp(local_candidate.resolve(), remote_name)
			except subprocess.TimeoutExpired:
				logger.error("Candidate %s upload timed out after 300 seconds", idx)
				result_dict["stderr"] = "Upload timeout"
				results.append(result_dict)
				continue

			if scp_proc.returncode != 0:
				logger.error("Candidate %s upload failed", idx)
				result_dict["stdout"] = scp_proc.stdout
				result_dict["stderr"] = scp_proc.stderr
				results.append(result_dict)
				continue

			remote_cmd = self._remote_run_python_to_files_command(remote_name, remote_dir)
			try:
				run_proc = self._run_ssh(remote_cmd)
			except subprocess.TimeoutExpired:
				logger.error("Candidate %s execution timed out after 600 seconds", idx)
				result_dict["stderr"] = "Execution timeout"
				results.append(result_dict)
				continue

			local_stdout_path = eval_dir / f"candidate_{idx}_program_stdout.txt"
			local_stderr_path = eval_dir / f"candidate_{idx}_program_stderr.txt"
			local_exit_code_path = eval_dir / f"candidate_{idx}_program_exit_code.txt"
			local_setup_log_path = eval_dir / f"candidate_{idx}_setup_log.txt"
			local_setup_exit_code_path = eval_dir / f"candidate_{idx}_setup_exit_code.txt"

			self._run_scp_from_remote(f"{remote_dir}/program_stdout.txt", local_stdout_path)
			self._run_scp_from_remote(f"{remote_dir}/program_stderr.txt", local_stderr_path)
			self._run_scp_from_remote(f"{remote_dir}/program_exit_code.txt", local_exit_code_path)
			self._run_scp_from_remote(f"{remote_dir}/setup_log.txt", local_setup_log_path)
			self._run_scp_from_remote(f"{remote_dir}/setup_exit_code.txt", local_setup_exit_code_path)

			prog_stdout = self._read_text_file(local_stdout_path)
			prog_stderr = self._read_text_file(local_stderr_path)
			exit_code_text = self._read_text_file(local_exit_code_path).strip()
			try:
				prog_rc = int(exit_code_text) if exit_code_text else 1
			except ValueError:
				prog_rc = 1

			enriched_stdout, util, spad_acc_lines = self._append_autocomp_hw_feedback_block(
				stdout_text=prog_stdout, stderr_text=prog_stderr
			)
			result_dict["stdout"] = enriched_stdout
			result_dict["stderr"] = prog_stderr if prog_stderr else (run_proc.stderr or "")
			result_dict["util"] = util
			result_dict["spad_acc_stats"] = spad_acc_lines

			# Always write a per-candidate combined output for debugging.
			output_path = eval_dir / f"candidate_{idx}_output.txt"
			with open(output_path, "w", encoding="utf-8", errors="replace") as f:
				f.write("=== PROGRAM STDOUT ===\n")
				f.write(enriched_stdout)
				f.write("\n=== PROGRAM STDERR ===\n")
				f.write(prog_stderr)
				f.write("\n=== SSH STDERR ===\n")
				f.write(run_proc.stderr or "")

			if prog_rc != 0:
				logger.error("Candidate %s failed on TPU (program exit %s). See %s.", idx, prog_rc, output_path)
				results.append(result_dict)
				continue

			latency = self._extract_latency(prog_stdout)
			if latency is None:
				logger.error("Candidate %s did not produce latency output. See %s.", idx, output_path)
				results.append(result_dict)
				continue

			result_dict["correct"] = True
			result_dict["latency"] = latency
			results.append(result_dict)

		# If we fell back to a single fixed file, replicate to preserve legacy behavior.
		if len(results) == 1 and len(code_strs) == 1:
			logger.info("Fixed file latency: %s", results[0].get("latency"))
		return results

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 2:
		print("Usage:")
		print("  python -m autocomp.backend.tpu.tpu_eval <file_path>")
		print("  python -m autocomp.backend.tpu.tpu_eval --ssh")
		print("")
		print("Direct SSH mode (no gcloud) example:")
		print("  AUTOCOMP_TPU_TRANSPORT=ssh AUTOCOMP_TPU_SSH_HOST=<HOST> AUTOCOMP_TPU_SSH_USER=<USER> \\")
		print(" python -m autocomp.backend.tpu.tpu_eval --ssh")
		sys.exit(1)

	backend = TpuEvalBackend(
		tpu_name="my-v6e-tpu",
		tpu_zone="us-east5-a",
	)

	if sys.argv[1] == "--ssh":
		raise SystemExit(backend.open_tpu_shell())

	file_path = sys.argv[1]

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