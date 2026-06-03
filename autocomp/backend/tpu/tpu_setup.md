# TPU Backend Setup

## TPU VM Setup

Autocomp evaluates generated Pallas kernels on a remote TPU VM via SSH.

To create a TPU VM via gcloud CLI:

```sh
gcloud alpha compute tpus tpu-vm create <tpu_name> \
    --zone=<zone> \
    --accelerator-type=v6e-1 \
    --version=v2-alpha-tpuv6e \
    --project=<project>
```

With **gcloud** transport (default), the backend auto-creates the VM using `AUTOCOMP_TPU_NAME`, `AUTOCOMP_TPU_ZONE`, `AUTOCOMP_TPU_PROJECT`, `AUTOCOMP_TPU_ACCELERATOR_TYPE`, and `AUTOCOMP_TPU_RUNTIME_VERSION`. For **direct SSH**:

```sh
export AUTOCOMP_TPU_TRANSPORT=ssh
export AUTOCOMP_TPU_SSH_HOST=10.0.0.42
export AUTOCOMP_TPU_SSH_USER=myuser
```

The backend uses Python 3.11 (override with `AUTOCOMP_TPU_PYTHON`) and, on the first run, creates a virtual environment on the TPU VM and installs `jax[tpu]==0.9.2` (plus `absl-py`) into it if the correct version is not present. The venv keeps the eval dependencies isolated from the system Python; its name defaults to `.autocomp_venv` (override with `AUTOCOMP_TPU_VENV`). Force reinstall with `AUTOCOMP_TPU_FORCE_PIP=1`.

## Connecting to TPU VMs Without External IPs

For security, you can run TPU VMs without external IP addresses (e.g. under a `constraints/compute.vmExternalIpAccess` org policy) and reach them over Identity-Aware Proxy (IAP) or their internal IP. Both are opt-in and off by default:

```sh
export AUTOCOMP_TPU_IAP=1           # tunnel ssh/scp through Cloud IAP
# or, for same-VPC / peered access:
export AUTOCOMP_TPU_INTERNAL_IP=1   # connect via the VM's internal IP
```

These are mutually exclusive. When either is set, the backend also creates new VMs with `--internal-ips`. `--tunnel-through-iap` is only available on the `gcloud alpha` track, so the backend switches tracks automatically when IAP is enabled. Optionally pin the network/subnetwork used at creation time with `AUTOCOMP_TPU_NETWORK` and `AUTOCOMP_TPU_SUBNETWORK`.

This requires one-time GCP setup on the project/subnet:

1. **Private Google Access** on the subnet, so the no-external-IP VM can reach Google APIs (including the JAX wheels at `storage.googleapis.com`):
   ```sh
   gcloud compute networks subnets update <subnet> \
       --region=<region> --enable-private-ip-google-access
   ```
2. **Firewall** allowing ingress on `tcp:22` from the IAP range `35.235.240.0/20`.
3. **IAM**: grant connecting users `roles/iap.tunnelResourceAccessor` (and `roles/tpu.admin`).

See [Connect to a TPU VM without a public IP address](https://cloud.google.com/tpu/docs/tpu-iap) for details.

## Running Autocomp on the TPU VM Itself

By default the orchestrator runs on your workstation and ships each evaluation to the TPU VM over scp + ssh. If you instead run Autocomp *on the TPU VM*, set the **local** transport so evaluations run as local subprocesses with no network roundtrip:

```sh
export AUTOCOMP_TPU_TRANSPORT=local
```

With `local` transport the backend copies scripts with a plain file copy, executes them via the local shell, and reads results directly from disk. VM creation (`ensure_tpu_vm`) is a no-op, and the IAP / internal-IP / SSH settings are ignored. Everything else is unchanged: it still creates the `.autocomp_venv` (override with `AUTOCOMP_TPU_VENV`) and installs `jax[tpu]==0.9.2` + `absl-py` into it on first use.

To set this up on the VM, install Autocomp into that same venv so both the orchestrator and the eval subprocess share it:

```sh
# on the TPU VM, from the repo root
~/.autocomp_venv/bin/python -m pip install -e .   # creates the venv first if needed
AUTOCOMP_TPU_TRANSPORT=local ~/.autocomp_venv/bin/python -m autocomp.backend.tpu.tpu_eval --bench sols/tpu/0_matmul_baseline.py
```

This avoids per-eval scp/ssh overhead, which is the main win during a long search. Note that the orchestrator's LLM API calls still need outbound internet on the VM — on a no-external-IP VM (the IAP scenario above) you would need Cloud NAT, since Private Google Access only covers Google APIs.

## How Evaluation Works

The evaluator builds a single batch script that:

1. Inlines the **test harness** (`harnesses/tpu/test{prob_id}.py`) with imports and `if __name__` stripped.
2. Base64-encodes each LLM-generated implementation and `exec`s it, which defines a `solution()` function.
3. Calls `_run_autocomp_harness()` (from the harness) which invokes `solution()` for warmup, correctness, and benchmarking.
4. Parses stdout for `Latency:` and `Utilization:` lines.

## Adding a New Problem

Each problem needs up to three files. See the matmul problem (`prob_id=0`) as a reference.

1. **Test harness** — `harnesses/tpu/test{prob_id}.py`: self-contained file defining inputs, reference, and `_run_autocomp_harness()`.
2. **Baseline solution** — `sols/tpu/{prob_id}_*.py`: defines `solution(...)` matching the harness signature. Starting point for the LLM.
3. **Context** (optional) — `harnesses/tpu/context{prob_id}.txt`: one-liner describing the problem for the LLM prompt.
4. **Wire up** `run_search.py`: set `prob_type="tpu"` and `prob_id="{prob_id}"`.

## CLI Usage

`tpu_eval.py` can be run directly for debugging:

```sh
python -m autocomp.backend.tpu.tpu_eval <file_path>           # run a single file
python -m autocomp.backend.tpu.tpu_eval --ssh                  # interactive SSH shell
python -m autocomp.backend.tpu.tpu_eval --bench <file1.py> ... # batch-evaluate
```

Set `AUTOCOMP_TPU_TRANSPORT=local` to run these directly on the TPU VM with no scp/ssh.
