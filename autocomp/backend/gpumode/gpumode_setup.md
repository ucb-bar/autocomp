# ⚙️ GPU Mode Backend Setup

## Reference Kernels

First, clone [reference-kernels](https://github.com/gpu-mode/reference-kernels.git), the official problem set repository for the [GPU MODE Leaderboard](https://gpumode.com).

```sh
git clone https://github.com/gpu-mode/reference-kernels.git
```

## Autocomp

In a different location, clone Autocomp and set up its Python dependencies:

```sh
git clone https://github.com/ucb-bar/autocomp
cd autocomp
pip install -e .
```

Then, point `GPU_MODE_DIR` in `autocomp/backend/gpumode/gpumode_eval.py` to the `problems` directory inside your clone of the reference-kernels repository.

### Adding a Problem

To register a new problem, add entries to the `prob_names` and `paths_to_probs` dicts at the top of `gpumode_eval.py`. For example, the `trimul` problem from the BioML competition is registered as:

```python
prob_names = {
    0: "trimul",
}
paths_to_probs = {
    0: GPU_MODE_DIR / "bioml" / "trimul",
}
```

Place the initial (unoptimized) solution file in `sols/gpumode/` following the naming convention `{prob_id}_{prob_name}.py` (e.g., `sols/gpumode/0_trimul.py`).

## Evaluation Modes

The backend supports two evaluation simulators, configured via the `simulator` parameter in `autocomp/search/search.py`:

### `gpumode-local`

Runs evaluation locally by calling the problem's `eval.py benchmark` script from the reference-kernels repo. This requires a local GPU and the appropriate PyTorch/CUDA environment. The generated code is written to `submission.py` inside the problem directory, and benchmark latencies are parsed from stdout.

### `gpumode-cli`

Runs evaluation remotely via the `popcorn-cli` tool, which submits code to the GPU MODE cloud infrastructure for benchmarking. Install `popcorn-cli` by following the [Getting Started guide](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/). Submissions are sent to the leaderboard specified in the CLI command, and results are written to an output file that is parsed for latency metrics.

## Running Autocomp

When running Autocomp, set the following in `autocomp/search/search.py`:
- `backend_name` should be `"gpumode"`.
- `agent_name` should be `"cuda"` (this is the default when `backend_name` is `"gpumode"`).
- `simulator` should be `"gpumode-local"` for local evaluation or `"gpumode-cli"` for remote evaluation.
- `prob_type` should be `"gpumode"`.
- `prob_id` should be the ID of the problem to optimize (e.g., `0` for `trimul`).

Autocomp will load the initial code from `sols/gpumode/`, and call the appropriate evaluation method to benchmark the generated code. The performance metric is latency (geometric mean across benchmark test cases), measured in microseconds.
