# ⚙️ Trainium Backend Setup

## AWS Instance Setup

We use UC Berkeley CS 152's [Lab 6 Setup Instructions](https://github.com/ucb-152/lab6?tab=readme-ov-file) as a reference for how to set up a Trainium instance.

Begin by launching a `trn1.2xlarge` instance by following steps 1-3 [here](https://github.com/ucb-152/lab6/blob/main/AWS_SETUP.md).

After `ssh`ing into the instance, add the following line to your `~/.bashrc` file to make sure your environment is always activated:

```sh
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
```

Also run the command inside your current shell.

## Autocomp

Next, inside the Trainium instance and with the environment activated, clone Autocomp and set up its Python dependencies:

```sh
git clone https://github.com/ucb-bar/autocomp
cd autocomp
pip install -e .
```

## NKI v1 vs NKI v2

NKI (Neuron Kernel Interface) has two API versions. The evaluation backend auto-detects which version a test file uses based on its imports — no manual configuration needed.

| | NKI v1 | NKI v2 |
|---|---|---|
| **Import** | `import neuronxcc.nki as nki` | `import nki` |
| **Package** | Bundled in `neuronxcc` (Neuron SDK) | Standalone `nki` package |
| **Execution** | Baremetal / numpy arrays | PyTorch/XLA tensors (`torch_xla`) |
| **Agents** | `built:trn1-nki1` (Trn1)<br>`built:trn2-nki1` (Trn2) | `built:trn2-nki2` (Trn2) |
| **Problem suffixes** | `trn-tutorial-nki1`, `trn-advanced-nki1` | `trn-tutorial-nki2`, `trn-advanced-nki2`, `trn-internal` |

The key difference for Autocomp evaluation is that NKI v1 (baremetal) supports decoupled compile-then-execute: compilation runs in parallel on CPU, then candidates execute sequentially on-device. NKI v2 runs through PyTorch/XLA, where compilation and execution are coupled — candidates run in parallel across NeuronCores but each includes both compilation and execution overhead.

## Available Problems

Trainium has the following problem types (`prob_type` in `run_search.py`):

| `prob_type` | Description |
|---|---|
| `trn-tutorial-nki1` | Tutorial NKI v1 kernels from the nki-samples repo |
| `trn-tutorial-nki2` | Tutorial NKI v2 kernels (same problems, NKI v2 API) |
| `trn-advanced-nki1` | Advanced NKI v1 kernels from the nki-samples repo |
| `trn-advanced-nki2` | Advanced NKI v2 kernels (same problems, NKI v2 API) |
| `trn-internal` | Internal NKI kernels (DeltaNet, OpenFold3, FlashVSR, etc.), NKI v2 |

### Adding a New Problem

To add a new Trainium problem, place the initial (unoptimized) solution file in `sols/{prob_type}/` following the naming convention `{prob_id}_{name}_ref.py`. The test harness goes in `harnesses/{prob_type}/{prob_id}_{name}_test.py` with a `// SUBSTITUTE HERE` marker where generated code should be inserted.
