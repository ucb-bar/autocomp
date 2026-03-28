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

## Available Problems

Trainium has the following problem types (`prob_type` in `run_search.py`):

| `prob_type` | Description |
|---|---|
| `trn-tutorial` | Tutorial NKI kernels from the nki-samples repo |
| `trn-advanced` | Advanced NKI kernels from the nki-samples repo |

### Adding a New Problem

To add a new Trainium problem, place the initial (unoptimized) solution file in `sols/{prob_type}/` following the naming convention `{prob_id}_{name}_ref.py`. The test harness goes in `tests/{prob_type}/{prob_id}_{name}_test.py` with a `// SUBSTITUTE HERE` marker where generated code should be inserted.
