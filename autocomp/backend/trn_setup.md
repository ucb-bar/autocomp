# ⚙️ Trainium Backend Setup

### ⚠️ Note: Trainium documentation is still under construction.

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
