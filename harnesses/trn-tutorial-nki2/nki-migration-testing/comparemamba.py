import os
import subprocess
import sys

import numpy as np


def main():
  np.random.seed(0)

  # Use the same shapes as `3_mamba_test.py` to exercise the intended tiling.
  batch = 1
  seq_len = 2048
  channels = 256
  state_size = 16
  dtype = np.float32

  delta = np.random.rand(batch, channels, seq_len).astype(dtype)
  u = np.random.rand(batch, channels, seq_len).astype(dtype)
  A = (-np.random.rand(channels, state_size)).astype(dtype)
  B = np.random.rand(batch, state_size, seq_len).astype(dtype)
  C = np.random.rand(batch, state_size, seq_len).astype(dtype)

  np.save("mamba_delta.npy", delta)
  np.save("mamba_u.npy", u)
  np.save("mamba_A.npy", A)
  np.save("mamba_B.npy", B)
  np.save("mamba_C.npy", C)

  subprocess.run([sys.executable, "beta1_mamba_ref.py"], check=True)

  env = os.environ.copy()
  env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
  subprocess.run([sys.executable, "beta2_mamba_test.py"], check=True, env=env)

  out1 = np.load("out_beta1_mamba.npy")
  out2 = np.load("out_beta2_mamba.npy")

  atol = 1e-3
  rtol = 1e-3
  print("allclose:", np.allclose(out1, out2, atol=atol, rtol=rtol))
  print("max abs diff:", np.max(np.abs(out1 - out2)))


if __name__ == "__main__":
  main()

