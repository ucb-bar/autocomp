import os
import subprocess
import sys

import numpy as np


def main():
  np.random.seed(0)

  # Keep shapes moderate so comparisons run quickly, but still exercise tiling/chunking.
  num_rows = 512
  num_cols = 4096

  x = np.random.randn(num_rows, num_cols).astype(np.float32)
  gamma = np.random.randn(num_cols).astype(np.float32)
  beta = np.random.randn(num_cols).astype(np.float32)

  np.save("x.npy", x)
  np.save("gamma.npy", gamma)
  np.save("beta.npy", beta)

  # Run Beta 1 reference.
  subprocess.run([sys.executable, "beta1_layernorm_ref.py"], check=True)

  # Run Beta 2 test with explicit target override.
  env = os.environ.copy()
  env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
  subprocess.run([sys.executable, "beta2_layernorm_test.py"], check=True, env=env)

  out1 = np.load("out_beta1_layernorm.npy")
  out2 = np.load("out_beta2_layernorm.npy")

  atol = 1e-3
  rtol = 5e-3
  print("allclose:", np.allclose(out1, out2, atol=atol, rtol=rtol))
  print("max abs diff:", np.max(np.abs(out1 - out2)))


if __name__ == "__main__":
  main()

