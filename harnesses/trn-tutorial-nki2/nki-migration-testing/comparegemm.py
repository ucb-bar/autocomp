import os
import subprocess
import sys

import numpy as np


def main():
  np.random.seed(0)

  # Shapes must satisfy NKI GEMM tiling constraints:
  # - K, M multiples of 128
  # - N multiple of 512
  K = 1024
  M = 512
  N = 1024

  lhsT = (np.random.randn(K, M) * (1.0 / np.sqrt(K))).astype(np.float16)
  rhs = (np.random.randn(K, N) * (1.0 / np.sqrt(K))).astype(np.float16)

  np.save("gemm_lhsT.npy", lhsT)
  np.save("gemm_rhs.npy", rhs)

  subprocess.run([sys.executable, "beta1_gemm_ref.py"], check=True)

  env = os.environ.copy()
  env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
  subprocess.run([sys.executable, "beta2_gemm_test.py"], check=True, env=env)

  out1 = np.load("out_beta1_gemm.npy")
  out2 = np.load("out_beta2_gemm.npy")

  # GEMM accumulates in fp32; output dtype is fp16, so allow a bit more error.
  atol = 5e-2
  rtol = 5e-2
  print("allclose:", np.allclose(out1, out2, atol=atol, rtol=rtol))
  print("max abs diff:", np.max(np.abs(out1.astype(np.float32) - out2.astype(np.float32))))


if __name__ == "__main__":
  main()

