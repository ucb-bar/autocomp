import os
import subprocess
import sys

import numpy as np


def main():
  np.random.seed(0)

  # Use smaller seqlen than tutorial default for faster checks.
  seqlen = 1024  # multiple of 128
  d_head = 64

  q = np.random.rand(seqlen, d_head).astype(np.float32)
  k = np.random.rand(seqlen, d_head).astype(np.float32)
  v = np.random.rand(seqlen, d_head).astype(np.float32)

  np.save("sdattn_q.npy", q)
  np.save("sdattn_k.npy", k)
  np.save("sdattn_v.npy", v)

  subprocess.run([sys.executable, "beta1_sdattention_ref.py"], check=True)

  env = os.environ.copy()
  env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
  subprocess.run([sys.executable, "beta2_sdattention_test.py"], check=True, env=env)

  out1 = np.load("out_beta1_sdattention.npy")
  out2 = np.load("out_beta2_sdattention.npy")

  atol = 1e-2
  rtol = 1e-2
  print("allclose:", np.allclose(out1, out2, atol=atol, rtol=rtol))
  print("max abs diff:", np.max(np.abs(out1.astype(np.float32) - out2.astype(np.float32))))


if __name__ == "__main__":
  main()

