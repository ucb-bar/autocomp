import os
import subprocess
import sys

import numpy as np


def main():
  np.random.seed(0)

  # Keep seqlen smaller than the tutorial default for faster iteration.
  d_head = 128
  seqlen = 1024  # multiple of 512 and 128

  q = (np.random.random_sample([d_head, seqlen]).astype(np.float32) - 0.5) * 2
  k = (np.random.random_sample([d_head, seqlen]).astype(np.float32) - 0.5) * 2
  v = (np.random.random_sample([d_head, seqlen]).astype(np.float32) - 0.5) * 2

  np.save("attn_q.npy", q)
  np.save("attn_k.npy", k)
  np.save("attn_v.npy", v)

  subprocess.run([sys.executable, "beta1_attention_ref.py"], check=True)

  env = os.environ.copy()
  env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
  subprocess.run([sys.executable, "beta2_attention_test.py"], check=True, env=env)

  out1 = np.load("out_beta1_attention.npy")
  out2 = np.load("out_beta2_attention.npy")

  atol = 1e-2
  rtol = 1e-2
  print("allclose:", np.allclose(out1, out2, atol=atol, rtol=rtol))
  print("max abs diff:", np.max(np.abs(out1.astype(np.float32) - out2.astype(np.float32))))


if __name__ == "__main__":
  main()

