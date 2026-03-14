import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
x = np.random.rand(2048, 2048).astype(np.float32)
np.save("cumsum_x.npy", x)

subprocess.run([sys.executable, "beta1_cumsum_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_cumsum_test.py"], check=True, env=env)

out1 = np.load("out_beta1_cumsum.npy")
out2 = np.load("out_beta2_cumsum.npy")
print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))
