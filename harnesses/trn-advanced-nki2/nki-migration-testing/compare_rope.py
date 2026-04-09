import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
x_in = np.random.rand(128, 4096).astype(np.float32)
cos = np.random.rand(64, 4096).astype(np.float32)
sin = np.random.rand(64, 4096).astype(np.float32)
np.save("rope_x_in.npy", x_in)
np.save("rope_cos.npy", cos)
np.save("rope_sin.npy", sin)

subprocess.run([sys.executable, "beta1_rope_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_rope_test.py"], check=True, env=env)

out1 = np.load("out_beta1_rope.npy")
out2 = np.load("out_beta2_rope.npy")
print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))
