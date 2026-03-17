import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
H, W = 4096, 4096
pool_size = 3
input_tensor = np.random.rand(H, W).astype(np.float32)
np.save("maxpool_input.npy", input_tensor)
np.save("maxpool_pool_size.npy", np.array(pool_size))

subprocess.run([sys.executable, "beta1_maxpool_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_maxpool_test.py"], check=True, env=env)

out1 = np.load("out_beta1_maxpool.npy")
out2 = np.load("out_beta2_maxpool.npy")
print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))
