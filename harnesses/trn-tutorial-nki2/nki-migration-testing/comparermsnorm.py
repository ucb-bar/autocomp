import os
import numpy as np
import subprocess
import sys

np.random.seed(0)
a = np.random.rand(512, 4096).astype(np.float32)
g = np.random.rand(4096).astype(np.float32)

np.save("a.npy", a)
np.save("g.npy", g)

# Run Beta 1 reference
subprocess.run([sys.executable, "beta1_ref.py"], check=True)

# Run Beta 2 test with explicit target override
env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"   # use trn2 on Trn2, trn3 on Trn3
subprocess.run([sys.executable, "beta2_test.py"], check=True, env=env)

out1 = np.load("out_beta1.npy")
out2 = np.load("out_beta2.npy")

print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))