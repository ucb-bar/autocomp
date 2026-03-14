import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
N, C, H, W = 8, 512, 1, 2048
C_out = C
H_f, W_f = 1, 3
padding = ((0, 0), (1, 1))
img = np.random.rand(N, C, H, W).astype(np.float32)
filt = np.random.rand(C_out, 1, H_f, W_f).astype(np.float32)
np.save("conv1d_img.npy", img)
np.save("conv1d_filt.npy", filt)

subprocess.run([sys.executable, "beta1_conv1d_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_conv1d_test.py"], check=True, env=env)

out1 = np.load("out_beta1_conv1d.npy")
out2 = np.load("out_beta2_conv1d.npy")
print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))
