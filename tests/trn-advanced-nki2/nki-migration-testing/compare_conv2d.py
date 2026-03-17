import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
batch_size, in_channels, height, width, out_channels, filter_h, filter_w = 16, 128, 128, 128, 512, 3, 3
img = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
filter_weights = np.random.rand(out_channels, in_channels, filter_h, filter_w).astype(np.float32)
np.save("conv2d_img.npy", img)
np.save("conv2d_filt.npy", filter_weights)

subprocess.run([sys.executable, "beta1_conv2d_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_conv2d_test.py"], check=True, env=env)

out1 = np.load("out_beta1_conv2d.npy")
out2 = np.load("out_beta2_conv2d.npy")
print("allclose:", np.allclose(out1, out2, atol=1e-4, rtol=1e-2))
print("max abs diff:", np.max(np.abs(out1 - out2)))
