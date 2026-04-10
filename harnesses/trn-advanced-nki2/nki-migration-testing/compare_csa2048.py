import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
d_head = 128
seq_len = 2048
NEG_INF = -9984.0
q = np.random.rand(seq_len, d_head).astype(np.float32)
k = np.random.rand(d_head, seq_len).astype(np.float32)
v = np.random.rand(seq_len, d_head).astype(np.float32)
np.save("csa2048_q.npy", q)
np.save("csa2048_k.npy", k)
np.save("csa2048_v.npy", v)

subprocess.run([sys.executable, "beta1_csa2048_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_csa2048_test.py"], check=True, env=env)

def load_f32(path):
    arr = np.load(path)
    if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
        # bfloat16 stored as void dtype — convert via uint16 shift
        return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)

o1 = load_f32("out_beta1_csa2048_o.npy")
l1 = load_f32("out_beta1_csa2048_l.npy")
m1 = load_f32("out_beta1_csa2048_m.npy")

o2 = load_f32("out_beta2_csa2048_o.npy")
l2 = load_f32("out_beta2_csa2048_l.npy")
m2 = load_f32("out_beta2_csa2048_m.npy")

# Normalized attention output: o / exp(l - m) = o / ps = softmax(qk) @ v
norm_o1 = o1 / np.exp(l1 - m1)
norm_o2 = o2 / np.exp(l2 - m2)

print("[normalized attn output] allclose (atol=0.01):", np.allclose(norm_o1, norm_o2, atol=0.01, rtol=0.0))
print("[normalized attn output] max abs diff:", np.max(np.abs(norm_o1 - norm_o2)))
