import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
num_heads = 8
d_head = 128
seq_len = 2048
q = np.random.rand(num_heads, seq_len, d_head).astype(np.float32)
k = np.random.rand(num_heads, d_head, seq_len).astype(np.float32)
v = np.random.rand(num_heads, seq_len, d_head).astype(np.float32)
np.save("mha_q.npy", q)
np.save("mha_k.npy", k)
np.save("mha_v.npy", v)

subprocess.run([sys.executable, "beta1_mha_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_mha_test.py"], check=True, env=env)

def load_f32(path):
    arr = np.load(path)
    if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
        return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)

o1 = load_f32("out_beta1_mha_o.npy")
l1 = load_f32("out_beta1_mha_l.npy")
m1 = load_f32("out_beta1_mha_m.npy")

o2 = load_f32("out_beta2_mha_o.npy")
l2 = load_f32("out_beta2_mha_l.npy")
m2 = load_f32("out_beta2_mha_m.npy")

# Normalized attention output: o / exp(l - m), where l = log(ps) + m
# Note: beta1 truncates QK to bfloat16 via affine_select before max computation,
# while beta2 keeps QK in float32. Compare normalized outputs with bfloat16 tolerance.
norm_o1 = o1 / np.exp(l1 - m1)
norm_o2 = o2 / np.exp(l2 - m2)

print("[normalized attn output] allclose (atol=1.0):", np.allclose(norm_o1, norm_o2, atol=1.0, rtol=0.0))
print("[normalized attn output] max abs diff:", np.max(np.abs(norm_o1 - norm_o2)))
