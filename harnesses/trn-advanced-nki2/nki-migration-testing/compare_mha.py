import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
num_heads = 8
d_head = 128
seq_len = 2048
NEG_INF = -9984.0
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

o2 = load_f32("out_beta2_mha_o.npy")
l2 = load_f32("out_beta2_mha_l.npy")
m2 = load_f32("out_beta2_mha_m.npy")

# Compute numpy float64 reference: causal softmax attention per head.
# Note: beta1_mha_ref has an SBUF aliasing bug with nl.affine_range double loops
# that causes it to compute incorrect results. We compare beta2 against the
# mathematically correct numpy reference instead.
ref_o = np.zeros((num_heads, seq_len, d_head), dtype=np.float64)
causal_mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float64))
for h in range(num_heads):
    qk = q[h].astype(np.float64) @ k[h].astype(np.float64)
    qk_masked = np.where(causal_mask_np, qk, NEG_INF)
    qk_max = np.max(qk_masked, axis=1, keepdims=True)
    exp_qk = np.exp(qk_masked - qk_max)
    softmax_qk = exp_qk / np.sum(exp_qk, axis=1, keepdims=True)
    ref_o[h] = softmax_qk @ v[h].astype(np.float64)

# Normalized attention output: o / exp(l - m) = o / ps = softmax(qk) @ v
norm_o2 = o2 / np.exp(l2 - m2)

print("[normalized attn output] allclose (atol=0.01):", np.allclose(norm_o2, ref_o, atol=0.01, rtol=0.0))
print("[normalized attn output] max abs diff:", np.max(np.abs(norm_o2 - ref_o)))
