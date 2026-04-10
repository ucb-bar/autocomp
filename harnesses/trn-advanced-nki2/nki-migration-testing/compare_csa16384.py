import numpy as np
import subprocess
import sys
import os

np.random.seed(42)
B_P_SIZE = 128
B_D_SIZE = 128
B_F_SIZE = 512
Q_TILE_SIZE = 128
seq_tile_size = 16384
num_q_tiles = seq_tile_size // B_P_SIZE

# Use a representative q_tile_idx
q_tile_idx = num_q_tiles // 2

q = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
k = np.random.rand(B_D_SIZE, seq_tile_size).astype(np.float32)
v = np.random.rand(B_P_SIZE, seq_tile_size // B_P_SIZE, B_D_SIZE).astype(np.float32)
olm_prev = np.random.rand(Q_TILE_SIZE, B_D_SIZE + 2).astype(np.float32)

tile_mask = np.zeros((B_P_SIZE, seq_tile_size), dtype=bool)
for i in range(B_P_SIZE):
    q_pos = q_tile_idx * B_P_SIZE + i
    tile_mask[i, :q_pos + 1] = True

np.save("csa16384_q_tile_idx.npy", np.array(q_tile_idx))
np.save("csa16384_q.npy", q)
np.save("csa16384_k.npy", k)
np.save("csa16384_v.npy", v)
np.save("csa16384_olm_prev.npy", olm_prev)
np.save("csa16384_tile_mask.npy", tile_mask)

subprocess.run([sys.executable, "beta1_csa16384_ref.py"], check=True)

env = os.environ.copy()
env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
subprocess.run([sys.executable, "beta2_csa16384_test.py"], check=True, env=env)

def load_f32(path):
    arr = np.load(path)
    if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
        return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)

olm1 = load_f32("out_beta1_csa16384.npy")
olm2 = load_f32("out_beta2_csa16384.npy")

B_D_SIZE = 128
# olm layout: [:, 0:128] = o, [:, 128] = l (running sum), [:, 129] = m (running max)
o1, l1 = olm1[:, 0:B_D_SIZE], olm1[:, B_D_SIZE:B_D_SIZE+1]
o2, l2 = olm2[:, 0:B_D_SIZE], olm2[:, B_D_SIZE:B_D_SIZE+1]

# Normalized output: o / l (l is the running sum of softmax weights, not log-form)
# beta1 uses float32 for QK (nl.where with acc_type=float32), so precision is similar
norm_o1 = o1 / l1
norm_o2 = o2 / l2

print("[normalized attn output] allclose (atol=0.1):", np.allclose(norm_o1, norm_o2, atol=0.1, rtol=0.0))
print("[normalized attn output] max abs diff:", np.max(np.abs(norm_o1 - norm_o2)))
