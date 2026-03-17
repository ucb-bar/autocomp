"""Diagnostic: verify nl.load with two loop variable offsets loads correct causal mask values."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def test_mask_load(causal_mask, seq_len=512, d_head=128):
    """Load causal_mask tiles and copy to output to verify two-var load."""
    B_P_SIZE = 128; B_F_SIZE = 512
    num_q_tiles = seq_len // B_P_SIZE
    num_k_tiles = seq_len // B_F_SIZE
    out_mask = nl.ndarray((seq_len, seq_len), dtype=np.float32, buffer=nl.shared_hbm)
    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_start = q_tile_idx * B_P_SIZE
        for k_i in nl.affine_range(num_k_tiles):
            k_start = k_i * B_F_SIZE
            tile = nl.load(causal_mask[q_start:q_start+B_P_SIZE, k_start:k_start+B_F_SIZE])
            nl.store(out_mask[q_start:q_start+B_P_SIZE, k_start:k_start+B_F_SIZE], value=tile)
    return out_mask

if __name__ == "__main__":
    seq_len = 1024  # 8 q tiles x 2 k tiles
    causal_np = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        causal_np[i, :i+1] = 1.0

    out_np = np.zeros_like(causal_np)
    result = test_mask_load(causal_np, seq_len=seq_len, d_head=128)

    loaded = np.array(result)
    print("Loaded matches original:", np.allclose(loaded, causal_np))
    print("Max diff:", np.max(np.abs(loaded - causal_np)))
    # Check specific tiles
    for q_tile in range(8):
        for k_tile in range(2):
            q_start = q_tile * 128
            k_start = k_tile * 512
            tile_orig = causal_np[q_start:q_start+128, k_start:k_start+512]
            tile_loaded = loaded[q_start:q_start+128, k_start:k_start+512]
            match = np.allclose(tile_orig, tile_loaded)
            if not match:
                print(f"q_tile={q_tile}, k_tile={k_tile}: MISMATCH! max_diff={np.max(np.abs(tile_orig-tile_loaded)):.3f}")
            else:
                print(f"q_tile={q_tile}, k_tile={k_tile}: OK (sum_orig={tile_orig.sum():.0f}, sum_loaded={tile_loaded.sum():.0f})")
