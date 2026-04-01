"""
Custom NKI flash attention kernel with FP32 softmax for FlashVSR.

Keeps exp(scores) in fp32 throughout the P@V matmul, matching CUDA flash
attention's precision. The standard nkilib attention_cte kernel truncates
exp to bf16, causing temporal quality degradation in one-step diffusion.

Design:
  - Flash attention tiling: processes K/V in chunks of 512 (K_TILE)
  - Online softmax: running max + running sum (Milakov & Gimelshein)
  - FP32 exp from computation through P@V matmul (no bf16 truncation)
  - V tiles upcast bf16->fp32 via tensor_copy before matmul
  - Non-causal only (no masking logic)
  - Uses intentional PSUM accumulation for P@V across V-tile chunks

Layout (matches attention_cte defaults):
  Q: (batch, seqlen_q, d_head)   -- bf16
  K: (batch, d_head, seqlen_kv)  -- bf16 (transposed)
  V: (batch, seqlen_kv, d_head)  -- bf16
  Output: (batch, seqlen_q, d_head) -- bf16
"""

import nki
import nki.language as nl
import nki.isa as nisa


# Tile size constants
PMAX = 128  # nl.tile_size.pmax -- partition dimension
K_TILE = 512  # K/V processing chunk (nl.tile_size.gemm_moving_fmax)
V_TILE = 128  # V sub-tile for matmul (nl.tile_size.gemm_stationary_fmax)
LARGE_NEG = -9.984e3  # Initial max value (within bf16 range, safe for fp32)


@nki.jit
def fp32_exp_attention(q, k, v, scale: float = 1.0):
    """Flash attention with fp32 softmax (non-causal).

    Args:
        q: (batch, seqlen_q, d_head) bf16
        k: (batch, d_head, seqlen_kv) bf16 (transposed)
        v: (batch, seqlen_kv, d_head) bf16
        scale: attention scale factor (typically 1/sqrt(d_head))

    Returns:
        output: (batch, seqlen_q, d_head) bf16
    """
    batch, seqlen_q, d_head = q.shape
    _, _, seqlen_kv = k.shape

    assert d_head == PMAX, f"d_head must be {PMAX}"
    assert seqlen_q % PMAX == 0, f"seqlen_q must be divisible by {PMAX}"
    assert seqlen_kv % K_TILE == 0, f"seqlen_kv must be divisible by {K_TILE}"

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE
    n_v_tiles_per_chunk = K_TILE // V_TILE  # 512 / 128 = 4

    # Output in HBM
    output = nl.ndarray((batch, seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # Process each batch independently
    for b in nl.affine_range(batch):
        # Process Q in tiles of PMAX=128
        for q_tile_idx in nl.affine_range(n_q_tiles):
            q_offset = q_tile_idx * PMAX

            # Load Q tile into SBUF: shape (PMAX, d_head) = (128, 128)
            q_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q[b, nl.ds(q_offset, PMAX), :])

            # Apply scale to Q and upcast to fp32 (fused scale + cast)
            q_scaled = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(q_scaled, q_tile, op0=nl.multiply, operand0=scale)

            # Running statistics for online softmax (persistent across KV chunks)
            running_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_max, LARGE_NEG)

            running_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_sum, 0.0)

            # Output accumulator: (PMAX, d_head) fp32 (persistent across KV chunks)
            out_acc = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(out_acc, 0.0)

            # Flash attention: iterate over K/V chunks
            # Use Python range() so the loop is unrolled at NKI trace time.
            for kv_chunk_idx in range(n_kv_chunks):
                kv_offset = kv_chunk_idx * K_TILE

                # --- Step 1: QK matmul ---
                k_tile = nl.ndarray((d_head, K_TILE), dtype=k.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k_tile, src=k[b, :, nl.ds(kv_offset, K_TILE)])

                q_scaled_bf16 = nl.ndarray(
                    (PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=q_scaled_bf16, src=q_scaled)

                qk_psum = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(qk_psum, q_scaled_bf16, k_tile)

                qk_sbuf = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                # --- Step 2: Online softmax ---
                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_max, nl.maximum, qk_sbuf, axis=1)

                new_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(new_max, running_max, chunk_max, op=nl.maximum)

                max_diff = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(max_diff, running_max, new_max, op=nl.subtract)
                correction = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(correction, nl.exp, max_diff)

                nisa.tensor_scalar(
                    out_acc, out_acc, op0=nl.multiply, operand0=correction
                )

                nisa.tensor_scalar(
                    running_sum, running_sum, op0=nl.multiply, operand0=correction
                )

                qk_centered = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    qk_centered, qk_sbuf, op0=nl.subtract, operand0=new_max
                )

                exp_scores = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(exp_scores, nl.exp, qk_centered)

                chunk_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_sum, nl.add, exp_scores, axis=1)
                nisa.tensor_tensor(running_sum, running_sum, chunk_sum, op=nl.add)

                nisa.tensor_copy(dst=running_max, src=new_max)

                # --- Step 3: P@V matmul (fp32 exp @ fp32 V) ---
                pv_psum = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.psum)

                for v_tile_idx in nl.affine_range(n_v_tiles_per_chunk):
                    v_offset = kv_offset + v_tile_idx * V_TILE

                    v_tile_bf16 = nl.ndarray(
                        (V_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=v_tile_bf16, src=v[b, nl.ds(v_offset, V_TILE), :])

                    exp_chunk_offset = v_tile_idx * V_TILE
                    exp_chunk_bf16 = nl.ndarray(
                        (PMAX, V_TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=exp_chunk_bf16,
                        src=exp_scores[:, nl.ds(exp_chunk_offset, V_TILE)],
                    )

                    nisa.nc_matmul(pv_psum, exp_chunk_bf16, v_tile_bf16)

                pv_sbuf = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                nisa.tensor_tensor(out_acc, out_acc, pv_sbuf, op=nl.add)

            # --- Step 4: Normalize by sum ---
            inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inv_sum, nl.reciprocal, running_sum)

            nisa.tensor_scalar(out_acc, out_acc, op0=nl.multiply, operand0=inv_sum)

            # --- Step 5: Cast to output dtype and store ---
            out_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=out_acc)

            nisa.dma_copy(dst=output[b, nl.ds(q_offset, PMAX), :], src=out_tile)

    return output
