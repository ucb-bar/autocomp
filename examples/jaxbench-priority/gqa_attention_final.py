
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import splash_attention as splash

CONFIG = {
    "name": "llama3_405b_gqa",
    "model": "Llama-3.1-405B",
    "operator": "gqa_attention",
    "batch": 1,
    "seq_len": 2048,
    "num_query_heads": 128,
    "num_kv_heads": 8,
    "head_dim": 128,
    "emb_dim": 16384,
}


@lru_cache(maxsize=None)
def _get_splash_gqa_kernel(seq_len: int, num_query_heads: int):
    """Build and cache a Splash Attention kernel for a given sequence/head shape.
    
    Uses maximized block sizes for seq_len=2048 to improve MXU utilization
    and reduce pipeline overhead on TPU v6e.
    
    VMEM budget analysis for block sizes of 1024:
    - Q tile: 1024 × 128 × 2 bytes (bf16) = 256 KB
    - K tile: 1024 × 128 × 2 bytes (bf16) = 256 KB
    - V tile: 1024 × 128 × 2 bytes (bf16) = 256 KB
    - Attention scores: 1024 × 1024 × 4 bytes (fp32) = 4 MB
    - Output accumulator: 1024 × 128 × 4 bytes (fp32) = 512 KB
    - Total per head: ~5.3 MB, fits comfortably in 16 MB VMEM
    """
    head_mask = splash.CausalMask(shape=(seq_len, seq_len))
    mask = splash.MultiHeadMask(masks=[head_mask] * num_query_heads)
    
    # Maximize block sizes for seq_len=2048 to reduce tile iterations
    # Using 1024 for both Q and KV reduces iterations from 4×4=16 to 2×2=4
    # This significantly improves arithmetic intensity and MXU utilization
    if seq_len == 2048:
        block_q = 1024
        block_kv = 1024
        block_kv_compute = 1024
        # Backward pass block sizes also maximized for consistency
        block_q_dkv = 1024
        block_kv_dkv = 1024
        block_kv_dkv_compute = 1024
        block_q_dq = 1024
        block_kv_dq = 1024
    elif seq_len >= 1024:
        # For larger sequences, use 512 to stay within VMEM
        block_q = 512
        block_kv = 512
        block_kv_compute = 512
        block_q_dkv = 512
        block_kv_dkv = 512
        block_kv_dkv_compute = 512
        block_q_dq = 512
        block_kv_dq = 512
    else:
        # For smaller sequences, use sequence length or 256
        block_q = min(seq_len, 256)
        block_kv = min(seq_len, 256)
        block_kv_compute = min(seq_len, 256)
        block_q_dkv = min(seq_len, 256)
        block_kv_dkv = min(seq_len, 256)
        block_kv_dkv_compute = min(seq_len, 256)
        block_q_dq = min(seq_len, 256)
        block_kv_dq = min(seq_len, 256)
    
    block_sizes = splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
    )
    
    return splash.make_splash_mha(
        mask=mask,
        head_shards=num_query_heads,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )


def workload(query, key, value):
    """GQA Attention optimized for TPU v6e.
    
    Assumes inputs are in BSHD layout (Batch, Seq, Head, Dim) as per original API.
    Transposes to BHSD for Splash Attention and transposes output back to BSHD.
    """
    B, S, Hq, D = query.shape
    _, Sk, Hkv, Dk = key.shape
    _, Sv, Hvv, Dv = value.shape

    if Sk != S or Sv != S:
        raise ValueError(
            f"Sequence length mismatch: query={S}, key={Sk}, value={Sv}"
        )
    if Dk != D or Dv != D:
        raise ValueError(
            f"Head dim mismatch: query={D}, key={Dk}, value={Dv}"
        )
    if Hvv != Hkv:
        raise ValueError(
            f"KV head mismatch: key has {Hkv} heads, value has {Hvv} heads"
        )
    if Hq % Hkv != 0:
        raise ValueError(
            f"num_query_heads ({Hq}) must be divisible by num_kv_heads ({Hkv})"
        )

    # Convert BSHD -> BHSD for Splash Attention.
    # Scale Q in fp32 for stable semantics, then cast back to the original dtype
    # expected by the Splash kernel.
    scale = jnp.asarray(D ** -0.5, dtype=jnp.float32)
    q_bhsd = (
        query.transpose(0, 2, 1, 3).astype(jnp.float32) * scale
    ).astype(query.dtype)
    k_bhsd = key.transpose(0, 2, 1, 3)
    v_bhsd = value.transpose(0, 2, 1, 3)

    # Native GQA path: pass Hkv-headed K/V directly.
    kernel = _get_splash_gqa_kernel(S, Hq)
    out_bhsd = jax.vmap(kernel, in_axes=(0, 0, 0), out_axes=0)(
        q_bhsd, k_bhsd, v_bhsd
    )

    # Convert BHSD -> BSHD.
    return out_bhsd.transpose(0, 2, 1, 3)
