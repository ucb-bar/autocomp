CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''import jax

import jax.numpy as jnp

from functools import partial

CONFIG = {
    \'name\': \'deepseek_v3_mla\',
    \'model\': \'DeepSeek-V3-671B\',
    \'operator\': \'mla_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'emb_dim\': 7168,
    \'num_heads\': 128,
    \'q_lora_rank\': 1536,
    \'kv_lora_rank\': 512,
    \'qk_nope_head_dim\': 128,
    \'qk_rope_head_dim\': 64,
    \'v_head_dim\': 128,
    \'rope_theta\': 10000,
}

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)

def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C[\'batch\'], C[\'seq_len\'], C[\'emb_dim\']
    H = C[\'num_heads\']
    ql, kvl = C[\'q_lora_rank\'], C[\'kv_lora_rank\']
    nope, rope, vd = C[\'qk_nope_head_dim\'], C[\'qk_rope_head_dim\'], C[\'v_head_dim\']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C[\'num_heads\']
    nope, rope, vd = C[\'qk_nope_head_dim\'], C[\'qk_rope_head_dim\'], C[\'v_head_dim\']
    kvl = C[\'kv_lora_rank\']
    # Query
    q = jnp.dot(jnp.dot(x, q_down_proj), q_up_proj)
    q = q.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    # KV compression
    kv = jnp.dot(x, kv_down_proj)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)
    # RoPE
    cos, sin = _compute_rope(rope, S, C[\'rope_theta\'], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)
    # Value
    v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
    # Attention
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    hd = nope + rope
    attn = jnp.einsum(\'bhqd,bhkd->bhqk\', q_full, k_full) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum(\'bhqk,bhkd->bhqd\', attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    return jnp.dot(out, o_proj)
''',
score=4.543,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 4.543 ms\n{"correct": true, "latency": 4.543, "error": "", "all_times_ms": [4.361, 4.364, 4.368, 4.371, 4.382, 4.383, 4.387, 4.387, 4.392, 4.392, 4.395, 4.4, 4.403, 4.405, 4.406, 4.409, 4.411, 4.413, 4.414, 4.414, 4.414, 4.42, 4.422, 4.424, 4.424, 4.434, 4.438, 4.44, 4.446, 4.446, 4.448, 4.456, 4.457, 4.458, 4.461, 4.462, 4.48, 4.482, 4.483, 4.485, 4.491, 4.522, 4.528, 4.53, 4.533, 4.533, 4.535, 4.537, 4.537, 4.539, 4.543, 4.545, 4.547, 4.548, 4.548, 4.554, 4.555, 4.556, 4.557, 4.558, 4.559, 4.562, 4.563, 4.565, 4.567, 4.58, 4.584, 4.588, 4.59, 4.591, 4.598, 4.599, 4.599, 4.614, 4.616, 4.63, 4.632, 4.634, 4.639, 4.645, 4.646, 4.647, 4.654, 4.655, 4.66, 4.67, 4.678, 4.685, 4.701, 4.705, 4.716, 4.723, 4.734, 4.745, 4.745, 4.761, 4.768, 4.774, 4.788, 4.87], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 3 — decompose matrix multiplications into blocked grid loops with `pl.BlockSpec` and a `pltpu.VMEM` accumulator**

### Why this is the best phase-1 change for this code on **TPU v6e-1**
The biggest, cleanest TPU-targetable pieces in `workload(...)` are the dense projections:

- `x @ q_down_proj`
- `(... ) @ q_up_proj`
- `x @ kv_down_proj`
- `k_latent @ k_up_proj`
- `k_latent @ v_up_proj`
- `out @ o_proj`

These are all regular GEMMs, and strategy 3 maps directly onto the TPU MXU with minimal semantic risk. On v6e-1 there is only **one TensorCore per chip**, so I would use a plain `pl.pallas_call` matmul kernel with no Megacore/core-splitting logic.

---

## Plan

### 1. Keep `workload`’s public signature unchanged
Do **not** change:

```python
def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    ...
```

Add internal helpers only, e.g. `_pallas_gemm` and `_gemm_kernel`.

---

### 2. Convert only the dense `jnp.dot` calls to a reusable TPU matmul kernel
Replace these six high-level matmuls with calls to `_pallas_gemm(...)`:

1. `jnp.dot(x, q_down_proj)`
2. `jnp.dot(q_low, q_up_proj)`
3. `jnp.dot(x, kv_down_proj)`
4. `jnp.dot(k_latent, k_up_proj)`
5. `jnp.dot(k_latent, v_up_proj)`
6. `jnp.dot(out, o_proj)`

Everything else in `workload` stays as-is in phase 1:
- `_compute_rope`
- `_apply_rope`
- concatenation/splitting/reshaping
- causal masking
- softmax
- attention `einsum`s

This keeps the scope strictly within strategy 3.

---

### 3. Flatten batch/sequence outside the kernel so each projection becomes a 2D GEMM
Before each projection, reshape outside Pallas:

- `x`: `(B, S, E)` → `(B*S, E)` = `(2048, 7168)`
- `q_low`: `(B, S, ql)` → `(2048, 1536)`
- `k_latent`: `(B, S, kvl)` → `(2048, 512)`
- `out`: `(B, S, H*vd)` → `(2048, 16384)`

This avoids doing unsupported reshape/transposition patterns inside the kernel.

Example replacements:

- `q_low_2d = _pallas_gemm(x_2d, q_down_proj)`
- `q_2d = _pallas_gemm(q_low_2d, q_up_proj)`
- `kv_2d = _pallas_gemm(x_2d, kv_down_proj)`
- `k_nope_2d = _pallas_gemm(k_latent_2d, k_up_proj)`
- `v_2d = _pallas_gemm(k_latent_2d, v_up_proj)`
- `final_2d = _pallas_gemm(out_2d, o_proj)`

Then reshape back in regular JAX.

---

### 4. Implement one blocked GEMM kernel with reduction on the **last grid axis**
Use a 3D grid:

```python
grid = (M // bm, N // bn, K // bk)
```

with:
- axis 0 = output row tiles
- axis 1 = output column tiles
- axis 2 = reduction tiles over `K`

The reduction axis **must be last**, so the accumulator survives across consecutive `(i, j, k)` iterations for the same output tile.

Use:

```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=("parallel", "parallel", "arbitrary")
)
```

This satisfies TPU ordering rules: parallel axes first, reduction axis last.

---

### 5. Use `BlockSpec` index maps for A, B, and C tiles
For a standard GEMM `C[M, N] = A[M, K] @ B[K, N]`:

```python
in_specs = [
    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),  # A tile
    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),  # B tile
]
out_specs = pl.BlockSpec((bm, bn), lambda i, j, k: (i, j))  # C tile
```

This is the exact TPU-friendly blocked matmul pattern.

---

### 6. Accumulate in a `pltpu.VMEM` scratch tile, initialized and flushed with `pl.when`
Kernel shape:

```python
def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    ...
    return None
```

Inside the kernel:

- explicitly read refs with `a_ref[...]`, `b_ref[...]`
- initialize the scratch accumulator on the first `k` tile:
  ```python
  @pl.when(pl.program_id(2) == 0)
  def _():
      acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
  ```
- accumulate:
  ```python
  acc_ref[...] = acc_ref[...] + jnp.dot(
      a_ref[...], b_ref[...], preferred_element_type=jnp.float32
  )
  ```
- write final output only on the last `k` tile:
  ```python
  @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
  def _():
      c_ref[...] = acc_ref[...].astype(c_ref.dtype)
  ```

This obeys the TPU requirements:
- no arithmetic on bare `Ref`s
- output buffer explicitly initialized indirectly via scratch
- reduction axis is innermost
- only the final reduction step writes `c_ref`

---

### 7. Use block sizes that satisfy TPU divisibility rules and fit comfortably in VMEM
For v6e-1, start with:

- **default**: `bm=128`, `bn=512`, `bk=128`

These are TPU-valid:
- second-to-last dims divisible by 8
- last dims divisible by 128

They also fit easily in VMEM with fp32 accumulation.

Approximate live tile footprint per invocation:
- `A`: `128 x 128` bf16
- `B`: `128 x 512` bf16
- `acc`: `128 x 512` f32
- `C`: `128 x 512` bf16

This is well under VMEM limits, even with buffering.

#### Special case: `kv_down_proj` output width = 576
`576` is **not divisible by 128**, so for that one matmul use the **full output width** as the block’s last dimension:

- `A` block: `(128, 128)`
- `B` block: `(128, 576)`  ← allowed because it spans the full array dimension
- `C` block: `(128, 576)`
- grid: `(2048 // 128, 1, 7168 // 128)`

That avoids illegal block shapes.

---

### 8. Apply the helper to each projection with shape-specific tiling
Concretely:

#### a) Query down projection
- `A`: `(2048, 7168)`
- `B`: `(7168, 1536)`
- use `(bm, bn, bk) = (128, 512, 128)` or `(128, 256, 128)`

#### b) Query up projection
- `A`: `(2048, 1536)`
- `B`: `(1536, 24576)`
- use `(128, 512, 128)`

#### c) KV down projection
- `A`: `(2048, 7168)`
- `B`: `(7168, 576)`
- use `(128, 576, 128)` with full-width `bn=576`

#### d) K up projection
- `A`: `(2048, 512)`
- `B`: `(512, 16384)`
- use `(128, 512, 128)`

#### e) V up projection
- same shape as K up

#### f) Final output projection
- `A`: `(2048, 16384)`
- `B`: `(16384, 7168)`
- use `(128, 512, 128)`

---

### 9. Keep bf16 storage, but accumulate in fp32
The original code uses bf16 inputs by default. The TPU kernel should:
- read bf16 tiles from HBM
- use `jnp.dot(..., preferred_element_type=jnp.float32)` for accumulation
- cast back to the original output dtype on the last reduction step

This preserves semantic equivalence within small numerical tolerance and matches TPU behavior well.

---

## What changes in `workload`
Only the projection lines change structurally. For example:

- replace:
  ```python
  q = jnp.dot(jnp.dot(x, q_down_proj), q_up_proj)
  ```
  with:
  ```python
  x2d = x.reshape(B * S, E)
  q_low2d = _pallas_gemm(x2d, q_down_proj, ...)
  q2d = _pallas_gemm(q_low2d, q_up_proj, ...)
  q = q2d.reshape(B, S, H, nope + rope)
  ```

- replace:
  ```python
  kv = jnp.dot(x, kv_down_proj)
  ```
  with:
  ```python
  kv2d = _pallas_gemm(x2d, kv_down_proj, ...)
  kv = kv2d.reshape(B, S, kvl + rope)
  ```

- replace:
  ```python
  k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)
  v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
  ```
  with flattened `_pallas_gemm` calls

- replace:
  ```python
  return jnp.dot(out, o_proj)
  ```
  with:
  ```python
  out2d = out.reshape(B * S, H * vd)
  return _pallas_gemm(out2d, o_proj, ...).reshape(B, S, E)
  ```

---

## Summary
For **phase 1**, I would apply **strategy 3 only**:

- introduce one reusable TPU Pallas GEMM helper
- rewrite the six dense `jnp.dot` projections in `workload` to use it
- tile them with `pl.BlockSpec`
- reduce over `K` on the innermost grid axis
- accumulate in `pltpu.VMEM` fp32 scratch
- initialize with `pl.when(program_id(k_axis) == 0)`
- flush on the final reduction step

This directly targets the most TPU-friendly high-cost parts of the function while keeping the rest of the algorithm unchanged.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    \'name\': \'deepseek_v3_mla\',
    \'model\': \'DeepSeek-V3-671B\',
    \'operator\': \'mla_attention\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'emb_dim\': 7168,
    \'num_heads\': 128,
    \'q_lora_rank\': 1536,
    \'kv_lora_rank\': 512,
    \'qk_nope_head_dim\': 128,
    \'qk_rope_head_dim\': 64,
    \'v_head_dim\': 128,
    \'rope_theta\': 10000,
}

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)

def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C[\'batch\'], C[\'seq_len\'], C[\'emb_dim\']
    H = C[\'num_heads\']
    ql, kvl = C[\'q_lora_rank\'], C[\'kv_lora_rank\']
    nope, rope, vd = C[\'qk_nope_head_dim\'], C[\'qk_rope_head_dim\'], C[\'v_head_dim\']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked GEMM kernel with reduction on innermost axis."""
    k_idx = pl.program_id(2)
    num_k = pl.num_programs(2)
    
    # Initialize accumulator on first k tile
    @pl.when(k_idx == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    
    # Read tiles and accumulate
    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)
    acc_ref[...] = acc_ref[...] + jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)
    
    # Write output on last k tile
    @pl.when(k_idx == num_k - 1)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=128, bn=512, bk=128):
    """
    Pallas-based GEMM: C[M, N] = A[M, K] @ B[K, N]
    Uses blocked tiling with reduction on innermost grid axis.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    # Determine actual block sizes, handling cases where N doesn\'t divide evenly
    # If N is not divisible by 128, use full N as block size (allowed by TPU rules)
    if N % 128 != 0:
        bn_actual = N
    else:
        bn_actual = bn
    
    # Ensure bm and bk are valid
    bm_actual = bm if M % bm == 0 else M
    bk_actual = bk if K % bk == 0 else K
    
    # Compute grid dimensions
    grid_m = M // bm_actual
    grid_n = N // bn_actual
    grid_k = K // bk_actual
    
    grid = (grid_m, grid_n, grid_k)
    
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)
    
    in_specs = [
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),  # A tile
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),  # B tile
    ]
    out_specs = pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j))  # C tile
    
    # Scratch for accumulator in VMEM
    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]
    
    result = pl.pallas_call(
        _gemm_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(a, b)
    
    return result


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C[\'num_heads\']
    nope, rope, vd = C[\'qk_nope_head_dim\'], C[\'qk_rope_head_dim\'], C[\'v_head_dim\']
    kvl = C[\'kv_lora_rank\']
    
    # Flatten batch and sequence for 2D GEMM
    x2d = x.reshape(B * S, E)
    
    # Query projections using Pallas GEMM
    # q_down_proj: (E, ql) = (7168, 1536)
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=128, bn=512, bk=128)
    
    # q_up_proj: (ql, H*(nope+rope)) = (1536, 24576)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=128, bn=512, bk=128)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    
    # KV compression using Pallas GEMM
    # kv_down_proj: (E, kvl+rope) = (7168, 576)
    # 576 is not divisible by 128, so _pallas_gemm will use full width
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=128, bn=576, bk=128)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    
    # K nope projection using Pallas GEMM
    # k_up_proj: (kvl, H*nope) = (512, 16384)
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=128, bn=512, bk=128)
    k_nope = k_nope2d.reshape(B, S, H, nope)
    
    # RoPE
    cos, sin = _compute_rope(rope, S, C[\'rope_theta\'], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)
    
    # Value projection using Pallas GEMM
    # v_up_proj: (kvl, H*vd) = (512, 16384)
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=128, bn=512, bk=128)
    v = v2d.reshape(B, S, H, vd)
    
    # Attention
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    hd = nope + rope
    attn = jnp.einsum(\'bhqd,bhkd->bhqk\', q_full, k_full) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum(\'bhqk,bhkd->bhqd\', attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    
    # Output projection using Pallas GEMM
    # o_proj: (H*vd, E) = (16384, 7168)
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=128, bn=512, bk=128)
    
    return final2d.reshape(B, S, E)
''',
score=25.773,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 25.773 ms\n{"correct": true, "latency": 25.773, "error": "", "all_times_ms": [25.749, 25.752, 25.755, 25.756, 25.757, 25.757, 25.758, 25.759, 25.76, 25.76, 25.76, 25.761, 25.761, 25.761, 25.761, 25.761, 25.762, 25.762, 25.762, 25.762, 25.763, 25.763, 25.764, 25.764, 25.764, 25.764, 25.765, 25.766, 25.766, 25.766, 25.766, 25.767, 25.767, 25.768, 25.768, 25.768, 25.768, 25.769, 25.769, 25.769, 25.77, 25.77, 25.771, 25.771, 25.771, 25.772, 25.772, 25.772, 25.773, 25.773, 25.773, 25.773, 25.774, 25.774, 25.774, 25.775, 25.776, 25.776, 25.776, 25.776, 25.777, 25.777, 25.777, 25.777, 25.778, 25.779, 25.779, 25.779, 25.779, 25.78, 25.78, 25.78, 25.78, 25.78, 25.781, 25.781, 25.781, 25.782, 25.782, 25.782, 25.783, 25.784, 25.784, 25.784, 25.785, 25.785, 25.787, 25.788, 25.788, 25.788, 25.79, 25.79, 25.79, 25.791, 25.792, 25.798, 25.8, 25.802, 25.805, 25.807], "max_diff": 0.015625, "max_rel_diff": 0.002151}',
stderr=''),
plan='''**Plan for Translation Phase 3**

**Strategy Selection:** Strategy 3: "Decompose matrix multiplications into blocked grid loops with `pl.BlockSpec` index maps and accumulate in a `pltpu.VMEM` scratch buffer using `jnp.dot(..., preferred_element_type=jnp.float32)`, initializing and flushing the accumulator with `pl.when` on the reduction axis program id."

**Target Hardware:** TPU v6e-1 (1 TensorCore, ~16 MiB VMEM, (8, 128) native tile).

**Detailed Plan:**

The original code implements a helper function `_pallas_gemm` which attempts to tiling matrix multiplication using `pl.pallas_call`. However, the implementation has specific issues regarding index mapping and loop ordering that violate the target hardware constraints (specifically regarding reduction grid axis order and accumulator initialization). The corresponding `workload` function performs several matrix multiplications using this helper for the MLA (Multi-head Latent Attention) operator.

To optimize this for the v6e hardware and align with Strategy 3, I will rewrite the GEMM logic to strictly enforce the "Reduction must be over the innermost (last) grid dimensions" rule to ensure VMEM persistence of accumulators. The significant matrices (like `q_up_proj` with N=24576) naturally exceed the VMEM capacity if processed without a reduction loop, making the 3D grid `(m, n, k)` essential.

**Key Modifications:**

1.  **Function Signature & Structure:** Keep `_pallas_gemm` and `workload` signatures the same. `_pallas_gemm` will encapsulate the optimized kernel.
2.  **Kernel definition (`_gemm_kernel`):** Update to read `a_ref` and `b_ref` as `jnp.float32` (upcasting bf16 inputs as required for native compute), perform the dot product accumulating in float32, and write back to `c_ref` casting to the output dtype.
3.  **Accumulator Logic:** Inside `_gemm_kernel`, use `pl.when(pl.program_id(2) == 0)` to initialize the VMEM scratch accumulator to zero on the first step of the K-reduction loop. Use `pl.when(pl.program_id(2) == num_k - 1)` to write the final accumulated result to the output `c_ref`.
4.  **Grid & BlockSpecs Configuration:**
    *   The `grid` will be `(M // bm, N // bn, K // bk)`.
    *   The `in_specs` will map `lambda i, j, k: (i, k)` for A and `lambda i, j, k: (k, j)` for B. The output `out_specs` will map `lambda i, j, k: (i, j)`. This ensures that for a fixed `(i, j)` (output block), the `k` iterator varies consecutively (innermost), allowing the compiler to keep the accumulator in VMEM.
    *   Ensure `block_shape` (last two dims) adhere to divisibility by (8, 128). Given N dimensions like 24576, a block size of 128 or 512 is suitable (24576 % 128 == 0). For N=576 (in `kv_down_proj`), the full width must be used or padded appropriately, but for this strategy, we adhere to the divisibility rules and set standard tiles.
5.  **Integration:** The `workload` function already calls `_pallas_gemm`. The optimized version will replace the existing logic, providing significant speedups by correctly saturating the MXU and utilizing pipelining/overlap.

**Implementation Details:**

*   **`_gemm_kernel`**:
    ```python
    def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
        k_id = pl.program_id(2)
        num_k = pl.num_programs(2)

        # Initialize accumulator on first k-step
        @pl.when(k_id == 0)
        def _():
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        # Load tiles (upcast to f32 for compute)
        a_tile = a_ref[...].astype(jnp.float32)
        b_tile = b_ref[...].astype(jnp.float32)

        # Accumulate dot product
        acc_ref[...] += jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)

        # Write back to output on last k-step
        @pl.when(k_id == num_k - 1)
        def _():
            c_ref[...] = acc_ref[...].astype(c_ref.dtype)
    ```

*   **`_pallas_gemm`**: Configure block sizes (bm=128, bn=128/512) to fit VMEM constraints (16MB). A 128x128 float32 tile is 64KB. 3 input + 1 output + 1 scratch fits easily.
    *   `dimension_semantics=("parallel", "parallel", "arbitrary")` enforces the M and N loops are parallelizable (or not dependent on K order), while K is the sequential reduction loop.

This approach directly applies Strategy 3 to the compute-heavy portions of the `workload`, ensuring the hardware-specific constraints of the v6e TPU (tile alignment, VMEM capacity, and ordering) are met for maximal throughput.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked GEMM kernel with K reduction on innermost grid axis."""
    k_id = pl.program_id(2)
    num_k = pl.num_programs(2)

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # Refs must be explicitly read. Upcast to f32 before computation.
    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        a_tile, b_tile, preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    """Choose a TPU-valid block size.

    Returns a block size that:
      * preferably matches `preferred`,
      * otherwise uses the largest divisor of `dim` <= preferred that is a multiple
        of `multiple`,
      * otherwise falls back to the full dimension (which is TPU-legal).
    """
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    # Full-dimension blocks are allowed on TPU even if not divisible by the tile multiple.
    return dim


def _pallas_gemm(a, b, bm=128, bn=512, bk=128):
    """
    Pallas GEMM for TPU v6e-1:
      C[M, N] = A[M, K] @ B[K, N]

    Uses a 3D grid:
      (m_block, n_block, k_block)

    with the K axis as the innermost reduction dimension so the accumulator
    can persist in VMEM across K steps.
    """
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    # Respect TPU blocked-layout constraints:
    #   - second-to-last block dim: divisible by 8 or full axis
    #   - last block dim: divisible by 128 or full axis
    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        # This should not happen due to _choose_tpu_block.
        raise ValueError(
            f"Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (M // bm_actual, N // bn_actual, K // bk_actual)

    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    in_specs = [
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
    ]
    out_specs = pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j))

    # Float32 accumulator tile lives in VMEM across the K reduction loop.
    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(a, b)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # Flatten batch and sequence for 2D GEMMs.
    x2d = x.reshape(B * S, E)

    # Query projections.
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=128, bn=512, bk=128)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=128, bn=512, bk=128)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV compression.
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=128, bn=512, bk=128)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    # K projection.
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=128, bn=512, bk=128)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # V projection.
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=128, bn=512, bk=128)
    v = v2d.reshape(B, S, H, vd)

    # Attention.
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    hd = nope + rope
    attn = jnp.einsum("bhqd,bhkd->bhqk", q_full, k_full) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=128, bn=512, bk=128)

    return final2d.reshape(B, S, E)
''',
score=25.76,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-4.7',
code_gen_model='gpt-5.4',
stdout='Latency: 25.760 ms\n{"correct": true, "latency": 25.76, "error": "", "all_times_ms": [25.739, 25.744, 25.745, 25.745, 25.745, 25.746, 25.747, 25.747, 25.747, 25.748, 25.748, 25.748, 25.749, 25.749, 25.749, 25.75, 25.751, 25.752, 25.752, 25.752, 25.753, 25.753, 25.753, 25.753, 25.753, 25.753, 25.755, 25.755, 25.755, 25.755, 25.755, 25.755, 25.756, 25.756, 25.756, 25.757, 25.757, 25.757, 25.757, 25.757, 25.758, 25.758, 25.758, 25.758, 25.758, 25.758, 25.758, 25.759, 25.759, 25.76, 25.76, 25.76, 25.76, 25.76, 25.761, 25.761, 25.761, 25.761, 25.762, 25.762, 25.762, 25.762, 25.762, 25.763, 25.763, 25.763, 25.764, 25.764, 25.764, 25.765, 25.766, 25.766, 25.766, 25.766, 25.767, 25.767, 25.767, 25.768, 25.769, 25.769, 25.77, 25.77, 25.77, 25.77, 25.77, 25.77, 25.77, 25.771, 25.772, 25.773, 25.773, 25.773, 25.775, 25.776, 25.78, 25.78, 25.782, 25.786, 25.787, 25.789], "max_diff": 0.015625, "max_rel_diff": 0.002151}',
stderr=''),
plan='''To optimize the DeepSeek-V3 MLA (Multi-Head Latent Attention) workload for the TPU v6e-1, we will apply **Strategy 3**. This strategy focuses on decomposing all matrix multiplications—including the large linear projections and the batch matmuls (BMM) within the attention mechanism—into blocked grid loops. We use `pl.BlockSpec` to map the multi-dimensional grid (covering batch, head, and spatial dimensions) to the on-chip VMEM buffers. Following the v6e hardware constraints, we ensure tile alignment (8, 128) and use a high-precision float32 accumulator in VMEM to mitigate rounding errors during the reduction loop over the inner dimension $K$.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)

def _matmul_kernel(a_ref, b_ref, c_ref, acc_ref):
    """General matmul kernel with K-reduction on the innermost grid axis."""
    # Reduction dimension is the last grid axis
    k_id = pl.program_id(pl.ndim_grid - 1)
    num_k = pl.num_programs(pl.ndim_grid - 1)

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    # Elementwise ops on TPU must be 32-bit; BF16 inputs are cast for computation.
    # jnp.dot accumulates in f32 when preferred_element_type is specified.
    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32), 
        b_ref[...].astype(jnp.float32), 
        preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)

def _choose_tpu_block(dim, preferred, multiple):
    """Ensure block size is valid for TPU (divisible by multiple or full dim)."""
    if dim <= preferred: return dim
    if preferred % multiple == 0 and dim % preferred == 0: return preferred
    # Fallback: largest divisor of dim that is a multiple of \'multiple\'
    for b in range(preferred - (preferred % multiple), multiple - 1, -multiple):
        if dim % b == 0: return b
    return dim

def _pallas_matmul(a, b, bm=128, bn=128, bk=128):
    """
    Pallas Batch Matmul for TPU v6e. Handles [..., M, K] @ [..., K, N].
    Grid handles batch/head dimensions + M, N, and K reduction.
    """
    a_shape, b_shape = a.shape, b.shape
    batch_dims = a_shape[:-2]
    M, K = a_shape[-2:]
    N = b_shape[-1]
    
    # TPU Layout Alignment: (8, 128)
    bm_p = _choose_tpu_block(M, bm, 8)
    bn_p = _choose_tpu_block(N, bn, 128)
    bk_p = _choose_tpu_block(K, bk, 128)
    
    # Grid setup: Batch axes + M-blocks + N-blocks + K-blocks (reduction)
    grid = (*batch_dims, M // bm_p, N // bn_p, K // bk_p)
    
    # Logic to map grid indices to array slices
    def idx_map_a(*args):
        # args: (*batch_indices, i_idx, j_idx, k_idx)
        return (*args[:-3], args[-3], args[-1])
        
    def idx_map_b(*args):
        return (*args[:-3], args[-1], args[-2])
        
    def idx_map_out(*args):
        return (*args[:-3], args[-3], args[-2])

    # Compiler hints: Parallelize batch and spatial blocks, keep K reduction sequential.
    semantics = ("parallel",) * (len(grid) - 1) + ("arbitrary",)

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((*batch_dims, M, N), a.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm_p, bk_p), idx_map_a),
            pl.BlockSpec((bk_p, bn_p), idx_map_b),
        ],
        out_specs=pl.BlockSpec((bm_p, bn_p), idx_map_out),
        scratch_shapes=[pltpu.VMEM((bm_p, bn_p), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=semantics)
    )(a, b)

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H, nope, rope, vd = C["num_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # Projections: Replace standard dots with Strategy 3 Pallas Matmuls
    q_low = _pallas_matmul(x, q_down_proj.reshape(1, E, -1))
    q = _pallas_matmul(q_low, q_up_proj.reshape(1, -1, H * (nope + rope)))
    q = q.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = _pallas_matmul(x, kv_down_proj.reshape(1, E, -1))
    kv = kv.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    k_nope = _pallas_matmul(k_latent.reshape(B, S, -1), k_up_proj.reshape(1, kvl, -1))
    k_nope = k_nope.reshape(B, S, H, nope)

    # RoPE calculation (remains JAX)
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope, k_rope = _apply_rope(q_rope, cos, sin), _apply_rope(k_rope, cos, sin)

    v = _pallas_matmul(k_latent.reshape(B, S, -1), v_up_proj.reshape(1, kvl, -1))
    v = v.reshape(B, S, H, vd)

    # Attention Mechanism: Strategy 3 BMMs
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3) # [B, H, S, D]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3) # [B, H, S, D]
    v = v.transpose(0, 2, 1, 3) # [B, H, S, D]

    # QK^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    attn = _pallas_matmul(q_full, k_full.transpose(0, 1, 3, 2)) * ((nope + rope) ** -0.5)
    
    # Softmax and Masking (remains JAX)
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    # Attn-V: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    out = _pallas_matmul(attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output Projection
    final = _pallas_matmul(out.reshape(B, S, -1), o_proj.reshape(1, H * vd, E))
    return final.reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked GEMM kernel with K reduction on innermost grid axis."""
    k_id = pl.program_id(2)
    num_k = pl.num_programs(2)

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        a_tile, b_tile, preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _bmm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Batched matmul kernel with K reduction on innermost grid axis.
    
    Grid: (batch, heads, m_blocks, n_blocks, k_blocks)
    """
    k_id = pl.program_id(4)
    num_k = pl.num_programs(4)

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        a_tile, b_tile, preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    """Choose a TPU-valid block size."""
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _pallas_gemm(a, b, bm=128, bn=512, bk=128):
    """Pallas GEMM for TPU v6e: C[M, N] = A[M, K] @ B[K, N]"""
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            f"Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (M // bm_actual, N // bn_actual, K // bk_actual)

    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    in_specs = [
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
    ]
    out_specs = pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j))

    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(a, b)


def _pallas_bmm(a, b, bm=128, bn=128, bk=128):
    """Pallas Batched Matmul for TPU v6e: C[B, H, M, N] = A[B, H, M, K] @ B[B, H, K, N]"""
    B, H, M, K = a.shape
    B2, H2, K2, N = b.shape
    
    if B != B2 or H != H2 or K != K2:
        raise ValueError(f"Shape mismatch: a={a.shape}, b={b.shape}")

    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            f"Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (B, H, M // bm_actual, N // bn_actual, K // bk_actual)

    out_shape = jax.ShapeDtypeStruct((B, H, M, N), a.dtype)

    in_specs = [
        pl.BlockSpec((1, 1, bm_actual, bk_actual), lambda b, h, i, j, k: (b, h, i, k)),
        pl.BlockSpec((1, 1, bk_actual, bn_actual), lambda b, h, i, j, k: (b, h, k, j)),
    ]
    out_specs = pl.BlockSpec((1, 1, bm_actual, bn_actual), lambda b, h, i, j, k: (b, h, i, j))

    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]

    def _bmm_kernel_inner(a_ref, b_ref, c_ref, acc_ref):
        k_id = pl.program_id(4)
        num_k = pl.num_programs(4)

        @pl.when(k_id == 0)
        def _init_acc():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        a_tile = a_ref[0, 0, :, :].astype(jnp.float32)
        b_tile = b_ref[0, 0, :, :].astype(jnp.float32)

        acc_ref[...] = acc_ref[...] + jnp.dot(
            a_tile, b_tile, preferred_element_type=jnp.float32
        )

        @pl.when(k_id == num_k - 1)
        def _store_out():
            c_ref[0, 0, :, :] = acc_ref[...].astype(c_ref.dtype)

    return pl.pallas_call(
        _bmm_kernel_inner,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary")
        ),
    )(a, b)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # Flatten batch and sequence for 2D GEMMs.
    x2d = x.reshape(B * S, E)

    # Query projections.
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=128, bn=512, bk=128)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=128, bn=512, bk=128)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV compression.
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=128, bn=512, bk=128)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    # K projection.
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=128, bn=512, bk=128)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # V projection.
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=128, bn=512, bk=128)
    v = v2d.reshape(B, S, H, vd)

    # Attention.
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    v = v.transpose(0, 2, 1, 3)  # [B, H, S, vd]

    hd = nope + rope
    
    # QK^T using batched matmul: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    k_t = k_full.transpose(0, 1, 3, 2)  # [B, H, D, S]
    attn = _pallas_bmm(q_full, k_t, bm=128, bn=128, bk=128)
    attn = attn * (hd ** -0.5)
    
    # Masking and softmax
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    # Attention @ V: [B, H, S, S] @ [B, H, S, vd] -> [B, H, S, vd]
    out = _pallas_bmm(attn, v, bm=128, bn=128, bk=128)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=128, bn=512, bk=128)

    return final2d.reshape(B, S, E)
''',
score=52.994,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 52.994 ms\n{"correct": true, "latency": 52.994, "error": "", "all_times_ms": [52.955, 52.961, 52.969, 52.97, 52.972, 52.979, 52.98, 52.98, 52.98, 52.981, 52.981, 52.981, 52.982, 52.982, 52.982, 52.983, 52.984, 52.984, 52.984, 52.984, 52.985, 52.985, 52.986, 52.986, 52.986, 52.987, 52.987, 52.987, 52.987, 52.987, 52.987, 52.988, 52.988, 52.988, 52.988, 52.989, 52.989, 52.989, 52.99, 52.99, 52.99, 52.99, 52.991, 52.991, 52.992, 52.992, 52.993, 52.993, 52.993, 52.993, 52.994, 52.995, 52.996, 52.996, 52.996, 52.996, 52.996, 52.997, 52.997, 52.998, 52.998, 52.998, 53.0, 53.001, 53.001, 53.001, 53.001, 53.002, 53.002, 53.002, 53.002, 53.002, 53.002, 53.003, 53.003, 53.003, 53.004, 53.006, 53.006, 53.008, 53.008, 53.009, 53.009, 53.009, 53.009, 53.009, 53.01, 53.01, 53.012, 53.012, 53.012, 53.013, 53.014, 53.014, 53.015, 53.016, 53.018, 53.02, 53.025, 53.026], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''To optimize the DeepSeek-V3 MLA (Multi-head Latent Attention) implementation for TPU v6e, we will apply **Strategy 3**: Decomposing matrix multiplications into blocked grid loops with `pl.BlockSpec` index maps, accumulating in a `pltpu.VMEM` scratch buffer, and using the double-buffered pipelining model.

The plan involves:
1.  **Optimizing `_pallas_gemm`**: Implementing a robust 2D matmul that utilizes `pl.Buffered(2)` for double buffering, enabling the overlap of HBM transfers and MXU computation. We ensure block sizes are aligned to the TPU v6e requirements (8 for second-to-last dimension, 128 for the last).
2.  **Optimizing `_pallas_bmm`**: Extending the GEMM logic to a 5D grid to handle batch and head dimensions efficiently. We use `pl.BlockSpec` with rank 4 to correctly tile the `[B, H, S, D]` tensors, ensuring the reduction axis is the innermost grid dimension to allow the accumulator to persist in VMEM.
3.  **Correct Initialization and Storage**: Using `pl.when` to zero the accumulator on the first reduction step and store it to HBM only on the final step, minimizing HBM bandwidth consumption.
4.  **Hardware Alignment**: Ensuring all intermediate element-wise operations (like casts) are performed on f32 for native VPU support and MXU accumulation.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj

def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    num_k = pl.num_programs(2)

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # MXU computes in bf16/f32; elementwise requires f32 on TPU
    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)
    acc_ref[...] += jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)

    @pl.when(k_id == num_k - 1)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)

def _bmm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(4)
    num_k = pl.num_programs(4)

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # Indices are rank 4 due to BlockSpec(1, 1, bm, bk)
    a_tile = a_ref[0, 0, :, :].astype(jnp.float32)
    b_tile = b_ref[0, 0, :, :].astype(jnp.float32)
    acc_ref[...] += jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)

    @pl.when(k_id == num_k - 1)
    def _store():
        c_ref[0, 0, :, :] = acc_ref[...].astype(c_ref.dtype)

def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    if dim % preferred == 0 and preferred % multiple == 0:
        return preferred
    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate
    return dim

def _pallas_gemm(a, b, bm=128, bn=512, bk=128):
    M, K = a.shape
    _, N = b.shape
    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)
    
    grid = (M // bm_actual, N // bn_actual, K // bk_actual)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(a, b)

def _pallas_bmm(a, b, bm=128, bn=128, bk=128):
    B, H, M, K = a.shape
    _, _, _, N = b.shape
    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    grid = (B, H, M // bm_actual, N // bn_actual, K // bk_actual)
    return pl.pallas_call(
        _bmm_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, M, N), a.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, 1, bm_actual, bk_actual), lambda b, h, i, j, k: (b, h, i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((1, 1, bk_actual, bn_actual), lambda b, h, i, j, k: (b, h, k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((1, 1, bm_actual, bn_actual), lambda b, h, i, j, k: (b, h, i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary")
        ),
    )(a, b)

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    x2d = x.reshape(B * S, E)
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=128, bn=512, bk=128)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=128, bn=512, bk=128)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=128, bn=512, bk=128)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=128, bn=512, bk=128)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=128, bn=512, bk=128)
    v = v2d.reshape(B, S, H, vd)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    hd = nope + rope
    k_t = k_full.transpose(0, 1, 3, 2)
    attn = _pallas_bmm(q_full, k_t, bm=128, bn=128, bk=128)
    attn = attn * (hd ** -0.5)
    
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    out = _pallas_bmm(attn, v, bm=128, bn=128, bk=128)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=128, bn=512, bk=128)
    return final2d.reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked GEMM kernel with K reduction on innermost grid axis."""
    k_id = pl.program_id(2)
    num_k = pl.num_programs(2)

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        a_tile, b_tile, preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _bmm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Batched matmul kernel with K reduction on innermost grid axis.
    
    Grid: (batch, heads, m_blocks, n_blocks, k_blocks)
    """
    k_id = pl.program_id(4)
    num_k = pl.num_programs(4)

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # BlockSpec has shape (1, 1, bm, bk), squeeze batch/head dims for dot
    a_tile = a_ref[0, 0, :, :].astype(jnp.float32)
    b_tile = b_ref[0, 0, :, :].astype(jnp.float32)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        a_tile, b_tile, preferred_element_type=jnp.float32
    )

    @pl.when(k_id == num_k - 1)
    def _store_out():
        c_ref[0, 0, :, :] = acc_ref[...].astype(c_ref.dtype)


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    """Choose a TPU-valid block size."""
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _pallas_gemm(a, b, bm=128, bn=512, bk=128):
    """Pallas GEMM for TPU v6e: C[M, N] = A[M, K] @ B[K, N]"""
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            f"Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (M // bm_actual, N // bn_actual, K // bk_actual)

    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)

    in_specs = [
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
    ]
    out_specs = pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j))

    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(a, b)


def _pallas_bmm(a, b, bm=128, bn=128, bk=128):
    """Pallas Batched Matmul for TPU v6e: C[B, H, M, N] = A[B, H, M, K] @ B[B, H, K, N]"""
    B, H, M, K = a.shape
    B2, H2, K2, N = b.shape
    
    if B != B2 or H != H2 or K != K2:
        raise ValueError(f"Shape mismatch: a={a.shape}, b={b.shape}")

    bm_actual = _choose_tpu_block(M, bm, 8)
    bk_actual = _choose_tpu_block(K, bk, 128)
    bn_actual = _choose_tpu_block(N, bn, 128)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            f"Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (B, H, M // bm_actual, N // bn_actual, K // bk_actual)

    out_shape = jax.ShapeDtypeStruct((B, H, M, N), a.dtype)

    in_specs = [
        pl.BlockSpec((1, 1, bm_actual, bk_actual), lambda b, h, i, j, k: (b, h, i, k)),
        pl.BlockSpec((1, 1, bk_actual, bn_actual), lambda b, h, i, j, k: (b, h, k, j)),
    ]
    out_specs = pl.BlockSpec((1, 1, bm_actual, bn_actual), lambda b, h, i, j, k: (b, h, i, j))

    scratch_shapes = [pltpu.VMEM((bm_actual, bn_actual), jnp.float32)]

    return pl.pallas_call(
        _bmm_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary")
        ),
    )(a, b)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # Flatten batch and sequence for 2D GEMMs.
    x2d = x.reshape(B * S, E)

    # Query projections.
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=128, bn=512, bk=128)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=128, bn=512, bk=128)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV compression.
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=128, bn=512, bk=128)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    # K projection.
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=128, bn=512, bk=128)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # V projection.
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=128, bn=512, bk=128)
    v = v2d.reshape(B, S, H, vd)

    # Attention.
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    v = v.transpose(0, 2, 1, 3)  # [B, H, S, vd]

    hd = nope + rope
    
    # QK^T using batched matmul: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    k_t = k_full.transpose(0, 1, 3, 2)  # [B, H, D, S]
    attn = _pallas_bmm(q_full, k_t, bm=128, bn=128, bk=128)
    attn = attn * (hd ** -0.5)
    
    # Masking and softmax
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    # Attention @ V: [B, H, S, S] @ [B, H, S, vd] -> [B, H, S, vd]
    out = _pallas_bmm(attn, v, bm=128, bn=128, bk=128)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=128, bn=512, bk=128)

    return final2d.reshape(B, S, E)
''',
score=52.997,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 52.997 ms\n{"correct": true, "latency": 52.997, "error": "", "all_times_ms": [52.958, 52.966, 52.969, 52.969, 52.971, 52.973, 52.974, 52.974, 52.977, 52.977, 52.977, 52.978, 52.979, 52.979, 52.979, 52.98, 52.98, 52.981, 52.981, 52.983, 52.984, 52.984, 52.984, 52.985, 52.985, 52.985, 52.986, 52.986, 52.986, 52.987, 52.987, 52.988, 52.988, 52.988, 52.988, 52.989, 52.99, 52.99, 52.992, 52.993, 52.993, 52.993, 52.993, 52.993, 52.995, 52.995, 52.996, 52.996, 52.996, 52.997, 52.997, 52.997, 52.998, 52.998, 52.998, 52.998, 52.999, 52.999, 52.999, 53.0, 53.0, 53.0, 53.0, 53.001, 53.001, 53.001, 53.002, 53.002, 53.002, 53.003, 53.003, 53.004, 53.004, 53.005, 53.005, 53.006, 53.007, 53.008, 53.008, 53.009, 53.01, 53.01, 53.012, 53.014, 53.014, 53.015, 53.017, 53.017, 53.02, 53.022, 53.022, 53.023, 53.024, 53.026, 53.03, 53.031, 53.031, 53.035, 53.04, 53.041], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''Looking at this code and the hardware target (TPU v6e with 1 TensorCore and no Megacore), I\'ll analyze which strategy makes the most sense for optimizing this DeepSeek-V3 MLA attention implementation.

**Plan: Apply Strategy #3 (Decompose matrix multiplications into blocked grid loops)**

## Analysis of Current Implementation:
The code already implements GEMM operations using custom Pallas implementations, at both 2D (`_pallas_gemm`) and batched 4D (`_pallas_bmm`) levels. However, there are several critical issues:

### Current Issues:
1. The current GEMM kernels don\'t properly handle the full TPU memory hierarchy
2. Block sizes are hardcoded without considering VMEM capacity constraints
3. No clear pipelining between memory movement and computation
4. Potentially inefficient use of VMEM for accumulator buffers

Assuming the code is too large for the VMEM capacity given the current block sizes, I need to apply Strategy #3:

**Select Strategy 3: Decompose matrix multiplications into a tiled grid of blocked operations.**

```python
def gemm_kernel(a, b, m, n, k, block_m, block_n, block_k):
    # Use scratch buffer for accumulation
    # Multi-stage blocking for VMEM capacity
    # Envelope the calculation with grid
```

**Implementation Plan:**

1. **Fix the GEMM kernel blocking**: Current block sizes (128x512x128) might not be optimal and may exceed VMEM capacity.

2. **Add proper VMEM management** with explicit scratch buffers for accumulation.

3. **Implement multi-stage blocking** to respect VMEM capacity of ~16MB.

The core transformation would be modifying the `_gemm_kernel` to:
- Use `grid` for tiling across M, N, and K dimensions
- Accumulate results in VMEM scratchpad
- Handle boundary conditions for non-divisible dimensions
- Use proper inline accumulation to respect accumulator partial results

Key changes needed:
1. Downsize block sizes to fit VMEM
2. Add accumulator accumulator buffer with explicit accumulation pattern
3. Ensure all loads/stores move through VMEM with proper tiling

This approach ensures the vast matrices in the MLA attention (e.g., 2048×2048 tiles with 7168-dim) are properly tiled to fit VMEM while maintaining arithmetic intensity.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


# Conservative per-kernel VMEM working-set target for v6e-1.
# This leaves headroom for compiler temporaries and pipeline buffering.
_V6E_GEMM_WORKING_SET_BUDGET = 2 * 1024 * 1024  # 2 MiB


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    """Choose a TPU-valid block size for one axis.

    Returns either:
      * a divisor of `dim` aligned to `multiple`, or
      * the full dimension `dim` (which is also legal on TPU block axes).
    """
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    """Approximate live VMEM footprint under TPU double-buffered pipelining."""
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    # Inputs and outputs are typically double buffered by the TPU pipeline.
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b

    # Accumulator scratch is float32 and must stay live across K-reduction steps.
    acc_bytes = bm * bn * 4

    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    """Generate legal block candidates, largest first."""
    candidates = {dim}  # full-dimension fallback is always legal on TPU
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    """Pick TPU-valid tiles that fit comfortably in v6e VMEM."""
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue

                # Maximize useful work first, then favor larger output tiles.
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    # Guaranteed-valid fallback.
    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked GEMM kernel with K-reduction on the innermost grid axis."""
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _bmm_kernel(a_ref, b_ref, c_ref, acc_ref):
    """Blocked batched matmul kernel.

    Grid: (batch, heads, m_blocks, n_blocks, k_blocks)
    """
    k_id = pl.program_id(4)
    last_k = pl.num_programs(4) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # Leading batch/head block dims are size-1 tiles.
    a_tile = a_ref[0, 0, :, :].astype(jnp.float32)
    b_tile = b_ref[0, 0, :, :].astype(jnp.float32)
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[0, 0, :, :] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    """Blocked TPU GEMM: C[M, N] = A[M, K] @ B[K, N]."""
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (M // bm_actual, N // bn_actual, K // bk_actual)

    outs = pl.pallas_call(
        _gemm_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype)],
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            # v6e-1 has a single TensorCore per chip; no Megacore splitting needed.
            # M/N are independent output tiles, K is a sequential reduction axis.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)
    return outs[0]


def _pallas_bmm(a, b, bm=256, bn=256, bk=256):
    """Blocked TPU batched matmul: C[B, H, M, N] = A[B, H, M, K] @ B[B, H, K, N]."""
    B, H, M, K = a.shape
    B2, H2, K2, N = b.shape

    if B != B2 or H != H2 or K != K2:
        raise ValueError(f"Shape mismatch: a={a.shape}, b={b.shape}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (B, H, M // bm_actual, N // bn_actual, K // bk_actual)

    outs = pl.pallas_call(
        _bmm_kernel,
        out_shape=[jax.ShapeDtypeStruct((B, H, M, N), a.dtype)],
        grid=grid,
        in_specs=[
            pl.BlockSpec(
                (1, 1, bm_actual, bk_actual),
                lambda b0, h0, i, j, k: (b0, h0, i, k),
            ),
            pl.BlockSpec(
                (1, 1, bk_actual, bn_actual),
                lambda b0, h0, i, j, k: (b0, h0, k, j),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (1, 1, bm_actual, bn_actual),
                lambda b0, h0, i, j, k: (b0, h0, i, j),
            ),
        ],
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
        ),
    )(a, b)
    return outs[0]


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    # v6e-1 tuned tile preferences:
    # - projection GEMMs: wider N tiles improve arithmetic intensity
    # - qk^T: square-ish tiles are a good fit for large attention score blocks
    # - attn @ V: N is only v_head_dim=128, so keep that axis narrow
    proj_bm, proj_bn, proj_bk = 256, 512, 256
    score_bm, score_bn, score_bk = 256, 256, 192
    value_bm, value_bn, value_bk = 256, 128, 256

    # Flatten batch and sequence for 2D GEMMs.
    x2d = x.reshape(B * S, E)

    # Query projections.
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV compression.
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    # K projection.
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # V projection.
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    v = v2d.reshape(B, S, H, vd)

    # Attention inputs.
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    v = v.transpose(0, 2, 1, 3)  # [B, H, S, vd]

    hd = nope + rope

    # QK^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    k_t = k_full.transpose(0, 1, 3, 2)
    attn = _pallas_bmm(q_full, k_t, bm=score_bm, bn=score_bn, bk=score_bk)
    attn = attn * (hd ** -0.5)

    # Causal masking + softmax.
    mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
    attn = jax.nn.softmax(attn, axis=-1)

    # Attention @ V: [B, H, S, S] @ [B, H, S, vd] -> [B, H, S, vd]
    out = _pallas_bmm(attn, v, bm=value_bm, bn=value_bn, bk=value_bk)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)

    return final2d.reshape(B, S, E)
''',
score=15.895,
translation_score=None,
hw_feedback=[],
plan_gen_model='deepseek.v3.2',
code_gen_model='gpt-5.4',
stdout='Latency: 15.895 ms\n{"correct": true, "latency": 15.895, "error": "", "all_times_ms": [15.857, 15.859, 15.87, 15.872, 15.875, 15.876, 15.876, 15.877, 15.877, 15.878, 15.878, 15.878, 15.88, 15.88, 15.881, 15.881, 15.883, 15.883, 15.883, 15.884, 15.886, 15.886, 15.886, 15.886, 15.886, 15.887, 15.887, 15.888, 15.888, 15.888, 15.888, 15.889, 15.889, 15.889, 15.89, 15.891, 15.891, 15.892, 15.892, 15.892, 15.892, 15.892, 15.893, 15.893, 15.893, 15.894, 15.894, 15.894, 15.894, 15.895, 15.895, 15.896, 15.897, 15.897, 15.897, 15.897, 15.898, 15.898, 15.898, 15.898, 15.899, 15.899, 15.9, 15.9, 15.901, 15.901, 15.901, 15.902, 15.902, 15.903, 15.903, 15.903, 15.904, 15.904, 15.905, 15.905, 15.905, 15.906, 15.906, 15.906, 15.906, 15.907, 15.91, 15.91, 15.91, 15.91, 15.911, 15.912, 15.912, 15.913, 15.915, 15.917, 15.917, 15.917, 15.919, 15.919, 15.932, 15.935, 15.939, 15.946], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''**Phase 1 strategy: fuse the entire causal attention subgraph into one streaming Pallas kernel**  
*(FlashAttention-style online softmax; distinct from loop tiling / software pipelining / larger blocks / aliasing / on-the-fly RoPE.)*

### Why this code is currently inefficient
The biggest inefficiency is here:

```python
k_t = k_full.transpose(0, 1, 3, 2)
attn = _pallas_bmm(q_full, k_t, ...)
attn = attn * (hd ** -0.5)
mask = jnp.tril(jnp.ones((S, S), dtype=bool))
attn = jnp.where(mask, attn, jnp.array(-1e9, dtype=attn.dtype))
attn = jax.nn.softmax(attn, axis=-1)
out = _pallas_bmm(attn, v, ...)
```

For your shapes, `attn` is:

- `[B, H, S, S] = [1, 128, 2048, 2048]`
- bf16 size ≈ **1 GiB**

So the current program:
1. computes `QK^T`,
2. writes a ~1 GiB score tensor to HBM,
3. reads/writes it again for masking/scaling,
4. reads/writes it again for softmax,
5. reads it again for `attn @ V`.

That makes attention heavily **HBM-traffic-bound**, which is especially bad on v6e where elementwise/softmax work is much cheaper than moving multi-GB intermediates.

---

## What to change
Replace the whole score-materialization path with a single helper, e.g.

```python
out = _pallas_fused_causal_attention(q_full, k_full, v)
```

and delete:
- `k_t`
- the first `_pallas_bmm` for scores
- explicit causal mask construction
- `jax.nn.softmax`
- the second `_pallas_bmm` for `attn @ V`

---

## How the fused kernel should work
Each kernel instance handles one output tile of shape:

- `Q_tile`: `[bm, hd]`
- output tile: `[bm, vd]`

Recommended starting tiles on **v6e-1**:
- `bm = 256`
- `bk = 256`

These are TPU-friendly:
- `bm` divisible by 8
- `bk` divisible by 128
- `hd = 192` can be used as the **full last dimension**, which is legal
- `vd = 128` is already TPU-native

### Grid
Use a grid over query tiles only:

```python
grid = (B, H, S // bm)
```

Each program:
1. loads one `[bm, hd]` query block,
2. iterates over K/V tiles along sequence,
3. updates a running online softmax state in **float32**,
4. writes one `[bm, vd]` output block.

Because `S=2048` and `bk=256`, the inner K/V loop is only **8 iterations**, which is small enough to unroll on TPU.

---

## Online softmax state
Inside the kernel, keep these in VMEM/f32:

- `m`: `[bm]` running row max
- `l`: `[bm]` running row sum of exp
- `acc`: `[bm, vd]` running numerator accumulator

For each K/V tile:
1. compute score block  
   `scores = Q_tile @ K_tile^T` in f32
2. apply scale `hd**-0.5`
3. apply causal mask for this tile
4. update online softmax:

```python
m_new = maximum(m, max(scores, axis=1))
alpha = exp(m - m_new)
p = exp(scores - m_new[:, None])
l = alpha * l + sum(p, axis=1)
acc = alpha[:, None] * acc + p @ V_tile
m = m_new
```

After all K/V tiles:
```python
out = acc / l[:, None]
```

Then cast when writing:
```python
o_ref[...] = out.astype(o_ref.dtype)
```

This is numerically equivalent to masked softmax attention up to normal floating-point tolerance, and it respects the TPU rule that softmax-related reductions/accumulation should be done in **float32**.

---

## Why this helps on v6e-1
### 1. Eliminates the worst intermediate
You stop materializing the `[1,128,2048,2048]` attention tensor in HBM.

That alone removes multiple GB of memory traffic.

### 2. Raises arithmetic intensity
Instead of:
- compute scores,
- write them out,
- read them back for elementwise ops,
- read them back again for `@ V`,

you do:
- compute scores,
- immediately consume them in softmax,
- immediately consume softmax weights in `@ V`,
- store only final output.

That moves the attention block closer to the compute-bound regime.

### 3. Better fit for TPU VMEM
A typical live working set with `bm=bk=256` is modest:

- Q tile: `256 x 192 x 2B` ≈ 96 KiB
- K tile: `256 x 192 x 2B` ≈ 96 KiB
- V tile: `256 x 128 x 2B` ≈ 64 KiB
- score tile f32: `256 x 256 x 4B` ≈ 256 KiB
- acc f32: `256 x 128 x 4B` ≈ 128 KiB
- `m`, `l`: tiny

So this comfortably fits v6e VMEM with headroom.

---

## Code-specific implementation notes
### Keep the same `workload(...)` signature
Only replace the attention core.

### Leave the projection GEMMs alone in phase 1
Do not touch:
- `q_down/q_up`
- `kv_down`
- `k_up`
- `v_up`
- `o_proj`

That keeps this phase narrowly scoped and low-risk.

### Use TPU-valid block shapes
Examples:
- `q_full`: block `(1, 1, bm, hd)` where `hd=192` is the full last dim
- `k_full`: block `(1, 1, bk, hd)`
- `v`: block `(1, 1, bk, 128)`
- output: `(1, 1, bm, 128)`

### Keep reductions in safe tile shapes
If you reduce over the score tile’s last axis, use `bk=256` or `128`, not odd widths.

### Avoid explicit `k_t`
Read K in `[B, H, S, D]` layout and contract over `D` directly in the fused kernel.

---

## Expected impact
This should be the **highest-leverage phase-1 change** because it targets the largest avoidable HBM traffic in the whole program.

In short:

- **Current path:** dense score materialization + mask + softmax + second BMM
- **Proposed path:** one fused causal attention kernel with online softmax

That is a major reduction in memory traffic and should reduce latency significantly on **TPU v6e-1**.

If you want, I can next rewrite `workload` to use a concrete `_pallas_fused_causal_attention(...)` helper while preserving the same function signature.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}


_V6E_GEMM_WORKING_SET_BUDGET = 2 * 1024 * 1024


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    grid = (M // bm_actual, N // bn_actual, K // bk_actual)

    out = pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)
    return out


def _causal_attention_jax(q, k, v):
    """Standard JAX causal attention - used as fallback."""
    B, H, S, hd = q.shape
    vd = v.shape[-1]
    
    scale = hd ** -0.5
    
    # Compute attention scores: [B, H, S, S]
    scores = jnp.einsum(\'bhid,bhjd->bhij\', q, k) * scale
    
    # Causal mask
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    
    # Softmax
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
    
    # Apply attention
    out = jnp.einsum(\'bhij,bhjd->bhid\', attn_weights, v)
    return out


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    # Query projections.
    q_low2d = _pallas_gemm(x2d, q_down_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV compression.
    kv2d = _pallas_gemm(x2d, kv_down_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]

    # K projection.
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d = _pallas_gemm(k_latent2d, k_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    k_nope = k_nope2d.reshape(B, S, H, nope)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # V projection.
    v2d = _pallas_gemm(k_latent2d, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    v = v2d.reshape(B, S, H, vd)

    # Attention inputs.
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)  # [B, H, S, D]
    v = v.transpose(0, 2, 1, 3)  # [B, H, S, vd]

    # Use JAX-based causal attention (avoids HBM blocking issues)
    out = _causal_attention_jax(q_full, k_full, v)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)

    # Output projection.
    out2d = out.reshape(B * S, H * vd)
    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)

    return final2d.reshape(B, S, E)
''',
score=10.459,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 10.459 ms\n{"correct": true, "latency": 10.459, "error": "", "all_times_ms": [10.438, 10.44, 10.442, 10.444, 10.444, 10.446, 10.447, 10.448, 10.448, 10.449, 10.449, 10.449, 10.45, 10.45, 10.45, 10.45, 10.451, 10.451, 10.451, 10.451, 10.451, 10.452, 10.452, 10.453, 10.453, 10.453, 10.453, 10.454, 10.454, 10.454, 10.454, 10.454, 10.454, 10.455, 10.455, 10.456, 10.456, 10.456, 10.457, 10.457, 10.457, 10.457, 10.457, 10.457, 10.457, 10.458, 10.458, 10.458, 10.459, 10.459, 10.459, 10.459, 10.46, 10.46, 10.46, 10.46, 10.46, 10.46, 10.46, 10.461, 10.462, 10.462, 10.462, 10.462, 10.462, 10.463, 10.463, 10.463, 10.464, 10.464, 10.464, 10.464, 10.464, 10.464, 10.464, 10.466, 10.466, 10.466, 10.466, 10.467, 10.467, 10.468, 10.468, 10.468, 10.469, 10.472, 10.474, 10.474, 10.474, 10.474, 10.475, 10.475, 10.477, 10.479, 10.482, 10.484, 10.484, 10.488, 10.489, 10.513], "max_diff": 0.015625, "max_rel_diff": 0.002151}',
stderr=''),
plan='''### Optimization Plan

The current implementation of the Multi-head Latent Attention (MLA) workload suffers from excessive HBM (High Bandwidth Memory) data movement. Specifically, several matrix multiplication branches operate on the same input activation tensors independently. For instance, the input $x$ is read from HBM twice: once to compute the query latent (`q_down_proj`) and once to compute the KV latent (`kv_down_proj`). Similarly, the compressed KV latent is read twice to compute the content keys (`k_up_proj`) and the values (`v_up_proj`).

We will apply **Strategy 1: reduce data movement** by fusing these parallel projection branches. By concatenating the weight matrices of projections that share a common input, we can perform a single larger GEMM instead of two smaller ones. This allows the TPU to read the activation tensor from HBM into on-chip memory (VMEM) once and use it to compute multiple outputs, effectively halving the activation read bandwidth for those sections of the model.

**Plan:**
1.  **Fuse Query and KV Down-Projections**: Concatenate `q_down_proj` and `kv_down_proj` along the column dimension (axis 1). Call the Pallas GEMM once with the input $x$ and the combined weight. Slice the resulting tensor to recover `q_low` and `kv_latent`.
2.  **Fuse Key and Value Up-Projections**: Concatenate `k_up_proj` and `v_up_proj` along the column dimension (axis 1). Call the Pallas GEMM once with the compressed `k_latent` and the combined weight. Slice the resulting tensor to recover `k_nope` and `v`.
3.  **Maintain Compatibility**: Ensure that the concatenated dimensions and the resulting GEMM tile selections remain valid according to TPU v6e constraints (tile sizes must be multiples of 128 or equal the full dimension).

### Optimized Code

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024  # Increased budget for better throughput

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    # TPU-friendly rotate: reshape to isolate pairs
    s = x.shape
    x_reshaped = x.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return jnp.stack([out1, out2], axis=-1).reshape(s)

def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred
    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate
    return dim

def _estimate_gemm_working_set_bytes(bm: int, bn: int, bk: int, in_dtype, out_dtype) -> int:
    in_b, out_b = jnp.dtype(in_dtype).itemsize, jnp.dtype(out_dtype).itemsize
    return 2 * (bm * bk * in_b + bk * bn * in_b + bm * bn * out_b) + (bm * bn * 4)

def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    # Heuristic to maximize tile size within VMEM budget
    for bm_c in sorted({_choose_tpu_block(M, x, 8) for x in (bm, 512, 256)}, reverse=True):
        for bn_c in sorted({_choose_tpu_block(N, x, 128) for x in (bn, 1024, 512)}, reverse=True):
            for bk_c in sorted({_choose_tpu_block(K, x, 128) for x in (bk, 512, 256)}, reverse=True):
                if _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype, dtype) <= _V6E_GEMM_WORKING_SET_BUDGET:
                    return bm_c, bn_c, bk_c
    return _choose_tpu_block(M, bm, 8), _choose_tpu_block(N, bn, 128), _choose_tpu_block(K, bk, 128)

def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    
    acc_ref[...] += jax.lax.dot_general(
        a_ref[...].astype(jnp.float32), 
        b_ref[...].astype(jnp.float32),
        (((1,), (0,)), ((), ())), 
        preferred_element_type=jnp.float32
    )
    
    @pl.when(k_id == pl.num_programs(2) - 1)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)

def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    N = b.shape[1]
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j))],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(a, b)

def _causal_attention_jax(q, k, v):
    B, H, S, hd = q.shape
    scores = jnp.einsum(\'bhid,bhjd->bhij\', q, k) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.einsum(\'bhij,bhjd->bhid\', attn_weights, v)

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H, nope, rope, vd = C["num_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl, ql = C["kv_lora_rank"], C["q_lora_rank"]
    x2d = x.reshape(B * S, E)

    # Strategy 1: Fuse parallel projections to reduce activation HBM reads
    # Fusion 1: q_down and kv_down share input x2d
    merged_down_weights = jnp.concatenate([q_down_proj, kv_down_proj], axis=1)
    combined_down = _pallas_gemm(x2d, merged_down_weights)
    q_low2d = combined_down[:, :ql]
    kv2d = combined_down[:, ql:]

    # Query up-projection (standalone)
    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # Prepare latent for K/V up-projections
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_latent2d = k_latent.reshape(B * S, kvl)

    # Fusion 2: k_up and v_up share input k_latent2d
    merged_up_weights = jnp.concatenate([k_up_proj, v_up_proj], axis=1)
    combined_up = _pallas_gemm(k_latent2d, merged_up_weights)
    k_nope2d = combined_up[:, :H * nope]
    v2d = combined_up[:, H * nope:]
    
    # RoPE and Attention
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope2d.reshape(B, S, H, nope), k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v2d.reshape(B, S, H, vd).transpose(0, 2, 1, 3)

    attn_out = _causal_attention_jax(q_full, k_full, v_full)
    attn_out_2d = attn_out.transpose(0, 2, 1, 3).reshape(B * S, H * vd)

    # Output projection
    final2d = _pallas_gemm(attn_out_2d, o_proj)
    return final2d.reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# Conservative-but-larger VMEM working-set target for v6e-class single-TensorCore chips.
_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    # Perform elementwise math in f32, then cast back.
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    # Full-dimension blocks are valid on TPU even if not divisible by the native multiple.
    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    # Double-buffered A/B/C plus f32 accumulator scratch.
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _causal_attention_jax(q, k, v):
    """Standard JAX causal attention in f32 with bf16 I/O."""
    B, H, S, hd = q.shape

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    scale = jnp.float32(hd) ** jnp.float32(-0.5)
    scores = jnp.einsum("bhid,bhjd->bhij", q_f32, k_f32) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v_f32)
    return out.astype(v.dtype)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA with fused parallel projections to reduce HBM activation traffic."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    ql = C["q_lora_rank"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    # Fuse q_down and kv_down: both consume x2d.
    down_merged = jnp.concatenate([q_down_proj, kv_down_proj], axis=1)
    down_out2d = _pallas_gemm(x2d, down_merged, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q_low2d = down_out2d[:, :ql]
    kv2d = down_out2d[:, ql:]

    # Query up-projection.
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # Split KV compressed state.
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    # Fuse k_up and v_up: both consume k_latent2d.
    k_latent2d = k_latent.reshape(B * S, kvl)
    up_merged = jnp.concatenate([k_up_proj, v_up_proj], axis=1)
    up_out2d = _pallas_gemm(k_latent2d, up_merged, bm=proj_bm, bn=proj_bn, bk=proj_bk)

    k_nope2d = up_out2d[:, : H * nope]
    v2d = up_out2d[:, H * nope :]

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # Attention inputs: [B, H, S, D]
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    # Causal attention fallback in JAX.
    out = _causal_attention_jax(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    # Output projection.
    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
''',
score=8.291,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='gpt-5.4',
stdout='Latency: 8.291 ms\n{"correct": true, "latency": 8.291, "error": "", "all_times_ms": [7.399, 7.834, 7.861, 8.051, 8.073, 8.116, 8.124, 8.125, 8.125, 8.157, 8.185, 8.194, 8.206, 8.209, 8.21, 8.21, 8.212, 8.214, 8.214, 8.216, 8.221, 8.227, 8.228, 8.234, 8.235, 8.238, 8.242, 8.244, 8.248, 8.25, 8.252, 8.256, 8.258, 8.26, 8.26, 8.262, 8.264, 8.264, 8.265, 8.267, 8.267, 8.268, 8.269, 8.269, 8.276, 8.283, 8.286, 8.289, 8.29, 8.291, 8.291, 8.294, 8.295, 8.295, 8.298, 8.299, 8.299, 8.302, 8.302, 8.304, 8.305, 8.305, 8.306, 8.307, 8.308, 8.309, 8.311, 8.311, 8.311, 8.313, 8.322, 8.324, 8.325, 8.327, 8.33, 8.33, 8.33, 8.331, 8.333, 8.337, 8.337, 8.34, 8.343, 8.343, 8.344, 8.347, 8.351, 8.352, 8.359, 8.364, 8.366, 8.387, 8.407, 8.448, 8.46, 8.472, 8.48, 8.576, 8.693, 8.808], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''**Selected strategy: 11. Replace concatenate-then-slice patterns with direct multi-output GEMM kernels writing to separate destination buffers.**

### Why this is the best next step for this code on **v6e-1**
The clearest waste in the current program is here:

```python
down_merged = jnp.concatenate([q_down_proj, kv_down_proj], axis=1)
down_out2d = _pallas_gemm(x2d, down_merged, ...)
q_low2d = down_out2d[:, :ql]
kv2d = down_out2d[:, ql:]
```

and here:

```python
up_merged = jnp.concatenate([k_up_proj, v_up_proj], axis=1)
up_out2d = _pallas_gemm(k_latent2d, up_merged, ...)
k_nope2d = up_out2d[:, : H * nope]
v2d = up_out2d[:, H * nope :]
```

Those patterns create large temporary merged weights and large temporary merged outputs that are immediately split again. On a v6e-1, that is expensive because the TensorCore is single-core and HBM traffic is often the bottleneck.

The worst offender is `up_out2d`:
- shape = `(2048, 32768)` bf16
- size ≈ **128 MiB**
- but the final consumers only need the two halves separately

So the plan is to keep the same `workload(...)` signature and semantics, but replace those two merged GEMMs with **dual-output Pallas GEMMs** that:
- load one `A` tile once,
- multiply it against two different `B` tiles,
- accumulate into two separate f32 scratch buffers,
- write directly into two output refs.

---

## Concrete plan

### 1) Add a dual-output GEMM kernel for the down-projection split
Replace:

```python
down_merged = jnp.concatenate([q_down_proj, kv_down_proj], axis=1)
down_out2d = _pallas_gemm(x2d, down_merged, ...)
q_low2d = down_out2d[:, :ql]
kv2d = down_out2d[:, ql:]
```

with something like:

```python
q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)
```

#### Kernel shape choice
For this specific case:
- `A = x2d`: `(2048, 7168)`
- `B0 = q_down_proj`: `(7168, 1536)`
- `B1 = kv_down_proj`: `(7168, 576)`

Use a grid over **M and K only**:
- `grid = (M // bm, K // bk)`

Reason: `N0=1536` and `N1=576` are different widths, so this kernel should compute a full row-block for both outputs at once.

Suggested block sizes for v6e-1:
- `bm = 256`
- `bk = 256`

Then use full-width output blocks:
- `q_low` block shape: `(256, 1536)`  
- `kv` block shape: `(256, 576)`  

This is legal on TPU because the last dimension may equal the full array dimension even if not divisible by 128 (`576` is allowed because it is the full axis).

#### Kernel structure
The Pallas kernel should:
- read `a_ref[...]`, `b0_ref[...]`, `b1_ref[...]`
- zero both accumulator refs on the first `k` iteration
- do two `dot_general(..., preferred_element_type=jnp.float32)`
- store both outputs on the last `k` iteration
- return `None`

Use:
- `dimension_semantics=("parallel", "arbitrary")`
- reduction axis last so the accumulators persist correctly across K tiles

---

### 2) Add a dual-output GEMM kernel for the K/V up-projection split
Replace:

```python
up_merged = jnp.concatenate([k_up_proj, v_up_proj], axis=1)
up_out2d = _pallas_gemm(k_latent2d, up_merged, ...)
k_nope2d = up_out2d[:, : H * nope]
v2d = up_out2d[:, H * nope :]
```

with:

```python
k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent2d, k_up_proj, v_up_proj)
```

#### Why this one is especially good
Here both outputs have the same width:
- `k_up_proj`: `(512, 16384)`
- `v_up_proj`: `(512, 16384)`

So this dual-output GEMM can use the normal 3D GEMM grid:
- `grid = (M // bm, N // bn, K // bk)`

Suggested starting tiles:
- `bm = 256`
- `bn = 512`
- `bk = 256` or `bk = 128` if your dual-output VMEM estimate is tighter

Use the same `N` tiling for both outputs.

#### Kernel structure
Same idea as above, but with two output refs and two accumulator refs:
- `acc_k_ref`
- `acc_v_ref`

and:
- `dimension_semantics=("parallel", "parallel", "arbitrary")`

Again, `K` must remain the innermost grid dimension.

---

### 3) Update tile selection to account for **two outputs**
Do **not** reuse `_pick_gemm_tiles` unchanged for the dual-output kernels, because it only budgets for one output/accumulator pair.

Add a dedicated working-set estimator for the dual-output cases:
- double-buffered `A`
- double-buffered `B0`
- double-buffered `B1`
- double-buffered `C0`
- double-buffered `C1`
- `f32` accumulator for output 0
- `f32` accumulator for output 1

For the down-projection split, the full-width row-block design above should still fit in v6e VMEM.  
For the K/V split, keep `bn` conservative enough that both accumulators fit comfortably.

---

### 4) Leave the rest of `workload(...)` unchanged
Keep these parts as they are in this phase:
- `_compute_rope`
- `_apply_rope`
- `q_up` GEMM
- JAX causal attention
- output projection GEMM

That keeps the optimization tightly scoped to just one strategy and minimizes regression risk.

---

## What changes in `workload`
Only these two sections change:

### Before
```python
down_merged = jnp.concatenate([q_down_proj, kv_down_proj], axis=1)
down_out2d = _pallas_gemm(x2d, down_merged, bm=proj_bm, bn=proj_bn, bk=proj_bk)
q_low2d = down_out2d[:, :ql]
kv2d = down_out2d[:, ql:]
```

### After
```python
q_low2d, kv2d = _pallas_dual_gemm_down(
    x2d, q_down_proj, kv_down_proj, bm=256, bk=256
)
```

and

### Before
```python
up_merged = jnp.concatenate([k_up_proj, v_up_proj], axis=1)
up_out2d = _pallas_gemm(k_latent2d, up_merged, bm=proj_bm, bn=proj_bn, bk=proj_bk)

k_nope2d = up_out2d[:, : H * nope]
v2d = up_out2d[:, H * nope :]
```

### After
```python
k_nope2d, v2d = _pallas_dual_gemm_same_n(
    k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
)
```

---

## Why this should reduce latency
This removes:
- the large merged weight tensors,
- the large intermediate merged output tensors,
- the extra split/slice traffic after GEMM.

In particular, it avoids materializing:
- `down_out2d` as a merged output buffer
- `up_out2d` as a **128 MiB** merged output buffer

while keeping the compute nearly identical.

So the expected win is primarily from **less HBM traffic and fewer large temporaries**, with minimal algorithmic risk.

---

## Numerical behavior
This remains semantically equivalent within normal bf16 tolerance because:
- inputs/outputs stay bf16,
- the two GEMMs still use `preferred_element_type=jnp.float32`,
- accumulation order across K tiles remains the same structure as the current `_pallas_gemm`.

If you want, I can next turn this plan into the actual rewritten code while keeping the same `workload(...)` signature.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# Conservative-but-larger VMEM working-set target for v6e-class single-TensorCore chips.
_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    # Perform elementwise math in f32, then cast back.
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    # Full-dimension blocks are valid on TPU even if not divisible by the native multiple.
    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    # Double-buffered A/B/C plus f32 accumulator scratch.
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _estimate_dual_gemm_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    """Estimate VMEM working set for dual-output GEMM with same N tiling."""
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    # Double-buffered A, B0, B1, C0, C1 plus two f32 accumulators
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_tiles_same_n(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    """Pick tiles for dual-output GEMM where both outputs have width N."""
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                # Two outputs of the same width
                working_set = _estimate_dual_gemm_working_set_bytes(
                    bm_c, bn_c, bn_c, bk_c, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    """Kernel for dual-output GEMM where both outputs have the same N dimension."""
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=256, bn=512, bk=256):
    """Dual-output GEMM: C0 = A @ B0, C1 = A @ B1, where B0 and B1 have the same shape."""
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    if N0 != N1:
        raise ValueError(f"Output widths must match for same_n variant: N0={N0}, N1={N1}")

    N = N0
    bm_actual, bn_actual, bk_actual = _pick_dual_gemm_tiles_same_n(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N), a.dtype),
            jax.ShapeDtypeStruct((M, N), a.dtype),
        ],
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _estimate_dual_gemm_down_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    """Estimate VMEM working set for dual-output down-projection GEMM."""
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    # Double-buffered A, B0, B1, C0, C1 plus two f32 accumulators
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_down_tiles(M: int, N0: int, N1: int, K: int, dtype, bm: int, bk: int):
    """Pick tiles for dual-output down-projection GEMM with full-width N blocks."""
    bm_candidates = _candidate_blocks(M, bm, 8)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bk_c in bk_candidates:
            # Full-width outputs
            working_set = _estimate_dual_gemm_down_working_set_bytes(
                bm_c, N0, N1, bk_c, dtype
            )
            if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                continue
            score = (bm_c * bk_c, bm_c, bk_c)
            if best_score is None or score > best_score:
                best_score = score
                best = (bm_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    """Kernel for dual-output down-projection GEMM with full-width N blocks."""
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=256, bk=256):
    """Dual-output GEMM for down-projection: C0 = A @ B0, C1 = A @ B1, with full-width N blocks."""
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")

    bm_actual, bk_actual = _pick_dual_gemm_down_tiles(M, N0, N1, K, a.dtype, bm, bk)

    if M % bm_actual != 0 or K % bk_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N0), a.dtype),
            jax.ShapeDtypeStruct((M, N1), a.dtype),
        ],
        grid=(M // bm_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, k: (i, k)),
            pl.BlockSpec((bk_actual, N0), lambda i, k: (k, 0)),
            pl.BlockSpec((bk_actual, N1), lambda i, k: (k, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, N0), lambda i, k: (i, 0)),
            pl.BlockSpec((bm_actual, N1), lambda i, k: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, N0), jnp.float32),
            pltpu.VMEM((bm_actual, N1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _causal_attention_jax(q, k, v):
    """Standard JAX causal attention in f32 with bf16 I/O."""
    B, H, S, hd = q.shape

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    scale = jnp.float32(hd) ** jnp.float32(-0.5)
    scores = jnp.einsum("bhid,bhjd->bhij", q_f32, k_f32) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v_f32)
    return out.astype(v.dtype)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA with fused parallel projections to reduce HBM activation traffic."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    ql = C["q_lora_rank"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    # Dual-output GEMM for down-projection: avoids concatenating weights and splitting output
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj, bm=256, bk=256)

    # Query up-projection.
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # Split KV compressed state.
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    # Dual-output GEMM for K/V up-projection: avoids concatenating weights and splitting output
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(
        k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
    )

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # RoPE.
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # Attention inputs: [B, H, S, D]
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    # Causal attention fallback in JAX.
    out = _causal_attention_jax(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    # Output projection.
    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
''',
score=8.205,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 8.205 ms\n{"correct": true, "latency": 8.205, "error": "", "all_times_ms": [7.313, 7.865, 8.003, 8.04, 8.055, 8.063, 8.087, 8.091, 8.095, 8.096, 8.096, 8.1, 8.101, 8.102, 8.105, 8.108, 8.11, 8.111, 8.118, 8.119, 8.126, 8.129, 8.132, 8.135, 8.135, 8.135, 8.136, 8.14, 8.14, 8.144, 8.149, 8.149, 8.156, 8.16, 8.16, 8.162, 8.166, 8.167, 8.169, 8.171, 8.174, 8.175, 8.175, 8.175, 8.176, 8.18, 8.18, 8.194, 8.197, 8.198, 8.205, 8.208, 8.215, 8.216, 8.222, 8.226, 8.226, 8.229, 8.232, 8.233, 8.236, 8.237, 8.238, 8.239, 8.239, 8.242, 8.242, 8.246, 8.246, 8.247, 8.249, 8.252, 8.252, 8.252, 8.253, 8.256, 8.261, 8.261, 8.266, 8.273, 8.276, 8.286, 8.301, 8.301, 8.301, 8.312, 8.316, 8.321, 8.321, 8.327, 8.329, 8.333, 8.336, 8.336, 8.347, 8.35, 8.354, 8.367, 8.4, 8.801], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''

Looking at the code, I\'ll apply Strategy 1: **reduce data movement** by implementing the attention computation as a custom Pallas kernel instead of using JAX operations that may execute less efficiently.

The key inefficiency is that `_causal_attention_jax` runs as standard JAX code, which requires moving data from VMEM to registers through JAX abstractions rather than using optimized Pallas primitives. This prevents efficient data reuse and may not fully overlap compute with memory transfers.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024

# Block sizes for attention kernel - aligned to 8x128 tile boundaries
_ATTN_BH = 8  # heads per block (sublanes)
_ATTN_BS = 128  # sequence length per block (lanes)


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred
    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate
    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(bm: int, bn: int, bk: int, in_dtype, out_dtype) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple
    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype, dtype)
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")
    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _estimate_dual_gemm_working_set_bytes(bm: int, bn0: int, bn1: int, bk: int, in_dtype) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_tiles_same_n(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_dual_gemm_working_set_bytes(bm_c, bn_c, bn_c, bk_c, dtype)
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=256, bn=512, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape
    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    if N0 != N1:
        raise ValueError(f"Output widths must match for same_n variant: N0={N0}, N1={N1}")
    N = N0
    bm_actual, bn_actual, bk_actual = _pick_dual_gemm_tiles_same_n(M, N, K, a.dtype, bm, bn, bk)
    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N), a.dtype),
            jax.ShapeDtypeStruct((M, N), a.dtype),
        ],
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _estimate_dual_gemm_down_working_set_bytes(bm: int, bn0: int, bn1: int, bk: int, in_dtype) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_down_tiles(M: int, N0: int, N1: int, K: int, dtype, bm: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bk_c in bk_candidates:
            working_set = _estimate_dual_gemm_down_working_set_bytes(bm_c, N0, N1, bk_c, dtype)
            if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                continue
            score = (bm_c * bk_c, bm_c, bk_c)
            if best_score is None or score > best_score:
                best_score = score
                best = (bm_c, bk_c)
    if best is not None:
        return best
    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=256, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape
    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    bm_actual, bk_actual = _pick_dual_gemm_down_tiles(M, N0, N1, K, a.dtype, bm, bk)
    if M % bm_actual != 0 or K % bk_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}"
        )
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N0), a.dtype),
            jax.ShapeDtypeStruct((M, N1), a.dtype),
        ],
        grid=(M // bm_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, k: (i, k)),
            pl.BlockSpec((bk_actual, N0), lambda i, k: (k, 0)),
            pl.BlockSpec((bk_actual, N1), lambda i, k: (k, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, N0), lambda i, k: (i, 0)),
            pl.BlockSpec((bm_actual, N1), lambda i, k: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, N0), jnp.float32),
            pltpu.VMEM((bm_actual, N1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b0, b1)


# === Custom Pallas attention kernel to reduce data movement ===
def _attention_kernel(q_ref, k_ref, v_ref, out_ref, scratch_qk_ref, acc_scratch_ref):
    """Pallas kernel for causal attention with fused softmax.
    
    Uses block-sparse attention where each block processes:
    - bh along sublanes (8 heads per block)
    - sq along lanes (128 sequence positions per block)
    
    This keeps QK^T and softmax intermediate results in VMEM scratch
    rather than going through slow JAX reductions.
    """
    B = q_ref.shape[0]
    H = q_ref.shape[1]
    S = q_ref.shape[2]
    D = q_ref.shape[3]  # head dim (nope + rope)
    
    bh_idx = pl.program_id(0)  # block over heads
    bs_idx = pl.program_id(1)  # block over sequence
    
    num_bh_blocks = pl.num_programs(0)
    num_bs_blocks = pl.num_programs(1)
    
    # Each block processes _ATTN_BH heads and _ATTN_BS sequence positions
    h_start = bh_idx * _ATTN_BH
    h_end = min(h_start + _ATTN_BH, H)
    s_start = bs_idx * _ATTN_BS
    s_end = min(s_start + _ATTN_BS, S)
    
    actual_bh = h_end - h_start
    actual_bs = s_end - s_start
    
    # Load block of Q: [B, actual_bh, actual_bs, D]
    # We\'ll process one batch element at a time for better memory reuse
    for b in range(B):
        q_block = q_ref[b, h_start:h_end, s_start:s_end, :]
        
        # Compute QK^T for all relevant key positions (causal: only previous positions)
        # For output at position s, we need keys from positions 0 to s
        #
        # We accumulate into scratch buffer: [actual_bs, s_end] in f32
        # This avoids materializing the full attention matrix
        
        # Initialize accumulation
        if bh_idx == 0 and bs_idx == 0:
            acc_scratch_ref[...] = jnp.full(
                (actual_bs, s_end), -jnp.inf, dtype=jnp.float32
            )
        
        # Loop over key blocks (reduction over K)
        for k_idx in range(num_bs_blocks):
            k_start = k_idx * _ATTN_BS
            k_end = min(k_start + _ATTN_BS, S)
            actual_k = k_end - k_start
            
            # Load K block: [B, H, actual_k, D]
            k_block = k_ref[b, :, k_start:k_end, :]
            
            # Compute Q @ K^T for this k block: [actual_bh, actual_bs] @ [actual_bh, actual_k] -> [actual_bs, actual_k]
            # But we need to sum over head dim - use einsum
            
            # Actually simpler: compute attention per head
            # q: [actual_bh, actual_bs, D], k: [actual_bh, actual_k, D]
            # result: [actual_bs, actual_k]
            
            # We need to scale by head_dim
            head_dim = D
            scale = jnp.float32(head_dim) ** jnp.float32(-0.5)
            
            # Take only the heads we\'re processing from k
            k_block_local = k_ref[b, h_start:h_end, k_start:k_end, :]
            
            # q_block: [actual_bh, actual_bs, D] -> transpose to [actual_bs, actual_bh, D]
            q_t = jnp.transpose(q_block, (1, 0, 2)).astype(jnp.float32)
            k_t = jnp.transpose(k_block_local, (1, 2, 0)).astype(jnp.float32)
            
            # q_t: [actual_bs, actual_bh, D], k_t: [actual_bh, D, actual_k]
            # We need to compute q_t @ k_t for each head and sum
            # This is expensive in VMEM, let\'s do simpler approach
            
        # Simple approach: compute full attention for this block\'s query positions
        # against all key positions up to current position
        q_f32 = q_block.astype(jnp.float32)  # [actual_bh, actual_bs, D]
        
        # Load all K for these heads
        k_full = k_ref[b, h_start:h_end, :s_end, :].astype(jnp.float32)  # [actual_bh, s_end, D]
        
        # q_f32: [actual_bh, actual_bs, D], k_full: [actual_bh, s_end, D]
        # Need scores[bh, sq, sk] = sum_d q[bh, sq, d] * k[bh, sk, d]
        
        # Compute QK^T
        # q_f32: [actual_bh, actual_bs, D]
        # Transpose k_full to [actual_bh, D, s_end]
        k_t = jnp.transpose(k_full, (0, 2, 1))
        
        # scores[bh, sq, sk] = sum_d q_f32[bh, sq, d] * k_t[bh, d, sk]
        # Use dot general: contract q\'s head dim with k\'s head dim
        scores = jax.lax.dot_general(
            q_f32, k_t,
            dimension_numbers=(((0,), (0,)), ((2,), (1,))),
            preferred_element_type=jnp.float32,
        )  # [actual_bh, actual_bs, s_end]
        
        # Transpose to [actual_bs, actual_bh, s_end]
        scores = jnp.transpose(scores, (1, 0, 2))
        
        # Apply causal mask - only valid for s_idx <= k_idx within current block
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((s_end, s_end), dtype=jnp.bool_))
        
        # Apply mask and scale
        scores_scaled = scores * jnp.float32(scale)
        
        # Mask is per batch/head
        scores_masked = jnp.where(
            causal_mask[None, None, :, :],
            scores_scaled,
            jnp.float32(-1e9)
        )
        
        # Softmax over last dimension (key positions)
        scores_max = jnp.max(scores_masked, axis=-1, keepdims=True)
        scores_exp = jnp.exp(scores_masked - scores_max)
        attn_weights = scores_exp / jnp.sum(scores_exp, axis=-1, keepdims=True)
        
        # Load V: [B, H, S, vd]
        v_full = v_ref[b, :, :, :].astype(jnp.float32)  # [H, s_end, vd]
        v_local = v_full[h_start:h_end, :s_end, :]  # [actual_bh, s_end, vd]
        
        # Transpose attention weights: [actual_bs, actual_bh, s_end] -> [actual_bh, s_end, actual_bs]
        attn_weights_t = jnp.transpose(attn_weights, (1, 2, 0))
        
        # Compute output: attn_weights [actual_bh, s_end, actual_bs] @ v_local [actual_bh, s_end, vd]
        # Result: [actual_bh, actual_bs, vd]
        
        # Actually we need to sum over key positions: out[bh, sq, vd] = sum_sk attn[bh, sq, sk] * v[bh, sk, vd]
        # transposed v: [actual_bh, vd, s_end]
        v_t = jnp.transpose(v_local, (0, 2, 1))
        
        out_block = jax.lax.dot_general(
            attn_weights_t, v_t,
            dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )  # [actual_bh, actual_bs, vd]
        
        # Write output
        out_ref[b, h_start:h_end, s_start:s_end, :] = out_block.astype(out_ref.dtype)


def _causal_attention_pallas(q, k, v):
    """Pallas kernel for causal attention.
    
    q: [B, H, S, D] where D = nope + rope
    k: [B, H, S, nope] 
    v: [B, H, S, vd]
    
    Returns: [B, H, S, vd]
    """
    B, H, S, _ = q.shape
    _, _, _, vd = v.shape
    
    # Block sizes aligned to 8x128
    bh_block = _ATTN_BH
    bs_block = _ATTN_BS
    
    # Ensure divisibility
    num_bh_blocks = (H + bh_block - 1) // bh_block
    num_bs_blocks = (S + bs_block - 1) // bs_block
    
    # Create padded versions if needed
    padded_H = num_bh_blocks * bh_block
    padded_S = num_bs_blocks * bs_block
    
    # Pad inputs
    def pad_to(x, target_h, target_s):
        # x: [B, H, S, D]
        pad_h = target_h - x.shape[1]
        pad_s = target_s - x.shape[2]
        return jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_s), (0, 0)), 
                       mode=\'constant\', constant_values=0)
    
    q_pad = pad_to(q, padded_H, padded_S)
    k_pad = pad_to(k, padded_H, padded_S)
    v_pad = pad_to(v, padded_H, padded_S)
    
    # Output shape matches input (we\'ll crop if padded)
    out = pl.pallas_call(
        _attention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, padded_H, padded_S, vd), q.dtype),
        grid=(num_bh_blocks, num_bs_blocks),
        in_specs=[
            pl.BlockSpec((1, bh_block, bs_block, q.shape[-1]), 
                        lambda i, j: (0, i, j, 0)),
            pl.BlockSpec((1, bh_block, bs_block, k.shape[-1]), 
                        lambda i, j: (0, i, j, 0)),
            pl.BlockSpec((1, bh_block, bs_block, v.shape[-1]), 
                        lambda i, j: (0, i, j, 0)),
        ],
        out_specs=pl.BlockSpec((1, bh_block, bs_block, vd),
                               lambda i, j: (0, i, j, 0)),
        scratch_shapes=[
            pltpu.VMEM((bh_block, bs_block, q.shape[-1]), jnp.float32),  # scratch_qk
            pltpu.VMEM((bs_block, padded_S), jnp.float32),  # acc_scratch
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
    )(q_pad, k_pad, v_pad)
    
    # Crop to original size
    return out[:, :H, :S, :]


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA with fused parallel projections and Pallas attention kernel."""
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    ql = C["q_lora_rank"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    # Dual-output GEMM for down-projection
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj, bm=256, bk=256)

    # Query up-projection
    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # Split KV compressed state
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    # Dual-output GEMM for K/V up-projection
    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(
        k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
    )

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # RoPE
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    # Attention inputs: [B, H, S, D]
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    # Use Pallas attention kernel instead of JAX
    out = _causal_attention_pallas(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    # Output projection
    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# Conservative VMEM working-set target for v6e-class single-TensorCore chips.
_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024

# Attention tiling for v6e-1.
# H block is on a non-minor axis, so 8 heads matches sublane granularity well.
# Query/key sequence blocks use 128, aligned to the lane dimension.
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 128
_ATTN_KEY_BLOCK = 128


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _estimate_dual_gemm_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_tiles_same_n(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_dual_gemm_working_set_bytes(
                    bm_c, bn_c, bn_c, bk_c, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=256, bn=512, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    if N0 != N1:
        raise ValueError(f"Output widths must match: N0={N0}, N1={N1}")

    N = N0
    bm_actual, bn_actual, bk_actual = _pick_dual_gemm_tiles_same_n(
        M, N, K, a.dtype, bm, bn, bk
    )

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N), a.dtype),
            jax.ShapeDtypeStruct((M, N), a.dtype),
        ],
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _estimate_dual_gemm_down_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_down_tiles(M: int, N0: int, N1: int, K: int, dtype, bm: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bk_c in bk_candidates:
            working_set = _estimate_dual_gemm_down_working_set_bytes(
                bm_c, N0, N1, bk_c, dtype
            )
            if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                continue
            score = (bm_c * bk_c, bm_c, bk_c)
            if best_score is None or score > best_score:
                best_score = score
                best = (bm_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=256, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")

    bm_actual, bk_actual = _pick_dual_gemm_down_tiles(M, N0, N1, K, a.dtype, bm, bk)

    if M % bm_actual != 0 or K % bk_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N0), a.dtype),
            jax.ShapeDtypeStruct((M, N1), a.dtype),
        ],
        grid=(M // bm_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, k: (i, k)),
            pl.BlockSpec((bk_actual, N0), lambda i, k: (k, 0)),
            pl.BlockSpec((bk_actual, N1), lambda i, k: (k, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, N0), lambda i, k: (i, 0)),
            pl.BlockSpec((bm_actual, N1), lambda i, k: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, N0), jnp.float32),
            pltpu.VMEM((bm_actual, N1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _causal_attention_jax(q, k, v):
    B, H, S, hd = q.shape

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    scale = jnp.float32(hd) ** jnp.float32(-0.5)
    scores = jnp.einsum("bhid,bhjd->bhij", q_f32, k_f32) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v_f32)
    return out.astype(v.dtype)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p,
        v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    num_k_blocks = k_ref.shape[2] // _ATTN_KEY_BLOCK

    q_f32 = q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** jnp.float32(-0.5)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK
        k_end = k_start + _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale

            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            scores = jnp.where(
                causal_mask,
                scores,
                jnp.full(scores.shape, jnp.float32(-1.0e30)),
            )

            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _attention_input_spec(block_shape, index_map):
    return pl.BlockSpec(
        block_shape,
        index_map,
        pipeline_mode=pl.Buffered(buffer_count=1),
    )


def _causal_attention_pallas(q, k, v):
    B, H, S, D = q.shape
    _, Hk, Sk, Dk = k.shape
    _, Hv, Sv, Dv = v.shape

    if H != Hk or H != Hv or S != Sk or S != Sv:
        raise ValueError("q, k, v must agree on batch/head/sequence dimensions")

    # This kernel is specialized for the target v6e-1 shape regime.
    # Fall back for incompatible shapes to preserve semantics.
    if (
        H % _ATTN_HEAD_BLOCK != 0
        or S % _ATTN_QUERY_BLOCK != 0
        or S % _ATTN_KEY_BLOCK != 0
    ):
        return _causal_attention_jax(q, k, v)

    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, D),
                lambda b, h, q_blk: (b, h, q_blk, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dk),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dv),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv),
            lambda b, h, q_blk: (b, h, q_blk, 0),
            pipeline_mode=pl.Buffered(buffer_count=1),
        ),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            # q block is last so consecutive iterations reuse the same full-S K/V tiles.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(q, k, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    ql = C["q_lora_rank"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    q_low2d, kv2d = _pallas_dual_gemm_down(
        x2d, q_down_proj, kv_down_proj, bm=256, bk=256
    )

    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(
        k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
    )

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    out = _causal_attention_pallas(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
''',
score=6.613,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='gpt-5.4',
stdout='Latency: 6.613 ms\n{"correct": true, "latency": 6.613, "error": "", "all_times_ms": [6.596, 6.596, 6.599, 6.599, 6.599, 6.599, 6.6, 6.6, 6.6, 6.6, 6.601, 6.603, 6.603, 6.603, 6.604, 6.605, 6.606, 6.606, 6.606, 6.606, 6.606, 6.606, 6.607, 6.608, 6.608, 6.608, 6.608, 6.609, 6.609, 6.609, 6.609, 6.61, 6.61, 6.61, 6.61, 6.61, 6.611, 6.611, 6.611, 6.611, 6.612, 6.612, 6.612, 6.612, 6.612, 6.613, 6.613, 6.613, 6.613, 6.613, 6.613, 6.613, 6.613, 6.614, 6.614, 6.614, 6.615, 6.615, 6.615, 6.615, 6.615, 6.616, 6.617, 6.617, 6.617, 6.617, 6.617, 6.617, 6.618, 6.618, 6.618, 6.618, 6.618, 6.618, 6.618, 6.619, 6.619, 6.619, 6.619, 6.62, 6.62, 6.62, 6.621, 6.621, 6.622, 6.623, 6.623, 6.623, 6.623, 6.623, 6.624, 6.624, 6.625, 6.626, 6.627, 6.628, 6.629, 6.631, 6.632, 6.663], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''The primary inefficiency in the provided code is the **lack of double buffering** for HBM↔VMEM transfers in the general matrix multiplication (`_gemm_kernel`) and dual matrix multiplication kernels. In the provided kernels, `pipeline_mode` is not specified in the `BlockSpec`s for the inputs and outputs. Since the default pipeline mode on TPU is typically double-buffered, explicitly configuring it confirms the intent, but inspecting the `_pallas_dual_gemm_down` function reveals a specific control flow issue that prevents optimal pipelining.

In `_pallas_dual_gemm_down`, the `index_map` for the output `BlockSpec`s is `lambda i, k: (i, 0)`. This means that for a fixed `i` (M-block) and changing `k` (K-block), the kernel writes to the **same** output block `(i, 0)` in VMEM.

The kernel accumulates results into this output block via `acc_ref` (which aliases the output block):
1. The `dimension_semantics` is `("parallel", "arbitrary")`, placing the reduction dimension `k` (the innermost grid axis) last. This is correct for maintaining the accumulator in VMEM.
2. Inside the kernel, the logic is: `acc_ref[...] += ...`.

However, without double buffering for the **output**, the runtime must ensure that the "ping" buffer is fully written by iteration `k` before it can be potentially read/transferred (conceptually) or overwritten. More critically, the **input** buffers (`a_ref`, `b0_ref`, `b1_ref`) are not double-buffered. This breaks the software pipelining capability. Software pipelining allows the DMA subunits to fetch data for iteration `k+1` while the TensorCore computes iteration `k`. Without double buffering on the inputs, the DMA engine cannot prefetch the next tile, forcing the compute unit to stall while waiting for HBM loads at the start of every iteration `k`.

The solution is to enable double buffering for all input arguments in the GEMM kernels, particularly in `_pallas_dual_gemm_down` where the K-dimension reduction is explicitly managed by the grid loop.

**Plan:**
1.  Modify `_pallas_gemm` to specify `pipeline_mode=pl.Buffered(buffer_count=2)` for all `BlockSpec`s in `in_specs` and `out_specs`.
2.  Modify `_pallas_dual_gemm_same_n` similarly.
3.  Modify `_pallas_dual_gemm_down` similarly. This specifically enables overlapping the `a`, `b0`, and `b1` loads for the next `k`-iteration with the current matrix multiplication.
4.  This change requires no modification to the kernel bodies (`_gemm_kernel`, etc.) as the buffering logic is handled by the `pallas_call` runtime and the explicit accumulator buffer management inside the kernels is already compatible (initializing on `k==0` and writing out on `k==last`).

This directly applies **Strategy 3 (double buffering)**. By allowing the hardware to overlap memory transfers with computation, we hide the HBM access latency (~819 GB/s vs compute throughput), reducing the overall execution time of the dominant GEMM operations in the DeepSeek MLA workload.

Reference implementation for the modification:

```python
# In _pallas_gemm
    in_specs=[
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
    ],
    out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),

# In _pallas_dual_gemm_same_n
    in_specs=[
        pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
    ],
    out_specs=[
        pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),
    ],

# In _pallas_dual_gemm_down
    in_specs=[
        pl.BlockSpec((bm_actual, bk_actual), lambda i, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bk_actual, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bk_actual, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
    ],
    out_specs=[
        pl.BlockSpec((bm_actual, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
        pl.BlockSpec((bm_actual, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
    ],
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# Conservative VMEM working-set target for v6e-class single-TensorCore chips.
_V6E_GEMM_WORKING_SET_BUDGET = 8 * 1024 * 1024

# Attention tiling for v6e-1.
# H block is on a non-minor axis, so 8 heads matches sublane granularity well.
# Query/key sequence blocks use 128, aligned to the lane dimension.
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 128
_ATTN_KEY_BLOCK = 128


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    # Double buffering: 2x for each input and output
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=256, bn=512, bk=256):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        out_specs=pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _estimate_dual_gemm_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    # Double buffering: 2x for each input and output
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_tiles_same_n(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_dual_gemm_working_set_bytes(
                    bm_c, bn_c, bn_c, bk_c, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=256, bn=512, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    if N0 != N1:
        raise ValueError(f"Output widths must match: N0={N0}, N1={N1}")

    N = N0
    bm_actual, bn_actual, bk_actual = _pick_dual_gemm_tiles_same_n(
        M, N, K, a.dtype, bm, bn, bk
    )

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N), a.dtype),
            jax.ShapeDtypeStruct((M, N), a.dtype),
        ],
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bk_actual, bn_actual), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bm_actual, bn_actual), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _estimate_dual_gemm_down_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    # Double buffering: 2x for each input and output
    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_down_tiles(M: int, N0: int, N1: int, K: int, dtype, bm: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bk_c in bk_candidates:
            working_set = _estimate_dual_gemm_down_working_set_bytes(
                bm_c, N0, N1, bk_c, dtype
            )
            if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                continue
            score = (bm_c * bk_c, bm_c, bk_c)
            if best_score is None or score > best_score:
                best_score = score
                best = (bm_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=256, bk=256):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")

    bm_actual, bk_actual = _pick_dual_gemm_down_tiles(M, N0, N1, K, a.dtype, bm, bk)

    if M % bm_actual != 0 or K % bk_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N0), a.dtype),
            jax.ShapeDtypeStruct((M, N1), a.dtype),
        ],
        grid=(M // bm_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec((bm_actual, bk_actual), lambda i, k: (i, k), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bk_actual, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bk_actual, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_actual, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
            pl.BlockSpec((bm_actual, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(buffer_count=2)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, N0), jnp.float32),
            pltpu.VMEM((bm_actual, N1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _causal_attention_jax(q, k, v):
    B, H, S, hd = q.shape

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    scale = jnp.float32(hd) ** jnp.float32(-0.5)
    scores = jnp.einsum("bhid,bhjd->bhij", q_f32, k_f32) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v_f32)
    return out.astype(v.dtype)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p,
        v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    num_k_blocks = k_ref.shape[2] // _ATTN_KEY_BLOCK

    q_f32 = q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** jnp.float32(-0.5)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK
        k_end = k_start + _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale

            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            scores = jnp.where(
                causal_mask,
                scores,
                jnp.full(scores.shape, jnp.float32(-1.0e30)),
            )

            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _attention_input_spec(block_shape, index_map):
    return pl.BlockSpec(
        block_shape,
        index_map,
        pipeline_mode=pl.Buffered(buffer_count=1),
    )


def _causal_attention_pallas(q, k, v):
    B, H, S, D = q.shape
    _, Hk, Sk, Dk = k.shape
    _, Hv, Sv, Dv = v.shape

    if H != Hk or H != Hv or S != Sk or S != Sv:
        raise ValueError("q, k, v must agree on batch/head/sequence dimensions")

    # This kernel is specialized for the target v6e-1 shape regime.
    # Fall back for incompatible shapes to preserve semantics.
    if (
        H % _ATTN_HEAD_BLOCK != 0
        or S % _ATTN_QUERY_BLOCK != 0
        or S % _ATTN_KEY_BLOCK != 0
    ):
        return _causal_attention_jax(q, k, v)

    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, D),
                lambda b, h, q_blk: (b, h, q_blk, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dk),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dv),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv),
            lambda b, h, q_blk: (b, h, q_blk, 0),
            pipeline_mode=pl.Buffered(buffer_count=1),
        ),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            # q block is last so consecutive iterations reuse the same full-S K/V tiles.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(q, k, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    ql = C["q_lora_rank"]
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 256, 512, 256

    x2d = x.reshape(B * S, E)

    q_low2d, kv2d = _pallas_dual_gemm_down(
        x2d, q_down_proj, kv_down_proj, bm=256, bk=256
    )

    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(
        k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
    )

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    out = _causal_attention_pallas(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
''',
score=6.609,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 6.609 ms\n{"correct": true, "latency": 6.609, "error": "", "all_times_ms": [6.591, 6.592, 6.594, 6.595, 6.596, 6.597, 6.597, 6.598, 6.598, 6.599, 6.599, 6.6, 6.6, 6.6, 6.601, 6.601, 6.601, 6.601, 6.602, 6.602, 6.602, 6.602, 6.602, 6.603, 6.603, 6.603, 6.603, 6.603, 6.604, 6.604, 6.605, 6.605, 6.605, 6.605, 6.605, 6.606, 6.606, 6.607, 6.607, 6.607, 6.607, 6.607, 6.607, 6.607, 6.608, 6.608, 6.608, 6.608, 6.608, 6.608, 6.609, 6.609, 6.609, 6.609, 6.609, 6.609, 6.61, 6.61, 6.61, 6.61, 6.61, 6.61, 6.61, 6.61, 6.611, 6.611, 6.611, 6.611, 6.612, 6.612, 6.612, 6.613, 6.613, 6.613, 6.613, 6.613, 6.614, 6.614, 6.615, 6.615, 6.615, 6.615, 6.615, 6.616, 6.616, 6.616, 6.617, 6.618, 6.618, 6.618, 6.618, 6.62, 6.62, 6.622, 6.623, 6.625, 6.626, 6.629, 6.638, 6.703], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''To optimize the MLA (Multi-Head Latent Attention) implementation for TPU v6e-1, we will apply **Strategy 7: Maximize block sizes**.

### Optimization Plan
The current implementation uses conservative block sizes for both GEMM operations and the attention kernel. On TPU v6e, the primary performance bottleneck for the provided sequence length (2048) and embedding dimension (7168) is often the overhead of memory transfers from HBM to VMEM. By increasing the block sizes, we improve the **arithmetic intensity** (FLOPs per byte) of the kernels, allowing the hardware to spend more time in the compute-bound regime and less time waiting for DMA transfers.

1.  **Increase GEMM VMEM Budget:** The existing `_V6E_GEMM_WORKING_SET_BUDGET` is set to 8 MB. Since the TPU v6e has approximately 16 MB of VMEM, we can safely increase this budget to **14 MB**. This allows the tile-picking logic to select significantly larger tiles (e.g., $512 	imes 1536$ instead of $512 	imes 512$), which reduces the number of HBM passes for the large projection weights.
2.  **Increase Attention Block Sizes:** The current attention query and key blocks are 128. For a total sequence length of 2048, this results in many outer loop iterations. We will increase `_ATTN_QUERY_BLOCK` and `_ATTN_KEY_BLOCK` to **256**.
    *   Increasing the query block reduces the redundant loading of the KV cache across the grid.
    *   Increasing the key block improves the utilization of the VPU/MXU by processing larger segments of the sequence in the inner loop.
3.  **Tile Constraints:** We will ensure all new blocks satisfy TPU hardware constraints (last dimension divisible by 128 or equal to the full dimension; second-to-last divisible by 8). Specifically, for attention, a block of 256 for the query dimension is divisible by 8, and the head dimension (128) is a native hardware lane width.

### Optimized Code

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# Increase VMEM working-set target to 14MB to allow larger, more efficient GEMM tiles.
_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024

# Increased Attention tiling for better arithmetic intensity.
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256  # Increased from 128
_ATTN_KEY_BLOCK = 256    # Increased from 128


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred
    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate
    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(bm: int, bn: int, bk: int, in_dtype, out_dtype) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)
    return 2 * (bm * bk * in_b + bk * bn * in_b + bm * bn * out_b) + (bm * bn * 4)


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8) if multiple == 8 else (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if 0 < seed <= dim:
            rounded = seed - (seed % multiple)
            while rounded >= multiple:
                if dim % rounded == 0:
                    candidates.add(rounded)
                    break
                rounded -= multiple
    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype, dtype)
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET: continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    return best if best else (_choose_tpu_block(M, bm, 8), _choose_tpu_block(N, bn, 128), _choose_tpu_block(K, bk, 128))


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    acc_ref[...] += jax.lax.dot_general(a_ref[...], b_ref[...], dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    @pl.when(k_id == pl.num_programs(2) - 1)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=1024, bk=512):
    M, K = a.shape
    K2, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2))],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(a, b)


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)
    a_tile = a_ref[...]
    acc0_ref[...] += jax.lax.dot_general(a_tile, b0_ref[...], dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    acc1_ref[...] += jax.lax.dot_general(a_tile, b1_ref[...], dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    @pl.when(k_id == pl.num_programs(1) - 1)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    bm_a, bk_a = _pick_gemm_tiles(M, b0.shape[1], K, a.dtype, bm, b0.shape[1], bk)[:3:2]
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, b0.shape[1]), a.dtype), jax.ShapeDtypeStruct((M, b1.shape[1]), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, b0.shape[1]), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, b1.shape[1]), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2))],
        out_specs=[pl.BlockSpec((bm_a, b0.shape[1]), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
                   pl.BlockSpec((bm_a, b1.shape[1]), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2))],
        scratch_shapes=[pltpu.VMEM((bm_a, b0.shape[1]), jnp.float32), pltpu.VMEM((bm_a, b1.shape[1]), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )(a, b0, b1)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=512, bk=512):
    M, K = a.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, b0.shape[1], K, a.dtype, bm, b0.shape[1], bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, bn_a), a.dtype), jax.ShapeDtypeStruct((M, bn_a), a.dtype)],
        grid=(M // bm_a, bn_a // bn_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2))],
        out_specs=[pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
                   pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2))],
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32), pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(a, b0, b1)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    m_prev, l_prev, acc_prev = m_ref[...], l_ref[...], acc_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(scores_f32, axis=-1))
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])
    acc_ref[...] = acc_prev * alpha[..., None] + jax.lax.dot_general(p, v_f32, dimension_numbers=(((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_idx = pl.program_id(2)
    q_start, q_f32 = q_idx * _ATTN_QUERY_BLOCK, q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** -0.5
    acc_ref[...], m_ref[...], l_ref[...] = 0.0, -jnp.inf, 0.0
    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(k_ref.shape[2] // _ATTN_KEY_BLOCK):
        k_start = kb * _ATTN_KEY_BLOCK
        k_f32, v_f32 = k_ref[0, :, k_start:k_start+_ATTN_KEY_BLOCK, :].astype(jnp.float32), v_ref[0, :, k_start:k_start+_ATTN_KEY_BLOCK, :].astype(jnp.float32)
        scores = jax.lax.dot_general(q_f32, k_f32, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32) * scale
        @pl.when(kb == q_idx)
        def _mask():
            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            scores[...] = jnp.where((q_pos[:, None] >= k_pos[None, :])[None, :, :], scores[...], -1e30)
        @pl.when(kb <= q_idx)
        def _update():
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)
    out_ref[0, :, :, :] = (acc_ref[...] / l_ref[...][..., None]).astype(out_ref.dtype)


def _causal_attention_pallas(q, k, v):
    B, H, S, Dv = q.shape[0], q.shape[1], q.shape[2], v.shape[-1]
    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, H // _ATTN_HEAD_BLOCK, S // _ATTN_QUERY_BLOCK),
        in_specs=[pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, q.shape[-1]), lambda b, h, qb: (b, h, qb, 0)),
                  pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, k.shape[-1]), lambda b, h, qb: (b, h, 0, 0)),
                  pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), lambda b, h, qb: (b, h, 0, 0))],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), lambda b, h, qb: (b, h, qb, 0)),
        scratch_shapes=[pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
                        pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
                        pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(q, k, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    B, S, E, C = *x.shape, CONFIG
    H, nope, rope, vd, ql, kvl = C["num_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"], C["q_lora_rank"], C["kv_lora_rank"]
    x2d = x.reshape(B * S, E)
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)
    q = _pallas_gemm(q_low2d, q_up_proj).reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent.reshape(B * S, kvl), k_up_proj, v_up_proj)
    k_nope, v = k_nope2d.reshape(B, S, H, nope), v2d.reshape(B, S, H, vd)
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    q_rope, k_rope = _apply_rope(q_rope, cos, sin), _apply_rope(jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope)), cos, sin)
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    out = _causal_attention_pallas(q_full, k_full, v.transpose(0, 2, 1, 3))
    out2d = out.transpose(0, 2, 1, 3).reshape(B * S, H * vd).astype(x.dtype)
    return _pallas_gemm(out2d, o_proj).reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

# v6e-1 has ~16 MiB VMEM per TensorCore. Use a larger but still conservative
# working-set budget to enable larger GEMM tiles.
_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024

# Larger attention tiles to increase arithmetic intensity on v6e-1.
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    y = jnp.stack([y1, y2], axis=-1).reshape(s)
    return y.astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _choose_tpu_block(dim: int, preferred: int, multiple: int) -> int:
    preferred = min(dim, preferred)
    if preferred > 0 and dim % preferred == 0 and preferred % multiple == 0:
        return preferred

    start = preferred - (preferred % multiple)
    for candidate in range(start, multiple - 1, -multiple):
        if candidate > 0 and dim % candidate == 0:
            return candidate

    return dim


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_gemm_working_set_bytes(
    bm: int, bn: int, bk: int, in_dtype, out_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(out_dtype)

    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * out_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)

    if multiple == 8:
        common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)

    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple

    return sorted(candidates, reverse=True)


def _pick_gemm_tiles(M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_gemm_working_set_bytes(
                    bm_c, bn_c, bk_c, dtype, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile,
        b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store_out():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} vs {K2}")

    bm_actual, bn_actual, bk_actual = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec(
                (bm_actual, bk_actual),
                lambda i, j, k: (i, k),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bk_actual, bn_actual),
                lambda i, j, k: (k, j),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        ],
        out_specs=pl.BlockSpec(
            (bm_actual, bn_actual),
            lambda i, j, k: (i, j),
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
        scratch_shapes=[pltpu.VMEM((bm_actual, bn_actual), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _estimate_dual_gemm_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_tiles_same_n(
    M: int, N: int, K: int, dtype, bm: int, bn: int, bk: int
):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                working_set = _estimate_dual_gemm_working_set_bytes(
                    bm_c, bn_c, bn_c, bk_c, dtype
                )
                if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (bm_c, bn_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(N, bn, 128),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_same_n_kernel(
    a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref
):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")
    if N0 != N1:
        raise ValueError(f"Output widths must match: N0={N0}, N1={N1}")

    N = N0
    bm_actual, bn_actual, bk_actual = _pick_dual_gemm_tiles_same_n(
        M, N, K, a.dtype, bm, bn, bk
    )

    if M % bm_actual != 0 or K % bk_actual != 0 or N % bn_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}; N={N}, bn={bn_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N), a.dtype),
            jax.ShapeDtypeStruct((M, N), a.dtype),
        ],
        grid=(M // bm_actual, N // bn_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec(
                (bm_actual, bk_actual),
                lambda i, j, k: (i, k),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bk_actual, bn_actual),
                lambda i, j, k: (k, j),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bk_actual, bn_actual),
                lambda i, j, k: (k, j),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (bm_actual, bn_actual),
                lambda i, j, k: (i, j),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bm_actual, bn_actual),
                lambda i, j, k: (i, j),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
            pltpu.VMEM((bm_actual, bn_actual), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _estimate_dual_gemm_down_working_set_bytes(
    bm: int, bn0: int, bn1: int, bk: int, in_dtype
) -> int:
    in_b = _dtype_nbytes(in_dtype)
    out_b = _dtype_nbytes(in_dtype)

    a_bytes = 2 * bm * bk * in_b
    b0_bytes = 2 * bk * bn0 * in_b
    b1_bytes = 2 * bk * bn1 * in_b
    c0_bytes = 2 * bm * bn0 * out_b
    c1_bytes = 2 * bm * bn1 * out_b
    acc0_bytes = bm * bn0 * 4
    acc1_bytes = bm * bn1 * 4
    return a_bytes + b0_bytes + b1_bytes + c0_bytes + c1_bytes + acc0_bytes + acc1_bytes


def _pick_dual_gemm_down_tiles(
    M: int, N0: int, N1: int, K: int, dtype, bm: int, bk: int
):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bk_candidates = _candidate_blocks(K, bk, 128)

    best = None
    best_score = None
    for bm_c in bm_candidates:
        for bk_c in bk_candidates:
            working_set = _estimate_dual_gemm_down_working_set_bytes(
                bm_c, N0, N1, bk_c, dtype
            )
            if working_set > _V6E_GEMM_WORKING_SET_BUDGET:
                continue
            score = (bm_c * bk_c, bm_c, bk_c)
            if best_score is None or score > best_score:
                best_score = score
                best = (bm_c, bk_c)

    if best is not None:
        return best

    return (
        _choose_tpu_block(M, bm, 8),
        _choose_tpu_block(K, bk, 128),
    )


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init_acc():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, dtype=jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, dtype=jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile,
        b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile,
        b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store_out():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    K0, N0 = b0.shape
    K1, N1 = b1.shape

    if K != K0 or K != K1:
        raise ValueError(f"Inner dimensions must match: K={K}, K0={K0}, K1={K1}")

    bm_actual, bk_actual = _pick_dual_gemm_down_tiles(M, N0, N1, K, a.dtype, bm, bk)

    if M % bm_actual != 0 or K % bk_actual != 0:
        raise ValueError(
            "Non-divisible tile selection: "
            f"M={M}, bm={bm_actual}; K={K}, bk={bk_actual}"
        )

    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((M, N0), a.dtype),
            jax.ShapeDtypeStruct((M, N1), a.dtype),
        ],
        grid=(M // bm_actual, K // bk_actual),
        in_specs=[
            pl.BlockSpec(
                (bm_actual, bk_actual),
                lambda i, k: (i, k),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bk_actual, N0),
                lambda i, k: (k, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bk_actual, N1),
                lambda i, k: (k, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (bm_actual, N0),
                lambda i, k: (i, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (bm_actual, N1),
                lambda i, k: (i, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm_actual, N0), jnp.float32),
            pltpu.VMEM((bm_actual, N1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b0, b1)


def _causal_attention_jax(q, k, v):
    B, H, S, hd = q.shape

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    scale = jnp.float32(hd) ** jnp.float32(-0.5)
    scores = jnp.einsum("bhid,bhjd->bhij", q_f32, k_f32) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - scores_max
    exp_scores = jnp.exp(scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v_f32)
    return out.astype(v.dtype)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p,
        v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    num_k_blocks = k_ref.shape[2] // _ATTN_KEY_BLOCK

    q_f32 = q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** jnp.float32(-0.5)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK
        k_end = k_start + _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)

            scores = jax.lax.dot_general(
                q_f32,
                k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale

            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            scores = jnp.where(
                causal_mask,
                scores,
                jnp.full(scores.shape, jnp.float32(-1.0e30)),
            )

            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _attention_input_spec(block_shape, index_map):
    return pl.BlockSpec(
        block_shape,
        index_map,
        pipeline_mode=pl.Buffered(buffer_count=1),
    )


def _causal_attention_pallas(q, k, v):
    B, H, S, D = q.shape
    _, Hk, Sk, Dk = k.shape
    _, Hv, Sv, Dv = v.shape

    if H != Hk or H != Hv or S != Sk or S != Sv:
        raise ValueError("q, k, v must agree on batch/head/sequence dimensions")

    if (
        H % _ATTN_HEAD_BLOCK != 0
        or S % _ATTN_QUERY_BLOCK != 0
        or S % _ATTN_KEY_BLOCK != 0
    ):
        return _causal_attention_jax(q, k, v)

    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, D),
                lambda b, h, q_blk: (b, h, q_blk, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dk),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
            _attention_input_spec(
                (1, _ATTN_HEAD_BLOCK, S, Dv),
                lambda b, h, q_blk: (b, h, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv),
            lambda b, h, q_blk: (b, h, q_blk, 0),
            pipeline_mode=pl.Buffered(buffer_count=1),
        ),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(q, k, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )
    kvl = C["kv_lora_rank"]

    proj_bm, proj_bn, proj_bk = 512, 2048, 512

    x2d = x.reshape(B * S, E)

    q_low2d, kv2d = _pallas_dual_gemm_down(
        x2d, q_down_proj, kv_down_proj, bm=512, bk=512
    )

    q2d = _pallas_gemm(q_low2d, q_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(
        k_latent2d, k_up_proj, v_up_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk
    )

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    out = _causal_attention_pallas(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj, bm=proj_bm, bn=proj_bn, bk=proj_bk)
    return final2d.reshape(B, S, E)
''',
score=4.764,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='gpt-5.4',
stdout='Latency: 4.764 ms\n{"correct": true, "latency": 4.764, "error": "", "all_times_ms": [4.749, 4.75, 4.75, 4.75, 4.751, 4.753, 4.753, 4.753, 4.753, 4.753, 4.754, 4.755, 4.757, 4.757, 4.757, 4.757, 4.757, 4.758, 4.759, 4.759, 4.759, 4.759, 4.759, 4.76, 4.76, 4.76, 4.76, 4.76, 4.761, 4.761, 4.761, 4.761, 4.761, 4.761, 4.762, 4.762, 4.762, 4.762, 4.762, 4.762, 4.762, 4.762, 4.762, 4.762, 4.763, 4.763, 4.763, 4.763, 4.763, 4.763, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.764, 4.765, 4.765, 4.765, 4.765, 4.766, 4.766, 4.766, 4.766, 4.766, 4.767, 4.767, 4.767, 4.768, 4.768, 4.768, 4.768, 4.768, 4.769, 4.769, 4.77, 4.771, 4.771, 4.771, 4.773, 4.774, 4.774, 4.774, 4.774, 4.774, 4.775, 4.776, 4.777, 4.777, 4.779, 4.779, 4.779, 4.791, 4.799], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''To optimize the MLA (Multi-Head Latent Attention) implementation for TPU v6e-1, we will apply **Strategy 5: Use `CompilerParams(allow_input_fusion=...)`**.

### Analysis of Inefficiencies
The current `workload` function materializes several large intermediate tensors in HBM between Pallas kernel calls:
1.  **GEMM Activations**: `q_low2d`, `q2d`, `k_latent2d`, `k_nope2d`, and `v2d` are stored in HBM.
2.  **Attention Inputs**: `q_full`, `k_full`, and `v_full` are created via `jnp.concatenate` and `transpose` operations. These tensors are large (e.g., $1 	imes 128 	imes 2048 	imes 192 	imes 2$ bytes $pprox 96$ MiB each), and writing/reading them creates significant memory bandwidth pressure.
3.  **RoPE**: `_apply_rope` performs elementwise operations in JAX, leading to additional HBM roundtrips.

### Optimization Plan
By setting `allow_input_fusion` to `True` for the activation inputs of our Pallas kernels, we instruct the Mosaic TPU compiler to fuse "cheap" producers (like `reshape`, `transpose`, `slice`, `concatenate`, `cast`, and basic arithmetic used in RoPE) directly into the kernel\'s DMA load operations. This allows the hardware to transform the data as it moves from HBM to VMEM, preventing the materialization of intermediate tensors.

Specifically, we will:
1.  Update `_pallas_gemm`, `_pallas_dual_gemm_same_n`, and `_pallas_dual_gemm_down` to enable fusion for their activation inputs.
2.  Update `_causal_attention_pallas` to enable fusion for `q`, `k`, and `v`, allowing the RoPE and concatenation logic to be fused into the attention kernel.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)

def _candidate_blocks(dim, preferred, multiple):
    candidates = {dim}
    capped = min(dim, preferred)
    common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8) if multiple == 8 else (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if seed <= 0 or seed > dim: continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple
    return sorted(candidates, reverse=True)

def _pick_gemm_tiles(M, N, K, dtype, bm, bn, bk):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                ws = (2 * bm_c * bk_c + 2 * bk_c * bn_c + 2 * bm_c * bn_c) * jnp.dtype(dtype).itemsize + bm_c * bn_c * 4
                if ws > _V6E_GEMM_WORKING_SET_BUDGET: continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    return best or (min(M, bm), min(N, bn), min(K, bk))

def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id, last_k = pl.program_id(2), pl.num_programs(2) - 1
    @pl.when(k_id == 0)
    def _(): acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    acc_ref[...] += jax.lax.dot_general(a_ref[...], b_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    @pl.when(k_id == last_k)
    def _(): c_ref[...] = acc_ref[...].astype(c_ref.dtype)

def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2))],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"),
                                             allow_input_fusion=(True, False)),
    )(a, b)

def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id, last_k = pl.program_id(2), pl.num_programs(2) - 1
    @pl.when(k_id == 0)
    def _():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)
    a_tile = a_ref[...]
    acc0_ref[...] += jax.lax.dot_general(a_tile, b0_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    acc1_ref[...] += jax.lax.dot_general(a_tile, b1_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    @pl.when(k_id == last_k)
    def _():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)

def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K, N = a.shape[0], a.shape[1], b0.shape[1]
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype)] * 2,
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2))],
        out_specs=[pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2))] * 2,
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)] * 2,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"),
                                             allow_input_fusion=(True, False, False)),
    )(a, b0, b1)

def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id, last_k = pl.program_id(1), pl.num_programs(1) - 1
    @pl.when(k_id == 0)
    def _():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)
    a_tile = a_ref[...]
    acc0_ref[...] += jax.lax.dot_general(a_tile, b0_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    acc1_ref[...] += jax.lax.dot_general(a_tile, b1_ref[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    @pl.when(k_id == last_k)
    def _():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)

def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K, N0, N1 = a.shape[0], a.shape[1], b0.shape[1], b1.shape[1]
    bm_a, bk_a = min(M, bm), min(K, bk)
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N0), a.dtype), jax.ShapeDtypeStruct((M, N1), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
                  pl.BlockSpec((bk_a, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2))],
        out_specs=[pl.BlockSpec((bm_a, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
                   pl.BlockSpec((bm_a, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2))],
        scratch_shapes=[pltpu.VMEM((bm_a, N0), jnp.float32), pltpu.VMEM((bm_a, N1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"),
                                             allow_input_fusion=(True, False, False)),
    )(a, b0, b1)

def _attention_update_state(scores, v, acc_ref, m_ref, l_ref):
    row_max = jnp.max(scores, axis=-1)
    m_prev, l_prev, acc_prev = m_ref[...], l_ref[...], acc_ref[...]
    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores - m_new[..., None])
    pv = jax.lax.dot_general(p, v, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new

def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_blk = pl.program_id(2)
    q_start = q_blk * _ATTN_QUERY_BLOCK
    num_k_blocks = k_ref.shape[2] // _ATTN_KEY_BLOCK
    q_f32 = q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** -0.5
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)
    for kb in range(num_k_blocks):
        k_s, k_e = kb * _ATTN_KEY_BLOCK, (kb + 1) * _ATTN_KEY_BLOCK
        @pl.when(kb <= q_blk)
        def _():
            k_f32, v_f32 = k_ref[0, :, k_s:k_e, :].astype(jnp.float32), v_ref[0, :, k_s:k_e, :].astype(jnp.float32)
            scores = jax.lax.dot_general(q_f32, k_f32, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32) * scale
            if kb == q_blk:
                k_pos = k_s + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
                scores = jnp.where(q_pos[None, :, None] >= k_pos[None, None, :], scores, -1e30)
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)
    denom = l_ref[...][..., None]
    out_ref[0, :, :, :] = jnp.where(denom > 0, acc_ref[...] / denom, 0.0).astype(out_ref.dtype)

def _causal_attention_pallas(q, k, v):
    B, H, S, D = q.shape
    Dv = v.shape[-1]
    num_h_blks, num_q_blks = H // _ATTN_HEAD_BLOCK, S // _ATTN_QUERY_BLOCK
    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blks, num_q_blks),
        in_specs=[pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, D), lambda b, h, q: (b, h, q, 0), pipeline_mode=pl.Buffered(1)),
                  pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, D), lambda b, h, q: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
                  pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), lambda b, h, q: (b, h, 0, 0), pipeline_mode=pl.Buffered(1))],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), lambda b, h, q: (b, h, q, 0), pipeline_mode=pl.Buffered(1)),
        scratch_shapes=[pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
                        pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
                        pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"),
                                             allow_input_fusion=(True, True, True)),
    )(q, k, v)

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H, nope, rope, vd, kvl = C["num_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"], C["kv_lora_rank"]
    x2d = x.reshape(B * S, E)
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)
    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent.reshape(B * S, kvl), k_up_proj, v_up_proj)
    k_nope, v = k_nope2d.reshape(B, S, H, nope), v2d.reshape(B, S, H, vd)
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope, k_rope = _apply_rope(q_rope, cos, sin), _apply_rope(k_rope, cos, sin)
    q_f, k_f = jnp.concatenate([q_nope, q_rope], -1).transpose(0, 2, 1, 3), jnp.concatenate([k_nope, k_rope], -1).transpose(0, 2, 1, 3)
    out = _causal_attention_pallas(q_f, k_f, v.transpose(0, 2, 1, 3))
    out2d = out.transpose(0, 2, 1, 3).reshape(B * S, H * vd).astype(x.dtype)
    return _pallas_gemm(out2d, o_proj).reshape(B, S, E)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    if multiple == 8:
        common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple
    return sorted(candidates, reverse=True)


def _estimate_gemm_working_set_bytes(bm, bn, bk, in_dtype):
    in_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * in_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _pick_gemm_tiles(M, N, K, dtype, bm, bn, bk):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                ws = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype)
                if ws > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (min(M, bm), min(N, bn), min(K, bk))


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile, b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False),
        ),
    )(a, b)


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b0.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype), jax.ShapeDtypeStruct((M, N), a.dtype)],
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32), pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    _, N0 = b0.shape
    _, N1 = b1.shape
    bm_a = min(M, bm)
    bk_a = min(K, bk)
    # Ensure divisibility
    while M % bm_a != 0 and bm_a > 8:
        bm_a -= 8
    while K % bk_a != 0 and bk_a > 128:
        bk_a -= 128
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N0), a.dtype), jax.ShapeDtypeStruct((M, N1), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, N0), jnp.float32), pltpu.VMEM((bm_a, N1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_block_kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    num_k_blocks = k_ref.shape[2] // _ATTN_KEY_BLOCK

    q_f32 = q_ref[0, :, :, :].astype(jnp.float32)
    scale = jnp.float32(q_ref.shape[-1]) ** jnp.float32(-0.5)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK
        k_end = k_start + _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            scores = jax.lax.dot_general(
                q_f32, k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_f32 = k_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_end, :].astype(jnp.float32)
            scores = jax.lax.dot_general(
                q_f32, k_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores = scores * scale
            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            scores = jnp.where(causal_mask, scores, jnp.float32(-1.0e30))
            _attention_update_state(scores, v_f32, acc_ref, m_ref, l_ref)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _causal_attention_pallas(q, k, v):
    B, H, S, D = q.shape
    Dv = v.shape[-1]
    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_block_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, D), lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, D), lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
        ],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, True, True),
        ),
    )(q, k, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    x2d = x.reshape(B * S, E)

    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)

    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent2d, k_up_proj, v_up_proj)

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v_full = v.transpose(0, 2, 1, 3)

    out = _causal_attention_pallas(q_full, k_full, v_full)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj)
    return final2d.reshape(B, S, E)
''',
score=4.419,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 4.419 ms\n{"correct": true, "latency": 4.419, "error": "", "all_times_ms": [4.403, 4.407, 4.407, 4.407, 4.408, 4.408, 4.409, 4.409, 4.409, 4.409, 4.41, 4.41, 4.41, 4.41, 4.411, 4.411, 4.411, 4.412, 4.412, 4.412, 4.413, 4.413, 4.413, 4.413, 4.413, 4.414, 4.414, 4.414, 4.414, 4.415, 4.415, 4.415, 4.415, 4.415, 4.415, 4.416, 4.416, 4.416, 4.416, 4.417, 4.417, 4.417, 4.417, 4.417, 4.417, 4.418, 4.418, 4.418, 4.419, 4.419, 4.419, 4.42, 4.42, 4.42, 4.42, 4.421, 4.421, 4.421, 4.421, 4.421, 4.422, 4.422, 4.422, 4.422, 4.422, 4.422, 4.422, 4.422, 4.422, 4.423, 4.423, 4.423, 4.423, 4.423, 4.423, 4.424, 4.424, 4.425, 4.425, 4.425, 4.425, 4.425, 4.425, 4.425, 4.425, 4.426, 4.426, 4.427, 4.427, 4.427, 4.429, 4.429, 4.429, 4.43, 4.43, 4.432, 4.434, 4.437, 4.438, 4.439], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''Based on the analysis of the provided code and the optimization strategies, I have identified a significant inefficiency in the implementation of the Multi-Latent Attention (MLA) forward pass.

The current code uses separate, blocked Pallas kernels (`_pallas_gemm`, `_pallas_dual_gemm_down`, etc.) for each matrix multiplication. This creates a "kernel boundary" between each layer of the MLP-like projection stack (Down -> Up). At each boundary:
1. Intermediate results (e.g., `q_low2d`, `kv2d`) are written from VMEM (on-chip) all the way back to HBM (off-chip).
2. The subsequent kernel must then wait for these results to be staged back from HBM into VMEM, incurring high latency and consuming memory bandwidth.

This violates the principles of data locality and overlapping compute with memory access for intermediate activations.

### Optimization Plan

**Selected Strategy:**  
**6. Keep `k_rope` head-shared all the way through the kernel instead of broadcasting `k_rope_raw` to `(B,S,H,rope)`; rotate the shared rope key once and broadcast logically during score computation.**

**Reasoning (as applied to this specific code):**  
The provided MLA implementation has a sequence of matrix multiplications: `Down-Proj -> Up-Proj -> Attention`. Currently, these are separate `pallas_call` kernels. As noted above, this causes intermediate results (`q_low2d`, `k_latent`, etc.) to round-trip to HBM.

To fix this, we should fuse these operations into a single kernel. The standard limitation of kernel fusion is the accumulation of VMEM pressure (running out of on-chip memory). However, **Strategy 6** highlights a specific structural optimization for MLA: the `k_rope` component of the Key matrix is effectively **shared across all heads**.

- In the current implementation, `k_rope_raw` (shape `(B, S, rope)`) is broadcast to `(B, S, H, rope)` before being passed to the attention kernel.
- This broadcast explicitly materializes `H` copies of the tensor, consuming `H * S * rope * sizeof(dtype)` bytes of VMEM in the attention kernel\'s input buffer.
- By modifying the fused kernel to keep `k_rope` **logical** (i.e., not materializing the broadcast), we can reduce the intermediate buffer footprint.

This memory saving is the enabling factor that allows us to fuse the `x -> DownProj` and `DownProj -> UpProj` steps into a single kernel pipeline without exhausting the 16 MB VMEM budget. By keeping `k_rope` head-shared, we save enough VMEM to stage the input/output blocks required for the fused projections, allowing the data to flow directly from one matmul unit to the next without returning to HBM.

### Changes to Make

1.  **Restructure the computation flow**: Merge the initial Down-Projection (`x -> q_low`, `x -> kv_latent`) and the subsequent Up-Projections (`q_low -> Q`, `kv_latent -> K_nope/V`) into a single "MLA Projector" kernel.
2.  **Modify the Attention Kernel**: Update `_causal_attention_block_kernel` to accept `k_rope` as a separate, non-broadcasted tensor `(S, rope)`. Inside the kernel, compute the attention scores by adding the `q_rope @ k_rope` term. This operation implicitly broadcasts the product, avoiding materializing the large input buffer.
3.  **Apply Pallas Pipelining**: Use `pl.pallas_call` with a `grid` covering the sequence of operations, utilizing `scratch_shapes` to hold intermediate activations (`q_low`, `kv_latent`) in VMEM between the fused matmul stages, rather than outputting them to HBM.

This plan specifically leverages Strategy 6 to unlock the performance gains of Strategy 2 (caching reused data in local memory) for the broader fused pipeline.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def _apply_rope_2d(x, cos, sin):
    """Apply RoPE to a 2D tensor (S, rope_dim)."""
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(s[0], s[1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)
    sin_f32 = sin.astype(jnp.float32)
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    if multiple == 8:
        common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple
    return sorted(candidates, reverse=True)


def _estimate_gemm_working_set_bytes(bm, bn, bk, in_dtype):
    in_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * in_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _pick_gemm_tiles(M, N, K, dtype, bm, bn, bk):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                ws = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype)
                if ws > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (min(M, bm), min(N, bn), min(K, bk))


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile, b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False),
        ),
    )(a, b)


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b0.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype), jax.ShapeDtypeStruct((M, N), a.dtype)],
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32), pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    _, N0 = b0.shape
    _, N1 = b1.shape
    bm_a = min(M, bm)
    bk_a = min(K, bk)
    while M % bm_a != 0 and bm_a > 8:
        bm_a -= 8
    while K % bk_a != 0 and bk_a > 128:
        bk_a -= 128
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N0), a.dtype), jax.ShapeDtypeStruct((M, N1), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, N0), jnp.float32), pltpu.VMEM((bm_a, N1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    """Update attention state given combined scores."""
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask):
    """Update attention state with causal mask."""
    scores_f32 = jnp.where(causal_mask, scores_f32, jnp.float32(-1.0e30))
    
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_kernel(q_nope_ref, q_rope_ref, k_nope_ref, k_rope_ref, v_ref, out_ref, acc_ref, m_ref, l_ref):
    """Attention kernel where k_rope is broadcast across heads."""
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    S = k_nope_ref.shape[2]
    num_k_blocks = S // _ATTN_KEY_BLOCK
    nope_dim = q_nope_ref.shape[-1]
    rope_dim = q_rope_ref.shape[-1]
    
    scale = jnp.float32(nope_dim + rope_dim) ** jnp.float32(-0.5)

    # Load Q blocks
    q_nope_f32 = q_nope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, nope)
    q_rope_f32 = q_rope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, rope)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            # Compute nope scores: (H_block, Q_block, nope) @ (H_block, K_block, nope).T
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            # Compute rope scores: (H_block, Q_block, rope) @ (H_block, K_block, rope).T
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores_f32 = (scores_nope + scores_rope) * scale
            _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores_f32 = (scores_nope + scores_rope) * scale
            
            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _causal_attention_pallas(q_nope, q_rope, k_nope, k_rope, v):
    """
    Attention with k_rope broadcast across heads.
    
    Args:
        q_nope: (B, H, S, nope_dim)
        q_rope: (B, H, S, rope_dim)
        k_nope: (B, H, S, nope_dim)
        k_rope: (B, H, S, rope_dim) - broadcast across heads
        v: (B, H, S, v_dim)
    
    Returns:
        output: (B, H, S, v_dim)
    """
    B, H, S, nope_dim = q_nope.shape
    rope_dim = q_rope.shape[-1]
    Dv = v.shape[-1]
    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, nope_dim), lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, rope_dim), lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, nope_dim), lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, rope_dim), lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
        ],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, True, True, True, True),
        ),
    )(q_nope, q_rope, k_nope, k_rope, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    x2d = x.reshape(B * S, E)

    # Down projections (fused)
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)

    # Q up projection
    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV processing
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]  # (B, S, rope)

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent2d, k_up_proj, v_up_proj)

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # Apply RoPE
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    
    # Apply RoPE to Q (per-head)
    q_rope = _apply_rope(q_rope, cos, sin)
    
    # Apply RoPE to K_rope and broadcast to all heads
    # k_rope_raw is (B, S, rope)
    k_rope_with_rope = _apply_rope_2d(k_rope_raw[0], cos, sin)  # (S, rope)
    # Broadcast to (B, H, S, rope) to match the expected shape
    k_rope_broadcast = jnp.broadcast_to(k_rope_with_rope[None, None, :, :], (B, H, S, rope))

    # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
    q_nope_t = q_nope.transpose(0, 2, 1, 3)
    q_rope_t = q_rope.transpose(0, 2, 1, 3)
    k_nope_t = k_nope.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Attention with broadcast k_rope
    out = _causal_attention_pallas(q_nope_t, q_rope_t, k_nope_t, k_rope_broadcast, v_t)

    # Output projection
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj)
    return final2d.reshape(B, S, E)
''',
score=4.038,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 4.038 ms\n{"correct": true, "latency": 4.038, "error": "", "all_times_ms": [4.024, 4.027, 4.028, 4.029, 4.029, 4.029, 4.029, 4.03, 4.03, 4.03, 4.03, 4.031, 4.031, 4.031, 4.031, 4.032, 4.032, 4.033, 4.033, 4.033, 4.034, 4.034, 4.034, 4.034, 4.035, 4.035, 4.035, 4.035, 4.035, 4.035, 4.035, 4.035, 4.036, 4.036, 4.036, 4.037, 4.037, 4.037, 4.037, 4.037, 4.037, 4.037, 4.037, 4.038, 4.038, 4.038, 4.038, 4.038, 4.038, 4.038, 4.038, 4.039, 4.039, 4.039, 4.039, 4.039, 4.039, 4.039, 4.039, 4.039, 4.04, 4.04, 4.04, 4.04, 4.04, 4.04, 4.04, 4.04, 4.041, 4.041, 4.041, 4.041, 4.041, 4.042, 4.042, 4.042, 4.042, 4.042, 4.042, 4.042, 4.043, 4.043, 4.043, 4.043, 4.044, 4.044, 4.045, 4.046, 4.046, 4.046, 4.047, 4.047, 4.048, 4.048, 4.048, 4.048, 4.05, 4.05, 4.061, 4.063], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr=''),
plan='''Looking at this code, I can identify a significant inefficiency in the attention implementation. Let me focus on applying **Strategy 8** to address it.

## Performance Inefficiency Analysis

The current implementation broadcasts `k_rope` to shape `(B, H, S, rope)`:
```python
k_rope_broadcast = jnp.broadcast_to(k_rope_with_rope[None, None, :, :], (B, H, S, rope))
```

This creates 128 identical copies (one per head) of the RoPE embeddings, wasting precious HBM bandwidth and VMEM space.

## Optimization Plan: Efficient `k_rope` Representation

The `k_rope` tensor has the same values across all heads (it\'s computed once per sequence position). Instead of materializing 128 copies, we should:

1. **Keep `k_rope` as a separate tensor** with shape `(B, S, rope)` 
2. **Pass it to the attention kernel separately** using a `BlockSpec` that doesn\'t tile over the head dimension
3. **Load it once and share across heads** in the kernel

This reduces memory traffic for `k_rope` from `B × H × S × rope × 2 bytes` to `B × S × rope × 2 bytes` (128× reduction).

### Concrete Changes:

1. **Remove the broadcast** in `workload()`:
   ```python
   k_rope_raw = kv[..., kvl:]
   k_rope_with_rope = _apply_rope_2d(k_rope_raw[0], cos, sin)  # (S, rope)
   ```

2. **Add `k_rope` as a separate input** to the attention kernel with shape `(B, S, rope)` using a `BlockSpec` that treats the head dimension as squeezed/constant.

3. **Update the kernel** to load `k_rope` independently:
   - The `BlockSpec` for `k_rope` would be something like:
     ```python
     pl.BlockSpec((1, _ATTN_QUERY_BLOCK, rope_dim), lambda b, h, q_blk: (b, q_blk, 0))
     ```
   - Inside the kernel, load `k_rope` once per query block and broadcast to heads when computing attention scores

4. **This naturally handles the broadcast semantics** during the attention computation:
   ```python
   # k_rope_f32 shape: (Q_block, rope) loaded once
   # Broadcast implicitly in matmul with q_rope_f32 (H_block, Q_block, rope)
   scores_rope = jax.lax.dot_general(
       q_rope_f32,  # (H_block, Q_block, rope)
       k_rope_f32,  # (K_block, rope)
       dimension_numbers=(((2,), (1,)), ((0,), ())),
       preferred_element_type=jnp.float32,
   )
   ```

This optimization also aligns with the soft constraint that singleton dimensions in block specs should be handled carefully - we\'re explicitly representing the shared dimension rather than materializing duplicates.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "deepseek_v3_mla",
    "model": "DeepSeek-V3-671B",
    "operator": "mla_attention",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 7168,
    "num_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000,
}

_V6E_GEMM_WORKING_SET_BUDGET = 14 * 1024 * 1024
_ATTN_HEAD_BLOCK = 8
_ATTN_QUERY_BLOCK = 256
_ATTN_KEY_BLOCK = 256


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(*s[:-1], s[-1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def _apply_rope_2d(x, cos, sin):
    """Apply RoPE to a 2D tensor (S, rope_dim)."""
    x_f32 = x.astype(jnp.float32)
    s = x_f32.shape
    x_pairs = x_f32.reshape(s[0], s[1] // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f32 = cos.astype(jnp.float32)
    sin_f32 = sin.astype(jnp.float32)
    y1 = x1 * cos_f32 - x2 * sin_f32
    y2 = x1 * sin_f32 + x2 * cos_f32
    return jnp.stack([y1, y2], axis=-1).reshape(s).astype(x.dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C["batch"], C["seq_len"], C["emb_dim"]
    H = C["num_heads"]
    ql, kvl = C["q_lora_rank"], C["kv_lora_rank"]
    nope, rope, vd = (
        C["qk_nope_head_dim"],
        C["qk_rope_head_dim"],
        C["v_head_dim"],
    )

    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def _dtype_nbytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _candidate_blocks(dim: int, preferred: int, multiple: int):
    candidates = {dim}
    capped = min(dim, preferred)
    if multiple == 8:
        common = (2048, 1024, 512, 256, 128, 64, 32, 16, 8)
    else:
        common = (2048, 1024, 512, 256, 128)
    for seed in (capped,) + common:
        if seed <= 0 or seed > dim:
            continue
        if seed == dim:
            candidates.add(seed)
            continue
        rounded = seed - (seed % multiple)
        while rounded >= multiple:
            if dim % rounded == 0:
                candidates.add(rounded)
                break
            rounded -= multiple
    return sorted(candidates, reverse=True)


def _estimate_gemm_working_set_bytes(bm, bn, bk, in_dtype):
    in_b = _dtype_nbytes(in_dtype)
    a_bytes = 2 * bm * bk * in_b
    b_bytes = 2 * bk * bn * in_b
    c_bytes = 2 * bm * bn * in_b
    acc_bytes = bm * bn * 4
    return a_bytes + b_bytes + c_bytes + acc_bytes


def _pick_gemm_tiles(M, N, K, dtype, bm, bn, bk):
    bm_candidates = _candidate_blocks(M, bm, 8)
    bn_candidates = _candidate_blocks(N, bn, 128)
    bk_candidates = _candidate_blocks(K, bk, 128)
    best, best_score = None, None
    for bm_c in bm_candidates:
        for bn_c in bn_candidates:
            for bk_c in bk_candidates:
                ws = _estimate_gemm_working_set_bytes(bm_c, bn_c, bk_c, dtype)
                if ws > _V6E_GEMM_WORKING_SET_BUDGET:
                    continue
                score = (bm_c * bn_c * bk_c, bm_c * bn_c, bn_c, bm_c, bk_c)
                if best_score is None or score > best_score:
                    best_score, best = score, (bm_c, bn_c, bk_c)
    if best is not None:
        return best
    return (min(M, bm), min(N, bn), min(K, bk))


def _gemm_kernel(a_ref, b_ref, c_ref, acc_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b_tile = b_ref[...]
    prod = jax.lax.dot_general(
        a_tile, b_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = acc_ref[...] + prod

    @pl.when(k_id == last_k)
    def _store():
        c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def _pallas_gemm(a, b, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False),
        ),
    )(a, b)


def _dual_gemm_same_n_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(2)
    last_k = pl.num_programs(2) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_same_n(a, b0, b1, bm=512, bn=2048, bk=512):
    M, K = a.shape
    _, N = b0.shape
    bm_a, bn_a, bk_a = _pick_gemm_tiles(M, N, K, a.dtype, bm, bn, bk)
    return pl.pallas_call(
        _dual_gemm_same_n_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N), a.dtype), jax.ShapeDtypeStruct((M, N), a.dtype)],
        grid=(M // bm_a, N // bn_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, j, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, bn_a), lambda i, j, k: (k, j), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, bn_a), lambda i, j, k: (i, j), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, bn_a), jnp.float32), pltpu.VMEM((bm_a, bn_a), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _dual_gemm_down_kernel(a_ref, b0_ref, b1_ref, c0_ref, c1_ref, acc0_ref, acc1_ref):
    k_id = pl.program_id(1)
    last_k = pl.num_programs(1) - 1

    @pl.when(k_id == 0)
    def _init():
        acc0_ref[...] = jnp.zeros(acc0_ref.shape, jnp.float32)
        acc1_ref[...] = jnp.zeros(acc1_ref.shape, jnp.float32)

    a_tile = a_ref[...]
    b0_tile = b0_ref[...]
    b1_tile = b1_ref[...]

    prod0 = jax.lax.dot_general(
        a_tile, b0_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    prod1 = jax.lax.dot_general(
        a_tile, b1_tile,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    acc0_ref[...] = acc0_ref[...] + prod0
    acc1_ref[...] = acc1_ref[...] + prod1

    @pl.when(k_id == last_k)
    def _store():
        c0_ref[...] = acc0_ref[...].astype(c0_ref.dtype)
        c1_ref[...] = acc1_ref[...].astype(c1_ref.dtype)


def _pallas_dual_gemm_down(a, b0, b1, bm=512, bk=512):
    M, K = a.shape
    _, N0 = b0.shape
    _, N1 = b1.shape
    bm_a = min(M, bm)
    bk_a = min(K, bk)
    while M % bm_a != 0 and bm_a > 8:
        bm_a -= 8
    while K % bk_a != 0 and bk_a > 128:
        bk_a -= 128
    return pl.pallas_call(
        _dual_gemm_down_kernel,
        out_shape=[jax.ShapeDtypeStruct((M, N0), a.dtype), jax.ShapeDtypeStruct((M, N1), a.dtype)],
        grid=(M // bm_a, K // bk_a),
        in_specs=[
            pl.BlockSpec((bm_a, bk_a), lambda i, k: (i, k), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N0), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bk_a, N1), lambda i, k: (k, 0), pipeline_mode=pl.Buffered(2)),
        ],
        out_specs=[
            pl.BlockSpec((bm_a, N0), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
            pl.BlockSpec((bm_a, N1), lambda i, k: (i, 0), pipeline_mode=pl.Buffered(2)),
        ],
        scratch_shapes=[pltpu.VMEM((bm_a, N0), jnp.float32), pltpu.VMEM((bm_a, N1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            allow_input_fusion=(True, False, False),
        ),
    )(a, b0, b1)


def _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref):
    """Update attention state given combined scores."""
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask):
    """Update attention state with causal mask."""
    scores_f32 = jnp.where(causal_mask, scores_f32, jnp.float32(-1.0e30))
    
    row_max = jnp.max(scores_f32, axis=-1)
    m_prev = m_ref[...]
    l_prev = l_ref[...]
    acc_prev = acc_ref[...]

    m_new = jnp.maximum(m_prev, row_max)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(scores_f32 - m_new[..., None])

    pv = jax.lax.dot_general(
        p, v_f32,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    acc_ref[...] = acc_prev * alpha[..., None] + pv
    l_ref[...] = alpha * l_prev + jnp.sum(p, axis=-1)
    m_ref[...] = m_new


def _causal_attention_kernel(
    q_nope_ref, q_rope_ref, k_nope_ref, k_rope_ref, v_ref, out_ref, acc_ref, m_ref, l_ref
):
    """Attention kernel where k_rope is already broadcast to have head dimension."""
    q_block = pl.program_id(2)
    q_start = q_block * _ATTN_QUERY_BLOCK
    S = k_nope_ref.shape[2]
    num_k_blocks = S // _ATTN_KEY_BLOCK
    nope_dim = q_nope_ref.shape[-1]
    rope_dim = q_rope_ref.shape[-1]
    
    scale = jnp.float32(nope_dim + rope_dim) ** jnp.float32(-0.5)

    # Load Q blocks: (H_block, Q_block, dim)
    q_nope_f32 = q_nope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, nope)
    q_rope_f32 = q_rope_ref[0, :, :, :].astype(jnp.float32)  # (H_block, Q_block, rope)

    acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)

    q_pos = q_start + jnp.arange(_ATTN_QUERY_BLOCK, dtype=jnp.int32)

    for kb in range(num_k_blocks):
        k_start = kb * _ATTN_KEY_BLOCK

        @pl.when(kb < q_block)
        def _full_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            # Compute nope scores: (H_block, Q_block, nope) @ (H_block, K_block, nope).T
            # -> (H_block, Q_block, K_block)
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            # Compute rope scores: (H_block, Q_block, rope) @ (H_block, K_block, rope).T
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_f32 = (scores_nope + scores_rope) * scale
            _attention_update_state(scores_f32, v_f32, acc_ref, m_ref, l_ref)

        @pl.when(kb == q_block)
        def _diag_k_block():
            k_nope_f32 = k_nope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            k_rope_f32 = k_rope_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            v_f32 = v_ref[0, :, k_start:k_start + _ATTN_KEY_BLOCK, :].astype(jnp.float32)
            
            scores_nope = jax.lax.dot_general(
                q_nope_f32, k_nope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_rope = jax.lax.dot_general(
                q_rope_f32, k_rope_f32,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            
            scores_f32 = (scores_nope + scores_rope) * scale
            
            k_pos = k_start + jnp.arange(_ATTN_KEY_BLOCK, dtype=jnp.int32)
            causal_mask = (q_pos[:, None] >= k_pos[None, :])[None, :, :]
            _attention_update_state_masked(scores_f32, v_f32, acc_ref, m_ref, l_ref, causal_mask)

    denom = l_ref[...][..., None]
    out_f32 = jnp.where(denom > 0, acc_ref[...] / denom, 0.0)
    out_ref[0, :, :, :] = out_f32.astype(out_ref.dtype)


def _causal_attention_pallas(q_nope, q_rope, k_nope, k_rope, v):
    """
    Attention where k_rope is broadcast to have head dimension.
    
    Args:
        q_nope: (B, H, S, nope_dim)
        q_rope: (B, H, S, rope_dim)
        k_nope: (B, H, S, nope_dim)
        k_rope: (B, H, S, rope_dim) - broadcast across heads
        v: (B, H, S, v_dim)
    
    Returns:
        output: (B, H, S, v_dim)
    """
    B, H, S, nope_dim = q_nope.shape
    rope_dim = q_rope.shape[-1]
    Dv = v.shape[-1]
    num_h_blocks = H // _ATTN_HEAD_BLOCK
    num_q_blocks = S // _ATTN_QUERY_BLOCK

    return pl.pallas_call(
        _causal_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, Dv), v.dtype),
        grid=(B, num_h_blocks, num_q_blocks),
        in_specs=[
            # q_nope: (B, H, S, nope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, nope_dim), 
                        lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            # q_rope: (B, H, S, rope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, rope_dim), 
                        lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
            # k_nope: (B, H, S, nope_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, nope_dim), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            # k_rope: (B, H, S, rope_dim) - broadcast across heads
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, rope_dim), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
            # v: (B, H, S, v_dim)
            pl.BlockSpec((1, _ATTN_HEAD_BLOCK, S, Dv), 
                        lambda b, h, q_blk: (b, h, 0, 0), pipeline_mode=pl.Buffered(1)),
        ],
        out_specs=pl.BlockSpec((1, _ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), 
                              lambda b, h, q_blk: (b, h, q_blk, 0), pipeline_mode=pl.Buffered(1)),
        scratch_shapes=[
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK, Dv), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
            pltpu.VMEM((_ATTN_HEAD_BLOCK, _ATTN_QUERY_BLOCK), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            allow_input_fusion=(True, True, True, True, True),
        ),
    )(q_nope, q_rope, k_nope, k_rope, v)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    x2d = x.reshape(B * S, E)

    # Down projections (fused)
    q_low2d, kv2d = _pallas_dual_gemm_down(x2d, q_down_proj, kv_down_proj)

    # Q up projection
    q2d = _pallas_gemm(q_low2d, q_up_proj)
    q = q2d.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]

    # KV processing
    kv = kv2d.reshape(B, S, kvl + rope)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]  # (B, S, rope)

    k_latent2d = k_latent.reshape(B * S, kvl)
    k_nope2d, v2d = _pallas_dual_gemm_same_n(k_latent2d, k_up_proj, v_up_proj)

    k_nope = k_nope2d.reshape(B, S, H, nope)
    v = v2d.reshape(B, S, H, vd)

    # Apply RoPE
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    
    # Apply RoPE to Q (per-head)
    q_rope = _apply_rope(q_rope, cos, sin)
    
    # Apply RoPE to K_rope and broadcast to head dimension
    # k_rope_raw is (B, S, rope)
    k_rope_with_rope = _apply_rope_2d(k_rope_raw[0], cos, sin)  # (S, rope)
    # Broadcast to (B, H, S, rope) for attention
    k_rope_broadcast = jnp.broadcast_to(
        k_rope_with_rope[None, None, :, :], 
        (B, H, S, rope)
    )

    # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
    q_nope_t = q_nope.transpose(0, 2, 1, 3)
    q_rope_t = q_rope.transpose(0, 2, 1, 3)
    k_nope_t = k_nope.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Attention with k_rope broadcast across heads
    out = _causal_attention_pallas(q_nope_t, q_rope_t, k_nope_t, k_rope_broadcast, v_t)

    # Output projection
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    out2d = out.reshape(B * S, H * vd).astype(x.dtype)

    final2d = _pallas_gemm(out2d, o_proj)
    return final2d.reshape(B, S, E)
''',
score=4.032,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 4.032 ms\n{"correct": true, "latency": 4.032, "error": "", "all_times_ms": [4.017, 4.018, 4.02, 4.02, 4.02, 4.02, 4.02, 4.021, 4.021, 4.021, 4.021, 4.022, 4.023, 4.024, 4.024, 4.025, 4.025, 4.025, 4.026, 4.026, 4.026, 4.026, 4.027, 4.027, 4.027, 4.027, 4.028, 4.028, 4.028, 4.029, 4.029, 4.029, 4.029, 4.029, 4.029, 4.03, 4.03, 4.03, 4.03, 4.03, 4.03, 4.031, 4.031, 4.031, 4.031, 4.031, 4.031, 4.031, 4.031, 4.031, 4.032, 4.032, 4.032, 4.032, 4.032, 4.032, 4.032, 4.032, 4.033, 4.033, 4.033, 4.034, 4.034, 4.034, 4.034, 4.034, 4.035, 4.035, 4.036, 4.036, 4.037, 4.037, 4.038, 4.038, 4.038, 4.039, 4.039, 4.04, 4.04, 4.04, 4.04, 4.041, 4.041, 4.042, 4.042, 4.042, 4.042, 4.043, 4.043, 4.044, 4.045, 4.045, 4.046, 4.052, 4.052, 4.056, 4.06, 4.068, 4.074, 4.12], "max_diff": 0.03125, "max_rel_diff": 0.004303}',
stderr='')