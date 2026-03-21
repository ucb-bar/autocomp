CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''import jax

import jax.numpy as jnp

from functools import partial

CONFIG = {
    \'name\': \'mamba2_2_7b_ssd\',
    \'model\': \'Mamba-2-2.7B\',
    \'operator\': \'state_space_duality\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'num_heads\': 64,
    \'head_dim\': 64,
    \'d_state\': 128,
    \'d_model\': 2560,
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG[\'batch\'], CONFIG[\'seq_len\']
    H, D = CONFIG[\'num_heads\'], CONFIG[\'head_dim\']
    # In Mamba-2 SSD: C maps to Q, B maps to K, x maps to V
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)  # C (output projection)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)  # B (input projection)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)  # x (hidden state)
    # A: input-dependent decay (after log-space parameterization)
    # Initialized negative (stable decay), per-head scalar
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log

def workload(query, key, value, A_log):
    """Mamba-2 SSD: structured linear attention with selective decay.

    y = (L ⊙ (C B^T)) x
    where L[i,j] = Π_{k=j+1}^{i} a_k for i > j, 1 for i=j, 0 for i<j
    and a_k = exp(A_log_k) is the selective (input-dependent) decay.
    """
    B, H, S, D = query.shape

    # Compute per-position decay: a = sigmoid(A_log) to keep in (0, 1)
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))  # (B, H, S)

    # Build causal mask L with cumulative decay
    # log(a) cumsum then exponentiate: L[i,j] = exp(Σ_{k=j+1}^{i} log(a_k))
    log_a = jnp.log(a + 1e-8)  # (B, H, S)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)  # (B, H, S)

    # L[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j, 0 for i < j
    diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

    # SSD attention: (L ⊙ CB^T) x
    # CB^T: (B, H, S, S) — "attention scores"
    scores = jnp.einsum(\'bhsd,bhtd->bhst\',
                        query.astype(jnp.float32),
                        key.astype(jnp.float32))

    # Apply selective decay mask
    scores = scores * L

    # Normalize
    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

    # Output
    output = jnp.einsum(\'bhst,bhtd->bhsd\', scores.astype(query.dtype), value)
    return output
''',
score=1.587,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 1.587 ms\n{"correct": true, "latency": 1.587, "error": "", "all_times_ms": [1.56, 1.562, 1.562, 1.563, 1.563, 1.564, 1.564, 1.565, 1.566, 1.566, 1.566, 1.567, 1.567, 1.568, 1.568, 1.569, 1.571, 1.571, 1.571, 1.571, 1.572, 1.573, 1.573, 1.574, 1.575, 1.575, 1.576, 1.576, 1.576, 1.576, 1.577, 1.577, 1.577, 1.578, 1.578, 1.58, 1.58, 1.58, 1.58, 1.581, 1.582, 1.582, 1.583, 1.584, 1.584, 1.584, 1.586, 1.586, 1.587, 1.587, 1.587, 1.589, 1.589, 1.59, 1.591, 1.591, 1.592, 1.592, 1.593, 1.595, 1.596, 1.598, 1.6, 1.603, 1.615, 1.616, 1.624, 1.634, 1.643, 1.674, 1.892, 1.893, 1.897, 1.902, 1.981, 1.985, 2.001, 2.003, 2.016, 2.022, 2.023, 2.031, 2.05, 2.155, 2.164, 2.293, 2.323, 2.549, 2.632, 2.664, 2.681, 2.707, 2.742, 2.744, 2.747, 2.761, 2.872, 2.895, 2.899, 2.923], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Plan:

1.  **Target Identification**: The `workload` function performs two core computations:
    *   `scores = jnp.einsum(\'bhsd,bhtd->bhst\', query, key)`: A vanilla matrix multiplication (batched GEMM).
    *   `output = jnp.einsum(\'bhst,bhtd->bhsd\', scores, value)`: A vanilla matrix multiplication (batched GEMM).

2.  **Strategy Selection**: I will apply **Strategy 3** ("Decompose matrix multiplications into blocked grid loops...") to implement these two GEMMs using `pl.pallas_call`. This strategy is ideal for maximizing throughput on the TPU v6e TensorCore by tiling the computation according to hardware constraints (8x128 tiles), explicitly managing accumulation in float32, and ensuring correct reduction ordering.

3.  **Detailed Conversion**:
    *   **MatMul 1 (`scores = query @ key`)**:
        *   **Geometry**: Inputs: `query` (B, H, S, D), `key` (B, H, S, D). Output: `scores` (B, H, S, S). We define tile sizes: `bm` (rows of output, tile S), `bn` (cols of output, tile S), `bk` (inner dim, tile D).
        *   **Grid**: `(B, H, S // bm, S // bn, D // bk)`. This grid ensures lexicographic ordering where the reduction dimension `D` tile index is innermost. This allows us to accumulate in a scratch buffer in VMEM across loop iterations effectively.
        *   **BlockSpecs**:
            *   `in_specs`: Maps grid indices `(b, h, i, j, k)` to slices of `query` `(bm, bk)` and `key` `(bk, bn)`.
            *   `out_specs`: Maps grid indices to output slices `(bm, bn)`.
        *   **Kernel**: Implements the GEMM body. The accumulator lives in VMEM (`pltpu.VMEM` scratch). We use `pl.when(pl.program_id(4) == 0)` to initialize the accumulator to zero only on the first iteration of the reduction loop (k=0). Subsequent iterations accumulate. The result is flushed to the output reference on the final iteration. Operations use `jnp.dot(..., preferred_element_type=jnp.float32)` to meet the accumulator precision requirement.

    *   **MatMul 2 (`output = scores @ value`)**:
        *   **Geometry**: Inputs: `scores` (B, H, S, S), `value` (B, H, S, D). Output: `output` (B, H, S, D). Tile sizes: `bm` (tile S), `bn` (tile D), `bk` (tile S).
        *   **Grid**: `(B, H, S // bm, D // bn, S // bk)`. Reduction dimension `bk` (inner S) is innermost.
        *   **Implementation**: Identical structure to MatMul 1, using appropriate tile shapes and indices.

    *   **Element-wise Passthrough**: The element-wise operations (computing `a`, `log_a`, masking `scores`, normalization) remain in JAX but are reordered and applied to the results of the kernels. This adheres to Strategy 3\'s focus on the matmul decomposition. The final cast to `query.dtype` (`bfloat16`) is handled in JAX.

4.  **Hardware Compliance**:
    *   **VMEM**: We rely on double-buffering (implicit in `pl.pallas_call`) and careful choice of `bm, bn, bk` (e.g., 64x64x64 or 128x128x32 fits in 16MB).
    *   **Alignment**: Tile dimensions will be chosen (e.g., 128) to align with the (8, 128) native tile constraints.
    *   **Single-threaded logic**: Kernel bodies are sequential loops (`for` loop over grid dimensions via `pallas_call` mechanism, with explicit accumulation logic).

5.  **Refactoring**:
    *   The function `workload` will be rewritten to use `jax.jit` and `pl.pallas_call`.
    *   Helper functions for kernel bodies (`matmul_t1_kernel`, `matmul_t2_kernel`) will be defined.
    *   Constants (BlockSpecs, Grids) will be computed dynamically inside `workload` or as a compiled configuration.

Resulting Code Structure:
```python
# Setup and helpers...

@partial(jax.jit, static_argnames=[\'bm\', \'bn\', \'bk\'])
def workload(query, key, value, A_log, *, bm=128, bn=128, bk=64):
    # ... (Element-wise logic for \'L\' mask remains in JAX) ...
    
    # ... MatMul 1 Kernel ...
    # ... MatMul 2 Kernel ...
    
    return output
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _reference_workload(query, key, value, A_log):
    """Original JAX formulation; used as a fallback for unsupported shapes."""
    B, H, S, D = query.shape

    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

    scores = jnp.einsum(
        "bhsd,bhtd->bhst",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )
    scores = scores * L

    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

    output = jnp.einsum("bhst,bhtd->bhsd", scores.astype(query.dtype), value)
    return output


def _qk_matmul_pallas(query, key, *, bm=256, bn=256):
    """Blocked QK^T -> scores, accumulated in f32."""
    B, H, S, D = query.shape
    k_tiles = 1  # D is taken as one full-width tile on v6e (D=64 in this workload).

    def kernel(q_ref, k_ref, o_ref, acc_ref):
        rk = pl.program_id(4)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        q = q_ref[...]
        k = k_ref[...]

        # (bm, D) @ (bn, D)^T -> (bm, bn), accumulate in f32.
        update = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H, S // bm, S // bn, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, D),
                index_map=lambda b, h, i, j, k: (b, h, i, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bn, D),
                index_map=lambda b, h, i, j, k: (b, h, j, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, bn),
            index_map=lambda b, h, i, j, k: (b, h, i, j),
        ),
        scratch_shapes=(pltpu.VMEM((bm, bn), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key)


def _sv_matmul_pallas(scores, value, *, bm=256, bk=256, out_dtype=jnp.bfloat16):
    """Blocked scores @ value -> output, accumulated in f32."""
    B, H, S, D = value.shape
    k_tiles = S // bk

    def kernel(s_ref, v_ref, o_ref, acc_ref):
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        s = s_ref[...]
        v = v_ref[...]

        # (bm, bk) @ (bk, D) -> (bm, D), accumulate in f32.
        update = jnp.dot(s, v, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), out_dtype),
        grid=(B, H, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, bk),
                index_map=lambda b, h, i, k: (b, h, i, k),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bk, D),
                index_map=lambda b, h, i, k: (b, h, k, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, D),
            index_map=lambda b, h, i, k: (b, h, i, 0),
        ),
        scratch_shapes=(pltpu.VMEM((bm, D), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores, value)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD rewritten to use TPU Pallas blocked GEMMs for the two einsums."""
    # Static shape checks. If the shape is not tile-friendly for this kernel layout,
    # fall back to the original formulation.
    if (
        query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or A_log.ndim != 3
        or query.shape != key.shape
        or query.shape != value.shape
        or A_log.shape != query.shape[:-1]
    ):
        return _reference_workload(query, key, value, A_log)

    B, H, S, D = query.shape

    # Tile choices tuned for v6e-friendly blocked execution.
    # S tiles are multiples of 128; D is taken as the full trailing dimension (64).
    bm_qk = 256
    bn_qk = 256
    bm_sv = 256
    bk_sv = 256

    if (S % bm_qk) != 0 or (S % bn_qk) != 0 or (S % bm_sv) != 0 or (S % bk_sv) != 0:
        return _reference_workload(query, key, value, A_log)

    # GEMM 1: scores = query @ key^T, output in f32.
    scores = _qk_matmul_pallas(query, key, bm=bm_qk, bn=bn_qk)

    # Original elementwise selective-decay logic stays in JAX.
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

    scores = scores * L

    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

    # Original code casts scores to query.dtype before the second einsum.
    scores_for_output = scores.astype(query.dtype)
    out_dtype = jnp.result_type(scores_for_output.dtype, value.dtype)

    # GEMM 2: output = scores @ value.
    output = _sv_matmul_pallas(
        scores_for_output,
        value,
        bm=bm_sv,
        bk=bk_sv,
        out_dtype=out_dtype,
    )
    return output
''',
score=5.905,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-4.7',
code_gen_model='gpt-5.4',
stdout='Latency: 5.905 ms\n{"correct": true, "latency": 5.905, "error": "", "all_times_ms": [5.887, 5.889, 5.89, 5.891, 5.891, 5.891, 5.892, 5.892, 5.893, 5.893, 5.894, 5.895, 5.896, 5.896, 5.896, 5.897, 5.897, 5.897, 5.897, 5.897, 5.898, 5.898, 5.899, 5.899, 5.899, 5.899, 5.899, 5.9, 5.9, 5.901, 5.901, 5.902, 5.902, 5.902, 5.903, 5.903, 5.903, 5.904, 5.904, 5.904, 5.904, 5.904, 5.904, 5.904, 5.905, 5.905, 5.905, 5.905, 5.905, 5.905, 5.905, 5.906, 5.906, 5.907, 5.907, 5.907, 5.907, 5.908, 5.908, 5.909, 5.909, 5.909, 5.909, 5.909, 5.909, 5.909, 5.91, 5.91, 5.91, 5.91, 5.91, 5.911, 5.911, 5.911, 5.912, 5.912, 5.912, 5.913, 5.913, 5.914, 5.914, 5.914, 5.915, 5.915, 5.915, 5.916, 5.917, 5.917, 5.919, 5.919, 5.919, 5.919, 5.921, 5.922, 5.923, 5.928, 5.929, 5.934, 5.934, 5.957], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 4 — replace large elementwise array code with pipelined TPU Pallas kernels**

The part of this program that still most needs conversion is the **post-`QKᵀ` path over the full `(B, H, S, S)` scores tensor**:

```python
diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

scores = scores * L

scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)
scores_for_output = scores.astype(query.dtype)
```

That section is still high-level JAX over a huge `S x S` tensor and materializes intermediates that are bad for v6e-1.

## Plan

### 1. Keep the public entrypoint unchanged
Keep:

```python
@jax.jit
def workload(query, key, value, A_log):
    ...
```

Do not change its signature.

Also remove the fallback path logic from `workload`; this phase should target the fixed tile-friendly workload directly rather than branching to `_reference_workload`.

---

### 2. Leave the two blocked GEMMs in place
Your existing `_qk_matmul_pallas(...)` and `_sv_matmul_pallas(...)` are already in the right style for TPU blocked MXU use.

Do **not** retarget those in this phase.

---

### 3. Keep only the small 3D prefix computation in JAX
These are relatively small and should stay high-level for now:

```python
a = jax.nn.sigmoid(A_log.astype(jnp.float32))
log_a = jnp.log(a + 1e-8)
log_a_cumsum = jnp.cumsum(log_a, axis=-1)
```

This works over `(B, H, S)`, not `(B, H, S, S)`, so it is not the memory problem.

---

### 4. Replace `diff/causal/L` materialization with a tiled Pallas kernel
Add a helper like `_apply_decay_pallas(scores, log_a_cumsum)`.

#### What it should do
For each tile of `scores`, compute the decay factor on-chip instead of building full dense `diff`, `causal`, and `L` arrays in HBM:

- load one `(1, 8, 128, 128)` scores tile
- load matching `(1, 8, 128)` row prefix tile from `log_a_cumsum`
- load matching `(1, 8, 128)` column prefix tile from `log_a_cumsum`
- inside the kernel:
  - read refs explicitly (`scores_ref[...]`, etc.)
  - compute
    - `diff = row_prefix[..., :, None] - col_prefix[..., None, :]`
    - causal mask from tile-local/global row/col indices
    - `decay = exp(where(causal, diff, -1e30))`
  - write `scores * decay` to the output ref

#### Why this fits v6e-1
Use a tile layout aligned to TPU rules:

- `scores` block shape: **`(1, 8, 128, 128)`**
- `log_a_cumsum` block shapes: **`(1, 8, 128)`**

This avoids singleton trailing dimensions and respects the last-two-dims rule:
- second-to-last dim divisible by 8
- last dim divisible by 128

#### Grid / semantics
Use a grid like:

```python
grid = (B, H // 8, S // 128, S // 128)
```

with:

```python
dimension_semantics = ("parallel", "parallel", "parallel", "parallel")
```

because each output tile is independent.

#### Input/output mapping
Use `pl.BlockSpec` so the kernel sees tiles, not full arrays:

- `scores`: `(1, 8, 128, 128)` mapped by `(b, hg, i, j) -> (b, hg, i, j)`
- `row_prefix`: `(1, 8, 128)` mapped by `(b, hg, i, j) -> (b, hg, i)`
- `col_prefix`: `(1, 8, 128)` mapped by `(b, hg, i, j) -> (b, hg, j)`

Use `pltpu.PrefetchScalarGridSpec` for the small scalar args/constants used by the kernel (e.g. epsilon / sentinel constants), while the large tensor tiles are moved through VMEM via `BlockSpec`.

---

### 5. Do not keep a trailing singleton normalization tensor
Right now the code uses `keepdims=True`, producing `(B, H, S, 1)`. That is a bad TPU shape because singleton trailing dimensions are wasteful.

Instead, in the rewritten flow, make the normalization factor a **3D tensor `(B, H, S)`**, not `(B, H, S, 1)`.

Even if the reduction itself stays in JAX in this phase, immediately squeeze it to `(B, H, S)` before feeding it to a Pallas kernel.

---

### 6. Replace the broadcast divide/cast with a second tiled Pallas kernel
Add a helper like `_normalize_scores_pallas(decayed_scores, row_sum, out_dtype)`.

#### What it should do
Tile over the same `(1, 8, 128, 128)` score blocks and a `(1, 8, 128)` row-sum block, then inside the kernel compute the existing scalar logic rowwise:

```python
safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
den = jnp.maximum(jnp.abs(safe_sum), 1.0)
normalized = decayed_scores / den[..., :, None]
```

Then cast on store:

```python
o_ref[...] = normalized.astype(out_dtype)
```

That preserves the current semantics while moving the expensive broadcasted elementwise work out of high-level JAX.

#### Important TPU details
- do all elementwise math in `float32`
- only cast to `query.dtype` at the output write
- read/write refs explicitly; no arithmetic on bare refs

---

### 7. Update `workload` to call the new kernels
The high-level structure becomes:

1. `scores = _qk_matmul_pallas(query, key, ...)`
2. `log_a_cumsum = ...` in JAX
3. `scores = _apply_decay_pallas(scores, log_a_cumsum)`
4. keep the row reduction as-is for now
5. `scores_for_output = _normalize_scores_pallas(scores, row_sum, query.dtype)`
6. `output = _sv_matmul_pallas(scores_for_output, value, ...)`

So the huge dense elementwise path is moved to TPU kernels, but the function name/signature remains unchanged.

---

### 8. Tile sizes to use
For v6e-1, use:

- head-group tile: `8`
- row tile: `128`
- col tile: `128`

So the main score kernel tile is `(1, 8, 128, 128)`.

This is a good fit because:
- `H=64` is divisible by 8
- `S=2048` is divisible by 128
- VMEM usage stays modest:
  - one f32 score tile is about 512 KiB
  - plus output tile and two small prefix tiles still fits comfortably

---

## In short
The specific code to convert in this phase is the **large `(B,H,S,S)` elementwise selective-decay and normalization path**. Replace it with **two pipelined TPU Pallas kernels** using `(1,8,128,128)` / `(1,8,128)` block specs and `PrefetchScalarGridSpec`, while keeping the existing blocked GEMMs and the small `(B,H,S)` prefix-scan logic unchanged.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _qk_matmul_pallas(query, key, *, bm=256, bn=256):
    """Blocked QK^T -> scores, accumulated in f32."""
    B, H, S, D = query.shape
    k_tiles = 1  # D is taken as one full-width tile on v6e (D=64 in this workload).

    def kernel(q_ref, k_ref, o_ref, acc_ref):
        rk = pl.program_id(4)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        q = q_ref[...]
        k = k_ref[...]

        # (bm, D) @ (bn, D)^T -> (bm, bn), accumulate in f32.
        update = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H, S // bm, S // bn, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, D),
                index_map=lambda b, h, i, j, k: (b, h, i, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bn, D),
                index_map=lambda b, h, i, j, k: (b, h, j, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, bn),
            index_map=lambda b, h, i, j, k: (b, h, i, j),
        ),
        scratch_shapes=(pltpu.VMEM((bm, bn), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key)


def _sv_matmul_pallas(scores, value, *, bm=256, bk=256, out_dtype=jnp.bfloat16):
    """Blocked scores @ value -> output, accumulated in f32."""
    B, H, S, D = value.shape
    k_tiles = S // bk

    def kernel(s_ref, v_ref, o_ref, acc_ref):
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        s = s_ref[...]
        v = v_ref[...]

        # (bm, bk) @ (bk, D) -> (bm, D), accumulate in f32.
        update = jnp.dot(s, v, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), out_dtype),
        grid=(B, H, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, bk),
                index_map=lambda b, h, i, k: (b, h, i, k),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bk, D),
                index_map=lambda b, h, i, k: (b, h, k, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, D),
            index_map=lambda b, h, i, k: (b, h, i, 0),
        ),
        scratch_shapes=(pltpu.VMEM((bm, D), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores, value)


def _apply_decay_pallas(scores, log_a_cumsum, *, bh=8, bm=128, bn=128):
    """
    Apply selective decay to scores in a tiled manner.
    
    Computes: scores * exp(where(causal, row_prefix - col_prefix, -1e30))
    where causal is the lower triangular mask.
    """
    B, H, S, _ = scores.shape
    
    # Reshape log_a_cumsum for tiled access: (B, H, S) -> (B, H//bh, bh, S//bm, bm)
    # But we\'ll use BlockSpec with appropriate indexing instead
    
    def kernel(scores_ref, row_prefix_ref, col_prefix_ref, out_ref):
        # Get grid indices
        b = pl.program_id(0)
        hg = pl.program_id(1)
        i = pl.program_id(2)
        j = pl.program_id(3)
        
        # Read tiles - scores_ref is (bh, bm, bn), prefix refs are (bh, bm) or (bh, bn)
        scores_tile = scores_ref[...].astype(jnp.float32)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)
        
        # Compute diff: row_prefix[:, :, None] - col_prefix[:, None, :]
        # row_prefix is (bh, bm), col_prefix is (bh, bn)
        # We want diff to be (bh, bm, bn)
        diff = row_prefix[:, :, None] - col_prefix[:, None, :]
        
        # Compute global indices for causal mask
        # Global row indices: i * bm + local_row
        # Global col indices: j * bn + local_col
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        
        global_rows = i * bm + local_rows  # (bm,)
        global_cols = j * bn + local_cols  # (bn,)
        
        # Causal mask: row >= col (lower triangular)
        # Shape: (bm, bn)
        causal_mask = global_rows[:, None] >= global_cols[None, :]
        
        # Broadcast causal mask to (bh, bm, bn)
        causal_mask_broadcast = jnp.broadcast_to(causal_mask[None, :, :], (bh, bm, bn))
        
        # Compute decay factor
        decay = jnp.exp(jnp.where(causal_mask_broadcast, diff, -1e30))
        
        # Apply decay to scores
        result = scores_tile * decay
        
        out_ref[...] = result
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            # scores: (B, H, S, S) -> block (1, bh, bm, bn)
            pl.BlockSpec(
                block_shape=(None, bh, bm, bn),
                index_map=lambda b, hg, i, j: (b, hg, i, j),
            ),
            # row_prefix (log_a_cumsum): (B, H, S) -> block (1, bh, bm)
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
            # col_prefix (log_a_cumsum): (B, H, S) -> block (1, bh, bn)
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, i, j: (b, hg, j),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, bn),
            index_map=lambda b, hg, i, j: (b, hg, i, j),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel"),
        ),
    )(scores, log_a_cumsum, log_a_cumsum)


def _normalize_scores_pallas(scores, row_sum, out_dtype, *, bh=8, bm=128, bn=128):
    """
    Normalize scores by row sum and cast to output dtype.
    
    Computes: (scores / max(|safe_sum|, 1.0)).astype(out_dtype)
    where safe_sum = where(|row_sum| < 1e-6, 1.0, row_sum)
    """
    B, H, S, _ = scores.shape
    
    def kernel(scores_ref, row_sum_ref, out_ref):
        # Read tiles
        scores_tile = scores_ref[...].astype(jnp.float32)  # (bh, bm, bn)
        row_sum_tile = row_sum_ref[...].astype(jnp.float32)  # (bh, bm)
        
        # Compute safe denominator
        # safe_sum = where(|row_sum| < 1e-6, 1.0, row_sum)
        safe_sum = jnp.where(jnp.abs(row_sum_tile) < 1e-6, 1.0, row_sum_tile)
        
        # den = max(|safe_sum|, 1.0)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)
        
        # Broadcast den to match scores shape: (bh, bm) -> (bh, bm, bn)
        den_broadcast = den[:, :, None]
        
        # Normalize
        normalized = scores_tile / den_broadcast
        
        # Cast to output dtype
        out_ref[...] = normalized.astype(out_dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), out_dtype),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            # scores: (B, H, S, S) -> block (1, bh, bm, bn)
            pl.BlockSpec(
                block_shape=(None, bh, bm, bn),
                index_map=lambda b, hg, i, j: (b, hg, i, j),
            ),
            # row_sum: (B, H, S) -> block (1, bh, bm)
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, bn),
            index_map=lambda b, hg, i, j: (b, hg, i, j),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel"),
        ),
    )(scores, row_sum)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD rewritten to use TPU Pallas blocked GEMMs and elementwise kernels."""
    B, H, S, D = query.shape

    # Tile choices tuned for v6e-friendly blocked execution.
    bm_qk = 256
    bn_qk = 256
    bm_sv = 256
    bk_sv = 256
    
    # Elementwise kernel tile sizes
    bh_elem = 8
    bm_elem = 128
    bn_elem = 128

    # GEMM 1: scores = query @ key^T, output in f32.
    scores = _qk_matmul_pallas(query, key, bm=bm_qk, bn=bn_qk)

    # Small 3D prefix computation stays in JAX (operates on (B, H, S), not (B, H, S, S))
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    # Apply selective decay using tiled Pallas kernel
    # This replaces: diff, causal, L computation and scores * L
    scores = _apply_decay_pallas(
        scores, log_a_cumsum, 
        bh=bh_elem, bm=bm_elem, bn=bn_elem
    )

    # Row reduction - keep in JAX for now, but squeeze to avoid singleton trailing dim
    # scores_sum shape: (B, H, S) instead of (B, H, S, 1)
    row_sum = jnp.sum(scores, axis=-1)  # (B, H, S)

    # Normalize scores and cast using tiled Pallas kernel
    scores_for_output = _normalize_scores_pallas(
        scores, row_sum, query.dtype,
        bh=bh_elem, bm=bm_elem, bn=bn_elem
    )

    out_dtype = jnp.result_type(scores_for_output.dtype, value.dtype)

    # GEMM 2: output = scores @ value.
    output = _sv_matmul_pallas(
        scores_for_output,
        value,
        bm=bm_sv,
        bk=bk_sv,
        out_dtype=out_dtype,
    )
    return output
''',
score=8.773,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 8.773 ms\n{"correct": true, "latency": 8.773, "error": "", "all_times_ms": [8.749, 8.753, 8.754, 8.757, 8.758, 8.758, 8.76, 8.76, 8.76, 8.761, 8.761, 8.762, 8.762, 8.762, 8.763, 8.763, 8.764, 8.764, 8.764, 8.764, 8.765, 8.765, 8.765, 8.766, 8.767, 8.767, 8.767, 8.768, 8.768, 8.768, 8.768, 8.768, 8.769, 8.769, 8.769, 8.769, 8.77, 8.77, 8.771, 8.771, 8.771, 8.771, 8.772, 8.772, 8.773, 8.773, 8.773, 8.773, 8.773, 8.773, 8.773, 8.773, 8.774, 8.774, 8.774, 8.775, 8.775, 8.775, 8.776, 8.776, 8.776, 8.776, 8.776, 8.777, 8.777, 8.777, 8.777, 8.777, 8.778, 8.778, 8.778, 8.778, 8.779, 8.779, 8.78, 8.78, 8.78, 8.78, 8.781, 8.783, 8.784, 8.784, 8.784, 8.786, 8.786, 8.786, 8.786, 8.787, 8.789, 8.79, 8.79, 8.79, 8.791, 8.795, 8.796, 8.796, 8.798, 8.799, 8.8, 8.808], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''I\'ll analyze the code and identify which strategy is most appropriate. Looking at the code, I can see it\'s already using Pallas kernels for the main matrix multiplications, but there are several JAX operations that could be optimized for TPU v6e hardware.

The most promising optimization opportunity is Strategy 5: Convert reduction operations into Pallas kernels. Specifically, I\'m looking at the row sum calculation:

```python
# Current JAX-based reduction
row_sum = jnp.sum(scores, axis=-1)  # (B, H, S, 1) -> (B, H, S)
```

This reduction operation is memory-intensive and would benefit significantly from being converted to a Pallas kernel. Using Strategy 5 would allow this reduction to be:
1. More efficient by operating on tiles in VMEM
2. Better optimized for the TPU memory hierarchy  
3. Prevent multiple passes over the data

**Plan to apply Strategy 5:**
I\'ll convert the reduction operation `jnp.sum(scores, axis=-1)` into a Pallas kernel where the reduction axis is processed in a blocked, tiled fashion. This is ideal because:
1. Matrix multiplication results in large 4D tensors that are expensive to reduce 
2. The reduction generates a new singleton dimension that gets explicitly squeezed
3. The Pallas kernel can accumulate directly into VMEM without extra data movement

The spectral distribution shows this sum reduction (axis=-1) could be optimized for v6e TPU by using:
- Tiled accumulation in scratch buffers
- Proper dimension semantics for parallel/arbitrary axes
- Reduction domain decomposition across VMEM/SMEM with proper tiling

The kernel would:
1. Tile the input scores tensor (B, H, S, S) into blocks
2. Accumulate partial sums across the reduction dimension
3. Write finalized sums for each grid of (B, H, S//bm) blocks

This specifically avoids moving the entire 4D tensor through memory multiple times.

**Additional Optimization Opportunities:**
- The elementwise operations like `sigmoid`, `log`, `cumsum`, etc., in the intermediate JAX functions could use Strategy 4 (pipelined kernels) for memory bandwidth efficiency
- BM/BK tiling for the row_sum operation could be (128, 128) or similar to align with 8x128 tile constraints
- Could fuse operations if we needed further optimization''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _qk_matmul_pallas(query, key, *, bm=256, bn=256):
    """Blocked QK^T -> scores, accumulated in f32."""
    B, H, S, D = query.shape
    k_tiles = 1  # Full-width D tile for this workload (D=64).

    def kernel(q_ref, k_ref, o_ref, acc_ref):
        rk = pl.program_id(4)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        q = q_ref[...]
        k = k_ref[...]

        update = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H, S // bm, S // bn, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, D),
                index_map=lambda b, h, i, j, k: (b, h, i, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bn, D),
                index_map=lambda b, h, i, j, k: (b, h, j, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, bn),
            index_map=lambda b, h, i, j, k: (b, h, i, j),
        ),
        scratch_shapes=(pltpu.VMEM((bm, bn), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key)


def _apply_decay_pallas(scores, log_a_cumsum, *, bh=8, bm=128, bn=128):
    """
    Apply selective decay to scores in a tiled manner.

    Computes:
      scores * exp(where(causal, row_prefix - col_prefix, -1e30))
    """
    B, H, S, _ = scores.shape

    def kernel(scores_ref, row_prefix_ref, col_prefix_ref, out_ref):
        i = pl.program_id(2)
        j = pl.program_id(3)

        scores_tile = scores_ref[...].astype(jnp.float32)     # (bh, bm, bn)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

        diff = row_prefix[:, :, None] - col_prefix[:, None, :]

        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        global_rows = i * bm + local_rows
        global_cols = j * bn + local_cols

        causal_mask = global_rows[:, None] >= global_cols[None, :]
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], (bh, bm, bn))

        decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
        out_ref[...] = scores_tile * decay

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, bn),
                index_map=lambda b, hg, i, j: (b, hg, i, j),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, i, j: (b, hg, j),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, bn),
            index_map=lambda b, hg, i, j: (b, hg, i, j),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel"),
        ),
    )(scores, log_a_cumsum, log_a_cumsum)


def _row_sum_pallas(scores, *, bh=8, bm=128, bk=128):
    """
    Reduce scores over the last axis using a tiled Pallas reduction.

    Input:
      scores: (B, H, S, S) float32
    Output:
      row_sum: (B, H, S) float32
    """
    B, H, S, _ = scores.shape
    k_tiles = S // bk

    def kernel(scores_ref, out_ref, acc_ref):
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        scores_tile = scores_ref[...].astype(jnp.float32)  # (bh, bm, bk)
        partial = jnp.sum(scores_tile, axis=-1)            # (bh, bm)
        acc_ref[...] = acc_ref[...] + partial

        @pl.when(rk == k_tiles - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S), jnp.float32),
        grid=(B, H // bh, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, bk),
                index_map=lambda b, hg, i, k: (b, hg, i, k),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm),
            index_map=lambda b, hg, i, k: (b, hg, i),
        ),
        scratch_shapes=(pltpu.VMEM((bh, bm), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores)


def _normalize_scores_pallas(scores, row_sum, out_dtype, *, bh=8, bm=128, bn=128):
    """
    Normalize scores by row sum and cast to output dtype.

    Computes:
      (scores / max(|safe_sum|, 1.0)).astype(out_dtype)
    where:
      safe_sum = where(|row_sum| < 1e-6, 1.0, row_sum)
    """
    B, H, S, _ = scores.shape

    def kernel(scores_ref, row_sum_ref, out_ref):
        scores_tile = scores_ref[...].astype(jnp.float32)   # (bh, bm, bn)
        row_sum_tile = row_sum_ref[...].astype(jnp.float32) # (bh, bm)

        safe_sum = jnp.where(jnp.abs(row_sum_tile) < 1e-6, 1.0, row_sum_tile)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)

        normalized = scores_tile / den[:, :, None]
        out_ref[...] = normalized.astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), out_dtype),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, bn),
                index_map=lambda b, hg, i, j: (b, hg, i, j),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, bn),
            index_map=lambda b, hg, i, j: (b, hg, i, j),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel"),
        ),
    )(scores, row_sum)


def _sv_matmul_pallas(scores, value, *, bm=256, bk=256, out_dtype=jnp.bfloat16):
    """Blocked scores @ value -> output, accumulated in f32."""
    B, H, S, D = value.shape
    k_tiles = S // bk

    def kernel(s_ref, v_ref, o_ref, acc_ref):
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        s = s_ref[...]
        v = v_ref[...]

        update = jnp.dot(s, v, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), out_dtype),
        grid=(B, H, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, bk),
                index_map=lambda b, h, i, k: (b, h, i, k),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bk, D),
                index_map=lambda b, h, i, k: (b, h, k, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, D),
            index_map=lambda b, h, i, k: (b, h, i, 0),
        ),
        scratch_shapes=(pltpu.VMEM((bm, D), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores, value)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using TPU Pallas blocked kernels, targeting v6e-class TPU execution."""
    B, H, S, D = query.shape

    # GEMM tiling.
    bm_qk = 256
    bn_qk = 256
    bm_sv = 256
    bk_sv = 256

    # Elementwise / reduction tiling.
    bh_elem = 8
    bm_elem = 128
    bn_elem = 128
    bk_reduce = 128

    # GEMM 1: scores = query @ key^T, output in f32.
    scores = _qk_matmul_pallas(query, key, bm=bm_qk, bn=bn_qk)

    # Prefix computation remains in JAX on the much smaller (B, H, S) tensor.
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    # Apply selective decay.
    scores = _apply_decay_pallas(
        scores,
        log_a_cumsum,
        bh=bh_elem,
        bm=bm_elem,
        bn=bn_elem,
    )

    # Pallas reduction replacing jnp.sum(scores, axis=-1).
    row_sum = _row_sum_pallas(
        scores,
        bh=bh_elem,
        bm=bm_elem,
        bk=bk_reduce,
    )

    # Normalize and cast.
    scores_for_output = _normalize_scores_pallas(
        scores,
        row_sum,
        query.dtype,
        bh=bh_elem,
        bm=bm_elem,
        bn=bn_elem,
    )

    out_dtype = jnp.result_type(scores_for_output.dtype, value.dtype)

    # GEMM 2: output = scores @ value.
    output = _sv_matmul_pallas(
        scores_for_output,
        value,
        bm=bm_sv,
        bk=bk_sv,
        out_dtype=out_dtype,
    )
    return output
''',
score=9.543,
translation_score=None,
hw_feedback=[],
plan_gen_model='deepseek.v3.2',
code_gen_model='gpt-5.4',
stdout='Latency: 9.543 ms\n{"correct": true, "latency": 9.543, "error": "", "all_times_ms": [9.522, 9.524, 9.526, 9.526, 9.528, 9.528, 9.529, 9.53, 9.532, 9.533, 9.533, 9.534, 9.534, 9.535, 9.535, 9.535, 9.535, 9.535, 9.535, 9.536, 9.536, 9.536, 9.537, 9.537, 9.537, 9.537, 9.538, 9.538, 9.539, 9.539, 9.539, 9.539, 9.539, 9.54, 9.54, 9.54, 9.54, 9.541, 9.541, 9.541, 9.542, 9.542, 9.542, 9.542, 9.542, 9.543, 9.543, 9.543, 9.543, 9.543, 9.543, 9.543, 9.543, 9.544, 9.544, 9.544, 9.544, 9.544, 9.545, 9.545, 9.545, 9.546, 9.546, 9.546, 9.547, 9.547, 9.547, 9.547, 9.548, 9.548, 9.548, 9.548, 9.55, 9.55, 9.55, 9.551, 9.552, 9.552, 9.553, 9.553, 9.554, 9.555, 9.556, 9.556, 9.556, 9.557, 9.559, 9.559, 9.559, 9.56, 9.561, 9.562, 9.563, 9.564, 9.566, 9.566, 9.566, 9.568, 9.569, 9.572], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''Plan: Apply **Strategy 4**: "Merge the decay application and row sum stages into a single kernel pass".

### Analysis
The current implementation computes `scores -> decayed_scores -> row_sum` in two separate Pallas kernels:
1.  `_apply_decay_pallas`: Loads `scores` tiles, computes the causal mask and decay factor, and writes the decayed scores to VMEM/HBM.
2.  `_row_sum_pallas`: Reloads the decayed scores tiles from memory, computes the row-wise reduction, and accumulates the result.

This is inefficient due to:
*   **Memory Bandwidth Saturation**: The large intermediate tensor `scores` (Shape: `[1, 64, 2048, 2048]`, ~1GB for f32, ~512MB for bf16) is written out and read back in entirely, saturating HBM bandwidth (~819 GB/s) twice for this stage.
*   **Kernel Launch Overhead**: Launching two kernels incurs fixed runtime overhead.

By fusing these two kernels, we:
1.  Load `scores` once per tile.
2.  Compute the decay factor and mask.
3.  Apply the decay to the scores.
4.  Immediately perform the reduction on the decayed values.
5.  Accumulate the result into the `out_ref`.

This cuts the memory traffic for this stage by approximately 50% and eliminates one kernel launch.

### Implementation Details
We will replace `_apply_decay_pallas` and `_row_sum_pallas` with a single kernel `_decay_and_row_sum_pallas`.

*   **Input**: `scores` (f32), `log_a_cumsum` (f32).
*   **Output**: `row_sum` (f32). Note: The decayed scores are transient and no longer written to global memory. The subsequent `_normalize_scores_pallas` will need to be modified to compute the decay factor again (which is cheap relative to loading the large scores matrix again) or fused later. However, strict adherence to Strategy 4 implies merging *these* two stages.
    *   *Refinement*: To maintain the exact data flow of the original code where `scores_for_output` is required for the final softmax and GEMM, Stage 1 (`_apply_decay_pallas`) *must* write its output if Stage 3 (`_normalize_scores_pallas`) still reads it.
    *   *Correction*: Strategy 4 explicitly says "reading the `scores` tile once... computing... and immediately accumulating". This implies the *output* of this combined stage is the reduction result (`row_sum`). The original code passes the *output* of stage 2 to stage 4.
    *   Wait, the original code uses `scores` (output of stage 1) in **Stage 3** (`_normalize_scores_pallas`) AND **Stage 4** (`_sv_matmul_pallas`).
    *   If we merge Stage 1 and 2, we lose the `decayed_scores` tensor needed for Stages 3 and 4.
    *   To strictly follow Strategy 4 "Merge ... into a single kernel pass", we must either:
        1.  Make the kernel write *both* the decayed scores and the row sum (Still saves the reload for the reduction).
        2.  Re-compute the decay in later stages (requires modifying later kernels).
    *   Given the prompt\'s focus on "identified inefficiencies" and this specific strategy, the most direct interpretation for the hardware (TPU v6e) optimizing this specific sequence is to perform the reduction *as soon as the data is loaded*, which effectively corresponds to fusing the load+decay+sum.
    *   However, since downstream stages *need* the decayed scores, simply discarding them breaks the program.
    *   Therefore, the merged kernel will implement a **Split-K/Reduction** logic:
        *   Grid: `(B, H//bh, S//bm, S//bn_k)` where `bn_k` is the chunk size for the reduction (e.g., `bk`).
        *   This is exactly the grid of `_row_sum_pallas`.
        *   We will load the `scores` and `log_a` slices corresponding to this reduction chunk.
        *   Compute decay, reduce, and accumulate.
        *   Crucially, we still need the full decayed matrix for the next steps. Since we can\'t reuse the "split-K" grid to write the full matrix (it would write multiple overlapping blocks with different partial sums), we must rely on the fact that `_row_sum_pallas` (the consumer of the intermediate) is the one being merged.
    *   Wait, looking at the flow: `scores` -> `[decayed_scores]` -> `row_sum`. The diagram shows `_apply_decay_pallas` producing `scores`. Then `_row_sum_pallas` consumes `scores`.
    *   If we merge them, we simply perform the load, decay, and summation in the loop that was originally `_row_sum_pallas`. This means we don\'t need `_apply_decay_pallas` to run at all, nor do we need its output array.
    *   But Stage 3 (`_normalize`) and Stage 4 (`GEMM`) consume `scores` (which was the output of Stage 1).
    *   If we delete Stage 1\'s output, we break Stage 3 and 4.
    *   Strategy 4 says: "Merge the decay application and row sum stages... reading the scores tile once... reduces total memory traffic". This implies we avoid the *write* of the intermediate matrix.
    *   This implies we must also **Move the decay computation** into Stages 3 and 4, or strictly the strategy allows us to optimize the branch where `row_sum` is computed (assuming we were just interested in the normalization).
    *   *Optimization decision*: I will implement the merged kernel for `row_sum` calculation. I will **assume** the user accepts that `scores_for_output` (which is decayed scores) needs to be regenerated in the later stages or that the merge implies we only care about the reduction efficiency here.
    *   *Better Decision*: The output of `_apply_decay_pallas` is assigned to `scores`. This variable is fed to `_row_sum_pallas`.
    *   I will merge them into `_decay_and_reduce_pallas`.
    *   To keep the code working (`scores` is needed later), I will actually define `_decay_and_reduce_pallas` to return `row_sum`.
    *   For `scores_for_output`, I will assume we can re-compute decay (cheap) in `_normalize_scores_pallas` (which takes the original `scores` anyway). *Correction*: `_normalize_scores_pallas` takes `scores` (which is now the output of QK GEMM, i.e., un-decayed scores).
    *   Ah! In the original code:
        ```python
        # GEMM 1
        scores = _qk_matmul_pallas(...) # Undecayed

        # Decay
        scores = _apply_decay_pallas(scores, log_a_cumsum) # Decayed

        # Sum
        row_sum = _row_sum_pallas(scores) # Reads Decayed

        # Normalize (Reads Decayed, Writes Decayed)
        scores_for_output = _normalize_scores_pallas(scores, row_sum, ...)
        ```
    *   If I merge Decay+Sum, I read Undecayed scores from HBM, compute decay in registers, sum it up, and only write `row_sum`. I **do not** write Decayed Scores to HBM.
    *   This frees up a massive amount of memory traffic.
    *   However, `_normalize_scores_pallas` needs Decayed Scores.
    *   I must modify `_normalize_scores_pallas` to:
        1.  Take `undecayed_scores` (from QK) and `log_a_cumsum`.
        2.  Recompute decay+mask (cheap compute) + normalize.
    *   This adheres strictly to Strategy 4 ("reading the scores tile once from VMEM... computing... accumulating") by eliminating the Round Trip (Write + Read) of the decayed matrix.
    *   Strategy 4 *only* mentions merging decay and row sum. It does not explicitly mention modifying normalize, but it is a logical consequence of removing the intermediate array creation.
    *   Plan:
        1.  Remove `_apply_decay_pallas`.
        2.  Create `_decay_and_sum_pallas` which takes `undecayed_scores` and computes the reduction.
        3.  Modify `_normalize_scores_pallas` to take `undecayed_scores` and compute decay locally.

### Code Changes

1.  **Define `_decay_and_sum_pallas`**:
    *   Grid: `(B, H//bh, S//bm, S//bk)` (Reduction grid).
    *   Loads: `scores` (undecayed), `log_a_cumsum`.
    *   Logic: Load tile -> Compute Decay -> Sum -> Accumulate in SMEM/VMEM scratch.
2.  **Modify `_normalize_scores_pallas`**:
    *   Add `log_a_cumsum` as an input.
    *   Logic: Load `undecayed_scores` tile + `log_a` tile -> Compute Decay -> Normalize -> Store.
3.  **Update `workload`**:
    *   Call `_decay_and_sum_pallas` with the raw output of `_qk_matmul_pallas`.
    *   Call updated `_normalize_scores_pallas` with the raw output of `_qk_matmul_pallas`.
    *   Remove the intermediate `scores` variable allocation (the large one).

This applies Strategy 4 perfectly: the large Decay op is effectively fused into the Sum and Normalize consumers, preventing the materialization of the intermediate huge tensor.

### Constraints Check
*   **v6e TPU**: Used Tiling parameters (bh=8, bm/bn/=128).
*   **BlockSpec**: Last two dims `(bh*bm, bk)`? No, reduction over last axis.
    *   `_decay_and_sum_pallas` input: `(None, bh, bm, bk)`.
        *   v6e constraint: Last two dims divisible by 8 and 128.
        *   `bh` (8) is div by 8. `bm` (128) is div by 128. `bk` (128) is div by 128.
        *   The block shape is effectively `(bh, bm, bk)`.
        *   V6e treats trailing dims as the "tile".
        *   ideally `(B*H*S*S)` is treated as 2D.
        *   I will ensure the mapping respects `(8, 128, x)` -> `(8, 128, y)` where x,y are multiples.
        *   In `_decay_and_sum_pallas`:
            *   `scores_ref`: `block_shape=(None, bh, bm, bk)`.
            *   `index_map`: `(b, hg, i, k)`.
            *   The effective 2D tile is `(bh, bk)`? No.
            *   The block is a slice. The hardware restriction applies to the shape of the block being loaded.
            *   Block shape: `(..., 8, 128, 128)`.
            *   Last 2 dims: `128, 128`. Divisible by `8`? `128` % `8` == 0. Yes.
            *   Wait, the rule is: "last two dimensions ... divisible by 8 and 128 respectively".
            *   Dims: `dim_N-2`, `dim_N-1`.
            *   If I have shape `(X, Y, Z)`, `Y` must be % 8, `Z` must be % 128.
            *   My block shape `(None, 8, 128, 128)`. `(8, 128, 128)`.
            *   Dim N-2: 128. Div by 8? Yes.
            *   Dim N-1: 128. Div by 128? Yes.
            *   This satisfies constraint.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _qk_matmul_pallas(query, key, *, bm=256, bn=256):
    """Blocked QK^T -> scores, accumulated in f32."""
    B, H, S, D = query.shape
    k_tiles = 1  # Full-width D tile for this workload (D=64).

    def kernel(q_ref, k_ref, o_ref, acc_ref):
        rk = pl.program_id(4)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        q = q_ref[...]
        k = k_ref[...]

        update = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H, S // bm, S // bn, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, D),
                index_map=lambda b, h, i, j, k: (b, h, i, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bn, D),
                index_map=lambda b, h, i, j, k: (b, h, j, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, bn),
            index_map=lambda b, h, i, j, k: (b, h, i, j),
        ),
        scratch_shapes=(pltpu.VMEM((bm, bn), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key)


def _decay_and_row_sum_pallas(scores, log_a_cumsum, *, bh=8, bm=128, bk=128):
    """
    Compute decayed scores on-the-fly and reduce to row sum.
    
    Computes:
      sum(scores * exp(where(causal, row_prefix - col_prefix, -inf)), axis=-1)
      
    This merges the decay application and reduction steps to avoid writing
    the large decayed scores matrix to memory.
    """
    B, H, S, _ = scores.shape
    k_tiles = S // bk

    def kernel(scores_ref, row_prefix_ref, col_prefix_ref, out_ref, acc_ref):
        # k is the reduction block index (col index)
        k = pl.program_id(3) 

        @pl.when(k == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        # Load tiles
        # scores: (..., bh, bm, bk)
        # row_prefix: (..., bh, bm)
        # col_prefix: (..., bh, bk) -- Loaded from col_prefix_ref which maps to (b, hg, k)
        
        scores_tile = scores_ref[...].astype(jnp.float32)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)

        # Compute decay: row_prefix[:, :, None] - col_prefix[:, None, :]
        # Shapes: (bh, bm, 1) - (bh, 1, bk) -> (bh, bm, bk)
        diff = row_prefix[:, :, None] - col_prefix[:, None, :]

        # Causal Mask
        # Grid indices: i (row block), k (col block)
        i = pl.program_id(2)
        
        # Global indices
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bk, dtype=jnp.int32)
        global_rows = i * bm + local_rows
        global_cols = k * bk + local_cols
        
        causal_mask = global_rows[:, None] >= global_cols[None, :]
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], (bh, bm, bk))

        decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
        
        # Apply decay and reduce
        decayed_scores = scores_tile * decay
        partial_sum = jnp.sum(decayed_scores, axis=-1) # Reduce over bk (last axis)
        
        acc_ref[...] = acc_ref[...] + partial_sum

        @pl.when(k == k_tiles - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S), jnp.float32),
        grid=(B, H // bh, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec( # Input scores (undecayed)
                block_shape=(None, bh, bm, bk),
                index_map=lambda b, hg, i, k: (b, hg, i, k),
            ),
            pl.BlockSpec( # Row Prefix (log_a_cumsum)
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, k: (b, hg, i),
            ),
            pl.BlockSpec''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _qk_matmul_pallas(query, key, *, bm=256, bn=256):
    """Blocked QK^T -> scores, accumulated in f32."""
    B, H, S, D = query.shape
    k_tiles = 1  # D=64 is used as a full-width tile on this workload.

    def kernel(q_ref, k_ref, o_ref, acc_ref):
        rk = pl.program_id(4)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        q = q_ref[...]
        k = k_ref[...]

        update = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), jnp.float32),
        grid=(B, H, S // bm, S // bn, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, D),
                index_map=lambda b, h, i, j, k: (b, h, i, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bn, D),
                index_map=lambda b, h, i, j, k: (b, h, j, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, bn),
            index_map=lambda b, h, i, j, k: (b, h, i, j),
        ),
        scratch_shapes=(pltpu.VMEM((bm, bn), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
        ),
    )(query, key)


def _decay_and_row_sum_pallas(scores, log_a_cumsum, *, bh=8, bm=128, bk=128):
    """
    Compute row sums of decayed scores without materializing the decayed matrix.

    For each tile this computes:
      decayed = scores * exp(where(causal, row_prefix - col_prefix, -1e30))
      row_sum = sum(decayed, axis=-1)
    """
    B, H, S, _ = scores.shape
    k_tiles = S // bk

    def kernel(scores_ref, row_prefix_ref, col_prefix_ref, out_ref, acc_ref):
        row_blk = pl.program_id(2)
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        scores_tile = scores_ref[...].astype(jnp.float32)      # (bh, bm, bk)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)   # (bh, bm)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)   # (bh, bk)

        diff = row_prefix[:, :, None] - col_prefix[:, None, :]

        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bk, dtype=jnp.int32)
        global_rows = row_blk * bm + local_rows
        global_cols = rk * bk + local_cols

        causal_mask = global_rows[:, None] >= global_cols[None, :]
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], diff.shape)

        decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
        partial = jnp.sum(scores_tile * decay, axis=-1)

        acc_ref[...] = acc_ref[...] + partial

        @pl.when(rk == k_tiles - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S), jnp.float32),
        grid=(B, H // bh, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, bk),
                index_map=lambda b, hg, i, k: (b, hg, i, k),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, k: (b, hg, i),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bk),
                index_map=lambda b, hg, i, k: (b, hg, k),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm),
            index_map=lambda b, hg, i, k: (b, hg, i),
        ),
        scratch_shapes=(pltpu.VMEM((bh, bm), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores, log_a_cumsum, log_a_cumsum)


def _normalize_scores_pallas(
    scores, log_a_cumsum, row_sum, out_dtype, *, bh=8, bm=128, bn=128
):
    """
    Recompute decay on-the-fly, normalize by row_sum, and cast to out_dtype.

    Computes:
      decayed = scores * exp(where(causal, row_prefix - col_prefix, -1e30))
      normalized = decayed / max(|safe_sum|, 1.0)

    where:
      safe_sum = where(|row_sum| < 1e-6, 1.0, row_sum)
    """
    B, H, S, _ = scores.shape

    def kernel(scores_ref, row_prefix_ref, col_prefix_ref, row_sum_ref, out_ref):
        row_blk = pl.program_id(2)
        col_blk = pl.program_id(3)

        scores_tile = scores_ref[...].astype(jnp.float32)      # (bh, bm, bn)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)   # (bh, bm)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)   # (bh, bn)
        row_sum_tile = row_sum_ref[...].astype(jnp.float32)    # (bh, bm)

        diff = row_prefix[:, :, None] - col_prefix[:, None, :]

        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        global_rows = row_blk * bm + local_rows
        global_cols = col_blk * bn + local_cols

        causal_mask = global_rows[:, None] >= global_cols[None, :]
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], diff.shape)

        decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
        decayed_scores = scores_tile * decay

        safe_sum = jnp.where(jnp.abs(row_sum_tile) < 1e-6, 1.0, row_sum_tile)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)

        out_ref[...] = (decayed_scores / den[:, :, None]).astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, S), out_dtype),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, bn),
                index_map=lambda b, hg, i, j: (b, hg, i, j),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, i, j: (b, hg, j),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, bn),
            index_map=lambda b, hg, i, j: (b, hg, i, j),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel"),
        ),
    )(scores, log_a_cumsum, log_a_cumsum, row_sum)


def _sv_matmul_pallas(scores, value, *, bm=256, bk=256, out_dtype=jnp.bfloat16):
    """Blocked scores @ value -> output, accumulated in f32."""
    B, H, S, D = value.shape
    k_tiles = S // bk

    def kernel(s_ref, v_ref, o_ref, acc_ref):
        rk = pl.program_id(3)

        @pl.when(rk == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

        s = s_ref[...]
        v = v_ref[...]

        update = jnp.dot(s, v, preferred_element_type=jnp.float32)
        acc_ref[...] = acc_ref[...] + update

        @pl.when(rk == k_tiles - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(out_dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), out_dtype),
        grid=(B, H, S // bm, k_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, bm, bk),
                index_map=lambda b, h, i, k: (b, h, i, k),
            ),
            pl.BlockSpec(
                block_shape=(None, None, bk, D),
                index_map=lambda b, h, i, k: (b, h, k, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, None, bm, D),
            index_map=lambda b, h, i, k: (b, h, i, 0),
        ),
        scratch_shapes=(pltpu.VMEM((bm, D), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(scores, value)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using TPU Pallas blocked kernels, tuned for v6e-class TPU execution."""
    B, H, S, D = query.shape

    # GEMM tiling.
    bm_qk = 256
    bn_qk = 256
    bm_sv = 256
    bk_sv = 256

    # Elementwise / reduction tiling.
    bh_elem = 8
    bm_elem = 128
    bn_elem = 128
    bk_reduce = 128

    # GEMM 1: raw_scores = query @ key^T, stored in f32.
    raw_scores = _qk_matmul_pallas(query, key, bm=bm_qk, bn=bn_qk)

    # Prefix computation on the smaller (B, H, S) tensor.
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    # Fused decay + row-reduction. This avoids materializing the decayed scores.
    row_sum = _decay_and_row_sum_pallas(
        raw_scores,
        log_a_cumsum,
        bh=bh_elem,
        bm=bm_elem,
        bk=bk_reduce,
    )

    # Recompute decay on-the-fly while normalizing.
    scores_for_output = _normalize_scores_pallas(
        raw_scores,
        log_a_cumsum,
        row_sum,
        query.dtype,
        bh=bh_elem,
        bm=bm_elem,
        bn=bn_elem,
    )

    out_dtype = jnp.result_type(scores_for_output.dtype, value.dtype)

    # GEMM 2: output = normalized_scores @ value.
    output = _sv_matmul_pallas(
        scores_for_output,
        value,
        bm=bm_sv,
        bk=bk_sv,
        out_dtype=out_dtype,
    )
    return output
''',
score=7.8,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-4.7',
code_gen_model='gpt-5.4',
stdout='Latency: 7.800 ms\n{"correct": true, "latency": 7.8, "error": "", "all_times_ms": [7.78, 7.781, 7.782, 7.783, 7.784, 7.784, 7.786, 7.786, 7.786, 7.786, 7.786, 7.788, 7.788, 7.79, 7.79, 7.79, 7.79, 7.79, 7.791, 7.791, 7.791, 7.792, 7.792, 7.792, 7.792, 7.793, 7.793, 7.793, 7.793, 7.794, 7.795, 7.795, 7.796, 7.796, 7.797, 7.797, 7.797, 7.797, 7.797, 7.798, 7.798, 7.798, 7.798, 7.799, 7.799, 7.799, 7.799, 7.8, 7.8, 7.8, 7.8, 7.801, 7.801, 7.801, 7.801, 7.801, 7.801, 7.801, 7.802, 7.802, 7.802, 7.802, 7.803, 7.803, 7.803, 7.804, 7.804, 7.804, 7.804, 7.804, 7.805, 7.806, 7.806, 7.806, 7.807, 7.807, 7.808, 7.809, 7.809, 7.809, 7.809, 7.809, 7.81, 7.81, 7.81, 7.81, 7.811, 7.811, 7.811, 7.812, 7.812, 7.812, 7.814, 7.818, 7.818, 7.819, 7.82, 7.82, 7.825, 7.839], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 1 — cache reused data in local memory instead of reloading from main memory**

### Why this is the best next step on **v6e-1**
The biggest inefficiency in the current code is not the math itself; it is the **HBM traffic from materializing and rereading full score tensors**:

- `raw_scores`: shape `(1, 64, 2048, 2048)`, `f32` → about **1.0 GiB**
- `scores_for_output`: same shape, likely `bf16` → about **0.5 GiB**

Today the program:
1. writes `raw_scores` to HBM,
2. reads `raw_scores` again for row sums,
3. reads `raw_scores` a second time for normalization,
4. writes normalized scores to HBM,
5. reads normalized scores again for `scores @ value`.

That is several GiB of avoidable HBM traffic per call. On a **single-TensorCore v6e-1**, removing those round-trips is a much better phase-2 optimization than trying to add more parallelism.

---

## Concrete plan

Keep the top-level API exactly the same:

```python
@jax.jit
def workload(query, key, value, A_log):
    ...
```

but replace the four helper kernels with **one fused tiled Pallas kernel** that keeps reused state in VMEM across the sweep over the sequence dimension.

### 1) Keep only the prefix precompute outside the kernel
Leave this part in `workload`:

```python
a = jax.nn.sigmoid(A_log.astype(jnp.float32))
log_a = jnp.log(a + 1e-8)
log_a_cumsum = jnp.cumsum(log_a, axis=-1)
```

This tensor is small enough compared to the score matrix and is already the right shape.

---

### 2) Replace `_qk_matmul_pallas`, `_decay_and_row_sum_pallas`, `_normalize_scores_pallas`, and `_sv_matmul_pallas`
Create one helper, e.g. `_ssd_fused_pallas(...)`, that computes the final `(B, H, S, D)` output directly.

### Kernel shape
Use a grid like:

```python
grid = (B, H // bh, S // bm, S // bn)
```

with:

- `bh = 8`
- `bm = 256`
- `bn = 256`

These divide the current shapes and fit comfortably in v6e VMEM.

### Compiler semantics
Use:

```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
)
```

The last axis (`S // bn`) must remain **arbitrary** because it is the reduction axis and the VMEM accumulators must persist across consecutive iterations.

---

### 3) Make the row tile stay resident in VMEM across the reduction
Set the `BlockSpec`s so that for a fixed `(b, hg, row_blk)`, the `query` tile and row-prefix tile do **not** change as `col_blk` advances.

That means:

- `query` block depends on `(b, hg, row_blk)` only
- `row_prefix` block depends on `(b, hg, row_blk)` only
- `key`, `value`, and `col_prefix` depend on `(b, hg, col_blk)`

Concretely:

- `query`: block `(None, bh, bm, D)`, index map `(b, hg, i, j) -> (b, hg, i, 0)`
- `key`: block `(None, bh, bn, D)`, index map `(b, hg, i, j) -> (b, hg, j, 0)`
- `value`: same as `key`
- `row_prefix`: block `(None, bh, bm)`, index map `(b, hg, i, j) -> (b, hg, i)`
- `col_prefix`: block `(None, bh, bn)`, index map `(b, hg, i, j) -> (b, hg, j)`
- output: block `(None, bh, bm, D)`, index map `(b, hg, i, j) -> (b, hg, i, 0)`

Because the last grid axis varies fastest, Pallas can keep the repeated `query` and `row_prefix` slices in VMEM instead of reloading them from HBM on every `col_blk`.

---

### 4) Use two VMEM scratch buffers
Allocate:

- `row_sum_acc`: `pltpu.VMEM((bh, bm), jnp.float32)`
- `out_acc`: `pltpu.VMEM((bh, bm, D), jnp.float32)`

These hold the per-row normalization denominator state and the final numerator accumulation.

---

### 5) Kernel logic
Inside the kernel:

- `col_blk = pl.program_id(3)`

On the first reduction step:

```python
@pl.when(col_blk == 0)
def _init():
    row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
    out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)
```

Then each iteration:

1. **Read** the refs explicitly:
   - `q = q_ref[...]`
   - `k = k_ref[...]`
   - `v = v_ref[...]`
   - `row_prefix = row_prefix_ref[...]`
   - `col_prefix = col_prefix_ref[...]`

2. Compute the score tile in `f32`:
   - per-head batched `q @ k^T`
   - use `jax.lax.dot_general` with head as a batch dimension

3. Form the decay and causal mask:
   - `diff = row_prefix[:, :, None] - col_prefix[:, None, :]`
   - build global row/col indices from `row_blk` and `col_blk`
   - `causal_mask = global_rows[:, None] >= global_cols[None, :]`
   - `decayed = score_tile * exp(where(causal_mask, diff, -1e30))`

4. Update both VMEM accumulators:
   - `row_sum_acc += sum(decayed, axis=-1)`
   - `out_acc += decayed @ v`

On the final reduction step:

```python
@pl.when(col_blk == (S // bn) - 1)
def _store():
    safe_sum = jnp.where(jnp.abs(row_sum_acc_ref[...]) < 1e-6, 1.0, row_sum_acc_ref[...])
    den = jnp.maximum(jnp.abs(safe_sum), 1.0)
    o_ref[...] = (out_acc_ref[...] / den[:, :, None]).astype(query.dtype)
```

---

## Why this is semantically valid
The current code computes:

\[
	ext{output}[r] = \sum_c \left(rac{	ext{decayed}[r,c]}{	ext{den}[r]}
ight) V[c]
\]

Since `den[r]` depends only on the row, this is algebraically:

\[
	ext{output}[r] = rac{\sum_c 	ext{decayed}[r,c] V[c]}{	ext{den}[r]}
\]

So accumulating the numerator in VMEM and dividing once at the end is equivalent up to normal floating-point reordering error, which is within the requested tolerance.

---

## Why this should reduce latency
This change removes the two huge intermediates from HBM:

- no HBM write/read of `raw_scores`
- no HBM write/read of `scores_for_output`

It also reuses the same `query` and row-prefix tiles across the full column sweep from VMEM.

That should cut a large fraction of the current runtime, because the existing implementation is paying for multiple full-sequence score-matrix round-trips through HBM.

---

## Summary of the code change
- **Keep**: `workload(query, key, value, A_log)` signature and prefix precompute.
- **Delete/replace**: the 4 helper kernels with 1 fused helper.
- **Use**: one reduction over column tiles with VMEM-cached row tiles and VMEM accumulators.
- **Target-specific**: single-core v6e-1, no Megacore assumptions.

If you want, I can next turn this plan into the actual rewritten Pallas code.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Fused SSD kernel that computes:
      1. scores = query @ key^T
      2. decayed_scores = scores * exp(causal_decay)
      3. row_sum = sum(decayed_scores, axis=-1)
      4. output = (decayed_scores @ value) / row_sum
    
    All intermediate score matrices stay in VMEM, avoiding HBM round-trips.
    """
    B, H, S, D = query.shape
    n_col_tiles = S // bn

    def kernel(
        q_ref,           # (bh, bm, D)
        k_ref,           # (bh, bn, D)
        v_ref,           # (bh, bn, D)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, bm, D)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, bm, D)
    ):
        row_blk = pl.program_id(2)
        col_blk = pl.program_id(3)

        # Initialize accumulators on first column tile
        @pl.when(col_blk == 0)
        def _init():
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        # Load tiles from VMEM refs
        q = q_ref[...].astype(jnp.float32)              # (bh, bm, D)
        k = k_ref[...].astype(jnp.float32)              # (bh, bn, D)
        v = v_ref[...].astype(jnp.float32)              # (bh, bn, D)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
        col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

        # Compute score tile: q @ k^T for each head in the batch
        # Shape: (bh, bm, D) @ (bh, D, bn) -> (bh, bm, bn)
        score_tile = jax.lax.dot_general(
            q, k,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )

        # Compute decay differences
        diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)

        # Build causal mask using global indices
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        global_rows = row_blk * bm + local_rows
        global_cols = col_blk * bn + local_cols

        causal_mask = global_rows[:, None] >= global_cols[None, :]  # (bm, bn)
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], (bh, bm, bn))

        # Apply causal decay
        decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
        decayed_scores = score_tile * decay  # (bh, bm, bn)

        # Accumulate row sums
        partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
        row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

        # Accumulate output: decayed_scores @ v
        # Shape: (bh, bm, bn) @ (bh, bn, D) -> (bh, bm, D)
        partial_out = jax.lax.dot_general(
            decayed_scores, v,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        out_acc_ref[...] = out_acc_ref[...] + partial_out

        # On final column tile, normalize and store
        @pl.when(col_blk == n_col_tiles - 1)
        def _store():
            row_sum = row_sum_acc_ref[...]  # (bh, bm)
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)
            normalized = out_acc_ref[...] / den[:, :, None]
            o_ref[...] = normalized.astype(query.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), query.dtype),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            # query: reused across all col_blk iterations
            pl.BlockSpec(
                block_shape=(None, bh, bm, D),
                index_map=lambda b, hg, i, j: (b, hg, i, 0),
            ),
            # key: changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, i, j: (b, hg, j, 0),
            ),
            # value: changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, i, j: (b, hg, j, 0),
            ),
            # row_prefix (log_a_cumsum): reused across col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
            # col_prefix (log_a_cumsum): changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, i, j: (b, hg, j),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, D),
            index_map=lambda b, hg, i, j: (b, hg, i, 0),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),      # row_sum_acc
            pltpu.VMEM((bh, bm, D), jnp.float32),   # out_acc
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key, value, log_a_cumsum, log_a_cumsum)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a single fused TPU Pallas kernel to minimize HBM traffic."""
    # Prefix computation on the smaller (B, H, S) tensor - kept outside kernel
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    # Fused kernel: QK^T -> decay -> row_sum -> normalize -> SV matmul
    # All intermediate score matrices stay in VMEM
    output = _ssd_fused_pallas(
        query, key, value, log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.899,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.899 ms\n{"correct": true, "latency": 0.899, "error": "", "all_times_ms": [0.891, 0.891, 0.891, 0.892, 0.892, 0.893, 0.893, 0.893, 0.893, 0.893, 0.893, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.895, 0.895, 0.895, 0.895, 0.895, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.897, 0.897, 0.897, 0.897, 0.897, 0.897, 0.897, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.901, 0.901, 0.901, 0.902, 0.902, 0.902, 0.902, 0.902, 0.903, 0.903, 0.903, 0.903, 0.903, 0.904, 0.905, 0.905, 0.905, 0.905, 0.907, 0.908, 0.908, 0.908, 0.909, 0.909, 0.91, 0.91, 0.911, 0.911, 0.911, 0.912, 0.916], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''**Selected strategy: 6. simplify or remove unnecessary code**

### Why this is the best single change here
Your kernel evaluates a full **8 × 8 tile grid** over `(row_blk, col_blk)` for each `(B, H-group)`. But because the operator is causal:

- **upper-triangular tiles** (`col_blk > row_blk`) contribute **exactly zero**
- **strictly lower-triangular tiles** (`col_blk < row_blk`) are **fully valid**, so they do **not need a causal mask**
- only **diagonal tiles** (`col_blk == row_blk`) need the expensive per-element causal mask

Right now the kernel still does all of this on every tile:
- `q @ k^T`
- mask index construction
- `jnp.where(...)`
- `exp(...)`
- reduction
- `decayed_scores @ v`

even for tiles that are known ahead of time to be all-zero or all-valid.

On **v6e-1**, that is wasted MXU/VPU work. With `S=2048`, `bm=bn=256`, there are 64 tile pairs total, but only 36 are in the causal lower triangle. So about **44% of tile iterations are useless compute** today.

---

## Plan

### 1) Keep the public API exactly the same
Do **not** change:
- `workload(query, key, value, A_log)`
- `_ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256)`

Do **not** change the grid shape or block sizes in this phase:
- `grid=(B, H // bh, S // bm, S // bn)`
- `bh=8, bm=256, bn=256`

These block sizes are already legal for TPU:
- last two block dims divisible by **8** and **128**
- reduction axis `col_blk` remains the innermost grid axis, so accumulator persistence stays correct

---

### 2) Specialize the kernel into 3 tile cases
Inside the kernel, compute:
```python
row_blk = pl.program_id(2)
col_blk = pl.program_id(3)
```

Then split behavior by tile relation:

#### Case A: `col_blk > row_blk`  → upper triangle
This tile is fully masked out by causality.

**Change:** make this a no-op compute tile:
- do not form `score_tile`
- do not form `diff`
- do not do `exp`
- do not update `row_sum_acc_ref`
- do not update `out_acc_ref`

Still keep:
- accumulator init when `col_blk == 0`
- final store when `col_blk == n_col_tiles - 1`

That preserves semantics because adding zero was the original effect.

---

#### Case B: `col_blk < row_blk`  → strictly lower triangle
This tile is fully causal-valid.

**Change:** remove all causal mask construction for this case.

Instead of:
```python
global_rows = ...
global_cols = ...
causal_mask = ...
decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
```

use:
```python
decay = jnp.exp(diff)
```

Then continue with:
- `partial_row_sum = jnp.sum(decayed_scores, axis=-1)`
- `partial_out = dot_general(decayed_scores, v, ...)`

This is mathematically identical for a fully valid tile.

---

#### Case C: `col_blk == row_blk`  → diagonal tile
Only here do you need the triangular mask.

Keep the existing masked logic, but restrict it to the diagonal case only.

That means the costly:
- `jnp.arange`
- global row/col index arithmetic
- causal compare
- masked `where`

runs only on **8 diagonal tiles**, instead of all **64** tiles.

---

### 3) Structure the kernel with `pl.when`
Use `pl.when(...)` to gate the three cases while keeping the kernel legal for TPU Refs.

Sketch:
```python
@pl.when(col_blk == 0)
def _init():
    row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
    out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

@pl.when(col_blk < row_blk)
def _lower():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    row_prefix = row_prefix_ref[...].astype(jnp.float32)
    col_prefix = col_prefix_ref[...].astype(jnp.float32)

    score_tile = ...
    diff = row_prefix[:, :, None] - col_prefix[:, None, :]
    decayed_scores = score_tile * jnp.exp(diff)
    row_sum_acc_ref[...] = row_sum_acc_ref[...] + jnp.sum(decayed_scores, axis=-1)
    out_acc_ref[...] = out_acc_ref[...] + ...

@pl.when(col_blk == row_blk)
def _diag():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    row_prefix = row_prefix_ref[...].astype(jnp.float32)
    col_prefix = col_prefix_ref[...].astype(jnp.float32)

    score_tile = ...
    diff = ...
    # build diagonal causal mask only here
    decayed_scores = ...
    row_sum_acc_ref[...] = ...
    out_acc_ref[...] = ...

@pl.when(col_blk == n_col_tiles - 1)
def _store():
    row_sum = row_sum_acc_ref[...]
    safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
    den = jnp.maximum(jnp.abs(safe_sum), 1.0)
    o_ref[...] = (out_acc_ref[...] / den[:, :, None]).astype(query.dtype)
```

Upper-triangle tiles are simply the fallthrough case: no compute branch runs.

---

## Why this should help
This single change removes unnecessary work in the hottest part of the kernel:

- skips **both matmuls** on upper-triangle tiles
- skips `exp`, `sum`, and accumulator updates on upper-triangle tiles
- removes mask construction entirely on strictly lower tiles
- reduces diagonal-mask construction from **64 tiles** to **8 tiles**

For your current configuration:
- total tile pairs per row/col grid: `8 * 8 = 64`
- useful causal tiles: `8 * 9 / 2 = 36`
- useless tiles today: `28`

So this plan targets a large fraction of the current latency without changing the interface or memory layout.

---

## Correctness / TPU safety
This remains valid on **TPU v6e-1** because:

- block shapes stay unchanged and legal: `(bh, bm, D)` and `(bh, bn, D)` with `bm=bn=256`
- reduction axis stays the last grid dimension (`col_blk`), so accumulator lifetime is still correct
- all Ref reads/writes remain explicit via `ref[...]`
- all elementwise math still happens in `float32`
- semantics are unchanged except for standard floating-point tolerance

---

## What to measure after this phase
Benchmark the same `workload(...)` end-to-end and compare against the original latency `0.899`.

Validation:
- compare outputs with the original implementation
- use `max_abs_err` / `allclose` with bf16-appropriate tolerances

If you want, I can next turn this plan into the actual rewritten code while keeping the same function names and signatures.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Fused SSD kernel that computes:
      1. scores = query @ key^T
      2. decayed_scores = scores * exp(causal_decay)
      3. row_sum = sum(decayed_scores, axis=-1)
      4. output = (decayed_scores @ value) / row_sum
    
    All intermediate score matrices stay in VMEM, avoiding HBM round-trips.
    
    Optimized: skips upper-triangle tiles entirely, removes causal mask
    construction for strictly lower-triangle tiles.
    """
    B, H, S, D = query.shape
    n_col_tiles = S // bn

    def kernel(
        q_ref,           # (bh, bm, D)
        k_ref,           # (bh, bn, D)
        v_ref,           # (bh, bn, D)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, bm, D)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, bm, D)
    ):
        row_blk = pl.program_id(2)
        col_blk = pl.program_id(3)

        # Initialize accumulators on first column tile
        @pl.when(col_blk == 0)
        def _init():
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        # Case B: col_blk < row_blk -> strictly lower triangle (fully valid, no mask needed)
        @pl.when(col_blk < row_blk)
        def _lower():
            # Load tiles from VMEM refs
            q = q_ref[...].astype(jnp.float32)              # (bh, bm, D)
            k = k_ref[...].astype(jnp.float32)              # (bh, bn, D)
            v = v_ref[...].astype(jnp.float32)              # (bh, bn, D)
            row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

            # Compute score tile: q @ k^T for each head in the batch
            # Shape: (bh, bm, D) @ (bh, D, bn) -> (bh, bm, bn)
            score_tile = jax.lax.dot_general(
                q, k,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )

            # Compute decay differences - no mask needed for lower triangle
            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            decay = jnp.exp(diff)
            decayed_scores = score_tile * decay  # (bh, bm, bn)

            # Accumulate row sums
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Accumulate output: decayed_scores @ v
            # Shape: (bh, bm, bn) @ (bh, bn, D) -> (bh, bm, D)
            partial_out = jax.lax.dot_general(
                decayed_scores, v,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        # Case C: col_blk == row_blk -> diagonal tile (needs causal mask)
        @pl.when(col_blk == row_blk)
        def _diag():
            # Load tiles from VMEM refs
            q = q_ref[...].astype(jnp.float32)              # (bh, bm, D)
            k = k_ref[...].astype(jnp.float32)              # (bh, bn, D)
            v = v_ref[...].astype(jnp.float32)              # (bh, bn, D)
            row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

            # Compute score tile: q @ k^T for each head in the batch
            score_tile = jax.lax.dot_general(
                q, k,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )

            # Compute decay differences
            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)

            # Build causal mask using local indices only (since row_blk == col_blk)
            local_rows = jnp.arange(bm, dtype=jnp.int32)
            local_cols = jnp.arange(bn, dtype=jnp.int32)
            causal_mask = local_rows[:, None] >= local_cols[None, :]  # (bm, bn)
            causal_mask = jnp.broadcast_to(causal_mask[None, :, :], (bh, bm, bn))

            # Apply causal decay
            decay = jnp.exp(jnp.where(causal_mask, diff, -1e30))
            decayed_scores = score_tile * decay  # (bh, bm, bn)

            # Accumulate row sums
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Accumulate output: decayed_scores @ v
            partial_out = jax.lax.dot_general(
                decayed_scores, v,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        # Case A: col_blk > row_blk -> upper triangle (no-op, contributes zero)
        # No explicit code needed - the pl.when guards above handle this by not executing

        # On final column tile, normalize and store
        @pl.when(col_blk == n_col_tiles - 1)
        def _store():
            row_sum = row_sum_acc_ref[...]  # (bh, bm)
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)
            normalized = out_acc_ref[...] / den[:, :, None]
            o_ref[...] = normalized.astype(query.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), query.dtype),
        grid=(B, H // bh, S // bm, S // bn),
        in_specs=[
            # query: reused across all col_blk iterations
            pl.BlockSpec(
                block_shape=(None, bh, bm, D),
                index_map=lambda b, hg, i, j: (b, hg, i, 0),
            ),
            # key: changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, i, j: (b, hg, j, 0),
            ),
            # value: changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, i, j: (b, hg, j, 0),
            ),
            # row_prefix (log_a_cumsum): reused across col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, i, j: (b, hg, i),
            ),
            # col_prefix (log_a_cumsum): changes with col_blk
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, i, j: (b, hg, j),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, D),
            index_map=lambda b, hg, i, j: (b, hg, i, 0),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),      # row_sum_acc
            pltpu.VMEM((bh, bm, D), jnp.float32),   # out_acc
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(query, key, value, log_a_cumsum, log_a_cumsum)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a single fused TPU Pallas kernel to minimize HBM traffic."""
    # Prefix computation on the smaller (B, H, S) tensor - kept outside kernel
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    # Fused kernel: QK^T -> decay -> row_sum -> normalize -> SV matmul
    # All intermediate score matrices stay in VMEM
    output = _ssd_fused_pallas(
        query, key, value, log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.755,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.755 ms\n{"correct": true, "latency": 0.755, "error": "", "all_times_ms": [0.741, 0.742, 0.742, 0.743, 0.744, 0.744, 0.744, 0.745, 0.746, 0.746, 0.746, 0.747, 0.748, 0.748, 0.748, 0.748, 0.748, 0.748, 0.748, 0.748, 0.749, 0.749, 0.749, 0.749, 0.749, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.751, 0.751, 0.751, 0.751, 0.752, 0.752, 0.752, 0.752, 0.752, 0.753, 0.753, 0.753, 0.753, 0.753, 0.754, 0.754, 0.755, 0.755, 0.755, 0.756, 0.757, 0.757, 0.757, 0.758, 0.758, 0.758, 0.759, 0.759, 0.76, 0.761, 0.761, 0.761, 0.762, 0.762, 0.762, 0.762, 0.762, 0.763, 0.763, 0.763, 0.763, 0.763, 0.763, 0.763, 0.765, 0.765, 0.765, 0.766, 0.766, 0.766, 0.766, 0.767, 0.767, 0.769, 0.77, 0.77, 0.771, 0.772, 0.772, 0.772, 0.772, 0.773, 0.773, 0.773, 0.785, 0.798], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''## Performance Analysis

Looking at this SSD (State Space Duality) kernel, I\'ll analyze the strategies and identify the most impactful optimization.

### Current Bottlenecks

1. **Upper triangle iterations are wasteful**: The kernel iterates over all `(row_blk, col_blk)` pairs but does nothing when `col_blk > row_blk`. With `S=2048` and `bn=256`, we have 8 column tiles per row, meaning roughly half the kernel invocations are wasted (upper triangle).

2. **Redundant decay computation**: The decay factor `exp(row_prefix[:, :, None] - col_prefix[:, None, :])` is computed identically for all heads within each `(row_blk, col_blk)` pair, but the current structure computes it per-head-group.

3. **Sequential column iteration**: The `"arbitrary"` dimension on the last axis forces sequential processing of column tiles.

### Selected Strategy: Triangular Iteration Pattern (Strategy 12)

**Strategy 12: Use triangular iteration pattern with dynamic grid bounds to reduce total kernel invocations from O(n²) to O(n²/2)**

This is highly applicable because:
- With `S=2048` and block size `256`, we have `8×8=64` `(row, col)` tile pairs
- Currently, ~28 of these pairs are upper triangle (no-op)
- Plus 8 diagonal tiles need full processing
- Only ~28 lower triangle tiles need full processing without masking

### Implementation Plan

**Approach**: Instead of a 4D grid `(B, H//bh, S//bm, S//bn)` where we skip upper triangle tiles at runtime, we\'ll use a **linearized triangular grid** that only iterates over valid `(row_blk, col_blk)` pairs where `col_blk <= row_blk`.

**Key Changes**:

1. **Compute triangular tile count**: For `n_tiles = S // bn = 8` row/col tiles, the number of valid pairs is `n_tiles * (n_tiles + 1) // 2 = 36` (lower triangle + diagonal).

2. **Linearize the grid**: Replace the last two grid dimensions `(S//bm, S//bn)` with a single dimension of size `n_tiles * (n_tiles + 1) // 2`.

3. **Inverse mapping**: In the kernel, convert the linear tile index back to `(row_blk, col_blk)` using the triangular number formula:
   - `row_blk = floor((sqrt(8*idx + 1) - 1) / 2)`
   - `col_blk = idx - row_blk * (row_blk + 1) / 2`

4. **Modify BlockSpec index_maps**: The index maps must compute the correct row/col block indices from the linearized tile index.

5. **Handle reduction across columns**: Since we\'re now iterating over column tiles non-contiguously (all columns for row 0, then all columns for row 1, etc.), the accumulation logic remains the same—we initialize on the first column tile for each row and write on the last.

**Performance Benefit**:
- Reduces kernel invocations from 64 to 36 per `(batch, head_group, row)` combination (44% reduction)
- Eliminates the runtime `pl.when` checks for upper triangle since those tiles are never visited
- Reduces total grid size from `1 × 8 × 8 × 8 = 512` to `1 × 8 × 36 = 288` invocations

**Additional Optimization**: Since we iterate row by row through the lower triangle, consecutive iterations for the same row will reuse the same query and row_prefix blocks (already in VMEM from the BlockSpec caching). The iteration order is: `(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...` which naturally groups by increasing row index.

**VMEM Budget Check**:
- q_ref: `(bh, bm, D) = (8, 256, 64)` @ bf16 = 256 KB
- k_ref, v_ref: same = 256 KB each
- row_prefix, col_prefix: `(bh, bm)` @ f32 = 8 KB each
- row_sum_acc: `(bh, bm)` @ f32 = 8 KB
- out_acc: `(bh, bm, D)` @ f32 = 512 KB
- Total: ~1.3 MB << 16 MB ✓''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _triangular_row_col(tri_idx):
    tri_idx = jnp.asarray(tri_idx, dtype=jnp.int32)
    tri_f = tri_idx.astype(jnp.float32)
    row_blk = jnp.floor((jnp.sqrt(8.0 * tri_f + 1.0) - 1.0) * 0.5).astype(jnp.int32)
    row_start = (row_blk * (row_blk + 1)) // 2
    col_blk = tri_idx - row_start
    return row_blk, col_blk


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Fused SSD kernel with triangular iteration over valid (row_blk, col_blk) pairs only.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This triangular SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This triangular SSD kernel requires S//bm == S//bn.")

    n_tri_tiles = n_row_tiles * (n_row_tiles + 1) // 2

    def kernel(
        q_ref,           # (bh, bm, D)
        k_ref,           # (bh, bn, D)
        v_ref,           # (bh, bn, D)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, bm, D)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, bm, D)
    ):
        tri_blk = pl.program_id(2)
        row_blk, col_blk = _triangular_row_col(tri_blk)

        def _load_common():
            q = q_ref[...].astype(jnp.float32)
            k = k_ref[...].astype(jnp.float32)
            v = v_ref[...].astype(jnp.float32)
            row_prefix = row_prefix_ref[...].astype(jnp.float32)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)

            score_tile = jax.lax.dot_general(
                q,
                k,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            diff = row_prefix[:, :, None] - col_prefix[:, None, :]
            return score_tile, diff, v

        def _accumulate(decayed_scores, v):
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            partial_out = jax.lax.dot_general(
                decayed_scores,
                v,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        @pl.when(col_blk == 0)
        def _init():
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        @pl.when(col_blk < row_blk)
        def _lower():
            score_tile, diff, v = _load_common()
            decayed_scores = score_tile * jnp.exp(diff)
            _accumulate(decayed_scores, v)

        @pl.when(col_blk == row_blk)
        def _diag_and_store():
            score_tile, diff, v = _load_common()

            local_rows = jnp.arange(bm, dtype=jnp.int32)
            local_cols = jnp.arange(bn, dtype=jnp.int32)
            causal_mask = local_rows[:, None] >= local_cols[None, :]

            decayed_scores = jnp.where(
                causal_mask[None, :, :],
                score_tile * jnp.exp(diff),
                jnp.zeros_like(score_tile),
            )
            _accumulate(decayed_scores, v)

            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)
            normalized = out_acc_ref[...] / den[:, :, None]
            o_ref[...] = normalized.astype(query.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), query.dtype),
        grid=(B, H // bh, n_tri_tiles),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, bm, D),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[0], 0),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[1], 0),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bn, D),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[1], 0),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[0]),
            ),
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[1]),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, bm, D),
            index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[0], 0),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, bm, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query, key, value, log_a_cumsum, log_a_cumsum)


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a single fused TPU Pallas kernel to minimize HBM traffic."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.648,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='gpt-5.4',
stdout='Latency: 0.648 ms\n{"correct": true, "latency": 0.648, "error": "", "all_times_ms": [0.643, 0.643, 0.644, 0.644, 0.644, 0.644, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.645, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.646, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.648, 0.649, 0.649, 0.649, 0.649, 0.649, 0.65, 0.65, 0.65, 0.65, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.653, 0.653, 0.653, 0.653, 0.653, 0.653, 0.654, 0.654, 0.655, 0.655, 0.655, 0.656, 0.656, 0.657, 0.657, 0.661, 0.662, 0.664, 0.669], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''## Analysis of Current Code

Looking at the current implementation, I can identify several aspects:

1. **Triangular iteration pattern**: The kernel iterates over `n_tri_tiles = n_row_tiles * (n_row_tiles + 1) // 2` tiles, processing only the lower triangular portion of the attention matrix.

2. **Block shapes**: `bh=8`, `bm=256`, `bn=256`, `D=64`. The blocks are `(bh, bm, D) = (8, 256, 64)` and `(bh, bn, D) = (8, 256, 64)`.

3. **Current inefficiency**: The `row_prefix` and `col_prefix` computations compute `diff = row_prefix[:, :, None] - col_prefix[:, None, :]` which creates a `(bh, bm, bn) = (8, 256, 256)` tensor. This computation is repeated in both `_lower()` and `_diag_and_store()` branches.

4. **Memory layout**: Looking at the last two dimensions of the blocks:
   - `(bh, bm, D) = (8, 256, 64)` → last two dims are (256, 64). 256 is divisible by 8, but 64 is not divisible by 128.
   - `(bh, bn) = (8, 256)` → last dim is 256, divisible by 128. ✓

## Selected Strategy: Strategy 5 - Align block shapes so last two dimensions are multiples of 8 and 128 respectively

The current block shape has `D=64` as the last dimension, which is not divisible by 128. This causes padding overhead and suboptimal memory access patterns.

## Plan

**Problem**: The current kernel uses blocks with shape `(bh, bm, D) = (8, 256, 64)` where `D=64` is the last dimension. Since 64 is not divisible by 128, the TPU must pad this to 128, wasting 50% of the memory bandwidth and compute resources for operations involving this dimension.

**Solution**: Restructure the block layout by:

1. **Merge the head dimension into the batch dimension** at the call site, so we process `(B*H, S, D)` arrays instead of `(B, H, S, D)`.

2. **Adjust BlockSpecs** to have the last two dimensions as `(S_block, D)` where we can control the padding better. Since `D=64`, we\'ll process multiple heads together in a way that the effective last dimension becomes 128 (by processing 2 heads together, giving `2*D=128`).

3. **Specifically**: Instead of blocking as `(bh, bm, D)` with `bh=8, bm=256, D=64`, we can use a different layout:
   - Process with blocks of shape `(bm, bh*D)` = `(256, 512)` by treating multiple heads\' data together
   - Or use `(bm, 128)` blocks by processing pairs of heads (`2*D=128`)

4. **Concrete change**: 
   - Reshape inputs from `(B, H, S, D)` to `(B*H, S, D)` 
   - Use `bm=256`, and process with block shape `(bm, D)` but adjust the head grouping so the effective last dimension is 128 (group 2 heads: `D*2 = 128`)
   - Grid becomes `(B, H//2, n_tri_tiles)` with blocks `(2, bm, D)` which has last two dims `(256, 64)` → still problematic

**Better approach**: Transpose the data layout so D is not the last dimension:
   - Work with transposed layout `(B, H, D, S)` so blocks have shape `(bh, D, bm)` = `(8, 64, 256)`, giving last two dims `(64, 256)`. Here 64 is divisible by 8 and 256 is divisible by 128. ✓

5. **Implementation**:
   - Transpose Q, K, V from `(B, H, S, D)` to `(B, H, D, S)` before the kernel
   - Adjust all BlockSpecs to work with this transposed layout
   - Update the `dot_general` dimension specifications accordingly
   - Transpose output back to `(B, H, S, D)` after the kernel

This alignment ensures optimal memory access patterns and eliminates padding waste for the main data arrays.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _triangular_row_col(tri_idx):
    tri_idx = jnp.asarray(tri_idx, dtype=jnp.int32)
    tri_f = tri_idx.astype(jnp.float32)
    row_blk = jnp.floor((jnp.sqrt(8.0 * tri_f + 1.0) - 1.0) * 0.5).astype(jnp.int32)
    row_start = (row_blk * (row_blk + 1)) // 2
    col_blk = tri_idx - row_start
    return row_blk, col_blk


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Fused SSD kernel with triangular iteration over valid (row_blk, col_blk) pairs only.
    Uses transposed layout (B, H, D, S) for better memory alignment on TPU.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This triangular SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This triangular SSD kernel requires S//bm == S//bn.")

    n_tri_tiles = n_row_tiles * (n_row_tiles + 1) // 2

    # Transpose inputs from (B, H, S, D) to (B, H, D, S) for better alignment
    # This makes the last two dims (D, S) where D=64 (div by 8) and S blocks are 256 (div by 128)
    query_t = jnp.transpose(query, (0, 1, 3, 2))  # (B, H, D, S)
    key_t = jnp.transpose(key, (0, 1, 3, 2))      # (B, H, D, S)
    value_t = jnp.transpose(value, (0, 1, 3, 2))  # (B, H, D, S)

    def kernel(
        q_ref,           # (bh, D, bm)
        k_ref,           # (bh, D, bn)
        v_ref,           # (bh, D, bn)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, D, bm)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, D, bm)
    ):
        tri_blk = pl.program_id(2)
        row_blk, col_blk = _triangular_row_col(tri_blk)

        def _load_common():
            # Load and convert to float32 for computation
            q = q_ref[...].astype(jnp.float32)  # (bh, D, bm)
            k = k_ref[...].astype(jnp.float32)  # (bh, D, bn)
            v = v_ref[...].astype(jnp.float32)  # (bh, D, bn)
            row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

            # Compute score: q^T @ k -> (bh, bm, bn)
            # q is (bh, D, bm), k is (bh, D, bn)
            # We want (bh, bm, bn) = (bh, bm, D) @ (bh, D, bn) in original layout
            # With transposed layout: q^T is (bh, bm, D), so we need:
            # score[h, m, n] = sum_d q[h, d, m] * k[h, d, n]
            # This is a batched matmul: q.transpose(0,2,1) @ k for each h
            # Using dot_general: contract over D (axis 1 of q and k), batch over h (axis 0)
            score_tile = jax.lax.dot_general(
                q,
                k,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            return score_tile, diff, v

        def _accumulate(decayed_scores, v):
            # decayed_scores: (bh, bm, bn)
            # v: (bh, D, bn)
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Compute partial_out: (bh, D, bm)
            # partial_out[h, d, m] = sum_n decayed_scores[h, m, n] * v[h, d, n]
            # This is: decayed_scores.transpose(0,2,1) @ v.transpose(0,2,1) then transpose back
            # Or: v @ decayed_scores^T for each h
            # Using dot_general: contract over n (axis 2 of decayed_scores, axis 2 of v), batch over h
            partial_out = jax.lax.dot_general(
                v,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        @pl.when(col_blk == 0)
        def _init():
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        @pl.when(col_blk < row_blk)
        def _lower():
            score_tile, diff, v = _load_common()
            decayed_scores = score_tile * jnp.exp(diff)
            _accumulate(decayed_scores, v)

        @pl.when(col_blk == row_blk)
        def _diag_and_store():
            score_tile, diff, v = _load_common()

            local_rows = jnp.arange(bm, dtype=jnp.int32)
            local_cols = jnp.arange(bn, dtype=jnp.int32)
            causal_mask = local_rows[:, None] >= local_cols[None, :]

            decayed_scores = jnp.where(
                causal_mask[None, :, :],
                score_tile * jnp.exp(diff),
                jnp.zeros_like(score_tile),
            )
            _accumulate(decayed_scores, v)

            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)
            # out_acc_ref: (bh, D, bm), den: (bh, bm)
            normalized = out_acc_ref[...] / den[:, None, :]
            o_ref[...] = normalized.astype(query.dtype)

    # Output shape in transposed layout
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_tri_tiles),
        in_specs=[
            # q_ref: (bh, D, bm) from query_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[0]),
            ),
            # k_ref: (bh, D, bn) from key_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bn),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[1]),
            ),
            # v_ref: (bh, D, bn) from value_t (B, H, D, S)
            pl.BlockSpec(
                block_shape=(None, bh, D, bn),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[1]),
            ),
            # row_prefix_ref: (bh, bm) from log_a_cumsum (B, H, S)
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[0]),
            ),
            # col_prefix_ref: (bh, bn) from log_a_cumsum (B, H, S)
            pl.BlockSpec(
                block_shape=(None, bh, bn),
                index_map=lambda b, hg, tri: (b, hg, _triangular_row_col(tri)[1]),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, D, bm),
            index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[0]),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)

    # Transpose output back from (B, H, D, S) to (B, H, S, D)
    output = jnp.transpose(out_t, (0, 1, 3, 2))
    return output


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a single fused TPU Pallas kernel to minimize HBM traffic."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.542,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.542 ms\n{"correct": true, "latency": 0.542, "error": "", "all_times_ms": [0.53, 0.532, 0.533, 0.534, 0.534, 0.534, 0.534, 0.534, 0.535, 0.535, 0.535, 0.535, 0.535, 0.535, 0.535, 0.536, 0.536, 0.536, 0.536, 0.537, 0.537, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.539, 0.539, 0.539, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.541, 0.541, 0.541, 0.541, 0.541, 0.541, 0.542, 0.542, 0.542, 0.542, 0.542, 0.542, 0.542, 0.542, 0.542, 0.542, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.544, 0.544, 0.544, 0.544, 0.544, 0.545, 0.545, 0.545, 0.545, 0.546, 0.547, 0.547, 0.548, 0.548, 0.548, 0.549, 0.549, 0.55, 0.55, 0.55, 0.55, 0.551, 0.551, 0.551, 0.552, 0.552, 0.553, 0.557, 0.558, 0.558, 0.573], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''

Looking at this SSD kernel code, I can identify a major inefficiency: **the query block and row_prefix are being reloaded from HBM for every triangular tile iteration**, even though they remain constant for a given row block as we iterate over column blocks.

Currently, the grid has `n_tri_tiles = n_row_tiles * (n_row_tiles + 1) // 2` iterations, where each iteration loads ALL inputs freshly. For a given `row_blk`, as we process `col_blk = 0, 1, ..., row_blk`, the query and row_prefix data is identical but gets reloaded each time.

Let me apply **Strategy 13**: Change ownership so one program owns a `row_blk` and loops over `col_blk <= row_blk` internally, keeping `q`, `row_prefix`, `row_sum`, and `out_acc` resident while only streaming `k/v/col_prefix`.

Here\'s the optimized plan:

1. **Change grid from triangular to row-based**: `(B, H // bh, n_row_tiles)` instead of `(B, H // bh, n_tri_tiles)`
2. **Keep q and row_prefix in registers/VMEM**: Load once per row_blk, reuse for all col_blk iterations
3. **Add inner loop in kernel**: Iterate `col_blk` from 0 to `row_blk` inside the kernel
4. **Stream k, v, col_prefix**: These change each iteration and must be reloaded
5. **Remove triangular index computation**: Now we have direct `row_blk` from program_id
6. **Simplify control flow**: Remove `pl.when` guards for triangular checks since the inner loop naturally handles valid pairs

The key insight is that the current implementation wastes HBM bandwidth reloading the same query and prefix data row_blk times. By restructuring to load once per row block and iterate internally, we achieve better memory efficiency.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Optimized SSD kernel: each program processes one row_blk and iterates over col_blk internally.
    Keeps q, row_prefix, row_sum, out_acc resident while streaming k/v/col_prefix.
    Uses transposed layout (B, H, D, S) for better memory alignment on TPU.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This SSD kernel requires S//bm == S//bn.")

    # Transpose inputs from (B, H, S, D) to (B, H, D, S) for better alignment
    query_t = jnp.transpose(query, (0, 1, 3, 2))  # (B, H, D, S)
    key_t = jnp.transpose(key, (0, 1, 3, 2))      # (B, H, D, S)
    value_t = jnp.transpose(value, (0, 1, 3, 2))  # (B, H, D, S)

    def kernel(
        q_ref,           # (bh, D, bm)
        k_ref,           # (bh, D, bn)
        v_ref,           # (bh, D, bn)
        row_prefix_ref,  # (bh, bm)
        col_prefix_ref,  # (bh, bn)
        o_ref,           # (bh, D, bm)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, D, bm)
    ):
        # Row block is now directly from program_id - no triangular computation needed
        row_blk = pl.program_id(2)
        
        # Initialize accumulators for this row block
        row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
        out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        # Load query and row_prefix ONCE and keep resident for all col_blk iterations
        # These don\'t change as we iterate over col_blk
        q = q_ref[...].astype(jnp.float32)  # (bh, D, bm)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)

        # Inner loop: iterate over col_blk = 0 to row_blk
        # We use a Python loop - Pallas will manage the iteration efficiently
        for col_blk in range(row_blk + 1):
            # Load k, v, col_prefix for current col_blk
            # These change each iteration, so must be reloaded
            k = k_ref[...].astype(jnp.float32)  # (bh, D, bn)
            v = v_ref[...].astype(jnp.float32)  # (bh, D, bn)
            col_prefix = col_prefix_ref[...].astype(jnp.float32)  # (bh, bn)

            # Compute score: q^T @ k -> (bh, bm, bn)
            # q is (bh, D, bm), k is (bh, D, bn)
            # contract over D (axis 1), batch over h (axis 0)
            score_tile = jax.lax.dot_general(
                q,
                k,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            
            # Check if this is the diagonal block
            is_diag = (col_blk == row_blk)
            
            if is_diag:
                # Diagonal block: apply causal mask
                local_rows = jnp.arange(bm, dtype=jnp.int32)
                local_cols = jnp.arange(bn, dtype=jnp.int32)
                causal_mask = local_rows[:, None] >= local_cols[None, :]
                
                decayed_scores = jnp.where(
                    causal_mask[None, :, :],
                    score_tile * jnp.exp(diff),
                    jnp.zeros_like(score_tile),
                )
            else:
                # Lower triangular block: all pairs valid
                decayed_scores = score_tile * jnp.exp(diff)

            # Accumulate row sums
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Compute partial_out and accumulate
            # partial_out[h, d, m] = sum_n decayed_scores[h, m, n] * v[h, d, n]
            partial_out = jax.lax.dot_general(
                v,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        # Normalize and store output
        row_sum = row_sum_acc_ref[...]
        safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)
        # out_acc_ref: (bh, D, bm), den: (bh, bm)
        normalized = out_acc_ref[...] / den[:, None, :]
        o_ref[...] = normalized.astype(query.dtype)

    # Output shape in transposed layout
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_row_tiles),  # Changed from n_tri_tiles to n_row_tiles
        in_specs=[
            # q_ref: (bh, D, bm) from query_t (B, H, D, S)
            # Loaded once per row_blk, stays resident for all col_blk iterations
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
            ),
            # k_ref: (bh, D, bn) - streamed for each col_blk
            # We need a sub-grid for col_blk, but BlockSpec doesn\'t support dynamic iteration
            # So we use a trick: make k_ref dimension larger and index into it
            # Actually, we need to pass col_blk somehow - let\'s handle this differently
            # The issue is BlockSpec is static - we need the kernel to iterate over col_blk
            # For now, we\'ll use the fact that the kernel receives the full block and we
            # need to slice appropriately. But wait - the kernel receives a single block
            # defined by BlockSpec, so we need a different approach...
            
            # Actually, looking at the original code more carefully:
            # The index_map for k/v uses _triangular_row_col(tri)[1] = col_blk
            # This means each program loads a different col_blk
            # 
            # In the new design, each program processes ALL col_blk for a given row_blk
            # So the BlockSpec for k/v should load ALL columns? That\'s too much data.
            #
            # Let me reconsider... We need to iterate over col_blk INSIDE the kernel
            # But we still need to load different k/v blocks for each col_blk
            #
            # The solution: use a 1D block for k/v (load all columns at once)
            # and slice inside the kernel. But this wastes memory bandwidth.
            #
            # Alternative: Keep the grid as (B, H//bh, n_row_tiles) but have the kernel
            # do the inner iteration. For each col_blk iteration, we need to load
            # the appropriate k/v/col_prefix slice. We can do this by:
            # 1. Loading the full k/v for the row (all columns) - expensive
            # 2. Using async copies inside the loop - complex
            # 3. Accepting that we reload k/v each time - still better than original
            #    because q/row_prefix are not reloaded
            #
            # Actually, the best approach is to keep grid=(B, H//bh, n_row_tiles) and
            # have the kernel iterate. For k/v loading, we can use the fact that
            # we have multiple buffer slots and can prefetch. But that\'s complex.
            #
            # Let me rethink: the key optimization is NOT reloading q and row_prefix.
            # We can still use the triangular grid but fix the index_map to avoid
            # recomputing triangular indices.
            #
            # Wait - the issue is that with grid=(B, H//bh, n_row_tiles),
            # we need to load different k/v blocks inside the loop.
            # The cleanest way is to use the original grid structure but:
            # 1. Keep q and row_prefix in accumulators (already inVMEM from previous iteration)
            # 2. Only reload k, v, col_prefix
            #
            # But this requires the compiler to recognize that q/row_prefix don\'t change...
            #
            # Let me try a different approach: Keep the triangular grid but
            # use a smem buffer to cache q and row_prefix across iterations.
        ],
        out_specs=...,
        ...
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)
```

Wait, I need to reconsider. The challenge with the proposed approach is that `pallas_call` with `BlockSpec` defines a static data movement pattern - each program gets fixed slices. To have a program iterate over multiple column blocks while keeping query resident requires a different approach.

Let me use a hybrid approach: keep the triangular grid but use SMEM to cache query and row_prefix, and rely on the compiler\'s double-buffering to avoid reloading them when possible. Actually, a cleaner solution is to restructure the grid to iterate over row blocks only, and use explicit indexing in the kernel to load different k/v slices. Let me revise:

The cleanest optimization that preserves the pallas model is:
1. Use grid = (B, H//bh, n_row_tiles) 
2. For k/v, use a larger block that spans all columns, and slice inside
3.OR keep triangular grid but note that consecutive iterations with same row_blk will benefit from VMEM caching

Actually, looking back at the reference patterns, the real win is to change the granularity. Let me implement a version that uses the row-based grid and loads k/v for each col_blk inside the kernel using a sub-block approach:

```python
# Revised kernel with row-based grid and inner col_blk loop
def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    B, H, S, D = query.shape
    # ... validation ...
    
    n_row_tiles = S // bm
    
    # Transpose
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    # Kernel with inner loop over col_blk
    def kernel(
        q_ref, k_ref, v_ref, row_prefix_ref, col_prefix_ref, o_ref,
        row_sum_acc_ref, out_acc_ref
    ):
        row_blk = pl.program_id(2)
        
        # Initialize
        ...
        
        # Load q ONCE
        q = q_ref[...].astype(jnp.float32)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)
        
        # For k, v, col_prefix - we need to iterate over col_blk
        # The trick: we load blocks from k_ref which has shape (bh, D, S)
        # and slice for each col_blk
        # But k_ref as passed has shape (bh, D, bn) from BlockSpec...
        
        # Actually, we CANNOT change the block shape dynamically inside kernel
        # The BlockSpec determines what data is loaded into the Ref
        
        # So the only viable approach with pallas model is:
        # Keep triangular grid, but recognize that q/row_prefix stay in VMEM
        # across iterations with same row_blk due to consecutive access pattern
        
    # Let\'s implement THIS version - it\'s semantically equivalent but
    # we note the optimization opportunity for the compiler
    
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_row_tiles * (n_row_tiles + 1) // 2),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, tri: (b, hg, 0, _triangular_row_col(tri)[0]),
            ),
            ...
        ],
        ...
    )
```

The issue is fundamental: with `BlockSpec`, each program gets a predetermined slice. To have one program iterate over multiple slices requires either (a) loading all data at once (wasteful) or (b) nested pipeline (complex).

Let me take a different, simpler optimization: **Strategy 12** - Keep tensors in transposed layout end-to-end. Currently the code transposes at start and end. If the producer (caller) already provides transposed data and consumer expects transposed, we save 3 transposes + 1 reshape. But we can\'t change the API.

Let me focus on a different, high-impact optimization: **Strategy 1 + 13 combined** - Use SMEM to cache indices and eliminate triangular computation, plus restructure to avoid redundant loads.

Actually, the simplest effective optimization is to **combine Strategy 13 with proper memory management**: Change to row-based grid and manually load k/v slices using pl.load with dynamic slicing inside the kernel:

```python
def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    ...
    n_row_tiles = S // bm
    
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    def kernel(
        q_ref,           # (bh, D, bm)
        k_full_ref,      # (bh, D, S) - full sequence for this head group
        v_full_ref,      # (bh, D, S)
        row_prefix_ref,  # (bh, bm)
        col_prefix_full_ref,  # (bh, S)
        o_ref,           # (bh, D, bm)
        row_sum_acc_ref, # scratch: (bh, bm)
        out_acc_ref,     # scratch: (bh, D, bm)
    ):
        row_blk = pl.program_id(2)
        
        # Initialize
        row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
        out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)
        
        # Load q and row_prefix ONCE
        q = q_ref[...].astype(jnp.float32)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)
        
        # Load full k, v, col_prefix once
        k_full = k_full_ref[...].astype(jnp.float32)  # (bh, D, S)
        v_full = v_full_ref[...].astype(jnp.float32)
        col_prefix_full = col_prefix_full_ref[...].astype(jnp.float32)  # (bh, S)
        
        # Iterate over col_blk
        for col_blk in range(row_blk + 1):
            col_start = col_blk * bn
            
            # Slice k, v, col_prefix for this col_blk
            k = pl.load(k_full_ref, (slice(None), slice(col_start, col_start + bn))).astype(jnp.float32)
            v = pl.load(v_full_ref, (slice(None), slice(col_start, col_start + bn))).astype(jnp.float32)
            col_prefix = pl.load(col_prefix_full_ref, (slice(None), slice(col_start, col_start + bn))).astype(jnp.float32)
            
            # Compute...
            ...
    
    # Grid is now (B, H//bh, n_row_tiles) - much simpler!
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_row_tiles),
        in_specs=[
            # q: single block per row
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
            ),
            # k: full sequence for this head group - loaded once per row_blk
            pl.BlockSpec(
                block_shape=(None, bh, D, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0, 0),
            ),
            # v: full sequence
            pl.BlockSpec(
                block_shape=(None, bh, D, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0, 0),
            ),
            # row_prefix: single block
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, row_blk: (b, hg, row_blk),
            ),
            # col_prefix_full: full sequence
            pl.BlockSpec(
                block_shape=(None, bh, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, D, bm),
            index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)
    
    output = jnp.transpose(out_t, (0, 1, 3, 2))
    return output
```

This is significantly better:
1. Grid reduced from n_tri_tiles (~2M) to n_row_tiles (~8)
2. q and row_prefix loaded ONCE per row block
3. k, v, col_prefix loaded once per row block, then sliced inside kernel
4. No triangular index computation
5. Inner loop is simple and compiler-friendly

The tradeoff is we load more data initially (full S instead of bn per tile), but we avoid the massive overhead of the original triangular grid with its repeated loads.

Wait, there\'s still an issue: the inner loop uses `pl.load` for slicing, but this may not be the most efficient. Let me adjust to use array slicing with Refs which is more idiomatic:

Actually, looking at the ref patterns, I should use `ref.at[slice]` for slicing refs. But the block is already loaded into the ref, so I can just use array indexing on the loaded data.

This is the right approach. Let me finalize the code:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Optimized SSD kernel: each program owns one row_blk and loops over col_blk internally.
    
    Key optimizations over triangular grid:
    1. Grid reduced from n_tri_tiles to n_row_tiles (8x fewer programs)
    2. Query and row_prefix loaded ONCE per row block, kept resident across col_blk iterations
    3. k, v, col_prefix loaded once per row block, sliced inside kernel
    4. No triangular index computation needed
    5. Simple inner loop without branch misprediction
    
    Uses transposed layout (B, H, D, S) for TPU alignment.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This SSD kernel requires S//bm == S//bn.")

    # Transpose inputs from (B, H, S, D) to (B, H, D, S) for better alignment
    query_t = jnp.transpose(query, (0, 1, 3, 2))  # (B, H, D, S)
    key_t = jnp.transpose(key, (0, 1, 3, 2))      # (B, H, D, S)
    value_t = jnp.transpose(value, (0, 1, 3, 2))  # (B, H, D, S)

    def kernel(
        q_ref,               # (bh, D, bm) - query block for this row
        k_full_ref,          # (bh, D, S) - full key sequence for this head group
        v_full_ref,          # (bh, D, S) - full value sequence for this head group
        row_prefix_ref,      # (bh, bm) - prefix sum for this row block
        col_prefix_full_ref, # (bh, S) - full prefix sum sequence
        o_ref,               # (bh, D, bm) - output block
        row_sum_acc_ref,     # scratch: (bh, bm) - row sum accumulator
        out_acc_ref,         # scratch: (bh, D, bm) - output accumulator
    ):
        row_blk = pl.program_id(2)
        
        # Initialize accumulators for this row block
        row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
        out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        # Load query and row_prefix ONCE - they stay constant across all col_blk iterations
        q = q_ref[...].astype(jnp.float32)  # (bh, D, bm)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)

        # Load full k, v, col_prefix once per row block
        k_full = k_full_ref[...].astype(jnp.float32)  # (bh, D, S)
        v_full = v_full_ref[...].astype(jnp.float32)  # (bh, D, S)
        col_prefix_full = col_prefix_full_ref[...].astype(jnp.float32)  # (bh, S)

        # Inner loop: iterate over col_blk = 0 to row_blk
        # For each iteration, slice k, v, col_prefix appropriately
        for col_blk in range(row_blk + 1):
            col_start = col_blk * bn
            
            # Slice k, v, col_prefix for this column block
            # Using array indexing on the loaded data (no DMA needed, just register remapping)
            k = k_full[:, :, col_start:col_start + bn]  # (bh, D, bn)
            v = v_full[:, :, col_start:col_start + bn]  # (bh, D, bn)
            col_prefix = col_prefix_full[:, col_start:col_start + bn]  # (bh, bn)

            # Compute score: q^T @ k -> (bh, bm, bn)
            # Contract over D (axis 1), batch over h (axis 0)
            score_tile = jax.lax.dot_general(
                q,
                k,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            
            # Determine if this is diagonal block (requires causal mask)
            is_diag = (col_blk == row_blk)
            
            if is_diag:
                # Diagonal block: apply causal mask
                # Create causal mask: rows >= cols within block
                local_rows = jnp.arange(bm, dtype=jnp.int32)
                local_cols = jnp.arange(bn, dtype=jnp.int32)
                # Broadcast to (bm, bn)
                causal_mask = local_rows[:, None] >= local_cols[None, :]
                
                decayed_scores = jnp.where(
                    causal_mask[None, :, :],
                    score_tile * jnp.exp(diff),
                    jnp.zeros_like(score_tile),
                )
            else:
                # Lower triangular block: all (m, n) pairs are valid
                decayed_scores = score_tile * jnp.exp(diff)

            # Accumulate row sums for normalization
            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            # Compute partial output and accumulate
            # partial_out[h, d, m] = sum_n decayed_scores[h, m, n] * v[h, d, n]
            partial_out = jax.lax.dot_general(
                v,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        # Normalize by row sums and store output
        row_sum = row_sum_acc_ref[...]
        # Avoid division by zero
        safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)  # (bh, bm)
        # Broadcast den across D dimension: den[:, None, :] for (bh, D, bm)
        normalized = out_acc_ref[...] / den[:, None, :]
        o_ref[...] = normalized.astype(query.dtype)

    # Output in transposed layout (B, H, D, S)
    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        # Changed from triangular grid (n_row_tiles * (n_row_tiles + 1) // 2)
        # to row-based grid (n_row_tiles) - ~8x fewer program invocations
        grid=(B, H // bh, n_row_tiles),
        in_specs=[
            # q_ref: (bh, D, bm) - single query block per row_blk
            # Loaded once, stays resident for all col_blk iterations
            pl.BlockSpec(
                block_shape=(None, bh, D, bm),
                index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
            ),
            # k_full_ref: (bh, D, S) - full key sequence for this head group
            # Loaded once per row_blk, sliced inside kernel for each col_blk
            pl.BlockSpec(
                block_shape=(None, bh, D, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0, 0),
            ),
            # v_full_ref: (bh, D, S) - full value sequence
            pl.BlockSpec(
                block_shape=(None, bh, D, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0, 0),
            ),
            # row_prefix_ref: (bh, bm) - prefix sum for this row block
            pl.BlockSpec(
                block_shape=(None, bh, bm),
                index_map=lambda b, hg, row_blk: (b, hg, row_blk),
            ),
            # col_prefix_full_ref: (bh, S) - full prefix sum sequence
            pl.BlockSpec(
                block_shape=(None, bh, S),
                index_map=lambda b, hg, row_blk: (b, hg, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, bh, D, bm),
            index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
        ),
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)

    # Transpose output back from (B, H, D, S) to (B, H, S, D)
    output = jnp.transpose(out_t, (0, 1, 3, 2))
    return output


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using optimized row-based kernel."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
```

This implementation:

1. **Reduces grid size**: From `n_row_tiles * (n_row_tiles + 1) // 2` (~2M for S=2048, bm=256) to `n_row_tiles` (~8). This dramatically reduces pipeline overhead.

2. **Eliminates redundant loads**: Query and row_prefix are loaded ONCE per row block instead of once per triangular tile. Since each row block previously loaded the same query ~row_blk times, this is a ~4x reduction in query HBM traffic for typical row_blk values.

3. **Simplifies indexing**: No need to compute triangular indices; row_blk comes directly from program_id.

4. **Better memory access pattern**: Loading full k/v sequences once and slicing in registers avoids the scattered access pattern of the triangular grid.

The key semantic difference: the original had ~2M program invocations with minimal per-invocation work (just one tile multiply). The new version has ~8 invocations with substantial per-invocation work (looping over col_blk). This moves toward compute-bound operation and away from the memory-bound pipeline overhead regime.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Row-owned SSD kernel for TPU v6e.

    Each program owns one output row block and loops over all valid column blocks
    internally. This keeps q and row_prefix resident while streaming/slicing K/V
    and prefix data from a head-group-wide VMEM tile.

    Input / output API matches the original function.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")
    if (S // bm) != (S // bn):
        raise ValueError("This SSD kernel requires S//bm == S//bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn

    # TPU-friendly layout: (B, H, D, S)
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    q_spec = pl.BlockSpec(
        block_shape=(None, bh, D, bm),
        index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
        pipeline_mode=pl.Buffered(buffer_count=1),
    )
    # Load full sequence for K/V once per head-group slice; row_blk is ignored in index_map.
    # With sequential row_blk traversal, repeated identical slices can stay resident.
    kv_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg, row_blk: (b, hg, 0, 0),
        pipeline_mode=pl.Buffered(buffer_count=1),
    )
    row_prefix_spec = pl.BlockSpec(
        block_shape=(None, bh, bm),
        index_map=lambda b, hg, row_blk: (b, hg, row_blk),
        pipeline_mode=pl.Buffered(buffer_count=1),
    )
    col_prefix_full_spec = pl.BlockSpec(
        block_shape=(None, bh, S),
        index_map=lambda b, hg, row_blk: (b, hg, 0),
        pipeline_mode=pl.Buffered(buffer_count=1),
    )
    out_spec = pl.BlockSpec(
        block_shape=(None, bh, D, bm),
        index_map=lambda b, hg, row_blk: (b, hg, 0, row_blk),
        pipeline_mode=pl.Buffered(buffer_count=1),
    )

    def kernel(
        q_ref,               # (bh, D, bm)
        k_full_ref,          # (bh, D, S)
        v_full_ref,          # (bh, D, S)
        row_prefix_ref,      # (bh, bm)
        col_prefix_full_ref, # (bh, S)
        o_ref,               # (bh, D, bm)
        row_sum_acc_ref,     # scratch: (bh, bm) f32
        out_acc_ref,         # scratch: (bh, D, bm) f32
    ):
        row_blk = pl.program_id(2)

        row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
        out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

        # Load once per owned row block.
        q = q_ref[...].astype(jnp.float32)                 # (bh, D, bm)
        row_prefix = row_prefix_ref[...].astype(jnp.float32)  # (bh, bm)

        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        causal_mask = local_rows[:, None] >= local_cols[None, :]  # (bm, bn)

        # Small static trip count; TPU will unroll this.
        for col_blk in range(n_col_tiles):

            @pl.when(row_blk >= col_blk)
            def _do_col():
                col_start = col_blk * bn
                col_stop = col_start + bn

                # Slice from VMEM-resident full-sequence refs.
                k = k_full_ref[:, :, col_start:col_stop].astype(jnp.float32)          # (bh, D, bn)
                v = v_full_ref[:, :, col_start:col_stop].astype(jnp.float32)          # (bh, D, bn)
                col_prefix = col_prefix_full_ref[:, col_start:col_stop].astype(jnp.float32)  # (bh, bn)

                # score[h, m, n] = sum_d q[h, d, m] * k[h, d, n]
                score_tile = jax.lax.dot_general(
                    q,
                    k,
                    dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )  # (bh, bm, bn)

                diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
                base_scores = score_tile * jnp.exp(diff)

                is_diag = row_blk == jnp.asarray(col_blk, dtype=jnp.int32)
                decayed_scores = jnp.where(
                    is_diag,
                    jnp.where(causal_mask[None, :, :], base_scores, jnp.zeros_like(base_scores)),
                    base_scores,
                )

                partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
                row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

                # partial_out[h, d, m] = sum_n v[h, d, n] * decayed_scores[h, m, n]
                partial_out = jax.lax.dot_general(
                    v,
                    decayed_scores,
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )  # (bh, D, bm)
                out_acc_ref[...] = out_acc_ref[...] + partial_out

        row_sum = row_sum_acc_ref[...]
        safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
        den = jnp.maximum(jnp.abs(safe_sum), 1.0)  # (bh, bm)

        normalized = out_acc_ref[...] / den[:, None, :]
        o_ref[...] = normalized.astype(query.dtype)

    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh, n_row_tiles),
        in_specs=[
            q_spec,
            kv_full_spec,
            kv_full_spec,
            row_prefix_spec,
            col_prefix_full_spec,
        ],
        out_specs=out_spec,
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum, log_a_cumsum)

    return jnp.transpose(out_t, (0, 1, 3, 2))


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a row-owned fused TPU Pallas kernel."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.558,
translation_score=None,
hw_feedback=[],
plan_gen_model='minimax.minimax-m2.5',
code_gen_model='gpt-5.4',
stdout='Latency: 0.558 ms\n{"correct": true, "latency": 0.558, "error": "", "all_times_ms": [0.551, 0.551, 0.553, 0.553, 0.553, 0.553, 0.554, 0.554, 0.554, 0.554, 0.554, 0.554, 0.554, 0.554, 0.555, 0.555, 0.555, 0.555, 0.555, 0.555, 0.555, 0.555, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.557, 0.558, 0.558, 0.558, 0.558, 0.558, 0.558, 0.558, 0.558, 0.558, 0.558, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.562, 0.564, 0.565, 0.565, 0.565, 0.565, 0.565, 0.565, 0.567, 0.567, 0.57, 0.571, 0.576, 0.576, 0.579, 0.594], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''## Analysis of Current Code

Looking at the current implementation, I can identify several key characteristics:

1. **Grid structure**: `(B, H // bh, n_row_tiles)` where each program owns one row block
2. **Memory access pattern**: K/V/prefix are loaded with full-sequence BlockSpecs `(bh, D, S)` per head-group
3. **Inner loop**: Iterates over `col_blk` from 0 to `n_col_tiles`

The current design relies on the Pallas runtime to recognize that consecutive `row_blk` programs access the same K/V/prefix data and keep it resident. However, this cross-program reuse is not guaranteed and depends on scheduler behavior.

## Selected Strategy: Strategy 8

**"Change ownership from \'one program per row tile\' to \'one program per (batch, head-group)\' and loop over row tiles inside that program, so full-sequence K/V/prefix data stays live in VMEM across all rows instead of depending on cross-program reuse."**

## Plan

### Rationale
The current code has `n_row_tiles` (2048/256 = 8) programs per (batch, head-group) pair. Each program loads the full K/V/prefix data for its head-group. While the BlockSpec uses `buffer_count=1` and the index_map ignores `row_blk`, the actual reuse depends on the runtime scheduler keeping data resident across consecutive grid invocations.

By restructuring to have **one program per (batch, head-group)** that loops over all row tiles internally, we guarantee that:
1. K/V/prefix data is loaded once and stays in VMEM for all row iterations
2. Query blocks and output blocks are streamed in/out per row iteration
3. No dependence on cross-program memory reuse heuristics

### Implementation Changes

1. **Change grid from `(B, H//bh, n_row_tiles)` to `(B, H//bh)`** - remove the row_blk dimension from grid

2. **Modify BlockSpecs**:
   - Q and output: Load/store entire sequence `(bh, D, S)` once, slice internally per row
   - K/V/prefix: Keep as full-sequence loads `(bh, D, S)` and `(bh, S)`
   - Row prefix: Load entire `(bh, S)` instead of `(bh, bm)` per row block

3. **Add outer loop over row tiles inside kernel**:
   - Loop `for row_blk in range(n_row_tiles):` 
   - Slice Q from `q_full_ref[:, :, row_blk*bm:(row_blk+1)*bm]`
   - Slice row_prefix from `row_prefix_full_ref[:, row_blk*bm:(row_blk+1)*bm]`
   - Write output to `o_full_ref[:, :, row_blk*bm:(row_blk+1)*bm]`

4. **Adjust scratch shapes**:
   - Keep `row_sum_acc_ref` as `(bh, bm)` - reset per row iteration
   - Keep `out_acc_ref` as `(bh, D, bm)` - reset per row iteration

5. **Update dimension_semantics**:
   - Change to `("parallel", "parallel")` for the two remaining grid dimensions

### Memory Budget Verification
For `bh=8, D=64, S=2048, bm=bn=256`:
- Q full: `(8, 64, 2048) * 2B = 2MB` bf16
- K full: `(8, 64, 2048) * 2B = 2MB` bf16  
- V full: `(8, 64, 2048) * 2B = 2MB` bf16
- Output full: `(8, 64, 2048) * 2B = 2MB` bf16
- Row prefix full: `(8, 2048) * 4B = 64KB` f32
- Col prefix full: `(8, 2048) * 4B = 64KB` f32
- Scratch row_sum: `(8, 256) * 4B = 8KB` f32
- Scratch out_acc: `(8, 64, 256) * 4B = 512KB` f32

Total: ~8.6MB, well within 16MB VMEM limit.

### Benefits
- Eliminates uncertainty about cross-program K/V reuse
- Reduces HBM bandwidth by guaranteeing single load of K/V per (batch, head-group)
- Simpler memory access pattern for the compiler to optimize
- Row iterations within a single program are guaranteed sequential''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Row-owned SSD kernel for TPU v6e.

    Restructured so that each program owns one (batch, head-group) and loops
    over all row tiles internally. This guarantees K/V/prefix data stays
    resident in VMEM across all row iterations without depending on
    cross-program reuse heuristics.
    """
    B, H, S, D = query.shape

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn

    # TPU-friendly layout: (B, H, D, S)
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    # Each program loads full sequence data for its (batch, head-group)
    # Grid is now (B, H // bh) - one program per (batch, head-group)
    q_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    k_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    v_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    prefix_full_spec = pl.BlockSpec(
        block_shape=(None, bh, S),
        index_map=lambda b, hg: (b, hg, 0),
    )
    out_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )

    def kernel(
        q_full_ref,          # (bh, D, S)
        k_full_ref,          # (bh, D, S)
        v_full_ref,          # (bh, D, S)
        prefix_full_ref,     # (bh, S)
        o_full_ref,          # (bh, D, S)
        row_sum_acc_ref,     # scratch: (bh, bm) f32
        out_acc_ref,         # scratch: (bh, D, bm) f32
    ):
        # Precompute causal mask for diagonal blocks
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        causal_mask = local_rows[:, None] >= local_cols[None, :]  # (bm, bn)

        # Loop over row tiles - this keeps K/V/prefix resident in VMEM
        for row_blk in range(n_row_tiles):
            row_start = row_blk * bm
            row_stop = row_start + bm

            # Reset accumulators for this row tile
            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

            # Load Q for this row block
            q = q_full_ref[:, :, row_start:row_stop].astype(jnp.float32)  # (bh, D, bm)
            row_prefix = prefix_full_ref[:, row_start:row_stop].astype(jnp.float32)  # (bh, bm)

            # Loop over column tiles
            for col_blk in range(n_col_tiles):
                # Only process lower triangular blocks (causal)
                @pl.when(row_blk >= col_blk)
                def _do_col():
                    col_start = col_blk * bn
                    col_stop = col_start + bn

                    # Slice K/V/prefix from VMEM-resident full-sequence refs
                    k = k_full_ref[:, :, col_start:col_stop].astype(jnp.float32)  # (bh, D, bn)
                    v = v_full_ref[:, :, col_start:col_stop].astype(jnp.float32)  # (bh, D, bn)
                    col_prefix = prefix_full_ref[:, col_start:col_stop].astype(jnp.float32)  # (bh, bn)

                    # score[h, m, n] = sum_d q[h, d, m] * k[h, d, n]
                    score_tile = jax.lax.dot_general(
                        q,
                        k,
                        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )  # (bh, bm, bn)

                    # Compute decay: exp(row_prefix - col_prefix)
                    diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
                    base_scores = score_tile * jnp.exp(diff)

                    # Apply causal mask for diagonal blocks
                    is_diag = row_blk == col_blk
                    decayed_scores = jnp.where(
                        is_diag,
                        jnp.where(causal_mask[None, :, :], base_scores, jnp.zeros_like(base_scores)),
                        base_scores,
                    )

                    # Accumulate row sum
                    partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
                    row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

                    # partial_out[h, d, m] = sum_n v[h, d, n] * decayed_scores[h, m, n]
                    partial_out = jax.lax.dot_general(
                        v,
                        decayed_scores,
                        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )  # (bh, D, bm)
                    out_acc_ref[...] = out_acc_ref[...] + partial_out

            # Normalize and write output for this row tile
            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)  # (bh, bm)

            normalized = out_acc_ref[...] / den[:, None, :]
            o_full_ref[:, :, row_start:row_stop] = normalized.astype(query.dtype)

    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), query.dtype),
        grid=(B, H // bh),
        in_specs=[
            q_full_spec,
            k_full_spec,
            v_full_spec,
            prefix_full_spec,
        ],
        out_specs=out_full_spec,
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum)

    return jnp.transpose(out_t, (0, 1, 3, 2))


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a row-owned fused TPU Pallas kernel."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.379,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 0.379 ms\n{"correct": true, "latency": 0.379, "error": "", "all_times_ms": [0.373, 0.373, 0.374, 0.374, 0.374, 0.374, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.376, 0.376, 0.376, 0.376, 0.376, 0.376, 0.376, 0.376, 0.376, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.379, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.381, 0.381, 0.381, 0.381, 0.381, 0.381, 0.381, 0.382, 0.382, 0.382, 0.383, 0.383, 0.383, 0.383, 0.383, 0.384, 0.384, 0.384, 0.384, 0.385, 0.385, 0.386, 0.387, 0.388, 0.389, 0.391, 0.392, 0.393, 0.397, 0.405, 0.406], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr=''),
plan='''## Analysis of Inefficiencies

Looking at the code, I can identify several performance bottlenecks:

1. **Strategy 10 (Split diagonal from off-diagonal tiles)** is highly relevant here. The current code uses `pl.when(row_blk >= col_blk)` and then inside that, applies a conditional causal mask: `jnp.where(is_diag, jnp.where(causal_mask[None, :, :], base_scores, jnp.zeros_like(base_scores)), base_scores)`. This means:
   - Every off-diagonal (strictly lower triangular) tile still evaluates the `is_diag` condition and the nested `jnp.where`
   - The causal mask computation and application happens even when it\'s not needed for off-diagonal blocks
   - This adds unnecessary branching and computation overhead

2. The kernel structure already has good properties (row-owned design, VMEM scratch for f32 accumulation), but the mixed handling of diagonal vs off-diagonal tiles is inefficient.

## Plan: Apply Strategy 10 - Split Diagonal from Off-Diagonal Tiles

### What to Change

I will restructure the inner column loop to explicitly separate the processing of:
1. **Strictly-lower triangular blocks** (`col_blk < row_blk`): These need no causal masking - all elements are valid
2. **Diagonal blocks** (`col_blk == row_blk`): These need the causal mask applied

### Specific Changes

1. **Replace the single column loop** that uses `pl.when(row_blk >= col_blk)` with conditional `jnp.where` inside, with **two separate processing paths**:

2. **First path - Off-diagonal blocks** (`col_blk` from 0 to `row_blk - 1`):
   - Remove the `is_diag` check and the `jnp.where(causal_mask...)` entirely
   - Compute `base_scores = score_tile * jnp.exp(diff)` directly without any masking
   - This eliminates the conditional overhead for the majority of blocks in lower rows

3. **Second path - Diagonal block** (only when `col_blk == row_blk`):
   - Apply the causal mask directly: `decayed_scores = jnp.where(causal_mask[None, :, :], base_scores, jnp.zeros_like(base_scores))`
   - No need for the `is_diag` check since we know this is the diagonal

4. **Implementation approach**:
   - Use a Python `for` loop (which gets unrolled) for off-diagonal blocks from `0` to `row_blk`
   - Since `row_blk` is a runtime value in the outer Python loop over `n_row_tiles`, we can structure this as:
     - Process off-diagonal blocks: `for col_blk in range(row_blk):` - no masking needed
     - Process diagonal block separately: `col_blk = row_blk` - apply causal mask

### Benefits

1. **Off-diagonal blocks** (which are the majority for lower rows) completely skip:
   - The `is_diag` comparison
   - The outer `jnp.where` evaluation
   - The causal mask lookup/application
   
2. **Diagonal blocks** have cleaner code with direct mask application

3. This optimization is especially impactful because:
   - For a sequence with `n_row_tiles` row tiles, there are `n_row_tiles * (n_row_tiles + 1) / 2` total blocks processed
   - Only `n_row_tiles` of these are diagonal blocks
   - For `n_row_tiles = 8` (with S=2048, bm=256), that\'s 36 total blocks but only 8 are diagonal
   - ~78% of blocks benefit from the simplified off-diagonal path''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG["batch"], CONFIG["seq_len"]
    H, D = CONFIG["num_heads"], CONFIG["head_dim"]

    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_fused_pallas(query, key, value, log_a_cumsum, *, bh=8, bm=256, bn=256):
    """
    Row-owned SSD kernel for TPU v6e.

    This version explicitly splits strictly-lower-triangular tiles from diagonal
    tiles. Off-diagonal tiles avoid the causal-mask path entirely, while the
    diagonal tile applies the mask directly.
    """
    B, H, S, D = query.shape
    out_dtype = query.dtype

    if H % bh != 0:
        raise ValueError(f"H={H} must be divisible by bh={bh}.")
    if S % bm != 0 or S % bn != 0:
        raise ValueError(f"S={S} must be divisible by bm={bm} and bn={bn}.")
    if bm != bn:
        raise ValueError("This SSD kernel requires bm == bn.")

    n_row_tiles = S // bm
    n_col_tiles = S // bn
    if n_row_tiles != n_col_tiles:
        raise ValueError("This SSD kernel assumes bm == bn, so row/col tile counts must match.")

    # TPU-friendly layout: (B, H, D, S)
    query_t = jnp.transpose(query, (0, 1, 3, 2))
    key_t = jnp.transpose(key, (0, 1, 3, 2))
    value_t = jnp.transpose(value, (0, 1, 3, 2))

    q_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    k_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    v_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )
    prefix_full_spec = pl.BlockSpec(
        block_shape=(None, bh, S),
        index_map=lambda b, hg: (b, hg, 0),
    )
    out_full_spec = pl.BlockSpec(
        block_shape=(None, bh, D, S),
        index_map=lambda b, hg: (b, hg, 0, 0),
    )

    def kernel(
        q_full_ref,       # (bh, D, S)
        k_full_ref,       # (bh, D, S)
        v_full_ref,       # (bh, D, S)
        prefix_full_ref,  # (bh, S)
        o_full_ref,       # (bh, D, S)
        row_sum_acc_ref,  # scratch: (bh, bm) f32
        out_acc_ref,      # scratch: (bh, D, bm) f32
    ):
        local_rows = jnp.arange(bm, dtype=jnp.int32)
        local_cols = jnp.arange(bn, dtype=jnp.int32)
        causal_mask = local_rows[:, None] >= local_cols[None, :]  # (bm, bn)

        def accumulate_block(q_block, row_prefix, col_blk, apply_causal_mask):
            col_start = col_blk * bn
            col_stop = col_start + bn

            k_block = k_full_ref[:, :, col_start:col_stop].astype(jnp.float32)      # (bh, D, bn)
            v_block = v_full_ref[:, :, col_start:col_stop].astype(jnp.float32)      # (bh, D, bn)
            col_prefix = prefix_full_ref[:, col_start:col_stop].astype(jnp.float32) # (bh, bn)

            score_tile = jax.lax.dot_general(
                q_block,
                k_block,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, bm, bn)

            diff = row_prefix[:, :, None] - col_prefix[:, None, :]  # (bh, bm, bn)
            base_scores = score_tile * jnp.exp(diff)

            if apply_causal_mask:
                decayed_scores = jnp.where(
                    causal_mask[None, :, :],
                    base_scores,
                    jnp.zeros_like(base_scores),
                )
            else:
                decayed_scores = base_scores

            partial_row_sum = jnp.sum(decayed_scores, axis=-1)  # (bh, bm)
            row_sum_acc_ref[...] = row_sum_acc_ref[...] + partial_row_sum

            partial_out = jax.lax.dot_general(
                v_block,
                decayed_scores,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )  # (bh, D, bm)
            out_acc_ref[...] = out_acc_ref[...] + partial_out

        for row_blk in range(n_row_tiles):
            row_start = row_blk * bm
            row_stop = row_start + bm

            row_sum_acc_ref[...] = jnp.zeros(row_sum_acc_ref.shape, jnp.float32)
            out_acc_ref[...] = jnp.zeros(out_acc_ref.shape, jnp.float32)

            q_block = q_full_ref[:, :, row_start:row_stop].astype(jnp.float32)      # (bh, D, bm)
            row_prefix = prefix_full_ref[:, row_start:row_stop].astype(jnp.float32) # (bh, bm)

            # Strictly lower-triangular blocks: no causal mask needed.
            for col_blk in range(row_blk):
                accumulate_block(q_block, row_prefix, col_blk, False)

            # Diagonal block: causal mask required.
            accumulate_block(q_block, row_prefix, row_blk, True)

            row_sum = row_sum_acc_ref[...]
            safe_sum = jnp.where(jnp.abs(row_sum) < 1e-6, 1.0, row_sum)
            den = jnp.maximum(jnp.abs(safe_sum), 1.0)  # (bh, bm)

            normalized = out_acc_ref[...] / den[:, None, :]
            o_full_ref[:, :, row_start:row_stop] = normalized.astype(out_dtype)

    out_t = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, D, S), out_dtype),
        grid=(B, H // bh),
        in_specs=[
            q_full_spec,
            k_full_spec,
            v_full_spec,
            prefix_full_spec,
        ],
        out_specs=out_full_spec,
        scratch_shapes=[
            pltpu.VMEM((bh, bm), jnp.float32),
            pltpu.VMEM((bh, D, bm), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.PARALLEL,
            ),
        ),
    )(query_t, key_t, value_t, log_a_cumsum)

    return jnp.transpose(out_t, (0, 1, 3, 2))


@jax.jit
def workload(query, key, value, A_log):
    """Mamba-2 SSD using a row-owned fused TPU Pallas kernel."""
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    output = _ssd_fused_pallas(
        query,
        key,
        value,
        log_a_cumsum,
        bh=8,
        bm=256,
        bn=256,
    )
    return output
''',
score=0.363,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='gpt-5.4',
stdout='Latency: 0.363 ms\n{"correct": true, "latency": 0.363, "error": "", "all_times_ms": [0.356, 0.356, 0.357, 0.357, 0.357, 0.357, 0.357, 0.357, 0.357, 0.357, 0.358, 0.358, 0.358, 0.359, 0.359, 0.359, 0.359, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.361, 0.361, 0.361, 0.361, 0.361, 0.361, 0.361, 0.361, 0.361, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.363, 0.363, 0.363, 0.363, 0.363, 0.363, 0.363, 0.363, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.364, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.365, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.366, 0.367, 0.367, 0.367, 0.368, 0.368, 0.368, 0.369, 0.369, 0.369, 0.37, 0.371, 0.372, 0.373, 0.374, 0.38, 0.393, 0.393], "max_diff": 0.03125, "max_rel_diff": 0.003998}',
stderr='')