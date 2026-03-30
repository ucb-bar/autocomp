CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''import jax

import jax.numpy as jnp

from functools import partial

CONFIG = {
    \'name\': \'mixtral_8x7b_moe\',
    \'model\': \'Mixtral-8x7B\',
    \'operator\': \'sparse_moe\',
    \'batch\': 1,
    \'seq_len\': 2048,
    \'emb_dim\': 4096,
    \'mlp_dim\': 14336,
    \'num_experts\': 8,
    \'num_experts_per_tok\': 2,
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = CONFIG[\'batch\'], CONFIG[\'seq_len\'], CONFIG[\'emb_dim\'], CONFIG[\'mlp_dim\']
    N = CONFIG[\'num_experts\']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k

def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    """Sparse MoE with einsum-based batched expert computation."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG[\'num_experts_per_tok\']
    # Routing
    logits = jnp.dot(x, router_weights)
    top_k_logits, top_k_indices = jax.lax.top_k(logits, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)
    # All experts in parallel
    gate_out = jax.nn.silu(jnp.einsum(\'bse,nem->bsnm\', x, expert_gate_kernels))
    up_out = jnp.einsum(\'bse,nem->bsnm\', x, expert_up_kernels)
    hidden = gate_out * up_out
    expert_outputs = jnp.einsum(\'bsnm,nme->bsne\', hidden, expert_down_kernels)
    # Weighted combination
    one_hot = jax.nn.one_hot(top_k_indices, N)
    weighted = one_hot * router_probs[..., None]
    expert_weights = weighted.sum(axis=2)
    output = jnp.einsum(\'bsne,bsn->bse\', expert_outputs, expert_weights)
    return output
''',
score=8.268,
translation_score=None,
hw_feedback=[],
plan_gen_model='None',
code_gen_model='None',
stdout='Latency: 8.268 ms\n{"correct": true, "latency": 8.268, "error": "", "all_times_ms": [8.249, 8.25, 8.251, 8.252, 8.254, 8.255, 8.255, 8.255, 8.255, 8.257, 8.258, 8.258, 8.259, 8.26, 8.26, 8.26, 8.261, 8.261, 8.262, 8.262, 8.263, 8.263, 8.263, 8.264, 8.264, 8.264, 8.264, 8.265, 8.265, 8.265, 8.265, 8.265, 8.265, 8.265, 8.266, 8.266, 8.266, 8.266, 8.266, 8.267, 8.267, 8.267, 8.267, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.268, 8.269, 8.269, 8.269, 8.269, 8.27, 8.27, 8.271, 8.271, 8.272, 8.272, 8.272, 8.272, 8.273, 8.274, 8.274, 8.275, 8.275, 8.275, 8.276, 8.276, 8.277, 8.277, 8.279, 8.279, 8.279, 8.281, 8.281, 8.282, 8.285, 8.288, 8.288, 8.29, 8.29, 8.291, 8.291, 8.291, 8.292, 8.293, 8.293, 8.293, 8.296, 8.297, 8.298, 8.299, 8.354, 8.379], "max_diff": 0.0, "max_rel_diff": 0.0}',
stderr=''),
plan='''**Selected strategy: 3**  
**Decompose the matmuls/einsums into blocked `pl.pallas_call` kernels with `pl.BlockSpec` index maps and a `pltpu.VMEM` float32 accumulator scratch buffer.**

### What to convert in this code
Convert these four contraction-heavy lines inside `workload(...)`:

1. `logits = jnp.dot(x, router_weights)`
2. `gate_out = jax.nn.silu(jnp.einsum(\'bse,nem->bsnm\', x, expert_gate_kernels))`
3. `up_out = jnp.einsum(\'bse,nem->bsnm\', x, expert_up_kernels)`
4. `expert_outputs = jnp.einsum(\'bsnm,nme->bsne\', hidden, expert_down_kernels)`

These are the parts that map naturally to TPU MXU-style blocked matmul kernels.

---

## Plan

### 1. Keep the public API unchanged
Keep:

```python
def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    ...
```

Do not change its signature.

Inside it, reshape only to expose matmul-friendly layouts:

- Flatten tokens once:
  - `T = B * S`
  - `x2 = x.reshape(T, E)`  
  This is allowed because it flattens leading dims onto the second-to-last dim.

---

### 2. Add one reusable reduction matmul kernel pattern
Create a Pallas kernel that:

- receives `Ref`s only
- reads tiles with `x_ref[...]`, `w_ref[...]`
- accumulates with:
  ```python
  acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
  ```
- zeros the VMEM accumulator on the first reduction step:
  ```python
  @pl.when(pl.program_id(k_axis) == 0)
  def _():
      acc_ref[...] = jnp.zeros_like(acc_ref)
  ```
- writes the result only on the last reduction step:
  ```python
  @pl.when(pl.program_id(k_axis) == num_k_steps - 1)
  def _():
      o_ref[...] = acc_ref[...].astype(o_ref.dtype)
  ```

This is the core conversion. The reduction axis must be the **last grid dimension** so the same output tile is visited consecutively and the accumulator can stay live in VMEM.

---

### 3. Replace the router projection with a blocked Pallas matmul
Convert:

```python
logits = jnp.dot(x, router_weights)
```

into a tiled matmul over `(T, E) @ (E, N) -> (T, N)`.

Suggested static tile sizes for this config on v6e-1:

- `bt = 128`
- `bk = 256`
- `N = 8` as the full last dimension

These divide evenly:

- `T = 2048`
- `E = 4096`
- `N = 8` (full dimension is allowed)

Suggested specs:

- `grid = (T // bt, E // bk)` with reduction axis last
- `x_spec = pl.BlockSpec((bt, bk), lambda t_blk, k_blk: (t_blk, k_blk))`
- `w_spec = pl.BlockSpec((bk, N), lambda t_blk, k_blk: (k_blk, 0))`
- `out_spec = pl.BlockSpec((bt, N), lambda t_blk, k_blk: (t_blk, 0))`
- `scratch_shapes = [pltpu.VMEM((bt, N), jnp.float32)]`

After the kernel, reshape back to `(B, S, N)` before `top_k`.

---

### 4. Replace the two expert input projections with one batched expert matmul helper
Convert both:

```python
jnp.einsum(\'bse,nem->bsnm\', x, expert_gate_kernels)
jnp.einsum(\'bse,nem->bsnm\', x, expert_up_kernels)
```

using the same blocked batched helper on:

- `x2`: shape `(T, E)`
- weights: shape `(N, E, M)`
- output layout: shape `(N, T, M)`

Using `(N, T, M)` is convenient because the expert axis becomes a grid axis, while the last two array dims remain TPU-friendly.

Suggested tiles:

- `bt = 128`
- `bk = 256`
- `bm = 512` or `1024`  
  (`14336` is divisible by both; pick one and keep it fixed)

Suggested grid/specs:

- `grid = (N, T // bt, M // bm, E // bk)` with `E // bk` last
- `x_spec = pl.BlockSpec((bt, bk), lambda n, t, m, k: (t, k))`
- `w_spec = pl.BlockSpec((None, bk, bm), lambda n, t, m, k: (n, k, m))`
- `out_spec = pl.BlockSpec((None, bt, bm), lambda n, t, m, k: (n, t, m))`
- `scratch_shapes = [pltpu.VMEM((bt, bm), jnp.float32)]`

Notes:

- `None` squeezes the expert block size of `1`, so the kernel sees 2D tiles for the matmul.
- All block shapes satisfy TPU rules:
  - second-to-last dims are multiples of `8`
  - last dims are multiples of `128`

Call this helper twice:
- once with `expert_gate_kernels`
- once with `expert_up_kernels`

Store outputs back in bf16 by casting on write.

---

### 5. Replace the expert down projection with the same pattern
Convert:

```python
expert_outputs = jnp.einsum(\'bsnm,nme->bsne\', hidden, expert_down_kernels)
```

by keeping `hidden` in `(N, T, M)` layout and computing:

- `(N, T, M) @ (N, M, E) -> (N, T, E)`

Suggested grid/specs:

- `grid = (N, T // bt, E // be, M // bk)` with `M // bk` last
- `hidden_spec = pl.BlockSpec((None, bt, bk), lambda n, t, e_blk, k: (n, t, k))`
- `down_spec = pl.BlockSpec((None, bk, be), lambda n, t, e_blk, k: (n, k, e_blk))`
- `out_spec = pl.BlockSpec((None, bt, be), lambda n, t, e_blk, k: (n, t, e_blk))`
- `scratch_shapes = [pltpu.VMEM((bt, be), jnp.float32)]`

with `be = 512` or `1024`, dividing `E = 4096`.

Then transpose/reshape the final expert result back to match the original combine logic, e.g. from `(N, T, E)` to `(B, S, N, E)` outside Pallas.

---

### 6. Leave the non-matmul logic unchanged in this phase
Do **not** convert these in this strategy:

- `top_k`
- `softmax`
- `silu`
- `gate_out * up_out`
- `one_hot`
- `weighted.sum(axis=2)`
- final expert-weighted combine

Keep them as ordinary JAX for now. This keeps the scope strictly on strategy 3.

---

## Why this fits v6e-1 TPU
- v6e-1 has **one TensorCore per chip**, so explicit blocked matmul kernels are the right representation.
- The chosen tiles align with TPU constraints:
  - second-to-last dimension divisible by `8`
  - last dimension divisible by `128` or equal to full dimension
- Accumulation stays in **float32 VMEM scratch**, which matches TPU matmul behavior and preserves numerical equivalence within tolerance.
- Example live-tile footprint is comfortably under VMEM limits:
  - lhs tile bf16
  - rhs tile bf16
  - f32 accumulator tile
  - plus pipeline buffers

---

## Expected code structure inside `workload`
In outline:

1. `x2 = x.reshape(B * S, E)`
2. `logits = router_pallas_matmul(x2, router_weights).reshape(B, S, N)`
3. Run existing routing:
   - `top_k`
   - `softmax`
4. `gate_proj = expert_batched_mm(x2, expert_gate_kernels)`  → `(N, T, M)`
5. `up_proj = expert_batched_mm(x2, expert_up_kernels)`      → `(N, T, M)`
6. `hidden = jax.nn.silu(gate_proj) * up_proj`
7. `expert_outputs_nte = expert_down_batched_mm(hidden, expert_down_kernels)` → `(N, T, E)`
8. Transpose/reshape to original logical layout
9. Keep the original weighted combine logic

---

## Semantics / numerical tolerance
- Use `jnp.dot(..., preferred_element_type=jnp.float32)` in every kernel.
- Cast back on store:
  ```python
  o_ref[...] = acc_ref[...].astype(o_ref.dtype)
  ```
- This preserves the original computation up to normal bf16/f32 TPU tolerance.

If you want, I can turn this plan into the actual Pallas skeleton for the router matmul and one expert batched matmul helper next.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# v6e-1-friendly tile sizes from the plan.
_ROUTER_BT = 128
_ROUTER_BK = 256

_EXPERT_BT = 128
_EXPERT_BK = 256
_EXPERT_BM = 512

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 512


def _make_router_kernel(num_k_steps: int):
    def kernel(x_ref, w_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(1) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            x_ref[...],
            w_ref[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(1) == num_k_steps - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _make_expert_input_kernel(num_k_steps: int):
    def kernel(x_ref, w_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(3) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            x_ref[...],
            w_ref[0, ...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(3) == num_k_steps - 1)
        def _store():
            o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _make_expert_down_kernel(num_k_steps: int):
    def kernel(hidden_ref, down_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(3) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            hidden_ref[0, ...],
            down_ref[0, ...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(3) == num_k_steps - 1)
        def _store():
            o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    num_k_steps = E // _ROUTER_BK
    kernel = _make_router_kernel(num_k_steps)

    out = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((T, N), x2.dtype)],
        grid=(T // _ROUTER_BT, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, N),
                lambda t_blk, k_blk: (t_blk, 0),
            )
        ],
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)

    return out[0]


def _expert_input_projection_pallas(
    x2: jax.Array,
    expert_kernels: jax.Array,
) -> jax.Array:
    T, E = x2.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert T % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0

    num_k_steps = E // _EXPERT_BK
    kernel = _make_expert_input_kernel(num_k_steps)

    out = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((N, T, M), x2.dtype)],
        grid=(N, T // _EXPERT_BT, M // _EXPERT_BM, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BK),
                lambda n_blk, t_blk, m_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _EXPERT_BK, _EXPERT_BM),
                lambda n_blk, t_blk, m_blk, k_blk: (n_blk, k_blk, m_blk),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (1, _EXPERT_BT, _EXPERT_BM),
                lambda n_blk, t_blk, m_blk, k_blk: (n_blk, t_blk, m_blk),
            )
        ],
        scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_input_projection",
    )(x2, expert_kernels)

    return out[0]


def _expert_down_projection_pallas(
    hidden_ntm: jax.Array,
    expert_down_kernels: jax.Array,
) -> jax.Array:
    N, T, M = hidden_ntm.shape
    N2, M2, E = expert_down_kernels.shape
    assert N == N2
    assert M == M2
    assert T % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0

    num_k_steps = M // _DOWN_BK
    kernel = _make_expert_down_kernel(num_k_steps)

    out = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((N, T, E), hidden_ntm.dtype)],
        grid=(N, T // _DOWN_BT, E // _DOWN_BE, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (1, _DOWN_BT, _DOWN_BK),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _DOWN_BK, _DOWN_BE),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, k_blk, e_blk),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                (1, _DOWN_BT, _DOWN_BE),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, e_blk),
            )
        ],
        scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_down_projection",
    )(hidden_ntm, expert_down_kernels)

    return out[0]


def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """Sparse MoE with Pallas-blocked TPU matmul kernels for contraction-heavy ops."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    # Expose matmul-friendly layout.
    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing logic kept in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    # 2) Gate projection: (T, E) x (N, E, M) -> (N, T, M)
    gate_proj_ntm = _expert_input_projection_pallas(x2, expert_gate_kernels)

    # 3) Up projection: (T, E) x (N, E, M) -> (N, T, M)
    up_proj_ntm = _expert_input_projection_pallas(x2, expert_up_kernels)

    # Non-matmul logic remains in regular JAX.
    gate_f32 = gate_proj_ntm.astype(jnp.float32)
    up_f32 = up_proj_ntm.astype(jnp.float32)
    hidden_ntm = (jax.nn.silu(gate_f32) * up_f32).astype(x.dtype)

    # 4) Down projection: (N, T, M) x (N, M, E) -> (N, T, E)
    expert_outputs_nte = _expert_down_projection_pallas(hidden_ntm, expert_down_kernels)

    # Back to original logical layout: (N, T, E) -> (B, S, N, E)
    expert_outputs = jnp.transpose(expert_outputs_nte, (1, 0, 2)).reshape(B, S, N, E)

    # Weighted combination kept in regular JAX.
    one_hot = jax.nn.one_hot(top_k_indices, N, dtype=jnp.float32)
    weighted = one_hot * router_probs[..., None]
    expert_weights = weighted.sum(axis=2)

    output = jnp.einsum(
        "bsne,bsn->bse",
        expert_outputs.astype(jnp.float32),
        expert_weights.astype(jnp.float32),
    )
    return output.astype(x.dtype)
''',
score=96.53,
translation_score=7.0,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='gpt-5.4',
stdout='Latency: 96.530 ms\n{"correct": true, "latency": 96.53, "error": "", "all_times_ms": [96.44, 96.444, 96.446, 96.461, 96.462, 96.468, 96.469, 96.472, 96.472, 96.48, 96.48, 96.483, 96.485, 96.486, 96.486, 96.487, 96.488, 96.489, 96.489, 96.49, 96.494, 96.495, 96.501, 96.502, 96.502, 96.505, 96.506, 96.507, 96.507, 96.507, 96.508, 96.509, 96.509, 96.51, 96.51, 96.513, 96.513, 96.513, 96.513, 96.514, 96.517, 96.519, 96.519, 96.522, 96.522, 96.525, 96.526, 96.526, 96.528, 96.529, 96.53, 96.535, 96.537, 96.537, 96.54, 96.54, 96.541, 96.541, 96.542, 96.542, 96.544, 96.546, 96.546, 96.548, 96.549, 96.549, 96.55, 96.551, 96.551, 96.552, 96.553, 96.553, 96.554, 96.554, 96.554, 96.555, 96.556, 96.556, 96.556, 96.559, 96.559, 96.56, 96.564, 96.564, 96.565, 96.565, 96.565, 96.57, 96.573, 96.575, 96.576, 96.58, 96.595, 96.596, 96.596, 96.599, 96.603, 96.605, 96.618, 96.628], "max_diff": 0.054258, "max_rel_diff": 0.004881}',
stderr=''),
plan='''Looking at this code, I need to identify an optimization opportunity using one of the available strategies.

## Analysis

The code implements a Mixture of Experts (MoE) workload with several Pallas kernels already in place:
1. `_router_pallas_matmul` - Router projection
2. `_expert_input_projection_pallas` - Gate and Up projections
3. `_expert_down_projection_pallas` - Down projection

However, I notice there\'s significant non-matmul computation happening in regular JAX between the kernels:

```python
# Non-matmul logic remains in regular JAX.
gate_f32 = gate_proj_ntm.astype(jnp.float32)
up_f32 = up_proj_ntm.astype(jnp.float32)
hidden_ntm = (jax.nn.silu(gate_f32) * up_f32).astype(x.dtype)
```

This is a classic SwiGLU activation pattern: `silu(gate) * up`. Currently this:
1. Reads gate_proj_ntm from HBM
2. Reads up_proj_ntm from HBM  
3. Computes silu and multiply
4. Writes hidden_ntm back to HBM
5. Then the down projection reads hidden_ntm from HBM again

## Strategy Selection

**Strategy 7** is the most impactful here: "Fuse post-matmul activations, transposes, or casts directly into the matmul kernel body by applying them to the accumulator on the final reduction step."

However, fusing SwiGLU requires both gate and up projections to be available simultaneously, which makes this complex since they\'re separate matmuls.

**Strategy 4** is more directly applicable: "Replace element-wise operations on large arrays with pipelined `pl.pallas_call` kernels that tile inputs into (8, 128)-aligned blocks via `pl.BlockSpec`."

The SwiGLU computation `(jax.nn.silu(gate_f32) * up_f32)` operates on arrays of shape `(N, T, M) = (8, 2048, 14336)` - that\'s ~2.2 GB of data being read twice and written once. This is memory-bound and should benefit from a tiled Pallas kernel.

## Plan: Apply Strategy 4 to the SwiGLU Activation

**Target code to convert:**
```python
gate_f32 = gate_proj_ntm.astype(jnp.float32)
up_f32 = up_proj_ntm.astype(jnp.float32)
hidden_ntm = (jax.nn.silu(gate_f32) * up_f32).astype(x.dtype)
```

**Implementation plan:**

1. **Create a new Pallas kernel `_swiglu_pallas`** that fuses the cast-to-f32, SiLU activation, elementwise multiply, and cast-back-to-bf16 into a single kernel.

2. **Choose block dimensions** aligned to (8, 128):
   - Shape is `(N=8, T=2048, M=14336)`
   - Use blocks of `(1, 128, 1024)` or similar - keeping N=1 per block since N=8 is small
   - This gives manageable VMEM usage: 2 inputs + 1 output × bf16 × (1, 128, 1024) = ~768 KB per block pair

3. **Grid structure**: 
   - Grid: `(N, T // BT, M // BM)` = `(8, 2048//128, 14336//1024)` = `(8, 16, 14)`
   - All dimensions are parallel (no reduction)
   - Use `dimension_semantics=("parallel", "parallel", "parallel")`

4. **Kernel implementation**:
   ```python
   def swiglu_kernel(gate_ref, up_ref, out_ref):
       # Read bf16 inputs, compute in f32
       gate_f32 = gate_ref[...].astype(jnp.float32)
       up_f32 = up_ref[...].astype(jnp.float32)
       # SiLU = x * sigmoid(x)
       result = (jax.nn.silu(gate_f32) * up_f32).astype(out_ref.dtype)
       out_ref[...] = result
   ```

5. **BlockSpecs**: Simple tiled access pattern with identity-like index_map:
   - `pl.BlockSpec((1, BT, BM), lambda n, t, m: (n, t, m))` for all inputs/outputs

6. **Replace** the three JAX lines with a single call to `_swiglu_pallas(gate_proj_ntm, up_proj_ntm)`.

This will reduce HBM traffic by fusing three memory operations into one pass and enable compute-memory overlap through pipelining.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# v6e-1-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 256

_EXPERT_BT = 128
_EXPERT_BK = 256
_EXPERT_BM = 512

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 512

# New fused SwiGLU elementwise tiling.
_SWIGLU_BT = 128
_SWIGLU_BM = 1024


def _make_router_kernel(num_k_steps: int):
    def kernel(x_ref, w_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(1) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            x_ref[...],
            w_ref[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(1) == num_k_steps - 1)
        def _store():
            o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _make_expert_input_kernel(num_k_steps: int):
    def kernel(x_ref, w_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(3) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            x_ref[...],
            w_ref[0, ...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(3) == num_k_steps - 1)
        def _store():
            o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _make_expert_down_kernel(num_k_steps: int):
    def kernel(hidden_ref, down_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(3) == 0)
        def _init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

        acc_ref[...] += jnp.dot(
            hidden_ref[0, ...],
            down_ref[0, ...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(3) == num_k_steps - 1)
        def _store():
            o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)

    return kernel


def _swiglu_kernel(gate_ref, up_ref, out_ref):
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32
    out_ref[...] = hidden_f32.astype(out_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    num_k_steps = E // _ROUTER_BK
    kernel = _make_router_kernel(num_k_steps)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


def _expert_input_projection_pallas(
    x2: jax.Array,
    expert_kernels: jax.Array,
) -> jax.Array:
    T, E = x2.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert T % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0

    num_k_steps = E // _EXPERT_BK
    kernel = _make_expert_input_kernel(num_k_steps)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), x2.dtype),
        grid=(N, T // _EXPERT_BT, M // _EXPERT_BM, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BK),
                lambda n_blk, t_blk, m_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _EXPERT_BK, _EXPERT_BM),
                lambda n_blk, t_blk, m_blk, k_blk: (n_blk, k_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _EXPERT_BT, _EXPERT_BM),
            lambda n_blk, t_blk, m_blk, k_blk: (n_blk, t_blk, m_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_input_projection",
    )(x2, expert_kernels)


def _swiglu_pallas(gate_proj_ntm: jax.Array, up_proj_ntm: jax.Array) -> jax.Array:
    N, T, M = gate_proj_ntm.shape
    assert up_proj_ntm.shape == (N, T, M)
    assert T % _SWIGLU_BT == 0
    assert M % _SWIGLU_BM == 0

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), gate_proj_ntm.dtype),
        grid=(N, T // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _SWIGLU_BT, _SWIGLU_BM),
            lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
        name="swiglu_activation",
    )(gate_proj_ntm, up_proj_ntm)


def _expert_down_projection_pallas(
    hidden_ntm: jax.Array,
    expert_down_kernels: jax.Array,
) -> jax.Array:
    N, T, M = hidden_ntm.shape
    N2, M2, E = expert_down_kernels.shape
    assert N == N2
    assert M == M2
    assert T % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0

    num_k_steps = M // _DOWN_BK
    kernel = _make_expert_down_kernel(num_k_steps)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, E), hidden_ntm.dtype),
        grid=(N, T // _DOWN_BT, E // _DOWN_BE, num_k_steps),
        in_specs=[
            pl.BlockSpec(
                (1, _DOWN_BT, _DOWN_BK),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _DOWN_BK, _DOWN_BE),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, k_blk, e_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _DOWN_BT, _DOWN_BE),
            lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, e_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_down_projection",
    )(hidden_ntm, expert_down_kernels)


def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """Sparse MoE with Pallas-blocked TPU matmul kernels for contraction-heavy ops."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    # Expose matmul-friendly layout.
    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing logic kept in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    # 2) Gate projection: (T, E) x (N, E, M) -> (N, T, M)
    gate_proj_ntm = _expert_input_projection_pallas(x2, expert_gate_kernels)

    # 3) Up projection: (T, E) x (N, E, M) -> (N, T, M)
    up_proj_ntm = _expert_input_projection_pallas(x2, expert_up_kernels)

    # 3.5) Fused SwiGLU in a tiled Pallas kernel:
    # hidden_ntm = (silu(gate_proj_ntm.astype(f32)) * up_proj_ntm.astype(f32)).astype(x.dtype)
    hidden_ntm = _swiglu_pallas(gate_proj_ntm, up_proj_ntm)

    # 4) Down projection: (N, T, M) x (N, M, E) -> (N, T, E)
    expert_outputs_nte = _expert_down_projection_pallas(hidden_ntm, expert_down_kernels)

    # Back to original logical layout: (N, T, E) -> (B, S, N, E)
    expert_outputs = jnp.transpose(expert_outputs_nte, (1, 0, 2)).reshape(B, S, N, E)

    # Weighted combination kept in regular JAX.
    one_hot = jax.nn.one_hot(top_k_indices, N, dtype=jnp.float32)
    weighted = one_hot * router_probs[..., None]
    expert_weights = weighted.sum(axis=2)

    output = jnp.einsum(
        "bsne,bsn->bse",
        expert_outputs.astype(jnp.float32),
        expert_weights.astype(jnp.float32),
    )
    return output.astype(x.dtype)
''',
score=97.095,
translation_score=8.0,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='gpt-5.4',
stdout='Latency: 97.095 ms\n{"correct": true, "latency": 97.095, "error": "", "all_times_ms": [96.992, 97.003, 97.013, 97.034, 97.045, 97.046, 97.048, 97.048, 97.053, 97.056, 97.057, 97.057, 97.058, 97.058, 97.06, 97.06, 97.064, 97.064, 97.064, 97.066, 97.068, 97.068, 97.069, 97.069, 97.069, 97.071, 97.072, 97.075, 97.076, 97.076, 97.077, 97.078, 97.079, 97.08, 97.081, 97.082, 97.082, 97.083, 97.085, 97.085, 97.087, 97.088, 97.088, 97.09, 97.09, 97.093, 97.094, 97.094, 97.094, 97.095, 97.095, 97.097, 97.097, 97.098, 97.098, 97.101, 97.103, 97.104, 97.104, 97.105, 97.105, 97.106, 97.106, 97.108, 97.109, 97.11, 97.11, 97.111, 97.112, 97.113, 97.114, 97.116, 97.116, 97.118, 97.118, 97.123, 97.123, 97.124, 97.131, 97.131, 97.137, 97.137, 97.14, 97.142, 97.145, 97.147, 97.148, 97.149, 97.151, 97.153, 97.157, 97.158, 97.161, 97.162, 97.167, 97.169, 97.175, 97.184, 97.185, 97.19], "max_diff": 0.054258, "max_rel_diff": 0.004881}',
stderr=''),
plan='''Looking at this code, I need to analyze what\'s happening and identify the best optimization strategy.

## Code Analysis

The code implements a Sparse Mixture of Experts (MoE) layer for Mixtral-8x7B. The current implementation already uses Pallas kernels for:
1. Router matmul (`_router_pallas_matmul`)
2. Expert input projections (gate and up) (`_expert_input_projection_pallas`)
3. SwiGLU activation (`_swiglu_pallas`)
4. Expert down projection (`_expert_down_projection_pallas`)

However, looking at the `workload` function, there\'s still a significant portion using regular JAX operations:

```python
# Weighted combination kept in regular JAX.
one_hot = jax.nn.one_hot(top_k_indices, N, dtype=jnp.float32)
weighted = one_hot * router_probs[..., None]
expert_weights = weighted.sum(axis=2)

output = jnp.einsum(
    "bsne,bsn->bse",
    expert_outputs.astype(jnp.float32),
    expert_weights.astype(jnp.float32),
)
```

This final weighted combination involves:
1. Creating a one-hot encoding
2. Element-wise multiplication with broadcasting
3. A reduction (sum over axis 2)
4. An einsum for the final weighted combination

## Strategy Selection

**Strategy 4** is most applicable here: "Replace element-wise operations on large arrays with pipelined `pl.pallas_call` kernels that tile inputs into (8, 128)-aligned blocks via `pl.BlockSpec`, using `pltpu.PrefetchScalarGridSpec` with appropriate `dimension_semantics` to overlap HBM↔VMEM transfers with VPU compute."

The weighted combination at the end involves:
- `expert_outputs` shape: (B, S, N, E) = (1, 2048, 8, 4096)
- `top_k_indices` shape: (B, S, K) = (1, 2048, 2)
- `router_probs` shape: (B, S, K) = (1, 2048, 2)

## Plan

**Fuse the weighted expert combination into a single Pallas kernel:**

1. **Create a new Pallas kernel `_weighted_expert_combine_pallas`** that takes:
   - `expert_outputs_nte`: (N, T, E) - the outputs from all experts
   - `top_k_indices`: (B, S, K) reshaped to (T, K) - which experts were selected
   - `router_probs`: (B, S, K) reshaped to (T, K) - the routing weights

2. **Kernel logic:**
   - Tile over T (tokens) and E (embedding dimension)
   - For each token, read the K selected expert indices and their weights
   - Gather the relevant expert outputs and compute the weighted sum
   - This fuses one_hot creation, multiplication, and summation into a single kernel pass

3. **Block specification:**
   - Grid: `(T // BT, E // BE)` where BT and BE are tile sizes (e.g., BT=128, BE=512)
   - Input `expert_outputs_nte`: BlockSpec `(N, BT, BE)` with index_map `lambda t, e: (0, t, e)` - load all N experts for the token block
   - Input `top_k_indices`: BlockSpec `(BT, K)` with index_map `lambda t, e: (t, 0)` - same indices for all E
   - Input `router_probs`: BlockSpec `(BT, K)` with index_map `lambda t, e: (t, 0)` - same probs for all E
   - Output: BlockSpec `(BT, BE)` with index_map `lambda t, e: (t, e)`

4. **Inside the kernel:**
   - Read expert_outputs block (N, BT, BE), indices block (BT, K), and probs block (BT, K)
   - Cast to float32 for computation
   - Use advanced indexing or loop over K to gather selected experts and multiply by weights
   - Sum across the K dimension to get (BT, BE) output
   - Cast back to output dtype

5. **Replace the JAX code** in `workload` with a call to this new kernel after reshaping inputs appropriately.

This approach eliminates multiple memory-bound passes (one-hot, multiply, sum, einsum) and replaces them with a single fused kernel that reads each element once.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# v6e-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 256

_EXPERT_BT = 128
_EXPERT_BK = 256
_EXPERT_BM = 512

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 512

_SWIGLU_BT = 128
_SWIGLU_BM = 1024

# Final fused combine tiles.
_COMBINE_BT = 128
_COMBINE_BE = 512


def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _expert_input_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _expert_down_kernel(hidden_ref, down_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        hidden_ref[0, ...],
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _swiglu_kernel(gate_ref, up_ref, out_ref):
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32
    out_ref[...] = hidden_f32.astype(out_ref.dtype)


def _weighted_expert_combine_kernel(expert_outputs_ref, topk_indices_ref, router_probs_ref, out_ref):
    """
    expert_outputs_ref: (N, BT, BE)
    topk_indices_ref:   (BT, K)
    router_probs_ref:   (BT, K)
    out_ref:            (BT, BE)

    Computes:
      out[t, e] = sum_k router_probs[t, k] * expert_outputs[topk_indices[t, k], t, e]
    """
    bt = out_ref.shape[0]
    be = out_ref.shape[1]
    num_experts = expert_outputs_ref.shape[0]
    k_per_tok = topk_indices_ref.shape[1]

    idx = topk_indices_ref[...]
    probs = router_probs_ref[...].astype(jnp.float32)

    acc = jnp.zeros((bt, be), dtype=jnp.float32)

    # N and K are tiny for Mixtral (8 experts, top-2), so full unrolling is acceptable.
    for n in range(num_experts):
        weight_n = jnp.zeros((bt,), dtype=jnp.float32)
        for k in range(k_per_tok):
            weight_n = weight_n + jnp.where(idx[:, k] == n, probs[:, k], 0.0)

        expert_slice = expert_outputs_ref[n, ...].astype(jnp.float32)
        acc = acc + expert_slice * weight_n[:, None]

    out_ref[...] = acc.astype(out_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


def _expert_input_projection_pallas(
    x2: jax.Array,
    expert_kernels: jax.Array,
) -> jax.Array:
    T, E = x2.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert T % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0

    return pl.pallas_call(
        _expert_input_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), x2.dtype),
        grid=(N, T // _EXPERT_BT, M // _EXPERT_BM, E // _EXPERT_BK),
        in_specs=[
            pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BK),
                lambda n_blk, t_blk, m_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _EXPERT_BK, _EXPERT_BM),
                lambda n_blk, t_blk, m_blk, k_blk: (n_blk, k_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _EXPERT_BT, _EXPERT_BM),
            lambda n_blk, t_blk, m_blk, k_blk: (n_blk, t_blk, m_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_input_projection",
    )(x2, expert_kernels)


def _swiglu_pallas(gate_proj_ntm: jax.Array, up_proj_ntm: jax.Array) -> jax.Array:
    N, T, M = gate_proj_ntm.shape
    assert up_proj_ntm.shape == (N, T, M)
    assert T % _SWIGLU_BT == 0
    assert M % _SWIGLU_BM == 0

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), gate_proj_ntm.dtype),
        grid=(N, T // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _SWIGLU_BT, _SWIGLU_BM),
            lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
        name="swiglu_activation",
    )(gate_proj_ntm, up_proj_ntm)


def _expert_down_projection_pallas(
    hidden_ntm: jax.Array,
    expert_down_kernels: jax.Array,
) -> jax.Array:
    N, T, M = hidden_ntm.shape
    N2, M2, E = expert_down_kernels.shape
    assert N == N2
    assert M == M2
    assert T % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0

    return pl.pallas_call(
        _expert_down_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, E), hidden_ntm.dtype),
        grid=(N, T // _DOWN_BT, E // _DOWN_BE, M // _DOWN_BK),
        in_specs=[
            pl.BlockSpec(
                (1, _DOWN_BT, _DOWN_BK),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _DOWN_BK, _DOWN_BE),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, k_blk, e_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _DOWN_BT, _DOWN_BE),
            lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, e_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_down_projection",
    )(hidden_ntm, expert_down_kernels)


def _weighted_expert_combine_pallas(
    expert_outputs_nte: jax.Array,
    top_k_indices_tk: jax.Array,
    router_probs_tk: jax.Array,
) -> jax.Array:
    """
    Fused replacement for:
      one_hot(top_k_indices) * router_probs[..., None]
      sum over K
      einsum("tne,tn->te")
    but operating directly on expert_outputs in (N, T, E) layout.
    """
    N, T, E = expert_outputs_nte.shape
    T2, K = top_k_indices_tk.shape
    T3, K2 = router_probs_tk.shape
    assert T == T2 == T3
    assert K == K2
    assert T % _COMBINE_BT == 0
    assert E % _COMBINE_BE == 0

    return pl.pallas_call(
        _weighted_expert_combine_kernel,
        out_shape=jax.ShapeDtypeStruct((T, E), expert_outputs_nte.dtype),
        grid=(T // _COMBINE_BT, E // _COMBINE_BE),
        in_specs=[
            pl.BlockSpec(
                (N, _COMBINE_BT, _COMBINE_BE),
                lambda t_blk, e_blk: (0, t_blk, e_blk),
            ),
            pl.BlockSpec(
                (_COMBINE_BT, K),
                lambda t_blk, e_blk: (t_blk, 0),
            ),
            pl.BlockSpec(
                (_COMBINE_BT, K),
                lambda t_blk, e_blk: (t_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_COMBINE_BT, _COMBINE_BE),
            lambda t_blk, e_blk: (t_blk, e_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="weighted_expert_combine",
    )(expert_outputs_nte, top_k_indices_tk, router_probs_tk)


def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """Sparse MoE with Pallas-blocked TPU kernels, including fused final expert combine."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    # Expose matmul-friendly layout.
    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing logic remains in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    # 2) Gate projection: (T, E) x (N, E, M) -> (N, T, M)
    gate_proj_ntm = _expert_input_projection_pallas(x2, expert_gate_kernels)

    # 3) Up projection: (T, E) x (N, E, M) -> (N, T, M)
    up_proj_ntm = _expert_input_projection_pallas(x2, expert_up_kernels)

    # 4) Fused SwiGLU: (N, T, M)
    hidden_ntm = _swiglu_pallas(gate_proj_ntm, up_proj_ntm)

    # 5) Down projection: (N, T, M) x (N, M, E) -> (N, T, E)
    expert_outputs_nte = _expert_down_projection_pallas(hidden_ntm, expert_down_kernels)

    # 6) Fused final combine directly from (N, T, E) plus top-k routing metadata.
    output_te = _weighted_expert_combine_pallas(
        expert_outputs_nte,
        top_k_indices.reshape(T, K),
        router_probs.reshape(T, K).astype(jnp.float32),
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=97.097,
translation_score=9.0,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='gpt-5.4',
stdout='Latency: 97.097 ms\n{"correct": true, "latency": 97.097, "error": "", "all_times_ms": [97.018, 97.023, 97.034, 97.035, 97.042, 97.044, 97.046, 97.052, 97.052, 97.054, 97.055, 97.056, 97.059, 97.061, 97.062, 97.063, 97.064, 97.064, 97.066, 97.066, 97.067, 97.068, 97.071, 97.072, 97.073, 97.074, 97.077, 97.078, 97.079, 97.079, 97.08, 97.081, 97.081, 97.082, 97.083, 97.083, 97.084, 97.085, 97.085, 97.086, 97.086, 97.086, 97.088, 97.088, 97.09, 97.091, 97.092, 97.092, 97.093, 97.095, 97.097, 97.098, 97.1, 97.101, 97.101, 97.103, 97.103, 97.105, 97.105, 97.106, 97.108, 97.109, 97.109, 97.109, 97.112, 97.113, 97.114, 97.117, 97.117, 97.117, 97.117, 97.119, 97.12, 97.121, 97.124, 97.125, 97.126, 97.131, 97.132, 97.132, 97.133, 97.133, 97.134, 97.135, 97.136, 97.138, 97.141, 97.144, 97.144, 97.146, 97.146, 97.148, 97.148, 97.153, 97.163, 97.164, 97.166, 97.169, 97.178, 97.203], "max_diff": 0.054258, "max_rel_diff": 0.004881}',
stderr=''),
plan='''### Optimization Plan

The current implementation uses relatively small block sizes (e.g., `_EXPERT_BK=256`, `_EXPERT_BM=512`), which results in a high number of grid iterations and underutilizes the memory reuse potential within the TensorCore\'s VMEM. For a "lite" class chip like the TPU v6e, maximizing arithmetic intensity by increasing block sizes is the primary lever for performance, provided we stay within the 16 MiB VMEM limit and respect tile alignment rules (last two dimensions divisible by 8 and 128).

**Strategy: Try new parameter values (Block Size Tuning)**

1.  **Increase Reduction Block Sizes ($BK$):** In the matrix multiplication kernels (`_router_kernel`, `_expert_input_kernel`, `_expert_down_kernel`), I will increase the block size along the reduction axis (from 256 to 512). This reduces the frequency of HBM memory transfers for the accumulator and weight blocks relative to the compute performed.
2.  **Increase Output Feature Block Sizes ($BM$, $BE$):** I will increase the block sizes for the MLP hidden dimension $M$ and the embedding dimension $E$ from 512 to 1024. This improves the reuse of the input activation blocks (tokens) across more weight features.
3.  **Adhere to V6e Alignment Constraints:** Ensure all updated block shapes satisfy the requirement that the last two dimensions are either equal to the full array dimension or divisible by 8 and 128 respectively.
    *   For example, in `_expert_input_projection_pallas`, the output feature block size will be 1024, which is divisible by 128. The token block size remains 128, which is divisible by 8.
4.  **Stay within VMEM limits:** With block sizes like $128 	imes 1024$ (f32) and $512 	imes 1024$ (bf16), the total memory per block set is approximately 1.5–2.0 MiB. Even with double-buffering, this remains well within the 16 MiB VMEM limit of the v6e TensorCore.

### Optimized Code

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}

# Optimized v6e tile sizes using Strategy 1.
# Increased BK to improve reduction efficiency and BM/BE to improve weight reuse.
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_SWIGLU_BT = 128
_SWIGLU_BM = 1024

_COMBINE_BT = 128
_COMBINE_BE = 1024


def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # Accumulate in f32 for matmul precision
    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _expert_input_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _expert_down_kernel(hidden_ref, down_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        hidden_ref[0, ...],
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _swiglu_kernel(gate_ref, up_ref, out_ref):
    # Upcast narrow types to 32-bit before elementwise ops
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32
    out_ref[...] = hidden_f32.astype(out_ref.dtype)


def _weighted_expert_combine_kernel(expert_outputs_ref, topk_indices_ref, router_probs_ref, out_ref):
    bt = out_ref.shape[0]
    be = out_ref.shape[1]
    num_experts = expert_outputs_ref.shape[0]
    k_per_tok = topk_indices_ref.shape[1]

    idx = topk_indices_ref[...]
    probs = router_probs_ref[...].astype(jnp.float32)

    acc = jnp.zeros((bt, be), dtype=jnp.float32)

    for n in range(num_experts):
        weight_n = jnp.zeros((bt,), dtype=jnp.float32)
        for k in range(k_per_tok):
            weight_n = weight_n + jnp.where(idx[:, k] == n, probs[:, k], 0.0)

        expert_slice = expert_outputs_ref[n, ...].astype(jnp.float32)
        acc = acc + expert_slice * weight_n[:, None]

    out_ref[...] = acc.astype(out_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec((_ROUTER_BT, _ROUTER_BK), lambda t, k: (t, k)),
            pl.BlockSpec((_ROUTER_BK, N), lambda t, k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((_ROUTER_BT, N), lambda t, k: (t, 0)),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="router_matmul",
    )(x2, router_weights)


def _expert_input_projection_pallas(x2: jax.Array, expert_kernels: jax.Array) -> jax.Array:
    T, E = x2.shape
    N, E2, M = expert_kernels.shape
    return pl.pallas_call(
        _expert_input_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), x2.dtype),
        grid=(N, T // _EXPERT_BT, M // _EXPERT_BM, E // _EXPERT_BK),
        in_specs=[
            pl.BlockSpec((_EXPERT_BT, _EXPERT_BK), lambda n, t, m, k: (t, k)),
            pl.BlockSpec((1, _EXPERT_BK, _EXPERT_BM), lambda n, t, m, k: (n, k, m)),
        ],
        out_specs=pl.BlockSpec((1, _EXPERT_BT, _EXPERT_BM), lambda n, t, m, k: (n, t, m)),
        scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
        name="expert_input_projection",
    )(x2, expert_kernels)


def _swiglu_pallas(gate_proj_ntm: jax.Array, up_proj_ntm: jax.Array) -> jax.Array:
    N, T, M = gate_proj_ntm.shape
    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), gate_proj_ntm.dtype),
        grid=(N, T // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec((1, _SWIGLU_BT, _SWIGLU_BM), lambda n, t, m: (n, t, m)),
            pl.BlockSpec((1, _SWIGLU_BT, _SWIGLU_BM), lambda n, t, m: (n, t, m)),
        ],
        out_specs=pl.BlockSpec((1, _SWIGLU_BT, _SWIGLU_BM), lambda n, t, m: (n, t, m)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        name="swiglu_activation",
    )(gate_proj_ntm, up_proj_ntm)


def _expert_down_projection_pallas(hidden_ntm: jax.Array, expert_down_kernels: jax.Array) -> jax.Array:
    N, T, M = hidden_ntm.shape
    N2, M2, E = expert_down_kernels.shape
    return pl.pallas_call(
        _expert_down_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, E), hidden_ntm.dtype),
        grid=(N, T // _DOWN_BT, E // _DOWN_BE, M // _DOWN_BK),
        in_specs=[
            pl.BlockSpec((1, _DOWN_BT, _DOWN_BK), lambda n, t, e, k: (n, t, k)),
            pl.BlockSpec((1, _DOWN_BK, _DOWN_BE), lambda n, t, e, k: (n, k, e)),
        ],
        out_specs=pl.BlockSpec((1, _DOWN_BT, _DOWN_BE), lambda n, t, e, k: (n, t, e)),
        scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
        name="expert_down_projection",
    )(hidden_ntm, expert_down_kernels)


def _weighted_expert_combine_pallas(expert_outputs_nte: jax.Array, top_k_indices_tk: jax.Array, router_probs_tk: jax.Array) -> jax.Array:
    N, T, E = expert_outputs_nte.shape
    T2, K = top_k_indices_tk.shape
    return pl.pallas_call(
        _weighted_expert_combine_kernel,
        out_shape=jax.ShapeDtypeStruct((T, E), expert_outputs_nte.dtype),
        grid=(T // _COMBINE_BT, E // _COMBINE_BE),
        in_specs=[
            pl.BlockSpec((N, _COMBINE_BT, _COMBINE_BE), lambda t, e: (0, t, e)),
            pl.BlockSpec((_COMBINE_BT, K), lambda t, e: (t, 0)),
            pl.BlockSpec((_COMBINE_BT, K), lambda t, e: (t, 0)),
        ],
        out_specs=pl.BlockSpec((_COMBINE_BT, _COMBINE_BE), lambda t, e: (t, e)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="weighted_expert_combine",
    )(expert_outputs_nte, top_k_indices_tk, router_probs_tk)


def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S
    x2 = x.reshape(T, E)

    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    top_k_logits, top_k_indices = jax.lax.top_k(logits.astype(jnp.float32), K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    gate_proj_ntm = _expert_input_projection_pallas(x2, expert_gate_kernels)
    up_proj_ntm = _expert_input_projection_pallas(x2, expert_up_kernels)
    hidden_ntm = _swiglu_pallas(gate_proj_ntm, up_proj_ntm)
    expert_outputs_nte = _expert_down_projection_pallas(hidden_ntm, expert_down_kernels)

    output_te = _weighted_expert_combine_pallas(
        expert_outputs_nte,
        top_k_indices.reshape(T, K),
        router_probs.reshape(T, K).astype(jnp.float32),
    )
    return output_te.reshape(B, S, E).astype(x.dtype)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# Tuned for TPU v6e-1:
# - larger K reduction tiles for better arithmetic intensity
# - larger feature tiles for better activation reuse
# - all trailing block dims obey TPU tiling constraints
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_SWIGLU_BT = 128
_SWIGLU_BM = 1024

_COMBINE_BT = 128
_COMBINE_BE = 1024


def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _expert_input_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _expert_down_kernel(hidden_ref, down_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        hidden_ref[0, ...],
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(3) == pl.num_programs(3) - 1)
    def _store():
        o_ref[0, ...] = acc_ref[...].astype(o_ref.dtype)


def _swiglu_kernel(gate_ref, up_ref, out_ref):
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32
    out_ref[...] = hidden_f32.astype(out_ref.dtype)


def _weighted_expert_combine_kernel(
    expert_outputs_ref, topk_indices_ref, router_probs_ref, out_ref
):
    """
    expert_outputs_ref: (N, BT, BE)
    topk_indices_ref:   (BT, K)
    router_probs_ref:   (BT, K)
    out_ref:            (BT, BE)

    Computes:
      out[t, e] = sum_k router_probs[t, k] * expert_outputs[topk_indices[t, k], t, e]
    """
    bt = out_ref.shape[0]
    be = out_ref.shape[1]
    num_experts = expert_outputs_ref.shape[0]
    k_per_tok = topk_indices_ref.shape[1]

    idx = topk_indices_ref[...]
    probs = router_probs_ref[...].astype(jnp.float32)

    acc = jnp.zeros((bt, be), dtype=jnp.float32)

    # Small static loops: Mixtral has 8 experts and top-2 routing.
    for n in range(num_experts):
        weight_n = jnp.zeros((bt,), dtype=jnp.float32)
        for k in range(k_per_tok):
            weight_n = weight_n + jnp.where(idx[:, k] == n, probs[:, k], 0.0)

        expert_slice = expert_outputs_ref[n, ...].astype(jnp.float32)
        acc = acc + expert_slice * weight_n[:, None]

    out_ref[...] = acc.astype(out_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


def _expert_input_projection_pallas(
    x2: jax.Array,
    expert_kernels: jax.Array,
) -> jax.Array:
    T, E = x2.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert T % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0

    return pl.pallas_call(
        _expert_input_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), x2.dtype),
        grid=(N, T // _EXPERT_BT, M // _EXPERT_BM, E // _EXPERT_BK),
        in_specs=[
            pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BK),
                lambda n_blk, t_blk, m_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _EXPERT_BK, _EXPERT_BM),
                lambda n_blk, t_blk, m_blk, k_blk: (n_blk, k_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _EXPERT_BT, _EXPERT_BM),
            lambda n_blk, t_blk, m_blk, k_blk: (n_blk, t_blk, m_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_input_projection",
    )(x2, expert_kernels)


def _swiglu_pallas(gate_proj_ntm: jax.Array, up_proj_ntm: jax.Array) -> jax.Array:
    N, T, M = gate_proj_ntm.shape
    assert up_proj_ntm.shape == (N, T, M)
    assert T % _SWIGLU_BT == 0
    assert M % _SWIGLU_BM == 0

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, M), gate_proj_ntm.dtype),
        grid=(N, T // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
            pl.BlockSpec(
                (1, _SWIGLU_BT, _SWIGLU_BM),
                lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _SWIGLU_BT, _SWIGLU_BM),
            lambda n_blk, t_blk, m_blk: (n_blk, t_blk, m_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
        name="swiglu_activation",
    )(gate_proj_ntm, up_proj_ntm)


def _expert_down_projection_pallas(
    hidden_ntm: jax.Array,
    expert_down_kernels: jax.Array,
) -> jax.Array:
    N, T, M = hidden_ntm.shape
    N2, M2, E = expert_down_kernels.shape
    assert N == N2
    assert M == M2
    assert T % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0

    return pl.pallas_call(
        _expert_down_kernel,
        out_shape=jax.ShapeDtypeStruct((N, T, E), hidden_ntm.dtype),
        grid=(N, T // _DOWN_BT, E // _DOWN_BE, M // _DOWN_BK),
        in_specs=[
            pl.BlockSpec(
                (1, _DOWN_BT, _DOWN_BK),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, k_blk),
            ),
            pl.BlockSpec(
                (1, _DOWN_BK, _DOWN_BE),
                lambda n_blk, t_blk, e_blk, k_blk: (n_blk, k_blk, e_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (1, _DOWN_BT, _DOWN_BE),
            lambda n_blk, t_blk, e_blk, k_blk: (n_blk, t_blk, e_blk),
        ),
        scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="expert_down_projection",
    )(hidden_ntm, expert_down_kernels)


def _weighted_expert_combine_pallas(
    expert_outputs_nte: jax.Array,
    top_k_indices_tk: jax.Array,
    router_probs_tk: jax.Array,
) -> jax.Array:
    N, T, E = expert_outputs_nte.shape
    T2, K = top_k_indices_tk.shape
    T3, K2 = router_probs_tk.shape
    assert T == T2 == T3
    assert K == K2
    assert T % _COMBINE_BT == 0
    assert E % _COMBINE_BE == 0

    return pl.pallas_call(
        _weighted_expert_combine_kernel,
        out_shape=jax.ShapeDtypeStruct((T, E), expert_outputs_nte.dtype),
        grid=(T // _COMBINE_BT, E // _COMBINE_BE),
        in_specs=[
            pl.BlockSpec(
                (N, _COMBINE_BT, _COMBINE_BE),
                lambda t_blk, e_blk: (0, t_blk, e_blk),
            ),
            pl.BlockSpec(
                (_COMBINE_BT, K),
                lambda t_blk, e_blk: (t_blk, 0),
            ),
            pl.BlockSpec(
                (_COMBINE_BT, K),
                lambda t_blk, e_blk: (t_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_COMBINE_BT, _COMBINE_BE),
            lambda t_blk, e_blk: (t_blk, e_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="weighted_expert_combine",
    )(expert_outputs_nte, top_k_indices_tk, router_probs_tk)


def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """Sparse MoE with larger v6e-friendly Pallas tiles and fused final expert combine."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing stays in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    # 2) Gate projection: (T, E) x (N, E, M) -> (N, T, M)
    gate_proj_ntm = _expert_input_projection_pallas(x2, expert_gate_kernels)

    # 3) Up projection: (T, E) x (N, E, M) -> (N, T, M)
    up_proj_ntm = _expert_input_projection_pallas(x2, expert_up_kernels)

    # 4) SwiGLU in float32 elementwise math, cast back on store.
    hidden_ntm = _swiglu_pallas(gate_proj_ntm, up_proj_ntm)

    # 5) Down projection: (N, T, M) x (N, M, E) -> (N, T, E)
    expert_outputs_nte = _expert_down_projection_pallas(hidden_ntm, expert_down_kernels)

    # 6) Fused final combine from (N, T, E) using top-k routing metadata.
    output_te = _weighted_expert_combine_pallas(
        expert_outputs_nte,
        top_k_indices.reshape(T, K),
        router_probs.reshape(T, K).astype(jnp.float32),
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=49.362,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='gpt-5.4',
stdout='Latency: 49.362 ms\n{"correct": true, "latency": 49.362, "error": "", "all_times_ms": [49.158, 49.172, 49.177, 49.18, 49.181, 49.186, 49.188, 49.189, 49.19, 49.206, 49.218, 49.223, 49.223, 49.228, 49.233, 49.249, 49.254, 49.262, 49.265, 49.265, 49.266, 49.275, 49.279, 49.284, 49.289, 49.304, 49.308, 49.311, 49.312, 49.317, 49.319, 49.319, 49.32, 49.321, 49.321, 49.322, 49.323, 49.324, 49.33, 49.332, 49.341, 49.346, 49.351, 49.352, 49.352, 49.354, 49.359, 49.36, 49.36, 49.361, 49.362, 49.363, 49.364, 49.365, 49.365, 49.367, 49.374, 49.381, 49.383, 49.383, 49.384, 49.385, 49.386, 49.391, 49.393, 49.396, 49.402, 49.403, 49.417, 49.419, 49.426, 49.426, 49.432, 49.439, 49.441, 49.445, 49.445, 49.447, 49.448, 49.45, 49.452, 49.465, 49.471, 49.483, 49.485, 49.485, 49.487, 49.489, 49.491, 49.499, 49.513, 49.515, 49.524, 49.528, 49.534, 49.535, 49.542, 49.551, 49.576, 49.68], "max_diff": 0.062513, "max_rel_diff": 0.005623}',
stderr=''),
plan='''**New phase-2 strategy: route-compacted expert execution**

Instead of running the expert MLP for **all 8 experts on all 2048 tokens** and only using the top-2 results at the end, **compact the routed token/expert pairs immediately after `top_k`** and run the expert path only on those routed pairs.

For Mixtral-8x7B here:

- original expert path computes on `N * T = 8 * 2048 = 16384` token-expert rows
- actual useful work is only `K * T = 2 * 2048 = 4096` routed rows

So the expert matmuls are doing about **4x more row work than necessary**.

---

## Why this helps on v6e-1

This code is dominated by the expert path:

1. gate projection
2. up projection
3. SwiGLU
4. down projection
5. final combine

Right now those run on dense tensors:

- `gate_proj_ntm`: `(N, T, M)`
- `up_proj_ntm`: `(N, T, M)`
- `hidden_ntm`: `(N, T, M)`
- `expert_outputs_nte`: `(N, T, E)`

That means you materialize huge HBM tensors for expert/token combinations that are never used.

### The new idea
After routing, build a packed route layout of length `R = T * K`, grouped by expert, padded to TPU-friendly blocks. Then run the expert kernels on:

- `packed_x`: `(R_cap, E)`
- `gate/up`: `(R_cap, M)`
- `hidden`: `(R_cap, M)`
- `packed_out`: `(R_cap, E)`

where `R_cap` is a **static upper bound** like:

```python
R = T * K
R_cap = ceil_to_multiple(R + N * (BT - 1), BT)
```

With `T=2048`, `K=2`, `N=8`, `BT=128`:

- `R = 4096`
- `R_cap <= 5120`

That is still far smaller than `N*T = 16384`.

---

## The key implementation trick

Make each `BT=128` route block contain rows from **only one expert**.

That lets the expert matmul kernels use:

- one packed input block `(BT, BK)`
- one expert id in SMEM
- one expert weight block `(1, BK, BM)`

This follows the sparse/reference pattern with `PrefetchScalarGridSpec`: pass a small `block_expert` table through SMEM and let the `BlockSpec` for the weight tensor select the right expert block.

Because route blocks are grouped by expert, **consecutive iterations reuse the same expert weight tiles**. On TPU, repeated consecutive accesses to the same input slice can skip HBM reloads, so this also improves cache/pipeline reuse beyond just reducing FLOPs.

---

## What to change

### 1) Keep router projection as-is
No change to:

- `_router_kernel`
- `_router_pallas_matmul`

Router logits/top-k/softmax stay outside Pallas for now.

---

### 2) Replace dense `(N, T, *)` expert execution with packed routed execution

Add a route packer after `top_k`:

```python
_PACK_BT = 128

def _ceil_div(a, b):
    return (a + b - 1) // b

def _pack_routes(x2, top_k_indices_tk, router_probs_tk):
    T, E = x2.shape
    K = top_k_indices_tk.shape[1]
    N = CONFIG["num_experts"]

    # Total real routes.
    R = T * K

    # Static capacity: real routes + at most (BT-1) padding rows per expert.
    R_cap = int((_ceil_div(R + N * (_PACK_BT - 1), _PACK_BT)) * _PACK_BT)

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)       # (R,)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)        # (R,)
    route_token = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)        # (R,)

    # Per-route rank inside its expert bucket.
    one_hot = jax.nn.one_hot(route_expert, N, dtype=jnp.int32)         # (R, N)
    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )                                                                  # (R,)

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)                 # (N,)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)                    # (N,)
    padded_counts = blocks_per_expert * _PACK_BT                       # (N,)

    expert_offsets = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
    ])                                                                  # (N,)

    packed_pos = expert_offsets[route_expert] + route_rank             # (R,)

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    # Expert id for each BT-sized route block.
    num_route_blocks = R_cap // _PACK_BT
    block_offsets = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(blocks_per_expert[:-1], dtype=jnp.int32),
    ])
    block_ids = jnp.arange(num_route_blocks, dtype=jnp.int32)[:, None]
    expert_ids = jnp.arange(N, dtype=jnp.int32)[None, :]
    block_mask = (
        (block_ids >= block_offsets[None, :]) &
        (block_ids < (block_offsets + blocks_per_expert)[None, :])
    )
    block_expert = jnp.sum(
        block_mask.astype(jnp.int32) * expert_ids,
        axis=1,
        dtype=jnp.int32,
    )  # (num_route_blocks,)

    return packed_x, packed_token, packed_prob, block_expert
```

This is semantically exact: every real route gets one packed row, padding rows are zeroed and later ignored.

---

### 3) Add a packed expert projection kernel
Use a 2D routed input `(R_cap, E)` instead of `(T, E)` plus explicit expert axis.

Kernel body can stay very close to your current `_expert_input_kernel`, but the weight expert is selected through SMEM metadata:

```python
def _packed_expert_input_kernel(block_expert, x_ref, w_ref, o_ref, acc_ref):
    del block_expert

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)
```

And wrap it like this:

```python
def _packed_expert_input_projection_pallas(
    packed_x: jax.Array,          # (R_cap, E)
    block_expert: jax.Array,      # (R_cap // BT,)
    expert_kernels: jax.Array,    # (N, E, M)
) -> jax.Array:
    R_cap, E = packed_x.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert R_cap % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0

    return pl.pallas_call(
        _packed_expert_input_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_BT, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BK),
                    lambda m_blk, r_blk, k_blk, block_expert: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert:
                        (block_expert[r_blk], k_blk, m_blk),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BM),
                lambda m_blk, r_blk, k_blk: (r_blk, m_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_input_projection",
    )(block_expert, packed_x, expert_kernels)
```

### Why the grid is ordered this way
`grid=(m_blk, r_blk, k_blk)` keeps the reduction axis last, while making `r_blk` faster-changing than `m_blk` among the parallel axes. Since route blocks are expert-grouped, that improves consecutive reuse of the same expert weight tile.

---

### 4) Reuse SwiGLU on the packed route tensor
Change `_swiglu_pallas` from `(N, T, M)` to `(R_cap, M)`:

```python
def _swiglu_routes_pallas(gate_rm: jax.Array, up_rm: jax.Array) -> jax.Array:
    R_cap, M = gate_rm.shape
    assert up_rm.shape == (R_cap, M)
    assert R_cap % _SWIGLU_BT == 0
    assert M % _SWIGLU_BM == 0

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((R_cap, M), gate_rm.dtype),
        grid=(R_cap // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec((_SWIGLU_BT, _SWIGLU_BM), lambda r_blk, m_blk: (r_blk, m_blk)),
            pl.BlockSpec((_SWIGLU_BT, _SWIGLU_BM), lambda r_blk, m_blk: (r_blk, m_blk)),
        ],
        out_specs=pl.BlockSpec(
            (_SWIGLU_BT, _SWIGLU_BM),
            lambda r_blk, m_blk: (r_blk, m_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="swiglu_routes",
    )(gate_rm, up_rm)
```

---

### 5) Add a packed down-projection kernel
Same idea as the packed input projection:

```python
def _packed_expert_down_kernel(block_expert, hidden_ref, down_ref, o_ref, acc_ref):
    del block_expert

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        hidden_ref[...],
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)
```

Wrapper:

```python
def _packed_expert_down_projection_pallas(
    hidden_rm: jax.Array,         # (R_cap, M)
    block_expert: jax.Array,      # (R_cap // BT,)
    expert_down_kernels: jax.Array,  # (N, M, E)
) -> jax.Array:
    R_cap, M = hidden_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert M == M2
    assert R_cap % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0

    return pl.pallas_call(
        _packed_expert_down_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _DOWN_BT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, r_blk, k_blk, block_expert:
                        (block_expert[r_blk], k_blk, e_blk),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_DOWN_BT, _DOWN_BE),
                lambda e_blk, r_blk, k_blk: (r_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), hidden_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_down_projection",
    )(block_expert, hidden_rm, expert_down_kernels)
```

---

### 6) Replace final combine with a simple float32 scatter-add
You no longer need `(N, T, E)` and the explicit expert loop. Just scatter the routed outputs back to tokens:

```python
def _scatter_routed_outputs(
    packed_out_re: jax.Array,   # (R_cap, E)
    packed_token: jax.Array,    # (R_cap,)
    packed_prob: jax.Array,     # (R_cap,)
    T: int,
) -> jax.Array:
    valid = packed_token >= 0
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[packed_token[valid]].add(weighted[valid])
    return out
```

This is also semantically exact up to normal float32 accumulation order differences.

---

## Updated `workload`

The public function signature stays the same:

```python
def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # Router.
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    # New: compact routed token/expert pairs.
    packed_x, packed_token, packed_prob, block_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K),
        router_probs.reshape(T, K),
    )

    # Expert path only on routed rows.
    gate_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_gate_kernels
    )
    up_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_up_kernels
    )
    hidden_rm = _swiglu_routes_pallas(gate_rm, up_rm)
    packed_out_re = _packed_expert_down_projection_pallas(
        hidden_rm, block_expert, expert_down_kernels
    )

    # Scatter back to tokens.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
```

---

## Why this should be faster

### 1) Much less expert matmul work
You go from dense expert rows:

- before: `N*T = 16384`
- after: `R_cap ≈ 5120`

That’s roughly a **3.2x reduction in row count**, and the true useful rows are exactly `4096`.

For the dominant `E↔M` expert matmuls, this is the biggest win in the whole program.

---

### 2) Much less HBM traffic
You stop materializing full:

- `(8, 2048, 14336)` gate
- `(8, 2048, 14336)` up
- `(8, 2048, 14336)` hidden
- `(8, 2048, 4096)` down output

and instead materialize route-packed versions with only ~`5120` rows total.

That cuts large intermediate tensors by about **3–4x**.

---

### 3) Better expert weight reuse
Because route blocks are grouped by expert, consecutive route blocks use the same expert weight slices. On TPU Pallas, repeated consecutive accesses to the same slice can avoid extra HBM transfers, so expert-grouping improves effective caching automatically.

This is especially valuable on v6e-1, where bandwidth pressure is often the limiter once the MXU is reasonably fed.

---

### 4) No TPU-invalid shapes
This rewrite keeps all kernel block shapes legal:

- route block second-to-last dims use `128`
- reduction dims use `512`
- output feature dims use `1024`
- no singleton trailing dims

So it remains v6e-safe.

---

## Expected impact

This is one of the highest-value phase-2 changes for this code because it removes the biggest structural inefficiency: **dense execution of a sparse MoE**.

I would expect this to beat the current 49.362 latency more than a tile-only retune, because it reduces both:

- MXU work
- HBM round-trips

without requiring risky kernel fusion.

If you want, I can next turn this plan into a **full rewritten program** with the helper functions and Pallas wrappers filled in end-to-end.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# TPU-v6e-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_SWIGLU_BT = 128
_SWIGLU_BM = 1024

_PACK_BT = 128

assert _PACK_BT == _EXPERT_BT == _DOWN_BT == _SWIGLU_BT


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


# ----------------------------
# Route packing
# ----------------------------

def _pack_routes(
    x2: jax.Array,              # (T, E)
    top_k_indices_tk: jax.Array,  # (T, K), int32
    router_probs_tk: jax.Array,   # (T, K), float32
):
    """
    Packs routed token/expert pairs into expert-grouped BT-sized blocks.

    Returns:
      packed_x:      (R_cap, E)
      packed_token:  (R_cap,) int32, -1 for padding rows
      packed_prob:   (R_cap,) float32, 0 for padding rows
      block_expert:  (R_cap // BT,) int32, expert id for each BT block
    """
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    assert T == T2
    N = CONFIG["num_experts"]

    R = T * K
    R_cap = int(_ceil_div(R + N * (_PACK_BT - 1), _PACK_BT) * _PACK_BT)

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)          # (R,)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)           # (R,)
    route_token = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(R)                                                          # (R,)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)  # (R, N)

    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )                                                                     # (R,)

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)                    # (N,)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)                       # (N,)
    padded_counts = blocks_per_expert * _PACK_BT                          # (N,)

    expert_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )                                                                     # (N,)

    packed_pos = expert_offsets[route_expert] + route_rank                # (R,)

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    num_route_blocks = R_cap // _PACK_BT
    block_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(blocks_per_expert[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )                                                                     # (N,)

    block_ids = jnp.arange(num_route_blocks, dtype=jnp.int32)[:, None]    # (B, 1)
    block_mask = (
        (block_ids >= block_offsets[None, :]) &
        (block_ids < (block_offsets + blocks_per_expert)[None, :])
    )                                                                     # (B, N)

    block_expert = jnp.sum(
        block_mask.astype(jnp.int32) * expert_ids[None, :],
        axis=1,
        dtype=jnp.int32,
    )                                                                     # (B,)

    return packed_x, packed_token, packed_prob, block_expert


# ----------------------------
# Packed expert input projections
# ----------------------------

def _packed_expert_input_kernel(block_expert_ref, x_ref, w_ref, o_ref, acc_ref):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_input_projection_pallas(
    packed_x: jax.Array,        # (R_cap, E)
    block_expert: jax.Array,    # (R_cap // BT,)
    expert_kernels: jax.Array,  # (N, E, M)
) -> jax.Array:
    R_cap, E = packed_x.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert R_cap % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0
    del N

    return pl.pallas_call(
        _packed_expert_input_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_BT, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BK),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BM),
                lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, m_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_input_projection",
    )(block_expert, packed_x, expert_kernels)


# ----------------------------
# Packed SwiGLU
# ----------------------------

def _swiglu_kernel(gate_ref, up_ref, out_ref):
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32
    out_ref[...] = hidden_f32.astype(out_ref.dtype)


def _swiglu_routes_pallas(gate_rm: jax.Array, up_rm: jax.Array) -> jax.Array:
    R_cap, M = gate_rm.shape
    assert up_rm.shape == (R_cap, M)
    assert R_cap % _SWIGLU_BT == 0
    assert M % _SWIGLU_BM == 0

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((R_cap, M), gate_rm.dtype),
        grid=(R_cap // _SWIGLU_BT, M // _SWIGLU_BM),
        in_specs=[
            pl.BlockSpec(
                (_SWIGLU_BT, _SWIGLU_BM),
                lambda r_blk, m_blk: (r_blk, m_blk),
            ),
            pl.BlockSpec(
                (_SWIGLU_BT, _SWIGLU_BM),
                lambda r_blk, m_blk: (r_blk, m_blk),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_SWIGLU_BT, _SWIGLU_BM),
            lambda r_blk, m_blk: (r_blk, m_blk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="swiglu_routes",
    )(gate_rm, up_rm)


# ----------------------------
# Packed expert down projection
# ----------------------------

def _packed_expert_down_kernel(block_expert_ref, hidden_ref, down_ref, o_ref, acc_ref):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        hidden_ref[...],
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_down_projection_pallas(
    hidden_rm: jax.Array,          # (R_cap, M)
    block_expert: jax.Array,       # (R_cap // BT,)
    expert_down_kernels: jax.Array,  # (N, M, E)
) -> jax.Array:
    R_cap, M = hidden_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert M == M2
    assert R_cap % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0
    del N

    return pl.pallas_call(
        _packed_expert_down_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _DOWN_BT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        e_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_DOWN_BT, _DOWN_BE),
                lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), hidden_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_down_projection",
    )(block_expert, hidden_rm, expert_down_kernels)


# ----------------------------
# Scatter routed outputs back to tokens
# ----------------------------

def _scatter_routed_outputs(
    packed_out_re: jax.Array,  # (R_cap, E)
    packed_token: jax.Array,   # (R_cap,)
    packed_prob: jax.Array,    # (R_cap,)
    T: int,
) -> jax.Array:
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out


# ----------------------------
# Public workload
# ----------------------------

def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """
    Sparse MoE with route-compacted expert execution.

    Semantics match the original dense-all-experts implementation up to normal
    floating-point accumulation differences.
    """
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)

    # 2) Compact routed token/expert pairs.
    packed_x, packed_token, packed_prob, block_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K).astype(jnp.int32),
        router_probs.reshape(T, K),
    )

    # 3) Expert path only on routed rows.
    gate_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_gate_kernels
    )
    up_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_up_kernels
    )
    hidden_rm = _swiglu_routes_pallas(gate_rm, up_rm)
    packed_out_re = _packed_expert_down_projection_pallas(
        hidden_rm, block_expert, expert_down_kernels
    )

    # 4) Scatter back to token space.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=16.002,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='gpt-5.4',
stdout='Latency: 16.002 ms\n{"correct": true, "latency": 16.002, "error": "", "all_times_ms": [15.964, 15.967, 15.968, 15.974, 15.974, 15.974, 15.977, 15.978, 15.978, 15.979, 15.981, 15.983, 15.983, 15.985, 15.985, 15.986, 15.987, 15.987, 15.987, 15.987, 15.988, 15.989, 15.991, 15.993, 15.993, 15.994, 15.994, 15.994, 15.995, 15.995, 15.995, 15.996, 15.996, 15.996, 15.997, 15.997, 15.997, 15.997, 15.997, 15.998, 15.998, 15.998, 15.999, 15.999, 15.999, 16.0, 16.0, 16.0, 16.0, 16.001, 16.002, 16.003, 16.003, 16.004, 16.004, 16.004, 16.005, 16.005, 16.005, 16.006, 16.006, 16.006, 16.007, 16.007, 16.007, 16.007, 16.008, 16.008, 16.008, 16.008, 16.009, 16.009, 16.009, 16.009, 16.009, 16.01, 16.01, 16.01, 16.011, 16.012, 16.012, 16.014, 16.014, 16.015, 16.015, 16.016, 16.018, 16.019, 16.02, 16.021, 16.022, 16.022, 16.023, 16.024, 16.028, 16.028, 16.033, 16.035, 16.04, 16.523], "max_diff": 0.062513, "max_rel_diff": 0.005623}',
stderr=''),
plan='''## Performance Optimization Plan

### Analysis of Inefficiency

The provided implementation creates **two intermediate tensors** (`gate_rm` and `up_rm`) of size `(R_cap, M)` where `R_cap ≈ 8192` and `M = 14336`. 

**Memory Impact Analysis:**
- Each tensor: `8192 × 14336 × 4 bytes = 469 MB` (in float32 accumulator space)
- Even accounting for pipelining and smaller effective VMEM usage, these tensors represent substantial memory pressure
- They must be written out completely by `_packed_expert_input_projection_pallas`, then read back entirely by `_swiglu_routes_pallas`, then read again by `_packed_expert_down_projection_pallas`

**Execution Flow:**
```
packed_x → gate_proj_pallas (write gate_rm) → ...
           up_proj_pallas   (write up_rm)   → ...
           swiglu_pallas    (read both, write hidden_rm) → ...
           down_proj_pallas (read hidden_rm)
```

This creates a **sequential dependency chain** where the down projection cannot start until both input projections complete AND SwiGLU completes. The intermediate tensors act as synchronization barriers.

### Optimization Strategy: Fusion of SwiGLU Activation

**Strategy 8** from the list: "Fuse SwiGLU activation directly into the down projection kernel\'s input load phase, eliminating the intermediate hidden_rm tensor"

By fusing SwiGLU into the down projection kernel, we:
1. Eliminate `hidden_rm` tensor entirely (one large intermediate tensor removed)
2. Eliminate the `_swiglu_routes_pallas` kernel call completely
3. Enable data to flow directly from `{gate_rm, up_rm} → {VREGs}` without ever touching VMEM for `hidden_rm`

### Implementation Plan

#### Step 1: Modify the hidden state loading in `_packed_expert_down_kernel`

Replace the direct load of `hidden_ref` with a fused computation:

```python
def _packed_expert_down_fused_kernel(block_expert_ref, gate_ref, up_ref, down_ref, o_ref, acc_ref):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # Fused SwiGLU: load gate and up, apply activation in VREGs
    gate_f32 = gate_ref[...].astype(jnp.float32)   # Load gate projection result
    up_f32 = up_ref[...].astype(jnp.float32)       # Load up projection result
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32    # SwiGLU activation in registers

    # Matrix multiplication with the intermediate hidden state
    acc_ref[...] += jnp.dot(
        hidden_f32,  # gpu: not a ref, directly using the computed array
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)
```

#### Step 2: Update the call signature and BlockSpecs

The down projection function now receives `gate_rm` and `up_rm` separately:

```python
def _packed_expert_down_projection_pallas(
    gate_rm: jax.Array,        # (R_cap, M) - gate projection output
    up_rm: jax.Array,          # (R_cap, M) - up projection output  
    block_expert: jax.Array,   # (R_cap // BT,)
    expert_down_kernels: jax.Array,  # (N, M, E)
) -> jax.Array:
    R_cap, M = gate_rm.shape
    N, M2, E = expert_down_kernels.shape
    # ... assertions ...
    
    return pl.pallas_call(
        _packed_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _DOWN_BT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, be: (r_blk, k_blk),
                ),
                # ← Second input: up_rm block spec (identical)
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, be: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, r_blk, k_blk, be: (be[r_blk], k_blk, e_blk),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_DOWN_BT, _DOWN_BE),
                lambda e_blk, r_blk, k_blk, be: (r_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_down_fused",
    )(block_expert, gate_rm, up_rm, expert_down_kernels)
```

#### Step 3: Update the workload function

Replace the separate SwiGLU + down projection calls:

```python
# BEFORE:
# hidden_rm = _swiglu_routes_pallas(gate_rm, up_rm)
# packed_out_re = _packed_expert_down_projection_pallas(hidden_rm, block_expert, expert_down_kernels)

# AFTER:
packed_out_re = _packed_expert_down_projection_pallas(
    gate_rm, up_rm, block_expert, expert_down_kernels
)
```

#### Step 4: Delete the unused `_swiglu_routes_pallas` function

The SwiGLU kernel is no longer called from the workload.

### Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Kernel launches | 3 (input×2, swiglu, down) | 3 (input×2, down+fused) | Same count, better overlap |
| Intermediate tensors | `gate_rm`, `up_rm`, `hidden_rm` | `gate_rm`, `up_rm` | -1 large tensor |
| VMEM pressure | Higher (3 live buffers) | Lower (2 live buffers) | Less spilling |
| Memory traffic | Write `hidden_rm` + Read `hidden_rm` | Neighbors fused | Eliminated |

**Estimated speedup:** 10-20% reduction in kernel execution time due to:
- Eliminated memory traffic for `hidden_rm` (read + write eliminated)
- Fused elementwise operations execute on data already in VREGs
- Improved pipeline overlap opportunities as the critical path is shorter

### Constraints Satisfied

1. **Block shape validity:** The `gate_rm` and `up_rm` blocks use `_DOWN_BT × _DOWN_BK`, which are `128 × 512`, satisfying the divisibility by 8 and 128 requirements.

2. **VMEM budget:** The scratch buffer remains `(_DOWN_BT, _DOWN_BE)` = `(128, 1024)` for float32 = 512 KB, well within limits.

3. **Numerical equivalence:** The mathematical formula remains `down(silu(gate) * up)`, just computed with different ordering of loads. Floating-point accumulation order may differ slightly but within standard tolerance.

4. **Semantics preserved:** The `pl.when(program_id == 0)` accumulator initialization remains correct for the `"arbitrary"` reduction dimension.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# TPU v6e-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_PACK_BT = 128

assert _PACK_BT == _EXPERT_BT == _DOWN_BT


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


# ----------------------------
# Route packing
# ----------------------------

def _pack_routes(
    x2: jax.Array,                # (T, E)
    top_k_indices_tk: jax.Array,  # (T, K), int32
    router_probs_tk: jax.Array,   # (T, K), float32
):
    """
    Packs routed token/expert pairs into expert-grouped BT-sized blocks.

    Returns:
      packed_x:      (R_cap, E)
      packed_token:  (R_cap,) int32, -1 for padding rows
      packed_prob:   (R_cap,) float32, 0 for padding rows
      block_expert:  (R_cap // BT,) int32, expert id for each BT block
    """
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    assert T == T2
    N = CONFIG["num_experts"]

    R = T * K
    R_cap = int(_ceil_div(R + N * (_PACK_BT - 1), _PACK_BT) * _PACK_BT)

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)

    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)
    padded_counts = blocks_per_expert * _PACK_BT

    expert_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    num_route_blocks = R_cap // _PACK_BT
    block_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(blocks_per_expert[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    block_ids = jnp.arange(num_route_blocks, dtype=jnp.int32)[:, None]
    block_mask = (
        (block_ids >= block_offsets[None, :]) &
        (block_ids < (block_offsets + blocks_per_expert)[None, :])
    )

    block_expert = jnp.sum(
        block_mask.astype(jnp.int32) * expert_ids[None, :],
        axis=1,
        dtype=jnp.int32,
    )

    return packed_x, packed_token, packed_prob, block_expert


# ----------------------------
# Packed expert input projections
# ----------------------------

def _packed_expert_input_kernel(block_expert_ref, x_ref, w_ref, o_ref, acc_ref):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_input_projection_pallas(
    packed_x: jax.Array,        # (R_cap, E)
    block_expert: jax.Array,    # (R_cap // BT,)
    expert_kernels: jax.Array,  # (N, E, M)
) -> jax.Array:
    R_cap, E = packed_x.shape
    N, E2, M = expert_kernels.shape
    assert E == E2
    assert R_cap % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0
    del N

    return pl.pallas_call(
        _packed_expert_input_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_BT, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BK),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_EXPERT_BT, _EXPERT_BM),
                lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, m_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_input_projection",
    )(block_expert, packed_x, expert_kernels)


# ----------------------------
# Packed expert down projection with fused SwiGLU
# ----------------------------

def _packed_expert_down_fused_kernel(
    block_expert_ref, gate_ref, up_ref, down_ref, o_ref, acc_ref
):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # TPU elementwise ops are natively 32-bit; compute fused SwiGLU in f32.
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32

    acc_ref[...] += jnp.dot(
        hidden_f32,
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_down_projection_pallas(
    gate_rm: jax.Array,             # (R_cap, M)
    up_rm: jax.Array,               # (R_cap, M)
    block_expert: jax.Array,        # (R_cap // BT,)
    expert_down_kernels: jax.Array, # (N, M, E)
) -> jax.Array:
    R_cap, M = gate_rm.shape
    R_cap2, M_up = up_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert (R_cap2, M_up) == (R_cap, M)
    assert M == M2
    assert R_cap % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0
    del N

    return pl.pallas_call(
        _packed_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _DOWN_BT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        e_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_DOWN_BT, _DOWN_BE),
                lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_down_fused",
    )(block_expert, gate_rm, up_rm, expert_down_kernels)


# ----------------------------
# Scatter routed outputs back to tokens
# ----------------------------

def _scatter_routed_outputs(
    packed_out_re: jax.Array,  # (R_cap, E)
    packed_token: jax.Array,   # (R_cap,)
    packed_prob: jax.Array,    # (R_cap,)
    T: int,
) -> jax.Array:
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out


# ----------------------------
# Public workload
# ----------------------------

def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """
    Sparse MoE with route-compacted expert execution.

    Semantics match the original dense-all-experts implementation up to normal
    floating-point accumulation differences.
    """
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)

    # 2) Compact routed token/expert pairs.
    packed_x, packed_token, packed_prob, block_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K).astype(jnp.int32),
        router_probs.reshape(T, K),
    )

    # 3) Expert path only on routed rows.
    gate_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_gate_kernels
    )
    up_rm = _packed_expert_input_projection_pallas(
        packed_x, block_expert, expert_up_kernels
    )
    packed_out_re = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, block_expert, expert_down_kernels
    )

    # 4) Scatter back to token space.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=15.873,
translation_score=None,
hw_feedback=[],
plan_gen_model='zai.glm-5',
code_gen_model='gpt-5.4',
stdout='Latency: 15.873 ms\n{"correct": true, "latency": 15.873, "error": "", "all_times_ms": [15.825, 15.839, 15.842, 15.844, 15.847, 15.847, 15.848, 15.85, 15.85, 15.851, 15.851, 15.852, 15.852, 15.856, 15.856, 15.856, 15.856, 15.856, 15.858, 15.858, 15.859, 15.86, 15.86, 15.86, 15.861, 15.861, 15.861, 15.861, 15.862, 15.862, 15.862, 15.863, 15.864, 15.866, 15.866, 15.867, 15.867, 15.868, 15.868, 15.868, 15.868, 15.868, 15.869, 15.869, 15.87, 15.871, 15.871, 15.871, 15.872, 15.873, 15.873, 15.874, 15.874, 15.874, 15.875, 15.875, 15.875, 15.875, 15.875, 15.877, 15.878, 15.88, 15.88, 15.881, 15.881, 15.881, 15.881, 15.882, 15.882, 15.883, 15.884, 15.884, 15.884, 15.885, 15.885, 15.886, 15.886, 15.886, 15.887, 15.888, 15.889, 15.889, 15.889, 15.89, 15.892, 15.896, 15.897, 15.9, 15.902, 15.904, 15.904, 15.907, 15.913, 15.913, 15.915, 15.919, 15.933, 15.957, 15.982, 15.987], "max_diff": 0.062513, "max_rel_diff": 0.005623}',
stderr=''),
plan='''Looking at this code, I need to identify a new optimization strategy beyond the listed ones.

**Analysis of Current Code:**

The code implements a Sparse Mixture of Experts (MoE) layer with:
1. Router matmul to compute expert routing
2. Route packing to group tokens by expert
3. Two input projections (gate and up) - **called separately**
4. Down projection with fused SwiGLU
5. Scatter back to token space

**Key Observation:**

The `gate_rm` and `up_rm` projections are computed by two **separate** calls to `_packed_expert_input_projection_pallas`:
```python
gate_rm = _packed_expert_input_projection_pallas(packed_x, block_expert, expert_gate_kernels)
up_rm = _packed_expert_input_projection_pallas(packed_x, block_expert, expert_up_kernels)
```

Both projections:
- Read the **same input** `packed_x` from HBM
- Use the **same block_expert** indices
- Have **identical grid structure** and blocking
- Compute `(R_cap, E) @ (E, M) -> (R_cap, M)`

This means `packed_x` is loaded from HBM **twice** - once for each projection.

**New Strategy: Kernel Fusion of Parallel Independent Computations**

Fuse the gate and up projections into a **single kernel** that:
1. Loads each block of `packed_x` **once** from HBM
2. Computes both `gate_rm` and `up_rm` outputs simultaneously
3. Writes two outputs back

This reduces HBM bandwidth by ~33% for the expert input projection phase (from 3 arrays read to 2 arrays worth of HBM traffic: one for input, amortized over two outputs, instead of two separate full reads).

**How to Implement:**

1. Create a new `_packed_expert_input_kernel_fused` that:
   - Takes both `expert_gate_kernels` and `expert_up_kernels` as inputs
   - Has two output refs and two accumulator scratch buffers
   - Loads `x_ref` once and computes both matmuls

2. Adjust BlockSpecs to handle two weight tensors and two outputs:
   - Both weight tensors use the same blocking pattern
   - Both outputs use the same blocking pattern
   - Grid iterates over `(M // BM, R_cap // BT, E // BK)` once for both

3. The kernel body computes:
   ```python
   acc_gate_ref[...] += jnp.dot(x_ref[...], gate_w_ref[0, ...], ...)
   acc_up_ref[...] += jnp.dot(x_ref[...], up_w_ref[0, ...], ...)
   ```

**Performance Improvement:**

- Reduces HBM reads by one full pass over `packed_x` (R_cap × E elements)
- For Mixtral-8x7B: R_cap ≈ 4096*2 tokens, E=4096, so ~64MB saved in bf16
- At ~819 GB/s HBM bandwidth, this saves ~0.08ms per forward pass
- Also reduces kernel launch overhead by eliminating one pallas_call

**VMEM Feasibility Check:**
- Current kernel uses: `_EXPERT_BT × _EXPERT_BK` input + `_EXPERT_BK × _EXPERT_BM` weight + `_EXPERT_BT × _EXPERT_BM` output + accumulator
- With current values (BT=128, BK=512, BM=1024): 64KB + 512KB + 128KB + 128KB ≈ 832KB
- Fused kernel doubles weight and output buffers: 64KB + 1024KB + 256KB + 256KB ≈ 1.6MB
- Well within 16MB VMEM limit

This fusion strategy is distinct from the listed strategies as it targets **horizontal fusion** of independent parallel computations sharing common inputs, rather than vertical fusion (like fusing SwiGLU into down projection which is already done) or the memory hierarchy optimizations listed.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# TPU v6e-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_PACK_BT = 128

assert _PACK_BT == _EXPERT_BT == _DOWN_BT


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


# ----------------------------
# Route packing
# ----------------------------

def _pack_routes(
    x2: jax.Array,                # (T, E)
    top_k_indices_tk: jax.Array,  # (T, K), int32
    router_probs_tk: jax.Array,   # (T, K), float32
):
    """
    Packs routed token/expert pairs into expert-grouped BT-sized blocks.

    Returns:
      packed_x:      (R_cap, E)
      packed_token:  (R_cap,) int32, -1 for padding rows
      packed_prob:   (R_cap,) float32, 0 for padding rows
      block_expert:  (R_cap // BT,) int32, expert id for each BT block
    """
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    assert T == T2
    N = CONFIG["num_experts"]

    R = T * K
    R_cap = int(_ceil_div(R + N * (_PACK_BT - 1), _PACK_BT) * _PACK_BT)

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)

    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)
    padded_counts = blocks_per_expert * _PACK_BT

    expert_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    num_route_blocks = R_cap // _PACK_BT
    block_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(blocks_per_expert[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    block_ids = jnp.arange(num_route_blocks, dtype=jnp.int32)[:, None]
    block_mask = (
        (block_ids >= block_offsets[None, :]) &
        (block_ids < (block_offsets + blocks_per_expert)[None, :])
    )

    block_expert = jnp.sum(
        block_mask.astype(jnp.int32) * expert_ids[None, :],
        axis=1,
        dtype=jnp.int32,
    )

    return packed_x, packed_token, packed_prob, block_expert


# ----------------------------
# Fused expert input projections (gate and up combined)
# ----------------------------

def _packed_expert_input_fused_kernel(
    block_expert_ref, x_ref, gate_w_ref, up_w_ref, 
    gate_o_ref, up_o_ref, acc_gate_ref, acc_up_ref
):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_gate_ref[...] = jnp.zeros(acc_gate_ref.shape, dtype=jnp.float32)
        acc_up_ref[...] = jnp.zeros(acc_up_ref.shape, dtype=jnp.float32)

    # Load input once, compute both matmuls
    x_block = x_ref[...]
    
    acc_gate_ref[...] += jnp.dot(
        x_block,
        gate_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )
    
    acc_up_ref[...] += jnp.dot(
        x_block,
        up_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        gate_o_ref[...] = acc_gate_ref[...].astype(gate_o_ref.dtype)
        up_o_ref[...] = acc_up_ref[...].astype(up_o_ref.dtype)


def _packed_expert_input_projection_fused_pallas(
    packed_x: jax.Array,            # (R_cap, E)
    block_expert: jax.Array,        # (R_cap // BT,)
    expert_gate_kernels: jax.Array, # (N, E, M)
    expert_up_kernels: jax.Array,   # (N, E, M)
):
    """Fused gate and up projections - loads packed_x once for both outputs."""
    R_cap, E = packed_x.shape
    N, E2, M = expert_gate_kernels.shape
    assert E == E2
    assert expert_up_kernels.shape == (N, E, M)
    assert R_cap % _EXPERT_BT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0
    del N

    gate_rm, up_rm = pl.pallas_call(
        _packed_expert_input_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_BT, E // _EXPERT_BK),
            in_specs=[
                # x_ref - same input for both projections
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BK),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                # gate_w_ref
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
                # up_w_ref
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
            ],
            out_specs=[
                # gate_o_ref
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, m_blk),
                ),
                # up_o_ref
                pl.BlockSpec(
                    (_EXPERT_BT, _EXPERT_BM),
                    lambda m_blk, r_blk, k_blk, block_expert_smem: (r_blk, m_blk),
                ),
            ],
            scratch_shapes=[
                pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32),  # acc_gate
                pltpu.VMEM((_EXPERT_BT, _EXPERT_BM), jnp.float32),  # acc_up
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),  # gate_rm
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),  # up_rm
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_input_fused",
    )(block_expert, packed_x, expert_gate_kernels, expert_up_kernels)
    
    return gate_rm, up_rm


# ----------------------------
# Packed expert down projection with fused SwiGLU
# ----------------------------

def _packed_expert_down_fused_kernel(
    block_expert_ref, gate_ref, up_ref, down_ref, o_ref, acc_ref
):
    del block_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # TPU elementwise ops are natively 32-bit; compute fused SwiGLU in f32.
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32

    acc_ref[...] += jnp.dot(
        hidden_f32,
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_down_projection_pallas(
    gate_rm: jax.Array,             # (R_cap, M)
    up_rm: jax.Array,               # (R_cap, M)
    block_expert: jax.Array,        # (R_cap // BT,)
    expert_down_kernels: jax.Array, # (N, M, E)
) -> jax.Array:
    R_cap, M = gate_rm.shape
    R_cap2, M_up = up_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert (R_cap2, M_up) == (R_cap, M)
    assert M == M2
    assert R_cap % _DOWN_BT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0
    del N

    return pl.pallas_call(
        _packed_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _DOWN_BT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (_DOWN_BT, _DOWN_BK),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, r_blk, k_blk, block_expert_smem: (
                        block_expert_smem[r_blk],
                        k_blk,
                        e_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_DOWN_BT, _DOWN_BE),
                lambda e_blk, r_blk, k_blk, block_expert_smem: (r_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_DOWN_BT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="packed_expert_down_fused",
    )(block_expert, gate_rm, up_rm, expert_down_kernels)


# ----------------------------
# Scatter routed outputs back to tokens
# ----------------------------

def _scatter_routed_outputs(
    packed_out_re: jax.Array,  # (R_cap, E)
    packed_token: jax.Array,   # (R_cap,)
    packed_prob: jax.Array,    # (R_cap,)
    T: int,
) -> jax.Array:
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out


# ----------------------------
# Public workload
# ----------------------------

def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """
    Sparse MoE with route-compacted expert execution.

    Semantics match the original dense-all-experts implementation up to normal
    floating-point accumulation differences.
    """
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)

    # 2) Compact routed token/expert pairs.
    packed_x, packed_token, packed_prob, block_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K).astype(jnp.int32),
        router_probs.reshape(T, K),
    )

    # 3) Expert path only on routed rows - FUSED gate and up projections
    gate_rm, up_rm = _packed_expert_input_projection_fused_pallas(
        packed_x, block_expert, expert_gate_kernels, expert_up_kernels
    )
    
    packed_out_re = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, block_expert, expert_down_kernels
    )

    # 4) Scatter back to token space.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=15.272,
translation_score=None,
hw_feedback=[],
plan_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 15.272 ms\n{"correct": true, "latency": 15.272, "error": "", "all_times_ms": [14.877, 14.88, 14.92, 14.924, 14.947, 14.986, 14.997, 15.006, 15.015, 15.025, 15.034, 15.043, 15.057, 15.063, 15.063, 15.067, 15.073, 15.074, 15.086, 15.094, 15.095, 15.096, 15.106, 15.107, 15.112, 15.116, 15.12, 15.154, 15.158, 15.16, 15.166, 15.169, 15.201, 15.204, 15.21, 15.211, 15.214, 15.215, 15.217, 15.218, 15.221, 15.221, 15.227, 15.244, 15.246, 15.247, 15.25, 15.251, 15.264, 15.264, 15.272, 15.28, 15.283, 15.288, 15.294, 15.299, 15.299, 15.307, 15.308, 15.31, 15.313, 15.321, 15.329, 15.339, 15.34, 15.342, 15.344, 15.348, 15.35, 15.355, 15.356, 15.357, 15.362, 15.365, 15.366, 15.368, 15.374, 15.374, 15.38, 15.38, 15.381, 15.386, 15.387, 15.394, 15.394, 15.398, 15.399, 15.4, 15.433, 15.435, 15.441, 15.454, 15.458, 15.468, 15.473, 15.505, 15.508, 15.531, 15.543, 15.592], "max_diff": 0.062513, "max_rel_diff": 0.005623}',
stderr=''),
plan='''**Selected strategy: 11 — reuse the loaded weight block across multiple consecutive routed blocks with the same expert**

### Why this is the best single change here

The code already does the most obvious fusion: gate+up are fused, and down includes fused SwiGLU.  
The remaining big inefficiency is **expert-weight rereads**.

After `_pack_routes`, routed rows are already **grouped by expert**. But the current expert kernels still run one `_EXPERT_BT=128` token block at a time:

- input fused kernel grid: `(M // 1024, R_cap // 128, E // 512)`
- down fused kernel grid: `(E // 1024, R_cap // 128, M // 512)`

So if expert `e` owns several consecutive route blocks, the same expert weight tile is fetched again for each 128-row block. On v6e-1, that is expensive because the expert weights are enormous and HBM traffic dominates.

---

## Plan

### 1) Change route packing so expert ranges are grouped in larger chunks
Add a fixed grouping factor, e.g.

```python
_EXPERT_RB = 4          # consecutive 128-row route blocks per expert group
_EXPERT_GBT = _EXPERT_RB * _EXPERT_BT   # 512 rows
```

Then modify `_pack_routes` so each expert is padded not just to a multiple of `_PACK_BT`, but to a multiple of `_EXPERT_RB * _PACK_BT`.

Today you compute:

- `blocks_per_expert = ceil_div(counts, _PACK_BT)`
- `padded_counts = blocks_per_expert * _PACK_BT`

Change that to:

- `groups_per_expert = ceil_div(blocks_per_expert, _EXPERT_RB)`
- `padded_blocks_per_expert = groups_per_expert * _EXPERT_RB`
- `padded_counts = padded_blocks_per_expert * _PACK_BT`

Also return a new metadata array:

- `group_expert: (num_groups,) int32`

where `num_groups = sum(groups_per_expert)`, and each entry tells which expert owns that 512-row packed group.

This preserves semantics because the extra rows are still padding:
- `packed_token = -1`
- `packed_prob = 0`
- `packed_x = 0`

So scatter behavior is unchanged up to normal floating-point tolerance.

---

### 2) Rewrite the fused gate/up kernel to operate on a whole expert-group tile
Instead of processing one `(128, 512)` activation tile per routed block, process one `(512, 512)` activation tile for a **single expert group**.

#### New logical kernel shape
For the input projections, change from:

- `x_ref`: `(_EXPERT_BT, _EXPERT_BK)` = `(128, 512)`

to:

- `x_ref`: `(_EXPERT_GBT, _EXPERT_BK)` = `(512, 512)`

and similarly:

- `gate_o_ref`: `(512, 1024)`
- `up_o_ref`: `(512, 1024)`
- scratch accumulators: `(512, 1024)` f32

#### New grid
Use:

```python
grid = (M // _EXPERT_BM, num_groups, E // _EXPERT_BK)
```

with the reduction axis still last, so accumulator lifetime remains correct.

#### New indexing
Use `group_expert` in scalar-prefetch SMEM exactly like the sparse examples use index tables:

- `x_ref` block comes from packed rows for group `g`
- `gate_w_ref` and `up_w_ref` both index expert `group_expert[g]`

Conceptually:

```python
x_spec      -> BlockSpec((_EXPERT_GBT, _EXPERT_BK), lambda m, g, k, ge: (g, k))
gate_w_spec -> BlockSpec((1, _EXPERT_BK, _EXPERT_BM),
                         lambda m, g, k, ge: (ge[g], k, m))
up_w_spec   -> same
out_spec    -> BlockSpec((_EXPERT_GBT, _EXPERT_BM), lambda m, g, k, ge: (g, m))
```

Because `_EXPERT_GBT = 512`, all block-shape rules are still TPU-legal:
- second-to-last dim divisible by 8
- last dim divisible by 128

#### Kernel body
The kernel itself stays simple and legal:

- zero scratch when `program_id(2) == 0`
- load `x_ref[...]` once
- do two `jnp.dot` calls using the same grouped input tile
- store on last `k`

So the main improvement is **not more math fusion**, but **weight-stationary reuse** across 4 consecutive routed blocks belonging to the same expert.

---

### 3) Apply the same grouped schedule to the down-projection kernel
Do the identical transformation for `_packed_expert_down_fused_kernel`:

- grouped gate tile: `(512, 512)`
- grouped up tile: `(512, 512)`
- down weight tile: `(1, 512, 1024)`
- output/accumulator tile: `(512, 1024)`

Grid becomes:

```python
grid = (E // _DOWN_BE, num_groups, M // _DOWN_BK)
```

with `k = M // _DOWN_BK` still the last axis.

Inside the kernel:

- upcast `gate_ref[...]` and `up_ref[...]` to f32
- compute `hidden_f32 = silu(gate_f32) * up_f32`
- multiply once against the expert’s down-weight tile for the whole 512-row group
- store on the last reduction step

Again, the key gain is that the expert’s down weights are loaded once per 512 routed rows, not once per 128 routed rows.

---

### 4) Keep the public API and semantics unchanged
Do **not** change:

```python
def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
```

Only internal helpers change.

You can replace `block_expert` with `group_expert` throughout the expert path, but the external signature and output remain the same.

---

## Why this should help on TPU v6e-1

For this specific config:

- `T = 2048`
- `K = 2`
- routed rows `R = 4096`
- `BT = 128` → about `32` route blocks total
- `N = 8` experts → about `4` route blocks/expert on average

So `_EXPERT_RB = 4` is a very natural grouping for this exact workload.

That means, approximately:

- gate weights: up to **4x fewer** HBM fetches for repeated expert tiles
- up weights: same
- down weights: same

This is especially attractive on v6e-1 because there is only one TensorCore per chip, so reducing redundant HBM traffic is often more valuable than trying to expose more parallelism.

---

## Why it remains correct

This is semantically equivalent because:

- routed rows are still grouped by expert in the same order
- only the **padding granularity** changes
- all newly added padded rows stay inert:
  - token = `-1`
  - probability = `0`
  - activation = `0`
- matmul accumulation order changes only slightly, so output differences should remain within normal floating-point tolerance

---

## Expected implementation impact

### Functions to change
- `_pack_routes`
- `_packed_expert_input_fused_kernel`
- `_packed_expert_input_projection_fused_pallas`
- `_packed_expert_down_fused_kernel`
- `_packed_expert_down_projection_pallas`

### New helper metadata
- `group_expert`
- optionally `num_groups`

### Functions that can remain unchanged
- `create_inputs`
- `_router_kernel`
- `_router_pallas_matmul`
- `_scatter_routed_outputs`
- `workload` signature

---

## One-line summary

**Repack routed tokens into larger expert-homogeneous groups (e.g. 4 consecutive 128-row blocks), then run the fused expert kernels on those grouped tiles so each expert weight tile is loaded once and reused across multiple routed blocks.**''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# TPU v6e-friendly tile sizes.
_ROUTER_BT = 128
_ROUTER_BK = 512

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 512
_DOWN_BE = 1024

_PACK_BT = 128

# New grouping factor: process 4 consecutive routed 128-row blocks for the same
# expert as a single 512-row expert-homogeneous group.
_EXPERT_RB = 4
_EXPERT_GBT = _EXPERT_RB * _EXPERT_BT

assert _PACK_BT == _EXPERT_BT == _DOWN_BT
assert _EXPERT_GBT % 8 == 0
assert _EXPERT_BK % 128 == 0
assert _EXPERT_BM % 128 == 0
assert _DOWN_BK % 128 == 0
assert _DOWN_BE % 128 == 0


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


# ----------------------------
# Route packing with expert-grouped 512-row chunks
# ----------------------------

def _pack_routes(
    x2: jax.Array,                # (T, E)
    top_k_indices_tk: jax.Array,  # (T, K), int32
    router_probs_tk: jax.Array,   # (T, K), float32
):
    """
    Packs routed token/expert pairs into expert-grouped `_EXPERT_GBT`-row chunks.

    Returns:
      packed_x:      (R_cap, E)
      packed_token:  (R_cap,) int32, -1 for padding rows
      packed_prob:   (R_cap,) float32, 0 for padding rows
      group_expert:  (R_cap // _EXPERT_GBT,) int32, expert id for each group
    """
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    assert T == T2
    N = CONFIG["num_experts"]

    R = T * K

    # Static-capacity upper bound on number of expert-homogeneous groups.
    # For nonnegative integers c_i with sum R:
    #   sum_i ceil(c_i / G) <= ceil(R / G) + N - 1
    # This keeps shapes static while avoiding the looser +N*(G-1) bound.
    num_group_slots = int(_ceil_div(R, _EXPERT_GBT) + (N - 1))
    R_cap = num_group_slots * _EXPERT_GBT

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)

    # Rank of each routed row within its expert.
    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)
    groups_per_expert = _ceil_div(blocks_per_expert, _EXPERT_RB)
    padded_blocks_per_expert = groups_per_expert * _EXPERT_RB
    padded_counts = padded_blocks_per_expert * _PACK_BT

    expert_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    group_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(groups_per_expert[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    group_ids = jnp.arange(num_group_slots, dtype=jnp.int32)[:, None]
    group_mask = (
        (group_ids >= group_offsets[None, :]) &
        (group_ids < (group_offsets + groups_per_expert)[None, :])
    )

    # Trailing unused groups (if any) default to expert 0; their rows are all
    # padded, so this is semantically inert.
    group_expert = jnp.sum(
        group_mask.astype(jnp.int32) * expert_ids[None, :],
        axis=1,
        dtype=jnp.int32,
    )

    return packed_x, packed_token, packed_prob, group_expert


# ----------------------------
# Fused expert input projections over grouped routed tiles
# ----------------------------

def _grouped_expert_input_fused_kernel(
    group_expert_ref,
    x_ref,
    gate_w_ref,
    up_w_ref,
    gate_o_ref,
    up_o_ref,
    acc_gate_ref,
    acc_up_ref,
):
    del group_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_gate_ref[...] = jnp.zeros(acc_gate_ref.shape, dtype=jnp.float32)
        acc_up_ref[...] = jnp.zeros(acc_up_ref.shape, dtype=jnp.float32)

    x_block = x_ref[...]

    acc_gate_ref[...] += jnp.dot(
        x_block,
        gate_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    acc_up_ref[...] += jnp.dot(
        x_block,
        up_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        gate_o_ref[...] = acc_gate_ref[...].astype(gate_o_ref.dtype)
        up_o_ref[...] = acc_up_ref[...].astype(up_o_ref.dtype)


def _packed_expert_input_projection_fused_pallas(
    packed_x: jax.Array,            # (R_cap, E)
    group_expert: jax.Array,        # (R_cap // _EXPERT_GBT,)
    expert_gate_kernels: jax.Array, # (N, E, M)
    expert_up_kernels: jax.Array,   # (N, E, M)
):
    """
    Fused gate+up projections over grouped routed rows.

    Each grid iteration in the group dimension processes `_EXPERT_GBT` rows for
    one expert, so the same expert weight tile is reused across 4 consecutive
    128-row routed blocks.
    """
    R_cap, E = packed_x.shape
    N, E2, M = expert_gate_kernels.shape
    assert E == E2
    assert expert_up_kernels.shape == (N, E, M)
    assert R_cap % _EXPERT_GBT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0
    assert group_expert.shape == (R_cap // _EXPERT_GBT,)
    del N

    gate_rm, up_rm = pl.pallas_call(
        _grouped_expert_input_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_GBT, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BK),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, m_blk),
                ),
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, m_blk),
                ),
            ],
            scratch_shapes=[
                pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32),
                pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="grouped_expert_input_fused",
    )(group_expert, packed_x, expert_gate_kernels, expert_up_kernels)

    return gate_rm, up_rm


# ----------------------------
# Grouped expert down projection with fused SwiGLU
# ----------------------------

def _grouped_expert_down_fused_kernel(
    group_expert_ref,
    gate_ref,
    up_ref,
    down_ref,
    o_ref,
    acc_ref,
):
    del group_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # TPU elementwise ops are natively 32-bit; upcast before elementwise work.
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32

    acc_ref[...] += jnp.dot(
        hidden_f32,
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_down_projection_pallas(
    gate_rm: jax.Array,             # (R_cap, M)
    up_rm: jax.Array,               # (R_cap, M)
    group_expert: jax.Array,        # (R_cap // _EXPERT_GBT,)
    expert_down_kernels: jax.Array, # (N, M, E)
) -> jax.Array:
    R_cap, M = gate_rm.shape
    R_cap2, M_up = up_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert (R_cap2, M_up) == (R_cap, M)
    assert M == M2
    assert R_cap % _EXPERT_GBT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0
    assert group_expert.shape == (R_cap // _EXPERT_GBT,)
    del N

    return pl.pallas_call(
        _grouped_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _EXPERT_GBT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _DOWN_BK),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (_EXPERT_GBT, _DOWN_BK),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        e_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_EXPERT_GBT, _DOWN_BE),
                lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="grouped_expert_down_fused",
    )(group_expert, gate_rm, up_rm, expert_down_kernels)


# ----------------------------
# Scatter routed outputs back to tokens
# ----------------------------

def _scatter_routed_outputs(
    packed_out_re: jax.Array,  # (R_cap, E)
    packed_token: jax.Array,   # (R_cap,)
    packed_prob: jax.Array,    # (R_cap,)
    T: int,
) -> jax.Array:
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out


# ----------------------------
# Public workload
# ----------------------------

def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """
    Sparse MoE with route-compacted expert execution.

    Semantics match the original implementation up to normal floating-point
    accumulation differences.
    """
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)

    # 2) Compact routed token/expert pairs into grouped expert-homogeneous chunks.
    packed_x, packed_token, packed_prob, group_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K).astype(jnp.int32),
        router_probs.reshape(T, K),
    )

    # 3) Expert path only on routed rows, using grouped schedule to reuse weight
    #    tiles across consecutive routed blocks of the same expert.
    gate_rm, up_rm = _packed_expert_input_projection_fused_pallas(
        packed_x, group_expert, expert_gate_kernels, expert_up_kernels
    )

    packed_out_re = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, group_expert, expert_down_kernels
    )

    # 4) Scatter back to token space.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=8.314,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='gpt-5.4',
stdout='Latency: 8.314 ms\n{"correct": true, "latency": 8.314, "error": "", "all_times_ms": [8.284, 8.291, 8.294, 8.294, 8.295, 8.295, 8.297, 8.299, 8.299, 8.299, 8.299, 8.299, 8.299, 8.3, 8.3, 8.3, 8.302, 8.302, 8.304, 8.304, 8.304, 8.305, 8.305, 8.305, 8.305, 8.306, 8.306, 8.307, 8.307, 8.307, 8.307, 8.307, 8.308, 8.308, 8.309, 8.309, 8.309, 8.309, 8.309, 8.31, 8.31, 8.311, 8.311, 8.311, 8.311, 8.311, 8.311, 8.312, 8.313, 8.313, 8.314, 8.316, 8.316, 8.316, 8.317, 8.317, 8.317, 8.318, 8.318, 8.319, 8.319, 8.32, 8.32, 8.321, 8.323, 8.326, 8.328, 8.328, 8.329, 8.331, 8.331, 8.332, 8.332, 8.333, 8.337, 8.342, 8.343, 8.344, 8.345, 8.35, 8.353, 8.355, 8.359, 8.359, 8.366, 8.368, 8.372, 8.372, 8.389, 8.392, 8.394, 8.41, 8.417, 8.418, 8.423, 8.423, 8.449, 8.464, 8.466, 8.494], "max_diff": 0.062513, "max_rel_diff": 0.005623}',
stderr=''),
plan='''**Selected strategy: 3 — reduce grid iterations by using larger blocks to minimize pipeline bubble overhead.**

### Why this is the right fit for this code on v6e-1
The biggest avoidable cost in this implementation is the number of **sequential reduction-grid iterations** in the Pallas matmuls:

- `_router_pallas_matmul` runs with:
  - `T // 128 = 16` token tiles
  - `E // 512 = 8` reduction tiles  
  - total: **128 kernel iterations**

- `_packed_expert_down_projection_pallas` runs with:
  - `E_out // 1024 = 4`
  - `R_cap // 512 ≈ 8–15`
  - `M // 512 = 28` reduction tiles  
  - the last axis is reduction and sequential, so **28 passes per output tile** is expensive.

On TPU v6e, each extra reduction-step carries pipeline prologue/epilogue and repeated HBM↔VMEM traffic scheduling overhead. The code already keeps the reduction axis innermost, which is good; the remaining opportunity is to **make that axis shorter** where VMEM allows it.

---

## Concrete plan

### 1) Make the router matmul much coarser
Increase the router tile sizes aggressively, because this kernel is VMEM-light:

- Change:
  - `_ROUTER_BT: 128 -> 512`
  - `_ROUTER_BK: 512 -> 4096`

This changes router execution from:

- old grid: `(2048/128, 4096/512) = (16, 8)` → **128 iterations**
- new grid: `(2048/512, 4096/4096) = (4, 1)` → **4 iterations**

No kernel-body rewrite is needed; `_router_kernel` can stay the same. Only the constants and resulting `BlockSpec`s / `grid` change.

### 2) Enlarge only the reduction block in the expert down projection
This is the densest sequential loop in the MoE path and should be shortened, but only within VMEM budget.

- Change:
  - `_DOWN_BK: 512 -> 1024`

Keep:
- `_DOWN_BE = 1024`
- `_EXPERT_GBT = 512`

This changes the down-projection reduction depth from:

- old: `14336 / 512 = 28`
- new: `14336 / 1024 = 14`

So the kernel still accumulates correctly over the innermost grid axis, but with **half as many sequential reduction steps**.

### 3) Do **not** enlarge the fused gate/up kernel in this phase
Leave these unchanged:

- `_EXPERT_BK = 512`
- `_EXPERT_BM = 1024`
- `_EXPERT_GBT = 512`

Reason: that kernel already holds **two f32 accumulators** plus two weight streams plus two outputs. On v6e-1, pushing its block sizes up is much more likely to exceed VMEM or cause spills, which would erase the benefit.

---

## VMEM sanity check for v6e-1

### Router kernel with `(BT, BK) = (512, 4096)`
Approximate live storage per buffer slot:

- `x_ref`: `512 x 4096 x bf16` ≈ **4 MiB**
- `w_ref`: `4096 x 8 x bf16` ≈ **64 KiB**
- `acc_ref`: `512 x 8 x f32` ≈ **16 KiB**
- `o_ref`: tiny

Even with double buffering on inputs, this is still comfortably within ~16 MiB VMEM.

### Down kernel with `_DOWN_BK = 1024`
Per slot:

- `gate_ref`: `512 x 1024 x bf16` ≈ **1 MiB**
- `up_ref`: `512 x 1024 x bf16` ≈ **1 MiB**
- `down_ref`: `1 x 1024 x 1024 x bf16` ≈ **2 MiB**
- `o_ref`: `512 x 1024 x bf16` ≈ **1 MiB**
- `acc_ref`: `512 x 1024 x f32` ≈ **2 MiB**

With default double buffering for inputs/outputs, this stays around **12 MiB + compiler overhead**, which is reasonable for v6e-1.

---

## What code changes to make
Only adjust the tiling constants:

```python
_ROUTER_BT = 512
_ROUTER_BK = 4096

_DOWN_BK = 1024
```

Keep the function names and signatures exactly the same. The existing `pallas_call` definitions will automatically pick up the larger `BlockSpec`s and smaller grids from those constants.

---

## Why this should help latency
This plan reduces overhead without changing algorithmic semantics:

- Router: **128 → 4** grid iterations
- Expert down reduction depth: **28 → 14**

That directly cuts:

- pipeline bubble overhead,
- repeated DMA setup/work per tile,
- number of sequential accumulator updates.

Since the arithmetic is unchanged, the result stays semantically equivalent, with only small floating-point differences from the changed reduction chunking order.

---

## Scope control
This plan uses **exactly one strategy**: **larger blocks to reduce grid iterations**.  
It does **not** change routing layout, scatter behavior, or accumulation structure in this phase.''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = (
        CONFIG["batch"],
        CONFIG["seq_len"],
        CONFIG["emb_dim"],
        CONFIG["mlp_dim"],
    )
    N = CONFIG["num_experts"]
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


# TPU v6e-friendly tile sizes - OPTIMIZED for larger blocks to reduce grid iterations
_ROUTER_BT = 512   # Increased from 128 to reduce grid iterations
_ROUTER_BK = 4096  # Increased from 512 to reduce reduction depth to 1

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 1024    # Increased from 512 to halve reduction depth
_DOWN_BE = 1024

_PACK_BT = 128

# Grouping factor: process 4 consecutive routed 128-row blocks for the same
# expert as a single 512-row expert-homogeneous group.
_EXPERT_RB = 4
_EXPERT_GBT = _EXPERT_RB * _EXPERT_BT

assert _PACK_BT == _EXPERT_BT == _DOWN_BT
assert _EXPERT_GBT % 8 == 0
assert _EXPERT_BK % 128 == 0
assert _EXPERT_BM % 128 == 0
assert _DOWN_BK % 128 == 0
assert _DOWN_BE % 128 == 0


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    acc_ref[...] += jnp.dot(
        x_ref[...],
        w_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    E2, N = router_weights.shape
    assert E == E2
    assert T % _ROUTER_BT == 0
    assert E % _ROUTER_BK == 0

    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec(
                (_ROUTER_BT, _ROUTER_BK),
                lambda t_blk, k_blk: (t_blk, k_blk),
            ),
            pl.BlockSpec(
                (_ROUTER_BK, N),
                lambda t_blk, k_blk: (k_blk, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (_ROUTER_BT, N),
            lambda t_blk, k_blk: (t_blk, 0),
        ),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        name="router_matmul",
    )(x2, router_weights)


# ----------------------------
# Route packing with expert-grouped 512-row chunks
# ----------------------------

def _pack_routes(
    x2: jax.Array,                # (T, E)
    top_k_indices_tk: jax.Array,  # (T, K), int32
    router_probs_tk: jax.Array,   # (T, K), float32
):
    """
    Packs routed token/expert pairs into expert-grouped `_EXPERT_GBT`-row chunks.

    Returns:
      packed_x:      (R_cap, E)
      packed_token:  (R_cap,) int32, -1 for padding rows
      packed_prob:   (R_cap,) float32, 0 for padding rows
      group_expert:  (R_cap // _EXPERT_GBT,) int32, expert id for each group
    """
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    assert T == T2
    N = CONFIG["num_experts"]

    R = T * K

    # Static-capacity upper bound on number of expert-homogeneous groups.
    num_group_slots = int(_ceil_div(R, _EXPERT_GBT) + (N - 1))
    R_cap = num_group_slots * _EXPERT_GBT

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)

    # Rank of each routed row within its expert.
    route_rank = jnp.sum(
        (jnp.cumsum(one_hot, axis=0) - 1) * one_hot,
        axis=1,
        dtype=jnp.int32,
    )

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)
    groups_per_expert = _ceil_div(blocks_per_expert, _EXPERT_RB)
    padded_blocks_per_expert = groups_per_expert * _EXPERT_RB
    padded_counts = padded_blocks_per_expert * _PACK_BT

    expert_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)

    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = x2[gather_idx]
    packed_x = packed_x * valid[:, None].astype(x2.dtype)

    group_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(groups_per_expert[:-1], dtype=jnp.int32),
        ],
        axis=0,
    )

    group_ids = jnp.arange(num_group_slots, dtype=jnp.int32)[:, None]
    group_mask = (
        (group_ids >= group_offsets[None, :]) &
        (group_ids < (group_offsets + groups_per_expert)[None, :])
    )

    group_expert = jnp.sum(
        group_mask.astype(jnp.int32) * expert_ids[None, :],
        axis=1,
        dtype=jnp.int32,
    )

    return packed_x, packed_token, packed_prob, group_expert


# ----------------------------
# Fused expert input projections over grouped routed tiles
# ----------------------------

def _grouped_expert_input_fused_kernel(
    group_expert_ref,
    x_ref,
    gate_w_ref,
    up_w_ref,
    gate_o_ref,
    up_o_ref,
    acc_gate_ref,
    acc_up_ref,
):
    del group_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_gate_ref[...] = jnp.zeros(acc_gate_ref.shape, dtype=jnp.float32)
        acc_up_ref[...] = jnp.zeros(acc_up_ref.shape, dtype=jnp.float32)

    x_block = x_ref[...]

    acc_gate_ref[...] += jnp.dot(
        x_block,
        gate_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    acc_up_ref[...] += jnp.dot(
        x_block,
        up_w_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        gate_o_ref[...] = acc_gate_ref[...].astype(gate_o_ref.dtype)
        up_o_ref[...] = acc_up_ref[...].astype(up_o_ref.dtype)


def _packed_expert_input_projection_fused_pallas(
    packed_x: jax.Array,            # (R_cap, E)
    group_expert: jax.Array,        # (R_cap // _EXPERT_GBT,)
    expert_gate_kernels: jax.Array, # (N, E, M)
    expert_up_kernels: jax.Array,   # (N, E, M)
):
    """
    Fused gate+up projections over grouped routed rows.
    """
    R_cap, E = packed_x.shape
    N, E2, M = expert_gate_kernels.shape
    assert E == E2
    assert expert_up_kernels.shape == (N, E, M)
    assert R_cap % _EXPERT_GBT == 0
    assert E % _EXPERT_BK == 0
    assert M % _EXPERT_BM == 0
    assert group_expert.shape == (R_cap // _EXPERT_GBT,)
    del N

    gate_rm, up_rm = pl.pallas_call(
        _grouped_expert_input_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, R_cap // _EXPERT_GBT, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BK),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
                pl.BlockSpec(
                    (1, _EXPERT_BK, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        m_blk,
                    ),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, m_blk),
                ),
                pl.BlockSpec(
                    (_EXPERT_GBT, _EXPERT_BM),
                    lambda m_blk, g_blk, k_blk, group_expert_smem: (g_blk, m_blk),
                ),
            ],
            scratch_shapes=[
                pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32),
                pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
            jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="grouped_expert_input_fused",
    )(group_expert, packed_x, expert_gate_kernels, expert_up_kernels)

    return gate_rm, up_rm


# ----------------------------
# Grouped expert down projection with fused SwiGLU
# ----------------------------

def _grouped_expert_down_fused_kernel(
    group_expert_ref,
    gate_ref,
    up_ref,
    down_ref,
    o_ref,
    acc_ref,
):
    del group_expert_ref

    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    # TPU elementwise ops are natively 32-bit; upcast before elementwise work.
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    hidden_f32 = jax.nn.silu(gate_f32) * up_f32

    acc_ref[...] += jnp.dot(
        hidden_f32,
        down_ref[0, ...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _packed_expert_down_projection_pallas(
    gate_rm: jax.Array,             # (R_cap, M)
    up_rm: jax.Array,               # (R_cap, M)
    group_expert: jax.Array,        # (R_cap // _EXPERT_GBT,)
    expert_down_kernels: jax.Array, # (N, M, E)
) -> jax.Array:
    R_cap, M = gate_rm.shape
    R_cap2, M_up = up_rm.shape
    N, M2, E = expert_down_kernels.shape
    assert (R_cap2, M_up) == (R_cap, M)
    assert M == M2
    assert R_cap % _EXPERT_GBT == 0
    assert M % _DOWN_BK == 0
    assert E % _DOWN_BE == 0
    assert group_expert.shape == (R_cap // _EXPERT_GBT,)
    del N

    return pl.pallas_call(
        _grouped_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, R_cap // _EXPERT_GBT, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec(
                    (_EXPERT_GBT, _DOWN_BK),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (_EXPERT_GBT, _DOWN_BK),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, k_blk),
                ),
                pl.BlockSpec(
                    (1, _DOWN_BK, _DOWN_BE),
                    lambda e_blk, g_blk, k_blk, group_expert_smem: (
                        group_expert_smem[g_blk],
                        k_blk,
                        e_blk,
                    ),
                ),
            ],
            out_specs=pl.BlockSpec(
                (_EXPERT_GBT, _DOWN_BE),
                lambda e_blk, g_blk, k_blk, group_expert_smem: (g_blk, e_blk),
            ),
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="grouped_expert_down_fused",
    )(group_expert, gate_rm, up_rm, expert_down_kernels)


# ----------------------------
# Scatter routed outputs back to tokens
# ----------------------------

def _scatter_routed_outputs(
    packed_out_re: jax.Array,  # (R_cap, E)
    packed_token: jax.Array,   # (R_cap,)
    packed_prob: jax.Array,    # (R_cap,)
    T: int,
) -> jax.Array:
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out


# ----------------------------
# Public workload
# ----------------------------

def workload(
    x,
    router_weights,
    expert_gate_kernels,
    expert_up_kernels,
    expert_down_kernels,
):
    """
    Sparse MoE with route-compacted expert execution.

    Semantics match the original implementation up to normal floating-point
    accumulation differences.
    """
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG["num_experts_per_tok"]
    T = B * S

    x2 = x.reshape(T, E)

    # 1) Router projection: (T, E) @ (E, N) -> (T, N)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, N)
    logits_f32 = logits.astype(jnp.float32)

    # Routing in regular JAX.
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)

    # 2) Compact routed token/expert pairs into grouped expert-homogeneous chunks.
    packed_x, packed_token, packed_prob, group_expert = _pack_routes(
        x2,
        top_k_indices.reshape(T, K).astype(jnp.int32),
        router_probs.reshape(T, K),
    )

    # 3) Expert path only on routed rows, using grouped schedule to reuse weight
    #    tiles across consecutive routed blocks of the same expert.
    gate_rm, up_rm = _packed_expert_input_projection_fused_pallas(
        packed_x, group_expert, expert_gate_kernels, expert_up_kernels
    )

    packed_out_re = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, group_expert, expert_down_kernels
    )

    # 4) Scatter back to token space.
    output_te = _scatter_routed_outputs(
        packed_out_re,
        packed_token,
        packed_prob,
        T,
    )

    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=8.199,
translation_score=None,
hw_feedback=[],
plan_gen_model='gpt-5.4',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 8.199 ms\n{"correct": true, "latency": 8.199, "error": "", "all_times_ms": [7.845, 7.974, 7.976, 7.999, 8.019, 8.023, 8.024, 8.025, 8.027, 8.031, 8.051, 8.054, 8.058, 8.066, 8.069, 8.07, 8.087, 8.088, 8.09, 8.091, 8.094, 8.097, 8.098, 8.101, 8.103, 8.113, 8.113, 8.116, 8.124, 8.124, 8.125, 8.126, 8.135, 8.138, 8.14, 8.141, 8.143, 8.16, 8.16, 8.165, 8.169, 8.171, 8.171, 8.171, 8.173, 8.177, 8.179, 8.182, 8.188, 8.196, 8.199, 8.203, 8.205, 8.213, 8.213, 8.215, 8.222, 8.228, 8.235, 8.24, 8.245, 8.249, 8.255, 8.261, 8.262, 8.266, 8.267, 8.267, 8.27, 8.273, 8.274, 8.275, 8.276, 8.28, 8.282, 8.283, 8.285, 8.291, 8.303, 8.305, 8.318, 8.32, 8.32, 8.322, 8.339, 8.343, 8.344, 8.346, 8.348, 8.351, 8.365, 8.365, 8.371, 8.387, 8.4, 8.401, 8.403, 8.407, 8.416, 8.49], "max_diff": 0.054258, "max_rel_diff": 0.004881}',
stderr=''),
plan='''### Optimization Plan

The current implementation uses a static upper bound for the number of expert-homogeneous groups (`num_group_slots`) to define the Pallas grid size. This leads to inefficient execution because the kernel iterates over many entirely padded groups that contain no routed tokens, wasting TensorCore cycles. 

My plan is to apply **Strategy 9** (Compute the exact number of routed expert groups and launch only those groups) to minimize redundant work:

1.  **Dynamically Calculate Grid Size**: In `_pack_routes`, I will calculate the exact number of 512-row groups required for each expert based on the actual router decisions (`groups_per_expert`). I will then compute the `total_groups` as the sum of these values.
2.  **Optimize Group-to-Expert Mapping**: I will replace the mask-based mapping logic with a more efficient `jnp.repeat` approach. This will create a `group_expert` array that maps each global group index (from `0` to `total_groups - 1`) directly to its corresponding `expert_id`.
3.  **Update Pallas Call Grids**: I will modify `_packed_expert_input_projection_fused_pallas` and `_packed_expert_down_projection_pallas` to accept `total_groups` as a parameter. This value will be used to define the second dimension of the Pallas grid, ensuring that the kernel only executes iterations for groups that actually contain data.
4.  **Preserve Static Shapes**: I will maintain a static upper bound for `num_group_slots` and `R_cap` to ensure that array shapes and `ShapeDtypeStruct` definitions remain compatible with JAX JIT compilation, while using the dynamic `total_groups` to control the hardware execution loop.

### Optimized Code

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}

_ROUTER_BT = 512
_ROUTER_BK = 4096

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 1024
_DOWN_BE = 1024

_PACK_BT = 128
_EXPERT_RB = 4
_EXPERT_GBT = _EXPERT_RB * _EXPERT_BT

def _ceil_div(a, b):
    return (a + b - 1) // b

# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    _, N = router_weights.shape
    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec((_ROUTER_BT, _ROUTER_BK), lambda t, k: (t, k)),
            pl.BlockSpec((_ROUTER_BK, N), lambda t, k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((_ROUTER_BT, N), lambda t, k: (t, 0)),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="router_matmul",
    )(x2, router_weights)

# ----------------------------
# Optimized Route packing
# ----------------------------

def _pack_routes(x2, top_k_indices_tk, router_probs_tk):
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    N = CONFIG["num_experts"]
    R = T * K
    
    # Static upper bound for shapes
    num_group_slots = int(_ceil_div(R, _EXPERT_GBT) + N)
    R_cap = num_group_slots * _EXPERT_GBT

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)
    route_rank = jnp.sum((jnp.cumsum(one_hot, axis=0) - 1) * one_hot, axis=1, dtype=jnp.int32)

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    groups_per_expert = _ceil_div(_ceil_div(counts, _PACK_BT), _EXPERT_RB)
    total_groups = jnp.sum(groups_per_expert)
    
    padded_counts = groups_per_expert * _EXPERT_GBT
    expert_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_counts[:-1], dtype=jnp.int32)])
    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)
    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = (x2[gather_idx] * valid[:, None].astype(x2.dtype))
    
    # Precise mapping: group_id -> expert_id
    group_expert = jnp.repeat(expert_ids, groups_per_expert, total_repeat_length=num_group_slots)

    return packed_x, packed_token, packed_prob, group_expert, total_groups

# ----------------------------
# Expert Projections
# ----------------------------

def _grouped_expert_input_fused_kernel(group_expert_ref, x_ref, gate_w_ref, up_w_ref, gate_o_ref, up_o_ref, acc_gate_ref, acc_up_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_gate_ref[...] = jnp.zeros(acc_gate_ref.shape, dtype=jnp.float32)
        acc_up_ref[...] = jnp.zeros(acc_up_ref.shape, dtype=jnp.float32)
    xb = x_ref[...]
    acc_gate_ref[...] += jnp.dot(xb, gate_w_ref[0, ...], preferred_element_type=jnp.float32)
    acc_up_ref[...] += jnp.dot(xb, up_w_ref[0, ...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        gate_o_ref[...] = acc_gate_ref[...].astype(gate_o_ref.dtype)
        up_o_ref[...] = acc_up_ref[...].astype(up_o_ref.dtype)

def _packed_expert_input_projection_fused_pallas(packed_x, group_expert, gate_k, up_k, total_groups):
    R_cap, E = packed_x.shape
    N, _, M = gate_k.shape
    return pl.pallas_call(
        _grouped_expert_input_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, total_groups, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BK), lambda m, g, k, ge: (g, k)),
                pl.BlockSpec((1, _EXPERT_BK, _EXPERT_BM), lambda m, g, k, ge: (ge[g], k, m)),
                pl.BlockSpec((1, _EXPERT_BK, _EXPERT_BM), lambda m, g, k, ge: (ge[g], k, m)),
            ],
            out_specs=[
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BM), lambda m, g, k, ge: (g, m)),
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BM), lambda m, g, k, ge: (g, m)),
            ],
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32)] * 2,
        ),
        out_shape=[jax.ShapeDtypeStruct((R_cap, M), packed_x.dtype)] * 2,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="expert_input",
    )(group_expert, packed_x, gate_k, up_k)

def _grouped_expert_down_fused_kernel(group_expert_ref, gate_ref, up_ref, down_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    h = jax.nn.silu(gate_ref[...].astype(jnp.float32)) * up_ref[...].astype(jnp.float32)
    acc_ref[...] += jnp.dot(h, down_ref[0, ...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def _packed_expert_down_projection_pallas(gate_rm, up_rm, group_expert, down_k, total_groups):
    R_cap, M = gate_rm.shape
    N, _, E = down_k.shape
    return pl.pallas_call(
        _grouped_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, total_groups, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec((_EXPERT_GBT, _DOWN_BK), lambda e, g, k, ge: (g, k)),
                pl.BlockSpec((_EXPERT_GBT, _DOWN_BK), lambda e, g, k, ge: (g, k)),
                pl.BlockSpec((1, _DOWN_BK, _DOWN_BE), lambda e, g, k, ge: (ge[g], k, e)),
            ],
            out_specs=pl.BlockSpec((_EXPERT_GBT, _DOWN_BE), lambda e, g, k, ge: (g, e)),
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((R_cap, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="expert_down",
    )(group_expert, gate_rm, up_rm, down_k)

def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    B, S, E = x.shape
    T = B * S
    K = CONFIG["num_experts_per_tok"]
    x2 = x.reshape(T, E)
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, -1)
    top_k_logits, top_k_indices = jax.lax.top_k(logits.astype(jnp.float32), K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)
    packed_x, packed_token, packed_prob, group_expert, total_groups = _pack_routes(
        x2, top_k_indices.reshape(T, K).astype(jnp.int32), router_probs.reshape(T, K))
    gate_rm, up_rm = _packed_expert_input_projection_fused_pallas(
        packed_x, group_expert, expert_gate_kernels, expert_up_kernels, total_groups)
    packed_out = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, group_expert, expert_down_kernels, total_groups)
    weighted = packed_out.astype(jnp.float32) * packed_prob[:, None]
    out = jnp.zeros((T, E), dtype=jnp.float32).at[jnp.maximum(packed_token, 0)].add(weighted)
    return out.reshape(B, S, E).astype(x.dtype)
```''',
code='''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}

_ROUTER_BT = 512
_ROUTER_BK = 4096

_EXPERT_BT = 128
_EXPERT_BK = 512
_EXPERT_BM = 1024

_DOWN_BT = 128
_DOWN_BK = 1024
_DOWN_BE = 1024

_PACK_BT = 128
_EXPERT_RB = 4
_EXPERT_GBT = _EXPERT_RB * _EXPERT_BT

def _ceil_div(a, b):
    return (a + b - 1) // b

# ----------------------------
# Router matmul
# ----------------------------

def _router_kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def _router_pallas_matmul(x2: jax.Array, router_weights: jax.Array) -> jax.Array:
    T, E = x2.shape
    _, N = router_weights.shape
    return pl.pallas_call(
        _router_kernel,
        out_shape=jax.ShapeDtypeStruct((T, N), x2.dtype),
        grid=(T // _ROUTER_BT, E // _ROUTER_BK),
        in_specs=[
            pl.BlockSpec((_ROUTER_BT, _ROUTER_BK), lambda t, k: (t, k)),
            pl.BlockSpec((_ROUTER_BK, N), lambda t, k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((_ROUTER_BT, N), lambda t, k: (t, 0)),
        scratch_shapes=[pltpu.VMEM((_ROUTER_BT, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="router_matmul",
    )(x2, router_weights)

# ----------------------------
# Optimized Route packing
# ----------------------------

def _pack_routes(x2, top_k_indices_tk, router_probs_tk):
    T, E = x2.shape
    T2, K = top_k_indices_tk.shape
    N = CONFIG["num_experts"]
    R = T * K
    
    # Static upper bound for shapes
    num_group_slots = int(_ceil_div(R, _EXPERT_GBT) + N)
    R_cap = num_group_slots * _EXPERT_GBT

    route_expert = top_k_indices_tk.reshape(R).astype(jnp.int32)
    route_prob = router_probs_tk.reshape(R).astype(jnp.float32)
    route_token = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)).reshape(R)

    expert_ids = jnp.arange(N, dtype=jnp.int32)
    one_hot = (route_expert[:, None] == expert_ids[None, :]).astype(jnp.int32)
    route_rank = jnp.sum((jnp.cumsum(one_hot, axis=0) - 1) * one_hot, axis=1, dtype=jnp.int32)

    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    blocks_per_expert = _ceil_div(counts, _PACK_BT)
    groups_per_expert = _ceil_div(blocks_per_expert, _EXPERT_RB)
    total_groups = jnp.sum(groups_per_expert)
    
    padded_counts = groups_per_expert * _EXPERT_GBT
    expert_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_counts[:-1], dtype=jnp.int32)])
    packed_pos = expert_offsets[route_expert] + route_rank

    packed_token = -jnp.ones((R_cap,), dtype=jnp.int32)
    packed_prob = jnp.zeros((R_cap,), dtype=jnp.float32)
    packed_token = packed_token.at[packed_pos].set(route_token)
    packed_prob = packed_prob.at[packed_pos].set(route_prob)

    valid = packed_token >= 0
    gather_idx = jnp.maximum(packed_token, 0)
    packed_x = (x2[gather_idx] * valid[:, None].astype(x2.dtype))
    
    # Precise mapping: group_id -> expert_id
    group_expert = jnp.repeat(expert_ids, groups_per_expert, total_repeat_length=num_group_slots)

    return packed_x, packed_token, packed_prob, group_expert, total_groups, num_group_slots

# ----------------------------
# Expert Projections
# ----------------------------

def _grouped_expert_input_fused_kernel(group_expert_ref, x_ref, gate_w_ref, up_w_ref, gate_o_ref, up_o_ref, acc_gate_ref, acc_up_ref):
    del group_expert_ref
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_gate_ref[...] = jnp.zeros(acc_gate_ref.shape, dtype=jnp.float32)
        acc_up_ref[...] = jnp.zeros(acc_up_ref.shape, dtype=jnp.float32)
    xb = x_ref[...]
    acc_gate_ref[...] += jnp.dot(xb, gate_w_ref[0, ...], preferred_element_type=jnp.float32)
    acc_up_ref[...] += jnp.dot(xb, up_w_ref[0, ...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        gate_o_ref[...] = acc_gate_ref[...].astype(gate_o_ref.dtype)
        up_o_ref[...] = acc_up_ref[...].astype(up_o_ref.dtype)

def _packed_expert_input_projection_fused_pallas(packed_x, group_expert, gate_k, up_k, total_groups, num_group_slots):
    R_cap, E = packed_x.shape
    N, _, M = gate_k.shape
    return pl.pallas_call(
        _grouped_expert_input_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // _EXPERT_BM, total_groups, E // _EXPERT_BK),
            in_specs=[
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BK), lambda m, g, k, ge: (g, k)),
                pl.BlockSpec((1, _EXPERT_BK, _EXPERT_BM), lambda m, g, k, ge: (ge[g], k, m)),
                pl.BlockSpec((1, _EXPERT_BK, _EXPERT_BM), lambda m, g, k, ge: (ge[g], k, m)),
            ],
            out_specs=[
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BM), lambda m, g, k, ge: (g, m)),
                pl.BlockSpec((_EXPERT_GBT, _EXPERT_BM), lambda m, g, k, ge: (g, m)),
            ],
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32), pltpu.VMEM((_EXPERT_GBT, _EXPERT_BM), jnp.float32)],
        ),
        out_shape=[jax.ShapeDtypeStruct((num_group_slots * _EXPERT_GBT, M), packed_x.dtype), jax.ShapeDtypeStruct((num_group_slots * _EXPERT_GBT, M), packed_x.dtype)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="expert_input",
    )(group_expert, packed_x, gate_k, up_k)

def _grouped_expert_down_fused_kernel(group_expert_ref, gate_ref, up_ref, down_ref, o_ref, acc_ref):
    del group_expert_ref
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
    gate_f32 = gate_ref[...].astype(jnp.float32)
    up_f32 = up_ref[...].astype(jnp.float32)
    h = jax.nn.silu(gate_f32) * up_f32
    acc_ref[...] += jnp.dot(h, down_ref[0, ...], preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def _packed_expert_down_projection_pallas(gate_rm, up_rm, group_expert, down_k, total_groups, num_group_slots):
    R_cap, M = gate_rm.shape
    N, _, E = down_k.shape
    return pl.pallas_call(
        _grouped_expert_down_fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(E // _DOWN_BE, total_groups, M // _DOWN_BK),
            in_specs=[
                pl.BlockSpec((_EXPERT_GBT, _DOWN_BK), lambda e, g, k, ge: (g, k)),
                pl.BlockSpec((_EXPERT_GBT, _DOWN_BK), lambda e, g, k, ge: (g, k)),
                pl.BlockSpec((1, _DOWN_BK, _DOWN_BE), lambda e, g, k, ge: (ge[g], k, e)),
            ],
            out_specs=pl.BlockSpec((_EXPERT_GBT, _DOWN_BE), lambda e, g, k, ge: (g, e)),
            scratch_shapes=[pltpu.VMEM((_EXPERT_GBT, _DOWN_BE), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((num_group_slots * _EXPERT_GBT, E), gate_rm.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="expert_down",
    )(group_expert, gate_rm, up_rm, down_k)

def _scatter_routed_outputs(packed_out_re, packed_token, packed_prob, T):
    weighted = packed_out_re.astype(jnp.float32) * packed_prob[:, None]
    scatter_idx = jnp.maximum(packed_token, 0)
    out = jnp.zeros((T, packed_out_re.shape[1]), dtype=jnp.float32)
    out = out.at[scatter_idx].add(weighted)
    return out

def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    B, S, E = x.shape
    T = B * S
    K = CONFIG["num_experts_per_tok"]
    x2 = x.reshape(T, E)
    
    logits = _router_pallas_matmul(x2, router_weights).reshape(B, S, -1)
    logits_f32 = logits.astype(jnp.float32)
    top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1).astype(jnp.float32)
    
    packed_x, packed_token, packed_prob, group_expert, total_groups, num_group_slots = _pack_routes(
        x2, top_k_indices.reshape(T, K).astype(jnp.int32), router_probs.reshape(T, K))
    
    gate_rm, up_rm = _packed_expert_input_projection_fused_pallas(
        packed_x, group_expert, expert_gate_kernels, expert_up_kernels, total_groups, num_group_slots)
    
    packed_out = _packed_expert_down_projection_pallas(
        gate_rm, up_rm, group_expert, expert_down_kernels, total_groups, num_group_slots)
    
    output_te = _scatter_routed_outputs(packed_out, packed_token, packed_prob, T)
    
    return output_te.reshape(B, S, E).astype(x.dtype)
''',
score=6.357,
translation_score=None,
hw_feedback=[],
plan_gen_model='gemini-3-flash-preview',
code_gen_model='us.anthropic.claude-opus-4-5-20251101-v1:0',
stdout='Latency: 6.357 ms\n{"correct": true, "latency": 6.357, "error": "", "all_times_ms": [6.333, 6.335, 6.337, 6.338, 6.339, 6.339, 6.34, 6.34, 6.342, 6.343, 6.343, 6.343, 6.344, 6.345, 6.345, 6.346, 6.346, 6.346, 6.346, 6.346, 6.346, 6.347, 6.347, 6.347, 6.347, 6.349, 6.349, 6.349, 6.349, 6.349, 6.35, 6.35, 6.351, 6.351, 6.351, 6.352, 6.352, 6.352, 6.352, 6.353, 6.353, 6.354, 6.354, 6.355, 6.355, 6.355, 6.355, 6.355, 6.355, 6.356, 6.357, 6.357, 6.357, 6.358, 6.358, 6.358, 6.358, 6.358, 6.358, 6.358, 6.359, 6.359, 6.36, 6.361, 6.361, 6.361, 6.361, 6.361, 6.361, 6.363, 6.363, 6.363, 6.363, 6.363, 6.363, 6.364, 6.364, 6.364, 6.365, 6.366, 6.366, 6.367, 6.367, 6.368, 6.369, 6.369, 6.373, 6.373, 6.375, 6.377, 6.379, 6.381, 6.382, 6.383, 6.383, 6.387, 6.387, 6.401, 6.423, 6.492], "max_diff": 0.054258, "max_rel_diff": 0.004881}',
stderr='')