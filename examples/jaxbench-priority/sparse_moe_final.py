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
