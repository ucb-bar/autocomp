## attention_kernels.py

```python
import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

####################################################################
# v1: toy example with 128 seqlen and nki.lang APIs
####################################################################
@nki.jit
def attn_fwd_v1(q, k, v):
    """nki.lang APIs"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    
    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128
    
    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
   
    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)
    
    # Q @ K, contract along d_head #
    qk: nt.tensor[seqlen_q, seqlen_kv] = nl.matmul(x=q_sbuf, y=k_sbuf, transpose_x=True)
    
    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.max(qk, axis=1)
    
    # subtract max from row
    norm_row = nl.subtract(qk, row_max)
    
    # exponentiation
    exp_row = nl.exp(norm_row)
    
    # sum of exp results
    sum_row = nl.sum(exp_row, axis=1)
    
    # divide exp results by sum
    scores: nt.tensor[seqlen_q, seqlen_kv] = nl.divide(exp_row, sum_row)
    
    # v has the wrong layout
    v_sbuf_t: nt.tensor[seqlen_kv, d_head] = nl.transpose(v_sbuf)
    
    # scores @ V, contract along seqlen_kv
    attn_out: nt.tensor[seqlen_q, d_head] = nl.matmul(scores, v_sbuf_t, transpose_x=False)
    
    # store output
    nl.store(dst=kernel_out, value=attn_out)
    return kernel_out


####################################################################
# v2: use nki.isa APIs
####################################################################
@nki.jit
def attn_fwd_v2(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128
    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)
    # Q @ K, contract along d_head #
    qk: nt.tensor[seqlen_q, seqlen_kv] = nisa.nc_matmul(stationary=q_sbuf,
                                                        moving=k_sbuf)
    # Softmax #
    # reduce max along seqlen_kv, dimension: [seqlen_q, 1]
    row_max = nisa.tensor_reduce(op=nl.max, data=qk, axis=1)
    # subtract max from row, dimension: [seqlen_q, seqlen_kv]
    norm_row = nisa.tensor_scalar(data=qk,
                                op0=nl.subtract,
                                operand0=row_max,
                                engine=nisa.vector_engine)
    # exponentiation, dimension: [seqlen_q, seqlen_kv]
    exp_row = nisa.activation(op=nl.exp, data=norm_row, bias=None, scale=1.0)
    # sum of exp results, dimension: [seqlen_q, 1]
    sum_row = nisa.tensor_reduce(op=nl.add,
                                data=exp_row,
                                axis=1)
    # reciprocal of sum_row, dimension: [seqlen_q, 1]
    inverse_sum_row = nisa.reciprocal(data=sum_row)
    scores: nt.tensor[seqlen_q, seqlen_kv] = nisa.tensor_scalar(data=exp_row,
                                    op0=nl.multiply,
                                    operand0=inverse_sum_row,
                                    engine=nisa.vector_engine,
                                    dtype=q.dtype)
    # v has the wrong layout
    v_psum_t = nisa.nc_transpose(v_sbuf)          # TensorE
    # dimension: [seqlen_kv, d_head]
    v_sbuf_t = nisa.tensor_copy(v_psum_t)         # ScalarE
    # scores has the wrong layout
    scores_psum_t = nisa.nc_transpose(scores)          # TensorE
    # dimension: [seqlen_kv, seqlen_q]
    scores_sbuf_t = nisa.tensor_copy(scores_psum_t)    # ScalarE
    # scores @ V, contract along seqlen_kv
    attn_out: nt.tensor[seqlen_q, d_head] = nisa.nc_matmul(stationary=scores_sbuf_t,
                                                           moving=v_sbuf_t)
    # store output
    nl.store(dst=kernel_out, value=attn_out)
    return kernel_out


####################################################################
# v6: instruction combination on ScalarE
####################################################################
@nki.jit
def attn_fwd_v6(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        
        # We leverage scalar engine's hardware capability of applying reduce after activation
        # with no extra performance cost to compute the max_val subtraction and sum reduction 
        # in one step, saving on extra loops that were previously required.
        #
        # At the same time the vector engine is freed up from compute, giving it more idle time
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.bfloat16)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out
```

## neuron-profiler-2-0-beta-user-guide.rst

```python
import jax
import os

# JAX Profiler - Context Manager Usage
with jax.profiler.trace(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]):
    # Code to profile
    pass

# JAX Custom Annotations
with jax.profiler.TraceAnnotation("my_label"+str(i)):
    # Code to annotate
    pass
```

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from torch_neuronx.experimental import profiler

class MLP(nn.Module):
    def __init__(self, input_size=10, output_size=2, layers=[5, 5]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# PyTorch Profiler - Context Manager Usage
with profiler.profile(
    port=9012,
    profile_type='system',
    target='neuron_profile_perfetto',
    output_dir=os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'],
    ms_duration=30000) as profiler_ctx:
    
    device = xm.xla_device()
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()
    
    model.train()
    optimizer.zero_grad()
    train_x = torch.randn(1, 10).to(device)
    train_label = torch.tensor([1]).to(device)
    
    loss = loss_fn(model(train_x), train_label)
    loss.backward()
    optimizer.step()
    xm.mark_step()
```

```c
#include <nrt/nrt_sys_trace.h>

// NeuronCore Filtering - API Usage
nrt_sys_trace_config_t *config;
nrt_sys_trace_config_allocate(&config);
nrt_sys_trace_config_set_defaults(config);

int num_cores = 128;
for (int i=0; i<num_cores; i++) {
  nrt_sys_trace_config_set_capture_enabled_for_nc(config, i, false);
}

nrt_sys_trace_config_set_capture_enabled_for_nc(config, 0, true);
nrt_sys_trace_config_set_capture_enabled_for_nc(config, 2, true);

nrt_sys_trace_start(config);

// Application code here

nrt_sys_trace_stop();
nrt_sys_trace_config_free(config);
```

```c
#include <nrt/nrt_sys_trace.h>

// Get Available Event Types
const char **event_types = nullptr;
size_t count = 0;
NRT_STATUS status = nrt_sys_trace_get_event_types(&event_types, &count);

if (status == NRT_SUCCESS) {
    for (size_t i = 0; i < count; ++i) {
        printf("  %s\n", event_types[i]);
    }
    
    for (size_t i = 0; i < count; ++i) {
        free((void*)event_types[i]);
    }
    free((void*)event_types);
}
```

```c
#include <nrt/nrt_sys_trace.h>

// Event Type Filtering - API Usage
nrt_sys_trace_config_t *config;
nrt_sys_trace_config_allocate(&config);
nrt_sys_trace_config_set_defaults(config);

nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "device_exec", false);

const char **all_event_types = nullptr;
size_t all_count = 0;
nrt_sys_trace_get_event_types(&all_event_types, &all_count);

for (size_t i = 0; i < all_count; ++i) {
    nrt_sys_trace_config_set_capture_enabled_for_event_type(config, all_event_types[i], false);
}

nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "model_load", true);
nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "nrt_execute", true);

const char **enabled_types = nullptr;
size_t enabled_count = 0;
nrt_sys_trace_config_get_enabled_event_types(config, &enabled_types, &enabled_count);

for (size_t i = 0; i < enabled_count; ++i) {
    free((void*)enabled_types[i]);
}
free((void*)enabled_types);

for (size_t i = 0; i < all_count; ++i) {
    free((void*)all_event_types[i]);
}
free((void*)all_event_types);

nrt_sys_trace_start(config);

// Application code here

nrt_sys_trace_stop();
nrt_sys_trace_config_free(config);
```

## fused_mamba.rst

```python
import torch
import neuron.ml as ml
import nki
import nki.language as nl
from nki.language import nisa
import numpy as np


# PyTorch Reference Implementation
def mamba_torch_reference(delta, u, A, B, C):
    """
    PyTorch reference implementation of Mamba layer computation.
    
    Input shapes:
    - delta: [batch, channels, seq_len]
    - u: [batch, channels, seq_len]
    - A: [channels, state_size]
    - B: [batch, state_size, seq_len]
    - C: [batch, state_size, seq_len]
    """
    batch_size, channels, seq_len = delta.shape
    state_size = A.shape[1]
    
    # Step 1 & 2: deltaA = exp(delta * A)
    deltaA = torch.exp(delta[:, :, None, :] * A[None, :, :, None])
    
    # Step 3: deltaBu = delta * B * u
    deltaBu = delta[:, :, None, :] * B[:, None, :, :] * u[:, :, None, :]
    
    # Step 4: Associative scan
    scan_res = torch.empty(batch_size, channels, state_size, seq_len,
                          device=deltaA.device, dtype=deltaA.dtype)
    for i in range(seq_len):
        prev_state = scan_res[..., i - 1] if i > 0 else 0
        scan_res[..., i] = deltaA[..., i] * prev_state + deltaBu[..., i]
    
    # Step 5: scanC = C * scan_res
    scanC = C[:, None, :, :] * scan_res
    
    # Step 6: Accumulate along state_size dimension
    output = scanC.sum(dim=-2)
    
    return output


# Initial NKI Kernel Implementation (mamba_v1)
@nki.jit
def mamba_v1(delta, u, A, B, C):
    """
    Initial NKI kernel for Mamba layer computation.
    
    Input shapes:
    - delta: [batch_size, channels, seq_len]
    - u: [batch_size, channels, seq_len]
    - A: [channels, state_size]
    - B: [batch_size, state_size, seq_len]
    - C: [batch_size, state_size, seq_len]
    """
    batch_size, channels, seq_len = delta.shape
    state_size = A.shape[1]
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        for i_state in nl.affine_range(state_size):
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize
                
                # Load inputs
                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                # Step 1 & 2: deltaA = exp(delta * A)
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                
                # Step 3: deltaBu = delta * B * u
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=ml.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=np.multiply, op1=np.add)
                
                # Step 5: scanC = C * scan_i
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
                
                # Step 6: Accumulate
                scanC_accum += scanC_i
            
            nl.store(output[i_batch, 0:channels, 0:seq_len], scanC_accum)
    
    return output


# Optimized NKI Kernel with Loop Reordering (mamba_v2)
@nki.jit
def mamba_v2(delta, u, A, B, C):
    """
    Optimized NKI kernel with loop reordering to minimize data reloading.
    
    Key optimization: Reorder loops to prioritize reuse of delta/u tensors
    which are larger than B/C tensors.
    """
    batch_size, channels, seq_len = delta.shape
    state_size = A.shape[1]
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            
            # Load delta and u once, reuse across all states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            
            # Accumulator for this channel tile
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            for i_state in nl.affine_range(state_size):
                # Load state-specific inputs
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                # Step 1 & 2: deltaA = exp(delta * A)
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                
                # Step 3: deltaBu = delta * B * u
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=ml.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=np.multiply, op1=np.add)
                
                # Step 5: scanC = C * scan_i
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
                
                # Step 6: Accumulate
                scanC_accum += scanC_i
            
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len], scanC_accum)
    
    return output


# Optimized NKI Kernel with seq_len Tiling (mamba_v3)
@nki.jit
def mamba_v3(delta, u, A, B, C):
    """
    Optimized NKI kernel with seq_len tiling to reduce SBUF spilling.
    
    Key optimization: Tile seq_len dimension to keep intermediate tensors
    small and maintain high VectorE utilization.
    """
    batch_size, channels, seq_len = delta.shape
    state_size = A.shape[1]
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    seq_len_fsize = 512  # Free dimension tile size for seq_len
    n_seq_len_tile = (seq_len + seq_len_fsize - 1) // seq_len_fsize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            
            # Accumulator for this channel tile
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            # Initialize scan state for loop-carried dependency
            scan_init = nl.zeros((nl.par_dim(channel_psize), 1), dtype=delta.dtype, buffer=nl.sbuf)
            
            for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                seq_start = i_seq_len_tile * seq_len_fsize
                seq_end = min(seq_start + seq_len_fsize, seq_len)
                seq_tile_len = seq_end - seq_start
                
                # Load delta and u for this seq_len tile
                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, seq_start:seq_end])
                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, seq_start:seq_end])
                
                for i_state in nl.affine_range(state_size):
                    # Load state-specific inputs
                    A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                    B_i = nl.load(B[i_batch, i_state:i_state+1, seq_start:seq_end])
                    C_i = nl.load(C[i_batch, i_state:i_state+1, seq_start:seq_end])
                    
                    # Step 1 & 2: deltaA = exp(delta * A)
                    deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                    
                    # Step 3: deltaBu = delta * B * u
                    deltaU = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
                    B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=ml.multiply)
                    
                    # Step 4: Associative scan with loop-carried dependency
                    scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                                                     op0=np.multiply, op1=np.add)
                    
                    # Step 5: scanC = C * scan_i
                    C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
                    
                    # Step 6: Accumulate
                    scanC_accum[:, seq_start:seq_end] += scanC_i
                
                # Update scan state for next seq_len tile
                scan_init = scan_i[:, seq_tile_len-1:seq_tile_len]
            
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len], scanC_accum)
    
    return output
```

## nki.errors.rst

```python
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# err_1d_arange_not_supported - Workaround with new axes
tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
i = nl.arange(64)[:, None]
c = nl.exp(tmp[i, 0])

# err_1d_arange_not_supported - Workaround with slicing
tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
c = nl.exp(tmp[0:64, 0])

# err_activation_bias_invalid_type - Valid examples
nisa.activation(op=nl.exp, data=data[...], bias=nisa.memset((128, 1), 1.2, dtype=np.float32))
nisa.activation(op=nl.exp, data=data[...], bias=nisa.memset((128, 1), 1.2, dtype=nl.bfloat16))

# err_activation_scale_invalid_type - Valid examples
nisa.activation(op=nl.exp, data=data[...], scale=1.2)
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 1), 1.2, dtype=np.float32))

# err_activation_scale_scalar_or_vector - Valid examples
nisa.activation(op=nl.exp, data=data[...], scale=1.2)
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 1), 1.2, dtype=np.float32))

# err_ambiguous_tensor_truth_value - Correct usage
def func(a, b: nt.Optional[nt.tensor]):
    ix, iy = nl.mgrid[0:128, 0:128]
    a_tile = nl.load(a[ix, iy])
    not_a_tile = ~(a_tile > 0)
    if b is not None:
        pass

# err_cannot_assign_to_index - Workaround with iota
_, x = nl.mgrid[0:1, 0:8]
y = nisa.iota(x, dtype=nl.uint32)
y[0, 5] = 1024

# err_cannot_update_immutable_parameter - Mutable parameter annotation
def kernel(in_tensor: nt.mutable_tensor):
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(in_tensor, value=y)
    return in_tensor

# err_cannot_update_immutable_parameter - Alternative with copy
def kernel(in_tensor):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                            buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_tensor, src=in_tensor)
    x = nl.load(out_tensor)
    y = x + 1
    nl.store(out_tensor, value=y)
    return out_tensor

# err_control_flow_condition_depending_on_arange - Workaround with mask
for j0 in nl.affine_range(4096):
    i1 = nl.arange(512)[None, :]
    j = j0 * 512 + i1
    y = nl.add(x[0, j], x[0, j - 2048], mask=j > 2048)

# err_copy_dynamic_indirect_indices_not_natively_supported - Workaround
data_tensor = nl.load(data_tensor)
idx_tile = nl.load(idx_tensor)
out_sbuf = nl.ndarray([8, 4], dtype=data_tensor.dtype, buffer=nl.sbuf)
iy, iz = nl.mgrid[0:8, 0:4]
out_sbuf[iy, iz] = nisa.tensor_copy_dynamic_src(data_tensor[0, idx_tile, iz])

# err_failed_to_infer_tile_from_local_tensor - Workaround with loop
a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
c = nl.ndarray((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
for i in range(4):
    c[i] = nl.add(a[i], 32)

# err_failed_to_infer_tile_from_local_tensor - Workaround with arange
ix = nl.arange(8)[:, None]
iy = nl.arange(8)[None, :]
c[i, ix, iy] = nl.add(a[i, ix, iy], 32)

# err_hbm_tensor_with_init_value_not_supported - Workaround
t = nl.ndarray((3, 128, 512), buffer=nl.shared_hbm)
for i in range(3):
    nl.store(dst=t[i, :, :], value=1.0)

# err_indirect_indices_free_dim - Correct usage with arange
i_p = nl.arange(64)[:, None]
i_f = nl.arange(512)[None, :]
data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f])

# err_local_variable_used_out_of_scope - Correct pattern
for i in range(4):
    tmp = nl.ndarray(shape=a.shape, dtype=a.dtype)
    if i < 2:
        tmp[...] = nl.load(a)
    else:
        tmp[...] = nl.load(b)
    nl.store(c, tmp)

# err_local_variable_used_out_of_scope - Correct pattern with update
data = nl.zeros((nl.par_dim(128), 128), dtype=np.float32)
for i in nl.sequential_range(4):
    i_tile = nisa.iota(i, dtype=nl.uint32).broadcast_to(data.shape)
    data[...] = data + i_tile
nl.store(ptr, value=data)

# err_mutable_parameter_not_returned - Correct pattern
def kernel(in_tensor: nt.mutable_tensor):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                            buffer=nl.shared_hbm)
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(out_tensor, value=y)
    nl.store(in_tensor, value=y)
    return out_tensor, in_tensor

# err_num_partition_mismatch - Workaround with broadcast_to
x = nl.zeros(shape=[128, 512], dtype=np.float32, buffer=nl.sbuf)
y0 = nl.zeros(shape=[1, 512], dtype=np.float32, buffer=nl.sbuf)
y1 = y0.broadcast_to([128, 512])
z = nisa.tensor_tensor(x, y1, op=nl.add)

# err_store_dst_shape_smaller_than_other_shape - Correct pattern
x = nl.zeros(shape=(128, 512), dtype=nl.float32, buffer=nl.sbuf)
y = nl.zeros(shape=(128, 1), dtype=nl.float32, buffer=nl.sbuf)
x[...] = y

# err_tensor_access_out_of_bound - Workaround with mask
x = nl.ndarray([128, 4000], dtype=np.float32, buffer=nl.hbm)
for i in nl.affine_range((4000 + 512 - 1) // 512):
    tile = nl.mgrid[0:128, 0:512]
    nl.store(x[tile.p, i * 512 + tile.x], value=0,
              mask=i * 512 + tile.x < 4000)

# err_tensor_output_not_written_to - Correct pattern with initialization
def memset_output(input, output, cnd):
    nl.store(output[i_p, i_f], value=0)
    while cnd:
        a = nl.load(input)
        nl.store(output, value=a)

# err_unexpected_output_dependencies - Correct pattern with affine_range
a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
for i in nl.affine_range(4):
    a[i] = 0

# err_unexpected_output_dependencies - Correct pattern with sequential_range
a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
for i in nl.sequential_range(4):
    a[0] = 0

# err_unsupported_expression_in_mask - Correct usage with affine expressions
def test():
    out = nl.ndarray([8], dtype=nl.int32, buffer=nl.shared_hbm)
    for i in range(8):
        nl.store(out, i, mask=i < 4)
    for i in range(8):
        for j in range(8):
            nl.store(out, i, mask=(2*i + 3*j + 1) < 20)
    return out

# err_unsupported_mixing_basic_advanced_tensor_indexing - Correct usage
a = nl.zeros((4, 4), dtype=nl.float32, buffer=nl.sbuf)
c = nl.exp(a[:, :])

i = nl.arange(4)[:, None]
j = nl.arange(4)[None, :]
c = nl.exp(a[i, j])

# err_while_loop_requires_unconditional_entry - Correct do-while pattern
def func(a):
    a_tile = nl.load(a[0, 0])
    a_scalar = nl.scalar(a_tile)
    cond = nl.scalar(True)
    while cond:
        a_tile = a_tile + 1
        a_scalar = nl.scalar(a_tile)
        cond = a_scalar < 10
```

## vllm-user-guide.rst

```python
import os
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

from vllm import LLM, SamplingParams
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    max_model_len=128,
    device="neuron",
    tensor_parallel_size=2)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(top_k=10, temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```python
import os
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

from vllm import LLM, SamplingParams

prompts = [
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(top_k=1)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=4,
    max_model_len=128,
    override_neuron_config={
        "enable_bucketing":False,
    },
    device="neuron",
    tensor_parallel_size=32)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model_name = models.data[0].id

max_tokens = 1024
temperature = 1.0
top_p = 1.0
top_k = 50
stream = False

prompt = "Hello, my name is Llama "
response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=int(max_tokens),
    temperature=float(temperature),
    top_p=float(top_p),
    stream=stream,
    extra_body={'top_k': top_k}
)

generated_text = ""
if stream:
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            generated_text += chunk.choices[0].delta.content
else:
    generated_text = response.choices[0].message.content
    
print(generated_text)
```

## perceiver-multimodal_compile.py

```python
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers import PerceiverForMultimodalAutoencoding
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverClassifierOutput, restructure
import torch_neuronx

class MultimodalPerceiverWrapper(nn.Module):
    def __init__(self, perceiver_model, nchunks, image_chunk_size, audio_chunk_size):
        super().__init__()
        self.perceiver_model = perceiver_model
        self.nchunks = nchunks
        self.image_chunk_size = image_chunk_size
        self.audio_chunk_size = audio_chunk_size
    
    def forward(self, inputs: torch.FloatTensor,
        neuron_decoder,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        output_attentions = output_attentions if output_attentions is not None else self.perceiver_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.perceiver_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.perceiver_model.config.use_return_dict
        
        if self.perceiver_model.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.perceiver_model.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            if inputs.size()[-1] != self.perceiver_model.config.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model:"
                    f" {self.perceiver_model.config.d_model}. Make sure to set config.d_model appropriately."
                )

        batch_size, seq_length, _ = inputs.size()
        device = inputs.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.perceiver_model.invert_attention_mask(attention_mask)

        head_mask = self.perceiver_model.get_head_mask(head_mask, self.perceiver_model.config.num_blocks * self.perceiver_model.config.num_self_attends_per_block)
        embedding_output = self.perceiver_model.embeddings(batch_size=batch_size)

        encoder_outputs = self.perceiver_model.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        reconstruction = {}
        for chunk_idx in range(self.nchunks):
            subsampled_output_points = {
            'image': torch.arange(
                self.image_chunk_size * chunk_idx, self.image_chunk_size * (chunk_idx + 1)).to(device),
            'audio': torch.arange(
                self.audio_chunk_size * chunk_idx, self.audio_chunk_size * (chunk_idx + 1)).to(device),
            'label': None,
            }
            
            logits = neuron_decoder(sequence_output, extended_attention_mask, 
                                             inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points)

            reconstruction['label'] = logits['label']
            if 'image' not in reconstruction:
                reconstruction['image'] = logits['image']
                reconstruction['audio'] = logits['audio']
            else:
                reconstruction['image'] = torch.cat(
                    [reconstruction['image'], logits['image']], dim=1)
                reconstruction['audio'] = torch.cat(
                    [reconstruction['audio'], logits['audio']], dim=1)
            
            del logits

        return reconstruction
```

```python
class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, embedding_output, inputs, extended_attention_mask):
        output = self.encoder(embedding_output, inputs=inputs, inputs_mask=extended_attention_mask)
        return output

class NeuronEncoder(nn.Module):
    def __init__(self, encoder_wrapper):
       super().__init__()
       self.encoder_wrapper = encoder_wrapper
    
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True):

        last_hidden_states = self.encoder_wrapper(hidden_states, inputs, inputs_mask)['last_hidden_state']
        return BaseModelOutputWithCrossAttentions(last_hidden_state=last_hidden_states)
```

```python
class DecoderWrapper(nn.Module):
    def __init__(self, decoder, decoder_query_audio, decoder_query_image, decoder_query_label, output_postprocessor):
        super().__init__()
        self.decoder = decoder
        self.decoder_query_audio = decoder_query_audio
        self.decoder_query_image = decoder_query_image
        self.decoder_query_label = decoder_query_label
        self.output_postprocessor = output_postprocessor
        self.num_query_channels = decoder.num_query_channels
    
    def forward(self, z, query_mask,
                audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
                image_input, image_input_without_pos, image_subsampled_point, image_padding,
                label_input, label_input_without_pos, label_padding):
        audio_query = self.decoder_query_audio(inputs=audio_input, inputs_without_pos=audio_input_without_pos, subsampled_points=audio_subsampled_point)
        image_query = self.decoder_query_image(inputs=image_input, inputs_without_pos=image_input_without_pos, subsampled_points=image_subsampled_point)
        label_query = self.decoder_query_label(inputs=label_input, inputs_without_pos=label_input_without_pos)

        def embed(x, pos):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            return torch.cat([x, pos], dim=2)

        audio_padded = embed(audio_query, audio_padding)
        image_padded = embed(image_query, image_padding)
        label_padded = embed(label_query, label_padding)

        decoder_query = torch.cat([audio_padded, image_padded, label_padded], dim=1)
        logits = self.decoder(decoder_query, z, query_mask).logits
        
        output_modality_sizes = {"audio": audio_subsampled_point.shape[0],
                                 "image": image_subsampled_point.shape[0],
                                 "label": 1}
        logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)
        return logits

class NeuronDecoder(nn.Module):
    def __init__(self, decoder_wrapper):
        super().__init__()
        self.decoder_wrapper = decoder_wrapper
        self.modalities = decoder_wrapper.decoder.modalities
        self.padding = decoder_wrapper.decoder.padding

    def forward(self, z, query_mask, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None, output_attentions=False):
        inputs = restructure(modality_sizes, inputs)

        assert(subsampled_points is not None)
        assert(inputs_without_pos is not None)

        for modality, decoder in self.modalities.items():
            if modality == "audio":
                audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding = inputs[modality], inputs_without_pos[modality], subsampled_points[modality].to(torch.float32), self.padding[modality]
            elif modality == "image":
                image_input, image_input_without_pos, image_subsampled_point, image_padding = inputs[modality], inputs_without_pos[modality], subsampled_points[modality].to(torch.float32), self.padding[modality]
            else:
                label_input, label_input_without_pos, label_padding = inputs[modality], inputs_without_pos[modality], self.padding[modality]

        assert(audio_input_without_pos is not None)
        assert(audio_subsampled_point is not None)
        assert(image_input_without_pos is not None)
        assert(image_subsampled_point is not None)
        assert(label_input_without_pos is not None)

        output = self.decoder_wrapper(z, query_mask, 
                                        audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
                                        image_input, image_input_without_pos, image_subsampled_point, image_padding,
                                        label_input, label_input_without_pos, label_padding)
        return output
```

```python
def custom_decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
    if self.position_encoding_type == "none":
        raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
    if subsampled_points is not None:
        def unravel_indices(indices, shape):
            coord = []
            for dim in reversed(shape):
                coord.append(indices % dim)
                indices = indices // dim
            coord = torch.stack(coord[::-1], dim=-1)
            return coord

        pos = unravel_indices(subsampled_points, self.output_index_dims)

        batch_size = inputs.shape[0]
        pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
        pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
        if self.position_encoding_type == "trainable":
            pos_emb = self.output_position_encodings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_emb = self.output_position_encodings(
                self.output_index_dims, batch_size=batch_size, device=inputs.device, dtype=inputs.dtype, pos=pos
            )

        pos_emb = self.positions_projection(pos_emb)
        pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
    else:
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[2:]

        if self.position_encoding_type == "trainable":
            pos_emb = self.output_position_encodings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_emb = self.output_position_encodings(
                index_dims, batch_size, device=inputs.device, dtype=inputs.dtype
            )

        pos_emb = self.positions_projection(pos_emb)

    if self.concat_preprocessed_input:
        if inputs_without_pos is None:
            raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
        pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)

    return pos_emb
```

```python
neuron_encoder.encoder_wrapper = torch_neuronx.trace(
  neuron_encoder.encoder_wrapper,
  (embedding_output, sample_inputs, extended_attention_mask),
  compiler_workdir=COMPILER_WORKDIR_ENCODER,
  compiler_args=[f"--temp-dir={COMPILER_WORKDIR_ENCODER}", "--auto-cast=none"]
)

neuron_decoder.decoder_wrapper = torch_neuronx.trace(
   neuron_decoder.decoder_wrapper,
   (z, query_mask, audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
        image_input, image_input_without_pos, image_subsampled_point, image_padding,
        label_input, label_input_without_pos, label_padding),
   compiler_workdir=COMPILER_WORKDIR_DECODER,
   compiler_args=[f"--temp-dir={COMPILER_WORKDIR_DECODER}", "--auto-cast=none"]
)
```

## matrix_multiplication.rst

```python
import nki
import nki.language as nl

@nki.jit
def nki_matmul_basic_():
    # Define indices to access LHS and RHS input tensors
    lhs_T = nl.ndarray(shape=(128, 64), dtype=nl.bfloat16)
    rhs = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16)
    output = nl.ndarray(shape=(64, 512), dtype=nl.bfloat16)
    
    # Load inputs from HBM to SBUF
    lhs_T_tile = nl.load(lhs_T)
    rhs_tile = nl.load(rhs)
    
    # Perform matrix multiplication with transpose_x=True
    result = nl.matmul(lhs_T_tile, rhs_tile, transpose_x=True)
    
    # Copy result from PSUM to SBUF
    result_sbuf = nl.copy(result, buffer=nl.sbuf)
    
    # Store result to HBM
    nl.store(output, result_sbuf)
```

```python
import nki
import nki.language as nl

@nki.jit
def nki_matmul_tiled_(lhs_T, rhs, output):
    M, K = lhs_T.shape
    K, N = rhs.shape
    
    # Tile LHS_T free dimension
    for m in nl.affine_range(0, M, 128):
        # Tile RHS free dimension
        for n in nl.affine_range(0, N, 512):
            # Zero-out the accumulator buffer
            psum_buf = nl.zeros((128, 512), dtype=nl.bfloat16, buffer=nl.psum)
            
            # Tile contraction dimension
            for k in nl.affine_range(0, K, 128):
                lhs_T_tile = nl.load(lhs_T[m : m+128, k : k+128])
                rhs_tile = nl.load(rhs[k : k+128, n : n+512])
                psum_buf += nl.matmul(lhs_T_tile, rhs_tile, transpose_x=True)
            
            # Copy result from PSUM to SBUF and store
            result_tile = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(output[m : m+128, n : n+512], result_tile)
```

```python
import nki
import nki.language as nl

@nki.jit
def nki_matmul_hoist_load_(lhs_T, rhs, output):
    M, K = lhs_T.shape
    K, N = rhs.shape
    
    # Tile LHS_T free dimension
    for m in nl.affine_range(0, M, 128):
        # Tile RHS free dimension
        for n in nl.affine_range(0, N, 512):
            # Hoist loads out of innermost loop
            rhs_tile = nl.load(rhs[:, n : n+512])
            
            # Zero-out the accumulator buffer
            psum_buf = nl.zeros((128, 512), dtype=nl.bfloat16, buffer=nl.psum)
            
            # Tile contraction dimension
            for k in nl.affine_range(0, K, 128):
                lhs_T_tile = nl.load(lhs_T[m : m+128, k : k+128])
                psum_buf += nl.matmul(lhs_T_tile, rhs_tile[k : k+128, :], transpose_x=True)
            
            # Copy result from PSUM to SBUF and store
            result_tile = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(output[m : m+128, n : n+512], result_tile)
```

```python
import nki
import nki.language as nl

@nki.jit
def nki_matmul_block_free_dimension_(lhs_T, rhs, output):
    M, K = lhs_T.shape
    K, N = rhs.shape
    
    NUM_BLOCK_M = 2
    NUM_BLOCK_N = 2
    BLOCK_M = M // NUM_BLOCK_M
    BLOCK_N = N // NUM_BLOCK_N
    
    # Block free dimensions
    for bm in nl.affine_range(0, NUM_BLOCK_M):
        for bn in nl.affine_range(0, NUM_BLOCK_N):
            m_start = bm * BLOCK_M
            n_start = bn * BLOCK_N
            
            # Load entire blocks
            lhs_T_tiles = nl.load(lhs_T[m_start : m_start+BLOCK_M, :])
            rhs_tiles = nl.load(rhs[:, n_start : n_start+BLOCK_N])
            
            # Zero-out the accumulator buffer
            psum_buf = nl.zeros((BLOCK_M, BLOCK_N), dtype=nl.bfloat16, buffer=nl.psum)
            
            # Tile contraction dimension
            for k in nl.affine_range(0, K, 128):
                psum_buf += nl.matmul(lhs_T_tiles[:, k : k+128], rhs_tiles[k : k+128, :], transpose_x=True)
            
            # Copy result from PSUM to SBUF and store
            result_tile = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(output[m_start : m_start+BLOCK_M, n_start : n_start+BLOCK_N], result_tile)
```

```python
import nki
import nki.language as nl

@nki.jit
def nki_matmul_fully_optimized_(lhs_T, rhs, output):
    M, K = lhs_T.shape
    K, N = rhs.shape
    
    NUM_BLOCK_M = M // 2048
    NUM_BLOCK_N = N // 1024
    NUM_BLOCK_K = K // 1024
    
    # Block all dimensions
    for bm in nl.affine_range(0, NUM_BLOCK_M):
        for bn in nl.affine_range(0, NUM_BLOCK_N):
            m_start = bm * 2048
            n_start = bn * 1024
            
            # Load blocks for free dimensions
            lhs_T_tiles = nl.load(lhs_T[m_start : m_start+2048, :])
            rhs_tiles = nl.load(rhs[:, n_start : n_start+1024])
            
            # Zero-out the accumulator buffer
            psum_buf = nl.zeros((2048, 1024), dtype=nl.bfloat16, buffer=nl.psum)
            
            # Block contraction dimension with sequential range
            for bk in nl.sequential_range(0, NUM_BLOCK_K):
                k_start = bk * 1024
                psum_buf += nl.matmul(lhs_T_tiles[:, k_start : k_start+1024], 
                                      rhs_tiles[k_start : k_start+1024, :], 
                                      transpose_x=True)
            
            # Copy result from PSUM to SBUF and store
            result_tile = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(output[m_start : m_start+2048, n_start : n_start+1024], result_tile)
```

## matrix_multiplication_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np


@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [64,512]
  """
  result = nl.ndarray((64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  i_lhsT_p, i_lhsT_f = nl.mgrid[0:128, 0:64]
  i_rhs_p, i_rhs_f = nl.mgrid[0:128, 0:512]
  i_out_p, i_out_f = nl.mgrid[0:64, 0:512]

  lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
  rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

  result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  nl.store(result[i_out_p, i_out_f], value=result_sbuf)

  return result
```

```python
@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)

  return result
```

```python
@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

  for m in nl.affine_range(M // TILE_M):
    lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    for k in nl.affine_range(K // TILE_K):
      lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(lhsT[k * TILE_K + i_lhsT.p,
                                                       m * TILE_M + i_lhsT.x])

    for n in nl.affine_range(N // TILE_N):

      rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)
      for k in nl.affine_range(K // TILE_K):
        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                     n * TILE_N + i_rhs.x])

      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        res_psum[...] += nl.matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True)

      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

  return result
```

```python
@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation while blocking the
     free dimensions of the LHS and RHS to improve memory access pattern.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  for m in nl.affine_range(M // BLOCK_M):
    lhsT_tiles = nl.ndarray(
        (TILES_IN_BLOCK_M, K // TILE_K, nl.par_dim(TILE_K), TILE_M),
        dtype=lhsT.dtype,
        buffer=nl.sbuf)
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      for k in nl.affine_range(K // TILE_K):
        lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x] = nl.load(
            lhsT[k * TILE_K + i_lhsT.p,
                 (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x])

    for n in nl.affine_range(N // BLOCK_N):
      rhs_tiles = nl.ndarray(
          (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
          dtype=rhs.dtype,
          buffer=nl.sbuf)
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        for k in nl.affine_range(K // TILE_K):
          rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
              rhs[k * TILE_K + i_rhs.p,
                  (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x])

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            res_psum += nl.matmul(lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x],
                                  rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                                  transpose_x=True)

          res_sb = nl.copy(res_psum, dtype=result.dtype)
          nl.store(result[(m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p,
                          (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x],
                   value=res_sb)

  return result
```

```python
@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    for k in nl.sequential_range(NUM_BLOCK_K):
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x])

      for m in nl.affine_range(NUM_BLOCK_M):
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype=lhsT.dtype,
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nisa.nc_matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                   dtype=result_tiles.dtype,
                                   buffer=nl.sbuf)

        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])

  return result
```

## mamba_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def mamba_v1(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    _, state_size = A.shape

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    for i_batch in nl.affine_range(batch_size):
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

        for i_state in nl.affine_range(state_size):

            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize

                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])

    return output
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def mamba_v2(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    for i_batch in nl.affine_range(batch_size):

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            for i_state in nl.affine_range(state_size):
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                scanC_accum[0:channel_psize, 0:seq_len] += scanC

            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])

    return output
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def mamba_v3(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    seq_len_fsize = 512
    n_seq_len_tile = seq_len // seq_len_fsize

    assert channels % channel_psize == 0
    assert seq_len % seq_len_fsize == 0

    for i_batch in nl.affine_range(batch_size):

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            for i_state in nl.affine_range(state_size):
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_len_start = i_seq_len_tile * seq_len_fsize

                    deltaA = nisa.activation(op=nl.exp,
                            data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            scale=A_i)

                    B_i = nl.load(B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    deltaU = nisa.tensor_tensor(delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            op=nl.multiply)
                    B_i_bcast = B_i.broadcast_to((channel_psize, seq_len_fsize))
                    deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                    scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                            op0=np.multiply, op1=np.add)
                    scan_init[...] = scan_res[0:channel_psize, seq_len_fsize-1]

                    C_i = nl.load(C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    C_i_bcast = C_i.broadcast_to((channel_psize, seq_len_fsize))
                    scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                    scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize] += scanC

            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    return output
```

## bert_model.py

```python
import tensorflow as tf
from tensorflow.neuron import fuse
from tensorflow.core.framework import attr_value_pb2

# Example 1: Using tensorflow.neuron.fuse for compiler optimization
fuser = fuse(compiler_args=['--fp32-cast', 'matmult'], timeout=360000)
bert.encoder = fuser(bert.encoder)

# Example 2: Creating TensorFlow placeholders with dynamic batch size
input_ids_ph_shape = input_ids.shape.as_list()
input_ids_ph_shape[0] = None
input_ids_ph = tf.placeholder(input_ids.dtype, input_ids_ph_shape, name='input_ids')

# Example 3: Building embedding layer with gather and reshape operations
with tf.name_scope('bert/embeddings'):
    expand_dims = tf.expand_dims(input_ids_ph, axis=-1)
    batch_size = tf.shape(input_ids_ph)[0]
    reshape = tf.reshape(expand_dims, [batch_size * seq_len])
    gatherv2 = tf.gather(weights_dict['bert/embeddings/word_embeddings:0'], reshape, axis=0)
    reshape_1 = tf.reshape(gatherv2, [batch_size, seq_len, hid_size])

# Example 4: Creating attention bias tensor with casting
with tf.name_scope('bert/encoder'):
    reshape = tf.reshape(input_mask_ph, [batch_size, 1, 1, seq_len])
    bias_tensor = tf.cast(reshape, tf.float32)
    bias_tensor = 1.0 - bias_tensor
    bias_tensor = bias_tensor * -10000.0
    bias_tensor = tf.cast(bias_tensor, dtype)

# Example 5: Modifying graph operations programmatically
for rts in dummy_reshapes:
    neuron_op = rts.consumers()[0]
    neuron_op._update_input(list(neuron_op.inputs).index(rts), rts.op.inputs[0])

# Example 6: Accessing and modifying NeuronOp attributes
neuron_op_node = [node for node in new_graph_def.node if node.op == 'NeuronOp'][0]
neuron_op_node.attr['input_batch_axis'].list.i[:] = [0, 0]
neuron_op_node.attr['output_batch_axis'].list.i[:] = [0]

# Example 7: Importing and validating compiled graph
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(new_graph_def, name='')
    neuron_op = [op for op in sess.graph.get_operations() if op.type == 'NeuronOp'][0]
    if not neuron_op.get_attr('executable'):
        raise AttributeError('Neuron executable (neff) is empty.')

# Example 8: Self-attention computation with multi-head reshaping
query_r = tf.reshape(query, [batch_size, seq_len, num_heads, head_size])
query_rt = tf.transpose(query_r, [0, 2, 1, 3])
query_key = tf.matmul(query_rt, key_rt, transpose_b=True)
softmax_weights = tf.nn.softmax(bias_query_key)

# Example 9: Layer normalization implementation
input_tensor = tf.cast(input_tensor, dtype)
mean = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True, name='mean')
residuals = tf.subtract(input_tensor, mean, name='residuals')
var = tf.reduce_mean(residuals * residuals, axis=[-1], keepdims=True, name='var')
rsqrt = tf.rsqrt(var + eps)
norm_output = tf.multiply(residuals, rsqrt, name='normalized')
output_tensor = norm_output * gamma + beta

# Example 10: GELU activation approximations
def gelu_tanh(tensor):
    pow3 = 0.044714998453855515 * tensor * tensor * tensor + tensor
    shifted = (tf.tanh(0.7978845834732056 * pow3) + 1.0) * tensor
    return tf.multiply(shifted, 0.5)

def gelu_sigmoid(tensor):
    return tf.sigmoid(1.702 * tensor) * tensor
```

## ssd300_model.py

```python
import numpy as np
import tensorflow as tf
from functools import partial

def decode_jpeg_resize(input_tensor, image_size):
    # decode jpeg
    tensor = tf.image.decode_png(input_tensor, channels=3)

    # resize
    decoded_shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    decoded_shape_hw = decoded_shape[0:2]
    decoded_shape_hw_float32 = tf.cast(decoded_shape_hw, tf.float32)
    tensor = tf.image.resize(tensor, image_size)

    # normalize
    tensor -= np.array([0.485, 0.456, 0.406]).astype(np.float32) * 255.0
    return tensor, decoded_shape_hw_float32[::-1]


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        tensor, bbox_scale_hw = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return tensor, bbox_scale_hw


def tf_Conv2d(input_tensor, module, first_conv=False):
    np_dtype = input_tensor.dtype.as_numpy_dtype
    kernel_np = module.weight.detach().numpy().transpose([2, 3, 1, 0])
    if first_conv:
        kernel_np /= (np.array([0.229, 0.224, 0.225]).astype(np.float32) * 255.0)[:, np.newaxis]
    kernel = tf.constant(kernel_np.astype(np_dtype))
    if any(module.padding):
        pad_h, pad_w = module.padding
        padding = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
        input_tensor = tf.pad(input_tensor, padding)
    stride_h, stride_w = module.stride
    tensor = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride_h, stride_w, 1], padding='VALID')
    if module.bias is not None:
        bias = tf.constant(module.bias.detach().numpy().astype(np_dtype))
        tensor = tf.nn.bias_add(tensor, bias)
    return tensor


def tf_BatchNorm2d(input_tensor, module):
    def _norm_np(ts):
        return ts.astype(input_tensor.dtype.as_numpy_dtype)
    mean = _norm_np(module.running_mean.detach().numpy())
    offset = _norm_np(module.bias.detach().numpy())
    inv_std = np.sqrt(module.running_var.detach().numpy() + module.eps)
    scale_inv_std = _norm_np(module.weight.detach().numpy() / inv_std)
    return scale_inv_std * (input_tensor - mean) + offset


def tf_MaxPool2d(input_tensor, module):
    pad = module.padding
    tensor = tf.pad(input_tensor, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool2d(tensor, ksize=module.kernel_size, strides=module.stride, padding='VALID')


def tf_Bottleneck(input_tensor, module):
    tensor = tf_Conv2d(input_tensor, module.conv1)
    tensor = tf_BatchNorm2d(tensor, module.bn1)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv2)
    tensor = tf_BatchNorm2d(tensor, module.bn2)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv3)
    tensor = tf_BatchNorm2d(tensor, module.bn3)
    if module.downsample is not None:
        input_tensor = tf_Conv2d(input_tensor, module.downsample[0])
        input_tensor = tf_BatchNorm2d(input_tensor, module.downsample[1])
    return tf.nn.relu(input_tensor + tensor)


def scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh):
    """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    with tf.name_scope('ScaleBackBatch'):
        bboxes_in = tf.transpose(bboxes_in, [0, 2, 1])
        scores_in = tf.transpose(scores_in, [0, 2, 1])

        bboxes_xy = bboxes_in[:, :, :2]
        bboxes_wh = bboxes_in[:, :, 2:]
        bboxes_xy *= scale_xy
        bboxes_wh *= scale_wh

        bboxes_xy = bboxes_xy * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        bboxes_wh = tf.exp(bboxes_wh) * dboxes_xywh[:, :, 2:]

        bboxes_wh_half = 0.5 * bboxes_wh
        bboxes_lt = bboxes_xy - bboxes_wh_half
        bboxes_rb = bboxes_xy + bboxes_wh_half

        bboxes_in = tf.concat([bboxes_lt, bboxes_rb], axis=-1)

        return bboxes_in, tf.nn.softmax(scores_in, axis=-1)
```

## ssd300_model.py

```python
import numpy as np
import tensorflow as tf
from functools import partial

def decode_jpeg_resize(input_tensor, image_size):
    # decode jpeg
    tensor = tf.image.decode_png(input_tensor, channels=3)

    # resize
    decoded_shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    decoded_shape_hw = decoded_shape[0:2]
    decoded_shape_hw_float32 = tf.cast(decoded_shape_hw, tf.float32)
    tensor = tf.image.resize(tensor, image_size)

    # normalize
    tensor -= np.array([0.485, 0.456, 0.406]).astype(np.float32) * 255.0
    return tensor, decoded_shape_hw_float32[::-1]


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        tensor, bbox_scale_hw = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return tensor, bbox_scale_hw


def tf_Conv2d(input_tensor, module, first_conv=False):
    np_dtype = input_tensor.dtype.as_numpy_dtype
    kernel_np = module.weight.detach().numpy().transpose([2, 3, 1, 0])
    if first_conv:
        kernel_np /= (np.array([0.229, 0.224, 0.225]).astype(np.float32) * 255.0)[:, np.newaxis]
    kernel = tf.constant(kernel_np.astype(np_dtype))
    if any(module.padding):
        pad_h, pad_w = module.padding
        padding = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
        input_tensor = tf.pad(input_tensor, padding)
    stride_h, stride_w = module.stride
    tensor = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride_h, stride_w, 1], padding='VALID')
    if module.bias is not None:
        bias = tf.constant(module.bias.detach().numpy().astype(np_dtype))
        tensor = tf.nn.bias_add(tensor, bias)
    return tensor


def tf_BatchNorm2d(input_tensor, module):
    def _norm_np(ts):
        return ts.astype(input_tensor.dtype.as_numpy_dtype)
    mean = _norm_np(module.running_mean.detach().numpy())
    offset = _norm_np(module.bias.detach().numpy())
    inv_std = np.sqrt(module.running_var.detach().numpy() + module.eps)
    scale_inv_std = _norm_np(module.weight.detach().numpy() / inv_std)
    return scale_inv_std * (input_tensor - mean) + offset


def tf_MaxPool2d(input_tensor, module):
    pad = module.padding
    tensor = tf.pad(input_tensor, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool2d(tensor, ksize=module.kernel_size, strides=module.stride, padding='VALID')


def tf_Bottleneck(input_tensor, module):
    tensor = tf_Conv2d(input_tensor, module.conv1)
    tensor = tf_BatchNorm2d(tensor, module.bn1)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv2)
    tensor = tf_BatchNorm2d(tensor, module.bn2)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv3)
    tensor = tf_BatchNorm2d(tensor, module.bn3)
    if module.downsample is not None:
        input_tensor = tf_Conv2d(input_tensor, module.downsample[0])
        input_tensor = tf_BatchNorm2d(input_tensor, module.downsample[1])
    return tf.nn.relu(input_tensor + tensor)


def scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh):
    """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    with tf.name_scope('ScaleBackBatch'):
        bboxes_in = tf.transpose(bboxes_in, [0, 2, 1])
        scores_in = tf.transpose(scores_in, [0, 2, 1])

        bboxes_xy = bboxes_in[:, :, :2]
        bboxes_wh = bboxes_in[:, :, 2:]
        bboxes_xy *= scale_xy
        bboxes_wh *= scale_wh

        bboxes_xy = bboxes_xy * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        bboxes_wh = tf.exp(bboxes_wh) * dboxes_xywh[:, :, 2:]

        bboxes_wh_half = 0.5 * bboxes_wh
        bboxes_lt = bboxes_xy - bboxes_wh_half
        bboxes_rb = bboxes_xy + bboxes_wh_half

        bboxes_in = tf.concat([bboxes_lt, bboxes_rb], axis=-1)

        return bboxes_in, tf.nn.softmax(scores_in, axis=-1)
```

## test_nki_isa_dma_copy.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

# Example 1: Basic DMA copy
@nki.jit(mode="simulation")
def nki_dma_copy(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
  nisa.dma_copy(dst=b, src=a)
  return b

# Example 2: Indirect load with out-of-bounds error handling
@nki.jit(mode="simulation")
def nki_indirect_load_oob_err(in_tensor):
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n//2, 0:m]

  expr_arange = 2*nl.arange(n//2)[:, None]
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
  nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)

  nl.store(out_tensor, value=out_tile)
  return out_tensor

# Example 3: Indirect load with out-of-bounds skip
@nki.jit(mode="simulation")
def nki_indirect_load_oob_skip(in_tensor):
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n//2, 0:m]

  expr_arange = 3*nl.arange(n//2)[:, None] 
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
  nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.skip)

  nl.store(out_tensor, value=out_tile)
  return out_tensor

# Example 4: Indirect store with read-modify-write
@nki.jit(mode="simulation")
def nki_indirect_store_rmw(in_tensor):
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  expr_arange = 2*nl.arange(n)[:, None]
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, dst_rmw_op=np.add)

  return out_tensor

# Example 5: Indirect store with out-of-bounds error handling
@nki.jit(mode="simulation")
def nki_indirect_store_oob_err(in_tensor):
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  expr_arange = 2*nl.arange(n)[:, None]
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)

  return out_tensor

# Example 6: Indirect store with out-of-bounds skip
@nki.jit(mode="simulation")
def nki_indirect_store_oob_skip(in_tensor):
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  expr_arange = 3*nl.arange(n)[:, None] 
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.skip)

  return out_tensor

# Example 7: DMA copy with SWDGE
@nki.jit(mode='simulation')
def nki_dma_copy_swdge(in_tensor):
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  nisa.dma_copy(dst=out_tensor, src=in_tensor, dge_mode=nisa.dge_mode.swdge)
  return out_tensor

# Example 8: DMA copy with HWDGE
@nki.jit(mode='simulation', platform_target='trn2')
def nki_dma_copy_hwdge(in_tensor):
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  inp_tile: tensor[128, 512] = nl.load(in_tensor)
  out_tile: tensor[128, 512] = nl.zeros_like(inp_tile, buffer=nl.sbuf)
  nisa.dma_copy(dst=out_tile, src=inp_tile, dge_mode=nisa.dge_mode.hwdge)
  nl.store(out_tensor, value=out_tile)
  return out_tensor
```

## sdxl_base_1024_compile.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import math
from diffusers.models.attention_processor import Attention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from packaging import version

def apply_neuron_attn_override(
    diffusers_pkg, get_attn_scores_func, neuron_scaled_dot_product_attention
):
    diffusers_version = version.parse(diffusers_pkg.__version__)
    use_new_diffusers = diffusers_version >= version.parse("0.18.0")
    if use_new_diffusers:
        diffusers_pkg.models.attention_processor.Attention.get_attention_scores = (
            get_attn_scores_func
        )
    else:
        diffusers_pkg.models.cross_attention.CrossAttention.get_attention_scores = (
            get_attn_scores_func
        )

    if hasattr(F, "scaled_dot_product_attention"):
        F.scaled_dot_product_attention = neuron_scaled_dot_product_attention

def get_attention_scores_neuron(self, query, key, attn_mask):    
    if query.size() == key.size():
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)
    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled

def neuron_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=None, is_causal=None
):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape
        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])

    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)

    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )

    return attn_out

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None
    ):
        out_tuple = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs=None,
        return_dict=False,
        cross_attention_kwargs=None,
    ):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)

class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple

class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(text_embeds=out_tuple[0], last_hidden_state=out_tuple[1], hidden_states=out_tuple[2])
```

```python
# Tracing and compiling text encoders
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)
torch.jit.save(neuron_text_encoder, text_encoder_filename)

# Tracing and compiling UNet
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)
torch.jit.save(unet_neuron, unet_filename)

# Tracing and compiling VAE decoder
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder')
)
torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, decoder_filename)
```

## neuron_profile_for_nki.rst

```python
import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "
```

```python
# prof-kernel.py - Full example referenced in documentation
# (literalinclude from examples/prof-kernel.py)
# Content not provided in source, but usage pattern shown:
# python3 prof-kernel.py
```

```python
# prof-kernel-profile.py - Full example using nki.profile API
# (literalinclude from examples/prof-kernel-profile.py)
# Content not provided in source, but usage pattern shown:
# python3 prof-kernel-profile.py
```

**Note:** The source document references two complete code examples (`prof-kernel.py` and `prof-kernel-profile.py`) via `literalinclude` directives, but their actual content is not included in the provided source text. Only the environment variable setup code and command-line usage patterns are explicitly shown. The actual NKI kernel implementations and `nki.profile` decorator usage would be found in those referenced example files.

## troubleshooting-guide.rst

```python
import os
import mxnet as mx
from concurrent import futures

NUM_PARALLEL = 4
os.environ['NEURONCORE_GROUP_SIZES'] = ','.join('1' for _ in range(NUM_PARALLEL))
   
data_iter = []
for i in range(NUM_PARALLEL):
    data_iter.append(mx.io.ImageRecordIter(
        path_imgrec=recfile_base, data_shape=(3, 224, 224), batch_size=1,            
        prefetch_buffer=1,
        num_parts=NUM_PARALLEL, part_index=i))

sym, args, auxs = mx.model.load_checkpoint('resnet-50_compiled', 0)

exec_list = []
for i in range(NUM_PARALLEL):
    exec = sym.bind(ctx=mx.neuron(i), args=args, aux_states=auxs, grad_req='null')
    exec_list.append(exec)

def single_thread_infer(i):
    for batch in data_iter[i]:
        img = batch.data[0]
        label = batch.label
        feed_dict = {'data': img}
        exe = exec_list[i]
        exe.copy_params_from(feed_dict)
        exe.forward()
        out = exe.outputs[0]

future_list = []
with futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
    for i in range(NUM_PARALLEL):
        future_list.append(executor.submit(single_thread_infer, i))
```

```python
import os

os.environ['NEURONCORE_GROUP_SIZES'] = '1'
```

## sd_attention_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  """
  Fused self attention kernel for small head dimension Stable Diffusion workload, 
  simplified for this tutorial. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV projection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (seq_q, d_head)
   - k_ptr: shape   (seq_k, d_head)
   - v_ptr: shape   (seq_v, d_head)
   - out_ptr: shape (seq_q, d_head)
   - We use seq_q and seq_k and seq_v just for clarity, this kernel requires 
   seq_q == seq_k == seq_v

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen,d_head), \
  f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
      v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
      q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]
      ],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                              stationary=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number nl.fp32.min to replace -inf similar to what neuronx-cc uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=nl.fp32.min, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

  return out_ref
```

## test_neuronperf.py

```python
import neuronperf

# Timer usage
timer = neuronperf.Timer()
with timer:
    time.sleep(1)

total_duration_s = timer.total_duration("s")
durations = timer.durations("s")
timestamps = timer.timestamps()
timer_length = len(timer)

# Timestamp conversion
converted = neuronperf.timestamp_convert(1, "s", "ms")
times_array = neuronperf.timestamp_convert(np.array([1, 2, 3]), "s", "ms")

# Model index creation
index = neuronperf.model_index.create("dummy_model.ext", model_name="dummy")
neuronperf.model_index.save(index, filename="index.json")
loaded_index = neuronperf.model_index.load("index.json")
neuronperf.model_index.delete("index.json")

# Model index operations
neuronperf.model_index.copy(index, "new_index.json", "new_models")
neuronperf.model_index.move("index.json", "new_index.json", "new_models")

# Model index combining
combined = neuronperf.model_index.append(index1, index2, index3)
filtered = neuronperf.model_index.filter(combined, filename="fake", performance_level=2)

# Benchmarking
benchmarker = neuronperf.benchmarking.Benchmarker(
    id=0, device_id=0, load_fn=load_fn, model_filename="test", inputs=[], workers_per_model=2
)
benchmarker.start()
benchmarker.stop()
status = benchmarker.status
n_infs = benchmarker.n_infs

# CPU benchmarking
results = neuronperf.cpu.benchmark(
    neuronperf.DummyModel,
    inputs=[np.array([1, 2, 3, 4])],
    duration=2,
    n_models=4,
    multiprocess=False,
    multiinterpreter=False,
    verbosity=2,
    return_timers=True,
)

# Generic benchmarking
reports = neuronperf.benchmark(
    load_fn=load_fn,
    model_filename="dummy_filename",
    inputs=[[1]],
    duration=2,
    n_models=4,
    multiprocess=False,
    multiinterpreter=False,
    verbosity=2,
)

# Reporting
reports = neuronperf.get_reports(benchmarker_results)
neuronperf.print_reports(reports)
csv_file = neuronperf.write_csv(reports)
json_file = neuronperf.write_json(reports)
```

## spmd_multiple_nc_tensor_addition.rst

```python
import nki
import nki.language as nl

@nki.jit
def nki_tensor_add_kernel_(a_input, b_input, c_output):
    """
    NKI kernel for tensor addition using SPMD execution across multiple Neuron Cores.
    
    Each worker operates on a tile of size [128, 512]:
    - First axis (partition-dimension): tiled into blocks of 128
    - Second axis (free-dimension): tiled into blocks of 512
    """
    # Calculate offsets based on program_id for SPMD execution
    program_id = nl.program_id(0)
    
    # Generate tile indices with offsets
    i = nl.arange(128) + program_id * 128
    j = nl.arange(512)
    
    # Load tiles from input tensors
    a_tile = a_input[i[:, None], j[None, :]]
    b_tile = b_input[i[:, None], j[None, :]]
    
    # Perform addition
    c_tile = a_tile + b_tile
    
    # Store result to shared HBM
    c_output[i[:, None], j[None, :]] = c_tile


def nki_tensor_add_nc2(a_input, b_input):
    """
    Helper function to launch tensor addition kernel across 2 Neuron Cores.
    
    Shards workload across 2 cores with:
    - Physical NC [0]: kernel[n, m] where n is 0 or even
    - Physical NC [1]: kernel[n, m] where n is odd
    """
    # Tile sizes
    tile_x = 128
    tile_y = 512
    num_cores = 2
    
    # Calculate grid dimensions accounting for 2 cores
    grid_x = a_input.shape[0] // (tile_x * num_cores)
    grid_y = a_input.shape[1] // tile_y
    
    # Create output tensor
    c_output = nl.zeros(a_input.shape, dtype=a_input.dtype)
    
    # Launch kernel with SPMD execution across 2 Neuron Cores
    nki_tensor_add_kernel_[nl.spmd_dim(grid_x, nl.nc(2)), grid_y](
        a_input, b_input, c_output
    )
    
    return c_output
```

## framework_custom_op.rst

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.randn(256, 1024, dtype=torch.float32).to(device)
b = torch.randn(256, 1024, dtype=torch.float32).to(device)
c = a + b
out = a * b * c
```

```python
device = xm.xla_device()
a = torch.randn(256, 1024, dtype=torch.float32).to(device)
b = torch.randn(256, 1024, dtype=torch.float32).to(device)
c = nki_tensor_add(a, b)
out = a * b * c
```

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_customop_tutorial(a, b):
    c = a + b
    out = a * b * c
    return out

seed = jax.random.PRNGKey(0)
seed_a, seed_b = jax.random.split(seed)
a = jax.random.normal(seed_a, (256, 1024), dtype=jnp.float32)
b = jax.random.normal(seed_b, (256, 1024), dtype=jnp.float32)
```

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_customop_tutorial(a, b):
    c = nki_tensor_add(a, b)
    out = a * b * c
    return out

seed = jax.random.PRNGKey(0)
seed_a, seed_b = jax.random.split(seed)
a = jax.random.normal(seed_a, (256, 1024), dtype=jnp.float32)
b = jax.random.normal(seed_b, (256, 1024), dtype=jnp.float32)
```

```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

class NkiAddFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return nki_tensor_add(a, b)

    @staticmethod
    def backward(ctx, dy, *args):
        return dy, dy

a = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
b = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
c = NkiAddFunc.apply(a, b)
out = a * b * c

loss = out.sum()
loss.backward()

xm.mark_step()
```

```python
import jax

@jax.custom_vjp
def nki_add_func(a, b):
    return nki_tensor_add(a, b)

def f_forward(a, b):
    return nki_add_func(a, b), (a, b)

def f_backward(res, grad):
    return grad, grad

nki_add_func.defvjp(f_forward, f_backward)

@jax.jit
def jax_customop_tutorial_and_grad(a, b):
    out = nki_add_func(a, b) * a * b
    grad = jax.grad(lambda x, y: (nki_add_func(x, y) * x * y).sum(), argnums=(0, 1))(a, b)
    return out, *grad
```

## spmd_tensor_addition.rst

```python
import neuronxcc.nki.language as nl
from neuronxcc.nki import nki_jit

@nki_jit
def nki_tensor_add_kernel_(a_input, b_input):
    """
    NKI kernel for tensor addition using SPMD programming model.
    Operates on tiles of size [128, 512].
    """
    # Allocate output tensor
    c_output = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype)
    
    # Get worker ID for SPMD execution
    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    
    # Define tile sizes
    tile_size_x = 128
    tile_size_y = 512
    
    # Calculate offsets based on program ID
    offset_x = pid_x * tile_size_x
    offset_y = pid_y * tile_size_y
    
    # Generate tile indices using advanced indexing
    indices_x = nl.arange(tile_size_x)[:, None]
    indices_y = nl.arange(tile_size_y)[None, :]
    
    x_idx = offset_x + indices_x
    y_idx = offset_y + indices_y
    
    # Load tiles from input tensors
    a_tile = nl.load(a_input[x_idx, y_idx])
    b_tile = nl.load(b_input[x_idx, y_idx])
    
    # Compute sum
    c_tile = a_tile + b_tile
    
    # Store result back to output tensor
    nl.store(c_output[x_idx, y_idx], c_tile)
    
    return c_output


def nki_tensor_add(a, b):
    """
    Helper function to launch the NKI tensor addition kernel with appropriate grid/block sizes.
    Assumes tensor sizes are multiples of maximum tile sizes.
    """
    tile_size_x = 128
    tile_size_y = 512
    
    # Calculate grid dimensions
    grid_x = a.shape[0] // tile_size_x
    grid_y = a.shape[1] // tile_size_y
    
    # Launch kernel with 2D grid
    return nki_tensor_add_kernel_[grid_x, grid_y](a, b)
```

## guide-torch-neuron-vs-torch-neuronx-inference.rst

```python
import torch
import torchvision
import torch_neuron

model = torchvision.models.resnet50(pretrained=True).eval()
image = torch.rand(1, 3, 224, 224)

trace = torch_neuron.trace(model, image)
```

```python
import torch
import torchvision
import torch_neuronx

model = torchvision.models.resnet50(pretrained=True).eval()
image = torch.rand(1, 3, 224, 224)

trace = torch_neuronx.trace(model, image)
```

## customop-mlp-training.rst

```python
import torch

torch.ops.load_library('librelu.so')

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ops.my_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        return torch.ops.my_ops.relu_backward(grad, input), None
```

```c++
torch::Tensor relu_forward(const torch::Tensor& t_in) {
    // ... implementation
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_in_acc[i][j] : 0.0;
    // ...
}

torch::Tensor relu_backward(const torch::Tensor& t_grad, const torch::Tensor& t_in) {
    // ... implementation
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_grad_acc[i][j] : 0.0;
    // ...
}

TORCH_LIBRARY(my_ops, m) {
    m.def("relu_forward", &relu_forward);
    m.def("relu_backward", &relu_backward);
}
```

```python
import torch.utils.cpp_extension
import os

torch.utils.cpp_extension.load(
    name='librelu',
    sources=['relu.cpp'],
    is_python_module=False,
    build_directory=os.getcwd()
)
```

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import my_ops

class MLP(nn.Module):
    def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        f1 = self.fc1(x)
        r1 = my_ops.Relu.apply(f1)
        f2 = self.fc2(r1)
        r2 = my_ops.Relu.apply(f2)
        f3 = self.fc3(r2)
        return torch.log_softmax(f3, dim=1)
```

```c++
torch::Tensor relu_fwd_shape(torch::Tensor t_in) {
    torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
    return t_out;
}

torch::Tensor relu_bwd_shape(torch::Tensor t_grad, torch::Tensor t_in) {
    torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
    return t_out;
}

NEURON_LIBRARY(my_ops, m) {
    m.def("relu_forward", &relu_fwd_shape, "relu_forward");
    m.def("relu_backward", &relu_bwd_shape, "relu_backward");
}
```

```python
from torch_neuronx.xla_impl import custom_op

custom_op.load(
    name='relu',
    compute_srcs=['relu.cpp'],
    shape_srcs=['shape.cpp'],
    build_directory=os.getcwd()
)
```

```python
import torch
import torch_neuronx
from torch_neuronx.xla_impl import custom_op

custom_op.load_library('librelu.so')

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ops.my_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        return torch.ops.my_ops.relu_backward(grad, input), None
```

## sd_15_512_compile.py

```python
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import copy
import time
import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from packaging import version
import diffusers
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention


def get_attention_scores(self, query, key, attn_mask):    
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = cust_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())
```

## rmsnorm.rst

```python
import nki
import nki.language as nl
import nki.isa as nisa
import math

@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, output):
    """
    Compute RMSNorm of a 2D tensor.
    
    Args:
        a_tensor: Input tensor of shape [num_rows, embedding_size]
        g_tensor: RMSNorm weight of shape [embedding_size]
        output: Output tensor of shape [num_rows, embedding_size]
    """
    num_rows = a_tensor.shape[0]
    embedding_size = a_tensor.shape[1]
    
    # Reshape g_tensor to 2D for SBUF compatibility
    g_tensor_2d = nl.reshape(g_tensor, (1, embedding_size))
    
    # Load g_tensor once outside the main loop for maximum reuse
    g_tile = nl.load(g_tensor_2d, mask=(0 < num_rows))
    
    # Iterate over tiles of a_tensor with partition axis size of 128
    for i in nl.affine_range(math.ceil(num_rows / 128)):
        # Load one tile of a_tensor
        a_tile = nl.load(
            a_tensor[i * 128:i * 128 + 128],
            mask=(i * 128 + nl.arange(128)[:, None] < num_rows)
        )
        
        # Compute RMS: sqrt(mean(a^2))
        a_squared = nl.multiply(a_tile, a_tile)
        rms = nl.sqrt(nl.mean(a_squared, axis=1, keepdims=True))
        
        # Compute RMS reciprocal for division
        rms_reciprocal = nl.reciprocal(rms)
        
        # Free-axis broadcast divide: a / RMS(a)
        out_tile = nl.multiply(a_tile, rms_reciprocal)
        
        # Partition-axis broadcast of g_tensor
        num_active_rows = nl.minimum(num_rows - i * 128, 128)
        g_broadcasted = nl.broadcast_to(
            g_tile,
            (num_active_rows, embedding_size)
        )
        
        # Element-wise multiply with broadcasted g_tensor
        out_tile = nl.multiply(out_tile, g_broadcasted)
        
        # Store result back to HBM
        nl.store(
            output[i * 128:i * 128 + 128],
            out_tile,
            mask=(i * 128 + nl.arange(128)[:, None] < num_rows)
        )
```

## example_app.cpp

```cpp
#include <vector>
#include <torch/script.h>

typedef std::vector<std::vector<long>> Input;

// construct a single input: input_ids, attention_mask, and token_type_ids from two input sentences
Input get_input(const std::string& sentence_1, const std::string& sentence_2)
{
    const size_t seq_len = 128;
    const long start_token = 101;
    const long end_token = 102;

    // ensure the concatenated sentences + separator tokens do not exceed the compiled sequence length
    assert(sentence_1.size() + sentence_2.size() + 3 <= seq_len);

    // tokenize the input sentence using the HuggingFace Tokenizers library
    std::vector<long> input_ids(seq_len, 0);
    input_ids[0] = start_token;
    size_t pos = 1; // current write position in input_ids

    // tokenize sentence_1 and copy to output buffer
    std::vector<uint32_t> buffer(seq_len, 0);
    remote_rust_encode(sentence_1.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
        input_ids[pos] = buffer[i];
    }

    // mark end of sentence_1
    input_ids[pos++] = end_token;
    const size_t sentence_2_start = pos;

    // tokenize sentence_2 and copy to output buffer
    std::fill(buffer.begin(), buffer.end(), 0);
    remote_rust_encode(sentence_2.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
        input_ids[pos] = buffer[i];
    }

    // mark end of sentence_2
    input_ids[pos++] = end_token;

    // construct attention mask
    std::vector<long> attention_mask(seq_len, 0);
    for (size_t i = 0; i < seq_len; ++i) attention_mask[i] = input_ids[i] ? 1 : 0;

    // token type ids are 0s for sentence_1 (incl. separators), 1s for sentence_2
    std::vector<long> token_type_ids(seq_len, 0);
    for (size_t i = sentence_2_start; i < seq_len; i++) {
        if (!attention_mask[i]) break;
        token_type_ids[i] = 1;
    }

    return {input_ids, attention_mask, token_type_ids};
}

// reshape a vector of inputs into a proper batch
std::vector<torch::jit::IValue> get_batch(const std::vector<Input>& inputs)
{
    const size_t batch_size = 6;
    const size_t seq_len = 128;

    // must be given a full batch
    assert(inputs.size() == batch_size);

    torch::Tensor input_ids_tensor = torch::zeros({batch_size, seq_len}, at::kLong);
    torch::Tensor attention_mask_tensor = torch::zeros({batch_size, seq_len}, at::kLong);
    torch::Tensor token_type_ids_tensor = torch::zeros({batch_size, seq_len}, at::kLong);

    const auto opts = torch::TensorOptions().dtype(torch::kLong);
    for (size_t i = 0; i < batch_size; i++) {
        input_ids_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][0].data(), {seq_len}, opts);
        attention_mask_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][1].data(), {seq_len}, opts);
        token_type_ids_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][2].data(), {seq_len}, opts);
    }

    return {input_ids_tensor, attention_mask_tensor, token_type_ids_tensor};
}
```

## layernorm.rst

```python
import nki
import nki.language as nl
import math

@nki.jit
def nki_layernorm_kernel_v1(input_tensor, gamma, beta, output_tensor, epsilon=1e-5):
    """
    LayerNorm kernel using nki.language APIs only.
    Performs layer normalization on a 2D tensor.
    
    Args:
        input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
        gamma: Affine transform parameter, shape [hidden_size]
        beta: Affine transform parameter, shape [hidden_size]
        output_tensor: Output tensor, same shape as input_tensor
        epsilon: Small constant for numerical stability
    """
    # Load gamma and beta, perform partition-axis broadcast
    shift_scale_tensor = nl.load(gamma)
    shift_scale_tensor = nl.broadcast_to(shift_scale_tensor, (nl.tile_size.pmax, gamma.shape[0]))
    
    beta_tensor = nl.load(beta)
    beta_tensor = nl.broadcast_to(beta_tensor, (nl.tile_size.pmax, beta.shape[0]))
    
    # Compute trip count for partition axis
    trip_count = math.ceil(input_tensor.shape[0] / nl.tile_size.pmax)
    
    for i in range(trip_count):
        # Load one tile of input_tensor
        i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
        mask = (i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
        
        input_tile = nl.load(
            input_tensor[i * nl.tile_size.pmax : (i + 1) * nl.tile_size.pmax],
            mask=mask
        )
        
        # Compute mean and variance
        mean = nl.mean(input_tile, axis=1, keepdims=True)
        variance = nl.mean(nl.square(input_tile - mean), axis=1, keepdims=True)
        
        # Normalize
        normalized = (input_tile - mean) * nl.rsqrt(variance + epsilon)
        
        # Scale and shift
        output_tile = normalized * shift_scale_tensor + beta_tensor
        
        # Store output
        nl.store(
            output_tensor[i * nl.tile_size.pmax : (i + 1) * nl.tile_size.pmax],
            output_tile,
            mask=mask
        )
```

```python
import nki
import nki.language as nl
import nki.isa as isa
import math

@nki.jit
def nki_layernorm_kernel_v2(input_tensor, gamma, beta, output_tensor, epsilon=1e-5):
    """
    Optimized LayerNorm kernel using nki.isa APIs for mean/variance calculation
    and shift/scale operations.
    
    Args:
        input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
        gamma: Affine transform parameter, shape [hidden_size]
        beta: Affine transform parameter, shape [hidden_size]
        output_tensor: Output tensor, same shape as input_tensor
        epsilon: Small constant for numerical stability
    """
    # Load gamma and beta, perform partition-axis broadcast
    shift_scale_tensor = nl.load(gamma)
    shift_scale_tensor = nl.broadcast_to(shift_scale_tensor, (nl.tile_size.pmax, gamma.shape[0]))
    
    beta_tensor = nl.load(beta)
    beta_tensor = nl.broadcast_to(beta_tensor, (nl.tile_size.pmax, beta.shape[0]))
    
    # Compute trip count for partition axis
    trip_count = math.ceil(input_tensor.shape[0] / nl.tile_size.pmax)
    
    for i in range(trip_count):
        # Load one tile of input_tensor
        i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
        mask = (i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
        
        input_tile = nl.load(
            input_tensor[i * nl.tile_size.pmax : (i + 1) * nl.tile_size.pmax],
            mask=mask
        )
        
        # Compute mean and variance using bn_stats and bn_aggr
        mean = nl.zeros((nl.tile_size.pmax, 1), dtype=input_tile.dtype)
        variance = nl.zeros((nl.tile_size.pmax, 1), dtype=input_tile.dtype)
        
        bn_stats_trip_count = math.ceil(input_tensor.shape[1] / nl.tile_size.bn_stats_fmax)
        
        for j in range(bn_stats_trip_count):
            start_col = j * nl.tile_size.bn_stats_fmax
            end_col = min((j + 1) * nl.tile_size.bn_stats_fmax, input_tensor.shape[1])
            
            input_slice = input_tile[:, start_col:end_col]
            
            # Calculate batch statistics
            stats = isa.bn_stats(input_slice)
            mean, variance = isa.bn_aggr(stats, mean, variance)
        
        # Shift and scale using tensor_scalar in a single instruction
        normalized = isa.tensor_scalar(
            input_tile,
            mean,
            variance,
            epsilon,
            operation='normalize'
        )
        
        # Scale and shift
        output_tile = normalized * shift_scale_tensor + beta_tensor
        
        # Store output
        nl.store(
            output_tensor[i * nl.tile_size.pmax : (i + 1) * nl.tile_size.pmax],
            output_tile,
            mask=mask
        )
```

## test_nki_isa_nc_match_replace8.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np


@nki.jit(mode="simulation")
def nki_nc_match_replace8():
  N = 4
  M = 16
  data_tile = nl.rand((N, M))
  max_vals = nisa.max8(src=data_tile)

  result = nisa.nc_match_replace8(data=data_tile[:, :], vals=max_vals, imm=float('-inf'))
  result_tensor = nl.ndarray([N, M], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(result_tensor, value=result)

  return result_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8(in_tensor: nt.tensor, imm: np.float32):
  n, m = in_tensor.shape
  out_tensor = nl.ndarray([n, m], dtype=in_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
  dst_idx = nl.ndarray((n, 8), dtype=idx_tensor.dtype)

  ix, iy = nl.mgrid[0:n, 0:8]

  inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  out_tile = nisa.nc_match_replace8(
    dst_idx=dst_idx[ix, iy], data=inp_tile[:, :], vals=max_vals, imm=imm
  )

  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor[ix, iy], value=dst_idx[ix, iy])
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_mask(in_tensor: nt.tensor, imm: np.float32):
  n, m = in_tensor.shape
  out_tensor = nl.ndarray([n, m], dtype=in_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)

  ix, iy = nl.mgrid[0:n, 0:m]
  inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile[ix, iy], mask=(ix < n //2 and iy < m//2))

  out_tile = nisa.nc_match_replace8(
    dst_idx=idx_tile[:, :],
    data=inp_tile[ix, iy],
    vals=max_vals,
    imm=imm,
    mask=(ix < n // 2 and iy < m // 2),
  )

  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_3d(data_tensor: nt.tensor):
  n, b, m = data_tensor.shape
  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)

  imm = 0.0
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)
  out_tile = nisa.memset(shape=(n, b, m), value=0, dtype=data_tensor.dtype)

  iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
  ip, io = nl.mgrid[0:n, 0:8]

  inp_tile = nl.load(data_tensor[iq, ir, iw])
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  out_tile[iq, ir, iw] = nisa.nc_match_replace8(
    dst_idx=idx_tile[ip, io],
    data=inp_tile[iq, ir, iw],
    vals=max_vals[ip, io],
    imm=imm,
  )

  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_3d_inplace(data_tensor: nt.tensor):
  n, b, m = data_tensor.shape
  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)

  imm = 0.0
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)

  iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
  ip, io = nl.mgrid[0:n, 0:8]

  inp_tile = nl.load(data_tensor[iq, ir, iw])
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  inp_tile[iq, ir, iw] = nisa.nc_match_replace8(
    dst_idx=idx_tile[ip, io],
    data=inp_tile[iq, ir, iw],
    vals=max_vals[ip, io],
    imm=imm,
  )

  nl.store(out_tensor, value=inp_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor
```

## sd2_inpainting_benchmark.py

```python
import torch
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.bfloat16().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled
```

## mrpc_feature.py

```python
# coding=utf-8
import os
import csv
import time
import numpy as np
import tokenization


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def text_pair_to_model_feed_dict(text_a, text_b, tokenizer):
    fake_tsv = [['index', '#1 ID', '#2 ID', '#1 String', '#2 String'],
                ['', '', '', text_a, text_b]]
    result = create_examples(fake_tsv, "test")
    example = result[0]
    label_list = ['0', '1']
    feature = convert_single_example(ex_index=0, example=example, label_list=label_list,
                                     max_seq_length=128, tokenizer=tokenizer)
    return {
        'input_ids': np.tile(np.int32(feature.input_ids), reps=[1, 1]),
        'input_mask': np.tile(np.int32(feature.input_mask), reps=[1, 1]),
        'segment_ids': np.tile(np.int32(feature.segment_ids), reps=[1, 1]),
    }
```

## test_nki_isa_range_select.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation", platform_target="trn2")
def nki_range_select_example(on_true, bound0, bound1, compare_op0, compare_op1, range_start):
    # Create output tensors
    select_res = nl.ndarray(on_true.shape, dtype=nl.float32, buffer=nl.hbm)
    reduce_result = nl.ndarray((on_true.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
    
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])

    reduce_res_tile = nl.ndarray((on_true.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(on_true.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    result[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_res=reduce_res_tile,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    nl.store(select_res[...], value=result[...])
    nl.store(reduce_result[...], value=reduce_res_tile[...])

    return result, reduce_result
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation", platform_target="trn2")
def nki_range_select_chaining(on_true, bound0, bound1, compare_op0, compare_op1, range_start):
    # Create output tensors
    select_res = nl.ndarray(on_true.shape, dtype=np.float32, buffer=nl.hbm)
    reduce_result = nl.ndarray((on_true.shape[0], 1), dtype=np.float32, buffer=nl.hbm)
    
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])

    reduce_res_sbuf = nl.ndarray((on_true.shape[0], 1), dtype=np.float32, buffer=nl.sbuf)
    result_sbuf = nl.ndarray(on_true.shape, dtype=np.float32, buffer=nl.sbuf)
    
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    ones = nl.full(on_true.shape, fill_value=1, dtype=np.float32, buffer=nl.sbuf)
    iteration_step_size = on_true_tile.shape[0]
    
    for i in range(1, 2):
        on_true_tile[...] = nl.add(on_true_tile, ones)
        
        result_sbuf[...] = nisa.range_select(
            on_true_tile=on_true_tile,
            comp_op0=compare_op0,
            comp_op1=compare_op1,
            bound0=bound0_tile,
            bound1=bound1_tile,
            reduce_cmd=nisa.reduce_cmd.reduce,
            reduce_op=np.max,
            range_start=range_start + (i * iteration_step_size),
            on_false_value=nl.fp32.min
        )

    range_start = range_start + (2 * iteration_step_size)
    
    on_true_tile[...] = nl.add(on_true_tile, ones)
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reduce,
        reduce_res=reduce_res_sbuf[...],
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    nl.store(select_res[...], value=result_sbuf[...])
    nl.store(reduce_result[...], value=reduce_res_sbuf[...])

    return select_res, reduce_result
```

## sd_4x_upscaler_compile.py

```python
import torch
import torch.nn as nn
import torch_neuronx

def apply_neuron_attn_override(
    diffusers_pkg, get_attn_scores_func, neuron_scaled_dot_product_attention
):
    diffusers_version = version.parse(diffusers_pkg.__version__)
    use_new_diffusers = diffusers_version >= version.parse("0.18.0")
    if use_new_diffusers:
        diffusers_pkg.models.attention_processor.Attention.get_attention_scores = (
            get_attn_scores_func
        )
    else:
        diffusers_pkg.models.cross_attention.CrossAttention.get_attention_scores = (
            get_attn_scores_func
        )

    if hasattr(F, "scaled_dot_product_attention"):
        F.scaled_dot_product_attention = neuron_scaled_dot_product_attention


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = cust_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = cust_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)

    return attention_probs


def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


def neuron_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=None, is_causal=None
):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape

        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])

        query, key, value = map(to3d, [query, key, value])

    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)

    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )

    return attn_out


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels,
        cross_attention_kwargs=None,
    ):
        out_tuple = self.unet(
            sample, timestep, encoder_hidden_states, class_labels, return_dict=False
        )
        return out_tuple


class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask=None):
        return [self.neuron_text_encoder(emb)["last_hidden_state"]]


# Compile text encoder
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder,
    emb,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

# Save the compiled text encoder
torch.jit.save(text_encoder_neuron, text_encoder_filename)


# Compile vae decoder
decoder_neuron = torch_neuronx.trace(
    decoder,
    decoder_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Save the compiled vae decoder
torch.jit.save(decoder_neuron, decoder_filename)


# Compile unet
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# Save compiled unet
torch.jit.save(unet_neuron, unet_filename)
```

## customop-mlp-perf-opt.rst

```c++
torch::Tensor relu_forward(const torch::Tensor& t_in) {
    size_t num_elem = t_in.numel();
    torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat); 

    static constexpr size_t buffer_size = 1024;
    float *tcm_buffer = (float*)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

    if (tcm_buffer != nullptr) {
        auto t_in_tcm_acc = t_in.tcm_accessor();
        auto t_out_tcm_acc = t_out.tcm_accessor();

        for (size_t i = 0; i < num_elem; i += buffer_size) {
            size_t remaining_elem = num_elem - i;
            size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

            t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, i, copy_size);
            for (size_t j = 0; j < copy_size; j++) {
                tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
            }
            t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, i, copy_size);
        }
    }
    torch::neuron::tcm_free(tcm_buffer);
    return t_out;
}
```

```c++
torch::Tensor relu_forward(const torch::Tensor& t_in) {
    size_t num_elem = t_in.numel();
    torch::Tensor t_out = get_dst_tensor();

    uint32_t cpu_id = get_cpu_id();
    uint32_t cpu_count = get_cpu_count();
    uint32_t partition = num_elem / cpu_count;
    if (cpu_id == cpu_count - 1) {
        partition = num_elem - partition * (cpu_count - 1);
    }

    static constexpr size_t buffer_size = 1024;
    float *tcm_buffer = (float*)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

    if (tcm_buffer != nullptr) {
        auto t_in_tcm_acc = t_in.tcm_accessor();
        auto t_out_tcm_acc = t_out.tcm_accessor();

        for (size_t i = 0; i < partition; i += buffer_size) {
            size_t remaining_elem = partition - i;
            size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

            t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, partition * cpu_id + i, copy_size);
            for (size_t j = 0; j < copy_size; j++) {
                tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
            }
            t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, partition * cpu_id + i, copy_size);
        }
    }
    torch::neuron::tcm_free(tcm_buffer);
    return t_out;
}
```

```python
custom_op.load(
    name='relu',
    compute_srcs=['relu.cpp'],
    shape_srcs=['shape.cpp'],
    build_directory=os.getcwd(),
    multicore=True,
    verbose=True
)
```

## nxd-examples-migration-guide.rst

```python
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

neuron_config = DbrxNeuronConfig()  # Provide args
config = DbrxInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)
```

```python
config.save(compiled_model_path)
```

```python
config = DbrxInferenceConfig.load(compiled_model_path)
```

```python
from transformers import GenerationConfig

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Init config, model, and tokenizer.

generation_config = GenerationConfig.from_pretrained(model_path)
generation_config_kwargs = {
    "do_sample": True,
    "top_k": 1,
    "pad_token_id": generation_config.eos_token_id,
    "max_length": neuron_config.max_length,
}
generation_config.update(**generation_config_kwargs)

inputs = tokenizer(prompts, padding=True, return_tensors="pt")
generation_model = HuggingFaceGenerationAdapter(model)
outputs = generation_model.generate(
    inputs.input_ids,
    generation_config=generation_config,
    attention_mask=inputs.attention_mask,
)
```

```python
NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
```

```python
class NeuronDbrxConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = True


class DbrxInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "d_model",
            "n_heads",
            "max_seq_len",
            "emb_pdrop",
            "resid_pdrop",
            "pad_token_id",
            "vocab_size",
            "attn_config",
            "ffn_config",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronDbrxConfig
```

## layernorm_nki_kernel.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import math

@nki.jit
def nki_layernorm_kernel_v1(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm.
    Used nki.language APIs only.
  """
  output_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype,
                             buffer=nl.shared_hbm)

  # Ensure that the shapes of tensors match
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Compute mean and variance
    mean = nl.mean(input_sb, axis=1)
    # Trick to calculate var with mean: mean(x^2) - mean(x)^2
    var = nl.mean(nl.square(input_sb), axis=1) - mean * mean

    # Normalize the input by shifting with the mean 
    # and scaling with rsqrt of variance and epsilon
    shift_scale_tensor = (input_sb - mean) * nl.rsqrt(var + epsilon)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import math

@nki.jit
def nki_layernorm_kernel_v2(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm.
    Used nki.isa APIs to calculate mean/variance and perform shift/scale.
  """
  output_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype,
                             buffer=nl.shared_hbm)

  # Ensure that the shapes of tensors match
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Tile free dimension of the input tensor by nl.tile_size.bn_stats_fmax, 
    # as bn_stats has a free dimension size limit
    i_f_bn = nl.arange(nl.tile_size.bn_stats_fmax)[None, :]
    i_f_stats = nl.arange(6)[None, :]
    num_bn_stats = math.ceil(input_tensor.shape[1]/nl.tile_size.bn_stats_fmax)
    stats_results = nl.ndarray((nl.tile_size.pmax, 6*num_bn_stats), dtype=np.float32)
    for j in nl.affine_range(num_bn_stats):
      stats_results[i_p_io, j * 6 + i_f_stats] = nisa.bn_stats(
              input_sb[i_p_io, j * nl.tile_size.bn_stats_fmax + i_f_bn],
              mask=(j * nl.tile_size.bn_stats_fmax + i_f_bn < input_tensor.shape[1]),
              dtype=np.float32)
      
    # Aggregate bn_stats results to compute mean and var
    i_f_aggr = nl.arange(6*num_bn_stats)[None, :]
    mean_var = nisa.bn_aggr(stats_results[i_p_io, i_f_aggr])
    mean = mean_var[i_p_io, 0]
    var = mean_var[i_p_io, 1]

    # Get reciprocal of sqrt(var + epsilon)
    scale_var = nl.rsqrt(var + epsilon)

    # Putting the shift and scale together in one line to trigger two alu_op tensor_vector instruction
    shift_scale_tensor = nisa.tensor_scalar(data=input_sb, op0=np.subtract,
                                            operand0=mean,
                                            op1=np.multiply,
                                            operand1=scale_var)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor
```

## sd2_512_compile.py

```python
import torch
import torch.nn as nn

# Optimized attention
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

# In the original badbmm the bias is all zeros, so only apply scale
def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled
```

```python
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)
```

```python
import torch.nn as nn

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]
```

## sd2_768_compile.py

```python
import torch
import torch.nn as nn

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]
```

```python
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled
```

## writing-tests.rst

```python
import torch

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

# Module to test.
class ExampleModule(torch.nn.Module):
    def __init__(self, distributed):
        super().__init__()
        if distributed:
            self.linear = ColumnParallelLinear(
                input_size=SAMPLE_SIZE,
                output_size=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )
        else:
            self.linear = torch.nn.Linear(
                in_features=SAMPLE_SIZE,
                out_features=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )

    def forward(self, x):
        return self.linear(x)
```

```python
import torch

from neuronx_distributed_inference.utils.testing import build_function, validate_accuracy


def example_sum(tensor):
    return torch.sum(tensor)
```

```python
import torch
from functools import partial

from neuronx_distributed_inference.utils.testing import build_module


def top_k(input: torch.Tensor, k: int, dim: int):
    return torch.topk(input, k, dim)


top_k_partial = partial(top_k, 1, 0)
model = build_function(top_k_partial, example_inputs=[(torch.rand(4)),])
output = model(torch.rand(4))
```

## sdxl_base_and_refiner_1024_compile.py

```python
import torch
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention

# Optimized attention
def get_attention_scores_neuron(self, query, key, attn_mask):    
    if query.size() == key.size():
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs


def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)
```

## test_nki_nl_load_store_indirect.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def example_indirect_load_1(data_tensor, idx_tensor):
  out_tensor = nl.ndarray([64, 512], dtype=data_tensor.dtype,
                          buffer=nl.shared_hbm)
  i_p = nl.arange(64)[:, None]
  i_f = nl.arange(512)[None, :]

  idx_tile = nl.load(idx_tensor[i_p])
  data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f]) 
  nl.store(out_tensor, value=data_tile)
  return out_tensor


@nki.jit(mode="simulation")
def example_indirect_load_2(data_tensor):
  out_tensor = nl.ndarray([64, 512], dtype=data_tensor.dtype,
                          buffer=nl.shared_hbm)
  n, m = data_tensor.shape
  assert n == 128 and m == 512
  i_f = nl.arange(512)[None, :]
  
  idx_expr = 2*nl.arange(64)[:, None]
  idx_tile = nisa.iota(idx_expr, dtype=np.int32)
  data_tile = nl.load(data_tensor[idx_tile, i_f]) 
  nl.store(out_tensor, value=data_tile)
  return out_tensor


@nki.jit(mode="simulation")
def example_indirect_save_1(in_tensor, idx_tensor):
  data_tensor = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                           buffer=nl.shared_hbm)
  data_tile = nl.load(in_tensor)
  i_p = nl.arange(64)[:, None]
  i_f = nl.arange(512)[None, :]
  idx_tile = nl.load(idx_tensor[i_p])
  nl.store(data_tensor[idx_tile[i_p, 0], i_f], value=data_tile[0:64, 0:512])
  return data_tensor


@nki.jit(mode="simulation")
def example_indirect_save_2(in_tensor):
  data_tensor = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                           buffer=nl.shared_hbm)
  n, m = in_tensor.shape
  i_f = nl.arange(m)[None, :]
  data_tile = nl.load(in_tensor)
  assert n == 64 and m == 512
  idx_expr = 2*nl.arange(64)[:, None]
  idx_tile = nisa.iota(idx_expr, dtype=np.int32)
  nl.store(data_tensor[idx_tile, i_f], value=data_tile[0:64, 0:512]) 
  return data_tensor
```

## sd_4x_upscaler_benchmark.py

```python
import torch
import torch.nn as nn
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels,
        cross_attention_kwargs=None,
    ):
        out_tuple = self.unet(
            sample, timestep, encoder_hidden_states, class_labels, return_dict=False
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels,
        cross_attention_kwargs=None,
        return_dict=False,
    ):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            class_labels,
        )[0]
        return UNet2DConditionOutput(sample=sample)


class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask=None):
        return [self.neuron_text_encoder(emb)["last_hidden_state"]]
```

## test_nki_isa_select_reduce.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_select_reduce_basic(predicate_data, on_true_data):
  # Example 1: Basic usage of select_reduce
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_select_reduce_with_reduction(predicate_data, on_true_data, on_false_data):
  # Example 2: Using select_reduce with reduction
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  reduce_tensor = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
  
  predicate = nl.load(predicate_data)
  on_true = nl.load(on_true_data)
  on_false = nl.load(on_false_data)

  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  reduce_res = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=on_false,
      reduce_cmd=nisa.reduce_cmd.reset_reduce,
      reduce_res=reduce_res,
      reduce_op=nl.max
  )
  
  nl.store(result_tensor, value=dst)
  nl.store(reduce_tensor, value=reduce_res)
  return result_tensor, reduce_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_select_reduce_reverse_pred(predicate_data, on_true_data):
  # Example 3: Using select_reduce with reverse_pred option
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
      reverse_pred=True
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor
```

## fused-self-attn.rst

```python
# NKI Kernel Implementation for Fused Self Attention
# Based on sd_attention_nki_kernels.py (NKI_EXAMPLE_31)

import nki
import nki.language as nl
from nki.language import par, seq
import numpy as np

@nki.jit
def sd_attention_kernel(q, k, v):
    """
    Fused self-attention kernel for Stable Diffusion 2.1
    
    Args:
        q: Query tensor of shape (seqlen, d_head)
        k: Key tensor of shape (seqlen, d_head)
        v: Value tensor of shape (seqlen, d_head)
    
    Returns:
        output: Attention result of shape (seqlen, d_head)
    """
    seqlen = q.shape[0]
    d_head = q.shape[1]
    
    # Tile dimensions
    tile_q = 128
    tile_k = 128
    
    # Initialize output
    output = nl.zeros((seqlen, d_head), dtype=q.dtype)
    
    # Loop over tiles in Q
    for i in nl.affine_range(seqlen // tile_q):
        q_start = i * tile_q
        q_end = q_start + tile_q
        
        # Extract Q tile
        q_tile = q[q_start:q_end, :]
        
        # Initialize accumulators for this Q tile
        row_max = nl.full((tile_q,), float('-inf'), dtype=nl.float32)
        row_sum = nl.zeros((tile_q,), dtype=nl.float32)
        result_accum = nl.zeros((tile_q, d_head), dtype=nl.float32)
        
        # Loop over tiles in K
        for j in nl.affine_range(seqlen // tile_k):
            k_start = j * tile_k
            k_end = k_start + tile_k
            
            # Extract K and V tiles
            k_tile = k[k_start:k_end, :]
            v_tile = v[k_start:k_end, :]
            
            # Compute S = Q * K.T
            s_tile = nl.matmul(q_tile, nl.transpose(k_tile))
            
            # Compute partial row-wise maximum
            partial_max = nl.max(s_tile, axis=1)
            
            # Update global row-wise maximum
            new_max = nl.maximum(row_max, partial_max)
            
            # Compute exponential with numerical stability
            exp_input = s_tile - nl.reshape(new_max, (tile_q, 1))
            exp_tile = nl.exp(exp_input)
            
            # Update row sum with correction factor
            correction = nl.exp(nl.reshape(row_max - new_max, (tile_q, 1)))
            row_sum = row_sum * nl.reshape(correction, (tile_q,)) + nl.sum(exp_tile, axis=1)
            
            # Accumulate result: S * V
            result_accum = result_accum * nl.reshape(correction, (tile_q, 1)) + nl.matmul(exp_tile, v_tile)
            
            # Update row max
            row_max = new_max
        
        # Normalize by row sum (delayed division)
        output[q_start:q_end, :] = result_accum / nl.reshape(row_sum, (tile_q, 1))
    
    return output
```

```python
# PyTorch Integration and Testing
# Based on sd_attention_torch.py (NKI_EXAMPLE_32)

import torch
import torch.nn as nn

def reference_attention(q, k, v):
    """
    Reference PyTorch implementation of self-attention
    
    Args:
        q: Query tensor of shape (seqlen, d_head)
        k: Key tensor of shape (seqlen, d_head)
        v: Value tensor of shape (seqlen, d_head)
    
    Returns:
        output: Attention result of shape (seqlen, d_head)
    """
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output

def test_attention_correctness(seqlen=4096, d_head=64):
    """
    Test NKI kernel against PyTorch reference
    
    Args:
        seqlen: Sequence length
        d_head: Head dimension
    """
    # Create test inputs
    q = torch.randn(seqlen, d_head, dtype=torch.bfloat16)
    k = torch.randn(seqlen, d_head, dtype=torch.bfloat16)
    v = torch.randn(seqlen, d_head, dtype=torch.bfloat16)
    
    # Compute reference output
    ref_output = reference_attention(q.float(), k.float(), v.float())
    
    # Compute NKI kernel output
    nki_output = sd_attention_kernel(q, k, v)
    
    # Compare results
    diff = torch.abs(ref_output - nki_output.float())
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    return max_diff, mean_diff
```

## optimize_for_inference.py

```python
import re
import copy
import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.tools.graph_transforms import TransformGraph

def clear_input(node):
  for i in range(len(node.input)):
    node.input.pop()

def replace_name(node, name):
  node.name = name
     
def replace_input(node, input_name, new_name):
  temp = []
  for i in node.input:
    temp.extend([new_name if i == input_name else i])
  clear_input(node)
  for i in temp:
    node.input.extend([i])

def swap_names(node1, node2):
  temp = node2.name
  node2.name = node1.name
  node1.name = temp

def get_const_node(const_node_name, const_by_name):
  name = re.sub("/read$", "", const_node_name)
  return const_by_name[name]

def get_const_ndarray(const_node_name, const_by_name):
  name = re.sub("/read$", "", const_node_name)
  node = const_by_name[name]
  return tf.make_ndarray(node.attr.get("value").tensor)

def adjust_bias_values(bias_node, fbn_node, const_by_name):
  bias_val = get_const_ndarray(bias_node.input[1], const_by_name)  
  gamma_val = get_const_ndarray(fbn_node.input[1], const_by_name)  
  mean_val = get_const_ndarray(fbn_node.input[3], const_by_name)  
  variance_val = get_const_ndarray(fbn_node.input[4], const_by_name) 
  new_bias = bias_val * gamma_val / np.sqrt(variance_val)
  new_tensor = tensor_util.make_tensor_proto(new_bias, new_bias.dtype, new_bias.shape)
  bias_const_node = get_const_node(bias_node.input[1], const_by_name)
  bias_const_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=new_tensor))

def MoveBiasAddAfterFusedBatchNorm(graphdef):
  sess = tf.compat.v1.Session(graph=tf.import_graph_def(graphdef))
  output_graph_def = tf.compat.v1.GraphDef()
  node_by_name = {}
  const_by_name = {}
  for node in graphdef.node:
    if node.op == "FusedBatchNormV3":
      node.op = "FusedBatchNorm"
      del(node.attr["U"])
    copied_node = node_def_pb2.NodeDef()
    copied_node.CopyFrom(node)
    node_by_name[node.name] = copied_node
    skip_add_node = False
    if node.op == "Const":
      const_by_name[node.name] = copied_node  
    elif node.op.startswith("FusedBatchNorm"):
      inputs = node.input
      for i in inputs:
        input_node = node_by_name[i]
        if input_node.op == "BiasAdd":
          output_graph_def.node.remove(input_node)
          input_node_input0 = input_node.input[0]
          adjust_bias_values(input_node, node, const_by_name)
          swap_names(copied_node, input_node)
          replace_input(copied_node, i, input_node_input0)
          replace_input(input_node, input_node_input0, copied_node.name)
          output_graph_def.node.extend([copied_node])
          output_graph_def.node.extend([input_node])
          skip_add_node = True
    if not skip_add_node:
      output_graph_def.node.extend([copied_node])
  return output_graph_def

def FoldFusedBatchNorm(graph_def):
  transformed_graph_def = TransformGraph (
         graph_def,
         ['input_1'],
         ['probs/Softmax'],
         [
            'add_default_attributes',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms',
            'strip_unused_nodes',
            'sort_by_execution_order',
         ])
  return transformed_graph_def

def load_graph(model_file):
  graph_def = tf.compat.v1.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def
```

## bert_benchmark_utils.py

```python
import torch
import torch.neuron
import csv
import numpy as np
from collections import Counter
import math

class BertTestDataset(torch.utils.data.Dataset):
    """Bert test dataset."""

    def __init__(self, tsv_file, tokenizer, max_length=128, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer (callable = hugging face tokenizer):  Takes a string and encodes to standard input tensor set
            max_length (int): Maximum length that all input tensors will be padded to
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(tsv_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            self.lines = list(reader)

        self.lines.pop(0)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s1_raw = self.lines[idx][3]
        if isinstance(s1_raw, bytes):
            s1_raw = s1_raw.decode("utf-8", "ignore")
        s2_raw = self.lines[idx][4]
        if isinstance(s2_raw, bytes):
            s2_raw = s2_raw.decode("utf-8", "ignore")

        quality = self.lines[idx][0]

        encoded = self.tokenizer.encode_plus(s1_raw, s2_raw, add_special_tokens=True,
                                             return_tensors='pt', max_length=self.max_length, 
                                             padding='max_length', truncation=True)

        sample = {'encoded': encoded, 'quality': quality}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

## test_nki_isa_nc_matmul.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_nc_matmul(a_tensor, b_tensor, d_tensor, e_tensor, g_tensor, h_tensor):
  c_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  f_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  i_tensor = nl.ndarray([16, 64, 512], dtype=nl.float32, buffer=nl.shared_hbm)

  # Example 1: Basic matrix multiplication
  a_mgrid = nl.mgrid[0:128, 0:128]
  b_mgrid = nl.mgrid[0:128, 0:512]
  c_mgrid = nl.mgrid[0:128, 0:512]

  a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
  b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])

  c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])

  nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)

  # Example 2: Matrix multiplication with psum accumulation
  d_mgrid = nl.mgrid[0:128, 0:128]
  e_mgrid = nl.mgrid[0:128, 0:512]
  f_mgrid = nl.mgrid[0:128, 0:512]

  f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

  for i_contract in nl.affine_range(2):
    d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
    e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
    f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
    
  nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)

  # Example 3: Batched matrix multiplication with tile positioning
  g_mgrid = nl.mgrid[0:64, 0:64]
  h_mgrid = nl.mgrid[0:64, 0:512]
  i_mgrid = nl.mgrid[0:64, 0:512]

  for i in nl.affine_range(4):
    for j in nl.affine_range(4):
      g = nl.load(g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
      h = nl.load(h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
      i_psum = nisa.nc_matmul(g, h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
      nl.store(i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], i_psum)

  return c_tensor, f_tensor, i_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation", platform_target='trn2')
def nki_nc_matmul_double_row_gen3(a_input, b_input):
  NUM_PARTITIONS_A, TWO_A, FREE_A = a_input.shape
  NUM_PARTITIONS_B, TWO_B, FREE_B = b_input.shape

  c_output = nl.ndarray([FREE_A, FREE_B], dtype=nl.float32, buffer=nl.shared_hbm)

  assert NUM_PARTITIONS_A == NUM_PARTITIONS_B and TWO_A == 2 and TWO_B == 2

  a_tile = nl.ndarray(
    (NUM_PARTITIONS_A, TWO_A, max(FREE_A, 16)), dtype=nl.float8_e5m2, buffer=nl.sbuf
  )
  a_mgrid = nl.mgrid[0:NUM_PARTITIONS_A, 0:TWO_A, 0:FREE_A]
  a_tile[a_mgrid.p, a_mgrid.x, a_mgrid.y] = nl.load(a_input.view(nl.float8_e5m2))
  b_tile = nl.load(b_input.view(nl.float8_e5m2))
  c_tile = nisa.nc_matmul(
    a_tile[a_mgrid.p, a_mgrid.x, a_mgrid.y], b_tile, perf_mode="double_row_gen3"
  )
  nl.store(c_output, value=c_tile)
  return c_output
```

## test_nki_nl_add.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit(mode="simulation")
def add_tensors(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:512])
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_tensor_scalar(a_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = 2.2
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_free_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:1])
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_par_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:512])
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_both_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:1])
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_each_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray([128, 512], dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:1])
  b = nl.load(b_tensor[0:1, 0:512])
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor
```

## pb2sm_compile.py

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
import tensorflow.neuron as tfn

def pb_to_saved_model(pb_path, input_names, output_names, model_dir):
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(open(pb_path, 'rb').read())
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        inputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in input_names.items()}
        outputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in output_names.items()}
        tf.saved_model.simple_save(sess, model_dir, inputs, outputs)
```

```python
import tensorflow.neuron as tfn

rslts = tfn.saved_model.compile(saved_model_dir, compiled_saved_model_dir,
               model_feed_dict={'input_1:0' : img_arr},
               compiler_workdir=workdir,
               dynamic_batch_size=True,
               compiler_args=compiler_args)

perc_on_inf = rslts['OnNeuronRatio'] * 100
```

## how-to-convolution-in-unet.rst

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki._private_kernels.conv import conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh

@nki.jit
def conv_wrap(img_ref, filter_ref, out_shape):
    out_arr = nl.ndarray(shape=out_shape, dtype=img_ref.dtype, buffer=nl.hbm)
    conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(img_ref, filter_ref, out_arr, **{
        'input': img_ref.shape,
        'filter': filter_ref.shape, 
        'output': out_shape,
        'in_perm': [0, 1, 2, 3],
        'kern_perm': [0, 1, 2, 3],
        'out_perm': [0, 1, 2, 3],
        'stride': (1, 1),
        'padding': ((1, 1), (1, 1))})
    return out_arr
```

```python
class BwdConv2dWithKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__()
        assert padding == 1
        assert bias == False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0.0, mode='fan_in', nonlinearity='leaky_relu')
```

```python
class DoubleConvWithKernel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            BwdConv2dWithKernel(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            BwdConv2dWithKernel(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

## t5_models.py

```python
import torch
import neuronx_distributed

from functools import partial
from transformers import T5Tokenizer, T5ForConditionalGeneration

from wrapper import EncoderWrapper, DecoderWrapper
from t5_model_layers import load_pretrained_with_parallel_attn


def get_wrapped_encoder(max_length, num_beams, tp_degree, model_name):
    
    model = load_pretrained_with_parallel_attn(model_name)

    encoder = EncoderWrapper(model.encoder, model.decoder, model.config, num_beams, max_length, "xla", num_beams, tp_degree=tp_degree)
    encoder.eval()
    
    # We are alaising the cache, so that way we keep the cache always on device.
    aliases = {}
    for i in range(len(encoder.past_key_values_sa)):
        aliases[encoder.past_key_values_sa[i]] = i
    
    for i in range(len(encoder.past_key_values_ca)):
        aliases[encoder.past_key_values_ca[i]] = len(encoder.past_key_values_sa) + i

    return encoder, aliases


def get_wrapped_decoder(max_length, num_beams, tp_degree, model_name):
    
    model = load_pretrained_with_parallel_attn(model_name)

    decoder = DecoderWrapper(decoder=model.decoder,
                             lm_head=model.lm_head,
                             model_config=model.config,
                             num_beams=num_beams,
                             max_length=max_length,
                             device="xla",
                             tp_degree=tp_degree)
    
    decoder.eval()
    num_outputs_from_trace = 3 if num_beams > 1 else 1
    aliases = {}
    for i in range(len(decoder.past_key_values_sa)):
        aliases[decoder.past_key_values_sa[i]] = i + num_outputs_from_trace
    for i in range(len(decoder.past_key_values_ca)):
        aliases[decoder.past_key_values_ca[i]] = len(decoder.past_key_values_sa) + i + num_outputs_from_trace

    return decoder, aliases
```

## test_nki_isa_tensor_copy_dynamic.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def example_tensor_copy_dynamic_src_0(src_tensor, offsets):
  out_tensor = nl.ndarray([128, 64], dtype=src_tensor.dtype,
                          buffer=nl.shared_hbm)
  
  # Load src_tensor and offsets into SBUF
  src_tensor_sbuf: nt.tensor[128, 512] = nl.load(src_tensor)
  offsets_sbuf: nt.tensor[1, 64] = nl.load(offsets)

  # Copy into output tensor in SBUF
  out_sbuf: nt.tensor[128, 64] = nl.ndarray([128, 64], dtype=src_tensor.dtype,
                                            buffer=nl.sbuf)

  # Static indices to access a tile of shape [128, 1];
  # Add dynamic offsets to iy for tensor_copy_dynamic_src
  ix, iy = nl.mgrid[0:128, 0:1]

  for idx in nl.affine_range(offsets_sbuf.shape[1]):
    out_sbuf[ix, idx] = nisa.tensor_copy_dynamic_src(
        src_tensor_sbuf[ix, offsets_sbuf[0, idx] + iy])

  nl.store(out_tensor, value=out_sbuf)
  return out_tensor


@nki.jit(mode="simulation")
def example_tensor_copy_dynamic_src_1(src_tensor, offsets):
  out_tensor = nl.ndarray([128, 8, 4], dtype=src_tensor.dtype,
                          buffer=nl.shared_hbm)
  
  # Load src_tensor and offsets into SBUF
  src_tensor_sbuf: nt.tensor[128, 512, 4] = nl.load(src_tensor)
  offsets_sbuf: nt.tensor[1, 8] = nl.load(offsets)

  # Copy into output tensor in SBUF
  out_sbuf: nt.tensor[128, 8, 4] = nl.ndarray([128, 8, 4], dtype=src_tensor.dtype,
                                              buffer=nl.sbuf)

  # Static indices to access a tile of shape [128, 1, 4];
  # Use dynamic offsets directly to index the second axis for tensor_copy_dynamic_src
  ix, _, iz = nl.mgrid[0:128, 0:1, 0:4]

  for idx in nl.affine_range(offsets.shape[1]):
    out_sbuf[ix, idx, iz] = nisa.tensor_copy_dynamic_src(
        src_tensor_sbuf[ix, offsets_sbuf[0, idx], iz])

  nl.store(out_tensor, value=out_sbuf)
  return out_tensor
```

## mamba_torch.py

```python
import torch

def associative_scan(deltaA, deltaB_u):
    """
    Args:
        deltaA: [batch_size, channels, state_size, seq_len]
        deltaB_u: [batch_size, channels, state_size, seq_len]

    Mamba uses an associative scan operator to aggregate information across
    time sequentially (sequence length, e.g. sequence of tokens),
    from the past to the present.
    """
    batch_size, channels, state_size, seq_len = deltaA.shape
    out = torch.empty(batch_size, channels, state_size, seq_len,
                        device=deltaA.device, dtype=deltaA.dtype)
    for i in range(seq_len):
        prev_state = out[..., i - 1] if i > 0 else 0
        out[..., i] = deltaA[..., i] * prev_state + deltaB_u[..., i]
    return out


def mamba_layer(delta, A, B, u, C):
    """
    Args:
        delta: [batch, channels, seq_len]
        u: [batch, channels, seq_len]
        A: [channels, state_size]
        B: [batch, state_size, seq_len]
        C: [batch, state_size, seq_len]
    """
    # expand the tensors so they all have the same dimensions and compute elementwise products (with broadcast)
    # deltaA and deltaB_u have shape [batch_size, channels, state_size, seq_len]
    deltaA = torch.exp(delta[:, :, None, :] * A[None, :, :, None])
    deltaB_u = delta[:, :, None, :] * B[:, None, :, :] * u[:, :, None, :]
    scan_res = associative_scan(deltaA, deltaB_u)
    # y sums over the `state_size` axis and has shape [batch_size, channels, seq_len]
    mamba_out = (C[:, None, :, :] * scan_res).sum(dim=-2)
    return mamba_out
```

## test_nki_isa_nc_stream_shuffle.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import numpy as np


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle(in_tensor):
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[32, 128] = nl.load(in_tensor)
  a_mgrid = nl.mgrid[0:32, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=a)
  return out_tensor


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle_broadcast_partition(in_tensor):
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[1, 128] = nl.load(in_tensor)
  b = nl.ndarray(shape=(32, 128), dtype=np.float32)
  dst_mgrid = nl.mgrid[0:32, 0:128]
  src_mgrid = nl.mgrid[0:1, 0:128]
  shuffle_mask = [0] * 32
  nisa.nc_stream_shuffle(src=a[0, src_mgrid.x], dst=b[dst_mgrid.p, dst_mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=b)
  return out_tensor


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle_broadcast_mask(in_tensor):
  out_tensor = nl.ndarray(shape=(128, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[128, 128] = nl.load(in_tensor)
  b = nl.ndarray(shape=(128, 128), dtype=np.float32)
  mgrid = nl.mgrid[0:128, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[mgrid.p, mgrid.x], dst=b[mgrid.p, mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=b)
  return out_tensor
```

## average_pool2d.rst

```python
# NKI Kernel Implementation
# From: average_pool2d_nki_kernels.py (NKI_EXAMPLE_37)

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki import nki_jit
import numpy as np

@nki_jit
def tensor_avgpool_kernel(in_tensor, pool_size):
    """
    2D Average Pooling kernel using NKI.
    
    Args:
        in_tensor: Input tensor of shape [C, H, W]
        pool_size: Size of the pooling window (pool_size x pool_size)
    
    Returns:
        Output tensor of shape [C, H//pool_size, W//pool_size]
    """
    # Implementation details would include:
    # - Mapping C axis to P dimension (parallel)
    # - Mapping H/W axes to F dimension (contraction)
    # - 4D memory access pattern in F dimension
    # - Reduction along two axes
    pass
```

```python
# PyTorch Kernel Launcher
# From: average_pool2d_torch.py (NKI_EXAMPLE_38)

import torch
from average_pool2d_nki_kernels import tensor_avgpool_kernel

def run_pytorch_avgpool(in_tensor, pool_size):
    """
    Launch AveragePool2D kernel from PyTorch.
    
    Args:
        in_tensor: PyTorch tensor of shape [C, H, W]
        pool_size: Size of the pooling window
    
    Returns:
        Output tensor after average pooling
    """
    output = tensor_avgpool_kernel(in_tensor, pool_size=pool_size)
    return output
```

```python
# JAX Kernel Launcher
# From: average_pool2d_jax.py (NKI_EXAMPLE_39)

import jax
import jax.numpy as jnp
from average_pool2d_nki_kernels import tensor_avgpool_kernel

def tensor_avgpool_kernel_jax(in_array, pool_size):
    """
    JAX wrapper for AveragePool2D kernel with compile-time constant pool_size.
    
    Args:
        in_array: JAX array of shape [C, H, W]
        pool_size: Size of the pooling window (compile-time constant)
    
    Returns:
        Output array after average pooling
    """
    return tensor_avgpool_kernel(in_array, pool_size=pool_size)
```

```python
# JAX Reference Implementation
# From: average_pool2d_jax.py (NKI_EXAMPLE_40)

import jax.numpy as jnp

def reference_avgpool2d(x, pool_size):
    """
    Reference JAX implementation of 2D Average Pooling.
    
    Args:
        x: Input array of shape [C, H, W]
        pool_size: Size of the pooling window
    
    Returns:
        Output array of shape [C, H//pool_size, W//pool_size]
    """
    C, H, W = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    
    x_reshaped = x.reshape(C, H_out, pool_size, W_out, pool_size)
    x_transposed = jnp.transpose(x_reshaped, (0, 1, 3, 2, 4))
    output = x_transposed.reshape(C, H_out, W_out, pool_size * pool_size)
    
    return jnp.mean(output, axis=3)
```

```python
# JAX Kernel Execution
# From: average_pool2d_jax.py (NKI_EXAMPLE_41)

import jax.numpy as jnp
from average_pool2d_nki_kernels import tensor_avgpool_kernel

def run_jax_avgpool(in_array, pool_size):
    """
    Execute AveragePool2D kernel from JAX.
    
    Args:
        in_array: JAX array of shape [C, H, W]
        pool_size: Size of the pooling window
    
    Returns:
        Output array after average pooling
    """
    output = tensor_avgpool_kernel(in_array, pool_size=pool_size)
    return output
```

## transpose2d.rst

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def tensor_transpose2D_kernel_(input_tensor, output_tensor, shape2D):
    """
    Transpose F1 and F2 axes of a 3D tensor [P, F1, F2].
    P axis is mapped to partitions, F1 and F2 are flattened in each partition.
    """
    # This is a placeholder showing the function signature and decorator usage
    # The actual implementation uses nki.language.copy with indexing manipulation
    # to transpose between two free-dimension axes
    pass
```

**Note:** The source document (transpose2d.rst) is a tutorial documentation file that references external code examples via markers (`.. nki_example::`) rather than containing the full source code inline. The actual kernel implementation code is located in `transpose2d_nki_kernels.py`, which is not provided in this source file. 

To extract the complete working code examples, you would need to access:
- `../examples/transpose2d/transpose2d_nki_kernels.py` (marked as NKI_EXAMPLE_33)
- `../examples/transpose2d/transpose2d_torch.py` (marked as NKI_EXAMPLE_34)
- `../examples/transpose2d/transpose2d_jax.py` (marked as NKI_EXAMPLE_36)

## test_nki_isa_iota.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def nki_iota():
  # Example 1: Generate tile a of 512 constant values in SBUF partition 0
  # that start at 0 and increment by 1:
  # a = [0, 1, ..., 511]
  expr_a = nl.arange(0, 512)[None, :]
  a: tensor[1, 512] = nisa.iota(expr_a, dtype=nl.int32)

  # Example 2: Generate tile b of 128 constant values across SBUF partitions
  # that start at 0 and increment by 1, with one value per partition:
  # b = [[0],
  #      [1],
  #      ...,
  #      [127]]
  expr_b = nl.arange(0, 128)[:, None]
  b: tensor[128, 1] = nisa.iota(expr_b, dtype=nl.int32)
  
  # Example 3: Generate tile c of 512 constant values in SBUF partition 0
  # that start at 0 and decrement by 1:
  # c = [0, -1, ..., -511]
  expr_c = expr_a * -1
  c: tensor[1, 512] = nisa.iota(expr_c, dtype=nl.int32)

  # Example 4: Generate tile d of 128 constant values across SBUF
  # partitions that start at 5 and increment by 2
  # d = [[5],
  #      [7],
  #      ...,
  #      [259]]
  expr_d = 5 + expr_b * 2
  d: tensor[128, 1] = nisa.iota(expr_d, dtype=nl.int32)

  # Example 5: Generate tile e of shape [128, 512] by
  # broadcast-add expr_a and expr_b
  # e = [[0, 1, ..., 511],
  #      [1, 2, ..., 512],
  #      ...
  #      [127, 2, ..., 638]]
  e: tensor[128, 512] = nisa.iota(expr_a + expr_b, dtype=nl.int32)

  a_tensor = nl.ndarray([1, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  b_tensor = nl.ndarray([128, 1], dtype=nl.float32, buffer=nl.shared_hbm)
  c_tensor = nl.ndarray([1, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  d_tensor = nl.ndarray([128, 1], dtype=nl.float32, buffer=nl.shared_hbm)
  e_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(a_tensor[0, expr_a], a)
  nl.store(b_tensor[expr_b, 0], b)
  nl.store(c_tensor[0, expr_a], c)  
  nl.store(d_tensor[expr_b, 0], d)
  nl.store(e_tensor[expr_b, expr_a], e)
  return a_tensor, b_tensor, c_tensor, d_tensor, e_tensor
```

## fp32tofp16.py

```python
import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import gfile

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util

def ConvertFP32ToOther(graphdef):
  """Converts an FP32 network by casting all constants (weights) to a lower
     precision floating point type (FP16) and updating the dtypes
     everywhere."""
  cast_type = "float16"
  sess = tf.Session(graph=tf.import_graph_def(graphdef))
  output_graph_def = graph_pb2.GraphDef()
  dummy_tensor = sess.run(tf.constant([0.1]))
  dummy_tensor_proto = tensor_util.make_tensor_proto(dummy_tensor, \
      dtype=cast_type, shape=dummy_tensor.shape)
  dummy_tensor32 = sess.run(tf.constant([0.1]))
  dummy_tensor_proto32 = tensor_util.make_tensor_proto(dummy_tensor, \
      dtype=tf.float32, shape=dummy_tensor.shape)
  dt_float_type_attr = attr_value_pb2.AttrValue(type=dummy_tensor_proto32.dtype)
  dt_half_type_attr = attr_value_pb2.AttrValue(type=dummy_tensor_proto.dtype)
  for node in graphdef.node:
    output_node = node_def_pb2.NodeDef()
    output_node.CopyFrom(node)
    if (node.op == "Const"):
      if (node.attr["dtype"] == dt_float_type_attr):
        a = tensor_util.MakeNdarray(node.attr["value"].tensor)
        a = tf.cast(a, cast_type)
        a = sess.run(a)
        output_node.attr["dtype"].CopyFrom(dt_half_type_attr)
        output_node.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(
              tensor=tensor_util.make_tensor_proto(a,\
                dtype=cast_type, shape=a.shape)))
    else:
      if ("T" in node.attr.keys()):
        if (output_node.attr["T"] == dt_float_type_attr):
          output_node.attr["T"].CopyFrom(dt_half_type_attr)
      if ("Tparams" in node.attr.keys()):
        if (output_node.attr["Tparams"] == dt_float_type_attr):
          output_node.attr["Tparams"].CopyFrom(dt_half_type_attr)
      if ("dtype" in node.attr.keys()):
        if (node.attr["dtype"] == dt_float_type_attr):
          output_node.attr["dtype"].CopyFrom(dt_half_type_attr)
      if ("SrcT" in node.attr.keys()):
        if (node.attr["SrcT"] == dt_float_type_attr):
          output_node.attr["SrcT"].CopyFrom(dt_half_type_attr)
      if ("DstT" in node.attr.keys()):
        if (node.attr["DstT"] == dt_float_type_attr):
          output_node.attr["DstT"].CopyFrom(dt_half_type_attr)
    output_graph_def.node.extend([output_node])
  return output_graph_def

def load_graph(model_file):
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())

  return graph_def
```

## test_nki_isa_tensor_scalar.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation")
def nki_tensor_scalar(a_tensor, c_tensor, e_tensor, f_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  d_tensor = nl.ndarray(c_tensor.shape, dtype=c_tensor.dtype,
                        buffer=nl.shared_hbm)
  g_tensor = nl.ndarray(e_tensor.shape, dtype=e_tensor.dtype,
                        buffer=nl.shared_hbm)
  
  # Example 1: subtract 1.0 from all elements of tile a
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)
  nl.store(b_tensor[i_p, i_f], b)

  # Example 2: broadcast 1.0 and subtract with tile c
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  c = nl.load(c_tensor[i_p, i_f])
  d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)
  nl.store(d_tensor[i_p, i_f], d)

  # Example 3: broadcast multiply tile e with vector f, then add scalar 2.5
  i_p_ef = nl.arange(64)[:, None]
  i_f_e = nl.arange(1024)[None, :]
  i_f_f = nl.arange(1)[None, :]
  e = nl.load(e_tensor[i_p_ef, i_f_e])
  f = nl.load(f_tensor[i_p_ef, i_f_f])
  g = nisa.tensor_scalar(e[i_p_ef, i_f_e], op0=np.multiply, operand0=f[i_p_ef, i_f_f], op1=np.add, operand1=2.5)
  nl.store(g_tensor[i_p_ef, i_f_e], g)
  
  return b_tensor, d_tensor, g_tensor
```

## test_psum_modulo_alloc.py

```python
from typing import Optional, Tuple
from functools import reduce
from operator import mul

def num_elems(shape):
  return reduce(mul, shape, 1)

def linearize(shape, indices):
  return sum(i * num_elems(shape[dim+1:]) for dim, i in enumerate(indices))

def modulo_allocate_func(base, allocate_shape, scale):
  def func(indices):
    if not allocate_shape:
      # default shape is always (1, 1, ...)
      allocate_shape_ = (1, ) * len(indices)
    else:
      allocate_shape_ = allocate_shape
    mod_idx = tuple(i % s for i, s in zip(indices, allocate_shape_))
    return linearize(shape=allocate_shape_, indices=mod_idx) * scale + base
  return func

def mod_alloc(base_addr: int, *, 
               base_bank: Optional[int] = 0,
               num_bank_tiles: Optional[Tuple[int]] = (),
               base_partition: Optional[int] = 0,
               num_par_tiles: Optional[Tuple[int]] = (),
               num_free_tiles: Optional[Tuple[int]] = ()):
  def psum_modulo_alloc_func(idx, pdim_size, fdim_size):
    # partial bank allocation is not allowed
    return (modulo_allocate_func(base_bank, num_bank_tiles, 1)(idx),
          modulo_allocate_func(base_partition, num_par_tiles, pdim_size)(idx),
          modulo_allocate_func(base_addr, num_free_tiles, fdim_size)(idx))
  return psum_modulo_alloc_func
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
import numpy as np

nki_jit = nki.trace

@nki_jit
def allocated_loop_transpose(a_ptr, tp_ptr):
  
  N, M = a_ptr.shape

  _M, _N = tp_ptr.shape
  assert _N == N and _M == M

  N0, N1 = N // 128, 128
  M0, M1 = M // 128, 128

  ix0 = nl.arange(0, M1)[:, None]
  iy0 = nl.arange(0, N1)[None, :]

  identity = nl.shared_constant(np.identity(n=128, dtype=np.int8), dtype=nl.bfloat16)

  for n0 in nl.affine_range(N0):
    for m0 in nl.affine_range(M0):
      ix0 = nl.arange(0, 128)[:, None]
      iy0 = nl.arange(0, 128)[None, :]
      a_local = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype, 
                           buffer=ncc.sbuf.mod_alloc(base_addr=1024))
      a_local[ix0, iy0] = nl.load(a_ptr[n0 * N1 + ix0, m0 * M1 + iy0])

      identity_load = nl.ndarray((nl.par_dim(128), 128), dtype=a_ptr.dtype, buffer=ncc.sbuf.mod_alloc(base_addr=0))
      identity_load[ix0, iy0] = nl.load(identity, dtype=a_ptr.dtype)

      a_local_transpose = nl.ndarray((nl.par_dim(M1), N1), dtype=a_ptr.dtype,
                                     buffer=ncc.psum.alloc(mod_alloc(base_addr=0)))
      a_local_transpose[ix0, iy0] = nisa.nc_matmul(a_local[ix0, iy0], identity_load)

      a_t_sbuf = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype,
                                     buffer=ncc.sbuf.mod_alloc(base_addr=2048))
      a_t_sbuf[ix0, iy0] = nl.copy(a_local_transpose[ix0, iy0])

      nl.store(tp_ptr[m0 * 128 + ix0, n0 * 128 + iy0], value=a_t_sbuf[ix0, iy0])
```

## test_nki_isa_bn_stats.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def nki_bn_stats_bn_aggr_1(a_tensor):
  mean_a_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  var_a_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a: tensor[128, 128] = nl.load(a_tensor)
  stats_a: tensor[128, 6] = nisa.bn_stats(a)
  mean_var_a: tensor[128, 2] = nisa.bn_aggr(stats_a)

  mean_a = mean_var_a[:, 0]
  var_a = mean_var_a[:, 1]
  nl.store(mean_a_tensor, mean_a)
  nl.store(var_a_tensor, var_a)

  return mean_a_tensor, var_a_tensor


@nki.jit(mode="simulation")
def nki_bn_stats_bn_aggr_2(b_tensor):
  mean_b_tensor = nl.ndarray([b_tensor.shape[0], 1], dtype=b_tensor.dtype, buffer=nl.shared_hbm)
  var_b_tensor = nl.ndarray([b_tensor.shape[0], 1], dtype=b_tensor.dtype, buffer=nl.shared_hbm)

  b: tensor[128, 1024] = nl.load(b_tensor)

  stats_b = nl.ndarray((128, 6 * 2), dtype=nl.float32)
  bn_tile = nl.tile_size.bn_stats_fmax
  ix, iy = nl.mgrid[0:128, 0:bn_tile]
  iz, iw = nl.mgrid[0:128, 0:6]

  for i in range(1024 // bn_tile):
    stats_b[iz, i * 6 + iw] = nisa.bn_stats(b[ix, i * bn_tile + iy], dtype=nl.float32)

  mean_var_b = nisa.bn_aggr(stats_b)

  mean_b = mean_var_b[:, 0]
  var_b = mean_var_b[:, 1]

  nl.store(mean_b_tensor, mean_b)
  nl.store(var_b_tensor, var_b)

  return mean_b_tensor, var_b_tensor
```

## test_attention.py

```python
"""
Copyright (c) 2025, Amazon.com. All Rights Reserved
"""
from attention_kernels import *
import neuronxcc.nki as nki
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

def numpy_attention(q, k, v):
    """NumPy reference implementation"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    
    # Not doing Q @ K.T due to NKI layout constraints which require
    # Q transposed for matmul since contraction dimension 
    # has to be mapped to the partition dimension
    # Shape: (seqlen_q, seqlen_kv)
    qk = np.matmul(q.T, k)  
    
    # Softmax
    # Shape: (seqlen_q, 1)
    row_max = np.max(qk, axis=1, keepdims=True) 
    
    # Shape: (seqlen_q, seqlen_kv)
    norm_row = qk - row_max
    exp_row = np.exp(norm_row)
    
    # Shape: (seqlen_q, 1)
    sum_row = np.sum(exp_row, axis=1, keepdims=True)  
    
    # Shape: (seqlen_q, seqlen_kv)
    scores = exp_row / sum_row  
    
    # V transpose
    v_t = v.T  # Shape: (seqlen_kv, d_head)
    
    # scores @ V
    attn_out = np.matmul(scores, v_t)  # Shape: (seqlen_q, d_head)
    
    return attn_out
```

## test_nki_isa_local_gather.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def nki_local_gather(src_buffer, index, num_elem_per_idx, num_valid_indices, output_shape):
  output = nl.ndarray(output_shape, dtype=src_buffer.dtype,
                      buffer=nl.shared_hbm)

  src_buffer_tile: tensor[128, 512, 4] = nl.load(src_buffer)
  index_tile: tensor[128, 4] = nl.load(index)
  output_tile: tensor[128, 4, 16, 4] = nisa.local_gather(
    src_buffer_tile, index_tile, num_elem_per_idx, num_valid_indices)

  nl.store(output, output_tile)

  return output
```

## test_sbuf_modulo_alloc.py

```python
from typing import Optional, Tuple
from functools import reduce
from operator import mul

def num_elms(shape):
  return reduce(mul, shape, 1)

def linearize(shape, indices):
  return sum(i * num_elms(shape[dim+1:]) for dim, i in enumerate(indices))

def modulo_allocate_func(base, allocate_shape, scale):
  def func(indices):
    if not allocate_shape:
      # default shape is always (1, 1, ...)
      allocate_shape_ = (1, ) * len(indices)
    else:
      allocate_shape_ = allocate_shape
    mod_idx = tuple(i % s for i, s in zip(indices, allocate_shape_))
    return linearize(shape=allocate_shape_, indices=mod_idx) * scale + base
  return func

def mod_alloc(base_addr: int, *, 
               base_partition: Optional[int] = 0,
               num_par_tiles: Optional[Tuple[int, ...]] = (),
               num_free_tiles: Optional[Tuple[int, ...]] = ()):
  def sbuf_modulo_alloc_func(idx, pdim_size, fdim_size):
    return (modulo_allocate_func(base_partition, num_par_tiles, pdim_size)(idx),
          modulo_allocate_func(base_addr, num_free_tiles, fdim_size)(idx))
  return sbuf_modulo_alloc_func
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
import numpy as np

nki_jit = nki.trace

@nki_jit
def allocated_loop_transpose(a_ptr, tp_ptr):
  
  N, M = a_ptr.shape

  _M, _N = tp_ptr.shape
  assert _N == N and _M == M

  N0, N1 = N // 128, 128
  M0, M1 = M // 128, 128

  ix0 = nl.arange(0, M1)[:, None]
  iy0 = nl.arange(0, N1)[None, :]

  identity = nl.shared_constant(np.identity(n=128, dtype=np.int8), dtype=nl.bfloat16)

  for n0 in nl.affine_range(N0):
    for m0 in nl.affine_range(M0):
      ix0 = nl.arange(0, 128)[:, None]
      iy0 = nl.arange(0, 128)[None, :]
      a_local = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype, 
                           buffer=ncc.sbuf.alloc(mod_alloc(base_addr=1024)))
      a_local[ix0, iy0] = nl.load(a_ptr[n0 * N1 + ix0, m0 * M1 + iy0])

      identity_load = nl.ndarray((nl.par_dim(128), 128), dtype=a_ptr.dtype, buffer=ncc.sbuf.alloc(mod_alloc(base_addr=0)))
      identity_load[ix0, iy0] = nl.load(identity, dtype=a_ptr.dtype)

      a_local_transpose = nl.ndarray((nl.par_dim(M1), N1), dtype=a_ptr.dtype,
                                     buffer=ncc.psum.mod_alloc(base_bank=0))
      a_local_transpose[ix0, iy0] = nisa.nc_matmul(a_local[ix0, iy0], identity_load)

      a_t_sbuf = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype,
                                     buffer=ncc.sbuf.alloc(mod_alloc(base_addr=2048)))
      a_t_sbuf[ix0, iy0] = nl.copy(a_local_transpose[ix0, iy0])

      nl.store(tp_ptr[m0 * 128 + ix0, n0 * 128 + iy0], value=a_t_sbuf[ix0, iy0])
```

## utils.cpp

```cpp
#include "utils.hpp"
#include "../tokenizers_binding/remote_rust_tokenizer.h"

#include <random>
#include <sstream>

#include <torch/csrc/jit/passes/inliner.h>
#include <ATen/ATen.h>

std::string get_visible_cores_str(size_t num_neuron_cores, size_t cores_per_model)
{
    std::ostringstream oss;
    oss << "0-" << ((num_neuron_cores * cores_per_model) - 1);
    return oss.str();
}

std::string get_uuid()
{
    static const char *chars = "0123456789abcdef";
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<> dist(0, 15);

    std::stringstream ss;
    for (size_t i = 0; i < 37; i++) {
        const int index = dist(mt);
        ss << chars[index];
    }

    std::stringstream variant_ss;
    size_t variant;
    variant_ss << std::hex << chars[dist(mt)];
    variant_ss >> variant;
    variant = 0x8 | (0x3 & variant);

    ss.seekp(9); ss << "-";
    ss.seekp(14); ss << "-4";
    ss.seekp(19); ss << "-" << std::hex << variant;
    ss.seekp(24); ss << "-";
    return ss.str();
}

torch::jit::script::Module get_model(const std::string& filename)
{
    torch::jit::script::Module model = torch::jit::load(filename);
    return model;
}
```

## benchmark_utils.py

```python
import math
from collections import Counter

import numpy as np

class Results():

    def __init__(self, batch_size, num_cores=1):
        self.latency_array = []
        self.end_times = []
        self.start_times = []
        self.batch_size = batch_size
        self.num_cores = num_cores

    def add_result(self, latency_array, end_times, start_times):
        self.latency_array.extend(latency_array)
        self.end_times.extend(end_times)
        self.start_times.extend(start_times)

    def report(self, f, window_size=1):
        assert(len(self.latency_array) != 0)
        p50_latency = np.percentile(self.latency_array, 50)
        p90_latency = np.percentile(self.latency_array, 90)
        p95_latency = np.percentile(self.latency_array, 95)
        p99_latency = np.percentile(self.latency_array, 99)
        p100_latency = np.percentile(self.latency_array, 100)

        def get_bucket(start, end):
            bucketed_start = math.floor(start / window_size) * window_size
            bucketed_end = math.ceil(end / window_size) * window_size
            if bucketed_end - bucketed_start == window_size:
                return bucketed_start
            else:
                return None
            
        bucketed_timestamps = [get_bucket(start, end)
                            for start, end in zip(self.start_times, self.end_times)]
        counted_buckets = Counter(
            item for item in bucketed_timestamps if item is not None)
        bucket_throughputs = [(key, value / window_size)
                            for key, value in sorted(counted_buckets.items())]
        
        busy_throughputs = [value for _, value in bucket_throughputs]
        max_throughput = max(busy_throughputs) * self.batch_size
        avg_throughput = sum(busy_throughputs) * self.batch_size / len(busy_throughputs)
```

## layernorm_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

# Reference torch implementation
def layernorm_layer(input_tensor, epsilon, gamma_vector, beta_vector):
    # Compute the mean and variance of the input tensor along the last dimension
    mean = input_tensor.mean(dim=-1, keepdim=True)
    variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
    # Subtract the mean from the input and divide by the square root of the variance plus epsilon
    normalized_input = (input_tensor - mean) / torch.sqrt(variance + epsilon)
    # Apply the affine transformation
    normalized_input = normalized_input * gamma_vector + beta_vector
    return normalized_input
```

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

# Copy tensors to NeuronDevice
input_tensor = input_tensor.to(device=device)
gamma_vector = gamma_vector.to(device=device)
beta_vector = beta_vector.to(device=device)

# Compute NKI layernorm kernel in NeuronDevice
xm.mark_step()
output_nki = func(input_tensor, epsilon, gamma_vector, beta_vector)
xm.mark_step()
output_nki = output_nki.to(device='cpu')
```

## average_pool2d_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
  """NKI kernel to compute a 2D avg-pool operation

  Args:
      in_tensor: an input tensor, of shape C x H x W
      pool_size: an integer representing a (square) pool-window size

  Return:
      out_tensor: the resulting output tensor, of shape C x (H/pool_size) x (W/pool_size)
  """

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_hout = sz_hin // pool_size
  sz_wout = sz_win // pool_size
  # Create output tensor shared between all SPMD instances as result tensor
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i0, i1, i2, i3, i4 = nl.mgrid[0:sz_p, 0:sz_hin//sz_pool, 0:sz_pool, 0:sz_win//sz_pool, 0:sz_pool]

  # Load input data from external memory to on-chip memory
  in_tile: tensor[sz_p, sz_hin, sz_win] = nl.load(in_tensor)

  # Perform the pooling operation:
  # We use numpy's advanced indexing, in order to extend in_tile to 5D, and then reduce-average two dimension.
  # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
  # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
  # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
  # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].
  out_tile : tensor[sz_p, sz_hout, sz_wout] = nl.sum(in_tile[i0, sz_pool*i1+i2, sz_pool*i3+i4],
                                                     axis=[2,4]) / (pool_size*pool_size)

  # Store the results back to hbm
  nl.store(out_tensor, value=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
```

## transpose2d_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
  """
  NKI kernel to reorder the elements on axis[1] of the input tensor.

  Every row of the input tensor is a flattened row-major 2D matrix.
  The shape2D argument defines the dimensions of the flattened matrices (#rows,#cols).
  Our goal in this kernel is to transpose these flattened 2D matrices, i.e. make them (#cols,#rows).

  Example:
      in_tensor = [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3]
      shape2D = (3,4)
  this means that in_tensor has 3 rows and 4 columns, i.e. can be represented as:
      [a0,a1,a2,a3]
      [b0,b1,b2,b3]
      [c0,c1,c2,c3]
  after transpose, we expect to get:
      [a0,b0,c0]
      [a1,b1,c1]
      [a2,b2,c2]
      [a3,b3,c3]
  Thus, out_tensor is expected to be [a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3]

  Args:
    in_tensor: an input tensor
    shape2D: tuple representing the dimensions to be transposed: (#rows, #cols)
    out_tensor: an output (transposed) tensor
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # We're going to need 3 indices to perform f1:f2 transpose.
  # - i_p0 is the parallel index
  # - i_f1 and i_f2 are both free-dim indices, and will be used to transpose between the f1/f2 axes
  i_p0 = nl.arange(sz_p)[:, None, None]
  i_f1 = nl.arange(sz_f1)[None, :, None]
  i_f2 = nl.arange(sz_f2)[None, None, :]

  # Perform the transposition via a SBUF-to-SBUF copy, with access-pattern manipulation
  # Note that we have 2D tensors and 3 indices, since we need to represent a 2D access pattern *per partition*
  # RHS traverses an F1 x F2 matrix in a row major manner
  # LHS traverses an F2 x F1 (new) matrix in a row major manner
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=out_tensor.dtype)
  out_tile[i_p0, i_f2*sz_f1+i_f1] = nl.copy(in_tile[i_p0, i_f1*sz_f2+i_f2])

  # Finally, we store out_tile to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor
```

## index-case-3.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def tensor_maxpool_kernel_(in_tensor, pool_size):
  """NKI kernel to compute a 2D max-pool operation

  Args:
      in_tensor: an input tensor, of dimensions C x H x W
      pool_size: integer P representing a (square) pool-window size
  Returns:
      out_tensor: the resulting output tensor, of dimensions C x (H/P) x (W/P)
  """

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_hout, sz_wout = sz_hin // pool_size, sz_win // pool_size
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Generate tensor h/w index patterns
  # 3D indexing according to [C, H, W]
  i_p = nl.arange(sz_p)[:, None, None]
  i_win = nl.arange(sz_win)[None, None, :]
  i_hin = nl.arange(sz_hin)[None, :, None]

  i_wout = nl.arange(sz_wout)[None, None, :]
  i_hout = nl.arange(sz_hout)[None, :, None]

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i_0 = nl.arange(sz_p)[:, None, None, None, None]
  i_1 = nl.arange(sz_hin//sz_pool)[None, :, None, None, None]
  i_2 = nl.arange(sz_pool)[None, None, :, None, None]
  i_3 = nl.arange(sz_win//sz_pool)[None, None, None, :, None]
  i_4 = nl.arange(sz_pool)[None, None, None, None, :]

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray([sz_p, sz_hin, sz_win], dtype=in_tensor.dtype)
  in_tile[:,:,:] = nl.load(in_tensor[i_p, i_hin, i_win])

  # Perform the pooling operation
  out_tile = nl.max(in_tile[i_0, sz_pool*i_1+i_2, sz_pool*i_3+i_4], axis=[2,4])

  # Store the results back to external memory
  nl.store(out_tensor[i_p, i_hout, i_wout], value=out_tile)

  return out_tensor
```

## rmsnorm_nki_kernels.py

```python
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
  # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
  # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
  # and N = a_tensor.shape[1]
  # Reduction (mean) is performed in the free (2nd) dimension
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Make sure shapes match
  assert a_tensor.shape[1] == g_tensor.shape[0]

  # Generate tensor indices to index input tensor
  ix = nl.arange(128)[:, None]
  iw = nl.arange(1)[:, None]
  iy = nl.arange(a_tensor.shape[1])[None, :]

  num_rows = a_tensor.shape[0]

  # Load RMSNorm weight once, reused by rows/tiles of a_tensor
  g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

  # Process 128 rows at a time due to 128-partition tile size limitation
  # Since we're not reducing across the first dimension
  # Tiles can be processed independently
  for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

    # Load input data from external memory to on-chip memory
    a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                    mask=(i * 128 + ix < num_rows))

    # Compute element-wise square of a_tensor
    in_square = nl.square(a_tile)

    # Calculate sum of squared elements, along last dimension
    square_sum = nl.sum(in_square, axis=[1])

    # Scale and get a reciprocal
    mean = square_sum / a_tensor.shape[1]

    # Take square root of mean and then reciprocal with
    # rsqrt API (one ISA instruction)
    rms_reciprocal = nl.rsqrt(mean)

    # Scale the input tensor
    out_tile = nl.multiply(a_tile, rms_reciprocal)

    # Broadcast weight along first axis to match tensor shape
    # num_rows_active = min(num_rows - i * 128, 128)
    g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

    # Multiply with the RMSNorm weight
    out_tile[...] = nl.multiply(out_tile, g_bcast,
                           mask=(i * 128 + ix < num_rows))

    # store the addition results back to external memory (out_tensor)
    nl.store(out_tensor[i * 128 + ix, iy], value=out_tile,
            mask=(i * 128 + ix < num_rows))

  return out_tensor
```

## parallel.py

```python
import mxnet as mx
from multiprocessing import Process, Manager


def consumer(model_file, sample_input, input_queue, result_queue):
    sym, args, aux = mx.model.load_checkpoint(model_file, 0)
    sample_input = {key: mx.nd.array(v) for key, v in sample_input.items()}
    args.update(sample_input)
    model = sym.bind(mx.cpu(), args=args, aux_states=aux, grad_req="null")

    while True:
        inputs, input_id = input_queue.get()
        input_queue.task_done()
        if inputs == "stop":
            break
        inputs = {key: mx.nd.array(v) for key, v in inputs.items()}
        results = model.forward(**inputs)
        results[0].wait_to_read()

        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]

        if input_id != -1:
            result_queue.put((results, input_id))


class NeuronSimpleDataParallel:
    def __init__(self, model_file, num_neuron_cores, sample_input):
        self.num_neuron_cores = num_neuron_cores
        self.sample_input = sample_input
        self.model_path = model_file
        manager = Manager()
        self.input_queue = manager.Queue(maxsize=num_neuron_cores * 16)
        self.result_queue = manager.Queue(maxsize=num_neuron_cores * 16)

        self.processes = [
            Process(
                target=consumer,
                args=(
                    self.model_path,
                    self.sample_input,
                    self.input_queue,
                    self.result_queue,
                ),
            )
            for _ in range(num_neuron_cores)
        ]
        self.input_id = 0
        self.input_dict = set()

    def start_continuous_inference(self):
        for p in self.processes:
            p.start()

    def infer(self, batch):
        self.input_id += 1
        self.input_dict.add(self.input_id)
        self.input_queue.put((batch, self.input_id))

    def stop(self):
        for _ in range(self.num_neuron_cores):
            self.input_queue.put(("stop", -1))
```

## test_nki_isa_sequence_bounds.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def nki_sequence_bounds(segment_ids):
  output = nl.ndarray([1, 2, 32], dtype=segment_ids.dtype, buffer=nl.shared_hbm)
  
  m, n = segment_ids.shape
  ix, iy, iz = nl.mgrid[0:m, 0:2, 0:n]
  
  out_tile = nl.ndarray([m, 2, n], dtype=segment_ids.dtype, buffer=nl.sbuf)
  seq_tile = nl.load(segment_ids)
  out_tile[ix, iy, iz] = nisa.sequence_bounds(segment_ids=seq_tile)
  
  nl.store(output, value=out_tile)
  return output
```

```python
import numpy as np

def compute_sequence_bounds(sequence):
  n = len(sequence)
  
  min_bounds = np.zeros(n, dtype=sequence.dtype)
  max_bounds = np.zeros(n, dtype=sequence.dtype)
  
  min_bound_pad = n
  max_bound_pad = -1
  
  min_bounds[0] = 0 if sequence[0] != 0 else min_bound_pad
  for i in range(1, n):
    if sequence[i] == 0:
      min_bounds[i] = min_bound_pad
    elif sequence[i] == sequence[i - 1]:
      min_bounds[i] = min_bounds[i - 1]
    else:
      min_bounds[i] = i
  
  max_bounds[-1] = n if sequence[-1] != 0 else max_bound_pad
  for i in range(n - 2, -1, -1):
    if sequence[i] == 0:
      max_bounds[i] = max_bound_pad
    elif sequence[i] == sequence[i + 1]:
      max_bounds[i] = max_bounds[i + 1]
    else:
      max_bounds[i] = i + 1
  
  return np.vstack((min_bounds, max_bounds))
```

## compile.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_neuron import trace  # or from torch_neuronx.xla_impl.trace import trace

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

max_length = 128
paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

batch_size = 6
example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Trace model for Neuron optimization
model_neuron = trace(model, example_inputs_paraphrase)

# Run inference
paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)

# Save compiled model
model_neuron.save(f'bert_neuron_b{batch_size}.pt')
```

## spmd_tensor_addition_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output sizes can be uniformly tiled to [128,512]

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor shared between all SPMD instances as result tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Calculate tile offsets based on current 'program'
  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 512

  # Generate tensor indices to index tensors a and b
  ix = offset_i_x + nl.arange(128)[:, None]
  iy = offset_i_y + nl.arange(512)[None, :]

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  a_tile = nl.load(a_input[ix, iy])
  b_tile = nl.load(b_input[ix, iy])

  # compute a + b
  c_tile = a_tile + b_tile

  # store the addition results back to device memory (c_output)
  nl.store(c_output[ix, iy], value=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output
```

```python
def nki_tensor_add(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  grid_x = a_input.shape[0] // 128
  grid_y = a_input.shape[1] // 512

  return nki_tensor_add_kernel_[grid_x, grid_y](a_input, b_input)
```

## gen_resnet50_keras.py

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from google.protobuf import text_format
import tensorflow.python.saved_model

# Keras global configurations
tf.keras.backend.set_learning_phase(0)
tf.keras.backend.set_image_data_format('channels_last')
tf.keras.backend.set_floatx('float32')

# Load pre-trained model
model = ResNet50(weights='imagenet')

# Extract model parameters
model_input = model.input.name.replace(':0', '')
model_output = model.output.name.replace(':0', '')
batch, height, width, channels = model.input.shape

# Obtain TensorFlow session
sess = tf.compat.v1.keras.backend.get_session()

# Save checkpoint files
ckpt_file = '/tmp/model/model.ckpt'
graph_file = '/tmp/model/model.pb'
tf.compat.v1.train.Saver().save(sess, ckpt_file)
tf.io.write_graph(sess.graph.as_graph_def(), logdir='.', name=graph_file, as_text=False)

# Freeze graph and convert variables to constants
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    saver = tf.compat.v1.train.import_meta_graph(ckpt_file + '.meta')
    saver.restore(sess, ckpt_file)
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, tf.compat.v1.get_default_graph().as_graph_def(), [model_output])
    output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(
        output_graph_def, protected_nodes=[model_output])
    with open('frozen_model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
```

## test_nki_isa_nc_transpose.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_nc_transpose(a_tensor, b_tensor):
  at_tensor = nl.ndarray([a_tensor.shape[1], a_tensor.shape[0]], dtype=a_tensor.dtype,
                         buffer=nl.shared_hbm)
  bt_tensor = nl.ndarray([b_tensor.shape[1], b_tensor.shape[0]], dtype=b_tensor.dtype,
                         buffer=nl.shared_hbm)
  
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])
  
  # Example 1: transpose tile a of shape (128, 64)
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  aT = nisa.nc_transpose(a[i_p_a, i_f_a])

  i_p_aT = nl.arange(64)[:, None]
  i_f_aT = nl.arange(128)[None, :]
  nl.store(at_tensor[i_p_aT, i_f_aT], aT)

  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b])
  
  # Example 2: transpose tile b of shape (32, 2) using Vector Engine
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)

  i_p_bT = nl.arange(2)[:, None]
  i_f_bT = nl.arange(32)[None, :]
  nl.store(bt_tensor[i_p_bT, i_f_bT], bT)
  return at_tensor, bt_tensor
```

## index-case-1.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def tensor_split_kernel_(in_tensor):
  """NKI kernel to split an input tensor into two output tensors, along the column axis.

  The even columns of the input tensor will be gathered into the first output tensor,
  and the odd columns of the input tensor will be gathered into the second output tensor.

  Args:
      in_tensor: an input tensor
  Returns:
      out_tensor_even: a first output tensor (will hold the even columns of the input tensor)
      out_tensor_odd: a second output tensor (will hold the odd columns of the input tensor)
  """

  # Extract tile sizes.
  sz_p, sz_f = in_tensor.shape
  sz_fout_even = sz_f - sz_f // 2
  sz_fout_odd = sz_f // 2
  out_tensor_even = nl.ndarray((sz_p, sz_fout_even), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  out_tensor_odd = nl.ndarray((sz_p, sz_fout_odd), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  # We assume that all three tensors have the same partition dimension size
  # and it does not exceed pmax
  assert in_tensor.shape[0] == out_tensor_even.shape[0] == out_tensor_odd.shape[0]
  assert in_tensor.shape[0] <= nl.tile_size.pmax

  # Make sure even/odd output tensors have correct free dimension size
  assert sz_fout_even == math.ceil(sz_f / 2)
  assert sz_fout_odd == math.floor(sz_f / 2)

  # Generate tensor indices for the input/output tensors
  i_p = nl.arange(sz_p)[:, None]
  i_f = nl.arange(sz_f)[None, :]
  i_fout_even = nl.arange(sz_fout_even)[None, :]
  i_fout_odd = nl.arange(sz_fout_odd)[None, :]

  # Split pattern:
  i_f_even = (2 * i_fout_even)
  i_f_odd = (2 * i_fout_odd + 1)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor[i_p, i_f])

  # Perform the split
  # these assignments invoke copy instructions under the hood
  # which can execute on either Scalar or Vector Engine
  # (decided by compiler instruction scheduler)
  out_tile_even = in_tile[i_p, i_f_even]
  out_tile_odd = in_tile[i_p, i_f_odd]

  # Store the results back to external memory
  nl.store(out_tensor_even[i_p, i_fout_even], value=out_tile_even)
  nl.store(out_tensor_odd[i_p, i_fout_odd], value=out_tile_odd)

  return out_tensor_even, out_tensor_odd
```

## train_monitor.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm

class MLP(nn.Module):
    def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
```

```python
device = 'xla'
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

model.train()
for idx, (train_x, train_label) in enumerate(train_loader):
    optimizer.zero_grad()
    train_x = train_x.view(train_x.size(0), -1)
    train_x = train_x.to(device)
    train_label = train_label.to(device)
    output = model(train_x)
    loss = loss_fn(output, train_label)
    loss.backward()
    optimizer.step()
    xm.mark_step()
```

```python
checkpoint = {'state_dict': model.state_dict()}
xm.save(checkpoint, 'checkpoints/checkpoint.pt')
```

## test_nki_spmd_grid.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_spmd_kernel(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
  i = nl.program_id(0)
  j = nl.program_id(1)
  
  a_tile = nl.load(a[i, j])
  nl.store(b[i, j], a_tile)

  return b
```

```python
# Example 1: Let compiler decide how to distribute the instances of spmd kernel
dst = nki_spmd_kernel[4, 2](src)
```

```python
# Example 2: Distribute SPMD kernel instances to physical NeuronCores with explicit annotations
dst = nki_spmd_kernel[nl.spmd_dim(nl.nc(2), 2), 2](src)
dst = nki_spmd_kernel[nl.nc(2) * 2, 2](src)  # syntactic sugar
```

```python
# Example 3: Distribute SPMD kernel instances to physical NeuronCores with explicit annotations
dst = nki_spmd_kernel[nl.spmd_dim(2, nl.nc(2)), 2](src)
dst = nki_spmd_kernel[2 * nl.nc(2), 2](src)  # syntactic sugar
```

## test_nki_nl_atomic_rmw.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def atomic_rmw_indirect_indices(in_tensor, indices_tensor, value_tensor):
  rmw_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  in_tile = nl.load(in_tensor)
  nl.store(rmw_tensor, in_tile)

  N = 128
  M = 512

  value: tensor[N, M] = nl.load(value_tensor)
  indices_tile: tensor[N, 1] = nl.load(indices_tensor)
  ix = nl.arange(M)[None, :]

  nl.atomic_rmw(rmw_tensor[indices_tile, ix], value=value, op=np.add)
  
  return rmw_tensor
```

## detect_instance.py

```python
import torch
import torch_neuronx
from typing import Optional

INSTANCETYPE_TO_NEURONCORES = {
    "inf1.xlarge": 4,
    "inf1.2xlarge": 4,
    "inf1.6xlarge": 16,
    "inf2.xlarge": 2,
    "inf2.8xlarge": 2,
    "inf2.24xlarge": 12,
    "inf2.48xlarge": 24,
    "inf1.24xlarge": 64,
    "trn1.2xlarge": 2,
    "trn1.32xlarge": 32,
}

def get_instance_type() -> str:
    """Try to obtain the instance type."""
    try:
        from urllib.request import Request, urlopen

        req = Request("http://169.254.169.254/latest/api/token", method="PUT")
        req.add_header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
        with urlopen(req) as response:
            token = response.read().decode("utf-8")

        req = Request("http://169.254.169.254/latest/meta-data/instance-type")
        req.add_header("X-aws-ec2-metadata-token", token)
        with urlopen(req) as response:
            instance_type = response.read().decode("utf-8")

        return instance_type
    except:
        return None

def get_num_neuroncores(instance_type: Optional[str] = None) -> int:
    """
    Try to obtain the maximum number of NeuronCores available on this instance.

    Args:
        instance_type: The Neuron instance type. Autodetermined from current instance
            if not provided.

    Returns:
        The number of NeuronCores (or 2 if the type is unknown).
    """
    try:
        if not instance_type:
            instance_type = get_instance_type()
        return INSTANCETYPE_TO_NEURONCORES[instance_type]
    except KeyError:
        num_cores = get_num_neuroncores_v3()
        return num_cores

def get_num_neuroncores_v3() -> int:
    """
    Retrieve the number of NeuronCores visible to this process.

    Returns:
        The number of visible neuron cores.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
    """
    runtime = torch.classes.neuron.Runtime()
    try:
        nc_count = runtime.get_visible_nc_count()
    except RuntimeError as e:
        raise RuntimeError(
            "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"
        ) from e
    return nc_count
```

## test_nki_nl_load_store.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def example_kernel(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # load from in_tensor[P, F] that is on HBM
  # copy into data_tile[P, F] that is on SBUF
  data_tile = nl.load(in_tensor)
  
  # store into out_tensor[P, F] that is on HBM
  # from data_tile[P, F] that is on SBUF
  nl.store(out_tensor, data_tile)
  return out_tensor


@nki.jit(mode="simulation")
def example_load_store_b(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, in_tensor.dtype,
                          buffer=nl.shared_hbm)
  for i_b in nl.affine_range(4):
    data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
    # load from in_tensor[4, 128, 512] one batch at a time
    # copy into data_tile[128, 512]
    i_p, i_f = nl.mgrid[0:128, 0:512]
    data_tile[i_p, i_f] = nl.load(in_tensor[i_b, i_p, i_f])
    
    # store into out_tensor[4, 128, 512] one batch at a time
    # from data_tile[128, 512] 
    i_p, i_f = nl.mgrid[0:128, 0:512]
    nl.store(out_tensor[i_b, i_p, i_f], value=data_tile[i_p, i_f]) 
  return out_tensor
```

## mlp_train.py

```python
import torch
from model import MLP
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch_xla.core.xla_model as xm

# XLA device usage
device = 'xla'
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

model.train()
for train_x, train_label in DataLoader(train_dataset, batch_size=32):
    optimizer.zero_grad()
    train_x = train_x.view(train_x.size(0), -1)
    train_x = train_x.to(device)
    train_label = train_label.to(device)
    output = model(train_x)
    loss = loss_fn(output, train_label)
    loss.backward()
    optimizer.step()
    xm.mark_step()  # XLA: collect ops and run them in XLA runtime

# XLA checkpoint saving
checkpoint = {'state_dict': model.state_dict()}
xm.save(checkpoint, 'checkpoints/checkpoint.pt')
```

## test_nki_isa_dropout.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def nki_dropout(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a: tensor[128, 512] = nl.load(a_tensor)
  b: tensor[128, 1] = nl.load(b_tensor)

  c: tensor[128, 512] = nisa.dropout(a, prob=b)

  nl.store(c_tensor, c)

  return c_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_dropout_scalar(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(in_tensor)

  b = nisa.dropout(a, prob=0.2)

  nl.store(out_tensor, b)

  return out_tensor
```

## test_nki_isa_partition_reduce.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

nki_jit = nki.trace

@nki_jit
def nki_par_reduce(a_tensor, b_tensor):
  a = nl.load(a_tensor[0:128, 0:32, 0:4])  
  b = nisa.tensor_partition_reduce(np.add, a)
  nl.store(b_tensor[0:1, 0:32, 0:4], b)

@nki_jit
def nki_par_reduce_nd_b(a_tensor, b_tensor):
  for i in nl.affine_range(a_tensor.shape[0]):
    a = nl.load(a_tensor[i])
    b = nisa.tensor_partition_reduce(np.add, a)
    nl.store(b_tensor[i], b)
```

## mm-nisa-spmd.py

```python
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc import nki

@nki.jit
def matmul_128x128x512_spmd(A_T, B):
  """NKI kernel to compute a 128x128x512 matrix multiplication operation
     Use SPMD program IDs to index into the full A and B input tensor to get tiles
     for 128x128x512 matrix multiplication.

  Args:
      A_T: an input tensor of shape [K=128,M=512],
         a left hand side argument of the matrix multiplication,
      B: an input tensor of shape [K=128,N=1024],
         a right hand side argument of the matrix multiplication
      result: the resulting output tensor of shape [M=512,N=1024]
  """
  K, N = A_T.shape
  K_, M = B.shape
  assert K == K_
  # Create output tensor shared between all SPMD instances as result tensor
  result = nl.ndarray((N, M), dtype=A_T.dtype, buffer=nl.shared_hbm)

  # Defining starting indexes for input A.T and B
  i_A_T_col = nl.program_id(0) * 128
  i_B_col = nl.program_id(1) * 512

  # Loading the inputs (HBM->SBUF)
  A_T_tile = nl.load(A_T[0:128, i_A_T_col:i_A_T_col+128])
  B_tile = nl.load(B[0:128, i_B_col:i_B_col+512])

  # Perform the matrix-multiplication
  result_psum = nisa.nc_matmul(A_T_tile, B_tile)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # Store back into result tile with the correct SPMD offsets.
  nl.store(result[i_A_T_col:i_A_T_col+128, i_B_col:i_B_col+512], value=result_sbuf)

  return result
```

## spmd_multiple_nc_tensor_addition_nki_kernels.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from spmd_tensor_addition_nki_kernels import nki_tensor_add_kernel_


def nki_tensor_add_nc2(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors using multiple Neuron cores.

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs.
  a_input and b_input are sharded across Neuron cores, directly utilizing Trn2 architecture capabilities

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  # Since we're sharding across neuron cores on the 1st dimension we want to do our slicing at 
  # 128 per core * 2 cores = 256
  grid_x = a_input.shape[0] // (128 * 2)
  grid_y = a_input.shape[1] // 512

  # In addition, we distribute the kernel to physical neuron cores around the first dimension
  # of the spmd grid.
  # This means:
  # Physical NC [0]: kernel[n, m] where n is even
  # Physical NC [1]: kernel[n, m] where n is odd
  # notice, by specifying this information in the SPMD grid, we can use multiple neuron cores
  # without updating the original `nki_tensor_add_kernel_` kernel.
  return nki_tensor_add_kernel_[nl.spmd_dim(grid_x, nl.nc(2)), grid_y](a_input, b_input)
```

## test_nki_nl_load_transpose2d.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def example_kernel_0(in_tensor):
  out_tensor = nl.ndarray([in_tensor.shape[1], in_tensor.shape[0]], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  N, M = in_tensor.shape
  local_tile: tensor[M, N] = nl.load_transpose2d(in_tensor)
  nl.store(out_tensor, value=local_tile)
  return out_tensor
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def example_kernel_1(in_tensor):
  out_tensor = nl.ndarray([in_tensor.shape[1], in_tensor.shape[0]], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  # always use the DMA engine
  N, M = in_tensor.shape
  local_tile: tensor[M, N] = nisa.dma_transpose(in_tensor)
  nl.store(out_tensor, value=local_tile)
  return out_tensor
```

## mm-nl-spmd.py

```python
import neuronxcc.nki.language as nl
from neuronxcc import nki


@nki.jit
def matmul_128x128x512_spmd(A, B):
  """NKI kernel to compute a 128x128x512 matrix multiplication operation.
     Use SPMD program IDs to index into the full A and B input tensor to get tiles
     for 128x128x512 matrix multiplication.

  Args:
      A: an input tensor of shape [M=512,K=128],
         a left hand side argument of the matrix multiplication,
      B: an input tensor of shape [K=128,N=1024],
         a right hand side argument of the matrix multiplication
      result: the resulting output tensor of shape [M=512,N=1024]
  """
  N, K = A.shape
  K_, M = B.shape
  assert K == K_
  # Create output tensor shared between all SPMD instances as result tensor
  result = nl.ndarray((N, M), dtype=A.dtype, buffer=nl.shared_hbm)

  # Defining starting indexes for input A and B
  i_A_row = nl.program_id(0) * 128
  i_B_col = nl.program_id(1) * 512

  # Loading the inputs (HBM->SBUF)
  A_tile = nl.load(A[i_A_row:i_A_row+128, 0:128])
  B_tile = nl.load(B[0:128, i_B_col:i_B_col+512])

  # Perform the matrix-multiplication
  # Note1: nl.matmul will invoke a transpose on A_tile before performing the actual matmul operation
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nl.matmul(A_tile, B_tile)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # The result of a [128,128] x [128,512] matrix multiplication has a shape of [128, 512].
  # This dictates which indices to use to address the result tile.
  nl.store(result[i_A_row:i_A_row+128, i_B_col:i_B_col+512], value=result_sbuf)

  return result
```

## test_nki_isa_tensor_tensor_scan.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_tensor_tensor_scan(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = nl.ndarray(shape=(128, 1024), dtype=nl.float32)

  c[:, 0:512] = nisa.tensor_tensor_scan(a[:, 0:512], b[:, 0:512],
                                        initial=0, op0=np.multiply, op1=np.add)

  c[:, 512:1024] = nisa.tensor_tensor_scan(a[:, 512:1024], b[:, 512:1024],
                                           initial=c[:, 511],
                                           op0=np.multiply, op1=np.add)

  nl.store(c_tensor, c)
  return c_tensor
```

## matrix_multiplication_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

device = xm.xla_device()
cpu = torch.device('cpu')

# Small workload with basic kernel
lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

output_small = nki_matmul_basic_(lhs_small.T, rhs_small)
output_small_torch = torch.matmul(lhs_small, rhs_small)

if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")

# Large workload with tiled kernels
lhs = torch.rand((4096, 1024), dtype=torch.bfloat16, device=device)
rhs = torch.rand((1024, 2048), dtype=torch.bfloat16, device=device)

output_torch = torch.matmul(lhs, rhs).to(device=cpu)

def check_match(nki_func):
    output = nki_func(lhs.T, rhs)
    output_nki = output.to(device=cpu)
    if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
        print("NKI and Torch match")

check_match(nki_matmul_tiled_)
check_match(nki_matmul_hoist_load_)
check_match(nki_matmul_block_free_dimension_)
check_match(nki_matmul_fully_optimized_)
```

## infer_resnet50_keras.py

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

def pb_to_saved_model(pb_path, input_names, output_names, model_dir):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(pb_path, 'rb').read())
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        inputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in input_names.items()}
        outputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in output_names.items()}
        tf.saved_model.simple_save(sess, model_dir, inputs, outputs)
```

## test_nki_isa_activation.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit(mode="simulation")
def nki_activation(a_tensor, b_tensor, c_tensor):
  a_act_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  b_act_tensor = nl.ndarray(b_tensor.shape, dtype=b_tensor.dtype, buffer=nl.shared_hbm)

  # Example 1: perform exponential function on matrix a of shape (128, 1024)
  a = nl.load(a_tensor)
  activated_a = nisa.activation(op=nl.exp, data=a)
  nl.store(a_act_tensor, activated_a)

  # Example 2: perform the following operations to matrix b of shape (128, 512)
  # using a single activation instruction: np.square(b * 2.0) + c
  # 1) compute `np.square(b * 2.0 + c)`
  # 2) cast 1) results into bfloat16
  b = nl.load(b_tensor)
  c = nl.load(c_tensor)
  activated_b = nisa.activation(op=np.square, data=b, bias=c, scale=2.0,
                                dtype=nl.bfloat16)
  nl.store(b_act_tensor, activated_b)

  return a_act_tensor, b_act_tensor
```

## test_nki_isa_copypredicated.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import numpy as np


@nki.jit(mode="simulation")
def nki_copy_predicated(predicate, on_true_tensor, on_false_tensor):
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=on_true_tensor.dtype,
                                            buffer=nl.shared_hbm)
  
  pre_tile: tensor[128, 512] = nl.load(predicate)
  src_tile: tensor[128, 512] = nl.load(on_true_tensor)

  ix, iy = nl.mgrid[0:128, 0:512]
  dst_tile: tensor[128, 512] = nl.zeros(shape=src_tile.shape, dtype=src_tile.dtype)
  dst_tile[ix, iy] = nl.load(on_false_tensor)

  nisa.tensor_copy_predicated(src=src_tile, dst=dst_tile, predicate=pre_tile)

  nl.store(out_tensor, dst_tile)
  return out_tensor
```

## test_nki_nl_gather_flattened.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit(mode="simulation")
def nki_gather_flattened():
    ##################################################################
    # Example 1: Gather values from a tensor using indices
    ##################################################################
    # Create source tensor
    N = 32
    M = 64
    data = nl.rand((N, M), dtype=nl.float32)

    # Create indices tensor - gather every 5th element
    indices = nl.zeros((N, 10), dtype=nl.uint32)
    for i in nl.static_range(N):
        for j in nl.static_range(10):
            indices[i, j] = j * 5

    # Gather values from data according to indices
    result = nl.gather_flattened(data=data, indices=indices)

    # Create output tensor and store result
    data_tensor = nl.ndarray([N, M], dtype=data.dtype, buffer=nl.shared_hbm)
    nl.store(data_tensor, value=data)
    indices_tensor = nl.ndarray([N, 10], dtype=nl.int32, buffer=nl.shared_hbm)
    nl.store(indices_tensor, value=indices)
    result_tensor = nl.ndarray([N, 10], dtype=data.dtype, buffer=nl.shared_hbm)
    nl.store(result_tensor, value=result)

    return data_tensor, indices_tensor, result_tensor
```

## test_nki_nl_mgrid.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def example_kernel(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  i_p, i_f = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[i_p, i_f])
  nl.store(out_tensor[i_p, i_f], tile)
  return out_tensor


@nki.jit(mode="simulation")
def example_kernel_1(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  grid = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[grid.p, grid.x])
  nl.store(out_tensor[grid.p, grid.x], tile)
  return out_tensor
```

## test_nki_isa_nc_find_index8.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import numpy as np


@nki.jit(mode="simulation")
def nki_max_index8():
  # Generate random data
  data = nl.rand((32, 128))

  # Find max 8 values per row
  max_vals = nisa.max8(src=data)

  # Create output tensor for indices
  indices_tensor = nl.ndarray([32, 8], dtype=nl.uint32, buffer=nl.shared_hbm)

  # Find indices of max values
  indices = nisa.nc_find_index8(data=data, vals=max_vals)

  # Store results
  nl.store(indices_tensor, value=indices)

  return indices_tensor
```

## distilbert-base-uncased_benchmark.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", return_dict=False)

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_1,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
inputs = (
    torch.cat([paraphrase["input_ids"]] * 9, 0),
    torch.cat([paraphrase["attention_mask"]] * 9, 0),
)
```

## getting_started_baremetal.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors
    """

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    # Return kernel output as function output
    return c_output
```

## distilbert-base-uncased_compile.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_batch(tokenizer, sequence_length, batch_size):
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        torch.cat([paraphrase["input_ids"]] * batch_size, 0),
        torch.cat([paraphrase["attention_mask"]] * batch_size, 0),
    )
    return inputs
```

```python
import torch.neuron
import neuronperf.torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", return_dict=False)

inputs = [get_batch(tokenizer, 128, batch_size) for batch_size in [9]]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=[9],
    pipeline_sizes=[1],
    filename="distilbert-base-uncased_sl128.json",
    model_name="distilbert-base-uncased",
)
```

## distilroberta-base_compile.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_batch(tokenizer, sequence_length, batch_size):
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        torch.cat([paraphrase["input_ids"]] * batch_size, 0),
        torch.cat([paraphrase["attention_mask"]] * batch_size, 0),
    )
    return inputs
```

```python
import torch.neuron
import neuronperf.torch

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", return_dict=False)

inputs = [get_batch(tokenizer, sequence_length=128, batch_size=6)]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=[6],
    pipeline_sizes=[1],
    filename="distilroberta-base_sl128.json",
    model_name="distilroberta-base",
)
```

## bert-base-uncased_compile.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_batch(tokenizer, sequence_length, batch_size):
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        torch.cat([paraphrase["input_ids"]] * batch_size, 0),
        torch.cat([paraphrase["attention_mask"]] * batch_size, 0),
    )
    return inputs
```

```python
import torch.neuron
import neuronperf.torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=False)

inputs = [get_batch(tokenizer, 128, batch_size) for batch_size in [6]]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=[6],
    pipeline_sizes=[1],
    filename="bert-base-uncased_sl128.json",
    model_name="bert-base-uncased",
)
```

## bert-base-cased_compile.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_batch(tokenizer, sequence_length, batch_size):
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        torch.cat([paraphrase["input_ids"]] * batch_size, 0),
        torch.cat([paraphrase["attention_mask"]] * batch_size, 0),
    )
    return inputs
```

```python
import torch.neuron
import neuronperf.torch

model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=filename,
    model_name=model_name,
)
```

## test_nki_isa_reduce.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation")
def nki_reduce(a_tensor):
  b_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(512)[None, :]
  
  a = nl.load(a_tensor[i_p_a, i_f_a])  

  b = nisa.tensor_reduce(np.add, a[i_p_a, i_f_a], axis=[1])

  i_p_b, i_f_b = nl.mgrid[0:128, 0:1]
  nl.store(b_tensor[i_p_b, i_f_b], b)
  return b_tensor
```

## prof-kernel.py

```python
import torch
from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  sz_p, sz_f = in_tensor.shape

  i_f = nl.arange(sz_f)[None, :]

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

    return out_tensor
```

## hf-openai-clip_benchmark.py

```python
import torch
import torch_neuronx
from transformers import CLIPProcessor, CLIPModel

# Model loading and preprocessing
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name, return_dict=False)

# Input preparation
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
image = inputs['pixel_values']
image = image.repeat(batch_size, 1, 1, 1)
inputs = (inputs['input_ids'], image)

# Model tracing for Neuron compilation
model.eval()
traced = torch_neuronx.trace(model, inputs, compiler_args='--enable-saturate-infinity')
torch.jit.save(traced, filename)
```

## test_nki_isa_tensor_tensor.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def nki_tensor_tensor(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)

  a: tensor[128, 512] = nl.load(a_tensor)
  b: tensor[128, 512] = nl.load(b_tensor)

  c: tensor[128, 512] = nisa.tensor_tensor(a, b, op=nl.add)

  nl.store(c_tensor, c)
  return c_tensor
```

## test_nki_nl_broadcast.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def test_nl_broadcast(in_tensor):
  out_tensor = nl.ndarray([128, 64], in_tensor.dtype,
                          buffer=nl.shared_hbm)
  in_tile = nl.load(in_tensor, dtype=in_tensor.dtype)
  out_tile = nl.broadcast_to(in_tile, shape=(128, in_tensor.shape[1]))
  nl.store(out_tensor, out_tile)
  return out_tensor
```

## inf2_benchmark.py

```python
import torch
import torch_neuronx
from transformers import AutoModel

class GPT2Neuron(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

model = AutoModel.from_pretrained(model_name, torchscript=True)
if 'gpt2' in model_name:
    model = GPT2Neuron(model)
model.eval()

example = (
    torch.zeros(batch_size, sequence_length, dtype=torch.int),
    torch.zeros(batch_size, sequence_length, dtype=torch.int),
)

traced = torch_neuronx.trace(model, example)
torch.jit.save(traced, filename)
```

## prof-kernel-profile.py

```python
from neuronxcc import nki
from neuronxcc.nki.typing import tensor
import neuronxcc.nki.language as nl
import math

@nki.profile(working_directory=WORKING_DIRECTORY, save_neff_name='file.neff', save_trace_name='profile.ntff', profile_nth=2)
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor
  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  sz_p, sz_f = in_tensor.shape
  i_f = nl.arange(sz_f)[None, :]
  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))
    out_tile = nl.exp(in_tile)
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

  return out_tensor
```

## bert_no_model.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

pred = tf.contrib.predictor.from_saved_model(input_saved_model)
no_fuse_ops = [op.name for op in pred.graph.get_operations()]

def force_fuse_condition(op_name):
    exclude_scopes = [
        'bert/encoder/strided_slice',
        'bert/encoder/ones',
        'bert/encoder/Reshape',
        'bert/encoder/Shape',
        'bert/encoder/Cast',
    ]
    for scope in exclude_scopes:
        if op_name == scope or op_name.startswith('{}/'.format(scope)):
            return False
    return op_name.startswith('bert/encoder') or op_name.startswith('bert/pooler')

force_fuse_ops = [op.name for op in pred.graph.get_operations() if force_fuse_condition(op.name)]

compilation_result = tfn.saved_model.compile(
    input_saved_model,
    output_saved_model,
    batch_size=batch_size,
    no_fuse_ops=no_fuse_ops,
    force_fuse_ops=force_fuse_ops,
)
```

## torch-neuron-dataparallel-example-disable-dynamic-batching.rst

```python
import torch
import torch_neuron
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch.neuron.trace(model, image)

# Create the DataParallel module and use 4 NeuronCores
model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1, 2, 3], dim=0)

# Disable dynamic batching
model_parallel.disable_dynamic_batching()

# Create a batched input (this will work)
batch_size = 4
image_batched = torch.rand([batch_size, 3, 224, 224])

# This will work because
# image_batched.shape[dim] / len(device_ids) == image.shape[dim]
output = model_parallel(image_batched)
```

## torch-neuronx-dataparallel-example-disable-dynamic-batching.rst

```python
import torch
import torch_neuronx
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch_neuronx.trace(model, image)

# Create the DataParallel module and use 2 NeuronCores
model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=[0, 1], dim=0)

# Disable dynamic batching
model_parallel.disable_dynamic_batching()

# Create a batched input (this will work)
batch_size = 2
image_batched = torch.rand([batch_size, 3, 224, 224])

# This will work because
# image_batched.shape[dim] / len(device_ids) == image.shape[dim]
output = model_parallel(image_batched)
```

## test_nki_isa_tensor_copy.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_tensor_copy(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  x = nl.load(in_tensor)
  x_copy = nisa.tensor_copy(x, engine=nisa.vector_engine)
  nl.store(out_tensor, value=x_copy)

  return out_tensor
```

## hf_pretrained_wav2vec2_conformer_rope_benchmark.py

```python
import torch
import torch_neuronx
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model.eval()

inputs = processor(audio_array, return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
example = (inputs,)

traced = torch_neuronx.trace(model, example, compiler_args='--model-type=transformer')
torch.jit.save(traced, 'model.pt')

model_neuron = torch.jit.load('model.pt')
output = model_neuron(inputs)
```

## prof-kernel-benchmark.py

```python
from neuronxcc import nki
from neuronxcc.nki.typing import tensor
import neuronxcc.nki.language as nl
import math


@nki.benchmark(save_neff_name='file.neff', save_trace_name='profile.ntff')
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor
  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  sz_p, sz_f = in_tensor.shape
  i_f = nl.arange(sz_f)[None, :]
  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))
    # perform the computation
    out_tile = nl.exp(in_tile)
    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

  return out_tensor
```

## getting_started_torch.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors
    """

    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    return c_output
```

## test_nki_mask.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation")
def nki_mask(in_tensor):
  out_tensor = nl.ndarray([64, 256], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  in_tile = nl.load(in_tensor[i_p, i_f])
  out_tile = nl.square(in_tile, mask=((i_p<64) & (i_f<256)))
  nl.store(out_tensor[i_p, i_f], out_tile[i_p, i_f],
           mask=((i_p < 64) & (i_f < 256)))
  return out_tensor
```

## test_nki_isa_memset.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_memset():
  a_tensor = nl.ndarray([128, 128], dtype=nl.float32, buffer=nl.shared_hbm)
  a = nisa.memset(shape=(128, 128), value=0.2, dtype=nl.float32)
  
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(128)[None, :]
  nl.store(a_tensor[i_p, i_f], a)
  return a_tensor
```

## test_nki_isa_dma_transpose.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def nki_dma_transpose_2d_hbm2sb(a):
  b = nisa.dma_transpose(a)
  return b

@nki.jit(mode="simulation")
def nki_dma_transpose_2d_sb2sb(a):
  a_sb = nl.load(a)
  b = nisa.dma_transpose(a_sb)
  return b
```

## hf-google-vit_benchmark.py

```python
import torch
import torch_neuronx
from transformers import ViTImageProcessor, ViTForImageClassification

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torchscript=True)
model.eval()

inputs = feature_extractor(images=image, return_tensors="pt")
inputs = inputs['pixel_values'].repeat([batch_size, 1, 1, 1])
example = (inputs,)

traced = torch_neuronx.trace(model, example, compiler_args="--model-type=transformer")
torch.jit.save(traced, filename)
```

## test_nki_isa_affine_select.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit(mode="simulation")
def nki_affine_select(a_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  ix, iy = nl.mgrid[0:128, 0:128]
  a = nl.load(a_tensor[ix, iy])

  b = nisa.affine_select(pred=(iy < ix), on_true_tile=a[ix, iy], on_false_value=nl.fp32.min)

  nl.store(b_tensor[ix, iy], b)

  return b_tensor
```

## torch-neuron-dataparallel-example-dim-neq-zero.rst

```python
import torch
import torch_neuron

# Create an example model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        return self.conv(x) + 1

model = Model()
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 8, 8])
model_neuron = torch.neuron.trace(model, image)

# Create the DataParallel module using 4 NeuronCores and dim = 2
model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1, 2, 3], dim=2)

# Create a batched input
batch_size = 4 * 8
image_batched = torch.rand([1, 3, batch_size, 8])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## getting_started_jax.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors
    """

    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    return c_output
```

## test_nki_isa_reciprocal.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def reciprocal_kernel(in_tensor):
  out_tensor = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  x = nl.load(in_tensor[nl.mgrid[0:128, 0:512]])
  
  y = nisa.reciprocal(x)

  nl.store(out_tensor[nl.mgrid[0:128, 0:512]], value=y)
  return out_tensor
```

## test_nki_isa_max8.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import numpy as np


@nki.jit(mode="simulation")
def nki_max8():
  ##################################################################
  # Example 1: Generate tile b of 32 * 128 random floating point values
  # and get the 8 largest values in each row:
  ##################################################################
  expr_a = nl.rand((32, 128))
  a = nisa.max8(src=expr_a)

  a_tensor = nl.ndarray([32, 8], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(a_tensor, value=a)

  return a_tensor
```

## sd_attention_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

from sd_attention_nki_kernels import fused_self_attn_for_SD_small_head_size


device = xm.xla_device()

def cpu_golden_attn(q, k, v):
    softmax_scale = 0.125
    q_scaled = q * softmax_scale
    raw_score = torch.matmul(q_scaled, k.transpose(1, 0))
    
    norm_score = torch.nn.functional.softmax(raw_score, dim=-1)

    return torch.matmul(norm_score, v)


q_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
k_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
v_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)

output_nki = fused_self_attn_for_SD_small_head_size(q_tensor, k_tensor, v_tensor)

output_torch = cpu_golden_attn(q_tensor, k_tensor, v_tensor)
```

## layout-violation.py

```python
import neuronxcc.nki.language as nl
from neuronxcc import nki


@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of shape [128,512]
  Returns:
      out_tensor: an output tensor of shape [128,512]
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Generate indices for the input/output tensors
  i_p = nl.arange(256)[:, None]
  i_f = nl.arange(512)[None, :]

  # Load input data from HBM to on-chip memory
  in_tile = nl.load(in_tensor[i_p, i_f])

  # perform the computation:
  out_tile = nl.exp(in_tile)

  # store the results back to HBM
  nl.store(out_tensor[i_p, i_f], value=out_tile)
```

## perceiver-vision_compile.py

```python
import torch
import transformers
import neuronperf as npf
import neuronperf.torch

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

model = getattr(transformers, "PerceiverForImageClassificationLearned").from_pretrained(
    "deepmind/vision-perceiver-learned"
)
inputs = [get_batch(batch_size) for batch_size in [1]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1],
    pipeline_sizes=[1],
    filename="vision-perceiver-learned.json",
    model_name="vision-perceiver-learned",
)
```

## layout-dynamic-loop.py

```python
import neuronxcc.nki.language as nl
from neuronxcc import nki
import math

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  sz_p, sz_f = in_tensor.shape
  out_tensor = nl.ndarray((sz_p, sz_f), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  i_f = nl.arange(sz_f)[None, :]

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))

    # perform the computation
    out_tile = nl.exp(in_tile, mask=(i_p<sz_p))

    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))

    return out_tensor
```

## torch-neuronx-dataparallel-example-dim-neq-zero.rst

```python
import torch
import torch_neuronx

# Create an example model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        return self.conv(x) + 1

model = Model()
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 8, 8])
model_neuron = torch_neuronx.trace(model, image)

# Create the DataParallel module using 2 NeuronCores and dim = 2
model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=[0, 1], dim=2)

# Create a batched input
batch_size = 2 * 8
image_batched = torch.rand([1, 3, batch_size, 8])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## trace_bert_neuron.py

```python
import torch
import torch_neuron

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "HuggingFace's headquarters are situated in Manhattan"

max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Trace model with torch_neuron
model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)

# Save the traced model
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
```

## trace_bert_neuron.py

```python
import torch
import torch_neuron

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "HuggingFace's headquarters are situated in Manhattan"

max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
```

## rmsnorm_torch.py

```python
import torch
from rmsnorm_nki_kernels import nki_rmsnorm_kernel

def torch_rmsnorm_kernel(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = a_tensor.pow(2)
  # Calculate means in the free dimension
  mean = in_square.mean(dim=1, keepdim=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * torch.rsqrt(mean)

  # Scale the output by the weight
  return tensor * g_tensor
```

## average_pool2d_jax.py

```python
import jax.numpy as jnp

# Reference JAX implementation
def jax_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return jnp.nanmean(reshaped, axis=(2, 4))
```

```python
from average_pool2d_nki_kernels import tensor_avgpool_kernel

out_nki = tensor_avgpool_kernel(in_array, pool_size=POOL_SIZE)
```

## layout-pass.py

```python
import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of shape [128,512]
  Returns:
      out_tensor: an output tensor of shape [128,512]
  """

  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Generate indices for the input/output tensors
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]

  # Load input data from HBM to on-chip memory
  in_tile = nl.load(in_tensor[i_p, i_f])

  # perform the computation:
  out_tile = nl.exp(in_tile)

  # store the results back to HBM
  nl.store(out_tensor[i_p, i_f], value=out_tile)

  return out_tensor
```

## rmsnorm_jax.py

```python
import jax
import jax.numpy as jnp
from rmsnorm_nki_kernels import nki_rmsnorm_kernel

def jax_rms_norm(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = jnp.square(a_tensor)
  # Calculate means in the free dimension
  mean = in_square.mean(axis=1, keepdims=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * jnp.reciprocal(jnp.sqrt(mean))

  # Scale the output by the weight
  return tensor * g_tensor
```

## unet_compile.py

```python
import torch
import neuronperf as npf
import neuronperf.torch

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

npf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=filename,
    model_name=model_name,
)
```

## average_pool2d_torch.py

```python
import torch
from torch_xla.core import xla_model as xm
from average_pool2d_nki_kernels import tensor_avgpool_kernel

device = xm.xla_device()

POOL_SIZE = 2
C, HIN, WIN = 2, 6, 6
HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

in_tensor = torch.arange(C * HIN * WIN, dtype=torch.bfloat16).reshape(C, HIN, WIN).to(device=device)
out_nki = tensor_avgpool_kernel(in_tensor, POOL_SIZE)
out_torch = torch.nn.functional.avg_pool2d(in_tensor, POOL_SIZE, POOL_SIZE)
```

## test_nki_nl_dslice.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit(mode="simulation")
def example_kernel(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  for i in nl.affine_range(in_tensor.shape[1] // 512):
    tile = nl.load(in_tensor[:, (i * 512):((i + 1) * 512)])
    # Same as above but use ds (dynamic slice) instead of the native
    # slice syntax
    tile = nl.load(in_tensor[:, nl.ds(i * 512, 512)])
    nl.store(out_tensor[:, nl.ds(i * 512, 512)], tile)

  return out_tensor
```

## torch-neuron-dataparallel-example-specify-ncs.rst

```python
import torch
import torch_neuron
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch.neuron.trace(model, image)

# Create the DataParallel module, run on the first three NeuronCores
# Equivalent to model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1, 2])
model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=['nc:0', 'nc:1', 'nc:2'])

# Create a batched input
batch_size = 5
image_batched = torch.rand([batch_size, 3, 224, 224])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## test_nki_memory_semantics.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode='simulation')
def simple_demo_kernel(a_ptr):
  
  B, N, M = a_ptr.shape

  a_loaded = nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.sbuf)
  exp_out =  nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.sbuf)
  out_ptr = nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.shared_hbm)

  for b in nl.affine_range(B):
    a_loaded[b] = nl.load(a_ptr[b])
    exp_out[b] = nl.exp(a_loaded[b])
    nl.store(out_ptr[b], value=exp_out[b])

  return out_ptr
```

## torch-neuronx-dataparallel-example-specify-ncs.rst

```python
import torch
import torch_neuronx
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch_neuronx.trace(model, image)

# Create the DataParallel module, run on the first two NeuronCores
# Equivalent to model_parallel = torch.neuron.DataParallel(model_neuron, device_ids=[0, 1])
model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=['nc:0', 'nc:1'])

# Create a batched input
batch_size = 5
image_batched = torch.rand([batch_size, 3, 224, 224])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## torch-neuronx-dataparallel-example-dynamic-batching.rst

```python
import torch
import torch_neuronx
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch_neuronx.trace(model, image)

# Create the DataParallel module
model_parallel = torch_neuronx.DataParallel(model_neuron)

# Create batched inputs and run inference on the same model
batch_sizes = [2, 3, 4, 5, 6]
for batch_size in batch_sizes:
    image_batched = torch.rand([batch_size, 3, 224, 224])
    
    # Run inference with a batched input
    output = model_parallel(image_batched)
```

## torch-neuron-dataparallel-example-dynamic-batching.rst

```python
import torch
import torch_neuron
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch.neuron.trace(model, image)

# Create the DataParallel module
model_parallel = torch.neuron.DataParallel(model_neuron)

# Create batched inputs and run inference on the same model
batch_sizes = [2, 3, 4, 5, 6]
for batch_size in batch_sizes:
    image_batched = torch.rand([batch_size, 3, 224, 224])
    
    # Run inference with a batched input
    output = model_parallel(image_batched)
```

## spmd_multiple_nc_tensor_addition_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.rand((512, 2048), dtype=torch.bfloat16).to(device=device)
b = torch.rand((512, 2048), dtype=torch.bfloat16).to(device=device)

output_nki = nki_tensor_add_nc2(a, b)
output_torch = a + b

allclose = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
```

## layout-loop.py

```python
import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit

@nki_jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of shape [256,512]
  Returns:
      out_tensor: an output tensor of shape [256,512]
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  i_f = nl.arange(512)[None, :]

  for k in nl.affine_range(2):
    # Generate tensor indices for the input/output tensors
    i_p = k * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

    # Load input data from HBM to on-chip memory
    in_tile = nl.load(in_tensor[i_p, i_f])

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to HBM
    nl.store(out_tensor[i_p, i_f], value=out_tile)

  return out_tensor
```

## spmd_multiple_nc_tensor_addition_jax.py

```python
import jax
import jax.numpy as jnp

from spmd_multiple_nc_tensor_addition_nki_kernels import nki_tensor_add_nc2

seed_a, seed_b = jax.random.split(jax.random.PRNGKey(42))
a = jax.random.uniform(seed_a, (512, 2048), dtype=jnp.bfloat16)
b = jax.random.uniform(seed_b, (512, 2048), dtype=jnp.bfloat16)

output_nki = nki_tensor_add_nc2(a, b)
output_jax = a + b

allclose = jnp.allclose(output_jax, output_nki, atol=1e-4, rtol=1e-2)
```

## resnet_compile.py

```python
import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

model = getattr(torchvision.models, "resnet18")(pretrained=True)
inputs = [get_batch(batch_size) for batch_size in [1, 8, 64]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1, 8, 64],
    pipeline_sizes=[1],
    filename="resnet18.json",
    model_name="resnet18",
)
```

## transpose2d_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

P, X, Y = 5, 3, 4
a = torch.arange(P*X*Y, dtype=torch.int8).reshape((P, X*Y)).to(device=device)
a_t_nki = torch.zeros((P, Y*X), dtype=torch.int8).to(device=device)

a_t_nki = tensor_transpose2D_kernel_(a, (X, Y))

a_t_torch = torch.transpose(a.reshape(P, X, Y), 1, 2).reshape(P, X * Y)
```

## spmd_tensor_addition_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)
b = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)

output_nki = nki_tensor_add(a, b)
output_torch = a + b

allclose = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
```

## vgg_compile.py

```python
import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

model = getattr(torchvision.models, "vgg11")(pretrained=True)
inputs = [get_batch(batch_size) for batch_size in [1, 8, 64]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1, 8, 64],
    pipeline_sizes=[1],
    filename="vgg11.json",
    model_name="vgg11",
)
```

## spmd_tensor_addition_jax.py

```python
import jax
import jax.numpy as jnp

from spmd_tensor_addition_nki_kernels import nki_tensor_add

seed_a, seed_b = jax.random.split(jax.random.PRNGKey(42))
a = jax.random.uniform(seed_a, (256, 1024), dtype=jnp.bfloat16)
b = jax.random.uniform(seed_b, (256, 1024), dtype=jnp.bfloat16)

output_nki = nki_tensor_add(a, b)
output_jax = a + b

allclose = jnp.allclose(output_jax, output_nki, atol=1e-4, rtol=1e-2)
```

## torch-neuronx-dataparallel-example-default.rst

```python
import torch
import torch_neuronx
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch_neuronx.trace(model, image)

# Create the DataParallel module
model_parallel = torch_neuronx.DataParallel(model_neuron)

# Create a batched input
batch_size = 5
image_batched = torch.rand([batch_size, 3, 224, 224])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## torch-neuron-dataparallel-example-default.rst

```python
import torch
import torch_neuron
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
model_neuron = torch.neuron.trace(model, image)

# Create the DataParallel module
model_parallel = torch.neuron.DataParallel(model_neuron)

# Create a batched input
batch_size = 5
image_batched = torch.rand([batch_size, 3, 224, 224])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## test_nki_simulate_kernel.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit
def print_kernel(a_tensor):
  b = nl.empty_like(a_tensor, buffer=nl.hbm)

  # Load tensor into sbuf
  a = nl.load(a_tensor)

  # Print tensor y
  nl.device_print("value of a:", a)

  # Directly store a into hbm
  nl.store(b, value=a)

  return b
```

```python
np.random.seed(0)
a = np.random.random_sample([3, 4]).astype(np.float32) * 10

b = nki.simulate_kernel(print_kernel, a)

assert np.allclose(a, b)
```

## resnet50_compile.py

```python
import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

model = torchvision.models.resnet50(pretrained=True)
inputs = [get_batch(batch_size) for batch_size in batch_sizes]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=filename,
    model_name=model_name,
)
```

## transpose2d_jax.py

```python
import jax
import jax.numpy as jnp

from transpose2d_nki_kernels import tensor_transpose2D_kernel_

P, X, Y = 5, 37, 44
a = jax.random.uniform(jax.random.PRNGKey(42), (P, X * Y))
a_t_nki = tensor_transpose2D_kernel_(a, shape2D=(X, Y))

a_t_jax = jnp.transpose(a.reshape(P, X, Y), axes=(0, 2, 1)).reshape(P, X * Y)
```

## test_resnet50_pt.py

```python
import torch
import torch_neuron

import neuronperf as npf
import neuronperf.torch

from torchvision import models


# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Construct example inputs
batch_sizes = [5, 6, 7]
inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32) for batch_size in batch_sizes]

# Compile
npf.torch.compile(
	model, 
	inputs, 
	batch_sizes=batch_sizes, 
	filename='resnet50.json',
)
```

## test_simple_pt.py

```python
import torch
import torch.neuron

class Model(torch.nn.Module):
    def forward(self, x):
        x = x * 3
        return x + 1

model = Model()
model.eval()

batch_sizes = [1]
inputs = [torch.ones((batch_size, 3, 224, 224)) for batch_size in batch_sizes]

model_neuron = torch.neuron.trace(model, inputs)
model_neuron.save("model_neuron_b1.pt")
```

## model.py

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(input_size, layers[0])
      self.fc2 = nn.Linear(layers[0], layers[1])
      self.fc3 = nn.Linear(layers[1], output_size)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return F.log_softmax(x, dim=1)
```