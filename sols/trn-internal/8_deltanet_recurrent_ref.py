"""NKI kernel for DeltaNet gated delta rule recurrent forward.

NKI v2 (SDK 2.28). Processes a SINGLE (batch, head) pair per kernel call.
The caller loops over (B, H) in PyTorch and calls this kernel for each pair.

Input layout: All inputs are 2D contiguous tensors (S, 128).
Each call processes one (batch, head) element's full sequence.

k_dim = v_dim = 128, which matches SBUF tile partition dimension exactly.
g and beta are scalars per token, expanded to (S, 128) by the caller.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width)
P_MAX = 128


@nki.jit
def test(
    query: nl.ndarray,  # (S, 128) float32
    key: nl.ndarray,  # (S, 128) float32
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 128) float32, log-decay broadcast to 128
    beta_in: nl.ndarray,  # (S, 128) float32, write gate broadcast to 128
) -> nl.ndarray:
    """NKI kernel for DeltaNet recurrent forward -- single (batch, head).

    Iterates over sequence tokens with sequential_range.
    State matrix (128 x 128) lives in SBUF.

    Args:
        query:    (S, 128) float32
        key:      (S, 128) float32
        value:    (S, 128) float32
        g_in:     (S, 128) float32
        beta_in:  (S, 128) float32

    Returns:
        output:   (S, 128) float32
    """
    seq_len, dim = query.shape

    # Output tensor in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)

    # Stride: for 2D (S, D), dim0 stride = D=128, dim1 stride = 1
    seq_stride = dim

    # Initialize recurrent state in SBUF: (128, 128)
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Sequential loop over tokens (state-dependent)
    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        # ---- Load inputs for token t ----
        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_t,
            src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_t,
            src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_t,
            src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_t,
            src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t,
            src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # ---- Step 1: Decay state -- state = state * exp(g_t) ----
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_g,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # ---- Step 2: Read memory -- kv_mem = state^T @ k_t ----
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_t)
        kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # ---- Step 3: delta = (v_t - kv_mem) * beta_t ----
        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=v_sub,
            op0=nl.multiply,
            operand0=beta_t,
            engine=nisa.vector_engine,
        )

        # ---- Step 4: state += outer(k_t, delta) ----
        # Compute outer product using nc_matmul with P=1 contraction.
        # nc_matmul computes: dst = stationary.T @ moving
        # stationary = k_row (1, 128): P=1, F_s=128 (k values along free dim)
        # moving = delta_row (1, 128): P=1, F_m=128 (delta values along free dim)
        # dst = (128, 1) @ (1, 128) = (128, 128)  -- this IS the outer product!

        # Transpose k_t (128, 1) -> k_row (1, 128)
        k_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_row_psum, data=k_t)
        k_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_row, src=k_row_psum)

        # Transpose delta (128, 1) -> delta_row (1, 128)
        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)
        delta_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row, src=delta_row_psum)

        # outer product: k_row.T @ delta_row = (128,128) in PSUM
        outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_psum, stationary=k_row, moving=delta_row)

        # Copy outer product from PSUM to SBUF
        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=outer_prod, src=outer_psum)

        # Accumulate into state
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- Step 5: o_t = state^T @ q_t ----
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        # ---- Store output for token t ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    return output
