## attention_kernels.py

```python
import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki.language import par_dim

# v1: toy example with 128 seqlen and nki.lang APIs
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


# v2: use nki.isa APIs
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


# v3: large sequence length with tiling
@nki.jit
def attn_fwd_v3(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # Tile along seqlen_q #
    qk = nl.ndarray((seqlen_q // PMAX, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                     dtype=nl.float32, buffer=nl.psum)
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

    # Softmax #
    row_max = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_q, i_tile_kv], axis=1)
        row_max[:, i_tile_q, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

    # subtract max from row
    norm_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv),
                       dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        norm_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_buf[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_q, i_tile_kv],
                op0=nl.subtract,
                operand0=row_max[:, i_tile_q, :],
                engine=nisa.vector_engine)
        nl.store(norm_row[i_tile_q], norm_buf[:,:])

    # exponentiation
    exp_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        exp_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        norm_buf = nl.load(norm_row[i_tile_q])
        exp_buf[:,:] = nisa.activation(op=nl.exp, data=norm_buf)
        nl.store(exp_row[i_tile_q], exp_buf[:,:])

    # sum of exp results
    sum_row = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        exp_buf = nl.load(exp_row[i_tile_q])
        sum_row[:, i_tile_q] = nisa.tensor_reduce(op=nl.add,
                                                         data=exp_buf,
                                                         axis=1)

    # reciprocal of sum_row
    inverse_sum_row = nisa.reciprocal(data=sum_row)
    
    scores = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        scores_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.load(exp_row[i_tile_q])
        scores_buf[:,:] = nisa.tensor_scalar(data=exp_buf,
                                               op0=nl.multiply,
                                               operand0=inverse_sum_row[:, i_tile_q],
                                               engine=nisa.vector_engine,
                                               dtype=nl.float32)
        nl.store(scores[i_tile_q], scores_buf[:,:])
        
    # v has the wrong layout
    v_t = nl.ndarray((seqlen_kv // PMAX, PMAX, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), d_head), dtype=nl.float32, buffer=nl.sbuf)
        v_sbuf_t[:, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)
        nl.store(v_t[i_tile_kv], v_sbuf_t[:,:])

    scores_t = nl.ndarray((seqlen_kv // PMAX, seqlen_q // PMAX, PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_buf = nl.load(scores[i_tile_q, :, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_psum_t = nisa.nc_transpose(scores_buf)
            scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
            scores_sbuf_t[:, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)
            nl.store(scores_t[i_tile_kv, i_tile_q, :, :], scores_sbuf_t)

    # scores @ V, contract along seqlen_kv
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        attn_out = nl.ndarray((nl.par_dim(PMAX), d_head),
                           dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_sbuf_t = nl.load(scores_t[i_tile_kv, i_tile_q, :, :])
            v_sbuf_t = nl.load(v_t[i_tile_kv, :, :])
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t,
                                            moving=v_sbuf_t)
        attn_out[:, :] = nisa.tensor_copy(attn_out_psum)
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:,:])

    return kernel_out


# v4: Loop fusion
@nki.jit
def attn_fwd_v4(q, k, v):
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
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

        # subtract max from row
        norm_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max,
                engine=nisa.vector_engine)

        # exponentiation
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            sum_row_kv[:, i_tile_kv] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=1)

        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_kv, axis=1)

        # reciprocal of sum_row
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        scores = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            scores[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)],
                op0=nl.multiply,
                operand0=inverse_sum_row,
                engine=nisa.vector_engine)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(scores[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out[...] = nisa.tensor_copy(attn_out_psum)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


# v5: softmax division delay
@nki.jit
def attn_fwd_v5(q, k, v):
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
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

        # subtract max from row
        norm_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max,
                engine=nisa.vector_engine)

        # exponentiation
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            sum_row_kv[:, i_tile_kv] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=1)

        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_kv, axis=1)

        # reciprocal of sum_row
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        # division is done on the final attention output
        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


# v6: instruction combination on ScalarE
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
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        
        # Leverage scalar engine's hardware capability of applying reduce after activation
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

        # reciprocal of sum_row
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


# v7: Downcast scores before transpose
@nki.jit
def attn_fwd_v7(q, k, v):
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
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.bfloat16)

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


# v8: Use tensor_scalar_reduce on VectorE
@nki.jit
def attn_fwd_v8(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            qk_sbuf[:, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_kv], op0=nl.multiply, operand0=1.0,
                                                                    reduce_op=nl.max, reduce_res=row_max_kv[:, i_tile_kv])
                                                                    
        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.bfloat16)

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


# v8a: refactor v8 to prepare for direct allocation and software pipelining
@nki.jit
def attn_fwd_v8a(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=v.dtype)
    identity_load = nl.ndarray((par_dim(128), 128), dtype=v.dtype,
                               buffer=nl.sbuf)
    identity_load[...] = nl.load(identity)

    identity_bf16 = nisa.tensor_copy(identity_load, dtype=nl.bfloat16)

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_sbuf = nl.ndarray((d_head, seqlen_q), dtype=q.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=k.dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=v.dtype, buffer=nl.sbuf)

    # load inputs into SBUF:
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)
    v_sbuf[...] = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16,
                          buffer=nl.sbuf)
    v_psum_t = nl.ndarray((seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                          buffer=nl.psum)

    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t[i_tile_kv] = nisa.nc_matmul(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_load,
                                             is_transpose=True, is_moving_onezero=True)
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)

    num_tile_q = seqlen_q // PMAX
    qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING),
                dtype=nl.float32, buffer=nl.sbuf)
    row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)

    exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                        dtype=nl.bfloat16, buffer=nl.sbuf)
    
    sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                                 buffer=nl.sbuf)

    scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
    attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                          dtype=nl.float32, buffer=nl.sbuf)

    qk = nl.ndarray((num_tile_q, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                    dtype=nl.float32, buffer=nl.psum)
    scores_psum_t = nl.ndarray((num_tile_q, seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX),
                               dtype=nl.float32, buffer=nl.psum)
    attn_out_psum = nl.ndarray((num_tile_q, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=nl.psum)

    attn_out_sbuf = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)

    return kernel_out
```

## kernel-optimization.rst

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn2")
def matrix_multiply_kernel(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation on a single tile

  Args:
    lhsT: an input tensor of shape [K,M], where both K and M are, at most, 
      128.  It is the left-hand-side argument of the matrix multiplication,
      delivered transposed for optimal performance.
    rhs: an input tensor of shape [K,N], where K is, at most, 128, and N
      is, at most, 512.  It is the right-hand-side argument of the matrix
      multiplication.
  Returns:
    result: the resulting output tensor of shape [M,N]
  """
  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape

  # Ensure that the contraction dimension matches
  assert K == K_, \
    f"Contraction demention {K} does not match {K_}, did you remember to transpose?"

  # Ensure the dimensions will fit within the constrins of matmul.
  assert K <= nl.tile_size.pmax, \
    f"Expected partition dimension in lhsT ({K}) to be less than {nl.tile_size.pmax}"
  assert M <= nl.tile_size.gemm_stationary_fmax, \
    f"Expected free dimension in lhsT ({M}) to be less than " \
    f"{nl.tile_size.gemm_stationary_fmax}"
  assert N <= nl.tile_size.gemm_moving_fmax, \
    f"Expected free dimension in rhs ({N}) to be less than " \
    f"{nl.tile_size.gemm_moving_fmax}"

  # Allocate tiles for lhsT and rhs on sbuf (uninitialized)
  lhsT_tile = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  # Copy the input matrices from HBM to SBUF
  nisa.dma_copy(dst=lhsT_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  # Perform matrix multiply, result will be written into PSUM
  result_tile = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=result_tile, stationary=lhsT_tile, moving=rhs_tile)

  # Copy result to SBUF (we cannot copy directly from PSUM to HBM)
  result_tmp = nl.ndarray(shape=result_tile.shape,
                          dtype=result_tile.dtype,
                          buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_tmp, src=result_tile)

  # Copy result to HBM
  result = nl.ndarray(shape=result_tmp.shape,
                      dtype=result_tmp.dtype,
                      buffer=nl.hbm)
  nisa.dma_copy(dst=result, src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn2")
def matrix_multiply_kernel(lhsT, rhs):
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

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
 
  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512
 
  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"
 
  # Create a space for the result in HBM (uninitialized)
  result = nl.ndarray(shape=(M, N), dtype=lhsT.dtype, buffer=nl.hbm)
 
  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM (uninitialized)
      result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                           dtype=nl.float32,
                           buffer=nl.psum)
 
      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF (uninitialized)
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                           dtype=lhsT.dtype,
                           buffer=nl.sbuf)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                          dtype=rhs.dtype,
                          buffer=nl.sbuf)
 
        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile, 
                  src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                           m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile,
                  src=rhs[k * TILE_K:(k + 1) * TILE_K,
                          n * TILE_N:(n + 1) * TILE_N])
 
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=result_tile, stationary=lhsT_tile, moving=rhs_tile)
 
      # Copy the result from PSUM back to SBUF, and cast to expected
      # output data-type
      result_tmp = nl.ndarray(shape=(TILE_M, TILE_N),
                          dtype=nl.float32,
                          buffer=nl.sbuf)
      nisa.tensor_copy(dst=result_tmp, src=result_tile)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                           n * TILE_N:(n + 1) * TILE_N],
                src=result_tmp)
 
  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn2")
def matrix_multiply_kernel(lhsT, rhs):
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

  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.hbm)

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_M numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      # Allocate space in SBUF for the tile (uninitialized)
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                           dtype=lhsT.dtype,
                           buffer=nl.sbuf)
      # Copy the tile from HBM to SBUF
      nisa.dma_copy(dst=lhsT_tile, 
                  src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                         m * TILE_M:(m + 1) * TILE_M])
      # Append the tile to the list of tiles.
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      # Load a whole column tiles from rhs (with K * TILE_N numbers)
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                          dtype=rhs.dtype,
                          buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=rhs_tile,
                  src=rhs[k * TILE_K:(k + 1) * TILE_K,
                          n * TILE_N:(n + 1) * TILE_N])
        # Append the tile to the list of tiles.
        rhs_tiles.append(rhs_tile)

      # Allocate a tile in PSUM for the result
      result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                           dtype=nl.float32,
                           buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=result_tile,
                   stationary=lhsT_tiles[k],
                   moving=rhs_tiles[k])

      # Copy the result from PSUM back to SBUF, and cast to expected
      # output data-type
      result_tmp = nl.ndarray(shape=(TILE_M, TILE_N),
                          dtype=nl.float32,
                          buffer=nl.sbuf)
      nisa.tensor_copy(dst=result_tmp, src=result_tile)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                           n * TILE_N:(n + 1) * TILE_N],
                src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn2")
def matrix_multiply_kernel(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation while blocking the
     free dimensions of the LHS and RHS to improve memory access pattern.
  
  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        1.    It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """
  
  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  
  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512
  
  # Configuring the blocking size for the free dimensions
  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2
  
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024
  
  # the size has to be multiple of block size
  assert M % BLOCK_M == 0, f"Expected M ({M}) to be divisible by BLOCK_M ({BLOCK_M})"
  assert N % BLOCK_N == 0, f"Expected N ({N}) to be divisible by BLOCK_N ({BLOCK_N})"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray(shape=(M, N), dtype=lhsT.dtype, buffer=nl.hbm)
  
  # Loop over blocks over the M dimension
  for m in nl.affine_range(M // BLOCK_M):
    # Load TILES_IN_BLOCK_M columns tiles from lhsT
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      # Inner tile array.
      lhsT_tiles_internal = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                               dtype=lhsT.dtype,
                               buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=lhsT_tile,
                src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                     (m * TILES_IN_BLOCK_M + bm) *
                     TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                     TILE_M])
        # Append the tile to the inner list of tiles.
        lhsT_tiles_internal.append(lhsT_tile)
      # Append the inner list of tiles into the outer list of tiles.
      lhsT_tiles.append(lhsT_tiles_internal)
  
    for n in nl.affine_range(N // BLOCK_N):
      # Load TILES_IN_BLOCK_N columns from rhs
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        # Inner tile array.
        rhs_tiles_internal = []
        for k in nl.affine_range(K // TILE_K):
          # Allocate space in SBUF for the tile (uninitialized)
          rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          # Copy the tile from HBM to SBUF
          nisa.dma_copy(dst=rhs_tile,
                src=rhs[k * TILE_K:(k + 1) * TILE_K,
                    (n * TILES_IN_BLOCK_N + bn) *
                    TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                    TILE_N])
          # Append the tile to the inner list of tiles.
          rhs_tiles_internal.append(rhs_tile)
        # Append the inner list of tiles into the outer list of tiles.
        rhs_tiles.append(rhs_tiles_internal)
  
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          # Allocate a tensor in PSUM
          result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                   dtype=nl.float32,
                                   buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            # Accumulate partial-sums into PSUM
            nisa.nc_matmul(dst=result_tile,
                           stationary=lhsT_tiles[bm][k],
                           moving=rhs_tiles[bn][k])
  
          # Copy the result from PSUM back to SBUF, and cast to expected
          # output data-type
          result_tmp = nl.ndarray(shape=result_tile.shape,
                                  dtype=result.dtype,
                                  buffer=nl.sbuf)
          nisa.tensor_copy(dst=result_tmp, src=result_tile)

          # Copy the result from SBUF to HBM.
          nisa.dma_copy(dst=result[(m * TILES_IN_BLOCK_M + bm) *
                                   TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                   TILE_N],
                        src=result_tmp)
  
  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn2")
def matrix_multiply_kernel(
    lhsT,
    rhs,
    # Meta-parameters
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

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Compute the block dimensions.
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray(shape=(M,N), dtype=nl.float32, buffer=nl.hbm)

  # Compute the number of blocks in each dimension
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    # Create the initial result tiles in SBUF and initialize each tile to
    # 0.0, since the final results will be accumulated here.
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          # Create the result tile (uninitialized)
          tile = nl.ndarray(shape=(TILE_M, TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)
          # Initialize the tile 0.0
          nisa.memset(dst=tile, value=0.0)
          # Append the tile to block_n array.
          block_n.append(tile)
        # Append block_n array to block_m array.
        block_m.append(block_n)
      # Append block_m array into result_tmps.
      result_tmps.append(block_m)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler to
    # change this loop by, for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs setting the load tile to
      # `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        # Allocate rhs_tile tensor, TILE_K x BLOCK_N
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        # Copy block tile from rhs, to rhs_tile.
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        # Append rhs_tile to rhs_tiles.
        rhs_tiles.append(rhs_tile)

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          # Allocate lhsT_tile tensor, BLOCK_M x TILE_K
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          # Copy block tile from lhsT, to lhsT_tile.
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          # Append lhsT_tile to lhsT_tiles.
          lhsT_tiles.append(lhsT_tile)

        # Compute the result tiles
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
          for bn in nl.affine_range(TILES_IN_BLOCK_N):
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              # Allocate a tensor in PSUM
              result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                       dtype=nl.float32,
                                       buffer=nl.psum)
              # Perform matrix multiply
              nisa.nc_matmul(dst=result_tile,
                             stationary=lhsT_tiles[bk][0:TILE_K,
                                                       bm * TILE_M:(bm + 1) * TILE_M],
                             moving=rhs_tiles[bk][0:TILE_K,
                                                  bn * TILE_N:(bn + 1) * TILE_N])

              # Add result to accumulated result
              nisa.tensor_tensor_add(dst=result_tmps[m][bm][bn],
                                     src0=result_tmps[m][bm][bn],
                                     src1=result_tile)

    # Write results to HBM
    for m_idx in nl.affine_range(NUM_BLOCK_M):
      for bm_idx in nl.affine_range(TILES_IN_BLOCK_M):
        for bn_idx in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.dma_copy(dst=result[(m_idx * TILES_IN_BLOCK_M + bm_idx) *
                                   TILE_M:((m_idx * TILES_IN_BLOCK_M + bm_idx) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn_idx) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn_idx) + 1) *
                                   TILE_N],
                        src=result_tmps[m_idx][bm_idx][bn_idx])

  return result
```

## rcnn-app-note.rst

```python
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_model():

    # Configure the R-CNN model
    CONFIG_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    WEIGHTS_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHTS_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cpu'  # Send to CPU for Neuron Tracing

    # Create the R-CNN predictor wrapper
    predictor = DefaultPredictor(cfg)
    return predictor
```

```python
import torch
import torch_neuron 

example = torch.rand([1, 3, 800, 800])

# Use `with torch.no_grad():` to avoid a jit tracing issue in the ResNet backbone
with torch.no_grad():
    neuron_backbone = torch_neuron.trace(predictor.model.backbone, example, strict=False)

backbone_filename = 'backbone.pt'
torch.jit.save(neuron_backbone, backbone_filename)
```

```python
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch.jit import ScriptModule

class NeuronRCNN(torch.nn.Module):
    """
    Creates a `NeuronRCNN` wrapper that injects the compiled backbone into
    the R-CNN model. It also stores the `size_divisibility` attribute from
    the original backbone.
    """

    def __init__(self, model: GeneralizedRCNN, neuron_backbone: ScriptModule) -> None:
        super().__init__()

        # Keep track of the backbone variables
        size_divisibility = model.backbone.size_divisibility

        # Load and inject the compiled backbone
        model.backbone = neuron_backbone

        # Set backbone variables
        setattr(model.backbone, 'size_divisibility', size_divisibility)

        self.model = model

    def forward(self, x):
        return self.model(x)
```

```python
def preprocess(original_image, predictor):
    """
    A basic preprocessing function that sets the input height=800 and 
    input width=800. The function is derived from the preprocessing
    steps in the Detectron2 `DefaultPredictor` module.
    """

    height, width = original_image.shape[:2]
    resize_func = predictor.aug.get_transform(original_image)
    resize_func.new_h = 800 # Override height
    resize_func.new_w = 800 # Override width
    image = resize_func.apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    return inputs
```

```python
import math

input_shape = [1, 3, 800, 800] # Overall input shape at inference time

# Create the list example of RPN inputs using the resizing logic from the RPN Head
features = list()
for i in [0, 1, 2, 3, 4]:
    ratio = 1 / (4 * 2**i)
    x_i_h = math.ceil(input_shape[2] * ratio)
    x_i_w = math.ceil(input_shape[3] * ratio)
    feature = torch.zeros(1, 256, x_i_h, x_i_w)
    features.append(feature)

# Extract and compile the RPN Head
neuron_rpn_head = torch_neuron.trace(predictor.model.proposal_generator.rpn_head, [features])
rpn_head_filename = 'rpn_head.pt'
torch.jit.save(neuron_rpn_head, rpn_head_filename)
```

```python
class NeuronFusedBackboneRPNHead(torch.nn.Module):
    """
    Wrapper to compile the fused ResNet backbone and RPN Head.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.rpn_head = model.proposal_generator.rpn_head
        self.in_features = model.proposal_generator.in_features

    def forward(self, x):
        features = self.backbone(x)
        features_ = [features[f] for f in self.in_features]
        return self.rpn_head(features_), features
```

```python
# Create the wrapper with the combined backbone and RPN Head
predictor = get_model()
backbone_rpn_wrapper = NeuronFusedBackboneRPNHead(predictor.model)
backbone_rpn_wrapper.eval()

# Compile the wrapper
example = torch.rand([1, 3, 800, 800])

with torch.no_grad():
    neuron_backbone_rpn_head = torch_neuron.trace(
        backbone_rpn_wrapper, example, strict=False)

backbone_rpn_filename = 'backbone_rpn.pt'
torch.jit.save(neuron_backbone_rpn_head, backbone_rpn_filename)
```

```python
class BackboneRPN(torch.nn.Module):
    """
    Wrapper that uses the compiled `neuron_backbone_rpn` instead
    of the original backbone and RPN Head. We copy the remainder
    of the RPN `forward` code (`predictor.model.proposal_generator.forward`)
    to create a "fused" backbone + RPN module.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.backbone_rpn_head = NeuronFusedBackboneRPNHead(model)
        self._rpn = model.proposal_generator
        self.in_features = model.proposal_generator.in_features

    def forward(self, images):
        preds, features = self.backbone_rpn_head(images.tensor)
        features_ = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = preds
        anchors = self._rpn.anchor_generator(features_)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self._rpn.anchor_generator.box_dim,
                   x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        proposals = self._rpn.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, features
```

```python
class NeuronRCNN(torch.nn.Module):
    """
    Wrapper that uses the fused backbone + RPN module and re-writes
    the rest of the R-CNN `model` `forward` function.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()

        # Use the fused Backbone + RPN
        self.backbone_rpn = BackboneRPN(model)

        self.roi_heads = model.roi_heads

        self.preprocess_image = model.preprocess_image
        self._postprocess = model._postprocess

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        proposals, features = self.backbone_rpn(images)
        results, _ = self.roi_heads(images, features, proposals, None)
        return self._postprocess(results, batched_inputs, images.image_sizes)
```

```python
class NeuronBoxHeadBoxPredictor(torch.nn.Module):
    """
    Wrapper that extracts the RoI Box Head and Box Predictor
    for compilation.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.roi_heads = model.roi_heads

    def forward(self, box_features):
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        return predictions
```

```python
# Create the NeuronBoxHeadBoxPredictor wrapper
predictor = get_model()
box_head_predictor = NeuronBoxHeadBoxPredictor(predictor.model)
box_head_predictor.eval()

# Compile the wrapper
example = torch.rand([1000, 256, 7, 7])
neuron_box_head_predictor = torch_neuron.trace(box_head_predictor, example)

roi_head_filename = 'box_head_predictor.pt'
torch.jit.save(neuron_box_head_predictor, roi_head_filename)
```

```python
class ROIHead(torch.nn.Module):
    """
    Wrapper that combines the compiled `roi_heads` into the
    rest of the RoI module. The `_forward_box` and `forward`
    functions are from the `predictor.model.roi_heads` module.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.roi_heads = model.roi_heads
        self.neuron_box_head_predictor = NeuronBoxHeadBoxPredictor(model)

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.roi_heads.box_in_features]
        box_features = self.roi_heads.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        predictions = self.neuron_box_head_predictor(box_features)
        pred_instances, _ = self.roi_heads.box_predictor.inference(
            predictions, proposals)
        return pred_instances

    def forward(self, images, features, proposals, targets=None):
        pred_instances = self._forward_box(features, proposals)
        pred_instances = self.roi_heads.forward_with_given_boxes(
            features, pred_instances)
        return pred_instances, {}
```

```python
from typing import Any, Union, Callable
import os

def compile(
    model: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    filename: str,
    **kwargs
) -> torch.nn.Module:
    """
    Compiles the model for Inf1 if it doesn't already exist and saves it as the provided filename. 
    
    model: A module or function which defines a torch model or computation.
    example_inputs: An example set of inputs which will be passed to the
        `model` during compilation.
    filename: Name of the compiled model
    kwargs: Extra `torch_neuron.trace` kwargs
    """

    if not os.path.exists(filename):
        with torch.no_grad():
            compiled_model = torch_neuron.trace(model, example_inputs, **kwargs)
        torch.jit.save(compiled_model, filename)
```

## fused_mamba.rst

```python
# Step 1: Element-wise multiplication of delta_i and A_i
deltaA_i = nisa.tensor_scalar(delta_i, op0=nl.multiply, operand0=A_i)
```

```python
# Step 1&2: nisa.activation
deltaA_i = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
```

```python
# Step 3: Element-wise multiplication of delta_i, B_i and u_i
deltaU_i = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
deltaBu_i = nisa.tensor_tensor(deltaU_i, B_i_bcast, op=ml.multiply)
```

```python
# Step 4: Associative scan between deltaA_i and deltaBu_i
scan_i = nl.ndarray((channels_tiled, seq_len), dtype=deltaA.dtype, buffer=nl.sbuf)
scan_i[0:channels_tiled, 0] = deltaBu[0:channels_tiled, 0]

for i in nl.sequential_range(seq_len - 1):
    scan_i[0:channels_tiled, i+1] = nisa.tensor_scalar(
        deltaA[0:channels_tiled, i+1],
        op0=nl.multiply,
        operand0=scan_i[0:channels_tiled, i],
        op1=nl.add,
        operand1=deltaBu[0:channels_tiled, i+1])
```

```python
# Step 4: Optimized associative scan using tensor_tensor_scan
scan_i = nisa.tensor_tensor_scan(deltaA_i, deltaBu_i, initial=0,
                                 op0=np.multiply, op1=np.add)
```

```python
# Step 5: Element-wise multiplication of C_i and scan_i
C_i_bcast = C_i.broadcast((nl.tile_size.pmax, seq_len))
scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
```

```python
# Step 6: Accumulation of scanC_i along state_size dimension
scanC_accum = nl.zeros(...)

for i_state in nl.affine_range(state_size):
    scanC_i = ...
    scanC_accum += scanC_i
```

```python
# Loop-carried dependency handling for seq_len tiling
scan_init = nl.zeros((channel_psize, 1), ...)

for i_seq_len_tile in static_range(seq_len // seq_len_fsize):
    scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                                     op0=np.multiply, op1=np.add)
    scan_init = scan_i[0:channel_psize, seq_len_fsize-1]
```

## wrapper.py

```python
import torch
import neuronx_distributed
import torch_xla.core.xla_model as xm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerCrossAttention
from transformers.generation.utils import ModelOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer

from optimum.neuron.generation import NeuronGenerationMixin

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation.utils import (
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
    GreedySearchOutput,
)


class T5Wrapper(T5ForConditionalGeneration, NeuronGenerationMixin):

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, 
        inputs_tensor: torch.Tensor, 
        model_kwargs, 
        model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        encoder = self.get_encoder()
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(inputs_tensor, model_kwargs["attention_mask"])
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    def _update_model_kwargs_for_xla_generation(
        self,
        model_kwargs: Dict[str, Any],
        batch_size: int,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        max_length: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        def _update_attention(model_kwargs, is_encoder_decoder):
            attention_mask_name = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
            attention_mask = model_kwargs.pop(attention_mask_name)
            attention_mask_update_slice = torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask[:, 1:], attention_mask_update_slice], dim=-1)
            mask = {attention_mask_name: attention_mask}
            return mask

        mask = _update_attention(model_kwargs, is_encoder_decoder)
        model_kwargs.update(mask)
        model_kwargs["past_key_values"] = torch.tensor([])

        return model_kwargs
    
    def _reorder_cache(self, past_key_values, beam_idx):
        self.beam_idx = beam_idx
        return past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        beam_scores = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        hidden_states = encoder_outputs["last_hidden_state"]

        if not hasattr(self, 'beam_idx'):
            num_beams = attention_mask.shape[0]
            self.beam_idx = torch.arange(0, num_beams, dtype=torch.int64)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            decoder_attention_mask,
            hidden_states,
            attention_mask,
            self.beam_idx,
            beam_scores
        )

        next_token_scores = decoder_outputs[0]
        next_tokens = decoder_outputs[1]
        next_indices = decoder_outputs[2]

        return next_token_scores, next_tokens, next_indices


class EncoderWrapper(torch.nn.Module):

    def __init__(self, 
                 encoder,
                 decoder, 
                 model_config, 
                 batch_size, 
                 max_length, 
                 device, 
                 num_beams,
                 tp_degree=None):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_config = model_config
        self.device = device
        self.num_beams = num_beams
        self.num_attention_heads_per_partition = model_config.num_heads
        self.tp_degree = tp_degree
        if self.tp_degree is not None:
            self.num_attention_heads_per_partition = model_config.num_heads // neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size()
            self.past_key_values_sa = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((self.num_beams,self.num_attention_heads_per_partition,self.max_length-1,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(model_config.num_decoder_layers * 2)])
            self.past_key_values_ca = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((self.num_beams,self.num_attention_heads_per_partition,self.max_length,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(model_config.num_decoder_layers * 2)])

    def forward(self, input_ids, attention_mask):
        encoder_output =  self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       output_attentions=False,
                                       output_hidden_states=False)

        last_hidden_state = encoder_output["last_hidden_state"]
        encoder_hidden_states = torch.concat([tensor.unsqueeze(0).repeat(self.num_beams, 1, 1) for tensor in last_hidden_state])

        decoder_blocks = self.decoder.block
        present_key_value_states_sa = []
        present_key_value_states_ca = []

        for i, block in enumerate(decoder_blocks):

            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                return states.view(self.batch_size, -1, self.num_attention_heads_per_partition, attention.key_value_proj_dim).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            if self.tp_degree is None:
                present_key_value_states_ca.append(key_states) 
                present_key_value_states_ca.append(value_states) 
                
                present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     
                                                                self.model_config.num_heads, 
                                                                self.max_length-1, 
                                                                self.model_config.d_kv), dtype=torch.float32, device=self.device)) 
                present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     
                                                                self.model_config.num_heads, 
                                                                self.max_length-1, 
                                                                self.model_config.d_kv), dtype=torch.float32, device=self.device))
            else:
                present_key_value_states_ca.append((self.past_key_values_ca[i*2] * 0) + key_states)
                present_key_value_states_ca.append((self.past_key_values_ca[i*2+1] * 0) + value_states)
                present_key_value_states_sa.append(self.past_key_values_sa[i*2]*torch.zeros((self.batch_size, self.num_attention_heads_per_partition, self.max_length-1, self.model_config.d_kv), dtype=torch.float32, device="xla"))
                present_key_value_states_sa.append(self.past_key_values_sa[i*2+1]*torch.zeros((self.batch_size, self.num_attention_heads_per_partition, self.max_length-1, self.model_config.d_kv), dtype=torch.float32, device="xla"))

        return present_key_value_states_sa + present_key_value_states_ca


class DecoderWrapper(torch.nn.Module):

    def __init__(self, 
                 decoder: T5Stack, 
                 lm_head: torch.nn.Linear,
                 model_config,
                 num_beams: int, 
                 max_length: int,
                 device: str,
                 tp_degree=None):
        super().__init__()        
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim=model_config.d_model
        self.device = device
        self.num_beams = num_beams
        self.batch_size = 1
        self.config = model_config

        num_heads=model_config.num_heads
        num_decoder_layers=model_config.num_decoder_layers

        self.num_attention_heads_per_partition = num_heads
        if tp_degree is not None:
            self.num_attention_heads_per_partition = num_heads // neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size()

        if device == "cpu":
            self.past_key_values_sa = [torch.ones((num_beams,num_heads,max_length-1,model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
            self.past_key_values_ca = [torch.ones((num_beams,num_heads,max_length,model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
        elif device == "xla":
            self.past_key_values_sa = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((num_beams,self.num_attention_heads_per_partition,max_length-1,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])
            self.past_key_values_ca = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((num_beams,self.num_attention_heads_per_partition,max_length,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])

    def update_past(self, past_key_values):
        new_past_sa = []
        new_past_ca = []
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past_sa += [new_past_layer[:2],]
            new_past_ca += [new_past_layer[2:],]
        return new_past_sa, new_past_ca
    
    def reorder_cache(self, past_key_values, beam_idx):
        for i in range(len(past_key_values)):
             past_key_values[i] = torch.index_select(past_key_values[i], 0, beam_idx)
        return past_key_values

    def forward(self,
                input_ids,
                decoder_attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                beam_idx,
                beam_scores,
                **kwargs):

        if self.num_beams > 1:
            past_key_values_sa = self.reorder_cache(self.past_key_values_sa, beam_idx)
            past_key_values_ca = self.reorder_cache(self.past_key_values_ca, beam_idx)
        else:
            past_key_values_sa = self.past_key_values_sa
            past_key_values_ca = self.past_key_values_ca

        past_key_values = [[*past_key_values_sa[i*2:i*2+2], *past_key_values_ca[i*2:i*2+2]] for i in range(0, int(len(past_key_values_ca)/2))]

        decoder_output = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False)

        last_hidden_state = decoder_output['last_hidden_state']
        past_key_values = decoder_output['past_key_values']

        if self.config.tie_word_embeddings:
            last_hidden_state = last_hidden_state * (self.model_dim**-0.5)

        lm_logits = self.lm_head(last_hidden_state)

        past_key_values_sa, past_key_values_ca = self.update_past(past_key_values)

        past_key_values_sa = [vec for kv_per_layer in past_key_values_sa for vec in kv_per_layer]
        past_key_values_ca = [vec for kv_per_layer in past_key_values_ca for vec in kv_per_layer]

        if self.device == "cpu":
            self.past_key_values_sa = past_key_values_sa
            self.past_key_values_ca = past_key_values_ca

        next_token_logits = lm_logits[:, -1, :]

        if self.num_beams > 1:
            logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
            logsumexp = torch.log(torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
            next_token_scores = next_token_logits - logit_max - logsumexp
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
            next_token_scores = next_token_scores * 1

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            ) 

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            return [next_token_scores, next_tokens, next_indices] + past_key_values_sa + past_key_values_ca
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            return [next_tokens] + past_key_values_sa + past_key_values_ca
```

## nki-language-guide.rst

```python
@nki.jit
def my_function(x : tensor, y : tensor) -> tensor:
  assert x.shape == y.shape, "expecting tensors of the same shape"
  assert x.dtype == y.dtype, "expecting tensors with the same element type"
  
  # allocate an output tensor for the result
  output = nki.language.ndarray(x.shape, x.dtype, buffer=sbuf)
  
  print(f"adding tensors of type {x.dtype} and {x.shape}")
  nki.isa.tensor_tensor(output, x, y, op=nki.langauge.add)
  return output
```

```python
l = [1,2,3]    # create a list with 3 elements 
l.append(4.1)  # append a value to the list
l.extend(("Hello", "List")) # extend list with multiple values
size = l.count() # return number of elements in list
third = l[2]  # get third element of list (index 2)

# search list for a specific value
if l.index(2):
  print("list contains 2")
   
# remove a specific value from a list (if present)
l.remove(1)

# print out list in reverse order
for x in l.reverse():
  print(x)
```

```python
d = dict() # create an empty dictionary
d['a'] = 1 # set a value in the dictionary

print(d.keys())  # print out keys in dictionary
print(d.items())  # print out values in dictionary

# print out dictionary
for k,v in d.values():
  print(k, v)

# remove value from dictionary if present
if d.pop('a'):
  print("removed 'a' from dictionary")

# fetch value of a, set to 1 if not present
a = d.setdefault('a', default=1)
```

```python
# assume t is a 3-dimensional tensor, we can iterate over the
# 2-D subtensors
for i in range(t.shape[0]):
  my_function(t[i])
```

```python
# A matrix of 128x128 16-bit float values in the SBUF memory
t = nl.ndarray((128,128), nl.float16, nl.sbuf)
assert t.shape = (128,128)
assert t.dtype == nl.float16
assert t.buffer == nl.sbuf
```

```python
# create an alternate view of t with shape 128x2x64
u = t.reshape((128,2,64))

# create an alternate view of t with shape 128x32
v = t.reshape((128,32))
```

```python
# create a memory region in the SBUF of size 128x64 bytes
region = sbuf.ptr(size=(128, 64))

# create a tensor of size 128x32 with float16 elementes
t = region.view(nl.float16, (128, 32))
```

```python
# equivalent to region.view above
t = nl.ndarray((128,32), nl.float16, buffer=region)
```

```python
# create a tensor at offset 128 bytes from the beginning of the SBUF memory.
region = sbuf.ptr(size=(128,64), offset=(0,128))
t = region.view(nl.float16, (128,32))
```

```python
region1 = sbuf.ptr(size=(128,64), offset=(0,0))
region2 = sbuf.ptr(size=(128,64), offset=(0,64))

t1 = region1.view(nl.float16, (128,32))
t2 = region2.view(nl.float16, (128,32))
```

```python
region = sbuf.ptr(size=(128,128))

region1 = region.ptr(size=(128,64), offset=(0,0))
region2 = region.ptr(size=(128,64), offset=(0,64))

t1 = region1.view(nl.float16, (128,32))
t2 = region2.view(nl.float16, (128,32))
```

```python
region2 = region.ptr(size=(128,64))

# t1 and t2 use the same underlying memory
t1 = region.view(nl.float16, (128,32))
t2 = region.view(nl.float16, (128,2,16))
```

```python
# this is just a short-hand
u = t.reshape(shape)

# for this
u = t.address.reshape(t.dtype, shape)
```

```python
# 10th element in partition 0
u = t[0,0,10]

# 65th element in partition 0
u = t[0,1,0]

# last element of the tensor
u = t[63,63,63]
```

```python
# All first 64 elements of every partition
u = t[0:64, 0, 0:64]

# Same as above, but using defaults
u = t[:, 0, :]

# Only the even elements of the third dimension
u = t[:, :, ::2]
```

```python
# the whole tensor t
u = t[...]

# same as above
u = t[:,...]

# use defaults for second dimension
# equivalent to t[0,0:64,0:64]
u = t[0,...,:]
```

```python
u = t[0,...]
assert u.shape = (64,64)

v = u[0:32, :]
assert v.shape = (32, 64)
```

```python
u = t[0,...]

# check hardware access pattern
print(u.offset)
print(u.pattern)
```

```python
# Specify HW access pattern directly
u = t.ap(offset = 0, pattern = [...])
```

```python
def kernel(outputs, inputs):
  for i in range(len(inputs)):
    if i % 2 == 0:
      nki.isa.nc_transpose(dst=outputs[i], data=inputs[i])
    else:
      nki.isa.reciprocal(dst=outputs[i], data=inputs[i])
```

```python
for i in sequential_range(...): ...
for i in static_range(...): ...
for i in affine_range(...): ...
```

```python
l = [1,2,3]
for x in l:
  print(x)

t = (1,2,3)
for x in t:
  print(x)
```

```python
# print the numbers 0-9
x = 0
while x < 10:
  print(x)
  x += 1
```

```python
# create a dynamic loop that runs "on chip"
for i in dynamic_range(10):
  process_tensor(t[i])
```

```python
count = nki.isa.register_alloc(count_tensor)
for i in dynamic_range(count):
  process_tensor(t[i])
```

```python
# allocate a new register with initial value
# either from constant integer, or a SBUF tensor
def register_alloc(x: int | tensor) -> register: ...

# store a constant integer into a register
def register_move(dst: imm: int): ...

# load a value from an SBUF tensor into a register
def register_load(dst: register, src: tensor): ...

# store the value of a register into an SBUF tensor
def register_store(dst: tensor, src: register): ...
```

```python
# suppose cond is an SBUF tensor, perhaps declared as
cond = nl.ndarray((1, 1), buffer=nl.shared_hbm, dtype=np.int32)

# allocate a register with initial value 1
reg = register_alloc(1)

# This while loop is dynamic because the condition is a register
while reg:
  # perform a calculation that updates cond
  ...
  nl.store(dst=cond[0], ...)
  # update register used in while-loop condition
  register_load(reg, cond)
```

```python
@dataclass 
class C(NKIObject):
  x : int
  y : bool = False
  
  def toggle(self):
    self.y = not self.y
    
c = C(1)
c.toggle()

# prints 1, True
print(c.x, c.y)
```

```python
# default if not provided by the user
def __init__(self, x = None, y = False):
  self.x = x
  self.y = y
  self.post_init()

# default if not provided by the user
def __post_init__(self):
  pass
```

```python
class A(NKIObject):
  x : int = 1
  def __init__(self, x):
    self.x = x

@nki.jit
def kernel(a : A): ...

kernel(A(1))
```

```python
# pseudo-code "copy constuct" A on NKI side
def kernel(python_a : A):
  # make a NKI instance of class A
  nki_a = new A
  # populate NKI instance from Python instance
  nki_a.__dict__ = python_a.__dict__
```

```python
class E(Enum):
  x = 1
  y = 2
  z = 3

def f(e : E):
  if e == E.x: ...
  else if e == E.y: ...
  else if e == E.z: ...
  
f(E.x)
```

```python
class E(NKIObject):
  x = E("x", 1)
  y = E("y", 2)
  z = E("z", 3)
  
  def __init__(self, name, value):
    self.name = name
    self.value = value
```

## onboarding-models.rst

```python
from neuronx_distributed_inference.models.config import NeuronConfig

class NeuronLlamaConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set any args/defaults
```

```python
from typing import List, Type
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

class LlamaInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]
        
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronLlamaConfig
```

```python
import torch
from typing import Optional, Tuple
from torch import nn
from transformers.activations import ACT2FN

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding

from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype

        self.tp_degree = parallel_state.get_tensor_model_parallel_size()

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.init_gqa_properties()
        self.init_rope()

    def init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronLlamaAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states, present_key_value = attn_outs
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value)


class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
        )

        self.layers = nn.ModuleList(
            [NeuronLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

```python
from neuronx_distributed_inference.models.llama import NeuronBaseForCausalLM

class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronLlamaModel
        
    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
```

```python
from neuronx_distributed_inference.utils.accuracy import generate_expected_logits, check_accuracy_logits_v2

# Init Neuron model, test inputs and HuggingFace generation config.

# Generating HuggingFace model outputs on CPU.
expected_logits = generate_expected_logits(
    neuron_model,
    inputs.input_ids,
    inputs.attention_mask,
    generation_config
)
# Alternatively, you can load the expected_logits from disk to save time.
# expected_logits = ...

check_accuracy_logits_v2(
    neuron_model,
    expected_logits,
    inputs.input_ids,
    inputs.attention_mask,
    generation_config=generation_config
)
```

```python
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits

# Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.

check_accuracy_logits(
    model,
    tokenizer,
    generation_config,
)
```

```python
from neuronx_distributed_inference.utils.accuracy import check_accuracy

# Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.

check_accuracy(
    model,
    tokenizer,
    generation_config,
)
```

```python
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

# Init Neuron model and HuggingFace generation config.

benchmark_sampling(model, generation_config)
```

```python
import logging

logging.getLogger().setLevel(logging.DEBUG)
```

## pytorch-neuron-debug.rst

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

device = xm.xla_device()
input1 = torch.randn(2,10).to(device)
# Defining 2 linear layers
linear1 = torch.nn.Linear(10,30).to(device)
linear2 = torch.nn.Linear(30,20).to(device)

# Running forward
output1 = linear1(input1)
output2 = linear2(output1)
print(output2)
```

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

device = xm.xla_device()
input1 = torch.randn(2,10).to(device)
# Defining 2 linear layers
linear1 = torch.nn.Linear(10,30).to(device)
linear2 = torch.nn.Linear(30,20).to(device)

# Running forward
output1 = linear1(input1)
output2 = linear2(output1)
xm.mark_step()
print(output2)
print(output1)
# Printing the metrics to check if compilation and execution occurred
print(met.metrics_report())
```

```python
import os
os.environ["NEURON_USE_EAGER_DEBUG_MODE"] = "1"

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

device = xm.xla_device()
input1 = torch.randn(2,10).to(device)
# Defining 2 linear layers
linear1 = torch.nn.Linear(10,30).to(device)
linear2 = torch.nn.Linear(30,20).to(device)

# Running forward
output1 = linear1(input1)
output2 = linear2(output1)

# Printing the metrics to check if compilation and execution occurred
print(met.metrics_report())

print(output2)
print(output1)
# Printing the metrics to check if compilation and execution occurred.
print(met.metrics_report())
```

```python
import os
os.environ["NEURON_USE_EAGER_DEBUG_MODE"] = "1"

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

os.environ['NEURON_CC_FLAGS'] = "--log_level=INFO"

device = xm.xla_device()
input1 = torch.randn(2,10).to(device)
# Defining 2 linear layers
linear1 = torch.nn.Linear(10,30).to(device)
linear2 = torch.nn.Linear(30,20).to(device)
linear3 = torch.nn.Linear(20,30).to(device)
linear4 = torch.nn.Linear(30,20).to(device)

# Running forward
output1 = linear1(input1)
output2 = linear2(output1)
output3 = linear3(output2)

# Note the number of compiles at this point and compare
# with the compiles in the next metrics print
print(met.metrics_report())

output4 = linear4(output3)
print(met.metrics_report())
```

```python
import torch_xla.debug.profiler as xp

for epoch in range(total_epochs):
    inputs = torch.randn(1,10).to(device)
    labels = torch.tensor([1]).to(device)
    with xp.Trace("model_build"):
        loss = model(inputs, labels)
    with xp.Trace("loss_backward"):
        loss.backward()
```

```python
import os
os.environ["XLA_FLAGS"] = "--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
import os
if os.environ.get("RANK", "0") == "0":
    os.environ["XLA_FLAGS"]="--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
import torch_xla.core.xla_model as xm

if xm.is_master_ordinal():
    os.environ["XLA_FLAGS"]="--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
import libneuronxla

def callback(name, addressable_device_index, execution_count):
    if execution_count == 2:
        return 'outputs'
    else:
        return ''

old_callback = libneuronxla.register_hlo_snapshot_callback(callback)
```

```python
import libneuronxla

step = 0
def callback(name, addressable_device_index, execution_count):
    if step == 5:
        return 'inputs'
    else:
        return ''

old_callback = libneuronxla.register_hlo_snapshot_callback(callback)

for epoch in range(EPOCHS):
    for idx, (train_x, train_label) in enumerate(train_loader):
        step += 1
```

## trn2-llama3.1-405b-speculative-tutorial.rst

```bash
# Compile model and run generation - Scenario 1 (bf16 weights)
MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct/"
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

NUM_CORES=128
TP_DEGREE=64
LNC=2

export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
export NEURON_RT_EXEC_TIMEOUT=600 

inference_demo \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path $MODEL_PATH \
    --compiled-model-path $COMPILED_MODEL_PATH \
    --torch-dtype bfloat16 \
    --start_rank_id 0 \
    --local_ranks_size $TP_DEGREE \
    --tp-degree $TP_DEGREE \
    --batch-size 1 \
    --max-context-length 12288 \
    --seq-len 12800 \
    --on-device-sampling \
    --top-k 1 \
    --fused-qkv \
    --sequence-parallel-enabled \
    --qkv-kernel-enabled \
    --attn-kernel-enabled \
    --mlp-kernel-enabled \
    --cc-pipeline-tiling-factor 1 \
    --pad-token-id 2 \
    --enable-bucketing \
    --context-encoding-buckets 2048 4096 10240 12288 \
    --token-generation-buckets 12800 \
    --prompt "What is annapurna labs?"
```

```bash
# Start vLLM server - Scenario 1 (bf16 weights)
export NEURON_RT_VIRTUAL_CORE_SIZE=2

MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct/"
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH

VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-num-seqs 1 \
    --max-model-len 12800 \
    --tensor-parallel-size 64 \
    --no-enable-prefix-caching \
    --port 8000
```

```bash
# Compile model with fp8 quantization and fused speculation - Scenario 2
MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled/"
DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1b-instruct/"    
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"
MTNC_FILE_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled/modules_to_not_convert.json"

NUM_CORES=128
TP_DEGREE=64
LNC=2

export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
export NEURON_RT_EXEC_TIMEOUT=600 
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1

inference_demo \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path $MODEL_PATH \
    --compiled-model-path $COMPILED_MODEL_PATH \
    --torch-dtype bfloat16 \
    --start_rank_id 0 \
    --local_ranks_size $TP_DEGREE \
    --tp-degree $TP_DEGREE \
    --batch-size 1 \
    --max-context-length 12288 \
    --seq-len 12800 \
    --on-device-sampling \
    --top-k 1 \
    --fused-qkv \
    --sequence-parallel-enabled \
    --qkv-kernel-enabled \
    --attn-kernel-enabled \
    --mlp-kernel-enabled \
    --cc-pipeline-tiling-factor 1 \
    --draft-model-path $DRAFT_MODEL_PATH \
    --enable-fused-speculation \
    --speculation-length 7 \
    --pad-token-id 2 \
    --quantized-mlp-kernel-enabled \
    --quantization-type per_channel_symmetric \
    --rmsnorm-quantize-kernel-enabled \
    --enable-bucketing \
    --prompt "What is annapurna labs?" \
    --modules-to-not-convert-file $MTNC_FILE_PATH \
    --context-encoding-buckets 2048 4096 10240 12288 \
    --token-generation-buckets 12800
```

```bash
# Start vLLM server with fp8 quantization and fused speculation - Scenario 2
export NEURON_RT_INSPECT_ENABLE=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1

MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled"
DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1b-instruct"
COMPILED_MODEL_PATH="/home/ubuntu/traced_models/Llama-3.1-405B-Instruct_fp8"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH

VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --max-num-seqs 1 \
    --max-model-len 12800 \
    --tensor-parallel-size 64 \
    --device neuron \
    --speculative-max-model-len 12800 \
    --speculative-model $DRAFT_MODEL_PATH \
    --num-speculative-tokens 7 \
    --use-v2-block-manager \
    --override-neuron-config "{\"enable_fused_speculation\":true, \"quantized-mlp-kernel-enabled\":true, \"quantization-type\":\"per_channel_symmetric\", \"skip_warmup\": true}" \
    --port 8000
```

## nki_direct_allocation_guide.rst

```python
import nki.language as nl
import nki.compiler as ncc

# Example 1: Automatic allocation
nki_tensor = nl.ndarray((16, nl.par_dim(128), 512), dtype=nl.bfloat16, buffer=ncc.sbuf.auto_alloc())
nki_tensor = nl.ndarray((16, nl.par_dim(128), 512), dtype=nl.bfloat16, buffer=nl.sbuf)

# Example 2: Direct allocation with ncc.sbuf.alloc()
nki_tensor = nl.ndarray((16, nl.par_dim(128), 512), dtype=nl.bfloat16, buffer=ncc.sbuf.alloc(...))

# Example 3: Direct allocation with ncc.sbuf.mod_alloc()
nki_tensor = nl.ndarray((16, nl.par_dim(128), 512), dtype=nl.bfloat16, buffer=ncc.sbuf.mod_alloc(...))
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 4: Simple 1D allocation function
def simple_1d_alloc_func(idx, pdim_size, fdim_size):
    idx, = idx  # unpack the tuple
    return (0, idx * fdim_size)

t = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
               buffer=ncc.sbuf.alloc(simple_1d_alloc_func))
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 5: Allocation factory with closure
next_addr = 0

def simple_1d_alloc_factory(total_fdim_size):
    global next_addr
    base_addr = next_addr
    next_addr += total_fdim_size

    def simple_1d_alloc_func(idx, pdim_size, fdim_size):
        idx, = idx
        start_partition = 0
        return (start_partition, base_addr + idx * fdim_size)

    return simple_1d_alloc_func

t0 = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                buffer=ncc.sbuf.alloc(simple_1d_alloc_factory(512*2*4)))
t1 = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                buffer=ncc.sbuf.alloc(simple_1d_alloc_factory(512*2*4)))
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 6: Modulo allocation
nki_tensor = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                        buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 7: Hoisting allocation outside loop (INCORRECT - serialized)
for i in nl.affine_range(8):
    t = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=ncc.sbuf.mod_alloc(base_addr=0))
    t[i] = ...
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 8: Hoisting allocation outside loop (CORRECT - parallelized)
t = nl.ndarray((8, 128, 512), dtype=nl.bfloat16,
               buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(8,)))
for i in nl.affine_range(8):
    t[i] = ...
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 9: Mixing direct allocation with automatic allocation (INCORRECT)
t = nl.load(input)  # t is a new tensor, this will fail
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 10: Correct way to use direct allocation with load
t = nl.ndarray(shape=..., dtype=..., buffer=ncc.sbuf.alloc(...))
t[...] = nl.load(input)
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 11: Lifetime conflict - overlapping memory addresses
t0 = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))

t1 = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                buffer=ncc.sbuf.mod_alloc(base_addr=1024, num_free_tiles=(2, )))
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 12: Lifetime conflict - insufficient physical tiles (INCORRECT)
t1 = nl.ndarray((8, nl.par_dim(128), 512), dtype=nl.bfloat16,
                buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))

for i in nl.affine_range(8):
    t1[i] = nl.load(...)

for i in nl.affine_range(8):
    result[i] = nl.exp(t1[i])
```

```python
import nki.language as nl
import nki.compiler as ncc

# Example 13: Correct way to avoid lifetime conflict
for i in nl.affine_range(4):
    t1 = nl.ndarray((2, nl.par_dim(128), 512), dtype=nl.bfloat16,
                    buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))
    for j in nl.affine_range(2):
        t1[j] = nl.load(...)
        result[i*2 + j] = nl.exp(t1[j])
```

## matrix_multiplication_nki_kernels.py

```python
import nki as nki
import nki.isa as nisa
import nki.language as nl
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
  K, M = lhsT.shape
  K_, N = rhs.shape

  assert K == K_, \
    f"Expected contraction dimension to match on both lhsT ({K}) and rhs ({K})"
  assert K == 128, f"Expected contraction dimension to be 128, but got {K}"
  assert M == 64, f"Expected lhsT matrix to have dimension M of 64, but got {M}"
  assert N == 512, f"Expected rhs matrix to have dimension N of 512, but got {N}"

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  nisa.dma_copy(dst=lhs_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)

  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)

  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=result.dtype)

  nisa.dma_copy(dst=result, src=result_sbuf)

  return result


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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result


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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m in nl.affine_range(M // TILE_M):
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=lhsT_tile, 
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                             m * TILE_M:(m + 1) * TILE_M])
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])
        rhs_tiles.append(rhs_tile)

      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])

      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result


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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N

  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m in nl.affine_range(M // BLOCK_M):
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      lhsT_tiles_internal = []
      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                               dtype=lhsT.dtype,
                               buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               (m * TILES_IN_BLOCK_M + bm) *
                               TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                               TILE_M])
        lhsT_tiles_internal.append(lhsT_tile)
      lhsT_tiles.append(lhsT_tiles_internal)

    for n in nl.affine_range(N // BLOCK_N):
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        rhs_tiles_internal = []
        for k in nl.affine_range(K // TILE_K):
          rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          nisa.dma_copy(dst=rhs_tile,
                        src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                (n * TILES_IN_BLOCK_N + bn) *
                                TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                TILE_N])
          rhs_tiles_internal.append(rhs_tile)
        rhs_tiles.append(rhs_tiles_internal)

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                   dtype=nl.float32,
                                   buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            nisa.nc_matmul(dst=result_tile,
                           stationary=lhsT_tiles[bm][k],
                           moving=rhs_tiles[bn][k])
  
          result_tmp = nl.ndarray(shape=result_tile.shape,
                                  dtype=result.dtype,
                                  buffer=nl.sbuf)
          nisa.tensor_copy(dst=result_tmp, src=result_tile)

          nisa.dma_copy(dst=result[(m * TILES_IN_BLOCK_M + bm) *
                                   TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                   TILE_N],
                        src=result_tmp)

  return result


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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  for n in nl.affine_range(NUM_BLOCK_N):
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
          nisa.memset(dst=tile, value=0.0)
          block_n.append(tile)
        block_m.append(block_n)
      result_tmps.append(block_m)

    for k in nl.sequential_range(NUM_BLOCK_K):
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        rhs_tiles.append(rhs_tile)

      for m in nl.affine_range(NUM_BLOCK_M):
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          lhsT_tiles.append(lhsT_tile)

        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
```

## pp_developer_guide.rst

```python
# Create torch model
config.return_dict = False
model = transformers.LlamaForCausalLM(config)
# Create pipeline cuts
pipeline_cuts = create_partition(config, args)
# Apply model wrapper
model = NxDPPModel(
    model,
    transformer_layer_cls=LlamaDecoderLayer,
    num_microbatches=args.num_microbatches,
    virtual_pipeline_size=1,
    output_loss_value_spec=(True, False),
    input_names=["input_ids", "attention_mask", "labels"],
    pipeline_cuts=pipeline_cuts,
    trace_file_path=args.trace_file_path,
    leaf_module_cls=[LlamaRMSNorm.__name__],
    autowrap_modules=[mappings],
    use_zero1_optimizer=args.use_zero1_optimizer,
    deallocate_pipeline_outputs=False,
)
model.move_model_to_device()
```

```python
def create_partition(config, args):
    """
    Evenly split the transformer layers between the PP ranks
    """
    assert config.num_hidden_layers % args.pipeline_parallel_size == 0
    num_layer_per_partition = config.num_hidden_layers  // args.pipeline_parallel_size
    pipeline_cuts = []
    current_cut = num_layer_per_partition - 1
    for i in range(args.pipeline_parallel_size-1):
        pipeline_cuts.append(f"model.layers.{current_cut}")
        current_cut += num_layer_per_partition
    if torch.distributed.get_rank() == 0:
        print(f"pipeline_cuts {pipeline_cuts}")
    return pipeline_cuts
```

```python
# replace loss, _ = model(input_ids, attention_mask, labels) with below
with torch.autocast(enabled=args.use_amp > 0, dtype=torch.bfloat16, device_type="cuda"):
    loss = model.run_train(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
```

```python
from torchdistx import deferred_init
# Instead of model = LlamaForCausalLM(config)
model = deferred_init.deferred_init(LlamaForCausalLM, config)
```

```python
from neuronx_distributed.utils.model_utils import init_on_device
with init_on_device(torch.device("meta")):
    model = LlamaForCausalLM(config)
```

```python
def init_weights(module):
    from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
    if isinstance(module, (nn.Linear, Conv1D)):
        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
        if module.padding_idx:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, (ParallelEmbedding, RowParallelLinear, ColumnParallelLinear)):
        module.init_weight_cpu()
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

model = NxDPPModel(...,param_init_fn=init_weights,...)
```

```python
from typing import Any, Dict, Iterator, Tuple
import torch.nn as nn
import torch
from torch_xla.utils.checkpoint import checkpoint as torch_checkpoint
from torch.distributed.utils import _replace_by_prefix

_CHECKPOINT_WRAPPED_MODULE = "mod"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."

class CheckPointWrapper(torch.nn.Module):
    def __init__(self, mod) -> None:
        super().__init__()
        self.mod = mod
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        ordered_args = list(args)
        for value in kwargs.values():
            ordered_args += [value]
        return torch_checkpoint(self.mod, *ordered_args, use_reentrant=True)
    
    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        for param_name, param in super().named_parameters(*args, **kwargs):
            updated_name = param_name.replace(_CHECKPOINT_PREFIX, "")
            yield updated_name, param
    
    def named_modules(self,*args,**kwargs):
        for module_name, module in super().named_modules(*args, **kwargs):
            updated_name = module_name.replace(_CHECKPOINT_PREFIX, "")
            yield updated_name, module

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        return state_dict
    
    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")

def apply_checkpoint(dist_model, layers_to_checkpoint=None):
    checkpoint_wrapper_added = False
    if layers_to_checkpoint is not None and len(layers_to_checkpoint) == 0:
        raise RuntimeError(
            f"invalid input layers_to_checkpoint {layers_to_checkpoint}, can't be empty"
        )
    for name, module in dist_model.local_module.named_children():
        if (layers_to_checkpoint and name in layers_to_checkpoint) or (
            not layers_to_checkpoint and type(module) == dist_model.transformer_layer_cls
        ):
            dist_model.local_module.add_module(name, CheckPointWrapper(module))
            checkpoint_wrapper_added = True
    if layers_to_checkpoint is not None and not checkpoint_wrapper_added:
        pass
    elif layers_to_checkpoint is None and not checkpoint_wrapper_added:
        pass

model = NxDPPModel(...)
apply_checkpoint(model)
```

```python
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMHF

# Keep the same class name as original one
class LlamaForCausalLM(LlamaForCausalLMHF):
    pass
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
neuron_config = dict(
    tp_degree=parallel_config.tensor_parallel_size,
    ctx_batch_size=1,
    batch_size=scheduler_config.max_num_seqs,
    max_context_length=scheduler_config.max_model_len,
    seq_len=scheduler_config.max_model_len,
    enable_bucketing=True,
    is_continuous_batching=True,
    quantized=False,
    torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
    padding_side="right"
)
```

```python
override_neuron_config={
    "enable_bucketing":False,
}
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

## torch-neuronx-profiling-with-tb.rst

```python
import os
import torch
import torch_neuronx
from torch_neuronx.experimental import profiler
import torch_xla.core.xla_model as xm

os.environ["NEURON_CC_FLAGS"] = "--cache_dir=./compiler_cache"

device = xm.xla_device()

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4, 2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))

with torch.no_grad():
    model = NN()
    inp = torch.rand(4, 4)
    output = model(inp)

    with torch_neuronx.experimental.profiler.profile(
        port=9012,
        profile_type='operator',
        ms_duration=10000):
        
        neuron_model = model.to(device)
        neuron_inp = inp.to(device)
        
        output_neuron = neuron_model(neuron_inp)
        xm.mark_step()
```

```python
import os
import torch
import torch_neuronx
from torch_neuronx.experimental import profiler

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4, 2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))

model = NN()
model.eval()

inp = torch.rand(4, 4)
output = model(inp)

with torch_neuronx.experimental.profiler.profile(
    port=9012,
    profile_type='operator',
    ms_duration=10000,
    traced_only=True):
    
    neuron_model = torch_neuronx.trace(model, inp, compiler_workdir="./compiler_cache")
    output_neuron = neuron_model(inp)
```

## pytorch-neuron-programming-guide.rst

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
```

```python
device = 'xla'
```

```python
model.to(device)
tensor.to(device)
```

```python
tensor.cpu()
```

```python
tensor.to('cpu')
```

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
# or
device = 'xla'
```

```python
xm.mark_step()
```

```python
device = 'cpu'
if not os.environ.get("DISABLE_XLA", None):
    device = 'xla'

# end of training step 
if not os.environ.get("DISABLE_XLA", None):
    xm.mark_step()
```

```python
import torch_xla.distributed.parallel_loader as pl
```

```python
import torch_xla.distributed.xla_backend
torch.distributed.init_process_group('xla')
```

```python
xm.optimizer_step(optimizer)
```

```python
parallel_loader = pl.MpDeviceLoader(dataloader, device)
```

```python
import torch_xla.core.xla_model as xm

devkind = xm.xla_device_kind()
print(devkind)
```

```python
import torch_xla.core.xla_model as xm

devices = xm.get_xla_supported_devices()
print(len(devices))
print(devices)
```

```python
from torch_neuronx.utils import get_platform_target

platform = get_platform_target()
print(platform)
```

```python
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"

# model is created
model.to(torch.bfloat16)
```

```python
running_loss = torch.zeros(1, dtype=torch.float).to(device)
```

```python
grad = p.grad.data.float()
```

```python
# model is created
model.to(torch.bfloat16)
```

```python
# keep a copy of weights in highprec
self.param_groups_highprec = []
for group in self.param_groups:
    params = group['params']
    param_groups_highprec = [p.data.float() for p in params]
    self.param_groups_highprec.append({'params': param_groups_highprec})
```

```python
os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
```

```python
with torch.autocast(dtype=torch.bfloat16, device_type='xla'):
    # forward pass
```

```python
def train_loop_fn(train_loader):
    for i, data in enumerate(train_loader):
        inputs = data[0]
        labels = data[3]
        outputs = model(inputs, labels=labels)
        loss = outputs.loss/ flags.grad_acc_steps
        loss.backward()
        optimizer.step()
        xm.mark_step()
```

```python
os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"

def train_loop_fn(train_loader):
    for i, data in enumerate(train_loader):
        torch.cuda.is_bf16_supported = lambda: True
        with torch.autocast(dtype=torch.bfloat16, device_type='xla'):
            inputs = data[0]
            labels = data[3]
            outputs = model(inputs, labels=labels)
        loss = outputs.loss/ flags.grad_acc_steps
        loss.backward()
        optimizer.step()
        xm.mark_step()
```

## perceiver-multimodal_benchmark.py

```python
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers import PerceiverForMultimodalAutoencoding
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverClassifierOutput
from transformers.models.perceiver.modeling_perceiver import restructure


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

## mlp.rst

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
# or
device = 'xla'
```

```python
model.to(device)
xm.mark_step()
loss.item()
```

```python
xm.save(checkpoint, path)
```

```python
import torch_xla.distributed.xla_backend
import torch.distributed

torch.distributed.init_process_group('xla')
```

```python
from torch_xla.distributed import MpDeviceLoader

test_loader = MpDeviceLoader(test_dataset, batch_size=32, drop_last=True)
```

```python
xm.optimizer_step(optimizer)
```

```python
world_size = xm.xrt_world_size()
```

```python
import torch_neuronx

if idx == 0:
    model = torch_neuronx.trace(model, test_x)
```

```python
from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)
```

## use-neuron-profile.rst

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import os

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"


@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
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

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler to change this loop by,
    # for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x])

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype=lhsT.dtype,
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        # Do matmul with all tiles in the blocks
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

            # Accumulate on corresponding SBUF tile
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                   dtype=result_tiles.dtype,
                                   buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])

  return result


class NKILinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NKILinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        weight_T = self.weight.t()
        x_T = x.t()
        output = nki_matmul_fully_optimized_(x_T, weight_T)
        return output + self.bias


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = NKILinear(2048, 2048)
        self.fc2 = NKILinear(2048, 1024)
        self.fc3 = NKILinear(1024, 1024)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
```

## hf_llama3_8B_DPO_ORPO.rst

```python
from datasets import load_dataset
from transformers import AutoTokenizer

def preference_data_format(example):

    system = "<|im_start|>\n" + example['system'] + "<|im_end|>\n"

    # Format instruction
    prompt = "<|im_start|> " + example['question'] + "<|im_end|>\n<|im_start|>assistant\n"

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Particular dataset with following fields: "system", "question", "chosen", "rejected"
dataset = load_dataset("json", data_files="orca_rlhf.jsonl", split="train")

# Save columns
original_columns = dataset.column_names

# Format dataset
dataset = dataset.map(
    preference_data_format,
    remove_columns=original_columns
    )

# save converted preference dataset
dataset.to_json("data_dpo.jsonl")
```

## disaggregated-inference.rst

```python
class NeuronConnector:
    def _keep_alive_ectd(self):
        # Add worker to etcd
        etcd_client.put(
            f"/workers/{self.role}/{self.local_ip}/{self.api_server_port}",
            json.dumps({"connections": []}),
            lease
        )
```

```python
def initialize_buffer(self):
    if self.config.is_kv_producer:
        self.static_buffer = SendBuffer(
            self.kv_caches,
            self.zmq_context,
            self.neuron_recv_ip,
            self.config.kv_ip,
            self.config.kv_port
        )
```

```python
def maybe_setup_buffer(self, remote_ip, remote_port):
    if self.static_buffer:
        return self.static_buffer

    key = "" if self.config.is_kv_producer else (remote_ip, remote_port)
    
    if key in self.connection_dict:
        return self.connection_dict[key]
```

```python
class NeuronTransferEngine:
    def transfer_neuron_tensors(self, tensors, offsets, lengths, peer_devices, ...):
        self.engine.queue_transfer_with_token(
            tensors, offsets, lengths, peer_devices, self.local_devices,
            self.comm_ids, completion_count, completion_token, use_queue,
            completion_time_out)
```

```python
def send_handler(self):
    while True:
        identity, request = self.router.recv_json()
    
        if request["type"] == "handshake":
            self.router.send_json(identity, {
                "status": "ok",
                "timestamp": time.time()
            })
            continue
    
        if request["type"] == "kv_map_init":
            # Set up transfer details
            continue
            
        if request["type"] == "lookup_all":
            self._process_lookup_all(identity, request)
            continue
```

```python
# Prefill Side - Starting Transfers
if request_id not in self.lookup_dict:
    self.router.send_json(identity, {"success": False})
    return

kv_caches, offsets, lengths, peer_devices = \
    self.generate_transfer_sequences(entry, remote_id=identity_str)

self.get_transfer_engine(remote_id=identity_str).transfer_neuron_tensors(
    kv_caches, offsets, lengths, peer_devices,
    completion_token=entry.completion_token)
```

```python
# Decode Side - Starting Transfers
entry.output_token = torch.tensor(
    response["output_token"]).unsqueeze(0)

kv_caches, offsets, lengths, peer_devices = \
     self.generate_transfer_sequences(entry)

self.get_transfer_engine().transfer_neuron_tensors(
    kv_caches, offsets, lengths, peer_devices,
    completion_token=entry.completion_token)
```

```python
prefill_task = asyncio.create_task(anext(prefill_response))
decode_task = asyncio.create_task(anext(decode_response))

await prefill_task
async for chunk in handle_prefill_response(prefill_response,
                                         streaming, endpoint,
                                         uid, request_time):
    yield chunk

await decode_task
async for chunk in handle_decode_response(decode_response,
                                        streaming, endpoint, uid,
                                        request_time):
    yield chunk
```

## vllm-user-guide-v1.rst

```python
import os
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=4,
    max_model_len=128,
    tensor_parallel_size=32
)

# Generate text
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.0)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

```python
import os
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    max_num_seqs=4,
    max_model_len=4096,
    tensor_parallel_size=64,
    additional_config=dict(
        override_neuron_config=dict(
            enable_bucketing=False,
        )
    ),
)

# Generate text
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.0)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

```python
from openai import OpenAI

# Client Setup
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model_name = models.data[0].id

# Sampling Parameters
max_tokens = 1024
temperature = 1.0
top_p = 1.0
top_k = 50
stream = False

# Chat Completion Request
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

# Parse the response
generated_text = ""
if stream:
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            generated_text += chunk.choices[0].delta.content
else:
    generated_text = response.choices[0].message.content
    
print(generated_text)
```

## hf_llama3_70B_pretraining.rst

```python
from huggingface_hub import login
from transformers import AutoTokenizer

login(token='your_own_hugging_face_token')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B')  
# For llama3 uncomment line below
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B')

tokenizer.save_pretrained(".")
```

## perceiver-multimodal_compile.py

```python
import torch
import torch.nn as nn
from typing import Optional
import torch_neuronx

# Define wrapper for tracing encoder
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
        return last_hidden_states
```

```python
import torch
import torch.nn as nn
from typing import Optional
import torch_neuronx

# Define wrapper for tracing decoder
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
```

```python
import torch
import torch_neuronx

# Compile Encoder with torch_neuronx.trace
neuron_encoder.encoder_wrapper = torch_neuronx.trace(
  neuron_encoder.encoder_wrapper,
  (embedding_output, sample_inputs, extended_attention_mask),
  compiler_workdir=COMPILER_WORKDIR_ENCODER,
  compiler_args=[f"--temp-dir={COMPILER_WORKDIR_ENCODER}", "--auto-cast=none"]
)

# Save compiled encoder
torch.jit.save(neuron_encoder.encoder_wrapper, encoder_fname)
```

```python
import torch
import torch_neuronx

# Compile Decoder with torch_neuronx.trace
neuron_decoder.decoder_wrapper = torch_neuronx.trace(
   neuron_decoder.decoder_wrapper,
   (z, query_mask, audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
        image_input, image_input_without_pos, image_subsampled_point, image_padding,
        label_input, label_input_without_pos, label_padding),
   compiler_workdir=COMPILER_WORKDIR_DECODER,
   compiler_args=[f"--temp-dir={COMPILER_WORKDIR_DECODER}", "--auto-cast=none"]
)

# Save compiled decoder
torch.jit.save(neuron_decoder.decoder_wrapper, decoder_fname)
```

## quickstart-implement-run-kernel.rst

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn1")
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """

    # Check both input tensor shapes are the same for element-wise operation.
    assert a_input.shape == b_input.shape

    # Check the first dimension's size to ensure it does not exceed on-chip
    # memory tile size, since this simple kernel does not tile inputs.
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate space for the input tensors in SBUF and copy the inputs from HBM
    # to SBUF with DMA copy. Note: 'sbuf' is a keyword in NKI.
    a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create a tensor in HBM and copy the result into HBM. Note: Simlar to
    # 'sbuf', 'hbm' is a keyword in NKI.
    c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.dma_copy(dst=c_output, src=c_tile)

    # Return kernel output as function output.
    return c_output
```

```python
import torch
from torch_xla.core import xla_model as xm
from add_kernel import nki_tensor_add_kernel

device = xm.xla_device()

a = torch.ones((4, 3), dtype=torch.float16).to(device=device)
b = torch.ones((4, 3), dtype=torch.float16).to(device=device)

c = nki_tensor_add_kernel(a, b)

print(c)
```

```python
import jax.numpy as jnp
from add_kernel import nki_tensor_add_kernel

a = jnp.ones((4, 3), dtype=jnp.float16)
b = jnp.ones((4, 3), dtype=jnp.float16)

c = nki_tensor_add_kernel(a, b)

print(c)
```

## design-rmsnorm-quant.rst

```python
def rmsnorm_quant_ref(inp: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """RMSNorm + Quantization reference impl.

    - inp: shape [B, S, H]
    - output[0]: shape [B, S, H] in fp8e4, representing the quantized RMSNorm output of input
    - output[1]: shape [B, S, 4] in fp32 representing the per-row dequantization scale
    """
    assert(len(inp.shape) == 3)
    inp = inp.astype(np.float32)
    gamma = gamma.astype(np.float32)

    # Perform RMSNorm
    rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True))
    norm = inp * np.reciprocal(rms + eps)
    norm *= gamma

    # Perform quantization
    norm_abs_max = np.abs(norm).max(axis=-1, keepdims=True)
    quant_scale = 240.0 / norm_abs_max
    norm_quant = norm * quant_scale
    assert(np.allclose(norm, norm_quant * np.reciprocal(quant_scale)))  # dequantization should yield same norm

    # Cast and return
    norm_quant = dt.static_cast(norm_quant, dt.float8_e4m3)
    dequant_scale = dt.static_cast(np.reciprocal(quant_scale), np.float32)

    return norm_quant, dequant_scale
```

## t5_model_layers.py

```python
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import BaseParallelLinear, ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.utils import divide

import torch
from torch import nn
from transformers import T5Config
from transformers.activations import ACT2FN
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5LayerNorm, T5LayerCrossAttention, T5LayerFF, T5DenseGatedActDense, T5DenseActDense
from transformers import T5ForConditionalGeneration
import neuronx_distributed


def prune_linear_layer(layer: BaseParallelLinear, index: torch.LongTensor,
                       dim: int = 0) -> BaseParallelLinear:
    """
    Prune a linear layer to keep only entries in index.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = ColumnParallelLinear(new_size[1],
                                     new_size[0],
                                     bias=layer.bias is not None,
                                     gather_output=False).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class ParallelAttention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        world_size = parallel_state.get_tensor_model_parallel_size()
        self.num_attention_heads_per_partition = divide(
            self.n_heads, world_size)
        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.key_value_proj_dim

        self.q = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.k = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.v = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.o = RowParallelLinear(self.inner_dim,
                                   self.d_model,
                                   bias=False,
                                   input_is_parallel=True)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = ParallelEmbedding(self.relative_attention_num_buckets, self.n_heads)
        self.n_heads = self.num_attention_heads_per_partition


class ParallelDenseActDense(T5DenseActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, gather_output=False, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, input_is_parallel=True, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class ParallelDenseGatedActDense(T5DenseGatedActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi_0 = ColumnParallelLinear(config.d_model,
                                      config.d_ff,
                                         gather_output=False,
                                      bias=False)
        self.wi_1 = ColumnParallelLinear(config.d_model,
                                      config.d_ff,
                                        gather_output=False,
                                      bias=False)
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    input_is_parallel=True,
                                    bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]
```

## neuronperf_benchmark_guide.rst

```python
def preprocess_fn(x):
    return x * 5
```

```python
def preprocess_fn(x, y, z):
    return x / 255, y / 255, z / 255
```

```python
def postprocess_fn(x):
    return x.argmax()
```

```python
reports = npf.torch.benchmark('model_neuron_b1.pt', ..., workers_per_model=1)
```

```python
workers_per_model = [1, 2]
reports = npf.torch.benchmark('model_neuron_b1.pt', ..., workers_per_model=workers_per_model)
```

```python
reports = npf.torch.benchmark('model_neuron_b1.pt', ..., n_models=6)
```

```python
n_models = list(range(1, 10))
reports = npf.torch.benchmark('model_neuron_b1.pt', ..., n_models=n_models)
```

```python
reports = npf.torch.benchmark('model_neuron_b1.pt', ..., pipeline_sizes=2)
```

```python
reports = npf.torch.benchmark('model_index.json', ..., pipeline_sizes=[1, 2, 3])
```

```python
reports = npf.torch.benchmark('model_index.json', ..., duration=10)
```

```python
results = npf.torch.benchmark('model_index.json', ..., return_timers=True)
```

```python
reports = npf.get_reports(results)
```

```python
reports = npf.torch.benchmark(..., n_models=1, duration=5, verbosity=2)
```

```python
reports = npf.torch.benchmark(..., multiprocess=False)
```

```python
reports = npf.torch.benchmark(..., multiinterpreter=True)
```

```python
cpu_reports = npf.cpu.benchmark(YourModelClass, ...)
```

```python
gpu_reports = npf.torch.benchmark(YourModelClass, ..., device_type="gpu")
```

```python
import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        model_name = "bert-base-cased"
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
        self.add_module(model_name, self.bert)

    def forward(self, *inputs):
        return self.bert(*inputs)
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

## bucketing-app-note.rst

```python
import numpy as np
import torch
from torchvision import models
import torch_neuron

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Define the bucket sizes that will be used for compilation and inference
bucket_sizes = [(500, 500), (600, 600), (700, 700), (800, 800)]

# Create the bucketed models by compiling a model for each bucket size
buckets = {}
for bucket_size in bucket_sizes:
    # Create an example input that is the desired bucket size
    h, w = bucket_size
    image = torch.rand([1, 3, h, w])

    # Compile with the example input to create the bucketed model
    model_neuron = torch.neuron.trace(model, image)

    # Run a warm up inference to load the model into Inferentia memory
    model_neuron(image)

    # Add the bucketed model based on its bucket size
    buckets[bucket_size] = model_neuron


def get_bucket_and_pad_image(image):
    # Determine which bucket size to use
    oh, ow = image.shape[-2:]
    target_bucket = None
    for bucket_size in bucket_sizes:
        # Choose a bucket that's larger in both the height and width dimensions
        if oh <= bucket_size[0] and ow <= bucket_size[1]:
            target_bucket = bucket_size
            break

    # Pad the image to match the size of the bucket
    h_delta = target_bucket[0] - oh
    w_delta = target_bucket[1] - ow

    b_pad = h_delta  # Bottom padding
    l_pad = 0  # Left padding
    t_pad = 0  # Top padding
    r_pad = w_delta  # Right padding

    # Pad the height and width of the image
    padding_amounts = (l_pad, r_pad, t_pad, b_pad)
    image_padded = torch.nn.functional.pad(image, padding_amounts, value=0)

    return image_padded, target_bucket


# Run inference on inputs with different shapes
for _ in range(10):
    # Create an image with a random height and width in range [400, 400] to [800, 800]
    h = int(np.random.uniform(low=400, high=800))
    w = int(np.random.uniform(low=400, high=800))
    image = torch.rand(1, 3, h, w)

    # Determine bucket and pad the image
    image_padded, target_bucket = get_bucket_and_pad_image(image)

    # Use the corresponding bucket to run inference
    output = buckets[target_bucket](image_padded)
```

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch_neuron

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)
model.eval()

# Define the bucket sizes that will be used for compilation and inference
bucket_sizes = [64, 128]

# Create the bucketed models by compiling a model for each bucket size
buckets = {}
for bucket_size in bucket_sizes:
    # Setup some example inputs
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "HuggingFace's headquarters are situated in Manhattan"

    # Create an example input that is the desired bucket size
    paraphrase = tokenizer.encode_plus(sequence_0,
                                    sequence_1,
                                    max_length=bucket_size,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt")

    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

    # Compile with the example input to create the bucketed model
    model_neuron = torch.neuron.trace(model, example_inputs_paraphrase)

    # Run a warm up inference to load the model into Inferentia memory
    model_neuron(*example_inputs_paraphrase)

    # Add the bucketed model based on its bucket size
    buckets[bucket_size] = model_neuron


def get_bucket_and_pad_paraphrase(paraphrase):
    # Determine which bucket size to use
    inputs = paraphrase['input_ids']
    attention = paraphrase['attention_mask']
    token_type = paraphrase['token_type_ids']
    paraphrase_len = inputs.shape[1]
    target_bucket = None
    for bucket_size in bucket_sizes:
        if paraphrase_len <= bucket_size:
            target_bucket = bucket_size
            break

    # Pad the paraphrase to match the size of the bucket
    delta = target_bucket - paraphrase_len
    zeros = torch.zeros([1, delta], dtype=torch.long)
    inputs = torch.cat([inputs, zeros], dim=1)
    attention = torch.cat([attention, zeros], dim=1)
    token_type = torch.cat([token_type, zeros], dim=1)

    paraphrase_padded = inputs, attention, token_type
    return paraphrase_padded, target_bucket


# Create two sample sequences
sequence_0 = ("The only other bear similar in size to the polar bear is the "
              "Kodiak bear, which is a subspecies of the brown bear. Adult male "
              "polar bears weigh 350–700 kg and measure 2.4–3 meters in total "
              "length. All bears are short-tailed, the polar bear's tail is "
              "relatively the shortest amongst living bears.")
sequence_1 = ("Around the Beaufort Sea, however, mature males reportedly "
              "average 450 kg. Adult females are roughly half the size of males "
              "and normally weigh 150–250 kg, measuring 1.8–2.4 meters in length. "
              "The legs are stocky and the ears and tail are small.")

# Run inference on inputs with different shapes
for _ in range(10):
    # Get random sequence lengths between 0 and 128
    paraphrase_len = int(np.random.uniform(128))

    # Crop the paraphrase
    paraphrase_cropped = tokenizer.encode_plus(sequence_0,
                                    sequence_1,
                                    max_length=paraphrase_len,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt")

    # Determine bucket and pad the paraphrase
    paraphrase_padded, target_bucket = get_bucket_and_pad_paraphrase(paraphrase_cropped)

    # Use the corresponding bucket to run inference
    output = buckets[target_bucket](*paraphrase_padded)
```

## matrix_multiplication.rst

```python
import nki
import nki.language as nl
from nki import nisa

@nl.kernel
def nki_matmul_basic_():
    # Define indices to access LHS and RHS input tensors
    lhs_T = nl.ndarray(shape=(128, 64), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    output = nl.ndarray(shape=(64, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Load inputs from HBM to SBUF
    nisa.dma_copy(lhs_T, nl.hbm_ref(lhs_T_hbm))
    nisa.dma_copy(rhs, nl.hbm_ref(rhs_hbm))
    
    # Perform matrix multiplication
    psum_result = nl.ndarray(shape=(64, 512), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_matmul(psum_result, lhs_T, rhs, transpose_lhs=True)
    
    # Copy result from PSUM to SBUF
    nisa.tensor_copy(output, psum_result)
    
    # Store result to HBM
    nisa.dma_copy(nl.hbm_ref(output_hbm), output)
```

```python
import nki
import nki.language as nl
from nki import nisa

@nl.kernel
def nki_matmul_tiled_(M, K, N):
    lhs_T = nl.ndarray(shape=(K, M), dtype=nl.bfloat16, buffer=nl.hbm)
    rhs = nl.ndarray(shape=(K, N), dtype=nl.bfloat16, buffer=nl.hbm)
    output = nl.ndarray(shape=(M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    
    psum_buf = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.psum)
    
    for m in nl.affine_range(M // 128):
        for n in nl.affine_range(N // 512):
            accum = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(accum, nl.hbm_ref(output[m*128:(m+1)*128, n*512:(n+1)*512]))
            
            for k in nl.affine_range(K // 128):
                lhs_tile = nl.ndarray(shape=(128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                rhs_tile = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                
                nisa.dma_copy(lhs_tile, nl.hbm_ref(lhs_T[k*128:(k+1)*128, m*128:(m+1)*128]))
                nisa.dma_copy(rhs_tile, nl.hbm_ref(rhs[k*128:(k+1)*128, n*512:(n+1)*512]))
                
                nisa.nc_matmul(psum_buf, lhs_tile, rhs_tile)
            
            nisa.tensor_copy(accum, psum_buf)
            nisa.dma_copy(nl.hbm_ref(output[m*128:(m+1)*128, n*512:(n+1)*512]), accum)
```

```python
import nki
import nki.language as nl
from nki import nisa

@nl.kernel
def nki_matmul_hoist_load_(M, K, N):
    lhs_T = nl.ndarray(shape=(K, M), dtype=nl.bfloat16, buffer=nl.hbm)
    rhs = nl.ndarray(shape=(K, N), dtype=nl.bfloat16, buffer=nl.hbm)
    output = nl.ndarray(shape=(M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    
    psum_buf = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.psum)
    
    for m in nl.affine_range(M // 128):
        for n in nl.affine_range(N // 512):
            accum = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(accum, nl.hbm_ref(output[m*128:(m+1)*128, n*512:(n+1)*512]))
            
            lhs_tiles = nl.ndarray(shape=(K // 128, 128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_tiles = nl.ndarray(shape=(K // 128, 128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            
            for k in nl.affine_range(K // 128):
                nisa.dma_copy(lhs_tiles[k], nl.hbm_ref(lhs_T[k*128:(k+1)*128, m*128:(m+1)*128]))
                nisa.dma_copy(rhs_tiles[k], nl.hbm_ref(rhs[k*128:(k+1)*128, n*512:(n+1)*512]))
            
            for k in nl.affine_range(K // 128):
                nisa.nc_matmul(psum_buf, lhs_tiles[k], rhs_tiles[k])
            
            nisa.tensor_copy(accum, psum_buf)
            nisa.dma_copy(nl.hbm_ref(output[m*128:(m+1)*128, n*512:(n+1)*512]), accum)
```

```python
import nki
import nki.language as nl
from nki import nisa

@nl.kernel
def nki_matmul_block_free_dimension_(M, K, N):
    lhs_T = nl.ndarray(shape=(K, M), dtype=nl.bfloat16, buffer=nl.hbm)
    rhs = nl.ndarray(shape=(K, N), dtype=nl.bfloat16, buffer=nl.hbm)
    output = nl.ndarray(shape=(M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    
    psum_buf = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.psum)
    
    NUM_BLOCK_M = 2
    NUM_BLOCK_N = 2
    
    for m_block in nl.affine_range(M // (128 * NUM_BLOCK_M)):
        for n_block in nl.affine_range(N // (512 * NUM_BLOCK_N)):
            result_tiles = nl.ndarray(shape=(NUM_BLOCK_M, NUM_BLOCK_N, 128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            
            for m in nl.affine_range(NUM_BLOCK_M):
                for n in nl.affine_range(NUM_BLOCK_N):
                    nisa.dma_copy(result_tiles[m, n], nl.hbm_ref(output[m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]))
            
            lhs_tiles = nl.ndarray(shape=(K // 128, NUM_BLOCK_M, 128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_tiles = nl.ndarray(shape=(K // 128, NUM_BLOCK_N, 128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            
            for k in nl.affine_range(K // 128):
                for m in nl.affine_range(NUM_BLOCK_M):
                    nisa.dma_copy(lhs_tiles[k, m], nl.hbm_ref(lhs_T[k*128:(k+1)*128, m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128]))
                for n in nl.affine_range(NUM_BLOCK_N):
                    nisa.dma_copy(rhs_tiles[k, n], nl.hbm_ref(rhs[k*128:(k+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]))
            
            for k in nl.affine_range(K // 128):
                for m in nl.affine_range(NUM_BLOCK_M):
                    for n in nl.affine_range(NUM_BLOCK_N):
                        nisa.nc_matmul(psum_buf, lhs_tiles[k, m], rhs_tiles[k, n])
                        nisa.tensor_copy(result_tiles[m, n], psum_buf)
            
            for m in nl.affine_range(NUM_BLOCK_M):
                for n in nl.affine_range(NUM_BLOCK_N):
                    nisa.dma_copy(nl.hbm_ref(output[m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]), result_tiles[m, n])
```

```python
import nki
import nki.language as nl
from nki import nisa

@nl.kernel
def nki_matmul_fully_optimized_(M, K, N):
    lhs_T = nl.ndarray(shape=(K, M), dtype=nl.bfloat16, buffer=nl.hbm)
    rhs = nl.ndarray(shape=(K, N), dtype=nl.bfloat16, buffer=nl.hbm)
    output = nl.ndarray(shape=(M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    
    psum_buf = nl.ndarray(shape=(128, 512), dtype=nl.bfloat16, buffer=nl.psum)
    
    NUM_BLOCK_M = 16
    NUM_BLOCK_N = 2
    NUM_BLOCK_K = 8
    
    for m_block in nl.affine_range(M // (128 * NUM_BLOCK_M)):
        for n_block in nl.affine_range(N // (512 * NUM_BLOCK_N)):
            result_tiles = nl.ndarray(shape=(NUM_BLOCK_M, NUM_BLOCK_N, 128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            
            for m in nl.affine_range(NUM_BLOCK_M):
                for n in nl.affine_range(NUM_BLOCK_N):
                    nisa.dma_copy(result_tiles[m, n], nl.hbm_ref(output[m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]))
            
            lhs_tiles = nl.ndarray(shape=(NUM_BLOCK_K, NUM_BLOCK_M, 128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_tiles = nl.ndarray(shape=(NUM_BLOCK_K, NUM_BLOCK_N, 128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            
            for k_block in nl.affine_range(K // (128 * NUM_BLOCK_K)):
                for k in nl.sequential_range(NUM_BLOCK_K):
                    for m in nl.affine_range(NUM_BLOCK_M):
                        nisa.dma_copy(lhs_tiles[k, m], nl.hbm_ref(lhs_T[k_block*128*NUM_BLOCK_K+k*128:(k_block*128*NUM_BLOCK_K+k+1)*128, m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128]))
                    for n in nl.affine_range(NUM_BLOCK_N):
                        nisa.dma_copy(rhs_tiles[k, n], nl.hbm_ref(rhs[k_block*128*NUM_BLOCK_K+k*128:(k_block*128*NUM_BLOCK_K+k+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]))
                
                for k in nl.sequential_range(NUM_BLOCK_K):
                    for m in nl.affine_range(NUM_BLOCK_M):
                        for n in nl.affine_range(NUM_BLOCK_N):
                            nisa.nc_matmul(psum_buf, lhs_tiles[k, m], rhs_tiles[k, n])
                            nisa.tensor_copy(result_tiles[m, n], psum_buf)
            
            for m in nl.affine_range(NUM_BLOCK_M):
                for n in nl.affine_range(NUM_BLOCK_N):
                    nisa.dma_copy(nl.hbm_ref(output[m_block*128*NUM_BLOCK_M+m*128:(m_block*128*NUM_BLOCK_M+m+1)*128, n_block*512*NUM_BLOCK_N+n*512:(n_block*512*NUM_BLOCK_N+n+1)*512]), result_tiles[m, n])
```

## bert_model.py

```python
import tensorflow as tf
from tensorflow.neuron import fuse

# Fusing encoder with compiler arguments
fuser = fuse(compiler_args=['--fp32-cast', 'matmult'], timeout=360000)
bert.encoder = fuser(bert.encoder)
```

```python
# Layer normalization implementation
def layer_norm(self, input_tensor, layer_name, force_float32=False):
    dtype = tf.float32 if force_float32 else self.layer_norm_dtype
    gamma = dtype.as_numpy_dtype(self.weights_dict['bert/{}/LayerNorm/gamma:0'.format(layer_name)])
    beta = dtype.as_numpy_dtype(self.weights_dict['bert/{}/LayerNorm/beta:0'.format(layer_name)])
    with tf.name_scope('bert/{}/LayerNorm'.format(layer_name)):
        input_tensor = tf.cast(input_tensor, dtype)
        mean = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True, name='mean')
        residuals = tf.subtract(input_tensor, mean, name='residuals')
        var = tf.reduce_mean(residuals * residuals, axis=[-1], keepdims=True, name='var')
        rsqrt = tf.rsqrt(var + dtype.as_numpy_dtype(self.eps))
        norm_output = tf.multiply(residuals, rsqrt, name='normalized')
        output_tensor = norm_output * gamma + beta
        output_tensor = tf.cast(output_tensor, self.dtype)
    return output_tensor
```

```python
# Self-attention implementation
def self_attention(self, input_tensor, bias_tensor, layer_name):
    query_kernel = self.weights_dict['bert/encoder/{}/attention/self/query/kernel:0'.format(layer_name)] * 0.125
    query_bias = self.weights_dict['bert/encoder/{}/attention/self/query/bias:0'.format(layer_name)] * 0.125
    key_kernel = self.weights_dict['bert/encoder/{}/attention/self/key/kernel:0'.format(layer_name)]
    key_bias = self.weights_dict['bert/encoder/{}/attention/self/key/bias:0'.format(layer_name)]
    value_kernel = self.weights_dict['bert/encoder/{}/attention/self/value/kernel:0'.format(layer_name)]
    value_bias = self.weights_dict['bert/encoder/{}/attention/self/value/bias:0'.format(layer_name)]
    output_kernel = self.weights_dict['bert/encoder/{}/attention/output/dense/kernel:0'.format(layer_name)]
    output_bias = self.weights_dict['bert/encoder/{}/attention/output/dense/bias:0'.format(layer_name)]
    with tf.name_scope('bert/encoder/{}/attention/self'.format(layer_name)):
        matmul = tf.matmul(input_tensor, query_kernel.astype(self.dtype.as_numpy_dtype))
        query = tf.nn.bias_add(matmul, query_bias.astype(self.dtype.as_numpy_dtype))
        query_r = tf.reshape(query, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
        query_rt = tf.transpose(query_r, [0, 2, 1, 3])
        matmul = tf.matmul(input_tensor, key_kernel.astype(self.dtype.as_numpy_dtype))
        key = tf.nn.bias_add(matmul, key_bias.astype(self.dtype.as_numpy_dtype))
        key_r = tf.reshape(key, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
        key_rt = tf.transpose(key_r, [0, 2, 1, 3])
        query_key = tf.matmul(query_rt, key_rt, transpose_b=True)
        bias_query_key = tf.add(query_key, bias_tensor)
        softmax_weights = tf.nn.softmax(bias_query_key)
        matmul = tf.matmul(input_tensor, value_kernel.astype(self.dtype.as_numpy_dtype))
        value = tf.nn.bias_add(matmul, value_bias.astype(self.dtype.as_numpy_dtype))
        value_r = tf.reshape(value, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
        value_rt = tf.transpose(value_r, [0, 2, 3, 1])
        weighted_value_rt = tf.matmul(softmax_weights, value_rt, transpose_b=True)
        weighted_value_r = tf.transpose(weighted_value_rt, [0, 2, 1, 3])
        weighted_value = tf.reshape(weighted_value_r, [self.batch_size * self.seq_len, self.hid_size])
    with tf.name_scope('bert/encoder/{}/attention/output'.format(layer_name)):
        matmul = tf.matmul(weighted_value, output_kernel.astype(self.dtype.as_numpy_dtype))
        unnorm_output = tf.nn.bias_add(matmul, output_bias.astype(self.dtype.as_numpy_dtype))
        output_tensor = tf.add(input_tensor, unnorm_output)
    return output_tensor
```

```python
# Fully connected layer implementation
def fully_connected(self, input_tensor, layer_name):
    inter_kernel = self.weights_dict['bert/encoder/{}/intermediate/dense/kernel:0'.format(layer_name)]
    inter_bias = self.weights_dict['bert/encoder/{}/intermediate/dense/bias:0'.format(layer_name)]
    out_kernel = self.weights_dict['bert/encoder/{}/output/dense/kernel:0'.format(layer_name)]
    out_bias = self.weights_dict['bert/encoder/{}/output/dense/bias:0'.format(layer_name)]
    with tf.name_scope('bert/encoder/{}/fully_connected/intermediate/dense'.format(layer_name)):
        matmul = tf.matmul(input_tensor, inter_kernel.astype(self.dtype.as_numpy_dtype))
        bias_add = tf.nn.bias_add(matmul, inter_bias.astype(self.dtype.as_numpy_dtype))
        gelu = self.gelu_sigmoid(bias_add) if self.crude_gelu else self.gelu_tanh(bias_add)
    with tf.name_scope('bert/encoder/{}/fully_connected/output/dense'.format(layer_name)):
        matmul = tf.matmul(gelu, out_kernel.astype(self.dtype.as_numpy_dtype))
        bias_add = tf.nn.bias_add(matmul, out_bias.astype(self.dtype.as_numpy_dtype))
        output_tensor = bias_add + input_tensor
    return output_tensor
```

## ssd300_model.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn
from functools import partial
import numpy as np

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


def tf_SequentialBottleneck(tensor, seq, resnet):
    with tf.name_scope('{}.Sequential'.format(seq)):
        for idx, module in enumerate(resnet[seq]):
            with tf.name_scope('{}.BasicBlock'.format(idx)):
                tensor = tf_Bottleneck(tensor, module)
    return tensor


def tf_bbox_view(detection_feed, modules, ndim):
    results = []
    for idx, (tensor, mod) in enumerate(zip(detection_feed, modules)):
        with tf.name_scope('branch{}'.format(idx)):
            tensor = tf_Conv2d(tensor, mod)
            tensor = tf.transpose(tensor, [0, 3, 1, 2])
            tensor = tf.cast(tensor, tf.float32)

            shape = tensor.shape.as_list()
            batch_size = -1 if shape[0] is None else shape[0]
            new_shape = [batch_size, ndim, np.prod(shape[1:]) // ndim]
            results.append(tf.reshape(tensor, new_shape))
    tensor = tf.concat(results, axis=-1)
    return tensor


def tf_feature_extractor(input_tensor, resnet):
    with tf.name_scope('FeatureExtractor'):
        with tf.name_scope('0.Conv2d'):
            tensor = tf_Conv2d(input_tensor, resnet[0], first_conv=True)
        with tf.name_scope('1.BatchNorm2d'):
            tensor = tf_BatchNorm2d(tensor, resnet[1])
        with tf.name_scope('2.ReLU'):
            tensor = tf.nn.relu(tensor)
        with tf.name_scope('3.MaxPool2d'):
            tensor = tf_MaxPool2d(tensor, resnet[3])
        tensor = tf_SequentialBottleneck(tensor, 4, resnet)
        tensor = tf_SequentialBottleneck(tensor, 5, resnet)
        tensor = tf_SequentialBottleneck(tensor, 6, resnet)
        tensor = tf.cast(tensor, tf.float16)
    return tensor


def tf_box_predictor(tensor, ssd300_torch):
    with tf.name_scope('BoxPredictor'):
        detection_feed = [tensor]
        for idx, block in enumerate(ssd300_torch.additional_blocks):
            with tf.name_scope('{}.Sequential'.format(idx)):
                tensor = tf_Conv2d(tensor, block[0])
                tensor = tf_BatchNorm2d(tensor, block[1])
                tensor = tf.nn.relu(tensor)
                tensor = tf_Conv2d(tensor, block[3])
                tensor = tf_BatchNorm2d(tensor, block[4])
                tensor = tf.nn.relu(tensor)
                detection_feed.append(tensor)
        with tf.name_scope('Boxes'):
            loc = tf_bbox_view(detection_feed, ssd300_torch.loc, ndim=4)
        with tf.name_scope('Probabilities'):
            conf = tf_bbox_view(detection_feed, ssd300_torch.conf, ndim=ssd300_torch.label_num)
    return loc, conf


@tfn.fuse(batch_size=1, dynamic_batch_size=True)
def tf_ssd300(input_tensor, ssd300_torch):
    with tf.name_scope('SSD300'):
        tensor = tf_feature_extractor(input_tensor, ssd300_torch.feature_extractor.feature_extractor)
        loc, conf = tf_box_predictor(tensor, ssd300_torch)
    return loc, conf


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


def select_nms_outputs(input_tensors):
    boxes_xywh, scores, classes, valid_detections = input_tensors
    return boxes_xywh[:valid_detections], scores[:valid_detections], classes[:valid_detections]


def postprocessor(ploc_ts, plabel_ts, bbox_scale_hw_ts, scale_xy, scale_wh, dboxes_xywh):
    with tf.name_scope('Postprocessor'):
        ploc_ts = tf.cast(ploc_ts, tf.float32)
        plabel_ts = tf.cast(plabel_ts, tf.float32)
        bboxes_ts, probs_ts = scale_back_batch(ploc_ts, plabel_ts, scale_xy, scale_wh, dboxes_xywh)
        bboxes_ts = bboxes_ts[:, :, tf.newaxis, :]
        probs_ts = probs_ts[:, :, 1:]
        nms_outputs = tf.image.combined_non_max_suppression(
            bboxes_ts,
            probs_ts,
            max_output_size_per_class=200,
            max_total_size=200,
            iou_threshold=0.5,
            score_threshold=0.05,
            pad_per_class=False,
            clip_boxes=False,
            name='CombinedNonMaxSuppression',
        )
        nmsed_boxes_x0y0x1y1, nmsed_scores, nmsed_classes, valid_detections = nms_outputs
        nmsed_boxes_x0y0 = nmsed_boxes_x0y0x1y1[..., :2]
        nmsed_boxes_x1y1 = nmsed_boxes_x0y0x1y1[..., 2:]
        bbox_scale_hw_ts = bbox_scale_hw_ts[:, tf.newaxis, :]
        nmsed_boxes_xy = nmsed_boxes_x0y0 * bbox_scale_hw_ts
        nmsed_boxes_wh = (nmsed_boxes_x1y1 - nmsed_boxes_x0y0) * bbox_scale_hw_ts
        nmsed_boxes_xywh = tf.concat([nmsed_boxes_xy, nmsed_boxes_wh], axis=-1)
        nmsed_boxes_xywh, nmsed_scores, nmsed_classes = tf.map_fn(
            select_nms_outputs, (nmsed_boxes_xywh, nmsed_scores, nmsed_classes, valid_detections),
            dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return nmsed_boxes_xywh, nmsed_scores, nmsed_classes
```

## hf_llama3_8B_SFT_LORA.rst

```python
import transformers

tokenizer_path="llama3_tokenizer"
model_weights_path="llama3-8B_hf_weights"
model_id = "meta-llama/Meta-Llama-3-8B"

t = transformers.AutoTokenizer.from_pretrained(model_id)
t.save_pretrained(tokenizer_path)

m = transformers.AutoModelForCausalLM.from_pretrained(model_id)
m.save_pretrained(model_weights_path)
```

```yaml
exp_manager:
    resume_from_checkpoint: /pretrained_ckpt

data:
    train_dir: /example_datasets/llama3_8b/training.jsonl
    val_dir: /example_datasets/llama3_8b/validation.json
    dev_choose_samples: 2250
    seq_length: 4096
    tokenizer:
        type: /llama3_tokenizer

model:
    weight_init_only: True

model_alignment_strategy:
    sft:
        packing: True
    peft:
        lora_rank: 16
        lora_alpha: 32
        lora_dropout: 0.05
        lora_bias: "none"
        lora_verbose: True
        target_modules: ["qkv_proj"]
```

## ssd300_model.py

```python
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
import tensorflow.neuron as tfn
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


def tf_SequentialBottleneck(tensor, seq, resnet):
    with tf.name_scope('{}.Sequential'.format(seq)):
        for idx, module in enumerate(resnet[seq]):
            with tf.name_scope('{}.BasicBlock'.format(idx)):
                tensor = tf_Bottleneck(tensor, module)
    return tensor


def tf_bbox_view(detection_feed, modules, ndim):
    results = []
    for idx, (tensor, mod) in enumerate(zip(detection_feed, modules)):
        with tf.name_scope('branch{}'.format(idx)):
            tensor = tf_Conv2d(tensor, mod)
            tensor = tf.transpose(tensor, [0, 3, 1, 2])
            tensor = tf.cast(tensor, tf.float32)

            shape = tensor.shape.as_list()
            batch_size = -1 if shape[0] is None else shape[0]
            new_shape = [batch_size, ndim, np.prod(shape[1:]) // ndim]
            results.append(tf.reshape(tensor, new_shape))
    tensor = tf.concat(results, axis=-1)
    return tensor


def tf_feature_extractor(input_tensor, resnet):
    with tf.name_scope('FeatureExtractor'):
        with tf.name_scope('0.Conv2d'):
            tensor = tf_Conv2d(input_tensor, resnet[0], first_conv=True)
        with tf.name_scope('1.BatchNorm2d'):
            tensor = tf_BatchNorm2d(tensor, resnet[1])
        with tf.name_scope('2.ReLU'):
            tensor = tf.nn.relu(tensor)
        with tf.name_scope('3.MaxPool2d'):
            tensor = tf_MaxPool2d(tensor, resnet[3])
        tensor = tf_SequentialBottleneck(tensor, 4, resnet)
        tensor = tf_SequentialBottleneck(tensor, 5, resnet)
        tensor = tf_SequentialBottleneck(tensor, 6, resnet)
        tensor = tf.cast(tensor, tf.float16)
    return tensor


def tf_box_predictor(tensor, ssd300_torch):
    with tf.name_scope('BoxPredictor'):
        detection_feed = [tensor]
        for idx, block in enumerate(ssd300_torch.additional_blocks):
            with tf.name_scope('{}.Sequential'.format(idx)):
                tensor = tf_Conv2d(tensor, block[0])
                tensor = tf_BatchNorm2d(tensor, block[1])
                tensor = tf.nn.relu(tensor)
                tensor = tf_Conv2d(tensor, block[3])
                tensor = tf_BatchNorm2d(tensor, block[4])
                tensor = tf.nn.relu(tensor)
                detection_feed.append(tensor)
        with tf.name_scope('Boxes'):
            loc = tf_bbox_view(detection_feed, ssd300_torch.loc, ndim=4)
        with tf.name_scope('Probabilities'):
            conf = tf_bbox_view(detection_feed, ssd300_torch.conf, ndim=ssd300_torch.label_num)
    return loc, conf


@tfn.fuse(batch_size=1, dynamic_batch_size=True)
def tf_ssd300(input_tensor, ssd300_torch):
    with tf.name_scope('SSD300'):
        tensor = tf_feature_extractor(input_tensor, ssd300_torch.feature_extractor.feature_extractor)
        loc, conf = tf_box_predictor(tensor, ssd300_torch)
    return loc, conf


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


def select_nms_outputs(input_tensors):
    boxes_xywh, scores, classes, valid_detections = input_tensors
    return boxes_xywh[:valid_detections], scores[:valid_detections], classes[:valid_detections]


def postprocessor(ploc_ts, plabel_ts, bbox_scale_hw_ts, scale_xy, scale_wh, dboxes_xywh):
    with tf.name_scope('Postprocessor'):
        ploc_ts = tf.cast(ploc_ts, tf.float32)
        plabel_ts = tf.cast(plabel_ts, tf.float32)
        bboxes_ts, probs_ts = scale_back_batch(ploc_ts, plabel_ts, scale_xy, scale_wh, dboxes_xywh)
        bboxes_ts = bboxes_ts[:, :, tf.newaxis, :]
        probs_ts = probs_ts[:, :, 1:]
        nms_outputs = tf.image.combined_non_max_suppression(
            bboxes_ts,
            probs_ts,
            max_output_size_per_class=200,
            max_total_size=200,
            iou_threshold=0.5,
            score_threshold=0.05,
            pad_per_class=False,
            clip_boxes=False,
            name='CombinedNonMaxSuppression',
        )
        nmsed_boxes_x0y0x1y1, nmsed_scores, nmsed_classes, valid_detections = nms_outputs
        nmsed_boxes_x0y0 = nmsed_boxes_x0y0x1y1[..., :2]
        nmsed_boxes_x1y1 = nmsed_boxes_x0y0x1y1[..., 2:]
        bbox_scale_hw_ts = bbox_scale_hw_ts[:, tf.newaxis, :]
        nmsed_boxes_xy = nmsed_boxes_x0y0 * bbox_scale_hw_ts
        nmsed_boxes_wh = (nmsed_boxes_x1y1 - nmsed_boxes_x0y0) * bbox_scale_hw_ts
        nmsed_boxes_xywh = tf.concat([nmsed_boxes_xy, nmsed_boxes_wh], axis=-1)
        nmsed_boxes_xywh, nmsed_scores, nmsed_classes = tf.map_fn(
            select_nms_outputs, (nmsed_boxes_xywh, nmsed_scores, nmsed_classes, valid_detections),
            dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return nmsed_boxes_xywh, nmsed_scores, nmsed_classes
```

## migrate-from-tnx-to-nxdi.rst

```python
import os

# Force vLLM framework to use neuronx-distributed-inference
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
```

```python
from neuronx_distributed.inference import NeuronConfig, NeuronLlamaForCausalLM, LlamaInferenceConfig
from neuronx_distributed.inference.config import load_pretrained_config

model_path = "/home/ubuntu/models/open_llama_3b"
compiled_model_path = "/home/ubuntu/compiled_models/open_llama_3b"

neuron_config = NeuronConfig(
    batch_size=1,
    tp_degree=8,
    seq_len=128
)

config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path)
)

model = NeuronLlamaForCausalLM(model_path, config)

# Compile the model, shard the weights, and save to the given path.
model.compile(compiled_model_path)
```

## tiling-overview.rst

```python
import nki.isa as nisa
import nki.language as nl
import nki

# The hardware supports up to 128 partitions
P_DIM = nki.language.tile_size.pmax

# allocating memory for input and output tiles
# note that memory allocation does not initialize
in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)

# process first tile from input to output
nki.isa.dma_copy(dst=in_tile, src=input[0:P_DIM, 0:256])
nki.isa.reciprocal(dst=out_tile, data=in_tile)
nki.isa.dma_copy(dst=output[0:P_DIM, 0:256], src=out_tile)

# process second tile
nki.isa.dma_copy(dst=in_tile, src=input[P_DIM:256, 0:256])
nki.isa.reciprocal(dst=out_tile, data=in_tile)
nki.isa.dma_copy(dst=output[P_DIM:256, 0:256], src=out_tile)
```

```python
# allocate memory for input and output tiles
in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
# process tiles
for i in range(input.shape[0] // P_DIM):
    s = nl.ds(i * P_DIM, P_DIM) # equivalent to i * P_DIM : (i + 1) * P_DIM
    nki.isa.dma_copy(dst=in_tile, src=input[s, 0:256])
    nki.isa.reciprocal(dst=out_tile, data=in_tile)
    nki.isa.dma_copy(dst=output[s, 0:256], src=out_tile)
```

```python
import nki.isa as nisa
import nki.language as nl
import nki

# The hardware supports up to 128 partitions
P_DIM = nki.language.tile_size.pmax

@nki.jit
def tensor_kernel(in_tensor):
    """NKI kernel to compute elementwise reciprocal of an input tensor
    Args:
        in_tensor: an input tensor of shape [128,512]
    Returns:
        out_tensor: an output tensor of shape [128,512]
    """
    X_SIZE = 128
    Y_SIZE = 512
    
    # allocate space for the result
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    # allocate space for tile memory
    in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
    out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)

    # Process first tile
    nki.isa.dma_copy(dst=in_tile, src=in_tensor[0:P_DIM, 0:256])
    nki.isa.reciprocal(dst=out_tile, data=in_tile)
    nki.isa.dma_copy(dst=out_tensor[0:P_DIM, 0:256], src=out_tile)
    
    return out_tensor
```

```python
import nki.isa as nisa
import nki.language as nl
import nki

# The hardware supports up to 128 partitions
P_DIM = nki.language.tile_size.pmax

@nki.jit
def tensor_exp_kernel_(in_tensor):
    """NKI kernel to compute elementwise exponential of an input tensor
    Args:
        in_tensor: an input tensor of shape [256,512]
    Returns:
        out_tensor: an output tensor of shape [256,512]
    """
    X_SIZE = 128
    Y_SIZE = 512
    assert in_tensor.shape == (X_SIZE, Y_SIZE)
    # allocate space for the result
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    # allocate space for tile memory
    in_tile = nl.ndarray((P_DIM, Y_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    out_tile = nl.ndarray((P_DIM, Y_SIZE), dtype=nl.float32, buffer=nl.sbuf)

    for k in nl.affine_range(in_tensor.shape[0] / nl.tile_size.pmax):
        # Generate tensor indices for the input/output tensors
        p_start = k * nl.tile_size.pmax
        i_p = nl.ds(p_start, nl.tile_size.pmax)

        # Process tile
        nki.isa.dma_copy(dst=in_tile, src=in_tensor[i_p, :])
        nki.isa.reciprocal(dst=out_tile, data=in_tile)
        nki.isa.dma_copy(dst=out_tensor[i_p, :], src=out_tile)
    
    return out_tensor
```

```python
import nki.isa as nisa
import nki.language as nl
import nki
import math

# The hardware supports up to 128 partitions
P_DIM = nki.language.tile_size.pmax

@nki.jit
def tensor_exp_kernel_(in_tensor):
    """NKI kernel to compute elementwise exponential of an input tensor
    Args:
        in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
    Returns:
        out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
    """

    sz_p, sz_f = in_tensor.shape
    assert sz_f < nl.tile_size.total_available_sbuf_size
    
    # allocate space for the result
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    # allocate space for tile memory
    in_tile = nl.ndarray((P_DIM, sz_f), dtype=nl.float32, buffer=nl.sbuf)
    out_tile = nl.ndarray((P_DIM, sz_f), dtype=nl.float32, buffer=nl.sbuf)
    
    for p in nl.affine_range(math.ceil(sz_p / P_DIM)):
        # Generate tensor indices for the input/output tensors
        p_start = p * P_DIM
        p_end = p_start + P_DIM
        i_p = slice(p_start, min(p_end, sz_p)) # same as nl.ds(p_start, min(p_end, sz_p) - p_start)

        # Process tile
        nki.isa.dma_copy(dst=in_tile, src=in_tensor[i_p, :])
        nki.isa.reciprocal(dst=out_tile, data=in_tile)
        nki.isa.dma_copy(dst=out_tensor[i_p, :], src=out_tile)
        
    return out_tensor
```

## trn2-llama3.1-405b-tutorial.rst

```python
import os
import torch

from vllm import LLM, SamplingParams

# Force vLLM framework to use neuronx-distributed-inference
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

model_path = "/home/ubuntu/models/Llama-3.1-405B-Instruct/"


def run_llama_generate():
    # Initialize vLLM.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=64,
        max_num_seqs=1,
        max_model_len=2048,
        block_size=2048,
        dtype=torch.bfloat16,
        enable_prefix_caching=False,
        additional_config={
            "override_neuron_config": {
                "skip_warmup": True,
                "max_context_length": 1024,
            },
        },
    )

    # Run vLLM to generate outputs.
    prompts = ["I believe the meaning of life is"]
    sampling_params = SamplingParams(top_k=50)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```python
import copy
import os
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig, NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

model_path = "/home/ubuntu/models/llama-3.1-405b-Instruct/"
draft_model_path = "/home/ubuntu/models/EAGLE-llama-3-405b/"
compiled_model_path = "/home/ubuntu/neuron_models/llama-3-405b-instruct-EAGLE/"

# Set environment variables for Trn2.
os.environ["XLA_DENSE_GATHER_FACTOR"] = "0"
os.environ["NEURON_RT_EXEC_TIMEOUT"] = "600"

def run_llama_generate():
    top_k = 1
    do_sample = False

    # Initialize tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize target model config.
    neuron_config = NeuronConfig(
        torch_dtype=torch.bfloat16,
        tp_degree=64,
        batch_size=1,
        max_context_length=1024,
        seq_len=2048,
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=False,
            do_sample=do_sample,
            top_k=top_k
        ),
        enable_eagle_speculation=True,
        enable_fused_speculation=True,
        speculation_length=6,
        trace_tokengen_model=False,
        enable_bucketing=True,
        fused_qkv=True,
        sequence_parallel_enabled=True,
        attn_kernel_enabled=True,
        qkv_kernel_enabled=True,
        mlp_kernel_enabled=True,
        cc_pipeline_tiling_factor=1,
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Initialize draft model config.
    draft_neuron_config = copy.deepcopy(neuron_config)
    draft_neuron_config.trace_tokengen_model = True
    draft_neuron_config.enable_fused_speculation = False
    draft_neuron_config.is_eagle_draft = True
    draft_config = LlamaInferenceConfig(
        draft_neuron_config,
        load_config=load_pretrained_config(draft_model_path)
    )

    # Initialize fused speculation config.
    fused_spec_config = FusedSpecNeuronConfig(
        NeuronLlamaForCausalLM._model_cls,
        draft_config=draft_config,
        draft_model_path=draft_model_path,
    )
    config.fused_spec_config = fused_spec_config
        
    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(compiled_model_path)
    tokenizer.save_pretrained(compiled_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlamaForCausalLM(compiled_model_path)
    model.load(compiled_model_path)
    tokenizer = AutoTokenizer.from_pretrained(compiled_model_path)

    # Initialize generation config.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": do_sample,
        "top_k": top_k,
        "pad_token_id": 0,
        "prompt_lookup_num_tokens": neuron_config.speculation_length,
    }
    generation_config.update(**generation_config_kwargs)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is"]
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
```

## test_nki_isa_dma_copy.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def nki_dma_copy(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
  nisa.dma_copy(dst=b, src=a)
  return b


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


@nki.jit(mode='simulation')
def nki_dma_copy_swdge(in_tensor):
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  nisa.dma_copy(dst=out_tensor, src=in_tensor, dge_mode=nisa.dge_mode.swdge)
  return out_tensor


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

## autobucketing-dev-guide.rst

```python
import torch
from typing import List

def sequence_length_bucket_kernel(tensor_list: List[torch.Tensor]):
    x = tensor_list[0]
    bucket_dim = 1
    x_shape = x.shape
    tensor_sequence_length = x_shape[bucket_dim]
    batch_size = x_shape[bucket_dim - 1]
    buckets = [128, 512]
    idx = 0
    num_inputs = 3
    bucket = buckets[0]
    reshaped_tensors: List[torch.Tensor] = []
    bucket_idx = 0
    for idx, bucket in enumerate(buckets):
        if tensor_sequence_length <= bucket:
            bucket_idx = idx
            for tensor in tensor_list:
                if num_inputs == 0:
                    break
                delta = bucket - tensor_sequence_length
                padding_shape: List[int] = [batch_size, delta]
                zeros = torch.zeros(padding_shape, dtype=x.dtype)
                reshaped_tensors.append(torch.cat([tensor, zeros], dim=bucket_dim))
                num_inputs -= 1
            break
    return reshaped_tensors, torch.tensor([bucket_idx])

def get_bucket_kernel(*_):
    bk = torch.jit.script(sequence_length_bucket_kernel)
    return bk
```

```python
import torch
from typing import List

def state_preprocessor(shapes_collection: List[List[List[int]]], states: List[torch.Tensor], bucket_idx_tensor: torch.Tensor) -> List[torch.Tensor]:
    bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
    shapes = shapes_collection[bucket_idx]
    sliced_state_tensors = []
    
    for i in range(len(shapes)):
        expected_shape = shapes[i]
        state_tensor = states[i]
        state_tensor_shape = state_tensor.shape
        for j, npos in enumerate(expected_shape):
            state_tensor_dim_length = state_tensor_shape[j]
            state_tensor = torch.ops.aten.slice(state_tensor, dim=j, start=state_tensor_dim_length-npos, end=state_tensor_dim_length)
        sliced_state_tensors.append(state_tensor)
    
    return sliced_state_tensors

def get_state_preprocessor():
    sp = torch.jit.script(state_preprocessor)
    return sp
```

```python
import torch
import torch_neuronx
from typing import List

bucket_config = torch_neuronx.BucketModelConfig(get_bucket_kernel, shared_state_buffer_preprocessor=get_state_preprocessor)
```

```python
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

def encode(tokenizer, *inputs, max_length=128, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
    )

def get_bert_model(*args):
    name = "bert-base-cased-finetuned-mrpc"
    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    return model, {}

def sequence_length_bucket_kernel(tensor_list: List[torch.Tensor]):
    x = tensor_list[0]
    bucket_dim = 1
    x_shape = x.shape
    tensor_sequence_length = x_shape[bucket_dim]
    batch_size = x_shape[bucket_dim - 1]
    buckets = [128, 512]
    idx = 0
    num_inputs = 3
    bucket = buckets[0]
    reshaped_tensors: List[torch.Tensor] = []
    bucket_idx = 0
    for idx, bucket in enumerate(buckets):
        if tensor_sequence_length <= bucket:
            bucket_idx = idx
            for tensor in tensor_list:
                if num_inputs == 0:
                    break
                delta = bucket - tensor_sequence_length
                padding_shape: List[int] = [batch_size, delta]
                zeros = torch.zeros(padding_shape, dtype=x.dtype)
                reshaped_tensors.append(torch.cat([tensor, zeros], dim=bucket_dim))
                num_inputs -= 1
            break
    return reshaped_tensors, torch.tensor([bucket_idx])

def get_bucket_kernel(*_):
    bk = torch.jit.script(sequence_length_bucket_kernel)
    return bk

bucket_config = torch_neuronx.BucketModelConfig(get_bucket_kernel)
bucket_trace_neuron = torch_neuronx.bucket_model_trace(get_bert_model, [paraphrase_s128, paraphrase_s512], bucket_config)
```

## performance-profiling-vllm.rst

```python
import transformers

model_id = "Qwen/Qwen3-8B-Base"
config = transformers.AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 4
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
output_dir = "4layer_qwen3"

model = transformers.AutoModelForCausalLM.from_pretrained(model_id, config=config)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

```python
import os
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

# Enable Neuron profiling via environment variables
os.environ['XLA_IR_DEBUG'] = "1"
os.environ['XLA_HLO_DEBUG'] = "1"
os.environ['NEURON_FRAMEWORK_DEBUG'] = "1"
os.environ['NEURON_RT_INSPECT_ENABLE'] = "1"
os.environ['NEURON_RT_INSPECT_SYSTEM_PROFILE'] = "1"
os.environ['NEURON_RT_INSPECT_DEVICE_PROFILE'] = "1"
os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'] = "./neuron_profiles"

from vllm import LLM, SamplingParams

prompts = [
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(top_k=1)

llm = LLM(
    model="4layer_qwen3",
    max_num_seqs=4,
    max_model_len=128,
    override_neuron_config={
        "enable_bucketing":False,
    },
    device="neuron",
    tensor_parallel_size=8)

outputs = llm.generate(prompts, sampling_params)
```

## neuron_profile_for_nki.rst

```python
import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "
```

## sdxl_base_1024_compile.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import math
import copy
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


class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple


# Compile text encoder with torch_neuronx.trace
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)
torch.jit.save(neuron_text_encoder, text_encoder_filename)


# Compile UNet with torch_neuronx.trace
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# Enable asynchronous and lazy loading
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

torch.jit.save(unet_neuron, unet_filename)


# Compile VAE decoder with torch_neuronx.trace
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder')
)

# Enable asynchronous loading
torch_neuronx.async_load(decoder_neuron)

torch.jit.save(decoder_neuron, decoder_filename)


# Compile VAE post_quant_conv with torch_neuronx.trace
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

# Enable asynchronous loading
torch_neuronx.async_load(post_quant_conv_neuron)

torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
```

## transformers-neuronx-developer-guide-for-continuous-batching.rst

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_num_seqs=8,
    # The max_model_len and block_size arguments are required to be same as max sequence length,
    # when targeting neuron device. Currently, this is a known limitation in continuous batching
    # support in transformers-neuronx.
    max_model_len=128,
    block_size=128,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection, or explicitly assigned.
    device="neuron",
    tensor_parallel_size=2)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_num_seqs=8,
    max_model_len=128,
    block_size=128,
    device="neuron",
    tensor_parallel_size=32,
    #Override or update the NeuronConfig
    override_neuron_config={"shard_over_sequence":True})
```

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    # The max_model_len, speculative_max_model_len, and block_size arguments are required to be same as max sequence length,
    # when targeting neuron device. Currently, this is a known limitation in continuous batching
    # support in transformers-neuronx.
    max_model_len=128,
    block_size=128,
    speculative_max_model_len=128,
    dtype="bfloat16",
    max_num_seqs=4,
    num_speculative_tokens=4,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection, or explicitly assigned.
    device="neuron",
    tensor_parallel_size=32,
    use_v2_block_manager=True,
)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## trace-vs-xla-lazytensor.rst

```python
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

# Create XLA device
device = xm.xla_device()

# Load example model and inputs to Neuron device
model = torch.nn.Sequential(
    torch.nn.Linear(784, 120),
    torch.nn.ReLU(),
    torch.nn.Linear(120, 10),
    torch.nn.Softmax(dim=-1),
)
model.eval()
model.to(device)
example = torch.rand((1, 784), device=device)

# Inference
with torch.no_grad():
    result = model(example)
    xm.mark_step()  # Compilation occurs here
    print(result.cpu())
```

```python
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

# Create XLA device
device = xm.xla_device()

# Load example model and inputs to Neuron device
model = torch.nn.Sequential(
    torch.nn.Embedding(num_embeddings=30522, embedding_dim=512),
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax(dim=-1),
)
model.eval()
model.to(device)

token_ids_1 = torch.tensor([
    [1, 28, 748, 0],
])
token_ids_2 = torch.tensor([
    [1, 13087, 10439, 1990, 18912, 0],
    [1, 12009, 7849, 2509, 3500, 0],
])

# Inference
with torch.no_grad():
    result = model(token_ids_1)
    xm.mark_step()
    print(result.cpu())

    result = model(token_ids_2)
    xm.mark_step()
    print(result.cpu())
```

```python
import torch
import torch_neuronx

# Create example model and inputs
model = torch.nn.Sequential(
    torch.nn.Linear(784, 120),
    torch.nn.ReLU(),
    torch.nn.Linear(120, 10),
    torch.nn.Softmax(dim=-1),
)
model.eval()
example = torch.rand((1, 784))

# Create fixed model trace
trace = torch_neuronx.trace(model, example)

# Inference
result = trace(example)
print(result)
```

```python
class TestModel(torch.nn.Module):
    def __init__(self, flag=1):
        super().__init__()
        self.flag = flag

    def forward(self, tensor):
        if self.flag:
            return tensor
        else:
            return tensor * 2
```

```python
import os
os.environ['PT_XLA_DEBUG_LEVEL'] = '2'
```

```python
import torch_xla
torch_xla._XLAC._set_allow_execution(False)
```

## training_llama_tp_zero1.rst

```python
from huggingface_hub import login
from transformers import AutoTokenizer

login(token='your_own_hugging_face_token')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')  
# For llama2 uncomment line below
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf') 

tokenizer.save_pretrained(".")
```

```python
from neuronx_distributed.parallel_layers import checkpointing

if global_step % every_n_steps_checkpoint == 0:
    state_dict = {
        "model": model.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "scheduler": scheduler.state_dict()
    }
    checkpointing.save(state_dict, flags.output_dir)
    optimizer.save_sharded_state_dict(flags.output_dir)
```

```python
from neuronx_distributed.parallel_layers import checkpointing

if global_step % every_n_steps_checkpoint == 0:
    state_dict = {
        "model": model.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "scheduler": scheduler.state_dict()
    }
    checkpointing.save(state_dict, flags.output_dir, down_cast_bf16=True)
```

```python
from neuronx_distributed.parallel_layers import checkpointing

if global_step % every_n_steps_checkpoint == 0:
    state_dict = {
        "model": model.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "scheduler": scheduler.state_dict()
    }
    checkpointing.save(state_dict, flags.output_dir, down_cast_bf16=True, save_xser=True)
```

```python
if global_step % every_n_steps_checkpoint == 0:
    optimizer.save_sharded_state_dict(flags.output_dir, num_workers_per_step=32)
```

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

## api-compilation-python-api.rst

```python
import torch
import torch_neuron

def foo(x, y):
    return 2 * x + y

# Run `foo` with the provided inputs and record the tensor operations
traced_foo = torch.neuron.trace(foo, (torch.rand(3), torch.rand(3)))

# `traced_foo` can now be run with the TorchScript interpreter or saved
# and loaded in a Python-free environment
torch.jit.save(traced_foo, 'foo.pt')
traced_foo = torch.jit.load('foo.pt')
```

```python
import torch
import torch_neuron
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x) + 1

n = Net()
n.eval()

inputs = torch.rand(1, 1, 3, 3)

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method
neuron_forward = torch.neuron.trace(n.forward, inputs)

# Trace a module (implicitly traces `forward`) and constructs a
# `ScriptModule` with a single `forward` method
neuron_net = torch.neuron.trace(n, inputs)
```

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
```

```python
import torch
import torch_neuron
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return {'conv': self.conv(x) + 1}

model = Model()
model.eval()

inputs = torch.rand(1, 1, 3, 3)

# use the strict=False kwarg to compile a model with dictionary outputs
# the model output format does not change
model_neuron = torch.neuron.trace(model, inputs, strict=False)
```

```python
import torch
import torch_neuron
from torchvision import models

# Load the model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input of batch size 1
image = torch.rand([1, 3, 224, 224])
model_neuron = torch.neuron.trace(model, image, dynamic_batch_size=True)

# Execute with a batch of 7 images
batch = torch.rand([7, 3, 224, 224])
results = model_neuron(batch)
```

```python
import torch
import torch_neuron
import torch.nn as nn

class ExampleConvolutionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x) + 1

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = ExampleConvolutionLayer()

    def forward(self, x):
        return self.layer(x) * 100

def subgraph_builder_function(node) -> bool:
    """Select if the node will be included in the Neuron graph"""

    # Node names are tuples of Module names.
    if 'ExampleConvolutionLayer' in node.name:
        return True

    # Ignore all operations not in the example convolution layer
    return False

model = Model()
model.eval()

inputs = torch.rand(1, 1, 3, 3)

# Log output shows that `aten::_convolution` and `aten::add` are compiled
# but `aten::mul` is not. This will seamlessly switch between Neuron/CPU
# execution in a single graph.
neuron_model = torch_neuron.trace(
    model,
    inputs,
    subgraph_builder_function=subgraph_builder_function
)
```

```python
import torch
import torch_neuron
from torchvision import models

# Load the model
model = models.resnet50(pretrained=True)
model.eval()

# Compile with an example input
image = torch.rand([1, 3, 224, 224])
#the models' output format does not change
model_neuron = torch.neuron.trace(model, image, separate_weights=True)
```

## pytorch-neuron-parallel-compile.rst

```python
import os

os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + "--cache_dir=<cache URL>"
```

```python
import os

os.environ['NEURON_CC_FLAGS'] = os.environ.get('NEURON_CC_FLAGS', '') + ' --retry_failed_compilation'
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
import time

# Timer usage
timer = neuronperf.Timer()
with timer:
    time.sleep(1)

total_duration = timer.total_duration("s")
durations = timer.durations("s")
timestamps = timer.timestamps()
```

```python
import neuronperf
import numpy as np

# Timestamp conversion
result_scalar = neuronperf.timestamp_convert(1, "s", "ms")
result_array = neuronperf.timestamp_convert(np.array([1, 2, 3]), "s", "ms")
```

```python
import neuronperf

# Model index creation
index = neuronperf.model_index.create("dummy_model.ext", model_name="dummy")
```

```python
import neuronperf

# Model index save and load
model_index = neuronperf.model_index.create("models/dummy.model", model_name="Dummy")
neuronperf.model_index.save(model_index, filename="dummy_index.json")
model_index_loaded = neuronperf.model_index.load("dummy_index.json")
neuronperf.model_index.delete("dummy_index.json")
```

```python
import neuronperf

# Model index copy
model_index = neuronperf.model_index.create("models/dummy.model", model_name="Dummy")
neuronperf.model_index.copy(model_index, "new_index.json", "new_models")
```

```python
import neuronperf

# Model index move
neuronperf.model_index.move("dummy_index.json", "new_index.json", "new_models")
```

```python
import neuronperf

# Model index append
model_indexes = [
    neuronperf.model_index.create(f"Dummy_{x}", model_name="Dummy") for x in range(10)
]
combined_index = neuronperf.model_index.append(*model_indexes)
```

```python
import neuronperf

# Model index filter
idx_1 = neuronperf.model_index.create("fake", performance_level=2, compile_s=1)
idx_2 = neuronperf.model_index.create("fake2", compile_s=2)
idx = neuronperf.model_index.append(idx_1, idx_2)

filtered = neuronperf.model_index.filter(idx, filename="fake")
filtered = neuronperf.model_index.filter(idx, performance_level=2)
```

## framework_custom_op.rst

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.randn(256, 1024, dtype=torch.float32).to(device)
b = torch.randn(256, 1024, dtype=torch.float32).to(device)
c = nki_tensor_add(a, b)
out = a * b * c
print(out)
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
print(jax_customop_tutorial(a, b))
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

## generative-llm-inference-with-neuron.rst

```python
# Vocabulary of tokens the model can parse. The position of each token in the 
# vocabulary is used as the token_id (an integer representing that token)
vocab = ["having", "I", "fun", "am", "learning", ".", "Neuron"]

# input token_ids: list of integers that represent the input tokens in this
# case: "I", "am", "having", "fun"
input_token_ids = [1, 3, 0, 2] 

# The LLM gets a vector of input token_ids, and generates a probability-distribution
# for what the output token_id should be (with a probability score for each token_id
# in the vocabulary)
output = LLM(input_token_ids) 

# by taking argmax on the output, we effectively perform a 'greedy sampling' process,
# i.e. we choose the token_id with the highest probability. Other sampling techniques
# also exist, e.g. Top-K. By choosing a probabilistic sampling method we enable the model
# to generate different outputs when called multiple times with the same input.
next_token_id = np.argmax(output) 

# map the token_id back into an output token
next_token = vocab[next_token_id]
```

```python
def generate(input_token_ids, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate): # decode loop
        output = LLM(input_token_ids) # model forward pass
    
        next_token_id = np.argmax(output) # greedy sampling
    
        if (next_token_id == EOS_TOK_ID):
            break # break if generated End Of Sentence (EOS)
    
        # append the prediction to the input, and continue to the next out_token
        input_token_ids.append(int(next_token_id)) 

    return input_token_ids[-n_tokens_to_generate :] # only return generated token_ids
```

```python
{
    "n_vocab": 50257, # number of tokens in our vocabulary
    "n_ctx": 2048, # maximum possible sequence length of the input
    "n_embd": 9216, # embedding dimension (determines the "width" of the network)
    "n_head": 72, # number of attention heads (n_embd must be divisible by n_head)
    "n_layer": 64 # number of layers (determines the "depth" of the network)
}
```

## spmd_tensor_addition.rst

```python
import neuronxcc.nki.language as nl
from neuronxcc.nki import nki_jit

@nki_jit
def nki_tensor_add_kernel_(a_input, b_input):
    # Allocate output tensor
    c_output = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype)
    
    # Get program ID for SPMD execution
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
    
    # Load tiles from input tensors
    a_tile = nl.load(a_input[offset_x + indices_x, offset_y + indices_y])
    b_tile = nl.load(b_input[offset_x + indices_x, offset_y + indices_y])
    
    # Compute sum
    c_tile = a_tile + b_tile
    
    # Store result back to output tensor
    nl.store(c_output[offset_x + indices_x, offset_y + indices_y], c_tile)
    
    return c_output
```

```python
def nki_tensor_add(a, b):
    # Calculate grid dimensions
    grid_x = (a.shape[0] + 127) // 128
    grid_y = (a.shape[1] + 511) // 512
    
    # Launch kernel with 2D grid
    return nki_tensor_add_kernel_[grid_x, grid_y](a, b)
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
    ...
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_in_acc[i][j] : 0.0;
    ...
}

torch::Tensor relu_backward(const torch::Tensor& t_grad, const torch::Tensor& t_in) {
    ...
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_grad_acc[i][j] : 0.0;
    ...
}

TORCH_LIBRARY(my_ops, m) {
    m.def("relu_forward", &relu_forward);
    m.def("relu_backward", &relu_backward);
}
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

## guide-torch-neuron-vs-torch-neuronx-inference.rst

```python
import torch
import torchvision
import torch_neuronx

model = torchvision.models.resnet50(pretrained=True).eval()
image = torch.rand(1, 3, 224, 224)

trace = torch_neuronx.trace(model, image)
```

## activation_memory_reduction_developer_guide.rst

```python
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

class GPTNeoXAttentionNxD(GPTNeoXAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            stride=3,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
```

```python
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from neuronx_distributed.parallel_layers import layer_norm

class GPTNeoXLayerNxD(GPTNeoXLayer):
    def __init__(self, config):
        super().__init__(config)
        self.input_layernorm = layer_norm.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            sequence_parallel_enabled=config.sequence_parallel_enabled
        )
        self.post_attention_layernorm = layer_norm.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            sequence_parallel_enabled=config.sequence_parallel_enabled
        )
```

```python
import torch

setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)
```

```python
import torch
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region

def allreduce_sequence_parallel_gradients(optimizer):
    """All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used."""
    grads = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        sequence_parallel_param = getattr(p, 'sequence_parallel_enabled', False)
                        if sequence_parallel_param:
                            grads.append(p.grad.data)
    for grad in grads:
        reduce_from_tensor_model_parallel_region(grad)
```

```python
from neuronx_distributed.parallel_layers.mappings import scatter_to_sequence_parallel_region

if self.config.sequence_parallel_enabled:
    hidden_states = hidden_states.transpose(0, 1).contiguous()
    hidden_states = scatter_to_sequence_parallel_region(hidden_states)
```

```python
if config.sequence_parallel_enabled:
    qkv = qkv.transpose(0, 1)

attn_output = attn_output.transpose(0, 1)
attn_output = self.dense(attn_output)
```

```python
from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region

if self.config.sequence_parallel_enabled:
    hidden_states = gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)
    hidden_states = hidden_states.transpose(0, 1).contiguous()
```

```python
import torch

if config.selective_activation_checkpointing_is_enabled:
    attn_output = torch.utils.checkpoint.checkpoint(self._attn, query, key, value, attention_mask, head_mask)
else:
    attn_output = self._attn(query, key, value, attention_mask, head_mask)
```

## neuron-cc-ops-pytorch.rst

```python
import torch.neuron
print(*torch.neuron.get_supported_operations(), sep='\n')
```

## tutorial-use-a-prebuilt-kernel.rst

```python
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import nki.language as nl
from nkilib.core.mlp.mlp import fused_mlp_isa_kernel
from nkilib.core.utils.common_types import ActFnType, NormType
```

```python
class MLPReference(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_output = torch.nn.functional.silu(self.gate_proj(hidden))
        up_output = self.up_proj(hidden)
        return self.down_proj(gate_output * up_output)
```

```python
model = MLPReference(hidden_size, intermediate_size, dtype=torch.bfloat16)
model.eval()
input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16) * 2
with torch.no_grad():
    reference_output = model(input_tensor)
```

```python
device = xm.xla_device()
nki_input = input_tensor.to(device=device, dtype=torch.bfloat16)
gate_w_xla = model.gate_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
up_w_xla = model.up_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
down_w_xla = model.down_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
```

```python
with torch.no_grad():
    nki_output = fused_mlp_isa_kernel[LNC_DEGREE](
        hidden=nki_input,
        gate_w=gate_w_xla,
        up_w=up_w_xla,
        down_w=down_w_xla,
        attn_output=None,
        norm_type=NormType.NO_NORM,
        dtype=nl.bfloat16,
        act_fn=ActFnType.SiLU,
    )
nki_output_cpu = nki_output.cpu()
```

```python
assert nki_output_cpu.shape == reference_output.shape, f"Shape mismatch: {nki_output_cpu.shape} vs {reference_output.shape}"
torch.testing.assert_close(nki_output_cpu, reference_output, rtol=1e-2, atol=1e-2)
```

## sd_15_512_compile.py

```python
import os
import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
```

```python
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
```

```python
def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled
```

```python
class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple
```

```python
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
```

```python
class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]
```

```python
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder, 
    emb, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(text_encoder_neuron)
torch.jit.save(text_encoder_neuron, text_encoder_filename)
```

```python
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, decoder_filename)
```

```python
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)
torch.jit.save(unet_neuron, unet_filename)
```

```python
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
```

```python
safety_model = torch_neuronx.trace(
    safety_model, 
    clip_input,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(safety_model)
torch.jit.save(safety_model, safety_model_neuron_filename)
```

## test_nki_isa_tensor_scalar_cumulative.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation")
def nki_tensor_scalar_cumulative_scalar(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  """Example 1: Basic usage of tensor scalar cumulative with scalar immediate values."""
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)
  src = nl.load(src_data[...])
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scalar_cumulative_vector(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  """Example 2: Basic usage of tensor scalar cumulative with vector immediate values."""
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)
  src = nl.load(src_data[...])
  imm0 = nl.load(imm0[...])
  imm1 = nl.load(imm1[...]) if imm1 else None
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scalar_cumulative_chain(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  """Example 3: Chain two tensor scalar cumulative operations together."""
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)
  src = nl.load(src_data[...])
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )
  
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=nisa.reduce_cmd.reduce
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scan(src_data, op, initial):
  """Example 4: Perform tensor scan using tensor scalar cumulative."""
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)
  src = nl.load(src_data[...])
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=nl.add,
    op1=op,
    imm0=np.float32(0.0),
    imm1=initial,
    reduce_cmd=nisa.reduce_cmd.load_reduce
  )
  
  nl.store(result_tensor, value=dst)
  return result_tensor
```

## how-to-use-fpem.rst

```python
import torch
from torch import nn
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

# Vision Model Definition
class VisionModel(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, config.vision_embedding_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Text Model Definition
class TextModel(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.embedding = nn.Linear(config.text_input_size, config.text_embedding_size)
        self.fusion = nn.Linear(
            config.vision_embedding_size + config.text_embedding_size,
            config.output_size
        )

    def forward(self, vision_features, text_input):
        text_features = self.embedding(text_input)
        combined = torch.cat([vision_features, text_features], dim=1)
        return self.fusion(combined)
```

```python
# Vision Model Wrapper - keeps output on device
class VisionModelWrapper(ModelWrapper):
    def __init__(self, config: InferenceConfig):
        super().__init__(
            config=config,
            model_cls=VisionModel,
            pipeline_execution=True,
            return_ranked_to_cpu=False,  # Keep output ranked for efficient pipeline
            tag="vision_model"
        )

    def input_generator(self):
        # Generate sample input for compilation
        x = torch.randn(
            self.neuron_config.batch_size,
            3,
            224,
            224
        )
        return [(x,)]

# Text Model Wrapper - returns final output to CPU
class TextModelWrapper(ModelWrapper):
    def __init__(self, config: InferenceConfig):
        super().__init__(
            config=config,
            model_cls=TextModel,
            pipeline_execution=True,
            return_ranked_to_cpu=True,  # Return final output to CPU
            tag="text_model"
        )

    def input_generator(self):
        # Generate sample inputs for compilation
        vision_features = torch.randn(
            self.neuron_config.batch_size,
            self.config.vision_embedding_size
        )
        text_input = torch.randn(
            self.neuron_config.batch_size,
            self.config.text_input_size
        )
        return [(vision_features, text_input)]
```

```python
# Application Classes
class VisionModelApp(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = VisionModelWrapper(config)
        self.models.append(self.model)

    def forward(self, x):
        return self.models[0].forward(x)

class TextModelApp(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = TextModelWrapper(config)
        self.models.append(self.model)

    def forward(self, vision_features, text_input):
        return self.models[0].forward(vision_features, text_input)
```

## nki_block_dimension_migration_guide.rst

```python
import nki
import nki.language as nl
from nki.language import bfloat16, float32

@nki.jit
def exp_func(inp):
    output = nl.ndarray((4, 8, 128, 2, 512), dtype=float32, 
      buffer=nl.shared_hbm)
    a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), dtype=float32, buffer=nl.sbuf)
    for i in range(4):
      for j in range(8):
        a[i, j] = nl.load(inp[i, j])
        a[i, j] = nl.exp(a[i, j])
        nl.store(output[i, j], value=result)
```

```python
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
    for i in range(8):
        nl.store(res[i], add_buf[i])
    return res

@nki.jit
def sb_blocks_migrated(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(128, 8, 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[0:128, i, 0:512] = nl.load(inp[i])
    for i in range(8):
        nl.store(res[i], add_buf[0:128, i, 0:512])
    return res
```

```python
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
        nl.store(res[i], add_buf[i])
    return res

@nki.jit
def sb_blocks_migrated(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    for i in range(8):
        add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
        add_buf[0:128, 0:512] = nl.load(inp[i])
        nl.store(res[i], add_buf[0:128, 0:512])
    return res

@nki.jit
def sb_blocks_migrated_incorrect(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[0:128, 0:512] = nl.load(inp[i])
        nl.store(res[i], add_buf[0:128, 0:512])
    return res
```

```python
def interleave_alloc_func(idx, pdim_size, fdim_size):
    idx, = idx
    start_partition = 0
    return (start_partition, (idx % 2) * fdim_size)

@nki.jit
def copy_func(inp):
    output = nl.ndarray((4, 128, 512), dtype=float32, buffer=nl.shared_hbm)
    a = nl.ndarray((4, nl.par_dim(128), 512), dtype=float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
    for i in range(4):
        a[i] = nl.load(inp[i])
        nl.store(output[i], value=a[i])
```

```python
def interleave_alloc_func(idx, pdim_size, fdim_size):
    assert idx == ()
    start_partition = 0
    return (start_partition, (idx % 2) * fdim_size)

@nki.compiler.skip_middle_end_transformations
@nki.jit
def exp_func(inp):
    output = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.ndarray((128, 2, 512), dtype=nl.float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
    for i in range(4):
        a[0:128, i % 2, 0:512] = nl.load(inp[i])
        nl.store(output[i], value=a[0:128, i % 2, 0:512])
```

## torch-neuronx-profiling-dev-guide.rst

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

import torch_neuronx
from torch_neuronx.experimental import profiler

os.environ["NEURON_CC_FLAGS"] = "--cache_dir=./compiler_cache"

# Declare 3-layer MLP Model
class MLP(nn.Module):
    def __init__(self, input_size = 10, output_size = 2, layers = [5, 5]):
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
# Basic profiler usage
torch.manual_seed(0)
device = xm.xla_device()

with torch_neuronx.experimental.profiler.profile(
    port=9012,
    profile_type='trace',
    ms_duration=15000) as profiler:
    
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()
    
    model.train()
    optimizer.zero_grad()
    train_x = torch.randn(1,10).to(device)
    train_label = torch.tensor([1]).to(device)
    
    loss = loss_fn(model(train_x), train_label)
    loss.backward()
    optimizer.step()
    
    xm.mark_step()
```

```python
# Profiler with named trace blocks
optimizer.zero_grad()
train_x = torch.randn(1,10).to(device)
train_label = torch.tensor([1]).to(device)

with xp.Trace("model_build"):
    loss = loss_fn(model(train_x), train_label)

with xp.Trace("loss_backward"):
    loss.backward()

with xp.Trace("optimizer_step"):
    optimizer.step()

xm.mark_step()
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
import neuron.nki as nki
import neuron.nki.language as nl
import math

@nki.jit
def nki_layernorm_kernel_v1(input_tensor, gamma, beta, epsilon):
  """
  LayerNorm kernel using nki.language APIs only.
  
  Args:
    input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
    gamma: 1D affine parameter of shape [hidden_size]
    beta: 1D affine parameter of shape [hidden_size]
    epsilon: Small constant for numerical stability
  
  Returns:
    Normalized output tensor of same shape as input_tensor
  """
  output_tensor = nl.zeros(input_tensor.shape, dtype=input_tensor.dtype)
  
  # Load gamma and beta, perform partition-axis broadcast
  shift_scale_tensor = nl.load(gamma)
  shift_scale_tensor = nl.broadcast_to(shift_scale_tensor, (nl.tile_size.pmax, input_tensor.shape[1]))
  
  beta_tensor = nl.load(beta)
  beta_tensor = nl.broadcast_to(beta_tensor, (nl.tile_size.pmax, input_tensor.shape[1]))
  
  # Compute loop over partition axis
  for i in range(math.ceil(input_tensor.shape[0] / nl.tile_size.pmax)):
    for i_p_io in nl.affine_range(nl.tile_size.pmax):
      # Load input tile with boundary guard
      input_tile = nl.load(
        input_tensor[i * nl.tile_size.pmax + i_p_io, :],
        mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
      )
      
      # Compute mean and variance
      mean = nl.mean(input_tile, axis=1)
      variance = nl.var(input_tile, axis=1)
      
      # Normalize: (x - mean) / sqrt(var + epsilon)
      normalized = (input_tile - mean) / nl.rsqrt(variance + epsilon)
      
      # Scale and shift: normalized * gamma + beta
      output_tile = normalized * shift_scale_tensor + beta_tensor
      
      # Store output with boundary guard
      nl.store(
        output_tensor[i * nl.tile_size.pmax + i_p_io, :],
        output_tile,
        mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
      )
  
  return output_tensor
```

```python
import neuron.nki as nki
import neuron.nki.language as nl
import neuron.nki.isa as nisa
import math

@nki.jit
def nki_layernorm_kernel_v2(input_tensor, gamma, beta, epsilon):
  """
  Optimized LayerNorm kernel using nki.isa APIs for mean/variance calculation
  and shift/scale operations.
  
  Args:
    input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
    gamma: 1D affine parameter of shape [hidden_size]
    beta: 1D affine parameter of shape [hidden_size]
    epsilon: Small constant for numerical stability
  
  Returns:
    Normalized output tensor of same shape as input_tensor
  """
  output_tensor = nl.zeros(input_tensor.shape, dtype=input_tensor.dtype)
  
  # Load gamma and beta
  gamma_loaded = nl.load(gamma)
  beta_loaded = nl.load(beta)
  
  # Compute loop over partition axis
  for i in range(math.ceil(input_tensor.shape[0] / nl.tile_size.pmax)):
    for i_p_io in nl.affine_range(nl.tile_size.pmax):
      # Load input tile
      input_tile = nl.load(
        input_tensor[i * nl.tile_size.pmax + i_p_io, :],
        mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
      )
      
      # Calculate mean and variance using bn_stats and bn_aggr
      mean_accum = nl.zeros((nl.tile_size.pmax,), dtype=input_tensor.dtype)
      var_accum = nl.zeros((nl.tile_size.pmax,), dtype=input_tensor.dtype)
      
      for j in range(math.ceil(input_tensor.shape[1] / nl.tile_size.bn_stats_fmax)):
        start_idx = j * nl.tile_size.bn_stats_fmax
        end_idx = min(start_idx + nl.tile_size.bn_stats_fmax, input_tensor.shape[1])
        
        input_slice = input_tile[:, start_idx:end_idx]
        
        # Use bn_stats to compute partial statistics
        stats = nisa.bn_stats(input_slice)
        mean_accum += stats[0]
        var_accum += stats[1]
      
      # Aggregate statistics
      mean = nisa.bn_aggr(mean_accum)
      variance = nisa.bn_aggr(var_accum)
      
      # Normalize and apply shift/scale in single instruction
      normalized = (input_tile - mean) / nl.rsqrt(variance + epsilon)
      output_tile = nisa.tensor_scalar(normalized, gamma_loaded, beta_loaded, "shift_scale")
      
      # Store output
      nl.store(
        output_tensor[i * nl.tile_size.pmax + i_p_io, :],
        output_tile,
        mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
      )
  
  return output_tensor
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

## torch-neuronx-graph-partitioner-app-note.rst

```python
import torch
import torch_neuronx
import torch.nn as nn

import logging

# adjust logger level to see what the partitioner is doing
logger = logging.getLogger("Neuron")

class MLP(nn.Module):
    def __init__(
        self, input_size=28 * 28, output_size=10, layers=[4096, 2048]
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        f1 = self.fc1(x)
        r1 = self.relu(f1)
        f2 = self.fc2(r1)
        r2 = self.relu(f2)
        f3 = self.fc3(r2)
        out = torch.log_softmax(f3, dim=1)
        sort_out,_ = torch.sort(out)
        return sort_out

n = MLP()
n.eval()

inputs = torch.rand(32,784)

# Configure the graph partitioner with the default values
partitioner_config = torch_neuronx.PartitionerConfig()

# Trace a neural network with graph partitioner enabled
neuron_net = torch_neuronx.trace(n, inputs, partitioner_config=partitioner_config)

# Run inference on the partitioned model
output = neuron_net(inputs)
```

```python
import torch
import torch_neuronx
import torch.nn as nn

import logging

# adjust logger level to see what the partitioner is doing
logger = logging.getLogger("Neuron")

class MLP(nn.Module):
    def __init__(
        self, input_size=28 * 28, output_size=10, layers=[4096, 2048]
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        f1 = self.fc1(x)
        r1 = self.relu(f1)
        sort_r1,_ = torch.sort(r1)
        f2 = self.fc2(sort_r1)
        r2 = self.relu(f2)
        f3 = self.fc3(r2)
        out = torch.log_softmax(f3, dim=1)
        return out

n = MLP()
n.eval()

inputs = torch.rand(32,784)

# Configure the graph partitioner with the default values
partitioner_config = torch_neuronx.PartitionerConfig(max_subgraph_count=2)

# This trace will fail since the min_subgraph_size requirement can't be satisfied by the graph partitioner
neuron_net = torch_neuronx.trace(n, inputs, partitioner_config=partitioner_config)
```

```python
import torch
import torch_neuronx
import torch.nn as nn

import logging

# adjust logger level to see what the partitioner is doing
logger = logging.getLogger("Neuron")
logger.setLevel(logging.INFO)

class MLP(nn.Module):
    def __init__(
        self, input_size=28 * 28, output_size=10, layers=[4096, 2048]
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        f1 = self.fc1(x)
        r1 = self.relu(f1)
        f2 = self.fc2(r1)
        r2 = self.relu(f2)
        f3 = self.fc3(r2)
        out = torch.log_softmax(f3, dim=1)
        sort_out,_ = torch.sort(out)
        return sort_out

n = MLP()
n.eval()

inputs = torch.rand(32,784)

# Configure the graph partitioner with the default values
partitioner_config = torch_neuronx.PartitionerConfig(min_operator_percentage_threshold=0.8,ops_to_partition=set(["aten::log_softmax"]))

# This trace succeeds
neuron_net = torch_neuronx.trace(n, inputs, partitioner_config=partitioner_config)
```

## sd2_inpainting_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
import os
import copy
from diffusers import StableDiffusionInpaintPipeline
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

    def forward(self, emb, attention_mask=None):
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


def trace_vae_encoder(model_id, height, width, compiler_workdir_root):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    vae_encoder = copy.deepcopy(pipe.vae.encoder)
    del pipe

    sample_input = torch.randn([1, 3, height, width])
    vae_encoder_neuron = torch_neuronx.trace(
            vae_encoder, 
            sample_input, 
            compiler_workdir=os.path.join(compiler_workdir_root, 'vae_encoder'),
            )

    vae_encoder_filename = os.path.join(compiler_workdir_root, 'vae_encoder/model.pt')
    torch.jit.save(vae_encoder_neuron, vae_encoder_filename)

    del vae_encoder
    del vae_encoder_neuron


def trace_unet(model_id, height, width, compiler_workdir_root):
    DTYPE = torch.bfloat16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

    Attention.get_attention_scores = get_attention_scores

    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    sample_1b = torch.randn([1, 9, height, width], dtype=DTYPE)
    timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=DTYPE)
    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=os.path.join(compiler_workdir_root, 'unet'),
        compiler_args=["--model-type=unet-inference", "--verbose=info"],
    )

    unet_filename = os.path.join(compiler_workdir_root, 'unet/model.pt')
    torch.jit.save(unet_neuron, unet_filename)

    del unet
    del unet_neuron
```

## test_nki_isa_range_select.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit(mode="simulation", platform_target="trn2")
def nki_range_select_example(on_true, bound0, bound1, compare_op0, compare_op1, range_start, dtype):
    # Create output tensors
    select_res = nl.ndarray(on_true.shape, dtype=dtype, buffer=nl.hbm)
    reduce_result = nl.ndarray((on_true.shape[0], 1), dtype=dtype, buffer=nl.hbm)
    
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])

    reduce_res_tile = nl.ndarray((on_true.shape[0], 1), dtype=dtype, buffer=nl.sbuf)
    result = nl.ndarray(on_true.shape, dtype=dtype, buffer=nl.sbuf)
    
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
        on_false_value=nl.fp32.min,
        dtype=dtype
    )

    nl.store(select_res[...], value=result[...])
    nl.store(reduce_result[...], value=reduce_res_tile[...])

    return result, reduce_result

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

## flex-eg.rst

```python
import mxnet as mx

# Load models (MXNet)
# loaded into the 2 cores starting with core 0
sym, args, aux = mx.model.load_checkpoint(mx_model0_file, 0)
model0 = sym.bind(ctx=mx.neuron(0), args=args, aux_states=aux, grad_req='null')
# loaded into the 4 cores starting with core 2
sym, args, aux = mx.model.load_checkpoint(mx_model1_file, 0)
model1 = sym.bind(ctx=mx.neuron(2), args=args, aux_states=aux, grad_req='null')
# loaded into the 3 cores starting with core 6
sym, args, aux = mx.model.load_checkpoint(mx_model2_file, 0)
model2 = sym.bind(ctx=mx.neuron(6), args=args, aux_states=aux, grad_req='null')
# loaded into the 4 cores starting with core 9
sym, args, aux = mx.model.load_checkpoint(mx_model3_file, 0)
model3 = sym.bind(ctx=mx.neuron(9), args=args, aux_states=aux, grad_req='null')

# run inference by simply calling the loaded model
results0 = model0.forward(data=inputs0)
results1 = model1.forward(data=inputs1)
results2 = model2.forward(data=inputs2)
results3 = model3.forward(data=inputs3)
```

```python
import os
import mxnet as mx

# Set Environment 
os.environ['NEURONCORE_GROUP_SIZES']='2,4,3,4'

# Load models (MXNet)
# loaded into the first group of NC0-NC1
sym, args, aux = mx.model.load_checkpoint(mx_model0_file, 0)
model0 = sym.bind(ctx=mx.neuron(0), args=args, aux_states=aux, grad_req='null')
# loaded into the second group of NC2-NC5
sym, args, aux = mx.model.load_checkpoint(mx_model1_file, 0)
model1 = sym.bind(ctx=mx.neuron(1), args=args, aux_states=aux, grad_req='null')
# loaded into the third group of NC6-NC8
sym, args, aux = mx.model.load_checkpoint(mx_model2_file, 0)
model2 = sym.bind(ctx=mx.neuron(2), args=args, aux_states=aux, grad_req='null')
# loaded into the fourth group of NC9-NC12
sym, args, aux = mx.model.load_checkpoint(mx_model3_file, 0)
model3 = sym.bind(ctx=mx.neuron(3), args=args, aux_states=aux, grad_req='null')

# run inference by simply calling the loaded model
results0 = model0.forward(data=inputs0)
results1 = model1.forward(data=inputs1)
results2 = model2.forward(data=inputs2)
results3 = model3.forward(data=inputs3)
```

## sd_4x_upscaler_compile.py

```python
import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

from diffusers import StableDiffusionUpscalePipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
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


# Compile text encoder
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder,
    emb,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

# Compile VAE decoder
decoder_neuron = torch_neuronx.trace(
    decoder,
    decoder_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Compile UNet with compiler args
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

# Compile VAE post_quant_conv
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv,
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)
```

## customop-mlp-perf-opt.rst

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import my_ops

class MLP(nn.Module):
    def __init__(self, input_size = 28 * 28, output_size = 10, layers = [4096, 2048]):
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

## hf_llama3_8B_pretraining.rst

```python
from huggingface_hub import login
from transformers import AutoTokenizer

login(token='your_own_hugging_face_token')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')  

tokenizer.save_pretrained(".")
```

## sd2_512_compile.py

```python
import torch
import torch.nn as nn
import torch_neuronx

# Define datatype
DTYPE = torch.bfloat16

# Model wrapper for custom return types
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

# Optimized attention computation
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = custom_badbmm(key, query.transpose(-1, -2))
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2))
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled

# Compile UNet with torch_neuronx
sample_1b = torch.randn([1, 4, 64, 64], dtype=DTYPE)
timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=DTYPE)
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
)

# Enable asynchronous and lazy loading
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

# Save compiled model
torch.jit.save(unet_neuron, unet_filename)

# Compile text encoder
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder, 
    emb, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)

torch_neuronx.async_load(text_encoder_neuron)
torch.jit.save(text_encoder_neuron, text_encoder_filename)

# Compile VAE decoder
decoder_in = torch.randn([1, 4, 64, 64], dtype=torch.float32)
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)

torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, decoder_filename)

# Compile VAE post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 64, 64], dtype=torch.float32)
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
```

## tp_developer_guide.rst

```python
from torch.utils.data import DataLoader, DistributedSampler
from neuronx_distributed.parallel_layers import parallel_state

def create_pretraining_dataset(
    input_file, max_pred_length, mini_batch_size, worker_init
):
    train_data = pretraining_dataset(
        input_file=input_file, max_pred_length=max_pred_length
    )
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
    )
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader
```

```python
import transformers
from neuronx_distributed.parallel_layers import ColumnParallelLinear, parallel_state

class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.query = ColumnParallelLinear(config.hidden_size,
                                          self.all_head_size,
                                          gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                          self.all_head_size,
                                          gather_output=False)
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
```

```python
import transformers
from neuronx_distributed.parallel_layers import RowParallelLinear

class ParallelSelfOutput(transformers.models.bert.modeling_bert.BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = RowParallelLinear(config.hidden_size,
                                       config.hidden_size,
                                       input_is_parallel=True)
```

```python
from neuronx_distributed.parallel_layers import parallel_state, clip_grad_norm, move_model_to_device
import neuronx_distributed
import torch_xla.core.xla_model as xm

neuronx_distributed.parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
dataloader = create_pretraining_dataset(
    input_file, max_pred_length, mini_batch_size, worker_init)

model = YourNewlyBuiltParallelModel(config)
move_model_to_device(model, device)

for inputs, labels in dataloader:
    output = model(*inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    clip_grad_norm(model.parameters(), max_norm)
    xm.optimizer_step(
        optimizer, 
        groups=parallel_state.get_data_parallel_group(as_list=True)
    )
    optimizer.zero_grad()
    scheduler.step()
```

```python
import neuronx_distributed

neuronx_distributed.parallel_layers.save({
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, PATH)
```

## sd2_768_compile.py

```python
import torch
import torch.nn as nn
import torch_neuronx

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
        sample = self.unetwrap(sample, timestep.to(dtype=torch.float32).expand((sample.shape[0],)), encoder_hidden_states)[0]
        from diffusers.models.unet_2d_condition import UNet2DConditionOutput
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

```python
# Compile UNet
sample_1b = torch.randn([1, 4, 96, 96], dtype=torch.float32)
timestep_1b = torch.tensor(999, dtype=torch.float32).expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=torch.float32)
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir='sd2_compile_dir_768/unet',
    compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
)

torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)
torch.jit.save(unet_neuron, 'sd2_compile_dir_768/unet/model.pt')
```

```python
# Compile text encoder
emb = torch.tensor([[49406, 18376, 525, 7496, 49407] + [0]*72])
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder, 
    emb, 
    compiler_workdir='sd2_compile_dir_768/text_encoder',
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)

torch_neuronx.async_load(text_encoder_neuron)
torch.jit.save(text_encoder_neuron, 'sd2_compile_dir_768/text_encoder/model.pt')
```

```python
# Compile VAE decoder
decoder_in = torch.randn([1, 4, 96, 96], dtype=torch.float32)
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir='sd2_compile_dir_768/vae_decoder',
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)

torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, 'sd2_compile_dir_768/vae_decoder/model.pt')
```

```python
# Compile VAE post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 96, 96], dtype=torch.float32)
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir='sd2_compile_dir_768/vae_post_quant_conv',
)

torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, 'sd2_compile_dir_768/vae_post_quant_conv/model.pt')
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


def test_validate_accuracy_basic_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(ExampleModule, example_inputs, module_init_kwargs={"distributed": True})

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)
```

```python
import torch

from neuronx_distributed_inference.utils.testing import build_function, validate_accuracy


def example_sum(tensor):
    return torch.sum(tensor)


def test_validate_accuracy_basic_function():
    inputs = [(torch.tensor([1, 2, 3], dtype=torch.float32),)]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(neuron_model, inputs, cpu_callable=example_sum)
```

```python
import torch
from functools import partial

from neuronx_distributed_inference.utils.testing import build_function


def top_k(input: torch.Tensor, k: int, dim: int):
    return torch.topk(input, k, dim)


top_k_partial = partial(top_k, 1, 0)
model = build_function(top_k_partial, example_inputs=[(torch.rand(4)),])
output = model(torch.rand(4))
```

## sd2_inpainting_inference.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

DTYPE = torch.bfloat16

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

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample.to(dtype=DTYPE), timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states.to(dtype=DTYPE))[0]
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

```python
# Loading compiled models with torch_neuronx
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.encoder = torch.jit.load(vae_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
```

## sdxl_base_and_refiner_1024_compile.py

```python
import torch
import torch.nn as nn
import torch_neuronx
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

```python
# Compile UNet with torch_neuronx.trace
sample_1b = torch.randn([1, 4, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 2048])
added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280]),
                        "time_ids": torch.randn([1, 6])}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet_base'),
    compiler_args=["--model-type=unet-inference"]
)

# Enable asynchronous and lazy loading
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

# Save compiled model
torch.jit.save(unet_neuron, unet_filename)
```

```python
# Compile VAE decoder
decoder_in = torch.randn([1, 4, 128, 128], dtype=DTYPE)
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder')
)

torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, decoder_filename)
```

```python
# Compile VAE post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=DTYPE)
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
```

## performance-tuning.rst

```python
import numpy as np
import tensorflow.neuron as tfn

# Compiling for batching optimization
batch_size = 5
example_input = np.zeros([batch_size,224,224,3], dtype='float16')

tfn.saved_model.compile("rn50_fp16",
                        "rn50_fp16_compiled/1",
                        model_feed_dict={'input_1:0': example_input },
                        dynamic_batch_size=True)
```

```python
import numpy as np
import tensorflow.neuron as tfn

# Compiling for pipeline optimization
compiler_args = ['--neuroncore-pipeline-cores', '16']
example_input = np.zeros([1,224,224,3], dtype='float16')
tfn.saved_model.compile("rn50_fp16",
                        "rn50_fp16_compiled/1",
                        model_feed_dict={'input_1:0': example_input },
                        compiler_args=compiler_args)
```

## new_model_guide.rst

```python
from transformers import GPTNeoXConfig
import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed_training.lightning_modules.model.base import BaseModelModule
from neuronx_distributed_training.utils.model_utils import get_param_groups_by_weight_decay
from modeling_gpt_neox_nxd import GPTNeoXForCausalLMNxD

class MyNewModel(BaseModelModule):

    def _get_model(self,):
        model_name = "EleutherAI/gpt-neox-20b"
        config = GPTNeoXConfig.from_pretrained(model_name)
        config.use_cache = False
        if self.config.model.get('num_layers', -1) != -1:
            config.num_hidden_layers = self.config.model.get('num_layers')
        if self.config.model.get('hidden_size', -1) != -1:
            config.hidden_size = self.config.model.get('hidden_size')
        config.sequence_parallel_enabled = self.config.distributed_strategy.get("sequence_parallel", False)
        return GPTNeoXForCausalLMNxD(config)

    def build_model(self):
        if self.config.model.get("activations_checkpoint_granularity", None) == "selective":
            self.nxd_config["activation_checkpoint_config"] = GPTNeoXMLPNxD
        elif self.config.model.get("activations_checkpoint_granularity", None) == "full":
            self.nxd_config["activation_checkpoint_config"] = "full"

        self.nxd_config["pipeline_config"].update(
            {
                "transformer_layer_cls": GPTNeoXLayerNxD,
                "output_loss_value_spec": (True, False),
                "input_names": ["input_ids", "attention_mask", "labels"],
                "leaf_module_cls": [LayerNorm.__name__],
            }
        )
        return nxd.initialize_parallel_model(self.nxd_config, self._get_model)

    def setup_optimizer_param_groups(self):
        no_decay = ["bias"]
        if self.config.model.get("do_layer_norm_weight_decay", False):
            no_decay.append("LayerNorm")
        self._optimizer_param_groups = get_param_groups_by_weight_decay(self.model, no_decay)

    def init_weights(self,):
        if isinstance(module, LayerNorm):
            module.weight.data.fill_(1.0)
        super().init_weights()
```

```python
from new_model_module import MyNewModel

data_module = HFDataModule(cfg, trainer)
model = MyNewModel(cfg, trainer)

trainer.fit(model, datamodule=data_module)
```

## sdxl_base_1024_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

DTYPE = torch.float32

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
                               timestep.to(dtype=DTYPE).expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)

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

class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple

# Load compiled models and deploy to Trainium
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE)

pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load("unet_model.pt"), device_ids, set_dynamic_batching=False)

pipe.vae.decoder = torch.jit.load("decoder_model.pt")
pipe.vae.post_quant_conv = torch.jit.load("post_quant_conv_model.pt")
pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load("text_encoder_model.pt"), pipe.text_encoder)
pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load("text_encoder_2_model.pt"), pipe.text_encoder_2)
```

## sdxl_base_and_refiner_1024_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

DTYPE = torch.float32

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
                               timestep.to(dtype=DTYPE).expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)
```

```python
# Load compiled UNet onto multiple neuron cores using DataParallel
pipe_base.unet = NeuronUNet(UNetWrap(pipe_base.unet))
device_ids = [0, 1]
pipe_base.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load(unet_base_filename), 
    device_ids, 
    set_dynamic_batching=False
)

# Load other compiled models onto neuron core
pipe_base.vae.decoder = torch.jit.load(decoder_filename)
pipe_base.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
```

## api-tracing-python-api.rst

```python
import tensorflow as tf
import tensorflow.neuron as tfn

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
model = tf.keras.Model(inputs=[input0], outputs=[dense0])
example_inputs = tf.random.uniform([1, 3])
model_neuron = tfn.trace(model, example_inputs)
print(model_neuron.on_neuron_ratio)

model_dir = './model_neuron'
model_neuron.save(model_dir)
model_neuron_reloaded = tf.keras.models.load_model(model_dir)
```

```python
import tensorflow as tf
import tensorflow.neuron as tfn

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
reshape0 = tf.keras.layers.Reshape([1, 3])(dense0)
output0 = tf.keras.layers.Dense(2)(reshape0)
model = tf.keras.Model(inputs=[input0], outputs=[output0])
example_inputs = tf.random.uniform([1, 3])

def subgraph_builder_function(node):
    return node.op == 'MatMul'

model_neuron = tfn.trace(
    model, example_inputs,
    subgraph_builder_function=subgraph_builder_function,
)
```

## neuronperf_evaluate_guide.rst

```python
import neuronperf as npf

reports = npf.torch.benchmark(
    model_index_or_path,
    dataset,
    n_models=1,
    workers_per_model=2,
    duration=0,
    eval_metrics=['accuracy', 'precision']
)
```

```python
import neuronperf as npf

reports = npf.torch.evaluate(model_index_or_path, dataset, metrics=['accuracy', 'precision'])
```

```python
import neuronperf as npf

reports = npf.torch.evaluate(model_index_or_path, dataset, metrics='accuracy', eval_target_col=1)
```

```python
import neuronperf as npf

npf.register_metric_from_existing("topk", "topk_3", k=3)
```

```python
import neuronperf as npf
from abc import ABC, abstractmethod
from typing import Any, Iterable

class MyCustomMetric(npf.BaseEvalMetric):
    def __init__(self):
        super().__init__()
        self.passing = 0
        self.processed = 0

    def process_record(self, outputs, target):
        self.processed += 1
        if outputs == target:
            self.passing += 1
    
    @staticmethod
    def aggregate(metrics):
        passing = 0
        processed = 0
        for metric in metrics:
            passing += metric.passing
            processed += metric.processed
        return passing / processed if processed else 0


npf.register_metric("MyCustomMetric", MyCustomMetric)
```

## pixart_sigma_benchmark.py

```python
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import torch
import torch_neuronx
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers import Transformer2DModel, PixArtSigmaPipeline


class InferenceTextEncoderWrapper(nn.Module):
  def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
    super().__init__()
    self.dtype = dtype
    self.device = t.device
    self.t = t
  def forward(self, text_input_ids, attention_mask=None):
    return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]


class InferenceTransformerWrapper(nn.Module):
  def __init__(self, transformer: Transformer2DModel):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, 
              encoder_attention_mask=None, added_cond_kwargs=None,
              return_dict=False):
    output = self.transformer(
      hidden_states, 
      encoder_hidden_states, 
      timestep, 
      encoder_attention_mask)
    return output


class SimpleWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, x):
    output = self.model(x)
    return output


# Load compiled models and integrate with pipeline
DTYPE = torch.bfloat16
text_encoder_filename = 'pixart_sigma_compile_dir/text_encoder/model.pt'
transformer_filename = 'pixart_sigma_compile_dir/transformer/model.pt'
decoder_filename = 'pixart_sigma_compile_dir/vae_decoder/model.pt'
post_quant_conv_filename = 'pixart_sigma_compile_dir/vae_post_quant_conv/model.pt'

pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", torch_dtype=DTYPE)

_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen=300)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)
pipe.transformer = _neuronTransformer

pipe.vae.decoder = SimpleWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load(post_quant_conv_filename))
```

## tfneuronx-python-tracing-api.rst

```python
import tensorflow as tf
import tensorflow_neuronx as tfnx

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
model = tf.keras.Model(inputs=[input0], outputs=[dense0])
example_inputs = tf.random.uniform([1, 3])
model_neuron = tfnx.trace(model, example_inputs)
print(model_neuron.on_neuron_ratio)

model_dir = './model_neuron'
model_neuron.save(model_dir)
model_neuron_reloaded = tf.keras.models.load_model(model_dir)
```

```python
import tensorflow as tf
import tensorflow_neuronx as tfnx

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
reshape0 = tf.keras.layers.Reshape([1, 3])(dense0)
output0 = tf.keras.layers.Dense(2)(reshape0)
model = tf.keras.Model(inputs=[input0], outputs=[output0])
example_inputs = tf.random.uniform([1, 3])

def subgraph_builder_function(node):
    return node.op == 'MatMul'

model_neuron = tfnx.trace(
    model, example_inputs,
    subgraph_builder_function=subgraph_builder_function,
)
```

## indexing-overview.rst

```python
import neuronxcc.nki as nl

# Basic integer indexing
x = nl.ndarray((2, 2, 2), dtype=nl.float32, buffer=nl.hbm)
assert x[1].shape == [2, 2]

# Slicing syntax
x = nl.ndarray((2, 128, 1024), dtype=nl.float32, buffer=nl.hbm)
assert x[1, :, :].shape == [128, 1024]
assert x[1, :, 0:512].shape == [128, 512]
assert x[:, 1, 0:2].shape == [2, 2]
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

## pixart_alpha_benchmark.py

```python
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import torch
import torch_neuronx
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers import Transformer2DModel

DTYPE = torch.bfloat16

class InferenceTextEncoderWrapper(nn.Module):
  def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
    super().__init__()
    self.dtype = dtype
    self.device = t.device
    self.t = t
  def forward(self, text_input_ids, attention_mask=None):
    return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]

class InferenceTransformerWrapper(nn.Module):
  def __init__(self, transformer: Transformer2DModel):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, 
              encoder_attention_mask=None, added_cond_kwargs=None,
              return_dict=False):
    output = self.transformer(
      hidden_states, 
      encoder_hidden_states, 
      timestep, 
      encoder_attention_mask)
    return output

class SimpleWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, x):
    output = self.model(x)
    return output

# Load compiled models
_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)
pipe.transformer = _neuronTransformer

pipe.vae.decoder = SimpleWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load(post_quant_conv_filename))
```

## sd_4x_upscaler_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import StableDiffusionUpscalePipeline
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

```python
# Load compiled models and deploy to Trainium
pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Load the compiled UNet onto two neuron cores with DataParallel
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load(unet_filename), 
    device_ids, 
    set_dynamic_batching=False
)

# Load other compiled models onto a single neuron core
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
```

## sd2_512_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

DTYPE = torch.bfloat16

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

class NeuronTypeConversionWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x.float())
```

```python
# Load compiled UNet with DataParallel across multiple neuron cores
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load compiled models onto neuron cores
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = NeuronTypeConversionWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = NeuronTypeConversionWrapper(torch.jit.load(post_quant_conv_filename))
```

## test_nki_isa_select_reduce.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def nki_select_reduce_basic(predicate_data, on_true_data):
  """Example 1: Basic usage of select_reduce"""
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


@nki.jit(mode="simulation")
def nki_select_reduce_with_reduction(predicate_data, on_true_data, on_false_data):
  """Example 2: Using select_reduce with reduction"""
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


@nki.jit(mode="simulation")
def nki_select_reduce_reverse_pred(predicate_data, on_true_data):
  """Example 3: Using select_reduce with reverse_pred option"""
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

## infer_resnet50_keras_loadtest.py

```python
import tensorflow as tf
import tensorflow.neuron
import os

# Set environment variables for NeuronCore allocation
os.environ['NEURON_MAX_NUM_INFERS'] = str(NUM_INFERS_IN_FLIGHT)
os.environ['NEURONCORE_GROUP_SIZES'] = ','.join(group_sizes)

# Load compiled model from saved model directory
pred = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)

# Prepare input data
model_feed_dict = {'input_1:0': img_arr3}

# Run inference
result = pred(model_feed_dict)
```

## sd_15_512_benchmark.py

```python
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline
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


# Load compiled models onto Neuron cores
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

# Load UNet with data parallelism across two Neuron cores
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load(unet_filename), 
    device_ids, 
    set_dynamic_batching=False
)

# Load other models onto single Neuron cores
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
pipe.safety_checker.vision_model = NeuronSafetyModelWrap(torch.jit.load(safety_model_neuron_filename))
```

## inference.rst

```python
import os
import torch
import torch_neuronx
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

import neuronx_distributed
from neuronx_distributed.parallel_layers import layers, parallel_state


def encode(tokenizer, *inputs, max_length=128, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
        torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
    )


def get_model():
    name = "bert-base-cased-finetuned-mrpc"
    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    
    class ParallelSelfAttention(BertSelfAttention):
        def __init__(self, config, position_embedding_type=None):
            super().__init__(config, position_embedding_type)
            self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
            self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()

    class ParallelSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = layers.RowParallelLinear(config.hidden_size,
                                    config.hidden_size,
                                    input_is_parallel=True)

    for layer in model.bert.encoder.layer:
        layer.attention.self = ParallelSelfAttention(model.config)
        layer.attention.output = ParallelSelfOutput(model.config)

    neuronx_distributed.parallel_layers.load("bert.pt", model, sharded=False)

    io_aliases = {}
    return model, io_aliases
```

```python
# Trace the model with tensor parallelism
model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2)

# Save the traced model
neuronx_distributed.trace.parallel_model_save(model, "tp_models")

# Load and run inference
model = neuronx_distributed.trace.parallel_model_load("tp_models")
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
  """fold_batch_norm function of TransformGraph is unable to fold Keras ResNet50
  because of BiasAdd between Conv2D and FusedBatchNorm (BiasAdd is not needed
  if FusedBatchNorm is used, but it exists in Keras ResNet50). Here, we 
  move BiasAdd to after FusedBatchNorm, and adjust bias value by gamma/sqrt(variance).
  """
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
  """Optimize training graph for inference:
    - Remove Identity and CheckNumerics nodes
    - Fold FusedBatchNorm constants into previous Conv2D weights
    - Fold other constants
    - Strip unused nodes
    - Sort by execution order
  """
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

## test_nki_isa_dma_transpose.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.isa.constants import dge_mode

@nki.jit(mode="simulation")
def nki_dma_transpose_2d_hbm2sb(a):
  b_sb = nisa.dma_transpose(a[:, :])
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b

@nki.jit(mode="simulation")
def nki_dma_transpose_2d_sb2sb(a):
  a_sb = nl.load(a)
  b_sb = nisa.dma_transpose(a_sb[:, :])
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b

@nki.jit(mode="simulation", platform_target="trn2")
def nki_dma_transpose_2d_hbm2sb_dge_xbar(a):
  b_sb = nisa.dma_transpose(a[:, :], dge_mode=dge_mode.hwdge)
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b

@nki.jit(mode="simulation", platform_target="trn2")
def nki_dma_transpose_2d_sb2sb_dge_xbar(a):
  a_sb = nl.load(a)
  b_sb = nisa.dma_transpose(a_sb[:, :], dge_mode=dge_mode.hwdge)
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b

@nki.jit(mode="simulation", platform_target="trn2")
def nki_dma_gather_transpose_3d_hbm2sb(src_tensor, idx_tensor):
  i_p = nl.arange(32)[:, None]
  idx = nl.load(idx_tensor)

  _, dim1, dim2 = src_tensor.shape

  iy = nl.arange(dim1)[None, :, None]
  iz = nl.arange(dim2)[None, None, :]

  dst = nisa.dma_transpose(src_tensor[idx[i_p, 0], iy, iz], axes=(2, 1, 0))
  dst_tensor = nl.ndarray(shape=(dim2, dim1, idx.shape[0]), dtype=src_tensor.dtype, buffer=nl.shared_hbm)
    
  nl.store(dst_tensor, dst)
  return dst_tensor

@nki.jit(mode="simulation", platform_target="trn2")
def nki_dma_gather_transpose_3d_sb2sb(src_tensor, idx_tensor):
  src = nl.load(src_tensor)
  idx = nl.load(idx_tensor)

  dim0, dim1, dim2 = src.shape
  
  iy = nl.arange(dim1)[None, :, None]
  iz = nl.arange(dim2)[None, None, :]

  dst = nisa.dma_transpose(src[idx, iy, iz], axes=(2, 1, 0))
  dst_tensor = nl.ndarray(shape=(dim2, dim1, dim0), dtype=src.dtype, buffer=nl.shared_hbm)
  
  nl.store(dst_tensor, dst)
  return dst_tensor
```

## tf-neuronx-auto-replication-api.rst

```python
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow_neuronx as tfnx

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
inputs = [input0]
outputs = [dense0]
model = tf.keras.Model(inputs=inputs, outputs=outputs)
input0_tensor = tf.random.uniform([1, 3])
model_neuron = tfnx.trace(model, input0_tensor)

# a trn1.2xlarge has 2 neuron cores
num_cores = 2
multicore_model = tfn.auto_multicore(model_neuron, input0_tensor, num_cores=num_cores)
multicore_model(input0_tensor)
```

```python
from tensorflow.python import saved_model
import tensorflow as tf
import tensorflow.neuron as tfn

input0_tensor = tf.random.uniform([1, 3])
num_cores = 4
reload_model = saved_model.load(model_dir)
multicore_model = tfn.auto_multicore(reload_model, input0_tensor, num_cores=num_cores)
```

## sd2_768_benchmark.py

```python
import torch
import torch_neuronx
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
import torch.nn as nn

DTYPE = torch.float32

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

# Load compiled models and deploy to Trainium
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=DTYPE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Deploy UNet across multiple Neuron cores with data parallelism
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load('sd2_compile_dir_768/unet/model.pt'), 
    device_ids, 
    set_dynamic_batching=False
)

# Deploy other models to single Neuron core
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load('sd2_compile_dir_768/text_encoder/model.pt')
pipe.vae.decoder = torch.jit.load('sd2_compile_dir_768/vae_decoder/model.pt')
pipe.vae.post_quant_conv = torch.jit.load('sd2_compile_dir_768/vae_post_quant_conv/model.pt')
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

## neuronperf_examples.rst

```python
import torch

import neuronperf as npf
import neuronperf.torch

# Construct dummy inputs
batch_sizes = 1
input_shape = (batch_sizes, 3, 224, 224)
inputs = torch.ones(input_shape)

# Benchmark and save results
reports = npf.torch.benchmark("your_model_file.pt", inputs, batch_sizes)
npf.print_reports(reports)
npf.write_json(reports)
```

```python
reports = npf.torch.benchmark(filename, inputs, batch_sizes, n_models=1, workers_per_model=[1, 2], duration=15)
```

```python
reports = npf.torch.benchmark(..., model_name="MyFancyModel")
```

```python
cpu_reports = npf.cpu.benchmark(YourModelClass, ...)
```

```python
gpu_reports = npf.torch.benchmark(YourModelClass, ..., device_type="gpu")
```

## index.rst

```python
# this is a Python function that calls 'kernel', which is a NKI kernel
def a_function(x, y, z):
    kernel(x, y, z)

# this is a NKI kernel that will be compiled by the NKI compiler and 
# integrated back into the overall model by the Neuron Graph compiler
@nki.jit
def kernel(x, y, z):
    # this is kernel code
    pass
```

## tutorial-tensorflowx-serving-NeuronRT-Visible-Cores.rst

```python
import tensorflow as tf
import tensorflow_neuronx as tfnx
import numpy as np

tf.keras.backend.set_learning_phase(0)
tf.keras.backend.set_image_data_format('channels_last')
image_sizes = [224, 224]
model = tf.keras.applications.ResNet50(weights='imagenet')
example_inputs = tf.random.uniform([1, *image_sizes, 3], dtype=tf.float32)

model_neuron = tfnx.trace(model, example_inputs)
# run the model once to define the forward pass and allow for saving
model_neuron(example_inputs)
tf.keras.models.save_model(model_neuron, './resnet50_neuron/1')
```

```python
import numpy as np
import grpc
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.applications.resnet50 import decode_predictions

tf.keras.backend.set_image_data_format('channels_last')

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
img_file = tf.keras.utils.get_file(
    "./kitten_small.jpg",
    "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")
img = image.load_img(img_file, target_size=(224, 224))
img_array = preprocess_input(image.img_to_array(img)[None, ...])
request = predict_pb2.PredictRequest()
request.model_spec.name = 'resnet50_neuron'
request.inputs['input_1'].CopyFrom(
    tf.make_tensor_proto(img_array, shape=img_array.shape))
result = stub.Predict(request)
prediction = tf.make_ndarray(result.outputs['output_1'])
print(decode_predictions(prediction))
```

## pb2sm_compile.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn
import numpy as np

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
batch_size = 5
img_arr = np.zeros([batch_size, 224, 224, 3], dtype='float16')

compiler_args = ['--neuroncore-pipeline-cores', '1']

rslts = tfn.saved_model.compile(
    saved_model_dir, 
    compiled_saved_model_dir,
    model_feed_dict={'input_1:0': img_arr},
    compiler_workdir='compiler_workdir',
    dynamic_batch_size=True,
    compiler_args=compiler_args
)

on_neuron_ratio = rslts['OnNeuronRatio'] * 100
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
```

```python
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


def parallel_trace_encoder(model_name: str,
                           max_length: int,
                           num_beams: int,
                           tp_degree: int):
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    get_encoder_callable = partial(get_wrapped_encoder, max_length, num_beams, tp_degree, model_name)

    batch_encoding = tokenizer("translate English to German: Lets go home now",
                               max_length=max_length, truncation=True, padding='max_length', return_tensors="pt")
    input_ids = batch_encoding['input_ids']
    attention_mask = batch_encoding['attention_mask']

    traced_encoder = neuronx_distributed.trace.parallel_model_trace(get_encoder_callable, (
            input_ids,
            attention_mask,
        ), 
        tp_degree=tp_degree, 
        compiler_workdir="/tmp/encoder/",
        )
    setattr(traced_encoder, 'main_input_name', 'input_ids')

    return traced_encoder


def parallel_trace_decoder(model: T5ForConditionalGeneration,
                           model_name: str,
                           num_beams: int,
                           max_length: int,
                           tp_degree: int):

    get_decoder_callable = partial(get_wrapped_decoder, max_length, num_beams, tp_degree, model_name)
  
    decoder_input_ids = torch.ones((num_beams, 1), dtype=torch.int64)
    decoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int32)
    encoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int64)
    encoder_hidden_states = torch.ones((num_beams, max_length, model.config.d_model), dtype=torch.float32)

    beam_idx = torch.arange(0, num_beams, dtype=torch.int64)
    beam_scores = torch.zeros((num_beams,), dtype=torch.float)

    traced_decoder = neuronx_distributed.trace.parallel_model_trace(get_decoder_callable, (
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            beam_idx,
            beam_scores
        ), 
        tp_degree=tp_degree,
        compiler_workdir="/tmp/decoder/",
        )

    return traced_decoder
```

## test_nki_isa_tensor_copy_dynamic.py

```python
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

## api-torch-neuronx-replace-weights.rst

```python
import torch
import torch_neuronx


class Network(torch.nn.Module):
    def __init__(self, hidden_size=4, layers=3) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            *(torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)))

    def forward(self, tensor):
        return self.layers(tensor)


# initialize two networks
network = Network()
network2 = Network()
network.eval()
network2.eval()

inp = torch.rand(2,4)

# trace weight separated model with first network
weight_separated_trace = torch_neuronx.trace(network,inp,inline_weights_to_neff=False)

# replace with weights from second network
torch_neuronx.replace_weights(weight_separated_trace,network2.state_dict())

# get outputs from neuron and cpu networks
out_network2 = network2(inp)
out_neuron = weight_separated_trace(inp)

# check that they are equal
print(out_network2,out_neuron)
```

```python
import torch
import torch_neuronx

from safetensors import safe_open
from safetensors.torch import save_model


class Network(torch.nn.Module):
    def __init__(self, hidden_size=4, layers=3) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            *(torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)))

    def forward(self, tensor):
        return self.layers(tensor)


# initialize two networks
network = Network()
network2 = Network()
network.eval()
network2.eval()

inp = torch.rand(2,4)

# trace weight separated model with first network
weight_separated_trace = torch_neuronx.trace(network,inp,inline_weights_to_neff=False)

# save network2 weights to safetensors
safetensor_path = f"{directory}/network2.safetensors"
save_model(network2,safetensor_path)

#load safetensors from network2 into traced_weight separated model
tensors = {}
with safe_open(safetensor_path,framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

# replace with weights from second network
torch_neuronx.replace_weights(weight_separated_trace,tensors)

# get outputs from neuron and cpu networks
out_network2 = network2(inp)
out_neuron = weight_separated_trace(inp)

# check that they are equal
print(out_network2,out_neuron)
```

## pipeline_parallelism_overview.rst

```python
import torch

# original NN module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(5)])

    def forward(self, x):
        for lin in self.linears:
            x = lin(x)
        return x

m = MyModule()
gm = torch.fx.symbolic_trace(m)
```

## mamba_torch.py

```python
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
import os

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --model-type=transformer --disable-dge "


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
```

```python
@nki.jit(mode="simulation")
def nki_nc_stream_shuffle(in_tensor):
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[32, 128] = nl.load(in_tensor)
  a_mgrid = nl.mgrid[0:32, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=a)
  return out_tensor
```

```python
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
```

```python
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

## api-compilation-python-api.rst

```python
import shutil
import tensorflow.neuron as tfn

saved_model_path = "<saved model path>"
compiled_saved_model_path = "<compiled saved model path>"
shutil.rmtree(compiled_saved_model_path, ignore_errors=True)
tfn.saved_model.compile(saved_model_path, compiled_saved_model_path)
```

## model_optimizer_wrapper_developer_guide.rst

```python
import neuronx_distributed as nxd
import torch

# Create training config with tensor parallel, pipeline parallel, ZeRO-1 optimizer,
# sequence parallel and activation checkpointing
nxd_config = nxd.neuronx_distributed_config(
    tensor_parallel_size=args.tensor_parallel_size,
    pipeline_parallel_size=args.pipeline_parallel_size,
    pipeline_config={
        "transformer_layer_cls": LlamaDecoderLayer,
        "num_microbatches": args.num_microbatches,
        "output_loss_value_spec": (True, False),
        "input_names": ["input_ids", "attention_mask", "labels"],
        "pipeline_cuts": pipeline_cuts,
        "trace_file_path": args.trace_file_path,
        "param_init_fn": None,
        "leaf_module_cls": [LlamaRMSNorm.__name__],
        "autowrap_modules": [mappings],
        "use_zero1_optimizer": args.use_zero1_optimizer > 0,
        "use_optimizer_wrapper": True,
    },
    optimizer_config={
        "zero_one_enabled": args.use_zero1_optimizer > 0,
        "grad_clipping": True,
        "max_grad_norm": 1.0,
    },
    sequence_parallel=args.use_sequence_parallel,
    activation_checkpoint_config=CoreAttention if args.use_selective_checkpoint > 0 else "full",
    model_init_config=model_init_config,
)

# Initialize parallel model
model = nxd.initialize_parallel_model(nxd_config, get_model, config)

# Run training iteration with pipeline parallel
loss = model.run_train(*inputs)

# Run training iteration without pipeline parallel
loss = model(*inputs)
loss.backward()

# Access wrapped model
model.local_module()

# Access model properties
model.dtype
model.config
model.name_or_path

# Initialize parallel optimizer
optimizer = nxd.initialize_parallel_optimizer(
    nxd_config, torch.optim.AdamW, param_groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
)

# Access grad norm from optimizer
grad_norm = optimizer.grad_norm
```

## new_dataloader_guide.rst

```python
from neuronx_distributed_training.lightning_modules.data.base import BaseDataModule

class NewDataModule(BaseDataModule):
    def __init__(self, cfg, trainer):
        """
        DataModule class for configuring the dataset/dataloader

        Args:
            cfg: `data` cfg in the yaml file.
            trainer: PyTorch-Lightning trainer.
        """
        super().__init__(cfg, trainer)
        # Users can use the cfg argument to pass down
        # arguments from the yaml file to the DataModule.

    def get_batch_length(self, batch):
        """
        Returns the length of the batch.
        """
        return len(batch["input_ids"])

    def process_global_batch(self, global_batch, global_batch_size=None):
        """ Any custom processing of batches can be done here.

        Args:
            global_batch: list of inputs, eg.[tokens, labels]
            global_batch_size: Length of tokens and labels
        """
        return global_batch

    def train_dataloader(self):
        """
        This API should return a torch.utils.data.dataloader.DataLoader object
        """
        ...

    def val_dataloader(self):
        """
        This API should return a torch.utils.data.dataloader.DataLoader object
        """
        ...

    def test_dataloader(self):
        """
        This API should return a torch.utils.data.dataloader.DataLoader object
        """
        ...
```

```python
from new_data_module import NewDataModule

data_module = NewDataModule(cfg, trainer)
model = HFLLamaModule(cfg, trainer)

trainer.fit(model, datamodule=data_module)
```

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
```

## cpu_mode_developer_guide.rst

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from neuronx_distributed.parallel_layers import layers
from neuronx_distributed.parallel_layers import initialize_model_parallel
from neuronx_distributed.utils import cpu_mode, get_device, master_print

# initialize the distributed environment inside PyTorch
cc_backend = "gloo" if cpu_mode() else "xla"
dist.init_process_group(backend=cc_backend)

# assuming sharding the model with TP=2
initialize_model_parallel(tensor_model_parallel_size=2)

hidden_size = 1024
rand_inputs = torch.rand(4, hidden_size)
model = nn.Sequential(
    layers.ColumnParallelLinear(
        hidden_size,
        hidden_size,
        bias=False,
        gather_output=False,
        keep_master_weight=True,
    ),
    layers.RowParallelLinear(
        hidden_size,
        hidden_size,
        bias=False,
        input_is_parallel=True,
        keep_master_weight=True,
    ),
)
model = model.to(get_device())
rand_inputs = rand_inputs.to(get_device())

outputs = model(rand_inputs)
master_print(f"Output sum is {outputs.sum()}")
```

## training-gpt-neox.rst

```python
from neuronx_distributed.optimizer import NeuronZero1Optimizer

optimizer = NeuronZero1Optimizer(
    optimizer_grouped_parameters,
    AdamW_FP32OptimParams,
    lr=flags.lr,
    pin_layout=False,
    sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
    grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
)
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
  
  # Example 1: subtract 1.0 from all elements of tile a of shape (128, 512)
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)
  nl.store(b_tensor[i_p, i_f], b)

  # Example 2: broadcast 1.0 into shape (128, 512) and subtract with tile c
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  c = nl.load(c_tensor[i_p, i_f])
  d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)
  nl.store(d_tensor[i_p, i_f], d)

  # Example 3: broadcast multiply tile e with vector f, then broadcast add with scalar 2.5
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

@nki.trace
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

## handler_bert_neuronx.py

```python
import os
import json
import torch
import torch_neuronx
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
from abc import ABC

os.environ['NEURON_RT_NUM_CORES'] = '1'

class BertEmbeddingHandler(BaseHandler, ABC):
    """
    Handler class for Bert Embedding computations.
    """
    def __init__(self):
        super(BertEmbeddingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = 'cpu'
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        with open('config.json') as fp:
            config = json.load(fp)
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        self.classes = ['not paraphrase', 'paraphrase']

        self.model = torch.jit.load(model_pt_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.initialized = True

    def preprocess(self, input_data):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for row in input_data:
            seq_0 = row['seq_0'].decode('utf-8')
            seq_1 = row['seq_1'].decode('utf-8')

            inputs = self.tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                    )

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            token_type_ids.append(inputs['token_type_ids'])

        batch = (torch.cat(input_ids, 0),
                torch.cat(attention_masks, 0),
                torch.cat(token_type_ids, 0))

        return batch

    def inference(self, inputs):
        assert(len(inputs) == 3)
        num_inferences = len(inputs[0])
        assert(num_inferences <= self.batch_size)

        padding = self.batch_size - num_inferences
        if padding > 0:
            pad = torch.nn.ConstantPad1d((0, 0, 0, padding), value=0)
            inputs = [pad(x) for x in inputs]

        outputs = self.model(*inputs)[0]
        predictions = []
        for i in range(num_inferences):
            prediction = self.classes[outputs[i].argmax(dim=-1).item()]
            predictions.append([prediction])
        return predictions

    def postprocess(self, inference_output):
        return inference_output
```

## handler_bert.py

```python
import os
import json
import torch
import torch_neuron
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
from abc import ABC

os.environ['NEURON_RT_NUM_CORES'] = '1'

class BertEmbeddingHandler(BaseHandler, ABC):
    """
    Handler class for Bert Embedding computations.
    """
    def __init__(self):
        super(BertEmbeddingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = 'cpu'
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        with open('config.json') as fp:
            config = json.load(fp)
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        self.classes = ['not paraphrase', 'paraphrase']

        self.model = torch.jit.load(model_pt_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.initialized = True

    def preprocess(self, input_data):
        """
        Tokenization pre-processing
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for row in input_data:
            seq_0 = row['seq_0'].decode('utf-8')
            seq_1 = row['seq_1'].decode('utf-8')

            inputs = self.tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                    )

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            token_type_ids.append(inputs['token_type_ids'])

        batch = (torch.cat(input_ids, 0),
                torch.cat(attention_masks, 0),
                torch.cat(token_type_ids, 0))

        return batch

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        assert(len(inputs) == 3)
        num_inferences = len(inputs[0])
        assert(num_inferences <= self.batch_size)

        padding = self.batch_size - num_inferences
        if padding > 0:
            pad = torch.nn.ConstantPad1d((0, 0, 0, padding), value=0)
            inputs = [pad(x) for x in inputs]

        outputs = self.model(*inputs)[0]
        predictions = []
        for i in range(num_inferences):
            prediction = self.classes[outputs[i].argmax().item()]
            predictions.append([prediction])
        return predictions

    def postprocess(self, inference_output):
        return inference_output
```

## test_nki_isa_bn_stats.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
```

```python
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
```

```python
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

## neuronperf_compile_guide.rst

```python
import neuronperf as npf
import neuronperf.torch

npf.torch.compile(model, inputs)  # compile for current instance type
npf.torch.compile(model, inputs, compiler_target="inf2")  # compile for inf2
```

```python
import torch
import neuronperf as npf
import neuronperf.torch

# Select a few batch sizes and pipeline configurations to test
batch_sizes = [1, 5, 10]
pipeline_sizes = [1, 2, 4]

# Construct example inputs
example_inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float16) for batch_size in batch_sizes]

# Compile all configurations
index = npf.torch.compile(
    model,
    example_inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
)
```

```python
import neuronperf as npf

# Compile with pipeline size 1 and vary batch dimension
batch_index = npf.torch.compile(
    model,
    example_inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=1,
)

# Compile with batch size 1 and vary pipeline dimension
pipeline_index = npf.torch.compile(
    model,
    example_inputs[0],
    batch_sizes=1,
    pipeline_sizes=pipeline_sizes,
)

index = npf.model_index.append(batch_index, pipeline_index)
npf.model_index.save(index, 'model_index.json')
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

@nki.trace
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

torch::jit::script::Module get_model(const std::string& filename)
{
    torch::jit::script::Module model = torch::jit::load(filename);
    return model;
}
```

## train_tb.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
device = 'xla'

# Move model to device and declare optimizer and loss function
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
```

```python
# Training loop with XLA
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
    xm.mark_step() # XLA: collect ops and run them in XLA runtime
```

```python
# XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
checkpoint = {'state_dict': model.state_dict()}
xm.save(checkpoint, 'checkpoints/checkpoint.pt')
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
# Copy tensors to NeuronDevice
device = xm.xla_device()
input_tensor = input_tensor.to(device=device)
gamma_vector = gamma_vector.to(device=device)
beta_vector = beta_vector.to(device=device)

# Compute NKI layernorm kernel in NeuronDevice
xm.mark_step()
output_nki = nki_layernorm_kernel(input_tensor, epsilon, gamma_vector, beta_vector)
xm.mark_step()
output_nki = output_nki.to(device='cpu')
```

## average_pool2d_nki_kernels.py

```python
import numpy as np
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

## test_nki_nl_load_store.py

```python
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def example_kernel(in_tensor, use_scalar=False):
  out_tensor = nl.ndarray(in_tensor.shape, in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # load from in_tensor[P, F] that is on HBM
  # copy into data_tile[P, F] that is on SBUF
  data_tile = nl.load(in_tensor)
  
  if use_scalar:
    scalar = 100
    # store scalar into out_tensor on HBM (effectively a memset)
    nl.store(out_tensor, scalar)
  else:
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

## transpose2d_nki_kernels.py

```python
import numpy as np
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
  i_p0, i_f1, i_f2 = nl.mgrid[:sz_p, :sz_f1, :sz_f2]

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

## test_nki_isa_sequence_bounds.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
```

```python
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

## compile.py

```python
import torch
from torch_neuron import trace
# or for trainium:
# from torch_neuronx.xla_impl.trace import trace

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Prepare example inputs for tracing
sequence_0 = "The company HuggingFace is based in New York City"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

example_inputs = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Compile model with torch.neuron.trace
model_neuron = trace(model, example_inputs)

# Run inference
output = model_neuron(*example_inputs)

# Save compiled model
model_neuron.save(f'bert_neuron_b{batch_size}.pt')
```

## spmd_tensor_addition_nki_kernels.py

```python
import numpy as np
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

## test_nki_isa_nc_transpose.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def nki_nc_transpose(a_tensor, b_tensor):
  at_tensor = nl.ndarray([a_tensor.shape[1], a_tensor.shape[0]], dtype=a_tensor.dtype,
                         buffer=nl.shared_hbm)
  bt_tensor = nl.ndarray([b_tensor.shape[1], b_tensor.shape[0]], dtype=b_tensor.dtype,
                         buffer=nl.shared_hbm)
  
  # Example 1: transpose tile a of shape (128, 64)
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])
  aT = nisa.nc_transpose(a[i_p_a, i_f_a])
  
  i_p_aT = nl.arange(64)[:, None]
  i_f_aT = nl.arange(128)[None, :]
  nl.store(at_tensor[i_p_aT, i_f_aT], aT)

  # Example 2: transpose tile b of shape (32, 2) using Vector Engine
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b])
  bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)
  
  i_p_bT = nl.arange(2)[:, None]
  i_f_bT = nl.arange(32)[None, :]
  nl.store(bt_tensor[i_p_bT, i_f_bT], bT)
  
  return at_tensor, bt_tensor
```

## opt_benchmark.py

```python
import os
import neuronperf as npf
import torch
from transformers import AutoTokenizer

BATCH_SIZE = 2
TP_DEGREE = 2
SEQ_LEN = 2048
TOKENIZER = AutoTokenizer.from_pretrained("facebook/opt-13b")
MODEL_DIR = "./opt-13b-split"


class Wrapper(torch.nn.Module):
    def __init__(self, filename):
        super().__init__()
        from transformers_neuronx.opt.model import OPTForSampling
        self.neuron_model = OPTForSampling.from_pretrained(
            filename, batch_size=BATCH_SIZE, tp_degree=TP_DEGREE, amp="f16"
        )
        self.neuron_model.to_neuron()

    def forward(self, *inputs):
        return self.neuron_model.sample(torch.concat(inputs), sequence_length=SEQ_LEN)


def load_fn(filename, **kwargs):
    return Wrapper(filename)


def env_setup_fn(*_):
    del os.environ["NEURON_RT_VISIBLE_CORES"]


def preprocess_fn(inputs):
    return [TOKENIZER.encode(text, return_tensors="pt") for text in inputs]


def postprocess_fn(outputs):
    return [TOKENIZER.decode(seq) for seq in outputs]


reports = npf.benchmark(
    load_fn,
    MODEL_DIR,
    [inputs],
    batch_sizes=1,
    n_models=1,
    max_infers=5,
    max_duration=0,
    workers_per_model=1,
    env_setup_fn=env_setup_fn,
    preprocess_fn=preprocess_fn,
    postprocess_fn=postprocess_fn,
)

report = reports[0]
npf.print_report(report)
npf.write_json(report)
```

## train_monitor.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Move model to XLA device
device = 'xla'
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

# Training loop with XLA
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
    xm.mark_step()  # Collect ops and run them in XLA runtime

# Save checkpoint using XLA
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
# Example 2: Distribute SPMD kernel instances to physical NeuronCores with
# explicit annotations. Expected physical NeuronCore assignments:
#   Physical NC [0]: kernel[0, 0], kernel[0, 1], kernel[1, 0], kernel[1, 1]
#   Physical NC [1]: kernel[2, 0], kernel[2, 1], kernel[3, 0], kernel[3, 1]
dst = nki_spmd_kernel[nl.spmd_dim(nl.nc(2), 2), 2](src)
dst = nki_spmd_kernel[nl.nc(2) * 2, 2](src)  # syntactic sugar
```

```python
# Example 3: Distribute SPMD kernel instances to physical NeuronCores with
# explicit annotations. Expected physical NeuronCore assignments:
#   Physical NC [0]: kernel[0, 0], kernel[0, 1], kernel[2, 0], kernel[2, 1]
#   Physical NC [1]: kernel[1, 0], kernel[1, 1], kernel[3, 0], kernel[3, 1]
dst = nki_spmd_kernel[nl.spmd_dim(2, nl.nc(2)), 2](src)
dst = nki_spmd_kernel[2 * nl.nc(2), 2](src)  # syntactic sugar
```

## tutorial-tensorboard-scalars-mnist.rst

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./output')
```

```python
writer.add_scalar("step loss", loss, idx)
```

```python
writer.flush()
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

  # dynamic indices have to be in SBUF, with shape [N, 1]
  indices_tile: tensor[N, 1] = nl.load(indices_tensor)

  ix = nl.arange(M)[None, :]

  # Atomic read-modify-write example:
  #   - read: values of rmw_tensor is indexed by values from indices_tile
  #   - modify: incremented by value
  #   - write: saved back into rmw_tensor
  # resulting in rmw_tensor = rmw_tensor + value
  nl.atomic_rmw(rmw_tensor[indices_tile, ix], value=value, op=np.add)
  
  return rmw_tensor
```

## detect_instance.py

```python
import torch
import torch_neuronx
from typing import Optional

def get_num_neuroncores_v3() -> int:
    """
    Retrieve the number of NeuronCores visible to this process.

    Returns:
        The number of visible neuron cores.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized. This most
            commonly occurs when executing on an instance with no Neuron
            devices available or when no Neuron devices are visible to the
            process.
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

## mlp_train.py

```python
import torch
import torch_xla.core.xla_model as xm
from model import MLP

# XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
device = 'xla'

# Move model to device and declare optimizer and loss function
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

# Training loop
model.train()
for epoch in range(EPOCHS):
    for idx, (train_x, train_label) in enumerate(train_loader):
        optimizer.zero_grad()
        train_x = train_x.view(train_x.size(0), -1)
        train_x = train_x.to(device)
        train_label = train_label.to(device)
        output = model(train_x)
        loss = loss_fn(output, train_label)
        loss.backward()
        optimizer.step()
        xm.mark_step()  # XLA: collect ops and run them in XLA runtime

# XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
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


@nki.jit(mode="simulation")
def nki_dropout_scalar(in_tensor):
  import neuronxcc.nki.language as nl
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

@nki.trace
def nki_par_reduce(a_tensor, b_tensor):
  """
  Example 1: reduce add tile a of shape (128, 32, 4)
  in the partition dimension and return
  reduction result in tile b of shape (1, 32, 4)
  """
  a = nl.load(a_tensor[0:128, 0:32, 0:4])  
  b = nisa.tensor_partition_reduce(np.add, a)
  nl.store(b_tensor[0:1, 0:32, 0:4], b)

@nki.trace
def nki_par_reduce_nd_b(a_tensor, b_tensor):
  """
  Example 2: reduce add tile a of shape (b, p, f1, ...)
  in the partition dimension p and return
  reduction result in tile b of shape (b, 1, f1, ...)
  """
  for i in nl.affine_range(a_tensor.shape[0]):
    a = nl.load(a_tensor[i])
    b = nisa.tensor_partition_reduce(np.add, a)
    nl.store(b_tensor[i], b)
```

## index-case-3.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def tensor_maxpool_kernel_(in_tensor, sz_pool):
  """NKI kernel to compute a 2D max-pool operation

  Args:
      in_tensor: an input tensor, of dimensions C x H x W
      sz_pool: integer P representing a (square) pool-window size
  Returns:
      out_tensor: the resulting output tensor, of dimensions C x (H/P) x (W/P)
  """

  # Get input/output dimensions
  sz_p, sz_hin, sz_win = in_tensor.shape
  sz_hout, sz_wout = sz_hin // sz_pool, sz_win // sz_pool
  out_tensor = nl.ndarray((sz_p, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i_0, i_1, i_2, i_3, i_4 = nl.mgrid[:sz_p, :sz_hout, :sz_pool, :sz_wout, :sz_pool]

  # Load input data from external memory to on-chip memory
  # Declare ndarray to force a 3D tensor (temporary requirement)
  in_tile = nl.ndarray((sz_p, sz_hin, sz_win), dtype=in_tensor.dtype)
  in_tile[...] = nl.load(in_tensor)

  # Perform the pooling operation:
  # We use advanced indexing, in order to extend in_tile to 5D, and then reduce-max two dimension.
  # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
  # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
  # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
  # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].
  out_tile = nl.max(in_tile[i_0, sz_pool*i_1+i_2, sz_pool*i_3+i_4], axis=[2,4])

  # Store the results back to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor
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

## EVRF016.rst

```python
def forward(self, input_tensor, indices_tensor, src_tensor):
    output = input_tensor.clone()
    
    output.scatter_reduce_(
        dim=1,
        index=indices_tensor,
        src=src_tensor,
        reduce='sum',
    )
    return output
```

## EOOM001.rst

```python
class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.query = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        # Since we shard the number of attention heads across tensor parallel
        # ranks, each rank would have a subset of heads, hence, we update
        # the num_attention_heads here.
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
```

## EOOM002.rst

```python
class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.query = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        # Since we shard the number of attention heads across tensor parallel
        # ranks, each rank would have a subset of heads, hence, we update
        # the num_attention_heads here.
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
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


@nki.jit(mode="simulation")
def example_kernel_1(in_tensor):
  out_tensor = nl.ndarray([in_tensor.shape[1], in_tensor.shape[0]], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  import neuronxcc.nki.isa as nisa
  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  # always use the DMA engine
  N, M = in_tensor.shape
  local_tile: tensor[M, N] = nisa.dma_transpose(in_tensor)
  nl.store(out_tensor, value=local_tile)
  return out_tensor
```

## api-tfn-analyze-model-api.rst

```python
import tensorflow as tf
import tensorflow.neuron as tfn

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
model = tf.keras.Model(inputs=[input0], outputs=[dense0])
example_inputs = tf.random.uniform([1, 3])
results = tfn.analyze_model(model, example_inputs)
print(results)
```

## tfnx-analyze-model-api.rst

```python
import tensorflow as tf
import tensorflow_neuron as tfnx

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
model = tf.keras.Model(inputs=[input0], outputs=[dense0])
example_inputs = tf.random.uniform([1, 3])
results = tfnx.analyze_model(model, example_inputs)
print(results)
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

## inference.py

```python
import os
import json
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def model_fn(model_dir):
    tokenizer_init = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model_file = os.path.join(model_dir, 'neuron_compiled_model.pt')
    model_neuron = torch.jit.load(model_file)
    return (model_neuron, tokenizer_init)

def predict_fn(input_data, models):
    model_bert, tokenizer = models
    sequence_0 = input_data[0] 
    sequence_1 = input_data[1]
    
    max_length = 128
    paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']  
    
    paraphrase_classification_logits_neuron = model_bert(*example_inputs_paraphrase)
    classes = ['not paraphrase', 'paraphrase']
    paraphrase_prediction = paraphrase_classification_logits_neuron[0][0].argmax().item()
    out_str = 'BERT says that "{}" and "{}" are {}'.format(sequence_0, sequence_1, classes[paraphrase_prediction])
    
    return out_str
```

## EXSP001.rst

```python
class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.query = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        # Since we shard the number of attention heads across tensor parallel
        # ranks, each rank would have a subset of heads, hence, we update
        # the num_attention_heads here.
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
```

**Note:** This example demonstrates tensor parallelism for memory optimization on Trainium, but is not specific to NKI kernel optimization. For NKI-specific kernel optimization examples, additional documentation would be needed.

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

## tokenizer_test.cpp

```cpp
#include <cstring>
#include <vector>

#include "remote_rust_tokenizer.h"

// Tokenizer API usage example
const uint32_t seq_len = 128;
const char *input_arr = "If everything goes smoothly, this text will be tokenized inside Rust.";
uint32_t* output_arr = new uint32_t[seq_len];
std::memset(output_arr, 0, sizeof(uint32_t) * seq_len);

// Call tokenizer
remote_rust_encode(input_arr, output_arr, seq_len);
```

## matrix_multiplication_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

device = xm.xla_device()
cpu = torch.device('cpu')

# Test the small workload with basic kernel
lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

# Run NKI kernel
output_small = nki_matmul_basic_(lhs_small.T, rhs_small)

# Run torch reference
output_small_torch = torch.matmul(lhs_small, rhs_small)

# Compare results
if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")
```

```python
# Test the large workload with tiled kernels
lhs = torch.rand((4096, 1024), dtype=torch.bfloat16, device=device)
rhs = torch.rand((1024, 2048), dtype=torch.bfloat16, device=device)

# Run torch reference
output_torch = torch.matmul(lhs, rhs).to(device=cpu)

# Run NKI kernels
output_tiled = nki_matmul_tiled_(lhs.T, rhs).to(device=cpu)
output_hoist = nki_matmul_hoist_load_(lhs.T, rhs).to(device=cpu)
output_block_free = nki_matmul_block_free_dimension_(lhs.T, rhs).to(device=cpu)
output_optimized = nki_matmul_fully_optimized_(lhs.T, rhs).to(device=cpu)

# Compare results
if torch.allclose(output_torch, output_tiled, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")
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

## EVRF009.rst

```python
class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.query = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        # Since we shard the number of attention heads across tensor parallel
        # ranks, each rank would have a subset of heads, hence, we update
        # the num_attention_heads here.
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
```

## EVRF015.rst

```python
def lowering(ctx, x_val):
    result_type = ir.RankedTensorType(x_val.type)
    return hlo.CustomCallOp(
        [result_type],
        [x_val],
        call_target_name="AwsNeuronSilu",
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
    ).results
```

## EHCA005.rst

```python
def lowering(ctx, x_val):
    result_type = ir.RankedTensorType(x_val.type)
    return hlo.CustomCallOp(
        [result_type],
        [x_val],
        call_target_name="AwsNeuronSilu",
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
    ).results
```

## parallel.py

```python
from concurrent import futures
import torch
import torch.neuron
import os
from queue import Queue

def consumer(model, input_queue):
    while True:
        inputs, input_id, callback_fn = input_queue.get()
        input_queue.task_done()
        # Stop execution if stopping condition is recieved
        if inputs == "stop":
            break
        results = model(*inputs)
        # Make the output iterable - if it is not already a tuple or list
        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]
        if callback_fn is not None:
            callback_fn(results, input_id)
              
class NeuronSimpleDataParallel():

    def __init__(self, model_file, num_neuron_cores, batch_size=1):
        self.num_neuron_cores = num_neuron_cores
        self.batch_size = batch_size
        
        os.environ['NEURON_RT_NUM_CORES'] = str(num_neuron_cores)
        
        # Construct a list of models
        self.models = [torch.jit.load(model_file)
                       for i in range(num_neuron_cores)]
        
        # Create shared input queue
        self.input_queue = Queue(maxsize=num_neuron_cores*16)

        self.executor = futures.ThreadPoolExecutor(
            max_workers=num_neuron_cores)

    def eval(self):
        for model in self.models:
            model.eval()
            
    def train(self):
        for model in self.models:
            model.train()
            
    def start_continuous_inference(self):
        for model in self.models:
            self.executor.submit(consumer, model, self.input_queue)
    
    def infer(self, batch, input_id, callback_fn):
        self.input_queue.put((batch, input_id, callback_fn))
        
    def stop(self):
        for _ in range(self.num_neuron_cores):
            self.input_queue.put(("stop", -1, None))
```

## test_nki_isa_copypredicated.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


@nki.jit(mode="simulation")
def nki_copy_predicated(predicate, on_true_tensor, on_false_tensor):
  """
  Conditionally copies elements from the `on_true` tile to 
  SBUF/PSUM destination tile using Vector Engine, where copying occurs 
  only at positions where the predicate evaluates to True.
  """
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

## distilbert-base-uncased-finetuned-sst-2-english_compile.py

```python
import numpy as np
import neuronperf as npf
import neuronperf.tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def get_batch(tokenizer, sequence_length, batch_size):
    sequence = "I am sorry. I really want to like it, but I just can not stand sushi."
    paraphrase = tokenizer.encode_plus(
        sequence,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    inputs = {
        "input_ids": np.concatenate([paraphrase["input_ids"]] * batch_size, axis=0),
        "attention_mask": np.concatenate([paraphrase["attention_mask"]] * batch_size, axis=0),
    }
    return inputs


# Compile a TensorFlow model for AWS Trainium
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    return_dict=False
)

inputs = [get_batch(tokenizer, sequence_length=128, batch_size=bs) for bs in [128]]

npf.tensorflow.compile(
    model,
    inputs,
    batch_sizes=[128],
    pipeline_sizes=[1],
    filename="distilbert_compiled.json",
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
)
```

## EVRF024.rst

```python
class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.query = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        self.value = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False)
        # Since we shard the number of attention heads across tensor parallel
        # ranks, each rank would have a subset of heads, hence, we update
        # the num_attention_heads here.
        tp_size = parallel_state.get_tensor_parallel_size()
        self.num_attention_heads = self.num_attention_heads // tp_size
        self.all_head_size = self.all_head_size // tp_size
```

## test_nki_nl_gather_flattened.py

```python
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
```

## EVRF031.rst

```python
import jax.numpy as jnp
from jax import lax

# Erroneous code example
operand = jnp.zeros((3, 4), dtype=jnp.float32)
indices = lax.iota(jnp.int32, 10)
indices = indices.reshape(10, 1)
updates = jnp.ones((10, 4), dtype=jnp.float32)

result = lax.scatter(
    operand,
    indices,
    updates,
    lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
)
```

```python
import jax.numpy as jnp
from jax import lax

# Fixed code example
N = 3
D = 4
operand = jnp.zeros((N, D), dtype=jnp.float32)

indices = lax.iota(jnp.int32, N)
indices = indices.reshape(N, 1)

updates = jnp.ones((N, D), dtype=jnp.float32)

result = lax.scatter(
    operand,
    indices,
    updates,
    lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
)
```

## test_nki_nl_mgrid.py

```python
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

## index-case-1.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

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

  # This example only works for tensors with a partition dimension that fits in the SBUF
  assert in_tensor.shape[0] <= nl.tile_size.pmax

  # Extract tile sizes.
  sz_p, sz_f = in_tensor.shape
  sz_fout_even = sz_f - sz_f // 2
  sz_fout_odd = sz_f // 2

  # create output tensors
  out_tensor_even = nl.ndarray((sz_p, sz_fout_even), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  out_tensor_odd = nl.ndarray((sz_p, sz_fout_odd), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Store the results back to external memory
  nl.store(out_tensor_even, value=in_tile[:, 0:sz_f:2])
  nl.store(out_tensor_odd,  value=in_tile[:, 1:sz_f:2])

  return out_tensor_even, out_tensor_odd
```

## test_nki_isa_nc_find_index8.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor


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

## distilbert-base-uncased-finetuned-sst-2-english_compile.py

```python
import torch
import torch.neuron

import neuronperf
import neuronperf.torch

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)
inputs = [
    get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes
]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=filename,
    model_name=model_name,
)
```

## distilbert-base-uncased_compile.py

```python
import torch
import torch.neuron

import neuronperf
import neuronperf.torch

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
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)

sequence_length = 128
batch_sizes = [9]
pipeline_sizes = [1]

inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=f"{model_name}_sl{sequence_length}.json",
    model_name=model_name,
)
```

## distilroberta-base_compile.py

```python
import torch
import torch.neuron

import neuronperf
import neuronperf.torch

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
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", return_dict=False)

inputs = [get_batch(tokenizer, sequence_length=128, batch_size=bs) for bs in [6]]

neuronperf.torch.compile(
    model,
    inputs,
    batch_sizes=[6],
    pipeline_sizes=[1],
    filename="distilroberta-base_sl128.json",
    model_name="distilroberta-base",
)
```

## EVRF019.rst

```python
import jax.lax as lax
import jax.numpy as jnp

# Correct usage: max pooling with reduce_window
max_pool = lax.reduce_window(
    x,         # single input tensor
    -jnp.inf,  # single initial value
    lax.max,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding='VALID'
)

# Correct usage: min pooling with reduce_window
min_pool = lax.reduce_window(
    x,        # single input tensor
    jnp.inf,  # single initial value
    lax.min,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding='VALID'
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
  
  # Example: reduce add tile a of shape (128, 512)
  # in the free dimension and return
  # reduction result in tile b of shape (128, 1)
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

# Trace the model for AWS Trainium
model.eval()
traced = torch_neuronx.trace(model, inputs, compiler_args='--enable-saturate-infinity')
filename = 'model.pt'
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

  # Example: add two tiles, a and b, of the same
  # shape (128, 512) element-wise and get
  # the addition result in tile c
  a: tensor[128, 512] = nl.load(a_tensor)
  b: tensor[128, 512] = nl.load(b_tensor)

  c: tensor[128, 512] = nisa.tensor_tensor(a, b, op=nl.add)

  nl.store(c_tensor, c)
  return c_tensor
```

## test_nki_nl_broadcast.py

```python
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

## prof-kernel-profile.py

```python
from neuronxcc import nki
from neuronxcc.nki.typing import tensor
import neuronxcc.nki.language as nl
import math
from pathlib import Path

WORKING_DIRECTORY = Path.home() / 'reports'

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

## bert_no_model.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

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

# Compile SavedModel with selective operator fusion
compilation_result = tfn.saved_model.compile(
    input_saved_model_dir,
    output_saved_model_dir,
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

  # Example: Copy over the tensor to another tensor using the Vector engine.
  x = nl.load(in_tensor)
  x_copy = nisa.tensor_copy(x, engine=nisa.vector_engine)
  nl.store(out_tensor, value=x_copy)

  return out_tensor
```

## hf_pretrained_wav2vec2_conformer_relpos_benchmark.py

```python
import torch
import torch_neuronx
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC

BATCH_SIZE = 1

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
model.eval()

# Prepare input data
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
inputs = inputs.repeat([BATCH_SIZE, 1])
example = (inputs,)

# Trace model for Trainium
traced = torch_neuronx.trace(model, example, compiler_args='--model-type=transformer')

# Save traced model
filename = 'model.pt'
torch.jit.save(traced, filename)

# Load and run inference
model_neuron = torch.jit.load(filename)
output = model_neuron(inputs)
```

## hf_pretrained_wav2vec2_conformer_rope_benchmark.py

```python
import torch
import torch_neuronx
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC

BATCH_SIZE = 1

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model.eval()

# Prepare input data
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
inputs = inputs.repeat([BATCH_SIZE, 1])
example = (inputs,)

# Trace model for Trainium
traced = torch_neuronx.trace(model, example, compiler_args='--model-type=transformer')

# Save traced model
filename = 'model.pt'
torch.jit.save(traced, filename)

# Load and run inference
model_neuron = torch.jit.load(filename)
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

## EVRF005.rst

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Convert to a supported type
input_tensor = torch.randn(1, 10).to(torch.float16)
```

## test_nki_mask.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

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
  # Example 1: Initialize a float32 tile a of shape (128, 128)
  # with a value of 0.2
  a = nisa.memset(shape=(128, 128), value=0.2, dtype=nl.float32)

  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(128)[None, :]
  nl.store(a_tensor[i_p, i_f], a)
  return a_tensor
```

## hf-google-vit_benchmark.py

```python
import torch
import torch_neuronx
from transformers import ViTImageProcessor, ViTForImageClassification

# Load model and processor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torchscript=True)
model.eval()

# Prepare example input
from PIL import Image
import requests
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")
inputs = inputs['pixel_values'].repeat([2, 1, 1, 1])
example = (inputs,)

# Trace model for Trainium
traced = torch_neuronx.trace(model, example, compiler_args="--model-type=transformer")

# Save traced model
torch.jit.save(traced, 'model.pt')
```

## test_nki_isa_affine_select.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def nki_affine_select(a_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  # Example: Take tile a of shape [128, 128] and replace its
  # upper triangle with nl.fp32.min
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
# Note that image_batched.shape[dim] / len(device_ids) == image.shape[dim]
batch_size = 4 * 8
image_batched = torch.rand([1, 3, batch_size, 8])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## EVRF010.rst

```python
import jax.numpy as jnp
from jax import lax

# Erroneous code example - simultaneous input and kernel dilation
x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

result = lax.conv_general_dilated(
    x,
    kernel,
    window_strides=(1, 1),
    padding=((2, 2), (2, 2)),
    lhs_dilation=(2, 2), # input dilation
    rhs_dilation=(2, 2), # kernel dilation
    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
)
```

```python
import jax.numpy as jnp
from jax import lax

# Corrected code example - kernel dilation only
x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

result = lax.conv_general_dilated(
    x,
    kernel,
    window_strides=(1, 1),
    padding=((2, 2), (2, 2)),
    lhs_dilation=(1, 1), # no input dilation
    rhs_dilation=(2, 2),
    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
)
```

## EVRF011.rst

```python
import jax.numpy as jnp
from jax import lax

# Erroneous code example - strided convolution with dilated input (not supported)
x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

result = lax.conv_general_dilated(
    x,
    kernel,
    window_strides=(2, 2),    # strided convolution
    padding=((2, 2), (2, 2)),
    lhs_dilation=(2, 2),      # and dilated input
    rhs_dilation=(1, 1),
    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
)
```

```python
import jax.numpy as jnp
from jax import lax

# Corrected code example - remove input dilation
x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

result = lax.conv_general_dilated(
    x, kernel,
    window_strides=(2, 2),  
    padding=((2, 2), (2, 2)),
    lhs_dilation=(1, 1),    # remove input dilation
    rhs_dilation=(1, 1),
    dimension_numbers=('NHWC', 'HWIO', 'NHWC')
)
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

q_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
k_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
v_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)

output_nki = fused_self_attn_for_SD_small_head_size(q_tensor, k_tensor, v_tensor)
```

## perceiver-vision_compile.py

```python
import torch
import transformers
import neuronperf as npf
import neuronperf.torch


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


# Compile a model for AWS Trainium
model = transformers.PerceiverForImageClassificationLearned.from_pretrained(
    "deepmind/vision-perceiver-learned"
)
inputs = [get_batch(batch_size) for batch_size in [1]]
batch_sizes = [1]
pipeline_sizes = [1]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename="perceiver_vision.json",
    model_name="vision-perceiver-learned",
)
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
# Note that image_batched.shape[dim] / len(device_ids) == image.shape[dim]
batch_size = 2 * 8
image_batched = torch.rand([1, 3, batch_size, 8])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## ESFH002.rst

```python
import jax
import jax.numpy as jnp

@jax.jit
def foo():
    # direct uint64 constant in arithmetic operation
    x = jnp.array([1, 2, 3], dtype=jnp.uint64)
    # large constant that exceeds uint32 max
    large_constant = jnp.uint64(5_000_000_000)
    return x + large_constant
```

```python
import jax
import jax.numpy as jnp

@jax.jit
def test():
    x = jnp.array([1, 2, 3], dtype=jnp.uint32)
    large_constant = jnp.uint32(5_000_000_000)
    return x + large_constant
```

## trace_bert_neuronx.py

```python
import torch
import torch_neuronx

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

# Trace model for AWS Trainium optimization
model_neuron_batch = torch_neuronx.trace(model, example_inputs_paraphrase)

# Save the optimized model
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
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

# Trace model with torch_neuron for AWS Neuron optimization
model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)

# Save the optimized model
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
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

# Trace model with torch_neuron for AWS Trainium optimization
model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)

# Save the optimized model
model_neuron_batch.save('bert_neuron_b{}.pt'.format(batch_size))
```

## rmsnorm_torch.py

```python
import torch
from rmsnorm_nki_kernels import nki_rmsnorm_kernel
from torch_xla.core import xla_model as xm

# Reference torch implementation
def torch_rmsnorm_kernel(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = a_tensor.pow(2)
  # Calculate means in the free dimension
  mean = in_square.mean(dim=1, keepdim=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * torch.rsqrt(mean)

  # Scale the output by the weight
  return tensor * g_tensor

device = xm.xla_device()

a_tensor = torch.rand((250, 512), dtype=torch.float32).to(device=device)
g_tensor = torch.rand((512), dtype=torch.float32).to(device=device)

output_nki = nki_rmsnorm_kernel(a_tensor, g_tensor)
output_torch = torch_rmsnorm_kernel(a_tensor, g_tensor)
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
out_nki = tensor_avgpool_kernel(in_array, pool_size=POOL_SIZE)
```

## ESPP004.rst

```python
import numpy as np
import jax.numpy as jnp
import jax
from jax._src import dtypes
from jax._src.lax import lax as lax_internal

# Unsupported data type example
dtype = np.dtype(dtypes.float4_e2m1fn)
val = lax_internal._convert_element_type(0, dtype, weak_type=False)
```

```python
import numpy as np
import jax.numpy as jnp
import jax
from jax._src import dtypes
from jax._src.lax import lax as lax_internal

# Supported data type example
dtype = jnp.bfloat16
val = lax_internal._convert_element_type(0, dtype, weak_type=False)
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

## EARG001.rst

```python
traced_model = torch_neuronx.trace(
   model,
   input,
   compiler_args=['--lnc', '2']  # ERROR: lnc=2 not supported on trn1
)
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

  # Load input data from HBM to on-chip memory
  in_tile = nl.load(in_tensor[0:256, 0:512])

  # perform the computation:
  out_tile = nl.exp(in_tile)

  # store the results back to HBM
  nl.store(out_tensor[0:256, 0:512], value=out_tile)
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

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    p_start = k * nl.tile_size.pmax
    p_end = p_start + nl.tile_size.pmax
    i_p = slice(p_start, min(p_end, sz_p))

    # Load input data from external memory to on-chip memory
    in_tile = nl.load(in_tensor[i_p, 0:sz_f])

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to external memory
    nl.store(out_tensor[i_p, 0:sz_f], value=out_tile)

    return out_tensor
```

## average_pool2d_torch.py

```python
import torch
from torch_xla.core import xla_model as xm
from average_pool2d_nki_kernels import tensor_avgpool_kernel

device = xm.xla_device()

# Set up average pool 2D parameters
POOL_SIZE = 2
C, HIN, WIN = 2, 6, 6
HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

# Create input tensor and move to device
in_tensor = torch.arange(C * HIN * WIN, dtype=torch.bfloat16).reshape(C, HIN, WIN).to(device=device)

# Call NKI kernel
out_nki = tensor_avgpool_kernel(in_tensor, POOL_SIZE)

# Compare with PyTorch reference implementation
out_torch = torch.nn.functional.avg_pool2d(in_tensor, POOL_SIZE, POOL_SIZE)
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

  # Load input data from HBM to on-chip memory
  in_tile = nl.load(in_tensor[0:128, 0:512])

  # perform the computation:
  out_tile = nl.exp(in_tile)

  # store the results back to HBM
  nl.store(out_tensor[0:128, 0:512], value=out_tile)

  return out_tensor
```

## test_nki_nl_dslice.py

```python
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

## EVRF017.rst

```python
import jax.lax as lax
import jax.numpy as jnp

# Erroneous code example - base dilation greater than 1 not supported
result = lax.reduce_window(
    x, -jnp.inf, lax.max,
    window_dimensions=(1, 1, 1, 1),
    window_strides=(1, 1, 1, 1),
    padding='VALID',
    base_dilation=(1, 2, 1, 1)
)

# Fixed code example - base dilation set to all 1s
result = lax.reduce_window(
    x, -jnp.inf, lax.max,
    window_dimensions=(1, 1, 1, 1),
    window_strides=(1, 1, 1, 1),
    padding='VALID',
    base_dilation=(1, 1, 1, 1)
)
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

  for k in nl.affine_range(2):
    # Generate tensor indices for the input/output tensors
    p_start = k * nl.tile_size.pmax
    p_end = p_start + nl.tile_size.pmax
    i_p = slice(p_start, p_end)

    # Load input data from HBM to on-chip memory
    in_tile = nl.load(in_tensor[i_p, 0:512])

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
```

## resnet_compile.py

```python
import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

model = getattr(torchvision.models, "resnet50")(pretrained=True)
inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32) for batch_size in [1, 8, 64]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1, 8, 64],
    pipeline_sizes=[1],
    filename="resnet50.json",
    model_name="resnet50",
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
```

## spmd_tensor_addition_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

a = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)
b = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)

output_nki = nki_tensor_add(a, b)
```

## vgg_compile.py

```python
import torch
import torchvision
import neuronperf as npf
import neuronperf.torch

model = getattr(torchvision.models, "vgg16")(pretrained=True)
inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32) for batch_size in [1, 8, 64]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1, 8, 64],
    pipeline_sizes=[1],
    filename="vgg16.json",
    model_name="vgg16",
)
```

## spmd_tensor_addition_jax.py

```python
import jax
import jax.numpy as jnp

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

## EVRF001.rst

```python
class Model(torch.nn.Module):
    def forward(self, A, b):
        # Although slower than triangular_solve, this is mathematically equivalent
        A_inv = torch.inverse(A)
        return A_inv @ b
```

## EUOC002.rst

```python
class Model(torch.nn.Module):
    def forward(self, A, b):
        return torch.triangular_solve(b, A)
```

```python
class Model(torch.nn.Module):
    def forward(self, A, b):
        # Although slower than triangular_solve, this is mathematically equivalent
        A_inv = torch.inverse(A)
        return A_inv @ b
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
import torch.neuron
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

## EVRF013.rst

```python
def forward(self, x):
    x = x.float()
    k = 5
    values, indices = torch.topk(x, k=k, dim=-1)
    return values, indices
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