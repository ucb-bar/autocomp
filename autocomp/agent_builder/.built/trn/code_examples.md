## attention_kernels.py

```python
import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki.language import par_dim
```

```python
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
```

```python
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
```

```python
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
```

```python
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
```

```python
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

        # Division delayed to final attention output
        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out
```

```python
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
```

```python
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

    # v has the wrong layout - downcast to bfloat16
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY):
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
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
```

```python
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
```

```python
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

## benchmarking.py

```python
from typing import Any, Callable, Dict, List, Union
import collections
import concurrent
import concurrent.futures
import copy
import functools
import logging
import multiprocessing
import os
import psutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import dill

from . import model_index
from .compile_constants import NEURONCORE_PIPELINE_CORES, FAST_MATH, FAST_MATH_OPTIONS
from .reporting import get_reports
from .scripts import run_benchmark_file
from .timing import Timer


log = logging.getLogger(__name__)

BenchmarkerErrorWrapper = collections.namedtuple("BenchmarkerErrorWrapper", "trace")

ERROR = "error"
SUPPORTED_DEVICE_TYPES = ["neuron", "cpu", "cuda", "gpu"]
BENCHMARK_SECS = 120


class Benchmarker(threading.Thread):
    r"""
    :class:`benchmarking:Benchmarker` benchmarks a single model.

    This class is a `threading.Thread`. Call `start` to launch a non-blocking
    benchmarking thread. Calling `stop` will end the benchmarking and block
    until all subroutines complete.

    An object of this class may be serialized and sent to multiple subprocesses
    for parallel use. After benchmarking, results can be obtained with
    `results`.
    """

    def __init__(
        self,
        id: int,
        device_id: int,
        load_fn: Callable[[str], Any],
        model_filename: str,
        inputs,
        workers_per_model: int,
        env_setup_fn: Callable[[int, Dict, Any], None] = None,
        setup_fn: Callable[[int, Dict, Any], None] = None,
        preprocess_fn: Callable[[Any], Any] = None,
        postprocess_fn: Callable[[Any], Any] = None,
        dataset_loader_fn: Callable[[Any, int], Any] = None,
        model_class_name: str = None,
        model_class_file: str = None,
    ):
        super().__init__()

        self.id = id
        self.device_id = device_id
        self.load_fn = load_fn
        self.model_filename = model_filename
        self.inputs = inputs
        self.input_iter = None
        self.input_lock = threading.Lock()
        self.workers_per_model = workers_per_model
        self.env_setup_fn = env_setup_fn
        self.setup_fn = setup_fn
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.dataset_loader_fn = dataset_loader_fn
        self.model_class_name = model_class_name
        self.model_class_file = model_class_file

        self.model = None
        self.benchmark_timer = Timer()
        self.env_setup_timer = Timer()
        self.setup_timer = Timer()
        self.load_timer = Timer()
        self.warmup_timer = Timer()
        self.input_timer = Timer()
        self.preprocess_timers = [Timer() for _ in range(workers_per_model)]
        self.infer_timers = [Timer() for _ in range(workers_per_model)]
        self.postprocess_timers = [Timer() for _ in range(workers_per_model)]
        self.e2e_timers = [Timer() for _ in range(workers_per_model)]
        self.worker_timers = [Timer() for _ in range(workers_per_model)]
        self.n_infs = [0] * workers_per_model
        self.process_id = 0
        self.benchmarking = False
        self.benchmarking_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.status = "ready"
        self.error = None

    def _status(self, status, error=None):
        """Update internal status, unless a previous error has occurred."""
        with self.status_lock:
            if self.status == ERROR:
                return
            self.status = status
            if error:
                self.error = error

    def next_input(self):
        self.input_lock.acquire()
        self.input_timer.start()
        try:
            return next(self.input_iter)
        finally:
            self.input_timer.stop()
            self.input_lock.release()

    def prepare_inputs(self):
        """Prepares input iterator; runs an optional custom setup function."""
        if self.dataset_loader_fn:

            def input_iter():
                dataset_loader = self.dataset_loader_fn(self.inputs, self.workers_per_model)
                while True:
                    inputs = next(dataset_loader)
                    yield inputs if isinstance(inputs, tuple) else (inputs,)

            self.input_iter = input_iter()
        else:

            def input_iter():
                inputs = self.inputs if isinstance(self.inputs, tuple) else (self.inputs,)
                while True:
                    yield inputs

            self.input_iter = input_iter()

    def load(self):
        """Loads the model that will be used for benchmarking."""
        with self.load_timer:
            self.model = self.load_fn(self.model_filename, device_id=self.device_id)

    def warmup(self):
        """Warmup the model with a single e2e inference."""
        with self.warmup_timer:
            inputs = self.next_input()
            if self.preprocess_fn:
                inputs = self.preprocess_fn(*inputs)
            outputs = self.model(*inputs if isinstance(inputs, tuple) else inputs)
            if self.postprocess_fn:
                self.postprocess_fn(outputs)
        self.n_infs[0] += 1

    def setup(self):
        """Perform all setup work prior to benchmarking."""
        self.prepare_inputs()

        if self.env_setup_fn:
            with self.env_setup_timer:
                self.env_setup_fn()

        self.load()

        if self.setup_fn:
            with self.setup_timer:
                self.setup_fn(self.model)

        self.warmup()

    def infer(self, worker_id) -> tuple:
        """Execute a single inference."""
        with self.e2e_timers[worker_id]:
            inputs = self.next_input()
            if self.preprocess_fn:
                with self.preprocess_timers[worker_id]:
                    inputs = self.preprocess_fn(*inputs)
            with self.infer_timers[worker_id]:
                outputs = self.model(*inputs if isinstance(inputs, tuple) else inputs)
            if self.postprocess_fn:
                with self.postprocess_timers[worker_id]:
                    outputs = self.postprocess_fn(outputs)
        return outputs

    def worker_thread(self, worker_id):
        """A single worker thread that runs inference until signalled to stop."""
        n_infs = 0
        try:
            log.debug(f"Benchmarker {self.id}, Worker {worker_id} started.")
            with self.worker_timers[worker_id]:
                while self.benchmarking and self.status != ERROR:
                    self.infer(worker_id)
                    n_infs += 1
            if self.status == ERROR:
                log.debug(
                    f"Benchmarker {self.id}, Worker {worker_id} stopped early due to an error after {n_infs} inferences."
                )
        except StopIteration:
            pass
        except:
            trace = "".join(traceback.format_exception(*sys.exc_info()))
            log.error(
                f"Benchmarker {self.id}, Worker {worker_id} encountered an error during benchmarking:\n{trace}"
            )
            self._status(ERROR, BenchmarkerErrorWrapper(trace))
        finally:
            self.n_infs[worker_id] += n_infs
            log.debug(
                f"Benchmarker {self.id}, Worker {worker_id} finished after {self.n_infs[worker_id]} inferences."
            )

    def run(self):
        with self.benchmarking_lock:
            if self.benchmarking:
                raise RuntimeError(
                    f"Benchmarker {self.id} can't start because it is already running."
                )
            self.benchmarking = True
            self._status("running")

        self.process_id = os.getpid()

        with self.benchmark_timer:
            try:
                self.setup()
            except:
                trace = "".join(traceback.format_exception(*sys.exc_info()))
                log.error(f"Benchmarker {self.id} encountered an error during prep:\n{trace}")
                self._status(ERROR, BenchmarkerErrorWrapper(trace))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_per_model) as exe:
                    for worker_id in range(self.workers_per_model):
                        exe.submit(self.worker_thread, worker_id)

        if self.benchmarking_lock.acquire(blocking=False):
            try:
                self.benchmarking = False
                self._status("finished")
            finally:
                self.benchmarking_lock.release()

    def stop(self):
        with self.benchmarking_lock:
            if not self.benchmarking:
                return
            self._status("stopping")
            self.benchmarking = False
            self.join()
            self._status("finished")

    def results(self) -> dict:
        with self.benchmarking_lock:
            if self.benchmarking:
                raise RuntimeError("Cannot produce results until benchmarking has completed.")
            return {
                "id": self.id,
                "device_id": self.device_id,
                "workers_per_model": self.workers_per_model,
                "n_infs": sum(self.n_infs),
                "status": self.status,
                "process_id": self.process_id,
                "total_s": self.benchmark_timer.total_duration("s"),
                "timers": {
                    "env_setup": [self.env_setup_timer],
                    "setup": [self.setup_timer],
                    "load": [self.load_timer],
                    "input": [self.input_timer],
                    "warmup": [self.warmup_timer],
                    "preprocess": self.preprocess_timers,
                    "infer": self.infer_timers,
                    "postprocess": self.postprocess_timers,
                    "e2e": self.e2e_timers,
                    "worker": self.worker_timers,
                },
            }


class StatsThread(threading.Thread):
    """A thread to collect some system metrics during benchmarking."""

    def __init__(self, interval: float):
        super().__init__()
        self.interval = interval
        self.cpu_percents = []
        self.mem_percents = []
        self.running = True

    def run(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=self.interval, percpu=False)
            mem_percent = psutil.virtual_memory()[2]
            self.cpu_percents.append(cpu_percent)
            self.mem_percents.append(mem_percent)

    def join(self, **kwargs):
        self.running = False
        super().join(**kwargs)


def _combine_results(results: List[dict]) -> dict:
    """Combines the results of multiple benchmarkers into a single results structure."""
    combined_results = {}
    for result in results:
        combined_results.setdefault("workers_per_model", result["workers_per_model"])
        combined_results["status"] = (
            result["status"] if combined_results.get("status", "") != ERROR else ERROR
        )
        combined_results["n_infs"] = combined_results.get("n_infs", 0) + result["n_infs"]
        combined_results["total_s"] = max(combined_results.get("total_s", 0), result["total_s"])
        timers = combined_results.get("timers", {})
        for k, v in result["timers"].items():
            timer_list = timers.get(k, [])
            timer_list.extend(v)
            timers[k] = timer_list
        combined_results["timers"] = timers
    return combined_results


def _get_num_workers(pipeline_size: int) -> int:
    """Returns a best-guess number of worker threads for a single benchmarking process."""
    return 2 if pipeline_size == 1 else pipeline_size - 1


def get_instance_type() -> str:
    """Try to obtain the maximum number of NeuronCores available on this instance."""
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-type"
        ) as response:
            instance_type = response.read().decode("utf-8")
        log.debug("Automatically determined instance type: {}".format(instance_type))
        return instance_type
    except:
        return None


def _get_cost_per_hour(instance_type: str) -> float:
    instancetype_to_cost = {
        "inf1.xlarge": 0.228,
        "inf1.2xlarge": 0.362,
        "inf1.6xlarge": 1.18,
        "inf1.24xlarge": 4.721,
    }
    try:
        return instancetype_to_cost[instance_type]
    except:
        return None


def _get_max_neuroncores(instance_type: str = None) -> int:
    """Try to obtain the maximum number of NeuronCores available on this instance."""
    instancetype_to_neuroncores = {
        "inf1.xlarge": 4,
        "inf1.2xlarge": 4,
        "inf1.6xlarge": 16,
        "inf1.24xlarge": 64,
    }
    try:
        if not instance_type:
            instance_type = get_instance_type()
        return instancetype_to_neuroncores[instance_type]
    except:
        num_cores = 2
        log.warning(f"Unknown Neuron device size. Assuming {num_cores} NeuronCores is the maximum.")
        return num_cores


def _get_num_gpus(instance_type: str = None) -> int:
    """Try to obtain the maximum number of GPUs available on this instance."""
    instancetype_to_gpus = {
        "g4dn.xlarge": 1,
        "g4dn.2xlarge": 1,
        "g4dn.4xlarge": 1,
        "g4dn.8xlarge": 1,
        "g4dn.16xlarge": 1,
        "g4dn.12xlarge": 4,
        "g4dn.metal": 8,
        "g4ad.xlarge": 1,
        "g4ad.2xlarge": 1,
        "g4ad.4xlarge": 1,
        "g4ad.8xlarge": 2,
        "g4ad.16xlarge": 4,
        "p4d.24xlarge": 8,
    }
    try:
        if not instance_type:
            instance_type = get_instance_type()
        return instancetype_to_gpus[instance_type]
    except:
        log.warning("Unknown GPU device size. Assuming 1 GPU is available.")
        return 1


def _get_num_devices(device_type: str, instance_type: str = None) -> int:
    """This is a stub, to be populated later for other instance types."""
    if device_type == "neuron":
        return _get_max_neuroncores(instance_type)
    elif device_type == "cpu":
        return multiprocessing.cpu_count()
    elif device_type == "cuda" or device_type == "gpu":
        return _get_num_gpus(instance_type)
    else:
        log.warning("An unknown device_type was passed: {}".format(device_type))
        return None


def _sanitize_inputs(inputs, batch_sizes: Union[int, List[int]], dataset_inputs=False) -> List[int]:
    """Return inputs and batch_sizes with matching lengths, or throw an error."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if not batch_sizes:
        log.warning(
            "Batch sizes were not provided, so assuming 1 and only the first input will be benchmarked."
        )
        batch_sizes = [1]
    if not dataset_inputs:
        if len(batch_sizes) < len(inputs):
            delta = len(inputs) - len(batch_sizes)
            log.warning(
                "Received {} inputs, but only {} batch sizes. Discarding last {} inputs.".format(
                    len(inputs), len(batch_sizes), delta
                )
            )
            inputs = inputs[: len(batch_sizes)]
        elif len(inputs) < len(batch_sizes):
            delta = len(batch_sizes) - len(inputs)
            log.warning(
                "Received {} batch sizes, but only {} inputs. Discarding last {} batch sizes.".format(
                    len(batch_sizes), len(inputs), delta
                )
            )
            batch_sizes = batch_sizes[: len(inputs)]
    return inputs, batch_sizes


def set_verbosity(verbosity: int):
    r"""
    Controls the verbosity of NeuronPerf logging.

    :param int verbosity: 0 = error, 1 = info, 2 = debug
    """
    if 0 == verbosity:
        log.setLevel(logging.ERROR)
    elif 1 == verbosity:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.DEBUG)


def compile(
    compile_fn,
    model,
    inputs,
    batch_sizes: Union[int, List[int]] = None,
    pipeline_sizes: Union[int, List[int]] = None,
    performance_levels: Union[str, List[int]] = None,
    models_dir: str = "models",
    model_name: str = None,
    filename: str = None,
    compiler_args: dict = None,
    verbosity: int = 1,
    **kwargs,
) -> str:
    r"""
    Compiles the provided model with each provided example input, pipeline size, and performance level.

    :param model: The model to compile.
    :param list inputs: A list of example inputs.
    :param Union[int, List[int]] batch_sizes: A list of batch sizes that correspond to the example inputs.
    :param Union[int, List[int]] pipeline_sizes: A list of pipeline sizes to use. See :ref:`neuroncore-pipeline`.
    :param Union[int, List[int]] performance_levels: A list of performance levels to try. Options are: 0 (max accuracy), 1, 2, 3 (max performance, default).  See :ref:`mixed-precision`.
    :param str models_dir: The directory where compilation artifacts will be stored.
    :param str model_name: An optional model name tag to apply to compiled artifacts.
    :param str filename: The name of the model index to write out. If not provided, a name will be generated and returned.
    :param dict compiler_args: Additional compiler arguments to be forwarded with every compilation.
    :param int verbosity: 0 = error, 1 = info, 2 = debug
    :return: A model index filename. If a configuration fails to compile, it will not be included in the index and an error will be logged.
    :rtype: str
    """
    set_verbosity(verbosity)

    if not pipeline_sizes:
        pipeline_sizes = [1]
    if not performance_levels:
        performance_levels = []
    if not compiler_args:
        compiler_args = {}
    if not model_name:
        if isinstance(model, str):
            model_name = model
        else:
            try:
                model_name = model.__name__
            except AttributeError:
                log.warning("Unable to determine a model name, using 'Model'.")
                model_name = "Model"
    if isinstance(pipeline_sizes, int):
        pipeline_sizes = [pipeline_sizes]
    if isinstance(performance_levels, int):
        performance_levels = [performance_levels]

    inputs, batch_sizes = _sanitize_inputs(inputs, batch_sizes)

    if NEURONCORE_PIPELINE_CORES in compiler_args:
        if pipeline_sizes:
            log.warning(
                (
                    "You provided NeuronCore Pipeline Core sizes using both "
                    "compiler_args and pipeline_sizes. Ignoring flag in compiler_args."
                )
            )
        else:
            pipeline_sizes = [compiler_args[NEURONCORE_PIPELINE_CORES]]
        del compiler_args[NEURONCORE_PIPELINE_CORES]

    if FAST_MATH in compiler_args:
        if performance_levels:
            log.warning(
                (
                    f"You provided performance_levels and {FAST_MATH}. "
                    "Ignoring flag in compiler_args."
                )
            )
        del compiler_args[FAST_MATH]

    max_performance = max(FAST_MATH_OPTIONS)
    performance_levels_invalid = list(
        filter(
            lambda level: level < min(FAST_MATH_OPTIONS) or level > max_performance,
            performance_levels,
        )
    )
    if performance_levels_invalid:
        log.warning(
            "You provided some invalid performance_levels. Ignoring: {}".format(
                performance_levels_invalid
            )
        )
        performance_levels = [
            level
            for level in performance_levels
            if (level in performance_levels) and (level not in performance_levels_invalid)
        ]

    if not performance_levels:
        performance_levels.append(max_performance)

    os.makedirs(models_dir, exist_ok=True)

    model_idxs = []

    def make_index():
        """Create a model index file that contains info about all compiled models."""
        index = model_index.append(*model_idxs)
        return model_index.save(index, filename=filename)

    compile_idx = 1
    n_compiles = len(inputs) * len(pipeline_sizes) * len(performance_levels)
    for input_idx, example_input in enumerate(inputs):
        batch_size = batch_sizes[input_idx]
        for pipeline_size in pipeline_sizes:
            for performance_level in performance_levels:
                _compiler_args = copy.copy(compiler_args)
                _compiler_args[FAST_MATH] = FAST_MATH_OPTIONS[performance_level]
                if pipeline_size != 1:
                    _compiler_args[NEURONCORE_PIPELINE_CORES] = str(pipeline_size)

                model_name_ex = "{}_b{}_p{}_{}".format(
                    model_name,
                    batch_size,
                    pipeline_size,
                    model_index.generate_id(),
                )
                log.info(
                    (
                        f"Compiling batch size {batch_size} for {pipeline_size} NeuronCore(s) with performance level "
                        f"{performance_level}/{max_performance}. [{compile_idx}/{n_compiles}]"
                    )
                )
                status = "ready"
                timer = Timer()
                with timer:
                    try:
                        model_filename = compile_fn(
                            model,
                            example_input,
                            models_dir,
                            model_name_ex,
                            compiler_args=_compiler_args,
                            **kwargs,
                        )
                        status = "finished"
                    except KeyboardInterrupt:
                        status = "error"
                        model_filename = None
                        log.error("Compilation interrupted, terminating.")
                        return make_index()
                    except:
                        status = "error"
                        model_filename = None
                        log.exception(
                            (
                                f"Failed to compile input={input_idx}, "
                                f"batch_size={batch_size}, "
                                f"pipeline_size={pipeline_size}, "
                                f"performance_level={performance_level}."
                            )
                        )
                    finally:
                        model_idx = model_index.create(
                            model_filename,
                            model_name=model_name,
                            batch_size=batch_size,
                            pipeline_size=pipeline_size,
                            performance_level=performance_level,
                            compile_s=round(timer.total_duration("s"), 2),
                            status=status,
                        )
                        model_idxs.append(model_idx)
                        filename = make_index()
                compile_idx += 1
    return filename


def run_benchmarker(benchmarker, duration, pipe=None):
    def _send(results):
        if pipe:
            pipe.send(results)
            pipe.close()
        else:
            return results

    try:
        log.debug(f"Benchmarker {benchmarker.id} started.")
        check_freq = 0.1
        start_time = time.time()
        benchmarker.start()
        elapsed = 0
        while (elapsed < duration) and benchmarker.benchmarking:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            time.sleep(min(check_freq, remaining))
        benchmarker.stop()
    except:
        trace = "".join(traceback.format_exception(*sys.exc_info()))
        error = BenchmarkerErrorWrapper(trace)
        return _send(error)
    else:
        results = benchmarker.results() if benchmarker.status != ERROR else benchmarker.error
        return _send(results)
    finally:
        log.debug(f"Benchmarker {benchmarker.id} finished.")


def _run_benchmarker_new_interpreter(benchmarker, duration):
    """
    This function is a workaround for frameworks that cannot be safely forked.
    The premise is to launch a new Python interpreter and run benchmarking
    from within the new interpreter. It works by writing serialized benchmarkers
    to temporary files, and then launching run_benchmark_file.py. The script
    writes back serialized results.
    """

    setattr(benchmarker, "_stderr", None)

    script = run_benchmark_file.__file__

    f = tempfile.NamedTemporaryFile(delete=False)
    log.debug("Dumping Benchmarker {} to file '{}'.".format(benchmarker.id, f.name))
    try:
        dill.dump(benchmarker, f)
    except dill.PicklingError:
        raise dill.PicklingError(
            (
                "NeuronPerf was unable to serialize the benchmarker. This is probably because your model "
                "could not be serialized. Make sure to use top-level classes instead of locals. You may "
                "need to wrap your model and manually load it using Python's importlib."
            )
        )
    f.close()

    command = [
        sys.executable,
        script,
        f.name,
        str(duration),
    ]

    if benchmarker.model_class_name and benchmarker.model_class_file:
        command.append(f"--model_class_name={benchmarker.model_class_name}")
        command.append(f"--model_class_file={benchmarker.model_class_file}")

    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )

    timeout = 60 + duration

    try:
        outs, errs = proc.communicate(timeout=timeout)
        with open(f.name, "rb") as fp:
            result = dill.load(fp)
        if isinstance(result, BenchmarkerErrorWrapper):
            raise ChildProcessError(
                "Benchmarker {} encountered an error:\n{}".format(benchmarker.id, result.trace)
            )
        if isinstance(result, Benchmarker):
            from pathlib import Path

            path = Path(f.name)
            logs = os.path.join(path.parent, "neuronperf_error_{}".format(str(path.stem)))
            if os.path.exists(logs):
                with open(logs, "rt") as logs_fp:
                    err_logs = logs_fp.readlines()
                os.unlink(logs)
                raise ChildProcessError(
                    "Benchmarker {} failed. Logs from child process:\n{}".format(
                        benchmarker.id, "".join(err_logs)
                    )
                )
            else:
                raise ChildProcessError(
                    (
                        "Benchmarker {} failed and no error logs were found. A child process may have "
                        "aborted. To obtain a stack trace, try running a single configuration inside a "
                        "single process by passing multiprocess=False, multiinterpreter=False"
                    )
                )

        return result
    except subprocess.TimeoutExpired:
        proc.kill()
        raise ChildProcessError(
            "Benchmarker {} stopped responding after {} seconds.".format(benchmarker.id, timeout)
        )
    finally:
        os.unlink(f.name)


def _run_benchmarkers_multiprocess(
    benchmarkers: List[Benchmarker], duration: int, benchmark_func=run_benchmarker
) -> dict:
    results = []
    pipes, procs = [], []
    for benchmarker in benchmarkers:
        parent_pipe, child_pipe = multiprocessing.Pipe()
        pipes.append(parent_pipe)
        proc = multiprocessing.Process(
            target=benchmark_func, args=(benchmarker, duration, child_pipe)
        )
        procs.append(proc)
    for proc in procs:
        proc.start()
    for id, (pipe, proc) in enumerate(zip(pipes, procs)):
        try:
            proc_result = pipe.recv()
            if isinstance(proc_result, BenchmarkerErrorWrapper):
                log.error("Child process encountered an error:\n{}".format(proc_result.trace))
                raise ChildProcessError()
            proc.join()
            results.append(proc_result)
        except KeyboardInterrupt:
            log.error("Benchmarking interrupted, terminating.")
            for proc in procs:
                proc.terminate()
            raise KeyboardInterrupt()
        except EOFError:
            log.error(
                (
                    f"Child process {id} was killed by the host OS during benchmarking.\n"
                    "You may have run out of memory.\n"
                    "Verify that your model can perform inference without NeuronPerf or try n_models=1."
                )
            )
    return _combine_results(results)


def _run_benchmarkers_multithreaded(
    benchmarkers: List[Benchmarker], duration: int, benchmark_func=run_benchmarker
) -> dict:
    results = []
    timeout = 60 + duration
    try:
        args = ((benchmarker, duration) for benchmarker in benchmarkers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(benchmarkers)) as exe:
            results.extend(exe.map(lambda arg: benchmark_func(*arg), args, timeout=timeout))
        for result in results:
            if isinstance(result, BenchmarkerErrorWrapper):
                raise RuntimeError("Worker thread encountered an error:\n{}".format(result.trace))
    except concurrent.futures.TimeoutError:
        log.error("Benchmarking timed out after {} seconds.".format(timeout))
    except KeyboardInterrupt:
        raise KeyboardInterrupt("Benchmarking interrupted, terminating.")
    return _combine_results(results)


def run_benchmarkers(
    benchmarkers: List[Benchmarker],
    duration: int,
    stats_interval: float = 0.5,
    multiprocess: bool = True,
    multiinterpreter: bool = False,
) -> dict:
    results = {}

    stats_thread = StatsThread(stats_interval)
    stats_thread.start()

    try:
        if multiinterpreter:
            if not sys.executable:
                raise ValueError(
                    (
                        "Unable to benchmark in multi-interpreter mode because "
                        "the Python interpreter cannot be located (sys.executable is empty)."
                    )
                )
            results = _run_benchmarkers_multithreaded(
                benchmarkers, duration, benchmark_func=_run_benchmarker_new_interpreter
            )
        elif multiprocess:
            results = _run_benchmarkers_multiprocess(benchmarkers, duration)
        else:
            results = _run_benchmarkers_multithreaded(benchmarkers, duration)
    finally:
        stats_thread.join()
        results["cpu_percents"] = stats_thread.cpu_percents
        results["mem_percents"] = stats_thread.mem_percents

    return results


def _get_env_setup_fn(benchmarker_id: int, benchmarker_config: dict, env_setup_fn):
    """Wrap an environment setup function with device-specific requirements."""
    device_type = str(benchmarker_config["device_type"]).lower().strip()
    legacy = bool(os.environ.get("NEURONCORE_GROUP_SIZES"))
    if "neuron" == device_type:

        @functools.wraps(env_setup_fn)
        def _env_setup_fn():
            import os

            id = benchmarker_id
            config = benchmarker_config
            pipeline_size = config["pipeline_size"]
            if config["multiprocess"] or config["multiinterpreter"]:
                min_core = pipeline_size * id
                max_core = min_core + (pipeline_size - 1)
                visible_cores = f"{min_core}-{max_core}"

                if legacy:
                    os.environ["NEURONCORE_GROUP_SIZES"] = str(pipeline_size)
                else:
                    os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores
            else:
                n_models = config["n_models"]
                if legacy:
                    os.environ["NEURONCORE_GROUP_SIZES"] = ",".join([str(pipeline_size)] * n_models)
                else:
                    os.environ["NEURON_RT_VISIBLE_CORES"] = "0-{}".format(
                        n_models * pipeline_size - 1
                    )

            if env_setup_fn:
                env_setup_fn(id, config)

        return _env_setup_fn
    elif device_type == "cpu":
        return env_setup_fn
    elif device_type == "cuda" or device_type == "gpu":

        @functools.wraps(env_setup_fn)
        def _env_setup_fn():
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = str(benchmarker_id)

            if env_setup_fn:
                env_setup_fn(benchmarker_id, benchmarker_config)

        return _env_setup_fn
    else:
        log.warning(
            (
                f"NeuronPerf does not implement a proper environment setup for {device_type}. "
                "You may need to provide your own."
            )
        )
        return env_setup_fn


def _get_setup_fn(benchmarker_id: int, benchmarker_config: dict, setup_fn):
    """Wraps a customer provided setup function with additional info from the benchmarker."""
    if not setup_fn:
        return None

    @functools.wraps(setup_fn)
    def _setup_fn(model):
        setup_fn(benchmarker_id, benchmarker_config, model)

    return _setup_fn


def _get_device_id(benchmarker_id: int, benchmarker_config: dict):
    """Calculate an appropriate device id for a benchmarker object."""
    device_id = benchmarker_id
    device_type = str(benchmarker_config["device_type"]).lower().strip()
    if device_type in SUPPORTED_DEVICE_TYPES:
        if not (benchmarker_config["multiprocess"] or benchmarker_config["multiinterpreter"]):
            device_id = benchmarker_id * benchmarker_config["pipeline_size"]
        return device_id
    else:
        log.warning(
            "Assuming device_id={} for benchmarker_id={} for unknown device_type={}".format(
                device_id, benchmarker_id, device_type
            )
        )
    return device_id
```

## feature-guide.rst

```python
# Configure, initialize, and compile a model.
model = NeuronLlamaForCausalLM(model_path, config)
model.compile(compiled_model_path)
```

```python
neuron_config = NeuronConfig(logical_nc_config=2)
```

```python
neuron_config = NeuronConfig(sequence_parallel_enabled=True)
```

```python
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Init Neuron model, HuggingFace tokenizer, HuggingFace and generation config.

# Run generation with HuggingFaceGenerationAdapter.
generation_model = HuggingFaceGenerationAdapter(model)
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
outputs = generation_model.generate(
    inputs.input_ids,
    generation_config=generation_config,
    attention_mask=inputs.attention_mask,
    max_length=model.neuron_config.max_length,
    **kwargs,
)

output_tokens = tokenizer.batch_decode(
    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("Generated outputs:")
for i, output_token in enumerate(output_tokens):
    print(f"Output {i}: {output_token}")
```

```python
on_device_sampling_config = OnDeviceSamplingConfig(global_topk=256)
neuron_config = NeuronConfig(on_device_sampling_config=on_device_sampling_config)
```

```python
on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True)
neuron_config = NeuronConfig(on_device_sampling_config=on_device_sampling_config)
```

```python
sampling_params = torch.tensor([[50, 0.5, 0.75], [5, 1.0, 1.0]])
```

```python
on_device_sampling_config = OnDeviceSamplingConfig(top_k=5)
```

```python
neuron_config = NeuronConfig(fused_qkv=True)
```

```python
neuron_config = NeuronConfig(enable_bucketing=True)
```

```python
neuron_config = NeuronConfig(
    enable_bucketing=True,
    context_encoding_buckets=[1024, 2048, 4096],
    token_generation_buckets=[8192]
)
```

```python
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = "/home/ubuntu/models/Llama-3.1-8B"
quantized_model_path = "/home/ubuntu/models/Llama-3.1-8B-quantized"

neuron_config = NeuronConfig(
    quantized=True,
    quantized_checkpoints_path=quantized_model_path,
    quantization_dtype="int8",
    quantization_type="per_tensor_symmetric"
)

config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path)
)

# Quantize the model and save it to `quantized_checkpoints_path`.
NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)

# Compile, load, and use the model.
model = NeuronLlamaForCausalLM(model_path, config)
```

```python
neuron_config = NeuronConfig(kv_cache_quant=True)
```

```python
import copy

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM
)
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

prompts = ["I believe the meaning of life is"]

model_path = "/home/ubuntu/models/Llama-3.1-70B"
draft_model_path = "/home/ubuntu/models/Llama-3.2-3B"
compiled_model_path = "/home/ubuntu/neuron_models/Llama-3.1-70B"
compiled_draft_model_path = "/home/ubuntu/neuron_models/Llama-3.2-3B"

# Initialize target model.
neuron_config = NeuronConfig(
    speculation_length=5,
    trace_tokengen_model=False
)
config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path)
)
model = NeuronLlamaForCausalLM(model_path, config)

# Initialize draft model.
draft_neuron_config = copy.deepcopy(neuron_config)
draft_neuron_config.speculation_length = 0
draft_neuron_config.trace_tokengen_model = True
draft_config = LlamaInferenceConfig(
    draft_neuron_config,
    load_config=load_pretrained_config(draft_model_path)
)
draft_model = NeuronLlamaForCausalLM(draft_model_path, draft_config)

# Compile and save models.
model.compile(compiled_model_path)
draft_model.compile(compiled_draft_model_path)

# Load models to the Neuron device.
model.load(compiled_model_path)
draft_model.load(compiled_draft_model_path)

# Load tokenizer and generation config.
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
generation_config = GenerationConfig.from_pretrained(model_path)

# Run generation.
_, output_tokens = get_generate_outputs(
    model,
    prompts,
    tokenizer,
    is_hf=False,
    draft_model=draft_model,
    generation_config=generation_config
)

print("Generated outputs:")
for i, output_token in enumerate(output_tokens):
    print(f"Output {i}: {output_token}")
```

```python
def load_json_file(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

medusa_tree = load_json_file("medusa_mc_sim_7b_63.json")

neuron_config = NeuronConfig(
    is_medusa=True,
    medusa_speculation_length=64,
    num_medusa_heads=4,
    medusa_tree=medusa_tree
)
```

```python
import copy

from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig,
    NeuronConfig,
    OnDeviceSamplingConfig
)
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
    NeuronLlamaModel
)
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from transformers import AutoTokenizer, GenerationConfig

prompt = "The future of AI is"

model_path = "/home/ubuntu/models/Llama-3.1-70B-Instruct"
draft_model_path = "/home/ubuntu/models/Llama-3.1-70B-Instruct-EAGLE-Draft"
compiled_model_path = "/home/ubuntu/neuron_models/Llama-3.1-70B-Instruct-EAGLE"
max_sequence_length = 1024

# Initialize on-device sampling configuration.
on_device_sampling_config = OnDeviceSamplingConfig(
    temperature=0.7,
    top_k=50,
    top_p=1.0,
)

# Initialize model configuration.
neuron_config = NeuronConfig(
    batch_size = 1,
    enable_eagle_speculation=True,
    enable_fused_speculation=True,
    max_context_length=max_sequence_length,
    max_length=max_sequence_length,
    on_device_sampling_config=on_device_sampling_config,
    seq_len=max_sequence_length,
    speculation_length=5,
    tp_degree=32,
    trace_tokengen_model=False
)

config = NeuronLlamaForCausalLM.get_config_cls()(
    neuron_config, load_config=load_pretrained_config(model_path)
)

# Initialize draft model configuration and set EAGLE-specific values.
draft_neuron_config = copy.deepcopy(neuron_config)
draft_neuron_config.trace_tokengen_model = True
draft_neuron_config.enable_fused_speculation = False
draft_neuron_config.is_eagle_draft = True
draft_neuron_config.sequence_parallel_enabled = False

draft_config = NeuronLlamaForCausalLM.get_config_cls()(
    draft_neuron_config, load_config=load_pretrained_config(draft_model_path))

# Initialize fused speculation configuration.
fused_spec_config = FusedSpecNeuronConfig(
    NeuronLlamaForCausalLM._model_cls,
    draft_config=draft_config,
    draft_model_path=draft_model_path,
)
config.fused_spec_config = fused_spec_config

# Initialize model from configuration.
model = NeuronLlamaForCausalLM(model_path, config)

# Compile and save model.
model.compile(compiled_model_path)

# Load model to the Neuron device.
model.load(compiled_model_path)

# Load tokenizer and generation config.
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.max_length = 1024
generation_config.pad_token_id = tokenizer.eos_token_id

# Run generation and print outputs.
_, output_tokens = get_generate_outputs(
    model,
    [prompt],
    tokenizer,
    is_hf=False,
    draft_model=None,
    generation_config=generation_config
)

print("Generated output:")
for _, output in enumerate(output_tokens):
    print(output)
```

```python
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

target_model_path = "Meta-Llama-3.1-70B-Instruct"
draft_model_path = "Llama-3.1-70B-Instruct-EAGLE-Draft"

DRAFT_MODEL_SAFETENSORS_NAME = "model.safetensors"
LM_HEAD_WEIGHT_TENSOR_NAME = "lm_head.weight"
TARGET_MODEL_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"

def find_lm_head_safetensors_location(model_dir):
    model_index_location_path = os.path.join(model_dir, TARGET_MODEL_SAFETENSORS_INDEX_NAME)

    with open(model_index_location_path, 'r') as f:
        model_index_locations = json.load(f)

    lm_head_safetensors_name = model_index_locations["weight_map"][LM_HEAD_WEIGHT_TENSOR_NAME]

    return lm_head_safetensors_name

# Find the target model `lm_head.weight` location in safetensors
target_lm_head_safetensors_name = find_lm_head_safetensors_location(target_model_path)
target_lm_head_safetensors_path = os.path.join(target_model_path, target_lm_head_safetensors_name)

# Open the target model.safetensor containing `lm_head.weight`
with safe_open(target_lm_head_safetensors_path, framework="pt") as f:
    target_lm_head = f.get_tensor(LM_HEAD_WEIGHT_TENSOR_NAME)

# Collect all tensors in the draft model
draft_model_safetensors_path = os.path.join(draft_model_path, DRAFT_MODEL_SAFETENSORS_NAME)
tensors = {}
with safe_open(draft_model_safetensors_path, framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Add the LM head weights and save out the new draft model.safetensors file
tensors[LM_HEAD_WEIGHT_TENSOR_NAME] = target_lm_head.type(torch.float16)
save_file(tensors, draft_model_safetensors_path)
```

```python
neuron_config = NeuronConfig(
    is_prefix_caching=True,
    is_block_kv_layout=True,
    pa_num_blocks=1024,
    pa_block_size=32,
)
```

```python
neuron_config = NeuronConfig(
    enable_bucketing=True,
    context_encoding_buckets=[512, 1024, 2048],
    prefix_buckets=[512, 1024],
    token_generation_buckets=[2048]
)
```

## neuronsetuphelper.py

```python
import json
import argparse
from packaging.version import Version, parse


class neuron_release_info:
    def __init__(self):
        self.release_frameworks_all = {}
        self.release_frameworks_main = {}
        self.release_packages_all ={}
        self.release_package_main={}
        self.release_frameworks_list=[]
        self.release_components_list = []
        self.release_tf_package_to_model_server_package={}
        self.release_os_install_list =[]
        self.python_ver=""


def cli_parse_arguments():
    __name__='neuron-install-helper.py'
    parser = argparse.ArgumentParser(prog=__name__
    ,usage='\npython3 %(prog)s --list {neuron_versions,packages,components,frameworks} [--neuron-version=X.Y.Z]  [--file FILE] \n'
    +'python3 %(prog)s --install {pytorch,tensorflow,mxnet} [--neuron-version=X.Y.Z] [--framework-version=FRAMEWORK-X.Y.Z] [options]\n'
    +'python3 %(prog)s --install {driver,runtime,tools} [--neuron-version=X.Y.Z] [options]\n'
    +'python3 %(prog)s --update {pytorch,tensorflow,mxnet} [--framework-version=framework-X.Y.Z]  [options]\n'
    +'python3 %(prog)s --update {driver,runtime,tools} [options]\n'
    +'options= [--file FILE] [--ami {dlami,non-dlami}] [--os {ubuntu,amazonlinux}]\n'
    ,description='Installer helper for Neuron SDK')

    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--neuron-version",metavar='X.Y.Z')
    group.add_argument("--list",choices=['neuron_versions','packages','components','frameworks'])
    group.add_argument("--install",choices=['pytorch','tensorflow','mxnet'])
    group.add_argument("--update",choices=['pytorch','tensorflow','mxnet'])
    parser.add_argument("--mode",choices=['develop','compile','deploy'],default='develop')
    parser.add_argument("--framework-version",metavar='framework-X.Y.Z')
    parser.add_argument("--os",choices=['ubuntu','amazonlinux'],default='ubuntu',help='default=ubuntu')
    parser.add_argument("--ami",choices=['dlami','non-dlami'],default='non-dlami',help='default=non-dlami')
    parser.add_argument("--file",default='neuron-releases-manifest.json',help='default=neuron-releases-manifest.json')

    return parser.parse_args()


def versiontuple(v):
   filled = []
   for point in v.split("."):
      filled.append(point.zfill(8))
   return tuple(filled)


def cli_validate(update,neuron_version,framework_version,is_latest_neuron,ami):
    if (update!=None) & (is_latest_neuron == False):
        print (__name__,": error: ","--update always update to latest Neuron versions, can't specify Neuron version")
        exit(-1)

    if (framework_version != None):
        if (framework_version not in  nr_setup.releases_info[neuron_version].release_frameworks_list):
            print (__name__,": error: "," " + framework_version + " is not a supported framework")
            exit(-1)
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
import torch
import numpy as np
import neuron.nki as nki
import neuron.nki.language as nl
import neuron.nki.isa as nisa
import neuron.nki.isa.math as ml
```

```python
@nki.jit
def mamba_v1(delta, u, A, B, C):
    """
    Initial NKI kernel implementation for Mamba layer.
    
    Args:
        delta: [batch_size, channels, seq_len]
        u: [batch_size, channels, seq_len]
        A: [channels, state_size]
        B: [batch_size, state_size, seq_len]
        C: [batch_size, state_size, seq_len]
    
    Returns:
        output: [batch_size, channels, seq_len]
    """
    batch_size = delta.shape[0]
    channels = delta.shape[1]
    seq_len = delta.shape[2]
    state_size = A.shape[1]
    
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
        
        for i_state in nl.affine_range(state_size):
            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize
                
                # Load inputs
                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                # Step 1&2: deltaA = exp(delta * A)
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                
                # Step 3: deltaBu = delta * B * u
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=ml.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=np.multiply, op1=np.add)
                
                # Step 5: scanC = C * scan
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
                
                # Step 6: Accumulate across states
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC_i
        
        # Store results
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])
    
    return output
```

```python
@nki.jit
def mamba_v2(delta, u, A, B, C):
    """
    Optimized NKI kernel with loop reordering to minimize data reloading.
    
    Args:
        delta: [batch_size, channels, seq_len]
        u: [batch_size, channels, seq_len]
        A: [channels, state_size]
        B: [batch_size, state_size, seq_len]
        C: [batch_size, state_size, seq_len]
    
    Returns:
        output: [batch_size, channels, seq_len]
    """
    batch_size = delta.shape[0]
    channels = delta.shape[1]
    seq_len = delta.shape[2]
    state_size = A.shape[1]
    
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            
            # Load delta and u once, reuse across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            for i_state in nl.affine_range(state_size):
                # Load state-specific inputs
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                # Step 1&2: deltaA = exp(delta * A)
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                
                # Step 3: deltaBu = delta * B * u
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=ml.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=ml.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=np.multiply, op1=np.add)
                
                # Step 5: scanC = C * scan
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=ml.multiply)
                
                # Step 6: Accumulate across states
                scanC_accum[0:channel_psize, 0:seq_len] += scanC_i
            
            # Store results
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    
    return output
```

```python
@nki.jit
def mamba_v3(delta, u, A, B, C):
    """
    Further optimized NKI kernel with seq_len tiling to mitigate spilling.
    
    Args:
        delta: [batch_size, channels, seq_len]
        u: [batch_size, channels, seq_len]
        A: [channels, state_size]
        B: [batch_size, state_size, seq_len]
        C: [batch_size, state_size, seq_len]
    
    Returns:
        output: [batch_size, channels, seq_len]
    """
    batch_size = delta.shape[0]
    channels = delta.shape[1]
    seq_len = delta.shape[2]
    state_size = A.shape[1]
    
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize
    seq_len_fsize = 512
    n_seq_len_tile = (seq_len + seq_len_fsize - 1) // seq_len_fsize
    
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype, buffer=nl.hbm)
    
    for i_batch in nl.affine_range(batch_size):
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            
            # Load delta and u once, reuse across states and seq_len tiles
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            
            for i_state in nl.affine_range(state_size):
                # Load state-specific inputs
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                scan_init = nl.zeros((nl.par_dim(channel_psize), 1), dtype=delta.dtype, buffer=nl.sbuf)
                
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_start = i_seq_len_tile * seq_len_fsize
                    seq_end = min(seq_start + seq_len_fsize, seq_len)
                    seq_tile_len = seq_end - seq_start
                    
                    # Extract tile slices
                    delta_tile = delta_i[0:channel_psize, seq_start:seq_end]
                    u_tile = u_i[0:channel_psize, seq_start:seq_end]
                    B_tile = B_i[0:1, seq_start:seq_end]
                    C_tile = C_i[0:1, seq_start:seq_end]
                    
                    # Step 1&2: deltaA = exp(delta * A)
                    deltaA = nisa.activation(op=nl.exp, data=delta_tile, scale=A_i)
                    
                    # Step 3: deltaBu = delta * B * u
                    deltaU = nisa.tensor_tensor(delta_tile, u_tile, op=ml.multiply)
                    B_tile_bcast = B_tile.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    deltaBu = nisa.tensor_tensor(deltaU, B_tile_bcast, op=ml.multiply)
                    
                    # Step 4: Associative scan with loop-carried dependency
                    scan_tile = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                                                        op0=np.multiply, op1=np.add)
                    
                    # Update scan_init for next iteration
                    scan_init = scan_tile[0:channel_psize, seq_tile_len-1:seq_tile_len]
                    
                    # Step 5: scanC = C * scan
                    C_tile_bcast = C_tile.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    scanC_tile = nisa.tensor_tensor(scan_tile, C_tile_bcast, op=ml.multiply)
                    
                    # Store partial results
                    nl.store(output[i_batch, channel_start:channel_start+channel_psize, seq_start:seq_end],
                            scanC_tile)
    
    return output
```

## bert.rst

```python
import torch
import torch.autocast

with torch.autocast(enabled=flags.enable_pt_autocast, dtype=torch.bfloat16, device_type='xla'):
    outputs = model(input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    labels=masked_lm_labels,
                    next_sentence_label=next_sentence_labels)
    loss = outputs.loss / flags.grad_accum_usteps
loss.backward()
running_loss += loss.detach()
```

```python
import torch_xla.core.xla_model as xm

cpu_data = xm._maybe_convert_to_cpu(data)
```

```python
import torch_xla.core.xla_model as xm

def _mp_fn(index, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_bert_hdf5(flags)
    xm.rendezvous("_mp_fn finished")
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
            """Updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""

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
                """projection"""
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

## nki.errors.rst

```python
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

# err_1d_arange_not_supported - Error case
tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
i = nl.arange(64)
c = nl.exp(tmp[i, 0])  # Error: indexing tensor `tmp` with 1d arange is not supported

# err_1d_arange_not_supported - Workaround 1
tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
i = nl.arange(64)[:, None]
c = nl.exp(tmp[i, 0])

# err_1d_arange_not_supported - Workaround 2
tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
c = nl.exp(tmp[0:64, 0])

# err_activation_bias_invalid_type
data = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.activation(op=nl.exp, data=data[...], bias=nisa.memset((128, 1), 1.2, dtype=np.float32))  # ok
nisa.activation(op=nl.exp, data=data[...], bias=nisa.memset((128, 1), 1.2, dtype=nl.bfloat16))  # ok
nisa.activation(op=nl.exp, data=data[...], bias=nisa.memset((128, 1), 1.2, dtype=np.int8))  # not supported

# err_activation_scale_invalid_type
nisa.activation(op=nl.exp, data=data[...], scale=1.2)  # ok
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 1), 1.2, dtype=np.float32))  # ok
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 1), 1.2, dtype=np.float16))  # not supported

# err_activation_scale_scalar_or_vector
nisa.activation(op=nl.exp, data=data[...], scale=1.2)  # ok
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 1), 1.2, dtype=np.float32))  # ok
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((1, 128), 1.2, dtype=np.float32))  # not supported
nisa.activation(op=nl.exp, data=data[...], scale=nisa.memset((128, 128), 1.2, dtype=np.float32))  # not supported

# err_ambiguous_tensor_truth_value - Error case
from typing import Optional
from neuronxcc.nki.typing import tensor

def func_error(a, b: Optional[tensor]):
    ix, iy = nl.mgrid[0:128, 0:128]
    a_tile: tensor[128, 128] = nl.load(a[ix, iy])
    not_a_tile = not (a_tile > 0)  # Error
    if b:  # Error
        pass

# err_ambiguous_tensor_truth_value - Correct usage
def func_correct(a, b: Optional[tensor]):
    ix, iy = nl.mgrid[0:128, 0:128]
    a_tile: tensor[128, 128] = nl.load(a[ix, iy])
    not_a_tile = ~(a_tile > 0)  # Element-wise negation
    if b is not None:  # Explicit None check
        pass

# err_annotation_shape_mismatch
import neuronxcc.nki.typing as nt
data: nt.tensor[128, 512] = nl.zeros((nl.par_dim(128), 128), dtype=np.float32)  # Error: shape mismatch

# err_cannot_assign_to_index
_, x = nl.mgrid[0:1, 0:8]
x[0, 5] = 1024  # Error: 'index' tensor does not support item assignment
y = nisa.iota(x, dtype=nl.uint32)
y[0, 5] = 1024  # works

# err_cannot_update_immutable_parameter - Error case
def kernel_error(in_tensor):
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(in_tensor, value=y)  # Error: Cannot update immutable parameter
    return in_tensor

# err_cannot_update_immutable_parameter - Correct usage with mutable annotation
def kernel_mutable(in_tensor: nt.mutable_tensor):
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(in_tensor, value=y)  # ok
    return in_tensor

# err_cannot_update_immutable_parameter - Correct usage with copy
def kernel_copy(in_tensor):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_tensor, src=in_tensor)
    x = nl.load(out_tensor)
    y = x + 1
    nl.store(out_tensor, value=y)  # ok
    return out_tensor

# err_control_flow_condition_depending_on_arange - Error case
x = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.sbuf)
for j0 in nl.affine_range(4096):
    i1 = nl.arange(512)[None, :]
    j = j0 * 512 + i1
    if j > 2048:  # Error
        y = nl.add(x[0, j], x[0, j - 2048])

# err_control_flow_condition_depending_on_arange - Workaround with mask
for j0 in nl.affine_range(4096):
    i1 = nl.arange(512)[None, :]
    j = j0 * 512 + i1
    y = nl.add(x[0, j], x[0, j - 2048], mask=j > 2048)

# err_copy_dynamic_indirect_indices_not_natively_supported - Error case
data_tensor = nl.zeros((128, 8, 4), dtype=nl.float32, buffer=nl.sbuf)
idx_tensor = nl.zeros((8, 1), dtype=nl.uint32, buffer=nl.sbuf)
out_sbuf = nl.ndarray([8, 4], dtype=data_tensor.dtype, buffer=nl.sbuf)
iy, iz = nl.mgrid[0:8, 0:4]
idx_tile = nl.load(idx_tensor)
out_sbuf[iy, iz] = data_tensor[0, idx_tile, iz]  # Error

# err_copy_dynamic_indirect_indices_not_natively_supported - Workaround
out_sbuf[iy, iz] = nisa.tensor_copy_dynamic_src(data_tensor[0, idx_tile, iz])

# err_exceed_max_supported_dimension
x = nl.zeros(shape=[64, 32, 2], dtype=np.float32, buffer=nl.sbuf)
b = nl.transpose(x)  # Error: exceed max supported number of dimensions

x = nl.zeros(shape=[64, 64], dtype=np.float32, buffer=nl.sbuf)
b = nl.transpose(x)  # Works

# err_failed_to_infer_tile_from_local_tensor - Error case
a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
c = nl.add(a, 32)  # Error

# err_failed_to_infer_tile_from_local_tensor - Workaround 1
c = nl.ndarray((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
for i in range(4):
    c[i] = nl.add(a[i], 32)  # works

# err_failed_to_infer_tile_from_local_tensor - Workaround 2
for i in range(4):
    ix = nl.arange(8)[:, None]
    iy = nl.arange(8)[None, :]
    c[i, ix, iy] = nl.add(a[i, ix, iy], 32)  # also works

# err_hbm_tensor_with_init_value_not_supported - Error case
t = nl.full((3, 128, 512), fill_value=1.0, buffer=nl.shared_hbm)  # Error

# err_hbm_tensor_with_init_value_not_supported - Workaround
t = nl.ndarray((3, 128, 512), buffer=nl.shared_hbm)
for i in range(3):
    nl.store(dst=t[i, :, :], value=1.0)

# err_indirect_indices_free_dim - Error case
i_p, i_f = nl.mgrid[0:64, 0:512]  # this won't work for dynamic access

# err_indirect_indices_free_dim - Correct usage
i_p = nl.arange(64)[:, None]  # this works for dynamic access
i_f = nl.arange(512)[None, :]
data_tensor = nl.zeros((128, 64, 512), dtype=nl.float32, buffer=nl.hbm)
idx_tile = nl.zeros((64, 1), dtype=nl.uint32, buffer=nl.sbuf)
data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f])

# err_local_variable_used_out_of_scope - Error case
a = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.sbuf)
b = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.sbuf)
c = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.sbuf)
for i in range(4):
    if i < 2:
        tmp = nl.load(a)
    else:
        tmp = nl.load(b)
    nl.store(c, tmp)  # Error

# err_local_variable_used_out_of_scope - Correct usage
for i in range(4):
    tmp = nl.ndarray(shape=a.shape, dtype=a.dtype, buffer=nl.sbuf)
    if i < 2:
        tmp[...] = nl.load(a)
    else:
        tmp[...] = nl.load(b)
    nl.store(c, tmp)

# err_local_variable_used_out_of_scope - Shadowing example error
data = nl.zeros((nl.par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
for i in nl.sequential_range(4):
    i_tile = nisa.iota(i, dtype=nl.uint32).broadcast_to(data.shape)
    data = data + i_tile  # Warning: shadowing
ptr = nl.zeros((nl.par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
nl.store(ptr, value=data)  # Error

# err_local_variable_used_out_of_scope - Shadowing fix
data = nl.zeros((nl.par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
for i in nl.sequential_range(4):
    i_tile = nisa.iota(i, dtype=nl.uint32).broadcast_to(data.shape)
    data[...] = data + i_tile
nl.store(ptr, value=data)

# err_mutable_parameter_not_returned - Error case
def kernel_error_mutable(in_tensor: nt.mutable_tensor):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(out_tensor, value=y)
    nl.store(in_tensor, value=y)
    return out_tensor  # Error: mutable parameter not returned

# err_mutable_parameter_not_returned - Correct usage
def kernel_correct_mutable(in_tensor: nt.mutable_tensor):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    x = nl.load(in_tensor)
    y = x + 1
    nl.store(out_tensor, value=y)
    nl.store(in_tensor, value=y)
    return out_tensor, in_tensor  # ok

# err_num_partition_exceed_arch_limit
x = nl.zeros(shape=[256, 1024], dtype=np.float32, buffer=nl.sbuf)  # Error
x = nl.zeros(shape=[128, 1024], dtype=np.float32, buffer=nl.sbuf)  # Works

# err_num_partition_mismatch
x = nl.zeros(shape=[128, 512], dtype=np.float32, buffer=nl.sbuf)
y0 = nl.zeros(shape=[1, 512], dtype=np.float32, buffer=nl.sbuf)
z = nisa.tensor_tensor(x, y0, op=nl.add)  # Error

y1 = y0.broadcast_to([128, 512])
z = nisa.tensor_tensor(x, y1, op=nl.add)  # works

# err_size_of_dimension_exceed_arch_limit
x = nl.zeros(shape=[128, 512], dtype=np.float32, buffer=nl.sbuf)
b = nl.transpose(x)  # Error

x = nl.zeros(shape=[128, 128], dtype=np.float32, buffer=nl.sbuf)
b = nl.transpose(x)  # Works

# err_store_dst_shape_smaller_than_other_shape
x = nl.zeros(shape=(128, 512), dtype=nl.float32, buffer=nl.sbuf)
y = nl.zeros(shape=(128, 1), dtype=nl.float32, buffer=nl.sbuf)
y[...] = x  # Error
x[...] = y  # ok

# err_tensor_access_out_of_bound - Error case
x = nl.ndarray([128, 4000], dtype=np.float32, buffer=nl.hbm)
for i in nl.affine_range((4000 + 512 - 1) // 512):
    tile_p, tile_x = nl.mgrid[0:128, 0:512]
    nl.store(x[tile_p, i * 512 + tile_x], value=0)  # Error

# err_tensor_access_out_of_bound - Workaround with mask
for i in nl.affine_range((4000 + 512 - 1) // 512):
    tile_p, tile_x = nl.mgrid[0:128, 0:512]
    nl.store(x[tile_p, i * 512 + tile_x], value=0, mask=i * 512 + tile_x < 4000)  # Ok

# err_tensor_output_not_written_to - Error case
def incorrect(tensor_in, tensor_out):
    M = 128
    N = M + 1
    for i in nl.affine_range(M // N):  # This evaluates to 0
        a = nl.load(tensor_in)
        nl.store(tensor_out, value=a)  # Never called

# err_tensor_output_not_written_to - Workaround
def memset_output(tensor_in, tensor_out, cnd):
    nl.store(tensor_out, value=0)  # Initialize output
    while cnd:
        a = nl.load(tensor_in)
        nl.store(tensor_out, value=a)

# err_unsupported_expression_in_mask - Error case
def test_mask_error(n):
    out = nl.ndarray([8], dtype=nl.int32, buffer=nl.shared_hbm)
    nl.store(out, 0, mask=n > 2)  # Error: n is a runtime value

# err_unsupported_expression_in_mask - Error case 2
def test_mask_error2():
    out = nl.ndarray([8], dtype=nl.int32, buffer=nl.shared_hbm)
    for i in range(8):
        nl.store(out, i, mask=(i % 4) < 2)  # Error: i % 4 is not affine

# err_unsupported_expression_in_mask - Correct usage
def test_mask_correct():
    out = nl.ndarray([8], dtype=nl.int32, buffer=nl.shared_hbm)
    for i in range(8):
        nl.store(out, i, mask=i < 4)  # Ok: i < 4 is affine
    for i in range(8):
        for j in range(8):
            nl.store(out, i, mask=(2*i + 3*j + 1) < 20)  # Ok: affine
    return out

# err_unsupported_memory - Error case
tmp = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.sbuf)
x = nl.load(tmp)  # Error: Expected 'src' to be in 'hbm'

tmp = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.hbm)
x = nl.exp(tmp)  # Error: Expected 'x' to be in 'psum|sbuf'

# err_unsupported_mixing_basic_advanced_tensor_indexing - Error case
a = nl.zeros((4, 4), dtype=nl.float32, buffer=nl.sbuf)
i = nl.arange(4)[:, None]
c = nl.exp(a[i, :])  # Error

# err_unsupported_mixing_basic_advanced_tensor_indexing - Correct usage
c = nl.exp(a[:, :])  # ok
i = nl.arange(4)[:, None]
j = nl.arange(4)[None, :]
c = nl.exp(a[i, j])  # also ok

# err_while_loop_requires_unconditional_entry - Error case
def func_while_error(a):
    a_tile = nl.load(a[0, 0])
    a_scalar = nl.scalar(a_tile)
    while a_scalar < 10:  # Error: traditional while loop
        a_tile = a_tile + 1
        a_scalar = nl.scalar(a_tile)

# err_while_loop_requires_unconditional_entry - Correct usage (do-while)
def func_while_correct(a):
    a_tile = nl.load(a[0, 0])
    a_scalar = nl.scalar(a_tile)
    cond = nl.scalar(True)  # Unconditional entry
    while cond:
        a_tile = a_tile + 1
        a_scalar = nl.scalar(a_tile)
        cond = a_scalar < 10  # Condition evaluated at end
```

## pytorch-neuron-debug.rst

```python
import os
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
import os
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
os.environ["XLA_FLAGS"] = "--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
if os.environ.get("RANK", "0") == "0":
    os.environ["XLA_FLAGS"]="--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
import torch_xla.core.xla_model as xm

if xm.is_master_ordinal():
    os.environ["XLA_FLAGS"]="--xla_dump_hlo_snapshots --xla_dump_to=./dump"
```

```python
def _dump_hlo_snapshot_callback(name: str, addressable_device_index: int, execution_count: int) -> str:
    return 'inputs'
```

```python
def callback(name, addressable_device_index, execution_count):
    if execution_count == 2:
        return 'outputs'
    else:
        return ''

import libneuronxla
old_callback = libneuronxla.register_hlo_snapshot_callback(callback)
```

```python
step = 0
def callback(name, addressable_device_index, execution_count):
    if step == 5:
        return 'inputs'
    else:
        return ''

import libneuronxla
old_callback = libneuronxla.register_hlo_snapshot_callback(callback)

for epoch in range(EPOCHS):
    for idx, (train_x, train_label) in enumerate(train_loader):
        step += 1
```

## neuron-gatherinfo.py

```python
import os
import re
import shutil
import subprocess
import sys


def get_os_version():
    ''' function to obtain the Linux version
        Args:

        Output:

        Returns:
            string with value 'Ubuntu' or 'RedHat'
    '''

    try:
        with open("/proc/version") as fdin:
            data = fdin.read()
            if data.find('Ubuntu') == -1:
                osver = 'RedHat'
            else:
                osver = 'Ubuntu'
    except FileNotFoundError:
        osver = 'Ubuntu'

    return osver


def get_files(*, basedir, matchfiles, verbose):
    ''' function to get the files based on a base directory and file extension

        Args:
            basedir     : base directory where files reside
            matchfiles  : set of files to match
            verbose : flag to indicate if verbose messages need to be displayed

        Output:

        Returns:
            list of files found

    '''

    myfiles = list()
    for dpath, _, files in os.walk(basedir):
        for mfile in files:
            if mfile in matchfiles:
                mfile = os.path.realpath(os.path.join(dpath, mfile))
                if os.path.isfile(mfile):
                    myfiles.append(mfile)
                else:
                    if verbose:
                        print("Warning: {} is not a file".format(mfile))

    return myfiles


def dump_compiler_info(*, outdir, location, allowmodel=False, addfldir=None, verbose=False):
    ''' function to gather the following information:
            Framework:
                - TensorFlow
                - MXNet
                - PyTorch
            Compiler:
        Args:
            outdir      : output directory
            location    : location of compiler-generated files
            allowmodel  : if True, allow gathering of additional files
            verbose : flag to indicate if verbose messages need to be displayed

        Output: compiler-generated files copied to outdir

        Returns:
    '''

    if location is not None:
        if allowmodel:  # copy the entire directory
            try:
                shutil.copytree(location, os.path.join(outdir, os.path.basename(location)),
                                ignore_dangling_symlinks=True)
            except shutil.Error:
                pass
        else:
            fileset = set(['graph_def.neuron-cc.log', 'all_metrics.csv', 'hh-tr-operand-tensortensor.json'])
            l1data = get_files(basedir=location, matchfiles=fileset, verbose=verbose)
            copy_files(outdir=outdir, basedir=location, filelist=l1data, verbose=verbose)

        if addfldir is not None:
            if os.path.isfile(addfldir):
                shutil.copy(addfldir, outdir)
            else:  # directory copy
                try:
                    shutil.copytree(addfldir, os.path.join(outdir, os.path.basename(addfldir)),
                                    ignore_dangling_symlinks=True)
                except shutil.Error:
                    pass


def copy_syslog(*, outdir, include_flag=False, verbose):
    '''
        function to copy contents of the syslog to the output directory

        Args:
            outdir          : output directory location where the syslog's contents
                              are to be copied
            include_flag    : if True, include lines that do not match
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            copy of syslog's contents with just "Neuron-specific" lines

        Returns:
    '''

    regex1 = re.compile(r'^(\S+)\s.*?({})'.format(r"nrtd|neuron|kernel:"))
    regex2 = re.compile(r'^(\S+)\s')

    osver = get_os_version()
    if osver == 'Ubuntu':
        syslog = '/var/log/syslog'
    else:
        syslog = '/var/log/messages'

    try:
        with open(syslog) as fdin,\
            open(os.path.join(outdir, 'copy-of-syslog'), 'w') as fdout:
            for line in fdin:
                match = regex1.search(line)
                if match is not None:
                    fdout.write(line)
                else:
                    if include_flag:
                        match = regex2.match(line)
                        if match is not None:
                            # exclude the rest of the line
                            fdout.write(match.group(1) + ' XXX contents elided XXX\n')
                        else:
                            print("Error in parsing this line: {}".format(line))
    except FileNotFoundError:
        print("Error, /var/log/syslog not found")


def dump_miscinfo(*, outdir, verbose):
    ''' function to dump miscellaneous information, including:
            - system info (uname -a)
            - package info (??? list of packages installed)
            - neuron-ls
            - neuron-top

        Args:
            outdir  : output directory
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Creates various reports in the outdir location

        Returns:

    '''

    osver = get_os_version()
    if osver == 'Ubuntu':
        pkgcmds = ["apt list | egrep '^aws'",
                   "pip list | egrep '^neuron|^numpy|^tensor|^scipy'"]
    else:
        pkgcmds = ["rpm -qa | egrep '^aws|^neuron|^numpy|^tensor|^scipy'"]

    cmds = ["lscpu", "lshw",
            "lspci | grep -i Amazon",
            "neuron-cc --version",
            "neuron-ls",
            "top -b -n 1",
            "uname -a", "uptime"] + pkgcmds

    for cmd in cmds:
        cmdname = cmd.split(' ')[0]  # get just the command name for creating the file
        cmdfile = os.path.join(outdir, "report-{}.txt".format(cmdname))

        with open(cmdfile, "w") as fdout:

            if verbose:
                print("Running cmd: {} and capturing output in file: {}".format(cmd, cmdfile))

            try:
                res = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, universal_newlines=True,
                                       shell=True)
                stdout, stderr = res.communicate()
                if stderr is not None:
                    fdout.write("Error in executing cmd: {}\nError: {}\n".format(cmd, str(stderr)))
                else:
                    fdout.write("Output from executing cmd: {}\n\n{}\n".format(cmd, str(stdout)))
            except (OSError, ValueError) as err:
                fdout.write("Error in executing cmd: {}\nError: {}\n".format(cmd, err))


def dump_proc_info(*, outdir, verbose):
    '''
        function to dump information related to "/proc"

        Args:
            outdir  : output directory
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Creates various reports in the outdir location

        Returns:

    '''

    proc_files = ["/proc/cmdline",
                  "/proc/cpuinfo",
                  "/proc/filesystems",
                  "/proc/interrupts",
                  "/proc/iomem",
                  "/proc/loadavg",
                  "/proc/meminfo",
                  "/proc/modules",
                  "/proc/mtrr",
                  "/proc/version"]

    for procfile in proc_files:
        fname = procfile.split('/')  # use the 2nd and 3rd items from this (canonical form)
        pfile = os.path.join(outdir, "report-{}-{}.txt".format(fname[1], fname[2]))
        if verbose:
            print("Copying contents of: {} to: {}".format(procfile, pfile))

        try:
            with open(pfile, "w") as fdout, open(procfile) as fdin:
                fdout.write("Contents of {}\n\n".format(procfile))
                fdout.write(fdin.read())
        except FileNotFoundError:
            print("Error: file {} not found\n".format(procfile))


def copy_files(*, outdir, basedir, filelist, verbose):
    '''
        function to copy files from the original source area
        into the destination. This is also the place for any
        massaging or eliding of file contents

        Args:
            outdir  : destination location
            basedir : base directory from where the files are to be copied
            filelist: list of files to be copied
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Copy of files (possibly altered) from the source

        Returns:

    '''

    for thisfile in filelist:
        myfile = '.' + thisfile[len(basedir):]
        mydir = os.path.dirname(os.path.join(outdir, myfile))
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
        shutil.copy(thisfile, mydir, follow_symlinks=True)


def write_miscinfo(*, outdir, data):
    '''
        function to write out the contents of the miscellaneous commands

        Args:
            outdir  : destination location
            data    : list of strings to be stored in a file

        Output:
            MISCINFO_FILE created with the contents of the output of the various
            commands
    '''

    flname = os.path.join(outdir, 'miscinfo.txt')

    with open(flname, "w") as fdout:
        fdout.write("\n".join(data))


def package_tarball(*, outdir, allowmodel, ccdir, verbose):
    '''
        function to package everything into a tarball

        Args:
            outdir      : output directory
            allowmodel  : flag to indicate whether the user has allowed
                          gathering of model data

        Output:
            A tar ball created in directory one level above outdir
            this would be the directory provided by the user

        Returns:
    '''

    mytarball = os.path.join(os.path.split(outdir)[0], 'neuron-gatherinfo')

    if verbose:
        print("Creating archive: {}".format(mytarball))

    archivefile = shutil.make_archive(mytarball, 'gztar', outdir)
    print("\n\n\t******\n\tArchive created at:\n\t\t{}\n\tFrom directory:\n\t\t{}\n\t******\n\n".format(archivefile, outdir))
```

## trn2-llama3.3-70b-tutorial.rst

```bash
# Replace this with the path where you downloaded and saved the model files.
MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
# This is where the compiled model will be saved. The same path
# should be used when launching vLLM server for inference.
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

NUM_CORES=128
TP_DEGREE=64
LNC=2

export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
export NEURON_RT_EXEC_TIMEOUT=600 
export XLA_DENSE_GATHER_FACTOR=0 
export NEURON_RT_INSPECT_ENABLE=0

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
        --do-sample \
        --fused-qkv \
        --sequence-parallel-enabled \
        --qkv-kernel-enabled \
        --attn-kernel-enabled \
        --mlp-kernel-enabled \
        --cc-pipeline-tiling-factor 1 \
        --pad-token-id 2 \
        --enable-bucketing \
        --context-encoding-buckets 2048 4096 8192 12288 \
        --token-generation-buckets 2048 4096 8192 12800 \
        --prompt "What is annapurna labs?" 2>&1 | tee log
```

```bash
export NEURON_RT_INSPECT_ENABLE=0 
export NEURON_RT_VIRTUAL_CORE_SIZE=2

# These should be the same paths used when compiling the model.
MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --max-num-seqs 1 \
    --max-model-len 12800 \
    --tensor-parallel-size 64 \
    --device neuron \
    --use-v2-block-manager \
    --override-neuron-config "{\"on_device_sampling_config\": {\"do_sample\": true}, \"skip_warmup\": true}" \
    --port 8000 &
PID=$!
echo "vLLM server started with PID $PID"
```

```bash
# This is the same path as in the previous scenario.
MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
# This is the path where the draft model is downaloded and saved.
DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1B-Instruct/"
# As in the previous scenario, this is where the compiled model will be saved.
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

NUM_CORES=128
TP_DEGREE=64
LNC=2

export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
export NEURON_RT_EXEC_TIMEOUT=600 
export XLA_DENSE_GATHER_FACTOR=0 
export NEURON_RT_INSPECT_ENABLE=0

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
        --enable-bucketing \
        --context-encoding-buckets 2048 4096 8192 12288 \
        --token-generation-buckets 2048 4096 8192 12800 \
        --prompt "What is annapurna labs?" 2>&1 | tee log
```

```bash
export NEURON_RT_INSPECT_ENABLE=0 
export NEURON_RT_VIRTUAL_CORE_SIZE=2

# These should be the same paths used when compiling the model.
MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1B-Instruct/"
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

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
    --override-neuron-config "{\"enable_fused_speculation\":true}" \
    --port 8000 &
PID=$!
echo PID=$PID
echo "vLLM server started with PID $PID"
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
from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.utils.logger import get_logger
from torch.distributed.utils import _replace_by_prefix

logger = get_logger()

_CHECKPOINT_WRAPPED_MODULE = "mod"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."

class CheckPointWrapper(torch.nn.Module):
    def __init__(self, mod) -> None:
        super().__init__()
        self.mod = mod
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        ordered_args = list(args)
        for value in kwargs.values():
            ordered_args += [value]

        # Note: checkpoint cannot accept kwargs
        return torch_checkpoint(self.mod, *ordered_args, use_reentrant=True)
    
    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
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
        """
        _post_state_dict_hook() is called after the state_dict() of this
        FSDP module is executed. For ``checkpoint_wrapper``, it will strip
        checkpoint-wrapped module prefix so that this module can be loaded into
        non-checkpointed modules. It would still be able to be loaded into
        checkpoint-wrapped modules as this class adds the prefix back before
        loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        return state_dict
    
    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
        is called. For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")

def apply_checkpoint(dist_model, layers_to_checkpoint=None):
    checkpoint_wrapper_added = False
    if layers_to_checkpoint is not None and len(layers_to_checkpoint) == 0:
        raise RuntimeError(
            rmsg(f"invalid input layers_to_checkpoint {layers_to_checkpoint}, can't be empty")
        )
    for name, module in dist_model.local_module.named_children():
        # checkpoint layers that are provided in input
        # if layers not provide in input, then checkpoint if it is transformer layer
        if (layers_to_checkpoint and name in layers_to_checkpoint) or (
            not layers_to_checkpoint and type(module) == dist_model.transformer_layer_cls
        ):
            # add_module replaces old module with our own custom module.
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
            dist_model.local_module.add_module(name, CheckPointWrapper(module))
            checkpoint_wrapper_added = True
    if layers_to_checkpoint is not None and not checkpoint_wrapper_added:
        logger.warning(
            rmsg(f"layers_to_checkpoint {layers_to_checkpoint} do not exist in the graph")
        )
    elif layers_to_checkpoint is None and not checkpoint_wrapper_added:
        logger.warning(
            rmsg(
                f"During applying activation checkpointing, transformer_layer_cls {dist_model.transformer_layer_cls.__name__} can not be found in stage {dist_model.pipeline_parallel_rank}, skipping..."
            )
        )

model = NxDPPModel(...)
# Will checkpoint every transformer layer
apply_checkpoint(model)
```

```python
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMHF

# Keep the same class name as original one
class LlamaForCausalLM(LlamaForCausalLMHF):
    ...
```

## yolo_v3_coco_saved_model.py

```python
import tensorflow as tf
from functools import partial
import numpy as np

STRIDES = [8, 16, 32]
ANCHORS = np.array([1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]).astype(np.float32).reshape([3, 3, 2])
ANCHOR_PER_SCALE = 3
BOX_SCORE_THRESH = 0.3
UPSAMPLE_METHOD = "resize"
NUM_CLASSES = 80


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, input_size, trainable):

        self.trainable        = trainable
        self.num_class        = NUM_CLASSES
        self.strides          = STRIDES
        self.anchors          = ANCHORS
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.box_score_thresh = BOX_SCORE_THRESH
        self.upsample_method  = UPSAMPLE_METHOD

        input_data, decoded_shape = preprocessor(input_data, [input_size, input_size])
        self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)

        def decode_boxes(bboxes_and_decoded_shape):
            conv_lbbox, conv_mbbox, conv_sbbox, decoded_shape = bboxes_and_decoded_shape
            conv_lbbox = tf.cast(conv_lbbox, tf.float32)
            conv_mbbox = tf.cast(conv_mbbox, tf.float32)
            conv_sbbox = tf.cast(conv_sbbox, tf.float32)
            conv_lbbox = conv_lbbox[tf.newaxis, ...]
            conv_mbbox = conv_mbbox[tf.newaxis, ...]
            conv_sbbox = conv_sbbox[tf.newaxis, ...]
            decoded_shape = decoded_shape[tf.newaxis, ...]
            with tf.variable_scope('pred_sbbox'):
                pred_sbbox_coors, pred_sbbox_class_scores = self.decode(conv_sbbox, self.anchors[0], self.strides[0], decoded_shape, input_size)

            with tf.variable_scope('pred_mbbox'):
                pred_mbbox_coors, pred_mbbox_class_scores = self.decode(conv_mbbox, self.anchors[1], self.strides[1], decoded_shape, input_size)

            with tf.variable_scope('pred_lbbox'):
                pred_lbbox_coors, pred_lbbox_class_scores = self.decode(conv_lbbox, self.anchors[2], self.strides[2], decoded_shape, input_size)

            with tf.variable_scope('pred_bbox_filter'):
                pred_bbox_coors = tf.concat([pred_sbbox_coors, pred_mbbox_coors, pred_lbbox_coors], axis=1)
                pred_bbox_class_scores = tf.concat([pred_sbbox_class_scores, pred_mbbox_class_scores, pred_lbbox_class_scores], axis=1)
                nms_top_k = 100
                nms_thresh= 0.45
                coors, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    pred_bbox_coors,
                    pred_bbox_class_scores,
                    max_output_size_per_class=nms_top_k,
                    max_total_size=nms_top_k,
                    iou_threshold=nms_thresh,
                    score_threshold=self.box_score_thresh,
                    pad_per_class=False,
                    clip_boxes=False,
                    name='CombinedNonMaxSuppression',
                )
                scores = scores[..., tf.newaxis]
                classes = classes[..., tf.newaxis]
            return coors[0], scores[0], classes[0]

        with tf.name_scope('Postprocessor'):
            coors, scores, classes = tf.map_fn(
                decode_boxes, [self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, decoded_shape],
                dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)

        with tf.variable_scope('pred_bbox'):
            self.pred_bbox_boxes = tf.identity(coors, name='boxes')
            self.pred_bbox_scores = tf.identity(scores[..., 0], name='scores')
            self.pred_bbox_classes = tf.identity(classes[..., 0], name='classes')

    def __build_nework(self, input_data):
        route_1, route_2, input_data = darknet53(input_data, self.trainable)

        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride, decoded_shape, input_size):
        conv_output = tf.cast(conv_output, tf.float32)
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_xywh = tf.reshape(pred_xywh, (-1, output_size*output_size*3, pred_xywh.shape[-1]))
        pred_conf = tf.reshape(pred_conf, (-1, output_size*output_size*3))
        pred_prob = tf.reshape(pred_prob, (-1, output_size*output_size*3, pred_prob.shape[-1]))

        return tf_postprocess_boxes(pred_xywh, pred_conf, pred_prob, decoded_shape, input_size, self.box_score_thresh)


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = convolutional(input_data, filters_shape=(3, 3, 32,  64), trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = convolutional(input_data, filters_shape=(3, 3,  64, 128), trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = convolutional(input_data, filters_shape=(3, 3, 128, 256), trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        weight = tf.cast(weight, tf.float16)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable,
                                                 fused=False)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            bias = tf.cast(bias, tf.float16)
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')
        residual_output = input_data + short_cut
    return residual_output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


def decode_jpeg_resize(input_tensor, image_size):
    tensor = tf.image.decode_png(input_tensor, channels=3)
    shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.resize_image_with_pad(tensor, image_size[0], image_size[1])
    tensor /= 255.0
    return tf.cast(tensor, tf.float16), shape


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        batch_tensor, batch_shape = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float16, tf.int32), back_prop=False, parallel_iterations=16)
    return batch_tensor, batch_shape


def tf_postprocess_boxes(pred_xywh, pred_conf, pred_prob, org_img_shape, input_size, score_threshold):
    batch_size = tf.shape(pred_xywh)[0]

    pred_coor = tf.concat([pred_xywh[:, :, :2] - pred_xywh[:, :, 2:] * 0.5,
                           pred_xywh[:, :, :2] + pred_xywh[:, :, 2:] * 0.5], axis=-1)
    org_wh = org_img_shape[:, tf.newaxis, 1::-1]
    org_whwh = tf.concat([org_wh, org_wh], axis=-1)
    org_whwh = tf.cast(org_whwh, tf.float32)
    input_size = np.float32(input_size)
    resize_ratio = input_size / tf.reduce_max(org_whwh, axis=-1)
    dwhwh = (input_size - resize_ratio * org_whwh) / 2
    pred_coor = (pred_coor - dwhwh) / resize_ratio

    scores = pred_conf * tf.reduce_max(pred_prob, axis=-1)
    score_mask = scores > score_threshold
    coors = pred_coor[score_mask]
    pred_conf = pred_conf[score_mask]
    pred_conf = tf.reshape(pred_conf, [batch_size, -1, 1])
    pred_prob = pred_prob[score_mask]
    pred_prob = tf.reshape(pred_prob, [batch_size, -1, pred_prob.shape[-1]])
    class_scores = pred_conf * pred_prob
    coors = tf.reshape(coors, [batch_size, -1, 1, coors.shape[-1]])
    class_scores = tf.reshape(class_scores, [batch_size, -1, class_scores.shape[-1]])
    return coors, class_scores
```

## disaggregated-inference-tutorial-1p1d.rst

```bash
#!/bin/bash
# compile.sh

while [[ $# -gt 0 ]]; do
   case $1 in
      --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
      --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
      --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
      *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path>"
            exit 1
            ;;
   esac
done

export COMPILED_MODEL_PATH="di_traced_model_tp${TP_DEGREE}_b${BATCH_SIZE}/"

inference_demo \
   --model-type llama \
   --task-type causal-lm \
   run \
   --model-path $MODEL_PATH \
   --compiled-model-path $COMPILED_MODEL_PATH \
   --torch-dtype bfloat16 \
   --tp-degree $TP_DEGREE \
   --batch-size $BATCH_SIZE \
   --ctx-batch-size 1 \
   --tkg-batch-size $BATCH_SIZE \
   --is-continuous-batching \
   --max-context-length 8192 \
   --seq-len 8192 \
   --on-device-sampling \
   --fused-qkv \
   --global-topk 256 --dynamic \
   --top-k 50 --top-p 0.9 --temperature 0.7 \
   --do-sample \
   --sequence-parallel-enabled \
   --qkv-kernel-enabled \
   --attn-kernel-enabled \
   --mlp-kernel-enabled \
   --cc-pipeline-tiling-factor 1 \
   --pad-token-id 2 \
   --logical-neuron-cores 2 \
   --context-encoding-buckets 256 512 1024 2048 4096 8192 \
   --token-generation-buckets 512 1024 2048 4096 8192 \
   --apply-seq-ids-mask \
   --enable-bucketing \
   --prompt "test prompt" \
   --save-sharded-checkpoint \
   --attn-block-tkg-nki-kernel-enabled \
   --attn-block-tkg-nki-kernel-cache-update \
   --k-cache-transposed \
   --async-mode \
   --compile-only
```

```bash
#!/bin/bash
# server.sh

while [[ $# -gt 0 ]]; do
   case $1 in
      --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
      --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
      --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
      --compiled-model-path)
            COMPILED_MODEL_PATH="$2"
            shift 2
            ;;
      --neuron-send-ip)
            SEND_IP="$2"
            shift 2
            ;;
      --neuron-recv-ip)
            RECV_IP="$2"
            shift 2
            ;;
      *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path> \
                           --compiled-model-path <path> --send-ip <ip> --recv-ip <ip>"
            exit 1
            ;;
   esac
done

export NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT=45645
export NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED=1
export NEURON_COMPILED_ARTIFACTS="$COMPILED_MODEL_PATH"
export NEURON_SEND_IP="$SEND_IP"
export NEURON_RECV_IP="$RECV_IP"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2

if [ "$SEND" = "1" ]; then
   PORT=8100
   if [ "$SINGLE_INSTANCE" = "1" ]; then
      export NEURON_RT_VISIBLE_CORES=0-31
   fi
   TRANSFER_CONFIG='{
            "kv_connector":"NeuronConnector",
            "kv_buffer_device":"cpu",
            "kv_role":"kv_producer",
            "kv_rank":0,
            "kv_parallel_size":2,
            "kv_buffer_size":2e11,
            "kv_ip":"'"$NEURON_SEND_IP"'",
            "neuron_core_offset": 0
      }'
   
else
   PORT=8200
   if [ "$SINGLE_INSTANCE" = "1" ]; then
      NC_OFFSET=32
      export NEURON_RT_VISIBLE_CORES=32-63
   else   
      NC_OFFSET=0
   fi
   TRANSFER_CONFIG='{
            "kv_connector":"NeuronConnector",
            "kv_buffer_device":"cpu",
            "kv_role":"kv_consumer",
            "kv_rank":1,
            "kv_parallel_size":2,
            "kv_buffer_size":2e11,
            "kv_ip":"'"$NEURON_SEND_IP"'",
            "neuron_core_offset": "'"$NC_OFFSET"'"
      }'
fi

python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --max-num-seqs "$BATCH_SIZE" \
      --max-model-len 8192 \
      --tensor-parallel-size "$TP_DEGREE" \
      --device neuron \
      --use-v2-block-manager \
      --override-neuron-config "{}" \
      --kv-transfer-config "$TRANSFER_CONFIG" \
      --port "$PORT"
```

```bash
#!/bin/bash
# llmperf.sh

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="mock_key"

python llmperf/token_benchmark_ray.py \
   --model=$MODEL_PATH \
   --tokenizer=$MODEL_PATH \
   --mean-input-tokens=1024 \
   --stddev-input-tokens=0\
   --mean-output-tokens=100 \
   --stddev-output-tokens=10 \
   --max-num-completed-requests=200 \
   --timeout=1720000 \
   --num-concurrent-requests=4 \
   --results-dir=llmperf_results \
   --llm-api=openai \
   --additional-sampling-params "{\"top_k\": 50, \"top_p\": 0.9, \"temperature\": 0.7}"
```

```bash
#!/bin/bash
# baseline_server.sh

while [[ $# -gt 0 ]]; do
   case $1 in
      --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
      --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
      --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
      --compiled-model-path)
            COMPILED_MODEL_PATH="$2"
            shift 2
            ;;
      *)  
            echo "Unknown parameter: $1"
            echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path> \
                           --compiled-model-path <path>"
            exit 1
            ;;
   esac
done

export NEURON_COMPILED_ARTIFACTS="$COMPILED_MODEL_PATH"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2

if [ "$SINGLE_INSTANCE" = "1" ]; then
   NEURON_RT_VISIBLE_CORES=0-31
fi

python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --max-num-seqs "$BATCH_SIZE" \
      --max-model-len 8192 \
      --tensor-parallel-size "$TP_DEGREE" \
      --device neuron \
      --use-v2-block-manager \
      --override-neuron-config "{}" \
      --port 8000
```

## torch-core-placement.rst

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc4
m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc5
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
m1 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc1
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-4-neuron-pipeline-cores.pt')  # Loads to nc0-nc3
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2
m1 = torch.jit.load('model-with-4-neuron-pipeline-cores.pt')  # Loads to nc3-nc6
m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc7
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3
m2 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
m3 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc2
m4 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc0
```

```python
import torch
import torch_neuron

m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1
m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3
m2 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2
```

```python
import torch
import torch_neuron

models = list()
for _ in range(4):
    model = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')
    models.append(model)
```

```python
import torch
import torch_neuron

# NOTE: Order of loads does NOT matter

with torch_neuron.experimental.neuron_cores_context(2):
    m1 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc2-nc3

with torch_neuron.experimental.neuron_cores_context(0):
    m2 = torch.jit.load('model-with-3-neuron-pipeline-cores.pt')  # Loads to nc0-nc2

with torch_neuron.experimental.neuron_cores_context(0):
    m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads to nc0-nc1

with torch_neuron.experimental.neuron_cores_context(3):
    m3 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads to nc3
```

```python
import torch
import torch_neuron

with torch_neuron.experimental.multicore_context():
    m0 = torch.jit.load('model-with-1-neuron-pipeline-cores.pt')  # Loads replications to nc0-nc7
```

```python
import torch
import torch_neuron

with torch_neuron.experimental.neuron_cores_context(start_nc=2, nc_count=4):
    m0 = torch.jit.load('model-with-2-neuron-pipeline-cores.pt')  # Loads replications to nc2-nc5
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

        self.layer1 = torch.nn.Linear(4,4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4,2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))

with torch.no_grad():

    model = NN()

    inp = torch.rand(4,4)
    output = model(inp)

    with torch_neuronx.experimental.profiler.profile(
        port=9012,
        profile_type='operator',
        ms_duration=10000 ):
        
        # IMPORTANT: the model has to be transferred to XLA within
        # the context manager, otherwise profiling won't work
        neuron_model = model.to(device)
        neuron_inp = inp.to(device)
        
        output_neuron = neuron_model(neuron_inp)
        xm.mark_step()
```

```python
import os
import time
import torch
import torch_neuronx
from torch_neuronx.experimental import profiler

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(4,4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4,2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))

model = NN()
model.eval()

inp = torch.rand(4,4)

output = model(inp)

with torch_neuronx.experimental.profiler.profile(
    port=9012,
    profile_type='operator',
    ms_duration=10000,
    traced_only=True):

    neuron_model = torch_neuronx.trace(model, inp, compiler_workdir="./compiler_cache")
    neuron_model(inp)
```

## perceiver-multimodal_benchmark.py

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
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


def custom_model_forward(
        self,
        nchunks,
        image_chunk_size,
        audio_chunk_size,
        neuron_decoder,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        perceiver_wrapper = MultimodalPerceiverWrapper(self.perceiver, nchunks, image_chunk_size, audio_chunk_size)
        outputs = perceiver_wrapper(
            inputs,
            neuron_decoder,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


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

## disaggregated-inference.rst

```python
# Proxy Server - Request Flow
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

```python
# Worker Discovery and Connection Manager
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
# Static 1P1D Mode - Buffer Initialization
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
# Dynamic xPyD Mode - Buffer Setup
def maybe_setup_buffer(self, remote_ip, remote_port):
    if self.static_buffer:
        return self.static_buffer

    key = "" if self.config.is_kv_producer else (remote_ip, remote_port)
    
    if key in self.connection_dict:
        return self.connection_dict[key]
```

```python
# Transfer Engine - KV Cache Transfer
class NeuronTransferEngine:
    def transfer_neuron_tensors(self, tensors, offsets, lengths, peer_devices, ...):
        self.engine.queue_transfer_with_token(
            tensors, offsets, lengths, peer_devices, self.local_devices,
            self.comm_ids, completion_count, completion_token, use_queue,
            completion_time_out)
```

```python
# Send Handler - Prefill Side
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
# Starting Transfers - Prefill Side
# ensure that the request is finished prefill
if request_id not in self.lookup_dict:
    self.router.send_json(identity, {"success": False})
    return

# After getting decode server request and prefill is finished
kv_caches, offsets, lengths, peer_devices = \
    self.generate_transfer_sequences(entry, remote_id=identity_str)

# Start transfer
self.get_transfer_engine(remote_id=identity_str).transfer_neuron_tensors(
    kv_caches, offsets, lengths, peer_devices,
    completion_token=entry.completion_token)
```

```python
# Starting Transfers - Decode Side
# receive prefill worker's output token
entry.output_token = torch.tensor(
    response["output_token"]).unsqueeze(0)

kv_caches, offsets, lengths, peer_devices = \
     self.generate_transfer_sequences(entry)

# do not wait for request completion for recv buffer
self.get_transfer_engine().transfer_neuron_tensors(
    kv_caches, offsets, lengths, peer_devices,
    completion_token=entry.completion_token)
```

## mlp.rst

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
# or
device = 'xla'
```

```python
optimizer.zero_grad()
```

```python
output = model(train_x)
```

```python
loss_fn(output, train_label)
```

```python
loss.backward()
```

```python
optimizer.step()
```

```python
import torch_xla.core.xla_model as xm

xm.mark_step()
```

```python
loss.item()
```

```python
import torch_xla.core.xla_model as xm

xm.save(checkpoint, path)
```

```python
import torch_xla.distributed.xla_backend
import torch.distributed

torch.distributed.init_process_group('xla')
```

```python
from torch_xla.distributed import MpDeviceLoader

data_loader = MpDeviceLoader(dataset, batch_size=32)
```

```python
import torch_xla.core.xla_model as xm

xm.optimizer_step(optimizer)
```

```python
import torch_xla.core.xla_model as xm

world_size = xm.xrt_world_size()
```

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
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

## finetune_hftrainer.rst

```python
import os
if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "1":
    from accelerate.accelerator import Accelerator
    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        return tensor
    Accelerator.pad_across_processes = pad_across_processes
```

```python
# Enable torchrun
import os
import torch
import torch_xla.distributed.xla_backend
from packaging import version
from transformers import __version__, Trainer
if version.parse(__version__) < version.parse("4.26.0") and os.environ.get("WORLD_SIZE"):
    torch.distributed.init_process_group('xla')

# Disable DDP for torchrun
import contextlib
if version.parse(__version__) < version.parse("4.20.0"):
    def _wrap_model(self, model, training=True):
        model.no_sync = lambda: contextlib.nullcontext()
        return model
else:
    def _wrap_model(self, model, training=True, dataloader=None):
        model.no_sync = lambda: contextlib.nullcontext()
        return model
Trainer._wrap_model = _wrap_model

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers
if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16
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

...

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
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import restructure
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
# Compile Encoder
neuron_encoder.encoder_wrapper = torch_neuronx.trace(
  neuron_encoder.encoder_wrapper,
  (embedding_output, sample_inputs, extended_attention_mask),
  compiler_workdir=COMPILER_WORKDIR_ENCODER,
  compiler_args=[f"--temp-dir={COMPILER_WORKDIR_ENCODER}", "--auto-cast=none"]
)

torch.jit.save(neuron_encoder.encoder_wrapper, encoder_fname)
```

```python
# Compile Decoder
neuron_decoder.decoder_wrapper = torch_neuronx.trace(
   neuron_decoder.decoder_wrapper,
   (z, query_mask, audio_input, audio_input_without_pos, audio_subsampled_point, audio_padding,
        image_input, image_input_without_pos, image_subsampled_point, image_padding,
        label_input, label_input_without_pos, label_padding),
   compiler_workdir=COMPILER_WORKDIR_DECODER,
   compiler_args=[f"--temp-dir={COMPILER_WORKDIR_DECODER}", "--auto-cast=none"]
)

torch.jit.save(neuron_decoder.decoder_wrapper, decoder_fname)
```

## matrix_multiplication.rst

```python
import neuron.nki as nki
import neuron.nki.language as nl
import numpy as np

@nki.jit
def nki_matmul_basic_(lhs, rhs):
    """
    Basic NKI matrix multiplication kernel.
    Computes: lhs [M, K] * rhs [K, N] = output [M, N]
    where M=64, K=128, N=512
    """
    # Define indices to access input tensors
    i_m, i_n = nl.mgrid[64, 512]
    i_k = nl.arange(128)
    
    # Load LHS in transposed form (adhering to layout considerations)
    lhs_tile = nl.load(lhs[i_k, i_m])
    
    # Load RHS
    rhs_tile = nl.load(rhs[i_k, i_n])
    
    # Perform matrix multiplication with transposed LHS
    output = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)
    
    # Copy result from PSUM to SBUF
    output_sbuf = nl.copy(output, buffer=nl.sbuf)
    
    # Store result to HBM
    nl.store(output_sbuf, output[i_m, i_n])
```

```python
import torch
import neuron.nki as nki

# Execute kernel and verify correctness
lhs = torch.randn(64, 128, dtype=torch.bfloat16)
rhs = torch.randn(128, 512, dtype=torch.bfloat16)

# Run NKI kernel
nki_output = nki_matmul_basic_(lhs, rhs)

# Compare with PyTorch
torch_output = torch.matmul(lhs, rhs)
assert torch.allclose(nki_output, torch_output, rtol=1e-2, atol=1e-2)
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

@nki.jit
def nki_matmul_tiled_(lhs_t, rhs, output_shape):
    """
    Tiled matrix multiplication kernel.
    Handles larger matrices by tiling across M, N, and K dimensions.
    """
    M, N = output_shape
    
    # Tile LHS_T free dimension (M)
    for m in nl.affine_range(0, M, 128):
        # Tile RHS free dimension (N)
        for n in nl.affine_range(0, N, 512):
            # Zero-out accumulator buffer
            i_m = nl.arange(128)
            i_n = nl.arange(512)
            psum_buf = nl.zeros((128, 512), buffer=nl.psum)
            
            # Tile contraction dimension (K)
            for k in nl.affine_range(0, 1024, 128):
                i_k = nl.arange(128)
                
                # Load tiles
                lhs_tile = nl.load(lhs_t[m : m+128, k : k+128])
                rhs_tile = nl.load(rhs[k : k+128, n : n+512])
                
                # Accumulate matmul results
                psum_buf += nl.matmul(lhs_tile, rhs_tile)
            
            # Copy from PSUM to SBUF and store
            result = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(result, output[m : m+128, n : n+512])
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

@nki.jit
def nki_matmul_hoist_load_(lhs_t, rhs, output_shape):
    """
    Optimization 1: Remove redundant loads by hoisting them out of innermost loop.
    """
    M, N = output_shape
    
    for m in nl.affine_range(0, M, 128):
        for n in nl.affine_range(0, N, 512):
            psum_buf = nl.zeros((128, 512), buffer=nl.psum)
            
            # Hoist RHS load outside K loop
            rhs_tile = nl.load(rhs[:, n : n+512])
            
            for k in nl.affine_range(0, 1024, 128):
                # Load LHS tile
                lhs_tile = nl.load(lhs_t[m : m+128, k : k+128])
                
                # Accumulate matmul results
                psum_buf += nl.matmul(lhs_tile, rhs_tile[k : k+128, :])
            
            result = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(result, output[m : m+128, n : n+512])
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

@nki.jit
def nki_matmul_block_free_dimension_(lhs_t, rhs, output_shape):
    """
    Optimization 2: Improve arithmetic intensity through blocking free dimensions.
    """
    M, N = output_shape
    
    # Block free dimensions
    for m_block in nl.affine_range(0, M, 256):
        for n_block in nl.affine_range(0, N, 1024):
            # Load larger blocks of LHS and RHS
            lhs_tiles = nl.load(lhs_t[m_block : m_block+256, :])
            rhs_tiles = nl.load(rhs[:, n_block : n_block+1024])
            
            psum_buf = nl.zeros((256, 1024), buffer=nl.psum)
            
            for k in nl.affine_range(0, 1024, 128):
                # Use blocked tiles
                lhs_tile = lhs_tiles[:, k : k+128]
                rhs_tile = rhs_tiles[k : k+128, :]
                
                psum_buf += nl.matmul(lhs_tile, rhs_tile)
            
            result = nl.copy(psum_buf, buffer=nl.sbuf)
            nl.store(result, output[m_block : m_block+256, n_block : n_block+1024])
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

@nki.jit
def nki_matmul_fully_optimized_(lhs_t, rhs, output_shape):
    """
    Optimization 3: Block all dimensions and optimize DMA efficiency.
    Optimized for large matrices: M, N multiples of 2048; K multiple of 512.
    """
    M, N = output_shape
    
    NUM_BLOCK_M = 16  # 2048 numbers in M dimension
    NUM_BLOCK_N = 2   # 1024 numbers in N dimension
    NUM_BLOCK_K = 8   # 1024 numbers in K dimension
    
    TILE_M = 128
    TILE_N = 512
    TILE_K = 128
    
    for m_block in nl.affine_range(0, M, NUM_BLOCK_M * TILE_M):
        for n_block in nl.affine_range(0, N, NUM_BLOCK_N * TILE_N):
            # Initialize result tiles for accumulation
            result_tiles = nl.zeros((NUM_BLOCK_M, TILE_M, NUM_BLOCK_N, TILE_N), buffer=nl.sbuf)
            
            # Block contraction dimension with sequential range
            for k_block in nl.sequential_range(0, 1024, NUM_BLOCK_K * TILE_K):
                # Load blocked tiles
                lhs_tiles = nl.load(lhs_t[m_block : m_block + NUM_BLOCK_M * TILE_M, 
                                          k_block : k_block + NUM_BLOCK_K * TILE_K])
                rhs_tiles = nl.load(rhs[k_block : k_block + NUM_BLOCK_K * TILE_K, 
                                        n_block : n_block + NUM_BLOCK_N * TILE_N])
                
                # Compute matmuls for all combinations
                for m_idx in nl.affine_range(NUM_BLOCK_M):
                    for n_idx in nl.affine_range(NUM_BLOCK_N):
                        psum_buf = nl.zeros((TILE_M, TILE_N), buffer=nl.psum)
                        
                        for k_idx in nl.affine_range(NUM_BLOCK_K):
                            lhs_tile = lhs_tiles[m_idx * TILE_M : (m_idx + 1) * TILE_M,
                                                 k_idx * TILE_K : (k_idx + 1) * TILE_K]
                            rhs_tile = rhs_tiles[k_idx * TILE_K : (k_idx + 1) * TILE_K,
                                                 n_idx * TILE_N : (n_idx + 1) * TILE_N]
                            
                            psum_buf += nl.matmul(lhs_tile, rhs_tile)
                        
                        result_tiles[m_idx, :, n_idx, :] += nl.copy(psum_buf, buffer=nl.sbuf)
            
            # Store results
            nl.store(result_tiles, output[m_block : m_block + NUM_BLOCK_M * TILE_M,
                                          n_block : n_block + NUM_BLOCK_N * TILE_N])
```

```python
import torch
import neuron.nki as nki

def test_correctness(kernel_fn, kernel_name, lhs, rhs):
    """
    Test correctness of NKI kernel against PyTorch implementation.
    """
    print(f"Checking correctness of {kernel_name}")
    
    # Run NKI kernel
    nki_output = kernel_fn(lhs, rhs)
    
    # Compare with PyTorch
    torch_output = torch.matmul(lhs, rhs)
    
    if torch.allclose(nki_output, torch_output, rtol=1e-2, atol=1e-2):
        print("NKI and Torch match")
    else:
        print("NKI and Torch do NOT match")
```

```python
import neuron.nki as nki

def benchmark_kernel(kernel_fn, lhs, rhs, num_iterations=100):
    """
    Benchmark NKI kernel performance.
    """
    import time
    
    # Warmup
    kernel_fn(lhs, rhs)
    
    # Measure
    start = time.time()
    for _ in range(num_iterations):
        kernel_fn(lhs, rhs)
    end = time.time()
    
    latency_ms = (end - start) / num_iterations * 1000
    return latency_ms
```

## matrix_multiplication_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
```

```python
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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N

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

  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

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

    Used to remove heads.

    Args:
        layer (`BaseParallelLinear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `BaseParallelLinear`: The pruned layer as a new layer with `requires_grad=True`.
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
        # Per attention head and per partition values
        world_size = parallel_state.get_tensor_model_parallel_size()
        self.num_attention_heads_per_partition = divide(
            self.n_heads, world_size)
        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
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

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads_per_partition, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.num_attention_heads_per_partition = self.num_attention_heads_per_partition - len(heads)
        self.hidden_size_per_partition = self.key_value_proj_dim * self.num_attention_heads_per_partition
        self.pruned_heads = self.pruned_heads.union(heads)

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        values = values[:, :, tp_rank * self.num_attention_heads_per_partition:(tp_rank + 1)
                                                                     * self.num_attention_heads_per_partition]

        values = values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        self.is_decoder = True
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.num_attention_heads_per_partition,
                               self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.hidden_size_per_partition)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states,
            past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states,
            past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads_per_partition, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (
                self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class ParallelSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=False)
        self.SelfAttention = ParallelAttention(config,
                                         has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class ParallelCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = ParallelAttention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


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


class ParallelFF(T5LayerFF):
    def __init__(self, config: T5Config):
        super().__init__(config)
        if config.is_gated_act:
            self.DenseReluDense = ParallelDenseGatedActDense(config)
        else:
            self.DenseReluDense = ParallelDenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


def load_pretrained_with_parallel_attn(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")

    # Parallel implementation of Attention modules.
    from t5_model_layers import ParallelSelfAttention, ParallelFF, ParallelCrossAttention

    for index, block in enumerate(model.decoder.block):
        if index == 0:
            block.layer[0] = ParallelSelfAttention(model.config,
                                                   has_relative_attention_bias=True)
        else:
            block.layer[0] = ParallelSelfAttention(model.config)
        block.layer[1] = ParallelCrossAttention(model.config)
        block.layer[2] = ParallelFF(model.config)
    # Load the weights into the parallel layers        
    neuronx_distributed.parallel_layers.load(model_name.split("/")[-1] + ".pt", model, sharded=False)

    return model
```

## mamba_nki_kernels.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
```

```python
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
# We create the variable shapes by randomly cropping the sequences
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

## bert_model.py

```python
import tensorflow as tf
from tensorflow.neuron import fuse

# Example 1: Using the fuse decorator for compiler optimization
fuser = fuse(compiler_args=['--fp32-cast', 'matmult'], timeout=360000)
bert.encoder = fuser(bert.encoder)
```

```python
import tensorflow as tf

# Example 2: Creating placeholders with dynamic batch size
input_ids_ph_shape = input_ids.shape.as_list()
input_ids_ph_shape[0] = None
input_ids_ph = tf.placeholder(input_ids.dtype, input_ids_ph_shape, name='input_ids')
```

```python
import tensorflow as tf

# Example 3: Modifying NeuronOp attributes for batch axis configuration
neuron_op_node = [node for node in new_graph_def.node if node.op == 'NeuronOp'][0]
neuron_op_node.attr['input_batch_axis'].list.i[:] = [0, 0]
neuron_op_node.attr['output_batch_axis'].list.i[:] = [0]
```

```python
import tensorflow as tf

# Example 4: Saving optimized model with inputs and outputs
tf.saved_model.simple_save(sess, args.output_saved_model, inputs, outputs)
```

```python
import tensorflow as tf

# Example 5: Self-attention implementation with multi-head attention
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
import tensorflow as tf

# Example 6: Layer normalization implementation
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


class DefaultBoxes(object):

    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios,
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = np.sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*np.sqrt(alpha), sk1/np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = np.array(self.default_boxes)
        self.dboxes = self.dboxes.clip(min=0, max=1)
        self.dboxes_ltrb = self.dboxes.copy()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
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

## ssd300_model.py

```python
import numpy as np
import tensorflow as tf
from functools import partial
import tensorflow.neuron as tfn


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


class DefaultBoxes(object):

    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios,
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = np.sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*np.sqrt(alpha), sk1/np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = np.array(self.default_boxes)
        self.dboxes = self.dboxes.clip(min=0, max=1)
        self.dboxes_ltrb = self.dboxes.copy()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
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
        # Configure NeuronConfig.
        override_neuron_config={
            "max_context_length": 1024,
            "skip_warmup": True,
        },
        device="neuron"
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
    draft_neuron_config.sequence_parallel_enabled = False
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
            state_tensor = torch.ops.aten.slice(state_tensor, dim=j, start=state_tensor_dim_length - npos, end=state_tensor_dim_length)
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

## sdxl_base_1024_compile.py

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import math
import copy
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
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
```

```python
# Compile Text Encoder with torch_neuronx.trace
neuron_text_encoder = torch_neuronx.trace(
    traceable_text_encoder,
    text_input_ids_1,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(neuron_text_encoder, text_encoder_filename)
```

```python
# Compile UNet with compiler arguments and optimization flags
unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)

torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
torch.jit.save(unet_neuron, unet_filename)
```

```python
# Compile VAE decoder with async loading
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder')
)

torch_neuronx.async_load(decoder_neuron)

decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)
```

## neuron_profile_for_nki.rst

```python
import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "
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
import numpy as np
import neuronperf

# Timer usage
timer = neuronperf.Timer()
with timer:
    pass  # code to time

total_duration_s = timer.total_duration("s")
durations = timer.durations("s")
timestamps = timer.timestamps()
timer_length = len(timer)
```

```python
import numpy as np
import neuronperf

# Timestamp conversion
result_scalar = neuronperf.timestamp_convert(1, "s", "ms")
times = np.array([1, 2, 3])
times_ms = neuronperf.timestamp_convert(times, "s", "ms")
```

```python
import neuronperf

# Model index creation
index = neuronperf.model_index.create("dummy_model.ext", model_name="dummy")
```

```python
import neuronperf

# Model index save, load, delete
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
model_indexes = [neuronperf.model_index.create(f"Dummy_{x}", model_name="Dummy") for x in range(10)]
combined_index = neuronperf.model_index.append(*model_indexes)
```

```python
import neuronperf

# Model index filter
idx_1 = neuronperf.model_index.create("fake", performance_level=2, compile_s=1)
idx_2 = neuronperf.model_index.create("fake2", compile_s=2)
idx = neuronperf.model_index.append(idx_1, idx_2)
filtered = neuronperf.model_index.filter(idx, filename="fake")
```

```python
import numpy as np
import neuronperf

# Benchmark with CPU
benchmarker_results = neuronperf.cpu.benchmark(
    neuronperf.DummyModel,
    [np.array([1, 2, 3, 4])],
    duration=2,
    n_models=4,
    multiprocess=False,
    multiinterpreter=False,
    verbosity=2,
    return_timers=True,
)
```

```python
import neuronperf

# Benchmark with custom load function
reports = neuronperf.benchmark(
    load_fn=lambda path, device_id: None,
    model_filename="dummy_filename",
    inputs=[[1]],
    duration=2,
    n_models=4,
    multiprocess=False,
    multiinterpreter=False,
    verbosity=2,
)
```

```python
import neuronperf

# Get and print reports
reports = neuronperf.get_reports(benchmarker_results)
neuronperf.print_reports(reports)
csv_file = neuronperf.write_csv(reports)
json_file = neuronperf.write_json(reports)
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

print(out)
```

```python
device = xm.xla_device()
a = torch.randn(256, 1024, dtype=torch.float32).to(device)
b = torch.randn(256, 1024, dtype=torch.float32).to(device)
c = nki_tensor_add(a, b) # calling a NKI kernel, instead of the built-in torch op
out = a * b * c
print(out)
```

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_customop_tutorial(a, b):
   c = a + b
   out = a * b * c
   return out
```

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_customop_tutorial(a, b):
   c = nki_tensor_add(a, b) # calling a NKI kernel, instead of the built-in jax op
   out = a * b * c
   return out
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
    # gradients for a and b
    return dy, dy

# now, let's define the compute graph
a = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
b = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
c = NkiAddFunc.apply(a, b)
out = a * b * c

# here we define a (dummy) loss-function, in prep for backward propagation
loss = out.sum()

# lastly, let's invoke the auto-grad engine
loss.backward()

xm.mark_step()
```

```python
import jax

@jax.custom_vjp
def nki_add_func(a, b):
   return nki_tensor_add(a, b)

def f_forward(a, b):
   # operator output and residual (same as input here)
   return nki_add_func(a, b), (a, b)

def f_backward(res, grad):
   # gradients for a and b
   return grad, grad

nki_add_func.defvjp(f_forward, f_backward)

@jax.jit
def jax_customop_tutorial_and_grad(a, b):
   out = nki_add_func(a, b) * x * y

   # use the same dummy loss function (output sum) as PyTorch example above
   grad = jax.grad(lambda x, y: (nki_add_func(x, y) * x * y).sum(), argnums=(0, 1))(a, b)
   return out, *grad
```

## spmd_tensor_addition.rst

```python
import nki
import nki.language as nl

@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
    # Allocate output tensor
    c_output = nl.zeros(a_input.shape, dtype=a_input.dtype)
    
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
    indices_x = nl.arange(tile_size_x)[:, None] + offset_x
    indices_y = nl.arange(tile_size_y)[None, :] + offset_y
    
    # Load tiles from input tensors
    a_tile = nl.load(a_input[indices_x, indices_y])
    b_tile = nl.load(b_input[indices_x, indices_y])
    
    # Compute sum
    c_tile = a_tile + b_tile
    
    # Store result back to output tensor
    nl.store(c_output[indices_x, indices_y], c_tile)
    
    return c_output
```

```python
def nki_tensor_add(a, b):
    # Get input tensor dimensions
    shape = a.shape
    
    # Define tile sizes
    tile_size_x = 128
    tile_size_y = 512
    
    # Calculate grid dimensions
    grid_x = shape[0] // tile_size_x
    grid_y = shape[1] // tile_size_y
    
    # Launch kernel with 2D grid
    return nki_tensor_add_kernel_[grid_x, grid_y](a, b)
```

```python
import torch
from spmd_tensor_addition_nki_kernels import nki_tensor_add

# Prepare input tensors
a = torch.rand(1024, 2048, dtype=torch.bfloat16)
b = torch.rand(1024, 2048, dtype=torch.bfloat16)

# Execute NKI kernel
output_nki = nki_tensor_add(a, b)

# Compute reference output using PyTorch
output_torch = a + b

# Verify correctness
assert torch.allclose(output_nki, output_torch), "NKI and Torch outputs do not match"
print("NKI and Torch match")
```

```python
import jax
import jax.numpy as jnp
from spmd_tensor_addition_nki_kernels import nki_tensor_add

# Prepare input arrays
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (1024, 2048), dtype=jnp.bfloat16)
b = jax.random.uniform(key, (1024, 2048), dtype=jnp.bfloat16)

# Execute NKI kernel
output_nki = nki_tensor_add(a, b)

# Compute reference output using JAX
output_jax = a + b

# Verify correctness
assert jnp.allclose(output_nki, output_jax), "NKI and JAX outputs do not match"
print("NKI and JAX match")
```

## torch-lstm-support.rst

```python
import torch
import torch_neuron

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

    def forward(self, inputs):
        output, (ht, ct) = self.lstm(inputs)
        return output, (ht, ct)
```

```python
import torch
import torch_neuron

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

    def forward(self, inputs, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths=lengths,
            enforce_sorted=True,
        )
        packed_result, (ht, ct) = self.lstm(packed_input)
        padded_result, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_result)
        return padded_result, ht, ct
```

```python
import torch
import torch_neuron

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

    def forward(self, inputs, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths=lengths,
            enforce_sorted=False,
        )
        packed_result, (ht, ct) = self.lstm(packed_input)
        padded_result, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_result)
        return padded_result, ht, ct
```

```python
import torch
import torch_neuron

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

    def forward(self, inputs, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths=lengths,
            enforce_sorted=True,
        )
        packed_output, (ht, ct) = self.lstm(packed_input)
        return ht, ct
```

## placement.py

```python
import contextlib
import torch
import torch_neuron.experimental

# Example 1: set_neuron_cores - Single Load
model = torch.jit.load('example_neuron_model.pt')
torch_neuron.experimental.set_neuron_cores(model, start_nc=0, nc_count=1)
model(example)  # Executes on NeuronCore 0

# Example 2: set_neuron_cores - Multiple Core Replication
model = torch.jit.load('example_neuron_model.pt')
torch_neuron.experimental.set_neuron_cores(model, start_nc=2, nc_count=2)
model(example)  # Executes on NeuronCore 2
model(example)  # Executes on NeuronCore 3
model(example)  # Executes on NeuronCore 2

# Example 3: set_neuron_cores - Multiple Model Load
model1 = torch.jit.load('example_neuron_model.pt')
torch_neuron.experimental.set_neuron_cores(model1, start_nc=2)
model2 = torch.jit.load('example_neuron_model.pt')
torch_neuron.experimental.set_neuron_cores(model2, start_nc=0)
model1(example)  # Executes on NeuronCore 2
model2(example)  # Executes on NeuronCore 0

# Example 4: set_multicore
model = torch.jit.load('example_neuron_model.pt')
torch_neuron.experimental.set_multicore(model)
model(example)  # Executes on NeuronCore 0
model(example)  # Executes on NeuronCore 1
model(example)  # Executes on NeuronCore 2

# Example 5: neuron_cores_context - Single Load
with torch_neuron.experimental.neuron_cores_context(start_nc=0, nc_count=1):
    model = torch.jit.load('example_neuron_model.pt')
model(example)  # Executes on NeuronCore 0

# Example 6: neuron_cores_context - Multiple Core Replication
with torch_neuron.experimental.neuron_cores_context(start_nc=2, nc_count=2):
    model = torch.jit.load('example_neuron_model.pt')
model(example)  # Executes on NeuronCore 2
model(example)  # Executes on NeuronCore 3
model(example)  # Executes on NeuronCore 2

# Example 7: neuron_cores_context - Multiple Model Load
with torch_neuron.experimental.neuron_cores_context(start_nc=2):
    model1 = torch.jit.load('example_neuron_model.pt')
with torch_neuron.experimental.neuron_cores_context(start_nc=0):
    model2 = torch.jit.load('example_neuron_model.pt')
model1(example)  # Executes on NeuronCore 2
model2(example)  # Executes on NeuronCore 0

# Example 8: multicore_context
with torch_neuron.experimental.multicore_context():
    model = torch.jit.load('example_neuron_model.pt')
model(example)  # Executes on NeuronCore 0
model(example)  # Executes on NeuronCore 1
model(example)  # Executes on NeuronCore 2
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
    // Implementation details omitted
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_in_acc[i][j] : 0.0;
}

torch::Tensor relu_backward(const torch::Tensor& t_grad, const torch::Tensor& t_in) {
    // Implementation details omitted
    t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_grad_acc[i][j] : 0.0;
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

## bert_server.py

```python
import os
import collections
import time
import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion
import pkg_resources
from threading import Lock
from multiprocessing.dummy import Pool
import grpc
from concurrent import futures
import mrpc_pb2_grpc


class BERTService(mrpc_pb2_grpc.mrpcServicer):

    def __init__(self, model_path, parallel, batch_size, bootstrap, vocab_txt, num_thread_per_predictor=2):
        num_queues = parallel * num_thread_per_predictor
        config = tf.ConfigProto(inter_op_parallelism_threads=num_queues, intra_op_parallelism_threads=1)
        tfn_version = LooseVersion(pkg_resources.get_distribution('tensorflow-neuron').version)
        if tfn_version >= LooseVersion('1.15.0.1.0.1333.0'):
            neuroncore_group_sizes = '{}x1'.format(parallel)
            predictor = tf.contrib.predictor.from_saved_model(model_path, config=config)
            self.predictor_list = [predictor for _ in range(num_queues)]
        else:
            neuroncore_group_sizes = ','.join('1' for _ in range(parallel))
            predictor_list = [tf.contrib.predictor.from_saved_model(model_path, config=config) for _ in range(parallel)]
            self.predictor_list = []
            for pred in predictor_list:
                self.predictor_list.extend(pred for _ in range(num_thread_per_predictor))
        os.environ['NEURONCORE_GROUP_SIZES'] = neuroncore_group_sizes
        if self.predictor_list[0].feed_tensors['input_ids'].shape.is_fully_defined():
            self.batch_size = self.predictor_list[0].feed_tensors['input_ids'].shape.as_list()[0]
        else:
            self.batch_size = batch_size
        self.output_name = list(self.predictor_list[0].fetch_tensors.keys())[0]
        self.result_map = {}
        self.alive = True

    def cleanup(self) -> None:
        for pred in self.predictor_list:
            pred.session.close()

    def process_input(self, idx: int) -> None:
        request_queue = self.request_queue_list[idx]
        predictor = self.predictor_list[idx]
        while self.alive:
            if len(request_queue) > 0:
                sublist = request_queue[:self.batch_size]
                request_queue[:self.batch_size] = []
                iid_list = [iid for iid, _ in sublist]
                model_feed_dict_list = [feed for _, feed in sublist]
                batch_feeds = {
                    key: np.concatenate([feed[key] for feed in model_feed_dict_list], axis=0)
                    for key in model_feed_dict_list[0].keys()
                }
                start = time.time()
                batch_predictions = predictor(batch_feeds)[self.output_name].argmax(-1)
                latency = time.time() - start
                self.result_map.update({iid: pred for iid, pred in zip(iid_list, batch_predictions)})
            time.sleep(0.001)

    def get_output(self, iid: int) -> int:
        while iid not in self.result_map:
            time.sleep(0.001)
        return self.result_map.pop(iid)
```

## activation_memory_reduction_developer_guide.rst

```python
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

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
setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)
```

```python
import torch
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region

def allreduce_sequence_parallel_gradients(optimizer):
    """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
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
```

```python
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

## training_llama_tp_pp.rst

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
sudo rm -rf /home/ubuntu/.cache/
```

```python
pip install -U datasets
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
class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())
```

```python
# Compile text encoder
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
# Compile vae decoder
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
# Compile unet
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
# Compile vae post_quant_conv
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
# Compile safety checker
safety_model = torch_neuronx.trace(
    safety_model, 
    clip_input,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model'),
    compiler_args=["--enable-fast-loading-neuron-binaries"]
)
torch_neuronx.async_load(safety_model)
torch.jit.save(safety_model, safety_model_neuron_filename)
```

## tf_neuron_check_model.py

```python
import os
import json
import sys
import struct
import argparse
import subprocess
from collections import Counter

class neuron_parser:
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('model_path', type=str, help='a TensorFlow SavedModel directory (currently supporting TensorFlow v1 SaveModel only).')
    self.parser.add_argument('--show_names', action='store_true', help='list operation by name instead of summarizing by type (caution: this option will generate many lines of output for a large model).')
    self.parser.add_argument('--expand_subgraph', action='store_true', help='show subgraph operations.')
    self.parser_args = self.parser.parse_args()
    self.neuronop_info = {}
    self.total_pipeline_cores = 0
    self.min_required_pipeline_cores = 0
    path = self.parser_args.model_path
    if os.path.exists(path + '-symbol.json'):
      self.load_mxnet_model(path)
    elif os.path.isdir(path):
      self.load_tensorflow_model(path)
    else:
      raise RuntimeError('Cannot determine framework type from model path argument.')
    self.supported = self.get_neuron_supported()
    self.supported.extend(self.addl_support)
    for name, executable, (sg_nodetypes, sg_nodenames) in self.neuron_nodes:
      num_cores, requested_cores, _ = self.get_cores_from_executable(executable)
      self.neuronop_info[name] = (num_cores, requested_cores, sg_nodetypes, sg_nodenames)
      self.total_pipeline_cores += num_cores
      if num_cores > self.min_required_pipeline_cores:
          self.min_required_pipeline_cores = num_cores

  def get_neuron_supported(self):
    exec_cmd = ["neuron-cc", "list-operators", "--framework", self.framework]
    oplist = subprocess.check_output(' '.join(exec_cmd), shell=True)
    oplist = str(oplist, 'utf-8')
    oplist = oplist.split("\n")
    return oplist[:-1]

  def get_tf_subgraph_types_names(self, node):
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(node.attr['graph_def'].s)
    sg_nodes = graph_def.node
    sg_nodes = [sg_node for sg_node in sg_nodes if sg_node.op not in self.excl_types]
    nodetypes = [sg_node.op for sg_node in sg_nodes]
    nodenames = [sg_node.name for sg_node in sg_nodes]
    return nodetypes, nodenames

  def load_tensorflow_model(self, path):
    import tensorflow as tf
    import tensorflow_hub as hub
    self.framework = 'TENSORFLOW'
    self.neuron_optype = "NeuronOp"
    self.excl_types = ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp', 'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2', 'MergeV2Checkpoints', 'RestoreV2']
    self.addl_support = ['FusedBatchNormV3', 'BatchMatMulV2', 'AddV2', 'StopGradient', self.neuron_optype]
    model = hub.load(path)
    graph_def = model.graph.as_graph_def()
    nodes = graph_def.node
    nodes = [node for node in nodes if node.op not in self.excl_types]
    self.nodetypes = [node.op for node in nodes]
    self.nodenames = [node.name for node in nodes]
    self.neuron_nodes = [(node.name, node.attr['executable'].s, self.get_tf_subgraph_types_names(node)) for node in nodes if node.op == self.neuron_optype]

  def get_mx_subgraph_types_names(self, node):
    nodetypes = []
    nodenames = []
    for sg in node['subgraphs']:
      filtered_nodes = [sg_node for sg_node in sg['nodes'] if sg_node['op'] not in self.excl_types]
      nodetypes.extend([sg_node['op'] for sg_node in filtered_nodes])
      nodenames.extend([sg_node['name'] for sg_node in filtered_nodes])
    return nodetypes, nodenames

  def load_mxnet_model(self, path):      
    import mxnet as mx
    if mx.__version__ != "1.5.1":
      try:
        import mxnetneuron as mxn
      except:
        raise "Please install mxnetneuron package."
    self.framework = 'MXNET'
    self.neuron_optype = "_neuron_subgraph_op"
    self.excl_types = ['null']
    self.addl_support = [self.neuron_optype]
    sym, args, auxs = mx.model.load_checkpoint(path, 0)
    nodes = json.loads(sym.tojson())["nodes"]
    nodes = [node for node in nodes if node['op'] not in self.excl_types]
    self.nodetypes = [node['op'] for node in nodes]
    self.nodenames = [node['name'] for node in nodes]
    neuron_nodes_tmp = [node for node in nodes if node['op'] == self.neuron_optype]
    self.neuron_nodes = [(node['name'], bytearray(args[node['name']+"_neuronbin"].asnumpy()), self.get_mx_subgraph_types_names(node)) for node in neuron_nodes_tmp]

  @staticmethod
  def get_cores_from_executable(executable):
    _NC_HEADER_SIZE = 544
    header = executable[:_NC_HEADER_SIZE]
    info = list(struct.unpack('168xI304xI64B', header))
    numCores = info.pop(0)
    numCoresRequested = info.pop(0)
    coresPerNode = info
    return numCores, numCoresRequested, coresPerNode
```

## mx_neuron_check_model.py

```python
import os
import json
import sys
import struct
import argparse
import subprocess
from collections import Counter

class neuron_parser:
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('model_path', type=str, help='path prefix to MXNet model (the part before -symbol.json).')
    self.parser.add_argument('--show_names', action='store_true', help='list operation by name instead of summarizing by type (caution: this option will generate many lines of output for a large model).')
    self.parser.add_argument('--expand_subgraph', action='store_true', help='show subgraph operations.')
    self.parser_args = self.parser.parse_args()
    self.neuronop_info = {}
    self.total_pipeline_cores = 0
    self.min_required_pipeline_cores = 0
    path = self.parser_args.model_path

    if os.path.exists(path + '-symbol.json'):
      self.load_mxnet_model(path)
    elif os.path.isdir(path):
      self.load_tensorflow_model(path)
    else:
      raise RuntimeError('Cannot determine framework type from model path argument.')
    self.supported = self.get_neuron_supported()
    self.supported.extend(self.addl_support)
    for name, executable, (sg_nodetypes, sg_nodenames) in self.neuron_nodes:
      num_cores, requested_cores, _ = self.get_cores_from_executable(executable)
      self.neuronop_info[name] = (num_cores, requested_cores, sg_nodetypes, sg_nodenames)
      self.total_pipeline_cores += num_cores
      if num_cores > self.min_required_pipeline_cores:
          self.min_required_pipeline_cores = num_cores

  def get_neuron_supported(self):
    exec_cmd = ["neuron-cc", "list-operators", "--framework", self.framework]
    oplist = subprocess.check_output(' '.join(exec_cmd), shell=True)
    oplist = str(oplist, 'utf-8')
    oplist = oplist.split("\n")
    return oplist[:-1]

  def get_tf_subgraph_types_names(self, node):
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(node.attr['graph_def'].s)
    sg_nodes = graph_def.node
    sg_nodes = [sg_node for sg_node in sg_nodes if sg_node.op not in self.excl_types]
    nodetypes = [sg_node.op for sg_node in sg_nodes]
    nodenames = [sg_node.name for sg_node in sg_nodes]
    return nodetypes, nodenames

  def load_tensorflow_model(self, path):
    import tensorflow as tf
    import tensorflow_hub as hub
    self.framework = 'TENSORFLOW'
    self.neuron_optype = "NeuronOp"
    self.excl_types = ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp', 'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2', 'MergeV2Checkpoints', 'RestoreV2']
    self.addl_support = ['FusedBatchNormV3', 'BatchMatMulV2', 'AddV2', 'StopGradient', self.neuron_optype]
    model = hub.load(path)
    graph_def = model.graph.as_graph_def()
    nodes = graph_def.node
    nodes = [node for node in nodes if node.op not in self.excl_types]
    self.nodetypes = [node.op for node in nodes]
    self.nodenames = [node.name for node in nodes]
    self.neuron_nodes = [(node.name, node.attr['executable'].s, self.get_tf_subgraph_types_names(node)) for node in nodes if node.op == self.neuron_optype]

  def get_mx_subgraph_types_names(self, node):
    nodetypes = []
    nodenames = []
    for sg in node['subgraphs']:
      filtered_nodes = [sg_node for sg_node in sg['nodes'] if sg_node['op'] not in self.excl_types]
      nodetypes.extend([sg_node['op'] for sg_node in filtered_nodes])
      nodenames.extend([sg_node['name'] for sg_node in filtered_nodes])
    return nodetypes, nodenames

  def load_mxnet_model(self, path):      
    import mxnet as mx
    if mx.__version__ != "1.5.1":
      try:
        import mx_neuron as mxn
      except:
        raise "Please install mxnetneuron package."
    self.framework = 'MXNET'
    self.neuron_optype = "_neuron_subgraph_op"
    self.excl_types = ['null']
    self.addl_support = [self.neuron_optype]
    sym, args, auxs = mx.model.load_checkpoint(path, 0)
    nodes = json.loads(sym.tojson())["nodes"]
    nodes = [node for node in nodes if node['op'] not in self.excl_types]
    self.nodetypes = [node['op'] for node in nodes]
    self.nodenames = [node['name'] for node in nodes]
    neuron_nodes_tmp = [node for node in nodes if node['op'] == self.neuron_optype]
    self.neuron_nodes = [(node['name'], bytearray(args[node['name']+"_neuronbin"].asnumpy()), self.get_mx_subgraph_types_names(node)) for node in neuron_nodes_tmp]

  @staticmethod
  def get_cores_from_executable(executable):
    _NC_HEADER_SIZE = 544
    header = executable[:_NC_HEADER_SIZE]
    info = list(struct.unpack('168xI304xI64B', header))
    numCores = info.pop(0)
    numCoresRequested = info.pop(0)
    coresPerNode = info
    return numCores, numCoresRequested, coresPerNode
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
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "utils.hpp"
#include "core_count.hpp"
#include "../tokenizers_binding/remote_rust_tokenizer.h"

typedef std::vector<std::vector<long>> Input;

// construct a single input: input_ids, attention_mask, and token_type_ids from two input sentences
Input get_input(const std::string& sentence_1, const std::string& sentence_2)
{
    const size_t seq_len = 128;
    
    // ensure the concatenated sentences + separator tokens do not exceed the compiled sequence length
    assert(sentence_1.size() + sentence_2.size() + 3 <= seq_len);

    const long start_token = 101;
    const long end_token = 102;

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

int sanity_check(const std::string& model_filename)
{
    // load the model
    auto model = get_model(model_filename);

    // construct some example inputs
    const std::string sentence_1 = "The company HuggingFace is based in New York City";
    const std::string sentence_2 = "Apples are especially bad for your health";
    const std::string sentence_3 = "HuggingFace's headquarters are situated in Manhattan";
    const auto paraphrase = get_input(sentence_1, sentence_3);
    const auto not_paraphrase = get_input(sentence_1, sentence_2);

    const size_t batch_size = 6;

    // batch the inputs 50/50 positive/negative
    std::vector<Input> inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (i < batch_size / 2) {
            inputs[i] = paraphrase;
        } else {
            inputs[i] = not_paraphrase;
        }
    }
    const auto batch = get_batch(inputs);

    // forward pass
    const auto output = model.forward(batch);

    // interpret output
    const auto output_tensor = output.toTuple()->elements()[0].toTensor();
    const auto paraphrase_probabilities = torch::softmax(output_tensor[0], 0);
    const auto not_paraphrase_probabilities = torch::softmax(output_tensor[batch_size-1], 0);
    const auto paraphrase_0 = std::round(paraphrase_probabilities[0].item<double>() * 100);
    const auto paraphrase_1 = std::round(paraphrase_probabilities[1].item<double>() * 100);
    const auto not_paraphrase_0 = std::round(not_paraphrase_probabilities[0].item<double>() * 100);
    const auto not_paraphrase_1 = std::round(not_paraphrase_probabilities[1].item<double>() * 100);

    if (paraphrase_0 >= paraphrase_1) return -1;
    if (not_paraphrase_0 <= not_paraphrase_1) return -2;

    return 0;
}
```

## layernorm.rst

```python
import neuron_nki as nki
import neuron_nki.language as nl
import neuron_nki.isa as nisa
import math

# Version 1: nki.language APIs only
def nki_layernorm_kernel_v1(
    input_tensor: nl.ndarray,
    gamma: nl.ndarray,
    beta: nl.ndarray,
    epsilon: float,
) -> nl.ndarray:
    """
    LayerNorm kernel using nki.language APIs.
    
    Args:
        input_tensor: Input tensor of shape (sequence_length, hidden_size)
        gamma: Affine transform parameter of shape (hidden_size,)
        beta: Affine transform parameter of shape (hidden_size,)
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
        for i_p_io in range(nl.tile_size.pmax):
            # Load one tile of input_tensor
            input_tile = nl.load(
                input_tensor,
                indices=(i * nl.tile_size.pmax + i_p_io, 0),
                mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
            )
            
            # Compute mean and variance
            mean = nl.mean(input_tile, axis=1)
            variance = nl.var(input_tile, axis=1)
            
            # Normalize
            normalized = (input_tile - mean) / nl.rsqrt(variance + epsilon)
            
            # Scale and shift
            output_tile = normalized * shift_scale_tensor + beta_tensor
            
            # Store output
            nl.store(
                output_tensor,
                indices=(i * nl.tile_size.pmax + i_p_io, 0),
                value=output_tile,
                mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
            )
    
    return output_tensor


# Version 2: nki.isa APIs for optimized mean/variance and shift/scale
def nki_layernorm_kernel_v2(
    input_tensor: nl.ndarray,
    gamma: nl.ndarray,
    beta: nl.ndarray,
    epsilon: float,
) -> nl.ndarray:
    """
    Optimized LayerNorm kernel using nki.isa APIs.
    
    Args:
        input_tensor: Input tensor of shape (sequence_length, hidden_size)
        gamma: Affine transform parameter of shape (hidden_size,)
        beta: Affine transform parameter of shape (hidden_size,)
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
        for i_p_io in range(nl.tile_size.pmax):
            # Load one tile of input_tensor
            input_tile = nl.load(
                input_tensor,
                indices=(i * nl.tile_size.pmax + i_p_io, 0),
                mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
            )
            
            # Compute mean and variance using bn_stats and bn_aggr
            mean = nl.zeros((nl.tile_size.pmax,), dtype=input_tensor.dtype)
            variance = nl.zeros((nl.tile_size.pmax,), dtype=input_tensor.dtype)
            
            for j in range(math.ceil(input_tensor.shape[1] / nl.tile_size.bn_stats_fmax)):
                start_idx = j * nl.tile_size.bn_stats_fmax
                end_idx = min(start_idx + nl.tile_size.bn_stats_fmax, input_tensor.shape[1])
                
                input_slice = input_tile[:, start_idx:end_idx]
                
                # Use bn_stats to compute statistics
                stats = nisa.bn_stats(input_slice)
                mean, variance = nisa.bn_aggr(stats, mean, variance)
            
            # Perform shift and scale using tensor_scalar
            normalized = nisa.tensor_scalar(
                input_tile,
                mean,
                variance,
                epsilon,
                gamma_loaded,
                beta_loaded
            )
            
            # Store output
            nl.store(
                output_tensor,
                indices=(i * nl.tile_size.pmax + i_p_io, 0),
                value=normalized,
                mask=(i * nl.tile_size.pmax + i_p_io < input_tensor.shape[0])
            )
    
    return output_tensor
```

```python
import torch
import torch.nn.functional as F

def pytorch_layernorm(input_tensor, gamma, beta, epsilon=1e-5):
    """
    PyTorch reference implementation of LayerNorm.
    
    Args:
        input_tensor: Input tensor of shape (sequence_length, hidden_size)
        gamma: Affine transform parameter of shape (hidden_size,)
        beta: Affine transform parameter of shape (hidden_size,)
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized output tensor of same shape as input_tensor
    """
    mean = input_tensor.mean(dim=1, keepdim=True)
    variance = input_tensor.var(dim=1, keepdim=True, unbiased=False)
    normalized = (input_tensor - mean) / torch.sqrt(variance + epsilon)
    output = normalized * gamma + beta
    return output
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
import torch_neuronx
import os
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
import copy

# Have to do this double wrapper trick to compile the unet, because
# of the special UNet2DConditionOutput output type.
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

## mrpc_feature.py

```python
import csv
import numpy as np
import tokenization


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
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

## ptl_developer_guide.rst

```python
from lightning.pytorch import LightningModule
from neuronx_distributed.lightning import NeuronLTModule
import torch_xla.core.xla_model as xm
from neuronx_distributed import parallel_state

class NeuronLlamaLTModule(NeuronLTModule):
    def training_step(self, batch, batch_idx):
        xm.mark_step()
        for logger in self.trainer.loggers:
            logger.print_step = -1
        self.should_print = False
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / self.grad_accum_steps
        loss.backward()
        self.averaged_loss += loss.detach()
        xm.mark_step()
        if not self.automatic_optimization and (batch_idx + 1) % self.grad_accum_steps == 0:
            self.should_print = True
            loss_div = self.averaged_loss / self.trainer.strategy.data_parallel_size
            loss_reduced = xm.all_reduce(
                xm.REDUCE_SUM,
                loss_div,
                groups=parallel_state.get_data_parallel_group(as_list=True),
            )
            loss_reduced_detached = loss_reduced.detach()
            self.averaged_loss.zero_()
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            xm.mark_step()
            self.loss = loss_reduced_detached
        return loss

    def configure_optimizers(self):
        param_groups = self.get_param_groups_by_weight_decay()
        optimizer = initialize_parallel_optimizer(
            self.nxd_config, self.opt_cls, param_groups, **self.opt_kwargs
        )
        optimizer.zero_grad()
        scheduler = self.scheduler_cls(optimizer, *self.scheduler_args, **self.scheduler_kwargs)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                }
            ],
        )

    def on_train_batch_end(self, *args, **kwargs):
        if self.should_print:
            if not self.automatic_optimization:
                self.log(
                    "loss",
                    self.loss.detach().cpu().item() if self.loss is not None else torch.zeros(1, device="cpu", requires_grad=False),
                    prog_bar=True,
                )
                self.log(
                    "global_step",
                    self.global_step,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                )
                for logger in self.trainer.loggers:
                    logger.print_step = self.global_step

    def get_param_groups_by_weight_decay(self):
        """Get param groups. Customers can override this to have their own way of weight_decay"""
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
```

```python
from lightning.pytorch.core.datamodule import LightningDataModule
from typing import Callable, Tuple, Dict

class NeuronLightningDataModule(LightningDataModule):
    def __init__(
        self, 
        dataloader_fn: Callable,
        data_dir: str, 
        batch_size: int,
        data_args: Tuple = (), 
        data_kwargs: Dict = {},
    ):
        super().__init__()
        self.dataloader_fn = dataloader_fn
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_args = data_args
        self.data_kwargs = data_kwargs

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return self.dataloader_fn(
            self.data_dir,
            self.batch_size,
            self.trainer.strategy.data_parallel_size,
            self.trainer.strategy.data_parallel_rank,
            *self.data_args,
            **self.data_kwargs
        )
```

```python
from lightning.pytorch import Trainer
from neuronx_distributed.lightning import NeuronXLAStrategy, NeuronXLAPrecisionPlugin, NeuronTQDMProgressBar, NeuronTensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

model = NeuronLlamaLTModule(
    model_fn=LlamaForCausalLM,
    nxd_config=nxd_config,
    model_args=(model_config,),
    opt_cls=optimizer_cls,
    scheduler_cls=configure_scheduler,
    opt_kwargs={
        "lr": flags.lr,
    },
    scheduler_args=(flags.warmup_steps, flags.max_steps),
    grad_accum_steps=flags.grad_accum_usteps,
    manual_opt=True,
)

dm = NeuronLightningDataModule(
    create_llama_pretraining_dataset,
    flags.data_dir,
    flags.batch_size,
    data_args=(flags.seed,),
)

strategy = NeuronXLAStrategy(
    nxd_config=nxd_config
)
plugins = []
plugins.append(NeuronXLAPrecisionPlugin())
callbacks = []
callbacks.append(NeuronTQDMProgressBar())

callbacks.append(
    ModelCheckpoint(
        save_top_k=flags.num_kept_checkpoint,
        monitor="global_step",
        mode="max",
        every_n_train_steps=flags.checkpoint_freq,
        dirpath=flags.checkpoint_dir,
    )
)

trainer = Trainer(
    strategy=strategy,
    max_steps=flags.steps_this_run,
    plugins=plugins,
    enable_checkpointing=flags.save_checkpoint,
    logger=NeuronTensorBoardLogger(save_dir=flags.log_dir),
    log_every_n_steps=1,
    callbacks=callbacks,
)
trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)
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
```

```python
# Compile text encoder with torch_neuronx.trace
text_encoder_neuron = torch_neuronx.trace(
    text_encoder.neuron_text_encoder,
    emb,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
)
torch.jit.save(text_encoder_neuron, text_encoder_filename)
```

```python
# Compile VAE decoder with torch_neuronx.trace
decoder_in = torch.randn([1, 4, 128, 128])
decoder_neuron = torch_neuronx.trace(
    decoder,
    decoder_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)
torch.jit.save(decoder_neuron, decoder_filename)
```

```python
# Compile UNet with torch_neuronx.trace and compiler arguments
sample_1b = torch.randn([1, 7, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 1024])
class_labels = torch.tensor([20])
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b, class_labels

unet_neuron = torch_neuronx.trace(
    unet,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
    compiler_args=["--model-type=unet-inference"]
)
torch.jit.save(unet_neuron, unet_filename)
```

## customop-mlp-perf-opt.rst

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import my_ops

# Declare 3-layer MLP for MNIST dataset                                                                
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

```cpp
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

```cpp
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

        t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, partition *cpu_id + i, copy_size);
        for (size_t j = 0; j < copy_size; j++) {
            tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
        }
        t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, partition *cpu_id + i, copy_size);
        }
    }
    torch::neuron::tcm_free(tcm_buffer);
    return t_out;
}
```

## finetuning_llama2_7b_ptl.rst

```python
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
torch.save(model.state_dict(), "llama-7b-hf-pretrained.pt")
```

## finetuning_llama3_8B_ptl_lora.rst

```python
import neuronx_distributed as nxd

nxd.save_checkpoint(
    checkpoint_dir_str="lora_checkpoint", 
    tag="lora", 
    model=model
)
```

```python
from peft import LoraConfig

target_modules = ["q_proj", "v_proj", "k_proj"] if flags.qkv_linear == 0 else ["qkv_proj"]      
lora_config = LoraConfig(
    enable_lora=flags.enable_lora,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    lora_verbose=True,
    target_modules=target_modules,
)
```

```python
from peft import LoraConfig

lora_config = LoraConfig(
    enable_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    lora_verbose=True,
    target_modules=target_modules,
    save_lora_base=False,
    merge_lora=False,
    save_lora_config_adapter=True,
)
```

## layernorm_nki_kernel.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import math
```

```python
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

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)
```

```python
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
```

```python
def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled
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
import copy
from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

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

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled

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
```

```python
# Compile UNet
unet = copy.deepcopy(pipe.unet.unetwrap)

sample_1b = torch.randn([1, 4, 96, 96], dtype=DTYPE)
timestep_1b = torch.tensor(999, dtype=DTYPE).expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 1024], dtype=DTYPE)
example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

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
# Compile text encoder
emb = torch.tensor([[49406, 18376, 525, 7496, 49407, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0]])

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
# Compile VAE decoder
decoder_in = torch.randn([1, 4, 96, 96], dtype=DTYPE)

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
# Compile VAE post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 96, 96], dtype=DTYPE)

post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

torch_neuronx.async_load(post_quant_conv_neuron)

torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
```

## migration-from-xla-downcast-bf16.rst

```python
import os
import torch

# Full BF16 with stochastic rounding enabled
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
model.to(torch.bfloat16)
```

```python
import torch

# Keep loss in FP32
running_loss = torch.zeros(1, dtype=torch.float).to(device)
```

```python
import torch

# Convert gradients to FP32 before optimizer computations
grad = p.grad.data.float()
```

```python
import torch

# BF16 in GPU-compatible mode: make FP32 copy of weights in optimizer initializer
self.param_groups_highprec = []
for group in self.param_groups:
    params = group['params']
    param_groups_highprec = [p.data.float() for p in params]
    self.param_groups_highprec.append({'params': param_groups_highprec})
```

```python
import torch

# Update FP32 copy of weights in optimizer step
for group, group_highprec in zip(self.param_groups, self.param_groups_highprec):
    for p, p_highprec in zip(group['params'], group_highprec['params']):
        grad = p.grad.data.float()
        # compute the exponential average and denominator using grad
        # ...
        p_highprec.data.addcdiv_(exponential_avg, denominator, value=-step_size)
```

```python
import os

# BF16 automatic mixed precision: disable compiler auto-cast
os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
```

```python
import torch

# BF16 autocast in training loop
def train_loop_fn(train_loader):
    for i, data in enumerate(train_loader):
        torch.cuda.is_bf16_supported = lambda: True
        with torch.autocast(dtype=torch.bfloat16, device_type='xla'):
            inputs = data[0]
            labels = data[3]
            outputs = model(inputs, labels=labels)
        loss = outputs.loss / flags.grad_acc_steps
        loss.backward()
        optimizer.step()
        xm.mark_step()
```

## zero1_gpt2.rst

```python
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
import torch
import torch_xla.core.xla_model as xm
from torch.optim import AdamW

# Example 1: Basic ZeRO-1 optimizer setup
device = xm.xla_device()
model = model.to(device)

optimizer = ZeroRedundancyOptimizer(model.parameters(), AdamW, lr=0.001)
```

```python
# Example 2: Training loop with ZeRO-1
loss.backward()
xm.mark_step()
optimizer.step()
xm.mark_step()
```

```python
# Example 3: ZeRO-1 optimizer with advanced configuration
optimizer = ZeroRedundancyOptimizer(
    model.parameters(),
    AdamW,
    lr=0.001,
    optimizer_dtype=torch.float32,
    grad_clipping=True,
    max_norm=1.0,
    use_grad_acc_hook=True,
    higher_cc_precision=False
)
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
from diffusers import StableDiffusionInpaintPipeline
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
device_ids = [0, 1]
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
    compiler_workdir='unet_base',
    compiler_args=["--model-type=unet-inference"]
)

# Enable asynchronous and lazy loading
torch_neuronx.async_load(unet_neuron)
torch_neuronx.lazy_load(unet_neuron)

# Save compiled model
torch.jit.save(unet_neuron, 'unet_base/model.pt')


# Compile VAE decoder
decoder_in = torch.randn([1, 4, 128, 128], dtype=torch.float32)
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir='vae_decoder'
)

torch_neuronx.async_load(decoder_neuron)
torch.jit.save(decoder_neuron, 'vae_decoder/model.pt')


# Compile VAE post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=torch.float32)
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir='vae_post_quant_conv',
)

torch_neuronx.async_load(post_quant_conv_neuron)
torch.jit.save(post_quant_conv_neuron, 'vae_post_quant_conv/model.pt')
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

# Load pipeline and compiled models
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

# Load the compiled UNet onto two neuron cores with DataParallel
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load(text_encoder_filename), pipe.text_encoder)
pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load(text_encoder_2_filename), pipe.text_encoder_2)
```

## yolo_v4_demo.rst

```python
import tensorflow as tf
from functools import partial

def decode_jpeg_resize(input_tensor, image_size):
    tensor = tf.image.decode_png(input_tensor, channels=3)
    shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.resize(tensor, image_size)
    tensor /= 255.0
    return tf.cast(tensor, tf.float16), shape

def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        tensor = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float16, tf.int32), back_prop=False, parallel_iterations=16)
    return tensor
```

```python
def filter_boxes_one_size(boxes, box_scores, conf_thresh):
    box_class_scores = tf.reduce_max(box_scores, axis=-1)
    keep = box_class_scores > conf_thresh
    boxes = boxes[keep]
    box_scores = box_scores[keep]
    return boxes, box_scores

def filter_boxes(outputs, conf_thresh, nms_thresh, nms_top_k):
    boxes_l, boxes_m, boxes_s, box_scores_l, box_scores_m, box_scores_s, image_shape = outputs
    boxes_l, box_scores_l = filter_boxes_one_size(boxes_l, box_scores_l, conf_thresh)
    boxes_m, box_scores_m = filter_boxes_one_size(boxes_m, box_scores_m, conf_thresh)
    boxes_s, box_scores_s = filter_boxes_one_size(boxes_s, box_scores_s, conf_thresh)
    boxes = tf.concat([boxes_l, boxes_m, boxes_s], axis=0)
    box_scores = tf.concat([box_scores_l, box_scores_m, box_scores_s], axis=0)
    image_shape_wh = image_shape[1::-1]
    image_shape_whwh = tf.concat([image_shape_wh, image_shape_wh], axis=-1)
    image_shape_whwh = tf.cast(image_shape_whwh, tf.float32)
    boxes *= image_shape_whwh
    boxes = tf.expand_dims(boxes, 0)
    box_scores = tf.expand_dims(box_scores, 0)
    boxes = tf.expand_dims(boxes, 2)
    nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes,
        box_scores,
        max_output_size_per_class=nms_top_k,
        max_total_size=nms_top_k,
        iou_threshold=nms_thresh,
        score_threshold=conf_thresh,
        pad_per_class=False,
        clip_boxes=False,
        name='CombinedNonMaxSuppression',
    )
    return nms_boxes[0], nms_scores[0], nms_classes[0]

def batch_yolo_out(outputs, anchors, masks, conf_thresh, nms_thresh, nms_top_k, batch_process_feats):
    with tf.name_scope('yolo_out'):
        b_output_lr, b_output_mr, b_output_sr, b_image_shape = outputs
        with tf.name_scope('process_feats'):
            b_boxes_l, b_box_scores_l = batch_process_feats(b_output_lr, anchors, masks[0])
        with tf.name_scope('process_feats'):
            b_boxes_m, b_box_scores_m = batch_process_feats(b_output_mr, anchors, masks[1])
        with tf.name_scope('process_feats'):
            b_boxes_s, b_box_scores_s = batch_process_feats(b_output_sr, anchors, masks[2])
        with tf.name_scope('filter_boxes'):
            b_nms_boxes, b_nms_scores, b_nms_classes = tf.map_fn(
                lambda x: filter_boxes(x, conf_thresh, nms_thresh, nms_top_k),
                [b_boxes_l, b_boxes_m, b_boxes_s, b_box_scores_l, b_box_scores_m, b_box_scores_s, b_image_shape],
                dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return b_nms_boxes, b_nms_scores, b_nms_classes
```

## sdxl_base_and_refiner_1024_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx

from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

# Define datatype
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


# Load base pipeline
pipe_base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, low_cpu_mem_usage=True)

# Load the compiled UNet onto two neuron cores using DataParallel
pipe_base.unet = NeuronUNet(UNetWrap(pipe_base.unet))
device_ids = [0, 1]
pipe_base.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load("unet_base/model.pt"), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core
pipe_base.vae.decoder = torch.jit.load("vae_decoder/model.pt")
pipe_base.vae.post_quant_conv = torch.jit.load("vae_post_quant_conv/model.pt")

# Load refiner pipeline sharing components with base
pipe_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe_base.text_encoder_2,
    vae=pipe_base.vae,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Load refiner UNet onto two neuron cores
pipe_refiner.unet = NeuronUNet(UNetWrap(pipe_refiner.unet))
device_ids = [0, 1]
pipe_refiner.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load("unet_refiner/model.pt"), device_ids, set_dynamic_batching=False)
```

## torch.py

```python
import functools
import itertools
import logging
import math
import os
import types

import torch
from .. import benchmarking


def _compile_fn(model, example_inputs, models_dir, model_name, **kwargs):
    import torch_neuron

    """Compiles a model for Neuron."""
    model_filename = os.path.join(models_dir, "{}.pt".format(model_name))
    model.eval()

    # NeuronPerf provides compiler_args as a dictionary, but framework expects a different format.
    compiler_args = kwargs.get("compiler_args", {})
    compiler_args_flattened = list(itertools.chain.from_iterable(compiler_args.items()))
    kwargs["compiler_args"] = compiler_args_flattened

    model_neuron = torch.neuron.trace(
        model,
        example_inputs,
        **kwargs,
    )
    model_neuron.save(model_filename)
    return model_filename


def _load_fn(model_filename, **kwargs):
    import torch_neuron

    model = torch.jit.load(model_filename)
    model.eval()
    return model


def _class_load_fn(model_class, **kwargs):
    model = model_class()
    model.eval()
    return model


def compile(model, inputs, *args, **kwargs):
    return benchmarking.compile(_compile_fn, model, inputs, *args, **kwargs)


def _get_dataset_loader_fn(dataset, loop):
    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        dataset = worker_info.dataset
        per_worker = int(math.ceil(len(dataset) / float(num_workers)))
        start = worker_id * per_worker
        end = min(start + per_worker, len(dataset))

        def _iter(self, start, end, loop):
            if loop:
                return itertools.cycle(range(start, end))
            else:
                return iter(range(start, end))

        __iter__ = functools.partial(_iter, start, end, loop)
        dataset.__iter__ = types.MethodType(__iter__, dataset)

    def dataset_loader_fn(dataset, num_workers):
        return iter(
            torch.utils.data.DataLoader(
                dataset, num_workers=num_workers, worker_init_fn=_worker_init_fn
            )
        )

    return dataset_loader_fn


def benchmark(model_filename, inputs, *args, dataset_inputs=False, loop_dataset=False, **kwargs):
    load_fn = _load_fn
    setup_fn = kwargs.get("setup_fn", lambda *args, **kwargs: None)
    preprocess_fn = kwargs.get("preprocess_fn", lambda *args: (*args,))

    device_type = kwargs.get("device_type", None)
    use_cuda = device_type and ("cuda" in device_type.lower() or "gpu" == device_type.lower())
    if use_cuda:
        if not torch.cuda.is_available():
            raise ValueError(
                "You requested CUDA benchmarking, but torch is unable to locate a CUDA device."
            )

        if "multiinterpreter" in kwargs and not kwargs["multiinterpreter"]:
            kwargs["multiinterpreter"] = True

        if not isinstance(model_filename, str):
            model_class = model_filename
            if not isinstance(model_class, type):
                raise TypeError("GPU benchmarking expects a model class to be provided instead of a filename.")

            import inspect

            try:
                model_class_file = inspect.getfile(model_class)
                kwargs["model_class_file"] = model_class_file
                kwargs["model_class_name"] = model_class.__name__
            except:
                raise ValueError(
                    (
                        "Your model class must be defined in a Python module so that it can be serialized properly.\n"
                        "Please add your model to a simple Python file along with any required imports."
                    )
                )

            @functools.wraps(_class_load_fn)
            def load_fn(*args, **kwargs):
                return _class_load_fn(model_class, **kwargs)

            model_filename = model_class.__name__

        @functools.wraps(setup_fn)
        def _setup_fn(id, config, model):
            setup_fn(id, config, model)
            model.to("cuda")

        kwargs["setup_fn"] = _setup_fn

        @functools.wraps(preprocess_fn)
        def _preprocess_fn(*inputs):
            inputs = preprocess_fn(*inputs)
            for input in inputs:
                input.to("cuda")
            return (*inputs,)

        kwargs["preprocess_fn"] = _preprocess_fn

    dataset_loader_fn = None
    if dataset_inputs:
        dataset_loader_fn = _get_dataset_loader_fn(example_inputs, loop_dataset)
    kwargs["dataset_loader_fn"] = dataset_loader_fn

    with torch.no_grad():
        return benchmarking.benchmark(
            load_fn,
            model_filename,
            inputs,
            *args,
            **kwargs,
        )
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

def get_pipe(resolution, dtype):
  if resolution == 256:
    transformer = Transformer2DModel.from_pretrained(
      "PixArt-alpha/PixArt-Sigma-XL-2-256x256", 
      subfolder='transformer', 
      torch_dtype=dtype,
    )
    return PixArtSigmaPipeline.from_pretrained(
      "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
      transformer=transformer,
      torch_dtype=dtype,
    )
  elif resolution == 512:
    transformer = Transformer2DModel.from_pretrained(
      "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
      subfolder='transformer', 
      torch_dtype=dtype,
    )
    return PixArtSigmaPipeline.from_pretrained(
      "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
      transformer=transformer,
      torch_dtype=dtype,
    )
  else:
    raise Exception(f"Unsupport resolution {resolution} for PixArt Sigma")

# Load pipeline and wrap compiled models
pipe = get_pipe(256, DTYPE)
seqlen = 300

_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load('pixart_sigma_compile_dir/text_encoder/model.pt')
pipe.text_encoder = _neuronTextEncoder

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(
    torch.jit.load('pixart_sigma_compile_dir/transformer/model.pt'), 
    device_ids, 
    set_dynamic_batching=False
)
pipe.transformer = _neuronTransformer

pipe.vae.decoder = SimpleWrapper(torch.jit.load('pixart_sigma_compile_dir/vae_decoder/model.pt'))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load('pixart_sigma_compile_dir/vae_post_quant_conv/model.pt'))
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

npf.list_metrics()
```

```python
import neuronperf as npf

npf.register_metric_from_existing("topk", "topk_3", k=3)
```

```python
import neuronperf as npf

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

# Define datatype
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

# Load compiled models and integrate with pipeline
_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(
    torch.jit.load(transformer_filename), 
    device_ids, 
    set_dynamic_batching=False
)
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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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

# Load compiled models
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=DTYPE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

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
  """
  Example 1: Basic usage of select_reduce
  Create source data, predicate, and destination tensors
  """
  # Create output tensor for result
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation - copy from on_true where predicate is true
  # and set to fp32.min where predicate is false
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
  )
  
  # Store result to HBM
  nl.store(result_tensor, value=dst)

  return result_tensor
```

```python
@nki.jit(mode="simulation")
def nki_select_reduce_with_reduction(predicate_data, on_true_data, on_false_data):
  """
  Example 2: Using select_reduce with reduction
  Perform selection and compute max reduction per partition
  """
  # Create output tensors for results
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  reduce_tensor = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data)
  on_true = nl.load(on_true_data)
  on_false = nl.load(on_false_data)

  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Create tensor for reduction results
  reduce_res = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation with reduction
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=on_false,
      reduce_cmd=nisa.reduce_cmd.reset_reduce,
      reduce_res=reduce_res,
      reduce_op=nl.max
  )
  
  # Store results to HBM
  nl.store(result_tensor, value=dst)
  nl.store(reduce_tensor, value=reduce_res)

  return result_tensor, reduce_tensor
```

```python
@nki.jit(mode="simulation")
def nki_select_reduce_reverse_pred(predicate_data, on_true_data):
  """
  Example 3: Using select_reduce with reverse_pred option
  Reverse the meaning of the predicate
  """
  # Create output tensor for result
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation with reverse_pred=True
  # This will select on_true where predicate is FALSE
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
      reverse_pred=True  # Reverse the meaning of the predicate
  )
  
  # Store result to HBM
  nl.store(result_tensor, value=dst)

  return result_tensor
```

## infer_resnet50_keras_loadtest.py

```python
import tensorflow as tf
import tensorflow.neuron
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
import numpy as np
import os

# Load and preprocess image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl, dtype='float16')
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = np.repeat(img_arr2, batch_size, axis=0)

# Configure NeuronCore allocation
os.environ['NEURON_MAX_NUM_INFERS'] = str(num_infers_in_flight)
os.environ['NEURONCORE_GROUP_SIZES'] = ','.join(group_sizes)

# Load compiled model
compiled_model_dir = "./rn50_fp16_compiled_b{}_nc{}/1".format(batch_size, neuroncore_pipeline_cores)
predictor = tf.contrib.predictor.from_saved_model(compiled_model_dir)

# Run inference
model_feed_dict = {'input_1:0': img_arr3}
result = predictor(model_feed_dict)
```

## tutorial-model-serving.rst

```python
from packaging import version
import numpy as np
import mxnet as mx

mxnet_version = version.parse(mx.__version__)
if mxnet_version >= version.parse("1.8"):
    import mx_neuron as neuron
else: 
    from mxnet.contrib import neuron

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
mx.test_utils.download(path+'synset.txt')

nn_name = "resnet-50"

# Load a model
sym, args, auxs = mx.model.load_checkpoint(nn_name, 0)

# Define compilation parameters
#  - input shape and dtype
inputs = {'data' : mx.nd.zeros([1,3,224,224], dtype='float32') }

# compile graph to inferentia target
csym, cargs, cauxs = neuron.compile(sym, args, auxs, inputs)

# save compiled model
mx.model.save_checkpoint(nn_name + "_compiled", 0, csym, cargs, cauxs)
```

```python
from packaging import version
import mxnet as mx

mxnet_version = version.parse(mx.__version__)
if mxnet_version >= version.parse("1.8"):
    import mx_neuron as neuron

self.mxnet_ctx = mx.neuron()
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


# Load compiled models and deploy to Trainium
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

# Load the compiled UNet onto two neuron cores with DataParallel
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load("unet/model.pt"), 
    device_ids, 
    set_dynamic_batching=False
)

# Load other compiled models onto neuron cores
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load("text_encoder/model.pt")
pipe.vae.decoder = torch.jit.load("vae_decoder/model.pt")
pipe.vae.post_quant_conv = torch.jit.load("vae_post_quant_conv/model.pt")
pipe.safety_checker.vision_model = NeuronSafetyModelWrap(torch.jit.load("safety_model/model.pt"))
```

## inference.rst

```python
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


# Trace the parallel model
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

## timing.py

```python
from typing import Any, Callable
import time
import numpy as np

time_unit_ratios = {
    'ns': { 'ns': 1, 'us': 1e-3, 'ms': 1e-6, 's': 1e-9 },
    'us': { 'ns': 1e3, 'us': 1, 'ms': 1e-3, 's': 1e-6 },
    'ms': { 'ns': 1e6, 'us': 1e3, 'ms': 1, 's': 1e-3 },
    's': { 'ns': 1e9, 'us': 1e6, 'ms': 1e3, 's': 1 }
}

def timestamp_convert(timestamps,
                      input_time_unit: str,
                      output_time_unit: str):
    """Convert timestamp(s) from one time unit to another.

    :param ts: A timestamp or iterable of timestamps.
    :param input_time_unit: A string specifying the input time unit.
    :param output_time_unit: A string specifying the output time unit.
    :returns: A single timestamp or container of timestamps in the output time unit.
    """
    try:
        ratio = time_unit_ratios[input_time_unit][output_time_unit]
    except:
        raise ValueError(f"Can't convert {input_time_unit} to {output_time_unit}")

    return timestamps * ratio


class Timer():
    def __init__(self,
                 timer_fn: Callable[[], Any] = time.perf_counter,
                 timer_unit: str = 's'):
        self.timer_fn = timer_fn
        self.timer_unit = timer_unit
        self._start = []
        self._end = []

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        if len(self._start) > len(self._end): self._start.pop()
        self._start.append(self.timer_fn())

    def stop(self):
        if 0 == len(self._start): return
        self._end.append(self.timer_fn())

    def next(self):
        """Manually advance the timer to the next timestamp measurement."""
        self.stop()
        self.start()

    def reset(self):
        self._start.clear()
        self._end.clear()

    def insert(self, timestamps: tuple, time_unit: str):
        """Manually insert a timestamp pair. Does not affect ongoing timing.

        :param timestamps: Timestamp pair to insert.
        :param time_unit: The time unit of the incoming timestamps.
        """
        if len(timestamps) != 2 or not time_unit: raise ValueError()
        timestamps = timestamp_convert(np.array(timestamps), time_unit, self.timer_unit)
        self._start.insert(0, timestamps[0])
        self._end.insert(0, timestamps[1])

    def durations(self, time_unit: str = None):
        """Returns an `ndarray` of timestamp deltas, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        :returns: An `ndarray` of timestamp deltas.
        """
        starts, ends = self.start_timestamps(), self.end_timestamps()
        return timestamp_convert(ends - starts[:len(ends)], self.timer_unit, time_unit)

    def total_duration(self, time_unit: str = None):
        """Returns total duration of all time measurements, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        """
        starts, ends = self.start_timestamps(), self.end_timestamps()
        total = np.sum(ends - starts[:len(ends)])
        return total if not time_unit else timestamp_convert(total, self.timer_unit, time_unit)

    def avg(self, time_unit: str = None):
        """Returns average duration, optionally converted into a provided time unit.

        :param time_unit: The time unit of the output timestamp(s). `None` will use the timer's native unit.
        :returns: The average duration.
        """
        return self.durations(time_unit).mean() if len(self._end) > 0 else 0
```

## tutorial-neuron-monitor-mnist.rst

```python
for run in range(0, 1000):
    print(f'Run {run}')
    model.train()
```

## bert_benchmark_utils.py

```python
import torch
import torch.neuron
import os
import sys
import csv
import math
from collections import Counter

import numpy as np

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

## sd2_768_benchmark.py

```python
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

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
```

```python
# Load compiled models and deploy to Trainium
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

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
  # add a and b element-wise and store in c[128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_tensor_scalar(a_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = 2.2
  # add constant b to each element in a
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_free_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:1])
  # broadcast on free dimension -- [128, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_par_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on partition dimension -- [1, 512] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_both_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:1])
  # broadcast on both dimensions -- [1, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_each_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray([128, 512], dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor[0:128, 0:1])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on each dimensions -- [128, 1] and [1, 512] are broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  return c_tensor
```

## ssd300_evaluation.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

def get_val_dataset(val_annotate, val_coco_root):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco
```

```python
# Load SavedModel and create predictor
predictor_list = [tf.contrib.predictor.from_saved_model(args.saved_model) for _ in range(args.num_sessions)]

# Run inference
def predict(pred, model_feed_dict):
    start = time.time()
    result = pred(model_feed_dict)
    latency_list.append(time.time() - start)
    return result

# Invoke predictor with feed dictionary
result = predictor_list[0]({'batch_image': [img_jpg_bytes]})
```

## ssd300_evaluation.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

def get_val_dataset(val_annotate, val_coco_root):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco
```

```python
# Load SavedModel and create predictor
predictor_list = [tf.contrib.predictor.from_saved_model(args.saved_model) for _ in range(args.num_sessions)]

# Run inference
def predict(pred, model_feed_dict):
    start = time.time()
    result = pred(model_feed_dict)
    latency_list.append(time.time() - start)
    return result

# Execute prediction with feed dict
result = predict(predictor_list[0], {'batch_image': [img_jpg_bytes]})

# Access results
boxes = result['boxes']
classes = result['classes']
scores = result['scores']
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
import neuronperf as npf

reports = npf.torch.benchmark(filename, inputs, batch_sizes, n_models=1, workers_per_model=[1, 2], duration=15)
```

```python
import neuronperf as npf

reports = npf.torch.benchmark(..., model_name="MyFancyModel")
```

```python
import neuronperf as npf

cpu_reports = npf.cpu.benchmark(YourModelClass, ...)
```

```python
import neuronperf as npf

gpu_reports = npf.torch.benchmark(YourModelClass, ..., device_type="gpu")
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


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle(in_tensor):
  """
  Example 1: Apply cross-partition data movement to a 32-partition tensor,
  in-place shuffling the data in partition[i] to partition[(i+1)%32].
  """
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[32, 128] = nl.load(in_tensor)
  a_mgrid = nl.mgrid[0:32, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=a)
  return out_tensor


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle_broadcast_partition(in_tensor):
  """
  Example 2: Broadcast data in 1 partition to 32 partitions.
  """
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
  """
  Example 3: When src and dst access more than one quadrant (32 partitions),
  the shuffle is applied to each quadrant independently with the same shuffle_mask.
  """
  out_tensor = nl.ndarray(shape=(128, 128), dtype=np.float32, buffer=nl.shared_hbm)
  a: tensor[128, 128] = nl.load(in_tensor)
  b = nl.ndarray(shape=(128, 128), dtype=np.float32)
  mgrid = nl.mgrid[0:128, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[mgrid.p, mgrid.x], dst=b[mgrid.p, mgrid.x], shuffle_mask=shuffle_mask)
  nl.store(out_tensor, value=b)
  return out_tensor
```

## model_optimizer_wrapper_developer_guide.rst

```python
import neuronx_distributed as nxd
import torch

# Create training config
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
wrapped_model = model.local_module()

# Access model properties
dtype = model.dtype
config = model.config
name_or_path = model.name_or_path

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
  
  # Example 1: subtract 1.0 from all elements of tile a of
  # shape (128, 512) and get the output tile in b
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)
  nl.store(b_tensor[i_p, i_f], b)

  # Example 2: broadcast 1.0 into a shape of (128, 512) and subtract
  # it with tile c to get output tile d
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  c = nl.load(c_tensor[i_p, i_f])
  d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)
  nl.store(d_tensor[i_p, i_f], d)

  # Example 3: broadcast multiply tile e with vector f and
  # then broadcast add with scalar 2.5;
  # tile e has a shape of (64, 1024) and vector f has a shape of (64, 1)
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

## handler_bert_neuronx.py

```python
import os
import json
import logging
from abc import ABC

import torch
import torch_neuronx

from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


os.environ['NEURON_RT_NUM_CORES'] = '1'

logger = logging.getLogger(__name__)

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
import logging
from abc import ABC

import torch
import torch_neuron
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


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
    // xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
    // M = version = 4, (4 bits, 0100 = 0x4)
    // N = variant = 1, (2 bits, 10XX = 0x{8, 9, A, B})

    static const char *chars = "0123456789abcdef";
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<> dist(0, 15);

    std::stringstream ss;
    for (size_t i = 0; i < 37; i++) {
        const int index = dist(mt);
        ss << chars[index];
    }

    // variant bits are 10XX
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
# XLA device setup for Trainium
device = 'xla'
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
```

```python
# Training loop with XLA mark_step
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
    xm.mark_step()  # XLA: collect ops and run them in XLA runtime
```

```python
# Checkpoint saving with XLA
checkpoint = {'state_dict': model.state_dict()}
xm.save(checkpoint, 'checkpoints/checkpoint.pt')
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

## standard_mixed_precision.rst

```python
mixed_precision_config = {
    "use_master_weights": True,
    "use_fp32_grad_acc": True,
    "use_master_weights_in_ckpt": False,
}

config = {
    "mixed_precision_config": mixed_precision_config,
}
```

```python
# same as `mixed_precision_config = None`
mixed_precision_config = {
    "use_master_weights": optimizer_config["zero_one_enabled"],
    "use_fp32_grad_acc": optimizer_config["zero_one_enabled"],
    "use_master_weights_in_ckpt": False,
}

config = {
    "mixed_precision_config": mixed_precision_config,
}
```

```python
mixed_precision_config = {
    "use_master_weights": False,
    "use_fp32_grad_acc": False,
    "use_master_weights_in_ckpt": False,
}

config = {
    "mixed_precision_config": mixed_precision_config,
}
```

## ssd300_detection.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

predictor = tf.contrib.predictor.from_saved_model(args.saved_model)
results = predictor(model_feed_dict)
boxes_np = results['boxes']
scores_np = results['scores']
classes_np = results['classes']
```

## ssd300_detection.py

```python
import tensorflow as tf
import tensorflow.neuron as tfn

predictor = tf.contrib.predictor.from_saved_model(args.saved_model)
results = predictor(model_feed_dict)
boxes_np = results['boxes']
scores_np = results['scores']
classes_np = results['classes']
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

```python
# Accuracy check: Compare the output tensors
allclose = torch.allclose(output_torch, output_nki, atol=1e-3, rtol=1e-2)
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
  i_p = nl.arange(sz_p)[:, None, None] # 3D for
  i_win = nl.arange(sz_win)[None, None, :]
  i_hin = nl.arange(sz_hin)[None, :, None]

  i_wout = nl.arange(sz_wout)[None, None, :]
  i_hout = nl.arange(sz_hout)[None, :, None]

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i_0 = nl.arange(sz_p)[:, None, None, None, None] #
  i_1 = nl.arange(sz_hin//sz_pool)[None, :, None, None, None] # y_outer
  i_2 = nl.arange(sz_pool)[None, None, :, None, None] # y_inner
  i_3 = nl.arange(sz_win//sz_pool)[None, None, None, :, None] # x_outer
  i_4 = nl.arange(sz_pool)[None, None, None, None, :] # x_inner

  # Load input data from external memory to on-chip memory
  # Declare ndarray to force a 3D tensor (temporary requirement)
  in_tile = nl.ndarray([sz_p, sz_hin, sz_win], dtype=in_tensor.dtype)
  in_tile[:,:,:] = nl.load(in_tensor[i_p, i_hin, i_win])

  # Perform the pooling operation:
  # We use numpy's advanced indexing, in order to extend in_tile to 5D, and then reduce-max two dimension.
  # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
  # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
  # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
  # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].
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
import mx_neuron
from multiprocessing import Process, Manager
from time import time


def consumer(model_file, sample_input, input_queue, result_queue):
    sym, args, aux = mx.model.load_checkpoint(model_file, 0)
    sample_input = {key: mx.nd.array(v) for key, v in sample_input.items()}
    args.update(sample_input)
    model = sym.bind(mx.cpu(), args=args, aux_states=aux, grad_req="null")

    while True:
        inputs, input_id = input_queue.get()
        input_queue.task_done()
        # Stop execution if stopping condition is recieved
        if inputs == "stop":
            break
        inputs = {key: mx.nd.array(v) for key, v in inputs.items()}
        start = time()
        results = model.forward(**inputs)
        results[0].wait_to_read()

        # Make the output iterable - if it is not already a tuple or list
        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]
        end = time()

        if input_id != -1:
            result_queue.put((results, start, end, input_id))


class NeuronSimpleDataParallel:
    def __init__(self, model_file, num_neuron_cores, sample_input):
        self.num_neuron_cores = num_neuron_cores
        self.sample_input = sample_input
        self.model_path = model_file
        # Create shared input queue and output queue
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

    def warmup(self, batch):
        self.input_queue.put((batch, -1))

    def infer(self, batch):
        self.input_id += 1
        self.input_dict.add(self.input_id)
        self.input_queue.put((batch, self.input_id))

    def stop(self):
        for _ in range(self.num_neuron_cores):
            self.input_queue.put(("stop", -1))

    def add_result(self, callback_fn):
        if not self.result_queue.empty():
            result, start, end, input_id = self.result_queue.get()
            self.input_dict.remove(input_id)
            self.result_queue.task_done()
            callback_fn(result, start, end)

    def add_all_results(self, callback_fn):
        results = []
        while len(self.input_dict):
            self.add_result(callback_fn)
        for p in self.processes:
            p.join()
```

## test_nki_isa_sequence_bounds.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import numpy as np


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
from torch_neuron import trace

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Prepare example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Trace model for Neuron compilation
model_neuron = trace(model, example_inputs_paraphrase)

# Run inference with compiled model
paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)

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
  # We refer to an indexed portion of a tensor as an intermediate tensor
  a_tile = nl.load(a_input[ix, iy])
  b_tile = nl.load(b_input[ix, iy])

  # compute a + b
  c_tile = a_tile + b_tile

  # store the addition results back to device memory (c_output)
  nl.store(c_output[ix, iy], value=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output


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

# Set Keras global configurations
tf.keras.backend.set_learning_phase(0)
tf.keras.backend.set_image_data_format('channels_last')
tf.keras.backend.set_floatx('float32')

# Load pre-trained model using Keras
model = ResNet50(weights='imagenet')

# Obtain model metadata
model_input = model.input.name.replace(':0', '')
model_output = model.output.name.replace(':0', '')
batch, height, width, channels = model.input.shape

# Obtain the TF session
sess = tf.compat.v1.keras.backend.get_session()

# Save checkpoint files
ckpt_file = '/tmp/resnet50_fp32_keras/resnet50_fp32_keras.ckpt'
graph_file = '/tmp/resnet50_fp32_keras/resnet50_fp32_keras.pb'
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

# NeuronPerf benchmark API usage
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
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import mnist
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

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


def training_loop():
    # Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = 'xla'

    # Move model to device and declare optimizer and loss function
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
        xm.mark_step()  # Collect ops and run them in XLA runtime


def save_checkpoint(model):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    # Use xm.save instead of torch.save to ensure states are moved back to cpu
    xm.save(checkpoint, 'checkpoints/checkpoint.pt')
```

## test_nki_spmd_grid.py

```python
import numpy as np
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


# Example 1: Let compiler decide how to distribute the instances of spmd kernel
dst = nki_spmd_kernel[4, 2](src)

# Example 2: Distribute SPMD kernel instances to physical NeuronCores with
# explicit annotations using spmd_dim
dst = nki_spmd_kernel[nl.spmd_dim(nl.nc(2), 2), 2](src)
dst = nki_spmd_kernel[nl.nc(2) * 2, 2](src)  # syntactic sugar

# Example 3: Distribute SPMD kernel instances to physical NeuronCores with
# explicit annotations using spmd_dim with different dimension ordering
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
    except:  # noqa: E722, there are various ways above code can fail and we don't care
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
            "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"  # noqa: E501
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
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader

# XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
device = 'xla'

# Move model to device and declare optimizer and loss function
model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

# Run the training loop
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
  ##################################################################
  # Example 1: reduce add tile a of shape (128, 32, 4)
  # in the partition dimension and return
  # reduction result in tile b of shape (1, 32, 4)
  ##################################################################
  a = nl.load(a_tensor[0:128, 0:32, 0:4])  
  b = nisa.tensor_partition_reduce(np.add, a)
  nl.store(b_tensor[0:1, 0:32, 0:4], b)

@nki.trace
def nki_par_reduce_nd_b(a_tensor, b_tensor):
  ##################################################################
  # Example 2: reduce add tile a of shape (b, p, f1, ...)
  # in the partition dimension p and return
  # reduction result in tile b of shape (b, 1, f1, ...)
  ##################################################################
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

## lora_finetune_developer_guide.rst

```python
import nxd

# Enable LoRA finetuning - Configuration setup
lora_config = nxd.modules.lora.LoraConfig(
    enable_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    lora_verbose=True,
    target_modules=["q_proj", "v_proj", "k_proj"],
    save_lora_base=False,
    merge_lora=False,
)

# Initialize NxD model with LoRA enabled
nxd_config = nxd.neuronx_distributed_config(
    lora_config=lora_config,
)
model = nxd.initialize_parallel_model(nxd_config)

# Save LoRA checkpoint
nxd.save_checkpoint(
    checkpoint_dir_str=checkpoint_dir,
    tag=tag,
    model=model
)

# Load LoRA checkpoint
lora_config = nxd.modules.lora.LoraConfig(
    enable_lora=True,
    load_lora_from_ckpt=True,
    lora_save_dir=checkpoint_dir,
    lora_load_tag=tag,
)
nxd_config = nxd.neuronx_distributed_config(
    lora_config=lora_config,
)
model = nxd.initialize_parallel_model(nxd_config)
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
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    tokenizer_init = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model_file = os.path.join(model_dir, 'neuron_compiled_model.pt')
    model_neuron = torch.jit.load(model_file)
    return (model_neuron, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)


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


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
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
else:
    print("NKI and Torch differ")
```

```python
device = xm.xla_device()
cpu = torch.device('cpu')

# Test the large workload with tiled kernels
lhs = torch.rand((4096, 1024), dtype=torch.bfloat16, device=device)
rhs = torch.rand((1024, 2048), dtype=torch.bfloat16, device=device)

# Run torch reference
output_torch = torch.matmul(lhs, rhs).to(device=cpu)

def check_match(nki_func):
    output = nki_func(lhs.T, rhs)
    output_nki = output.to(device=cpu)
    if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
        print("NKI and Torch match")
    else:
        print("NKI and Torch differ")

check_match(nki_matmul_tiled_)
check_match(nki_matmul_hoist_load_)
check_match(nki_matmul_block_free_dimension_)
check_match(nki_matmul_fully_optimized_)
```

## infer_resnet50_keras.py

```python
import numpy as np
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

```python
# Load and preprocess image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(np.repeat(img_arr2, 1, axis=0))

# Load model
predictor_host = tf.contrib.predictor.from_saved_model(SAVED_MODEL_DIR)

# Run inference
model_feed_dict = {'input_1:0': img_arr3}
infa_rslts = predictor_host(model_feed_dict)
predictions = resnet50.decode_predictions(infa_rslts[output_tname], top=5)[0]
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

## parallel.py

```python
from concurrent import futures
import torch
import torch.neuron
import os
from time import time
from queue import Queue

def consumer(model, input_queue):
    while True:
        inputs, input_id, callback_fn = input_queue.get()
        input_queue.task_done()
        # Stop execution if stopping condition is recieved
        if inputs == "stop":
            break
        start = time()
        results = model(*inputs)
        # Make the output iterable - if it is not already a tuple or list
        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]
        end = time()
        if callback_fn is not None:
            callback_fn(results, input_id, start, end)
              
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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)

# Prepare inputs
inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]

# Compile for Trainium
npf.tensorflow.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename=filename,
    model_name=model_name,
)
```

## test_nki_nl_gather_flattened.py

```python
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import neuronxcc.nki as nki

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
    
    return result
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

## distilbert-base-uncased-finetuned-sst-2-english_benchmark.py

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
```

```python
inputs = [
    get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes
]

reports = neuronperf.torch.benchmark(filename, inputs)

neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
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
  ##################################################################
  # Example 1: Generate tile b of 32 * 128 random floating point values,
  # find the 8 largest values in each row, then find their indices:
  ##################################################################
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
inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]
```

```python
reports = neuronperf.torch.benchmark(filename, inputs)
neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
```

## distilroberta-base_benchmark.py

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
# Load model and tokenizer
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)

# Prepare inputs
sequence_length = 128
batch_size = 6
inputs = get_batch(tokenizer, sequence_length, batch_size)

# Benchmark
filename = f"{model_name}_sl{sequence_length}.json"
reports = neuronperf.torch.benchmark(filename, [inputs])

# View and save results
neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
```

## bert-base-uncased_benchmark.py

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
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=False)
```

```python
inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]
reports = neuronperf.torch.benchmark(filename, inputs)
neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
```

## distilbert-base-uncased-finetuned-sst-2-english_benchmark.py

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
```

## bert-base-cased_benchmark.py

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
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=False)
```

```python
inputs = [get_batch(tokenizer, sequence_length, batch_size) for batch_size in batch_sizes]
reports = neuronperf.torch.benchmark(filename, inputs)
neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
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

## bert-base-uncased_compile.py

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
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=False)

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

## bert-base-cased_compile.py

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
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=False)

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

## neuronperf_overview.rst

```python
import neuronperf
import neuronperf.torch
```

```python
reports = neuronperf.torch.benchmark(model, inputs, ...)
```

```python
model_index = neuronperf.torch.compile(model, inputs, ...)
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
  
  # Example 1: reduce add tile a of shape (128, 512)
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
import neuronperf
import neuronperf.torch
import torch_neuronx
import os

from torchvision.datasets import CIFAR100
from transformers import CLIPProcessor, CLIPModel

def benchmark(model_name, batch_size):
    # Build the model, preprocessor, and dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, return_dict=False)

    # Prepare a sample input
    image = cifar100[0][0]
    text = []
    for c in cifar100.classes:
        text.append(f'a photo of a {c}')

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    image = inputs['pixel_values']
    # (b, c, h, w)
    image = image.repeat(batch_size, 1, 1, 1)
    inputs = (inputs['input_ids'], image)

    # Trace the model
    model.eval()
    traced = torch_neuronx.trace(model, inputs, compiler_args='--enable-saturate-infinity')
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [inputs], batch_sizes=[batch_size])
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

  # Example 1: add two tiles, a and b, of the same
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
  
  # Load from in_tensor and broadcast into out_tile
  in_tile = nl.load(in_tensor, dtype=in_tensor.dtype)
  out_tile = nl.broadcast_to(in_tile, shape=(128, in_tensor.shape[1]))

  # Store output
  nl.store(out_tensor, out_tile)
  return out_tensor
```

## inf2_benchmark.py

```python
import torch
import neuronperf
import neuronperf.torch
import torch_neuronx
from transformers import AutoModel

class GPT2Neuron(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

# Load and prepare model
model = AutoModel.from_pretrained('gpt2', torchscript=True)
model = GPT2Neuron(model)
model.eval()

# Create example inputs
example = (
    torch.zeros(16, 256, dtype=torch.int),  # input_ids
    torch.zeros(16, 256, dtype=torch.int),  # attention_mask
)

# Trace model for Neuron
traced = torch_neuronx.trace(model, example)

# Save traced model
filename = 'model.pt'
torch.jit.save(traced, filename)

# Benchmark the model
reports = neuronperf.torch.benchmark(filename, [example])

# View results
neuronperf.print_reports(reports)
neuronperf.write_csv(reports)
neuronperf.write_json(reports)
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

# Compile SavedModel for Trainium
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

  # Example 1: Copy over the tensor to another tensor using the Vector engine.
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

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
model.eval()

# Prepare input data
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
inputs = inputs.repeat([1, 1])
example = (inputs,)

# Trace model for Trainium
traced = torch_neuronx.trace(model, example, compiler_args='--model-type=transformer')

# Save and load traced model
filename = 'model.pt'
torch.jit.save(traced, filename)
model_neuron = torch.jit.load(filename)

# Run inference
output = model_neuron(inputs)
```

## hf_pretrained_wav2vec2_conformer_rope_benchmark.py

```python
import torch
import torch_neuronx
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model.eval()

# Prepare input data
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
inputs = inputs.repeat([1, 1])
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
import neuronperf
import neuronperf.torch
import torch_neuronx

from PIL import Image
import requests
from transformers import ViTImageProcessor, ViTForImageClassification


def benchmark(batch_size):
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torchscript=True)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].repeat([batch_size, 1, 1, 1])
    example = (inputs,)

    traced = torch_neuronx.trace(model, example, compiler_args="--model-type=transformer")
    filename = 'model.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [example], batch_sizes=[batch_size])
```

## test_nki_isa_affine_select.py

```python
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit(mode="simulation")
def nki_affine_select(a_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  # Example 1: Take tile a of shape [128, 128] and replace its
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

## perceiver-vision_benchmark.py

```python
import torch
import neuronperf as npf
import neuronperf.torch


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)
```

```python
inputs = [get_batch(batch_size) for batch_size in batch_sizes]
reports = npf.torch.benchmark(filename, inputs, n_models=n_models, workers_per_model=workers_per_model)
```

```python
npf.print_reports(reports)
npf.write_csv(reports)
npf.write_json(reports)
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

allclose = torch.allclose(output_torch, output_nki, atol=1e-5, rtol=1e-3)
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
  i_p = nl.arange(256)[:, None] # Previously nl.arange(128)
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


# Compile a model for Trainium
model = transformers.PerceiverForImageClassificationLearned.from_pretrained(
    "deepmind/vision-perceiver-learned"
)
inputs = [get_batch(batch_size) for batch_size in [1]]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=[1],
    pipeline_sizes=[1],
    filename="model.json",
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
# Note that image_batched.shape[dim] / len(device_ids) == image.shape[dim]
batch_size = 2 * 8
image_batched = torch.rand([1, 3, batch_size, 8])

# Run inference with a batched input
output = model_parallel(image_batched)
```

## trace_bert_neuronx.py

```python
import torch
import torch_neuronx

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup some example inputs
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

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
model_neuron_batch = torch_neuronx.trace(model, example_inputs_paraphrase)

# Save the batched model
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

# Setup some example inputs
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

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)

# Save the batched model
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

# Trace model with torch_neuron
model_neuron_batch = torch_neuron.trace(model, example_inputs_paraphrase)

# Save the traced model
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
# NKI kernel usage
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

# Reference JAX implementation
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

# Compile
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

# Set up parameters
POOL_SIZE = 2
C, HIN, WIN = 2, 6, 6
HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

# Create input tensor on device
in_tensor = torch.arange(C * HIN * WIN, dtype=torch.bfloat16).reshape(C, HIN, WIN).to(device=device)

# Call NKI kernel
out_nki = tensor_avgpool_kernel(in_tensor, POOL_SIZE)

# Compare with PyTorch reference
out_torch = torch.nn.functional.avg_pool2d(in_tensor, POOL_SIZE, POOL_SIZE)

# Verify results
if (out_nki == out_torch).all():
    print("NKI and Torch match")
else:
    print("NKI and Torch differ")
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

## resnet_benchmark.py

```python
import torch
import neuronperf as npf
import neuronperf.torch


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)
```

```python
inputs = [get_batch(batch_size) for batch_size in batch_sizes]
filename = f"{model_name}.json"

reports = npf.torch.benchmark(filename, inputs, n_models=n_models, workers_per_model=workers_per_model)

npf.print_reports(reports)
npf.write_csv(reports)
npf.write_json(reports)
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

## vgg_benchmark.py

```python
import torch
import neuronperf as npf
import neuronperf.torch


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)
```

```python
inputs = [get_batch(batch_size) for batch_size in batch_sizes]
filename = f"{model_name}.json"

reports = npf.torch.benchmark(filename, inputs, n_models=n_models, workers_per_model=workers_per_model)

npf.print_reports(reports)
npf.write_csv(reports)
npf.write_json(reports)
```

## spmd_multiple_nc_tensor_addition_jax.py

```python
import jax
import jax.numpy as jnp

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

# Compile a model
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

allclose = torch.allclose(a_t_torch, a_t_nki)
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


def get_batch(batch_size):
    return torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)


# Compile a model with neuronperf
model = getattr(torchvision.models, "vgg16")(pretrained=True)
inputs = [get_batch(batch_size) for batch_size in [1, 8, 64]]

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


# Compile model with neuronperf
model = torchvision.models.resnet50(pretrained=True)
batch_sizes = [1, 6]
pipeline_sizes = [1]
inputs = [get_batch(batch_size) for batch_size in batch_sizes]

npf.torch.compile(
    model,
    inputs,
    batch_sizes=batch_sizes,
    pipeline_sizes=pipeline_sizes,
    filename="resnet50.json",
    model_name="resnet50",
)
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

# Select a few batch sizes to test
batch_sizes = [5, 6, 7]

# Construct example inputs
inputs = [torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32) for batch_size in batch_sizes]

# Compile
npf.torch.compile(
	model, 
	inputs, 
	batch_sizes=batch_sizes, 
	filename='resnet50.json',
)

# Benchmark
reports = npf.torch.benchmark('resnet50.json', inputs)

# View and save results
npf.print_reports(reports)
npf.write_csv(reports, 'resnet50_results.csv')
npf.write_json(reports, 'resnet50_results.json')
```

## test_simple_pt.py

```python
import torch
import torch.neuron

import neuronperf as npf
import neuronperf.torch


# Define a simple model
class Model(torch.nn.Module):
    def forward(self, x):
        x = x * 3
        return x + 1


# Instantiate
model = Model()
model.eval()

# Define some inputs
batch_sizes = [1]
inputs = [torch.ones((batch_size, 3, 224, 224)) for batch_size in batch_sizes]

# Compile for Neuron
model_neuron = torch.neuron.trace(model, inputs)
model_neuron.save("model_neuron_b1.pt")

# Benchmark
reports = npf.torch.benchmark("model_neuron_b1.pt", inputs, batch_sizes)

# View and save results
npf.print_reports(reports)
npf.write_csv(reports, "model_neuron_b1.csv")
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

## neuron-setup-example.py

```python
from neuronsetuphelper import neuron_setup_helper

nr_setup = neuron_setup_helper(manifest_file='default', neuron_version='latest')

setup_cmd = nr_setup.instructions(
    framework='tensorflow',
    action='Install',
    os='ubuntu',
    ami='non-dlami',
    mode='develop',
    framework_version='latest'
)
```