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
    # for this example we assume that seqlen_q is divisible by PMAX and 
    # seqlen_kv is divisible by FMAX_MOVING, otherwise need to use mask or "final multiplication"
    qk = nl.ndarray((seqlen_q // PMAX, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                     dtype=nl.float32, buffer=nl.psum)
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

    # Softmax #
    # reduce max along seqlen_k
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

    # reciprocal of sum_row, tile shape is [PMAX, seqlen_q // PMAX]
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
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), d_head), dtype=nl.float32, buffer=nl.sbuf)
        v_sbuf_t[:, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)                   # ScalarE
        nl.store(v_t[i_tile_kv], v_sbuf_t[:,:])

    # scores has the wrong layout
    # PMAX restriction on both free and partition dimension when performing transpose.
    # scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, seqlen_q // PMAX, PMAX),
    #                            dtype=nl.float32, buffer=nl.sbuf)
    scores_t = nl.ndarray((seqlen_kv // PMAX, seqlen_q // PMAX, PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_buf = nl.load(scores[i_tile_q, :, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_psum_t = nisa.nc_transpose(scores_buf) # TensorE
            scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
            scores_sbuf_t[:, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE
            nl.store(scores_t[i_tile_kv, i_tile_q, :, :], scores_sbuf_t)

    # scores @ V, contract along seqlen_kv
    # d_head == P_MAX, no need to tile there
    for i_tile_q in nl.affine_range(seqlen_q // PMAX): # loop on stationary free
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        attn_out = nl.ndarray((nl.par_dim(PMAX), d_head),
                           dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
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
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # per i_tile_q we finish a partial block matrix for qk
        # total blocks are # (seqlen_q // FMAX_STATIONARY) * (seqlen_kv // FMAX_MOVING)
        # we do the operations of attn_fwd_v3 on each block since they are independent row-wise.
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

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        # has recriprocals of 128 rows at a time, akin to the block of
        # output each q-tile is responsible for.
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
            scores_psum_t = nisa.nc_transpose(scores[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
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
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

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

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # CHANGE OF LOGIC COMPARED TO attn_fwd_v4, here we delay the division

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        # notice how here the division is done on the final attention output
        # directly comparing to the previous implementation, we save on having to 
        # loop all the i_tile_kvs, meaning we do less divsion operations as our
        # attention block is already collapsed.
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
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

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
                            dtype=nl.float32, buffer=nl.sbuf)
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
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

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
        qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # previously the entire qk_sbuf row would be processed at once, so PSUM would be occupied for longer
            # here PSUM gets evicted a bit earlier, allowing us to queue the tensor engine earlier as well.
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
                                             is_transpose=True, is_moving_onezero=True)          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)     # ScalarE

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

    for i_tile_q in nl.affine_range(num_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            qk[i_tile_q, i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

    for i_tile_q in nl.affine_range(num_tile_q):
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            qk_sbuf[:, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q, i_tile_kv], op0=nl.multiply, operand0=1.0,
                                                                    reduce_op=nl.max, reduce_res=row_max_kv[:, i_tile_kv])

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_q, i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
            )
        sum_row[:, :] = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        inverse_sum_row[:, :] = nisa.reciprocal(data=sum_row)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t[i_tile_q, i_tile_kv, :, :] = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t[i_tile_q, i_tile_kv, :, :], dtype=nl.bfloat16)

        attn_out_psum[i_tile_q, :, :] = nl.zeros((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.psum)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                                             moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out_sbuf[:, :] = nisa.tensor_scalar(data=attn_out_psum[i_tile_q, :, :], op0=nl.multiply,
                                                  operand0=inverse_sum_row, engine=nisa.vector_engine)

        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out_sbuf[:, :])

    return kernel_out
```

## fused_mamba.rst

```python
import torch
import numpy as np
import neuron.nki as nki
import neuron.nki.language as nl
import neuron.nki.isa as nisa
from neuron.nki import nki_jit


@nki_jit
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
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nisa.np.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nisa.np.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=nisa.np.multiply, op1=nisa.np.add)
                
                # Step 5: scanC = C * scan
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=nisa.np.multiply)
                
                # Step 6: Accumulate across states
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC_i
        
        # Store results
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])
    
    return output


@nki_jit
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
            
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            # Load delta and u once, reuse across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            
            for i_state in nl.affine_range(state_size):
                # Load state-specific inputs
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                # Step 1&2: deltaA = exp(delta * A)
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)
                
                # Step 3: deltaBu = delta * B * u
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nisa.np.multiply)
                B_i_bcast = B_i.broadcast_to((nl.tile_size.pmax, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nisa.np.multiply)
                
                # Step 4: Associative scan
                scan_i = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                                                 op0=nisa.np.multiply, op1=nisa.np.add)
                
                # Step 5: scanC = C * scan
                C_i_bcast = C_i.broadcast_to((nl.tile_size.pmax, seq_len))
                scanC_i = nisa.tensor_tensor(scan_i, C_i_bcast, op=nisa.np.multiply)
                
                # Step 6: Accumulate across states
                scanC_accum[0:channel_psize, 0:seq_len] += scanC_i
            
            # Store results
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    
    return output


@nki_jit
def mamba_v3(delta, u, A, B, C):
    """
    Optimized NKI kernel with seq_len tiling to mitigate spilling.
    
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
            
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype, buffer=nl.sbuf)
            
            # Load delta and u once, reuse across states and seq_len tiles
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            
            for i_state in nl.affine_range(state_size):
                # Load state-specific inputs
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state:i_state+1])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])
                
                scan_init = nl.zeros((channel_psize, 1), dtype=delta.dtype, buffer=nl.sbuf)
                
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_start = i_seq_len_tile * seq_len_fsize
                    seq_end = min(seq_start + seq_len_fsize, seq_len)
                    seq_tile_len = seq_end - seq_start
                    
                    # Extract tiles for this seq_len chunk
                    delta_tile = delta_i[0:channel_psize, seq_start:seq_end]
                    u_tile = u_i[0:channel_psize, seq_start:seq_end]
                    B_tile = B_i[0:1, seq_start:seq_end]
                    C_tile = C_i[0:1, seq_start:seq_end]
                    
                    # Step 1&2: deltaA = exp(delta * A)
                    deltaA = nisa.activation(op=nl.exp, data=delta_tile, scale=A_i)
                    
                    # Step 3: deltaBu = delta * B * u
                    deltaU = nisa.tensor_tensor(delta_tile, u_tile, op=nisa.np.multiply)
                    B_tile_bcast = B_tile.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    deltaBu = nisa.tensor_tensor(deltaU, B_tile_bcast, op=nisa.np.multiply)
                    
                    # Step 4: Associative scan with loop-carried dependency
                    scan_tile = nisa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                                                        op0=nisa.np.multiply, op1=nisa.np.add)
                    
                    # Update scan_init for next iteration
                    scan_init = scan_tile[0:channel_psize, seq_tile_len-1:seq_tile_len]
                    
                    # Step 5: scanC = C * scan
                    C_tile_bcast = C_tile.broadcast_to((nl.tile_size.pmax, seq_tile_len))
                    scanC_tile = nisa.tensor_tensor(scan_tile, C_tile_bcast, op=nisa.np.multiply)
                    
                    # Step 6: Accumulate across states
                    scanC_accum[0:channel_psize, seq_start:seq_end] += scanC_tile
            
            # Store results
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    
    return output
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
  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape

  # Check that the contraction dimension matches and all dimensions
  #are what were expected.
  assert K == K_, \
    f"Expected contraction dimension to match on both lhsT ({K}) and rhs ({K})"
  assert K == 128, f"Expected contraction dimension to be 128, but got {K}"
  assert M == 64, f"Expected lhsT matrix to have dimension M of 64, but got {M}"
  assert N == 512, f"Expected rhs matrix to have dimension N of 512, but got {N}"

  # Create a tensor to write the result into (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Creating a tensor in SBUF to load the inputs into (not initialized)
  lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  nisa.dma_copy(dst=lhs_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  # Create a tensor in PSUM to accumulate the result in (uninitialized)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)

  # Perform the matrix-multiplication
  # Note: A NKI matmul instruction always writes to PSUM in float32 data-type
  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)

  # Create a tensor in SBUF and copy the result from PSUM back to SBUF, 
  # and cast to expected output data-type
  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=result.dtype)

  # The result of [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nisa.dma_copy(dst=result, src=result_sbuf)

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

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

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

  # Verify that the lhsT and rhs are the expected sizes.
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

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_M numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      # Allocate space in SBUF for the tile (uninitialized)
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
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
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])
        # Append the tile to the list of tiles.
        rhs_tiles.append(rhs_tile)

      # Allocate a tile in PSUM for the result (uninitialized)
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

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
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Loop over blocks over the M dimension
  for m in nl.affine_range(M // BLOCK_M):
    # Load TILES_IN_BLOCK_M columns tiles by TILES_K rows from lhsT
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
      # Load TILES_IN_BLOCK_N columns from rhs by TILES_K rows from rhs
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

  # Verify the size is a multiple of block size
  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Compute the number of blocks in each dimension
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    # Create the initial result tiles in SBUF and initialize each tile to
    # 0.0, since the final results will be accumulated here. Results in 3-d array.
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          # Create the result tile (uninitialized)
          tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
          # Initialize the tile 0.0
          nisa.memset(dst=tile, value=0.0)
          # Append the tile to block_n array.
          block_n.append(tile)
        # Append block_n array to block_m array.
        block_m.append(block_n)
      # Append block_m array into result_tmps.
      result_tmps.append(block_m)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler
    # to change this loop by, for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
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
          # Allocate lhsT_tile in SBUF (uninitialized)
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          # Copy block tile from lhsT to lhsT_tile
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          # Append to list of lhsT tiles.
          lhsT_tiles.append(lhsT_tile)

        # Do matmul with all tiles in the blocks
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            # Allocate result_tile in PSUM (uninitialized)
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              # Perform matrix multiply on a tile.
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            # Accumulate the result into the result_tmps tile.
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        # coalesce result tiles for better DMA performance
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        # Copy packed result from SBUF to HBM.
        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
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
xm.xrt_world_size()
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

## matrix_multiplication.rst

```python
import nki
import nki.language as nl
from nki import nisa
import numpy as np

@nl.kernel
def nki_matmul_basic_(lhs: nl.ndarray(64, 128, dtype=nl.bfloat16),
                      rhs: nl.ndarray(128, 512, dtype=nl.bfloat16),
                      output: nl.ndarray(64, 512, dtype=nl.bfloat16)):
    # Define indices to access the LHS and RHS input tensors
    m_idx = nl.arange(64)
    n_idx = nl.arange(512)
    k_idx = nl.arange(128)
    
    # Load LHS in transposed form to map contraction axis to P-dimension
    lhs_tile = lhs[k_idx, m_idx]
    rhs_tile = rhs[k_idx, n_idx]
    
    # Allocate SBUF tiles for inputs
    lhs_sbuf = nl.ndarray((128, 64), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs_sbuf = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_sbuf = nl.ndarray((64, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Load inputs from HBM to SBUF
    nisa.dma_copy(lhs_sbuf, lhs_tile)
    nisa.dma_copy(rhs_sbuf, rhs_tile)
    
    # Perform matrix multiplication
    psum_buf = nl.ndarray((64, 512), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_matmul(psum_buf, lhs_sbuf, rhs_sbuf, transpose_lhs=True)
    
    # Copy result from PSUM back to SBUF
    nisa.tensor_copy(output_sbuf, psum_buf)
    
    # Store result to HBM
    nisa.dma_copy(output, output_sbuf)
```

```python
import nki
import nki.language as nl
from nki import nisa
import numpy as np

@nl.kernel
def nki_matmul_tiled_(lhs_t: nl.ndarray(dtype=nl.bfloat16),
                      rhs: nl.ndarray(dtype=nl.bfloat16),
                      output: nl.ndarray(dtype=nl.bfloat16)):
    # Get dimensions
    K, M = lhs_t.shape
    K, N = rhs.shape
    
    # Tile sizes
    M_TILE = 128
    N_TILE = 512
    K_TILE = 128
    
    # Allocate SBUF buffers
    lhs_t_sbuf = nl.ndarray((K_TILE, M_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs_sbuf = nl.ndarray((K_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_sbuf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Allocate PSUM accumulator
    psum_buf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Tile LHS_T free dimension
    for m in nl.affine_range(M // M_TILE):
        # Tile RHS free dimension
        for n in nl.affine_range(N // N_TILE):
            # Zero-out the accumulator buffer
            nisa.zero(psum_buf)
            
            # Tile contraction dimension
            for k in nl.affine_range(K // K_TILE):
                # Load tiles
                nisa.dma_copy(lhs_t_sbuf, lhs_t[k * K_TILE:(k + 1) * K_TILE, m * M_TILE:(m + 1) * M_TILE])
                nisa.dma_copy(rhs_sbuf, rhs[k * K_TILE:(k + 1) * K_TILE, n * N_TILE:(n + 1) * N_TILE])
                
                # Accumulate matmul results
                nisa.nc_matmul(psum_buf, lhs_t_sbuf, rhs_sbuf, transpose_lhs=True)
            
            # Copy result from PSUM to SBUF
            nisa.tensor_copy(output_sbuf, psum_buf)
            
            # Store to HBM
            nisa.dma_copy(output[m * M_TILE:(m + 1) * M_TILE, n * N_TILE:(n + 1) * N_TILE], output_sbuf)
```

```python
import nki
import nki.language as nl
from nki import nisa
import numpy as np

@nl.kernel
def nki_matmul_hoist_load_(lhs_t: nl.ndarray(dtype=nl.bfloat16),
                           rhs: nl.ndarray(dtype=nl.bfloat16),
                           output: nl.ndarray(dtype=nl.bfloat16)):
    # Get dimensions
    K, M = lhs_t.shape
    K, N = rhs.shape
    
    # Tile sizes
    M_TILE = 128
    N_TILE = 512
    K_TILE = 128
    
    # Allocate SBUF buffers
    lhs_t_sbuf = nl.ndarray((K_TILE, M_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs_sbuf = nl.ndarray((K_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_sbuf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Allocate PSUM accumulator
    psum_buf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Tile LHS_T free dimension
    for m in nl.affine_range(M // M_TILE):
        # Load LHS tile once
        nisa.dma_copy(lhs_t_sbuf, lhs_t[:, m * M_TILE:(m + 1) * M_TILE])
        
        # Tile RHS free dimension
        for n in nl.affine_range(N // N_TILE):
            # Zero-out the accumulator buffer
            nisa.zero(psum_buf)
            
            # Load RHS tile once
            nisa.dma_copy(rhs_sbuf, rhs[:, n * N_TILE:(n + 1) * N_TILE])
            
            # Tile contraction dimension
            for k in nl.affine_range(K // K_TILE):
                # Accumulate matmul results
                nisa.nc_matmul(psum_buf, lhs_t_sbuf[k * K_TILE:(k + 1) * K_TILE, :], 
                              rhs_sbuf[k * K_TILE:(k + 1) * K_TILE, :], transpose_lhs=True)
            
            # Copy result from PSUM to SBUF
            nisa.tensor_copy(output_sbuf, psum_buf)
            
            # Store to HBM
            nisa.dma_copy(output[m * M_TILE:(m + 1) * M_TILE, n * N_TILE:(n + 1) * N_TILE], output_sbuf)
```

```python
import nki
import nki.language as nl
from nki import nisa
import numpy as np

@nl.kernel
def nki_matmul_block_free_dimension_(lhs_t: nl.ndarray(dtype=nl.bfloat16),
                                     rhs: nl.ndarray(dtype=nl.bfloat16),
                                     output: nl.ndarray(dtype=nl.bfloat16)):
    # Get dimensions
    K, M = lhs_t.shape
    K, N = rhs.shape
    
    # Tile sizes
    M_TILE = 128
    N_TILE = 512
    K_TILE = 128
    
    # Blocking factors
    NUM_BLOCK_M = 2
    NUM_BLOCK_N = 2
    
    # Allocate SBUF buffers for blocked tiles
    lhs_t_tiles = nl.ndarray((NUM_BLOCK_M, K, M_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs_tiles = nl.ndarray((NUM_BLOCK_N, K, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    result_tiles = nl.ndarray((NUM_BLOCK_M, NUM_BLOCK_N, M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Allocate PSUM accumulator
    psum_buf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Load blocks of LHS_T
    for bm in nl.affine_range(NUM_BLOCK_M):
        m_start = bm * M_TILE
        nisa.dma_copy(lhs_t_tiles[bm], lhs_t[:, m_start:m_start + M_TILE])
    
    # Load blocks of RHS
    for bn in nl.affine_range(NUM_BLOCK_N):
        n_start = bn * N_TILE
        nisa.dma_copy(rhs_tiles[bn], rhs[:, n_start:n_start + N_TILE])
    
    # Compute blocked matrix multiplication
    for bm in nl.affine_range(NUM_BLOCK_M):
        for bn in nl.affine_range(NUM_BLOCK_N):
            nisa.zero(psum_buf)
            
            for k in nl.affine_range(K // K_TILE):
                nisa.nc_matmul(psum_buf, lhs_t_tiles[bm, k * K_TILE:(k + 1) * K_TILE, :],
                              rhs_tiles[bn, k * K_TILE:(k + 1) * K_TILE, :], transpose_lhs=True)
            
            nisa.tensor_copy(result_tiles[bm, bn], psum_buf)
    
    # Store results to HBM
    for bm in nl.affine_range(NUM_BLOCK_M):
        for bn in nl.affine_range(NUM_BLOCK_N):
            m_start = bm * M_TILE
            n_start = bn * N_TILE
            nisa.dma_copy(output[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_tiles[bm, bn])
```

```python
import nki
import nki.language as nl
from nki import nisa
import numpy as np

@nl.kernel
def nki_matmul_fully_optimized_(lhs_t: nl.ndarray(dtype=nl.bfloat16),
                                rhs: nl.ndarray(dtype=nl.bfloat16),
                                output: nl.ndarray(dtype=nl.bfloat16)):
    # Get dimensions
    K, M = lhs_t.shape
    K, N = rhs.shape
    
    # Tile sizes
    M_TILE = 128
    N_TILE = 512
    K_TILE = 128
    
    # Blocking configuration for large matrices
    NUM_BLOCK_M = M // 2048
    NUM_BLOCK_N = N // 1024
    NUM_BLOCK_K = K // 1024
    
    # Allocate SBUF buffers
    lhs_t_tiles = nl.ndarray((NUM_BLOCK_M, NUM_BLOCK_K, 2048, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    rhs_tiles = nl.ndarray((NUM_BLOCK_N, NUM_BLOCK_K, 1024, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    result_tiles = nl.ndarray((NUM_BLOCK_M, NUM_BLOCK_N, 2048, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Allocate PSUM accumulator
    psum_buf = nl.ndarray((M_TILE, N_TILE), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Load all LHS_T blocks
    for bm in nl.affine_range(NUM_BLOCK_M):
        for bk in nl.affine_range(NUM_BLOCK_K):
            m_start = bm * 2048
            k_start = bk * 1024
            nisa.dma_copy(lhs_t_tiles[bm, bk], lhs_t[k_start:k_start + 1024, m_start:m_start + 2048])
    
    # Load all RHS blocks
    for bn in nl.affine_range(NUM_BLOCK_N):
        for bk in nl.affine_range(NUM_BLOCK_K):
            n_start = bn * 1024
            k_start = bk * 1024
            nisa.dma_copy(rhs_tiles[bn, bk], rhs[k_start:k_start + 1024, n_start:n_start + 1024])
    
    # Compute fully optimized matrix multiplication
    for bm in nl.affine_range(NUM_BLOCK_M):
        for bn in nl.affine_range(NUM_BLOCK_N):
            nisa.zero(psum_buf)
            
            # Use sequential_range for hand-optimized K blocking loop
            for bk in nl.sequential_range(NUM_BLOCK_K):
                for m_tile in nl.affine_range(2048 // M_TILE):
                    for n_tile in nl.affine_range(1024 // N_TILE):
                        for k_tile in nl.affine_range(1024 // K_TILE):
                            m_idx = m_tile * M_TILE
                            n_idx = n_tile * N_TILE
                            k_idx = k_tile * K_TILE
                            
                            nisa.nc_matmul(psum_buf, 
                                          lhs_t_tiles[bm, bk, k_idx:k_idx + K_TILE, m_idx:m_idx + M_TILE],
                                          rhs_tiles[bn, bk, k_idx:k_idx + K_TILE, n_idx:n_idx + N_TILE],
                                          transpose_lhs=True)
            
            nisa.tensor_copy(result_tiles[bm, bn], psum_buf)
    
    # Store results to HBM
    for bm in nl.affine_range(NUM_BLOCK_M):
        for bn in nl.affine_range(NUM_BLOCK_N):
            m_start = bm * 2048
            n_start = bn * 1024
            nisa.dma_copy(output[m_start:m_start + 2048, n_start:n_start + 1024], result_tiles[bm, bn])
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

## spmd_tensor_addition.rst

```python
import nki
import nki.language as nl

@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
    """
    NKI kernel for tensor addition using SPMD programming model.
    Operates on tiles of size [128, 512].
    """
    # Allocate output tensor
    c_output = nl.zeros(a_input.shape, dtype=a_input.dtype)
    
    # Get worker ID for SPMD execution
    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    
    # Define tile sizes
    tile_size_x = 128  # Partition dimension (hardware restricted)
    tile_size_y = 512  # Free dimension
    
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
    """
    Helper function to launch the NKI tensor addition kernel with appropriate grid/block sizes.
    Assumes tensor sizes are multiples of maximum tile sizes.
    """
    # Define tile sizes
    tile_size_x = 128
    tile_size_y = 512
    
    # Calculate grid dimensions
    grid_x = a.shape[0] // tile_size_x
    grid_y = a.shape[1] // tile_size_y
    
    # Launch kernel with 2D grid
    return nki_tensor_add_kernel_[grid_x, grid_y](a, b)
```

```python
import torch

# Prepare input tensors
a = torch.rand((1024, 2048), dtype=torch.bfloat16, device='xla:0')
b = torch.rand((1024, 2048), dtype=torch.bfloat16, device='xla:0')

# Execute NKI kernel
output_nki = nki_tensor_add(a, b)

# Compute reference result using PyTorch
output_torch = a + b

# Verify correctness
assert torch.allclose(output_nki, output_torch), "NKI and Torch outputs do not match"
print("NKI and Torch match")
```

```python
import jax
import jax.numpy as jnp

# Prepare input arrays
a = jax.random.uniform(jax.random.PRNGKey(0), (1024, 2048), dtype=jnp.bfloat16)
b = jax.random.uniform(jax.random.PRNGKey(1), (1024, 2048), dtype=jnp.bfloat16)

# Execute NKI kernel
output_nki = nki_tensor_add(a, b)

# Compute reference result using JAX
output_jax = a + b

# Verify correctness
assert jnp.allclose(output_nki, output_jax), "NKI and JAX outputs do not match"
print("NKI and JAX match")
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
nki_input = input_tensor.to(device=device, dtype=torch.bfloat16)
gate_w_xla = model.gate_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
up_w_xla = model.up_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
down_w_xla = model.down_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)

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

## layernorm.rst

# Extracted Code Examples from layernorm.rst

## Version 1: nki.language APIs only

```python
import nki
import nki.language as nl
import math

@nki.jit
def nki_layernorm_kernel_v1(
    input_tensor: nki.tensor,
    gamma: nki.tensor,
    beta: nki.tensor,
    output_tensor: nki.tensor,
    epsilon: float = 1e-5,
):
    """
    LayerNorm kernel using nki.language APIs.
    
    Args:
        input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
        gamma: 1D affine parameter of shape [hidden_size]
        beta: 1D affine parameter of shape [hidden_size]
        output_tensor: 2D output tensor of shape [sequence_length, hidden_size]
        epsilon: Small constant for numerical stability
    """
    seq_len = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    
    # Load gamma and beta, perform partition-axis broadcast
    shift_scale_tensor = nl.load(gamma)
    shift_scale_tensor = nl.broadcast_to(shift_scale_tensor, (nl.tile_size.pmax, hidden_size))
    
    beta_tensor = nl.load(beta)
    beta_tensor = nl.broadcast_to(beta_tensor, (nl.tile_size.pmax, hidden_size))
    
    # Compute loop over partition axis
    for i in nl.affine_range(math.ceil(seq_len / nl.tile_size.pmax)):
        # Load one tile of input_tensor
        input_tile = nl.load(
            input_tensor,
            indices=[i * nl.tile_size.pmax, 0],
            mask=(i * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None] < seq_len)
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
            output_tensor,
            indices=[i * nl.tile_size.pmax, 0],
            value=output_tile,
            mask=(i * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None] < seq_len)
        )
```

## Version 2: nki.isa APIs for optimized mean/variance and shift/scale

```python
import nki
import nki.language as nl
import nki.isa as isa
import math

@nki.jit
def nki_layernorm_kernel_v2(
    input_tensor: nki.tensor,
    gamma: nki.tensor,
    beta: nki.tensor,
    output_tensor: nki.tensor,
    epsilon: float = 1e-5,
):
    """
    Optimized LayerNorm kernel using nki.isa APIs for mean/variance calculation
    and shift/scale operations.
    
    Args:
        input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
        gamma: 1D affine parameter of shape [hidden_size]
        beta: 1D affine parameter of shape [hidden_size]
        output_tensor: 2D output tensor of shape [sequence_length, hidden_size]
        epsilon: Small constant for numerical stability
    """
    seq_len = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    
    # Load gamma and beta, perform partition-axis broadcast
    shift_scale_tensor = nl.load(gamma)
    shift_scale_tensor = nl.broadcast_to(shift_scale_tensor, (nl.tile_size.pmax, hidden_size))
    
    beta_tensor = nl.load(beta)
    beta_tensor = nl.broadcast_to(beta_tensor, (nl.tile_size.pmax, hidden_size))
    
    # Compute loop over partition axis
    for i in nl.affine_range(math.ceil(seq_len / nl.tile_size.pmax)):
        # Load one tile of input_tensor
        input_tile = nl.load(
            input_tensor,
            indices=[i * nl.tile_size.pmax, 0],
            mask=(i * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None] < seq_len)
        )
        
        # Calculate mean and variance using bn_stats and bn_aggr
        mean_accum = nl.zeros((nl.tile_size.pmax, 1), dtype=input_tile.dtype)
        var_accum = nl.zeros((nl.tile_size.pmax, 1), dtype=input_tile.dtype)
        
        for j in nl.affine_range(math.ceil(hidden_size / nl.tile_size.bn_stats_fmax)):
            start_idx = j * nl.tile_size.bn_stats_fmax
            end_idx = min(start_idx + nl.tile_size.bn_stats_fmax, hidden_size)
            tile_size = end_idx - start_idx
            
            input_slice = input_tile[:, start_idx:end_idx]
            
            # Use bn_stats to compute partial statistics
            stats = isa.bn_stats(input_slice)
            mean_accum += stats[0]
            var_accum += stats[1]
        
        # Aggregate statistics
        mean = isa.bn_aggr(mean_accum, hidden_size)
        variance = isa.bn_aggr(var_accum, hidden_size)
        
        # Perform shift and scale in a single instruction
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
            output_tensor,
            indices=[i * nl.tile_size.pmax, 0],
            value=output_tile,
            mask=(i * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None] < seq_len)
        )
```

## PyTorch Reference Implementation

```python
import torch
import torch.nn.functional as F

def layernorm_torch_reference(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    PyTorch reference implementation of LayerNorm.
    
    Args:
        input_tensor: 2D input tensor of shape [sequence_length, hidden_size]
        gamma: 1D affine parameter of shape [hidden_size]
        beta: 1D affine parameter of shape [hidden_size]
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized output tensor of shape [sequence_length, hidden_size]
    """
    # Compute mean and variance along the hidden_size dimension (axis=1)
    mean = torch.mean(input_tensor, dim=1, keepdim=True)
    variance = torch.var(input_tensor, dim=1, keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (input_tensor - mean) / torch.sqrt(variance + epsilon)
    
    # Scale and shift
    output = normalized * gamma + beta
    
    return output
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
```

```python
# Trace the distributed model with tensor parallelism
model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2)

# Save the traced model
neuronx_distributed.trace.parallel_model_save(model, "tp_models")

# Load and run inference
model = neuronx_distributed.trace.parallel_model_load("tp_models")
```

## tutorial-neuron-monitor-mnist.rst

```python
for run in range(0, 1000):
    print(f'Run {run}')
    model.train()
```

**Note:** The file references a full training script at `/src/examples/pytorch/mnist_mlp/train_monitor.py` via `literalinclude` directive, but the actual code content is not provided in the excerpt. The only explicit code example shown is the training loop repetition above. The rest of the document contains setup instructions for Prometheus and Grafana (bash/yml configuration) and usage guidance, but no additional Python API code examples for NKI kernel optimization or Neuron-specific APIs are present in the provided excerpt.

## tensorflow-neuronx-serving-tutorial.rst

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
        if inputs == "stop":
            break
        inputs = {key: mx.nd.array(v) for key, v in inputs.items()}
        start = time()
        results = model.forward(**inputs)
        results[0].wait_to_read()

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

## compile.py

```python
import torch
from torch_neuron import trace
from torch_neuronx.xla_impl.trace import trace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Configure NeuronCore allocation
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Prepare example inputs for tracing
sequence_0 = "The company HuggingFace is based in New York City"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
max_length = 128
batch_size = 6

paraphrase = tokenizer.encode_plus(
    sequence_0, 
    sequence_2, 
    max_length=max_length, 
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

example_inputs_paraphrase = (
    torch.cat([paraphrase['input_ids']] * batch_size, 0),
    torch.cat([paraphrase['attention_mask']] * batch_size, 0),
    torch.cat([paraphrase['token_type_ids']] * batch_size, 0)
)

# Compile model for Trainium/Inferentia using trace
model_neuron = trace(model, example_inputs_paraphrase)

# Execute compiled model
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
import tensorflow
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

## matrix_multiplication_torch.py

```python
import torch
from torch_xla.core import xla_model as xm

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_
```

```python
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

## getting_started_baremetal.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl
```

```python
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

def jax_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return jnp.nanmean(reshaped, axis=(2, 4))
```

```python
out_nki = tensor_avgpool_kernel(in_array, pool_size=POOL_SIZE)
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
```

```python
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

## transpose2d_torch.py

```python
import torch
from torch_xla.core import xla_model as xm
```

```python
def transpose2d_pytorch_usage():
    device = xm.xla_device()

    P, X, Y = 5, 3, 4
    a = torch.arange(P*X*Y, dtype=torch.int8).reshape((P, X*Y)).to(device=device)
    a_t_nki = torch.zeros((P, Y*X), dtype=torch.int8).to(device=device)

    a_t_nki = tensor_transpose2D_kernel_(a, (X, Y))

    a_t_torch = torch.transpose(a.reshape(P, X, Y), 1, 2).reshape(P, X * Y)
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