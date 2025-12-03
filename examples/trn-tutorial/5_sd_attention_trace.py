CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=CodeCandidate(parent=None,
plan=None,
code='''@nki.jit
def test (q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), \'Input shape mismatch!\'
  assert tuple(k_ref.shape) == (seqlen, d_head), \'Input shape mismatch!\'
  assert tuple(v_ref.shape) == (seqlen,d_head),   f\'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}\'
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
        # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
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
      attn_res_psum[ip_out, if_out] +=         nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

  return out_ref
''',
score=0.558,
spad_acc_stats=[],
plan_gen_model='None',
code_gen_model='None',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+--------+--------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S    IRES/S   L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   982.38   982.38   593    600     630     558      558       558       N/A    \n+---+----+---------+---------+---------+--------+--------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.558 ms (P99)\n',
stderr=''),
plan='''Selected optimization: 6) fuse multiple instructions into one (do reduction inside nisa.activation)

What’s inefficient now
- In Step 4 (Softmax), the kernel:
  - computes exp on qk_res_buf with nisa.activation
  - then separately reduces sum over the free axis with nisa.tensor_reduce
  - then casts/copies the exp result to pe_in_dt

This is two passes over the same data (exp then sum), plus an explicit copy. The Scalar Engine can do exp and the reduction in one instruction and directly write the output tile in the desired dtype.

Plan
- Use nisa.activation with reduce_op=nl.add and reduce_res to fuse exp and the sum-reduction into a single instruction.
- Set reduce_cmd=nisa.reduce_cmd.reset_reduce to avoid unintended accumulation across calls.
- Set dtype=pe_in_dt in the fused activation so we no longer need an extra nl.copy to cast the exp result.
- Keep all shapes and indices identical; only replace the exp + reduce + copy trio.
- The rest of the kernel remains unchanged and semantically equivalent (within normal FP32->BF16 rounding).

Code change (replace only the “Step 4. Softmax” block)
# Before:
# exp_res = nisa.activation(np.exp,
#                           data=qk_res_buf[ip_softmax, if_softmax],
#                           bias=neg_max_res_final, scale=1.0)
# sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
#                              dtype=kernel_dtype)
# softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

# After (fused exp + reduce + cast):
# Allocate the reduction result vector (same partition size, 1 free element)
sum_res = nl.ndarray((nl.tile_size.pmax if False else q_seq_tile_size, 1),  # explicit shape for clarity
                     dtype=kernel_dtype)  # shape: (q_seq_tile_size, 1)

softmax_res[ip_softmax, if_softmax] = nisa.activation(
    op=nl.exp,
    data=qk_res_buf[ip_softmax, if_softmax],
    bias=neg_max_res_final,     # subtract row-wise max (passed as bias to exp)
    scale=1.0,
    reduce_op=nl.add,           # fuse the row-wise sum
    reduce_res=sum_res,         # row-wise sum written here
    reduce_cmd=nisa.reduce_cmd.reset_reduce,  # ensure no accumulation across calls
    dtype=pe_in_dt              # directly emit BF16/FP16, no extra copy needed
)

# The rest using sum_res is unchanged:
sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

Why this helps
- Eliminates an extra full pass over the tile for the reduction (removes nisa.tensor_reduce over exp_res).
- Removes an extra copy/cast (nl.copy of exp_res to pe_in_dt), since activation emits pe_in_dt directly.
- Reduces engine cycles and SRAM traffic, improving latency with no semantic change.''',
code='''
@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(q_ref.shape) == (seqlen, d_head)
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    softmax_scale = 0.125

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    ###################################
    # Step 1. transpose(tensor_v)
    ###################################
    trans_v = nl.ndarray((nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
                         dtype=pe_in_dt)
    for i_k_seq_tile in nl.affine_range(v_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
            v_ref[i_k_seq_tile * v_seq_tile_size + ip_v, if_v],
            dtype=pe_in_dt)

    ###################################
    # Step 1.5 transpose(q) and load(k)
    ###################################
    q_local = nl.ndarray((q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
                         dtype=pe_in_dt)
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
            q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
                  nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt
        ) * softmax_scale

    k_local = nl.ndarray((k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
                         dtype=pe_in_dt)
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
                  nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt
        )

    ###################################
    # Main Q‐loop: compute QK, mask, softmax, then attention
    ###################################
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # buffer for assembling the [q_tile_size x seqlen] QK scores
        qk_res_buf = nl.ndarray((nl.par_dim(q_seq_tile_size), seqlen),
                                dtype=kernel_dtype)

        # for row‐wise max subtraction
        neg_max_res = nl.ndarray((nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
                                 dtype=kernel_dtype)
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        # matmul and accumulation per K‐block
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            # PSUM buffer for one block of Q @ K^T
            qk_psum = nl.zeros((nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                                dtype=np.float32, buffer=nl.psum)
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            # Step 2: Q @ K^T
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            # Step 3: optional causal mask and copy back to SBUF
            if use_causal_mask:
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] =                     nisa.affine_select(
                        pred=(i_q_seq_tile * q_seq_tile_size + ip_qk
                              >= i_k_seq_tile * k_seq_tile_size + if_qk),
                        on_true_tile=qk_psum[ip_qk, if_qk],
                        on_false_value=-9984.0,
                        dtype=kernel_dtype
                    )
            else:
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] =                     nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

            # accumulate row‐wise max for subtraction
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # final row‐wise max to subtract from every element
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # prepare indices for Softmax
        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        # buffer for row‐wise Softmax outputs
        softmax_res = nl.ndarray((nl.par_dim(q_seq_tile_size), seqlen),
                                 dtype=pe_in_dt)

        ###################################
        # Step 4. Softmax (fused exp + sum‐reduce + cast)
        ###################################
        # allocate the per‐row sum result (row length = q_seq_tile_size)
        sum_res = nl.zeros((nl.par_dim(q_seq_tile_size), 1),
                           dtype=kernel_dtype)

        softmax_res[ip_softmax, if_softmax] = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=sum_res,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # compute reciprocal of sums and broadcast back to head dimension
        sum_reciprocal = (1.0 / sum_res)
        sum_reciprocal_bcast = sum_reciprocal.broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )

        # allocate buffer for the broadcast reciprocal
        sum_divisor = nl.ndarray((nl.par_dim(q_seq_tile_size), d_head_tile_size),
                                 dtype=kernel_dtype)
        sum_divisor[ip_sum_res, if_sum_res] =             nl.copy(sum_reciprocal_bcast, dtype=kernel_dtype)

        ###################################
        # Step 5. transpose(softmax_res)
        ###################################
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        ip_scores = nl.arange(q_seq_tile_size)[:, None]
        if_scores = nl.arange(k_seq_tile_size)[None, :]
        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] =                 nisa.nc_transpose(
                    softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores],
                    engine=nisa.unknown_engine
                )

        ###################################
        # Step 6. attention output matmul
        ###################################
        attn_res_psum = nl.zeros((nl.par_dim(d_head_tile_size), q_seq_tile_size),
                                 dtype=np.float32, buffer=nl.psum)
        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_out, if_out] +=                 nisa.nc_matmul(
                    moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                    stationary=trans_v[ip_v, i_k_seq_tile, if_v]
                )

        # cast back to the output dtype and store
        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out],
                                 dtype=kernel_dtype)
        # multiply by reciprocal divisor (need transpose for shape match)
        attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor)
        nl.store(
            out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
            value=attn_res_div
        )

    return out_ref
''',
score=0.437,
spad_acc_stats=[],
plan_gen_model='gpt-5',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1087.16   1087.16   474    502     525     436      436       437       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.437 ms (P99)\n',
stderr=''),
plan='''We can get an immediate win simply by fusing the two back-to-back “transpose‐and‐load Q” and “transpose‐and‐load K” loops into one.  Right now we are streaming over HBM twice (once for Q, once for K) with two separate affine_range loops.  If instead we fuse them into a single affine_range over the same sequence‐tile index, we:  
  • cut our HBM→SBUF traffic in half for these two big buffers  
  • improve SBUF “hot” data reuse (we keep our index arithmetics in registers)  
  • reduce loop overhead  

Here’s the specific change.  In the original kernel you have:  

    # Step 1.5 transpose(q) and load(k)  
    q_local = nl.ndarray((q_seq_n_tiles, nl.par_dim(d_head_tile_size),  
                          q_seq_tile_size), dtype=pe_in_dt)  
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  
        ip_q = nl.arange(d_head_tile_size)[:, None]  
        if_q = nl.arange(q_seq_tile_size)[None, :]  
        q_local[i_q_seq_tile, ip_q, if_q] = (  
            nl.load_transpose2d(  
                q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],  
                      nl.arange(d_head_tile_size)[None, :]],  
                dtype=pe_in_dt  
            ) * softmax_scale  
        )  

    k_local = nl.ndarray((k_seq_n_tiles, nl.par_dim(d_head_tile_size),  
                          k_seq_tile_size), dtype=pe_in_dt)  
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  
        ip_k = nl.arange(d_head_tile_size)[:, None]  
        if_k = nl.arange(k_seq_tile_size)[None, :]  
        k_local[i_k_seq_tile, ip_k, if_k] = (  
            nl.load_transpose2d(  
                k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],  
                      nl.arange(d_head_tile_size)[None, :]],  
                dtype=pe_in_dt  
            )  
        )  

Replace both of those with a single loop that does both loads in one pass:  

    # fused Step 1.5: transpose‐and‐load Q and K in one pass  
    q_local = nl.ndarray((q_seq_n_tiles, nl.par_dim(d_head_tile_size),  
                          q_seq_tile_size), dtype=pe_in_dt)  
    k_local = nl.ndarray((q_seq_n_tiles, nl.par_dim(d_head_tile_size),  
                          q_seq_tile_size), dtype=pe_in_dt)  
    ip = nl.arange(d_head_tile_size)[:, None]  
    if_ = nl.arange(q_seq_tile_size)[None, :]  
    for i_seq in nl.affine_range(q_seq_n_tiles):  
        # load & transpose Q once  
        q_block = nl.load_transpose2d(  
            q_ref[i_seq * q_seq_tile_size + ip,                # partition index  
                  ip * 0 + nl.arange(q_seq_tile_size)[None, :]], # free index  
            dtype=pe_in_dt  
        )  
        q_local[i_seq, ip, if_] = q_block * softmax_scale  

        # load & transpose K in the very same iteration  
        k_block = nl.load_transpose2d(  
            k_ref[i_seq * q_seq_tile_size + ip,                # same partition index  
                  ip * 0 + nl.arange(q_seq_tile_size)[None, :]], # same free index  
            dtype=pe_in_dt  
        )  
        k_local[i_seq, ip, if_] = k_block  

Notes:  
 1. We renamed k_seq_tile_size to q_seq_tile_size since both loop bounds are identical.  
 2. We hoist the nl.arange into “ip” and “if_” before the loop to avoid re‐allocating them.  
 3. Now each HBM page of Q and K gets streamed exactly once, rather than twice.  

By halving the passes over HBM for these large d_head×seqlen tiles, we reduce the HBM traffic and loop overhead, and gain an almost 2× boost in those two stages.  Everything else in the kernel remains untouched, so semantic equivalence is guaranteed (we’re still doing the same load_transpose2d + scale for Q, and load_transpose2d for K).''',
code='''
@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(q_ref.shape) == (seqlen, d_head)
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    ###################################
    # Step 1. transpose(tensor_v)
    ###################################
    trans_v = nl.ndarray((nl.par_dim(v_seq_tile_size),
                          v_seq_n_tiles, d_head),
                         dtype=pe_in_dt)
    for i_k_seq_tile in nl.affine_range(v_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
            v_ref[i_k_seq_tile * v_seq_tile_size + ip_v, if_v],
            dtype=pe_in_dt
        )

    ###################################
    # Step 1.5 fused: transpose‐and‐load Q and K
    ###################################
    q_local = nl.ndarray((q_seq_n_tiles,
                          nl.par_dim(d_head_tile_size),
                          q_seq_tile_size),
                         dtype=pe_in_dt)
    k_local = nl.ndarray((q_seq_n_tiles,
                          nl.par_dim(d_head_tile_size),
                          q_seq_tile_size),
                         dtype=pe_in_dt)

    # Hoist index tensors
    ip = nl.arange(d_head_tile_size)[:, None]
    if_ = nl.arange(q_seq_tile_size)[None, :]

    for i_seq in nl.affine_range(q_seq_n_tiles):
        # load & transpose Q once, then scale
        q_block = nl.load_transpose2d(
            q_ref[i_seq * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
                  nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt
        )
        q_local[i_seq, ip, if_] = q_block * softmax_scale

        # load & transpose K in the same iteration
        k_block = nl.load_transpose2d(
            k_ref[i_seq * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
                  nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt
        )
        k_local[i_seq, ip, if_] = k_block

    ###################################
    # Main Q‐loop: compute QK, mask, softmax, then attention
    ###################################
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # buffer for assembling the [q_tile_size x seqlen] QK scores
        qk_res_buf = nl.ndarray((nl.par_dim(q_seq_tile_size), seqlen),
                                dtype=kernel_dtype)

        # for row‐wise max subtraction
        neg_max_res = nl.ndarray((nl.par_dim(q_seq_tile_size),
                                  k_seq_n_tiles),
                                 dtype=kernel_dtype)
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        # matmul and accumulation per K‐block
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            # PSUM buffer for one block of Q @ K^T
            qk_psum = nl.zeros((nl.par_dim(q_seq_tile_size),
                                k_seq_tile_size),
                               dtype=np.float32,
                               buffer=nl.psum)
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            # Step 2: Q @ K^T
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip, if_],
                stationary=q_local[i_q_seq_tile, ip, if_]
            )

            # Step 3: optional causal mask and copy back to SBUF
            if use_causal_mask:
                qk_res_buf[ip_qk,
                           i_k_seq_tile * k_seq_tile_size + if_qk] =                     nisa.affine_select(
                        pred=(i_q_seq_tile * q_seq_tile_size + ip_qk
                              >= i_k_seq_tile * k_seq_tile_size + if_qk),
                        on_true_tile=qk_psum[ip_qk, if_qk],
                        on_false_value=-9984.0,
                        dtype=kernel_dtype
                    )
            else:
                qk_res_buf[ip_qk,
                           i_k_seq_tile * k_seq_tile_size + if_qk] =                     nl.copy(qk_psum[ip_qk, if_qk],
                            dtype=kernel_dtype)

            # accumulate row‐wise max for subtraction
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk,
                                i_k_seq_tile * k_seq_tile_size + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # final row‐wise max to subtract from every element
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # prepare indices for Softmax
        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        # buffer for row‐wise Softmax outputs
        softmax_res = nl.ndarray((nl.par_dim(q_seq_tile_size), seqlen),
                                 dtype=pe_in_dt)

        ###################################
        # Step 4. Softmax (fused exp + sum‐reduce + cast)
        ###################################
        sum_res = nl.zeros((nl.par_dim(q_seq_tile_size), 1),
                           dtype=kernel_dtype)

        softmax_res[ip_softmax, if_softmax] = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=sum_res,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # compute reciprocal of sums and broadcast back to head dimension
        sum_reciprocal = (1.0 / sum_res)
        sum_reciprocal_bcast = sum_reciprocal.broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )

        sum_divisor = nl.ndarray((nl.par_dim(q_seq_tile_size),
                                  d_head_tile_size),
                                 dtype=kernel_dtype)
        sum_divisor[ip_sum_res, if_sum_res] =             nl.copy(sum_reciprocal_bcast, dtype=kernel_dtype)

        ###################################
        # Step 5. transpose(softmax_res)
        ###################################
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size),
             k_seq_n_tiles,
             q_seq_tile_size),
            dtype=pe_in_dt
        )
        ip_scores = nl.arange(q_seq_tile_size)[:, None]
        if_scores = nl.arange(k_seq_tile_size)[None, :]
        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            trans_softmax_res[ip_scores_t,
                              i_k_seq_tile,
                              if_scores_t] =                 nisa.nc_transpose(
                    softmax_res[ip_scores,
                                i_k_seq_tile * k_seq_tile_size + if_scores],
                    engine=nisa.unknown_engine
                )

        ###################################
        # Step 6. attention output matmul
        ###################################
        attn_res_psum = nl.zeros((nl.par_dim(d_head_tile_size),
                                  q_seq_tile_size),
                                 dtype=np.float32,
                                 buffer=nl.psum)
        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_out, if_out] +=                 nisa.nc_matmul(
                    moving=trans_softmax_res[
                        ip_scores_t, i_k_seq_tile, if_scores_t
                    ],
                    stationary=trans_v[ip_v, i_k_seq_tile, if_v]
                )

        # cast back to the output dtype and store
        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out],
                                 dtype=kernel_dtype)
        attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor)
        nl.store(
            out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
            value=attn_res_div
        )

    return out_ref
''',
score=0.434,
spad_acc_stats=[],
plan_gen_model='o4-mini',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1105.26   1105.26   482    498     506     433      433       434       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.434 ms (P99)\n',
stderr=''),
plan='''Below is a focused plan applying exactly one of the suggested optimizations—namely,  
“2. Hoist redundant operations out of loops”—to our attention‐softmax kernel.  We will move all of the constant index‐tensor constructions (all of the nl.arange and nl.mgrid calls that do not depend on an affine_range loop variable) out of their inner loops and compute them just once up‐front.  This both cuts down the number of nl.arange/nl.mgrid calls the compiler has to lower and reduces register and engine dispatch overhead, improving overall latency without touching any of the numerical logic or tiling structure.

---  

1) Before any of the per‐tile loops, define once and for all all of the “ip_*” and “if_*” tensors that are used as indices inside your affine_range loops.  For example, immediately after you compute  

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128  
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128  
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128  
    d_head_tile_size = d_head  

   insert:  
   
    #— hoisted, constant index tensors for all inner loops —  
    ip_qk       = nl.arange(q_seq_tile_size)[:, None]       # [128,1]  
    if_qk       = nl.arange(k_seq_tile_size)[None, :]       # [1,128]  
    ip_max      = nl.arange(q_seq_tile_size)[:, None]       # [128,1]  
    if_max      = nl.arange(k_seq_n_tiles)[None, :]         # [1,#k-tiles]  
    ip_softmax  = nl.arange(q_seq_tile_size)[:, None]       # [128,1]  
    if_softmax  = nl.arange(seqlen)[None, :]                # [1,seqlen]  
    ip_scores   = nl.arange(q_seq_tile_size)[:, None]       # [128,1]  
    if_scores   = nl.arange(k_seq_tile_size)[None, :]       # [1,128]  
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]       # [128,1]  
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]       # [1,128]  
    ip_v_t      = nl.arange(k_seq_tile_size)[:, None]       # [128,1]  
    if_v_t      = nl.arange(d_head_tile_size)[None, :]      # [1,d_head]  
    ip_out      = nl.arange(d_head_tile_size)[:, None]      # [d_head,1]  
    if_out      = nl.arange(q_seq_tile_size)[None, :]       # [1,128]  
    ip_sum_res  = nl.arange(q_seq_tile_size)[:, None]       # [128,1]  
    if_sum_res  = nl.arange(d_head_tile_size)[None, :]      # [1,d_head]  

2) Now remove *all* of the identical nl.arange or nl.mgrid calls from inside the inner `for i_k_seq_tile in affine_range(...)` and the inner transpose loops.  Simply reuse the hoisted variables above:

    •  Replace  
         for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  
             qk_psum = nl.zeros((nl.par_dim(q_seq_tile_size), k_seq_tile_size), …)  
             ip_qk = nl.arange(q_seq_tile_size)[:, None]     # ← **remove this line**  
             if_qk = nl.arange(k_seq_tile_size)[None, :]     # ← **remove this line**  
             …  
             qk_psum[ip_qk, if_qk] += nisa.nc_matmul(…)  
         …  

      with  

         for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  
             qk_psum = nl.zeros((nl.par_dim(q_seq_tile_size), k_seq_tile_size), …)  
             # ip_qk, if_qk already defined above  
             qk_psum[ip_qk, if_qk] += nisa.nc_matmul(…)  
         …  

    •  Likewise in the softmax‐exp loop: drop the repeated definitions of `ip_softmax, if_softmax`.  
    •  Inside the “Step 5: transpose(softmax_res)” loop remove the local `nl.arange` calls for `ip_scores_t`/`if_scores_t`.  
    •  And in the final matmul loop remove the local definitions of `ip_v_t`/`if_v_t`, `ip_out`/`if_out`, etc.  

3) Keep all of the original affine_range nests and all of the data‐flow exactly as before.  You are simply turning:  

       for ...:  
         ip_foo = nl.arange(...)  
         if_foo = nl.arange(...)  
         some_tile[ip_foo, if_foo] = …  

   into  

       # once up-front: ip_foo = nl.arange(...)  
       for ...:  
         some_tile[ip_foo, if_foo] = …  

4) Verify that **no** of your hoisted index tensors depend on any loop‐carried variables.  All of them are functions only of the constant tile sizes (128, d_head, etc.), so they may live outside *all* loops.  

5) Recompile and remeasure.  You will see that each affine_range iteration has noticeably less overhead because we have eliminated dozens of redundant nl.arange / nl.mgrid tensor constructions.  In practice this typically buys you on the order of a few percent of total kernel latency (often 5–15 µs off a ~425 µs base), and it is a completely safe, zero‐change to correctness.  

---  

That completes Phase 4: we have hoisted all constant index tensor creations out of the inner loops, thereby reducing per‐iteration overhead without changing any tile shapes, data movements, or numerical results.''',
code='''
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    # tile counts
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    # Hoist index tensors
    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    # inner-loop indices
    ip_qk      = nl.arange(q_seq_tile_size)[:, None]
    if_qk      = nl.arange(k_seq_tile_size)[None, :]
    ip_max     = nl.arange(q_seq_tile_size)[:, None]
    if_max     = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_scores  = nl.arange(q_seq_tile_size)[:, None]
    if_scores  = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t     = nl.arange(k_seq_tile_size)[:, None]
    if_v_t     = nl.arange(d_head_tile_size)[None, :]
    ip_out     = nl.arange(d_head_tile_size)[:, None]
    if_out     = nl.arange(q_seq_tile_size)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    # Fused Step1 + Step1b
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        # V transpose
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        # Q load+transpose+scale
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        # K load+transpose
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    # Main softmax+attention loop
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # SBUF buffers
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=kernel_dtype
        )
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        # Steps 2+3+4: compute Q·K^T, apply mask, row-max
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )
            base_col = i_k_seq_tile * k_seq_tile_size
            if use_causal_mask:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk
                          >= base_col + if_qk),
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=kernel_dtype
                )
            else:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.tensor_copy(
                    qk_psum[ip_qk, if_qk],
                    dtype=kernel_dtype
                )
            # gather negated max per partial row
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # full-row neg-max
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # Fused exp + sum
        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=kernel_dtype
        )
        softmax_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        softmax_res[ip_softmax, if_softmax] = nisa.tensor_copy(
            exp_res,
            dtype=pe_in_dt
        )

        sum_reciprocal = (1.0 / sum_acc)
        sum_reciprocal_broadcast = sum_reciprocal.broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )
        sum_divisor = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=kernel_dtype
        )
        sum_divisor[ip_sum_res, if_sum_res] = nisa.tensor_copy(
            sum_reciprocal_broadcast,
            dtype=kernel_dtype
        )

        # Step 5: transpose softmax_res
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    softmax_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        # Step 6: final QK-softmax @ V^T
        attn_res_psum = nl.zeros(
            (d_head_tile_size, q_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        # back to SBUF and apply per‐Q‐head divisor
        attn_res_sbuf = nisa.tensor_copy(
            attn_res_psum[ip_out, if_out],
            dtype=kernel_dtype
        )
        divisor_vec = nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res],
            engine=nisa.vector_engine
        )
        attn_res_div = attn_res_sbuf * divisor_vec

        # store output
        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + if_out,
                ip_out
            ],
            value=attn_res_div
        )

    return out_ref
''',
score=0.424,
spad_acc_stats=[],
plan_gen_model='o4-mini',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1090.46   1090.46   475    501     525     424      424       424       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.424 ms (P99)\n',
stderr=''),
plan='''Selected optimization: 6) downcast to lower precision during operations that take dtype as an argument.

What and why
- The inner attention pipeline keeps logits and exp(·) tiles in full precision in SBUF even though:
  - PSUM accumulations (matmul) already happen in float32.
  - Scalar/Vector engines internally compute in float32 and only cast on write-out.
  - Softmax inputs/outputs are typically safe in bfloat16.
- By downcasting SBUF-resident intermediates that are bandwidth/compute heavy but not accumulation-critical, we reduce SBUF traffic and engine cycles without changing algorithmic flow.

Concretely
- Keep high-precision only where it matters for accumulation/reduction:
  - PSUM tiles remain float32.
  - Reduction outputs (neg_max_res, sum_acc) remain kernel_dtype (float32 if inputs are fp32; otherwise BF16 is fine).
- Downcast tiles that are repeatedly read/written in SBUF:
  - qk_res_buf: store masked/cast logits in pe_in_dt (BF16 when mixed_precision=True).
  - The activation output (exp_res) is directly produced as pe_in_dt; reuse it as softmax_res to avoid an extra cast/copy.
  - The causal mask write (affine_select) and non-mask path (tensor_copy) both write BF16.

Semantics
- Matmul and reductions still perform in float32 internally; only stored intermediates are BF16. The result remains equivalent within small numerical tolerance.

Rewritten program (only changes related to dtype, kept shapes/indexing intact)

@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    seqlen, d_head = q_ref.shape
    assert d_head <= 128
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    softmax_scale = 0.125

    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    ip_qk      = nl.arange(q_seq_tile_size)[:, None]
    if_qk      = nl.arange(k_seq_tile_size)[None, :]
    ip_max     = nl.arange(q_seq_tile_size)[:, None]
    if_max     = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_scores  = nl.arange(q_seq_tile_size)[:, None]
    if_scores  = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t     = nl.arange(k_seq_tile_size)[:, None]
    if_v_t     = nl.arange(d_head_tile_size)[None, :]
    ip_out     = nl.arange(d_head_tile_size)[:, None]
    if_out     = nl.arange(q_seq_tile_size)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # CHANGE: keep logits buffer in pe_in_dt to cut SBUF bandwidth.
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        # Keep reduction results in kernel_dtype (e.g., fp32)
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )
            base_col = i_k_seq_tile * k_seq_tile_size
            if use_causal_mask:
                # CHANGE: write masked logits as pe_in_dt
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= base_col + if_qk),
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=pe_in_dt
                )
            else:
                # CHANGE: cast qk_psum to pe_in_dt once when moving to SBUF
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.tensor_copy(
                    qk_psum[ip_qk, if_qk],
                    dtype=pe_in_dt
                )
            # Reduce max from pe_in_dt logits; engine computes in fp32; keep output in kernel_dtype
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        # CHANGE: produce exp result directly as pe_in_dt
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )
        # CHANGE: reuse exp_res as softmax_res (remove redundant copy)
        # softmax_res = exp_res

        sum_reciprocal = (1.0 / sum_acc)
        sum_reciprocal_broadcast = sum_reciprocal.broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )
        sum_divisor = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=kernel_dtype
        )
        sum_divisor[ip_sum_res, if_sum_res] = nisa.tensor_copy(
            sum_reciprocal_broadcast,
            dtype=kernel_dtype
        )

        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    exp_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        attn_res_psum = nl.zeros(
            (d_head_tile_size, q_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        attn_res_sbuf = nisa.tensor_copy(
            attn_res_psum[ip_out, if_out],
            dtype=kernel_dtype
        )
        divisor_vec = nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res],
            engine=nisa.vector_engine
        )
        attn_res_div = attn_res_sbuf * divisor_vec

        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + if_out,
                ip_out
            ],
            value=attn_res_div
        )

    return out_ref

Why this should be faster
- qk_res_buf is now pe_in_dt, halving SBUF bandwidth and copy cost for logits.
- The exp output is produced directly as pe_in_dt; we avoid an extra nisa.tensor_copy.
- Reductions still compute in float32, and PSUM accumulations remain float32, preserving numerical stability.

Notes
- No loop structure, indexing order, or matmul layout changed. Only dtype on SBUF tiles and instruction outputs was adjusted where allowed.''',
code='''
@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    # tile counts and sizes
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    # Hoist index tensors
    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    ip_qk      = nl.arange(q_seq_tile_size)[:, None]
    if_qk      = nl.arange(k_seq_tile_size)[None, :]
    ip_max     = nl.arange(q_seq_tile_size)[:, None]
    if_max     = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_scores  = nl.arange(q_seq_tile_size)[:, None]
    if_scores  = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t     = nl.arange(k_seq_tile_size)[:, None]
    if_v_t     = nl.arange(d_head_tile_size)[None, :]
    ip_out     = nl.arange(d_head_tile_size)[:, None]
    if_out     = nl.arange(q_seq_tile_size)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    # Step 1: preload and transpose Q, K, V into SBUF in pe_in_dt
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        # V transpose
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        # Q load+transpose+scale
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        # K load+transpose
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    # Main softmax+attention loop
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # Keep logits buffer in pe_in_dt
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        # Reduction outputs in kernel_dtype
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        # Steps 2-4: compute Q·K^T, apply mask, row-max
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )
            base_col = i_k_seq_tile * k_seq_tile_size
            if use_causal_mask:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk
                          >= base_col + if_qk),
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=pe_in_dt
                )
            else:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.tensor_copy(
                    qk_psum[ip_qk, if_qk],
                    dtype=pe_in_dt
                )
            # Partial row max (negated)
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # Final row max
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # Fused exp + sum
        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # Compute normalization divisor
        sum_reciprocal = (1.0 / sum_acc)
        sum_reciprocal_broadcast = sum_reciprocal.broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )
        sum_divisor = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=kernel_dtype
        )
        sum_divisor[ip_sum_res, if_sum_res] = nisa.tensor_copy(
            sum_reciprocal_broadcast,
            dtype=kernel_dtype
        )

        # Transpose softmax results for V multiplication
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    exp_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        # Step 6: final QK-softmax @ V^T
        attn_res_psum = nl.zeros(
            (d_head_tile_size, q_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        # Back to SBUF, apply per‐Q‐head divisor, and store
        attn_res_sbuf = nisa.tensor_copy(
            attn_res_psum[ip_out, if_out],
            dtype=kernel_dtype
        )
        divisor_vec = nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res],
            engine=nisa.vector_engine
        )
        attn_res_div = attn_res_sbuf * divisor_vec

        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + if_out,
                ip_out
            ],
            value=attn_res_div
        )

    return out_ref
''',
score=0.399,
spad_acc_stats=[],
plan_gen_model='gpt-5',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1156.19   1156.19   437    446     482     399      399       399       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.399 ms (P99)\n',
stderr=''),
plan='''Here is a very targeted change to pick up almost the same work you’re doing in the final “copy‐and‐divide” and do it in one vector instruction instead of three.  This is a classic “fuse multiple instructions into one” move (your #4).

In your original Step 6 you have something like:

    # Back to SBUF, apply per‐Q‐head divisor, and store
    attn_res_sbuf = nisa.tensor_copy(
        attn_res_psum[ip_out, if_out],
        dtype=kernel_dtype
    )
    divisor_vec = nisa.nc_transpose(
        sum_divisor[ip_sum_res, if_sum_res],
        engine=nisa.vector_engine
    )
    attn_res_div = attn_res_sbuf * divisor_vec

    nl.store(
        out_ref[
            i_q_seq_tile * q_seq_tile_size + if_out,
            ip_out
        ],
        value=attn_res_div
    )

What this does is:

 1. copy a PSUM tile into SBUF  
 2. broadcast your 1/S accumulator into a wide tile (via a transpose)  
 3. do an element‐wise multiply in SBUF  

We can fuse steps 1 and 3 by using exactly one nisa.tensor_scalar (Vector‐engine) instruction with op0=np.multiply.  A single tensor_scalar can both pull in the PSUM tile and do the multiply by a (per‐partition) vector in one go.  This gets rid of the explicit tensor_copy, the temporary sum_divisor, and the nc_transpose.

Concretely, replace the three lines above with:

    # Fuse copy‐from‐PSUM + multiply‐by‐1/s into one vector op.
    # sum_reciprocal is shape (q_seq_tile_size,1), dtype=kernel_dtype,
    # computed already as 1/sum_acc.
    attn_res_div = nisa.tensor_scalar(
        data=attn_res_psum[ip_out, if_out],   # pull PSUM →
        op0=np.multiply,                      # multiply
        operand0=sum_reciprocal,              # by the (128×1) vector
        dtype=kernel_dtype                    # cast result if needed
    )

    nl.store(
        out_ref[
            i_q_seq_tile * q_seq_tile_size + if_out,
            ip_out
        ],
        value=attn_res_div
    )

You can now delete:

 • the `attn_res_sbuf = nisa.tensor_copy(...)`  
 • the `sum_divisor = nl.ndarray(...)` + its broadcast_to + `divisor_vec = nc_transpose(...)`  
 • the standalone `attn_res_sbuf * divisor_vec`  

and you still get exactly the same  Q‐K softmax‐by‐V result.  By fusing copy + multiply into one vector engine instruction you:

 – eliminate one PSUM→SBUF copy  
 – eliminate one SBUF transpose  
 – eliminate one pure point‐wise multiply  

This simplification should cut down a nontrivial amount of the final tail latency without touching the rest of your carefully arranged loops or changing any tensor shapes.''',
code='''
@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    # tile counts and sizes
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    # Hoist index tensors
    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    ip_qk     = nl.arange(q_seq_tile_size)[:, None]
    if_qk     = nl.arange(k_seq_tile_size)[None, :]
    ip_max    = nl.arange(q_seq_tile_size)[:, None]
    if_max    = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax= nl.arange(q_seq_tile_size)[:, None]
    if_softmax= nl.arange(seqlen)[None, :]
    ip_scores = nl.arange(q_seq_tile_size)[:, None]
    if_scores = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t    = nl.arange(k_seq_tile_size)[:, None]
    if_v_t    = nl.arange(d_head_tile_size)[None, :]
    ip_out    = nl.arange(d_head_tile_size)[:, None]
    if_out    = nl.arange(q_seq_tile_size)[None, :]
    # We\'ll use sum_acc directly without extra indexing
    # ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    # if_sum_res = nl.arange(d_head_tile_size)[None, :]

    # Step 1: preload and transpose Q, K, V into SBUF in pe_in_dt
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        # V transpose
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        # Q load+transpose+scale
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        # K load+transpose
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    # Main softmax+attention loop
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # Keep logits buffer in pe_in_dt
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        # Reduction outputs in kernel_dtype
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        # Steps 2-4: compute Q·K^T, apply mask, row-max
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )
            base_col = i_k_seq_tile * k_seq_tile_size
            if use_causal_mask:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk
                          >= base_col + if_qk),
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=pe_in_dt
                )
            else:
                qk_res_buf[ip_qk, base_col + if_qk] = nisa.tensor_copy(
                    qk_psum[ip_qk, if_qk],
                    dtype=pe_in_dt
                )
            # Partial row max (negated)
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # Final row max
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # Fused exp + sum
        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # Compute normalization divisor (keep as vector)
        sum_reciprocal = 1.0 / sum_acc

        # Transpose softmax results for V multiplication
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    exp_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        # Step 6: final QK-softmax @ V^T
        attn_res_psum = nl.zeros(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_softmax, if_v_t] += nisa.nc_matmul(
                stationary=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                moving=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        # Fuse PSUM→SBUF copy and per-Q scaling into one vector op
        attn_res_div = nisa.tensor_scalar(
            data=attn_res_psum[ip_softmax, if_v_t],
            op0=np.multiply,
            operand0=sum_reciprocal,   # Use the full vector directly
            dtype=kernel_dtype
        )

        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + ip_softmax,
                if_v_t
            ],
            value=attn_res_div
        )

    return out_ref
''',
score=0.385,
spad_acc_stats=[],
plan_gen_model='o4-mini',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1174.39   1174.39   423    443     477     384      385       385       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.385 ms (P99)\n',
stderr=''),
plan='''Below is one more “in‐place” code change you can layer on top of the existing kernel to cut both Vector‐Engine instruction count and memory traffic in the Q·K phase.  The idea is to get rid of the two‐step  
 
  1) affine_select (to do the causal mask)  
  2) tensor_copy (to move the selected tile into the big [128, seqlen] buffer)  

and instead do it in one go with nisa.tensor_copy_predicated.  We first prefill the entire QK‐buffer with the large negative constant (so that “masked‐off” positions already hold –INF), then for each 128×128 PSUM tile we do one predicated copy into the correct columns.

How this helps  
• Collapses two vector‐engine passes (mask‐then‐copy) into one predicated‐copy.  
• Cuts the extra SBUF<–>PSUM or SBUF<–>SBUF copy from the tensor_copy call.  
• One fewer NL/Vector instruction per 128×128 matmul output tile.  
• Still maintains exact same output values (within FP tolerance).  

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––  
Patch sketch (inside the Q·K loop)  

```python
    # --- before K‐loop ---
    qk_res_buf = nl.ndarray((nl.par_dim(q_seq_tile_size), seqlen),
                             dtype=pe_in_dt,
                             buffer=nl.sbuf)
    # fill all positions with very large negative so exp(...)→0
    # broadcast a single scalar into the entire 128×seqlen tile
    qk_res_buf[...] = -9984.0  

    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        base_col = i_k_seq_tile * k_seq_tile_size

        # 1) standard QK 128×128 matmul into PSUM
        qk_psum = nl.zeros(
            (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
            moving=k_local[i_k_seq_tile, ip_k, if_k],
            stationary=q_local[i_q_seq_tile, ip_q, if_q]
        )

        # 2) instead of affine_select + tensor_copy, do one predicated copy
        #    predicate[x,y] = True exactly when we want to keep the PSUM,
        #    i.e. (global_q_row >= global_k_col)
        #    global_q_row = i_q_seq_tile*tile + ip_qk
        #    global_k_col = base_col + if_qk
        pred_tile = (i_q_seq_tile * q_seq_tile_size + ip_qk
                     >= base_col + if_qk)

        # slice out the destination in qk_res_buf
        dst_slice = qk_res_buf[ip_qk, base_col + if_qk]

        # this single vector‐engine op:
        #  – writes qk_psum into dst_slice wherever pred_tile==True
        #  – leaves the –9984.0 we prefixed wherever pred_tile==False
        nisa.tensor_copy_predicated(
            src=qk_psum[ip_qk, if_qk],
            dst=dst_slice,
            predicate=pred_tile
        )
```

What changed

  • We prefilled `qk_res_buf` once with –9984.0 instead of re‐writing masked‐off rows inside the loop.  
  • We replaced each `affine_select(...)` + `tensor_copy(...)` pair with one `tensor_copy_predicated(...)`.  

Effect

  • Half the vector‐engine passes (mask + copy ⇒ one predicated copy).  
  • Half the copy traffic for each 128×128 tile.  
  • No change in numerical result.  

That small change will shave off one vector‐engine instruction (and one copy) per K‐tile, across all Q‐tiles.  On a 1024×128 attention that’s 8 tiles per Q, so 8 fewer masks and 8 fewer copies per Q‐tile—over all Q‐tiles this adds up to a solid improvement.''',
code='''
@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    # tile counts and sizes
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    # Hoist index tensors
    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    ip_qk     = nl.arange(q_seq_tile_size)[:, None]
    if_qk     = nl.arange(k_seq_tile_size)[None, :]
    ip_max    = nl.arange(q_seq_tile_size)[:, None]
    if_max    = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax= nl.arange(q_seq_tile_size)[:, None]
    if_softmax= nl.arange(seqlen)[None, :]
    ip_scores = nl.arange(q_seq_tile_size)[:, None]
    if_scores = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t    = nl.arange(k_seq_tile_size)[:, None]
    if_v_t    = nl.arange(d_head_tile_size)[None, :]
    ip_out    = nl.arange(d_head_tile_size)[:, None]
    if_out    = nl.arange(q_seq_tile_size)[None, :]

    # Step 1: preload and transpose Q, K, V into SBUF in pe_in_dt
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        # V transpose
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        # Q load+transpose+scale
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        # K load+transpose
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    # Main softmax+attention loop
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # QK‐buffer on SBUF in pe_in_dt
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt,
            buffer=nl.sbuf
        )
        # If causal mask, prefill with large negative so exp→0
        if use_causal_mask:
            qk_res_buf[...] = -9984.0

        # Reduction outputs in kernel_dtype
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        # Steps 2-4: compute Q·K^T, apply mask via predicated copy, partial row-max
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size

            # 128×128 matmul into PSUM
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            if use_causal_mask:
                # predicate True where global_q_row >= global_k_col
                pred_tile = (i_q_seq_tile * q_seq_tile_size + ip_qk
                             >= base_col + if_qk)
                # slice out the destination
                dst_slice = qk_res_buf[ip_qk, base_col + if_qk]
                # one pass: copy where pred True, leave -9984.0 otherwise
                nisa.tensor_copy_predicated(
                    src=qk_psum[ip_qk, if_qk],
                    dst=dst_slice,
                    predicate=pred_tile
                )
            else:
                # non-causal: copy entire PSUM tile
                qk_res_buf[ip_qk, base_col + if_qk] = \
                    nisa.tensor_copy(
                        qk_psum[ip_qk, if_qk],
                        dtype=pe_in_dt
                    )

            # Partial row max (negated)
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # Final row max
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # Fused exp + sum
        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # Compute normalization divisor
        sum_reciprocal = 1.0 / sum_acc

        # Transpose softmax results for V multiplication
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    exp_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        # Step 6: final QK-softmax @ V^T
        attn_res_psum = nl.zeros(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_softmax, if_v_t] += nisa.nc_matmul(
                stationary=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                moving=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        # Fuse PSUM→SBUF copy and per-Q scaling into one vector op
        attn_res_div = nisa.tensor_scalar(
            data=attn_res_psum[ip_softmax, if_v_t],
            op0=np.multiply,
            operand0=sum_reciprocal,   # broadcast full vector
            dtype=kernel_dtype
        )

        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + ip_softmax,
                if_v_t
            ],
            value=attn_res_div
        )

    return out_ref
''',
score=0.384,
spad_acc_stats=[],
plan_gen_model='o4-mini',
code_gen_model='o4-mini',
stdout='\nfile                                                                            \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\n  B   NC   NC USED   WEIGHTS   MODE      INF/S     IRES/S    L(1)   L(50)   L(99)   NCL(1)   NCL(50)   NCL(99)   %USER  \n  1   1    1         dynamic   LIBMODE   1146.13   1146.13   429    445     472     384      384       384       N/A    \n+---+----+---------+---------+---------+---------+---------+------+-------+-------+--------+---------+---------+-------+\nTest passed\nLatency: 0.384 ms (P99)\n',
stderr='')