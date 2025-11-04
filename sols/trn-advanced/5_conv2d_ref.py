def div_ceil(n, d):
  return (n + d - 1) // d

def create_indices(*tripcounts):
  rank = len(tripcounts)
  # rank needs to be reduced by 1 if last dim is 1
  # Note: may need to find all 1s toward the end
  if tripcounts[-1] == 1:
    assert tripcounts[-2] != 1, "Unhandled case"
    rank-=1
  indices = map(lambda c: nl.arange(c) if c > 1 else 0, tripcounts)

  indices = []
  colon = slice(None, None, None)
  cur_rank = 0
  for c in tripcounts:
    if c > 1:
      access = [None] * rank
      access[cur_rank] = colon
      indices.append(nl.arange(c)[tuple(access)])
    else:
      indices.append(0)
    cur_rank += 1
  return indices

def transpose_to_last_dim(src, dim, dst=None):
  if dst is None:
    new_shape = get_3d_shape(src, dim)
    transposed_shape = (new_shape[0], new_shape[2], new_shape[1])
    dst = nl.ndarray(shape=transposed_shape, buffer=nl.hbm, dtype=src.dtype)
  transpose_to_last_dim_kernel(src, dim, dst)
  return dst

def transpose_to_last_dim_kernel(ref, dim, dst):
  assert len(ref.shape) >= 2
  assert dim != len(ref.shape) - 1

  ref = ref.reshape(get_3d_shape(ref, dim))
  transposed_shape = (ref.shape[0], ref.shape[2], ref.shape[1])
  transpose_nonlocal = dst.reshape(transposed_shape)

  D0, B, N = ref.shape
  B_tile_size = min(128, B)
  N_tile_size = min(128, N)
  B_num_tiles = div_ceil(B, B_tile_size)
  N_num_tiles = div_ceil(N, N_tile_size)
  for d0 in nl.affine_range(D0):
    for b_out_tile in nl.affine_range(B_num_tiles):
      for n_out_tile in nl.affine_range(N_num_tiles):
        _local = nl.ndarray(shape=(B_tile_size, N_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='local')
        transposed_local = nl.ndarray(shape=(par_dim(N_tile_size), B_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='transposed_local')
        i = nl.arange(0, B_tile_size)[:, None]
        j = nl.arange(0, N_tile_size)[None, :]
        mask = (b_out_tile * B_tile_size + i < B) & (n_out_tile * N_tile_size + j < N)
        #TODO: maybe better performance by refetching the ref tensor
        _local[i, j] = nl.load(ref[d0, b_out_tile * B_tile_size + i, n_out_tile * N_tile_size + j], mask=mask)

        p = nl.arange(0, N_tile_size)[:, None]
        q = nl.arange(0, B_tile_size)[None, :]
        transposed_local[p, q] = nisa.nc_transpose(_local[i, j], mask=mask)

        mask = (b_out_tile * B_tile_size + q < B) & (n_out_tile * N_tile_size + p < N)
        nl.store(transpose_nonlocal[d0, n_out_tile * N_tile_size + p, b_out_tile * B_tile_size + q], transposed_local[p, q], mask=mask)


def get_3d_shape(ref, dim):
  new_shape = [int(np.prod(ref.shape[:dim])),
                ref.shape[dim],
                int(np.prod(ref.shape[dim+1:]))]
  return new_shape

# get the shape after applying dilation on the filter, 
def canonicalize_filter_shape(H_f, W_f, rhs_dilation=None):
  if rhs_dilation:
    H_f = (H_f - 1) * rhs_dilation[0] + 1
    W_f = (W_f - 1) * rhs_dilation[1] + 1
  return H_f, W_f

def replication_factor(rep_from, rep_to, dilation=1):
  max_rep = 0
  for i in range(rep_from, 0, -1):
    if i * rep_to <= 128 and i % dilation == 0:
      max_rep = i
      break
  return max(max_rep, 1)

def tile(tripcount, tile_size):
  if not tile_size:
    return tripcount, 1, 0
  return div_ceil(tripcount, tile_size), min(tripcount, tile_size), tile_size // tripcount

def tile_with_stride(tripcount, size, stride):
  if size < stride:
    # not enlarge size if:
    # (1) only the first elem is accessed in each tile
    # (2) TODO: if stride is too big - loading too many unused data
    return tripcount, 1, 0
  if size % stride != 0:
  # adjust tilesize so that stride will not across two tiles
    size = div_ceil(size, stride) * stride
  n_tiles, tile_size, remaining = tile(tripcount, size)
  assert tile_size % stride == 0, "wrong tilesize for striding"
  return n_tiles, tile_size, remaining


def is_negative_padding(padding):
  return any([p < 0 for p in padding])


@nki.jit
def test(img_ref, filter_ref, **kwargs):
    
    padding = kwargs['padding']
    H_padding_l, H_padding_r = padding[0]
    W_padding_l, W_padding_r = padding[1]
    srcs_shapes = kwargs.get('srcs_shapes', None)
    dsts_shapes = kwargs.get('dsts_shapes', None)
    stride = kwargs.get('stride', [1, 1])
    h_stride, w_stride = stride
    lhs_dilation = kwargs.get('lhs_dilation', None) # unsupported
    rhs_dilation = kwargs.get('rhs_dilation', None)
    in_perm = kwargs.get('in_perm', None)
    kern_perm = kwargs.get('kern_perm', None)
    out_perm = kwargs.get('out_perm', None)

    nchw_in = in_perm == [0, 1, 2, 3]
    nchw_out = out_perm == [0, 1, 2, 3]

    kernel_dtype = img_ref.dtype

    if rhs_dilation is None:
        rhs_dilation = (1, 1)

    if srcs_shapes:
        img_ref, filter_ref = reshape_all([img_ref, filter_ref], srcs_shapes)

    
    
    C_out, C_in, H_f, W_f = filter_ref.shape
    _weight = nl.ndarray(shape=(C_in, H_f, W_f, C_out), dtype=kernel_dtype, buffer=nl.hbm, name='weight_transposed')
    transpose_to_last_dim(filter_ref.reshape((C_out, C_in*H_f*W_f)), dim=0, dst=_weight)

    # transpose to 3, 0, 1, 2 - C_in, H, W, N
  
    assert all([s == 1 for s in stride]), "unsupported perm with strides"
    N, C_in, H, W = img_ref.shape
    _ifmap = img_ref

    
    canonical_H_f, canonical_W_f = canonicalize_filter_shape(H_f, W_f, rhs_dilation)
    
    # Create output tensor internally
    # Determine output shape based on input shapes and parameters
  
    N, C_in, H, W = img_ref.shape
        
   
    # Calculate output dimensions
    K0 = (H + H_padding_l + H_padding_r - canonical_H_f) // h_stride + 1
    K1 = (W + W_padding_l + W_padding_r - canonical_W_f) // w_stride + 1
    
    
    out_shape = (N, C_out, K0, K1)
    
    out_ref = nl.ndarray(out_shape, dtype=kernel_dtype, buffer=nl.shared_hbm)

  
    conv_out = out_ref

 

    # need to add tiling to remove the belowed restriction
    # avoid predicates in the inner tile of replication if we have dilation in rhs
    #  instead, allow tiling on the outer tile of replication
    H_REP = replication_factor(canonical_H_f, C_in, rhs_dilation[0])
    # either no replication or divisible
    assert H_REP == 1 or H_REP % rhs_dilation[0] == 0

   
    H_OUTER_NUM_TILES, H_OUTER_TILE_SIZES, _ = tile(canonical_H_f, H_REP)

    # computation tiles
    COUT_NUM_TILES, COUT_TILE_SIZES, _ = tile(C_out, 128)

    # tiling for lhs
    tile_size = 512


    # N cannot be chosen as LHS free, so only tile on K0 and K1
    N_COMP_NUM_TILES, N_COMP_TILE_SIZES = 1, 1 # computation tile
    N_DMA_NUM_TILES, N_DMA_TILE_SIZES = 1, 1 # only prefetch on H and W
    N_OUTER_NUM_TILES, N_OUTER_TILE_SIZES = N, 1


    K1_NUM_TILES, K1_TILE_SIZES, tile_size = tile(K1, tile_size)
    K0_NUM_TILES, K0_TILE_SIZES, _ = tile_with_stride(K0*h_stride, tile_size, h_stride)
    if K0_TILE_SIZES == 1: # stride happens inter tile
        K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0, 1
        h_stride_intra_tile = h_stride

    else:                  # stride happens intra tile
        K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0_NUM_TILES, div_ceil(K0_TILE_SIZES, h_stride)
        h_stride_intra_tile = 1

    # prefetching tiles
    # only tile on prefetching W_f for simplicity
    PREFETCH_TILE_SIZE = 512*16 # TODO: pick a better tile size here
    WF_NUM_TILES, WF_TILE_SIZES = (W_f, 1) if C_out > PREFETCH_TILE_SIZE / 2 else \
        (div_ceil(W_f, PREFETCH_TILE_SIZE // C_out), min(W_f, PREFETCH_TILE_SIZE // C_out))
    print(f'W_f: {W_f}: {WF_NUM_TILES} * {WF_TILE_SIZES}')

    # for debugging only: we can determine the lhs from above:
    lhs_frees = list(map(lambda p: f'{p[0]} - {p[1]}', 
                         filter(lambda p: p[1] > 1, 
                                zip([('K0', K0), ('K1', K1), ('N', N)], [K0_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES]))))
    print(f'config: {C_in}, {N}, {H}, {W}; LHS: {(K0, K1, N)}, LHS_FREES: {lhs_frees}')
    name = f'{C_in}, {N}, {H}, {W}; {lhs_frees}'.replace('"', '').replace('\'', '')

    for n_outer_tile in nl.affine_range(N_OUTER_NUM_TILES):
        out_sb = nl.zeros((COUT_NUM_TILES, K0_COMP_NUM_TILES, K1_NUM_TILES, N_COMP_NUM_TILES, par_dim(COUT_TILE_SIZES), K0_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES), 
                          dtype=kernel_dtype, buffer=nl.sbuf, name=f'a0_sb_{name}')
    
        for h_outer in nl.affine_range(H_OUTER_NUM_TILES):
            # if there is negative padding on W: load a larger image, then make a copy to trim the padding 
            W_l_pos_padding = max(0,W_padding_l)
            W_r_pos_padding = max(0,W_padding_r)
            img_local_prefetch_raw = nl.zeros(shape=(N_DMA_NUM_TILES, K0_NUM_TILES, nl.par_dim(C_in*H_REP), K0_TILE_SIZES, W+W_l_pos_padding+W_r_pos_padding, N_DMA_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf, name='a0_img_local_prefetch')
            img_local_prefetch = nl.zeros(shape=(N_COMP_NUM_TILES, K0_NUM_TILES, nl.par_dim(C_in*H_REP), K0_TILE_SIZES, W+W_padding_l+W_padding_r, N_DMA_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf, name='a0_img_local_prefetch_neg')
            for n_tile in nl.affine_range(N_DMA_NUM_TILES):
                for k0_tile in nl.affine_range(K0_NUM_TILES):
                    for h_rep in nl.affine_range(H_REP):
                        # we cannot handle NEGATIVE padding on i_w because it will result in predicate i_w >= -w_padding_l or i_w < W+W_padding_r and bubble in free dim
                        # so we need to have a tensor copy to make this legal
                        i_cin, i_k0, i_w, i_n = create_indices(C_in, K0_TILE_SIZES, W, N_DMA_TILE_SIZES)

                        h = h_outer * H_OUTER_TILE_SIZES + h_rep
                        k0 = k0_tile * K0_TILE_SIZES * h_stride_intra_tile + i_k0
                        n = n_outer_tile * N_OUTER_TILE_SIZES + n_tile * N_DMA_TILE_SIZES + i_n

                        # replication on h, implicit padding on H, explicit padding on W, prefetchig on W
                        # all W padding is non-negative, no boundary check needed: 
                        #   i_w+W_l_pos_padding>=0 and i_w+W_l_pos_padding<W+W_l_pos_padding+W_r_pos_padding
                        mask = (h+k0-H_padding_l < H) & (h+k0-H_padding_l >= 0) & (n < N)

                        if nchw_in:
                            img_local_prefetch_raw[n_tile, k0_tile, i_cin + h_rep * C_in, i_k0, i_w+W_l_pos_padding, i_n] = nl.load(_ifmap[n, i_cin, h+k0-H_padding_l, i_w], mask=mask) 
                        else:
                            img_local_prefetch_raw[n_tile, k0_tile, i_cin + h_rep * C_in, i_k0, i_w+W_l_pos_padding, i_n] = nl.load(_ifmap[i_cin, h+k0-H_padding_l, i_w, n], mask=mask)

                
                img_local_prefetch = img_local_prefetch_raw

            # The filter is usually bigger in training, and we will need to tile on W_f. Consider the filter's free axes are H_f, W_f and C_out. 
            # And the free axes for image is H, W, N. We have H_f ~= H, W_f ~= W, and C_out >> N. (N is always small)
            # When C_out is too big, we may overflow SB when prefetching the whole W_f. We need to tile it, and better to fuse it the matmul.
            # In inference, H_f and W_f is small, but since N is also small so we likely can afford prefetching on the whole W on the image
            # Therefore, we prefetch in the image above, and then fuse filter dma with matmul. 
            for wf_tile in nl.affine_range(WF_NUM_TILES):
                filter_local_prefetch = nl.zeros((par_dim(C_in*H_REP), WF_TILE_SIZES, C_out), dtype=kernel_dtype, name='a0_filter_local_prefetch')

                if H_REP == 1:  # rhs_dilation happen on H_OUTER
                    H_REP_NUM_TILES = 1
                    
                else:  # rhs_dilation happen on H_REP
                    assert H_REP // rhs_dilation[0]
                    
                    H_REP_NUM_TILES = H_REP // rhs_dilation[0] # should be always divisible

                for h_rep in nl.affine_range(H_REP_NUM_TILES): 
                    i_cin = nl.arange(C_in)[:, None, None]
                    i_w_f = nl.arange(WF_TILE_SIZES)[None, :, None] # prefetching on W_f and C_out
                    i_cout = nl.arange(C_out)[None, None, :]

                    h = h_outer * H_REP_NUM_TILES + h_rep
                    wf = wf_tile * WF_TILE_SIZES + i_w_f
                    mask = (h < H_f) & (wf < W_f)

                    # this following dilates on W because (1) filter_local_prefetch is
                    # memset to zero (2) h_rep represent a tile of [C_in, C_in*0, C_in*0]
                    # where *0 from rhs_dilation.
                    filter_local_prefetch[i_cin + C_in * h_rep * rhs_dilation[0], i_w_f, i_cout] = nl.load(_weight[i_cin, h, wf, i_cout], mask=mask)

                for k0_tile in nl.affine_range(K0_COMP_NUM_TILES):
                    for k1_tile in nl.affine_range(K1_NUM_TILES):
                        for n_tile in nl.affine_range(N_COMP_NUM_TILES):
                            for c_out_tile in nl.affine_range(COUT_NUM_TILES):
                                ps = nl.zeros(shape=(par_dim(COUT_TILE_SIZES), K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES), dtype=np.float32, buffer=nl.psum, name=f'a0_psum_{name}')
                                for w in nl.affine_range(WF_TILE_SIZES):
                                    i_cin, i_k0, i_k1, i_n = create_indices(C_in*H_REP, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)

                                    k1 = k1_tile * K1_TILE_SIZES + i_k1
                                    n = n_tile * N_COMP_TILE_SIZES + i_n
                                    wf = wf_tile * WF_TILE_SIZES + w
                                    
                                    
                                    img_local = img_local_prefetch[n_tile, k0_tile, i_cin, i_k0 * h_stride, wf*rhs_dilation[1]+k1 * w_stride, i_n] # strided by w_stride

                                    _i_cin = nl.arange(C_in*H_REP)[:, None] # replicated
                                    i_cout = nl.arange(COUT_TILE_SIZES)[None, :]
                                    c_out = c_out_tile*COUT_TILE_SIZES+i_cout
                                    filter_local = filter_local_prefetch[_i_cin, w, c_out]

                                    i_cout_out = nl.arange(COUT_TILE_SIZES)[:, None, None]
                                    ps[i_cout_out, i_k0, i_k1, i_n] += nisa.nc_matmul(
                                        filter_local[c_out < C_out],
                                        img_local[k1 < K1][n < N][wf < W_f],
                                    )

                                i_cout_out, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)
                                out_sb[c_out_tile, k0_tile, k1_tile, n_tile, i_cout_out, i_k0, i_k1, i_n] += ps[i_cout_out, i_k0, i_k1, i_n]

        # storing the compute results
        for k0_tile in nl.affine_range(K0_COMP_NUM_TILES):
            for k1_tile in nl.affine_range(K1_NUM_TILES):
                for n_tile in nl.affine_range(N_COMP_NUM_TILES):
                    for c_out_tile in nl.affine_range(COUT_NUM_TILES):
                        i_cout, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)

                        c_out = c_out_tile * COUT_TILE_SIZES + i_cout
                        k0 = k0_tile * K0_COMP_TILE_SIZES + i_k0
                        k1 = k1_tile * K1_TILE_SIZES + i_k1
                        n = n_outer_tile * N_OUTER_TILE_SIZES + n_tile * N_COMP_TILE_SIZES + i_n
                        mask = (c_out < C_out) & (k0 < K0) & (k1 < K1) & (n < N)

                        nl.store(
                            out_ref[n, c_out, k0, k1] if nchw_out else conv_out[c_out, k0, k1, n],
                            out_sb[c_out_tile, k0_tile, k1_tile, n_tile, i_cout, i_k0, i_k1, i_n],
                            mask=mask
                        )

    if nchw_out:
        return out_ref
    if out_perm == [2, 3, 0, 1]:
        transpose_to_last_dim(conv_out, dim=0, dst=out_ref)
        return out_ref
    else:
        assert out_perm == [1, 2, 3, 0]
        return out_ref