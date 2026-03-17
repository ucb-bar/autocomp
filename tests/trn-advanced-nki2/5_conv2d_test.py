
import math
import nki
import numpy as np
import nki.isa as nisa
import nki.language as nl
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE


def div_ceil(n, d):
    return (n + d - 1) // d


def get_3d_shape(ref, dim):
    before = 1
    for s in ref.shape[:dim]:
        before *= s
    after = 1
    for s in ref.shape[dim+1:]:
        after *= s
    return [before, ref.shape[dim], after]


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
        return tripcount, 1, 0
    if size % stride != 0:
        size = div_ceil(size, stride) * stride
    n_tiles, tile_size, remaining = tile(tripcount, size)
    assert tile_size % stride == 0
    return n_tiles, tile_size, remaining


def transpose_kernel_beta2(ref, dim, dst):
    """Transpose a tensor along 'dim' to last position using beta2 API."""
    ref = ref.reshape(get_3d_shape(ref, dim))
    transposed_shape = (ref.shape[0], ref.shape[2], ref.shape[1])
    transpose_nonlocal = dst.reshape(transposed_shape)

    D0, B, N = ref.shape
    B_tile_size = min(128, B)
    N_tile_size = min(128, N)
    B_num_tiles = div_ceil(B, B_tile_size)
    N_num_tiles = div_ceil(N, N_tile_size)

    for d0 in nl.affine_range(D0):
        for b_tile in nl.affine_range(B_num_tiles):
            b_start = b_tile * B_tile_size
            b_end = min(b_start + B_tile_size, B)
            tile_b = b_end - b_start
            for n_tile in nl.affine_range(N_num_tiles):
                n_start = n_tile * N_tile_size
                n_end = min(n_start + N_tile_size, N)
                tile_n = n_end - n_start

                _local = nl.ndarray((B_tile_size, N_tile_size), dtype=ref.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=_local[0:tile_b, 0:tile_n],
                              src=ref[d0, b_start:b_end, n_start:n_end])
                t_psum = nl.ndarray((N_tile_size, B_tile_size), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=t_psum[0:tile_n, 0:tile_b],
                                  data=_local[0:tile_b, 0:tile_n])
                t_local = nl.ndarray((N_tile_size, B_tile_size), dtype=ref.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=t_local[0:tile_n, 0:tile_b],
                                 src=t_psum[0:tile_n, 0:tile_b])
                nisa.dma_copy(dst=transpose_nonlocal[d0, n_start:n_end, b_start:b_end],
                              src=t_local[0:tile_n, 0:tile_b])


@nki.jit
def ref(img_ref, filter_T_ref, padding=None, stride=None, rhs_dilation=None, in_perm=None, kern_perm=None, out_perm=None):
    # filter_T_ref shape: (C_in, H_f, W_f, C_out) — pre-transposed outside the kernel
    if stride is None:
        stride = [1, 1]
    H_padding_l, H_padding_r = padding[0]
    W_padding_l, W_padding_r = padding[1]
    h_stride, w_stride = stride

    nchw_in = in_perm == [0, 1, 2, 3]
    nchw_out = out_perm == [0, 1, 2, 3]

    kernel_dtype = img_ref.dtype

    if rhs_dilation is None:
        rhs_dilation = (1, 1)

    C_in, H_f, W_f, C_out = filter_T_ref.shape

    assert stride[0] == 1 and stride[1] == 1, "unsupported perm with strides"
    N, C_in, H, W = img_ref.shape

    canonical_H_f, canonical_W_f = canonicalize_filter_shape(H_f, W_f, rhs_dilation)

    K0 = (H + H_padding_l + H_padding_r - canonical_H_f) // h_stride + 1
    K1 = (W + W_padding_l + W_padding_r - canonical_W_f) // w_stride + 1

    out_ref = nl.ndarray((N, C_out, K0, K1), dtype=kernel_dtype, buffer=nl.shared_hbm)

    H_REP = replication_factor(canonical_H_f, C_in, rhs_dilation[0])
    assert H_REP == 1 or H_REP % rhs_dilation[0] == 0

    H_OUTER_NUM_TILES, H_OUTER_TILE_SIZES, _ = tile(canonical_H_f, H_REP)
    COUT_NUM_TILES, COUT_TILE_SIZES, _ = tile(C_out, 128)
    tile_size = 512

    N_COMP_NUM_TILES, N_COMP_TILE_SIZES = 1, 1
    N_DMA_NUM_TILES, N_DMA_TILE_SIZES = 1, 1
    N_OUTER_NUM_TILES, N_OUTER_TILE_SIZES = N, 1

    K1_NUM_TILES, K1_TILE_SIZES, tile_size = tile(K1, tile_size)
    K0_NUM_TILES, K0_TILE_SIZES, _ = tile_with_stride(K0 * h_stride, tile_size, h_stride)
    if K0_TILE_SIZES == 1:
        K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0, 1
        h_stride_intra_tile = h_stride
    else:
        K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0_NUM_TILES, div_ceil(K0_TILE_SIZES, h_stride)
        h_stride_intra_tile = 1

    PREFETCH_TILE_SIZE = 512 * 16
    WF_NUM_TILES, WF_TILE_SIZES = (W_f, 1) if C_out > PREFETCH_TILE_SIZE / 2 else \
        (div_ceil(W_f, PREFETCH_TILE_SIZE // C_out), min(W_f, PREFETCH_TILE_SIZE // C_out))

    W_l_pos_padding = max(0, W_padding_l)
    W_r_pos_padding = max(0, W_padding_r)
    padded_W = W + W_l_pos_padding + W_r_pos_padding

    if H_REP == 1:
        H_REP_NUM_TILES = 1
    else:
        H_REP_NUM_TILES = H_REP // rhs_dilation[0]

    for n_outer_tile in nl.affine_range(N_OUTER_NUM_TILES):
        n = n_outer_tile * N_OUTER_TILE_SIZES

        for k0_tile in nl.affine_range(K0_COMP_NUM_TILES):
            k0_start = k0_tile * K0_COMP_TILE_SIZES

            for k1_tile in nl.affine_range(K1_NUM_TILES):
                k1_start = k1_tile * K1_TILE_SIZES
                k1_end = min(k1_start + K1_TILE_SIZES, K1)
                tile_k1 = k1_end - k1_start

                for c_out_tile in nl.affine_range(COUT_NUM_TILES):
                    c_out_start = c_out_tile * COUT_TILE_SIZES
                    c_out_end = min(c_out_start + COUT_TILE_SIZES, C_out)
                    tile_cout = c_out_end - c_out_start

                    out_sbuf = nl.ndarray(
                        (COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES),
                        dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=out_sbuf, value=0.0)

                    for h_outer in nl.affine_range(H_OUTER_NUM_TILES):
                        img_local = nl.ndarray(
                            (C_in * H_REP, K0_TILE_SIZES, padded_W),
                            dtype=kernel_dtype, buffer=nl.sbuf)
                        nisa.memset(dst=img_local, value=0.0)

                        for h_rep in nl.affine_range(H_REP):
                            h = h_outer * H_OUTER_TILE_SIZES + h_rep
                            for i_k0 in nl.affine_range(K0_TILE_SIZES):
                                k0_row = k0_tile * K0_TILE_SIZES * h_stride_intra_tile + i_k0
                                src_h = h + k0_row - H_padding_l
                                if src_h >= 0 and src_h < H:
                                    img_strip = nl.ndarray((C_in, W), dtype=kernel_dtype, buffer=nl.sbuf)
                                    if nchw_in:
                                        nisa.dma_copy(dst=img_strip[0:C_in, 0:W],
                                                      src=img_ref[n, 0:C_in, src_h, 0:W])
                                    else:
                                        nisa.dma_copy(dst=img_strip[0:C_in, 0:W],
                                                      src=img_ref[0:C_in, src_h, 0:W, n])
                                    nisa.tensor_copy(
                                        dst=img_local[h_rep * C_in:(h_rep + 1) * C_in,
                                                      i_k0, W_l_pos_padding:W_l_pos_padding+W],
                                        src=img_strip[0:C_in, 0:W])

                        for wf_tile in nl.affine_range(WF_NUM_TILES):
                            filter_slice = nl.ndarray(
                                (C_in * H_REP, WF_TILE_SIZES, COUT_TILE_SIZES),
                                dtype=kernel_dtype, buffer=nl.sbuf)
                            nisa.memset(dst=filter_slice, value=0.0)

                            for h_rep in nl.affine_range(H_REP_NUM_TILES):
                                h = h_outer * H_REP_NUM_TILES + h_rep
                                wf_start = wf_tile * WF_TILE_SIZES
                                for i_wf in nl.affine_range(WF_TILE_SIZES):
                                    wf_abs = wf_start + i_wf
                                    if h < H_f and wf_abs < W_f:
                                        w_strip = nl.ndarray((C_in, COUT_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf)
                                        nisa.dma_copy(dst=w_strip[0:C_in, 0:COUT_TILE_SIZES],
                                                      src=filter_T_ref[0:C_in, h, wf_abs, c_out_start:c_out_start+COUT_TILE_SIZES])
                                        nisa.tensor_copy(
                                            dst=filter_slice[h_rep * C_in * rhs_dilation[0]:(h_rep + 1) * C_in * rhs_dilation[0]:rhs_dilation[0], i_wf, 0:COUT_TILE_SIZES],
                                            src=w_strip[0:C_in, 0:COUT_TILE_SIZES])

                            for w in nl.affine_range(WF_TILE_SIZES):
                                wf = wf_tile * WF_TILE_SIZES + w
                                img_col_start = wf * rhs_dilation[1] + k1_start * w_stride

                                filt_vec = nl.ndarray((C_in * H_REP, COUT_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf)
                                nisa.tensor_copy(
                                    dst=filt_vec[0:C_in*H_REP, 0:COUT_TILE_SIZES],
                                    src=filter_slice[0:C_in*H_REP, w, 0:COUT_TILE_SIZES])

                                for i_k0 in nl.affine_range(K0_COMP_TILE_SIZES):
                                    img_block = nl.ndarray((C_in * H_REP, K1_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf)
                                    nisa.tensor_copy(
                                        dst=img_block[0:C_in*H_REP, 0:K1_TILE_SIZES],
                                        src=img_local[0:C_in*H_REP, i_k0, img_col_start:img_col_start+K1_TILE_SIZES])

                                    contrib = nl.ndarray((COUT_TILE_SIZES, K1_TILE_SIZES), dtype=nl.float32, buffer=nl.psum)
                                    nisa.nc_matmul(dst=contrib, stationary=filt_vec, moving=img_block)
                                    contrib_sbuf = nl.ndarray((COUT_TILE_SIZES, K1_TILE_SIZES), dtype=nl.float32, buffer=nl.sbuf)
                                    nisa.tensor_copy(dst=contrib_sbuf, src=contrib)

                                    cur_k0 = nl.ndarray((COUT_TILE_SIZES, K1_TILE_SIZES), dtype=nl.float32, buffer=nl.sbuf)
                                    nisa.tensor_copy(dst=cur_k0, src=out_sbuf[0:COUT_TILE_SIZES, i_k0, 0:K1_TILE_SIZES])
                                    nisa.tensor_tensor(dst=cur_k0, data1=cur_k0, data2=contrib_sbuf, op=nl.add)
                                    nisa.tensor_copy(dst=out_sbuf[0:COUT_TILE_SIZES, i_k0, 0:K1_TILE_SIZES], src=cur_k0)

                    # Store accumulated results to output HBM
                    for i_k0 in nl.affine_range(K0_COMP_TILE_SIZES):
                        k0 = k0_start + i_k0
                        if k0 < K0:
                            out_strip = nl.ndarray((COUT_TILE_SIZES, K1_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf)
                            nisa.tensor_copy(
                                dst=out_strip[0:tile_cout, 0:tile_k1],
                                src=out_sbuf[0:tile_cout, i_k0, 0:tile_k1])
                            nisa.dma_copy(
                                dst=out_ref[n, c_out_start:c_out_end, k0, k1_start:k1_end],
                                src=out_strip[0:tile_cout, 0:tile_k1])

    return out_ref


def _pre_transpose_filter(filter_weights):
    """Pre-transpose filter: (C_out, C_in, H_f, W_f) -> (C_in, H_f, W_f, C_out)"""
    C_out, C_in, H_f, W_f = filter_weights.shape
    return filter_weights.reshape(C_out, C_in * H_f * W_f).T.reshape(C_in, H_f, W_f, C_out)


def test_nki(ref_func, test_func):
    device = xm.xla_device()
    test_configs = [
        (16, 128, 128, 128, 512, 3, 3, ((1, 1), (1, 1)), (1, 1)),
    ]
    for i in range(len(test_configs)):
        batch_size, in_channels, height, width, out_channels, filter_h, filter_w, padding, stride = test_configs[i]
        img_np = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
        filter_weights_np = np.random.rand(out_channels, in_channels, filter_h, filter_w).astype(np.float32)
        filter_T_np = _pre_transpose_filter(filter_weights_np)
        img = torch.from_numpy(img_np).to(device=device)
        filter_T = torch.from_numpy(filter_T_np).to(device=device)
        result = test_func(img, filter_T, padding=padding, stride=stride, rhs_dilation=(1, 1), in_perm=[0, 1, 2, 3], kern_perm=[0, 1, 2, 3], out_perm=[0, 1, 2, 3])
        ref_result = ref_func(img, filter_T, padding=padding, stride=stride, rhs_dilation=(1, 1), in_perm=[0, 1, 2, 3], kern_perm=[0, 1, 2, 3], out_perm=[0, 1, 2, 3])
        if not np.allclose(result.detach().cpu().numpy(), ref_result.detach().cpu().numpy(), atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    device = xm.xla_device()
    test_configs = [
        (16, 128, 128, 128, 512, 3, 3, ((1, 1), (1, 1)), (1, 1)),
    ]

    for i, (batch_size, in_channels, height, width, out_channels, filter_h, filter_w, padding, stride) in enumerate(test_configs):
        img_np = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
        filter_weights_np = np.random.rand(out_channels, in_channels, filter_h, filter_w).astype(np.float32)
        filter_T_np = _pre_transpose_filter(filter_weights_np)
        img = torch.from_numpy(img_np).to(device=device)
        filter_T = torch.from_numpy(filter_T_np).to(device=device)

        bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
        bench_func(img, filter_T,
                   padding=padding,
                   stride=stride,
                   rhs_dilation=(1, 1),
                   in_perm=[0, 1, 2, 3],
                   kern_perm=[0, 1, 2, 3],
                   out_perm=[0, 1, 2, 3])

        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))


if __name__ == "__main__":
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1" 
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
