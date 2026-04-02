import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import time

# SUBSTITUTE HERE


@nki.jit
def ref(img_ref, filter_ref, **kwargs):
    def _div_ceil(n, d):
        return (n + d - 1) // d
    def _create_indices(*tripcounts):
        rank = len(tripcounts)
        if tripcounts[-1] == 1:
            assert tripcounts[-2] != 1, "Unhandled case"
            rank -= 1
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
    padding = kwargs['padding']
    W_padding_l, W_padding_r = padding[1]

    N, C_in, H, W = img_ref.shape        # bf01
    C_out, _, H_f, W_f = filter_ref.shape

    # Compute output spatial sizes from inputs + padding + filter
    K0 = H - H_f + 1                      # here H==1 and H_f==1 -> 1
    K1 = W + W_padding_l + W_padding_r - W_f + 1
    out_image_size = K0 * K1
    image_size = H * (W + W_padding_l + W_padding_r)
    window_size = H_f * W_f
    dtype = img_ref.dtype

    # Allocate output in HBM and return it later
    out_hbm = nl.ndarray((N, C_out, K0, K1), dtype=dtype, buffer=nl.shared_hbm)

    C_NUM_TILES, C_TILE_SIZE = _div_ceil(C_in, 128), min(C_in, 128)

    img_local_prefetch_raw = nl.zeros(shape=(N, C_NUM_TILES, nl.par_dim(C_TILE_SIZE), image_size),
                                      dtype=dtype, buffer=nl.sbuf, name='a0_img_local_prefetch')
    for i_n in nl.affine_range(N):
        for c_tile in nl.affine_range(C_NUM_TILES):
            i_cin_tile, i_w = _create_indices(C_TILE_SIZE, W)
            i_cin = i_cin_tile + c_tile * 128
            i_h = 0
            i_image = W_padding_l + i_w
            img_local_prefetch_raw[i_n, c_tile, i_cin_tile, i_image] = nl.load(img_ref[i_n, i_cin, i_h, i_w])

    filter_local = nl.zeros(shape=(C_NUM_TILES, nl.par_dim(C_TILE_SIZE), window_size),
                            dtype=dtype, buffer=nl.sbuf, name='a0_filter_local')
    for c_tile in nl.affine_range(C_NUM_TILES):
        i_cin_tile, i_w = _create_indices(C_TILE_SIZE, W_f)
        i_cin = i_cin_tile + c_tile * 128
        i_h = 0
        filter_local[c_tile, i_cin_tile, i_w * H_f + i_h] = nl.load(filter_ref[i_cin, i_h, i_h, i_w])

    out_sb = nl.zeros((N, C_NUM_TILES, nl.par_dim(C_TILE_SIZE), out_image_size),
                      dtype=dtype, buffer=nl.sbuf, name='output')

    for i_n in nl.affine_range(N):
        for c_tile in nl.affine_range(C_NUM_TILES):
            for i_out in nl.affine_range(out_image_size):
                i_p_a = nl.arange(C_TILE_SIZE)[:, None]
                i_f_a = nl.arange(W_f)[None, :]
                prod = nisa.tensor_tensor(
                    img_local_prefetch_raw[i_n, c_tile, i_p_a, i_f_a + i_out],
                    filter_local[c_tile, i_p_a, i_f_a],
                    np.multiply
                )
                out_sb[i_n, c_tile, i_p_a, i_out] = nisa.tensor_reduce(np.add, prod[i_p_a, i_f_a], axis=[1])

    # Write SBUF -> HBM result and return it
    for n in nl.affine_range(N):
        for c_tile in nl.affine_range(C_NUM_TILES):
            i_cout, i_k0, i_k1 = _create_indices(C_TILE_SIZE, K0, K1)
            c_out = c_tile * C_TILE_SIZE + i_cout
            i_out = i_k1 * K0 + i_k0
            mask = (c_out < C_out)
            nl.store(out_hbm[n, c_out, i_k0, i_k1], out_sb[n, c_tile, i_cout, i_out], mask=mask)

    return out_hbm

def test_nki(ref_func, test_func):
    for _ in range(2):
        N, C, H, W = 8, 512, 1, 2048
        C_out = C
        H_f, W_f = 1, 3
        padding = ((0, 0), (1, 1))  # ((top,bottom),(left,right))
        img = np.random.rand(N, C, H, W).astype(np.float32)
        filt = np.random.rand(C_out, 1, H_f, W_f).astype(np.float32)
        result_1 = ref_func(img, filt, padding=padding)
        result_2 = test_func(img, filt, padding=padding)
        if not np.allclose(result_1, result_2, atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    N, C, H, W = 8, 512, 1, 2048
    C_out = C
    H_f, W_f = 1, 3
    padding = ((0, 0), (1, 1))  # ((top,bottom),(left,right))
    img = np.random.rand(N, C, H, W).astype(np.float32)
    filt = np.random.rand(C_out, 1, H_f, W_f).astype(np.float32)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(img, filt, padding=padding)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)