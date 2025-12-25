import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(Q, K, V, past_key_value, attention_mask):
    """
    NKI implementation of token generation attention.

    Shapes (constant for this kernel):
      Q: [32, 16, 1, 64]
      K: [32, 4, 1, 64]
      V: [32, 4, 1, 64]
      past_key_value[0]: [32, 4, 512, 64] (K_prior)
      past_key_value[1]: [32, 4, 512, 64] (V_prior)
      attention_mask: [32, 16, 1, 512] (bool)
    """
    kernel_output = nl.ndarray(Q.shape, dtype=Q.dtype, buffer=nl.shared_hbm)

    # Constants (compile-time for fixed shapes)
    B_SZ = 32
    NUM_HEADS = 16
    NUM_KV_HEADS = 4
    HEAD_DIM = 64
    SEQ_LEN = 512
    GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS  # 4

    inv_sqrt_scale = 1.0 / 8.0  # 1/sqrt(64)
    neg_inf_val = -1.0e4

    pkv_k = past_key_value[0]
    pkv_v = past_key_value[1]

    for b in nl.affine_range(B_SZ):
        for kv_h in nl.affine_range(NUM_KV_HEADS):

            # ------------------------------------------------------------
            # 1) Build K_prior^T in SBUF: [64(P), 512(F)]
            # ------------------------------------------------------------
            k_prior_T = nl.ndarray((nl.par_dim(HEAD_DIM), SEQ_LEN),
                                   dtype=pkv_k.dtype, buffer=nl.sbuf)

            for i in nl.affine_range(SEQ_LEN // 128):
                offset = i * 128

                # Load [128(P), 64(F)]
                k_chunk = nl.load(pkv_k[b, kv_h, nl.ds(offset, 128)])

                # Transpose to [64(P), 128(F)] (PSUM -> SBUF)
                k_chunk_T_psum = nisa.nc_transpose(k_chunk, engine=nisa.tensor_engine)
                k_chunk_T_sb = nisa.tensor_copy(k_chunk_T_psum, dtype=pkv_k.dtype)

                # Scatter into [64, 512]
                k_prior_T[:, nl.ds(offset, 128)] = k_chunk_T_sb

            # ------------------------------------------------------------
            # 2) Active K^T: K_active^T [64(P), 1(F)]
            # ------------------------------------------------------------
            k_active_tile = nl.load(K[b, kv_h, 0:1, :])  # [1(P), 64(F)]
            k_active_T_psum = nisa.nc_transpose(k_active_tile, engine=nisa.tensor_engine)  # [64,1]
            k_active_T = nisa.tensor_copy(k_active_T_psum, dtype=K.dtype)

            # ------------------------------------------------------------
            # 3) Process 4 Q heads for this KV head
            # ------------------------------------------------------------
            h_start = kv_h * GROUP_SIZE

            # Q group: [4(P), 64(F)]
            q_group = nl.load(Q[b, nl.ds(h_start, GROUP_SIZE), 0, :])

            # Stationary for matmul: Q^T as [64(P), 4(F)]
            q_stat_psum = nisa.nc_transpose(q_group, engine=nisa.tensor_engine)
            q_stat = nisa.tensor_copy(q_stat_psum, dtype=Q.dtype)

            # ------------------------------------------------------------
            # Scores
            # ------------------------------------------------------------
            scores_prior_psum = nisa.nc_matmul(q_stat, k_prior_T)   # [4,512] in PSUM
            scores_prior = nisa.tensor_copy(scores_prior_psum, dtype=nl.float32)  # [4,512] SBUF fp32

            scores_active_psum = nisa.nc_matmul(q_stat, k_active_T)  # [4,1] in PSUM
            scores_active = nisa.tensor_copy(scores_active_psum, dtype=nl.float32)  # [4,1] SBUF fp32

            # Scale
            scores_prior = nl.multiply(scores_prior, inv_sqrt_scale)
            scores_active = nl.multiply(scores_active, inv_sqrt_scale)

            # Mask prior scores
            mask_tile = nl.load(attention_mask[b, nl.ds(h_start, GROUP_SIZE), 0, :])  # [4,512] bool
            scores_prior = nl.where(mask_tile, scores_prior, neg_inf_val)

            # ------------------------------------------------------------
            # Softmax (numerically stable) but DELAY normalization division
            # ------------------------------------------------------------
            max_prior = nl.max(scores_prior, axis=1, keepdims=True)  # [4,1]
            max_active = scores_active                               # [4,1]
            max_global = nl.maximum(max_prior, max_active)           # [4,1]

            exp_prior = nl.exp(nl.subtract(scores_prior, max_global))    # [4,512] fp32
            exp_active = nl.exp(nl.subtract(scores_active, max_global))  # [4,1] fp32

            sum_prior = nl.sum(exp_prior, axis=1, keepdims=True)  # [4,1] fp32
            sum_active = exp_active                               # [4,1] fp32
            denominator = nl.add(sum_prior, sum_active)           # [4,1] fp32

            # Use unnormalized exp() values for value accumulation (cast once)
            exp_prior_w = nisa.tensor_copy(exp_prior, dtype=Q.dtype)     # [4,512] bf16/fp16
            exp_active_w = nisa.tensor_copy(exp_active, dtype=Q.dtype)   # [4,1] bf16/fp16

            # ------------------------------------------------------------
            # Value accumulation: (exp * V) / denom
            # ------------------------------------------------------------
            out_accum = nl.zeros((GROUP_SIZE, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)  # [4,64] fp32

            # Prior V contribution: sum over 512 in 128-chunks
            for i in nl.affine_range(SEQ_LEN // 128):
                offset = i * 128

                # weights chunk [4,128] -> transpose to stationary [128,4]
                w_chunk = exp_prior_w[:, nl.ds(offset, 128)]
                w_stat_psum = nisa.nc_transpose(w_chunk, engine=nisa.tensor_engine)  # [128,4]
                w_stat = nisa.tensor_copy(w_stat_psum, dtype=Q.dtype)

                # V chunk [128,64]
                v_chunk = nl.load(pkv_v[b, kv_h, nl.ds(offset, 128)])

                # [4,128] @ [128,64] -> [4,64]
                out_accum += nisa.nc_matmul(w_stat, v_chunk)

            # Active V contribution: [4,1] @ [1,64] -> [4,64]
            w_active_stat_psum = nisa.nc_transpose(exp_active_w, engine=nisa.tensor_engine)  # [1,4]
            w_active_stat = nisa.tensor_copy(w_active_stat_psum, dtype=Q.dtype)

            v_active = nl.load(V[b, kv_h, 0:1, :])  # [1,64]
            out_accum += nisa.nc_matmul(w_active_stat, v_active)

            # ------------------------------------------------------------
            # Normalize once at the end: out = out_accum / denominator
            # ------------------------------------------------------------
            out_fp32 = nisa.tensor_copy(out_accum, dtype=nl.float32)     # PSUM -> SBUF fp32
            out_norm_fp32 = nl.divide(out_fp32, denominator)            # broadcast denom [4,1] -> [4,64]
            out_tile = nisa.tensor_copy(out_norm_fp32, dtype=Q.dtype)    # cast to output dtype

            nl.store(kernel_output[b, nl.ds(h_start, GROUP_SIZE), 0, :], value=out_tile)

    return kernel_output

def get_test_data(dtype):
    """Create test data for attention kernel."""
    batch = 32
    num_heads = 16
    num_kv_heads = 4
    seqlen_kv = 512
    head_dim = 64
    
    # Q: [32, 16, 1, 64]
    Q = np.random.randn(batch, num_heads, 1, head_dim).astype(dtype)
    # K: [32, 4, 1, 64]
    K = np.random.randn(batch, num_kv_heads, 1, head_dim).astype(dtype)
    # V: [32, 4, 1, 64]
    V = np.random.randn(batch, num_kv_heads, 1, head_dim).astype(dtype)
    # past_k: [32, 4, 512, 64]
    past_k = np.random.randn(batch, num_kv_heads, seqlen_kv, head_dim).astype(dtype)
    # past_v: [32, 4, 512, 64]
    past_v = np.random.randn(batch, num_kv_heads, seqlen_kv, head_dim).astype(dtype)
    # attention_mask: [32, 16, 1, 512]
    attention_mask = np.random.randint(0, 2, (batch, num_heads, 1, seqlen_kv)).astype(np.bool_)
    return (Q, K, V, (past_k, past_v), attention_mask)

def compare_outputs(reference_out, test_out, atol=1e-3, rtol=1e-3):
    """Compare test output against reference output."""
    ref_f32 = reference_out.astype(nl.float32)
    test_f32 = test_out.astype(nl.float32)
    if not np.allclose(ref_f32, test_f32, atol=atol, rtol=rtol):
        print("reference_out[:8]: %s", ref_f32.flatten()[:8])
        print("test_out[:8]: %s", test_f32.flatten()[:8])
        diff = np.abs(ref_f32 - test_f32)
        print("max_diff: %s", np.max(diff))
        print("mean_diff: %s", np.mean(diff))
        print("FAIL: test output does not match reference")
        return False
    return True

def test_nki(ref_func, test_func):
    np.random.seed(0)
    dtype = nl.bfloat16
    
    for _ in range(2):
        test_data = get_test_data(dtype)
        ref_out = ref_func(*test_data)
        test_out = test_func(*test_data)
        if not compare_outputs(ref_out, test_out):
            return False
    return True

def benchmark_nki(nki_func):
    dtype = nl.bfloat16
    test_data = get_test_data(dtype)
    
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(*test_data)
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
