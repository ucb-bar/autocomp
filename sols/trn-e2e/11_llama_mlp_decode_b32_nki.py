@nki.jit
def test(
    x_dev: nt.tensor,          # [B, 1, K_IN]  (no external transpose)
    up_w: nt.tensor,           # [K_IN, K_INTER]
    gate_w: nt.tensor,         # [K_IN, K_INTER]
    down_w: nt.tensor          # [K_INTER, N_OUT]
) -> nt.tensor:                # [B, 1, N_OUT]

    # Constant shapes (LLM MLP)
    B = 32
    K_IN = 2048
    K_INTER = 4096
    N_OUT = 2048

    # Tile sizes (TensorE-friendly)
    T_K = 128
    T_INTER = 128
    T_OUT_CHUNK = 128

    NUM_KIN = K_IN // T_K                 # 16
    NUM_KI = K_INTER // T_INTER           # 32
    NUM_OUT_CHUNKS = N_OUT // T_OUT_CHUNK # 16

    out_dev = nl.ndarray((B, 1, N_OUT), dtype=x_dev.dtype, buffer=nl.shared_hbm)

    # PSUM accumulation buffer for all batches and output chunks.
    # Layout: [B, J, P=128, F=1] (tile per (b,j) is a valid tile when indexed)
    out_psum_all = nl.zeros(
        (B, NUM_OUT_CHUNKS, nl.par_dim(T_OUT_CHUNK), 1),
        dtype=nl.float32,
        buffer=nl.psum
    )

    # NOTE:
    # - Keep ki sequential to avoid schedule hazards (same as original).
    # - down_strip is reused across all b for this ki, saving HBM bandwidth.
    for ki in nl.sequential_range(NUM_KI):
        # down_w block: [T_INTER, N_OUT] = [128, 2048] loaded once per ki
        down_strip = nl.load(down_w[nl.ds(ki * T_INTER, T_INTER), 0:N_OUT])  # SBUF [128,2048]

        # For each batch element, compute up/gate projections then down-proj
        for b in nl.affine_range(B):
            # Accumulate (1 x 128) in PSUM for better matmul accumulate behavior
            up_acc = nl.zeros((1, T_INTER), dtype=nl.float32, buffer=nl.psum)
            gate_acc = nl.zeros((1, T_INTER), dtype=nl.float32, buffer=nl.psum)

            # K_IN reduction
            for kin in nl.affine_range(NUM_KIN):
                # Load x directly as [1,128] from [B,1,K_IN]
                x_flat = nl.load(x_dev[b, 0:1, nl.ds(kin * T_K, T_K)])  # SBUF [1,128]

                # In-kernel transpose to make a [128,1] tile (partition axis=128) for matmul API.
                # nc_transpose: PSUM output, then copy to SBUF.
                x_T_psum = nisa.nc_transpose(x_flat, engine=nisa.tensor_engine)  # PSUM [128,1]
                x_tile = nisa.tensor_copy(x_T_psum, dtype=x_dev.dtype, engine=nisa.vector_engine)  # SBUF [128,1]

                # Load corresponding weight tiles
                up_tile = nl.load(
                    up_w[nl.ds(kin * T_K, T_K), nl.ds(ki * T_INTER, T_INTER)]
                )  # SBUF [128,128]
                gate_tile = nl.load(
                    gate_w[nl.ds(kin * T_K, T_K), nl.ds(ki * T_INTER, T_INTER)]
                )  # SBUF [128,128]

                # Compute: (x^T @ W) where x_tile is [K,1] and treated as x^T when transpose_x=True
                # => [1,K] @ [K,128] -> [1,128]
                up_acc += nl.matmul(x_tile, up_tile, transpose_x=True)
                gate_acc += nl.matmul(x_tile, gate_tile, transpose_x=True)

            # Stage 2: SwiGLU: SiLU(gate) * up
            # Convert [1,128] accumulators to dtype and transpose to [128,1] for elementwise ops
            up_sbuf_1x128 = nisa.tensor_copy(up_acc, dtype=x_dev.dtype, engine=nisa.vector_engine)      # SBUF [1,128]
            gate_sbuf_1x128 = nisa.tensor_copy(gate_acc, dtype=x_dev.dtype, engine=nisa.vector_engine)  # SBUF [1,128]

            up_T_psum = nisa.nc_transpose(up_sbuf_1x128, engine=nisa.tensor_engine)       # PSUM [128,1]
            gate_T_psum = nisa.nc_transpose(gate_sbuf_1x128, engine=nisa.tensor_engine)   # PSUM [128,1]

            gate_act = nisa.activation(op=nl.silu, data=gate_T_psum, dtype=x_dev.dtype)   # SBUF [128,1]
            up_T = nisa.tensor_copy(up_T_psum, dtype=x_dev.dtype, engine=nisa.vector_engine)  # SBUF [128,1]
            act = nl.multiply(gate_act, up_T)  # SBUF [128,1]

            # Stage 3: Down projection (for this ki block) into all output chunks
            # Each out_psum_all[b, j] is a PSUM tile [128,1], valid to accumulate into.
            for j in nl.affine_range(NUM_OUT_CHUNKS):
                down_chunk = down_strip[:, nl.ds(j * T_OUT_CHUNK, T_OUT_CHUNK)]  # SBUF view [128,128]
                out_psum_all[b, j] += nisa.nc_matmul(stationary=down_chunk, moving=act)  # PSUM [128,1]

    # Pack/store outputs per batch: convert each [128,1] chunk into [1,128] and write into result row
    for b in nl.affine_range(B):
        result_sbuf = nl.zeros((1, N_OUT), dtype=x_dev.dtype, buffer=nl.sbuf)  # SBUF [1,2048]

        for j in nl.affine_range(NUM_OUT_CHUNKS):
            acc_sbuf_128x1 = nisa.tensor_copy(out_psum_all[b, j], dtype=x_dev.dtype, engine=nisa.vector_engine)  # SBUF [128,1]
            acc_T_psum_1x128 = nisa.nc_transpose(acc_sbuf_128x1, engine=nisa.tensor_engine)                      # PSUM [1,128]
            acc_T_sbuf_1x128 = nisa.tensor_copy(acc_T_psum_1x128, dtype=x_dev.dtype, engine=nisa.vector_engine)  # SBUF [1,128]

            # Basic indexing only (ds slice) to avoid mixing indexing modes
            result_sbuf[:, nl.ds(j * T_OUT_CHUNK, T_OUT_CHUNK)] = acc_T_sbuf_1x128

        nl.store(out_dev[b, 0:1, 0:N_OUT], value=result_sbuf)

    return out_dev
