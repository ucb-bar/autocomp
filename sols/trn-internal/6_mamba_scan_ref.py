@nki.jit
def solution(
    dA_exp_t,
    dBx_t,
    C_t,
    Dx_t,
    x_t,
    hd_range,
    ss_range,
):
    """Mamba2 selective scan using tensor_tensor_scan (Granite 4.0-H-Small).

    O(L) hardware-accelerated recurrence: state[t] = exp(dA[t]) * state[t-1] + dBx[t].
    Inputs are pre-transposed to partition-first layout by host-side prepare_scan_inputs().

    Args:
        dA_exp_t: (NH, SL) float32 — pre-computed exp(dt * A), transposed.
        dBx_t: (NH * HD * SS, SL) float32 — flattened input contributions, transposed.
        C_t: (NH * SS, SL) float32 — observation matrix, flattened+transposed.
        Dx_t: (NH * HD, SL) float32 — pre-computed D*x skip connection, flattened+transposed.
        x_t: (NH * HD, SL) float32 — input, flattened+transposed (unused, shape derivation).
        hd_range: (HD,) float32 — dummy tensor for head_dim shape.
        ss_range: (SS,) float32 — dummy tensor for ssm_state_size shape.

    Returns:
        y_out: (NH * HD, SL) float32 — scan output.
        final_state_out: (NH * HD * SS, 1) float32 — final hidden state.
    """
    P_MAX = 128
    NH = dA_exp_t.shape[0]
    SL = dA_exp_t.shape[1]
    HD = hd_range.shape[0]
    SS = ss_range.shape[0]

    y_out = nl.ndarray((NH * HD, SL), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (NH * HD * SS, 1), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # Load dA_exp once: (num_heads, seq_len) -> SBUF
    dA_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dA_sb, value=0.0)
    nisa.dma_copy(
        dst=dA_sb[0:NH, 0:SL],
        src=dA_exp_t[0:NH, 0:SL],
    )

    for d in nl.affine_range(HD):
        # Load pre-computed D*x as initial y accumulator
        y_acc_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=y_acc_sb, value=0.0)
        Dx_row_start = d * NH
        nisa.dma_copy(
            dst=y_acc_sb[0:NH, 0:SL],
            src=Dx_t[Dx_row_start : Dx_row_start + NH, 0:SL],
        )

        # Accumulate over ssm_state_size
        for s in nl.affine_range(SS):
            # Load dBx for this (d, s)
            dBx_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=dBx_sb, value=0.0)
            dBx_row = (d * SS + s) * NH
            nisa.dma_copy(
                dst=dBx_sb[0:NH, 0:SL],
                src=dBx_t[dBx_row : dBx_row + NH, 0:SL],
            )

            # Run scan: state[h, t] = dA[h, t] * state[h, t-1] + dBx[h, t]
            init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=init_sb, value=0.0)

            state_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=state_sb[0:NH, 0:SL],
                data0=dA_sb[0:NH, 0:SL],
                data1=dBx_sb[0:NH, 0:SL],
                initial=init_sb[0:NH, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            # Save final state column
            final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=final_sb[0:NH, 0:1],
                src=state_sb[0:NH, SL - 1 : SL],
            )
            fs_row = (d * SS + s) * NH
            nisa.dma_copy(
                dst=final_state_out[fs_row : fs_row + NH, 0:1],
                src=final_sb[0:NH, 0:1],
            )

            # Load C for this state dim s
            C_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=C_sb, value=0.0)
            C_row = s * NH
            nisa.dma_copy(
                dst=C_sb[0:NH, 0:SL],
                src=C_t[C_row : C_row + NH, 0:SL],
            )

            # y_acc += C * state
            Cs_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=Cs_sb[0:NH, 0:SL],
                data1=C_sb[0:NH, 0:SL],
                data2=state_sb[0:NH, 0:SL],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=y_acc_sb[0:NH, 0:SL],
                data1=y_acc_sb[0:NH, 0:SL],
                data2=Cs_sb[0:NH, 0:SL],
                op=nl.add,
            )

        # Store y_acc for this d
        y_row = d * NH
        nisa.dma_copy(
            dst=y_out[y_row : y_row + NH, 0:SL],
            src=y_acc_sb[0:NH, 0:SL],
        )

    return y_out, final_state_out
