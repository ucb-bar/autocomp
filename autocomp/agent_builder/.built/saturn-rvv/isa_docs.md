# RVV Intrinsics Reference

## Type System

Vector types: `v{int|uint|float}{SEW}m{LMUL}_t`
- SEW (element width): 8, 16, 32, 64
- LMUL (register grouping): mf8, mf4, mf2, m1, m2, m4, m8
- Examples: `vfloat32m4_t` (32-bit float, LMUL=4), `vint16m2_t` (16-bit signed int, LMUL=2)
- Fractional LMUL: `vfloat32mf2_t` (LMUL=1/2, uses half a register)

Mask types: `vbool{N}_t` where N = SEW/LMUL ratio
- Examples: `vbool8_t` (e32m4 produces vbool8), `vbool4_t` (e32m8 produces vbool4)

Tuple types for segment loads: `v{type}m{LMUL}x{NF}_t`
- Example: `vfloat32m1x3_t` (3 fields of vfloat32m1, for vlseg3)

Scalar types in signatures: `float32_t`, `float64_t`, `int32_t`, `uint32_t`, etc.

In signatures below, `m<LMUL>` and `<N>` are placeholders — substitute your LMUL and mask width.

## Policy Variants

All operations support policy suffixes that change the signature. Using `vfadd` with e32m4 as an example:

```c
// No suffix: unmasked, tail-agnostic (default)
vfloat32m4_t __riscv_vfadd_vv_f32m4(vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);

// _tu: tail-undisturbed — adds vd parameter (preserves tail elements from vd)
vfloat32m4_t __riscv_vfadd_vv_f32m4_tu(vfloat32m4_t vd, vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);

// _m: masked, tail-agnostic, mask-agnostic — adds vm parameter
vfloat32m4_t __riscv_vfadd_vv_f32m4_m(vbool8_t vm, vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);

// _tum: masked, tail-undisturbed, mask-agnostic — adds vm and vd
vfloat32m4_t __riscv_vfadd_vv_f32m4_tum(vbool8_t vm, vfloat32m4_t vd, vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);

// _mu: masked, tail-agnostic, mask-undisturbed — adds vm and vd
vfloat32m4_t __riscv_vfadd_vv_f32m4_mu(vbool8_t vm, vfloat32m4_t vd, vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);

// _tumu: masked, tail-undisturbed, mask-undisturbed — adds vm and vd
vfloat32m4_t __riscv_vfadd_vv_f32m4_tumu(vbool8_t vm, vfloat32m4_t vd, vfloat32m4_t vs2, vfloat32m4_t vs1, size_t vl);
```

Key: `_tu`/`_tumu` preserve inactive tail/mask elements from `vd`. Use tail-agnostic (no suffix or `_m`) when you don't need to preserve inactive elements — it avoids a read dependency on the destination register.

## Rounding Modes

Fixed-point rounding (`vxrm`): `__RISCV_VXRM_RNU` (0), `__RISCV_VXRM_RNE` (1), `__RISCV_VXRM_RDN` (2), `__RISCV_VXRM_ROD` (3)
FP rounding (`frm`): `__RISCV_FRM_RNE` (0), `__RISCV_FRM_RTZ` (1), `__RISCV_FRM_RDN` (2), `__RISCV_FRM_RUP` (3), `__RISCV_FRM_RMM` (4)

## IEEE-754 Behavioral Notes

- **Subnormals**: fully supported, not flushed to zero.
- **NaN**: invalid operations produce the canonical quiet NaN. Input NaN payloads are NOT propagated.
- **FMA**: `vfmacc`/`vfmsac`/`vfnmacc`/`vfnmsac`/`vfmadd`/`vfmsub`/`vfnmadd`/`vfnmsub` are truly fused — one rounding at the end, not two.
- **Min/Max**: `vfmin`/`vfmax` use IEEE-754 `minNum`/`maxNum` semantics. If one operand is NaN, the non-NaN value is returned. `vfmin(-0, +0) = -0`, `vfmax(-0, +0) = +0`.

## Reciprocal Estimate Refinement Pattern

`vfrec7` and `vfrsqrt7` provide 7-bit accuracy estimates. Newton-Raphson iterations refine:
- 0 iterations ≈ bfloat16 accuracy
- 1 iteration ≈ FP16, 2 iterations ≈ FP32, 3 iterations ≈ FP64

```c
// Newton-Raphson for 1/x (one iteration):
vfloat32m4_t est = __riscv_vfrec7_v_f32m4(x, vl);
vfloat32m4_t two = __riscv_vfmv_v_f_f32m4(2.0f, vl);
two = __riscv_vfnmsac_vv_f32m4(two, x, est, vl);  // 2.0 - x * est
est = __riscv_vfmul_vv_f32m4(est, two, vl);        // refined est
```

## Arithmetic

### `__riscv_vwadd`
Operand forms: vv, vx, wv, wx
```c
vuint16m<LMUL>_t __riscv_vwaddu_vv_*(vuint8m<LMUL>_t op1, vuint8m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = sext(vs2[i]) + sext(vs1[i])` (widening, also vwaddu)

### `__riscv_vwsub`
Operand forms: vv, vx, wv, wx
```c
vuint16m<LMUL>_t __riscv_vwsubu_vv_*(vuint8m<LMUL>_t op1, vuint8m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = sext(vs2[i]) - sext(vs1[i])` (widening, also vwsubu)

## Bitwise|AND

### `__riscv_vand`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vand_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] & vs1[i]` (bitwise AND)

## Bitwise|OR

### `__riscv_vor`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vor_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] | vs1[i]` (bitwise OR)

## Bitwise|Shift

### `__riscv_vnsra`
Operand forms: wv, wx
```c
vint32m<LMUL>_t __riscv_vnsra_wv_*(vint64m<LMUL>_t op1, vuint32m<LMUL>_t shift, size_t vl);
```
Semantics: `vd[i] = narrow(vs2[i] >> shift)` (narrowing shift right arithmetic)

### `__riscv_vnsrl`
Operand forms: wv, wx
```c
vuint16m<LMUL>_t __riscv_vnsrl_wv_*(vuint32m<LMUL>_t op1, vuint16m<LMUL>_t shift, size_t vl);
```
Semantics: `vd[i] = narrow(vs2[i] >> shift)` (narrowing shift right logical)

### `__riscv_vsll`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vsll_vv_*(vint32m<LMUL>_t op1, vuint32m<LMUL>_t shift, size_t vl);
```
Semantics: `vd[i] = vs2[i] << vs1[i]` (shift left logical)

### `__riscv_vsra`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vsra_vv_*(vint32m<LMUL>_t op1, vuint32m<LMUL>_t shift, size_t vl);
```
Semantics: `vd[i] = vs2[i] >> vs1[i]` (shift right arithmetic, sign-fill)

### `__riscv_vsrl`
Operand forms: vv, vx
```c
vuint16m<LMUL>_t __riscv_vsrl_vv_*(vuint16m<LMUL>_t op1, vuint16m<LMUL>_t shift, size_t vl);
```
Semantics: `vd[i] = vs2[i] >> vs1[i]` (shift right logical, zero-fill)

## Bitwise|XOR

### `__riscv_vxor`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vxor_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] ^ vs1[i]` (bitwise XOR)

## Conversion

### `__riscv_vncvt_x_x_w`
```c
vint32m<LMUL>_t __riscv_vncvt_x_x_w_*(vint64m<LMUL>_t src, size_t vl);
```
Semantics: `res[i] = (int32_t) src[i];`

### `__riscv_vwcvt_x_x_v`
```c
vint32m<LMUL>_t __riscv_vwcvt_x_x_v_*(vint16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = sext(vs[i])` (widening convert, signed)

## Conversion|Extension

### `__riscv_vzext`
Operand forms: vf
```c
vuint16m<LMUL>_t __riscv_vzext_vf2_*(vuint8m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = zero_extend(vs[i])` (integer zero extension, vf2/vf4/vf8)

## Conversion|Float

### `__riscv_vfcvt_f_x_v`
```c
vfloat32m<LMUL>_t __riscv_vfcvt_f_x_v_*(vint32m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (float)vs[i]` (signed int to FP)

### `__riscv_vfcvt_f_xu_v`
```c
vfloat32m<LMUL>_t __riscv_vfcvt_f_xu_v_*(vuint32m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (float)vs[i]` (unsigned int to FP)

### `__riscv_vfcvt_rtz_x_f_v`
```c
vint32m<LMUL>_t __riscv_vfcvt_rtz_x_f_v_*(vfloat32m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (int)trunc(vs[i])` (FP to int, round toward zero)

### `__riscv_vfcvt_rtz_xu_f_v`
```c
vuint16m<LMUL>_t __riscv_vfcvt_rtz_xu_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (uint)trunc(vs[i])` (FP to uint, round toward zero)

### `__riscv_vfcvt_x_f_v`
```c
vint32m<LMUL>_t __riscv_vfcvt_x_f_v_*(vfloat32m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (int)vs[i]` (FP to signed int, current rounding mode)

### `__riscv_vfcvt_xu_f_v`
```c
vuint16m<LMUL>_t __riscv_vfcvt_xu_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = (uint)vs[i]` (FP to unsigned int)

### `__riscv_vfncvt_f_f_w`
```c
vfloat32m<LMUL>_t __riscv_vfncvt_f_f_w_*(vfloat64m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = narrow(vs[i])` (narrowing FP to smaller FP)

### `__riscv_vfncvt_f_x_w`
```c
vfloat32m<LMUL>_t __riscv_vfncvt_f_x_w_*(vint64m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = narrow((float)vs[i])` (narrowing int to FP)

### `__riscv_vfncvt_f_xu_w`
```c
vfloat32m<LMUL>_t __riscv_vfncvt_f_xu_w_*(vuint64m<LMUL>_t src, size_t vl);
```
Semantics: `res[i] = (float32_t) src[i];`

### `__riscv_vfncvt_rod_f_f_w`
```c
vfloat32m<LMUL>_t __riscv_vfncvt_rod_f_f_w_*(vfloat64m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = narrow_rod(vs[i])` (narrowing FP, round-to-odd)

### `__riscv_vfncvt_rtz_x_f_w`
```c
vint32m<LMUL>_t __riscv_vfncvt_rtz_x_f_w_*(vfloat64m<LMUL>_t src, size_t vl);
```
Semantics: `res[i] = (int32_t) src[i];`

### `__riscv_vfncvt_rtz_xu_f_w`
```c
vuint16m<LMUL>_t __riscv_vfncvt_rtz_xu_f_w_*(vfloat32m<LMUL>_t src, size_t vl);
```
Semantics: `res[i] = (uint16_t) src[i];`

### `__riscv_vfncvt_x_f_w`
```c
vint32m<LMUL>_t __riscv_vfncvt_x_f_w_*(vfloat64m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = narrow((int)vs[i])` (narrowing FP to int)

### `__riscv_vfncvt_xu_f_w`
```c
vuint16m<LMUL>_t __riscv_vfncvt_xu_f_w_*(vfloat32m<LMUL>_t src, size_t vl);
```
Semantics: `res[i] = (uint16_t) src[i];`

### `__riscv_vfwcvt_f_f_v`
```c
vfloat32m<LMUL>_t __riscv_vfwcvt_f_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen(vs[i])` (widening FP to larger FP)

### `__riscv_vfwcvt_f_x_v`
```c
vfloat32m<LMUL>_t __riscv_vfwcvt_f_x_v_*(vint16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((float)vs[i])` (widening int to FP)

### `__riscv_vfwcvt_f_xu_v`
```c
vfloat32m<LMUL>_t __riscv_vfwcvt_f_xu_v_*(vuint16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((float)vs[i])` (widening uint to FP)

### `__riscv_vfwcvt_rtz_x_f_v`
```c
vint32m<LMUL>_t __riscv_vfwcvt_rtz_x_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((int)trunc(vs[i]))` (widening FP to int, RTZ)

### `__riscv_vfwcvt_rtz_xu_f_v`
```c
vuint32m<LMUL>_t __riscv_vfwcvt_rtz_xu_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((uint)trunc(vs[i]))` (widening FP to uint, RTZ)

### `__riscv_vfwcvt_x_f_v`
```c
vint32m<LMUL>_t __riscv_vfwcvt_x_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((int)vs[i])` (widening FP to int)

### `__riscv_vfwcvt_xu_f_v`
```c
vuint32m<LMUL>_t __riscv_vfwcvt_xu_f_v_*(vfloat16m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = widen((uint)vs[i])` (widening FP to uint)

## Conversion|Reinterpret

### `__riscv_vreinterpret_v`
```c
vint32m<LMUL>_t __riscv_vreinterpret_v_*(vfloat32m<LMUL>_t src);
```
Semantics: reinterpret vector register bits as different type (zero cost)

## Fixed-point|Averaging

### `__riscv_vaadd`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vaadd_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = (vs2[i] + vs1[i] + round) >> 1` (averaging add, also vaaddu)

### `__riscv_vasub`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vasub_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = (vs2[i] - vs1[i] + round) >> 1` (averaging sub, also vasubu)

## Fixed-point|Clip

### `__riscv_vnclip`
Operand forms: wv, wx
```c
vint32m<LMUL>_t __riscv_vnclip_wv_*(vint64m<LMUL>_t op1, vuint32m<LMUL>_t shift, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = sat(vs2[i] >> shift)` (narrowing clip, also vnclipu)

## Fixed-point|Saturating add

### `__riscv_vsadd`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vsadd_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = sat(vs2[i] + vs1[i])` (saturating add, also vsaddu)

## Fixed-point|Saturating subtract

### `__riscv_vssub`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vssub_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = sat(vs2[i] - vs1[i])` (saturating sub, also vssubu)

## Float|Absolute

### `__riscv_vfabs_v`
```c
vfloat32m<LMUL>_t __riscv_vfabs_v_*(vfloat32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = |vs[i]|` (FP absolute value)

## Float|Add

### `__riscv_vfadd`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfadd_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] + vs1[i]` (FP add)

## Float|Compare

### `__riscv_vmfeq`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmfeq_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] == vs1[i])` (FP equal)

### `__riscv_vmfge`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmfge_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] >= vs1[i])` (FP greater-equal)

### `__riscv_vmfgt`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmfgt_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] > vs1[i])` (FP greater-than)

### `__riscv_vmfle`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmfle_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] <= vs1[i])` (FP less-equal)

### `__riscv_vmflt`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmflt_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] < vs1[i])` (FP less-than)

### `__riscv_vmfne`
Operand forms: vf, vv
```c
vbool<N>_t __riscv_vmfne_vf_*_b8(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] != vs1[i])` (FP not-equal)

## Float|Divide

### `__riscv_vfdiv`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfdiv_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] / vs1[i]` (FP divide, SLOW iterative)

## Float|Fused multiply-add

### `__riscv_vfmacc`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmacc_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = +(vs1[i] * vs2[i]) + vd[i]` (FP fused multiply-accumulate)

### `__riscv_vfmadd`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmadd_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = +(vd[i] * vs1[i]) + vs2[i]` (FP fused multiply-add)

### `__riscv_vfmsac`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmsac_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = +(vs1[i] * vs2[i]) - vd[i]` (FP fused multiply-subtract-accumulate)

### `__riscv_vfmsub`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmsub_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = +(vd[i] * vs1[i]) - vs2[i]` (FP fused multiply-subtract)

### `__riscv_vfwmacc`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfwmacc_vf_*(vfloat32m<LMUL>_t vd, float16_t vs1, vfloat16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] += widen(vs1[i]) * widen(vs2[i])` (widening FP MAC)

### `__riscv_vfwmsac`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfwmsac_vf_*(vfloat32m<LMUL>_t vd, float16_t vs1, vfloat16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = widen(vs1[i]) * widen(vs2[i]) - vd[i]`

### `__riscv_vfwnmacc`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfwnmacc_vf_*(vfloat32m<LMUL>_t vd, float16_t vs1, vfloat16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(widen(vs1[i]) * widen(vs2[i])) - vd[i]`

### `__riscv_vfwnmsac`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfwnmsac_vf_*(vfloat32m<LMUL>_t vd, float16_t vs1, vfloat16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(widen(vs1[i]) * widen(vs2[i])) + vd[i]`

## Float|Min/Max

### `__riscv_vfmax`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmax_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = fmax(vs2[i], vs1[i])` (FP maximum)

### `__riscv_vfmin`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmin_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = fmin(vs2[i], vs1[i])` (FP minimum)

## Float|Multiply

### `__riscv_vfmul`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfmul_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] * vs1[i]` (FP multiply)

## Float|Negate

### `__riscv_vfneg_v`
```c
vfloat32m<LMUL>_t __riscv_vfneg_v_*(vfloat32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = -vs[i]` (FP negate)

## Float|Reciprocal

### `__riscv_vfrec7_v`
```c
vfloat32m<LMUL>_t __riscv_vfrec7_v_*(vfloat32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] ≈ 1/vs2[i]` (7-bit reciprocal estimate)

## Float|Reciprocal sqrt

### `__riscv_vfrsqrt7_v`
```c
vfloat32m<LMUL>_t __riscv_vfrsqrt7_v_*(vfloat32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] ≈ 1/sqrt(vs2[i])` (7-bit reciprocal sqrt estimate)

## Float|Sqrt

### `__riscv_vfsqrt_v`
```c
vfloat32m<LMUL>_t __riscv_vfsqrt_v_*(vfloat32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = sqrt(vs2[i])` (SLOW iterative)

## Float|Subtract

### `__riscv_vfsub`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfsub_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] - vs1[i]` (FP subtract)

## Float|Widen|Add

### `__riscv_vfwadd`
Operand forms: vf, vv, wf, wv
```c
vfloat32m<LMUL>_t __riscv_vfwadd_vf_*(vfloat16m<LMUL>_t op1, float16_t op2, size_t vl);
```
Semantics: `vd[i] = widen(vs2[i]) + widen(vs1[i])` (widening FP add)

## Float|Widen|Multiply

### `__riscv_vfwmul`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfwmul_vf_*(vfloat16m<LMUL>_t op1, float16_t op2, size_t vl);
```
Semantics: `vd[i] = widen(vs2[i]) * widen(vs1[i])` (widening FP multiply)

## Float|Widen|Subtract

### `__riscv_vfwsub`
Operand forms: vf, vv, wf, wv
```c
vfloat32m<LMUL>_t __riscv_vfwsub_vf_*(vfloat16m<LMUL>_t op1, float16_t op2, size_t vl);
```
Semantics: `vd[i] = widen(vs2[i]) - widen(vs1[i])` (widening FP subtract)

## Fold|Reduction

### `__riscv_vfredmax`
Operand forms: vs
```c
vfloat32m<LMUL>_t __riscv_vfredmax_vs_*(vfloat32m<LMUL>_t vector, vfloat32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = fmax(vs2[0..vl-1], vs1[0])` (result in vd[0])

### `__riscv_vfredmin`
Operand forms: vs
```c
vfloat32m<LMUL>_t __riscv_vfredmin_vs_*(vfloat32m<LMUL>_t vector, vfloat32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = fmin(vs2[0..vl-1], vs1[0])` (result in vd[0])

### `__riscv_vfredosum`
Operand forms: vs
```c
vfloat32m<LMUL>_t __riscv_vfredosum_vs_*(vfloat32m<LMUL>_t vector, vfloat32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = (...((vs1[0] + vs2[0]) + vs2[1]) + ...)` (FP ordered sum, slower, result in vd[0])

### `__riscv_vfredusum`
Operand forms: vs
```c
vfloat32m<LMUL>_t __riscv_vfredusum_vs_*(vfloat32m<LMUL>_t vector, vfloat32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = vs2[0] + vs2[1] + ... + vs1[0]` (FP unordered sum, result in vd[0], extract with vfmv.f.s)

### `__riscv_vredand`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredand_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = vs2[0] & vs2[1] & ... & vs1[0]` (result in vd[0])

### `__riscv_vredmax`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredmax_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = max(vs2[0..vl-1], vs1[0])` (result in vd[0], also vredmaxu)

### `__riscv_vredmin`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredmin_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = min(vs2[0..vl-1], vs1[0])` (result in vd[0], also vredminu)

### `__riscv_vredor`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredor_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = vs2[0] | vs2[1] | ... | vs1[0]` (result in vd[0])

### `__riscv_vredsum`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredsum_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = vs2[0] + vs2[1] + ... + vs2[vl-1] + vs1[0]` (result in vd[0], extract with vmv.x.s)

### `__riscv_vredxor`
Operand forms: vs
```c
vint32m<LMUL>_t __riscv_vredxor_vs_*(vint32m<LMUL>_t vector, vint32m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = vs2[0] ^ vs2[1] ^ ... ^ vs1[0]` (result in vd[0])

## Initialize|Create tuple

### `__riscv_vcreate_v`
```c
vint32m<LMUL>_t __riscv_vcreate_v_*(vint32m<LMUL>_t v0, vint32m<LMUL>_t v1, vint32m<LMUL>_t v2, vint32m<LMUL>_t v3);
```
Semantics: create tuple from individual registers (zero cost)

## Initialize|Extract

### `__riscv_vmv_x_s`
```c
int32_t __riscv_vmv_x_s_*_i32(vint32m<LMUL>_t src);
```
Semantics: `scalar = vs[0]` (extract element 0 to scalar, SCALAR WRITEBACK)

## Initialize|Float move

### `__riscv_vfmv_f_s_f`
```c
float32_t __riscv_vfmv_f_s_*_f32(vfloat32m<LMUL>_t src);
```
Semantics: `res[0] = scalar; return res;`

### `__riscv_vfmv_s_f`
```c
vfloat32m<LMUL>_t __riscv_vfmv_s_f_*(float32_t src, size_t vl);
```
Semantics: `vd[0] = fp_scalar` (set element 0 from FP scalar)

### `__riscv_vfmv_v_f`
```c
vfloat32m<LMUL>_t __riscv_vfmv_v_f_*(float32_t src, size_t vl);
```
Semantics: `vd[i] = fp_scalar` (broadcast FP scalar to all elements)

## Initialize|Get from tuple

### `__riscv_vget_v`
```c
vint32m<LMUL>_t __riscv_vget_v_*(vint32m<LMUL>_t src, size_t index);
```
Semantics: extract register from tuple (zero cost)

## Initialize|Insert

### `__riscv_vmv_s_x`
```c
vint32m<LMUL>_t __riscv_vmv_s_x_*(int32_t src, size_t vl);
```
Semantics: `vd[0] = scalar` (set element 0 from scalar)

## Initialize|Merge

### `__riscv_vmerge`
Operand forms: vvm, vxm
```c
vint32m<LMUL>_t __riscv_vmerge_vvm_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, vbool<N>_t mask, size_t vl);
```
Semantics: `vd[i] = mask[i] ? vs1[i] : vs2[i]` (merge under mask)

## Initialize|Move

### `__riscv_vmv_v_v`
```c
vint32m<LMUL>_t __riscv_vmv_v_v_*(vint32m<LMUL>_t src, size_t vl);
```
Semantics: `vd = vs` (vector copy)

### `__riscv_vmv_v_x`
```c
vint32m<LMUL>_t __riscv_vmv_v_x_*(int32_t src, size_t vl);
```
Semantics: `vd[i] = scalar` (broadcast scalar to all elements)

## Initialize|Set in tuple

### `__riscv_vset_v`
```c
vint32m<LMUL>_t __riscv_vset_v_*(vint32m<LMUL>_t dest, size_t index, vint32m<LMUL>_t val);
```
Semantics: insert register into tuple (zero cost)

## Initialize|Set specific vl

### `__riscv_vsetvl_e16m`
```c
size_t __riscv_vsetvl_e16m1(size_t avl);
```
Semantics: `vlmax = vlmax(e16, m1);  if (avl <= vlmax) {  } else if (vlmax < avl < vlmax*2) {  } else {`

### `__riscv_vsetvl`
```c
size_t __riscv_vsetvl_e16mf2(size_t avl);
```
Semantics: sets vl for given SEW/LMUL. If avl <= VLMAX, vl = avl. If avl >= 2*VLMAX, vl = VLMAX. In between, vl is implementation-defined in [ceil(avl/2), VLMAX].

### `__riscv_vsetvl_e32m`
```c
size_t __riscv_vsetvl_e32m1(size_t avl);
```
Semantics: `vlmax = vlmax(e32, m1);  if (avl <= vlmax) {  } else if (vlmax < avl < vlmax*2) {  } else {`

### `__riscv_vsetvl_e64m`
```c
size_t __riscv_vsetvl_e64m1(size_t avl);
```
Semantics: `vlmax = vlmax(e64, m1);  if (avl <= vlmax) {  } else if (vlmax < avl < vlmax*2) {  } else {`

### `__riscv_vsetvl_e8m`
```c
size_t __riscv_vsetvl_e8m1(size_t avl);
```
Semantics: `vlmax = vlmax(e8, m1);  if (avl <= vlmax) {  } else if (vlmax < avl < vlmax*2) {  } else {`

## Initialize|Set to vlmax

### `__riscv_vsetvlmax_e16m`
```c
size_t __riscv_vsetvlmax_e16m1();
```
Semantics: `return vlmax(e16, m1);`

### `__riscv_vsetvlmax`
```c
size_t __riscv_vsetvlmax_e16mf2();
```
Semantics: `vl = VLMAX` — set vector length to maximum for given SEW/LMUL

### `__riscv_vsetvlmax_e32m`
```c
size_t __riscv_vsetvlmax_e32m1();
```
Semantics: `return vlmax(e32, m1);`

### `__riscv_vsetvlmax_e64m`
```c
size_t __riscv_vsetvlmax_e64m1();
```
Semantics: `return vlmax(e64, m1);`

### `__riscv_vsetvlmax_e8m`
```c
size_t __riscv_vsetvlmax_e8m1();
```
Semantics: `return vlmax(e8, m1);`

## Initialize|Undefined

### `__riscv_vundefined`
```c
vint32m<LMUL>_t __riscv_vundefined_*();
```
Semantics: undefined value placeholder (zero cost, no instruction)

## Integer|Add

### `__riscv_vadd`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vadd_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] + vs1[i]` (or scalar)

## Integer|Divide

### `__riscv_vdiv`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vdiv_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] / vs1[i]` (also vdivu)

## Integer|Min/Max

### `__riscv_vmax`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmax_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = max(vs2[i], vs1[i])` (signed, also unsigned vmaxu)

### `__riscv_vmin`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmin_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = min(vs2[i], vs1[i])` (signed, also unsigned vminu)

## Integer|Multiply

### `__riscv_vmul`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmul_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] * vs1[i]` (low bits)

## Integer|Multiply-add

### `__riscv_vfnmacc`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfnmacc_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(vs1[i] * vs2[i]) - vd[i]` (FP negated fused multiply-accumulate)

### `__riscv_vfnmadd`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfnmadd_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(vd[i] * vs1[i]) - vs2[i]` (FP negated fused multiply-add)

### `__riscv_vfnmsac`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfnmsac_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(vs1[i] * vs2[i]) + vd[i]` (FP negated fused multiply-subtract-accumulate)

### `__riscv_vfnmsub`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfnmsub_vf_*(vfloat32m<LMUL>_t vd, float32_t rs1, vfloat32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(vd[i] * vs1[i]) + vs2[i]` (FP negated fused multiply-subtract)

### `__riscv_vmacc`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmacc_vv_*(vint32m<LMUL>_t vd, vint32m<LMUL>_t vs1, vint32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] += vs1[i] * vs2[i]` (multiply-accumulate)

### `__riscv_vmadd`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmadd_vv_*(vint32m<LMUL>_t vd, vint32m<LMUL>_t vs1, vint32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = vd[i] * vs1[i] + vs2[i]` (multiply-add)

### `__riscv_vnmsac`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vnmsac_vv_*(vint32m<LMUL>_t vd, vint32m<LMUL>_t vs1, vint32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] -= vs1[i] * vs2[i]` (negate multiply-subtract-accumulate)

### `__riscv_vnmsub`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vnmsub_vv_*(vint32m<LMUL>_t vd, vint32m<LMUL>_t vs1, vint32m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = -(vd[i] * vs1[i]) + vs2[i]`

### `__riscv_vwmacc`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vwmacc_vv_*(vint32m<LMUL>_t vd, vint16m<LMUL>_t vs1, vint16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] += vs2[i] * vs1[i]` (widening MAC, also vwmaccu/vwmaccsu/vwmaccus)

## Integer|Multiply|Widening

### `__riscv_vwmul`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vwmul_vv_*(vint16m<LMUL>_t op1, vint16m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] * vs1[i]` (widening, also vwmulu/vwmulsu)

## Integer|Negate

### `__riscv_vneg_v`
```c
vint32m<LMUL>_t __riscv_vneg_v_*(vint32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = -vs[i]` (integer negate)

## Integer|Remainder

### `__riscv_vrem`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vrem_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] % vs1[i]` (also vremu)

## Integer|Subtract

### `__riscv_vsub`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vsub_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] - vs1[i]` (or scalar)

## Mask|Find first set

### `__riscv_vfirst_m`
```c
long __riscv_vfirst_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: `scalar = index of first set bit, or -1` (SCALAR WRITEBACK)

## Mask|Logical

### `__riscv_vmand`
Operand forms: mm
```c
vbool<N>_t __riscv_vmand_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = vs2 & vs1` (mask AND)

### `__riscv_vmandn`
Operand forms: mm
```c
vbool<N>_t __riscv_vmandn_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = vs2 & ~vs1` (mask AND-NOT)

### `__riscv_vmnand`
Operand forms: mm
```c
vbool<N>_t __riscv_vmnand_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = ~(vs2 & vs1)` (mask NAND)

### `__riscv_vmnor`
Operand forms: mm
```c
vbool<N>_t __riscv_vmnor_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = ~(vs2 | vs1)` (mask NOR)

### `__riscv_vmnot_m`
```c
vbool<N>_t __riscv_vmnot_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: `vd = ~vs` (mask NOT)

### `__riscv_vmor`
Operand forms: mm
```c
vbool<N>_t __riscv_vmor_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = vs2 | vs1` (mask OR)

### `__riscv_vmorn`
Operand forms: mm
```c
vbool<N>_t __riscv_vmorn_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = vs2 | ~vs1` (mask OR-NOT)

### `__riscv_vmxnor`
Operand forms: mm
```c
vbool<N>_t __riscv_vmxnor_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = ~(vs2 ^ vs1)` (mask XNOR)

### `__riscv_vmxor`
Operand forms: mm
```c
vbool<N>_t __riscv_vmxor_mm_b1(vbool<N>_t op1, vbool<N>_t op2, size_t vl);
```
Semantics: `vd = vs2 ^ vs1` (mask XOR)

## Mask|Population count

### `__riscv_vcpop_m`
```c
unsigned long __riscv_vcpop_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: `scalar = popcount(vs)` (count set mask bits, SCALAR WRITEBACK)

## Memory|Fault-only-first load

### `__riscv_vleNff_v`
```c
vfloat16m<LMUL>_t __riscv_vle16ff_v_*(const float16_t * base, size_t * new_vl, size_t vl);
```
Semantics: fault-only-first load — loads up to vl elements, sets new_vl on fault

## Memory|Indexed|Load/gather ordered

### `__riscv_vloxeiN_v`
```c
vint32m<LMUL>_t __riscv_vloxei16_v_*(const int32_t * base, vuint16m<LMUL>_t bindex, size_t vl);
```
Semantics: indexed (gather) load — `vd[i] = mem[base + bindex[i]]` (ordered)

## Memory|Indexed|Load/gather unordered

### `__riscv_vluxeiN_v`
```c
vint32m<LMUL>_t __riscv_vluxei16_v_*(const int32_t * base, vuint16m<LMUL>_t bindex, size_t vl);
```
Semantics: indexed (gather) load — `vd[i] = mem[base + bindex[i]]` (unordered)

## Memory|Indexed|Store/scatter ordered

### `__riscv_vsoxeiN_v`
```c
void __riscv_vsoxei16_v_*(int32_t * base, vuint16m<LMUL>_t bindex, vint32m<LMUL>_t value, size_t vl);
```
Semantics: indexed (scatter) store — `mem[base + bindex[i]] = vs[i]` (ordered)

## Memory|Indexed|Store/scatter unordered

### `__riscv_vsuxeiN_v`
```c
void __riscv_vsuxei16_v_*(int32_t * base, vuint16m<LMUL>_t bindex, vint32m<LMUL>_t value, size_t vl);
```
Semantics: indexed (scatter) store — `mem[base + bindex[i]] = vs[i]` (unordered)

## Memory|Load

### `__riscv_vleN_v`
```c
vfloat16m<LMUL>_t __riscv_vle16_v_*(const float16_t * base, size_t vl);
```
Semantics: loads vl contiguous elements from memory (unit-stride)

## Memory|Mask

### `__riscv_vlm_v`
```c
vbool<N>_t __riscv_vlm_v_b1(const uint8_t * base, size_t vl);
```
Semantics: load mask register from memory

### `__riscv_vsm_v`
```c
void __riscv_vsm_v_b1(uint8_t * base, vbool<N>_t value, size_t vl);
```
Semantics: store mask register to memory

## Memory|Store

### `__riscv_vseN_v`
```c
void __riscv_vse16_v_*(float16_t * base, vfloat16m<LMUL>_t value, size_t vl);
```
Semantics: stores vl contiguous elements to memory (unit-stride)

## Memory|Strided|Load

### `__riscv_vlseN_v`
```c
vfloat16m<LMUL>_t __riscv_vlse16_v_*(const float16_t * base, ptrdiff_t bstride, size_t vl);
```
Semantics: loads vl elements with stride bytes between each (strided load)

## Memory|Strided|Store

### `__riscv_vsseN_v`
```c
void __riscv_vsse16_v_*(float16_t * base, ptrdiff_t bstride, vfloat16m<LMUL>_t value, size_t vl);
```
Semantics: stores vl elements with stride bytes between each (strided store)

## Other

### `__riscv_vadc`
Operand forms: vvm, vxm
```c
vint32m<LMUL>_t __riscv_vadc_vvm_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, vbool<N>_t carryin, size_t vl);
```
Semantics: `vd[i] = vs2[i] + vs1[i] + carry[i]` (add with carry)

### `__riscv_vcpop_v`
```c
vuint16m<LMUL>_t __riscv_vcpop_v_*(vuint16m<LMUL>_t vs2, size_t vl);
```
Vector Basic Bit-manipulation - Vector Population Count

### `__riscv_vfclass_v`
```c
vuint16m<LMUL>_t __riscv_vfclass_v_*(vfloat16m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = classify(vs2[i])` (returns 10-bit class mask: NaN, inf, normal, subnormal, zero)

### `__riscv_vfmerge`
Operand forms: vfm
```c
vfloat32m<LMUL>_t __riscv_vfmerge_vfm_*(vfloat32m<LMUL>_t op1, float32_t op2, vbool<N>_t mask, size_t vl);
```
Semantics: `vd[i] = mask[i] ? fp_scalar : vs2[i]` (FP merge under mask)

### `__riscv_vfrdiv`
Operand forms: vf
```c
vfloat32m<LMUL>_t __riscv_vfrdiv_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = scalar / vs2[i]` (FP reverse divide, SLOW)

### `__riscv_vfrsub`
Operand forms: vf
```c
vfloat32m<LMUL>_t __riscv_vfrsub_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = scalar - vs2[i]` (FP reverse subtract)

### `__riscv_vfsgnj`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfsgnj_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = |vs2[i]| * sign(vs1[i])` (copy sign)

### `__riscv_vfsgnjn`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfsgnjn_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = |vs2[i]| * -sign(vs1[i])` (negate sign)

### `__riscv_vfsgnjx`
Operand forms: vf, vv
```c
vfloat32m<LMUL>_t __riscv_vfsgnjx_vf_*(vfloat32m<LMUL>_t op1, float32_t op2, size_t vl);
```
Semantics: `vd[i] = vs2[i] * sign(vs1[i])` (XOR signs = conditional negate)

### `__riscv_vfslide1down`
Operand forms: vf
```c
vfloat32m<LMUL>_t __riscv_vfslide1down_vf_*(vfloat32m<LMUL>_t src, float32_t value, size_t vl);
```
Semantics: `vd[i] = vs2[i+1]; vd[vl-1] = fp_scalar` (FP slide down by 1)

### `__riscv_vfslide1up`
Operand forms: vf
```c
vfloat32m<LMUL>_t __riscv_vfslide1up_vf_*(vfloat32m<LMUL>_t src, float32_t value, size_t vl);
```
Semantics: `vd[0] = fp_scalar; vd[i+1] = vs2[i]` (FP slide up by 1)

### `__riscv_vfwredosum`
Operand forms: vs
```c
vfloat64m<LMUL>_t __riscv_vfwredosum_vs_*(vfloat32m<LMUL>_t vector, vfloat64m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = widen ordered sum` (widening FP ordered sum, result in vd[0])

### `__riscv_vfwredusum`
Operand forms: vs
```c
vfloat64m<LMUL>_t __riscv_vfwredusum_vs_*(vfloat32m<LMUL>_t vector, vfloat64m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = widen(vs2[0]) + ... + vs1[0]` (widening FP unordered sum, result in vd[0])

### `__riscv_vghsh`
Operand forms: vv
```c
vuint32m<LMUL>_t __riscv_vghsh_vv_*(vuint32m<LMUL>_t vd, vuint32m<LMUL>_t vs2, vuint32m<LMUL>_t vs1, size_t vl);
```
Vector GCM/GMAC

### `__riscv_vid_v`
```c
vuint16m<LMUL>_t __riscv_vid_v_*(size_t vl);
```
Semantics: `vd[i] = i` (element index vector)

### `__riscv_viota_m`
```c
vuint16m<LMUL>_t __riscv_viota_m_*(vbool<N>_t op1, size_t vl);
```
Semantics: `vd[i] = popcount(vs[0..i-1])` (prefix popcount of mask)

### `__riscv_vlmul_ext_v`
```c
vint32m<LMUL>_t __riscv_vlmul_ext_v_*(vint32m<LMUL>_t op1);
```
Semantics: extend to higher LMUL (zero cost, upper bits undefined)

### `__riscv_vlmul_trunc_v`
```c
vint32m<LMUL>_t __riscv_vlmul_trunc_v_*(vint32m<LMUL>_t op1);
```
Semantics: truncate to lower LMUL (zero cost, discard upper registers)

### `__riscv_vloxsegNFeiN_v`
```c
vint32m4x2_t __riscv_vloxseg2ei16_v_*x2(const int32_t * base, vuint16m<LMUL>_t bindex, size_t vl);
```
Semantics: indexed segment load (ordered)

### `__riscv_vlsegNFeN_v`
```c
vfloat16m1x2_t __riscv_vlseg2e16_v_*x2(const float16_t * base, size_t vl);
```
Semantics: segment load — loads NF interleaved fields per element (AOS to SOA)

### `__riscv_vlsegNFeNff_v`
```c
vfloat16m1x2_t __riscv_vlseg2e16ff_v_*x2(const float16_t * base, size_t * new_vl, size_t vl);
```
Semantics: fault-only-first segment load

### `__riscv_vlssegNFeN_v`
```c
vfloat16m1x2_t __riscv_vlsseg2e16_v_*x2(const float16_t * base, ptrdiff_t bstride, size_t vl);
```
Semantics: strided segment load

### `__riscv_vluxsegNFeiN_v`
```c
vint32m4x2_t __riscv_vluxseg2ei16_v_*x2(const int32_t * base, vuint16m<LMUL>_t bindex, size_t vl);
```
Semantics: indexed segment load (unordered)

### `__riscv_vmadc`
Operand forms: vv, vvm, vx, vxm
```c
vbool<N>_t __riscv_vmadc_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = carry_out(vs2[i] + vs1[i])` (carry-out test)

### `__riscv_vmclr_m`
```c
vbool<N>_t __riscv_vmclr_m_b1(size_t vl);
```
Semantics: `vd = 0` (clear all mask bits)

### `__riscv_vmmv_m`
```c
vbool<N>_t __riscv_vmmv_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: `vd = vs` (mask move/copy)

### `__riscv_vmsbc`
Operand forms: vv, vvm, vx, vxm
```c
vbool<N>_t __riscv_vmsbc_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = borrow_out(vs2[i] - vs1[i])` (borrow-out test)

### `__riscv_vmsbf_m`
```c
vbool<N>_t __riscv_vmsbf_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: set mask bits before first set bit in vs

### `__riscv_vmseq`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmseq_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] == vs1[i])` (also unsigned)

### `__riscv_vmset_m`
```c
vbool<N>_t __riscv_vmset_m_b1(size_t vl);
```
Semantics: `vd = all 1s` (set all mask bits)

### `__riscv_vmsge`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmsge_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] >= vs1[i])` (signed, also vmsgeu)

### `__riscv_vmsgt`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmsgt_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] > vs1[i])` (signed, also vmsgtu)

### `__riscv_vmsif_m`
```c
vbool<N>_t __riscv_vmsif_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: set mask bits before and including first set bit in vs

### `__riscv_vmsle`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmsle_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] <= vs1[i])` (signed, also vmsleu)

### `__riscv_vmslt`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmslt_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] < vs1[i])` (signed, also vmsltu)

### `__riscv_vmsne`
Operand forms: vv, vx
```c
vbool<N>_t __riscv_vmsne_vv_*_b8(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] != vs1[i])` (also unsigned)

### `__riscv_vmsof_m`
```c
vbool<N>_t __riscv_vmsof_m_b1(vbool<N>_t op1, size_t vl);
```
Semantics: set only the first set bit in vs

### `__riscv_vmulh`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vmulh_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, size_t vl);
```
Semantics: `vd[i] = (vs2[i] * vs1[i]) >> SEW` (high bits, also vmulhu/vmulhsu)

### `__riscv_vnot_v`
```c
vint32m<LMUL>_t __riscv_vnot_v_*(vint32m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = ~vs[i]` (bitwise NOT)

### `__riscv_vrev8_v`
```c
vuint16m<LMUL>_t __riscv_vrev8_v_*(vuint16m<LMUL>_t vs2, size_t vl);
```
Semantics: `vd[i] = byte_reverse(vs[i])` (reverse bytes within each element)

### `__riscv_vrsub`
Operand forms: vx
```c
vint32m<LMUL>_t __riscv_vrsub_vx_*(vint32m<LMUL>_t op1, int32_t op2, size_t vl);
```
Semantics: `vd[i] = scalar - vs2[i]` (reverse subtract)

### `__riscv_vsbc`
Operand forms: vvm, vxm
```c
vint32m<LMUL>_t __riscv_vsbc_vvm_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, vbool<N>_t borrowin, size_t vl);
```
Semantics: `vd[i] = vs2[i] - vs1[i] - borrow[i]` (subtract with borrow)

### `__riscv_vsext`
Operand forms: vf
```c
vint32m<LMUL>_t __riscv_vsext_vf2_*(vint16m<LMUL>_t op1, size_t vl);
```
Semantics: `vd[i] = sign_extend(vs[i])` (integer sign extension, vf2/vf4/vf8)

### `__riscv_vslide1down`
Operand forms: vx
```c
vint32m<LMUL>_t __riscv_vslide1down_vx_*(vint32m<LMUL>_t src, int32_t value, size_t vl);
```
Semantics: `vd[i] = vs2[i+1]; vd[vl-1] = scalar` (slide down by 1, insert scalar)

### `__riscv_vslide1up`
Operand forms: vx
```c
vint32m<LMUL>_t __riscv_vslide1up_vx_*(vint32m<LMUL>_t src, int32_t value, size_t vl);
```
Semantics: `vd[0] = scalar; vd[i+1] = vs2[i]` (slide up by 1, insert scalar)

### `__riscv_vsmul`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vsmul_vv_*(vint32m<LMUL>_t op1, vint32m<LMUL>_t op2, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = (vs2[i] * vs1[i] + round) >> (SEW-1)` (fractional multiply)

### `__riscv_vsoxsegNFeiN_v`
```c
void __riscv_vsoxseg2ei16_v_*x2(int32_t * base, vuint16m<LMUL>_t bindex, vint32m4x2_t v_tuple, size_t vl);
```
Semantics: indexed segment store (ordered)

### `__riscv_vssegNFeN_v`
```c
void __riscv_vsseg2e16_v_*x2(float16_t * base, vfloat16m1x2_t v_tuple, size_t vl);
```
Semantics: segment store — stores NF interleaved fields per element (SOA to AOS)

### `__riscv_vssra`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vssra_vv_*(vint32m<LMUL>_t op1, vuint32m<LMUL>_t shift, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = (vs2[i] >> shift + round)` (scaling shift right arithmetic)

### `__riscv_vssrl`
Operand forms: vv, vx
```c
vuint16m<LMUL>_t __riscv_vssrl_vv_*(vuint16m<LMUL>_t op1, vuint16m<LMUL>_t shift, unsigned int vxrm, size_t vl);
```
Semantics: `vd[i] = (vs2[i] >> shift + round)` (scaling shift right logical)

### `__riscv_vsssegNFeN_v`
```c
void __riscv_vssseg2e16_v_*x2(float16_t * base, ptrdiff_t bstride, vfloat16m1x2_t v_tuple, size_t vl);
```
Semantics: strided segment store

### `__riscv_vsuxsegNFeiN_v`
```c
void __riscv_vsuxseg2ei16_v_*x2(int32_t * base, vuint16m<LMUL>_t bindex, vint32m4x2_t v_tuple, size_t vl);
```
Semantics: indexed segment store (unordered)

### `__riscv_vwcvtu_x_x_v`
```c
vuint16m<LMUL>_t __riscv_vwcvtu_x_x_v_*(vuint8m<LMUL>_t src, size_t vl);
```
Semantics: `vd[i] = zext(vs[i])` (widening convert, unsigned)

### `__riscv_vwredsum`
Operand forms: vs
```c
vint64m<LMUL>_t __riscv_vwredsum_vs_*(vint32m<LMUL>_t vector, vint64m<LMUL>_t scalar, size_t vl);
```
Semantics: `vd[0] = widen(vs2[0]) + ... + vs1[0]` (widening int sum, result in vd[0], also vwredsumu)

## Permutation|Compress

### `__riscv_vcompress_vm`
```c
vint32m<LMUL>_t __riscv_vcompress_vm_*(vint32m<LMUL>_t src, vbool<N>_t mask, size_t vl);
```
Semantics: packs vs2 elements where mask=1 contiguously into vd (SLOW 1 elem/cycle)

## Permutation|Gather

### `__riscv_vrgather`
Operand forms: vv, vx
```c
vint32m<LMUL>_t __riscv_vrgather_vv_*(vint32m<LMUL>_t op1, vuint32m<LMUL>_t index, size_t vl);
```
Semantics: `vd[i] = (vs1[i] < VLMAX) ? vs2[vs1[i]] : 0` (register gather, out-of-range indices yield 0, SLOW 1 elem/cycle)

### `__riscv_vrgatherei16`
Operand forms: vv
```c
vint32m<LMUL>_t __riscv_vrgatherei16_vv_*(vint32m<LMUL>_t op1, vuint16m<LMUL>_t index, size_t vl);
```
Semantics: `vd[i] = (vs1[i] < VLMAX) ? vs2[vs1[i]] : 0` (gather with 16-bit indices, out-of-range yields 0)

## Permutation|Slide

### `__riscv_vslidedown`
Operand forms: vx
```c
vint32m<LMUL>_t __riscv_vslidedown_vx_*(vint32m<LMUL>_t src, size_t offset, size_t vl);
```
Semantics: `vd[i] = vs2[i+offset]` (slide elements down by offset)

### `__riscv_vslideup`
Operand forms: vx
```c
vint32m<LMUL>_t __riscv_vslideup_vx_*(vint32m<LMUL>_t dest, vint32m<LMUL>_t src, size_t offset, size_t vl);
```
Semantics: `vd[i+offset] = vs2[i]` (slide elements up by offset)
