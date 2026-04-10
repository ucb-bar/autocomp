## Data Types and Constants

### __RISCV_VXRM

enum __RISCV_VXRM {
  __RISCV_VXRM_RNU = 0,
  __RISCV_VXRM_RNE = 1,
  __RISCV_VXRM_RDN = 2,
  __RISCV_VXRM_ROD = 3,
};

---

### __RISCV_FRM

enum __RISCV_FRM {
  __RISCV_FRM_RNE = 0,
  __RISCV_FRM_RTZ = 1,
  __RISCV_FRM_RDN = 2,
  __RISCV_FRM_RUP = 3,
  __RISCV_FRM_RMM = 4,
};

---

### vundefined

[[pseudo-vundefined]]
=== `vundefined`
The `vundefined` intrinsics are placeholders to represent unspecified values in variable initialization, or as arguments of `vset` and `vcreate`. These pseudo intrinsics are not mapped to any instruction.


## Vector Configuration and Control

### vsetvl

[[pseudo-vsetvl]]
=== `vsetvl`

The `vsetvl` intrinsics return the number of elements processed in a stripmining loop when  provided with the element width and LMUL in the intrinsic suffix. This pseudo intrinsic is typically mapped to `vsetvli` or `vsetivli` instructions.

NOTE: The implementation must respect the ratio between SEW and LMUL given to the intrinsic. On the other hand, as mentioned in <<control-of-vl>>, the `vsetvl` intrinsics do not necessarily map to the emission a `vsetvli` or `vsetivli` instruction of that exact SEW and LMUL provided. The actual value written to the `vl` control register is an implementation defined behavior and typically not known until runtime.


---

### vsetvlmax

[[pseudo-vsetvlmax]]
=== `vsetvlmax`

The `vsetvlmax` intrinsics return `VLMAX` when provided with the element width and LMUL in the intrinsic suffix. This pseudo intrinsic is typically mapped to the `vsetvli` instruction.

NOTE: As mentioned in <<control-of-vl>>, the `vsetvlmax` intrinsics do not necessarily map to the emission a `vsetvli` instruction of that exact SEW and LMUL provided. The actual value written to the `vl` control register is an implementation defined behavior and typically not known until runtime.


---

### vlenb

[[pseudo-vlenb]]
=== `vlenb`

The `vlenb` intrinsic returns what is held inside the read-only CSR `vlenb`, which is the vector register length in bytes. This pseudo intrinsic is mapped to a `csrr` instruction that reads from the CSR `vlenb`.

[,c]
----
unsigned __riscv_vlenb();
----

## Vector Register Manipulation

### vreinterpret

[[pseudo-vreinterpret]]
=== `vreinterpret`

The `vreinterpret` intrinsics are provided for users to transition across the strongly-typed scheme. The intrinsic is limited to conversion between types operating upon the same number of registers.
These intrinsics are not mapped to any instruction because reinterpretation of registers is a no-operation.

These pseudo intrinsics do not alter the bits held by a register. Please use `vfcvt/v(f)wcvt/v(f)ncvt` intrinsics if you seek to extend, narrow, or perform real float/interger type conversions for the values.


---

### vget

[[pseudo-vget]]
=== `vget`

The `vget` intrinsics allow users to obtain small LMUL values from larger LMUL ones. The `vget` intrinsics also allows users to extract non-tuple (`NFIELD=1`) types from tuple (`NFIELD>1`) types after segment load intrinsics. The index provided must be a constant known at compile time.

These pseudo intrinsics do not map to any real instruction. The compiler may emit zero or more instructions to implement the semantics of these pseudo intrinsics. The precise set of instructions emitted is a compiler optimization issue.


---

### vset

[[pseudo-vset]]
=== `vset`

The `vset` intrinsics allow users to combine small LMUL values into larger LMUL ones. The `vset` intrinsics also allows users to combine non-tuple (`NFIELD=1`) types to tuple (`NFIELD>1`) types for segment store intrinsics. The index provided must be a constant known at compile time.

These pseudo intrinsics do not map to any real instruction. The compiler may emit zero or more instructions to implement the semantics of these pseudo intrinsics. The precise set of instructions emitted is a compiler optimization issue.


---

### vlmul_trunc

[[pseudo-vlmul_trunc]]
=== `vlmul_trunc`

The `vlmul_trunc` intrinsics are syntactic sugar for RVV vector types other than tuples and have the same semantic as `vget` with the `index` operand having the value `0`.


---

### vlmul_ext

[[pseudo-vlmul_ext]]
=== `vlmul_ext`

The `vlmul_ext` intrinsics are syntactic sugar for RVV vector types other than tuples and have the same semantic as `vset` with the `index` operand having the value `0`.


---

### vcreate

[[pseudo-vcreate]]
=== `vcreate`

The `vcreate` intrinsics are syntactic sugar for the creation of values of RVV types. They have the same semantic as multiple `vset` pseudo intrinsics filling in values accordingly.

.Pseudo intrinsic `vcreate` used to build a SEW=32, LMUL=4 `float` vector (`vfloat32m4`) from two SEW=32, LMUL=2 `float` vectors (`vfloat32m2`).
====
[,c]
----
// Given the following declarations
vfloat32m4_t dest;
vfloat32m2_t v0 = ...;
vfloat32m2_t v1 = ...;

// this pseudo intrinsic
dest = __riscv_vcreate_v_f32m2_f32m4(v0, v1);

// is semantically equivalent to
dest = __riscv_vset_v_f32m2_f32m4(__riscv_vundefined_f32m4(), 0, v0);
dest = __riscv_vset_v_f32m2_f32m4(dest, 1, v1);
----
====

## Naming Schemes and Conventions

### Policy and masked naming scheme

[[policy-and-masked-naming-scheme]]
=== Policy and masked naming scheme

With the default policy scheme mentioned under <<control-of-policy>>, each intrinsic provides corresponding variants for their available control of `vm`, `vta` and `vma`. The following list enumerates the possible suffixes.

* No suffix: Represents an unmasked (`vm=1`) vector operation with tail-agnostic (`vta=1`)
* `_tu` suffix: Represents an unmasked (`vm=1`) vector operation with tail-undisturbed (`vta=0`) policy
* `_m` suffix: Represents a masked (`vm=0`) vector operation with tail-agnostic (`vta=1`), mask-agnostic (`vma=1`) policy
* `_tum` suffix: Represents a masked (`vm=0`) vector operation with tail-undisturbed (`vta=0`), mask-agnostic (`vma=1`) policy
* `_mu` suffix: Represents a masked (`vm=0`) vector operation with tail-agnostic (`vta=1`), mask-undisturbed (`vma=0`) policy
* `_tumu` suffix: Represents a masked (`vm=0`) vector operation with tail-undisturbed (`vta=0`), mask-undisturbed (`vma=0`) policy

Using `vadd` with EEW=32 and EMUL=1 as an example, the variants are:

[,c]
----
// vm=1, vta=1
vint32m1_t __riscv_vadd_vv_i32m1(vint32m1_t vs2, vint32m1_t vs1, size_t vl);
// vm=1, vta=0
vint32m1_t __riscv_vadd_vv_i32m1_tu(vint32m1_t vd, vint32m1_t vs2,
                                    vint32m1_t vs1, size_t vl);
// vm=0, vta=1, vma=1
vint32m1_t __riscv_vadd_vv_i32m1_m(vbool32_t vm, vint32m1_t vs2, vint32m1_t vs1,
                                   size_t vl);
// vm=0, vta=0, vma=1
vint32m1_t __riscv_vadd_vv_i32m1_tum(vbool32_t vm, vint32m1_t vd,
                                     vint32m1_t vs2, vint32m1_t vs1, size_t vl);
// vm=0, vta=1, vma=0
vint32m1_t __riscv_vadd_vv_i32m1_mu(vbool32_t vm, vint32m1_t vd, vint32m1_t vs2,
                                    vint32m1_t vs1, size_t vl);
// vm=0, vta=0, vma=0
vint32m1_t __riscv_vadd_vv_i32m1_tumu(vbool32_t vm, vint32m1_t vd,
                                      vint32m1_t vs2, vint32m1_t vs1,
                                      size_t vl);
----

---

### Explicit (Non-overloaded) naming scheme

[[explicit-naming-scheme]]
=== Explicit (Non-overloaded) naming scheme

In general, the intrinsics are encoded as the following. The intrinsics under this naming scheme are the "non-overloaded intrinsics", which in parallel we have the "overloaded intrinsics" defined under <<implicit-naming-scheme>>.

The naming rules are as follows.

[,c]
----
__riscv_{V_INSTRUCTION_MNEMONIC}_{OPERAND_MNEMONIC}_{RETURN_TYPE}_{ROUND_MODE}_{POLICY}{(...)
----

* `OPERAND_MNEMONIC` are like `v`, `vv`, `vx`, `vs`, `vvm`, `vxm`
* `RETURN_TYPE` depends on whether the return type of the vector instruction is a mask register...
** For intrinsics that represents instructions with a non-mask destination register:
*** `EEW` is one of `i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | f16 | f32 | f64`.
*** `EMUL` is one of `m1 | m2 | m4 | m8 | mf2 | mf4 | mf8`.
*** <<type-system>> explains the limited enumeration of EEW-EMUL pairs.
** For intrinsics that represent intrinsics with a mask destination register:
*** `RETURN_TYPE` is one of `b1 | b2 | b4 | b8 | b16 | b32 | b64`, which is derived from the ratio `EEW`/`EMUL`.
* `V_INSTRUCTION_MNEMONIC` are like `vadd`, `vfmacc`, `vsadd`.
* `ROUND_MODE` is the `_rm` suffix mentioned in <<explicit-frm>>. Other intrinsics do not have this suffix.
* `POLICY` are enumerated under <<policy-and-masked-naming-scheme>>.

The general naming scheme is not sufficient to express all intrinsics. The exceptions are enumerated in the proceeding section <<explicit-exception-naming>>.


---

### Scalar move instructions (Explicit)

==== Scalar move instructions

Only encoding the return type will cause naming collisions for the scalar move instruction intrinsics. The intrinsics encode the input vector type and the output scalar type in the suffix.

[,c]
----
int8_t __riscv_vmv_x_s_i8m1_i8 (vint8m1_t vs2);
int8_t __riscv_vmv_x_s_i8m2_i8 (vint8m2_t vs2);
int8_t __riscv_vmv_x_s_i8m4_i8 (vint8m4_t vs2);
int8_t __riscv_vmv_x_s_i8m8_i8 (vint8m8_t vs2);
----

---

### Reduction instructions (Explicit)


Only encoding the return type will cause naming collisions for the reduction instruction intrinsics. The intrinsics encode the input vector type and the output vector type in the suffix.

[,c]
----
vint8m1_t __riscv_vredsum_vs_i8m1_i8m1(vint8m1_t vs2, vint8m1_t vs1, size_t vl);
vint8m1_t __riscv_vredsum_vs_i8m2_i8m1(vint8m2_t vs2, vint8m1_t vs1, size_t vl);
vint8m1_t __riscv_vredsum_vs_i8m4_i8m1(vint8m4_t vs2, vint8m1_t vs1, size_t vl);
vint8m1_t __riscv_vredsum_vs_i8m8_i8m1(vint8m8_t vs2, vint8m1_t vs1, size_t vl);
----

---

### Add-with-carry / Subtract-with-borrow instructions (Explicit)

==== Add-with-carry / Subtract-with-borrow instructions

Only encoding the return type will cause naming collisions for the carry/borrow instruction intrinsics. The intrinsics encode the input vector type and the output mask vector type in the suffix.

[,c]
----
vbool64_t __riscv_vmadc_vvm_i8mf8_b64(vint8mf8_t vs2, vint8mf8_t vs1,
                                      vbool64_t v0, size_t vl);
vbool64_t __riscv_vmadc_vvm_i16mf4_b64(vint16mf4_t vs2, vint16mf4_t vs1,
                                       vbool64_t v0, size_t vl);
vbool64_t __riscv_vmadc_vvm_i32mf2_b64(vint32mf2_t vs2, vint32mf2_t vs1,
                                       vbool64_t v0, size_t vl);
vbool64_t __riscv_vmadc_vvm_i64m1_b64(vint64m1_t vs2, vint64m1_t vs1,
                                      vbool64_t v0, size_t vl);
----

---

### vreinterpret, vlmul_trunc/vlmul_ext, and vset/vget (Explicit)

==== `vreinterpret`, `vlmul_trunc`/`vlmul_ext`, and `vset`/`vget`

Only encoding the return type will cause naming collisions for these pseudo instructions. The intrinsics encode the input vector type before the return type in the suffix.

The following shows an example with `__riscv_vreinterpret_v` of `vint32m1_t` input vector type.

[,c]
----
vfloat32m1_t __riscv_vreinterpret_v_i32m1_f32m1 (vint32m1_t src);
vuint32m1_t __riscv_vreinterpret_v_i32m1_u32m1 (vint32m1_t src);
vint8m1_t __riscv_vreinterpret_v_i32m1_i8m1 (vint32m1_t src);
vint16m1_t __riscv_vreinterpret_v_i32m1_i16m1 (vint32m1_t src);
vint64m1_t __riscv_vreinterpret_v_i32m1_i64m1 (vint32m1_t src);
vbool64_t __riscv_vreinterpret_v_i32m1_b64 (vint32m1_t src);
vbool32_t __riscv_vreinterpret_v_i32m1_b32 (vint32m1_t src);
vbool16_t __riscv_vreinterpret_v_i32m1_b16 (vint32m1_t src);
vbool8_t __riscv_vreinterpret_v_i32m1_b8 (vint32m1_t src);
vbool4_t __riscv_vreinterpret_v_i32m1_b4 (vint32m1_t src);
----

---

### Vector BFloat instructions (Explicit)

=== Vector BFloat instructions

For vector bfloat instructions, appending additional input type before output type to disambiguate between `zvfhmin` instructions, e.g. `vfloat32mf2_t __riscv_vfwadd_vv_f32mf2(vfloat16mf4_t vs2, vfloat16mf4_t vs1, size_t vl);`.

==== Vector Widening BFloat Add instruction
[,c]
----
vfloat32mf2_t __riscv_vfwadd_vv_bf16mf4_f32mf2(vbfloat16mf4_t vs2, vbfloat16mf4_t vs1, size_t vl);
----

==== Vector Widening BFloat Sub instruction
[,c]
----
vfloat32mf2_t __riscv_vfwsub_vv_bf16mf4_f32mf2(vbfloat16mf4_t vs2, vbfloat16mf4_t vs1, size_t vl);
----

==== Vector Widening BFloat Mul instruction
[,c]
----
vfloat32mf2_t __riscv_vfwmul_vv_bf16mf4_f32mf2(vbfloat16mf4_t vs2, vbfloat16mf4_t vs1, size_t vl);
----

==== Vector Widening BFloat Fused Multiply-Add instructions
[,c]
----
vfloat32mf2_t __riscv_vfwmacc_vv_bf16mf4_f32mf2(vfloat32mf2_t vd, vbfloat16mf4_t vs1, vbfloat16mf4_t vs2, size_t vl);
vfloat32mf2_t __riscv_vfwnmacc_vv_bf16mf4_f32mf2(vfloat32mf2_t vd, vbfloat16mf4_t vs1, vbfloat16mf4_t vs2, size_t vl);
vfloat32mf2_t __riscv_vfwmsac_vv_bf16mf4_f32mf2(vfloat32mf2_t vd, vbfloat16mf4_t vs1, vbfloat16mf4_t vs2, size_t vl);
vfloat32mf2_t __riscv_vfwnmsac_vv_bf16mf4_f32mf2(vfloat32mf2_t vd, vbfloat16mf4_t vs1, vbfloat16mf4_t vs2, size_t vl);
----

==== Vector Widening BFloat Type-Convert instruction
[,c]
----
vfloat32mf2_t __riscv_vfwcvt_f_f_v_bf16mf4_f32mf2(vbfloat16mf4_t vs2, size_t vl);
----

==== Vector Narrowing BFloat Type-Convert instruction
[,c]
----
vint8mf8_t __riscv_vfncvt_x_f_w_bf16mf4_i8mf8(vbfloat16mf4_t vs2, size_t vl);
vint8mf8_t __riscv_vfncvt_rtz_x_f_w_bf16mf4_i8mf8(vbfloat16mf4_t vs2, size_t vl);
----

==== Vector BFloat Classify instruction
[,c]
----
vuint16mf4_t __riscv_vfclass_v_bf16mf4_u16mf4(vbfloat16mf4_t vs2, size_t vl);
----

---

### Implicit (Overloaded) naming scheme

[[implicit-naming-scheme]]
=== Implicit (Overloaded) naming scheme

The implicit (overloaded) interface aims to provide a generic interface that takes values of different EEW and EMUL as the input. Therefore, the implicit intrinsics omit the EEW and EMUL encoded in the function name. The `_rm` prefix for explicit FP rounding mode intrinsics (<<control-of-frm>>) is also omitted. The intrinsics under this scheme are the "overloaded intrinsics", which in parallel we have the "non-overloaded intrinsics" defined under <<explicit-naming-scheme>>.

Take the vector addition (`vadd`) instruction intrinsics as an example, stripping off the operand mnemonics and encoded EEW, EMUL information, the intrinsics provides the following overloaded interfaces.

[,c]
----
vint32m1_t __riscv_vadd(vint32m1_t v0, vint32m1_t v1, size_t vl);
vint16m4_t __riscv_vadd(vint16m4_t v0, vint16m4_t v1, size_t vl);
----

Since the main intent is to let the users put different value(s) of EEW and EMUL as input argument(s), the overloaded intrinsics do not omit the policy suffix. That is, the suffix listed under <<control-of-policy>> is not omitted and is still encoded in the function name.

The masked variants with the default policy shares the same interface with the unmasked variants with the default policy. They do not have any trailing suffixes.

Take the vector floating-point add (`vfadd`) as an example, the intrinsics provides the following overloaded interfaces.

[,c]
----
vfloat32m1_t __riscv_vfadd(vfloat32m1_t vs2, vfloat32m1_t vs1, size_t vl);
vfloat32m1_t __riscv_vfadd(vbool32_t vm, vfloat32m1_t vs2, vfloat32m1_t vs1,
                           size_t vl);
vfloat32m1_t __riscv_vfadd(vfloat32m1_t vs2, vfloat32m1_t vs1, unsigned int frm,
                           size_t vl);
vfloat32m1_t __riscv_vfadd(vbool32_t vm, vfloat32m1_t vs2, vfloat32m1_t vs1,
                           unsigned int frm, size_t vl);
vfloat32m1_t __riscv_vfadd_tu(vfloat32m1_t vd, vfloat32m1_t vs2,
                              vfloat32m1_t vs1, size_t vl);
vfloat32m1_t __riscv_vfadd_tum(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                               vfloat32m1_t vs1, size_t vl);
vfloat32m1_t __riscv_vfadd_tumu(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                                vfloat32m1_t vs1, size_t vl);
vfloat32m1_t __riscv_vfadd_mu(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                              vfloat32m1_t vs1, size_t vl);
vfloat32m1_t __riscv_vfadd_tu(vfloat32m1_t vd, vfloat32m1_t vs2,
                              vfloat32m1_t vs1, unsigned int frm, size_t vl);
vfloat32m1_t __riscv_vfadd_tum(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                               vfloat32m1_t vs1, unsigned int frm, size_t vl);
vfloat32m1_t __riscv_vfadd_tumu(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                                vfloat32m1_t vs1, unsigned int frm, size_t vl);
vfloat32m1_t __riscv_vfadd_mu(vbool32_t vm, vfloat32m1_t vd, vfloat32m1_t vs2,
                              vfloat32m1_t vs1, unsigned int frm, size_t vl);
----

The naming scheme to prune everything except the instruction mnemonics is not available for all the intrinsics. Please see <<implicit-exception-naming>> for overloaded intrinsics with irregular naming patterns.

Due to the limitations of the C language (without the aid of features like C++ templates), some intrinsics do not have an overloaded version. Therefore these intrinsics do not possess a simplified, EEW/EMUL-omitted interface. Please see <<unsupported-implicit-naming>> for more detail.


---

### Widening instructions (Implicit)

==== Widening instructions

Widening instruction intrinsics (e.g. `vwadd`) have the same return type but different types of arguments. The operand mnemonics are encoded into their overloaded versions to help distinguish them.

[,c]
----
vint32m1_t __riscv_vwadd_vv(vint16mf2_t vs2, vint16mf2_t vs1, size_t vl);
vint32m1_t __riscv_vwadd_vx(vint16mf2_t vs2, int16_t rs1, size_t vl);
vint32m1_t __riscv_vwadd_wv(vint32m1_t vs2, vint16mf2_t vs1, size_t vl);
vint32m1_t __riscv_vwadd_wx(vint32m1_t vs2, int16_t rs1, size_t vl);
----

---

### Widening BFloat Type-Convert instructions (Implicit)

==== Widening BFloat Type-Convert instructions

Add `_bf16` suffix to disambiguate between `zvfhmin` instructions, e.g. `vfloat16mf4_t __riscv_vfwcvt_f(vint8mf8_t vs2, size_t vl);`.

[,c]
----
vbfloat16mf4_t __riscv_vfwcvt_f_bf16(vint8mf8_t vs2, size_t vl);
----

---

### Narrowing BFloat Type-Convert instructions (Implicit)

==== Narrowing BFloat Type-Convert instructions

Add `_bf16` suffix to disambiguate between `zvfhmin` instructions, e.g. `vfloat16mf4_t __riscv_vfncvt_f(vfloat32mf2_t vs2, size_t vl);`.

[,c]
----
vbfloat16mf4_t __riscv_vfncvt_f_bf16(vfloat32mf2_t vs2, size_t vl);
vbfloat16mf4_t __riscv_vfncvt_rod_f_bf16(vfloat32mf2_t vs2, size_t vl);
----

---

### Type-convert instructions (Implicit)

==== Type-convert instructions

Type-convert instruction intrinsics (e.g. `vfcvt.x.f`, `vfcvt.xu.f`, `vfcvt.rtz.xu.f`) encode the returning type mnemonics into their overloaded variants to help distinguish them.

The following shows how `_x`, `_rtz_x`, `_xu`, and `_rtz_xu` are appended to the suffix for distinction.

[,c]
----
vint32m1_t __riscv_vfcvt_x (vfloat32m1_t src, size_t vl);
vint32m1_t __riscv_vfcvt_rtz_x (vfloat32m1_t src, size_t vl);
vuint32m1_t __riscv_vfcvt_xu (vfloat32m1_t src, size_t vl);
vuint32m1_t __riscv_vfcvt_rtz_xu (vfloat32m1_t src, size_t vl);
----

---

### vreinterpret, LMUL truncate/extension, and vset/vget (Implicit)

==== `vreinterpret`, LMUL truncate/extension, and `vset`/`vget`

These pseudo intrinsics encode the return type (e.g. `__riscv_vreinterpret_b8`) into their overloaded variants to help distinguish them.

The following shows how the return type is appended to the suffix for distinction.

[,c]
----
vfloat32m1_t __riscv_vreinterpret_f32m1 (vint32m1_t src);
vuint32m1_t __riscv_vreinterpret_u32m1 (vint32m1_t src);
vint8m1_t __riscv_vreinterpret_i8m1 (vint32m1_t src);
vint16m1_t __riscv_vreinterpret_i16m1 (vint32m1_t src);
vint64m1_t __riscv_vreinterpret_i64m1 (vint32m1_t src);
vbool64_t __riscv_vreinterpret_b64 (vint32m1_t src);
vbool32_t __riscv_vreinterpret_b32 (vint32m1_t src);
vbool16_t __riscv_vreinterpret_b16 (vint32m1_t src);
vbool8_t __riscv_vreinterpret_b8 (vint32m1_t src);
vbool4_t __riscv_vreinterpret_b4 (vint32m1_t src);
----