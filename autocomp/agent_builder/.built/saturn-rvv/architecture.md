**Hardware Architecture Summary: Saturn RISC-V Vector (RVV) Microarchitecture**

**Overview and Programming Model**
Saturn is a parameterized, short-vector microarchitecture implementing the RISC-V Vector (RVV) extension, designed primarily for domain-specialized, DSP, and embedded cores. It employs a Decoupled Access-Execute (DAE) design where the Vector Load-Store Unit (VLSU) and Vector Datapath (VU) operate independently. Saturn executes vector instructions strictly post-commit relative to the scalar core, meaning vector instructions are non-speculative. It relies on efficient dynamic scheduling of short-chime vector instructions and limited out-of-order execution between independent sequencing paths, rather than costly register renaming or deep out-of-order scalar integration. The programming model uses standard RVV intrinsics, heavily leveraging dynamic vector length (`vl`), vector type (`vtype`), and register grouping (`LMUL`).

**Memory Hierarchy**
*   **Standard Memory Interface:** The VLSU typically bypasses the scalar L1 cache to access a coherent backing memory or a high-bandwidth, software-managed Tightly-Coupled Memory (TCM). It processes unit-stride accesses at full memory bandwidth.
*   **Scatter-Gather TCM (SGTCM):** For high-throughput indexed (scatter/gather) accesses, Saturn can integrate a specialized, non-cacheable, deeply-banked SGTCM with parallel byte-wide ports. Without SGTCM, standard strided and indexed memory operations are bottlenecked to generating only one element address per cycle.
*   **Segmented Accesses:** Saturn features dedicated double-buffered segment buffers (LSB/SSB) that efficiently handle segmented loads/stores (e.g., `vlseg`, `vsseg`). These instructions perform on-the-fly AOS↔SOA repacking and should always be preferred over manual repacking. Throughput depends on NF×ELEN relative to DLEN: when NF×ELEN = DLEN, full bandwidth; when NF×ELEN < DLEN, the segment buffer may underutilize memory bandwidth.
*   **Memory Disambiguation:** Hardware performs precise early-stage scalar-vector and vector-vector memory disambiguation. However, vector memory instructions cannot begin execution if there are pending older scalar stores in the scalar store buffer.

**Compute Units**
*   **Datapath Width (DLEN):** The fundamental unit of compute and register access is the "element group," which is `DLEN` bits wide. The hardware processes 1 element group per cycle, regardless of element width (ELEN).
*   **Chime Length:** The base occupancy of a vector instruction is `VLEN/DLEN` cycles. Using register grouping (`LMUL`) extends this to `LMUL * (VLEN/DLEN)` cycles.
*   **Sequencers:** The backend is divided into independent, single-issue, in-order sequencers: Load (VLS), Store (VSS), Execute/Arithmetic (VXS), and Special (VPS - for index generation, slides, gathers, compress, reductions).
*   **Functional Units:** FUs are pipelined but lack direct FU-to-FU bypasses. Typical latencies: Integer ALU/Shift/Bitmanip (1-2 stages), Integer Multiply (3 stages), FMA (4 stages). Divide and square root use iterative, non-pipelined units.
*   **Issue Topologies:** Depending on the specific Saturn configuration, integer and floating-point operations may share a single sequencer (Unified), use separate sequencers fed by a shared queue (Shared), or use fully independent sequencers and queues (Split/Multi-ALU). **On Shared configs, explicit int/FP instruction interleaving is critical** since both share one issue queue — without interleaving, only one FU type is active. **On Split configs, software load-balancing matters less** because each sequencer has its own queue and the hardware schedules independently.

**Key Constraints and Code Optimization Guidelines**
*   **Maximize LMUL:** Because Saturn is a short-vector machine, low `LMUL` (e.g., 1) results in very short chimes (e.g., 2-4 cycles), which can expose pipeline latencies (e.g., a 4-stage FMA will stall dependent instructions if the chime is only 2 cycles). Always use the largest `LMUL` possible that avoids vector register spilling to increase chime length, hide pipeline latencies, and reduce scalar instruction fetch pressure. **If higher LMUL is not possible** (register pressure), schedule independent FMAs back-to-back as an alternative — two independent 4-stage FMAs at LMUL=1 (chime=2) will keep the pipeline busy without stalling.
*   **Leverage Chaining via Instruction Interleaving:** Saturn supports vector chaining at the `DLEN` (element-group) granularity through the vector register file. Because sequencers are in-order, chaining only occurs between instructions occupying *different* sequencers (e.g., a load chaining into an arithmetic operation). Interleave independent memory and arithmetic intrinsics to maximize concurrent sequencer utilization.
*   **Avoid Vector-to-Scalar Writes in Inner Loops:** Because vector instructions execute post-commit, any vector instruction that writes to a scalar register (e.g., `vfmv.f.s`, or vector reductions yielding a scalar) will cause a Read-After-Write (RAW) hazard that severely stalls the scalar pipeline. Keep reductions and scalar extractions outside of performance-critical inner loops.
*   **Minimize `vsetvl` Bubbles:** Depending on the host scalar core (e.g., Rocket), changing `vtype` or `vl` can introduce pipeline bubbles. Group operations of the same element width and LMUL together to minimize the frequency of `vsetvl` transitions.
*   **Prefer Segmented Loads over Manual Repacking:** Use RVV segmented load/store intrinsics for interleaved data (like complex numbers or RGB pixels) rather than loading raw vectors and manually permuting them, as Saturn's segment buffers handle this at near full memory bandwidth.
*   **Avoid Plain Strided/Indexed Accesses:** Unless the target system explicitly features an SGTCM, avoid plain strided (`vlse`/`vsse`) and indexed (`vluxei`/`vsuxei`) intrinsics, as they generate only 1 element address per cycle. Note: segmented strided/indexed accesses (`vlsseg`/`vluxseg`) are still preferable to manual unpacking even though they are slower than unit-stride segments, because the segment buffer handles repacking efficiently.
*   **Defer Reductions to Tail Code:** Saturn cannot maintain full FU utilization during element-wise reductions due to limited accumulator registers. Express reduction loops as element-wise vector operations (e.g., `vfadd` accumulating into a vector) in the main loop, then perform the final horizontal reduction (`vfredusum`/`vredsum`) once after the loop.
*   **Issue Queue Back-Pressure:** Long sequences of instructions targeting the same sequencer can fill the shallow issue queues, back-pressuring the decoupling queue and stalling the scalar core's frontend. Balance instruction mix across sequencers to avoid this.
*   **Scalar Stores Block Vector Dispatch:** The PFC stalls ALL vector instruction dispatch until the scalar store buffer is empty. Avoid scalar stores (e.g., `*ptr = value`) inside vectorized inner loops — hoist them before or after the vector computation region.
*   **Masking Cost Varies by Access Type:** Masked unit-stride loads are relatively cheap — the mask is applied at VRF writeback, so the load itself proceeds at full bandwidth. Masked unit-stride stores use byte masks in the store merge unit. However, masked strided/indexed memory operations fall to the iterative fault checker (IFC) and execute element-by-element, making them very expensive. Prefer VL-predication over masking for leading-element restriction when possible.
*   **Overlap Scalar Bookkeeping with Vector Execution:** Schedule scalar address calculations, loop counter updates, and pointer increments to overlap with in-flight vector operations. Saturn's decoupled execution means the scalar core can run ahead while vector ops execute. Short-chime kernels benefit significantly from Shuttle (2-3 issue, vsetvl bypass) over Rocket (single-issue, 2-cycle vsetvl bubble).

**Intrinsic → Saturn Pipeline Mapping**

Use this table to reason about instruction scheduling, chaining opportunities, and latency hiding. Chaining only occurs between instructions on DIFFERENT sequencers (VLS, VSS, VXS, VPS). Instructions on the SAME sequencer execute in-order with no overlap.

*Memory Operations (VLS / VSS)*
| Intrinsic Pattern | Assembly | Sequencer | Throughput | Notes |
|---|---|---|---|---|
| `__riscv_vle{n}_v_*` | `vle{n}.v` | VLS | DLEN/cycle | Unit-stride load, best throughput |
| `__riscv_vse{n}_v_*` | `vse{n}.v` | VSS | DLEN/cycle | Unit-stride store, best throughput |
| `__riscv_vlse{n}_v_*` | `vlse{n}.v` | VLS | 1 elem/cycle | Strided load, slow without SGTCM |
| `__riscv_vsse{n}_v_*` | `vsse{n}.v` | VSS | 1 elem/cycle | Strided store, slow without SGTCM |
| `__riscv_vluxei{n}_v_*` | `vluxei{n}.v` | VLS+VPS | 1 elem/cycle | Indexed load, VPS generates indices |
| `__riscv_vsuxei{n}_v_*` | `vsuxei{n}.v` | VSS+VPS | 1 elem/cycle | Indexed store |
| `__riscv_vlseg{nf}e{n}_v_*` | `vlseg{nf}e{n}.v` | VLS | NF×ELEN dependent | Segment load, uses segment buffer |
| `__riscv_vle{n}ff_v_*` | `vle{n}ff.v` | VLS | DLEN/cycle | Fault-only-first load |

*Floating-Point Arithmetic (VXS → FMAPipe, 4-stage pipeline)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_vfadd_{vv,vf}_*` | `vfadd` | 4 | FP add |
| `__riscv_vfsub_{vv,vf}_*` | `vfsub` | 4 | FP subtract |
| `__riscv_vfmul_{vv,vf}_*` | `vfmul` | 4 | FP multiply |
| `__riscv_vfmadd_{vv,vf}_*` | `vfmadd` | 4 | vd = vd×vs1 + vs2 |
| `__riscv_vfmacc_{vv,vf}_*` | `vfmacc` | 4 | vd = vs1×vs2 + vd |
| `__riscv_vfmsub_{vv,vf}_*` | `vfmsub` | 4 | vd = vd×vs1 - vs2 |
| `__riscv_vfmsac_{vv,vf}_*` | `vfmsac` | 4 | vd = vs1×vs2 - vd |
| `__riscv_vfnmadd_{vv,vf}_*` | `vfnmadd` | 4 | Negated fused multiply-add |
| `__riscv_vfnmacc_{vv,vf}_*` | `vfnmacc` | 4 | Negated fused multiply-accumulate |
| `__riscv_vfnmsub_{vv,vf}_*` | `vfnmsub` | 4 | Negated fused multiply-subtract |
| `__riscv_vfnmsac_{vv,vf}_*` | `vfnmsac` | 4 | Negated fused multiply-subtract-accumulate |
| `__riscv_vfw{mul,macc,add,sub}_{vv,vf}_*` | widening variants | 4 | Widening FP ops, same pipeline |

*Floating-Point Convert/Compare (VXS → FPConvPipe, 2-stage pipeline)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_vfcvt_{x_f,f_x,xu_f,f_xu}_v_*` | `vfcvt.*` | 2 | FP↔int conversion |
| `__riscv_vfncvt_*` | `vfncvt.*` | 2 | Narrowing FP convert |
| `__riscv_vfwcvt_*` | `vfwcvt.*` | 2 | Widening FP convert |
| `__riscv_vmf{eq,ne,lt,le,gt,ge}_*` | `vmf*` | 2 | FP compares |
| `__riscv_vf{min,max}_{vv,vf}_*` | `vfmin/vfmax` | 2 | FP min/max |
| `__riscv_vfsgnj{,n,x}_*` | `vfsgnj*` | 2 | FP sign injection |
| `__riscv_vfclass_v_*` | `vfclass.v` | 2 | FP classify |

*FP Divide/Sqrt (VXS → FPDivSqrtUnit, iterative — AVOID in inner loops)*
| Intrinsic Pattern | Assembly | Notes |
|---|---|---|
| `__riscv_vfdiv_{vv,vf}_*` | `vfdiv` | 1 elem/cycle, extremely slow |
| `__riscv_vfsqrt_v_*` | `vfsqrt.v` | 1 elem/cycle, extremely slow |
| `__riscv_vfrsqrt7_v_*` | `vfrsqrt7.v` | Reciprocal sqrt estimate (also iterative) |
| `__riscv_vfrec7_v_*` | `vfrec7.v` | Reciprocal estimate (also iterative) |

*Integer Arithmetic (VXS → IntPipe, 1-stage; saturating ops 2-stage)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_v{add,sub,rsub}_{vv,vx,vi}_*` | `vadd/vsub/vrsub` | 1 | Integer add/sub |
| `__riscv_v{min,max,minu,maxu}_{vv,vx}_*` | `vmin/vmax` | 1 | Integer min/max |
| `__riscv_v{sadd,ssub}_{vv,vx}_*` | `vsadd/vssub` | 2 | Saturating add/sub (stage 2 writeback) |
| `__riscv_vw{add,sub}_{vv,vx,wv,wx}_*` | `vwadd/vwsub` | 1 | Widening add/sub |
| `__riscv_vmerge_{vvm,vxm}_*` | `vmerge` | 1 | Merge under mask |
| `__riscv_vmv_v_{v,x}_*` | `vmv.v.v/vmv.v.x` | 1 | Vector move / scalar splat |
| `__riscv_vfmv_v_f_*` | `vfmv.v.f` | 1 | **FP splat uses IntPipe, NOT FMAPipe** — can run concurrently with FMA |
| `__riscv_vfmerge_vfm_*` | `vfmerge.vfm` | 1 | FP merge uses IntPipe |
| `__riscv_vms{eq,ne,lt,le,gtu}_{vv,vx}_*` | `vmseq/vmsne/...` | 1 | Integer compares |

*Integer Shift (VXS → ShiftPipe, 2-stage pipeline)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_vs{ll,rl,ra}_{vv,vx,vi}_*` | `vsll/vsrl/vsra` | 2 | Shift operations |
| `__riscv_vns{rl,ra}_w{v,x,i}_*` | `vnsrl/vnsra` | 2 | Narrowing shifts |
| `__riscv_vnclip{,u}_w{v,x,i}_*` | `vnclip/vnclipu` | 2 | Narrowing clip (shift + saturate) |

*Integer Bitwise (VXS → BitwisePipe, 1-stage pipeline)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_v{and,or,xor}_{vv,vx,vi}_*` | `vand/vor/vxor` | 1 | Bitwise ops |
| `__riscv_vm{and,nand,or,xor}_mm_*` | `vmand/vmor/...` | 1 | Mask ops |

*Integer Multiply (VXS → MulPipe, 3-stage pipeline)*
| Intrinsic Pattern | Assembly | Depth | Notes |
|---|---|---|---|
| `__riscv_vmul_{vv,vx}_*` | `vmul` | 3 | Integer multiply |
| `__riscv_vmul{h,hu}_{vv,vx}_*` | `vmulh/vmulhu` | 3 | Multiply high |
| `__riscv_v{macc,madd,nmsac,nmsub}_{vv,vx}_*` | `vmacc/vmadd/...` | 3 | Integer MAC |
| `__riscv_vw{mul,macc}_{vv,vx}_*` | `vwmul/vwmacc` | 3 | Widening multiply |

*Integer Divide (VXS → IntDivide, iterative — AVOID in inner loops)*
| Intrinsic Pattern | Assembly | Notes |
|---|---|---|
| `__riscv_v{div,divu,rem,remu}_{vv,vx}_*` | `vdiv/vrem` | 1 elem/cycle |

*Permutation / Special (VPS + VXS)*
| Intrinsic Pattern | Assembly | Throughput | Notes |
|---|---|---|---|
| `__riscv_vslide{up,down}_vx_*` | `vslideup/down` | DLEN/cycle | Fast — uses rotation circuit |
| `__riscv_vslide1{up,down}_vx_*` | `vslide1up/down` | DLEN/cycle | Slide by 1 element |
| `__riscv_vrgather_vv_*` | `vrgather.vv` | 1 elem/cycle | SLOW — element-wise |
| `__riscv_vcompress_vm_*` | `vcompress.vm` | 1 elem/cycle | SLOW — element-wise |

*Reductions (VPS + VXS)*
| Intrinsic Pattern | Assembly | Notes |
|---|---|---|
| `__riscv_v{red,fred}{sum,max,min,and,or,xor}_vs_*` | `vredsum/vfredmax/...` | Uses VPS accumulation buffer |
| `__riscv_vfredusum_vs_*` | `vfredusum.vs` | FP unordered sum (faster) |
| `__riscv_vfredosum_vs_*` | `vfredosum.vs` | FP ordered sum (slower, sequential) |

*Scalar Writebacks (VPS → PrefixUnit) — AVOID IN INNER LOOPS*
| Intrinsic Pattern | Assembly | Notes |
|---|---|---|
| `__riscv_vmv_x_s_*` | `vmv.x.s` | Vector→scalar, causes scalar pipeline stall |
| `__riscv_vfmv_f_s_*` | `vfmv.f.s` | FP vector→scalar, causes scalar pipeline stall |
| `__riscv_vcpop_m_*` | `vcpop.m` | Mask popcount→scalar |
| `__riscv_vfirst_m_*` | `vfirst.m` | Find-first→scalar |

*Zero-Cost (compiler-only, no instruction emitted)*
| Intrinsic Pattern | Notes |
|---|---|
| `__riscv_vreinterpret_v_*` | Type reinterpretation, zero cost |
| `__riscv_vundefined_*` / `__riscv_vcreate_v_*` | No instruction emitted |
| `__riscv_vget_v_*` / `__riscv_vset_v_*` | Sub-register group access, no instruction |

*Vector Configuration (scalar pipeline)*
| Intrinsic Pattern | Assembly | Notes |
|---|---|---|
| `__riscv_vsetvl_e{n}m{m}` | `vsetvli` | 2-cycle bubble on Rocket, 1-cycle on Shuttle |
| `__riscv_vsetvlmax_e{n}m{m}` | `vsetvli` (AVL=~0) | Same bubble cost as vsetvl |

**Scheduling Rules Summary**
1. **Chaining**: VLS → VXS → VSS can all chain. Two VXS instructions on the **same** execute sequencer cannot chain. However, in Split or Multi-ALU configs with separate integer and FP execute sequencers, independent int and FP arithmetic ops CAN chain across execute sequencers.
2. **FMAPipe saturation**: Need LMUL >= ceil(4 / chime_length) = 2 to hide 4-stage FMA latency.
3. **Interleave for chaining**: Load → FP op → Store → Load lets all three sequencers overlap.
4. **vfmv.v.f uses IntPipe**: FP scalar splats go through IntPipe (1 stage), NOT FMAPipe. They can execute concurrently with FMA operations on split/shared configs.
5. **FPConvPipe vs FMAPipe**: Different functional units within VXS. In split config with separate int/fp sequencers, converts and FMAs can overlap.