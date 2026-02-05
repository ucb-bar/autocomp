"""
Saturn ISA generator for Saturn Vector Unit.

Generates documentation about the RISC-V Vector extension and Saturn microarchitecture
to include in LLM prompts for code optimization.

Supports LLM-based automatic selection of relevant ISA sections based on code analysis.
"""

from typing import Iterable, Optional
import json

from autocomp.common import logger, LLMClient
from autocomp.search.prob import Prob
from autocomp.agents.saturn.saturn_config import SaturnConfig

# Map problem types to relevant kernel categories
workload_to_kernel_dict = {
    "gemm": ["matmul", "reduction"],
    "conv": ["matmul", "sliding_window"],
    "softmax": ["reduction", "elementwise"],
    "dotprod": ["reduction", "elementwise"],
    "fft": ["permutation", "elementwise"],
    "transpose": ["permutation"],
}

class SaturnIsaGenerator:
    """Generates Saturn/RVV ISA documentation for LLM prompts.
    
    Supports automatic selection of relevant ISA sections using an LLM to analyze
    the code and determine which documentation sections are most relevant.
    """

    # Available ISA documentation sections
    AVAILABLE_SECTIONS = [
        "architecture",  # Saturn architecture overview
        "vsetvl",        # Vector configuration (vsetvl)
        "memory",        # Memory operations (loads/stores)
        "arithmetic",    # Arithmetic operations (add, mul, fma, etc.)
        "reduction",     # Reduction operations (sum, max, min)
        "permutation",   # Permutation operations (slide, gather, compress)
        "mask",          # Mask operations (comparisons, predication)
        "optimization_guide",  # Saturn-specific optimization tips
    ]

    # Section descriptions for LLM selection
    SECTION_DESCRIPTIONS = {
        "architecture": "Saturn Vector Unit architecture overview: VLEN, DLEN, LMUL, chime length, memory system, execution units, chaining, issue queues",
        "vsetvl": "Vector configuration (vsetvl) intrinsics and stripmining patterns for setting vector length and type",
        "memory": "Vector memory operations: unit-stride, strided, indexed (gather/scatter), and segmented loads/stores",
        "arithmetic": "Vector arithmetic: integer ops, floating-point ops, FMA, widening operations, min/max",
        "reduction": "Vector reductions: sum, max, min reductions with optimized accumulation patterns",
        "permutation": "Vector permutations: slide operations, register gather, compress, broadcast/move",
        "mask": "Vector masking: comparisons, masked operations, mask manipulation, predicated execution",
        "optimization_guide": "Saturn optimization tips: LMUL tuning, chaining, sequencer balancing, memory patterns, FMA saturation",
    }

    def __init__(self, config: SaturnConfig = None, llm_client: Optional[LLMClient] = None):
        """Initialize the ISA generator.
        
        Args:
            config: Saturn hardware configuration. Uses defaults if not provided.
            llm_client: Optional LLM client for automatic section selection.
                       If None, all sections are included by default.
        """
        self.config = config or SaturnConfig()
        self.llm_client = llm_client
        self.isa_dict = self._build_isa_dict()

    def _build_isa_dict(self) -> dict:
        """Build the ISA documentation dictionary."""
        return {
            "architecture": {
                "description": self._get_architecture_description(),
            },
            "vsetvl": self._get_vsetvl_docs(),
            "memory": self._get_memory_docs(),
            "arithmetic": self._get_arithmetic_docs(),
            "reduction": self._get_reduction_docs(),
            "permutation": self._get_permutation_docs(),
            "mask": self._get_mask_docs(),
            "optimization_guide": self._get_optimization_guide(),
        }

    def _get_architecture_description(self) -> str:
        cfg = self.config
        return f"""## Saturn Vector Unit Architecture

Saturn is a RISC-V Vector (RVV) implementation targeting DSP and domain-specialized cores.
It implements a "short-vector" SIMD-style microarchitecture with efficient dynamic scheduling.

### This Instance
- VLEN: {cfg.vlen} bits (vector register length)
- DLEN: {cfg.dlen} bits (datapath width)
- MLEN: {cfg.mlen} bits (memory interface width)
- Chime length: {cfg.chime_length} cycles (VLEN/DLEN)
- Issue queue: {cfg.issue_queue}
- FMA latency: {cfg.fma_latency} cycles (need LMUL ≥ {cfg.min_lmul_for_fma_saturation} to saturate)

### Key Parameters
- LMUL: Length multiplier (1/8, 1/4, 1/2, 1, 2, 4, 8) - groups consecutive vector registers
- SEW: Selected Element Width (8, 16, 32, 64 bits)

### Chime Length
- Fundamental occupancy = VLEN/DLEN cycles per vector instruction
- With LMUL > 1: effective chime = LMUL × VLEN/DLEN cycles
- Higher LMUL reduces dynamic instruction count and improves utilization

### Memory System
- Vector Load-Store Unit (VLSU) with independent load/store paths
- Supports unit-stride, strided, and indexed memory access patterns
- Segmented loads/stores for array-of-structs to struct-of-arrays conversion

### Execution Units
- Integer pipeline: add/sub/shift/bitwise operations ({cfg.int_latency}-cycle latency)
- Multiply pipeline: integer multiply ({cfg.mul_latency}-cycle latency)  
- FMA pipeline: floating-point multiply-add ({cfg.fma_latency}-cycle latency)
- Iterative units: divide, sqrt (element-wise, variable latency)

### Vector Chaining
- Saturn supports chaining at DLEN (element-group) granularity through the VRF
- Dependent instructions can begin as soon as first element group is written back
- Chaining occurs between instructions in different sequencers (load/store/execute)
- Critical for achieving high utilization with short vector lengths

### Issue Queue Configurations
- Unified: Single sequencer for all arithmetic (lowest area, no parallel int/fp)
- Shared: Separate int/fp sequencers with shared issue queue (requires interleaving)
- Split: Separate int/fp sequencers with independent queues (most flexible)
"""

    def _get_vsetvl_docs(self) -> str:
        return """## Vector Configuration (vsetvl)

Vector configuration instructions set the vector length (vl) and type (vtype).

### Intrinsics
```c
// Set vector length for element width and LMUL
size_t vl = __riscv_vsetvl_e8m1(avl);   // 8-bit elements, LMUL=1
size_t vl = __riscv_vsetvl_e16m2(avl);  // 16-bit elements, LMUL=2
size_t vl = __riscv_vsetvl_e32m4(avl);  // 32-bit elements, LMUL=4
size_t vl = __riscv_vsetvl_e64m8(avl);  // 64-bit elements, LMUL=8

// Set vector length with explicit max
size_t vl = __riscv_vsetvlmax_e32m1();  // Get max vl for e32m1
```

### Usage Pattern (Stripmining)
```c
size_t avl = n;  // Application vector length
for (size_t vl; avl > 0; avl -= vl) {
    vl = __riscv_vsetvl_e32m4(avl);
    // Vector operations with vl elements
    ptr += vl;
}
```

### Performance Notes
- vsetvl in inner loops can cause pipeline bubbles (especially with Rocket core)
- Shuttle core has vsetvl bypass network to reduce bubbles
- Use highest LMUL that avoids register spilling to reduce vsetvl frequency
- Mixed-precision code requires frequent vsetvl; consider Shuttle integration
"""

    def _get_memory_docs(self) -> str:
        return """## Vector Memory Operations

### Unit-Stride Loads/Stores (Highest Throughput)
```c
// Load vl elements contiguously
vfloat32m4_t vec = __riscv_vle32_v_f32m4(ptr, vl);
vint64m8_t vec = __riscv_vle64_v_i64m8(ptr, vl);

// Store vl elements contiguously  
__riscv_vse32_v_f32m4(ptr, vec, vl);
__riscv_vse64_v_i64m8(ptr, vec, vl);
```

### Strided Loads/Stores (One Address Per Cycle)
```c
// Load with constant stride (in bytes)
vfloat32m4_t vec = __riscv_vlse32_v_f32m4(ptr, stride, vl);

// Store with constant stride
__riscv_vsse32_v_f32m4(ptr, stride, vec, vl);
```

### Indexed (Gather/Scatter) Loads/Stores (One Address Per Cycle)
```c
// Gather: load from ptr + indices[i]
vuint32m4_t indices = ...;
vfloat32m4_t vec = __riscv_vluxei32_v_f32m4(ptr, indices, vl);

// Scatter: store to ptr + indices[i]
__riscv_vsuxei32_v_f32m4(ptr, indices, vec, vl);
```

### Segmented Loads/Stores (Array-of-Structs)
```c
// Load NF fields interleaved in memory into NF vector registers
// e.g., RGB pixels: [R0,G0,B0,R1,G1,B1,...] -> vr, vg, vb
vfloat32m2x3_t rgb = __riscv_vlseg3e32_v_f32m2x3(ptr, vl);
vfloat32m2_t vr = __riscv_vget_v_f32m2x3_f32m2(rgb, 0);
vfloat32m2_t vg = __riscv_vget_v_f32m2x3_f32m2(rgb, 1);
vfloat32m2_t vb = __riscv_vget_v_f32m2x3_f32m2(rgb, 2);
```

### Performance Notes
- Unit-stride saturates memory bandwidth; prefer over strided/indexed
- Strided and indexed: 1 element address per cycle (memory-bound)
- Segmented loads never worse than equivalent manual unpacking
- Masked unit-stride loads ignore mask (apply at VRF write); stores use mask
"""

    def _get_arithmetic_docs(self) -> str:
        return """## Vector Arithmetic Operations

### Integer Operations
```c
// Add/subtract
vint32m4_t c = __riscv_vadd_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vsub_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vadd_vx_i32m4(a, scalar, vl);  // vector + scalar

// Multiply
vint32m4_t c = __riscv_vmul_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vmacc_vv_i32m4(acc, a, b, vl);  // acc += a * b
vint32m4_t c = __riscv_vmadd_vv_i32m4(a, b, c, vl);   // a = a * b + c

// Shift
vint32m4_t c = __riscv_vsll_vv_i32m4(a, shift, vl);   // left shift
vint32m4_t c = __riscv_vsra_vv_i32m4(a, shift, vl);   // arithmetic right shift
vint32m4_t c = __riscv_vsrl_vv_i32m4(a, shift, vl);   // logical right shift

// Min/Max
vint32m4_t c = __riscv_vmin_vv_i32m4(a, b, vl);
vint32m4_t c = __riscv_vmax_vv_i32m4(a, b, vl);
```

### Floating-Point Operations
```c
// Add/subtract/multiply/divide
vfloat32m4_t c = __riscv_vfadd_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfsub_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfmul_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfdiv_vv_f32m4(a, b, vl);  // iterative, slow

// Fused multiply-add (4-cycle pipeline, critical for GEMM)
vfloat32m4_t c = __riscv_vfmacc_vv_f32m4(acc, a, b, vl);  // acc += a * b
vfloat32m4_t c = __riscv_vfmadd_vv_f32m4(a, b, c, vl);   // a = a * b + c
vfloat32m4_t c = __riscv_vfnmacc_vv_f32m4(acc, a, b, vl); // acc -= a * b

// Scalar operand versions (broadcast scalar to all lanes)
vfloat32m4_t c = __riscv_vfmul_vf_f32m4(a, scalar, vl);
vfloat32m4_t c = __riscv_vfmacc_vf_f32m4(acc, scalar, b, vl);

// Min/Max
vfloat32m4_t c = __riscv_vfmin_vv_f32m4(a, b, vl);
vfloat32m4_t c = __riscv_vfmax_vv_f32m4(a, b, vl);

// Square root (iterative)
vfloat32m4_t c = __riscv_vfsqrt_v_f32m4(a, vl);
```

### Widening Operations (Double Output Width)
```c
// Widening multiply: i16 × i16 → i32
vint32m4_t c = __riscv_vwmul_vv_i32m4(a_i16, b_i16, vl);

// Widening multiply-add
vint32m4_t c = __riscv_vwmacc_vv_i32m4(acc, a_i16, b_i16, vl);
```

### Performance Notes
- FMA has 4-cycle latency; need LMUL ≥ 4 or interleaved independent FMAs to saturate
- Integer multiply has 3-cycle latency
- Division and sqrt are iterative (one element at a time) - avoid in inner loops
- Use vfmacc_vf with scalar broadcast for matrix-vector products
"""

    def _get_reduction_docs(self) -> str:
        return """## Vector Reduction Operations

Reductions combine all vector elements into a single scalar result.

### Sum Reduction
```c
// Initialize scalar accumulator
vfloat32m1_t scalar_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);

// Reduce vector to scalar (acc = acc + sum(vec))
scalar_acc = __riscv_vfredusum_vs_f32m4_f32m1(vec, scalar_acc, vl);

// Extract scalar result
float result = __riscv_vfmv_f_s_f32m1_f32(scalar_acc);
```

### Other Reductions
```c
// Max reduction
vfloat32m1_t max_val = __riscv_vfredmax_vs_f32m4_f32m1(vec, init, vl);

// Min reduction  
vfloat32m1_t min_val = __riscv_vfredmin_vs_f32m4_f32m1(vec, init, vl);

// Integer sum
vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(vec, init, vl);

// Integer max/min
vint32m1_t max_val = __riscv_vredmax_vs_i32m4_i32m1(vec, init, vl);
vint32m1_t min_val = __riscv_vredmin_vs_i32m4_i32m1(vec, init, vl);
```

### Performance Notes
- Reductions cannot maintain full utilization due to limited accumulator registers
- Better pattern: element-wise ops across vector, then final reduction in tail code
- Example: For dot product, use vfmacc to accumulate in vector, then vredsum once

### Optimized Reduction Pattern
```c
// Instead of reducing each iteration:
// BAD: for each chunk: acc = vredsum(vmul(a, b), acc)

// Better: accumulate in vector, reduce once at end:
vfloat32m4_t vec_acc;
for (first chunk) {
    vec_acc = __riscv_vfmul_vv_f32m4(a, b, vl);
}
for (remaining chunks) {
    vec_acc = __riscv_vfmacc_vv_f32m4(vec_acc, a, b, vl);
}
// Single reduction at end
vfloat32m1_t result = __riscv_vfredusum_vs_f32m4_f32m1(vec_acc, zero, vl);
```
"""

    def _get_permutation_docs(self) -> str:
        return """## Vector Permutation Operations

### Slide Operations
```c
// Slide down: shift elements toward lower indices
vfloat32m4_t c = __riscv_vslidedown_vx_f32m4(src, offset, vl);

// Slide up: shift elements toward higher indices
vfloat32m4_t c = __riscv_vslideup_vx_f32m4(dst, src, offset, vl);

// Slide by immediate (more efficient for constant offsets)
vfloat32m4_t c = __riscv_vslidedown_vi_f32m4(src, 1, vl);  // slide by 1
```

### Gather (Register-Register)
```c
// Gather elements from src using indices in idx
vuint32m4_t idx = ...;
vfloat32m4_t c = __riscv_vrgather_vv_f32m4(src, idx, vl);
```

### Compress
```c
// Compress: pack elements where mask is true
vbool8_t mask = ...;
vfloat32m4_t c = __riscv_vcompress_vm_f32m4(src, mask, vl);
```

### Move Operations
```c
// Broadcast scalar to all elements
vfloat32m4_t c = __riscv_vfmv_v_f_f32m4(scalar, vl);

// Move between vector and scalar
float s = __riscv_vfmv_f_s_f32m4_f32(vec);  // extract element 0
vfloat32m4_t c = __riscv_vfmv_s_f_f32m4(scalar, vl);  // set element 0
```

### Performance Notes
- Slides proceed at DLEN bits/cycle (efficient for convolution stencils)
- Register gathers are element-wise (slower than slides)
- Compress is element-wise
- Use slides instead of indexed loads for regular access patterns
"""

    def _get_mask_docs(self) -> str:
        return """## Vector Mask Operations

Masks enable conditional execution on individual vector elements.

### Creating Masks
```c
// Compare operations produce masks
vbool8_t mask = __riscv_vmflt_vv_f32m4_b8(a, b, vl);  // a < b
vbool8_t mask = __riscv_vmfle_vv_f32m4_b8(a, b, vl);  // a <= b
vbool8_t mask = __riscv_vmfeq_vv_f32m4_b8(a, b, vl);  // a == b
vbool8_t mask = __riscv_vmfgt_vf_f32m4_b8(a, 0.0f, vl);  // a > 0

// Integer comparisons
vbool8_t mask = __riscv_vmslt_vv_i32m4_b8(a, b, vl);  // signed a < b
vbool8_t mask = __riscv_vmsltu_vv_u32m4_b8(a, b, vl); // unsigned a < b
```

### Masked Operations
```c
// Masked load (inactive elements undefined)
vfloat32m4_t c = __riscv_vle32_v_f32m4_m(mask, ptr, vl);

// Masked store (only store where mask is true)
__riscv_vse32_v_f32m4_m(mask, ptr, vec, vl);

// Masked arithmetic
vfloat32m4_t c = __riscv_vfadd_vv_f32m4_m(mask, a, b, vl);

// Merge: select from two sources based on mask
vfloat32m4_t c = __riscv_vmerge_vvm_f32m4(false_val, true_val, mask, vl);
```

### Mask Manipulation
```c
// Mask AND/OR/XOR
vbool8_t c = __riscv_vmand_mm_b8(mask1, mask2, vl);
vbool8_t c = __riscv_vmor_mm_b8(mask1, mask2, vl);
vbool8_t c = __riscv_vmnot_m_b8(mask, vl);

// Count set bits in mask
unsigned count = __riscv_vcpop_m_b8(mask, vl);

// Find first set bit
long first = __riscv_vfirst_m_b8(mask, vl);  // -1 if none set
```

### Performance Notes
- Masks stored in v0 register (shadow copy avoids extra VRF read port)
- Masked unit-stride loads ignore mask, apply at writeback
- Masked stores generate byte mask for memory request
"""

    def _get_optimization_guide(self) -> str:
        cfg = self.config
        return f"""## Saturn Optimization Guide

### 1. Maximize LMUL
- Use highest LMUL that avoids register spilling
- Higher LMUL = longer chimes = better amortization of overheads
- LMUL=8 with e32 uses all 32 registers as one group

### 2. Enable Chaining
- Interleave loads and arithmetic to enable chaining between sequencers
- Example: load A, load B, compute on A, load C, compute on B, ...
- Chaining requires instructions in different sequencers (load vs execute)

### 3. Balance Across Sequencers
- This instance uses {cfg.issue_queue} configuration
- Split configuration: int and fp have separate queues (most flexible)
- Shared configuration: interleave int and fp operations
- Unified configuration: no parallel int/fp execution

### 4. Minimize vsetvl Overhead
- Keep vsetvl outside inner loops when possible
- Use consistent element width to avoid reconfiguration

### 5. Memory Access Patterns
- Unit-stride loads/stores saturate bandwidth
- Strided/indexed: 1 address per cycle (use only when necessary)
- Use segmented loads for AoS data instead of manual unpacking

### 6. Reduction Strategy
- Accumulate in vector registers across loop iterations
- Single reduction at the end (not per iteration)
- Use vfmacc for dot products, vredsum only at end

### 7. Avoid Pipeline Stalls
- FMA latency = {cfg.fma_latency} cycles; need LMUL ≥ {cfg.min_lmul_for_fma_saturation} or independent FMAs to saturate
- Avoid division/sqrt in inner loops (iterative, element-wise)
- Schedule scalar bookkeeping to overlap with vector execution

### 8. Register Blocking for GEMM
- Block to fit working set in vector registers
- Typical pattern: load A tile, stream B columns, accumulate in C
- Use multiple accumulators to hide FMA latency

---

## Example: Optimized Dot Product
Shows: LMUL=8, vector accumulation, single reduction at end
```c
int64_t dotp_v64b(int64_t *a, int64_t *b, uint64_t avl) {{
    size_t orig_avl = avl;
    size_t vl = __riscv_vsetvl_e64m8(avl);

    vint64m8_t acc, buf_a, buf_b;
    vint64m1_t red = __riscv_vmv_s_x_i64m1(0, vl);

    for (; avl > 0; avl -= vl) {{
        vl = __riscv_vsetvl_e64m8(avl);
        buf_a = __riscv_vle64_v_i64m8(a, vl);
        buf_b = __riscv_vle64_v_i64m8(b, vl);
        if (avl == orig_avl) {{
            acc = __riscv_vmul_vv_i64m8(buf_a, buf_b, vl);
        }} else {{
            acc = __riscv_vmacc_vv_i64m8(acc, buf_a, buf_b, vl);
        }}
        a += vl;
        b += vl;
    }}

    red = __riscv_vredsum_vs_i64m8_i64m1(acc, red, vl);
    return __riscv_vmv_x_s_i64m1_i64(red);
}}
```

## Example: Register-Blocked GEMM
Shows: register blocking, multiple accumulators, scalar broadcast (vfmacc_vf)
```c
// C[M,N] = A[M,K] * B[K,N]
void sgemm_blocked(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc) {{
    const size_t TILE_M = 4;  // Rows of C tile

    for (size_t i = 0; i < M; i += TILE_M) {{
        size_t m_tile = (i + TILE_M <= M) ? TILE_M : (M - i);

        for (size_t j = 0; j < N; ) {{
            size_t vl = __riscv_vsetvl_e32m4(N - j);

            // Initialize C accumulators
            vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (size_t k = 0; k < K; ++k) {{
                vfloat32m4_t b_row = __riscv_vle32_v_f32m4(&B[k * ldb + j], vl);

                // vfmacc_vf broadcasts scalar A element
                if (m_tile > 0) c0 = __riscv_vfmacc_vf_f32m4(c0, A[(i+0)*lda+k], b_row, vl);
                if (m_tile > 1) c1 = __riscv_vfmacc_vf_f32m4(c1, A[(i+1)*lda+k], b_row, vl);
                if (m_tile > 2) c2 = __riscv_vfmacc_vf_f32m4(c2, A[(i+2)*lda+k], b_row, vl);
                if (m_tile > 3) c3 = __riscv_vfmacc_vf_f32m4(c3, A[(i+3)*lda+k], b_row, vl);
            }}

            // Store C tile
            if (m_tile > 0) __riscv_vse32_v_f32m4(&C[(i+0)*ldc+j], c0, vl);
            if (m_tile > 1) __riscv_vse32_v_f32m4(&C[(i+1)*ldc+j], c1, vl);
            if (m_tile > 2) __riscv_vse32_v_f32m4(&C[(i+2)*ldc+j], c2, vl);
            if (m_tile > 3) __riscv_vse32_v_f32m4(&C[(i+3)*ldc+j], c3, vl);

            j += vl;
        }}
    }}
}}
```

## Example: Convolution with vslidedown
Shows: vslidedown for stencil access, scalar broadcast for filter weights
```c
// 3x3 convolution - key pattern using vslidedown
// Load input row with padding, then slide for each filter column
vfloat64m2_t row = __riscv_vle64_v_f64m2(in_ptr, vl_padded);

// Column 0: no slide
vfloat64m2_t out = __riscv_vfmul_vf_f64m2(row, filter[0], vl);

// Column 1: slide by 1
vfloat64m2_t row_s1 = __riscv_vslidedown_vx_f64m2(row, 1, vl);
out = __riscv_vfmacc_vf_f64m2(out, filter[1], row_s1, vl);

// Column 2: slide by 2
vfloat64m2_t row_s2 = __riscv_vslidedown_vx_f64m2(row, 2, vl);
out = __riscv_vfmacc_vf_f64m2(out, filter[2], row_s2, vl);
```
"""

    def generate_isa(self, prob: Prob = None, code: str = None, use_llm_selection: bool = False) -> str:
        """Generate ISA documentation string for the given problem.
        
        Args:
            prob: Problem specification (optional)
            code: Source code to analyze for relevant sections (optional)
            use_llm_selection: If True and code is provided, use LLM to select
                              relevant sections. Requires llm_client to be set.
        
        Returns:
            Concatenated ISA documentation string with relevant sections.
        """
        # Determine which sections to include
        if use_llm_selection and code and self.llm_client:
            sections_to_include = self.select_relevant_sections(code)
            logger.info(f"LLM selected sections: {sections_to_include}")
        else:
            # Default: include all sections
            sections_to_include = self.AVAILABLE_SECTIONS
        
        sections = []
        
        # Always include architecture overview first
        if "architecture" in sections_to_include:
            sections.append(self.isa_dict["architecture"]["description"])
        
        # Always include vsetvl (fundamental to RVV)
        if "vsetvl" in sections_to_include:
            sections.append(self.isa_dict["vsetvl"])
        
        # Include other sections based on selection
        if "memory" in sections_to_include:
            sections.append(self.isa_dict["memory"])
        
        if "arithmetic" in sections_to_include:
            sections.append(self.isa_dict["arithmetic"])
        
        if "reduction" in sections_to_include:
            sections.append(self.isa_dict["reduction"])
        
        if "permutation" in sections_to_include:
            sections.append(self.isa_dict["permutation"])
        
        if "mask" in sections_to_include:
            sections.append(self.isa_dict["mask"])
        
        # Always include optimization guide last
        if "optimization_guide" in sections_to_include:
            sections.append(self.isa_dict["optimization_guide"])
        
        return "\n\n".join(sections)

    def select_relevant_sections(self, code: str) -> list[str]:
        """Use LLM to analyze code and select relevant ISA documentation sections.
        
        Args:
            code: Source code to analyze
            
        Returns:
            List of section names that are relevant to the code.
        """
        if not self.llm_client:
            logger.warning("No LLM client configured, returning all sections")
            return self.AVAILABLE_SECTIONS
        
        prompt = self._build_section_selection_prompt(code)
        
        try:
            response = self.llm_client.chat(
                prompt=prompt,
                num_candidates=1,
                temperature=0.0  # Deterministic for consistency
            )[0]
            
            selected = self._parse_section_selection_response(response)
            
            # Ensure we always have at least architecture and optimization_guide
            if "architecture" not in selected:
                selected.insert(0, "architecture")
            if "optimization_guide" not in selected:
                selected.append("optimization_guide")
            
            return selected
            
        except Exception as e:
            logger.warning(f"LLM section selection failed: {e}, returning all sections")
            return self.AVAILABLE_SECTIONS

    def select_relevant_sections_batch(self, codes: list[str]) -> list[list[str]]:
        """Batch version of select_relevant_sections for multiple code samples.
        
        Args:
            codes: List of source code strings to analyze
            
        Returns:
            List of section name lists, one per code sample.
        """
        if not self.llm_client:
            logger.warning("No LLM client configured, returning all sections for all codes")
            return [self.AVAILABLE_SECTIONS for _ in codes]
        
        prompts = [self._build_section_selection_prompt(code) for code in codes]
        
        try:
            responses = self.llm_client.chat_async(
                prompts_lst=prompts,
                num_candidates=1,
                temperature=0.0
            )
            
            results = []
            for response_list in responses:
                response = response_list[0] if response_list else ""
                selected = self._parse_section_selection_response(response)
                
                # Ensure minimum sections
                if "architecture" not in selected:
                    selected.insert(0, "architecture")
                if "optimization_guide" not in selected:
                    selected.append("optimization_guide")
                
                results.append(selected)
            
            return results
            
        except Exception as e:
            logger.warning(f"LLM batch section selection failed: {e}, returning all sections")
            return [self.AVAILABLE_SECTIONS for _ in codes]

    def _build_section_selection_prompt(self, code: str) -> str:
        """Build prompt for LLM to select relevant ISA sections."""
        section_list = "\n".join(
            f"- {name}: {desc}" 
            for name, desc in self.SECTION_DESCRIPTIONS.items()
        )
        
        return f"""Analyze the following RISC-V Vector (RVV) code and determine which ISA documentation sections would be most helpful for optimizing it.

CODE:
```c
{code}
```

AVAILABLE SECTIONS:
{section_list}

Based on the operations used in the code and potential optimization opportunities, select the most relevant sections.

Rules:
1. Select sections that cover operations actually used in the code
2. Select sections that cover operations that could be used to optimize the code
3. Always include "architecture" and "optimization_guide" 
4. Be safe - include sections that aren't sure about but could be relevant
5. Consider what transformations might improve performance

Respond with a JSON object containing a single key "sections" with a list of section names.
Example: {{"sections": ["architecture", "memory", "arithmetic", "optimization_guide"]}}

Your response (JSON only):"""

    def _parse_section_selection_response(self, response: str) -> list[str]:
        """Parse LLM response to extract selected sections."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(response)
            sections = data.get("sections", [])
            
            # Validate section names
            valid_sections = [s for s in sections if s in self.AVAILABLE_SECTIONS]
            
            if not valid_sections:
                logger.warning(f"No valid sections in LLM response: {sections}")
                return self.AVAILABLE_SECTIONS
            
            return valid_sections
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract section names from plain text
            return self._extract_sections_from_text(response)

    def _extract_sections_from_text(self, text: str) -> list[str]:
        """Fallback: extract section names from plain text response."""
        found = []
        text_lower = text.lower()
        
        for section in self.AVAILABLE_SECTIONS:
            if section.lower() in text_lower:
                found.append(section)
        
        return found if found else self.AVAILABLE_SECTIONS

    def get_workload_kernels(self, workload: str) -> list[str]:
        """Get relevant kernel categories for a workload."""
        return workload_to_kernel_dict.get(workload, [])
