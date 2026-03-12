from typing import Iterable

from autocomp.common import logger
from autocomp.search.prob import Prob

workload_to_kernel_dict = {
    "attention": ["gemm", "softmax"], # for now, attention is a combination of gemm and softmax
    "attention_decoder": ["gemm", "softmax", "causal_mask"],
    "llama_mlp": ["gemm", "softmax"],
    "llama_attention": ["gemm", "softmax"],
    "llama_logits": ["transpose", "gemm"],
}

prob_to_name = {
    "trn-tutorial": {
        0: "rmsnorm",
        1: "layernorm",
        2: "gemm",
        3: "mamba",
        4: "attention",
        5: "attention",
    },
    "trn-advanced": {
        0: "cumsum",
        1: "transpose",
        2: "maxpool",
        3: "rope",
        4: "conv1d",
        5: "conv2d",
        6: "attention_decoder",
        7: "attention_decoder",
        8: "attention_decoder",
    },
    "trn-e2e": {
        0: "llama_mlp",
        1: "llama_mlp",
        2: "llama_mlp",
        3: "llama_mlp",
        4: "llama_attention",
        5: "llama_attention",
        6: "llama_logits",
        7: "llama_logits",
        8: "llama_attention",
        9: "llama_attention",
        10: "llama_mlp",
        11: "llama_mlp",
        12: "llama_logits",
        13: "llama_logits",
    },
}

nki_isa_dict = {
    "architecture": {
        "description": """Kernel structure: Each NKI kernel has 3 stages — load data from HBM → SBUF, compute on NeuronCore, store results from SBUF → HBM.
NKI Beta 2: Use namespace nki.* and nki.isa (not neuronxcc.nki.*). Mark top-level kernels with @nki.jit; remove @nki.jit from sub-kernels called from other kernels. All nki.isa APIs require explicit dst as first argument; nl.load/nl.store replaced by nisa.dma_copy; nl.arange and mask deprecated. See NKI Migration Guide (Beta 1 to Beta 2).

NeuronCore: 2 per Trainium. Each has SBUF + PSUM SRAMs and 4 engines (Tensor, Vector, Scalar, GpSimd).

Tensor model:
    Tensor = NKI array in HBM/SBUF/PSUM.
    First dim = partition dimension (maps to 128 SBUF partitions).
    Can mark explicitly via nl.par_dim().
    A Tensor whose first dim is the partition dim = Tile → required input/output for all NKI compute APIs.
    If not Tile, index to extract per-partition Tile (e.g., x[i]).

Axes concepts:
    Partition (P) + Free (F) = physical layout in 2D SRAM.
    Parallel + Contraction = logical compute axes.

Layout constraints:
    [LC#1] Matmul: contraction axis (K) → P dim → use [K, M] × [K, N] in nki.isa.nc_matmul.
    [LC#2] Non-matmul ops: parallel axis → P dim.

Tile-size constraints:
    [TC#1] partition dimension P ≤ 128 (SBUF/PSUM).
    [TC#2] PSUM combined free dimensions ≤ 512.
    [TC#3] Matmul: LHS free dimension ≤ 128, RHS free dimension ≤ 512.
    limit of 192KB per partition on SBUF buffer.

Indexing:
    Standard Python-style indexing/slicing returns views.
    Indexing fewer dims (e.g. x[1]) can produce a valid Tile.
    For contiguous access use regular integer slicing (e.g. t[0:128, 0:512]); nl.arange for indexing is removed in NKI Beta 2.
    List indices must be integers or slices. They cannot be computed from affine_range loop variables.
    Slice with variable size is not supported.
    Tile on SBUF/PSUM must have at least 2 dimensions as described here. If using a 1D tile on SBUF/PSUM, users may get an “Insufficient rank” error. Workaround this by creating a 2D tile, e.g.,
    buf = nl.zeros((128, ), dtype=dtype, buffer=nl.sbuf)  # this won't work
    buf = nl.zeros((128, 1), dtype=dtype, buffer=nl.sbuf) # this works
    Users must index their [N, 1] or [1, M] shaped 2D buffers with both indices, do my_sbuf[0:N, 0] or my_sbuf[0, 0:M] to access them, since accessing in 1D my_sbuf[0:N] won’t work.
    Use HBM to/from SBUF: nisa.dma_copy; dynamic access: tensor.ap(). Contiguous access: use integer slicing (e.g. t[0:128, 0:512]). See NKI migration guide.
    If indexing with [0, 0] gets internal errors, try using [0:1, 0:1] slicing instead.
    If indexing with [0:1, ...] gets internal errors, try using [0, ...] instead.
    nki.language.ds(start, size) constructs a dynamic slice for simple tensor indexing.
    Example (Beta 2): load tiles from HBM with nisa.dma_copy:
        def example_kernel(in_tensor):
            out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)  # output on HBM
            for i in nl.affine_range(in_tensor.shape[1] // 512):  # loop over free-dimension tiles (512 elements each)
                tile = nl.ndarray((in_tensor.shape[0], 512), dtype=in_tensor.dtype, buffer=nl.sbuf)  # SBUF tile for this chunk
                nisa.dma_copy(dst=tile, src=in_tensor[:, (i * 512):((i + 1) * 512)])  # load HBM -> SBUF
    Use basic indexing or slicing consistently; mixing with advanced indexing is not supported.
        c = nl.exp(a[:, :])  # ok
        c = nl.exp(a[0:4, 0:4])  # also ok (slicing)

NKI Beta 2: Masking deprecated. The mask parameter is no longer supported. Use Python min() and slice() to build in-bounds indexing (e.g. p_end = min(sz_p, p_start + 128); use t[p_start:p_end, f_start:f_end]).

    Matmul: use slicing for tile bounds (e.g. lhs[0:min(M,64), :], rhs[:, 0:min(N,256)]).

    For non-divisible tile sizes use min() and slice: p_end = min(sz_p, p_start + 128); f_end = min(sz_f, f_start + 512); then index with [p_start:p_end, f_start:f_end].

    Example (Beta 2: nisa.dma_copy for load/store, no mask):
    def tensor_exp_kernel_(in_tensor, out_tensor):
        sz_p, sz_f = in_tensor.shape
        trip_count = math.ceil(sz_p / nl.tile_size.pmax)
        for p in nl.affine_range(trip_count):
            p_start = p * nl.tile_size.pmax
            p_end = min(sz_p, p_start + nl.tile_size.pmax)  # handle partial last tile
            in_tile = nl.ndarray((p_end - p_start, sz_f), dtype=in_tensor.dtype, buffer=nl.sbuf)  # SBUF tile for this chunk
            nisa.dma_copy(dst=in_tile, src=in_tensor[p_start:p_end, 0:sz_f])  # load HBM -> SBUF
            out_tile = nl.exp(in_tile)  # compute exp (language API; for nisa.activation use explicit dst)
            nisa.dma_copy(dst=out_tensor[p_start:p_end, 0:sz_f], src=out_tile)  # store SBUF -> HBM

Performance tip:
    Access HBM sequentially; only F-dim striding is hardware-efficient.

Scope rules:
    Tensors in NKI are not allowed to be used outside of their parent scope.
    Tensors in NKI have a stricter scope rules than Python. In NKI, control blocks in if/else/for statements will introduce their own scope for tensors. A tensor defined in if/else/for control blocks are not allowed to be used outside of the scope.

    for i in range(4):
        if i < 2:
            nisa.dma_copy(dst=sbuf_tile, src=a)  # dst must be pre-allocated
        else:
            nisa.dma_copy(dst=sbuf_tile, src=b)
        nisa.dma_copy(dst=c, src=sbuf_tile)  # Error if sbuf_tile is defined inside if/else (scope)

    To fix: define tmp once and use it in both branches:
    for i in range(4):
        tmp = nl.ndarray(shape=a.shape, dtype=a.dtype, buffer=nl.sbuf)
        if i < 2:
            nisa.dma_copy(dst=tmp, src=a)
        else:
            nisa.dma_copy(dst=tmp, src=b)
        nisa.dma_copy(dst=c, src=tmp)

Other constraints:
    & is not supported for scalar.
""",
    },
    "ElementWiseMath": [
        {"header": "nl.add(x: tile | scalar, y: tile | scalar, dtype=None)", 
         "description": """Element-wise addition. 
x.shape and y.shape must be broadcastable to a common shape, that will become the shape of the output. 
dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision.""",
         "examples": """# Beta 2: use nisa.dma_copy for HBM <-> SBUF (nl.load/nl.store removed)
a = nl.ndarray((128, 512), dtype=a_tensor.dtype, buffer=nl.sbuf)
b = nl.ndarray((128, 512), dtype=b_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:512])
nisa.dma_copy(dst=b, src=b_tensor[0:128, 0:512])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)

# add constant b to each element in a
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:512])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)

# broadcast on free dimension -- [128, 1] broadcasted to [128, 512]
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:512])
nisa.dma_copy(dst=b, src=b_tensor[0:128, 0:1])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)

# broadcast on partition dimension -- [1, 512] broadcasted to [128, 512]
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:512])
nisa.dma_copy(dst=b, src=b_tensor[0:1, 0:512])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)

# broadcast on both dimensions
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:512])
nisa.dma_copy(dst=b, src=b_tensor[0:1, 0:1])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)

# [128, 1] and [1, 512] broadcasted to [128, 512]
nisa.dma_copy(dst=a, src=a_tensor[0:128, 0:1])
nisa.dma_copy(dst=b, src=b_tensor[0:1, 0:512])
c = nl.add(a, b)
nisa.dma_copy(dst=c_tensor[0:128, 0:512], src=c)"""},
        {"header": "nl.subtract(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Element-wise subtraction."},
        {"header": "nl.multiply(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Element-wise multiplication."},
        {"header": "nl.divide(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Element-wise division."},
        {"header": "nl.power(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Elements of x raised to powers of y, element-wise."},
        {"header": "nl.maximum(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Maximum of the inputs, element-wise."},
        {"header": "nl.minimum(x: tile | scalar, y: tile | scalar, dtype=None)", "description": "Minimum of the inputs, element-wise."},
        {"header": "nl.abs(x: tile, dtype=None)", "description": "Element-wise absolute value."},
        {"header": "nl.exp(x: tile, dtype=None)", "description": "Element-wise exponential (e**x)."},
        {"header": "nl.log(x: tile, dtype=None)", "description": "Element-wise natural logarithm."},
        {"header": "nl.sqrt(x: tile, dtype=None)", "description": "Element-wise non-negative square root."},
        {"header": "nl.rsqrt(x: tile, dtype=None)", "description": "Element-wise reciprocal of the square root (1/sqrt(x))."},
        {"header": "nl.square(x: tile, dtype=None)", "description": "Element-wise square (x*x)."},
        {"header": "nl.reciprocal(x: tile, dtype=None)", "description": "Element-wise reciprocal (1/x)."},
        {"header": "nl.sin(x: tile, dtype=None)", "description": "Element-wise sine."},
        {"header": "nl.cos(x: tile, dtype=None)", "description": "Element-wise cosine."},
        {"header": "nl.tanh(x: tile, dtype=None)", "description": "Element-wise hyperbolic tangent."},
        {"header": "nl.ceil(x: tile, dtype=None)", "description": "Element-wise ceiling."},
        {"header": "nl.floor(x: tile, dtype=None)", "description": "Element-wise floor."},
        {"header": "nl.sign(x: tile, dtype=None)", "description": "Element-wise sign of a number."},
        {"header": "nl.negative(x: tile, dtype=None)", "description": "Element-wise negative."},
        {"header": "nl.trunc(x: tile, dtype=None)", "description": "Element-wise truncation."},
    ],
    "ActivationFunctions": [
        {"header": "nl.relu(x: tile, dtype=None)", "description": "Rectified Linear Unit."},
        {"header": "nl.sigmoid(x: tile, dtype=None)", "description": "Sigmoid activation."},
        {"header": "nl.softmax(x: tile, axis: int|tuple, dtype=None)", "description": "Softmax activation along a specified axis."},
        {"header": "nl.gelu(x: tile, dtype=None)", "description": "Gaussian Error Linear Unit."},
        {"header": "nl.silu(x: tile, dtype=None)", "description": "Sigmoid Linear Unit (Swish)."}
    ],
    "ReductionOperations": [
        {"header": "nl.sum(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Sum of elements along a specified free axis. Beta 2: mask parameter removed; use in-bounds slicing."},
        {"header": "nl.prod(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Product of elements along the specified free axis (or free axes) of the input. Beta 2: mask removed."},
        {"header": "nl.all(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Product of elements along the specified free axis (or free axes) of the input. Beta 2: mask removed."},
        {"header": "nl.max(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Maximum of elements along the specific free axis (or free axes) of the input. Beta 2: mask removed."},
        {"header": "nl.min(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Minimum of elements along the specific free axis (or free axes) of the input. Beta 2: mask removed."},
        {"header": "nl.mean(x: tile, axis: int|tuple, dtype=None, keepdims=False)", "description": "Mean of elements along the specific free axis (or free axes) of the input. Beta 2: mask removed."},
        # {"header": "nl.all_reduce(x: tile, op: binary_op, program_axes: int|tuple)", "description": "Performs a reduction (e.g., sum, max) across multiple SPMD programs."}
    ],
    "LogicalBitwise": [
        {"header": "nl.equal(x: tile|scalar, y: tile|scalar)", "description": "Element-wise comparison (x == y)."},
        {"header": "nl.not_equal(x: tile|scalar, y: tile|scalar)", "description": "Element-wise comparison (x != y)."},
        {"header": "nl.greater(x: tile|scalar, y: tile|scalar)", "description": "Element-wise comparison (x > y)."},
        {"header": "nl.less(x: tile|scalar, y: tile|scalar)", "description": "Element-wise comparison (x < y)."},
        {"header": "nl.bitwise_and(x: tile|scalar, y: tile|scalar)", "description": "Element-wise bitwise AND."},
        {"header": "nl.bitwise_or(x: tile|scalar, y: tile|scalar)", "description": "Element-wise bitwise OR."},
        {"header": "nl.invert(x: tile)", "description": "Element-wise bitwise NOT (~x)."}
    ],
    "ShapeAndSelection": [
        {"header": "nl.where(condition: tile[bool], x: tile, y: tile|scalar, dtype=None)", "description": "Return a tile with elements from x where condition is True, and elements from y otherwise. Note x must be a tile."},
        {"header": "nl.broadcast_to(src: tile, *, shape: tuple=None)", "description": "Broadcasts a tile to a new shape. Returns a new tile broadcast along the partition dimension of src, this new tile will be in SBUF, but can be also assigned to a PSUM tensor."},
        {"header": "nki.meta.tensor.broadcast_to(shape: tuple)", "description": "The tensor object must be a tile or can be implicitly converted to a tile. A tensor can be implicitly converted to a tile iff the partition dimension is the highest dimension. Returns a new view of the tile, no copy will occur. (Beta 2: nki.tensor moved to nki.meta.tensor)"},
        {"header": "nl.expand_dims(data: tile, axis: int|tuple)", "description": "Inserts a new dimension of size 1 into the tile's shape."},
        {"header": "nisa.nc_n_gather / nl.gather_flattened", "description": "Beta 2: nl.gather_flattened replaced by nisa.nc_n_gather (free partition limited to 512). Gathers elements from data's flattened free dimension using indices."}
    ],
    "nki.language.affine_range": {
        "header": "nl.affine_range(num_iterations: int):",
        "description": """Create a sequence of numbers for use as parallel loop iterators in NKI. affine_range should be the default loop iterator choice, when there is no loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. A concrete example of associative reduction is multiple nl.matmul or nisa.nc_matmul calls accumulating into the same output buffer defined outside of this loop level (see code example #2 below).
Overlapping load (nisa.dma_copy) outputs in SBUF are considered loop dependencies and are not allowed; buffers can be allocated inside affine_range or outputs can be indexed/sliced to avoid this.
When the dst of an operation inside of affine_range is indexed using [:] (or [, :], or [:, :], etc.), the operation is likely to have an illegal loop carried dependency.
When the above conditions are not met, we recommend using sequential_range instead.
Notes:
    Using affine_range prevents Neuron compiler from unrolling the loops until entering compiler backend, which typically results in better compilation time.
    Using affine_range also allows Neuron compiler to perform additional loop-level optimizations, such as loop vectorization.
    Since each kernel instance only runs on a single NeuronCore, affine_range does not parallelize different loop iterations across multiple NeuronCores. However, different iterations could be parallelized/pipelined on different compute engines within a NeuronCore depending on the invoked instructions (engines) and data dependency in the loop body.
""",
        "examples": """# Example 1: No loop carried dependency
# Input/Output tensor shape: [128, 2048]
# Load one tile ([128, 512]) at a time, square the tensor element-wise,
# and store it into output tile

# Every loop instance works on an independent input/output tile.
# No data dependency between loop instances. Beta 2: use nisa.dma_copy for load/store.
for i_input in nl.affine_range(input.shape[1] // 512):
  offset = i_input * 512
  input_sb = nl.ndarray((input.shape[0], 512), dtype=input.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=input_sb, src=input[0:input.shape[0], offset:offset+512])
  result = nl.multiply(input_sb, input_sb)
  nisa.dma_copy(dst=output[0:input.shape[0], offset:offset+512], src=result)

# Example 2: Matmul output buffer accumulation, a type of associative reduction
# Input tensor shapes for nl.matmul: xT[K=2048, M=128] and y[K=2048, N=128]
# Load one tile ([128, 128]) from both xT and y at a time, matmul and
# accumulate into the same output buffer

result_psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
for i_K in nl.affine_range(xT.shape[0] // 128):
  offset = i_K * 128
  xT_sbuf = nl.ndarray((128, xT.shape[1]), dtype=xT.dtype, buffer=nl.sbuf)
  y_sbuf = nl.ndarray((128, y.shape[1]), dtype=y.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=xT_sbuf, src=xT[offset:offset+128, 0:xT.shape[1]])
  nisa.dma_copy(dst=y_sbuf, src=y[offset:offset+128, 0:y.shape[1]])

  result_psum += nl.matmul(xT_sbuf, y_sbuf, transpose_x=True)""",
    },
    "nki.language.sequential_range": {
        "header": "nl.sequential_range(num_iterations: int):",
        "description": "Create a sequence of numbers for use as sequential loop iterators in NKI. sequential_range should be used when there is a loop carried dependency.",
        "examples": """# Example 1: Loop carried dependency from tiling tensor_tensor_scan
# Both sbuf tensor input0 and input1 shapes: [128, 2048]
# Perform a scan operation between the two inputs using a tile size of [128, 512]
# Store the scan output to another [128, 2048] tensor

# Loop iterations communicate through this init tensor
init = nl.zeros((128, 1), dtype=input0.dtype)

# This loop will only produce correct results if the iterations are performed in order
for i_input in nl.sequential_range(input0.shape[1] // 512):
  offset = i_input * 512

  # Depends on scan result from the previous loop iteration; explicit dst required
  result = nl.ndarray((128, 512), dtype=input0.dtype, buffer=nl.sbuf)
  nisa.tensor_tensor_scan(dst=result, data0=input0[:, offset:offset+512],
                         data1=input1[:, offset:offset+512],
                         initial=init, op0=nl.multiply, op1=nl.add)

  nisa.dma_copy(dst=output[0:input0.shape[0], offset:offset+512], src=result)

  # Prepare initial result for scan in the next loop iteration
  init[:, :] = result[:, 511]""",
    },
    "nki.compiler.sbuf.mod_alloc": {
        "header": "nki.compiler.sbuf.mod_alloc(*, base_addr, base_partition=0, num_par_tiles=(), num_free_tiles=())",
        "description": """Deprecated in NKI Beta 2. Use the improved allocation API (e.g. nl.ndarray with address=) and avoid block dimensions; see NKI Migration Guide.

Allocate SBUF memory space for each logical tile in a tensor through modulo allocation. This is one of the NKI direct allocation APIs (Beta 1). When direct allocation is used, all tensors, including those written by ISA instructions (Beta 2: use explicit dst), in that kernel must also use direct allocation.
When direct allocation is used, HBM tensors cannot be declared unless they are used as kernel outputs.
When direct allocation is used, compute APIs that introduce implicit tensors such as nc_transpose with engine=nisa.tensor_engine will fail.

Parameters:
    base_addr – the base address in the free(F) dimension of the SBUF in bytes.
    base_partition – the partition where the physical tile starts from. Must be 0 in the current version.
    num_par_tiles – the number of physical tiles on the partition dimension of SBUF allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor.
    num_free_tiles – the number of physical tiles on the free dimension of SBUF allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor.
""",
        "examples": """Here’s Beta 1 style (deprecated in Beta 2; block dims removed). Example:

nki_tensor = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.bfloat16,
                        buffer=nki.compiler.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))

for i_block in nl.affine_range(4):
  nisa.dma_copy(dst=nki_tensor[i_block, :, :], src=...)
  ... = nl.exp(nki_tensor[i_block, :, :])

This produces the following allocation:
Logical Tile Index: (0, ); Physical Tile start_partition: 0; Physical Tile byte_addr: 0 + (0 % 2) * 512 * sizeof(nl.bfloat16) = 0
Logical Tile Index: (1, ); Physical Tile start_partition: 0; Physical Tile byte_addr: 0 + (1 % 2) * 512 * sizeof(nl.bfloat16) = 1024
Logical Tile Index: (2, ); Physical Tile start_partition: 0; Physical Tile byte_addr: 0 + (2 % 2) * 512 * sizeof(nl.bfloat16) = 0
Logical Tile Index: (3, ); Physical Tile start_partition: 0; Physical Tile byte_addr: 0 + (3 % 2) * 512 * sizeof(nl.bfloat16) = 1024

With above scheme, we are able to implement double buffering in nki_tensor, such that nisa.dma_copy in one iteration can write to one physical tile while nl.exp of the previous iteration can read from the other physical tile simultaneously.

Here's an example SBufAllocator class that uses mod_alloc to allocate SBUF memory space, incrementing base_addr each time a new allocation is made.

sb_mod = nki.compiler.sbuf.mod_alloc

class SBufAllocator:
    def __init__(self):
        self.offset = 0

    def get_dtype_size(self, dtype):
        if dtype == nl.float32:
            return 4
        elif dtype == nl.bfloat16:
            return 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def allocate(self, size, dtype, num_buffers=1):
        addr = self.offset
        self.offset += size * num_buffers * self.get_dtype_size(dtype)
        return sb_mod(base_addr=addr, num_free_tiles=(num_buffers, ))
""",
    },
    "nki.compiler.psum.mod_alloc": {
        "header": "nki.compiler.psum.mod_alloc(*, base_bank, base_addr=0, base_partition=0, num_bank_tiles=(), num_par_tiles=(), num_free_tiles=())",
        "description": """Deprecated in NKI Beta 2. Use improved allocation API; see NKI Migration Guide.

Allocate PSUM memory space for each logical block in a tensor through modulo allocation. This is one of the NKI direct allocation APIs (Beta 1).

Parameters:
    base_addr – the base address in bytes along the free(F) dimension of the PSUM bank. Must be 0 in the current version.
    base_bank – the base bank ID that the physical tiles start from.
    num_bank_tiles – the number of PSUM banks allocated for the tensor.
    base_partition – the partition ID the physical tiles start from. Must be 0 in the current version.
    num_par_tiles – the number of physical tiles along the partition dimension allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor. Currently must be an empty tuple or (1, 1, …).
    num_free_tiles – the number of physical tiles on the free dimension per PSUM bank allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor. Currently must be an empty tuple or (1, 1, …).
""",
        "examples": """Here’s Beta 1 style (deprecated in Beta 2; block dims removed). Example:

psum_tensor = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.float32,
                         buffer=nki.compiler.psum.mod_alloc(base_bank=0,
                                                    base_addr=0,
                                                    num_bank_tiles=(2,)))

for i_block in nl.affine_range(4):
  nisa.nc_matmul(dst=psum_tensor[i_block, :, :], stationary=..., moving=...)
  ...                 = nl.exp(psum_tensor[i_block, :, :])

This produces the following allocation:

Logical Tile Index: (0, ); Physical Tile bank_id: 0; Physical Tile start_partition: 0; Physical Tile byte_addr: 0
Logical Tile Index: (1, ); Physical Tile bank_id: 1; Physical Tile start_partition: 0; Physical Tile byte_addr: 0
Logical Tile Index: (2, ); Physical Tile bank_id: 0; Physical Tile start_partition: 0; Physical Tile byte_addr: 0
Logical Tile Index: (3, ); Physical Tile bank_id: 1; Physical Tile start_partition: 0; Physical Tile byte_addr: 0

With above scheme, we are able to implement double buffering in nki_tensor, such that nisa.nc_matmul in one iteration can write to one physical tile while nl.exp of the previous iteration can read from the other physical tile simultaneously.
""",
    },
    "nki.language.mgrid": {
        "header": "nl.mgrid[start:end, start:end, ...]",
        "description": "Same as NumPy mgrid: “An instance which returns a dense (or fleshed out) mesh-grid when indexed, so that each returned argument has the same shape. The dimensions and number of the output arrays are equal to the number of indexing dimensions.”",
    },
    "nki.language.ndarray": {
        "header": "nl.ndarray(shape: tuple, dtype: nki_dtype, buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf) -> a new tensor allocated on the buffer",
        "description": "Create a new tensor of given shape and dtype on the specified buffer. (Similar to numpy.ndarray)",
    },
    "nki.language.zeros": {
        "header": "nl.zeros(shape: tuple, dtype: nki_dtype, buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf) -> a new tensor allocated on the buffer",
        "description": "Create a new tensor of given shape and dtype on the specified buffer, filled with zeros. (Similar to numpy.zeros)",
    },
    "nki.language.par_dim": {
        "header": "nl.par_dim(dim: int)",
        "description": "Mark a dimension explicitly as a partition dimension. Beta 2: block dimensions (tensors with par_dim not leftmost) are removed; use a list of tensors or fold the block dimension into the free dimension (partition dim must be leftmost).",
        "examples": """# Example 1 (sb_mod is deprecated in Beta 2; use nl.ndarray(..., buffer=nl.sbuf) or improved allocation API):
identity_load = nl.ndarray((64, nl.par_dim(128)), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca))

# Example 2:
# Most outer loop with batch_size, parallel_for
for i_batch in nl.affine_range(batch_size):
    # partial accumulated scanC result with processed states
    scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)
    ...

# NKI requires inputs of all compute APIs to be valid tiles with the first dimension being the partition dimension.
# We mark the second dimension as the partition dimension
a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
c = nl.add(a, 32) # Error: Failed to infer tile from tensor 'a',

# To fix the problem you can use index tensor a to generate a tile whose first dimension is the partition dimension
# We mark the second dimension of tensor a as the partition dimension
a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
c = nl.ndarray((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
for i in range(4):
  # result of `a[i]` is a tile with shape (8, 8) and the first dimension is the partition dimension
  c[i] = nl.add(a[i], 32) # works
  # Beta 2: nl.arange removed. Use integer slicing for contiguous access, e.g. c[i, 0:8, 0:8] = nl.add(a[i, 0:8, 0:8], 32)
  # Or use nisa.iota(dst=..., pattern=..., offset=...) for index generation; see NKI Migration Guide.""",
    },
    "nki.language.tile_size": {
        "header": """class nl.tile_size, attributes: 
bn_stats_fmax: Maximum free dimension of BN_STATS
gemm_moving_fmax: Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine.
gemm_stationary_fmax: Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine.
pmax: Maximum partition dimension of a tile.
psum_fmax: Maximum free dimension of a tile on PSUM buffer.
psum_min_align: The minimum byte alignment requirement for PSUM free dimension address.
sbuf_min_align: The minimum byte alignment requirement for SBUF free dimension address.
total_available_sbuf_size: The total SBUF available size""",
    },
    
#    "nki.language.load": {
#        "header": "nl.load(src: tile[HBM]) -> tile[SBUF] (same shape as src)",
#        "description": "[Beta 1 only; removed in Beta 2] Use nisa.dma_copy(dst=sbuf_tile, src=hbm_tile) instead.",
#        "examples": """# Partition dimension has to be the first dimension in the index tuple of a tile. Therefore, data may need to be split into multiple batches to load/store, for example:
#for i_b in nl.affine_range(4):
#  data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
#  # load from in_tensor[4, 128, 512] one batch at a time
#  # copy into data_tile[128, 512]
#  i_p, i_f = nl.mgrid[0:128, 0:512]
#  data_tile[i_p, i_f] = nl.load(in_tensor[i_b, i_p, i_f])
#  ...

# Also supports indirect DMA access with dynamic index values:
#"""# Indirect DMA read example 1:
# - data_tensor on HBM has shape [128 x 512].
# - idx_tensor on HBM has shape [64] (with values [0, 2, 4, 6, ...]).
# - idx_tensor values read from HBM and stored in SBUF idx_tile of shape [64 x 1]
# - data_tensor values read from HBM indexed by values in idx_tile
#   and store into SBUF data_tile of shape [64 x 512].
#i_p = nl.arange(64)[:, None]
#i_f = nl.arange(512)[None, :]
#idx_tile = nl.load(idx_tensor[i_p]) # indices have to be in SBUF
#data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f])
#...
# Indirect DMA read example 2:
# - data_tensor on HBM has shape [128 x 512].
# - idx_tile on SBUF has shape [64 x 1] (with values [[0], [2], [4], ...] generated by iota)
# - data_tensor values read from HBM indexed by values in idx_tile 
#   and store into SBUF data_tile of shape [64 x 512].
# i_f = nl.arange(512)[None, :]

# idx_expr = 2*nl.arange(64)[:, None]
# idx_tile = nisa.iota(idx_expr, dtype=np.int32)
# data_tile = nl.load(data_tensor[idx_tile, i_f]) 
# ...""",
#    },
    
#     "nki.language.store": {
#        "header": "nl.store(dst: tile[HBM], value: tile[SBUF])",
#        "description": """[Beta 1 only; removed in Beta 2] Use nisa.dma_copy(dst=hbm_tile, src=sbuf_tile) instead. Store into a tensor on device memory (HBM) from on-chip memory (SBUF).
#Parameters:
#    dst – HBM tensor to store the data into.
#    value – An SBUF tile that contains the values to store. If the tile is in PSUM, an extra copy will be performed to move the tile to SBUF first.""",
#        "examples": """# Partition dimension has to be the first dimension in the index tuple of a tile. Therefore, data may need to be split into multiple batches to load/store, for example:
# for i_b in nl.affine_range(4):
#  data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 

#...
#    store into out_tensor[4, 128, 512] one batch at a time
# from data_tile[128, 512] 
#i_p, i_f = nl.mgrid[0:128, 0:512]
#nl.store(out_tensor[i_b, i_p, i_f], value=data_tile[i_p, i_f]) 

# Also supports indirect DMA access with dynamic index values:
# Indirect DMA write example 1:
#  - data_tensor has shape [128 x 512].
#  - idx_tensor on HBM has shape [64] (with values [0, 2, 4, 6, ...]).
#  - idx_tensor values read from HBM and stored in SBUF idx_tile.
#  - data_tile of shape [64 x 512] values written into
#    HBM data_tensor indexed by values in idx_tile.
# i_p = nl.arange(64)[:, None]
# i_f = nl.arange(512)[None, :]
# idx_tile = nl.load(idx_tensor[i_p]) # indices have to be in SB

# nl.store(data_tensor[idx_tile[i_p, 0], i_f], value=data_tile[0:64, 0:512])

# Indirect DMA write example 2:
#  - data_tensor has shape [128 x 512].
#  - idx_tile on SBUF has shape [64 x 1] (with values [[0], [2], [4], ...] generated by iota)
#  - data_tile of shape [64 x 512] values written into
#    HBM data_tensor indexed by values in idx_tile.
# idx_expr = 2*nl.arange(64)[:, None]
# idx_tile = nisa.iota(idx_expr, dtype=np.int32)

# nl.store(data_tensor[idx_tile, i_f], value=data_tile[0:64, 0:512])""",
#     },
    "nki.isa.activation": {
        "header": "nisa.activation(dst: tile[SBUF|PSUM], op: activation_function, data: tile[SBUF|PSUM], bias: tile[vector]=None, scale: scalar|tile[vector]=1.0, reduce_op: reduce_function=None, reduce_res: tile[vector]=None, reduce_cmd: nisa.reduce_cmd=nisa.reduce_cmd.idle, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation before the activation and an optional reduction operation after the activation in the same instruction.

The activation function is specified in the op input field (see Supported Activation Functions for NKI ISA for a list of supported activation functions and their valid input ranges).

nisa.activation can optionally multiply the input data by a scalar or vector scale and then add another vector bias before the activation function is applied. output = op(data * scale + bias)

When the scale is a scalar, it must be a compile-time constant. In this case, the scale is broadcasted to all the elements in the input data tile. When the scale/bias is a vector, it must have the same partition axis size as the input data tile and only one element per partition. In this case, the element of scale/bias within each partition is broadcasted to elements of the input data tile in the same partition.
There are 128 registers on the scalar engine for storing reduction results, corresponding to the 128 partitions of the input. The scalar engine can reduce along free dimensions without extra performance penalty, and store the result of reduction into these registers. The reduction is done after the activation function is applied.

output = f_act(data * scale + bias)
accu_registers = reduce_op(accu_registers, reduce_op(output, axis = <FreeAxis>))

These registers are shared between activation and activation_accu calls, and the state of them can be controlled via the reduce_cmd parameter.
    nisa.reduce_cmd.reset: Reset the accumulators to zero
    nisa.reduce_cmd.idle: Do not use the accumulators
    nisa.reduce_cmd.reduce: keeps accumulating over the current value of the accumulator
    nisa.reduce_cmd.reset_reduce: Resets the accumulators then immediately accumulate the results of the current instruction into the accumulators

We can choose to read out the current values stored in the register by passing in a tensor in the reduce_res arguments. Reading out the accumulator will incur a small overhead.
Note that activation_accu can also change the state of the registers. It’s user’s responsibility to ensure correct ordering. It’s recommended to not mixing the use of activation_accu and activation, when reduce_cmd is not set to idle.
Note, the Scalar Engine always performs the math operations in float32 precision. Therefore, the engine automatically casts the input data tile to float32 before performing multiply/add/activate specified in the activation instruction. The engine is also capable of casting the float32 math results into another output data type specified by the dtype field at no additional performance cost. If dtype field is not specified, Neuron Compiler will set output data type of the instruction to be the same as input data type of data. On the other hand, the scale parameter must have a float32 data type, while the bias parameter can be float32/float16/bfloat16.
The input data tile can be an SBUF or PSUM tile. Similarly, the instruction can write the output tile into either SBUF or PSUM, which is specified using the buffer field. If not specified, nki.language.sbuf is selected by default.

Pipelined Multiply-Add
Each ScalarE compute lane also supports an additional multiply-add before the non-linear function (func) is applied in a pipeline fashion. Mathematically, ScalarE implements:
# Case 1: scale is SBUF/PSUM vector
# Input: 2D in_tile, 1D scale, 1D bias
# Output: 2D out_tile
for lane_id in range(in_tile.shape[0]):
   for k in range(in_tile.shape[1])
    out_tile[lane_id][k] = func(in_tile[lane_id][k] * scale[lane_id]
                                    + bias[lane_id])
# Case 2: scale is a compile-time scalar constant in the instruction
for lane_id in range(in_tile.shape[0]):
   for k in range(in_tile.shape[1])
    out_tile[lane_id][k] = func(in_tile[lane_id][k] * scale
                                    + bias[lane_id])
This functionality can be invoked using the nki.isa.activation API by specifying a scale for multiplication and bias for addition. The scale can either be a tile from SBUF/PSUM with one element/partition or a compile-time constant. On the other hand, the bias can only be a tile from SBUF/PSUM with one element/partition. A useful mental model for this capability is combining a nki.isa.tensor_scalar instruction with a non-linear function evaluation into a single instruction (2x speed-up than two separate instructions).

Pipelined Reduction
Each ScalarE compute lane also supports reduction after the non-linear function (func) is applied in a pipeline fashion. On NeuronCore-v2, the reduction operator can only be addition.
Mathematically, ScalarE with accumulation enabled implements:
# Input: 2D in_tile, 1D scale (similarly for scalar scale), 1D bias
# Output: 2D out_tile, 1D reduce_res
for lane_id in range(in_tile.shape[0]):
  for k in range(in_tile.shape[1]):
    out_tile[lane_id][k] = func(in_tile[lane_id][k] * scale[lane_id]
                                 + bias[lane_id])
    reduce_res[lane_id] += out_tile[lane_id][k]
This functionality can be invoked using the nki.isa.activation API by specifying reduce_op as nki.language.add and reduce_res as the output reduction tile, passed by reference.
A useful mental model for this capability is combining a nki.isa.activation instruction with a nki.isa.tensor_reduce into a single API, which returns results from both APIs. Note, this invokes two back-to-back ISA instructions on hardware, Activate and ActReadAccumulator. The Activate instruction performs the regular computation as specified in nki.isa.activation and also reduction at no additional cost. The reduction result is cached inside ScalarE after Activate. The ActReadAccumulator instruction is a low cost (roughly 64 ScalarE cycles on NeuronCore-v2) instruction to write the internal reduction result back to SBUF/PSUM, one element per partition.

Estimated instruction cost:
max(MIN_II, N) Scalar Engine cycles, where
    N is the number of elements per partition in data.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Parameters:
    op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)
    data – the input tile; layout: (partition axis <= 128, free axis)
    bias – a vector with the same partition axis size as data for broadcast add (after broadcast multiply with scale)
    scale – a scalar or a vector with the same partition axis size as data for broadcast multiply
    reduce_op – the reduce operation to perform on the free dimension of the activation result
    reduce_res – a tile of shape (data.shape[0], 1), where data.shape[0] is the partition axis size of the input data tile. The result of sum(ReductionResult) is written in-place into the tensor.
    reduce_cmd – an enum member from nisa.reduce_cmd to control the state of reduction registers
    dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Result is written to dst (same layout as input data tile); the instruction does not return a value.
""",
        "examples": """# Example 1: perform exponential function on matrix a of shape (128, 1024)
# Beta 2: use nisa.dma_copy for load; nisa.activation requires explicit dst.
a = nl.ndarray((128, 1024), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor)
activated_a = nl.ndarray((128, 1024), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.activation(dst=activated_a, op=nl.exp, data=a)
nisa.dma_copy(dst=a_act_tensor, src=activated_a)

# Example 2: np.square(b * 2.0 + c), cast to bfloat16
b = nl.ndarray((128, 512), dtype=b_tensor.dtype, buffer=nl.sbuf)
c = nl.ndarray((128, 512), dtype=c_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=b, src=b_tensor)
nisa.dma_copy(dst=c, src=c_tensor)
activated_b = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
nisa.activation(dst=activated_b, op=np.square, data=b, bias=c, scale=2.0, dtype=nl.bfloat16)
nisa.dma_copy(dst=b_act_tensor, src=activated_b)""",

# # Example 3: Compute softmax with exp, bias subtraction, and reduction
# # Applies exp(qk_sbuf - row_max) and accumulates results into sum_row_tiles
# exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
#                         dtype=nl.bfloat16, buffer=nl.sbuf)
# qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING),
#                 dtype=nl.float32, buffer=nl.sbuf)
# row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
#                          buffer=nl.sbuf)
# sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
#                        buffer=nl.sbuf)
# def exp_row_sum(i_tile_q):
#     for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
#         nisa.activation(
#             dst=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)],
#             op=nl.exp,
#             data=qk_sbuf[:, i_tile_kv, :],
#             bias=row_max[:, :],
#             reduce_op=nl.add,
#             reduce_res=sum_row_tiles[:, i_tile_kv],
#             reduce_cmd=nisa.reduce_cmd.reset_reduce,
#             dtype=nl.bfloat16
#             )
# """,
    },
    "nki.isa.reduce_cmd": {
        "header": "nisa.reduce_cmd",
        "description": """Engine Register Reduce commands
.idle: Not using the accumulator registers
.reset: Resets the accumulator registers to its initial state
.reset_reduce: Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators
.reduce: keeps accumulating over the current value of the accumulator registers"""
    },
    "nki.isa.activation_reduce": {
        "header": "nisa.activation_reduce(dst: tile[SBUF|PSUM], op: activation_function, data: tile[SBUF|PSUM], reduce_op: reduce_function, reduce_res: tile[vector], bias: tile[vector]=None, scale: scalar|tile[vector]=1.0, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Perform the same computation as nisa.activation and also a reduction along the free dimension of the nisa.activation result using Scalar Engine. The reduction result is stored in reduce_res in-place.

This API is equivalent to calling nisa.activation with reduce_cmd=nisa.reduce_cmd.reset_reduce and passing in reduce_res. We recommend using nisa.activation for new code. Refer to nisa.activation for semantics of op/data/bias/scale.

In addition to nisa.activation computation, this API performs a reduction along the free dimension(s) of the activation result. The reduction result is written in-place into reduce_res, which must be a SBUF/PSUM tile with the same partition axis size as data and one element per partition. On NeuronCore-v2, reduce_op must be nl.add. Reduction axis is not configurable; if the input has multiple free axes, the API reduces across all of them.

Parameters:
  dst – output tile of the activation; same layout as input data tile
  op – activation function (see Supported Activation Functions for NKI ISA)
  data – input tile; layout: (partition axis <= 128, free axis)
  reduce_op – reduce operation on the free dimension of the activation result
  reduce_res – tile of shape (data.shape[0], 1); reduction result written in-place
  bias – optional vector for broadcast add (after scale)
  scale – optional scalar or vector for broadcast multiply""",
        "examples": "",
    },
    "nki.isa.affine_select": {
        "header": "nisa.affine_select(dst: tile[SBUF], pattern: list, offset: int32, channel_multiplier: int32, on_true_tile: tile[SBUF], on_false_value: scalar, cmp_op: comparison_op, dtype: nki_dtype=on_true_tile.dtype, name=None)",
        "description": """Select elements between an input tile on_true_tile and a scalar value on_false_value according to a boolean predicate tile using GpSimd Engine.

        The predicate tile is calculated on-the-fly by evaluating an affine expression element-by-element. The affine expression is defined by pattern, offset, and channel_multiplier, similar to nisa.iota. The pattern field is a list of lists in the form [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]. When fewer than 4D pattern is provided, NKI compiler automatically pads remaining dimensions with size of 1.

        Given a 4D pattern (padded if needed), the instruction generates a predicate: for each position, affine_value = offset + (channel_id * channel_multiplier) + (w*step_w + z*step_z + y*step_y + x*step_x); predicate = cmp_op(affine_value, 0). If predicate is True, dst gets on_true_tile; else dst gets on_false_value.

        A common use case is to apply a causal mask on attention scores for transformer decoder models.

        Memory types: The output dst tile must be in SBUF. The input on_true_tile must also be in SBUF.

        Data types: on_true_tile and dst can be any valid NKI data type. If they differ, selected input elements are cast to FP32 then to dst.dtype. on_false_value must be float32.

        Layout: The partition dimension determines the number of active channels. on_true_tile, the predicate, and dst must have the same partition dimension size and same number of elements per partition.

        Tile size: Partition dimension size of dst and on_true_tile must be the same and must not exceed 128. Total elements in pattern must match elements per partition in dst and on_true_tile.

Parameters:
  dst – the output tile in SBUF to store the selected values
  pattern – list of [step, num] for up to 4D tensor sizes and strides for affine expression
  offset – int32 offset added to every generated affine value
  channel_multiplier – int32 multiplier applied to the channel (partition) ID
  on_true_tile – input tile for selection when predicate is True
  on_false_value – scalar value for selection when predicate is False
  cmp_op – comparison operator for predicate evaluation (default: nl.equal)
""",
        "examples": """# Example 1: Take tile a of shape [128, 128] and replace its
# upper triangle with nl.fp32.min. Beta 2: use nisa.dma_copy for load/store; affine_select takes pattern and cmp_op.
ix, iy = nl.mgrid[0:128, 0:128]
a = nl.ndarray((128, 128), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor[ix, iy])
b = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
nisa.affine_select(dst=b, pattern=..., cmp_op=..., on_true_tile=a, on_false_value=nl.fp32.min)
nisa.dma_copy(dst=b_tensor[ix, iy], src=b)
""",
    },
    "nki.isa.bn_aggr": {
        "header": "nisa.bn_aggr(dst: tile[SBUF|PSUM], data: tile[SBUF|PSUM], dtype: nki_dtype=data.dtype, name=None)",
        "description": """Aggregate one or multiple bn_stats outputs to generate a mean and variance per partition using Vector Engine.

The input data tile effectively has an array of (count, mean, variance*count) tuples per partition produced by bn_stats instructions. Therefore, the number of elements per partition of data must be a multiple of three.

Note, if you need to aggregate multiple bn_stats instruction outputs, it is recommended to declare a SBUF tensor and then make each bn_stats instruction write its output into the SBUF tensor at different offsets.

Vector Engine performs the statistics aggregation in float32 precision. The engine automatically casts the input data to float32 before computation. The float32 results are cast to dst.dtype at no additional performance cost.

Parameters:
  dst – an output tile with two elements per partition: a mean followed by a variance
  data – an input tile with results of one or more bn_stats""",
    },
    "nki.isa.bn_stats": {
        "header": "nisa.bn_stats(dst: tile[SBUF|PSUM], data: tile[SBUF|PSUM], dtype: nki_dtype=data.dtype, name=None)",
        "description": """Compute mean- and variance-related statistics for each partition of an input tile data in parallel using Vector Engine.

The output tile of the instruction has 6 elements per partition: the count of the even elements, the mean of the even elements, variance*count of the even elements, the count of the odd elements, the mean of the odd elements, variance*count of the odd elements.

To get the final mean and variance of the input tile, pass the bn_stats output into the bn_aggr instruction, which outputs two elements per partition: mean and variance.

Due to hardware limitation, the number of elements per partition (free dimension size) of the input data must not exceed 512 (nl.tile_size.bn_stats_fmax). To calculate per-partition mean/variance of a tensor with more than 512 elements in free dimension, invoke bn_stats on each 512-element tile and use a single bn_aggr to aggregate the outputs.

Vector Engine performs the statistics calculation in float32 precision. The engine automatically casts the input data to float32. The float32 results are cast to dst.dtype at no additional performance cost.

Parameters:
  dst – an output tile with 6-element statistics per partition
  data – the input tile (up to 512 elements per partition)""",
        "examples": """# Example 1: Calculate the mean and variance for each partition
# of tile a with shape (128, 128). Beta 2: use nisa.dma_copy for load/store.
a = nl.ndarray((128, 128), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor)
stats_a = nl.ndarray((128, 6), dtype=nl.float32, buffer=nl.sbuf)
nisa.bn_stats(dst=stats_a, data=a)
mean_var_a = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
nisa.bn_aggr(dst=mean_var_a, data=stats_a)
mean_a = mean_var_a[:, 0]
var_a = mean_var_a[:, 1]
nisa.dma_copy(dst=mean_a_tensor, src=mean_a)
nisa.dma_copy(dst=var_a_tensor, src=var_a)

# Example 2: Calculate the mean and variance for each partition of
# tile b with shape [128, 1024]
b = nl.ndarray((128, 1024), dtype=b_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=b, src=b_tensor)

# Run bn_stats in two tiles because b has 1024 elements per partition,
# but bn_stats has a limitation of nl.tile_size.bn_stats_fmax
# Initialize a bn_stats output tile with shape of [128, 6*2] to
# hold outputs of two bn_stats instructions
stats_b = nl.ndarray((128, 6 * 2), dtype=nl.float32)
bn_tile = nl.tile_size.bn_stats_fmax
ix, iy = nl.mgrid[0:128, 0:bn_tile]
iz, iw = nl.mgrid[0:128, 0:6]

for i in range(1024 // bn_tile):
  nisa.bn_stats(dst=stats_b[:, i * 6:(i + 1) * 6], data=b[:, i * bn_tile:(i + 1) * bn_tile])

mean_var_b = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
nisa.bn_aggr(dst=mean_var_b, data=stats_b)

# Extract mean and variance
mean_b = mean_var_b[:, 0]
var_b = mean_var_b[:, 1]

nisa.dma_copy(dst=mean_b_tensor, src=mean_b)
nisa.dma_copy(dst=var_b_tensor, src=var_b)""",
    },
    "nki.isa.dma_copy": {
        "header": "nisa.dma_copy(dst: tile[HBM|SBUF], src: tile[HBM|SBUF], dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, unique_indices: bool=True, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Copy data from src to dst using DMA engines with optional read-modify-write operations.

This instruction performs data movement between memory locations (SBUF or HBM) using DMA engines. The basic operation copies data from the source tensor to the destination tensor: dst = src. Optionally, a read-modify-write operation can be performed where the source data is combined with existing destination data using a specified operation: dst = dst_rmw_op(dst, src).

Currently, only nl.add is supported for dst_rmw_op when performing read-modify-write operations. When dst_rmw_op=None, the source data directly overwrites the destination data.

nisa.dma_copy supports different modes of DMA descriptor generation (DGE): nisa.dge_mode.none (Neuron Runtime generates DMA descriptors into HBM before NEFF execution), nisa.dge_mode.swdge (GpSimd Engine generates descriptors during execution), nisa.dge_mode.hwdge (Sync/Scalar Engine sequencers invoke DGE hardware during execution). See Trainium2 arch guide and Introduction to DMA with NKI for more.

When either sw_dge or hw_dge mode is used, src and dst can have a dynamic start address. When sw_dge is selected, nisa.dma_copy can also perform gather or scatter using a list of unique dynamic indices from SBUF. Out-of-bound address checking is on by default (oob_mode.error); use oob_mode.skip to skip out-of-bound indices. If dst_rmw_op is specified for dynamic modes, only oob_mode.error is allowed.

Memory types: Both src and dst tiles can be in HBM or SBUF. If both are in SBUF, consider nisa.tensor_copy, nisa.nc_n_gather, or nisa.local_gather for better performance.

Data types: Both src and dst can be any supported NKI data type. DMA engines handle type conversion via float32. If dst_rmw_op is used, inputs are cast to float32 and the result is cast to output dtype.

Tile size: The total number of data elements in src must match that of dst.

Parameters:
  dst – the destination tensor to copy data into
  src – the source tensor to copy data from
  dst_rmw_op – optional read-modify-write operation (currently only nl.add supported)
  unique_indices – optional; only used when dst_rmw_op is used; indicates whether scatter indices are unique
  dge_mode – optional; nki.isa.dge_mode.none, .swdge, .hwdge, or .unknown (default)
  oob_mode – optional; oob_mode.error (default) raises error on out-of-bounds; oob_mode.skip skips them""",
        "examples": """# Beta 2: use nki.* namespace; import nki and nki.isa as nisa.
from nki.typing import tensor

# Example 1: Simple load from HBM to SBUF and store back (dst required).
x = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=x, src=in_tensor)
# ... compute on x ...
nisa.dma_copy(dst=out_tensor, src=x)

# Example 2: Load elements from HBM with indirect addressing. If addressing 
# results out-of-bound access, the operation will fail.
# Beta 2: nisa.iota(dst, pattern, offset); nisa.memset(dst, value).
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]

idx_tile = nl.ndarray((n//2, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[2, n//2]], offset=0)
out_tile = nl.ndarray((n//2, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=-1)
nisa.dma_copy(dst=out_tile[ix, iy], src=in_tensor[idx_tile, iy], oob_mode=nisa.oob_mode.error)

# Example 3: Load elements from HBM with indirect addressing. If addressing 
# results in out-of-bounds access, the operation will fail.
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]
idx_tile = nl.ndarray((n//2, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[3, n//2]], offset=0)
out_tile = nl.ndarray((n//2, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=-1)
nisa.dma_copy(dst=out_tile[ix, iy], src=in_tensor[idx_tile, iy], oob_mode=nisa.oob_mode.error)

# Example 4: Load elements from HBM with indirect addressing. If addressing 
# results in out-of-bounds access, the operation will skip indices.
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]
idx_tile = nl.ndarray((n//2, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[3, n//2]], offset=0)
out_tile = nl.ndarray((n//2, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=-1)
nisa.dma_copy(dst=out_tile[ix, iy], src=in_tensor[idx_tile, iy], oob_mode=nisa.oob_mode.skip)

# Example 5: Store elements to HBM with indirect addressing and with 
# read-modify-write operation. Beta 2: use nisa.dma_copy for load/store.
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]
inp_tile = nl.ndarray((n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=inp_tile, src=in_tensor[ix, iy])
idx_tile = nl.ndarray((n, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[2, n]], offset=0)
out_tile = nl.ndarray((2*n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=1)
nisa.dma_copy(dst=out_tensor[0:2*n, 0:m], src=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, dst_rmw_op=np.add)

# Example 6: Store elements to HBM with indirect addressing. If indirect 
# addressing results out-of-bound access, the operation will fail.
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]
inp_tile = nl.ndarray((n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=inp_tile, src=in_tensor[ix, iy])
idx_tile = nl.ndarray((n, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[2, n]], offset=0)
out_tile = nl.ndarray((2*n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=-1)
nisa.dma_copy(dst=out_tensor[0:2*n, 0:m], src=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)

# Example 7: Store elements to HBM with indirect addressing. If indirect 
# addressing results out-of-bounds access, the operation will skip indices.
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]
inp_tile = nl.ndarray((n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=inp_tile, src=in_tensor[ix, iy])
idx_tile = nl.ndarray((n, 1), dtype=np.int32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[3, n]], offset=0)
out_tile = nl.ndarray((2*n, m), dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.memset(dst=out_tile, value=-1)
nisa.dma_copy(dst=out_tensor[0:2*n, 0:m], src=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)
""",
    },
    "nki.isa.dma_transpose": {
        "header": "nisa.dma_transpose(dst: tile[HBM|SBUF], src: tile[HBM|SBUF], axes: tuple=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Perform a transpose on input src using DMA Engine.

    The permutation of transpose follow the rules described below:
        For 2-d input tile, the permutation will be [1, 0]
        For 3-d input tile, the permutation will be [2, 1, 0]
        For 4-d input tile, the permutation will be [3, 1, 2, 0]

    DMA Transpose Constraints
    The only valid dge_mode s are unknown and hwdge. If hwdge, this instruction will be lowered to a Hardware DGE transpose. This has additional restrictions:
        src.shape[0] == 16
        src.shape[-1] % 128 == 0
        dtype is 2 bytes

Parameters:
  dst – the transpose output
  src – the source of transpose, must be a tile in HBM or SBUF
  axes – transpose axes; supported: (1, 0), (2, 1, 0), (3, 1, 2, 0)
  dge_mode – optional; nki.isa.dge_mode.none, .swdge, .hwdge, or .unknown (default)
  oob_mode – optional; oob_mode.error (default) or oob_mode.skip for out-of-bounds indices
""",
        "examples": "",
    },
    "nki.isa.dropout": {
        "header": "nisa.dropout(dst: tile, data: tile, prob: scalar|tile[vector], dtype: nki_dtype=data.dtype, name=None)",
        "description": """Randomly replace some elements of the input tile data with zeros based on input probabilities using Vector Engine. The probability of replacing elements with zeros (drop probability) is specified by prob: if 1.0, all elements are replaced with zeros; if 0.0, all are kept.

prob can be a scalar constant or a tile of shape (data.shape[0], 1), where each partition has one drop probability applicable to that partition. Data type of data can be any valid NKI type; prob has restrictions: if data is integer type, prob must be float32; if data is float type, prob can be any valid float type. dst.dtype must match data.dtype.

Parameters:
  dst – output tile of the dropout result
  data – the input tile
  prob – scalar or tile (data.shape[0], 1) for probability of replacing elements with zeros""",
        "examples": "",
    },
    "nki.isa.get_nc_version": {
        "header": "nisa.get_nc_version() -> nisa.nc_version",
        "description": """Returns the nc_version of the current target context.""",
        "examples": "",
    },
    "nki.isa.iota": {
        "header": "nisa.iota(dst: tile[SBUF], pattern: list, offset: int32, channel_multiplier: int32=0, dtype: nki_dtype=dst.dtype, name=None)",
        "description": """Generate a constant literal pattern into SBUF using GpSimd Engine.

The pattern is defined by an int32 offset, a tensor access pattern of up to 4D pattern and an int32 channel_multiplier. The pattern field is a list of lists in the form [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]. When fewer than 4D pattern is provided, NKI compiler automatically pads remaining dimensions with size of 1.

Given a 4D pattern (padded if needed), the instruction generates: for each channel_id, w, z, y, x: value = offset + (channel_id * channel_multiplier) + (w*step_w + z*step_z + y*step_y + x*step_x); dst[channel_id, w, z, y, x] = value.

Memory types: The output dst tile must be in SBUF.

Data types: Generated values are computed in 32-bit integer arithmetic. GpSimd Engine can cast to any valid NKI data type before writing; output type is determined by dst.dtype.

Layout: The partition dimension determines the number of active channels.

Tile size: Partition dimension size of dst must not exceed 128. Number of elements per partition of dst must not exceed physical SBUF partition size. Total elements in pattern must match elements per partition in dst.

Parameters:
  dst – the output tile in SBUF to store the generated pattern
  pattern – list of [step, num] to describe up to 4D tensor sizes and strides
  offset – int32 offset value added to every generated value
  channel_multiplier – int32 multiplier applied to the channel (partition) ID""",
        "examples": "",
    },
    "nki.isa.local_gather": {
        "header": "nisa.local_gather(dst: tile[SBUF], src_buffer: tile[SBUF], index: tile[SBUF], num_elem_per_idx: int=1, num_valid_indices: int=None, dtype: nki_dtype=src_buffer.dtype, name=None)",
        "description": """Gather SBUF data in src_buffer using index on GpSimd Engine.

Each of the eight GpSimd cores connects to 16 contiguous SBUF partitions and performs gather from those 16 partitions independently. Indices used for gather on each core should come from the same 16 connected SBUF partitions. For gathering within a single partition, consider nisa.nc_n_gather.

Each core reads a 16-partition slice from index, flattens indices to 1D, and uses them as partition offsets to gather from the connected slice of src_buffer. num_elem_per_idx allows gathering multiple contiguous elements per index. num_valid_indices can specify valid index count per core when not a multiple of 16. Out-of-bound index access is undefined. local_gather preserves input data types; no casting. index must be uint16.

Constraints: src_buffer.shape[0] == index.shape[0] and multiple of 16; num_elem_per_idx in [1, 2, 4, 8, 16, 32]; indices per core <= 4096.

Parameters:
  dst – output tile of the gathered data
  src_buffer – input tile for gathering
  index – input tile with indices (uint16)
  num_elem_per_idx – optional; contiguous elements per index per partition; default 1
  num_valid_indices – optional; valid indices per GpSimd core""",
        "examples": "",
    },
    "nki.isa.max8": {
        "header": "nisa.max8(dst: tile, src: tile, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Find the 8 largest values in each partition of the source tile using Vector Engine.

Reads input elements, converts to fp32 internally, and outputs the 8 largest values in descending order per partition. Outputs are converted to dst.dtype automatically. Source can be up to 5-dimensional; output is always 2D. Elements per partition must be between 8 and 16,384 inclusive. Output has exactly 8 elements per partition. Source and output must have the same partition dimension size: source [par_dim, ...], output [par_dim, 8].

Parameters:
  dst – 2D tile [par_dim, 8] with the 8 largest values per partition in descending order
  src – source tile to find maximum values from""",
        "examples": "",
    },
    "nki.isa.memset": {
        "header": "nisa.memset(dst: tile[SBUF], value: scalar, engine=engine.unknown, dtype: nki_dtype=dst.dtype, name=None)",
        "description": """Initialize dst by filling it with a compile-time constant value, using Vector or GpSimd Engine. The memset instruction supports all valid NKI dtypes (see Supported Data Types).

Parameters:
  dst – destination tile to initialize
  value – the constant value to initialize with
  engine – specify which engine to use: nki.isa.vector_engine or nki.isa.gpsimd_engine; nki.isa.unknown_engine by default""",
        "examples": "",
    },
    "nki.isa.nc_find_index8": {
        "header": "nisa.nc_find_index8(dst: tile, data: tile, vals: tile, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Find indices of the 8 given values in each partition of the data tensor using Vector Engine.

Loads the 8 values from vals, then loads the data tensor and outputs the indices (starting at 0) of the first occurrence of each value in data, per partition. data can be up to 5-dimensional; vals must be up to 3-dimensional. data must have between 8 and 16,384 elements per partition. vals must have exactly 8 elements per partition. Output has exactly 8 elements per partition and is uint16 or uint32 (default uint32). Behavior is undefined if a value in vals is not in data. If provided, mask is applied only to the data tensor.
Behavior is undefined if vals tensor contains values that are not in the data tensor.

Parameters:
  dst – 2D tile [par_dim, 8] containing indices (uint16 or uint32) of the 8 values per partition
  data – data tensor to search
  vals – tensor with 8 values per partition whose indices are found""",
        "examples": "",
    },
    "nki.isa.nc_match_replace8": {
        "header": "nisa.nc_match_replace8(dst: tile, data: tile, vals: tile, imm: scalar, dst_idx: tile=None, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Replace first occurrence of each value in vals with imm in data using Vector Engine; write result to dst. If dst_idx is provided, write indices of matched values to dst_idx.

Reads input data, replaces the first occurrence of each of the 8 values (from vals) with the immediate constant, and optionally outputs indices of matches to dst_idx. Free dimensions of data and vals are flattened for the operation but preserved in output and dst_idx. Partition dimension is the parallelization boundary. data can be up to 5D; vals up to 3D. vals must have exactly 8 elements per partition; data at most 16,384 per partition. Output shape matches input data. data and vals can be SBUF or PSUM. Behavior undefined if a value in vals is not in data.

Parameters:
  dst – modified data tensor (same shape as data)
  data – data tensor to modify
  vals – tensor with 8 values per partition to replace
  imm – float32 constant to replace matched values with
  dst_idx – optional tile for flattened indices of matched values""",
        "examples": "",
    },
    "nki.language.matmul": {
        "header": "nl.matmul(x, y, *, transpose_x=False)",
        "description": """x @ y matrix multiplication of x and y. (Similar to numpy.matmul).
Parameters:
        x – a tile on SBUF (partition dimension <= 128, free dimension <= 128), x’s free dimension must match y’s partition dimension.
        y – a tile on SBUF (partition dimension <= 128, free dimension <= 512)
        transpose_x – Defaults to False. If True, x is treated as already transposed. If False, an additional transpose will be inserted to make x’s partition dimension the contract dimension of the matmul to align with the Tensor Engine.
Result: x @ y or x.T @ y if transpose_x=True. Beta 2: mask removed.
""",
    },
    "nki.isa.nc_matmul": {
        "header": "nisa.nc_matmul(dst: tile[PSUM], stationary: tile[SBUF], moving: tile[SBUF], is_stationary_onezero: bool=False, is_moving_onezero: bool=False, is_transpose: bool=False, tile_position: tuple=(), tile_size: tuple=(), perf_mode=matmul_perf_mode.none, dtype: nki_dtype=dst.dtype, name=None)",
        "description": """Compute dst = stationary.T @ moving using the Tensor Engine.

Make the contraction dimension as close as possible to 128 without exceeding it. The stationary tensor should be transposed so that the contraction dimension is in the first (K, M) position. Both stationary and moving inputs must be SBUF tiles; dst must be a PSUM tile.
If the contraction dimension exceeds 128, accumulate multiple nc_matmul outputs into the same PSUM tile.
The nc_matmul instruction currently supports float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32 input data types.

In NKI, to perform a multiplication of two matrices, x[M, K] and y[K, N], allocate a PSUM tile for the result and call nisa.nc_matmul(dst=..., stationary=..., moving=...). The result has shape [M, N]. At the hardware level, TensorE requires both input tiles to have the contraction dimension K in the SBUF partition dimension, that is, the first dimension of input shapes (LC #1 as discussed in NKI Programming Model). This ISA requirement is reflected in the low-level API nki.isa.nc_matmul, which takes stationary and moving matrices as input parameters. Therefore, nki.isa.nc_matmul(x, y) is a two-step computation: invoking nki.isa.nc_transpose(x) to get stationary and then nki.isa.nc_matmul(stationary, moving) to get the final result. In other words, nki.isa.nc_matmul(stationary[K,M], moving[K,N]) performs a stationary.T @ moving calculation, which will result in an output with dimensions [M,N].
For every nki.isa.nc_matmul(stationary, moving) call, TensorE executes two distinct Neuron ISA instructions:
    LoadStationary (short for LS): This instruction loads the stationary from SBUF and caches it in internal storage of TensorE
    MultiplyMoving (short for MM): This instruction loads the moving from SBUF and multiplies moving across the pre-loaded stationary matrix from the previous LoadStationary instruction. The output of this instruction is the output of the nki.isa.nc_matmul call written to PSUM.

Alternative Use Case
One interesting use case of TensorE is low-latency data reshape within NeuronCore, which typically involves multiplying a matrix to be reshaped with a compile-time constant matrix filled with zeros and ones.
As an example, we can perform a 128x128 matrix transposition (i.e., swap the free and partition axis of the matrix) using nki.isa.nc_matmul(transpose_input, identity), where transpose_input is the matrix to be transposed and identity is a 128x128 identity matrix. In fact, this is exactly what nki.isa.nc_transpose() does, when TensorE is chosen as the compute engine.
Similarly, we can broadcast a vector occupying a single partition to M (M <= 128) partitions using nki.isa.nc_matmul(ones, broadcast_input, is_stationary_onezero=True), where ones is a 1xM vector filled with ones and broadcast_input is the vector to be broadcast. In fact, NKI invokes such matmul under the hood when broadcast_input.broadcast_to((M, broadcast_input.shape[1])) is called.
In general, we can achieve many more complex data reshapes in TensorE, such as shuffling partitions of a SBUF tensor, by constructing appropriate zero/one patterns as one of the matmul inputs.
Finally, we can also leverage TensorE for data summation across SBUF partitions (P-dim summation). For example, a vector laid out across SBUF partitions can be reduced into a single sum using TensorE as shown in the diagram below. Note, this utilizes only a single PE column of the TensorE; therefore, depending on the surrounding operators, this may not be the best use of TensorE. If you can do summation within each partition (F-dim summation), see nki.isa.tensor_reduce for an alternative reduction implementation on Vector Engine. It is recommended to choose the engine based on the natural layout of your input data to avoid any transpositions.
As TensorE is the most performant compute engine of the NeuronCore in terms of FLOPS, the goal is to have it execute meaningful computation at high utilization as much as possible. The above “alternative use cases” stop TensorE from performing useful computations at high throughput and therefore, should generally be avoided. However, there are situations where it is advisable to use them:
    Operators that do not require heavy matmuls anyhow, e.g. normalization, softmax.
    Layout conflicts between producer and consumer engines where broadcast/transpose are absolutely unavoidable (see example in fused attention tutorial).

Performance Consideration
As a rule of thumb, TensorE can achieve the best throughput when it runs many back-to-back nki.isa.nc_matmul with both input matrices at the largest possible tiles sizes (stationary is 128x128 and moving is 128x512). In this ideal scenario, TensorE sees the below instruction sequence:
    LoadStationary (LS[0]) (128x128)
    MultiplyMoving (MM[0]) (128x512)
    LoadStationary (LS[1]) (128x128)
    MultiplyMoving (MM[1]) (128x512)
    ...
Cost Model: TensorE is a deeply pipelined engine; therefore, the engine can have several LS&MM instruction pairs in-flight at a given time. Due to this pipelining nature, it is often not useful to use end-to-end execution latency of a single instruction when estimating the instruction cost. Instead, we can focus on the initiation interval of such instructions, that is, the number of cycles between successive instruction launches. Therefore, we can estimate the cost of an instruction I by how soon TensorE can issue the next instruction after I.
For the sake of discussion, let’s assume we have many back-to-back MM instructions with BF16/FP16/TF32/cFP8 input data type that reuse a single pre-loaded stationary inside TensorE. The initiation interval between subsequent MM instructions in this case is roughly max(N, MM_INIT_LATENCY), where MM_INIT_LATENCY is 64 TensorE cycles on NeuronCore-v2, and N is the free axis size of moving of current MM (typically set to 512). For FP32 input data type, the instruction cost is roughly 4x higher than BF16/FP16/TF32/cFP8. Therefore, whenever possible, we recommend down-casting FP32 input matrix data type to one of BF16/FP16/TF32/cFP8 before performing matrix multiplications.
Background LoadStationary: In typical workloads, TensorE would be alternating between LS and MM instructions with different input matrices. In order to optimize TensorE’s utilization, we also enable a “background LoadStationary” capability, which allows loading of the next stationary tensor in parallel to the computation on the current stationary tensor.
As a result, depending on the relative sizes of the stationary and moving matrices, the overall TensorE performance can be bounded by either LS or MM instructions.
Fast LoadStationary: Since LoadStationary is a pure data movement with no computation, TensorE can perform LoadStationary up to 4x faster than a MultiplyMoving with the same free axis size. Fast LoadStationary has an important performance implication on nki.isa.nc_matmul: When one of the input matrices has a small free axis size and the other has a large free axis size, we prefer to put the matrix with large free axis as the stationary matrix. For example, if we try to do a vector-matrix multiplication, it is recommended to put the matrix as stationary matrix and vector as moving matrix to get the best performance out of TensorE.

Cost (Tensor Engine Cycles)
if input data type is one of float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32, cost is max(min(64, N_stationary), N_moving)
if input data type is float32, cost is 4 * max(min(64, N_stationary), N_moving)
where,
    N_stationary is the number of elements per partition in stationary tile.
    N_moving is the number of elements per partition in moving tile.

Args:
    stationary – the stationary operand on SBUF; layout: (partition axis <= 128, free axis <= 128)
    moving – the moving operand on SBUF; layout: (partition axis <= 128, free axis <= 512)
    is_stationary_onezero – hints to the compiler whether the stationary operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if stationary tile is in float32; the field has no impact for non-float32 stationary.
    is_moving_onezero – hints to the compiler if the moving operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if moving tile is in float32; the field has no impact for non-float32 moving.
    is_transpose – hints to the compiler that this is a transpose operation with moving as an identity matrix.
    tile_position – a 2D tuple (row, column) for the start PE tile position to run nc_matmul.
    tile_size – a 2D tuple (row, column) for the PE tile size to hold by nc_matmul starting from tile_position.
Result is written to dst (PSUM tile; layout: partition axis from free axis of stationary, free axis from free axis of moving); the instruction does not return a value.
""",
        "examples": """# Example 1: multiply matrix a (128, 128) and b (128, 512) -> c in PSUM (128, 512)
# Beta 2: use nisa.dma_copy for load/store (nl.load/nl.store removed).
a_mgrid = nl.mgrid[0:128, 0:128]
b_mgrid = nl.mgrid[0:128, 0:512]
c_mgrid = nl.mgrid[0:128, 0:512]
a = nl.ndarray((128, 128), dtype=a_tensor.dtype, buffer=nl.sbuf)
b = nl.ndarray((128, 512), dtype=b_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor[a_mgrid.p, a_mgrid.x])
nisa.dma_copy(dst=b, src=b_tensor[b_mgrid.p, b_mgrid.x])
c_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
nisa.nc_matmul(dst=c_psum, stationary=a, moving=b)
nisa.dma_copy(dst=c_tensor[c_mgrid.p, c_mgrid.x], src=c_psum)

# Example 2: d (256,128) @ e (256,512) -> f (128,512) with psum accumulation
d_mgrid = nl.mgrid[0:128, 0:128]
e_mgrid = nl.mgrid[0:128, 0:512]
f_mgrid = nl.mgrid[0:128, 0:512]
f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)
tmp_psum = nl.ndarray((128, 512), nl.float32, buffer=nl.psum)
for i_contract in nl.affine_range(2):
  d = nl.ndarray((128, 128), dtype=d_tensor.dtype, buffer=nl.sbuf)
  e = nl.ndarray((128, 512), dtype=e_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=d, src=d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
  nisa.dma_copy(dst=e, src=e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
  nisa.nc_matmul(dst=tmp_psum, stationary=d, moving=e)
  f_psum_sbuf = nl.ndarray((128, 512), nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=f_psum_sbuf, src=f_psum)
  nisa.tensor_tensor(dst=f_psum, data1=f_psum_sbuf, data2=tmp_psum, op=nl.add)
nisa.dma_copy(dst=f_tensor[f_mgrid.p, f_mgrid.x], src=f_psum)

# Example 3:
# perform batched matrix multiplication on matrix g of shape (16, 64, 64) 
# and matrix h of shape (16, 64, 512) to get matrix i of (16, 64, 512) 
# using Tensor Engine PE tiling mode. 
g_mgrid = nl.mgrid[0:64, 0:64]
h_mgrid = nl.mgrid[0:64, 0:512]
i_mgrid = nl.mgrid[0:64, 0:512]
for i in nl.affine_range(4):
  for j in nl.affine_range(4):
    g = nl.ndarray((64, 64), dtype=g_tensor.dtype, buffer=nl.sbuf)
    h = nl.ndarray((64, 512), dtype=h_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g, src=g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
    nisa.dma_copy(dst=h, src=h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
    i_psum = nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=i_psum, stationary=g, moving=h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
    nisa.dma_copy(dst=i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], src=i_psum)""",
    },
    "nki.isa.nc_transpose": {
        "header": "nisa.nc_transpose(dst: tile[SBUF|PSUM], data: tile[SBUF|PSUM], dtype: nki_dtype=data.dtype, engine=engine.unknown, name=None)",
        "description": """Perform a 2D transpose between the partition axis and the free axis of input data, i.e., a PF-transpose, using Tensor or Vector Engine. If the data tile has more than one free axes, this API implicitly collapses all free axes into one axis and then performs a 2D PF-transpose.

In NeuronCore, both Tensor and Vector Engine can perform a PF-transpose, but they support different input shapes. Tensor Engine nc_transpose can handle an input tile of shape (128, 128) or smaller, while Vector Engine can handle shape (32, 32) or smaller. Therefore, when the input tile shape is (32, 32) or smaller, we have an option to run it on either engine, which is controlled by the engine field. If no engine is specified, Neuron Compiler will automatically select an engine based on the input shape. Note, similar to other Tensor Engine instructions, the Tensor Engine nc_transpose must read the input tile from SBUF and write the transposed result to PSUM. On the other hand, Vector Engine nc_transpose can read/write from/to either SBUF or PSUM.

Note, PF-transpose on Tensor Engine is done by performing a matrix multiplication between data as the stationary tensor and an identity matrix as the moving tensor. See architecture guide for more information. On NeuronCore-v2, such matmul-style transpose is not bit-accurate if the input data contains NaN/Inf. You may consider replacing NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix before calling nc_transpose(engine=nki.isa.constants.engine.tensor).
Parameters:
    data – the input tile to be transposed
    dtype – if specified and it’s different from the data type of input tile data, an additional nki.isa.cast instruction will be inserted to cast the transposed data into the target dtype (see Supported Data Types for more information)
    engine – specify which engine to use for transpose: nki.isa.tensor_engine or nki.isa.vector_engine ; by default, the best engine will be selected for the given input tile shape
Result is written to dst (transposed result of input data); the instruction does not return a value.

Cost (Engine Cycles)
if engine is vector, max(MIN_II, N)
if engine is tensor and assuming many back-to-back nc_transpose of the same shape, max(P, min(64, F))
where,
    N is the number of elements per partition in data.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
    P is partition axis size of data.
    F is the number of elements per partition in data.
""",
        "examples": """# Example 1: transpose tile a of shape (128, 64). Beta 2: use slicing; dst required.
aT = nl.ndarray((64, 128), dtype=a.dtype, buffer=nl.sbuf)
nisa.nc_transpose(dst=aT, data=a[0:128, 0:64])

# Example 2: transpose tile b (32, 2) using Vector Engine
bT = nl.ndarray((2, 32), dtype=b.dtype, buffer=nl.sbuf)
nisa.nc_transpose(dst=bT, data=b[0:32, 0:2], engine=nisa.vector_engine)
""",
    },
    "nki.isa.range_select": {
        "header": "nisa.range_select(dst: tile, on_true_tile: tile, comp_op0: comparison_op, comp_op1: comparison_op, bound0: tile[vector], bound1: tile[vector], reduce_cmd: nisa.reduce_cmd=idle, reduce_res: tile[vector]=None, reduce_op: reduce_function=np.amax, range_start: int=0, on_false_value: scalar=fp32.min, dtype: nki_dtype=on_true_tile.dtype, name=None)",
        "description": """Select elements from on_true_tile based on comparison with bounds using Vector Engine. Available only on NeuronCore-v3 and newer.

For each element in on_true_tile, compares (free dimension index + range_start) against bound0 and bound1 using comp_op0 and comp_op1. If both comparisons are True, copy the element to output; otherwise use on_false_value. Also performs a reduction (reduce_op) on the results, storing in reduce_res. on_false_value is always treated as FP32_MIN for numerical stability; reduce_cmd is always reset_reduce.

Comparison operators must be one of: nl.equal, nl.less, nl.less_equal, nl.greater, nl.greater_equal. Partition dim sizes must match across on_true_tile, bound0, bound1; bound0 and bound1 have one element per partition; on_true_tile must be FP dtype, bound0/bound1 FP32. reduce_op currently only nl.maximum supported.

Parameters:
  dst – output tile with selected elements
  on_true_tile – input tile to select from
  comp_op0, comp_op1 – comparison operators
  bound0, bound1 – tiles with one element per partition (FP32)
  reduce_res – optional tile for reduction results
  range_start – base offset for free-dim index; compile-time integer, default 0""",
        "examples": "",
    },
    "nki.isa.reciprocal": {
        "header": "nisa.reciprocal(dst: tile[SBUF|PSUM], data: tile[SBUF|PSUM], dtype: nki_dtype=data.dtype, name=None)",
        "description": """Compute element-wise reciprocal (1.0/x) of the input data tile using Vector Engine.

Memory types: Both the input data and output dst tiles can be in SBUF or PSUM.

Data types: The input data tile can be any valid NKI data type. The Vector Engine automatically casts the input to float32 and performs the reciprocal in float32 math. The float32 results are cast to the data type of dst.

Layout: The partition dimension of the input data is the parallel compute dimension.

Tile size: The partition dimension size of input data and output dst tiles must be the same and must not exceed 128. The number of elements per partition of dst must match that of data and must not exceed the physical size of each SBUF partition.

Parameters:
  dst – the output tile
  data – the input tile""",
        "examples": "",
    },
    "nki.isa.scalar_tensor_tensor": {
        "header": "nisa.scalar_tensor_tensor(dst: tile, data: tile, op0: binary_op, operand0: scalar|tile[vector], op1: binary_op, operand1: tile, reverse0: bool=False, reverse1: bool=False, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Apply two math operators in sequence using Vector Engine: (data op0 operand0) op1 operand1.

Equivalent to: temp = tensor_scalar(data, op0, operand0); dst = tensor_tensor(temp, operand1, op1). operand0 can be a compile-time scalar (broadcast) or a tile (data.shape[0], 1). operand1 must have the same shape as data. Scalar broadcast in the first op has no extra cost; latency is approximately that of a single tensor_tensor. Both op0 and op1 must be arithmetic operators (see Supported Math Operators); bitvec not supported. Use reverse0/reverse1 for non-commutative operand order.

Memory: data and operand1 can be SBUF or PSUM but not both in PSUM; operand0 can be SBUF, PSUM, or scalar; dst can be SBUF or PSUM. Engine casts inputs to float32 and casts result to dst.dtype. Partition size of data, operand1, dst must match and not exceed 128; elements per partition must match.

Layout: The parallel computation dimension of nisa.scalar_tensor_tensor is along the partition dimension.

Tile size: The partition dimension size of input data, operand1, and output dst tiles must be the same and must not exceed 128. The total number of elements per partition of input data, operand1, and output dst tiles must be the same and must not exceed the physical size of each SBUF partition. If operand0 is not a scalar, the partition dimension size of operand0 must be the same as that of data and the number of elements per partition of operand0 must be 1.

Parameters:
  dst – output tile
  data – input tile
  op0, operand0 – first operator and operand (scalar or tile (P, 1))
  op1, operand1 – second operator and tile (same shape as data)
  reverse0, reverse1 – reverse operand order for non-commutative ops""",
        "examples": "",
    },
    "nki.isa.select_reduce": {
        "header": "nisa.select_reduce(dst: tile, predicate: tile, on_true: tile, on_false: scalar|tile[vector], reduce_res: tile[vector]=None, reduce_cmd: nisa.reduce_cmd=idle, reduce_op: reduce_function=np.amax, reverse_pred: bool=False, dtype: nki_dtype=on_true.dtype, name=None)",
        "description": """Selectively copy elements from on_true or on_false to dst based on predicate using Vector Engine, with optional max reduction.

Where predicate is True, copy from on_true to dst; where False, copy from on_false. With reduction enabled, max of each partition of the result is computed and stored in reduce_res. reduce_op only supports max. Memory: both on_true and predicate in SBUF allowed; either on_true or predicate may be in PSUM but not both; dst can be SBUF or PSUM. on_true, dst, predicate must have identical shapes. predicate dtype: int8, uint8, int16, uint16. reduce_res shape (on_true.shape[0], 1), float type. reduce_cmd controls accumulator: reset_reduce, reduce, or idle. Accumulator registers are shared with e.g. range_select; use reset_reduce after idle for consistent behavior.

Parameters:
  dst – destination tile for selected values
  predicate – tile determining on_true vs on_false
  on_true – tile used when predicate is True
  on_false – scalar or vector (on_true.shape[0], 1) when predicate is False
  reduce_res – optional tile for reduction results
  reduce_cmd – nisa.reduce_cmd (idle, reset_reduce, reduce)
  reverse_pred – reverse meaning of predicate""",
        "examples": "",
    },
    "nki.isa.sequence_bounds": {
        "header": "nisa.sequence_bounds(dst: tile, segment_ids: tile, dtype: nki_dtype=segment_ids.dtype, name=None)",
        "description": """Compute the sequence bounds for a given set of segment IDs using GpSimd Engine.

Given a tile of segment IDs, identifies where each segment begins and ends. For each element returns [start_index, end_index] for the segment that element belongs to. All segment IDs must be non-negative. Padding elements (segment ID zero) get start index n and end index -1, where n is the length of segment_ids. Output tile has two values per input element (start and end); partition dimension must be 1. Example: input (1, 512) -> output (1, 2, 512). segment_ids and dst must be nl.float32 or nl.int32.

Parameters:
  dst – tile containing sequence bounds (start, end per element)
  segment_ids – tile of segment IDs; elements with ID=0 are padding""",
        "examples": "",
    },
    "nki.isa.tensor_copy": {
        "header": "nisa.tensor_copy(dst: tile[SBUF|PSUM], src: tile[SBUF|PSUM], engine=engine.unknown, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Create a copy of src tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

The output tile has the same partition axis size and the same number of elements per partition as the input tile src.

tensor_copy casting behavior: When src and dst data types are the same, tensor_copy performs a bit-accurate copy. When they differ, tensor_copy performs an intermediate FP32 cast.

Since GpSimd Engine cannot access PSUM, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details). On NeuronCore v2, tensor_copy is not supported on the Scalar Engine; use nisa.activation with op=nl.copy instead.

Estimated instruction cost:
max(MIN_II, N) engine cycles, where N is the number of elements per partition in the input tile, and MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
Constraints. Supported engines: NeuronCore v2: Vector, GpSimd. NeuronCore v3+: Vector, Scalar, GpSimd. Since GpSimd cannot access PSUM, src and dst must be in SBUF when using GpSimd Engine.

Parameters:
  dst – a tile with the same content and partition axis size as the src tile
  src – the source of copy, must be a tile in SBUF or PSUM
  engine – optional; nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default)
""",
        "examples": """# Example 1: Copy over the tensor to another tensor using the Vector engine.
# Beta 2: dst required; use nisa.dma_copy for HBM<->SBUF.
x = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=x, src=in_tensor)
x_copy = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
nisa.tensor_copy(dst=x_copy, src=x, engine=nisa.vector_engine)
nisa.dma_copy(dst=out_tensor, src=x_copy)
""",
    },
    "nki.isa.tensor_copy_dynamic_dst": {
        "header": "nisa.tensor_copy_dynamic_dst(dst: tile, src: tile, engine: nisa.engine=unknown, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Copy from src to dst when the destination has a dynamic layout. The destination tile dst can have a runtime-determined shape or addressing. See NKI API for dynamic tensor handling.""",
        "examples": "",
    },
    "nki.isa.tensor_copy_dynamic_src": {
        "header": "nisa.tensor_copy_dynamic_src(dst: tile, src: tile, engine: nisa.engine=unknown, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Copy from src to dst when the source has a dynamic layout. The source tile src can have a runtime-determined shape or addressing. Result is written to dst. See NKI API for dynamic tensor handling.""",
        "examples": "",
    },
    "nki.isa.tensor_copy_predicated": {
        "header": "nisa.tensor_copy_predicated(dst: tile[SBUF|PSUM], src: tile[SBUF|PSUM]|scalar, predicate: tile[SBUF|PSUM], reverse_pred: bool=False, dtype: nki_dtype=src.dtype, name=None)",
        "description": """Conditionally copy elements from the src tile to the destination tile on SBUF/PSUM based on a predicate using Vector Engine.

This instruction provides low-level control over conditional data movement. Either src or predicate may be in PSUM, but not both. Both may be in SBUF.

Shape and data type constraints: src (if tensor), dst, and predicate must have the same number of partitions and same number of elements per partition. predicate must be uint8, uint16, or uint32. src and dst must share the same data type.

Estimated instruction cost (Vector Engine Cycles):
max(MIN_II, N), if src is from SBUF and predicate is from PSUM or the other way around
max(MIN_II, 2N), if both src and dst are in SBUF
N is the number of elements per partition in src tile
MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
Behavior: Where predicate is True, corresponding elements from src are copied to dst (or the scalar if src is a scalar). Where predicate is False, dst is unmodified.


Parameters:
  dst – The destination tile to copy elements to
  src – The source tile or scalar to copy from when predicate is True
  predicate – A tile that determines which elements to copy
  reverse_pred – A boolean that reverses the effect of predicate
""",
        "examples": """# Example 1: Conditionally copy from on_true into dst where predicate is True.
# Beta 2: use nisa.dma_copy for HBM->SBUF load.
pre_tile = nl.ndarray((128, 512), dtype=predicate.dtype, buffer=nl.sbuf)
src_tile = nl.ndarray((128, 512), dtype=on_true_tensor.dtype, buffer=nl.sbuf)
dst_tile = nl.zeros(shape=(128, 512), dtype=on_true_tensor.dtype)
nisa.dma_copy(dst=pre_tile, src=predicate)
nisa.dma_copy(dst=src_tile, src=on_true_tensor)
nisa.dma_copy(dst=dst_tile, src=on_false_tensor)
nisa.tensor_copy_predicated(src=src_tile, dst=dst_tile, predicate=pre_tile)
""",
    },
    "nki.isa.tensor_partition_reduce": {
        "header": "nisa.tensor_partition_reduce(dst: tile, op: reduce_function, data: tile, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Apply a reduction operation across partitions of the input data tile using GpSimd Engine. Reduces along the partition dimension (unlike tensor_reduce which reduces along free dimensions).

Parameters:
  dst – output tile with reduced result
  op – reduction operator (e.g. add, max, bitwise_or, bitwise_and)
  data – input tile to be reduced""",
        "examples": "",
    },
    "nki.isa.tensor_reduce": {
        "header": "nisa.tensor_reduce(dst: tile[SBUF|PSUM], op: reduce_function, data: tile[SBUF|PSUM], axis: int|tuple, negate: bool=False, keepdims: bool=False, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Apply a reduction operation to the free axes of an input data tile using Vector Engine.
The reduction operator is specified in the op input field (see Supported Math Operators for NKI ISA for a list of supported reduction operators). There are two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or) and 2) arithmetic operators (e.g., add, subtract, multiply). For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, there is no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the reduction operation in float32 math. The float32 reduction results are cast to the target data type specified in the dtype field before written into the output tile. If the dtype field is not specified, it is default to be the same as input tile data type.
When the reduction op is an arithmetic operator, the instruction can also multiply the output reduction results by -1.0 before writing into the output tile, at no additional performance cost. This behavior is controlled by the negate input field.
The reduction axes are specified in the axis field using a list of integer(s) to indicate axis indices. The reduction axes can contain up to four free axes and must start at the most minor free axis. Since axis 0 is the partition axis in a tile, the reduction axes must contain axis 1 (most-minor). In addition, the reduction axes must be consecutive: e.g., [1, 2, 3, 4] is a legal axis field, but [1, 3, 4] is not.
Since this instruction only supports free axes reduction, the output tile must have the same partition axis size as the input data tile. To perform a partition axis reduction, we can either:
    invoke a nki.isa.nc_transpose instruction on the input tile and then this reduce instruction to the transposed tile, or
    invoke nki.isa.nc_matmul instructions to multiply a nki.language.ones([128, 1], dtype=data.dtype) vector with the input tile.

Estimated instruction cost (Vector Engine Cycles):
N/2, ifboth input and output data types are bfloat16 and the reduction operator is add or maximum
N, otherwise
where,
    N is the number of elements per partition in data.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Parameters:
        op – the reduction operator (see Supported Math Operators for NKI ISA for supported reduction operators)
        data – the input tile to be reduced
        axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4]
        dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
        negate – if True, reduction result is multiplied by -1.0; only applicable when op is an arithmetic operator
        keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
Result is written to dst; the instruction does not return a value.""",
        "examples": """# Example 1: reduce add tile a (128, 512) along free dim -> b (128, 1)
# Beta 2: use slicing for contiguous access; explicit dst required.
b = nl.ndarray((128, 1), dtype=a.dtype, buffer=nl.sbuf)
nisa.tensor_reduce(dst=b, op=np.add, data=a[0:128, 0:512], axis=[1])""",
    },
    "nki.isa.tensor_scalar": {
        "header": "nisa.tensor_scalar(dst: tile[SBUF|PSUM], data: tile[SBUF|PSUM], op0: binary_op, operand0: scalar|tile[vector], reverse0: bool=False, op1: binary_op=None, operand1: scalar|tile[vector]=None, reverse1: bool=False, dtype: nki_dtype=data.dtype, engine=engine.unknown, name=None)",
        "description": """Apply up to two math operators to the input data tile by broadcasting scalar/vector operands in the free dimension using Vector or Scalar or GpSimd Engine: (data <op0> operand0) <op1> operand1.
The input data tile can be an SBUF or PSUM tile. Both operand0 and operand1 can be SBUF or PSUM tiles of shape (data.shape[0], 1), i.e., vectors, or compile-time constant scalars.
op1 and operand1 are optional, but must be None (default values) when unused. Note, performing one operator has the same performance cost as performing two operators in the instruction.
When the operators are non-commutative (e.g., subtract), we can reverse ordering of the inputs for each operator through:
    reverse0 = True: tmp_res = operand0 <op0> data
    reverse1 = True: operand1 <op1> tmp_res
The tensor_scalar instruction supports two types of operators: 1) bitvec operators (e.g., bitwise_and) and 2) arithmetic operators (e.g., add). See Supported Math Operators for NKI ISA for the full list of supported operators. The two operators, op0 and op1, in a tensor_scalar instruction must be of the same type (both bitvec or both arithmetic). If bitvec operators are used, the tensor_scalar instruction must run on Vector Engine. Also, the input/output data types must be integer types, and input elements are treated as bit patterns without any data type casting.
If arithmetic operators are used, the tensor_scalar instruction can run on Vector or Scalar or GpSimd Engine. However, each engine supports limited arithmetic operators (see :ref:tbl-aluop). The Scalar Engine on trn2 only supports a subset of the operator combination:
        op0=np.multiply and op1=np.add
        op0=np.multiply and op1=None
        op0=add and op1=None
Also, arithmetic operators impose no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the operators in float32 math. The float32 computation results are cast to the target data type specified in the dtype field before written into the output tile, at no additional performance cost. If the dtype field is not specified, it is default to be the same as input tile data type.

Estimated instruction cost:
max(MIN_II, N) Vector or Scalar Engine cycles, where
    N is the number of elements per partition in data.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Parameters:
        data – the input tile
        op0 – the first math operator used with operand0 (see Supported Math Operators for NKI ISA for supported operators)
        operand0 – a scalar constant or a tile of shape (data.shape[0], 1), where data.shape[0] is the partition axis size of the input data tile. operand0's free dimension must be 1 (a vector).
        reverse0 – reverse ordering of inputs to op0; if false, operand0 is the rhs of op0; if true, operand0 is the lhs of op0
        op1 – the second math operator used with operand1 (see Supported Math Operators for NKI ISA for supported operators); this operator is optional
        operand1 – a scalar constant or a tile of shape (data.shape[0], 1), where data.shape[0] is the partition axis size of the input data tile
        reverse1 – reverse ordering of inputs to op1; if false, operand1 is the rhs of op1; if true, operand1 is the lhs of op1
        dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
        engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine (only allowed for rsqrt) or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

Result is written to dst; the instruction does not return a value.""",
        "examples": """
# Example 1: subtract 1.0 from all elements of tile a (128, 512). Beta 2: use slicing; explicit dst.
b = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
nisa.tensor_scalar(dst=b, data=a[0:128, 0:512], op0=np.subtract, operand0=1.0)


# Example 2: (1.0 - c) with reverse0=True
d = nl.ndarray((128, 512), dtype=c.dtype, buffer=nl.sbuf)
nisa.tensor_scalar(dst=d, data=c[0:128, 0:512], op0=np.subtract, operand0=1.0, reverse0=True)


# Example 3: e (64, 1024) * f (64, 1) + 2.5
g = nl.ndarray((64, 1024), dtype=e.dtype, buffer=nl.sbuf)
nisa.tensor_scalar(dst=g, data=e[0:64, 0:1024], op0=np.multiply, operand0=f[0:64, 0:1], op1=np.add, operand1=2.5)""",
    },
    "nki.isa.tensor_scalar_reduce": {
        "header": "nisa.tensor_scalar_reduce(dst: tile, data: tile, op0: binary_op, operand0: scalar|tile[vector], reduce_op: reduce_function, reduce_res: tile[vector], reverse0: bool=False, dtype: nki_dtype=data.dtype, name=None)",
        "description": """Perform the same computation as nisa.tensor_scalar with one math operator and also a reduction along the free dimension of the nisa.tensor_scalar result using Vector Engine.
Refer to nisa.tensor_scalar for semantics of data/op0/operand0. Unlike regular nisa.tensor_scalar where two operators are supported, only one operator is supported in this API. Also, op0 can only be arithmetic operation in Supported Math Operators for NKI ISA. Bitvec operators are not supported in this API.
In addition to nisa.tensor_scalar computation, this API also performs a reduction along the free dimension(s) of the nisa.tensor_scalar result, at a small additional performance cost. The reduction result is returned in reduce_res in-place, which must be a SBUF/PSUM tile with the same partition axis size as the input tile data and one element per partition. The reduce_op can be any of nl.add, nl.subtract, nl.multiply, nl.max or nl.min.
Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will reduce across all of them.

result = data <op0> operand0
reduce_res = reduce_op(dst, axis=<FreeAxis>)

Estimated instruction cost:
max(MIN_II, N) + MIN_II Vector Engine cycles, where
    N is the number of elements per partition in data, and
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Parameters:
        data – the input tile
        op0 – the math operator used with operand0 (any arithmetic operator in Supported Math Operators for NKI ISA is allowed)
        operand0 – a scalar constant or a tile of shape (data.shape[0], 1), where data.shape[0] is the partition axis size of the input data tile
        reverse0 – (not supported yet) reverse ordering of inputs to op0; if false, operand0 is the rhs of op0; if true, operand0 is the lhs of op0. <– currently not supported yet.
        reduce_op – the reduce operation to perform on the free dimension of data <op0> operand0
        reduce_res – a tile of shape (data.shape[0], 1), where data.shape[0] is the partition axis size of the input data tile. The result of reduce_op(data <op0> operand0) is written in-place into the tile.
        dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
Result is written to dst; reduction result is written in-place to reduce_res; the instruction does not return a value.""",
    },
    "nki.isa.tensor_tensor": {
        "header": "nisa.tensor_tensor(dst: tile[SBUF|PSUM], data1: tile[SBUF|PSUM], data2: tile[SBUF|PSUM], op: binary_op, dtype: nki_dtype=None, engine=engine.unknown, name=None)",
        "description": """Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine. The two tiles must have the same partition axis size and the same number of elements per partition.
The element-wise operator is specified using the op field and can be any binary operator supported by NKI that runs on the Vector Engine, or it can be power or integer add, multiply, or subtract which run on the GpSimd Engine. For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, there is no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the element-wise operation in float32 math (unless it is one of the supported integer ops mentioned above). The float32 results are cast to the target data type specified in the dtype field before written into the output tile. If the dtype field is not specified, it is default to be the same as the data type of data1 or data2, whichever has the higher precision.
Since GpSimd Engine cannot access PSUM, the input or output tiles cannot be in PSUM if op is one of the GpSimd operations mentioned above. (see NeuronCore-v2 Compute Engines for details). Otherwise, the output tile can be in either SBUF or PSUM. However, the two input tiles, data1 and data2 cannot both reside in PSUM. The three legal cases are:
    Both data1 and data2 are in SBUF.
    data1 is in SBUF, while data2 is in PSUM.
    data1 is in PSUM, while data2 is in SBUF.
Note, if you need broadcasting capability in the free dimension for either input tile, you should consider using nki.isa.tensor_scalar API instead, which has better performance than nki.isa.tensor_tensor in general.

Estimated instruction cost:
See below table for tensor_tensor performance when it runs on Vector Engine.
Cost (Vector Engine Cycles)
max(MIN_II, N), if one input tile is in PSUM and the other is in SBUF
max(MIN_II, N), if all of the below:
    both input tiles are in SBUF,
    input/output data types are all bfloat16,
    the operator is add, multiply or subtract,
    Input tensor data is contiguous along the free dimension (that is, stride in each partition is 1 element)
max(MIN_II, 2N), otherwise

where,
    N is the number of elements per partition in data1/data2.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Parameters:
    data1 – lhs input operand of the element-wise operation
    data2 – rhs input operand of the element-wise operation
    op – a binary math operator (see Supported Math Operators for NKI ISA for supported operators)
    dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).
Result is written to dst; the instruction does not return a value.""",
        "examples": """# Example 1: add two tiles a and b element-wise. Beta 2: use nisa.dma_copy for load; explicit dst.
a = nl.ndarray((128, 512), dtype=a_tensor.dtype, buffer=nl.sbuf)
b = nl.ndarray((128, 512), dtype=b_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=a, src=a_tensor)
nisa.dma_copy(dst=b, src=b_tensor)
c = nl.ndarray((128, 512), dtype=a_tensor.dtype, buffer=nl.sbuf)
nisa.tensor_tensor(dst=c, data1=a, data2=b, op=nl.add)""",
    },
    "nki.isa.tensor_tensor_scan": {
        "header": "nisa.tensor_tensor_scan(dst: tile[SBUF|PSUM], data0: tile[SBUF|PSUM], data1: tile[SBUF|PSUM], initial: scalar|tile[vector], op0: binary_op, op1: binary_op, reverse0: bool=False, reverse1: bool=False, dtype: nki_dtype=None, name=None)",
        "description": """Perform a scan operation of two input tiles using Vector Engine.
Mathematically, the tensor_tensor_scan instruction on Vector Engine performs the following computation per partition:

# Let's assume we work with numpy, and data0 and data1 are 2D (with shape[0] being the partition axis)
import numpy as np
result = np.ndarray(data0.shape, dtype=data0.dtype)
result[:, 0] = op1(op0(data0[:. 0], initial), data1[:, 0])
for i in range(1, data0.shape[1]):
    result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i])

The two input tiles (data0 and data1) must have the same partition axis size and the same number of elements per partition. The third input initial can either be a float32 compile-time scalar constant that will be broadcasted in the partition axis of data0/data1, or a tile with the same partition axis size as data0/data1 and one element per partition.
The two input tiles, data0 and data1 cannot both reside in PSUM. The three legal cases are:
    Both data1 and data2 are in SBUF.
    data1 is in SBUF, while data2 is in PSUM.
    data1 is in PSUM, while data2 is in SBUF.
The scan operation supported by this API has two programmable math operators in op0 and op1 fields. Both op0 and op1 can be any binary arithmetic operator supported by NKI (see Supported Math Operators for NKI ISA for details). We can optionally reverse the input operands of op0 by setting reverse0 to True (or op1 by setting reverse1). Reversing operands is useful for non-commutative operators, such as subtract.

Input/output data types can be any supported NKI data type (see Supported Data Types), but the engine automatically casts input data types to float32 and performs the computation in float32 math. The float32 results are cast to the target data type specified in the dtype field before written into the output tile. If the dtype field is not specified, it is default to be the same as the data type of data0 or data1, whichever has the highest precision.

Estimated instruction cost:
max(MIN_II, 2N) Vector Engine cycles, where
    N is the number of elements per partition in data0/data1.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
Parameters:
        data0 – lhs input operand of the scan operation
        data1 – rhs input operand of the scan operation
        initial – starting state of the scan; can be a SBUF/PSUM tile with 1 element/partition or a scalar compile-time constant
        op0 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)
        op1 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)
        reverse0 – reverse ordering of inputs to op0; if false, data0 is the lhs of op0; if true, data0 is the rhs of op0
        reverse1 – reverse ordering of inputs to op1; if false, data1 is the rhs of op1; if true, data1 is the lhs of op1
        dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
Result is written to dst; the instruction does not return a value.""",
        "examples": """# Example 1: scan two tiles, a and b, of the same
# shape (128, 1024) using multiply/add and get
# the scan result in tile c; explicit dst required.
c = nl.ndarray(shape=(128, 1024), dtype=nl.float32)

nisa.tensor_tensor_scan(dst=c[:, 0:512], data0=a[:, 0:512], data1=b[:, 0:512],
                        initial=0, op0=np.multiply, op1=np.add)

nisa.tensor_tensor_scan(dst=c[:, 512:1024], data0=a[:, 512:1024], data1=b[:, 512:1024],
                        initial=c[:, 511], op0=np.multiply, op1=np.add)""",
    }
}

kernel_insts_dict = {
    "standard": [
        "architecture",
        "nki.language.par_dim",
        "nki.language.affine_range",
        "nki.language.sequential_range",
        "nki.language.mgrid",
        "nki.language.ndarray",
        "nki.language.zeros",
        "nki.language.tile_size",
        # "nki.compiler.sbuf.mod_alloc",
        # "nki.compiler.psum.mod_alloc",
        "nki.isa.dma_copy",
        #"nki.language.load",
        #"nki.language.store",
        "nki.isa.tensor_copy",
    ],
    "gemm": [
        "nki.language.matmul",
        "nki.isa.nc_matmul",
        "nki.isa.nc_transpose",
    ],
    "layernorm": [
        "ElementWiseMath",
        "ShapeAndSelection",
        "nki.isa.bn_stats",
        "nki.isa.bn_aggr",
        "nki.isa.tensor_scalar",
    ],
    "mamba": [
        "ElementWiseMath",
        "ShapeAndSelection",
        "ActivationFunctions",
        "nki.isa.activation",
        "nki.isa.tensor_scalar",
        "nki.isa.tensor_tensor",
        "nki.isa.tensor_tensor_scan",
    ],
    "softmax": [
        "ElementWiseMath",
        "ActivationFunctions",
        "ReductionOperations",
        "ShapeAndSelection",
        "nki.isa.activation",
        "nki.isa.tensor_reduce",
        "nki.isa.tensor_scalar",
        "nki.isa.tensor_scalar_reduce",
    ],
    "causal_mask": [
        "nki.isa.affine_select",
        "nki.isa.tensor_copy_predicated",
    ],
    "rmsnorm": [
        "ElementWiseMath",              
        "ReductionOperations",          
        "ShapeAndSelection",            
        "nki.isa.tensor_scalar",        
        "nki.isa.tensor_reduce",        
        "nki.isa.bn_stats",
        "nki.isa.bn_aggr",
    ],
    "conv1d": [
        "ElementWiseMath",
        "ReductionOperations",
        "nki.isa.tensor_copy_predicated",
        "nki.isa.dma_transpose",
        "nki.isa.tensor_scalar",
        "nki.isa.tensor_tensor",
        "nki.isa.tensor_reduce",
    ],
    "maxpool": [
        "ElementWiseMath",
        "ReductionOperations",
        "ShapeAndSelection",
        "nki.isa.tensor_copy_predicated",
        "nki.isa.tensor_reduce",
        "nki.isa.select_reduce",
        "nki.isa.tensor_scalar",
        "nki.isa.local_gather",
    ],
    "conv2d": [
        "ElementWiseMath",              
        "ShapeAndSelection",            
        "ReductionOperations",         
        "nki.isa.dma_transpose",
        "nki.isa.tensor_copy_predicated", 
        "nki.isa.nc_transpose",         
        "nki.isa.nc_matmul",           
        "nki.isa.tensor_scalar",        
        "nki.isa.tensor_tensor",        
        "nki.isa.select_reduce",        
        "nki.isa.local_gather",         
        "nki.isa.activation",
    ],
    "cumsum": [
        "ElementWiseMath",                 
        "ShapeAndSelection",               
        "nki.isa.tensor_scalar",           
        "nki.isa.tensor_tensor_scan",      
    ],
    "rope": [
        "ElementWiseMath",
        "ShapeAndSelection",
        "nki.isa.tensor_copy_predicated",  
        "nki.isa.tensor_scalar",
        "nki.isa.tensor_tensor",
    ],
    "tensor_add": 
    [
        "ElementWiseMath",              
        "ShapeAndSelection",            
        "nki.isa.tensor_scalar",        
        "nki.isa.tensor_tensor",        
    ],
    "transpose": [
        "ShapeAndSelection",
        "nki.isa.tensor_copy_predicated",
        # "nki.isa.dma_transpose",
        "nki.isa.nc_transpose",
    ]
}

class NkiIsaGenerator:
    def __init__(self):
        self.isa_dict = nki_isa_dict
        self.kernel_insts_dict = kernel_insts_dict
        self.workload_to_kernel_dict = workload_to_kernel_dict
        self.prob_to_name = prob_to_name

    def generate_isa_string(self, insts: Iterable[str]):
        # First expand those that are lists into individual dictionaries
        dicts = []
        for inst in insts:
            if inst in self.isa_dict:
                if isinstance(self.isa_dict[inst], list):
                    for item in self.isa_dict[inst]:
                        dicts.append(item)
                else:
                    dicts.append(self.isa_dict[inst])
            else:
                raise ValueError(f"Instruction {inst} not found in isa_dict")

        # Then generate the isa string from the expanded dictionaries
        isa_string = ""
        for dic in dicts:
            header = dic.get("header", "")
            if header:
                isa_string += header + "\n"
            description = dic.get("description", "")
            if description:
                isa_string += description + "\n"
            examples = dic.get("examples", "")
            if examples:
                isa_string += examples + "\n"
            isa_string += "\n"
        return isa_string

    def generate_isa(self, prob_or_name: Prob | int | str):
        if isinstance(prob_or_name, Prob):
            name = self.prob_to_name[prob_or_name.prob_type][prob_or_name.prob_id]
        elif isinstance(prob_or_name, int):
            name = self.prob_to_name[prob_or_name]
        elif isinstance(prob_or_name, str):
            name = prob_or_name
        else:
            raise ValueError(f"Invalid input type: {type(prob_or_name)}")
        logger.debug(f"Generating ISA for problem type: {name}")
        kernels = self.workload_to_kernel_dict.get(name, [name]) # if not found, then <name> is a kernel
        kernels = ["standard"] + kernels # always include standard instructions
        insts = []
        seen = set()
        for kernel in kernels:
            for inst in self.kernel_insts_dict[kernel]:
                if inst not in seen:
                    insts.append(inst)
                    seen.add(inst)
        return self.generate_isa_string(insts)
