from typing import Iterable

nki_isa_dict = {
    "architecture": {
        "description": """Kernel structure: Each NKI kernel has 3 stages — load data from HBM → SBUF, compute on NeuronCore, store results from SBUF → HBM.

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
    Advanced indexing via nl.arange enables efficient striding along F dim (not P).
    List indices must be integers or slices. They cannot be computed from affine_range loop variables.
    Slice with variable size is not supported.
    Tile on SBUF/PSUM must have at least 2 dimensions as described here. If using a 1D tile on SBUF/PSUM, users may get an “Insufficient rank” error. Workaround this by creating a 2D tile, e.g.,
    buf = nl.zeros((128, ), dtype=dtype, buffer=nl.sbuf)  # this won't work
    buf = nl.zeros((128, 1), dtype=dtype, buffer=nl.sbuf) # this works
    Users must index their [N, 1] or [1, M] shaped 2D buffers with both indices, do my_sbuf[0:N, 0] or my_sbuf[0, 0:M] to access them, since accessing in 1D my_sbuf[0:N] won’t work.
    Use nl.arange for indirect load/store access indexing, nl.mgrid won’t work. See code examples in nl.load and nl.store.
    If indexing with [0, 0] gets internal errors, try using [0:1, 0:1] or nl.mgrid[0:1, 0:1] instead.
    If indexing with [0:1, ...] gets internal errors, try using [0, ...] instead.

Performance tip:
    Access HBM sequentially; only F-dim striding is hardware-efficient.

Scope rules:
    Tensors in NKI are not allowed to be used outside of their parent scope.
    Tensors in NKI have a stricter scope rules than Python. In NKI, control blocks in if/else/for statements will introduce their own scope for tensors. A tensor defined in if/else/for control blocks are not allowed to be used outside of the scope.

    for i in range(4):
    if i < 2:
        tmp = nl.load(a)
    else:
        tmp = nl.load(b)
    nl.store(c, tmp) # Error: Local variable 'tmp' is referenced outside of its parent scope ...

    To fix the problem, you can rewrite the above code as:

    for i in range(4):
    tmp = nl.ndarray(shape=a.shape, dtype=a.dtype)
    if i < 2:
        tmp[...] = nl.load(a)
    else:
        tmp[...] = nl.load(b)
    nl.store(c, tmp)

Other constraints:
    & is not supported for scalar.
""",
    },
    "ElementWiseMath": [
        {"header": "nl.add(x: tile | scalar, y: tile | scalar)", "description": "Element-wise addition. x.shape and y.shape must be broadcastable to a common shape, that will become the shape of the output.", "examples": """a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:128, 0:512])
# add a and b element-wise and store in c[128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)

a = nl.load(a_tensor[0:128, 0:512])
b = 2.2
# add constant b to each element in a
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:128, 0:1])
# broadcast on free dimension -- [128, 1] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:1, 0:512])
# broadcast on partition dimension -- [1, 512] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:1, 0:1])
# broadcast on both dimensions -- [1, 1] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)

a = nl.load(a_tensor[0:128, 0:1])
b = nl.load(b_tensor[0:1, 0:512])
# broadcast on each dimensions -- [128, 1] and [1, 512] are broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)"""},
        {"header": "nl.subtract(x: tile | scalar, y: tile | scalar)", "description": "Element-wise subtraction."},
        {"header": "nl.multiply(x: tile | scalar, y: tile | scalar)", "description": "Element-wise multiplication."},
        {"header": "nl.divide(x: tile | scalar, y: tile | scalar)", "description": "Element-wise division."},
        {"header": "nl.power(x: tile | scalar, y: tile | scalar)", "description": "Elements of x raised to powers of y, element-wise."},
        {"header": "nl.maximum(x: tile | scalar, y: tile | scalar)", "description": "Maximum of the inputs, element-wise."},
        {"header": "nl.minimum(x: tile | scalar, y: tile | scalar)", "description": "Minimum of the inputs, element-wise."},
        {"header": "nl.abs(x: tile)", "description": "Element-wise absolute value."},
        {"header": "nl.exp(x: tile)", "description": "Element-wise exponential (e**x)."},
        {"header": "nl.log(x: tile)", "description": "Element-wise natural logarithm."},
        {"header": "nl.sqrt(x: tile)", "description": "Element-wise non-negative square root."},
        {"header": "nl.rsqrt(x: tile)", "description": "Element-wise reciprocal of the square root (1/sqrt(x))."},
        {"header": "nl.square(x: tile)", "description": "Element-wise square (x*x)."},
        {"header": "nl.reciprocal(x: tile)", "description": "Element-wise reciprocal (1/x)."},
        {"header": "nl.sin(x: tile)", "description": "Element-wise sine."},
        {"header": "nl.cos(x: tile)", "description": "Element-wise cosine."},
        {"header": "nl.tanh(x: tile)", "description": "Element-wise hyperbolic tangent."},
        {"header": "nl.ceil(x: tile)", "description": "Element-wise ceiling."},
        {"header": "nl.floor(x: tile)", "description": "Element-wise floor."},
        {"header": "nl.sign(x: tile)", "description": "Element-wise sign of a number."},
        {"header": "nl.negative(x: tile)", "description": "Element-wise negative."},
        {"header": "nl.trunc(x: tile)", "description": "Element-wise truncation."},
    ],
    "ActivationFunctions": [
        {"header": "nl.relu(x: tile)", "description": "Rectified Linear Unit."},
        {"header": "nl.sigmoid(x: tile)", "description": "Sigmoid activation."},
        {"header": "nl.softmax(x: tile, axis: int|tuple)", "description": "Softmax activation along a specified axis."},
        {"header": "nl.gelu(x: tile)", "description": "Gaussian Error Linear Unit."},
        {"header": "nl.silu(x: tile)", "description": "Sigmoid Linear Unit (Swish)."}
    ],
    "ReductionOperations": [
        {"header": "nl.sum(x: tile, axis: int|tuple, keepdims=False)", "description": "Sum of elements along a specified free axis."},
        {"header": "nl.prod(x: tile, axis: int|tuple, keepdims=False)", "description": "Product of elements along the specified axis (or axes) of the input."},
        {"header": "nl.all(x: tile, axis: int|tuple, keepdims=False)", "description": "Product of elements along the specified axis (or axes) of the input."},
        {"header": "nl.max(x: tile, axis: int|tuple, keepdims=False)", "description": "Maximum of elements along a specified free axis."},
        {"header": "nl.min(x: tile, axis: int|tuple, keepdims=False)", "description": "Minimum of elements along a specified free axis."},
        {"header": "nl.mean(x: tile, axis: int|tuple, keepdims=False)", "description": "Mean of elements along a specified free axis."},
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
        {"header": "nl.where(condition: tile[bool], x: tile, y: tile|scalar)", "description": "Selects elements from x or y based on a condition."},
        {"header": "nl.broadcast_to(src: tile, *, shape: tuple=None)", "description": "Broadcasts a tile to a new shape. Returns a new tile broadcast along the partition dimension of src, this new tile will be in SBUF, but can be also assigned to a PSUM tensor."},
        {"header": "nki.tensor.broadcast_to(shape: tuple)", "description": "The tensor object must be a tile or can be implicitly converted to a tile. A tensor can be implicitly converted to a tile iff the partition dimension is the highest dimension. Returns a new view of the tile, no copy will occur."},
        {"header": "nl.expand_dims(data: tile, axis: int|tuple)", "description": "Inserts a new dimension of size 1 into the tile's shape."},
        {"header": "nl.gather_flattened(data: tile, indices: tile[uint32])", "description": "Gathers elements from data's flattened free dimension using indices."}
    ],
    "nki.language.affine_range": {
        "header": "nl.affine_range(num_iterations: int):",
        "description": """Create a sequence of numbers for use as parallel loop iterators in NKI. affine_range should be the default loop iterator choice, when there is no loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. A concrete example of associative reduction is multiple nl.matmul or nisa.nc_matmul calls accumulating into the same output buffer defined outside of this loop level (see code example #2 below).
Overlapping nl.load outputs in SBUF are considered loop dependencies and are not allowed; buffers can be allocated inside affine_range or outputs can be indexed/sliced to avoid this.
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
# No data dependency between loop instances.
for i_input in nl.affine_range(input.shape[1] // 512):
  offset = i_input * 512
  input_sb = nl.load(input[0:input.shape[0], offset:offset+512])
  result = nl.multiply(input_sb, input_sb)
  nl.store(output[0:input.shape[0], offset:offset+512], result)

# Example 2: Matmul output buffer accumulation, a type of associative reduction
# Input tensor shapes for nl.matmul: xT[K=2048, M=128] and y[K=2048, N=128]
# Load one tile ([128, 128]) from both xT and y at a time, matmul and
# accumulate into the same output buffer

result_psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
for i_K in nl.affine_range(xT.shape[0] // 128):
  offset = i_K * 128
  xT_sbuf = nl.load(offset:offset+128, 0:xT.shape[1]])
  y_sbuf = nl.load(offset:offset+128, 0:y.shape[1]])

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

  # Depends on scan result from the previous loop iteration
  result = nisa.tensor_tensor_scan(input0[:, offset:offset+512],
                                   input1[:, offset:offset+512],
                                   initial=init,
                                   op0=nl.multiply, op1=nl.add)

  nl.store(output[0:input0.shape[0], offset:offset+512], result)

  # Prepare initial result for scan in the next loop iteration
  init[:, :] = result[:, 511]""",
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
        "description": "Mark a dimension explicitly as a partition dimension.",
        "examples": """# Example 1:
identity_load = nl.ndarray((64, nl.par_dim(128)), dtype=nl.bfloat16, buffer=sb_mod(base_addr=sca))

# Example 2:
# Most outer loop with batch_size, parallel_for
for i_batch in nl.affine_range(batch_size):
    # partial accumulated scanC result with processed states
    scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)
    ...
""",
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
    "nki.isa.activation": {
        "header": "nisa.activation(op: activation_function, data: tile[SBUF|PSUM], bias: tile[vector]=None, scale: scalar|tile[vector]=1.0, reduce_op: reduce_function=None, reduce_res: tile[vector]=None, reduce_cmd: nisa.reduce_cmd=idle, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (same shape as data)",
        "examples": """# Example 1: perform exponential function on matrix a of shape (128, 1024)
a = nl.load(a_tensor)
activated_a = nisa.activation(op=nl.exp, data=a)
nl.store(a_act_tensor, activated_a)

# Example 2: perform the following operations to matrix b of shape (128, 512)
# using a single activation instruction: np.square(b * 2.0) + c
# 1) compute `np.square(b * 2.0 + c)`
# 2) cast 1) results into bfloat16
b = nl.load(b_tensor)
c = nl.load(c_tensor)
activated_b = nisa.activation(op=np.square, data=b, bias=c, scale=2.0,
                              dtype=nl.bfloat16)
nl.store(b_act_tensor, activated_b)
""",
    },
    "nki.isa.activation_reduce": {
        "header": "nisa.activation_reduce(op: activation_function, data: tile[SBUF|PSUM], reduce_op: reduce_function, reduce_res: tile[vector], bias: tile[vector]=None, scale: scalar|tile[vector]=1.0, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (same shape as data)",
        "examples": "",
    },
    "nki.isa.affine_select": {
        "header": "nisa.affine_select(pred: affine_expression, on_true_tile: tile, on_false_value: scalar, dtype: nki_dtype=on_true_tile.dtype, mask: predicate=None) -> tile (same shape as on_true_tile)",
        "examples": "",
    },
    "nki.isa.bn_aggr": {
        "header": "nisa.bn_aggr(data: tile, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (shape: [par_dim, 2])",
        "description": """Aggregate one or multiple bn_stats outputs to generate a mean and variance per partition using Vector Engine.
The input data tile effectively has an array of (count, mean, variance*count) tuples per partition produced by bn_stats instructions. Therefore, the number of elements per partition of data must be a modulo of three.
Note, if you need to aggregate multiple bn_stats instruction outputs, it is recommended to declare a SBUF tensor and then make each bn_stats instruction write its output into the SBUF tensor at different offsets (see example implementation in Example 2 in bn_stats).
Vector Engine performs the statistics aggregation in float32 precision. Therefore, the engine automatically casts the input data tile to float32 before performing float32 computation and is capable of casting the float32 computation results into another data type specified by the dtype field, at no additional performance cost. If dtype field is not specified, the instruction will cast the float32 results back to the same data type as the input data tile.

Estimated instruction cost:
max(MIN_II, 13*(N/3)) Vector Engine cycles, where N is the number of elements per partition in data and MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.

Returns:
    an output tile with two elements per partition: a mean followed by a variance
""",
    },
    "nki.isa.bn_stats": {
        "header": "nisa.bn_stats(data: tile, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (shape: [par_dim, 6])",
        "description": """Compute mean- and variance-related statistics for each partition of an input tile data in parallel using Vector Engine.
The output tile of the instruction has 6 elements per partition:
    - the count of the even elements (of the input tile elements from the same partition)
    - the mean of the even elements
    - variance * count of the even elements
    - the count of the odd elements
    - the mean of the odd elements
    - variance * count of the odd elements
To get the final mean and variance of the input tile, we need to pass the above bn_stats instruction output into the bn_aggr instruction, which will output two elements per partition:
    - mean (of the original input tile elements from the same partition)
    - variance
Due to hardware limitation, the number of elements per partition (i.e., free dimension size) of the input data must not exceed 512 (nl.tile_size.bn_stats_fmax). To calculate per-partition mean/variance of a tensor with more than 512 elements in free dimension, we can invoke bn_stats instructions on each 512-element tile and use a single bn_aggr instruction to aggregate bn_stats outputs from all the tiles. Refer to Example 2 for an example implementation.
Vector Engine performs the above statistics calculation in float32 precision. Therefore, the engine automatically casts the input data tile to float32 before performing float32 computation and is capable of casting the float32 computation results into another data type specified by the dtype field, at no additional performance cost. If dtype field is not specified, the instruction will cast the float32 results back to the same data type as the input data tile.

Estimated instruction cost:
max(MIN_II, N) Vector Engine cycles, where N is the number of elements per partition in data and MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
""",
        "examples": """# Example 1: Calculate the mean and variance for each partition
# of tile a with shape (128, 128)
a: tensor[128, 128] = nl.load(a_tensor)
stats_a: tensor[128, 6] = nisa.bn_stats(a)
mean_var_a: tensor[128, 2] = nisa.bn_aggr(stats_a)

# Extract mean and variance
mean_a = mean_var_a[:, 0]
var_a = mean_var_a[:, 1]
nl.store(mean_a_tensor, mean_a)
nl.store(var_a_tensor, var_a)

# Example 2: Calculate the mean and variance for each partition of
# tile b with shape [128, 1024]
b: tensor[128, 1024] = nl.load(b_tensor)

# Run bn_stats in two tiles because b has 1024 elements per partition,
# but bn_stats has a limitation of nl.tile_size.bn_stats_fmax
# Initialize a bn_stats output tile with shape of [128, 6*2] to
# hold outputs of two bn_stats instructions
stats_b = nl.ndarray((128, 6 * 2), dtype=nl.float32)
bn_tile = nl.tile_size.bn_stats_fmax
ix, iy = nl.mgrid[0:128, 0:bn_tile]
iz, iw = nl.mgrid[0:128, 0:6]

for i in range(1024 // bn_tile):
  stats_b[iz, i * 6 + iw] = nisa.bn_stats(b[ix, i * bn_tile + iy], dtype=nl.float32)

mean_var_b = nisa.bn_aggr(stats_b)

# Extract mean and variance
mean_b = mean_var_b[:, 0]
var_b = mean_var_b[:, 1]

nl.store(mean_b_tensor, mean_b)
nl.store(var_b_tensor, var_b)""",
    },
    "nki.isa.dma_copy": {
        "header": "nisa.dma_copy(dst: tile[HBM|SBUF], src: tile[HBM|SBUF], dst_rmw_op: rmw_op=None, oob_mode: nisa.oob_mode=error, dge_mode: nisa.dge_mode=unknown, mask: predicate=None) -> None",
        "description": "Copy data from src to dst using DMA engine. Both src and dst tiles can be in device memory (HBM) or SBUF. However, if both src and dst tiles are in SBUF, consider using nisa.tensor_copy instead for better performance.",
        "examples": """from neuronxcc.nki.typing import tensor

# Example 2: Load elements from HBM with indirect addressing. If addressing 
# results out-of-bound access, the operation will fail.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]

expr_arange = 2*nl.arange(n//2)[:, None]
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)

# Example 3: Load elements from HBM with indirect addressing. If addressing 
# results in out-of-bounds access, the operation will fail.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]

# indices are out of range on purpose to demonstrate the error
expr_arange = 3*nl.arange(n//2)[:, None] 
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)

# Example 4: Load elements from HBM with indirect addressing. If addressing 
# results in out-of-bounds access, the operation will skip indices.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n//2, 0:m]

# indices are out of range on purpose
expr_arange = 3*nl.arange(n//2)[:, None] 
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.skip)

# Example 5: Store elements to HBM with indirect addressing and with 
# read-modifed-write operation.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]

expr_arange = 2*nl.arange(n)[:, None]
inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=1, dtype=in_tensor.dtype)
nl.store(out_tensor, value=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, dst_rmw_op=np.add)

# Example 6: Store elements to HBM with indirect addressing. If indirect 
# addressing results out-of-bound access, the operation will fail.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]

expr_arange = 2*nl.arange(n)[:, None]
inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
nl.store(out_tensor, value=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)

# Example 7: Store elements to HBM with indirect addressing. If indirect 
# addressing results out-of-bounds access, the operation will skip indices.
...
n, m = in_tensor.shape
ix, iy = nl.mgrid[0:n, 0:m]

# indices are out of range on purpose to demonstrate the error
expr_arange = 3*nl.arange(n)[:, None] 
inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
nl.store(out_tensor, value=out_tile)
nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)
""",
    },
    "nki.isa.dma_transpose": {
        "header": "nisa.dma_transpose(src: tile[HBM|SBUF], axes: tuple=auto, dtype: nki_dtype=src.dtype, mask: predicate=None) -> tile (transposed)",
        "description": """Perform a transpose on input src using DMA Engine. The permutation of transpose follow the rules described below:
    For 2-d input tile, the permutation will be [1, 0]
    For 3-d input tile, the permutation will be [2, 1, 0]
    For 4-d input tile, the permutation will be [3, 1, 2, 0]
""",
    },
    "nki.isa.dropout": {
        "header": "nisa.dropout(data: tile, prob: scalar|tile[vector], dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (same shape as data)",
        "examples": "",
    },
    "nki.isa.get_nc_version": {
        "header": "nisa.get_nc_version() -> nisa.nc_version",
        "examples": "",
    },
    "nki.isa.iota": {
        "header": "nisa.iota(expr: affine_expression, dtype: nki_dtype, mask: predicate=None) -> tile[SBUF]",
        "examples": "",
    },
    "nki.isa.local_gather": {
        "header": "nisa.local_gather(src_buffer: tile[SBUF], index: tile[SBUF], num_elem_per_idx: int=1, num_valid_indices: int=None, mask: predicate=None) -> tile (gathered data)",
        "examples": "",
    },
    "nki.isa.max8": {
        "header": "nisa.max8(src: tile, dtype: nki_dtype=src.dtype, mask: predicate=None) -> tile (shape: [par_dim, 8])",
        "examples": "",
    },
    "nki.isa.memset": {
        "header": "nisa.memset(shape: tuple, value: scalar, dtype: nki_dtype, engine: nisa.engine=unknown, mask: predicate=None) -> tile",
        "examples": "",
    },
    "nki.isa.nc_find_index8": {
        "header": "nisa.nc_find_index8(data: tile, vals: tile, dtype: nki_dtype=uint32, mask: predicate=None) -> tile (shape: [par_dim, 8])",
        "examples": "",
    },
    "nki.isa.nc_match_replace8": {
        "header": "nisa.nc_match_replace8(data: tile, vals: tile, imm: scalar, dst_idx: tile=None, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (modified data tensor)",
        "examples": "",
    },
    "nki.isa.nc_matmul": {
        "header": "nisa.nc_matmul(stationary: tile[SBUF], moving: tile[SBUF], is_stationary_onezero: bool=False, is_moving_onezero: bool=False, is_transpose: bool=False, tile_position: tuple=(), tile_size: tuple=(), mask: predicate=None) -> tile[PSUM](shape:(M, N))",
        "description": """Compute stationary.T @ moving using the Tensor Engine.
Make the contraction dimension as close as possible to 128 without exceeding it. The stationary tensor should be transposed before or during the call so that the contraction dimension is in the first (K, M) position instead of (M, K).
Both stationary and moving inputs must be SBUF tiles, and the output must be a PSUM tile.
If the contraction dimension exceeds 128, accumulate multiple nc_matmul outputs into the same PSUM tile.

Cost (Tensor Engine Cycles)
if input data type is one of float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32, cost is max(min(64, N_stationary), N_moving)
if input data type is float32, cost is 4 * max(min(64, N_stationary), N_moving)
where,
    N_stationary is the number of elements per partition in stationary tile.
    N_moving is the number of elements per partition in moving tile.

Args:
    stationary – the stationary operand on SBUF; layout: (partition axis <= 128, free axis <= 128)
    moving – the moving operand on SBUF; layout: (partition axis <= 128, free axis <= 512)
    mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)
    is_stationary_onezero – hints to the compiler whether the stationary operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if stationary tile is in float32; the field has no impact for non-float32 stationary.
    is_moving_onezero – hints to the compiler if the moving operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if moving tile is in float32; the field has no impact for non-float32 moving.
    is_transpose – hints to the compiler that this is a transpose operation with moving as an identity matrix.
    tile_position – a 2D tuple (row, column) for the start PE tile position to run nc_matmul.
    tile_size – a 2D tuple (row, column) for the PE tile size to hold by nc_matmul starting from tile_position.
Returns:
    a tile on PSUM that has the result of matrix multiplication of stationary and moving tiles; layout: partition axis comes from free axis of stationary, while free axis comes from free axis of moving.
""",
        "examples": """# Example 1:
# multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
# to get matrix c in PSUM of shape (128, 512)
a_mgrid = nl.mgrid[0:128, 0:128]
b_mgrid = nl.mgrid[0:128, 0:512]
c_mgrid = nl.mgrid[0:128, 0:512]

a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])

c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])

nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)

# Example 2:
# multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
# to get matrix f in PSUM of shape (128, 512) using psum accumulation
d_mgrid = nl.mgrid[0:128, 0:128]
e_mgrid = nl.mgrid[0:128, 0:512]
f_mgrid = nl.mgrid[0:128, 0:512]

f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

for i_contract in nl.affine_range(2):
  d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
  e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
  f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
  
nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)

# Example 3:
# perform batched matrix multiplication on matrix g of shape (16, 64, 64) 
# and matrix h of shape (16, 64, 512) to get matrix i of (16, 64, 512) 
# using Tensor Engine PE tiling mode. 
g_mgrid = nl.mgrid[0:64, 0:64]
h_mgrid = nl.mgrid[0:64, 0:512]
i_mgrid = nl.mgrid[0:64, 0:512]

for i in nl.affine_range(4):
  for j in nl.affine_range(4):
    g = nl.load(g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
    h = nl.load(h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
    i_psum = nisa.nc_matmul(g, h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
    nl.store(i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], i_psum)

return c_tensor, f_tensor, i_tensor""",
    },
    "nki.isa.nc_transpose": {
        "header": "nisa.nc_transpose(data, dtype: nki_dtype=data.dtype, mask: predicate=None, engine: nisa.engine=unknown) -> tile (transposed)",
        "description": """Perform a 2D transpose between the partition axis and the free axis of input data, i.e., a PF-transpose, using Tensor or Vector Engine. If the data tile has more than one free axes, this API implicitly collapses all free axes into one axis and then performs a 2D PF-transpose.

In NeuronCore, both Tensor and Vector Engine can perform a PF-transpose, but they support different input shapes. Tensor Engine nc_transpose can handle an input tile of shape (128, 128) or smaller, while Vector Engine can handle shape (32, 32) or smaller. Therefore, when the input tile shape is (32, 32) or smaller, we have an option to run it on either engine, which is controlled by the engine field. If no engine is specified, Neuron Compiler will automatically select an engine based on the input shape. Note, similar to other Tensor Engine instructions, the Tensor Engine nc_transpose must read the input tile from SBUF and write the transposed result to PSUM. On the other hand, Vector Engine nc_transpose can read/write from/to either SBUF or PSUM.

Note, PF-transpose on Tensor Engine is done by performing a matrix multiplication between data as the stationary tensor and an identity matrix as the moving tensor. See architecture guide for more information. On NeuronCore-v2, such matmul-style transpose is not bit-accurate if the input data contains NaN/Inf. You may consider replacing NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix before calling nc_transpose(engine=nki.isa.constants.engine.tensor).
Parameters:
    data – the input tile to be transposed
    mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)
    dtype – if specified and it’s different from the data type of input tile data, an additional nki.isa.cast instruction will be inserted to cast the transposed data into the target dtype (see Supported Data Types for more information)
    engine – specify which engine to use for transpose: nki.isa.tensor_engine or nki.isa.vector_engine ; by default, the best engine will be selected for the given input tile shape
Returns:
    a tile with transposed result of input data tile

Cost (Engine Cycles)
if engine is vector, max(MIN_II, N)
if engine is tensor and assuming many back-to-back nc_transpose of the same shape, max(P, min(64, F))
where,
    N is the number of elements per partition in data.
    MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.
    P is partition axis size of data.
    F is the number of elements per partition in data.
""",
        "examples": """# Example 1: transpose tile a of shape (128, 64)
i_p_a = nl.arange(128)[:, None]
i_f_a = nl.arange(64)[None, :]
aT = nisa.nc_transpose(a[i_p_a, i_f_a])

# Example 2: transpose tile b of shape (32, 2) using Vector Engine
i_p_b = nl.arange(32)[:, None]
i_f_b = nl.arange(2)[None, :]
bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)
""",
    },
    "nki.isa.range_select": {
        "header": "nisa.range_select(on_true_tile: tile, comp_op0: comparison_op, comp_op1: comparison_op, bound0: tile[vector], bound1: tile[vector], on_false_value: scalar=fp32.min, range_start: int=0, reduce_op: reduce_function=np.amax, reduce_res: tile[vector]=None, reduce_cmd: nisa.reduce_cmd=idle, dtype: nki_dtype=on_true_tile.dtype, mask: predicate=None) -> tile (selected elements)",
        "examples": "",
    },
    "nki.isa.reciprocal": {
        "header": "nisa.reciprocal(data: tile, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (reciprocal results)",
        "examples": "",
    },
    "nki.isa.scalar_tensor_tensor": {
        "header": "nisa.scalar_tensor_tensor(data: tile, op0: binary_op, operand0: scalar|tile[vector], op1: binary_op, operand1: tile, reverse0: bool=False, reverse1: bool=False, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (computation result)",
        "examples": "",
    },
    "nki.isa.select_reduce": {
        "header": "nisa.select_reduce(dst: tile, predicate: tile, on_true: tile, on_false: scalar|tile[vector], reduce_res: tile[vector]=None, reduce_cmd: nisa.reduce_cmd=idle, reduce_op: reduce_function=np.amax, reverse_pred: bool=False, dtype: nki_dtype=on_true.dtype, mask: predicate=None) -> None",
        "examples": "",
    },
    "nki.isa.sequence_bounds": {
        "header": "nisa.sequence_bounds(segment_ids: tile, dtype: nki_dtype=segment_ids.dtype) -> tile (shape: (1, 2, N))",
        "examples": "",
    },
    "nki.isa.tensor_copy": {
        "header": "nisa.tensor_copy(src: tile[SBUF|PSUM], engine: nisa.engine=unknown, dtype: nki_dtype=src.dtype, mask: predicate=None) -> tile (copy of src)",
        "description": """Create a copy of src tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.
The output tile has the same partition axis size and also the same number of elements per partition as the input tile src.
All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior is slightly different across engines:
    Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from FP32 to the output dtype (dtype, or src.dtype if dtype is not specified). Therefore, users should be cautious with assigning this instruction to Scalar Engine when the input data type cannot be precisely cast to FP32 (e.g., INT32).
    Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.
In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details). By default, this API returns a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

Estimated instruction cost:
max(MIN_II, N) engine cycles, where N is the number of elements per partition in the input tile, and MIN_II is the minimum instruction initiation interval for small input tiles. MIN_II is roughly 64 engine cycles.""",
        "examples": """# Example 1: Copy over the tensor to another tensor using the Vector engine.
x = nl.load(in_tensor)
x_copy = nisa.tensor_copy(x, engine=nisa.vector_engine)
nl.store(out_tensor, value=x_copy)
""",
    },
    "nki.isa.tensor_copy_dynamic_dst": {
        "header": "nisa.tensor_copy_dynamic_dst(dst: tile, src: tile, engine: nisa.engine=unknown, dtype: nki_dtype=src.dtype, mask: predicate=None) -> None",
        "examples": "",
    },
    "nki.isa.tensor_copy_dynamic_src": {
        "header": "nisa.tensor_copy_dynamic_src(src: tile, engine: nisa.engine=unknown, dtype: nki_dtype=src.dtype, mask: predicate=None) -> tile (copy of src)",
        "examples": "",
    },
    "nki.isa.tensor_copy_predicated": {
        "header": "nisa.tensor_copy_predicated(src: tile|scalar, dst: tile, predicate: tile, reverse_pred: bool=False, dtype: nki_dtype=src.dtype, mask: predicate=None) -> None",
        "examples": "",
    },
    "nki.isa.tensor_partition_reduce": {
        "header": "nisa.tensor_partition_reduce(op: reduce_function, data: tile, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (reduced result)",
        "examples": "",
    },
    "nki.isa.tensor_reduce": {
        "header": "nisa.tensor_reduce(op: reduce_function, data: tile, axis: int|tuple, negate: bool=False, keepdims: bool=False, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (reduced result)",
        "examples": "",
    },
    "nki.isa.tensor_scalar": {
        "header": "nisa.tensor_scalar(data: tile, op0: binary_op, operand0: scalar|tile[vector], reverse0: bool=False, op1: binary_op=None, operand1: scalar|tile[vector]=None, reverse1: bool=False, engine: nisa.engine=unknown, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (computation result)",
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
        mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)
        engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine (only allowed for rsqrt) or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

Returns:
    an output tile of (data <op0> operand0) <op1> operand1 computation""",
        "examples": """
# Example 1: subtract 1.0 from all elements of tile a of
# shape (128, 512) and get the output tile in b
i_p = nl.arange(128)[:, None]
i_f = nl.arange(512)[None, :]

b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)


# Example 2: broadcast 1.0 into a shape of (128, 512) and subtract
# it with tile c to get output tile d
i_p = nl.arange(128)[:, None]
i_f = nl.arange(512)[None, :]

d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)


# Example 3: broadcast multiply tile e with vector f and
# then broadcast add with scalar 2.5;
# tile e has a shape of (64, 1024) and vector f has a shape of (64, 1)
i_p_ef = nl.arange(64)[:, None]
i_f_e = nl.arange(1024)[None, :]
i_f_f = nl.arange(1)[None, :]

g = nisa.tensor_scalar(e[i_p_ef, i_f_e], op0=np.multiply, operand0=f[i_p_ef, i_f_f], op1=np.add, operand1=2.5)""",
    },
    "nki.isa.tensor_scalar_reduce": {
        "header": "nisa.tensor_scalar_reduce(data: tile, op0: binary_op, operand0: scalar|tile[vector], reduce_op: reduce_function, reduce_res: tile[vector], reverse0: bool=False, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (result of tensor-scalar op)",
        "examples": "",
    },
    "nki.isa.tensor_tensor": {
        "header": "nisa.tensor_tensor(data1: tile, data2: tile, op: binary_op, engine: nisa.engine=unknown, dtype: nki_dtype=type_promotion, mask: predicate=None) -> tile (element-wise result)",
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
    mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)
    dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).
Returns:
    an output tile of the element-wise operation""",
        "examples": """# Example 1: add two tiles, a and b, of the same
# shape (128, 512) element-wise and get
# the addition result in tile c
a: tensor[128, 512] = nl.load(a_tensor)
b: tensor[128, 512] = nl.load(b_tensor)

c: tensor[128, 512] = nisa.tensor_tensor(a, b, op=nl.add)""",
    },
    "nki.isa.tensor_tensor_scan": {
        "header": "nisa.tensor_tensor_scan(data0: tile, data1: tile, initial: scalar|tile[vector], op0: binary_op, op1: binary_op, reverse0: bool=False, reverse1: bool=False, dtype: nki_dtype=type_promotion, mask: predicate=None) -> tile (scan result)",
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
        mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)
        dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
Returns:
    an output tile of the scan operation""",
        "examples": """# Example 1: scan two tiles, a and b, of the same
# shape (128, 1024) using multiply/add and get
# the scan result in tile c
c = nl.ndarray(shape=(128, 1024), dtype=nl.float32)

c[:, 0:512] = nisa.tensor_tensor_scan(a[:, 0:512], b[:, 0:512],
                                      initial=0, op0=np.multiply, op1=np.add)

c[:, 512:1024] = nisa.tensor_tensor_scan(a[:, 512:1024], b[:, 512:1024],
                                         initial=c[:, 511],
                                         op0=np.multiply, op1=np.add)""",
    }
}

kernel_insts_dict = {
    "gemm": [
        "architecture",
        "nki.language.affine_range",
        "nki.language.sequential_range",
        "nki.language.mgrid",
        "nki.language.ndarray",
        "nki.language.zeros",
        "nki.language.tile_size",
        "nki.isa.dma_copy",
        "nki.isa.nc_matmul",
        "nki.isa.nc_transpose",
        "nki.isa.tensor_copy",
    ],
    "layernorm": [
        "architecture",
        "ElementWiseMath",
        "ShapeAndSelection",
        "nki.language.affine_range",
        "nki.language.sequential_range",
        "nki.language.mgrid",
        "nki.language.ndarray",
        "nki.language.zeros",
        "nki.language.tile_size",
        "nki.isa.dma_copy",
        "nki.isa.tensor_copy",
        "nki.isa.bn_stats",
        "nki.isa.bn_aggr",
        "nki.isa.tensor_scalar",
    ],
    "mamba": [
        "architecture",
        "ElementWiseMath",
        "ShapeAndSelection",
        "ActivationFunctions",
        "nki.language.affine_range",
        "nki.language.sequential_range",
        "nki.language.mgrid",
        "nki.language.ndarray",
        "nki.language.zeros",
        "nki.language.tile_size",
        "nki.isa.activation",
        "nki.isa.dma_copy",
        "nki.isa.tensor_copy",
        "nki.isa.tensor_scalar",
        "nki.isa.tensor_tensor",
        "nki.isa.tensor_tensor_scan",
    ],
}
workload_to_kernel_dict = {
}

prob_id_to_name = {
    0: "gemm",
    1: "gemm",
    2: "layernorm",
    3: "mamba",
}

class NkiIsaGenerator:
    def __init__(self):
        self.isa_dict = nki_isa_dict
        self.kernel_insts_dict = kernel_insts_dict
        self.workload_to_kernel_dict = workload_to_kernel_dict
        self.prob_id_to_name = prob_id_to_name

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
        return isa_string

    def generate_isa(self, id_or_name: int | str):
        if isinstance(id_or_name, int):
            name = self.prob_id_to_name[id_or_name]
        else:
            name = id_or_name
        if name in self.kernel_insts_dict:
            insts = self.kernel_insts_dict[name]
        else:
            kernels = self.workload_to_kernel_dict[name]
            insts = []
            seen = set()
            for kernel in kernels:
                for inst in self.kernel_insts_dict[kernel]:
                    if inst not in seen:
                        insts.append(inst)
                        seen.add(inst)
        return self.generate_isa_string(insts)