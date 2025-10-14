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
    [TC#1] P ≤ 128 (SBUF/PSUM).
    [TC#2] PSUM F ≤ 512.
    [TC#3] Matmul: LHS F ≤ 128, RHS F ≤ 512.

Indexing:
    Standard Python-style indexing/slicing returns views.
    Indexing fewer dims (e.g. x[1]) can produce a valid Tile.
    Advanced indexing via nl.arange enables efficient striding along F dim (not P).
    List indices must be integers or slices, not computed from loop variables.

Performance tip:
    Access HBM sequentially; only F-dim striding is hardware-efficient.
""",
    },
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
        "examples": "",
    },
    "nki.isa.bn_stats": {
        "header": "nisa.bn_stats(data: tile, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (shape: [par_dim, 6])",
        "examples": "",
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
In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details). By default, this API returns a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.""",
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
        "examples": "",
    },
    "nki.isa.tensor_scalar_reduce": {
        "header": "nisa.tensor_scalar_reduce(data: tile, op0: binary_op, operand0: scalar|tile[vector], reduce_op: reduce_function, reduce_res: tile[vector], reverse0: bool=False, dtype: nki_dtype=data.dtype, mask: predicate=None) -> tile (result of tensor-scalar op)",
        "examples": "",
    },
    "nki.isa.tensor_tensor": {
        "header": "nisa.tensor_tensor(data1: tile, data2: tile, op: binary_op, engine: nisa.engine=unknown, dtype: nki_dtype=type_promotion, mask: predicate=None) -> tile (element-wise result)",
        "examples": "",
    },
    "nki.isa.tensor_tensor_scan": {
        "header": "nisa.tensor_tensor_scan(data0: tile, data1: tile, initial: scalar|tile[vector], op0: binary_op, op1: binary_op, reverse0: bool=False, reverse1: bool=False, dtype: nki_dtype=type_promotion, mask: predicate=None) -> tile (scan result)",
        "examples": "",
    }
}

class NkiIsaGenerator:
    def __init__(self):
        self.isa_dict = nki_isa_dict

    def generate_isa_string(self, insts: list[str]):
        isa_string = ""
        for inst in insts:
            header = self.isa_dict[inst].get("header", "")
            if header:
                isa_string += header + "\n"
            description = self.isa_dict[inst].get("description", "")
            if description:
                isa_string += description + "\n"
            examples = self.isa_dict[inst].get("examples", "")
            if examples:
                isa_string += examples + "\n"
        return isa_string

    def generate_gemm_isa(self):
        return self.generate_isa_string([
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
        ])