## Data Types and Constants

### nki.language.float8_e4m3

nki.language.float8_e4m3 = dtype(float8_e4m3)#
    

8-bit floating-point number (1S,4E,3M)

---

### nki.language.float8_e5m2

nki.language.float8_e5m2 = dtype(float8_e5m2)#
    

8-bit floating-point number (1S,5E,2M)

---

### nki.language.bfloat16

nki.language.bfloat16 = dtype(bfloat16)#
    

16-bit floating-point number (1S,8E,7M)

---

### nki.language.tfloat32

nki.language.tfloat32 = dtype('V4')#
    

32-bit floating-point number (1S,8E,10M)

---

### nki.language.fp32

class nki.language.fp32
    

FP32 Constants

Attributes

`min` | FP32 Bit pattern (0xff7fffff) representing the minimum (or maximum negative) FP32 value  

---

### nki.language.tile_size

class nki.language.tile_size
    

Tile size constants.

Attributes

`bn_stats_fmax` | Maximum free dimension of BN_STATS  
---|---  
`gemm_moving_fmax` | Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine.  
`gemm_stationary_fmax` | Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine.  
`pmax` | Maximum partition dimension of a tile.  
`psum_fmax` | Maximum free dimension of a tile on PSUM buffer.  
`psum_min_align` | The minimum byte alignment requirement for PSUM free dimension address.  
`sbuf_min_align` | The minimum byte alignment requirement for SBUF free dimension address.  
`total_available_sbuf_size` | The total SBUF available size  

---

### nki.language.par_dim

nki.language.par_dim = Ellipsis#
    

Mark a dimension explicitly as a partition dimension.

---

### nki.isa.nc_version

class nki.isa.nc_version(value)
    

NeuronCore version

__init__()#
    

Attributes

`gen2` | Trn1/Inf2 target  
---|---  
`gen3` | Trn2 target  
  

---

### nki.isa.get_nc_version

nki.isa.get_nc_version()
    

Returns the `nc_version` of the current target context.

---

### nki.isa.engine

class nki.isa.engine(value)
    

Neuron Device engines

Attributes

`tensor` | Tensor Engine  
---|---  
`vector` | Vector Engine  
`scalar` | Scalar Engine  
`gpsimd` | GpSIMD Engine  
`sync` | Sync Engine  
`unknown` | Unknown Engine  

## Tensor Creation and Initialization

### nki.language.ndarray

nki.language.ndarray(shape, dtype, *, buffer=None, name='', **kwargs)
    

Create a new tensor of given shape and dtype on the specified buffer.

((Similar to numpy.ndarray))

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a new tensor allocated on the buffer.

---

### nki.language.zeros

nki.language.zeros(shape, dtype, *, buffer=None, name='', **kwargs)
    

Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.

((Similar to numpy.zeros))

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a new tensor allocated on the buffer.

---

### nki.language.ones

nki.language.ones(shape, dtype, *, buffer=None, name='', **kwargs)
    

Create a new tensor of given shape and dtype on the specified buffer, filled with ones.

((Similar to numpy.ones))

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a new tensor allocated on the buffer.

---

### nki.language.full

nki.language.full(shape, fill_value, dtype, *, buffer=None, name='', **kwargs)
    

Create a new tensor of given shape and dtype on the specified buffer, filled with initial value.

((Similar to numpy.full))

Parameters:
    

  * shape – the shape of the tensor.

  * fill_value – the initial value of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a new tensor allocated on the buffer.

---

### nki.language.zeros_like

nki.language.zeros_like(a, dtype=None, *, buffer=None, name='', **kwargs)
    

Create a new tensor of zeros with the same shape and type as a given tensor.

((Similar to numpy.zeros_like))

Parameters:
    

  * a – the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a tensor of zeros with the same shape and type as a given tensor.

---

### nki.language.empty_like

nki.language.empty_like(a, dtype=None, *, buffer=None, name='', **kwargs)
    

Create a new tensor with the same shape and type as a given tensor.

((Similar to numpy.empty_like))

Parameters:
    

  * a – the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor.

Returns:
    

a tensor with the same shape and type as a given tensor.

---

### nki.language.shared_constant

nki.language.shared_constant(constant, dtype=None, **kwargs)
    

Create a new tensor filled with the data specified by data array.

Parameters:
    

constant – the constant data to be filled into a tensor

Returns:
    

a tensor which contains the constant data

---

### nki.language.shared_identity_matrix

nki.language.shared_identity_matrix(n, dtype=<class 'numpy.uint8'>, **kwargs)
    

Create a new identity tensor with specified data type.

This function has the same behavior to nki.language.shared_constant but is preferred if the constant matrix is an identity matrix. The compiler will reuse all the identity matrices of the same dtype in the graph to save space.

Parameters:
    

  * n – the number of rows(and columns) of the returned identity matrix

  * dtype – the data type of the tensor, default to be `np.uint8` (see Supported Data Types for more information).

Returns:
    

a tensor which contains the identity tensor

---

### nki.language.rand

nki.language.rand(shape, dtype=<class 'numpy.float32'>, **kwargs)
    

Generate a tile of given shape and dtype, filled with random values that are sampled from a uniform distribution between 0 and 1.

Parameters:
    

  * shape – the shape of the tile.

  * dtype – the data type of the tile (see Supported Data Types for more information).

Returns:
    

a tile with random values.

---

### nki.language.arange

nki.language.arange(*args)
    

Return contiguous values within a given interval, used for indexing a tensor to define a tile.

((Similar to numpy.arange))

arange can be called as:
    

  * `arange(stop)`: Values are generated within the half-open interval `[0, stop)` (the interval including zero, excluding stop).

  * `arange(start, stop)`: Values are generated within the half-open interval `[start, stop)` (the interval including start, excluding stop).

---

### nki.language.mgrid

# nki.language.mgrid
nki.language.mgrid = Ellipsis#
    

Same as NumPy mgrid: “An instance which returns a dense (or fleshed out) mesh-grid when indexed, so that each returned argument has the same shape. The dimensions and number of the output arrays are equal to the number of indexing dimensions.”

Complex numbers are not supported in the step length.

((Similar to numpy.mgrid))
    
    
    import neuronxcc.nki.language as nl
    ...
    
    
    i_p, i_f = nl.mgrid[0:128, 0:512]
    tile = nl.load(in_tensor[i_p, i_f])
    ...
    nl.store(out_tensor[i_p, i_f], tile)
    
    
    
    
    import neuronxcc.nki.language as nl
    ...
    
    
    grid = nl.mgrid[0:128, 0:512]
    tile = nl.load(in_tensor[grid.p, grid.x])
    ...
    nl.store(out_tensor[grid.p, grid.x], tile)

---

### nki.isa.iota

nki.isa.iota(expr, dtype, *, mask=None, **kwargs)
    

Build a constant literal in SBUF using GpSimd Engine, rather than transferring the constant literal values from the host to device.

The iota instruction takes an affine expression of `nki.language.arange()` indices as the input pattern to generate constant index values (see examples below for more explanation). The index values are computed in 32-bit integer math. The GpSimd Engine is capable of casting the integer results into any desirable data type (specified by `dtype`) before writing them back to SBUF, at no additional performance cost.

Estimated instruction cost:

`150 + N` GpSimd Engine cycles, where `N` is the number of elements per partition in the output tile.

Parameters:
    

  * expr – an input affine expression of `nki.language.arange()`

  * dtype – output data type of the generated constant literal (see Supported Data Types for more information)

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

an output tile in SBUF

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Generate tile a of 512 constant values in SBUF partition 0
    # that start at 0 and increment by 1:
    ##################################################################
    # a = [0, 1, ..., 511]
    expr_a = nl.arange(0, 512)[None, :]
    a: tensor[1, 512] = nisa.iota(expr_a, dtype=nl.int32)
    
    ##################################################################
    # Example 2: Generate tile b of 128 constant values across SBUF partitions
    # that start at 0 and increment by 1, with one value per partition:
    # b = [[0],
    #      [1],
    #      ...,
    #      [127]]
    ##################################################################
    expr_b = nl.arange(0, 128)[:, None]
    b: tensor[128, 1] = nisa.iota(expr_b, dtype=nl.int32)
    
    ##################################################################
    # Example 3: Generate tile c of 512 constant values in SBUF partition 0
    # that start at 0 and decrement by 1:
    # c = [0, -1, ..., -511]
    ##################################################################
    expr_c = expr_a * -1
    c: tensor[1, 512] = nisa.iota(expr_c, dtype=nl.int32)
    
    ##################################################################
    # Example 4: Generate tile d of 128 constant values across SBUF
    # partitions that start at 5 and increment by 2
    ##################################################################
    # d = [[5],
    #      [7],
    #      ...,
    #      [259]]
    expr_d = 5 + expr_b * 2
    d: tensor[128, 1] = nisa.iota(expr_d, dtype=nl.int32)
    
    ##################################################################
    # Example 5: Generate tile e of shape [128, 512] by
    # broadcast-add expr_a and expr_b
    # e = [[0, 1, ..., 511],
    #      [1, 2, ..., 512],
    #      ...
    #      [127, 2, ..., 638]]
    ##################################################################
    e: tensor[128, 512] = nisa.iota(expr_a + expr_b, dtype=nl.int32)
    

---

### nki.isa.memset

nki.isa.memset(shape, value, dtype, *, mask=None, engine=engine.unknown, **kwargs)
    

Initialize a tile filled with a compile-time constant value using Vector or GpSimd Engine. The shape of the tile is specified in the `shape` field and the initialized value in the `value` field. The memset instruction supports all valid NKI dtypes (see Supported Data Types).

Parameters:
    

  * shape – the shape of the output tile; layout: (partition axis, free axis). Note that memset ignores nl.par_dim() and always treats the first dimension as the partition dimension.

  * value – the constant value to initialize with

  * dtype – data type of the output tile (see Supported Data Types for more information)

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * engine – specify which engine to use for memset: `nki.isa.vector_engine` or `nki.isa.gpsimd_engine` ; `nki.isa.unknown_engine` by default, lets compiler select the best engine for the given input tile shape

Returns:
    

a tile with shape shape whose elements are initialized to value.

Estimated instruction cost:

Given `N` is the number of elements per partition in the output tile, and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

  * If the initialized value is zero and output data type is bfloat16/float16, `max(MIN_II, N/2)` Vector Engine cycles;

  * Otherwise, `max(MIN_II, N)` Vector Engine cycles

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    ##################################################################
    # Example 1: Initialize a float32 tile a of shape (128, 128)
    # with a value of 0.2
    ##################################################################
    a = nisa.memset(shape=(128, 128), value=0.2, dtype=nl.float32)
    

## Memory Allocation

### nki.language.sbuf

# nki.language.sbuf
nki.language.sbuf = Ellipsis#
    

State Buffer - Only visible to each individual kernel instance in the SPMD grid, alias of `nki.compiler.sbuf.auto_alloc()`

---

### nki.language.psum

nki.language.psum = Ellipsis#
    

PSUM - Only visible to each individual kernel instance in the SPMD grid, alias of `nki.compiler.psum.auto_alloc()`

---

### nki.compiler.sbuf.alloc

# nki.compiler.sbuf.alloc
nki.compiler.sbuf.alloc(func)
    

Allocate SBUF memory space for each logical block in a tensor using a customized allocation method.

This is one of the NKI direction allocation APIs. We recommend reading NKI Direct Allocation Developer Guide before using these APIs.

In NKI, a SBUF tensor (declared using NKI tensor creation APIs) can have three kinds of dimensions, in order: logical block(B), partition(P), and free(F). The partition and free dimensions directly map to the SBUF dimensions. Both B and F can be multi-dimensional, while P must be one-dimensional per Neuron ISA constraints. The block dimension describes how many (P, F) logical tiles this tensor has, but does not reflect the number of physical tiles being allocated.

`ncc.sbuf.alloc` should be assigned to the `buffer` field of a NKI tensor declaration API. For example,
    
    
    nki_tensor = nl.ndarray((4, 8, nl.par_dim(128), 4, 32), dtype=nl.bfloat16, buffer=ncc.sbuf.alloc(...))
    

`ncc.sbuf.alloc` allows programmers to specify the physical location of each logical tile in the tensor. The API accepts a single input `func` parameter, which is a callable object that takes in:

  1. a tuple of integers `idx` representing a logical block index,

  2. an integer `pdim_size` for the number of partitions the logical tile has, and

  3. an integer `fdim_size` for the number of bytes the logical tile has per partition.

The number of integers in `idx` must match the number of B dimensions the SBUF tensor has. For example, for the above `nki_tensor`, we expect the `idx` tuple to have two integers for a 2D block index.

`pdim_size` should match the partition dimension size of the NKI tensor exactly. `fdim_size` should be the total size of F dimension shapes of each logical tile in the tensor, multiplied by the data type size in bytes. For the above `sbuf_tensor`, `pdim_size` should be 128, and `fdim_size` should be `4*32*sizeof(nl.bfloat16) = 256` bytes.

The `func` callable must return a tuple of two integers `(start_partition, byte_addr)` indicating the physical tile location for the input logical block index. `start_partition` indicates the lowest partition the physical tile allocation starts from and must follow the these ISA rules:

  * If `64 < pdim_size <= 128`, `start_partition` must be 0

  * If `32 < pdim_size <= 64`, `start_partition` must be 0 or 64

  * If `0 < pdim_size <= 32`, `start_partition` must be one of 0/32/64/96

The `byte_addr` indicates the byte offset into each partition the physical tile starts from. On NeuronCore-v2, a valid `byte_addr` can be any integer values from 0 (inclusive) to 192KiB-16KiB=(192-16)*1024 (exclusive). 192KiB is the physical size of a SBUF partition (defined in architecture guide) and 16KiB is allocated for compiler internal usage. In addition, the `base_addr` must be aligned to `nki.language.constants.sbuf_min_align`.

Note

In current release, programmers cannot mix NKI tensor declarations using automatic allocation (`ncc.sbuf.auto_alloc()` or the PSUM variant) and direction allocation APIs (`ncc.sbuf.alloc()`, `ncc.sbuf.mod_alloc()` or the PSUM variants) in the same kernel.

Parameters:
    

func – a callable object to specify how to place the logical block in SBUF memory.


---

### nki.compiler.sbuf.mod_alloc

nki.compiler.sbuf.mod_alloc(*, base_addr, base_partition=0, num_par_tiles=(), num_free_tiles=())
    

Allocate SBUF memory space for each logical tile in a tensor through modulo allocation.

This is one of the NKI direction allocation APIs. We recommend reading NKI Direct Allocation Developer Guide before using these APIs.

This API is equivalent to calling nisa.compiler.alloc() with a callable `psum_modulo_alloc_func` as defined below.
    
    
     1from typing import Optional, Tuple
     2from functools import reduce
     3from operator import mul
     4import unittest
     5
     6def num_elms(shape):
     7  return reduce(mul, shape, 1)
     8
     9def linearize(shape, indices):
    10  return sum(i * num_elms(shape[dim+1:]) for dim, i in enumerate(indices))
    11
    12def modulo_allocate_func(base, allocate_shape, scale):
    13  def func(indices):
    14    if not allocate_shape:
    15      # default shape is always (1, 1, ...)
    16      allocate_shape_ = (1, ) * len(indices)
    17    else:
    18      allocate_shape_ = allocate_shape
    19    mod_idx = tuple(i % s for i, s in zip(indices, allocate_shape_))
    20    return linearize(shape=allocate_shape_, indices=mod_idx) * scale + base
    21  return func
    22
    23def mod_alloc(base_addr: int, *, 
    24               base_partition: Optional[int] = 0,
    25               num_par_tiles: Optional[Tuple[int, ...]] = (),
    26               num_free_tiles: Optional[Tuple[int, ...]] = ()):
    27  def sbuf_modulo_alloc_func(idx, pdim_size, fdim_size):
    28    return (modulo_allocate_func(base_partition, num_par_tiles, pdim_size)(idx),
    29          modulo_allocate_func(base_addr, num_free_tiles, fdim_size)(idx))
    30  return sbuf_modulo_alloc_func
    31
    

Here’s an example usage of this API:
    
    
    nki_tensor = nl.ndarray((4, par_dim(128), 512), dtype=nl.bfloat16,
                            buffer=nki.compiler.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))
    
    for i_block in nl.affine_range(4):
      nki_tensor[i_block, :, :] = nl.load(...)
      ...                       = nl.exp(nki_tensor[i_block, :, :])
    

This produces the following allocation:

Table 4 Modulo Allocation Example# Logical Tile Index | Physical Tile `start_partition` | Physical Tile `byte_addr`  
---|---|---  
(0, ) | 0 | 0 + (0 % 2) * 512 * sizeof(nl.bfloat16) = 0  
(1, ) | 0 | 0 + (1 % 2) * 512 * sizeof(nl.bfloat16) = 1024  
(2, ) | 0 | 0 + (2 % 2) * 512 * sizeof(nl.bfloat16) = 0  
(3, ) | 0 | 0 + (3 % 2) * 512 * sizeof(nl.bfloat16) = 1024  
  
With above scheme, we are able to implement double buffering in `nki_tensor`, such that `nl.load` in one iteration can write to one physical tile while `nl.exp` of the previous iteration can read from the other physical tile simultaneously.

Note

In current release, programmers cannot mix NKI tensor declarations using automatic allocation (`ncc.sbuf.auto_alloc()` or the PSUM variant) and direction allocation APIs (`ncc.sbuf.alloc()`, `ncc.sbuf.mod_alloc()` or the PSUM variants).

Parameters:
    

  * base_addr – the base address in the free(F) dimension of the SBUF in bytes.

  * base_partition – the partition where the physical tile starts from. Must be 0 in the current version.

  * num_par_tiles – the number of physical tiles on the partition dimension of SBUF allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor.

  * num_free_tiles – the number of physical tiles on the free dimension of SBUF allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor.

---

### nki.compiler.sbuf.auto_alloc

nki.compiler.sbuf.auto_alloc()
    

Returns a maker to indicate the tensor should be automatically allocated by compiler. All SBUF tensors in a kernel must either all be marked as `auto_alloc()`, or all be allocated with `alloc` or `mod_alloc`.

Initialize a tensor with `buffer=nl.sbuf` is equivalent to `buffer=ncc.sbuf.auto_alloc()`.

---

### nki.compiler.psum.alloc

nki.compiler.psum.alloc(func)
    

Allocate PSUM memory space for each logical block in a tensor using a customized allocation method.

This is one of the NKI direction allocation APIs. We recommend reading NKI Direct Allocation Developer Guide before using these APIs.

In NKI, a PSUM tensor (declared using NKI tensor creation APIs) can have three kinds of dimensions, in order: logical block(B), partition(P), and free(F). The partition and free dimensions directly map to the PSUM dimensions. Both B and F can be multi-dimensional, while P must be one-dimensional per Neuron ISA constraints. The block dimension describes how many (P, F) logical tiles this tensor has, but does not reflect the number of physical tiles being allocated.

`ncc.psum.alloc` should be assigned to the `buffer` field of a NKI tensor declaration API. For example,
    
    
    nki_tensor = nl.ndarray((2, 4, nl.par_dim(128), 512), dtype=nl.float32, buffer=ncc.psum.alloc(...))
    

`ncc.psum.alloc` allows programmers to specify the physical location of each logical tile in the tensor. The API accepts a single input `func` parameter, which is a callable object that takes in:

  1. a tuple of integers `idx` representing a logical block index,

  2. an integer `pdim_size` for the number of partitions the logical tile has, and

  3. an integer `fdim_size` for the number of bytes the logical tile has per partition.

The number of integers in `idx` must match the number of B dimensions the PSUM tensor has. For example, for the above `nki_tensor`, we expect the `idx` tuple to have two integers for a 2D block index.

`pdim_size` should match the partition dimension size of the NKI tensor exactly. `fdim_size` should be the total size of F dimension shapes of each logical tile in the tensor, multiplied by the data type size in bytes. For the above `nki_tensor`, `pdim_size` should be 128, and `fdim_size` should be `512*sizeof(nl.float32) = 2048` bytes.

Note

In current release, `fdim_size` cannot exceed 2KiB, which is the size of a single PSUM bank per partition. Therefore, a physical PSUM tile cannot span multiple PSUM banks. Check out Trainium/Inferentia2 Architecture Guide for NKI for more information on PSUM banks.

The `func` returns a tuple of three integers `(bank_id, start_partition, byte_addr)` indicating the physical tile location for the input logical block index.

`bank_id` indicates the PSUM bank ID of the physical tile. `start_partition` indicates the lowest partition the physical tile allocation starts from. The `byte_addr` indicates the byte offset into each PSUM bank per partition the physical tile starts from.

Note

In current release, `start_partition` and `byte_addr` must both be 0.

Note

In current release, programmers cannot mix NKI tensor declarations using automatic allocation (`ncc.psum.auto_alloc()` or the SBUF variant) and direction allocation APIs (`ncc.psum.alloc()`, `ncc.psum.mod_alloc()` or the SBUF variants) in the same kernel.

Parameters:
    

func – a callable object to specify how to place the logical block in PSUM memory.

---

### nki.compiler.psum.mod_alloc

nki.compiler.psum.mod_alloc(*, base_bank, base_addr=0, base_partition=0, num_bank_tiles=(), num_par_tiles=(), num_free_tiles=())
    

Allocate PSUM memory space for each logical block in a tensor through modulo allocation.

This is one of the NKI direction allocation APIs. We recommend reading NKI Direct Allocation Developer Guide before using these APIs.

This API is equivalent to calling nki.compiler.psum.alloc() with a callable `psum_modulo_alloc_func` as defined below.
    
    
     1from typing import Optional, Tuple
     2from functools import reduce
     3from operator import mul
     4import unittest
     5
     6def num_elems(shape):
     7  return reduce(mul, shape, 1)
     8
     9def linearize(shape, indices):
    10  return sum(i * num_elems(shape[dim+1:]) for dim, i in enumerate(indices))
    11
    12def modulo_allocate_func(base, allocate_shape, scale):
    13  def func(indices):
    14    if not allocate_shape:
    15      # default shape is always (1, 1, ...)
    16      allocate_shape_ = (1, ) * len(indices)
    17    else:
    18      allocate_shape_ = allocate_shape
    19    mod_idx = tuple(i % s for i, s in zip(indices, allocate_shape_))
    20    return linearize(shape=allocate_shape_, indices=mod_idx) * scale + base
    21  return func
    22
    23def mod_alloc(base_addr: int, *, 
    24               base_bank: Optional[int] = 0,
    25               num_bank_tiles: Optional[Tuple[int]] = (),
    26               base_partition: Optional[int] = 0,
    27               num_par_tiles: Optional[Tuple[int]] = (),
    28               num_free_tiles: Optional[Tuple[int]] = ()):
    29  def psum_modulo_alloc_func(idx, pdim_size, fdim_size):
    30    # partial bank allocation is not allowed
    31    return (modulo_allocate_func(base_bank, num_bank_tiles, 1)(idx),
    32          modulo_allocate_func(base_partition, num_par_tiles, pdim_size)(idx),
    33          modulo_allocate_func(base_addr, num_free_tiles, fdim_size)(idx))
    34  return psum_modulo_alloc_func
    35
    

Here’s an example usage of this API:
    
    
    psum_tensor = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.float32,
                             buffer=ncc.psum.mod_alloc(base_bank=0,
                                                        base_addr=0,
                                                        num_bank_tiles=(2,)))
    
    for i_block in nl.affine_range(4):
      psum[i_block, :, :] = nisa.nc_matmul(...)
      ...                 = nl.exp(psum[i_block, :, :])
    

This produces the following allocation:

Table 5 Modulo Allocation Example# Logical Tile Index | Physical Tile `bank_id` | Physical Tile `start_partition` | Physical Tile `byte_addr`  
---|---|---|---  
(0, ) | 0 | 0 | 0  
(1, ) | 1 | 0 | 0  
(2, ) | 0 | 0 | 0  
(3, ) | 1 | 0 | 0  
  
With above scheme, we are able to implement double buffering in `nki_tensor`, such that `nisa.nc_matmul` in one iteration can write to one physical tile while `nl.exp` of the previous iteration can read from the other physical tile simultaneously.

Note

In current release, programmers cannot mix NKI tensor declarations using automatic allocation (`ncc.psum.auto_alloc()` or the SBUF variant) and direction allocation APIs (`ncc.psum.alloc()`, `ncc.psum.mod_alloc()` or the SBUF variants).

Parameters:
    

  * base_addr – the base address in bytes along the free(F) dimension of the PSUM bank. Must be 0 in the current version.

  * base_bank – the base bank ID that the physical tiles start from.

  * num_bank_tiles – the number of PSUM banks allocated for the tensor.

  * base_partition – the partition ID the physical tiles start from. Must be 0 in the current version.

  * num_par_tiles – the number of physical tiles along the partition dimension allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor. Currently must be an empty tuple or (1, 1, …).

  * num_free_tiles – the number of physical tiles on the free dimension per PSUM bank allocated for the tensor. The length of the tuple must be empty or equal to the length of block dimension for the tensor. Currently must be an empty tuple or (1, 1, …).


---

### nki.compiler.psum.auto_alloc

nki.compiler.psum.auto_alloc()
    

Returns a maker to indicate the tensor should be automatically allocated by compiler. All PSUM tensors in a kernel must either all be marked as `auto_alloc()`, or all be allocated with `alloc` or `mod_alloc`.

Initialize a tensor with `buffer=nl.psum` is equivalent to `buffer=ncc.psum.auto_alloc()`.

## Memory Load/Store and DMA

### nki.language.load

nki.language.load(src, *, mask=None, dtype=None, **kwargs)
    

Load a tensor from device memory (HBM) into on-chip memory (SBUF).

See Memory hierarchy for detailed information.

Parameters:
    

  * src – HBM tensor to load the data from.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile on SBUF with values from `src`.
    
    
    import neuronxcc.nki.language as nl
    
    # load from in_tensor[P, F] that is on HBM
    # copy into data_tile[P, F] that is on SBUF
    data_tile = nl.load(in_tensor)
    ...
    

Note

Partition dimension size can’t exceed the hardware limitation of `nki.language.tile_size.pmax`, see Tile size considerations.

Partition dimension has to be the first dimension in the index tuple of a tile. Therefore, data may need to be split into multiple batches to load/store, for example:
    
    
    import neuronxcc.nki.language as nl
    
    for i_b in nl.affine_range(4):
      data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
      # load from in_tensor[4, 128, 512] one batch at a time
      # copy into data_tile[128, 512]
      i_p, i_f = nl.mgrid[0:128, 0:512]
      data_tile[i_p, i_f] = nl.load(in_tensor[i_b, i_p, i_f])
      ...
    

Also supports indirect DMA access with dynamic index values:
    
    
    import neuronxcc.nki.language as nl
    ...
    
    
    ############################################################################################
    # Indirect DMA read example 1:
    # - data_tensor on HBM has shape [128 x 512].
    # - idx_tensor on HBM has shape [64] (with values [0, 2, 4, 6, ...]).
    # - idx_tensor values read from HBM and stored in SBUF idx_tile of shape [64 x 1]
    # - data_tensor values read from HBM indexed by values in idx_tile 
    #   and store into SBUF data_tile of shape [64 x 512].
    ############################################################################################
    i_p = nl.arange(64)[:, None]
    i_f = nl.arange(512)[None, :]
    
    idx_tile = nl.load(idx_tensor[i_p]) # indices have to be in SBUF
    data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f]) 
    ...
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    
    ############################################################################################
    # Indirect DMA read example 2:
    # - data_tensor on HBM has shape [128 x 512].
    # - idx_tile on SBUF has shape [64 x 1] (with values [[0], [2], [4], ...] generated by iota)
    # - data_tensor values read from HBM indexed by values in idx_tile 
    #   and store into SBUF data_tile of shape [64 x 512].
    ############################################################################################
    i_f = nl.arange(512)[None, :]
    
    idx_expr = 2*nl.arange(64)[:, None]
    idx_tile = nisa.iota(idx_expr, dtype=np.int32)
    data_tile = nl.load(data_tensor[idx_tile, i_f]) 
    ...
    

---

### nki.language.store

nki.language.store(dst, value, *, mask=None, **kwargs)
    

Store into a tensor on device memory (HBM) from on-chip memory (SBUF).

See Memory hierarchy for detailed information.

Parameters:
    

  * dst – HBM tensor to store the data into.

  * value – An SBUF tile that contains the values to store. If the tile is in PSUM, an extra copy will be performed to move the tile to SBUF first.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    
    
    
    import neuronxcc.nki.language as nl
    
    ...
    # store into out_tensor[P, F] that is on HBM
    # from data_tile[P, F] that is on SBUF
    nl.store(out_tensor, data_tile)
    

Note

Partition dimension size can’t exceed the hardware limitation of `nki.language.tile_size.pmax`, see Tile size considerations.

Partition dimension has to be the first dimension in the index tuple of a tile. Therefore, data may need to be split into multiple batches to load/store, for example:
    
    
    import neuronxcc.nki.language as nl
    
    for i_b in nl.affine_range(4):
      data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
    
    ...
    # store into out_tensor[4, 128, 512] one batch at a time
    # from data_tile[128, 512] 
    i_p, i_f = nl.mgrid[0:128, 0:512]
    nl.store(out_tensor[i_b, i_p, i_f], value=data_tile[i_p, i_f]) 
    

Also supports indirect DMA access with dynamic index values:
    
    
    import neuronxcc.nki.language as nl
    ...
    
    
    ##################################################################################
    # Indirect DMA write example 1:
    #  - data_tensor has shape [128 x 512].
    #  - idx_tensor on HBM has shape [64] (with values [0, 2, 4, 6, ...]).
    #  - idx_tensor values read from HBM and stored in SBUF idx_tile.
    #  - data_tile of shape [64 x 512] values written into
    #    HBM data_tensor indexed by values in idx_tile.
    ##################################################################################
    i_p = nl.arange(64)[:, None]
    i_f = nl.arange(512)[None, :]
    idx_tile = nl.load(idx_tensor[i_p]) # indices have to be in SB
    
    nl.store(data_tensor[idx_tile[i_p, 0], i_f], value=data_tile[0:64, 0:512])
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    
    #############################################################################################
    # Indirect DMA write example 2:
    #  - data_tensor has shape [128 x 512].
    #  - idx_tile on SBUF has shape [64 x 1] (with values [[0], [2], [4], ...] generated by iota)
    #  - data_tile of shape [64 x 512] values written into
    #    HBM data_tensor indexed by values in idx_tile.
    #############################################################################################
    idx_expr = 2*nl.arange(64)[:, None]
    idx_tile = nisa.iota(idx_expr, dtype=np.int32)
    
    nl.store(data_tensor[idx_tile, i_f], value=data_tile[0:64, 0:512]) 
    

---

### nki.language.load_transpose2d

nki.language.load_transpose2d(src, *, mask=None, dtype=None, **kwargs)
    

Load a tensor from device memory (HBM) and 2D-transpose the data before storing into on-chip memory (SBUF).

Parameters:
    

  * src – HBM tensor to load the data from.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile on SBUF with values from `src` 2D-transposed.
    
    
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    ...
    
    
    # load from in_tensor[F, P] that is on HBM
    # transpose and copy into local_tile[P, F] that is on SBUF
    N, M = in_tensor.shape
    local_tile: tensor[M, N] = nl.load_transpose2d(in_tensor)
    ...
    

Note

Partition dimension size can’t exceed the hardware limitation of `nki.language.tile_size.pmax`, see Tile size considerations.

---

### nki.language.atomic_rmw

nki.language.atomic_rmw(dst, value, op, *, mask=None, **kwargs)
    

Perform an atomic read-modify-write operation on HBM data `dst = op(dst, value)`

Parameters:
    

  * dst – HBM tensor with subscripts, only supports indirect dynamic indexing currently.

  * value – tile or scalar value that is the operand to `op`.

  * op – atomic operation to perform, only supports `np.add` currently.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    
    
    
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    ...
    
    value: tensor[N, M] = nl.load(value_tensor)
    
    # dynamic indices have to be in SBUF, with shape [N, 1]
    indices_tile: tensor[N, 1] = nl.load(indices_tensor)
    
    ix = nl.arange(M)[None, :]
    
    ########################################################################
    # Atomic read-modify-write example:
    #   - read: values of rmw_tensor is indexed by values from indices_tile
    #   - modify: incremented by value
    #   - write: saved back into rmw_tensor
    # resulting in rmw_tensor = rmw_tensor + value
    ########################################################################
    nl.atomic_rmw(rmw_tensor[indices_tile, ix], value=value, op=np.add)
    

---

### nki.isa.dma_copy

nki.isa.dma_copy(*, dst, src, mask=None, dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown)
    

Copy data from `src` to `dst` using DMA engine. Both `src` and `dst` tiles can be in device memory (HBM) or SBUF. However, if both `src` and `dst` tiles are in SBUF, consider using nisa.tensor_copy instead for better performance.

Parameters:
    

  * src – the source of copy.

  * dst – the dst of copy.

  * dst_rmw_op – the read-modify-write operation to be performed at the destination. Currently only `np.add` is supported, which adds the source data to the existing destination data. If `None`, the source data directly overwrites the destination. If `dst_rmw_op` is specified, only `oob_mode=oob_mode.error` is allowed. For best performance with Descriptor Generation Engine (DGE), unique dynamic offsets must be used to access `dst`. Multiple accesses to the same offset will cause a data hazard. If duplicated offsets are present, the compiler automatically adds synchronization to avoid hazards, which slows down computation.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * mode – 

(optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

    * `oob_mode.error`: (Default) Raises an error when encountering out-of-bounds indices.

    * `oob_mode.skip`: Silently skips any operations involving out-of-bounds indices.

For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for copy: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). HWDGE is only supported for NeuronCore-v3+.

A cast will happen if the `src` and `dst` have different dtype.

Example:
    
    
    import neuronxcc.nki.isa as nisa
    
    ############################################################################
    # Example 1: Copy over the tensor to another tensor
    ############################################################################
    nisa.dma_copy(dst=b, src=a)
    
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 2: Load elements from HBM with indirect addressing. If addressing 
    # results out-of-bound access, the operation will fail.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n//2, 0:m]
    
    expr_arange = 2*nl.arange(n//2)[:, None]
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
    nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 3: Load elements from HBM with indirect addressing. If addressing 
    # results in out-of-bounds access, the operation will fail.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n//2, 0:m]
    
    # indices are out of range on purpose to demonstrate the error
    expr_arange = 3*nl.arange(n//2)[:, None] 
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
    nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)
    
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 4: Load elements from HBM with indirect addressing. If addressing 
    # results in out-of-bounds access, the operation will skip indices.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n//2, 0:m]
    
    # indices are out of range on purpose
    expr_arange = 3*nl.arange(n//2)[:, None] 
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
    nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.skip)
    
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 5: Store elements to HBM with indirect addressing and with 
    # read-modifed-write operation.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n, 0:m]
    
    expr_arange = 2*nl.arange(n)[:, None]
    inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=1, dtype=in_tensor.dtype)
    nl.store(out_tensor, value=out_tile)
    nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, dst_rmw_op=np.add)
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 6: Store elements to HBM with indirect addressing. If indirect 
    # addressing results out-of-bound access, the operation will fail.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n, 0:m]
    
    expr_arange = 2*nl.arange(n)[:, None]
    inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
    nl.store(out_tensor, value=out_tile)
    nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ############################################################################
    # Example 7: Store elements to HBM with indirect addressing. If indirect 
    # addressing results out-of-bounds access, the operation will skip indices.
    ############################################################################
    
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
    
    
    
    
    ############################################################################
    # Example 8: Store elements to HBM with indirect addressing. If indirect 
    # addressing results out-of-bounds access, the operation will skip indices.
    ############################################################################
    
    ...
    n, m = in_tensor.shape
    ix, iy = nl.mgrid[0:n, 0:m]
    
    # indices are out of range on purpose
    expr_arange = 3*nl.arange(n)[:, None] 
    inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
    idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)
    
    out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
    nl.store(out_tensor, value=out_tile)
    nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.skip)
    

---

### nki.isa.dma_transpose

nki.isa.dma_transpose(src, *, axes=None, mask=None, dtype=None, **kwargs)
    

Perform a transpose on input `src` using DMA Engine.

The permutation of transpose follow the rules described below:

  1. For 2-d input tile, the permutation will be [1, 0]

  2. For 3-d input tile, the permutation will be [2, 1, 0]

  3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

Parameters:
    

  * src – the source of transpose, must be a tile in HBM or SBUF.

  * axes – transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source. Supported axes are `(1, 0)`, `(2, 1, 0)`, and `(3, 1, 2, 0)`.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for copy: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). HWDGE is only supported for NeuronCore-v3+.

Returns:
    

a tile with transposed content

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ############################################################################
    # Example 1: Simple 2D transpose (HBM->SB)
    ############################################################################
    def nki_dma_transpose_2d_hbm2sb(a):
      b = nisa.dma_transpose(a)
      return b
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ############################################################################
    # Example 2: Simple 2D transpose (SB->SB)
    ############################################################################
    def nki_dma_transpose_2d_sb2sb(a):
      a_sb = nl.load(a)
      b = nisa.dma_transpose(a_sb)
      return b
    

## Indexing and Slicing

### nki.language.ds

nki.language.ds(start, size)
    

Construct a dynamic slice for simple tensor indexing.
    
    
    import neuronxcc.nki.language as nl
    ...
    
    
    
    @nki.jit(mode="simulation")
    def example_kernel(in_tensor):
      out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                              buffer=nl.shared_hbm)
      for i in nl.affine_range(in_tensor.shape[1] // 512):
        tile = nl.load(in_tensor[:, (i * 512):((i + 1) * 512)])
        # Same as above but use ds (dynamic slice) instead of the native
        # slice syntax
        tile = nl.load(in_tensor[:, nl.ds(i * 512, 512)])

---

### nki.language.gather_flattened

nki.language.gather_flattened(data, indices, *, mask=None, dtype=None, **kwargs)
    

Gather elements from `data` according to the `indices`.

This instruction gathers elements from the `data` tensor using integer indices provided in the `indices` tensor. For each element in the `indices` tensor, it retrieves the corresponding value from the `data` tensor using the index value to select from the free dimension of `data`. The gather instruction effectively performs up to 128 parallel gather operations, with each operation using the corresponding partition of `data` and `indices`.

The output tensor has the same shape as the `indices` tensor, with each output element containing the value from `data` at the position specified by the corresponding index. Out of bounds indices will return garbage values.

Both `data` and `indices` must be 2-, 3-, or 4-dimensional. The `indices` tensor must contain uint32 values.

For indexing purposes, all free dimensions are flattened and indexed as the same “row”. Consider this example:
    
    
    data =
    [[[1., 2.],
     [3., 4.]],
    [[5., 6.],
     [7., 8.]]]
    indices =
    [[[0, 1],
      [1, 3]],
     [[3, 3],
      [1, 0]]]
    nl.gather_flattened(data, indices) produces this result:
    [[[1., 2.],
      [2., 4.]],
     [[8., 8.],
      [6., 5.]]]
    

With the exception of handling out-of-bounds indices, this behavior is equivalent to:
    
    
    indices_flattened = indices.reshape(indices.shape[0], -1)
    data_flattened = data.reshape(data.shape[0], -1)
    result = np.take_along_axis(data_flattened, indices_flattened, axis=-1)
    result.reshape(indices.shape)
    

((Similar to torch.gather))

Parameters:
    

  * data – the source tensor to gather values from

  * indices – tensor containing uint32 indices to gather across the flattened free dimension.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tensor with the same shape as indices containing gathered values from data

Example:
    
    
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Gather values from a tensor using indices
    ##################################################################
    # Create source tensor
    N = 32
    M = 64
    data = nl.rand((N, M), dtype=nl.float32)
    
    # Create indices tensor - gather every 5th element
    indices = nl.zeros((N, 10), dtype=nl.uint32)
    for i in nl.static_range(N):
        for j in nl.static_range(10):
            indices[i, j] = j * 5
    
    # Gather values from data according to indices
    result = nl.gather_flattened(data=data, indices=indices)
    

---

### nki.isa.local_gather

nki.isa.local_gather(src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, *, mask=None)
    

Gather SBUF data in `src_buffer` using `index` on GpSimd Engine.

Each of the eight GpSimd cores in GpSimd Engine connects to 16 contiguous SBUF partitions (e.g., core[0] connected to partition[0:16]) and performs gather from the connected 16 SBUF partitions independently in parallel. The indices used for gather on each core should also come from the same 16 connected SBUF partitions.

During execution of the instruction, each GpSimd core reads a 16-partition slice from `index`, flattens all indices into a 1D array `indices_1d` (along the partition dimension first). By default with no `num_valid_indices` specified, each GpSimd core will treat all indices from its corresponding 16-partition `index` slice as valid indices. However, when the number of valid indices per core is not a multiple of 16, users can explicitly specify the valid index count per core in `num_valid_indices`. Note, `num_valid_indices` must not exceed the total element count in each 16-partition `index` slice (i.e., `num_valid_indices <= index.size / (index.shape[0] / 16)`).

Next, each GpSimd core uses the flattened `indices_1d` indices as partition offsets to gather from the connected 16-partition slice of `src_buffer`. Optionally, this API also allows gathering of multiple contiguous elements starting at each index to improve gather throughput, as indicated by `num_elem_per_idx`. Behavior of out-of-bound index access is undefined.

Even though all eight GpSimd cores can gather with completely different indices, a common use case for this API is to make all cores gather with the same set of indices (i.e., partition offsets). In this case, users can generate indices into 16 partitions, replicate them eight times to 128 partitions and then feed them into `local_gather`.

As an example, if `src_buffer` is (128, 512) in shape and `index` is (128, 4) in shape, where the partition dimension size is 128, `local_gather` effectively performs the following operation:
    
    
    num_gpsimd_cores = 8
    num_partitions_per_core = 16
    
    src_buffer = np.random.random_sample([128, 512, 4]).astype(np.float32) * 100
    index_per_core = np.random.randint(low=0, high=512, size=(16, 4), dtype=np.uint16)
    # replicate 8 times for 8 GpSimd cores
    index = np.tile(index_per_core, (num_gpsimd_cores, 1))
    num_elem_per_idx = 4
    index_hw = index * num_elem_per_idx
    num_valid_indices = 64
    output_shape = (128, 4, 16, 4)
    
    num_active_cores = index.shape[0] / num_partitions_per_core
    num_valid_indices = num_valid_indices if num_valid_indices \
      else index.size / num_active_cores
    
    output_np = np.ndarray(shape=(128, num_valid_indices, num_elem_per_idx),
                           dtype=src_buffer.dtype)
    
    for i_core in range(num_gpsimd_cores):
      start_par = i_core * num_partitions_per_core
      end_par = (i_core + 1) * num_partitions_per_core
      indices_1d = index[start_par:end_par].flatten(order='F')[0: num_valid_indices]
    
      output_np[start_par:end_par, :, :] = np.take(
        src_buffer[start_par:end_par],
        indices_1d, axis=1)
    
    output_np = output_np.reshape(output_shape)
    

`local_gather` preserves the input data types from `src_buffer` in the gather output. Therefore, no data type casting is allowed in this API. The indices in `index` tile must be uint16 types.

This API has three tile size constraints [subject to future relaxation]:

  1. The partition axis size of `src_buffer` must match that of `index` and must be a multiple of 16. In other words, `src_buffer.shape[0] == index.shape[0] and src_buffer.shape[0] % 16 == 0`.

  2. The number of contiguous elements to gather per index per partition `num_elem_per_idx` must be one of the following values: `[1, 2, 4, 8, 16, 32]`.

  3. The number of indices for gather per core must be less than or equal to 4096.

Estimated instruction cost:

`150 + (num_valid_indices * num_elem_per_idx)/C` GpSimd Engine cycles, where `C` can be calculated using `((28 + t * num_elem_per_idx)/(t * num_elem_per_idx)) / min(4/dtype_size, num_elem_per_idx)`. `dtype_size` is the size of `src_buffer.dtype` in bytes. Currently, `t` is a constant 4, but subject to change in future software implementation.

Parameters:
    

  * src_buffer – an input tile for gathering.

  * index – an input tile with indices used for gathering.

  * num_elem_per_idx – an optional integer value to read multiple contiguous elements per index per partition; default is 1.

  * num_valid_indices – an optional integer value to specify the number of valid indices per GpSimd core; default is `index.size / (index.shape[0] / 16)`.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

an output tile of the gathered data

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    
    
    ##################################################################
    # Example 1: gather src_buffer using index
    # Gather input: src_buffer_tile with shape (128, 512, 4)
    # Gather indices: index_tile with shape (128, 4)
    # We use num_valid_indices indices per core, and read num_elem_per_idx
    # contiguous elements per partition.
    ##################################################################
    src_buffer_tile: tensor[128, 512, 4] = nl.load(src_buffer)
    index_tile: tensor[128, 4] = nl.load(index)
    output_tile: tensor[128, 4, 16, 4] = nisa.local_gather(
      src_buffer_tile, index_tile, num_elem_per_idx, num_valid_indices)
    
    nl.store(output, output_tile)
    

Click `here` to download the full NKI code example with equivalent numpy implementation.

## Tensor Copy and Movement

### nki.language.copy

nki.language.copy(src, *, mask=None, dtype=None, **kwargs)
    

Create a copy of the src tile.

Parameters:
    

  * src – the source of copy, must be a tile in SBUF or PSUM.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile with the same layout as src, this new tile will be in SBUF, but can be also assigned to a PSUM tensor.

---

### nki.isa.tensor_copy

nki.isa.tensor_copy(src, *, mask=None, dtype=None, engine=engine.unknown, **kwargs)
    

Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

The output tile has the same partition axis size and also the same number of elements per partition as the input tile `src`.

All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior is slightly different across engines:

  * Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from FP32 to the output dtype (`dtype`, or src.dtype if `dtype` is not specified). Therefore, users should be cautious with assigning this instruction to Scalar Engine when the input data type cannot be precisely cast to FP32 (e.g., INT32).

  * Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.

In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details). By default, this API returns a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

Estimated instruction cost:

`max(MIN_II, N)` engine cycles, where `N` is the number of elements per partition in the input tile, and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * src – the source of copy, must be a tile in SBUF or PSUM.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, compiler selects best engine based on engine workload).

Returns:
    

a tile with the same content and partition axis size as the `src` tile.

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    
    ############################################################################
    # Example 1: Copy over the tensor to another tensor using the Vector engine.
    ############################################################################
    x = nl.load(in_tensor)
    x_copy = nisa.tensor_copy(x, engine=nisa.vector_engine)
    nl.store(out_tensor, value=x_copy)
    

---

### nki.isa.tensor_copy_dynamic_src

nki.isa.tensor_copy_dynamic_src(src, *, mask=None, dtype=None, engine=engine.unknown, **kwargs)
    

Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine, with `src` located at a dynamic offset within each partition.

Both source and destination tiles can be in either SBUF or PSUM. By default, this API returns a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

The source and destination tiles must also have the same number of partitions and the same number of elements per partition.

The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets for gathering tiles in SBUF/PSUM, you may loop over each offset and call `tensor_copy_dynamic_src` once per offset.

Estimated instruction cost:

`max(MIN_II_DYNAMIC, N)` engine cycles, where:

  * `N` is the number of elements per partition in the `src` tile,

  * `MIN_II_DYNAMIC` is the minimum instruction initiation interval for instructions with dynamic source location. `MIN_II_DYNAMIC` is roughly 600 engine cycles.

Parameters:
    

  * src – the source of copy, must be a tile in SBUF or PSUM that is dynamically indexed within each partition.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine, nki.isa.scalar_engine or nki.isa.unknown_engine (default, let compiler select best engine).

  * return – the modified destination of copy.

Example:
    
    
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    
    #########################################################################################
    # TensorCopyDynamicSrc example 0:
    # - src_tensor in HBM of shape [128, 512]
    # - offsets in HBM of shape [1, 64] (with values [4, 5, 6, 7, ...])
    # - Gather tiles of shape [128, 1] from src_tensor into out_tensor using offsets
    #########################################################################################
    
    # Load src_tensor and offsets into SBUF
    src_tensor_sbuf: nt.tensor[128, 512] = nl.load(src_tensor)
    offsets_sbuf: nt.tensor[1, 64] = nl.load(offsets)
    
    # Copy into output tensor in SBUF
    out_sbuf: nt.tensor[128, 64] = nl.ndarray([128, 64], dtype=src_tensor.dtype,
                                              buffer=nl.sbuf)
    
    # Static indices to access a tile of shape [128, 1];
    # Add dynamic offsets to iy for tensor_copy_dynamic_src
    ix, iy = nl.mgrid[0:128, 0:1]
    
    for idx in nl.affine_range(offsets_sbuf.shape[1]):
      out_sbuf[ix, idx] = nisa.tensor_copy_dynamic_src(
          src_tensor_sbuf[ix, offsets_sbuf[0, idx] + iy])
    
    nl.store(out_tensor, value=out_sbuf)
    ...
    
    
    
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    
    #########################################################################################
    # TensorCopyDynamicSrc example 1:
    # - src_tensor in HBM of shape [128, 512, 4]
    # - offsets in HBM of shape [1 x 8] (with values [4, 5, 6, 7, ...]) to index into
    #   second axis of src_tensor
    # - Gather tiles of shape [128, 4] from src_tensor into out_tensor using offsets
    #########################################################################################
    
    # Load src_tensor and offsets into SBUF
    src_tensor_sbuf: nt.tensor[128, 512, 4] = nl.load(src_tensor)
    offsets_sbuf: nt.tensor[1, 8] = nl.load(offsets)
    
    # Copy into output tensor in SBUF
    out_sbuf: nt.tensor[128, 8, 4] = nl.ndarray([128, 8, 4], dtype=src_tensor.dtype,
                                                buffer=nl.sbuf)
    
    # Static indices to access a tile of shape [128, 1, 4];
    # Use dynamic offsets directly to index the second axis for tensor_copy_dynamic_src
    ix, _, iz = nl.mgrid[0:128, 0:1, 0:4]
    
    for idx in nl.affine_range(offsets.shape[1]):
      out_sbuf[ix, idx, iz] = nisa.tensor_copy_dynamic_src(
          src_tensor_sbuf[ix, offsets_sbuf[0, idx], iz])
    
    nl.store(out_tensor, value=out_sbuf)
    ...

---

### nki.isa.tensor_copy_dynamic_dst

nki.isa.tensor_copy_dynamic_dst(*, dst, src, mask=None, dtype=None, engine=engine.unknown, **kwargs)
    

Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine, with `dst` located at a dynamic offset within each partition.

Both source and destination tiles can be in either SBUF or PSUM.

The source and destination tiles must also have the same number of partitions and the same number of elements per partition.

The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets for scattering tiles in SBUF/PSUM, you may loop over each offset and call `tensor_copy_dynamic_dst` once per offset.

Estimated instruction cost:

`max(MIN_II_DYNAMIC, N)` engine cycles, where:

  * `N` is the number of elements per partition in the `src` tile,

  * `MIN_II_DYNAMIC` is the minimum instruction initiation interval for instructions with dynamic destination location. `MIN_II_DYNAMIC` is roughly 600 engine cycles.

Parameters:
    

  * dst – the destination of copy, must be a tile in SBUF of PSUM that is dynamically indexed within each dimension.

  * src – the source of copy, must be a tile in SBUF or PSUM.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine, nki.isa.scalar_engine or nki.isa.unknown_engine (default, let compiler select best engine).


---

### nki.isa.tensor_copy_predicated

nki.isa.tensor_copy_predicated(*, src, dst, predicate, reverse_pred=False, mask=None, dtype=None, **kwargs)
    

Conditionally copy elements from the `src` tile to the destination tile on SBUF / PSUM based on a `predicate` using Vector Engine.

This instruction provides low-level control over conditional data movement on NeuronCores, optimized for scenarios where only selective copying of elements is needed. Either `src` or `predicate` may be in PSUM, but not both simultaneously. Both `src` and `predicate` are permitted to be in SBUF.

Shape and data type constraints:

  1. `src` (if it is a tensor), `dst`, and `predicate` must occupy the same number of partitions and same number of elements per partition.

  2. `predicate` must be of type `uint8`, `uint16`, or `uint32`.

  3. `src` and `dst` must share the same data type.

Behavior:

  * Where predicate is True: The corresponding elements from src are copied to dst tile. If src is a scalar, the scalar is copied to the dst tile.

  * Where predicate is False: The corresponding values in dst tile are unmodified

Estimated instruction cost:

Cost `(Vector Engine Cycles)` | Condition  
---|---  
`max(MIN_II, N)` | If `src` is from SBUF and `predicate` is from PSUM or the other way around  
`max(MIN_II, 2N)` | If both `src` and `dst` are in SBUF  
  
  * `N` is the number of elements per partition in `src` tile

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * src – The source tile or number to copy elements from when `predicate` is True

  * dst – The destination tile to copy elements to

  * predicate – A tile that determines which elements to copy

  * reverse_pred – A boolean that reverses the effect of `predicate`.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Conditionally copies elements from the `on_true` tile to 
    # SBUF/PSUM destination tile using Vector Engine, where copying occurs 
    # only at positions where the predicate evaluates to True.
    ##################################################################
    
    ...
    pre_tile: tensor[128, 512] = nl.load(predicate)
    src_tile: tensor[128, 512] = nl.load(on_true_tensor)
    
    ix, iy = nl.mgrid[0:128, 0:512]
    dst_tile: tensor[128, 512] = nl.zeros(shape=src_tile.shape, dtype=src_tile.dtype)
    dst_tile[ix, iy] = nl.load(on_false_tensor)
    
    nisa.tensor_copy_predicated(src=src_tile, dst=dst_tile, predicate=pre_tile)
    

---

### nki.isa.nc_stream_shuffle

nki.isa.nc_stream_shuffle(src, dst, shuffle_mask, *, dtype=None, mask=None, **kwargs)
    

Apply cross-partition data movement within a quadrant of 32 partitions from source tile `src` to destination tile `dst` using Vector Engine.

Both source and destination tiles can be in either SBUF or PSUM, and passed in by reference as arguments. In-place shuffle is allowed, i.e., `dst` same as `src`. `shuffle_mask` is a 32-element list. Each mask element must be in data type int or affine expression. `shuffle_mask[i]` indicates which input partition the output partition [i] copies from within each 32-partition quadrant. The special value `shuffle_mask[i]=255` means the output tensor in partition [i] will be unmodified. `nc_stream_shuffle` can be applied to multiple of quadrants. In the case with more than one quadrant, the shuffle is applied to each quadrant independently, and the same `shuffle_mask` is used for each quadrant. `mask` applies to `dst`, meaning that locations masked out by `mask` will be unmodified. For more information about the cross-partition data movement, see Cross-partition Data Movement.

This API has 3 constraints on `src` and `dst`:

  1. `dst` must have same data type as `src`.

  2. `dst` must have the same number of elements per partition as `src`.

  3. The access start partition of `src` (`src_start_partition`), does not have to match or be in the same quadrant as that of `dst` (`dst_start_partition`). However, `src_start_partition`/`dst_start_partition` needs to follow some special hardware rules with the number of active partitions `num_active_partitions`. `num_active_partitions = ceil(max(src_num_partitions, dst_num_partitions)/32) * 32`, where `src_num_partitions` and `dst_num_partitions` refer to the number of partitions the `src` and `dst` tensors access respectively. `src_start_partition`/`dst_start_partition` is constrained based on the value of `num_active_partitions`:

>   * If `num_active_partitions` is 96/128, `src_start_partition`/`dst_start_partition` must be 0.
> 
>   * If `num_active_partitions` is 64, `src_start_partition`/`dst_start_partition` must be 0/64.
> 
>   * If `num_active_partitions` is 32, `src_start_partition`/`dst_start_partition` must be 0/32/64/96.
> 
> 

Estimated instruction cost:

`max(MIN_II, N)` Vector Engine cycles, where `N` is the number of elements per partition in `src`, and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * src – the source tile

  * dst – the destination tile

  * shuffle_mask – a 32-element list that specifies the shuffle source and destination partition

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    #####################################################################
    # Example 1: 
    # Apply cross-partition data movement to a 32-partition tensor,
    # in-place shuffling the data in partition[i] to partition[(i+1)%32].
    #####################################################################
    
    ...
    a: tensor[32, 128] = nl.load(in_tensor)
    a_mgrid = nl.mgrid[0:32, 0:128]
    shuffle_mask = [(i - 1) % 32 for i in range(32)]
    nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
    
    nl.store(out_tensor, value=a)
    
    
    
    #####################################################################
    # Example 2: 
    # Broadcast data in 1 partition to 32 partitions.
    #####################################################################
    
    ...
    a: tensor[1, 128] = nl.load(in_tensor)
    b = nl.ndarray(shape=(32, 128), dtype=np.float32)
    dst_mgrid = nl.mgrid[0:32, 0:128]
    src_mgrid = nl.mgrid[0:1, 0:128]
    shuffle_mask = [0] * 32
    nisa.nc_stream_shuffle(src=a[0, src_mgrid.x], dst=b[dst_mgrid.p, dst_mgrid.x], shuffle_mask=shuffle_mask)
    
    nl.store(out_tensor, value=b)
    
    
    
    #####################################################################
    # Example 3: 
    # In the case where src and dst access more than one quadrant (32 
    # partitions), the shuffle is applied to each quadrant independently, 
    # and the same shuffle_mask is used for each quadrant.
    #####################################################################
    
    ...
    a: tensor[128, 128] = nl.load(in_tensor)
    b = nl.ndarray(shape=(128, 128), dtype=np.float32)
    mgrid = nl.mgrid[0:128, 0:128]
    shuffle_mask = [(i - 1) % 32 for i in range(32)]
    nisa.nc_stream_shuffle(src=a[mgrid.p, mgrid.x], dst=b[mgrid.p, mgrid.x], shuffle_mask=shuffle_mask)
    
    nl.store(out_tensor, value=b)
    

## Shape Manipulation

### nki.language.broadcast_to

nki.language.broadcast_to(src, *, shape, **kwargs)
    

Broadcast the `src` tile to a new shape based on numpy broadcast rules. The `src` may also be a tensor object which may be implicitly converted to a tile. A tensor can be implicitly converted to a tile if the partition dimension is the outermost dimension. If `src.shape` is already the same as `shape`, this operation will simply return `src`.

Parameters:
    

  * src – the source of broadcast, a tile in SBUF or PSUM. May also be a tensor object.

  * shape – the target shape for broadcasting.

Returns:
    

a new tile broadcast along the partition dimension of `src`, this new tile will be in SBUF, but can be also assigned to a PSUM tensor.
    
    
    import neuronxcc.nki.language as nl
    
    ##################################################################
    # Example 1: Load from in_tensor[P, F] that is on HBM and
    # copy into out_tile[P, F] that is on SBUF by broadcasting
    ##################################################################
    ...
    
    ...
    # broadcast into out_tile[P, F] that is on SBUF
    # from data_tile[P, F] that is on SBUF
    in_tile = nl.load(in_tensor, dtype=in_tensor.dtype)
    out_tile = nl.broadcast_to(in_tile, shape=(128, in_tensor.shape[1]))
    
    # store output
    nl.store(out_tensor, out_tile)
    

---

### nki.language.expand_dims

nki.language.expand_dims(data, axis)
    

Expand the shape of a tile. Insert a new axis that will appear at the `axis` position in the expanded tile shape. Currently only supports expanding dimensions after the last index of the tile.

((Similar to numpy.expand_dims))

Parameters:
    

  * data – a tile input

  * axis – int or tuple/list of ints. Position in the expanded axes where the new axis (or axes) is placed; must be free dimensions, not partition dimension (0); Currently only supports axis (or axes) after the last index.

Returns:
    

a tile with view of input `data` with the number of dimensions increased.

---

### nki.language.transpose

nki.language.transpose(x, *, dtype=None, mask=None, **kwargs)
    

Transposes a 2D tile between its partition and free dimension.

Parameters:
    

  * x – 2D input tile

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has the values of the input tile with its partition and free dimensions swapped.

---

### nki.isa.nc_transpose

nki.isa.nc_transpose(data, *, mask=None, dtype=None, engine=engine.unknown, **kwargs)
    

Perform a 2D transpose between the partition axis and the free axis of input `data`, i.e., a PF-transpose, using Tensor or Vector Engine. If the `data` tile has more than one free axes, this API implicitly collapses all free axes into one axis and then performs a 2D PF-transpose.

In NeuronCore, both Tensor and Vector Engine can perform a PF-transpose, but they support different input shapes. Tensor Engine `nc_transpose` can handle an input tile of shape (128, 128) or smaller, while Vector Engine can handle shape (32, 32) or smaller. Therefore, when the input tile shape is (32, 32) or smaller, we have an option to run it on either engine, which is controlled by the `engine` field. If no `engine` is specified, Neuron Compiler will automatically select an engine based on the input shape. Note, similar to other Tensor Engine instructions, the Tensor Engine `nc_transpose` must read the input tile from SBUF and write the transposed result to PSUM. On the other hand, Vector Engine `nc_transpose` can read/write from/to either SBUF or PSUM.

Note, PF-transpose on Tensor Engine is done by performing a matrix multiplication between `data` as the stationary tensor and an identity matrix as the moving tensor. See architecture guide for more information. On NeuronCore-v2, such matmul-style transpose is not bit-accurate if the input `data` contains NaN/Inf. You may consider replacing NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix before calling `nc_transpose(engine=nki.isa.constants.engine.tensor)`.

Estimated instruction cost:

Cost (Engine Cycles) | Condition  
---|---  
`max(MIN_II, N)` | `engine` set to `nki.isa.constants.engine.vector`  
`max(P, min(64, F))` | `engine` set to `nki.isa.constants.engine.tensor` and assuming many back-to-back `nc_transpose` of the same shape on Tensor Engine  
  
where,

  * `N` is the number of elements per partition in `data`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

  * `P` is partition axis size of `data`.

  * `F` is the number of elements per partition in `data`.

Parameters:
    

  * data – the input tile to be transposed

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – if specified and it’s different from the data type of input tile `data`, an additional nki.isa.cast instruction will be inserted to cast the transposed data into the target `dtype` (see Supported Data Types for more information)

  * engine – specify which engine to use for transpose: `nki.isa.tensor_engine` or `nki.isa.vector_engine` ; by default, the best engine will be selected for the given input tile shape

Returns:
    

a tile with transposed result of input `data` tile

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    ##################################################################
    # Example 1: transpose tile a of shape (128, 64)
    ##################################################################
    i_p_a = nl.arange(128)[:, None]
    i_f_a = nl.arange(64)[None, :]
    aT = nisa.nc_transpose(a[i_p_a, i_f_a])
    
    
    ##################################################################
    # Example 2: transpose tile b of shape (32, 2) using Vector Engine
    ##################################################################
    i_p_b = nl.arange(32)[:, None]
    i_f_b = nl.arange(2)[None, :]
    bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)
    

## Element-wise Arithmetic

### nki.language.add

nki.language.add(x, y, *, dtype=None, mask=None, **kwargs)
    

Add the inputs, element-wise.

((Similar to numpy.add))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has `x + y`, element-wise.

Examples:
    
    
    import neuronxcc.nki.language as nl
    
    a = nl.load(a_tensor[0:128, 0:512])
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
    nl.store(c_tensor[0:128, 0:512], c)
    

Note

Broadcasting in the partition dimension is generally more expensive than broadcasting in free dimension. It is recommended to align your data to perform free dimension broadcast whenever possible.

---

### nki.language.subtract

nki.language.subtract(x, y, *, dtype=None, mask=None, **kwargs)
    

Subtract the inputs, element-wise.

((Similar to numpy.subtract))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has `x - y`, element-wise.

---

### nki.language.multiply

nki.language.multiply(x, y, *, dtype=None, mask=None, **kwargs)
    

Multiply the inputs, element-wise.

((Similar to numpy.multiply))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has `x * y`, element-wise.

---

### nki.language.divide

nki.language.divide(x, y, *, dtype=None, mask=None, **kwargs)
    

Divide the inputs, element-wise.

((Similar to numpy.divide))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has `x / y`, element-wise.

---

### nki.language.negative

nki.language.negative(x, *, dtype=None, mask=None, **kwargs)
    

Numerical negative of the input, element-wise.

((Similar to numpy.negative))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has numerical negative values of `x`.

---

### nki.language.power

nki.language.power(x, y, *, dtype=None, mask=None, **kwargs)
    

Elements of x raised to powers of y, element-wise.

((Similar to numpy.power))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has values `x` to the power of `y`.

---

### nki.language.square

nki.language.square(x, *, dtype=None, mask=None, **kwargs)
    

Square of the input, element-wise.

((Similar to numpy.square))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has square of `x`.

---

### nki.language.sqrt

nki.language.sqrt(x, *, dtype=None, mask=None, **kwargs)
    

Non-negative square-root of the input, element-wise.

((Similar to numpy.sqrt))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has square-root values of `x`.

---

### nki.language.rsqrt

nki.language.rsqrt(x, *, dtype=None, mask=None, **kwargs)
    

Reciprocal of the square-root of the input, element-wise.

((Similar to torch.rsqrt))

`rsqrt(x) = 1 / sqrt(x)`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has reciprocal square-root values of `x`.

---

### nki.language.reciprocal

nki.language.reciprocal(x, *, dtype=None, mask=None, **kwargs)
    

Reciprocal of the the input, element-wise.

((Similar to numpy.reciprocal))

`reciprocal(x) = 1 / x`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has reciprocal values of `x`.

---

### nki.language.abs

nki.language.abs(x, *, dtype=None, mask=None, **kwargs)
    

Absolute value of the input, element-wise.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has absolute values of `x`.

---

### nki.language.sign

nki.language.sign(x, *, dtype=None, mask=None, **kwargs)
    

Sign of the numbers of the input, element-wise.

((Similar to numpy.sign))

The sign function returns `-1` if `x < 0`, `0` if `x==0`, `1` if `x > 0`.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has sign values of `x`.

---

### nki.language.floor

nki.language.floor(x, *, dtype=None, mask=None, **kwargs)
    

Floor of the input, element-wise.

((Similar to numpy.floor))

The floor of the scalar x is the largest integer i, such that i <= x.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has floor values of `x`.

---

### nki.language.ceil

nki.language.ceil(x, *, dtype=None, mask=None, **kwargs)
    

Ceiling of the input, element-wise.

((Similar to numpy.ceil))

The ceil of the scalar x is the smallest integer i, such that i >= x.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has ceiling values of `x`.

---

### nki.language.trunc

nki.language.trunc(x, *, dtype=None, mask=None, **kwargs)
    

Truncated value of the input, element-wise.

((Similar to numpy.trunc))

The truncated value of the scalar x is the nearest integer i which is closer to zero than x is. In short, the fractional part of the signed number x is discarded.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has truncated values of `x`.

---

### nki.language.fmod

nki.language.fmod(x, y, dtype=None, mask=None, **kwargs)
    

Floor-mod of `x / y`, element-wise.

The remainder has the same sign as the dividend x. It is equivalent to the Matlab(TM) rem function and should not be confused with the Python modulus operator x % y.

((Similar to numpy.fmod))

Parameters:
    

  * x – a tile. If x is a scalar value it will be broadcast to the shape of y. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has values `x fmod y`.

---

### nki.language.mod

nki.language.mod(x, y, dtype=None, mask=None, **kwargs)
    

Integer Mod of `x / y`, element-wise

Computes the remainder complementary to the floor_divide function. It is equivalent to the Python modulus x % y and has the same sign as the divisor y.

((Similar to numpy.mod))

Parameters:
    

  * x – a tile. If x is a scalar value it will be broadcast to the shape of y. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has values `x mod y`.

---

### nki.language.log

nki.language.log(x, *, dtype=None, mask=None, **kwargs)
    

Natural logarithm of the input, element-wise.

((Similar to numpy.log))

It is the inverse of the exponential function, such that: `log(exp(x)) = x` . The natural logarithm base is `e`.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has natural logarithm values of `x`.

---

### nki.language.exp

nki.language.exp(x, *, dtype=None, mask=None, **kwargs)
    

Exponential of the input, element-wise.

((Similar to numpy.exp))

The `exp(x)` is `e^x` where `e` is the Euler’s number = 2.718281…

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has exponential values of `x`.

---

### nki.isa.reciprocal

nki.isa.reciprocal(data, *, dtype=None, mask=None, **kwargs)
    

Compute reciprocal of each element in the input `data` tile using Vector Engine.

Estimated instruction cost:

`max(MIN_II, 8*N)` Vector Engine cycles, where `N` is the number of elements per partition in `data`, and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – the input tile

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

an output tile of reciprocal computation

Example:
    
    
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    ...
    
    x = nl.load(in_tensor[nl.mgrid[0:128, 0:512]])
    
    y = nisa.reciprocal(x)
    
    

## Trigonometric Functions

### nki.language.sin

nki.language.sin(x, *, dtype=None, mask=None, **kwargs)
    

Sine of the input, element-wise.

((Similar to numpy.sin))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has sine values of `x`.

---

### nki.language.cos

nki.language.cos(x, *, dtype=None, mask=None, **kwargs)
    

Cosine of the input, element-wise.

((Similar to numpy.cos))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has cosine values of `x`.

---

### nki.language.tan

nki.language.tan(x, *, dtype=None, mask=None, **kwargs)
    

Tangent of the input, element-wise.

((Similar to numpy.tan))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has tangent values of `x`.

---

### nki.language.tanh

nki.language.tanh(x, *, dtype=None, mask=None, **kwargs)
    

Hyperbolic tangent of the input, element-wise.

((Similar to numpy.tanh))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has hyperbolic tangent values of `x`.

---

### nki.language.arctan

nki.language.arctan(x, *, dtype=None, mask=None, **kwargs)
    

Inverse tangent of the input, element-wise.

((Similar to numpy.arctan))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has inverse tangent values of `x`.

## Activation Functions

### nki.language.relu

nki.language.relu(x, *, dtype=None, mask=None, **kwargs)
    

Rectified Linear Unit activation function on the input, element-wise.

`relu(x) = (x)+ = max(0,x)`

((Similar to torch.nn.functional.relu))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has relu of `x`.

---

### nki.language.sigmoid

nki.language.sigmoid(x, *, dtype=None, mask=None, **kwargs)
    

Logistic sigmoid activation function on the input, element-wise.

((Similar to torch.nn.functional.sigmoid))

`sigmoid(x) = 1/(1+exp(-x))`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has sigmoid of `x`.

---

### nki.language.silu

nki.language.silu(x, *, dtype=None, mask=None, **kwargs)
    

Sigmoid Linear Unit activation function on the input, element-wise.

((Similar to torch.nn.functional.silu))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has silu of `x`.

---

### nki.language.silu_dx

nki.language.silu_dx(x, *, dtype=None, mask=None, **kwargs)
    

Derivative of Sigmoid Linear Unit activation function on the input, element-wise.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has silu_dx of `x`.

---

### nki.language.gelu

nki.language.gelu(x, *, dtype=None, mask=None, **kwargs)
    

Gaussian Error Linear Unit activation function on the input, element-wise.

((Similar to torch.nn.functional.gelu))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has gelu of `x`.

---

### nki.language.gelu_apprx_tanh

nki.language.gelu_apprx_tanh(x, *, dtype=None, mask=None, **kwargs)
    

Gaussian Error Linear Unit activation function on the input, element-wise, with tanh approximation.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has gelu of `x`.

---

### nki.language.gelu_apprx_sigmoid

nki.language.gelu_apprx_sigmoid(x, *, dtype=None, mask=None, **kwargs)
    

Gaussian Error Linear Unit activation function on the input, element-wise, with sigmoid approximation.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has gelu of `x`.

---

### nki.language.gelu_dx

nki.language.gelu_dx(x, *, dtype=None, mask=None, **kwargs)
    

Derivative of Gaussian Error Linear Unit (gelu) on the input, element-wise.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has gelu_dx of `x`.

---

### nki.language.softplus

nki.language.softplus(x, *, dtype=None, mask=None, **kwargs)
    

Softplus activation function on the input, element-wise.

Softplus is a smooth approximation to the ReLU activation, defined as:

`softplus(x) = log(1 + exp(x))`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has softplus of `x`.

---

### nki.language.mish

nki.language.mish(x, *, dtype=None, mask=None, **kwargs)
    

Mish activation function on the input, element-wise.

Mish: A Self Regularized Non-Monotonic Neural Activation Function is defined as:

\\[mish(x) = x * tanh(softplus(x))\\]

see: https://arxiv.org/abs/1908.08681

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has mish of `x`.

---

### nki.language.erf

nki.language.erf(x, *, dtype=None, mask=None, **kwargs)
    

Error function of the input, element-wise.

((Similar to torch.erf))

`erf(x) = 2/sqrt(pi)*integral(exp(-t**2), t=0..x)` .

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has erf of `x`.

---

### nki.language.erf_dx

nki.language.erf_dx(x, *, dtype=None, mask=None, **kwargs)
    

Derivative of the Error function (erf) on the input, element-wise.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has erf_dx of `x`.

---

### nki.language.softmax

nki.language.softmax(x, axis, *, dtype=None, compute_dtype=None, mask=None, **kwargs)
    

Softmax activation function on the input, element-wise.

((Similar to torch.nn.functional.softmax))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * compute_dtype – (optional) dtype for the internal computation - currently `dtype` and `compute_dtype` behave the same, both sets internal compute and return dtype.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has softmax of `x`.

---

### nki.language.dropout

nki.language.dropout(x, rate, *, dtype=None, mask=None, **kwargs)
    

Randomly zeroes some of the elements of the input tile given a probability rate.

Parameters:
    

  * x – a tile.

  * rate – a scalar value or a tile with 1 element, with the probability rate.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with randomly zeroed elements of `x`.

---

### nki.isa.activation

nki.isa.activation(op, data, *, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, mask=None, dtype=None, **kwargs)
    

Apply an activation function on every element of the input tile using Scalar Engine. The activation function is specified in the `op` input field (see Supported Activation Functions for NKI ISA for a list of supported activation functions and their valid input ranges).

The activation instruction can optionally multiply the input `data` by a scalar or vector `scale` and then add another vector `bias` before the activation function is applied, at no additional performance cost:

\\[output = f_{act}(data * scale + bias)\\]

When the scale is a scalar, it must be a compile-time constant. In this case, the scale is broadcasted to all the elements in the input `data` tile. When the scale/bias is a vector, it must have the same partition axis size as the input `data` tile and only one element per partition. In this case, the element of scale/bias within each partition is broadcasted to elements of the input `data` tile in the same partition.

There are 128 registers on the scalar engine for storing reduction results, corresponding to the 128 partitions of the input. The scalar engine can reduce along free dimensions without extra performance penalty, and store the result of reduction into these registers. The reduction is done after the activation function is applied.

\\[output = f_{act}(data * scale + bias) accu\\_registers = reduce\\_op(accu\\_registers, reduce\\_op(output, axis=<FreeAxis>))\\]

These registers are shared between `activation` and `activation_accu` calls, and the state of them can be controlled via the `reduce_cmd` parameter.

  * `nisa.reduce_cmd.reset`: Reset the accumulators to zero

  * `nisa.reduce_cmd.idle`: Do not use the accumulators

  * `nisa.reduce_cmd.reduce`: keeps accumulating over the current value of the accumulator

  * `nisa.reduce_cmd.reset_reduce`: Resets the accumulators then immediately accumulate the results of the current instruction into the accumulators

We can choose to read out the current values stored in the register by passing in a tensor in the `reduce_res` arguments. Reading out the accumulator will incur a small overhead.

Note that `activation_accu` can also change the state of the registers. It’s user’s responsibility to ensure correct ordering. It’s recommended to not mixing the use of `activation_accu` and `activation`, when `reduce_cmd` is not set to idle.

Note, the Scalar Engine always performs the math operations in float32 precision. Therefore, the engine automatically casts the input `data` tile to float32 before performing multiply/add/activate specified in the activation instruction. The engine is also capable of casting the float32 math results into another output data type specified by the `dtype` field at no additional performance cost. If `dtype` field is not specified, Neuron Compiler will set output data type of the instruction to be the same as input data type of `data`. On the other hand, the `scale` parameter must have a float32 data type, while the `bias` parameter can be float32/float16/bfloat16.

The input `data` tile can be an SBUF or PSUM tile. Similarly, the instruction can write the output tile into either SBUF or PSUM, which is specified using the `buffer` field. If not specified, `nki.language.sbuf` is selected by default.

Estimated instruction cost:

`max(MIN_II, N)` Scalar Engine cycles, where

  * `N` is the number of elements per partition in `data`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)

  * data – the input tile; layout: (partition axis <= 128, free axis)

  * bias – a vector with the same partition axis size as `data` for broadcast add (after broadcast multiply with `scale`)

  * scale – a scalar or a vector with the same partition axis size as `data` for broadcast multiply

  * reduce_op – the reduce operation to perform on the free dimension of the activation result

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `sum(ReductionResult)` is written in-place into the tensor.

  * reduce_cmd – an enum member from `nisa.reduce_cmd` to control the state of reduction registers

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

output tile of the activation instruction; layout: same as input `data` tile

Example:
    
    
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    
    ##################################################################
    # Example 1: perform exponential function on matrix a of shape (128, 1024)
    ##################################################################
    a = nl.load(a_tensor)
    activated_a = nisa.activation(op=nl.exp, data=a)
    nl.store(a_act_tensor, activated_a)
    
    ##################################################################
    # Example 2: perform the following operations to matrix b of shape (128, 512)
    # using a single activation instruction: np.square(b * 2.0) + c
    # 1) compute `np.square(b * 2.0 + c)`
    # 2) cast 1) results into bfloat16
    ##################################################################
    b = nl.load(b_tensor)
    c = nl.load(c_tensor)
    activated_b = nisa.activation(op=np.square, data=b, bias=c, scale=2.0,
                                  dtype=nl.bfloat16)
    nl.store(b_act_tensor, activated_b)
    

---

### nki.isa.activation_reduce

nki.isa.activation_reduce(op, data, *, reduce_op, reduce_res, bias=None, scale=1.0, mask=None, dtype=None, **kwargs)
    

Perform the same computation as `nisa.activation` and also a reduction along the free dimension of the `nisa.activation` result using Scalar Engine. The results for the reduction is stored in the reduce_res.

This API is equivalent to calling `nisa.activation` with `reduce_cmd=nisa.reduce_cmd.reset_reduce` and passing in reduce_res. This API is kept for backward compatibility, we recommend using `nisa.activation` moving forward.

Refer to nisa.activation for semantics of `op/data/bias/scale`.

In addition to nisa.activation computation, this API also performs a reduction along the free dimension(s) of the nisa.activation result, at a small additional performance cost. The reduction result is returned in `reduce_res` in-place, which must be a SBUF/PSUM tile with the same partition axis size as the input tile `data` and one element per partition. On NeuronCore-v2, the `reduce_op` can only be an addition, `np.add` or `nl.add`.

There are 128 registers on the scalar engine for storing reduction results, corresponding to the 128 partitions of the input. These registers are shared between `activation` and `activation_accu` calls. This instruction first resets those registers to zero, performs the reduction on the value after activation function is applied, stores the results into the registers, then reads out the reduction results from the register, eventually store them into `reduce_res`.

Note that `nisa.activation` can also change the state of the register. It’s user’s responsibility to ensure correct ordering. It’s the best practice to not mixing the use of `activation_reduce` and `activation`.

Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will reduce across all of them.

Mathematically, this API performs the following computation:

\\[\begin{split}output = f_{act}(data * scale + bias) \\\ reduce\\_res = reduce\\_op(output, axis=<FreeAxis>)\end{split}\\]

Estimated instruction cost:

`max(MIN_II, N) + MIN_II` Scalar Engine cycles, where

  * `N` is the number of elements per partition in `data`, and

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)

  * data – the input tile; layout: (partition axis <= 128, free axis)

  * reduce_op – the reduce operation to perform on the free dimension of the activation result

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `sum(ReductionResult)` is written in-place into the tensor.

  * bias – a vector with the same partition axis size as `data` for broadcast add (after broadcast multiply with `scale`)

  * scale – a scalar or a vector with the same partition axis size as `data` for broadcast multiply

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

output tile of the activation instruction; layout: same as input `data` tile

---

### nki.isa.dropout

nki.isa.dropout(data, prob, *, mask=None, dtype=None, **kwargs)
    

Randomly replace some elements of the input tile `data` with zeros based on input probabilities using Vector Engine. The probability of replacing input elements with zeros (i.e., drop probability) is specified using the `prob` field: \- If the probability is 1.0, all elements are replaced with zeros. \- If the probability is 0.0, all elements are kept with their original values.

The `prob` field can be a scalar constant or a tile of shape `(data.shape[0], 1)`, where each partition contains one drop probability value. The drop probability value in each partition is applicable to the input `data` elements from the same partition only.

Data type of the input `data` tile can be any valid NKI data types (see Supported Data Types for more information). However, data type of `prob` has restrictions based on the data type of `data`:

  * If data type of `data` is any of the integer types (e.g., int32, int16), `prob` data type must be float32

  * If data type of data is any of the float types (e.g., float32, bfloat16), `prob` data can be any valid float type

The output data type of this instruction is specified by the `dtype` field. The output data type must match the input data type of `data` if input data type is any of the integer types. Otherwise, output data type can be any valid NKI data types. If output data type is not specified, it is default to be the same as input data type.

Estimated instruction cost:

`max(MIN_II, N)` Vector Engine cycles, where `N` is the number of elements per partition in `data`, and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – the input tile

  * prob – a scalar or a tile of shape `(data.shape[0], 1)` to indicate the probability of replacing elements with zeros

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

an output tile of the dropout result

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ###########################################################################
    # Example 1: From an input tile a of shape [128, 512], dropout its values
    # with probabilities in tile b of shape [128, 1] and store the result in c.
    ###########################################################################
    a: tensor[128, 512] = nl.load(a_tensor)
    b: tensor[128, 1] = nl.load(b_tensor)
    
    c: tensor[128, 512] = nisa.dropout(a, prob=b)
    
    nl.store(c_tensor, c)
    
    ######################################################
    # Example 2: From an input tile a, dropout its values 
    # with probability of 0.2 and store the result in b.
    ######################################################
    a = nl.load(in_tensor)
    
    b = nisa.dropout(a, prob=0.2)
    
    nl.store(out_tensor, b)
    

## Comparison and Logical Operations

### nki.language.equal

nki.language.equal(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x == y.

((Similar to numpy.equal))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x == y` element-wise.

---

### nki.language.not_equal

nki.language.not_equal(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x != y.

((Similar to numpy.not_equal))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x != y` element-wise.

---

### nki.language.less

nki.language.less(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x < y.

((Similar to numpy.less))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x < y` element-wise.

---

### nki.language.less_equal

nki.language.less_equal(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x <= y.

((Similar to numpy.less_equal))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x <= y` element-wise.

---

### nki.language.greater

nki.language.greater(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x > y.

((Similar to numpy.greater))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x > y` element-wise.

---

### nki.language.greater_equal

nki.language.greater_equal(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x >= y.

((Similar to numpy.greater_equal))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x >= y` element-wise.

---

### nki.language.logical_and

nki.language.logical_and(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x AND y.

((Similar to numpy.logical_and))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x AND y` element-wise.

---

### nki.language.logical_or

nki.language.logical_or(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x OR y.

((Similar to numpy.logical_or))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x OR y` element-wise.

---

### nki.language.logical_xor

nki.language.logical_xor(x, y, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of x XOR y.

((Similar to numpy.logical_xor))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `x XOR y` element-wise.

---

### nki.language.logical_not

nki.language.logical_not(x, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Element-wise boolean result of NOT x.

((Similar to numpy.logical_not))

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with boolean result of `NOT x` element-wise.

---

### nki.language.where

nki.language.where(condition, x, y, *, dtype=None, mask=None, **kwargs)
    

Return elements chosen from x or y depending on condition.

((Similar to numpy.where))

Parameters:
    

  * condition – if True, yield x, otherwise yield y.

  * x – a tile with values from which to choose if condition is True.

  * y – a tile or a numerical value from which to choose if condition is False.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with elements from x where condition is True, and elements from y otherwise.

---

### nki.language.maximum

nki.language.maximum(x, y, *, dtype=None, mask=None, **kwargs)
    

Maximum of the inputs, element-wise.

((Similar to numpy.maximum))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has the maximum of each elements from x and y.

---

### nki.language.minimum

nki.language.minimum(x, y, *, dtype=None, mask=None, **kwargs)
    

Minimum of the inputs, element-wise.

((Similar to numpy.minimum))

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has the minimum of each elements from x and y.

## Bitwise Operations

### nki.language.left_shift

nki.language.left_shift(x, y, *, dtype=None, mask=None, **kwargs)
    

Bitwise left-shift x by y, element-wise.

((Similar to numpy.left_shift))

Computes the bit-wise left shift of the underlying binary representation of the integers in the input tiles. This function implements the C/Python operator `<<`

Parameters:
    

  * x – a tile or a scalar value of integer type.

  * y – a tile or a scalar value of integer type. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has values `x << y`.

---

### nki.language.bitwise_and

nki.language.bitwise_and(x, y, *, dtype=None, mask=None, **kwargs)
    

Bitwise AND of the two inputs, element-wise.

((Similar to numpy.bitwise_and))

Computes the bit-wise AND of the underlying binary representation of the integers in the input tiles. This function implements the C/Python operator `&`

Parameters:
    

  * x – a tile or a scalar value of integer type.

  * y – a tile or a scalar value of integer type. `x.shape` and `y.shape` must be broadcastable to a common shape, that will become the shape of the output.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile that has values `x & y`.

---

### nki.language.invert

nki.language.invert(x, *, dtype=None, mask=None, **kwargs)
    

Bitwise NOT of the input, element-wise.

((Similar to numpy.invert))

Computes the bit-wise NOT of the underlying binary representation of the integers in the input tile. This ufunc implements the C/Python operator `~`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with bitwise NOT `x` element-wise.

## Reduction Operations

### nki.language.sum

nki.language.sum(x, axis, *, dtype=None, mask=None, keepdims=False, **kwargs)
    

Sum of elements along the specified axis (or axes) of the input.

((Similar to numpy.sum))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

Returns:
    

a tile with the sum of elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.prod

nki.language.prod(x, axis, *, dtype=None, mask=None, keepdims=False, **kwargs)
    

Product of elements along the specified axis (or axes) of the input.

((Similar to numpy.prod))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

Returns:
    

a tile with the product of elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.max

nki.language.max(x, axis, *, dtype=None, mask=None, keepdims=False, **kwargs)
    

Maximum of elements along the specified axis (or axes) of the input.

((Similar to numpy.max))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

Returns:
    

a tile with the maximum of elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.min

nki.language.min(x, axis, *, dtype=None, mask=None, keepdims=False, **kwargs)
    

Minimum of elements along the specified axis (or axes) of the input.

((Similar to numpy.min))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

Returns:
    

a tile with the minimum of elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.mean

nki.language.mean(x, axis, *, dtype=None, mask=None, keepdims=False, **kwargs)
    

Arithmetic mean along the specified axis (or axes) of the input.

((Similar to numpy.mean))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with the average of elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed. `float32` intermediate and return values are used for integer inputs.

---

### nki.language.var

nki.language.var(x, axis, *, dtype=None, mask=None, **kwargs)
    

Variance along the specified axis (or axes) of the input.

((Similar to numpy.var))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a tile with the variance of the elements along the provided axis. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.all

nki.language.all(x, axis, *, dtype=<class 'bool'>, mask=None, **kwargs)
    

Whether all elements along the specified axis (or axes) evaluate to True.

((Similar to numpy.all))

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

a boolean tile with the result. This return tile will have a shape of the input tile’s shape with the specified axes removed.

---

### nki.language.loop_reduce

nki.language.loop_reduce(x, op, loop_indices, *, dtype=None, mask=None, **kwargs)
    

Apply reduce operation over a loop. This is an ideal instruction to compute a high performance reduce_max or reduce_min.

Note: The destination tile is also the rhs input to `op`. For example,
    
    
    b = nl.zeros((N_TILE_SIZE, M_TILE_SIZE), dtype=float32, buffer=nl.sbuf)
    for k_i in affine_range(NUM_K_BLOCKS):
    
      # Skipping over multiple nested loops here.
      # a, is a psum tile from a matmul accumulation group.
      b = nl.loop_reduce(a, op=np.add, loop_indices=[k_i], dtype=nl.float32)
    

is the same as:
    
    
    b = nl.zeros((N_TILE_SIZE, M_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    for k_i in affine_range(NUM_K_BLOCKS):
    
      # Skipping over multiple nested loops here.
      # a, is a psum tile from a matmul accumulation group.
      b = nisa.tensor_tensor(data1=b, data2=a, op=np.add, dtype=nl.float32)
    

If you are trying to use this instruction only for accumulating results on SBUF, consider simply using the `+=` operator instead.

The `loop_indices` list enables the compiler to recognize which loops this reduction can be optimized across as part of any aggressive loop-level optimizations it may perform.

Parameters:
    

  * x – a tile.

  * op – numpy ALU operator to use to reduce over the input tile.

  * loop_indices – a single loop index or a tuple of loop indices along which the reduction operation is performed. Can be numbers or loop_index objects coming from `nl.affine_range`.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

the reduced resulting tile

---

### nki.isa.tensor_reduce

nki.isa.tensor_reduce(op, data, axis, *, mask=None, dtype=None, negate=False, keepdims=False, **kwargs)
    

Apply a reduction operation to the free axes of an input `data` tile using Vector Engine.

The reduction operator is specified in the `op` input field (see Supported Math Operators for NKI ISA for a list of supported reduction operators). There are two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or) and 2) arithmetic operators (e.g., add, subtract, multiply). For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, there is no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the reduction operation in float32 math. The float32 reduction results are cast to the target data type specified in the `dtype` field before written into the output tile. If the `dtype` field is not specified, it is default to be the same as input tile data type.

When the reduction `op` is an arithmetic operator, the instruction can also multiply the output reduction results by `-1.0` before writing into the output tile, at no additional performance cost. This behavior is controlled by the `negate` input field.

The reduction axes are specified in the `axis` field using a list of integer(s) to indicate axis indices. The reduction axes can contain up to four free axes and must start at the most minor free axis. Since axis 0 is the partition axis in a tile, the reduction axes must contain axis 1 (most-minor). In addition, the reduction axes must be consecutive: e.g., [1, 2, 3, 4] is a legal `axis` field, but [1, 3, 4] is not.

Since this instruction only supports free axes reduction, the output tile must have the same partition axis size as the input `data` tile. To perform a partition axis reduction, we can either:

  1. invoke a `nki.isa.nc_transpose` instruction on the input tile and then this `reduce` instruction to the transposed tile, or

  2. invoke `nki.isa.nc_matmul` instructions to multiply a `nki.language.ones([128, 1], dtype=data.dtype)` vector with the input tile.

Estimated instruction cost:

Cost (Vector Engine Cycles) | Condition  
---|---  
`N/2` | both input and output data types are `bfloat16` and the reduction operator is add or maximum  
`N` | otherwise  
  
where,

  * `N` is the number of elements per partition in `data`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * op – the reduction operator (see Supported Math Operators for NKI ISA for supported reduction operators)

  * data – the input tile to be reduced

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * negate – if True, reduction result is multiplied by `-1.0`; only applicable when op is an arithmetic operator

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

Returns:
    

output tile of the reduction result

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import numpy as np
    ...
    
    ##################################################################
    # Example 1: reduce add tile a of shape (128, 512)
    # in the free dimension and return
    # reduction result in tile b of shape (128, 1)
    ##################################################################
    i_p_a = nl.arange(128)[:, None]
    i_f_a = nl.arange(512)[None, :]
    
    b = nisa.tensor_reduce(np.add, a[i_p_a, i_f_a], axis=[1])
    

---

### nki.isa.tensor_partition_reduce

nki.isa.tensor_partition_reduce(op, data, *, mask=None, dtype=None, **kwargs)
    

Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine.

Parameters:
    

  * op – the reduction operator (add, max, bitwise_or, bitwise_and)

  * data – the input tile to be reduced

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

output tile with reduced result

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import numpy as np
    ...
    
    ##################################################################
    # Example 1: reduce add tile a of shape (128, 32, 4)
    # in the partition dimension and return
    # reduction result in tile b of shape (1, 32, 4)
    ##################################################################
    a = nl.load(a_tensor[0:128, 0:32, 0:4])  
    b = nisa.tensor_partition_reduce(np.add, a)
    nl.store(b_tensor[0:1, 0:32, 0:4], b)
    
    ##################################################################
    # Example 2: reduce add tile a of shape (b, p, f1, ...)
    # in the partition dimension p and return
    # reduction result in tile b of shape (b, 1, f1, ...)
    ##################################################################
    for i in nl.affine_range(a_tensor.shape[0]):
      a = nl.load(a_tensor[i])
      b = nisa.tensor_partition_reduce(np.add, a)
      nl.store(b_tensor[i], b)
    

## Tensor-Tensor and Tensor-Scalar ISA Operations

### nki.isa.tensor_tensor

nki.isa.tensor_tensor(data1, data2, op, *, dtype=None, mask=None, engine=engine.unknown, **kwargs)
    

Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine. The two tiles must have the same partition axis size and the same number of elements per partition.

The element-wise operator is specified using the `op` field and can be any binary operator supported by NKI (see Supported Math Operators for NKI ISA for details) that runs on the Vector Engine, or it can be `power` or integer `add`, `multiply``, or `subtract` which run on the GpSimd Engine. For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, there is no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the element-wise operation in float32 math (unless it is one of the supported integer ops mentioned above). The float32 results are cast to the target data type specified in the `dtype` field before written into the output tile. If the `dtype` field is not specified, it is default to be the same as the data type of `data1` or `data2`, whichever has the higher precision.

Since GpSimd Engine cannot access PSUM, the input or output tiles cannot be in PSUM if `op` is one of the GpSimd operations mentioned above. (see NeuronCore-v2 Compute Engines for details). Otherwise, the output tile can be in either SBUF or PSUM. However, the two input tiles, `data1` and `data2` cannot both reside in PSUM. The three legal cases are:

  1. Both `data1` and `data2` are in SBUF.

  2. `data1` is in SBUF, while `data2` is in PSUM.

  3. `data1` is in PSUM, while `data2` is in SBUF.

Note, if you need broadcasting capability in the free dimension for either input tile, you should consider using nki.isa.tensor_scalar API instead, which has better performance than `nki.isa.tensor_tensor` in general.

Estimated instruction cost:

See below table for tensor_tensor performance when it runs on Vector Engine.

Cost (Vector Engine Cycles) | Condition  
---|---  
`max(MIN_II, N)` | one input tile is in PSUM and the other is in SBUF  
`max(MIN_II, N)` | all of the below:

  * both input tiles are in SBUF,
  * input/output data types are all `bfloat16`,
  * the operator is add, multiply or subtract,
  * Input tensor data is contiguous along the free dimension (that is, stride in each partition is 1 element)

  
`max(MIN_II, 2N)` | otherwise  
  
where,

  * `N` is the number of elements per partition in `data1`/`data2`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data1 – lhs input operand of the element-wise operation

  * data2 – rhs input operand of the element-wise operation

  * op – a binary math operator (see Supported Math Operators for NKI ISA for supported operators)

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

Returns:
    

an output tile of the element-wise operation

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    ...
    
    ##################################################################
    # Example 1: add two tiles, a and b, of the same
    # shape (128, 512) element-wise and get
    # the addition result in tile c
    ##################################################################
    a: tensor[128, 512] = nl.load(a_tensor)
    b: tensor[128, 512] = nl.load(b_tensor)
    
    c: tensor[128, 512] = nisa.tensor_tensor(a, b, op=nl.add)
    
    


---

### nki.isa.tensor_scalar

nki.isa.tensor_scalar(data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, *, dtype=None, mask=None, engine=engine.unknown, **kwargs)
    

Apply up to two math operators to the input `data` tile by broadcasting scalar/vector operands in the free dimension using Vector or Scalar or GpSimd Engine: `(data <op0> operand0) <op1> operand1`.

The input `data` tile can be an SBUF or PSUM tile. Both `operand0` and `operand1` can be SBUF or PSUM tiles of shape `(data.shape[0], 1)`, i.e., vectors, or compile-time constant scalars.

`op1` and `operand1` are optional, but must be `None` (default values) when unused. Note, performing one operator has the same performance cost as performing two operators in the instruction.

When the operators are non-commutative (e.g., subtract), we can reverse ordering of the inputs for each operator through:

>   * `reverse0 = True`: `tmp_res = operand0 <op0> data`
> 
>   * `reverse1 = True`: `operand1 <op1> tmp_res`
> 
> 

The `tensor_scalar` instruction supports two types of operators: 1) bitvec operators (e.g., bitwise_and) and 2) arithmetic operators (e.g., add). See Supported Math Operators for NKI ISA for the full list of supported operators. The two operators, `op0` and `op1`, in a `tensor_scalar` instruction must be of the same type (both bitvec or both arithmetic). If bitvec operators are used, the `tensor_scalar` instruction must run on Vector Engine. Also, the input/output data types must be integer types, and input elements are treated as bit patterns without any data type casting.

If arithmetic operators are used, the `tensor_scalar` instruction can run on Vector or Scalar or GpSimd Engine. However, each engine supports limited arithmetic operators (see :ref:`tbl-aluop`). The Scalar Engine on trn2 only supports a subset of the operator combination:

>   * `op0=np.multiply` and `op1=np.add`
> 
>   * `op0=np.multiply` and `op1=None`
> 
>   * `op0=add` and `op1=None`
> 
> 

Also, arithmetic operators impose no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the operators in float32 math. The float32 computation results are cast to the target data type specified in the `dtype` field before written into the output tile, at no additional performance cost. If the `dtype` field is not specified, it is default to be the same as input tile data type.

Estimated instruction cost:

`max(MIN_II, N)` Vector or Scalar Engine cycles, where

  * `N` is the number of elements per partition in `data`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – the input tile

  * op0 – the first math operator used with operand0 (see Supported Math Operators for NKI ISA for supported operators)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse0 – reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`

  * op1 – the second math operator used with operand1 (see Supported Math Operators for NKI ISA for supported operators); this operator is optional

  * operand1 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse1 – reverse ordering of inputs to `op1`; if false, `operand1` is the rhs of `op1`; if true, `operand1` is the lhs of `op1`

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine (only allowed for rsqrt) or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

Returns:
    

an output tile of `(data <op0> operand0) <op1> operand1` computation

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import numpy as np
    ...
    
    ##################################################################
    # Example 1: subtract 1.0 from all elements of tile a of
    # shape (128, 512) and get the output tile in b
    ##################################################################
    i_p = nl.arange(128)[:, None]
    i_f = nl.arange(512)[None, :]
    
    b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)
    
    
    ##################################################################
    # Example 2: broadcast 1.0 into a shape of (128, 512) and subtract
    # it with tile c to get output tile d
    ##################################################################
    i_p = nl.arange(128)[:, None]
    i_f = nl.arange(512)[None, :]
    
    d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)
    
    
    ##################################################################
    # Example 3: broadcast multiply tile e with vector f and
    # then broadcast add with scalar 2.5;
    # tile e has a shape of (64, 1024) and vector f has a shape of (64, 1)
    ##################################################################
    i_p_ef = nl.arange(64)[:, None]
    i_f_e = nl.arange(1024)[None, :]
    i_f_f = nl.arange(1)[None, :]
    
    g = nisa.tensor_scalar(e[i_p_ef, i_f_e], op0=np.multiply, operand0=f[i_p_ef, i_f_f], op1=np.add, operand1=2.5)  
    

---

### nki.isa.scalar_tensor_tensor

nki.isa.scalar_tensor_tensor(*, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, dtype=None, mask=None, **kwargs)
    

Apply up to two math operators using Vector Engine: `(data <op0> operand0) <op1> operand1`.

`data` input can be an SBUF or PSUM tile of 2D shape. `operand0` can be SBUF or PSUM tile of shape `(data.shape[0], 1)`, i.e., vector, or a compile-time constant scalar. `operand1` can be SBUF or PSUM tile of shape `(data.shape[0], data.shape[1])` (i.e., has to match `data` shape), note that `operand1` and `data` can’t both be on PSUM.

Estimated instruction cost:

Cost (Vector Engine Cycles) | Condition  
---|---  
`N` | `data` and `operand1` are both `bfloat16`, `op0=nl.subtract` and `op1=nl.multiply`, and `N` is even  
`2*N` | otherwise  
  
where,

  * `N` is the number of elements per partition in `data`.

Parameters:
    

  * data – the input tile

  * op0 – the first math operator used with operand0 (see Supported Math Operators for NKI ISA for supported operators)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile.

  * reverse0 – reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`.

  * op1 – the second math operator used with operand1 (see Supported Math Operators for NKI ISA for supported operators).

  * operand1 – a tile of shape with the same partition and free dimension as `data` input.

  * reverse1 – reverse ordering of inputs to `op1`; if false, `operand1` is the rhs of `op1`; if true, `operand1` is the lhs of `op1`.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

an output tile of `(data <op0> operand0) <op1> operand1` computation

---

### nki.isa.tensor_tensor_scan

nki.isa.tensor_tensor_scan(data0, data1, initial, op0, op1, reverse0=False, reverse1=False, *, dtype=None, mask=None, **kwargs)
    

Perform a scan operation of two input tiles using Vector Engine.

Mathematically, the tensor_tensor_scan instruction on Vector Engine performs the following computation per partition:
    
    
    # Let's assume we work with numpy, and data0 and data1 are 2D (with shape[0] being the partition axis)
    import numpy as np
    
    result = np.ndarray(data0.shape, dtype=data0.dtype)
    result[:, 0] = op1(op0(data0[:. 0], initial), data1[:, 0])
    
    for i in range(1, data0.shape[1]):
        result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i])
    

The two input tiles (`data0` and `data1`) must have the same partition axis size and the same number of elements per partition. The third input `initial` can either be a float32 compile-time scalar constant that will be broadcasted in the partition axis of `data0`/`data1`, or a tile with the same partition axis size as `data0`/`data1` and one element per partition.

The two input tiles, `data0` and `data1` cannot both reside in PSUM. The three legal cases are:

  1. Both `data1` and `data2` are in SBUF.

  2. `data1` is in SBUF, while `data2` is in PSUM.

  3. `data1` is in PSUM, while `data2` is in SBUF.

The scan operation supported by this API has two programmable math operators in `op0` and `op1` fields. Both `op0` and `op1` can be any binary arithmetic operator supported by NKI (see Supported Math Operators for NKI ISA for details). We can optionally reverse the input operands of `op0` by setting `reverse0` to True (or `op1` by setting `reverse1`). Reversing operands is useful for non-commutative operators, such as subtract.

Input/output data types can be any supported NKI data type (see Supported Data Types), but the engine automatically casts input data types to float32 and performs the computation in float32 math. The float32 results are cast to the target data type specified in the `dtype` field before written into the output tile. If the `dtype` field is not specified, it is default to be the same as the data type of `data0` or `data1`, whichever has the highest precision.

Estimated instruction cost:

`max(MIN_II, 2N)` Vector Engine cycles, where

  * `N` is the number of elements per partition in `data0`/`data1`.

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data0 – lhs input operand of the scan operation

  * data1 – rhs input operand of the scan operation

  * initial – starting state of the scan; can be a SBUF/PSUM tile with 1 element/partition or a scalar compile-time constant

  * op0 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)

  * op1 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)

  * reverse0 – reverse ordering of inputs to `op0`; if false, `data0` is the lhs of `op0`; if true, `data0` is the rhs of `op0`

  * reverse1 – reverse ordering of inputs to `op1`; if false, `data1` is the rhs of `op1`; if true, `data1` is the lhs of `op1`

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

an output tile of the scan operation

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ##################################################################
    # Example 1: scan two tiles, a and b, of the same
    # shape (128, 1024) using multiply/add and get
    # the scan result in tile c
    ##################################################################
    c = nl.ndarray(shape=(128, 1024), dtype=nl.float32)
    
    c[:, 0:512] = nisa.tensor_tensor_scan(a[:, 0:512], b[:, 0:512],
                                          initial=0, op0=np.multiply, op1=np.add)
    
    c[:, 512:1024] = nisa.tensor_tensor_scan(a[:, 512:1024], b[:, 512:1024],
                                             initial=c[:, 511],
                                             op0=np.multiply, op1=np.add)
    

---

### nki.isa.tensor_scalar_reduce

nki.isa.tensor_scalar_reduce(*, data, op0, operand0, reduce_op, reduce_res, reverse0=False, dtype=None, mask=None, **kwargs)
    

Perform the same computation as `nisa.tensor_scalar` with one math operator and also a reduction along the free dimension of the `nisa.tensor_scalar` result using Vector Engine.

Refer to nisa.tensor_scalar for semantics of `data/op0/operand0`. Unlike regular `nisa.tensor_scalar` where two operators are supported, only one operator is supported in this API. Also, `op0` can only be arithmetic operation in Supported Math Operators for NKI ISA. Bitvec operators are not supported in this API.

In addition to nisa.tensor_scalar computation, this API also performs a reduction along the free dimension(s) of the nisa.tensor_scalar result, at a small additional performance cost. The reduction result is returned in `reduce_res` in-place, which must be a SBUF/PSUM tile with the same partition axis size as the input tile `data` and one element per partition. The `reduce_op` can be any of `nl.add`, `nl.subtract`, `nl.multiply`, `nl.max` or `nl.min`.

Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will reduce across all of them.

\\[\begin{split}result = data <op0> operand0 \\\ reduce\\_res = reduce\\_op(dst, axis=<FreeAxis>)\end{split}\\]

Estimated instruction cost:

`max(MIN_II, N) + MIN_II` Vector Engine cycles, where

  * `N` is the number of elements per partition in `data`, and

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – the input tile

  * op0 – the math operator used with operand0 (any arithmetic operator in Supported Math Operators for NKI ISA is allowed)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse0 – (not supported yet) reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`. <– currently not supported yet.

  * reduce_op – the reduce operation to perform on the free dimension of `data <op0> operand0`

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `reduce_op(data <op0> operand0)` is written in-place into the tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

an output tile of `(data <op0> operand0)` computation


---

### nki.isa.select_reduce

nki.isa.select_reduce(*, dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=<function amax>, reverse_pred=False, mask=None, dtype=None, **kwargs)
    

Selectively copy elements from either `on_true` or `on_false` to the destination tile based on a `predicate` using Vector Engine, with optional reduction (max).

The operation can be expressed in NumPy as:
    
    
    # Select:
    predicate = ~predicate if reverse_pred else predicate
    result = np.where(predicate, on_true, on_false)
    
    # With Reduce:
    reduction_result = np.max(result, axis=1, keepdims=True)
    

Memory constraints:

  * Both `on_true` and `predicate` are permitted to be in SBUF

  * Either `on_true` or `predicate` may be in PSUM, but not both simultaneously

  * The destination `dst` can be in either SBUF or PSUM

Shape and data type constraints:

  * `on_true`, `dst`, and `predicate` must have identical shapes (same number of partitions and elements per partition)

  * `on_true` can be any supported dtype except `tfloat32`, `int32`, `uint32`

  * `on_false` dtype must be `float32` if `on_false` is a scalar.

  * `on_false` has to be either scalar or vector of shape `(on_true.shape[0], 1)`

  * `predicate` dtype can be any supported integer type `int8`, `uint8`, `int16`, `uint16`

  * `reduce_res` must be a vector of shape `(on_true.shape[0], 1)`

  * `reduce_res` dtype must of float type

  * `reduce_op` only supports `max`

Behavior:

  * Where predicate is True: The corresponding elements from `on_true` are copied to `dst`

  * Where predicate is False: The corresponding elements from `on_false` are copied to `dst`

  * When reduction is enabled, the max value from each partition of the `result` is computed and stored in `reduce_res`

Accumulator behavior:

The Vector Engine maintains internal accumulator registers that can be controlled via the `reduce_cmd` parameter:

  * `nisa.reduce_cmd.reset_reduce`: Reset accumulators to -inf, then accumulate the current results

  * `nisa.reduce_cmd.reduce`: Continue accumulating without resetting (useful for multi-step reductions)

  * `nisa.reduce_cmd.idle`: No accumulation performed (default)

Note

Even when `reduce_cmd` is set to `idle`, the accumulator state may still be modified. Always use `reset_reduce` after any operations that ran with `idle` mode to ensure consistent behavior.

Note

The accumulator registers are shared for other Vector Engine accumulation instructions such nki.isa.range_select

Parameters:
    

  * dst – The destination tile to write the selected values to

  * predicate – Tile that determines which value to select (on_true or on_false)

  * on_true – Tile to select from when predicate is True

  * on_false – Value to use when predicate is False, can be a scalar value or a vector tile of `(on_true.shape[0], 1)`

  * reduce_res – (optional) Tile to store reduction results, must have shape `(on_true.shape[0], 1)`

  * reduce_cmd – (optional) Control accumulator behavior using `nisa.reduce_cmd` values, defaults to idle

  * reduce_op – (optional) Reduction operator to apply (only `np.max` is supported)

  * reverse_pred – (optional) Reverse the meaning of the predicate condition, defaults to False

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Example 1: Basic selection
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ##################################################################
    # Example 1: Basic usage of select_reduce
    # Create source data, predicate, and destination tensors
    ##################################################################
    # Create output tensor for result
    result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
    
    # Load input data to SBUF
    predicate = nl.load(predicate_data[...])
    on_true = nl.load(on_true_data[...])
    
    # Create destination tensor
    dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform select operation - copy from on_true where predicate is true
    # and set to fp32.min where predicate is false
    nisa.select_reduce(
        dst=dst,
        predicate=predicate,
        on_true=on_true,
        on_false=nl.fp32.min,
    )
    
    # Store result to HBM
    nl.store(result_tensor, value=dst)
    

Example 2: Selection with reduction
    
    
    ##################################################################
    # Example 2: Using select_reduce with reduction
    # Perform selection and compute max reduction per partition
    ##################################################################
    # Create output tensors for results
    result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
    reduce_tensor = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
    
    # Load input data to SBUF
    predicate = nl.load(predicate_data)
    on_true = nl.load(on_true_data)
    on_false = nl.load(on_false_data)
    
    # Create destination tensor
    dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    # Create tensor for reduction results
    reduce_res = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform select operation with reduction
    nisa.select_reduce(
        dst=dst,
        predicate=predicate,
        on_true=on_true,
        on_false=on_false,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_res=reduce_res,
        reduce_op=nl.max
    )
    
    # Store results to HBM
    nl.store(result_tensor, value=dst)
    nl.store(reduce_tensor, value=reduce_res)
    

Example 3: Selection with reversed predicate
    
    
    ##################################################################
    # Example 3: Using select_reduce with reverse_pred option
    # Reverse the meaning of the predicate
    ##################################################################
    # Create output tensor for result
    result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
    
    # Load input data to SBUF
    predicate = nl.load(predicate_data[...])
    on_true = nl.load(on_true_data[...])
    
    # Create destination tensor
    dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform select operation with reverse_pred=True
    # This will select on_true where predicate is FALSE
    nisa.select_reduce(
        dst=dst,
        predicate=predicate,
        on_true=on_true,
        on_false=nl.fp32.min,
        reverse_pred=True  # Reverse the meaning of the predicate
    )
    
    # Store result to HBM
    nl.store(result_tensor, value=dst)
    

---

### nki.isa.range_select

nki.isa.range_select(*, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.idle, reduce_res=None, reduce_op=<function amax>, range_start=0, on_false_value=<property object>, mask=None, dtype=None, **kwargs)
    

Select elements from `on_true_tile` based on comparison with bounds using Vector Engine.

Note

Available only on NeuronCore-v3 and beyond.

For each element in `on_true_tile`, compares its free dimension index + `range_start` against `bound0` and `bound1` using the specified comparison operators (`comp_op0` and `comp_op1`). If both comparisons evaluate to True, copies the element to the output; otherwise uses `on_false_value`.

Additionally performs a reduction operation specified by `reduce_op` on the results, storing the reduction result in `reduce_res`.

Note on numerical stability:

In self-attention, we often have this instruction sequence: `range_select` (VectorE) -> `reduce_res` -> `activation` (ScalarE). When `range_select` outputs a full row of `fill_value`, caution is needed to avoid NaN in the activation instruction that subtracts the output of `range_select` by `reduce_res` (max value):

  * If `dtype` and `reduce_res` are both FP32, we should not hit any NaN issue since `FP32_MIN - FP32_MIN = 0`. Exponentiation on 0 is stable (1.0 exactly).

  * If `dtype` is FP16/BF16/FP8, the fill_value in the output tile will become `-INF` since HW performs a downcast from FP32_MIN to a smaller dtype. In this case, you must make sure reduce_res uses FP32 `dtype` to avoid NaN in `activation`. NaN can be avoided because `activation` always upcasts input tiles to FP32 to perform math operations: `-INF - FP32_MIN = -INF`. Exponentiation on `-INF` is stable (0.0 exactly).

Constraints:

The comparison operators must be one of:

  * np.equal

  * np.less

  * np.less_equal

  * np.greater

  * np.greater_equal

Partition dim sizes must match across `on_true_tile`, `bound0`, and `bound1`:

  * `bound0` and `bound1` must have one element per partition

  * `on_true_tile` must be one of the FP dtypes, and `bound0/bound1` must be FP32 types.

The comparison with `bound0`, `bound1`, and free dimension index is done in FP32. Make sure `range_start` \+ free dimension index is within 2^24 range.

Estimated instruction cost:

`max(MIN_II, N)` Vector Engine cycles, where:

  * `N` is the number of elements per partition in `on_true_tile`, and

  * `MIN_II` is the minimum instruction initiation interval for small input tiles.

  * `MIN_II` is roughly 64 engine cycles.

Numpy equivalent:
    
    
    indices = np.zeros(on_true_tile.shape)
    indices[:] = range_start + np.arange(on_true_tile[0].size)
    
    mask = comp_op0(indices, bound0) & comp_op1(indices, bound1)
    select_out_tile = np.where(mask, on_true_tile, on_false_value)
    reduce_tile = reduce_op(select_out_tile, axis=1, keepdims=True)
    

Parameters:
    

  * on_true_tile – input tile containing elements to select from

  * on_false_value – constant value to use when selection condition is False. Due to HW constraints, this must be FP32_MIN FP32 bit pattern

  * comp_op0 – first comparison operator

  * comp_op1 – second comparison operator

  * bound0 – tile with one element per partition for first comparison

  * bound1 – tile with one element per partition for second comparison

  * reduce_op – reduction operator to apply on across the selected output. Currently only `np.max` is supported.

  * reduce_res – optional tile to store reduction results.

  * range_start – starting base offset for index array for the free dimension of `on_true_tile` Defaults to 0, and must be a compiler time integer.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

output tile with selected elements

Example:
    
    
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import numpy as np
    ...
    
    ##################################################################
    # Example 1: # Select elements where 
    # bound0 <= range_start + index < bound1 and compute max reduction
    # 
    # on_false_value must be nl.fp32.min
    ##################################################################
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])
    
    reduce_res_tile = nl.ndarray((on_true.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(on_true.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    result[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_res=reduce_res_tile,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )
    
    nl.store(select_res[...], value=result[...])
    nl.store(reduce_result[...], value=reduce_res_tile[...])
    

Alternatively, `reduce_cmd` can be used to chain multiple calls to the same accumulation register to accumulate across multiple range_select calls. For example:
    
    
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import numpy as np
    ...
    
    ##################################################################
    # Example 2.a: Initialize reduction with first range_select
    # Notice we don't pass reduce_res since the accumulation
    # register keeps track of the accumulation until we're ready to 
    # read it. Also we use reset_reduce in order to "clobber" or zero
    # out the accumulation register before we start accumulating.
    #
    # Note: Since the type of these tensors are fp32, we use nl.fp32.min
    # for on_false_value due to HW constraints.
    ##################################################################
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])
    
    reduce_res_sbuf = nl.ndarray((on_true.shape[0], 1), dtype=np.float32, buffer=nl.sbuf)
    result_sbuf = nl.ndarray(on_true.shape, dtype=np.float32, buffer=nl.sbuf)
    
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )
    
    ##################################################################
    # Example 2.b: Chain multiple range_select operations 
    # with reduction in an affine loop. Adding ones just lets us ensure the reduction 
    # gets updated with new values.
    ##################################################################
    ones = nl.full(on_true.shape, fill_value=1, dtype=np.float32, buffer=nl.sbuf)
    # we are going to loop as if we're tiling on the partition dimension    
    iteration_step_size = on_true_tile.shape[0]
    
    # Perform chained operations using an affine loop index for range_start
    for i in range(1, 2):
        # Update input values
        on_true_tile[...] = nl.add(on_true_tile, ones)
        
        # Continue reduction with updated values
        # notice, we still don't have reduce_res specified
        result_sbuf[...] = nisa.range_select(
            on_true_tile=on_true_tile,
            comp_op0=compare_op0,
            comp_op1=compare_op1,
            bound0=bound0_tile,
            bound1=bound1_tile,
            reduce_cmd=nisa.reduce_cmd.reduce,
            reduce_op=np.max,
            # we can also use index expressions for setting the start of the range
            range_start=range_start + (i * iteration_step_size),
            on_false_value=nl.fp32.min
        )
    
    range_start = range_start + (2 * iteration_step_size)
    ##################################################################
    # Example 2.c: Final iteration, we actually want the results to 
    # return to the user so we pass reduce_res argument so the 
    # reduction  will be written from the accumulation 
    # register to reduce_res_tile
    ##################################################################
    on_true_tile[...] = nl.add(on_true_tile, ones)
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reduce,
        reduce_res=reduce_res_sbuf[...],
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )
    
    nl.store(select_res[...], value=result_sbuf[...])
    nl.store(reduce_result[...], value=reduce_res_sbuf[...])
    

---

### nki.isa.affine_select

nki.isa.affine_select(pred, on_true_tile, on_false_value, *, mask=None, dtype=None, **kwargs)
    

Select elements between an input tile `on_true_tile` and a scalar value `on_false_value` according to a boolean predicate tile using GpSimd Engine. The predicate tile is calculated on-the-fly in the engine by evaluating an affine expression element-by-element as indicated in `pred`.

`pred` must meet the following requirements:

>   * It must not depend on any runtime variables that can’t be resolved at compile-time.
> 
>   * It can’t be multiple masks combined using logical operators such as `&` and `|`.
> 
> 

For a complex predicate that doesn’t meet the above requirements, consider using nl.where.

The input tile `on_true_tile`, the calculated boolean predicate tile expressed by `pred`, and the returned output tile of this instruction must have the same shape. If the predicate value of a given position is `True`, the corresponding output element will take the element from `on_true_tile` in the same position. If the predicate value of a given position is `False`, the corresponding output element will take the value of `on_false_value`.

A common use case for `affine_select` is to apply a causal mask on the attention scores for transformer decoder models.

This instruction allows any float or 8-bit/16-bit integer data types for both the input data tile and output tile (see Supported Data Types for more information). The output tile data type is specified using the `dtype` field. If `dtype` is not specified, the output data type will be the same as the input data type of `data`. However, the data type of `on_false_value` must be float32, regardless of the input/output tile data types.

Estimated instruction cost:

`GPSIMD_START + N` GpSimd Engine cycles, where `N` is the number of elements per partition in `on_true_tile` and `GPSIMD_START` is the instruction startup overhead on GpSimdE, roughly 150 engine cycles.

Parameters:
    

  * pred – an affine expression that defines the boolean predicate

  * on_true_tile – an input tile for selection with a `True` predicate value

  * on_false_value – a scalar value for selection with a `False` predicate value

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

an output tile with values selected from either `on_true_tile` or `on_false_value` according to the following equation: output[x] = (pred[x] > 0) ? on_true_tile[x] : on_false_value

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ##################################################################
    # Example 1: Take tile a of shape [128, 128] and replace its
    # upper triangle with nl.fp32.min;
    ##################################################################
    ix, iy = nl.mgrid[0:128, 0:128]
    a = nl.load(a_tensor[ix, iy])
    
    b = nisa.affine_select(pred=(iy <ix), on_true_tile=a[ix, iy], on_false_value=nl.fp32.min)
    
    nl.store(b_tensor[ix, iy], b)
    

## Matrix Multiplication

### nki.language.matmul

nki.language.matmul(x, y, *, transpose_x=False, mask=None, **kwargs)
    

`x @ y` matrix multiplication of `x` and `y`.

((Similar to numpy.matmul))

Note

For optimal performance on hardware, use `nki.isa.nc_matmul()` or call `nki.language.matmul` with `transpose_x=True`. Use `nki.isa.nc_matmul` also to access low-level features of the Tensor Engine.

Note

Implementation details: `nki.language.matmul` calls `nki.isa.nc_matmul` under the hood. `nc_matmul` is neuron specific customized implementation of matmul that computes `x.T @ y`, as a result, `matmul(x, y)` lowers to `nc_matmul(transpose(x), y)`. To avoid this extra transpose instruction being inserted, use `x.T` and `transpose_x=True` inputs to this `matmul`.

Parameters:
    

  * x – a tile on SBUF (partition dimension `<= 128`, free dimension `<= 128`), `x`’s free dimension must match `y`’s partition dimension.

  * y – a tile on SBUF (partition dimension `<= 128`, free dimension `<= 512`)

  * transpose_x – Defaults to False. If `True`, `x` is treated as already transposed. If `False`, an additional transpose will be inserted to make `x`’s partition dimension the contract dimension of the matmul to align with the Tensor Engine.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

`x @ y` or `x.T @ y` if `transpose_x=True`

---

### nki.isa.nc_matmul

nki.isa.nc_matmul(stationary, moving, *, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), mask=None, **kwargs)
    

Compute `stationary.T @ moving` matrix multiplication using Tensor Engine.

The `nc_matmul` instruction must read inputs from SBUF and write outputs to PSUM. Therefore, the `stationary` and `moving` must be SBUF tiles, and the result tile is a PSUM tile.

The nc_matmul instruction currently supports `float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32` input data types as listed in Supported Data Types. The matmul accumulation and results are always in float32.

The Tensor Engine imposes special layout constraints on the input tiles. First, the partition axis sizes of the `stationary` and `moving` tiles must be identical and `<=128`, which corresponds to the contraction dimension of the matrix multiplication. Second, the free axis sizes of `stationary` and `moving` tiles must be `<= 128` and `<=512`, respectively, For example, `stationary.shape = (128, 126)`; `moving.shape = (128, 512)` and `nc_matmul(stationary,moving)` returns a tile of `shape = (126, 512)`. For more information about the matmul layout, see Tensor Engine.

Fig. 15 MxKxN Matrix Multiplication Visualization.#

If the contraction dimension of the matrix multiplication exceeds `128`, you may accumulate multiple `nc_matmul` instruction output tiles into the same PSUM tile. See example code snippet below.

Estimated instruction cost:

The Tensor Engine has complex performance characteristics given its data flow and pipeline design. The below formula is the average nc_matmul cost assuming many `nc_matmul` instructions of the same shapes running back-to-back on the engine:

Cost (Tensor Engine Cycles) | Condition  
---|---  
`max(min(64, N_stationary), N_moving)` | input data type is one of `float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32`  
`4 * max(min(64, N_stationary), N_moving)` | input data type is `float32`  
  
where,

  * `N_stationary` is the number of elements per partition in `stationary` tile.

  * `N_moving` is the number of elements per partition in `moving` tile.

The Tensor Engine, as a systolic array with 128 rows and 128 columns of processing elements (PEs), could be underutilized for small `nc_matmul` instructions, i.e., the `stationary` tile has small free axis size or small partition axis size (e.g. 32, 64). In such a case, the Tensor Engine allows PE tiling, i.e., multiple small `nc_matmul` instructions to execute in parallel on the PE array, to improve compute throughput. PE tiling is enabled by setting `tile_position` and `tile_size`. `tile_position` indicates the PE tile starting position (row position, column position) for a `nc_matmul` instruction in the PE array. `tile_size` indicates the PE tile size (row size, column size) to hold by a `nc_matmul` instruction starting from the `tile_position`. For example, setting `tile_position` to (0, 0) and `tile_size` to (128, 128) means using full PE array.

Requirements on `tile_position` and `tile_size` are:

  1. `tile_position` and `tile_size` must be both set to enable PE tiling.

  2. The type of values in `tile_position` and `tile_size` must be integer or affine expression.

  3. Values in `tile_position` and `tile_size` must be multiple of 32.

  4. `tile_size` must be larger than or equal to accessed `stationary` tile size.

  5. Both the row and column sizes in `tile_size` cannot be 32 for NeuronCore-v2.

Parameters:
    

  * stationary – the stationary operand on SBUF; layout: (partition axis `<= 128`, free axis `<= 128`)

  * moving – the moving operand on SBUF; layout: (partition axis `<= 128`, free axis `<= 512`)

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * is_stationary_onezero – hints to the compiler whether the `stationary` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `stationary` tile is in float32; the field has no impact for non-float32 `stationary`.

  * is_moving_onezero – hints to the compiler if the `moving` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `moving` tile is in float32; the field has no impact for non-float32 `moving`.

  * is_transpose – hints to the compiler that this is a transpose operation with `moving` as an identity matrix.

  * tile_position – a 2D tuple (row, column) for the start PE tile position to run `nc_matmul`.

  * tile_size – a 2D tuple (row, column) for the PE tile size to hold by `nc_matmul` starting from `tile_position`.

Returns:
    

a tile on PSUM that has the result of matrix multiplication of `stationary` and `moving` tiles; layout: partition axis comes from free axis of `stationary`, while free axis comes from free axis of `moving`.

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    
    ##################################################################
    # Example 1:
    # multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
    # to get matrix c in PSUM of shape (128, 512)
    ##################################################################
    a_mgrid = nl.mgrid[0:128, 0:128]
    b_mgrid = nl.mgrid[0:128, 0:512]
    c_mgrid = nl.mgrid[0:128, 0:512]
    
    a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
    b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])
    
    c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])
    
    nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)
    
    ##################################################################
    # Example 2:
    # multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
    # to get matrix f in PSUM of shape (128, 512) using psum accumulation
    ##################################################################
    d_mgrid = nl.mgrid[0:128, 0:128]
    e_mgrid = nl.mgrid[0:128, 0:512]
    f_mgrid = nl.mgrid[0:128, 0:512]
    
    f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)
    
    for i_contract in nl.affine_range(2):
      d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
      e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
      f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
      
    nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)
    
    ##################################################################
    # Example 3:
    # perform batched matrix multiplication on matrix g of shape (16, 64, 64) 
    # and matrix h of shape (16, 64, 512) to get matrix i of (16, 64, 512) 
    # using Tensor Engine PE tiling mode. 
    ##################################################################
    g_mgrid = nl.mgrid[0:64, 0:64]
    h_mgrid = nl.mgrid[0:64, 0:512]
    i_mgrid = nl.mgrid[0:64, 0:512]
    
    for i in nl.affine_range(4):
      for j in nl.affine_range(4):
        g = nl.load(g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
        h = nl.load(h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
        i_psum = nisa.nc_matmul(g, h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
        nl.store(i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], i_psum)
    
    return c_tensor, f_tensor, i_tensor
    

## Normalization and Statistics

### nki.language.rms_norm

nki.language.rms_norm(x, w, axis, n, epsilon=1e-06, *, dtype=None, compute_dtype=None, mask=None, **kwargs)
    

Apply Root Mean Square Layer Normalization.

Parameters:
    

  * x – input tile

  * w – weight tile

  * axis – axis along which to compute the root mean square (rms) value

  * n – total number of values to calculate rms

  * epsilon – epsilon value used by rms calculation to avoid divide-by-zero

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * compute_dtype – (optional) dtype for the internal computation - currently `dtype` and `compute_dtype` behave the same, both sets internal compute and return dtype.

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

Returns:
    

`` x / RMS(x) * w ``

---

### nki.isa.bn_stats

nki.isa.bn_stats(data, *, mask=None, dtype=None, **kwargs)
    

Compute mean- and variance-related statistics for each partition of an input tile `data` in parallel using Vector Engine.

The output tile of the instruction has 6 elements per partition:

  * the `count` of the even elements (of the input tile elements from the same partition)

  * the `mean` of the even elements

  * `variance * count` of the even elements

  * the `count` of the odd elements

  * the `mean` of the odd elements

  * `variance * count` of the odd elements

To get the final mean and variance of the input tile, we need to pass the above `bn_stats` instruction output into the bn_aggr instruction, which will output two elements per partition:

  * mean (of the original input tile elements from the same partition)

  * variance

Due to hardware limitation, the number of elements per partition (i.e., free dimension size) of the input `data` must not exceed 512 (nl.tile_size.bn_stats_fmax). To calculate per-partition mean/variance of a tensor with more than 512 elements in free dimension, we can invoke `bn_stats` instructions on each 512-element tile and use a single `bn_aggr` instruction to aggregate `bn_stats` outputs from all the tiles. Refer to Example 2 for an example implementation.

Vector Engine performs the above statistics calculation in float32 precision. Therefore, the engine automatically casts the input `data` tile to float32 before performing float32 computation and is capable of casting the float32 computation results into another data type specified by the `dtype` field, at no additional performance cost. If `dtype` field is not specified, the instruction will cast the float32 results back to the same data type as the input `data` tile.

Estimated instruction cost:

`max(MIN_II, N)` Vector Engine cycles, where `N` is the number of elements per partition in `data` and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – the input tile (up to 512 elements per partition)

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

an output tile with 6-element statistics per partition

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Calculate the mean and variance for each partition
    # of tile a with shape (128, 128)
    ##################################################################
    a: tensor[128, 128] = nl.load(a_tensor)
    stats_a: tensor[128, 6] = nisa.bn_stats(a)
    mean_var_a: tensor[128, 2] = nisa.bn_aggr(stats_a)
    
    # Extract mean and variance
    mean_a = mean_var_a[:, 0]
    var_a = mean_var_a[:, 1]
    nl.store(mean_a_tensor, mean_a)
    nl.store(var_a_tensor, var_a)
    
    # ##################################################################
    # # Example 2: Calculate the mean and variance for each partition of
    # # tile b with shape [128, 1024]
    # ##################################################################
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
    nl.store(var_b_tensor, var_b)
    

---

### nki.isa.bn_aggr

nki.isa.bn_aggr(data, *, mask=None, dtype=None, **kwargs)
    

Aggregate one or multiple `bn_stats` outputs to generate a mean and variance per partition using Vector Engine.

The input `data` tile effectively has an array of `(count, mean, variance*count)` tuples per partition produced by bn_stats instructions. Therefore, the number of elements per partition of `data` must be a modulo of three.

Note, if you need to aggregate multiple `bn_stats` instruction outputs, it is recommended to declare a SBUF tensor and then make each `bn_stats` instruction write its output into the SBUF tensor at different offsets (see example implementation in Example 2 in bn_stats).

Vector Engine performs the statistics aggregation in float32 precision. Therefore, the engine automatically casts the input `data` tile to float32 before performing float32 computation and is capable of casting the float32 computation results into another data type specified by the `dtype` field, at no additional performance cost. If `dtype` field is not specified, the instruction will cast the float32 results back to the same data type as the input `data` tile.

Estimated instruction cost:

`max(MIN_II, 13*(N/3))` Vector Engine cycles, where `N` is the number of elements per partition in `data` and `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

Parameters:
    

  * data – an input tile with results of one or more bn_stats

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

an output tile with two elements per partition: a mean followed by a variance

## Search and Replace

### nki.isa.nc_find_index8

nki.isa.nc_find_index8(*, data, vals, mask=None, dtype=None, **kwargs)
    

Find indices of the 8 given vals in each partition of the data tensor.

This instruction first loads the 8 values, then loads the data tensor and outputs the indices (starting at 0) of the first occurrence of each value in the data tensor, for each partition.

The data tensor can be up to 5-dimensional, while the vals tensor must be up to 3-dimensional. The data tensor must have between 8 and 16,384 elements per partition. The vals tensor must have exactly 8 elements per partition. The output will contain exactly 8 elements per partition and will be uint16 or uint32 type. Default output type is uint32.

Behavior is undefined if vals tensor contains values that are not in the data tensor.

If provided, a mask is applied only to the data tensor.

Estimated instruction cost:

`N` engine cycles, where:

  * `N` is the number of elements per partition in the data tensor

Parameters:
    

  * data – the data tensor to find indices from

  * vals – tensor containing the 8 values per partition whose indices will be found

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – uint16 or uint32

Returns:
    

a 2D tile containing indices (uint16 or uint32) of the 8 values in each partition with shape [par_dim, 8]

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Generate tile b of 32 * 128 random floating point values,
    # find the 8 largest values in each row, then find their indices:
    ##################################################################
    # Generate random data
    data = nl.rand((32, 128))
    
    # Find max 8 values per row
    max_vals = nisa.max8(src=data)
    
    # Create output tensor for indices
    indices_tensor = nl.ndarray([32, 8], dtype=nl.uint32, buffer=nl.shared_hbm)
    
    # Find indices of max values
    indices = nisa.nc_find_index8(data=data, vals=max_vals)
    
    # Store results
    nl.store(indices_tensor, value=indices)
    

---

### nki.isa.nc_match_replace8

nki.isa.nc_match_replace8(*, data, vals, imm, dst_idx=None, mask=None, dtype=None, **kwargs)
    

Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced tensor. If `dst_idx` tile is provided, the indices of the matched values are written to `dst_idx`.

This instruction reads the input `data`, replaces the first occurrence of each of the given values (from `vals` tensor) with the specified immediate constant and, optionally, output indices of matched values to `dst_idx`. When performing the operation, the free dimensions of both `data` and `vals` are flattened. However, these dimensions are preserved in the replaced output tensor and in `dst_idx` respectively. The partition dimension defines the parallelization boundary. Match, replace, and index generation operations execute independently within each partition.

The `data` tensor can be up to 5-dimensional, while the `vals` tensor can be up to 3-dimensional. The `vals` tensor must have exactly 8 elements per partition. The data tensor must have no more than 16,384 elements per partition. The replaced output will have the same shape as the input data tensor. `data` and `vals` must have the same number of partitions. Both input tensors can come from SBUF or PSUM.

Behavior is undefined if vals tensor contains values that are not in the data tensor.

If provided, a mask is applied to the data tensor.

Estimated instruction cost:

`min(MIN_II, N)` engine cycles, where:

  * `N` is the number of elements per partition in the data tensor

  * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.

NumPy equivalent:
    
    
    # Let's assume we work with NumPy, and ``data``, ``vals`` are 2-dimensional arrays
    # (with shape[0] being the partition axis) and imm is a constant float32 value.
    
    import numpy as np
    
    # Get original shapes
    data_shape = data.shape
    vals_shape = vals.shape
    
    # Reshape to 2D while preserving first dimension
    data_2d = data.reshape(data_shape[0], -1)
    vals_2d = vals.reshape(vals_shape[0], -1)
    
    # Initialize output array for indices
    indices = np.zeros(vals_2d.shape, dtype=np.uint32)
    
    for i in range(data_2d.shape[0]):
      for j in range(vals_2d.shape[1]):
        val = vals_2d[i, j]
        # Find first occurrence of val in data_2d[i, :]
        matches = np.where(data_2d[i, :] == val)[0]
        if matches.size > 0:
          indices[i, j] = matches[0]  # Take first match
          data_2d[i, matches[0]] = imm
    
    output = data_2d.reshape(data.shape)
    indices = indices.reshape(vals.shape) # Computed only if ``dst_idx`` is specified
    

Parameters:
    

  * data – the data tensor to modify

  * dst_idx – (optional) the destination tile to write flattened indices of matched values

  * vals – tensor containing the 8 values per partition to replace

  * imm – float32 constant to replace matched values with

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

the modified data tensor

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    
    
    ##################################################################
    # Example 1: Generate tile a of random floating point values,
    # get the 8 largest values in each row, then replace their first
    # occurrences with -inf:
    ##################################################################
    N = 4
    M = 16
    data_tile = nl.rand((N, M))
    max_vals = nisa.max8(src=data_tile)
    
    result = nisa.nc_match_replace8(data=data_tile[:, :], vals=max_vals, imm=float('-inf'))
    result_tensor = nl.ndarray([N, M], dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(result_tensor, value=result)
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    
    
    ##################################################################
    # Example 2: Read the 8 largest values in each row of the tensor,
    # replace the first occurrence with imm, write indices, and return
    # the replaced output.
    ##################################################################
    n, m = in_tensor.shape
    
    dst_idx = nl.ndarray((n, 8), dtype=idx_tensor.dtype)
    
    ix, iy = nl.mgrid[0:n, 0:8]
    
    inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
    max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)
    
    out_tile = nisa.nc_match_replace8(
      dst_idx=dst_idx[ix, iy], data=inp_tile[:, :], vals=max_vals, imm=imm
    )
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    
    
    ##################################################################
    # Example 3: Read the 8 largest values in each row of the tensor,
    # after applying the specified mask, replace the first occurrence
    # with imm, write indices, and return the replaced output.
    ##################################################################
    n, m = in_tensor.shape
    
    idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)
    
    ix, iy = nl.mgrid[0:n, 0:m]
    inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
    max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile[ix, iy], mask=(ix < n //2 and iy < m//2))
    
    out_tile = nisa.nc_match_replace8(
      dst_idx=idx_tile[:, :],
      data=inp_tile[ix, iy],
      vals=max_vals,
      imm=imm,
      mask=(ix < n // 2 and iy < m // 2),  # mask applies to `data`
    )
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    
    
    ##################################################################
    # Example 4: Read the 8 largest values in each row of the tensor,
    # replace the first occurrence with 0.0, write indices, and return 
    # the replaced output.
    ##################################################################
    n, b, m = data_tensor.shape
    
    n, b, m = data_tensor.shape
    
    out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
    idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
    
    imm = 0.0
    idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)
    out_tile = nisa.memset(shape=(n, b, m), value=0, dtype=data_tensor.dtype)
    
    iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
    ip, io = nl.mgrid[0:n, 0:8]
    
    inp_tile = nl.load(data_tensor[iq, ir, iw])
    max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)
    
    out_tile[iq, ir, iw] = nisa.nc_match_replace8(
      dst_idx=idx_tile[ip, io],
      data=inp_tile[iq, ir, iw],
      vals=max_vals[ip, io],
      imm=imm,
    )
    
    
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    
    
    ##################################################################
    # Example 5: Read the 8 largest values in each row of the tensor,
    # replace the first occurrence with 0.0 in-place and write indices.
    ##################################################################
    n, b, m = data_tensor.shape
    
    n, b, m = data_tensor.shape
    
    out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
    idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
    
    imm = 0.0
    idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)
    
    iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
    ip, io = nl.mgrid[0:n, 0:8]
    
    inp_tile = nl.load(data_tensor[iq, ir, iw])
    max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)
    
    inp_tile[iq, ir, iw] = nisa.nc_match_replace8(
      dst_idx=idx_tile[ip, io],
      data=inp_tile[iq, ir, iw],
      vals=max_vals[ip, io],
      imm=imm,
    )
    

---

### nki.isa.max8

nki.isa.max8(*, src, mask=None, dtype=None, **kwargs)
    

Find the 8 largest values in each partition of the source tile.

This instruction reads the input elements, converts them to fp32 internally, and outputs the 8 largest values in descending order for each partition. By default, returns the same dtype as the input tensor.

The source tile can be up to 5-dimensional, while the output tile is always 2-dimensional. The number of elements read per partition must be between 8 and 16,384 inclusive. The output will always contain exactly 8 elements per partition. The source and output must have the same partition dimension size:

  * source: [par_dim, …]

  * output: [par_dim, 8]

Estimated instruction cost:

`N` engine cycles, where:

  * `N` is the number of elements per partition in the source tile

Parameters:
    

  * src – the source tile to find maximum values from

  * mask – (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see NKI API Masking for details)

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a 2D tile containing the 8 largest values per partition in descending order with shape [par_dim, 8]

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ##################################################################
    # Example 1: Generate tile b of 32 * 128 random floating point values
    # and get the 8 largest values in each row:
    ##################################################################
    expr_a = nl.rand((32, 128))
    a = nisa.max8(src=expr_a)
    
    a_tensor = nl.ndarray([32, 8], dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(a_tensor, value=a)
    

---

### nki.isa.sequence_bounds

nki.isa.sequence_bounds(*, segment_ids, dtype=None)
    

Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.

Given a tile of segment IDs, this function identifies where each segment begins and ends. For each element, it returns a pair of values: [start_index, end_index] indicating the boundaries of the segment that element belongs to. All segment IDs must be non-negative integers. Padding elements (with segment ID of zero) receive special boundary values: a start index of n and an end index of (-1), where n is the length of `segment_ids`.

The output tile contains two values per input element: the start index (first column) and end index (second column) of each segment. The partition dimension must always be 1. For example, with input shape (1, 512), the output shape becomes (1, 2, 512), where the additional dimension holds the start and end indices for each element.

The input tile (`segment_ids`) must have data type np.float32 or np.int32. The output tile data type is specified using the `dtype` field (must be np.float32 or np.int32). If `dtype` is not specified, the output data type will be the same as the input data type of `segment_ids`.

NumPy equivalent:
    
    
    def compute_sequence_bounds(sequence):
      n = len(sequence)
    
      min_bounds = np.zeros(n, dtype=sequence.dtype)
      max_bounds = np.zeros(n, dtype=sequence.dtype)
    
      min_bound_pad = n
      max_bound_pad = -1
    
      min_bounds[0] = 0 if sequence[0] != 0 else min_bound_pad
      for i in range(1, n):
        if sequence[i] == 0:
          min_bounds[i] = min_bound_pad
        elif sequence[i] == sequence[i - 1]:
          min_bounds[i] = min_bounds[i - 1]
        else:
          min_bounds[i] = i
    
      max_bounds[-1] = n if sequence[-1] != 0 else max_bound_pad
      for i in range(n - 2, -1, -1):
        if sequence[i] == 0:
          max_bounds[i] = max_bound_pad
        elif sequence[i] == sequence[i + 1]:
          max_bounds[i] = max_bounds[i + 1]
        else:
          max_bounds[i] = i + 1
    
      return np.vstack((min_bounds, max_bounds))
    
    b = (
      np.apply_along_axis(
        compute_sequence_bounds, axis=1, arr=reshaped_segment_ids
      )
      .reshape(m, 2, n)
      .astype(np.float32)
    )
    

Parameters:
    

  * segment_ids – tile containing the segment IDs. Elements with ID=0 are treated as padding.

  * dtype – data type of the output (must be np.float32 or np.int32)

Returns:
    

tile containing the sequence bounds.

Example:
    
    
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.typing import tensor
    
    ######################################################################
    # Example 1: Generate tile of boundaries of sequence for each element:
    ######################################################################
    # Input example
    # segment_ids = np.array([[0, 1, 1, 2, 2, 2, 0, 3, 3]], dtype=np.int32)
    
    # Expected output for this example:
    # [[
    #   [9, 1, 1, 3, 3, 3, 9, 7, 7]       # start index
    #   [-1, 3, 3, 6, 6, 6, -1, 9, 9]     # end index
    #   ]]
    m, n = segment_ids.shape
    
    ix, iy, iz = nl.mgrid[0:m, 0:2, 0:n]
    
    out_tile = nl.ndarray([m, 2, n], dtype=segment_ids.dtype, buffer=nl.sbuf)
    seq_tile = nl.load(segment_ids)
    out_tile[ix, iy, iz] = nisa.sequence_bounds(segment_ids=seq_tile)
    

## Control Flow and Loop Iterators

### nki.language.affine_range

# nki.language.affine_range

##  Contents 

# nki.language.affine_range
nki.language.affine_range(*args, **kwargs)
    

Create a sequence of numbers for use as parallel loop iterators in NKI. `affine_range` should be the default loop iterator choice, when there is no loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. A concrete example of associative reduction is multiple nl.matmul or nisa.nc_matmul calls accumulating into the same output buffer defined outside of this loop level (see code example #2 below).

When the above conditions are not met, we recommend using sequential_range instead.

Notes:

  * Using `affine_range` prevents Neuron compiler from unrolling the loops until entering compiler backend, which typically results in better compilation time compared to the fully unrolled iterator static_range.

  * Using `affine_range` also allows Neuron compiler to perform additional loop-level optimizations, such as loop vectorization in current release. The exact type of loop-level optimizations applied is subject to changes in future releases.

  * Since each kernel instance only runs on a single NeuronCore, affine_range does not parallelize different loop iterations across multiple NeuronCores. However, different iterations could be parallelized/pipelined on different compute engines within a NeuronCore depending on the invoked instructions (engines) and data dependency in the loop body.

    
    
     1import neuronxcc.nki.language as nl
     2
     3#######################################################################
     4# Example 1: No loop carried dependency
     5# Input/Output tensor shape: [128, 2048]
     6# Load one tile ([128, 512]) at a time, square the tensor element-wise,
     7# and store it into output tile
     8#######################################################################
     9
    10# Every loop instance works on an independent input/output tile.
    11# No data dependency between loop instances.
    12for i_input in nl.affine_range(input.shape[1] // 512):
    13  offset = i_input * 512
    14  input_sb = nl.load(input[0:input.shape[0], offset:offset+512])
    15  result = nl.multiply(input_sb, input_sb)
    16  nl.store(output[0:input.shape[0], offset:offset+512], result)
    17
    18#######################################################################
    19# Example 2: Matmul output buffer accumulation, a type of associative reduction
    20# Input tensor shapes for nl.matmul: xT[K=2048, M=128] and y[K=2048, N=128]
    21# Load one tile ([128, 128]) from both xT and y at a time, matmul and
    22# accumulate into the same output buffer
    23#######################################################################
    24
    25result_psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
    26for i_K in nl.affine_range(xT.shape[0] // 128):
    27  offset = i_K * 128
    28  xT_sbuf = nl.load(offset:offset+128, 0:xT.shape[1]])
    29  y_sbuf = nl.load(offset:offset+128, 0:y.shape[1]])
    30
    31  result_psum += nl.matmul(xT_sbuf, y_sbuf, transpose_x=True)
    

---

### nki.language.sequential_range

nki.language.sequential_range(*args, **kwargs)
    

Create a sequence of numbers for use as sequential loop iterators in NKI. `sequential_range` should be used when there is a loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. See affine_range for an example of such associative reduction.

Notes:

  * Inside a NKI kernel, any use of Python `range(...)` will be replaced with `sequential_range(...)` by Neuron compiler.

  * Using `sequential_range` prevents Neuron compiler from unrolling the loops until entering compiler backend, which typically results in better compilation time compared to the fully unrolled iterator static_range.

  * Using `sequential_range` informs Neuron compiler to respect inter-loop dependency and perform much more conservative loop-level optimizations compared to `affine_range`.

  * Using `affine_range` instead of `sequential_range` in case of loop carried dependency incorrectly is considered unsafe and could lead to numerical errors.

    
    
     1import neuronxcc.nki.language as nl
     2
     3#######################################################################
     4# Example 1: Loop carried dependency from tiling tensor_tensor_scan
     5# Both sbuf tensor input0 and input1 shapes: [128, 2048]
     6# Perform a scan operation between the two inputs using a tile size of [128, 512]
     7# Store the scan output to another [128, 2048] tensor
     8#######################################################################
     9
    10# Loop iterations communicate through this init tensor
    11init = nl.zeros((128, 1), dtype=input0.dtype)
    12
    13# This loop will only produce correct results if the iterations are performed in order
    14for i_input in nl.sequential_range(input0.shape[1] // 512):
    15  offset = i_input * 512
    16
    17  # Depends on scan result from the previous loop iteration
    18  result = nisa.tensor_tensor_scan(input0[:, offset:offset+512],
    19                                   input1[:, offset:offset+512],
    20                                   initial=init,
    21                                   op0=nl.multiply, op1=nl.add)
    22
    23  nl.store(output[0:input0.shape[0], offset:offset+512], result)
    24
    25  # Prepare initial result for scan in the next loop iteration
    26  init[:, :] = result[:, 511]
    

---

### nki.language.static_range

nki.language.static_range(*args)
    

Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop. Unlike affine_range or sequential_range, Neuron compiler will fully unroll the loop during NKI kernel tracing.

Notes:

  * Due to loop unrolling, compilation time may go up significantly compared to affine_range or sequential_range.

  * On-chip memory (SBUF) usage may also go up significantly compared to affine_range or sequential_range.

  * No loop-level optimizations will be performed in the compiler.

  * `static_range` should only be used as a fall-back option for debugging purposes when affine_range or sequential_range is giving functionally incorrect results or undesirable performance characteristics.

---

### nki.language.program_id

nki.language.program_id(axis)
    

Index of the current SPMD program along the given axis in the launch grid.

Parameters:
    

axis – The axis of the ND launch grid.

Returns:
    

The program id along `axis` in the launch grid

---

### nki.language.num_programs

nki.language.num_programs(axes=None)
    

Number of SPMD programs along the given axes in the launch grid. If `axes` is not provided, returns the total number of programs.

Parameters:
    

axes – The axes of the ND launch grid. If not provided, returns the total number of programs along the entire launch grid.

Returns:
    

The number of SPMD(single process multiple data) programs along `axes` in the launch grid

## Compiler Directives and Kernel Decorators

### nki.jit

nki.jit(func=None, mode='auto', **kwargs)
    

This decorator compiles a function to run on NeuronDevices.

This decorator tries to automatically detect the current framework and compile the function as a custom operator of the current framework. To bypass the framework detection logic, you may specify the `mode` parameter explicitly.

Parameters:
    

  * func – The function that define the custom op

  * mode – The compilation mode, possible values: “jax”, “torchxla”, “baremetal”, “benchmark”, “simulation” and “auto”

Listing 13 An Example#
    
    
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    
    @nki.jit
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
      a = nl.load(a_tensor)
      b = nl.load(b_tensor)
    
      c = a + b
    
      nl.store(c_tensor, c)
    
      return c_tensor
    

---

### nki.baremetal

nki.baremetal(kernel=None, **kwargs)
    

Compile and run a NKI kernel on NeuronDevice without involving ML frameworks such as PyTorch and JAX. If you decorate your NKI kernel function with decorator `@nki.baremetal(...)`, you may call the NKI kernel function directly just like any other Python function. You must run this API on a Trn/Inf instance with NeuronDevices (v2 or beyond) attached.

Note

The decorated function using `nki.baremetal` expects numpy.ndarray as input/output tensors instead of ML framework tensor objects.

This decorator compiles the NKI kernel into an executable on NeuronDevices (`NEFF`) and also collects an execution trace (`NTFF`) by running the `NEFF` on the local NeuronDevice. See Profiling NKI kernels with Neuron Profile for more information on how to visualize the execution trace for profiling purposes.

Since `nki.baremetal` runs the compiled NEFF without invoking any ML framework, it is the fastest way to compile and run any NKI kernel standalone on NeuronDevice. Therefore, this decorator is useful for quickly iterating an early implementation of a NKI kernel to reach functional correctness before porting it to the ML framework and injecting the kernel into the full ML model. To iterate over NKI kernel performance quickly, NKI also provides nki.benchmark decorator which uses the same underlying mechanism as `nki.baremetal` but additionally collects latency statistics in different percentiles.

Parameters:
    

  * save_neff_name – A file path to save your NEFF file. By default, this is unspecified, and the NEFF file will be deleted automatically after execution.

  * save_trace_name – A file path to save your NTFF file. By default, this is unspecified, and the NTFF file will be deleted automatically after execution. Known issue: if `save_trace_name` is specified, `save_neff_name` must be set to “file.neff”.

  * additional_compile_opt – Additional Neuron compiler flags to pass in when compiling the kernel.

  * artifacts_dir – A directory path to save Neuron compiler artifacts. The directory must be empty before running the kernel. A non-empty directory would lead to a compilation error.

Returns:
    

None

Listing 16 An Example#
    
    
    from neuronxcc.nki import baremetal
    import neuronxcc.nki.language as nl
    import numpy as np
    
    @baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
      a = nl.load(a_tensor)
      b = nl.load(b_tensor)
    
      c = a + b
    
      nl.store(c_tensor, c)
    
      return c_tensor
    
    a = np.zeros([128, 1024], dtype=np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = nki_tensor_tensor_add(a, b)
    
    assert np.allclose(c, a + b)
    

---

### nki.benchmark

nki.benchmark(kernel=None, **kwargs)
    

Benchmark a NKI kernel on a NeuronDevice by using `nki.benchmark` as a decorator. You must run this API on a Trn/Inf instance with NeuronDevices (v2 or beyond) attached and also `aws-neuronx-tools` installed on the host using the following steps:
    
    
    # on Ubuntu
    sudo apt-get install aws-neuronx-tools=2.* -y
    
    # on Amazon Linux
    sudo dnf install aws-neuronx-tools-2.* -y
    

You may specify a path to save your NEFF file through input parameter `save_neff_name` and a path to save your NTFF file through `save_trace_name`. See Profiling NKI kernels with Neuron Profile for more information on how to visualize the execution trace for profiling purposes.

Note

Similar to `nki.baremetal`, The decorated function using `nki.benchmark` expects numpy.ndarray as input/output tensors instead of ML framework tensor objects.

In additional to generating NEFF/NTFF files, this decorator also invokes `neuron-bench` to collect execution latency statistics of the NEFF file and prints the statistics to the console.

`neuron-bench` is a tool that launches the NEFF file on a NeuronDevice in a loop to collect end-to-end latency statistics. You may specify the number of warm-up iterations to skip benchmarking in input parameter `warmup`, and the number of benchmarking iterations in `iters`. Currently, `nki.benchmark` only supports benchmarking on a single NeuronCore, since NKI not yet supports collective compute. Note, `neuron-bench` measures not only the device latency but also the time taken to transfer data between host and device. However, the tool does not rely on any ML framework to launch the NEFF and therefore reports NEFF latency without any framework overhead.

Parameters:
    

  * warmup – The number of iterations for warmup execution (10 by default).

  * iters – The number of iterations for benchmarking (100 by default).

  * save_neff_name – Save the compiled neff file if specify a name (unspecified by default).

  * save_trace_name – Save the trace (profile) file if specified a name (unspecified by default); at the moment, it requires that the save_neff_name is unspecified or specified as ‘file.neff’.

  * additional_compile_opt – Additional Neuron compiler flags to pass in when compiling the kernel.

Returns:
    

A function object that wraps the decorating function. A property `benchmark_result.nc_latency` is available after invocation. `get_latency_percentile(int)` of the property returns the specified percentile latency in microsecond(us). Available percentiles: [0, 1, 10, 25, 50, 90, 99, 100]

Listing 14 An Example#
    
    
    from neuronxcc.nki import benchmark
    import neuronxcc.nki.language as nl
    import numpy as np
    
    @benchmark(warmup=10, iters = 100, save_neff_name='file.neff', save_trace_name='profile.ntff')
    def nki_tensor_tensor_add(a_tensor, b_tensor):
      c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
      a = nl.load(a_tensor)
      b = nl.load(b_tensor)
    
      c = a + b
    
      nl.store(c_tensor, c)
    
      return c_tensor
    
    a = np.zeros([128, 1024], dtype=np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = nki_tensor_tensor_add(a, b)
    
    metrics = nki_tensor_tensor_add.benchmark_result.nc_latency
    print("latency.p50 = " + str(metrics.get_latency_percentile(50)))
    print("latency.p99 = " + str(metrics.get_latency_percentile(99)))
    

---

### nki.compiler.enable_stack_allocator

nki.compiler.enable_stack_allocator(func=None, log_level=50)
    

Use stack allocator to allocate the psum and sbuf tensors in the kernel.

Must use together with skip_middle_end_transformations.
    
    
    from neuronxcc import nki
    
    @nki.compiler.enable_stack_allocator
    @nki.compiler.skip_middle_end_transformations
    @nki.jit
    def kernel(...):
      ...
    

---

### nki.compiler.force_auto_alloc

nki.compiler.force_auto_alloc(func=None)
    

Force automatic allocation to be turned on in the kernel.

This will ignore any direct allocation inside the kernel


---

### nki.compiler.skip_middle_end_transformations

nki.compiler.skip_middle_end_transformations(func=None)
    

Skip all middle end transformations on the kernel


---

### nki.isa.reduce_cmd

class nki.isa.reduce_cmd(value)
    

Engine Register Reduce commands

Attributes

`idle` | Not using the accumulator registers  
---|---  
`reset` | Resets the accumulator registers to its initial state  
`reset_reduce` | Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators  
`reduce` | keeps accumulating over the current value of the accumulator registers  