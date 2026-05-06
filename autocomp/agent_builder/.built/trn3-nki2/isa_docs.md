## Kernel Launch and Simulation

### nki.jit

nki.jit(func=None, mode='auto', **kwargs)
    

This decorator compiles a top-level NKI function to run on NeuronDevices.

This decorator tries to automatically detect the current framework and compile the function as a custom operator. To bypass the framework detection logic, you can specify the `mode` parameter explicitly.

You might need to explicitly set the target platform using the `NEURON_PLATFORM_TARGET_OVERRIDE` environment variable. Supported values are “trn1”/”gen2”, “trn2”/”gen3”, and “trn3”/”gen4”.

Parameters:
    

  * func – Function that defines the custom operation.

  * mode – Compilation mode. Supported values are “jax”, “torchxla”, and “auto”. (Default: “auto”.)

Listing 6 Writing an addition kernel using `@nki.jit`#
    
    
     @nki.jit()
     def nki_tensor_add_kernel(a_input, b_input):
         # Check both input tensor shapes are the same for element-wise operation.
         assert a_input.shape == b_input.shape
    
         # Check the first dimension's size to ensure it does not exceed on-chip
         # memory tile size, since this simple kernel does not tile inputs.
         assert a_input.shape[0] <= nl.tile_size.pmax
    
         # Allocate space for the input tensors in SBUF and copy the inputs from HBM
         # to SBUF with DMA copy.
         a_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
         nisa.dma_copy(dst=a_tile, src=a_input)
    
         b_tile = nl.ndarray(dtype=b_input.dtype, shape=b_input.shape, buffer=nl.sbuf)
         nisa.dma_copy(dst=b_tile, src=b_input)
    
         # Allocate space for the result and use tensor_tensor to perform
         # element-wise addition. Note: the first argument of 'tensor_tensor'
         # is the destination tensor.
         c_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
         nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
    
         # Create a tensor in HBM and copy the result into HBM.
         c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.hbm)
         nisa.dma_copy(dst=c_output, src=c_tile)
    
         # Return kernel output as function output.
         return c_output

### nki.jit

    @nki.jit
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """
    

Add the `nki_tensor_add_kernel` function definition above. Make sure you annotate it with the `@nki.jit` decorator as in the example above.

### nki.jit

    @nki.jit
    def kernel(x,y,z):
      # this is NKI code

---

### nki.simulate

### NKI CPU Simulator
NKI 0.3.0 introduces `nki.simulate(kernel)`, which executes NKI kernels entirely on CPU without requiring NeuronDevice hardware. The simulator interprets NKI operations using NumPy, producing numerically equivalent results to on-device execution (with minor floating-point differences due to CPU vs NeuronCore arithmetic). This enables local development, debugging, and functional correctness testing on any machine — including laptops and CI environments.

Note

The NKI CPU Simulator is experimental in NKI 0.3.0.

The simulator can be invoked in two ways:

  1. Set the environment variable `NKI_SIMULATOR=1` to run existing kernels without code changes:

    
    
    NKI_SIMULATOR=1 python my_script.py
    

  2. Wrap the kernel call with `nki.simulate`:

    
    
    import nki
    import numpy as np
    
    @nki.jit
    def my_kernel(X, Y):
        ...
    
    # Run on CPU — no Neuron device needed
    X = np.random.randn(128, 512).astype(np.float16)
    Y = np.zeros((128, 512), dtype=np.float16)
    nki.simulate(my_kernel)(X, Y)
    

## Data Types and Constants

### bool_

`bool_` | Boolean (True or False) stored as a byte  

---

### int8

`int8` | 8-bit signed integer number  

---

### int16

`int16` | 16-bit signed integer number  

---

### int32

`int32` | 32-bit signed integer number  

---

### uint8

`uint8` | 8-bit unsigned integer number  

---

### uint16

`uint16` | 16-bit unsigned integer number  

---

### uint32

`uint32` | 32-bit unsigned integer number  

---

### float16

`float16` | 16-bit floating-point number  

---

### float32

`float32` | 32-bit floating-point number  

---

### bfloat16

`bfloat16` | 16-bit floating-point number (1S,8E,7M)  

---

### tfloat32

`tfloat32` | 32-bit floating-point number (1S,8E,10M)  

---

### float8_e4m3

`float8_e4m3` | 8-bit floating-point number (1S,4E,3M)  

---

### float8_e5m2

`float8_e5m2` | 8-bit floating-point number (1S,5E,2M)  

---

### float8_e4m3fn

`float8_e4m3fn` | no inf, NaN represented by 0bS111'1111  

---

### float8_e5m2_x4

`float8_e5m2_x4` | 4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### float8_e4m3fn_x4

`float8_e4m3fn_x4` | 4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### float4_e2m1fn_x4

`float4_e2m1fn_x4` | 4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### nki.language.int8

# nki.language.int8
nki.language.int8 = 'int8'#
    

8-bit signed integer number

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.bfloat16

# nki.language.bfloat16
nki.language.bfloat16 = 'bfloat16'#
    

16-bit floating-point number (1S,8E,7M)

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.tfloat32

nki.language.tfloat32 = 'tfloat32'#
    

32-bit floating-point number (1S,8E,10M)

---

### nki.language.float8_e4m3

# nki.language.float8_e4m3
nki.language.float8_e4m3 = 'float8_e4m3'#
    

8-bit floating-point number (1S,4E,3M)

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.float8_e5m2

# nki.language.float8_e5m2
nki.language.float8_e5m2 = 'float8_e5m2'#
    

8-bit floating-point number (1S,5E,2M)

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.float8_e4m3fn

# nki.language.float8_e4m3fn
nki.language.float8_e4m3fn = 'float8_e4m3fn'#
    

no inf, NaN represented by 0bS111’1111

Type:
    

8-bit floating-point number (1S,4E,3M), Extended range

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.float8_e4m3fn_x4

nki.language.float8_e4m3fn_x4 = 'float8_e4m3fn_x4'#
    

4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

---

### nki.language.float8_e5m2_x4

# nki.language.float8_e5m2_x4
nki.language.float8_e5m2_x4 = 'float8_e5m2_x4'#
    

4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.float4_e2m1fn_x4

nki.language.float4_e2m1fn_x4 = 'float4_e2m1fn_x4'#
    

4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

---

### Supported Data Types by NKI

## Supported Data Types
Supported Data Types by NKI below lists all supported data types by NKI. Almost all of the NKI APIs accept a data type field, dtype, which must be a nki.language data type.

Table 15 Supported Data Types by NKI# | Data Type | Accepted `dtype` Field by NKI APIs  
---|---|---  
Integer | 8-bit unsigned integer | `nki.language.uint8`  
8-bit signed integer | `nki.language.int8`  
16-bit unsigned integer | `nki.language.uint16`  
16-bit signed integer | `nki.language.int16`  
32-bit unsigned integer | `nki.language.uint32`  
32-bit signed integer | `nki.language.int32`  
Float | float8_e4m3 (1S,4E,3M) [2] | `nki.language.float8_e4m3`  
float8_e5m2 (1S,5E,2M) | `nki.language.float8_e5m2`  
float16 (1S,5E,10M) | `nki.language.float16`  
bfloat16 (1S,8E,7M) | `nki.language.bfloat16`  
tfloat32 (1S,8E,10M) | `nki.language.tfloat32`  
float32 (1S,8E,23M) | `nki.language.float32`  
Boolean | boolean stored as uint8 | `nki.language.bool_`  

## Memory Buffers and Regions

### psum

`psum` | Memory region constants for NKI tensors.  

---

### sbuf

`sbuf` | Memory region constants for NKI tensors.  

---

### hbm

`hbm` | Memory region constants for NKI tensors.  

---

### private_hbm

`private_hbm` | Memory region constants for NKI tensors.  

---

### shared_hbm

`shared_hbm` | Memory region constants for NKI tensors.  

---

### nl.shared_hbm

    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)
    

You use `nl.ndarray` with `buffer=nl.shared_hbm` to create tensors in HBM, similar to how you allocated space in SBUF with `buffer=nl.sbuf`. You then copy the result in `c_tile` into `c_output`. Remember that `c_output` is the destination and `c_tile` is the source for the `dma_copy` instruction. The copy is needed because outputs, like inputs, need to be in HBM.

### nl.shared_hbm

    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

### nl.shared_hbm

### Output Tensors Must Use `nl.shared_hbm`
All kernel output (return) tensors must be allocated with `buffer=nl.shared_hbm`. Using `nl.hbm` for output tensors will cause compilation failures.
    
    
    # Beta 2
    output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.hbm)
    
    # NKI 0.3.0
    output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.shared_hbm)
    

---

### nl.sbuf

    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    
    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)
    

The `nl.ndarray` function allows you to allocate tensors in SBUF. Here you allocate `a_tile` and `b_tile` and use the `nisa.dma_copy` instruction to copy tensors between HBM and SBUF memories. You first supply the destination for the copy, `a_tile` and `b_tile`. Then you provide the source for the copy, `a_input` and `b_input`, as seen in this example.

### nl.sbuf

    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

---

### is_psum

`is_psum` | Check if buffer is PSUM.  

---

### is_sbuf

`is_sbuf` | Check if buffer is SBUF.  

---

### is_hbm

`is_hbm` | Check if buffer is any HBM type.  

---

### is_on_chip

`is_on_chip` | Check if buffer is on-chip (SBUF or PSUM).  

---

### nki.language.psum

# nki.language.psum
nki.language.psum = MemoryRegion.psum#
    

Memory region constants for NKI tensors.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.hbm

# nki.language.hbm
nki.language.hbm = MemoryRegion.private_hbm#
    

Memory region constants for NKI tensors.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.private_hbm

# nki.language.private_hbm
nki.language.private_hbm = MemoryRegion.private_hbm#
    

Memory region constants for NKI tensors.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.shared_hbm

# nki.language.shared_hbm
nki.language.shared_hbm = MemoryRegion.shared_hbm#
    

Memory region constants for NKI tensors.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.is_psum

nki.language.is_psum(buffer)
    

Check if buffer is PSUM.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.is_sbuf

nki.language.is_sbuf(buffer)
    

Check if buffer is SBUF.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.is_hbm

nki.language.is_hbm(buffer)
    

Check if buffer is any HBM type.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.is_on_chip

nki.language.is_on_chip(buffer)
    

Check if buffer is on-chip (SBUF or PSUM).

This document is relevant for: `Trn2`, `Trn3`

## Hardware Constants and Tile Sizes

### tile_size

tile_size | Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)  

---

### nl.tile_size.pmax

    assert a_input.shape[0] <= nl.tile_size.pmax
    

The first assertion checks that `a_input` and `b_input` have the same shape. The second assertion checks that the inputs will fit in within the tile size of the on-chip memory. If an input is larger than the on-chip tile size, you must tile the input. To keep this example simple we will avoid discussing tiling further in this quick start.

---

### nki.language.tile_size

class nki.language.tile_size
    

Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)

Attributes

`pmax` | Maximum partition dimension of a tile  
---|---  
`psum_fmax` | Maximum free dimension of a tile on PSUM buffer  
`gemm_stationary_fmax` | Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine  
`gemm_moving_fmax` | Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine  
`bn_stats_fmax` | Maximum free dimension of BN_STATS  
`psum_min_align` | Minimum byte alignment requirement for PSUM free dimension address  
`sbuf_min_align` | Minimum byte alignment requirement for SBUF free dimension address  
`total_available_sbuf_size` | Usable SBUF size per partition (total minus reserved bytes).  

## Tensor Allocation and Initialization

### ndarray

`ndarray` | Create a new tensor of given shape and dtype on the specified buffer.  

---

### zeros

`zeros` | Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.  

---

### ones

`ones` | Create a new tensor of given shape and dtype on the specified buffer, filled with ones.  

---

### full

`full` | Create a new tensor of given shape and dtype on the specified buffer, filled with initial value.  

---

### zeros_like

`zeros_like` | Create a new tensor of zeros with the same shape and type as a given tensor.  

---

### empty_like

`empty_like` | Create a new tensor with the same shape and type as a given tensor.  

---

### rand

`rand` | Create a new tensor of given shape and dtype on the specified buffer, filled with random values.  

---

### random_seed

`random_seed` | Set the random seed for random number generation.  

---

### shared_identity_matrix

`shared_identity_matrix` | Create an identity matrix in SBUF with the specified data type.  

---

### nl.ndarray

    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    
    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)
    

The `nl.ndarray` function allows you to allocate tensors in SBUF. Here you allocate `a_tile` and `b_tile` and use the `nisa.dma_copy` instruction to copy tensors between HBM and SBUF memories. You first supply the destination for the copy, `a_tile` and `b_tile`. Then you provide the source for the copy, `a_input` and `b_input`, as seen in this example.

### nl.ndarray

    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

### nl.ndarray

    a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result",
      address=(0, 128), buffer=nl.sbuf)

---

### nl.ndarray with address parameter

    # creates your buffer on parition 0, offset by 128 elements of your data type
    a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result",
      address=(0, 128), buffer=nl.sbuf)
    

---

### PSUM tensor allocation

For example, the following code will allocate a PSUM tensor on bank 3:
    
    
    bank_id = 3
    PSUM_BANK_SIZE = 2048
    psum_t = nl.ndarray(dtype=nl.bfloat16, shape=(128, 1024),
      address=(0, bank_id*PSUM_BANK_SIZE))
    

---

### nki.language.ndarray

nki.language.ndarray(shape, dtype, buffer=MemoryRegion.sbuf, name='', address=None)
    

Create a new tensor of given shape and dtype on the specified buffer.

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor.

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

  * address – optional memory address `(partition_offset, free_offset)`.

Returns:
    

a new `NkiTensor` allocated on the buffer.

### nki.language.ndarray

### Address Placement
The `address` parameter was added to `nki.language.ndarray` as an optional parameter for explicit memory placement.
    
    
    buf = nl.ndarray((128, 512), dtype=nl.float16, address=(p_off, f_off))  # explicit placement
    

---

### nki.language.zeros

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

  * `nki.language.ndarray` — `buffer` default changed from `None` to `nl.sbuf`

  * `nki.language.zeros` — `buffer` default changed from `None` to `nl.sbuf`

  * `nki.language.sequential_range` — `stop` and `step` now have default values (`None` and `1`)

### nki.language.zeros

nki.language.zeros(shape, dtype, buffer=MemoryRegion.sbuf, name='')
    

Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.

((Similar to numpy.zeros))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor.

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` allocated on the buffer.

---

### nki.language.ones

nki.language.ones(shape, dtype, buffer=MemoryRegion.sbuf, name='')
    

Create a new tensor of given shape and dtype on the specified buffer, filled with ones.

((Similar to numpy.ones))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor.

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` allocated on the buffer.

---

### nki.language.full

nki.language.full(shape, fill_value, dtype, buffer=MemoryRegion.sbuf, name='')
    

Create a new tensor of given shape and dtype on the specified buffer, filled with initial value.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * shape – the shape of the tensor.

  * fill_value – the value to fill the tensor with.

  * dtype – the data type of the tensor.

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` allocated on the buffer.

---

### nki.language.zeros_like

nki.language.zeros_like(x, dtype=None, buffer=None, name='')
    

Create a new tensor of zeros with the same shape and type as a given tensor.

((Similar to numpy.zeros_like))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – the tensor.

  * dtype – the data type of the tensor.

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` of zeros with the same shape as `x`.

---

### nki.language.empty_like

nki.language.empty_like(x, dtype=None, buffer=None, name='')
    

Create a new tensor with the same shape and type as a given tensor.

((Similar to numpy.empty_like))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – the tensor.

  * dtype – the data type of the tensor (default: same as `x`).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), (default: same as `x`).

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` with the same shape and type as `x`.

---

### nki.language.rand

nki.language.rand(shape, dtype, buffer=MemoryRegion.sbuf, name='')
    

Create a new tensor of given shape and dtype on the specified buffer, filled with random values.

Values are sampled from a uniform distribution between 0 and 1.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor, used in scheduling.

Returns:
    

a new `NkiTensor` allocated on the buffer with random values.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.rand -- generate random values in [0, 1)
    a = nl.rand((128, 512), dtype=nl.float32)
    

---

### nki.language.random_seed

nki.language.random_seed(seed)
    

Set the random seed for random number generation.

Using the same seed will generate the same sequence of random numbers when used with `rand()`.

Warning

This API is experimental and may change in future releases.

Parameters:
    

seed – a [1,1] tensor on SBUF or PSUM with a 32-bit seed value.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.random_seed -- set seed for reproducible random values
    seed = nl.full((1, 1), 42, dtype=nl.int32, buffer=nl.sbuf)
    nl.random_seed(seed)
    a = nl.rand((128, 512), dtype=nl.float32)
    
    # nki.language.random_seed -- same seed produces same values
    seed = nl.full((1, 1), 42, dtype=nl.int32, buffer=nl.sbuf)
    nl.random_seed(seed)
    a = nl.rand((128, 512), dtype=nl.float32)
    nl.random_seed(seed)
    b = nl.rand((128, 512), dtype=nl.float32)
    assert nl.equal(a, b)
    

---

### nki.language.shared_identity_matrix

nki.language.shared_identity_matrix(n, dtype='uint8', dst=None)
    

Create an identity matrix in SBUF with the specified data type.

The compiler will reuse all identity matrices of the same dtype in the graph to save space.

Parameters:
    

  * n – the number of rows (and columns) of the returned identity matrix

  * dtype – the data type of the tensor, default to be `nl.uint8` (see Supported Data Types for more information).

Returns:
    

a new `NkiTensor` which contains the identity tensor

Examples:
    
    
    import nki.language as nl
    
    # nki.language.shared_identity_matrix -- 128x128 identity matrix
    identity = nl.shared_identity_matrix(n=128, dtype=nl.float32)
    expected = nl.load(expected_tensor[0:128, 0:128])
    assert nl.equal(identity, expected)
    nl.store(actual_tensor[0:128, 0:128], identity)
    

## Memory Load and Store (DMA)

### load

`load` | Load a tensor from device memory (HBM) into on-chip memory (SBUF).  

---

### store

`store` | Store into a tensor on device memory (HBM) from on-chip memory (SBUF).  

---

### load_transpose2d

`load_transpose2d` | Load a tensor from device memory (HBM) and 2D-transpose the data before storing into on-chip memory (SBUF).  

---

### nki.language.load

nki.language.load(src, dtype=None)
    

Load a tensor from device memory (HBM) into on-chip memory (SBUF).

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * src – HBM tensor to load the data from.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile on SBUF with values from `src`.

---

### nki.language.store

nki.language.store(dst, value)
    

Store into a tensor on device memory (HBM) from on-chip memory (SBUF).

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * dst – HBM tensor to store the data into.

  * value – an SBUF tile that contains the values to store.


---

### nki.language.load_transpose2d

nki.language.load_transpose2d(src, dtype=None)
    

Load a tensor from device memory (HBM) and 2D-transpose the data before storing into on-chip memory (SBUF).

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * src – HBM tensor to load the data from.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile on SBUF with values from `src` 2D-transposed.

---

### nisa.dma_copy

Every DMA operation (`nisa.dma_copy`, `nisa.dma_transpose`) needs a descriptor that tells the hardware the source address, destination address, transfer shape, and stride pattern. We can specify when and where those descriptors are produced—on the host before execution, on the GpSimd engine at runtime, or on a dedicated hardware block. Each choice has different performance characteristics and capability constraints. DGE (Descriptor Generation Engine) is the umbrella term for the strategies that control this.

In the NKI API, there are three concrete strategies—plus an `unknown` mode that lets the compiler choose—exposed through the `nki.isa.dge_mode` enum. The rest of this document describes each mode, its constraints, and when to use each one.

## DGE Modes
### `unknown` — let the compiler decide
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.unknown)
    

The default. The compiler selects the best mode based on the target hardware, tensor shapes, and surrounding instruction schedule. Use this unless you have a specific reason to force a specific mode.

### `none` — pre-computed descriptors in HBM
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.none)
    

DMA descriptors are pre-computed on the Trainium host before NEFF execution. The pre-computed descriptors are stored them in HBM. At runtime the DMA engine reads the pre-built descriptor directly—no on-device generation is needed.

When to use:

  * Fully static transfer patterns where source/destination addresses are known at compile time.

  * When you want to avoid any on-device descriptor generation overhead.

Trade-offs:

  * Descriptors consume HBM capacity (one per DMA instruction instance).

  * Cannot handle dynamic (runtime-computed) addresses or indices.

### `swdge` — software DGE
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.swdge)
    

The GpSimd Engine generates DMA descriptors during NEFF execution. This is the only mode that supports indirect (gather/scatter) operations with dynamic indices from SBUF.

When to use:

  * Dynamic addresses that depend on runtime values.

  * Gather or scatter operations using `vector_offset` (indirect indexing).

  * Indirect transpose (`dma_transpose` with indirect `src`).

Trade-offs:

  * Consumes GpSimd Engine cycles for descriptor generation.

  * May compete with other GpSimd workloads.

Importantly, `swdge` has additional constraints for indirect transpose:

  * `src.shape[-1] <= 128`

  * `src.dtype` must be 2 bytes (`float16` / `bfloat16`)

  * `src` must be on HBM

  * `src.shape[0]` must be divisible by 16

  * When `src` is 4D: `src.shape[1]` or `src.shape[2]` must be 1

  * Index tensor must be 2-D, on SBUF, with dtype `uint32`

  * `indices.shape[0]` must be in `[16, 128]` and divisible by 16

  * When `indices.shape[1] > 1`: `indices.shape[0]` must be exactly 128

  * Only available on NeuronCore-v3 (Trainium2) or newer only

### `hwdge` — hardware DGE
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge)
    

A dedicated hardware block on the NeuronCore generates descriptors on demand, triggered by the Scalar Engine or Sync Engine sequencer. Each TRN2 NeuronCore has two DGE instances.

When to use:

  * Dynamic or semi-dynamic transfer patterns on NeuronCore-v3+.

  * When GpSimd Engine is busy with other work (avoids `swdge` contention).

  * Overlapping descriptor generation with compute via Scalar Engine pipelining.

Trade-offs:

  * Each hardware-DGE DMA instruction takes approximately 600 ns to execute.

  * Does not support indirect (gather/scatter) operations.

Note, for `dma_copy` with `hwdge`, the `engine` parameter can optionally select which sequencer triggers the DGE block:
    
    
    # Let Scalar Engine trigger DGE (can overlap with earlier compute)
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.scalar)
    
    # Let Sync Engine trigger DGE
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.sync)
    

Only `nisa.engine.scalar` and `nisa.engine.sync` are valid when `dge_mode=hwdge`.

### nisa.dma_copy

### `nisa.dma_copy` — Reading from PSUM
`nisa.dma_copy` no longer supports reading directly from PSUM. Copy the PSUM tensor to SBUF first using `nisa.tensor_copy`.
    
    
    # Beta 2
    nisa.dma_copy(dst=hbm_tensor, src=psum_tensor[0:TILE, 0:N])
    
    # NKI 0.3.0
    sbuf_temp = nl.ndarray((TILE, PSUM_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sbuf_temp[0:TILE, 0:N], src=psum_tensor[0:TILE, 0:N])
    nisa.dma_copy(dst=hbm_tensor, src=sbuf_temp[0:TILE, 0:N])
    

### `nisa.dma_copy` — `dge_mode` Type Matching
NKI 0.3.0 enforces that source and destination element types must match when using `dge_mode=dge_mode.hwdge`. Beta 2 did not validate this, allowing mismatched types to pass silently.

The DMA hardware moves raw bytes — HWDGE generates descriptors without interpreting data content, so no type casting occurs. To reinterpret data as a different type, use `.view()` to match types before the copy.
    
    
    # Beta 2 (no validation, undefined behavior)
    nisa.dma_copy(dst=dst_f4, src=src_ui16, dge_mode=nisa.dge_mode.hwdge)
    
    # NKI 0.3.0 — use .view() to reinterpret
    nisa.dma_copy(dst=dst_f4, src=src_ui16.view(nl.float4_e2m1fn_x4), dge_mode=nisa.dge_mode.hwdge)
    

Alternatively, use `dge_mode.swdge` or `dge_mode.none` if type casting is intended.

### nisa.dma_copy

    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)
    
     if nl.num_programs() == 1:
       # Not using multiple cores, process two tiles
       for i in range(2):
         nisa.dma_copy(in_tile, input[i])
         nisa.reciprocal(out_tile, in_tile)
         nisa.dma_copy(output[i], out_tile)
     else:
       # Using multiple cores, process tiles in parallel, one per core
       i = nl.program_id(0)
       nisa.dma_copy(in_tile, input[i])
       nisa.reciprocal(out_tile, in_tile)
       nisa.dma_copy(output[i], out_tile)
     return output

### nisa.dma_copy

    nisa.dma_copy(dst=a_tile, src=a_input)
    
    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)
    

The `nl.ndarray` function allows you to allocate tensors in SBUF. Here you allocate `a_tile` and `b_tile` and use the `nisa.dma_copy` instruction to copy tensors between HBM and SBUF memories. You first supply the destination for the copy, `a_tile` and `b_tile`. Then you provide the source for the copy, `a_input` and `b_input`, as seen in this example.

### nisa.dma_copy

  * `nl.load` and `nl.store` have been removed, use `nisa.dma_copy`

---

### nki.isa.dma_copy

nki.isa.dma_copy(dst, src, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, engine=engine.unknown, name=None)
    

Copy data from `src` to `dst` using DMA engines.

This instruction performs data movement between memory locations (SBUF or HBM) using DMA engines. The operation copies data from the source tensor to the destination tensor: `dst = src`.

`nisa.dma_copy` supports different modes of DMA descriptor generation (DGE):

  * `nisa.dge_mode.none`: Neuron Runtime generates DMA descriptors and stores them into HBM before NEFF execution.

  * `nisa.dge_mode.swdge`: Gpsimd Engine generates DMA descriptors as part of the `nisa.dma_copy` instruction during NEFF execution.

  * `nisa.dge_mode.hwdge`: Sync Engine or Scalar Engine sequencers invoke DGE hardware block to generate DMA descriptors as part of the `nisa.dma_copy` instruction during NEFF execution.

See Trainium2 arch guide and Introduction to DMA with NKI for more discussion.

When either `sw_dge` or `hw_dge` mode is used, the `src` and `dst` tensors can have a dynamic start address which depends on a variable that cannot be resolved at compile time. When `sw_dge` is selected, `nisa.dma_copy` can also perform a gather or scatter operation, using a list of dynamic indices from SBUF. In both of these dynamic modes, out-of-bound address checking is turned on automatically during execution. By default a runtime error is raised (`oob_mode=oob_mode.error` as default setting). Developers can disable this error and make the `nisa.dma_copy` instruction skip the DMA transfer for a given dynamic address or index when it is out of bound using `oob_mode=oob_mode.skip`.

Memory types.

Both `src` and `dst` tiles can be in HBM or SBUF. However, if both tiles are in SBUF, consider using an alternative for better performance:

  * nisa.tensor_copy for direct copies

  * nisa.nc_n_gather to gather elements within each partition independently

  * nisa.local_gather to gather elements within groups of partitions

Data types.

Both `src` and `dst` tiles can be any supported NKI data types (see Supported Data Types for more information).

The DMA engines automatically handle data type conversion when `src` and `dst` have different data types. The conversion is performed through a two-step process: first casting from `src.dtype` to float32, then from float32 to `dst.dtype`.

Tile size.

The total number of data elements in `src` must match that of `dst`.

Indirect addressing (gather/scatter).

`nisa.dma_copy` supports indirect addressing for dynamic row selection at runtime. This enables gather (read from dynamic rows) and scatter (write to dynamic rows) patterns. Indirect addressing is activated by calling `.ap()` on `src` or `dst` with a `vector_offset` or `scalar_offset` parameter.

There are two types of indirect addressing:

Vector indirection provides per-partition dynamic offsets. Each of the hardware partitions gets its own index, enabling gather/scatter where different partitions access different rows. Use `.ap(pattern=..., vector_offset=idx_tensor, indirect_dim=0)` where `idx_tensor` is an SBUF tensor of shape `(P, 1)` containing one row index per partition. The tensor being indexed (the one `.ap()` is called on) must be in HBM.

Scalar indirection provides a single dynamic offset applied uniformly to all partitions. Use `.ap(pattern=..., scalar_offset=reg_or_tensor, indirect_dim=N)` where the offset is either a 1x1 SBUF tensor or a `VirtualRegister` from `nisa.register_alloc()`.

`vector_offset` and `scalar_offset` are mutually exclusive.

Indirect gather example (`vector_offset` on `src`):
    
    
    import nki
    import nki.isa as nisa
    import nki.language as nl
    
    @nki.jit
    def indirect_gather_kernel(data, indices):
        P, F = indices.shape[0], data.shape[1]
        output = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.shared_hbm)
    
        idx = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=indices)
    
        dst = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dst,
            src=data.ap(
                pattern=[[F, P], [1, F]],
                vector_offset=idx,
                indirect_dim=0,
            ),
        )
    
        nisa.dma_copy(dst=output, src=dst)
        return output
    

Indirect scatter example (`vector_offset` on `dst`):
    
    
    import nki
    
    @nki.jit
    def indirect_scatter_kernel(src_data, indices, output):
        P, F = src_data.shape
    
        src = nl.ndarray((P, F), dtype=src_data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src, src=src_data)
    
        idx = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=indices)
    
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[F, P], [1, F]],
                vector_offset=idx,
                indirect_dim=0,
            ),
            src=src,
        )
        return output
    

Parameters:
    

  * dst – the destination tensor to copy data into

  * src – the source tensor to copy data from

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See Trainium2 arch guide for more information.

  * oob_mode – 

(optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

    * `oob_mode.error`: (Default) Raises an error when encountering out-of-bounds indices.

    * `oob_mode.skip`: Silently skips any operations involving out-of-bounds indices.

For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

  * engine – (optional) the engine to use for HWDGE descriptor generation: `nki.isa.engine.sync` or `nki.isa.engine.scalar`. Only valid when `dge_mode=nisa.dge_mode.hwdge`. `nki.isa.engine.unknown` by default.

---

### nisa.dma_copy with access pattern

      nisa.dma_copy(src=A.ap(
        pattern=[[512, 128], [1, 256]], offset=0,
        scalar_offset=batch_idx, indirect_dim=0
        ),
        dst=result[...])

---

### nisa.dge_mode.hwdge

### `hwdge` — hardware DGE
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge)
    

A dedicated hardware block on the NeuronCore generates descriptors on demand, triggered by the Scalar Engine or Sync Engine sequencer. Each TRN2 NeuronCore has two DGE instances.

When to use:

  * Dynamic or semi-dynamic transfer patterns on NeuronCore-v3+.

  * When GpSimd Engine is busy with other work (avoids `swdge` contention).

  * Overlapping descriptor generation with compute via Scalar Engine pipelining.

Trade-offs:

  * Each hardware-DGE DMA instruction takes approximately 600 ns to execute.

  * Does not support indirect (gather/scatter) operations.

Note, for `dma_copy` with `hwdge`, the `engine` parameter can optionally select which sequencer triggers the DGE block:
    
    
    # Let Scalar Engine trigger DGE (can overlap with earlier compute)
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.scalar)
    
    # Let Sync Engine trigger DGE
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                  dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.sync)
    

Only `nisa.engine.scalar` and `nisa.engine.sync` are valid when `dge_mode=hwdge`.

Hardware DGE constraints for `dma_transpose`:

  * `src.shape[0] == 16`

  * `src.shape[-1] % 128 == 0`

  * `src.dtype` must be 2 bytes (`float16` / `bfloat16`)

---

### nki.isa.dma_transpose

nki.isa.dma_transpose(dst, src, axes=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None)
    

Perform a transpose on input `src` using DMA Engine.

The permutation of transpose follow the rules described below:

  1. For 2-d input tile, the permutation will be [1, 0]

  2. For 3-d input tile, the permutation will be [2, 1, 0]

  3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

DMA Direct Transpose Constraints

The only valid `dge_mode` s are `unknown` and `hwdge`. If `hwdge`, this instruction will be lowered to a Hardware DGE transpose. This has additional restrictions:

  1. `src.shape[0] == 16`

  2. `src.shape[-1] % 128 == 0`

  3. `src.dtype` is 2 bytes

DMA Indirect Transpose Constraints

The only valid `dge_mode` s are `unknown` and `swdge`. This instruction will be lowered to a Software DGE transpose (`dma_gather_transpose`). This has additional restrictions:

  1. When `src` is 4D: `len(src[1])` or `len(src[2])` must be 1

  2. `src.shape[-1] <= 128`

  3. `src.dtype` is 2 bytes

  4. `src` tensor must be on HBM

  5. `indices` must be 2-d

  6. `indices.shape[0] * indices.shape[1]` must be `>=` `src.shape[0]`

  7. `src.shape[0]` must be divisible by 16

  8. `indices.shape[0]` must be in `[16, 128]` and divisible by 16

  9. When `indices.shape[1] > 1`: `indices.shape[0]` must be exactly 128

  10. `indices.dtype` is `np.uint32`

  11. `indices` tensor must be on SBUF

  12. TRN2+ only

Indirect transpose effectively performs the following operation: `flat_indices = indices.T.flatten()[:src.shape[0]]` `gathered = src[flat_indices, :]` `dst = gathered.T`

Indirect transpose example (`vector_offset` on `src`):
    
    
    import nki
    import nki.isa as nisa
    import nki.language as nl
    
    @nki.jit
    def gather_transpose_kernel(src_hbm, idx_hbm):
        P, F = 128, 128
        output = nl.ndarray((F, P), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    
        idx_sb = nl.load(idx_hbm)
    
        dst_sb = nl.ndarray((F, P), dtype=src_hbm.dtype, buffer=nl.sbuf)
        nisa.memset(dst=dst_sb, value=0)
    
        src_ap = src_hbm.ap(
            pattern=[[F, P], [1, F]],
            vector_offset=idx_sb,
            indirect_dim=0,
        )
        nisa.dma_transpose(dst=dst_sb, src=src_ap, axes=(1, 0))
    
        nisa.dma_copy(dst=output, src=dst_sb)
        return output
    

Parameters:
    

  * dst – the destination of transpose, must be a tile in SBUF.

  * src – the source of transpose, must be a tile in HBM or SBUF. `src.dtype == dst.dtype`

  * axes – transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source. Supported axes are `(1, 0)`, `(2, 1, 0)`, and `(3, 1, 2, 0)`.

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See Trainium2 arch guide for more information.

  * oob_mode – 

(optional) Specifies how to handle runtime out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

    * `oob_mode.error`: (Default) Raises an error when encountering runtime out-of-bounds indices.

    * `oob_mode.skip`: Silently skips any operations involving out-of-bounds indices. Only valid when `src` uses indirect indexing.


---

### dma_copy

`dma_copy` | Copy data from `src` to `dst` using DMA engines.  

---

### dma_transpose

`dma_transpose` | Perform a transpose on input `src` using DMA Engine.  

---

### name parameter

## The `name` Parameter
Both `dma_copy` and `dma_transpose` accept an optional `name` string:
    
    
    nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor, name="load_weights")
    

This label appears in profiling traces and compiler debug output. It does not affect execution. Assigning meaningful names makes it significantly easier to identify specific DMA operations when analyzing performance with Neuron profiling tools.

---

### vector_offset parameter

        [[512, 64], [1, 512]], 0, vector_offset=dynamic_idx_legal, indirect_dim=0

---

### vector_offset access pattern

## How `.ap()` Affects DGE Mode
When you use `.ap()` with `vector_offset` for indirect (gather/scatter) access, the DGE mode is constrained to `swdge`:

Access Pattern | `dma_copy` | `dma_transpose`  
---|---|---  
Static (no `.ap()`, or `.ap()` without offsets) | Any mode | `none`, `hwdge`, or compiler-selected  
`.ap()` with `scalar_offset` | Any mode | Any mode  
`.ap()` with `vector_offset` | `unknown` or `swdge` | `unknown` or `swdge`  
  
If you specify `dge_mode=unknown` (the default) with `vector_offset`, the compiler will automatically select `swdge`.

---

### DGE Mode Selection Summary

## Mode Selection Summary
Mode | Descriptor Source | Min HW | Indirect Support | Best For  
---|---|---|---|---  
`none` | Host (pre-computed in HBM) | Any | No | Fully static patterns, zero on-device overhead  
`swdge` | GpSimd Engine | Any (indirect: v3+) | Yes | Gather/scatter, dynamic indices  
`hwdge` | Hardware DGE block | NeuronCore-v3+ | No | Dynamic patterns without GpSimd contention  
`unknown` | Compiler decides | Any | Depends | Default—recommended unless tuning  

---

### DGE Mode Performance Implications

## Performance Implications
In essence, the choice comes down to where you want to spend your overhead budget:

  * ``none`` — Lowest per-transfer latency (descriptor already in HBM), but each descriptor consumes HBM bandwidth on first fetch and HBM capacity permanently.

  * ``swdge`` — Flexible but uses GpSimd cycles. In GpSimd-bound kernels this can become a bottleneck.

  * ``hwdge`` — ~600 ns per instruction. When triggered from Scalar Engine, descriptor generation overlaps with earlier compute instructions in the pipeline, effectively hiding the cost. Frees GpSimd for other work.

  * ``unknown`` — The compiler applies heuristics to pick the best mode for the target and workload. Start here and only override after profiling.

In summary, use `unknown` until profiling tells you otherwise, then switch to the specific mode that addresses the bottleneck you observe.

---

### nki.isa.dge_mode

class nki.isa.dge_mode(value)
    

Descriptor Generation Engine mode.

Attributes

`unknown` | Unknown DGE mode, i.e., let compiler decide the DGE mode  
---|---  
`swdge` | Software DGE  
`hwdge` | Hardware DGE  
`none` | Not using DGE  

---

### dge_mode

`dge_mode` | Descriptor Generation Engine mode.  

## DMA Compute

### nisa.dma_compute

### `nisa.dma_copy` — `dst_rmw_op` and `unique_indices` Removed
`nisa.dma_copy` no longer supports read-modify-write operations. The `dst_rmw_op` and `unique_indices` parameters have been removed. Use `nisa.dma_compute` instead.
    
    
    # Beta 2 — simple read-modify-write
    nisa.dma_copy(dst, src, dst_rmw_op=nl.add)
    
    # NKI 0.3.0 — use dma_compute
    nisa.dma_compute(dst, [src], reduce_op=nl.add)
    

For accumulation loops with indirect indexing:
    
    
    # Beta 2
    for k_idx in range(K):
        dst_rmw_op = None if k_idx == 0 else nl.add
        nisa.dma_copy(
            src=input.ap(...),
            dst=reduced_sb[:, :],
            dst_rmw_op=dst_rmw_op,
            unique_indices=True,
        )
    
    # NKI 0.3.0 — split into dma_copy + dma_compute
    for k_idx in range(K):
        src_access = input.ap(...)
        if k_idx == 0:
            nisa.dma_copy(dst=reduced_sb[:, :], src=src_access)
        else:
            nisa.dma_compute(
                dst=reduced_sb[:, :],
                srcs=[src_access, reduced_sb[:, :]],
                reduce_op=nl.add,
                unique_indices=True,
            )
    

---

### dma_compute

`dma_compute` | Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.  

---

### nki.isa.dma_compute

nki.isa.dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True, name=None)
    

Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.

This instruction leverages the compute capabilities within DMA engines to perform scaled element-wise operations followed by reduction across multiple source tensors. The computation follows the pattern: `dst = reduce_op(srcs[0] * scales[0], srcs[1] * scales[1], ...)`, where each source tensor is first multiplied by its corresponding scale factor, then all scaled results are combined using the specified reduction operation. Currently, only `nl.add` is supported for `reduce_op`, and all values in `scales` must be `1.0` (or `scales` can be `None` which defaults to all 1.0).

The DMA engines perform all computations in float32 precision internally. Input tensors are automatically cast from their source data types to float32 before computation, and the final float32 result is cast to the output data type in a pipelined fashion.

Read-Modify-Write with vector_offset (scatter and gather).

When one of the source tensors has a `vector_offset` (indirect indexing), `dma_compute` performs read-modify-write with two modes:

Scatter RMW: `dst(HBM)[indices] = dst(HBM)[indices] + src(SB)`
    

  * `dst` is in HBM with indirect indexing

  * One source matches `dst` and has `vector_offset`

  * The other source is data in SBUF

Gather RMW: `dst(SB) = dst(SB) + src(HBM)[indices]`
    

  * `dst` is in SBUF

  * One source is data in HBM with `vector_offset`

  * The other source matches `dst`

Both modes require:
    

  * Exactly 2 source tensors

  * All `scales` must be `1.0` (or `None`)

  * `unique_indices` must be `True` (non-unique indices not yet supported)

Memory types.

Both input `srcs` tensors and output `dst` tensor can be in HBM or SBUF. Both `srcs` and `dst` tensors must have compile-time known addresses (unless using vector_offset for indirect access).

Data types.

All input `srcs` tensors and the output `dst` tensor can be any supported NKI data types (see Supported Data Types for more information). The DMA engines automatically cast input data types to float32 before performing the scaled reduction computation. The float32 computation results are then cast to the data type of `dst` in a pipelined fashion.

Layout.

The computation is performed element-wise across all tensors, with the reduction operation applied across the scaled source tensors at each element position.

Tile size.

The element count of each tensor in `srcs` and `dst` must match exactly. The max number of source tensors in `srcs` is 16.

Parameters:
    

  * dst – the output tensor to store the computed results

  * srcs – a list of input tensors to be scaled and reduced

  * reduce_op – the reduction operation to apply (currently only `nl.add` is supported)

  * scales – (optional) a list of scale factors corresponding to each tensor in `srcs`. Must be all 1.0 if provided. Defaults to None (equivalent to [1.0, 1.0, …]).

  * unique_indices – (optional) Whether scatter indices are unique. Must be True when using vector_offset (non-unique not yet supported). Default: True.

## Tensor Copy and Move

### copy

`copy` | Create a copy of the input tile.  

---

### nki.language.copy

nki.language.copy(x, dtype=None)
    

Create a copy of the input tile.

Warning

This API is experimental and may change in future releases.

Uses the Scalar Engine via `activation(op=copy)`. Note that the Scalar Engine internally casts through FP32, which may be lossy for integer types with values exceeding FP32 precision (e.g. int32 values > 2^23).

Parameters:
    

  * x – the source of copy, must be a tile in SBUF or PSUM.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a new tile with the same layout as `x`, allocated on the same buffer as `x` (SBUF or PSUM).

---

### nisa.tensor_copy

## Deprecated and Removed APIs
### `nki.isa.tensor_copy_dynamic_src` / `nki.isa.tensor_copy_dynamic_dst`
Deprecated and scheduled for removal. Use `nisa.tensor_copy()` with `.ap()` and `scalar_offset` instead.

### nisa.tensor_copy

        nisa.tensor_copy(result[p, f], t[p, f])

---

### tensor_copy

`tensor_copy` | Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.  

---

### nki.isa.tensor_copy

nki.isa.tensor_copy(dst, src, engine=engine.unknown, name=None)
    

Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

The output tile has the same partition axis size and also the same number of elements per partition as the input tile `src`.

All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior is slightly different across engines:

  * Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from FP32 to `dst.dtype`. Users should be cautious with assigning this instruction to Scalar Engine when the input data type cannot be precisely cast to FP32 (e.g., INT32).

  * Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.

In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details). By default, this API returns a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

On NeuronCore v2, `tensor_copy` is not supported on the Scalar Engine. Instead, use nisa.activation with `op=nl.copy`.

Parameters:
    

  * dst – a tile with the same content and partition axis size as the `src` tile.

  * src – the source of copy, must be a tile in SBUF or PSUM.

  * engine – (optional) the engine to use for the operation: nki.isa.engine.vector, nki.isa.engine.scalar, nki.isa.engine.gpsimd or nki.isa.engine.unknown (default, compiler selects best engine based on engine workload).


---

### tensor_copy_predicated

`tensor_copy_predicated` | Conditionally copy elements from the `src` tile to the destination tile on SBUF / PSUM based on a `predicate` using Vector Engine.  

---

### nki.isa.tensor_copy_predicated

# nki.isa.tensor_copy_predicated
nki.isa.tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None)
    

Conditionally copy elements from the `src` tile to the destination tile on SBUF / PSUM based on a `predicate` using Vector Engine.

This instruction provides low-level control over conditional data movement on NeuronCores, optimized for scenarios where only selective copying of elements is needed. Either `src` or `predicate` may be in PSUM, but not both simultaneously. Both `src` and `predicate` are permitted to be in SBUF.

Shape and data type constraints:

  1. `src` (if it is a tensor), `dst`, and `predicate` must occupy the same number of partitions and same number of elements per partition.

  2. `predicate` must be of type `uint8`, `uint16`, or `uint32`.

  3. `src` and `dst` must share the same data type.

Behavior:

  * Where predicate is True: The corresponding elements from src are copied to dst tile. If src is a scalar, the scalar is copied to the dst tile.

  * Where predicate is False: The corresponding values in dst tile are unmodified

Parameters:
    

  * src – The source tile or number to copy elements from when `predicate` is True

  * dst – The destination tile to copy elements to

  * predicate – A tile that determines which elements to copy

  * reverse_pred – A boolean that reverses the effect of `predicate`.


## Access Patterns and Indexing

### ds

`ds` | Create a dynamic slice for tensor indexing.  

---

### nki.language.ds

nki.language.ds(start, size)
    

Create a dynamic slice for tensor indexing.

Parameters:
    

  * start – the start index of the slice.

  * size – the size of the slice.

Returns:
    

a dynamic slice object for use in tensor indexing.

---

### nl.ndarray.ap

The NKI API for access pattern is a direct reflection of the hardware capability. The `nl.ndarray` has an `ap` method.
    
    
    def ap(self, pattern: List[Tuple[int, int]],
       offset: Optional[int] = 0,
       scalar_offset: Optional[Access] = None,
       vector_offset: Optional[Access] = None,
       indirect_dim: int = 0
       dtype: Optional[Dtype] = None):
       pass
    

The parameters have the following definitions:

  * `pattern`: A list of two-element tuples, each tuple describes the access on one dimension. The first element represents the element stepping and the second element represents the number of elements in each dimension. This tuple is referred to as `[step, num]` going forward.

    * The shape of a pattern is the collection of num. For example, given pattern `[[w_step, w_num], [z_step, z_num], [y_step, y_num], [x_step, x_num]]`, the shape is `[w_num, z_num, y_num, x_num]`.

    * It is worth mentioning that the order of the pattern specified here is in the opposite order to what is actually accepted by the hardware. Therefore, the order of the tuples shown on the profiler will be to the opposite order of what is specified here.

  * `offset`: The offset to start the access in terms of number of elements from the beginning of the tensor. The default value is 0.

  * `scalar_offset`: An SBUF memory location that specifies the location to start the access in terms of number of elements on the `indirect_dim` of the access pattern. At most one of the `scalar_offset` and `vector_offset` can be specified.

  * `vector_offset`: An SBUF memory location that specifies the location to start the access in terms of number of elements from the beginning of the indirect dimension specified by `indirect_dim`. At most one of the `scalar_offset` and `vector_offset` can be specified.

  * `indirect_dim`: The indirect dimension on which to apply `scalar_offset` and `vector_offset`.

  * `dtype`: The data type of the access pattern. The default value is the `dtype` of the tensor being accessed.

## Semantics of the Access Pattern
Access patterns can be thought of as compact representations of a loop. The offset is an integer indicating the start offset in terms of elements with respect to the beginning of the tensor. Each two-element list `[step, num]` represents the stride in terms of elements and the number of iterations of each level of the loop. The semantics are explored through the following example.

Given a tensor, the Access Pattern conceptually flattens the tensor to 1d, and then uses a loop to fetch elements from the tensor to construct a view. Consider the following NKI code:
    
    
    t = nl.ndarray((p_count, N), dtype=nl.float32, buffer=nl.sbuf)
    access = t.ap(
      pattern=[[N, p_size], [z_step, z_num], [
      y_step, y_num], [x_step, x_num]],
      offset)
    

The above represents the following access on the tensor `t`, written below in pseudo-code.
    
    
    access = nl.ndarray((p_size, z_num, y_num, x_num), dtype=nl.float32, buffer=nl.sbuf)
    for w in range(p_size):
      for z in range(z_num):
        for y in range(y_num):
          for x in range(x_num):
            t_flatten = t.flatten() # first flatten the tensor to 1d
            access[w, z, y, x] = [offset + (w * N) + (z * z_step)
                      + (y * y_step) + (x * x_step)]
    

The access pattern has the following properties:

1\. Recall from the hardware capability, the access pattern in each partition must be identical. Therefore, the step of the first tuple in the AP must be equal to the number of elements in the free dimension of the tensor. 2\. The shape of the result view is always the same as the shape of the pattern.

Note that calling `.ap` on a tensor does not do any computation directly. It describes how to get data. The engines will consume data when the AP is passed into a `nki.isa` instruction.
    
    
    src = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)
    dst = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)
    src_access = src.ap([32, 16], [1, 32]) # no computation happens
    dst_access = dst.ap([32, 16], [1, 32]) # no computation happens
    
    # Engine reads both src_access and dst_access and performs the copy
    nisa.dma_copy(dst_access, src_access)
    

---

### SBUF/PSUM Tensor Restrictions

## Restriction on SBUF/PSUM Tensors
For SBUF/PSUM tensors, the first tuple must always be the access for the partition dimension. On NeuronCore v2/v3/v4, the access on the partition dimension must be continuous, meaning that the step of the leading dimension must be the element count of the entire free dimension of the tensor. Therefore, given a tensor of shape `(p_dim, f_dim0, fdim1)`, the step of the leading dimension must be `f_dim0 * f_dim1`.

The following example is not allowed because it reads every other partition.
    
    
    t = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)
    
    # The following is illegal, because the first stride is 16*2 and reads every other partition
    t.ap(pattern=[[64, 8], [1, 32]], offset=0)
    

---

### Nested Indexing Restriction

## Restriction on Nested Indexing
The `.ap` method is only allowed on `nl.ndarray` and cannot be called on a tile produced by it. For example, the following would result in an error.
    
    
    t = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
    t.ap(pattern=[[256, 128],[1, 256]], offset=0).ap(pattern=[[256, 64], [1, 64]], offset=0)
         ^-- cannot specify an access pattern on an already indexed tensor
    

---

### Reinterpret Cast with ap

## Reinterpret Cast with `ap`
The `dtype` parameter can be used for reinterpret casting the tensor. Since both the pattern and the offset are in terms of number of elements, not bytes, the count must be computed accordingly. See the following example of reinterpret cast from `INT32` to `BF16`.
    
    
    t = nl.ndarray((128, 256), dtype=nl.int32, buffer=nl.sbuf)
    cast_to_bf16 = t.ap(pattern=[
      [512, 128], [1, 512]
     ], # notice the number of elements is doubled due to dtype size change
    offset = 0, dtype=nl.bfloat16) # cast_to_bf16 has shape (128, 512)
    

---

### expand_dims

`expand_dims` | Expand the shape of a tile.  

---

### nki.language.expand_dims

nki.language.expand_dims(x, axis)
    

Expand the shape of a tile.

((Similar to numpy.expand_dims))

Warning

This API is experimental and may change in future releases.

Insert a new axis that will appear at the axis position in the expanded tile shape.

Parameters:
    

  * x – a tile.

  * axis – position in the expanded axes where the new axis is placed.

Returns:
    

a tile with view of input data with the number of dimensions increased.

---

### broadcast_to

`broadcast_to` | Broadcast a tile to a new shape following numpy broadcasting rules.  

---

### nki.language.broadcast_to

nki.language.broadcast_to(x, shape, dtype=None)
    

Broadcast a tile to a new shape following numpy broadcasting rules.

((Similar to numpy.broadcast_to))

Warning

This API is experimental and may change in future releases.

If `x.shape` is already the same as `shape`, returns `x` unchanged (or a dtype-cast copy if `dtype` differs).

Parameters:
    

  * x – the source tile in SBUF or PSUM.

  * shape – the target shape. Must have the same rank as `x`. Each dimension must either match or be broadcast from size 1.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile with the target shape containing broadcast values from `x`.

---

### gather_flattened

`gather_flattened` | Gather elements from data tensor using indices after flattening.  

---

### nki.language.gather_flattened

nki.language.gather_flattened(data, indices, axis=0, dtype=None)
    

Gather elements from data tensor using indices after flattening.

This instruction gathers elements from the data tensor using integer indices provided in the indices tensor. For each element in the indices tensor, it retrieves the corresponding value from the data tensor using the index value to select from the free dimension of data.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * data – input tensor to gather from.

  * indices – indices to gather.

  * axis – axis along which to gather.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

gathered tensor.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.gather_flattened -- gather elements by index
    data = nl.load(data_tensor[0:128, 0:512])
    indices = nl.load(indices_tensor[0:128, 0:512])
    result = nl.gather_flattened(data, indices)
    nl.store(actual_tensor[0:128, 0:512], result)
    

## Element-wise Math and Arithmetic

### abs

`abs` | Absolute value of the input, element-wise.  

---

### add

`add` | Add the inputs, element-wise.  

---

### subtract

`subtract` | Subtract the inputs, element-wise.  

---

### multiply

`multiply` | Multiply the inputs, element-wise.  

---

### negative

`negative` | Numerical negative of the input, element-wise.  

---

### power

`power` | Elements of x raised to powers of y, element-wise.  

---

### reciprocal

`reciprocal` | Compute element-wise reciprocal (1.0/x) of the input `data` tile using Vector Engine.  

### reciprocal

`reciprocal` | Reciprocal of the input, element-wise.  

---

### square

`square` | Square of the input, element-wise.  

---

### sqrt

`sqrt` | Non-negative square-root of the input, element-wise.  

---

### rsqrt

`rsqrt` | Reciprocal of the square-root of the input, element-wise.  

---

### ceil

`ceil` | Ceiling of the input, element-wise.  

---

### floor

`floor` | Floor of the input, element-wise.  

---

### trunc

`trunc` | Truncated value of the input, element-wise.  

---

### sign

`sign` | Sign of the numbers of the input, element-wise.  

---

### log

`log` | Natural logarithm of the input, element-wise.  

---

### exp

`exp` | Exponential of the input, element-wise.  

---

### nki.language.abs

nki.language.abs(x, dtype=None)
    

Absolute value of the input, element-wise.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has absolute values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.abs
    a = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.abs(a)
    expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    
    # nki.language.abs with explicit dtype
    a = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.abs(a, dtype=nl.float16)
    expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.add

nki.language.add(x, y, dtype=None)
    

Add the inputs, element-wise.

((Similar to numpy.add))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has `x + y`, element-wise.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.add -- element-wise addition of two tiles
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.add(a, b)
    
    expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.add -- adding a scalar to every element of a tile
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.add(a, 2.0)
    expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.subtract

nki.language.subtract(x, y, dtype=None)
    

Subtract the inputs, element-wise.

((Similar to numpy.subtract))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has `x - y`, element-wise.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.subtract -- element-wise subtraction of two tiles
    a = nl.full((128, 512), 10.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.subtract(a, b)
    expected = nl.full((128, 512), 7.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.subtract -- subtracting a scalar from every element
    a = nl.full((128, 512), 10.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.subtract(a, 3.0)
    expected = nl.full((128, 512), 7.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.multiply

nki.language.multiply(x, y, dtype=None)
    

Multiply the inputs, element-wise.

((Similar to numpy.multiply))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has `x * y`, element-wise.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.multiply -- element-wise multiplication of two tiles
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 4.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.multiply(a, b)
    expected = nl.full((128, 512), 12.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.multiply -- scaling every element by a scalar
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.multiply(a, 4.0)
    expected = nl.full((128, 512), 12.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.negative

nki.language.negative(x, dtype=None)
    

Numerical negative of the input, element-wise.

((Similar to numpy.negative))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has numerical negative values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.negative -- negates 5.0 to -5.0
    a = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.negative(a)
    expected = nl.full((128, 512), -5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.negative -- negates -3.0 to 3.0
    a = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.negative(a)
    expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)

---

### nki.language.power

nki.language.power(x, y, dtype=None)
    

Elements of x raised to powers of y, element-wise.

((Similar to numpy.power))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has values `x` to the power of `y`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.power -- element-wise exponentiation of two tiles
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.power(a, b)
    expected = nl.full((128, 512), 9.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.reciprocal

nki.language.reciprocal(x, dtype=None)
    

Reciprocal of the input, element-wise.

((Similar to numpy.reciprocal))

Warning

This API is experimental and may change in future releases.

`reciprocal(x) = 1 / x`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has reciprocal values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.reciprocal -- reciprocal(4.0) = 0.25
    a = nl.full((128, 512), 4.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.reciprocal(a)
    expected = nl.full((128, 512), 0.25, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.square

nki.language.square(x, dtype=None)
    

Square of the input, element-wise.

((Similar to numpy.square))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has square of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.square -- square(3.0) = 9.0
    a = nl.full((128, 512), 3.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.square(a)
    expected = nl.full((128, 512), 9.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.sqrt

nki.language.sqrt(x, dtype=None)
    

Non-negative square-root of the input, element-wise.

((Similar to numpy.sqrt))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has square-root values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.sqrt -- sqrt(4.0) = 2.0
    a = nl.full((128, 512), 4.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.sqrt(a)
    expected = nl.full((128, 512), 2.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.rsqrt

# nki.language.rsqrt
nki.language.rsqrt(x, dtype=None)
    

Reciprocal of the square-root of the input, element-wise.

((Similar to torch.rsqrt))

Warning

This API is experimental and may change in future releases.

`rsqrt(x) = 1 / sqrt(x)`

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has reciprocal square-root values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.rsqrt -- rsqrt(4.0) = 0.5
    a = nl.full((128, 512), 4.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.rsqrt(a)
    expected = nl.full((128, 512), 0.5, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.ceil

# nki.language.ceil
nki.language.ceil(x, dtype=None)
    

Ceiling of the input, element-wise.

((Similar to numpy.ceil))

Warning

This API is experimental and may change in future releases.

The ceil of the scalar x is the smallest integer i, such that i >= x.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has ceiling values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.ceil -- rounds 3.2 up to 4.0
    a = nl.full((128, 512), 3.2, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.ceil(a)
    expected = nl.full((128, 512), 4.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.ceil -- rounds -3.7 up to -3.0
    a = nl.full((128, 512), -3.7, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.ceil(a)
    expected = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.floor

# nki.language.floor
nki.language.floor(x, dtype=None)
    

Floor of the input, element-wise.

((Similar to numpy.floor))

Warning

This API is experimental and may change in future releases.

The floor of the scalar x is the largest integer i, such that i <= x.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has floor values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.floor -- rounds 3.7 down to 3.0
    a = nl.full((128, 512), 3.7, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.floor(a)
    expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.floor -- rounds -3.2 down to -4.0
    a = nl.full((128, 512), -3.2, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.floor(a)
    expected = nl.full((128, 512), -4.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.trunc

nki.language.trunc(x, dtype=None)
    

Truncated value of the input, element-wise.

((Similar to numpy.trunc))

Warning

This API is experimental and may change in future releases.

The truncated value of the scalar x is the nearest integer i which is closer to zero than x is. In short, the fractional part of the signed number x is discarded.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has truncated values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.trunc -- truncates 3.7 toward zero to 3.0
    a = nl.full((128, 512), 3.7, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.trunc(a)
    expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.trunc -- truncates -3.7 toward zero to -3.0
    a = nl.full((128, 512), -3.7, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.trunc(a)
    expected = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.sign

nki.language.sign(x, dtype=None)
    

Sign of the numbers of the input, element-wise.

((Similar to numpy.sign))

Warning

This API is experimental and may change in future releases.

The sign function returns `-1` if `x < 0`, `0` if `x==0`, `1` if `x > 0`.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has sign values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.sign -- sign(-5.0) = -1.0
    a = nl.full((128, 512), -5.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.sign(a)
    expected = nl.full((128, 512), -1.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.log

nki.language.log(x, dtype=None)
    

Natural logarithm of the input, element-wise.

((Similar to numpy.log))

Warning

This API is experimental and may change in future releases.

It is the inverse of the exponential function, such that: `log(exp(x)) = x` . The natural logarithm base is `e`.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has natural logarithm values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.log -- log(1.0) = 0.0
    a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.log(a)
    expected = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.exp

nki.language.exp(x, dtype=None)
    

Exponential of the input, element-wise.

((Similar to numpy.exp))

Warning

This API is experimental and may change in future releases.

The `exp(x)` is `e^x` where `e` is the Euler’s number = 2.718281…

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has exponential values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.exp -- exp(0.0) = 1.0
    a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.exp(a)
    expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nisa.reciprocal

    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)
    
     if nl.num_programs() == 1:
       # Not using multiple cores, process two tiles
       for i in range(2):
         nisa.dma_copy(in_tile, input[i])
         nisa.reciprocal(out_tile, in_tile)
         nisa.dma_copy(output[i], out_tile)
     else:
       # Using multiple cores, process tiles in parallel, one per core
       i = nl.program_id(0)
       nisa.dma_copy(in_tile, input[i])
       nisa.reciprocal(out_tile, in_tile)
       nisa.dma_copy(output[i], out_tile)
     return output

### nisa.reciprocal

    nisa.reciprocal(dst=result[...], src)

---

### nki.isa.reciprocal

nki.isa.reciprocal(dst, data, name=None)
    

Compute element-wise reciprocal (1.0/x) of the input `data` tile using Vector Engine.

Memory types.

Both the input `data` and output `dst` tiles can be in SBUF or PSUM.

Data types.

The input `data` tile can be any valid NKI data type (see Supported Data Types for more information). The Vector Engine automatically casts the input data type to float32 and performs the reciprocal computation in float32 math. The float32 results are cast to the data type of `dst`.

Layout.

The partition dimension of the input `data` is considered the parallel compute dimension.

Tile size.

The partition dimension size of input `data` and output `dst` tiles must be the same and must not exceed 128. The number of elements per partition of `dst` must match that of `data` and must not exceed the physical size of each SBUF partition.

Parameters:
    

  * dst – the output tile

  * data – the input tile

---

### Supported Math Operators by NKI ISA

## Supported Math Operators for NKI ISA
Supported Math Operators by NKI ISA below lists all the mathematical operator primitives supported by NKI. Many nki.isa APIs (instructions) allow programmable operators through the `op` field. The supported operators fall into two categories: bitvec and arithmetic. In general, instructions using bitvec operators expect integer data types and treat input elements as bit patterns. On the other hand, instructions using arithmetic operators accept any valid NKI data type and convert input elements into float32 before performing the operators.

Table 16 Supported Math Operators by NKI ISA# | Operator | `op` | Legal Reduction `op`  
---|---|---|---  
Bitvec | Bitwise Not | `nki.language.invert` | N  
Bitwise And | `nki.language.bitwise_and` | Y  
Bitwise Or | `nki.language.bitwise_or` | Y  
Bitwise Xor | `nki.language.bitwise_xor` | Y  
Arithmetic Shift Left | `nki.language.left_shift` | N  
Arithmetic Shift Right | Not supported | N  
Logical Shift Left | `nki.language.left_shift` | N  
Logical Shift Right | `nki.language.right_shift` | N  
Arithmetic | Add | `nki.language.add` | Y  
Subtract | `nki.language.subtract` | Y  
Multiply | `nki.language.multiply` | Y  
Max | `nki.language.maximum` | Y  
Min | `nki.language.minimum` | Y  
Is Equal to | `nki.language.equal` | N  
Is Not Equal to | `nki.language.not_equal` | N  
Is Greater than or Equal to | `nki.language.greater_equal` | N  
Is Greater than to | `nki.language.greater` | N  
Is Less than or Equal to | `nki.language.less_equal` | N  
Is Less than | `nki.language.less` | N  
Logical And | `nki.language.logical_and` | Y  
Logical Or | `nki.language.logical_or` | Y  
Logical Xor | `nki.language.logical_xor` | Y  
Reverse Square Root | `nki.language.rsqrt` | N  
Reciprocal | `nki.language.reciprocal` | N  
Absolute | `nki.language.abs` | N  
Power | `nki.language.power` | N  

---

### NKI Engine Selection for Operators Supported on Multiple Engines

## NKI Engine Selection for Operators Supported on Multiple Engines
There is a tradeoff between precision and speed on different engines for operators with multiple engine options. Users can select which engine to map to based on their needs. We take reciprocal and reverse square root as two examples and explain the tradeoff below.

  1. Reciprocal can run on Scalar Engine or Vector Engine:

> Reciprocal can run on Vector Engine with `nki.isa.reciprocal` or on Scalar Engine with `nki.isa.activation(nl.reciprocal)`. Vector Engine performs reciprocal at a higher precision compared to Scalar Engine; however, the computation throughput of reciprocal on Vector Engine is about 8x lower than Scalar Engine for large input tiles. For input tiles with a small number of elements per partition (less than 64, processed one per cycle), instruction initiation interval (roughly 64 cycles) dominates performance so Scalar Engine and Vector Engine have comparable performance. In this case, we suggest using Vector Engine to achieve better precision.
> 
> Estimated cycles on different engines:
> 
> Cost (Engine Cycles) | Condition  
> ---|---  
> `max(MIN_II, N)` | mapped to Scalar Engine `nki.isa.scalar_engine`  
> `max(MIN_II, 8*N)` | mapped to Vector Engine `nki.isa.vector_engine`  
>   
> where,
> 
>   * `N` is the number of elements per partition in the input tile.
> 
>   * `MIN_II` is the minimum instruction initiation interval for small input tiles. `MIN_II` is roughly 64 engine cycles.
> 
> 

> 
> Note `nki.isa.activation(op=nl.reciprocal)` doesn’t support setting bias on NeuronCore-v2.

  2. Reverse square root can run on GpSIMD Engine or Scalar Engine:

> Reverse square root can run on GpSIMD Engine with `nki.isa.tensor_scalar(op0=nl.rsqrt, operand0=0.0)` or on Scalar Engine with `nki.isa.activation(nl.rsqrt)`. GpSIMD Engine performs reverse square root at a higher precision compared to Scalar Engine; however, the computation throughput of reverse square root on GpSIMD Engine is 4x lower than Scalar Engine.

## Trigonometric Functions

### sin

`sin` | Sine of the input, element-wise.  

---

### cos

`cos` | Cosine of the input, element-wise.  

---

### tan

`tan` | Tangent of the input, element-wise.  

---

### arctan

`arctan` | Inverse tangent of the input, element-wise.  

---

### nki.language.sin

nki.language.sin(x, dtype=None)
    

Sine of the input, element-wise.

((Similar to numpy.sin))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has sine values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.sin -- sin(0.0) = 0.0
    a = nl.full((128, 512), 0.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.sin(a)
    expected = nl.full((128, 512), 0.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.cos

nki.language.cos(x, dtype=None)
    

Cosine of the input, element-wise.

((Similar to numpy.cos))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has cosine values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.cos -- cos(0.0) = 1.0
    a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.cos(a)
    expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.tan

nki.language.tan(x, dtype=None)
    

Tangent of the input, element-wise.

((Similar to numpy.tan))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has tangent values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.tan -- tan(0.0) = 0.0
    a = nl.full((128, 512), 0.0, dtype=nl.float32,
                buffer=nl.sbuf)
    b = nl.tan(a)
    expected = nl.full((128, 512), 0.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

---

### nki.language.arctan

# nki.language.arctan
nki.language.arctan(x, dtype=None)
    

Inverse tangent of the input, element-wise.

((Similar to numpy.arctan))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has inverse tangent values of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.arctan -- arctan(0.0) = 0.0
    a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.arctan(a)
    expected = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(b, expected)
    

## Comparison and Logical Operations

### equal

`equal` | Return (x == y) element-wise.  

---

### not_equal

`not_equal` | Return (x != y) element-wise.  

---

### less

`less` | Return (x < y) element-wise.  

---

### less_equal

`less_equal` | Return (x <= y) element-wise.  

---

### greater

`greater` | Return (x > y) element-wise.  

---

### greater_equal

`greater_equal` | Return (x >= y) element-wise.  

---

### logical_and

`logical_and` | Compute the logical AND of two tiles element-wise.  

---

### logical_or

`logical_or` | Compute the logical OR of two tiles element-wise.  

---

### logical_xor

`logical_xor` | Compute the logical XOR of two tiles element-wise.  

---

### logical_not

`logical_not` | Compute the logical NOT element-wise.  

---

### nki.language.equal

nki.language.equal(x, y, dtype=None)
    

Return (x == y) element-wise.

((Similar to numpy.equal))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where equal, 0 otherwise.

---

### nki.language.not_equal

nki.language.not_equal(x, y, dtype=None)
    

Return (x != y) element-wise.

((Similar to numpy.not_equal))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where not equal, 0 otherwise.

---

### nki.language.less

nki.language.less(x, y, dtype=None)
    

Return (x < y) element-wise.

((Similar to numpy.less))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where x < y, 0 otherwise.

---

### nki.language.less_equal

nki.language.less_equal(x, y, dtype=None)
    

Return (x <= y) element-wise.

((Similar to numpy.less_equal))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where x <= y, 0 otherwise.

---

### nki.language.greater

nki.language.greater(x, y, dtype=None)
    

Return (x > y) element-wise.

((Similar to numpy.greater))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where x > y, 0 otherwise.

---

### nki.language.greater_equal

nki.language.greater_equal(x, y, dtype=None)
    

Return (x >= y) element-wise.

((Similar to numpy.greater_equal))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype. Use `dtype=nl.uint8` for a boolean-like result.

Returns:
    

a tile with 1 where x >= y, 0 otherwise.

---

### nki.language.logical_and

nki.language.logical_and(x, y, dtype=None)
    

Compute the logical AND of two tiles element-wise.

((Similar to numpy.logical_and))

Warning

This API is experimental and may change in future releases.

Inputs should be boolean-like (0 or 1 values).

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the logical AND result.

---

### nki.language.logical_or

nki.language.logical_or(x, y, dtype=None)
    

Compute the logical OR of two tiles element-wise.

((Similar to numpy.logical_or))

Warning

This API is experimental and may change in future releases.

Inputs should be boolean-like (0 or 1 values).

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the logical OR result.

---

### nki.language.logical_xor

nki.language.logical_xor(x, y, dtype=None)
    

Compute the logical XOR of two tiles element-wise.

((Similar to numpy.logical_xor))

Warning

This API is experimental and may change in future releases.

Inputs should be boolean-like (0 or 1 values).

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the logical XOR result.

---

### nki.language.logical_not

nki.language.logical_not(x, dtype=None)
    

Compute the logical NOT element-wise.

((Similar to numpy.logical_not))

Warning

This API is experimental and may change in future releases.

Implemented as XOR with 1, so inputs should be boolean-like (0 or 1 values). For non-boolean inputs, use `nl.equal(x, 0)` instead.

Parameters:
    

  * x – a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile with the logical NOT result.

## Bitwise Operations

### bitwise_and

`bitwise_and` | Compute the bitwise AND of two tiles element-wise.  

---

### bitwise_or

`bitwise_or` | Compute the bitwise OR of two tiles element-wise.  

---

### bitwise_xor

`bitwise_xor` | Compute the bitwise XOR of two tiles element-wise.  

---

### invert

`invert` | Compute the bitwise NOT element-wise.  

---

### left_shift

`left_shift` | Left shift the bits of x by y positions element-wise.  

---

### right_shift

`right_shift` | Right shift the bits of x by y positions element-wise.  

---

### nki.language.bitwise_and

nki.language.bitwise_and(x, y, dtype=None)
    

Compute the bitwise AND of two tiles element-wise.

((Similar to numpy.bitwise_and))

Warning

This API is experimental and may change in future releases.

Inputs must be integer typed.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the bitwise AND result.

---

### nki.language.bitwise_or

nki.language.bitwise_or(x, y, dtype=None)
    

Compute the bitwise OR of two tiles element-wise.

((Similar to numpy.bitwise_or))

Warning

This API is experimental and may change in future releases.

Inputs must be integer typed.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the bitwise OR result.

---

### nki.language.bitwise_xor

nki.language.bitwise_xor(x, y, dtype=None)
    

Compute the bitwise XOR of two tiles element-wise.

((Similar to numpy.bitwise_xor))

Warning

This API is experimental and may change in future releases.

Inputs must be integer typed.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the bitwise XOR result.

---

### nki.language.left_shift

nki.language.left_shift(x, y, dtype=None)
    

Left shift the bits of x by y positions element-wise.

((Similar to numpy.left_shift))

Warning

This API is experimental and may change in future releases.

Inputs must be integer typed.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the left-shifted result.

---

### nki.language.right_shift

nki.language.right_shift(x, y, dtype=None)
    

Right shift the bits of x by y positions element-wise.

((Similar to numpy.right_shift))

Warning

This API is experimental and may change in future releases.

Inputs must be integer typed.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value. At least one of x, y must be a tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile with the right-shifted result.

## Activation Functions

### relu

`relu` | ReLU activation, element-wise.  

---

### sigmoid

`sigmoid` | Sigmoid activation, element-wise.  

---

### silu

`silu` | SiLU (Swish) activation, element-wise.  

---

### silu_dx

`silu_dx` | Derivative of SiLU activation, element-wise.  

---

### gelu

`gelu` | GELU activation, element-wise.  

---

### gelu_dx

`gelu_dx` | Derivative of GELU activation, element-wise.  

---

### gelu_apprx_sigmoid

`gelu_apprx_sigmoid` | GELU approximation using sigmoid, element-wise.  

---

### gelu_apprx_sigmoid_dx

`gelu_apprx_sigmoid_dx` | Derivative of sigmoid-approximated GELU, element-wise.  

---

### gelu_apprx_tanh

`gelu_apprx_tanh` | GELU approximation using tanh, element-wise.  

---

### mish

`mish` | Mish activation, element-wise.  

---

### softplus

`softplus` | Softplus activation, element-wise.  

---

### tanh

`tanh` | Hyperbolic tangent, element-wise.  

---

### erf

`erf` | Error function, element-wise.  

---

### erf_dx

`erf_dx` | Derivative of error function, element-wise.  

---

### nki.language.relu

nki.language.relu(x, dtype=None)
    

ReLU activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.sigmoid

nki.language.sigmoid(x, dtype=None)
    

Sigmoid activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.silu

nki.language.silu(x, dtype=None)
    

SiLU (Swish) activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.silu_dx

nki.language.silu_dx(x, dtype=None)
    

Derivative of SiLU activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.gelu

nki.language.gelu(x, dtype=None)
    

GELU activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.gelu_dx

nki.language.gelu_dx(x, dtype=None)
    

Derivative of GELU activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.gelu_apprx_sigmoid

nki.language.gelu_apprx_sigmoid(x, dtype=None)
    

GELU approximation using sigmoid, element-wise.

---

### nki.language.gelu_apprx_sigmoid_dx

nki.language.gelu_apprx_sigmoid_dx(x, dtype=None)
    

Derivative of sigmoid-approximated GELU, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.gelu_apprx_tanh

nki.language.gelu_apprx_tanh(x, dtype=None)
    

GELU approximation using tanh, element-wise.

---

### nki.language.mish

nki.language.mish(x, dtype=None)
    

Mish activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.softplus

nki.language.softplus(x, dtype=None)
    

Softplus activation, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.tanh

nki.language.tanh(x, dtype=None)
    

Hyperbolic tangent, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.erf

nki.language.erf(x, dtype=None)
    

Error function, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### nki.language.erf_dx

nki.language.erf_dx(x, dtype=None)
    

Derivative of error function, element-wise.

This document is relevant for: `Trn2`, `Trn3`

---

### Supported Activation Functions by NKI ISA

## Supported Activation Functions for NKI ISA
Supported Activation Functions by NKI ISA below lists all the activation function supported by the `nki.isa.activation` API. These activation functions are approximated with piece-wise polynomials on Scalar Engine. NOTE: if input values fall outside the supported Valid Input Range listed below, the Scalar Engine will generate invalid output results.

Table 17 Supported Activation Functions by NKI ISA# Function Name | Accepted `op` by Scalar Engine | Valid Input Range  
---|---|---  
Identity | `nki.language.copy` | `[-inf, inf]`  
Square | `nki.language.square` | `[-inf, inf]`  
Sigmoid | `nki.language.sigmoid` | `[-inf, inf]`  
Relu | `nki.language.relu` | `[-inf, inf]`  
Gelu | `nki.language.gelu` | `[-inf, inf]`  
Gelu Derivative | `nki.language.gelu_dx` | `[-inf, inf]`  
Gelu with Tanh Approximation | `nki.language.gelu_apprx_tanh` | `[-inf, inf]`  
Gelu with Sigmoid Approximation | `nki.language.gelu_apprx_sigmoid` | `[-inf, inf]`  
Gelu with Sigmoid Approximation Derivative | `nki.language.gelu_apprx_sigmoid_dx` | `[-inf, inf]`  
Silu | `nki.language.silu` | `[-inf, inf]`  
Silu Derivative | `nki.language.silu_dx` | `[-inf, inf]`  
Tanh | `nki.language.tanh` | `[-inf, inf]`  
Softplus | `nki.language.softplus` | `[-inf, inf]`  
Mish | `nki.language.mish` | `[-inf, inf]`  
Erf | `nki.language.erf` | `[-inf, inf]`  
Erf Derivative | `nki.language.erf_dx` | `[-inf, inf]`  
Exponential | `nki.language.exp` | `[-inf, inf]`  
Natural Log | `nki.language.log` | `[2^-64, 2^64]`  
Sine | `nki.language.sin` | `[-PI, PI]`  
Arctan | `nki.language.arctan` | `[-PI/2, PI/2]`  
Square Root | `nki.language.sqrt` | `[2^-116, 2^118]`  
Reverse Square Root | `nki.language.rsqrt` | `[2^-87, 2^97]`  
Reciprocal | `nki.language.reciprocal` | `±[2^-42, 2^42]`  
Sign | `nki.language.sign` | `[-inf, inf]`  
Absolute | `nki.language.abs` | `[-inf, inf]`  

## Reduction Operations

### max

`max` | Maximum of elements along the specified axis (or axes) of the input.  

---

### min

`min` | Minimum of elements along the specified axis (or axes) of the input.  

---

### sum

`sum` | Sum of elements along the specified axis (or axes) of the input.  

---

### mean

`mean` | Arithmetic mean along the specified axis (or axes) of the input.  

---

### prod

`prod` | Product of elements along the specified axis (or axes) of the input.  

---

### var

`var` | Variance along the specified axis (or axes) of the input.  

---

### all

`all` | Whether all elements along the specified axis (or axes) evaluate to True.  

---

### nki.language.max

nki.language.max(x, axis, dtype=None, keepdims=False)
    

Maximum of elements along the specified axis (or axes) of the input.

((Similar to numpy.max))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – if True, the reduced axes are kept as size-one dimensions.

Returns:
    

a tile with the maximum along the provided axis.

---

### nki.language.min

nki.language.min(x, axis, dtype=None, keepdims=False)
    

Minimum of elements along the specified axis (or axes) of the input.

((Similar to numpy.min))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – if True, the reduced axes are kept as size-one dimensions.

Returns:
    

a tile with the minimum along the provided axis.

---

### nki.language.sum

nki.language.sum(x, axis, dtype=None, keepdims=False)
    

Sum of elements along the specified axis (or axes) of the input.

((Similar to numpy.sum))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – if True, the reduced axes are kept as size-one dimensions.

Returns:
    

a tile with the sum along the provided axis.

---

### nki.language.mean

nki.language.mean(x, axis, dtype=None, keepdims=False)
    

Arithmetic mean along the specified axis (or axes) of the input.

((Similar to numpy.mean))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – if True, the reduced axes are kept as size-one dimensions.

Returns:
    

a tile with the average of elements along the provided axis. Float32 intermediate values are used for the computation.

---

### nki.language.prod

nki.language.prod(x, axis, dtype=None, keepdims=False)
    

Product of elements along the specified axis (or axes) of the input.

((Similar to numpy.prod))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – if True, the reduced axes are kept as size-one dimensions.

Returns:
    

a tile with the product along the provided axis.

---

### nki.language.var

nki.language.var(x, axis, dtype=None, keepdims=False)
    

Variance along the specified axis (or axes) of the input.

((Similar to numpy.var))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * keepdims – currently ignored; result always has keepdims=True shape.

Returns:
    

a tile with the variance of the elements along the provided axis.

---

### nki.language.all

nki.language.all(x, axis, dtype=None)
    

Whether all elements along the specified axis (or axes) evaluate to True.

((Similar to numpy.all))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile with the logical AND reduction along the provided axis.

## Normalization and Softmax

### softmax

`softmax` | Softmax activation function on the input, element-wise.  

---

### rms_norm

`rms_norm` | Apply Root Mean Square Layer Normalization.  

---

### nki.language.softmax

# nki.language.softmax
nki.language.softmax(x, axis=-1, dtype=None)
    

Softmax activation function on the input, element-wise.

((Similar to torch.nn.functional.softmax))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4]

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has softmax of `x`.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.softmax -- uniform input produces uniform output
    a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    result = nl.softmax(a, axis=1)
    

---

### nki.language.rms_norm

nki.language.rms_norm(x, w, axis, n, epsilon=1e-06, dtype=None, compute_dtype=None)
    

Apply Root Mean Square Layer Normalization.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – input tile.

  * w – weight tile.

  * axis – axis along which to compute the root mean square (rms) value.

  * n – total number of values to calculate rms.

  * epsilon – epsilon value used by rms calculation to avoid divide-by-zero.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

  * compute_dtype – (optional) dtype for the internal computation.

Returns:
    

`x / RMS(x) * w`

Examples:
    
    
    import nki.language as nl
    
    # nki.language.rms_norm -- normalize with unit weights
    x = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
    w = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    result = nl.rms_norm(x, w, axis=1, n=512)
    

## Dropout

### dropout

`dropout` | Randomly replace some elements of the input tile `data` with zeros based on input probabilities using Vector Engine.  

### dropout

`dropout` | Randomly zeroes some of the elements of the input tile given a probability rate.  

---

### nki.language.dropout

nki.language.dropout(x, rate, dtype=None)
    

Randomly zeroes some of the elements of the input tile given a probability rate.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile.

  * rate – the probability of zeroing each element. Can be a scalar constant or a tile of shape `(x.shape[0], 1)` for per-partition drop probabilities.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile with randomly zeroed elements of `x`.

---

### nki.isa.dropout

nki.isa.dropout(dst, data, prob, name=None)
    

Randomly replace some elements of the input tile `data` with zeros based on input probabilities using Vector Engine. The probability of replacing input elements with zeros (i.e., drop probability) is specified using the `prob` field: \- If the probability is 1.0, all elements are replaced with zeros. \- If the probability is 0.0, all elements are kept with their original values.

The `prob` field can be a scalar constant or a tile of shape `(data.shape[0], 1)`, where each partition contains one drop probability value. The drop probability value in each partition is applicable to the input `data` elements from the same partition only.

Data type of the input `data` tile can be any valid NKI data types (see Supported Data Types for more information). However, data type of `prob` has restrictions based on the data type of `data`:

  * If data type of `data` is any of the integer types (e.g., int32, int16), `prob` data type must be float32

  * If data type of data is any of the float types (e.g., float32, bfloat16), `prob` data can be any valid float type

The output data type `dst.dtype` must match the input data type `data.dtype`.

Parameters:
    

  * dst – an output tile of the dropout result

  * data – the input tile

  * prob – a scalar or a tile of shape `(data.shape[0], 1)` to indicate the probability of replacing elements with zeros


## Conditional and Selection

### where

`where` | Return elements chosen from x or y depending on condition.  

---

### nki.language.where

nki.language.where(condition, x, y, dtype=None)
    

Return elements chosen from x or y depending on condition.

((Similar to numpy.where))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * condition – condition tile with float values (1.0 for True, 0.0 for False).

  * x – tensor from which to take elements where condition is True.

  * y – tensor from which to take elements where condition is False.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

tensor with elements from x or y based on condition.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.where -- select 10.0 where condition is 1, else 0.0
    cond = nl.full((128, 512), 1.0, dtype=nl.float32,
                   buffer=nl.sbuf)
    x = nl.full((128, 512), 10.0, dtype=nl.float32,
                buffer=nl.sbuf)
    y = nl.full((128, 512), 0.0, dtype=nl.float32,
                buffer=nl.sbuf)
    result = nl.where(cond, x, y)
    expected = nl.full((128, 512), 10.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(result, expected)
    
    # nki.language.where -- select 5.0 where condition is 0
    cond = nl.full((128, 512), 0.0, dtype=nl.float32,
                   buffer=nl.sbuf)
    x = nl.full((128, 512), 10.0, dtype=nl.float32,
                buffer=nl.sbuf)
    y = nl.full((128, 512), 5.0, dtype=nl.float32,
                buffer=nl.sbuf)
    result = nl.where(cond, x, y)
    expected = nl.full((128, 512), 5.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(result, expected)

---

### nki.isa.affine_select

nki.isa.affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, cmp_op=<function equal>, offset=0, name=None)
    

Select elements between an input tile `on_true_tile` and a scalar value `on_false_value` according to a boolean predicate tile using GpSimd Engine.

The predicate tile is calculated on-the-fly in the engine by evaluating an affine expression element-by-element. The affine expression is defined by a `pattern`, `offset`, and `channel_multiplier`, similar to `nisa.iota`. The `pattern` field is a list of lists in the form of `[[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]`. When fewer than 4D `pattern` is provided, NKI compiler automatically pads remaining dimensions with size of 1.

Given a 4D pattern (padded if needed), the instruction generates a predicate using the following pseudo code:
    
    
    num_partitions = dst.shape[0]
    [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]] = pattern
    
    for channel_id in range(num_partitions):
      for w in range(num_w):
        for z in range(num_z):
          for y in range(num_y):
            for x in range(num_x):
              affine_value = offset + (channel_id * channel_multiplier) +
                            (w * step_w) + (z * step_z) + (y * step_y) + (x * step_x)
    
              predicate = cmp_op(affine_value, 0)  # Compare with 0 using cmp_op
    
              if predicate:
                  dst[channel_id, w, z, y, x] = on_true_tile[channel_id, w, z, y, x]
              else:
                  dst[channel_id, w, z, y, x] = on_false_value
    

The above pseudo code assumes `dst` has the same size in every dimension `x/y/z/w` for simplicity. However, the instruction allows any sizes in the free dimension, as long as the number of elements per partition in `dst` matches the product: `num_w * num_z * num_y * num_x`.

A common use case for `affine_select` is to apply a causal mask on the attention scores for transformer decoder models.

Memory types.

The output `dst` tile must be in SBUF. The input `on_true_tile` must also be in SBUF.

Data types.

The input `on_true_tile` and output `dst` tile can be any valid NKI data type (see Supported Data Types for more information). If the data type of `on_true_tile` differs from that of `dst`, the input elements in `on_true_tile`, if selected, are first cast to FP32 before converting to the output data type in `dst`. The `on_false_value` must be float32, regardless of the input/output tile data types.

Layout.

The partition dimension determines the number of active channels for parallel pattern generation and selection. The input tile `on_true_tile`, the calculated boolean predicate tile, and the returned output tile must have the same partition dimension size and.

Tile size.

  * The partition dimension size of `dst` and `on_true_tile` must be the same and must not exceed 128.

  * The number of elements per partition of `dst` and `on_true_tile` must not exceed the physical size of each SBUF partition.

  * The total number of elements in `pattern` must match the number of elements per partition in the `dst` and `on_true_tile` tiles.

Parameters:
    

  * dst – the output tile in SBUF to store the selected values

  * pattern – a list of [step, num] to describe up to 4D tensor sizes and strides for affine expression generation

  * offset – an int32 offset value to be added to every generated affine value

  * channel_multiplier – an int32 multiplier to be applied to the channel (partition) ID

  * on_true_tile – an input tile for selection with a `True` predicate value

  * on_false_value – a scalar value for selection with a `False` predicate value

  * cmp_op – comparison operator to use for predicate evaluation (default: nl.equal)


---

### affine_select

`affine_select` | Select elements between an input tile `on_true_tile` and a scalar value `on_false_value` according to a boolean predicate tile using GpSimd Engine.  

---

### nki.isa.range_select

nki.isa.range_select(dst, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.reset_reduce, reduce_res=None, reduce_op=<function maximum>, range_start=0, on_false_value=-3.4028235e+38, name=None)
    

Select elements from `on_true_tile` based on comparison with bounds using Vector Engine.

Note

Available only on NeuronCore-v3 and newer.

For each element in `on_true_tile`, compares its free dimension index + `range_start` against `bound0` and `bound1` using the specified comparison operators (`comp_op0` and `comp_op1`). If both comparisons evaluate to True, copies the element to the output; otherwise uses `on_false_value`.

Additionally performs a reduction operation specified by `reduce_op` on the results, storing the reduction result in `reduce_res`.

Note on numerical stability:

In self-attention, we often have this instruction sequence: `range_select` (VectorE) -> `reduce_res` -> `activation` (ScalarE). When `range_select` outputs a full row of `fill_value`, caution is needed to avoid NaN in the activation instruction that subtracts the output of `range_select` by `reduce_res` (max value):

  * If `dst.dtype` and `reduce_res.dtype` are both FP32, we should not hit any NaN issue since `FP32_MIN - FP32_MIN = 0`. Exponentiation on 0 is stable (1.0 exactly).

  * If `dst.dtype` is FP16/BF16/FP8, the fill_value in the output tile will become `-INF` since HW performs a downcast from FP32_MIN to a smaller dtype. In this case, you must make sure `reduce_res.dtype` is FP32 to avoid NaN in `activation`. NaN can be avoided because `activation` always upcasts input tiles to FP32 to perform math operations: `-INF - FP32_MIN = -INF`. Exponentiation on `-INF` is stable (0.0 exactly).

Constraints:

The comparison operators must be one of:

  * nl.equal

  * nl.less

  * nl.less_equal

  * nl.greater

  * nl.greater_equal

Partition dim sizes must match across `on_true_tile`, `bound0`, and `bound1`:

  * `bound0` and `bound1` must have one element per partition

  * `on_true_tile` must be one of the FP dtypes, and `bound0/bound1` must be FP32 types.

The comparison with `bound0`, `bound1`, and free dimension index is done in FP32. Make sure `range_start` \+ free dimension index is within 2^24 range.

Numpy equivalent:
    
    
    indices = np.zeros_like(on_true_tile, dtype=np.float32)
    indices[:] = range_start + np.arange(on_true_tile[0].size)
    
    mask = comp_op0(indices, bound0) & comp_op1(indices, bound1)
    select_out_tile = np.where(mask, on_true_tile, on_false_value)
    reduce_tile = reduce_op(select_out_tile, axis=1, keepdims=True)
    

Parameters:
    

  * dst – output tile with selected elements

  * on_true_tile – input tile containing elements to select from

  * on_false_value – constant value to use when selection condition is False. Due to hardware constraints, this must be `FP32_MIN` (`-3.4028235e+38`). See the numerical stability note above for guidance on output dtype selection.

  * comp_op0 – first comparison operator

  * comp_op1 – second comparison operator

  * bound0 – tile with one element per partition for first comparison

  * bound1 – tile with one element per partition for second comparison

  * reduce_op – reduction operator to apply on across the selected output. Currently only `nl.maximum` is supported.

  * reduce_cmd – controls the state of the Vector Engine accumulator registers. Defaults to `reduce_cmd.reset_reduce`. See nki-reduce-cmd for supported values.

  * reduce_res – optional tile to store reduction results.

  * range_start – starting base offset for index array for the free dimension of `on_true_tile`. Defaults to 0, and must be a compile-time integer.

---

### nisa.range_select

### `nisa.range_select` — Parameter Fixes
Beta 2 silently overrode `on_false_value` to `FP32_MIN` and `reduce_cmd` to `reset_reduce`, regardless of user input. In NKI 0.3.0:

  * `reduce_cmd` now works as expected (default `reset_reduce`)

  * `on_false_value` must be `FP32_MIN` due to hardware constraints, but is now documented as a constraint rather than silently ignored

---

### range_select

`range_select` | Select elements from `on_true_tile` based on comparison with bounds using Vector Engine.  

---

### nki.isa.select_reduce

nki.isa.select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=<function maximum>, reverse_pred=False, name=None)
    

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

  * reduce_op – (optional) Reduction operator to apply (only `nl.maximum` is supported)

  * reverse_pred – (optional) Reverse the meaning of the predicate condition, defaults to False


---

### select_reduce

`select_reduce` | Selectively copy elements from either `on_true` or `on_false` to the destination tile based on a `predicate` using Vector Engine, with optional reduction (max).  

## Matrix Multiplication

### matmul

`matmul` | x @ y matrix multiplication of x and y.  

---

### nki.language.matmul

nki.language.matmul(x, y, transpose_x=False)
    

x @ y matrix multiplication of x and y.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile on SBUF (partition dimension <= 128, free dimension <= 128), x’s free dimension must match y’s partition dimension.

  * y – a tile on SBUF (partition dimension <= 128, free dimension <= 512).

  * transpose_x – defaults to False. If True, x is treated as already transposed. If False, an additional transpose will be inserted to make x’s partition dimension the contract dimension of the matmul to align with the Tensor Engine.

Returns:
    

x @ y or x.T @ y if transpose_x=True.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.matmul -- identity.T @ ones = ones
    x = nl.shared_identity_matrix(n=128, dtype=nl.float32)
    y = nl.full((128, 128), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    result_psum = nl.matmul(x, y, transpose_x=True)
    result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(result, result_psum)
    expected = nl.full((128, 128), 1.0, dtype=nl.float32,
                       buffer=nl.sbuf)
    assert nl.equal(result, expected)
    

---

### nisa.nc_matmul

### Matmul Accumulation
`nc_matmul` and `nc_matmul_mx` now have an `accumulate` parameter that controls whether the operation overwrites or accumulates on the destination PSUM tile. The default (`accumulate=None`) auto-detects: the first write to a PSUM location overwrites, and subsequent writes accumulate. This matches Beta 2 behavior.
    
    
    nisa.nc_matmul(dst, stationary, moving, accumulate=True)
    nisa.nc_matmul_mx(dst, stationary, moving, stat_scale, mov_scale, accumulate=True)
    

---

### nc_matmul

`nc_matmul` | Compute `dst = stationary.T @ moving` matrix multiplication using Tensor Engine.  

---

### nki.isa.nc_matmul

nki.isa.nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None)
    

Compute `dst = stationary.T @ moving` matrix multiplication using Tensor Engine.

The figure below illustrates how to map a matrix multiplication from a mathematical definition to `nisa.nc_matmul` on Tensor Engine. The stationary tensor is loaded into the systolic array first and stays in place, while the moving tensor streams through the array during computation. For more detailed discussion of Tensor Engine capabilities, see Trainium arch guide.

Fig. 92 MxKxN Matrix Multiplication Visualization.#

Performance mode.

On NeuronCore-v2, performance mode is not supported. On NeuronCore-v3 and NeuronCore-v4, Tensor Engine supports FP8 double performance mode, enabled by setting performance mode to `double_row`. See Trainium2 arch guide for more details. `double_row` performance mode cannot be combined with Tensor Engine column tiling mode (details below).

Tiling mode. NeuronCore Tensor Engine is built upon a systolic array with 128 rows and 128 columns of processing elements (PEs). Tensor Engine supports both row and column tiling modes, which allow multiple `nc_matmul` instructions with a stationary tile size smaller than [128, 128] to run in parallel to improve hardware utilization. Row tiling mode slices the 128 PE rows into 2x 64 row tiles (NeuronCore-v2 or newer), or 4x 32 row tiles (NeuronCore-v3 or newer). Column tiling mode slices the 128 PE columns in the same fashion. The row and column tile sizes can be set independently in the `tile_size` field as a tuple `(row_size, column_size)`. The stationary tile size must not exceed the chosen `tile_size`.

In addition, a given `nc_matmul` can also pick the exact row and column tile within the 128x128 systolic array, by specifying the starting row and starting column in `tile_position` as a tuple `(start_row, start_column)`. The `start_row` must be a multiple of `row_size` specified in `tile_size` and must not exceed 128. Similarly, the `start_column` must be a multiple of `column_size` and must not exceed 128.

For example, setting `tile_position` to (64, 0) and `tile_size` to (64, 128) means using the bottom half of the systolic array.

Note, `tile_position` and `tile_size` must both be set to enable tiling mode. If they are not set, the default is to use the full systolic array, which is equivalent to `tile_position=(0, 0)` and `tile_size=(128, 128)`. The values in `tile_position` and `tile_size` tuples can be integers or affine expressions.

Accumulation mode.

The `accumulate` parameter controls whether the matmul result should overwrite or accumulate on top of the `dst` PSUM tile. When `accumulate=False`, the result overwrites the existing content. When `accumulate=True`, the result is added to the existing content. When `accumulate=None` (default), the behavior is auto-detected: the first write to a PSUM location overwrites, and subsequent writes to the same location accumulate. Multiple `nc_matmul` instructions with `accumulate=True` can form an accumulation group before the PSUM tile content is evicted back to SBUF.

Transpose mode.

Tensor Engine can transpose a tile in SBUF by loading it as a stationary tile and using an identity matrix as the moving tile. Starting NeuronCore-v3, turning on transpose mode by setting `is_transpose=True` enables bit-accurate data transpose, which can transpose tensors with NaN/Inf values properly. See Trainium2 arch guide for more details.

On NeuronCore-v2, Tensor Engine does not support transpose mode natively. However, setting `is_transpose=True` ensures neuron-profile identifies this instruction as a transpose for performance metric accounting purposes.

Memory types.

The `nc_matmul` instruction must read inputs from SBUF and write outputs to PSUM. Therefore, the `stationary` and `moving` must be SBUF tiles, and `dst` tile must be a PSUM tile.

Data types.

The input `stationary` and `moving` tiles can be one of these supported data types: `float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32`. The `stationary` and `moving` tiles can have different data types, with one exception: if one of the input tiles is `tfloat32/float32`, the other tile must also be `tfloat32/float32`. On NeuronCore-v3 and NeuronCore-v4, when performance mode is `double_row`, `stationary` and `moving` tiles must be one of `float8_e4m3` or `float8_e5m2`, but the two input tiles can have different float8 formats.

The accumulation precision internal to Tensor Engine is float32. The `dst` tile must be a float32 tile in NeuronCore-v2 and NeuronCore-v3. Starting NeuronCore-v4, `dst` can either be a float32 or bfloat16 tile.

Layout.

If performance mode is off, the contraction dimension of the matmul must be along the partition dimension in both `stationary` and `moving` tiles.

If performance mode is `double_row`, the contraction dimension of the matmul is split between the partition dimension and the first free dimension after the partition dimension in both `stationary` and `moving` tiles. The first free dimension must be 2. For example, to perform a matmul of `[1, 256]@[256, 3]=[1, 3]`, the stationary tile is of shape `[128, 2, 1]`, while the moving tile is of shape `[128, 2, 3]`.

Regardless of performance mode, the free dimension of the `stationary` tile matches the partition dimension of the output `dst` tile in size, while the free dimension of the `moving` tile matches the free dimension of the `dst` tile in size.

Tile size.

The partition dimension sizes of the `stationary` and `moving` tiles must be identical. They must not exceed 128 when tiling mode is off or `row_size` specified in `tile_size` when tiling mode is on. The free dimension size of `stationary` must not exceed 128 when tiling mode is off or `column_size` in `tile_size` when tiling mode is on.

On NeuronCore-v2 and -v3, the free dimension size of `moving` tile must not exceed 512, matching the maximum number of float32 elements per PSUM bank. Starting NeuronCore-v4, the free dimension size of `moving` tile can go up to 4096 for float32 `dst` or 8192 for bfloat16 `dst`, matching the size of 8x PSUM banks (the entire PSUM).

Explicit tiling is required when the high-level matmul operation exceeds the tile size limits of `nc_matmul`.

Profiler view syntax.

Each `nc_matmul` call lowers to two ISA instructions in the profiler: a load instruction (to load the stationary operand into the Tensor Engine) followed by a multiply instruction. Both instructions will appear in profiler output for a single `nc_matmul` call.

The multiply instruction operands are displayed in a compact ISA syntax:
    
    
    src=<dtype>@<address>[<strides>][<num_elem>]
    dst=<dtype>@<address>[<strides>][<num_elem>]
    <M>*<K> acc_flags=<flags> psum_zero=<val>
    

Where:

  * `<dtype>`: data type (e.g., `bfloat16`, `fp8e4`, `fp8e5`)

  * `<address>`: hex memory address in SBUF (for src) or PSUM (for dst)

  * `[<strides>]`: element strides per dimension (multi-dimensional)

  * `[<num_elem>]`: number of elements per dimension (multi-dimensional)

  * `<M>*<K>`: matmul dimensions (M rows × K contraction)

  * `acc_flags`: accumulator control flags (e.g., `2` = reset accumulator)

  * `psum_zero`: PSUM zero-initialization control value

Parameters:
    

  * dst – the matmul output

  * stationary – the stationary operand

  * moving – the moving operand

  * is_stationary_onezero – hints to the compiler whether the `stationary` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `stationary` tile is in float32; the field has no impact for non-float32 `stationary`

  * is_moving_onezero – hints to the compiler whether the `moving` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `moving` tile is in float32; the field has no impact for non-float32 `moving`

  * is_transpose – controls Tensor Engine transpose mode on/off starting NeuronCore-v3

  * accumulate – if True, accumulate the matmul result into the existing `dst` PSUM tile content; if False, overwrite the existing content; if None (default), auto-detect based on whether this PSUM location was previously written. Not exposed for `nc_transpose`.

  * tile_position – a 2D tuple (start_row, start_column) to control starting row in Tensor Engine tiling mode; start_column must be 0

  * tile_size – a 2D tuple (row_size, column_size) to control row tile size in Tensor Engine tiling mode; column_size must be 128

  * perf_mode – controls Tensor Engine FP8 double performance mode on/off starting NeuronCore-v3: `matmul_perf_mode.none` (default) disables double FP8 mode; `matmul_perf_mode.double_row` enables double FP8 mode which achieves 2x matmul throughput by packing two FP8 weight/ifmap element pairs and computing two multiplications in parallel per cycle; cannot be combined with column tiling mode. See the Trainium2 arch guide for more information.


---

### nki.isa.nc_matmul_mx

nki.isa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None, accumulate=None, name=None)
    

Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.

Note

Available only on NeuronCore-v4 and newer.

The NeuronCore-v4 Tensor Engine supports matrix multiplication of MXFP8/MXFP4 quantized matrices as defined in the OCP Microscaling standard. This instruction performs matrix multiplication between quantized `stationary` and `moving` matrices while applying dequantization scales during computation. The micro-scaling group size is 32 elements in groups of 8 partitions × 4 elements per partition of both `stationary` and `moving` tensors. See Trainium3 arch guide for more detailed discussion.

Tiling Mode.

NeuronCore Tensor Engine is built upon a systolic array with 128 rows and 128 columns of processing elements (PEs). For `nc_matmul_mx`, Tensor Engine supports only row tiling mode, which allows multiple `nc_matmul_mx` instructions with a stationary partition dimension size smaller than 128 to run in parallel to improve hardware utilization. Row tiling mode slices the 128 PE rows into 2x 64 row tiles or 4x 32 row tiles.

The row tile size can be set in the `tile_size` field as a tuple `(row_size, column_size)`, where `column_size` must be 128. The stationary tile size must not exceed the chosen `tile_size`.

A given `nc_matmul_mx` can pick the exact row tile within the 128x128 systolic array by specifying the starting row in `tile_position` as a tuple `(start_row, start_column)`, where `start_column` must be 0. The `start_row` must be a multiple of `row_size` specified in `tile_size` and must not exceed 128.

For example, setting `tile_position` to (64, 0) and `tile_size` to (64, 128) means using the bottom half of the systolic array.

Note, `tile_position` and `tile_size` must both be set to enable tiling mode. If they are not set, the default is to use the full systolic array, which is equivalent to `tile_position=(0, 0)` and `tile_size=(128, 128)`. The values in `tile_position` and `tile_size` tuples can be integers or affine expressions.

Memory types.

The `nc_matmul_mx` instruction must read inputs from SBUF and write outputs to PSUM. Therefore, the `stationary`, `moving`, `stationary_scale`, and `moving_scale` must be SBUF tiles, and `dst` tile must be a PSUM tile.

Data types.

The input `stationary` and `moving` tiles must be float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4 (4-packed quantized data types). The `stationary_scale` and `moving_scale` tiles must be uint8. The `dst` tile can be float32 or bfloat16.

Layout.

The contraction dimension of the matrix multiplication is along the partition dimension of `stationary` and `moving` tensors and also the x4 dimension within each packed data type element (float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4).

The free dimension of the `stationary` tile matches the partition dimension of the output `dst` tile in size, while the free dimension of the `moving` tile matches the free dimension of the `dst` tile in size.

The scale tensors follow a special layout requirement. See more details in `nisa.quantize_mx` API doc.

Tile size

  * The partition dimension size of `stationary` and `moving` must be identical and be a multiple of 32, not exceeding 128.

  * The free dimension size of `stationary` must be even and not exceed 128.

  * The free dimension size of `moving` must not exceed 512 when `dst` is in float32 or 1024 when `dst` is in bfloat16.

  * The scale tensors have partition dimensions that depend on whether the data tensors span multiple quadrants. See more details in `nisa.quantize_mx` API doc.

Profiler view syntax.

`nc_matmul_mx` uses the same profiler output format as nisa.nc_matmul, except the source access pattern is interpreted as an MX-quantized tensor: `src=<dtype>@$MX[<data_addr>,<scale_addr>,<start_scale_partition>]@[<step_elem>][<num_elem>]`.

Parameters:
    

  * dst – the matrix multiplication output (PSUM tile)

  * stationary – the stationary quantized matrix (SBUF tile)

  * moving – the moving quantized matrix (SBUF tile)

  * stationary_scale – the dequantization scales for stationary matrix (SBUF tile)

  * moving_scale – the dequantization scales for moving matrix (SBUF tile)

  * tile_position – a 2D tuple (start_row, start_column) to control starting row and column in Tensor Engine tiling mode

  * tile_size – a 2D tuple (row_size, column_size) to control row and column tile sizes in Tensor Engine tiling mode

  * accumulate – if True, accumulate the matmul result into the existing `dst` PSUM tile content; if False, overwrite the existing content; if None (default), auto-detect based on whether this PSUM location was previously written


---

### nc_matmul_mx

`nc_matmul_mx` | Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.  

## Transpose

### transpose

`transpose` | Transposes a 2D tile between its partition and free dimension.  

---

### nki.language.transpose

nki.language.transpose(x, dtype=None)
    

Transposes a 2D tile between its partition and free dimension.

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – 2D input tile.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.

Returns:
    

a tile that has the values of the input tile with its partition and free dimensions swapped.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.transpose -- transpose of identity is identity
    x = nl.shared_identity_matrix(n=128, dtype=nl.float32)
    result_psum = nl.transpose(x)
    result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(result, result_psum)
    assert nl.equal(result, x)
    

---

### nc_transpose

`nc_transpose` | Perform a 2D transpose between the partition axis and the free axis of input `data` using Tensor or Vector Engine.  

---

### nki.isa.nc_transpose

# nki.isa.nc_transpose
nki.isa.nc_transpose(dst, data, engine=engine.unknown, name=None)
    

Perform a 2D transpose between the partition axis and the free axis of input `data` using Tensor or Vector Engine.

If the `data` tile has more than one free axis, this API implicitly flattens all free axes into one axis and then performs a 2D transpose.

2D transpose on Tensor Engine is implemented by performing a matrix multiplication between `data` as the stationary tensor and an identity matrix as the moving tensor. This is equivalent to calling `nisa.nc_matmul` directly with `is_transpose=True`. See architecture guide for more information. On NeuronCore-v2, Tensor Engine transpose is not bit-accurate if the input `data` contains NaN/Inf. You may consider replacing NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix. Starting NeuronCore-v3, all Tensor Engine transpose is bit-accurate.

Memory types.

Tensor Engine `nc_transpose` must read the input tile from SBUF and write the transposed result to PSUM. Vector Engine `nc_transpose` can read/write from/to either SBUF or PSUM.

Data types.

The input `data` tile can be any valid NKI data type (see Supported Data Types for more information). The output `dst` tile must have the same data type as that of `data`.

Layout. The partition dimension of `data` tile becomes the free dimension of the `dst` tile. Similarly, the free dimension of the `data` tile becomes the partition dimension of the `dst` tile.

Tile size. Tensor Engine `nc_transpose` can handle an input tile of shape [128, 128] or smaller, while Vector Engine can handle shape [32, 32] or smaller. If no `engine` is specified, Neuron Compiler will automatically select an engine based on the input shape.

Parameters:
    

  * dst – the transpose output

  * data – the input tile to be transposed

  * engine – specify which engine to use for transpose: `nki.isa.engine.tensor` or `nki.isa.engine.vector`; by default, the best engine will be selected for the given input tile shape

## ISA Tensor-Tensor and Tensor-Scalar Operations

### nisa.tensor_tensor

    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
    

As in step 4, you allocate a space for the `c_tile` in SBUF, using `nl.ndarray`. Since the shape of the output will be the same shape as the inputs, you can use the `a_input` data type and shape for the allocation. You use the `nisa.tensor_tensor` instruction to perform element-wise calculation on two tensors. The first argument of `tensor_tensor` is the destination tensor, `c_tile`, and the sources, `a_tile` and `b_tile`, follow it. You must also provide an op which tells `tensor_tensor` which operation to perform on the inputs. In this case, you use `op=nl.add` to specify addition.

---

### tensor_tensor

`tensor_tensor` | Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine.  

---

### nki.isa.tensor_tensor

nki.isa.tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None)
    

Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine. The two tiles must have the same partition axis size and the same number of elements per partition.

The element-wise operator is specified using the `op` field. Valid choices for `op`:

  1. Any supported binary operator that runs on the Vector Engine. (See Supported Math Operators for NKI ISA for details.)

  2. `nl.power`. (Which runs on the GpSimd engine.)

For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, the behavior depends on the data types:

  * Float types: The engine casts input data types to float32 and performs the element-wise operation in float32 math. The float32 results are cast to `dst.dtype` at no additional performance cost.

  * int32/uint32 types: When all input/output tiles are int32 or uint32, the operation defaults to GpSimd Engine, which uses native integer arithmetic. This ensures exact results for all 32-bit integer values. You may override this by passing `engine=nki.isa.engine.vector` explicitly.

Since GpSimd Engine cannot access PSUM, the input/output tiles cannot be in PSUM if `op` is `nl.power`. Similarly, the automatic GpSimd dispatch for int32/uint32 falls back to Vector Engine when any operand resides in PSUM. (See NeuronCore-v2 Compute Engines for details.)

Otherwise, the output tile can be in either SBUF or PSUM. However, the two input tiles, `data1` and `data2` cannot both reside in PSUM. The three legal cases are:

  1. Both `data1` and `data2` are in SBUF.

  2. `data1` is in SBUF, while `data2` is in PSUM.

  3. `data1` is in PSUM, while `data2` is in SBUF.

Note, if you need broadcasting capability in the free dimension for either input tile, you should consider using nki.isa.tensor_scalar API instead, which has better performance than `nki.isa.tensor_tensor` in general.

Parameters:
    

  * dst – an output tile of the element-wise operation

  * data1 – lhs input operand of the element-wise operation

  * data2 – rhs input operand of the element-wise operation

  * op – a binary math operator (see Supported Math Operators for NKI ISA for supported operators)

  * engine – (optional) the engine to use for the operation: nki.isa.engine.vector, nki.isa.engine.gpsimd or nki.isa.engine.unknown (default, let compiler select best engine based on the input tile shape).


---

### nisa.tensor_scalar

      nisa.tensor_scalar(cond, cond, nl.add, -1)

---

### tensor_scalar

`tensor_scalar` | Apply up to two math operators to the input `data` tile by broadcasting scalar/vector operands in the free dimension using Vector or Scalar or GpSimd Engine: `(data <op0> operand0) <op1> operand1`.  

---

### nki.isa.tensor_scalar

nki.isa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=engine.unknown, name=None)
    

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

If arithmetic operators are used, the `tensor_scalar` instruction can run on Vector or Scalar or GpSimd Engine. However, each engine supports limited arithmetic operators (see :ref:`tbl-aluop`). The Scalar Engine on trn2 only supports some operator combinations:

>   * `op0=nl.multiply` and `op1=nl.add`
> 
>   * `op0=nl.multiply` and `op1=None`
> 
>   * `op0=nl.add` and `op1=None`
> 
> 

Also, arithmetic operators impose no restriction on the data types of input tensor `data` and output tensor `dst`, but the operand0 and operand1 (if used) must be float32. The compute engine automatically casts `data.dtype` to float32 and performs the operators in float32 math. The float32 computation results are cast to `dst.dtype` at no additional performance cost.

Parameters:
    

  * dst – an output tile of `(data <op0> operand0) <op1> operand1` computation

  * data – the input tile

  * op0 – the first math operator used with operand0 (see Supported Math Operators for NKI ISA for supported operators)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse0 – reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`

  * op1 – the second math operator used with operand1 (see Supported Math Operators for NKI ISA for supported operators); this operator is optional

  * operand1 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse1 – reverse ordering of inputs to `op1`; if false, `operand1` is the rhs of `op1`; if true, `operand1` is the lhs of `op1`

  * engine – (optional) the engine to use for the operation: nki.isa.engine.vector, nki.isa.engine.scalar, nki.isa.engine.gpsimd (only allowed for rsqrt) or nki.isa.engine.unknown (default, let compiler select best engine based on the input tile shape).


---

### scalar_tensor_tensor

`scalar_tensor_tensor` | Apply two math operators in sequence using Vector Engine: `(data <op0> operand0) <op1> operand1`.  

---

### nki.isa.scalar_tensor_tensor

nki.isa.scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None)
    

Apply two math operators in sequence using Vector Engine: `(data <op0> operand0) <op1> operand1`.

This instruction is equivalent to running two operations back-to-back: 1\. `temp_result = tensor_scalar(data, op0, operand0)` \- broadcast `operand0` and apply `op0` 2\. `dst = tensor_tensor(temp_result, op1, operand1)` \- element-wise operation with `operand1`

The `operand0` can be either a compile-time constant scalar for broadcast across all elements of `data` or a tile of shape `(data.shape[0], 1)` for broadcast along the free dimension. The `operand1` tile must have the same shape as `data` for element-wise operation.

The scalar broadcasting in the first operation is performed at no additional performance cost, making this instruction have approximately the same latency as a regular `tensor_tensor` instruction.

Both `op0` and `op1` must be arithmetic operators (see Supported Math Operators for NKI ISA for supported operators). Bitvec operators are not supported. When the operators are non-commutative (e.g., subtract), operand ordering can be reversed using `reverse0` and `reverse1` flags.

Memory types.

The input `data` tile can be an SBUF or PSUM tile. The `operand0` can be an SBUF or PSUM tile or a compile-time constant scalar. The `operand1` must be an SBUF or PSUM tile. However, `data` and `operand1` cannot both reside in PSUM. The output `dst` tile can be written to either SBUF or PSUM.

Data types.

All input tiles can be any supported NKI data type (see Supported Data Types for more information). The Vector Engine automatically casts input data types to float32 and performs all computations in float32 math. The float32 results are cast to the data type of output `dst`.

Layout.

The parallel computation dimension of `nisa.scalar_tensor_tensor` is along the partition dimension.

Tile size.

The partition dimension size of input `data`, `operand1`, and output `dst` tiles must be the same and must not exceed 128. The total number of elements per partition of input `data`, `operand1`, and output `dst` tiles must be the same and must not exceed the physical size of each SBUF partition. If operand0 is not a scalar, the partition dimension size of `operand0` must be the same as that of `data` and the number of elements per partition of `operand0` must be 1.

Parameters:
    

  * dst – the output tile

  * data – the input tile

  * op0 – the first math operator used with operand0 (see Supported Math Operators for NKI ISA for supported operators)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse0 – reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`

  * op1 – the second math operator used with operand1 (see Supported Math Operators for NKI ISA for supported operators)

  * operand1 – a tile with the same size as `data` for element-wise operation

  * reverse1 – reverse ordering of inputs to `op1`; if false, `operand1` is the rhs of `op1`; if true, `operand1` is the lhs of `op1`


---

### tensor_reduce

`tensor_reduce` | Apply a reduction operation to the free axes of an input `data` tile using Vector Engine.  

---

### nki.isa.tensor_reduce

nki.isa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None)
    

Apply a reduction operation to the free axes of an input `data` tile using Vector Engine.

The reduction operator is specified in the `op` input field (see Supported Math Operators for NKI ISA for a list of supported reduction operators). `nisa.tensor_reduce` supports two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or) and 2) arithmetic operators (e.g., add, subtract, multiply).

The reduction axes are specified in the `axis` field as an int or list of ints indicating which dimensions to reduce. The reduction axes must be the last contiguous free dimension(s) of the tile, ending at the final dimension. Axis 0 (partition axis) cannot be reduced.

For example, given a 4D tile `(P, D1, D2, D3)`:

  * `axis=(3,)` reduces only `D3`

  * `axis=(2, 3)` reduces `D2` and `D3`

  * `axis=(1, 2, 3)` reduces `D1`, `D2`, and `D3`

When the reduction `op` is an arithmetic operator, the instruction can also multiply the output reduction results by `-1.0` before writing into the output tile, at no additional performance cost. This behavior is controlled by the `negate` input field.

Memory types.

Both the input `data` and `dst` tiles can be in SBUF or PSUM.

Data types.

For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, the input/output data types can be any supported NKI data types, but the engine automatically casts input data types to float32 and performs the reduction operation in float32 math. The float32 reduction results are cast to the data type of `dst`.

Layout.

`nisa.tensor_reduce` only supports free axes reduction. Therefore, the partition dimension of the input `data` is considered the parallel compute dimension. To perform a partition axis reduction, we can either:

  1. invoke a `nisa.nc_transpose` instruction on the input tile and then this `nisa.tensor_reduce` on the transposed tile, or

  2. invoke `nisa.nc_matmul` instructions to multiply a `nl.ones([128, 1], dtype=data.dtype)` tile as a stationary tensor with the input tile as a moving tensor. See more discussion on Tensor Engine alternative usage in Trainium architecture guide.

Tile size.

The partition dimension size of input `data` and output `dst` tiles must be the same and must not exceed 128. The number of elements per partition of `data` must not exceed the physical size of each SBUF partition. The number of elements per partition in `dst` must be consistent with the `axis` field. For example, if `axis` indicates all free dimensions of `data` are reduced, the number of elements per partition in `dst` must be 1.

Parameters:
    

  * dst – output tile of the reduction result

  * op – the reduction operator (see Supported Math Operators for NKI ISA for supported reduction operators)

  * data – the input tile to be reduced

  * axis – int or tuple/list of ints. The axis (or axes) along which to reduce; must be the last contiguous free dimension(s) ending at the final dim. For example, for a 4D tile `(P, D1, D2, D3)`: valid values are `(3,)`, `(2, 3)`, or `(1, 2, 3)`. Axis 0 (partition dim) cannot be reduced.

  * negate – if True, reduction result is multiplied by `-1.0`; only applicable when op is an arithmetic operator

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

---

### tensor_partition_reduce

`tensor_partition_reduce` | Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine.  

---

### nki.isa.tensor_partition_reduce

nki.isa.tensor_partition_reduce(dst, op, data, name=None)
    

Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine.

Parameters:
    

  * dst – output tile with reduced result

  * op – the reduction operator (add, max, bitwise_or, bitwise_and)

  * data – the input tile to be reduced


---

### tensor_scalar_reduce

`tensor_scalar_reduce` | Perform the same computation as `nisa.tensor_scalar` with one math operator and also a reduction along the free dimension of the `nisa.tensor_scalar` result using Vector Engine.  

---

### nki.isa.tensor_scalar_reduce

nki.isa.tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, name=None)
    

Perform the same computation as `nisa.tensor_scalar` with one math operator and also a reduction along the free dimension of the `nisa.tensor_scalar` result using Vector Engine.

Refer to nisa.tensor_scalar for semantics of `data/op0/operand0`. Unlike regular `nisa.tensor_scalar` where two operators are supported, only one operator is supported in this API. Also, `op0` can only be arithmetic operation in Supported Math Operators for NKI ISA. Bitvec operators are not supported in this API.

In addition to nisa.tensor_scalar computation, this API also performs a reduction along the free dimension(s) of the nisa.tensor_scalar result, at a small additional performance cost. The reduction result is returned in `reduce_res` in-place, which must be a SBUF/PSUM tile with the same partition axis size as the input tile `data` and one element per partition. The `reduce_op` can be any of `nl.add`, `nl.subtract`, `nl.multiply`, `nl.max` or `nl.min`.

Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will reduce across all of them.

\\[\begin{split}result = data <op0> operand0 \\\ reduce\\_res = reduce\\_op(dst, axis=<FreeAxis>)\end{split}\\]

Parameters:
    

  * dst – an output tile of `(data <op0> operand0)` computation

  * data – the input tile

  * op0 – the math operator used with operand0 (any arithmetic operator in Supported Math Operators for NKI ISA is allowed)

  * operand0 – a scalar constant or a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile

  * reverse0 – (not supported yet) reverse ordering of inputs to `op0`; if false, `operand0` is the rhs of `op0`; if true, `operand0` is the lhs of `op0`. <– currently not supported yet.

  * reduce_op – the reduce operation to perform on the free dimension of `data <op0> operand0`

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `reduce_op(data <op0> operand0)` is written in-place into the tile.


---

### tensor_scalar_cumulative

`tensor_scalar_cumulative` | Perform tensor-scalar arithmetic operation with cumulative reduction using Vector Engine.  

---

### nki.isa.tensor_scalar_cumulative

nki.isa.tensor_scalar_cumulative(dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce, name=None)
    

Perform tensor-scalar arithmetic operation with cumulative reduction using Vector Engine.

The operation applies a scalar operation to each tensor element, then performs a cumulative reduction, storing the cumulative results in the destination tensor.

The operation can be expressed in pseudocode as:
    
    
    if reduce_cmd == reset_reduce:
        if op1 == add or op1 == subtract:
            reg = 0
        elif op1 == mult:
            reg = 1
        elif op1 == max:
            reg = -inf
        elif op1 == min:
            reg = +inf
    elif reduce_cmd == reduce:
        reg = reg
    elif reduce_cmd == load_reduce:
        reg = imm1
    
    for i in len(in_tensor):
        if not reverse0:
            reg = op1(op0(in_tensor[i], imm0), reg)
            out_tensor[i] = reg
        else:
            reg = op1(op0(imm0, in_tensor[i]), reg)
            out_tensor[i] = reg
    

Operation constraints:

  * Scalar operation (`op0`) must be an arithmetic op (e.g., add, mult, max)

  * Reduction operation (`op1`) is limited to add, subtract, mult, max, min

  * Input / output dtypes are restricted to BF16, FP16, FP32, FP8, UINT8, UINT16, INT8, INT16
    
    * INT32/UINT32 are not supported as input/output dtypes (ISA limitation)

Accumulator behavior:

The Vector Engine maintains internal accumulator registers controlled via `reduce_cmd`:

  * `reset_reduce`: Reset accumulator based on reduction operation type

  * `load_reduce`: Initialize accumulator with `imm1` value

  * `reduce`: Continue with existing accumulator value

Parameters:
    

  * dst – The destination tensor to write cumulative results to

  * src – The source tensor to process

  * op0 – Scalar arithmetic operation to apply to each element

  * op1 – Cumulative arithmetic operation for cumulative computation

  * imm0 – Scalar or vector value for tensor-scalar operation. Must be FP32 datatype

  * imm1 – (optional) Initial scalar or vector value for the accumulator when `load_reduce` is specified as the `reduce_cmd`. Must be FP32 datatype

  * reduce_cmd – (optional) Control accumulator behavior using `nisa.reduce_cmd` values, defaults to `reset_reduce`


---

### tensor_tensor_scan

`tensor_tensor_scan` | Perform a scan operation of two input tiles using Vector Engine.  

---

### nki.isa.tensor_tensor_scan

nki.isa.tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None)
    

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

Input/output data types can be any supported NKI data type (see Supported Data Types), but the engine automatically casts input data types to float32 and performs the computation in float32 math. The float32 computation results are cast to `dst.dtype` at no additional performance cost.

Parameters:
    

  * dst – an output tile of the scan operation

  * data0 – lhs input operand of the scan operation

  * data1 – rhs input operand of the scan operation

  * initial – starting state of the scan; can be a SBUF/PSUM tile with 1 element/partition or a scalar compile-time constant

  * op0 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)

  * op1 – a binary arithmetic math operator (see Supported Math Operators for NKI ISA for supported operators)

  * reverse0 – reverse ordering of inputs to `op0`; if false, `data0` is the lhs of `op0`; if true, `data0` is the rhs of `op0`

  * reverse1 – reverse ordering of inputs to `op1`; if false, `data1` is the rhs of `op1`; if true, `data1` is the lhs of `op1`


## ISA Activation and Activation-Reduce

### activation

`activation` | Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation before the activation and an optional reduction operation after the activation in the same instruction.  

---

### nki.isa.activation

nki.isa.activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None)
    

Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation before the activation and an optional reduction operation after the activation in the same instruction.

The activation function is specified in the `op` input field (see Supported Activation Functions for NKI ISA for a list of supported activation functions and their valid input ranges).

`nisa.activation` can optionally multiply the input `data` by a scalar or vector `scale` and then add another vector `bias` before the activation function is applied.

After the activation function is applied, Scalar Engine can also reduce along the free dimensions of the activated data per lane, using `reduce_op` operation. `reduce_op` must be `nl.add`.

The reduction result is then either stored into or reduced on top of a set of internal engine registers called `reduce_regs` (one 32-bit register per compute lane, 128 registers in total), controlled by the `reduce_cmd` field:

  * `nisa.reduce_cmd.reset`: Reset `reduce_regs` to zero only.

  * `nisa.reduce_cmd.idle`: Do not modify `reduce_regs`.

  * `nisa.reduce_cmd.reduce`: Reduce activated data over existing values in `reduce_regs`.

  * `nisa.reduce_cmd.reset_reduce`: Reset `reduce_regs` to zero and then store the reduction result of the activated data.

`nisa.activation` can also emit another instruction to read out `reduce_regs` by passing an SBUF/PSUM tile in the `reduce_res` arguments. The `reduce_regs` state can persist across multiple `nisa.activation` instructions without the need to be evicted back to SBUF/PSUM (`reduce_res` tile).

The following is the pseudo code for `nisa.activation`:
    
    
    output = op(data * scale + bias)
    
    if reduce_cmd == nisa.reduce_cmd.reset or reduce_cmd == nisa.reduce_cmd.reset_reduce:
        reduce_regs = 0
    
    result = reduce_op(reduce_regs, reduce_op(output, axis=<FreeAxis>))
    
    if reduce_cmd == nisa.reduce_cmd.reduce or reduce_cmd == nisa.reduce_cmd.reset_reduce:
        reduce_regs += result
    
    if reduce_res:
        reduce_res = reduce_regs
    

All these optional operations incur no further performance penalty compared to only applying the activation function, except reading out `reduce_regs` into `reduce_res` will have a small overhead due to an extra instruction.

Memory types.

The input `data` tile can be an SBUF or PSUM tile. Similarly, the instruction can write the output `dst` tile into either SBUF or PSUM.

Data types.

Both input `data` and output `dst` tiles can be in any valid NKI data type (see Supported Data Types for more information). The Scalar Engine always performs the math operations in float32 precision. Therefore, the engine automatically casts the input `data` tile to float32 before performing multiply/add/activate specified in the activation instruction. The engine is also capable of casting the float32 math results into another output data type in `dst` at no additional performance cost. The `scale` parameter must have a float32 data type, while the `bias` parameter can be any supported dtype except tfloat32.

Layout.

The `scale` can either be a compile-time constant scalar or a `[N, 1]` vector from SBUF/PSUM. `N` must be the same as the partition dimension size of `data`. In NeuronCore-v2, the `bias` must be a `[N, 1]` vector, but starting NeuronCore-v3, `bias` can either be a compile-time constant scalar or a `[N, 1]` vector similar to `scale`.

When the `scale` (or similarly, `bias`) is a scalar, the scalar is broadcasted to all the elements in the input `data` tile to perform the computation. When the `scale` (or `bias`) is a vector, the `scale` (or `bias`) value in each partition is broadcast along the free dimension of the `data` tile.

Tile size.

The partition dimension size of input `data` and output `dst` tiles must be the same and must not exceed 128. The number of elements per partition of `data` and `dst` tiles must be the same and must not exceed the physical size of each SBUF partition.

Parameters:
    

  * dst – the activation output

  * op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)

  * data – the input tile; layout: (partition axis <= 128, free axis)

  * scale – a scalar or a vector for multiplication

  * bias – a scalar (NeuronCore-v3 or newer) or a vector for addition

  * reduce_op – the reduce operation to perform on the free dimension of the activated data

  * reduce_res – a tile of shape `(data.shape[0], 1)` to hold the final state of `reduce_regs`.

  * reduce_cmd – an enum member from `nisa.reduce_cmd` to control the state of `reduce_regs`.

---

### activation_reduce

`activation_reduce` | Perform the same computation as `nisa.activation` and also a reduction along the free dimension of the `nisa.activation` result using Scalar Engine.  

---

### nki.isa.activation_reduce

nki.isa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None)
    

Perform the same computation as `nisa.activation` and also a reduction along the free dimension of the `nisa.activation` result using Scalar Engine. The results for the reduction is stored in the reduce_res.

This API is equivalent to calling `nisa.activation` with `reduce_cmd=nisa.reduce_cmd.reset_reduce` and passing in reduce_res. This API is kept for backward compatibility, we recommend using `nisa.activation` moving forward.

Refer to nisa.activation for semantics of `op/data/bias/scale`.

In addition to nisa.activation computation, this API also performs a reduction along the free dimension(s) of the nisa.activation result, at a small additional performance cost. The reduction result is returned in `reduce_res` in-place, which must be a SBUF/PSUM tile with the same partition axis size as the input tile `data` and one element per partition. On NeuronCore-v2, the `reduce_op` must be `nl.add`.

There are 128 registers on the scalar engine for storing reduction results, corresponding to the 128 partitions of the input. These registers are shared between `activation` and `activation_accu` calls. This instruction first resets those registers to zero, performs the reduction on the value after activation function is applied, stores the results into the registers, then reads out the reduction results from the register, eventually store them into `reduce_res`.

Note that `nisa.activation` can also change the state of the register. It’s user’s responsibility to ensure correct ordering. It’s the best practice to not mixing the use of `activation_reduce` and `activation`.

Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will reduce across all of them.

Mathematically, this API performs the following computation:
    
    
    output = op(data * scale + bias)
    reduce_res = reduce_op(output, axis=<FreeAxis>)
    

Parameters:
    

  * dst – output tile of the activation instruction; layout: same as input `data` tile

  * op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)

  * data – the input tile; layout: (partition axis <= 128, free axis)

  * reduce_op – the reduce operation to perform on the free dimension of the activation result

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `sum(ReductionResult)` is written in-place into the tensor.

  * bias – a vector with the same partition axis size as `data` for broadcast add (after broadcast multiply with `scale`)

  * scale – a scalar or a vector with the same partition axis size as `data` for broadcast multiply


---

### nki.isa.exponential

### New `nki.isa` APIs
  * `nki.isa.exponential` — Dedicated exponential instruction with max subtraction, faster than `nisa.activation(op=nl.exp)` and useful for Softmax calculation. Trn3 (NeuronCore-v4) only.

## Batch Normalization Statistics

### bn_stats

`bn_stats` | Compute mean- and variance-related statistics for each partition of an input tile `data` in parallel using Vector Engine.  

---

### nki.isa.bn_stats

nki.isa.bn_stats(dst, data, name=None)
    

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

Due to hardware limitation, the number of elements per partition (i.e., free dimension size) of the input `data` must not exceed 512 (nl.tile_size.bn_stats_fmax). To calculate per-partition mean/variance of a tensor with more than 512 elements in free dimension, we can invoke `bn_stats` instructions on each 512-element tile and use a single `bn_aggr` instruction to aggregate `bn_stats` outputs from all the tiles.

Vector Engine performs the above statistics calculation in float32 precision. The engine automatically casts the input `data` to float32 before performing computation. The float32 computation results are cast to `dst.dtype` at no additional performance cost.

Parameters:
    

  * dst – an output tile with 6-element statistics per partition

  * data – the input tile (up to 512 elements per partition)


---

### bn_aggr

`bn_aggr` | Aggregate one or multiple `bn_stats` outputs to generate a mean and variance per partition using Vector Engine.  

---

### nki.isa.bn_aggr

nki.isa.bn_aggr(dst, data, name=None)
    

Aggregate one or multiple `bn_stats` outputs to generate a mean and variance per partition using Vector Engine.

The input `data` tile effectively has an array of `(count, mean, variance*count)` tuples per partition produced by bn_stats instructions. Therefore, the number of elements per partition of `data` must be a modulo of three.

Note, if you need to aggregate multiple `bn_stats` instruction outputs, it is recommended to declare a SBUF tensor and then make each `bn_stats` instruction write its output into the SBUF tensor at different offsets.

Vector Engine performs the statistics aggregation in float32 precision. The engine automatically casts the input `data` to float32 before performing computation. The float32 computation results are cast to `dst.dtype` at no additional performance cost.

Parameters:
    

  * dst – an output tile with two elements per partition: a mean followed by a variance

  * data – an input tile with results of one or more bn_stats


## Quantization

### quantize_mx

`quantize_mx` | Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.  

---

### nki.isa.quantize_mx

nki.isa.quantize_mx(dst, src, dst_scale, name=None)
    

Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.

Note

Available only on NeuronCore-v4 and newer.

The resulting MXFP8 tensors, `dst` and `dst_scale` are as defined in the OCP Microscaling standard. This instruction calculates the required scales for each group of 32 values in `src`, divides them by the calculated scale, and casts to the target MXFP8 datatype. The output layout is suitable for direct consumption by the `nisa.nc_matmul_mx` API running on Tensor Engine.

Memory types.

All input `src` and output tiles (`dst` and `dst_scale`) must be in SBUF.

Data types.

The input `src` tile must be float16 or bfloat16. The output `dst` tile must be float8_e5m2_x4 or float8_e4m3fn_x4 (4-packed FP8 data types). The `dst_scale` tile must be uint8.

The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4) are 32-bit data types that pack four 8-bit float8_e5m2/float8_e4m3fn values.

Layout.

The quantization operates on groups of 32 elements from the input `src` tile, where each group consists of 8 partitions × 4 elements per partition. For each 32-element group, the instruction produces:

  * Quantized FP8 data in `dst`

  * One shared scale value in `dst_scale` per group

Tile size.

  * The partition dimension size of `src` must be a multiple of 32 and must not exceed 128.

  * The free dimension size of `src` must be a multiple of 4 and must not exceed the physical size of each SBUF partition.

  * The `dst` tile has the same partition dimension size as `src` but a free dimension size that is 1/4 of `src` free dimension size due to the special 4-packed FP8 data types.

Parameters:
    

  * dst – the quantized MXFP8 output tile

  * src – the input FP16/BF16 tile to be quantized

  * dst_scale – the output scale tile

## ISA Memory Set and Iota

### nisa.memset

### `nisa.memset` — Strict Type Matching
NKI 0.3.0 enforces that the `value` argument must match the destination tensor’s dtype. Beta 2 silently cast float values to the destination type. For integer-typed tensors, pass an integer literal.
    
    
    # Beta 2
    buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=buf, value=2.0)
    
    # NKI 0.3.0
    buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=buf, value=2)
    

### `nisa.tensor_reduce` — Axis Handling Fix
NKI 0.3.0 fixes incorrect axis handling that existed in Beta 2. Beta 2 incorrectly allowed `axis=1` to refer to the last free dimension even for 3D/4D tensors. NKI 0.3.0 corrects this so that axis values correspond to the actual tensor dimensions.

Kernels that relied on the Beta 2 behavior (e.g., using `axis=1` to mean the last dimension of a 3D/4D tensor) will produce errors in NKI 0.3.0.

### `nisa.dma_compute` — Parameter Reorder
The `scales` and `reduce_op` parameters swapped positions. `scales` is now optional, and `unique_indices` was added (moved from `dma_copy`).
    
    
    # Beta 2
    nisa.dma_compute(dst, srcs, scales, reduce_op)
    
    # NKI 0.3.0
    nisa.dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True)
    

### `nisa.sendrecv` — `dma_engine` Enum
The boolean `use_gpsimd_dma` parameter is replaced by the `dma_engine` enum.
    
    
    # Beta 2
    nisa.sendrecv(..., use_gpsimd_dma=True)
    
    # NKI 0.3.0
    from nki.isa import dma_engine
    nisa.sendrecv(..., dma_engine=dma_engine.gpsimd_dma)
    nisa.sendrecv(..., dma_engine=dma_engine.dma)      # was use_gpsimd_dma=False
    

### `nisa.affine_select` — `offset` Parameter Moved
The `offset` parameter moved from the 3rd positional argument to a keyword argument with default `0`. Existing positional call sites will break.
    
    
    # Beta 2
    nisa.affine_select(dst, pattern, offset, channel_multiplier, on_true, on_false)
    
    # NKI 0.3.0
    nisa.affine_select(dst, pattern, channel_multiplier, on_true, on_false, offset=offset)
    

### `nisa.register_move` — `imm` Renamed to `src`
The `imm` parameter has been renamed to `src` and now accepts a `VirtualRegister` instead of a compile-time constant. To move a compile-time constant into a register, first allocate a register with the constant value.
    
    
    # Beta 2
    nisa.register_move(dst, imm=42)
    
    # NKI 0.3.0
    src = nisa.register_alloc(x=42)
    nisa.register_move(dst, src=src)
    

### Collectives — `num_channels` Removed
`num_channels` removed from `collective_permute_implicit_current_processing_rank_id`. The high-level `collective_permute_implicit()` now accepts a `channel_ids` list directly.
    
    
    # Beta 2
    rank_id = ncc.collective_permute_implicit_current_processing_rank_id(
        iteration_id=0, channel_id=ch, num_channels=N, replica_group=rg
    )
    
    # NKI 0.3.0
    rank_id = ncc.collective_permute_implicit_current_processing_rank_id(
        iteration_id=0, channel_id=ch, replica_group=rg
    )
    
    ncc.collective_permute_implicit(
        srcs_by_channel=[[src0], [src1]],
        dsts_by_channel=[[dst0], [dst1]],
        replica_group=rg,
        channel_ids=[0, 1],  # replaces num_channels=2
    )
    

### Output Tensors Must Use `nl.shared_hbm`
All kernel output (return) tensors must be allocated with `buffer=nl.shared_hbm`. Using `nl.hbm` for output tensors will cause compilation failures.
    
    
    # Beta 2
    output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.hbm)
    
    # NKI 0.3.0
    output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.shared_hbm)
    

### Integer Enum Constants No Longer Supported
Raw integer values (e.g., `dge_mode=2`) are no longer accepted for enum parameters. Use the named enum members instead: `nki.isa.engine`, `nki.isa.dge_mode`, `nki.isa.oob_mode`, `nki.isa.reduce_cmd`, and `nki.isa.nc_version`.
    
    
    # Beta 2
    nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=2)
    
    # NKI 0.3.0
    nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=nisa.dge_mode.hwdge)
    

### String Buffer Names No Longer Supported
`nl.ndarray`, `nl.zeros`, and other creation ops no longer accept strings for the `buffer` parameter. Use buffer objects from `nki.language` instead.
    
    
    # Beta 2
    buf = nl.ndarray((128, 512), dtype=nl.float16, buffer='sbuf')
    
    # NKI 0.3.0
    buf = nl.ndarray((128, 512), dtype=nl.float16)  # buffer defaults to sbuf
    buf = nl.ndarray((128, 512), dtype=nl.float16, buffer=nl.sbuf)
    

Table 14 Buffer type mapping# Beta 2 (string) | NKI 0.3.0 (object)  
---|---  
`"sbuf"` | `nl.sbuf`  
`"psum"` | `nl.psum`  
`"hbm"` | `nl.hbm`  
`"private_hbm"` | `nl.private_hbm`  
`"shared_hbm"` | `nl.shared_hbm`  
  
### `nki.isa.dma_engine` Alias Repurposed
The Beta 2 `nki.isa.dma_engine` module-level alias was unused and did not map correctly to a valid engine. In NKI 0.3.0, it has been replaced with the `nki.isa.dma_engine` enum, which provides explicit control over DMA transfer engines (`dma_engine.dma` for shared DMA, `dma_engine.gpsimd_dma` for GPSIMD’s internal DMA engine).

## Language Restrictions
The NKI 0.3.0 compiler has stricter validation. The following patterns require changes for NKI 0.3.0.

### Remove Keyword-Only Argument Separator (`*`)
The NKI 0.3.0 compiler does not support the `*` separator in kernel function signatures. Move all parameters with defaults to the end of the signature.
    
    
    # Beta 2
    @nki.jit
    def my_kernel(X: nl.ndarray, *, flag: bool = True, scale: float = 1.0):
        ...
    
    # NKI 0.3.0
    @nki.jit
    def my_kernel(X: nl.ndarray, flag: bool = True, scale: float = 1.0):
        ...
    

### Replace `is` / `is not` with `==` / `!=`
The NKI 0.3.0 compiler does not support Python’s `is` / `is not` operators. These operators check object identity, which is not meaningful during NKI compilation tracing. Use `==` / `!=` instead.
    
    
    # Beta 2
    if some_flag is True:
        ...
    
    # NKI 0.3.0
    if some_flag == True:
        ...
    

### Replace List Kernel Arguments with Tuples
The NKI 0.3.0 compiler does not support `list` as a kernel argument type. Convert list arguments to tuples at the call site.

Tuples are immutable and hashable, which more accurately reflects the semantics of compiled kernels and enables the compiler to cache compilations based on the kernel’s arguments.
    
    
    # Beta 2
    @nki.jit
    def my_kernel(img, in_perm, stride=[1, 1]):
        ...
    my_kernel(img, in_perm=[0, 3, 1, 2], stride=[1, 1])
    
    # NKI 0.3.0
    @nki.jit
    def my_kernel(img, in_perm, stride=(1, 1)):
        ...
    my_kernel(img, in_perm=(0, 3, 1, 2), stride=(1, 1))
    

## API Improvements
These changes improve correctness or usability but are non-breaking for most kernels.

### `nisa.memset` — x4 Packed Type Restriction
x4 packed types (`float8_e4m3fn_x4`, `float8_e5m2_x4`, `float4_e2m1fn_x4`) now enforce `value=0`. The ISA memset instruction fills the destination with a single u32 value and has no notion of the sub-elements packed inside, so only zero is valid. To initialize x4 packed tensors with non-zero values, use `nisa.dma_copy` to load pre-computed x4 data from an HBM kernel argument.
    
    
    # Zero-fill works directly
    buf = nl.ndarray((128, 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    nisa.memset(dst=buf, value=0)
    
    # Non-zero: pass pre-computed x4 data as a kernel argument from HBM
    # and use nisa.dma_copy to load it into SBUF
    nisa.dma_copy(dst=buf, src=precomputed_x4_hbm_tensor)
    

### nisa.memset

      nisa.memset(batch_idx, value=3*128)

---

### memset

`memset` | Initialize `dst` by filling it with a compile-time constant `value`, using Vector or GpSimd Engine.  

---

### nki.isa.memset

nki.isa.memset(dst, value, engine=engine.unknown, name=None)
    

Initialize `dst` by filling it with a compile-time constant `value`, using Vector or GpSimd Engine. The memset instruction supports all valid NKI dtypes (see Supported Data Types).

Parameters:
    

  * dst – destination tile to initialize.

  * value – the constant value to initialize with

  * engine – specify which engine to use for memset: `nki.isa.engine.vector` or `nki.isa.engine.gpsimd` ; `nki.isa.engine.unknown` by default, lets compiler select the best engine for the given input tile shape

Note

For x4 packed types (`float8_e4m3fn_x4`, `float8_e5m2_x4`, `float4_e2m1fn_x4`), only `value=0` is supported.


---

### nisa.iota

      nisa.iota(dynamic_idx_legal, [[1, 1]], 0, 2)

---

### iota

`iota` | Generate a constant literal pattern into SBUF using GpSimd Engine.  

---

### nki.isa.iota

nki.isa.iota(dst, pattern, offset=0, channel_multiplier=0, name=None)
    

Generate a constant literal pattern into SBUF using GpSimd Engine.

The pattern is defined by an int32 `offset`, a tensor access pattern of up to 4D `pattern` and an int32 `channel_multiplier`. The `pattern` field is a list of lists in the form of `[[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]`. When fewer than 4D `pattern` is provided, NKI compiler automatically pads remaining dimensions with size of 1.

Given a 4D pattern (padded if needed), the instruction generates a stream of values using the following pseudo code:
    
    
    num_partitions = dst.shape[0]
    [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]] = pattern
    
    for channel_id in range(num_partitions):
        for w in range(num_w):
            for z in range(num_z):
                for y in range(num_y):
                    for x in range(num_x):
                        value = offset + (channel_id * channel_multiplier) +
                                (w * step_w) + (z * step_z) + (y * step_y) + (x * step_x)
    
                        dst[channel_id, w, z, y, x] = value
    

The above pseudo code assumes `dst` has the same size in every dimension `x/y/z/w` for simplicity. However, the instruction allows any sizes in the free dimension, as long as the number of elements per partition in `dst` matches the product: `num_w * num_z * num_y * num_x`.

Memory types.

The output `dst` tile must be in SBUF.

Data types.

The generated values are computed in 32-bit integer arithmetic. The GpSimd Engine can cast these integer results to any valid NKI data type (see Supported Data Types for more information) before writing to the output tile. The output data type is determined by the `dst` tile’s data type.

Layout.

The partition dimension determines the number of active channels for parallel pattern generation.

Tile size.

The partition dimension size of `dst` must not exceed 128. The number of elements per partition of `dst` must not exceed the physical size of each SBUF partition. The total number of elements in `pattern` must match the number of elements per partition in the `dst` tile.

Parameters:
    

  * dst – the output tile in SBUF to store the generated pattern

  * pattern – a list of [step, num] to describe up to 4D tensor sizes and strides

  * offset – an int32 offset value to be added to every generated value

  * channel_multiplier – an int32 multiplier to be applied to the channel (parition) ID

### nki.isa.iota

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

## Gather, Scatter, and Shuffle

### local_gather

`local_gather` | Gather SBUF data in `src_buffer` using `index` on GpSimd Engine.  

---

### nki.isa.local_gather

nki.isa.local_gather(dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None)
    

Gather SBUF data in `src_buffer` using `index` on GpSimd Engine.

Each of the eight GpSimd cores in GpSimd Engine connects to 16 contiguous SBUF partitions (e.g., core[0] connected to partition[0:16]) and performs gather from the connected 16 SBUF partitions independently in parallel. The indices used for gather on each core should also come from the same 16 connected SBUF partitions. If you only need to gather elements within a partition, consider using nisa.nc_n_gather instead, which supports gathering more indices.

During execution of the instruction, each GpSimd core reads a 16-partition slice from `index`, flattens all indices into a 1D array `indices_1d` (along the partition dimension first). By default with no `num_valid_indices` specified, each GpSimd core will treat all indices from its corresponding 16-partition `index` slice as valid indices. However, when the number of valid indices per core is not a multiple of 16, users can explicitly specify the valid index count per core in `num_valid_indices`. Note, `num_valid_indices` must not exceed the total element count in each 16-partition `index` slice (i.e., `num_valid_indices <= index.size / (index.shape[0] / 16)`).

Next, each GpSimd core uses the flattened `indices_1d` indices as partition offsets to gather from the connected 16-partition slice of `src_buffer`. Optionally, this API also allows gathering of multiple contiguous elements starting at each index to improve gather throughput, as indicated by `num_elem_per_idx`. Behavior of out-of-bound index access is undefined.

Even though all eight GpSimd cores can gather with completely different indices, a common use case for this API is to make all cores gather with the same set of indices (i.e., partition offsets). In this case, users can generate indices into 16 partitions, replicate them eight times to 128 partitions and then feed them into `local_gather`.

As an example, if `src_buffer` is (128, 512) in shape and `index` is (128, 4) in shape, where the partition dimension size is 128, `local_gather` effectively performs the following operation:

`local_gather` preserves the input data types from `src_buffer` in the gather output. Therefore, no data type casting is allowed in this API. The indices in `index` tile must be uint16 types.

This API has three tile size constraints [subject to future relaxation]:

  1. The partition axis size of `src_buffer` must match that of `index` and must be a multiple of 16. In other words, `src_buffer.shape[0] == index.shape[0] and src_buffer.shape[0] % 16 == 0`.

  2. The number of contiguous elements to gather per index per partition `num_elem_per_idx` must be one of the following values: `[1, 2, 4, 8, 16, 32]`.

  3. The number of indices for gather per core must be less than or equal to 4096.

Parameters:
    

  * dst – an output tile of the gathered data

  * src_buffer – an input tile for gathering.

  * index – an input tile with indices used for gathering.

  * num_elem_per_idx – an optional integer value to read multiple contiguous elements per index per partition; default is 1.

  * num_valid_indices – an optional integer value to specify the number of valid indices per GpSimd core; default is `index.size / (index.shape[0] / 16)`.


---

### nc_n_gather

`nc_n_gather` | Gather elements from `data` according to `indices` using GpSimd Engine.  

---

### nki.isa.nc_n_gather

nki.isa.nc_n_gather(dst, data, indices, name=None)
    

Gather elements from `data` according to `indices` using GpSimd Engine.

This instruction performs a gather operation where elements are selected from the input `data` tile based on flattened indices specified in the `indices` tile. The free dimensions of `data` are treated as if they were flattened into a single dimension for indexing purposes, while the partition dimension defines the parallel compute boundary.

The gather operation works independently within each partition. For each partition, the free dimensions of `data` are conceptually flattened, and elements are gathered according to the corresponding flattened indices from the same partition in `indices`. If you need to gather elements across partitions (within groups of partitions), consider using nisa.local_gather.

The `n` in `nc_n_gather` indicates that this instruction corresponds to `n` groups of instructions in the underlying ISA, where `n = ceil(elems_per_partition / 512)`.

Alternatively, we could gather elements by calling nisa.dma_copy with an indirect access pattern derived from `indices`. However, this is less efficient than `nc_n_gather`, which uses GpSimd Engine to perform local data movement within SBUF, without using DMA engines.

Memory types.

All input and output tiles (`data`, `indices`, and `dst`) must be in SBUF. GpSimd Engine cannot access PSUM (see NeuronCore-v2 Compute Engines for details).

Data types.

The input `data` tile can be any valid NKI data type (see Supported Data Types for more information). The output `dst` tile must have the same data type as `data`. The `indices` tile must be uint32.

Layout.

The partition dimension of `data`, `indices`, and `dst` must be the same. Within each partition, the free dimensions of `data` are flattened for indexing. The free dimensions of `indices` determine the shape of the output `dst`.

Tile size.

The partition dimension size of `data`, `indices`, and `dst` must be the same and must not exceed 128. The number of elements per partition in `dst` must match the number of elements per partition in `indices`. The indices’ values must be within the range `[0, data.size / data.shape[0])`.

Parameters:
    

  * dst – output tile containing the gathered elements

  * data – the input tile to gather elements from

  * indices – the indices tile (uint32) specifying which elements to gather


---

### nc_stream_shuffle

`nc_stream_shuffle` | Apply cross-partition data movement within a quadrant of 32 partitions from source tile `src` to destination tile `dst` using Vector Engine.  

---

### nki.isa.nc_stream_shuffle

nki.isa.nc_stream_shuffle(dst, src, shuffle_mask, name=None)
    

Apply cross-partition data movement within a quadrant of 32 partitions from source tile `src` to destination tile `dst` using Vector Engine.

Both source and destination tiles can be in either SBUF or PSUM, and passed in by reference as arguments. In-place shuffle is allowed, i.e., `dst` same as `src`. `shuffle_mask` is a 32-element list. Each mask element must be in data type int or affine expression. `shuffle_mask[i]` indicates which input partition the output partition [i] copies from within each 32-partition quadrant. The special value `shuffle_mask[i]=255` means the output tensor in partition [i] will be unmodified. `nc_stream_shuffle` can be applied to multiple of quadrants. In the case with more than one quadrant, the shuffle is applied to each quadrant independently, and the same `shuffle_mask` is used for each quadrant. For more information about the cross-partition data movement, see Cross-partition Data Movement.

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

## Search and Replace

### max8

`max8` | Find the 8 largest values in each partition of the source tile.  

---

### nki.isa.max8

nki.isa.max8(dst, src, name=None)
    

Find the 8 largest values in each partition of the source tile.

This instruction reads the input elements, converts them to fp32 internally, and outputs the 8 largest values in descending order for each partition. Outputs are converted to `dst.dtype` automatically.

The source tile can be up to 5-dimensional, while the output tile is always 2-dimensional. The number of elements read per partition must be between 8 and 16,384 inclusive. The output will always contain exactly 8 elements per partition. The source and output must have the same partition dimension size:

  * source: [par_dim, …]

  * output: [par_dim, 8]

Parameters:
    

  * dst – a 2D tile containing the 8 largest values per partition in descending order with shape [par_dim, 8]

  * src – the source tile to find maximum values from


---

### nc_find_index8

`nc_find_index8` | Find indices of the 8 given vals in each partition of the data tensor.  

---

### nki.isa.nc_find_index8

# nki.isa.nc_find_index8
nki.isa.nc_find_index8(dst, data, vals, name=None)
    

Find indices of the 8 given vals in each partition of the data tensor.

This instruction first loads the 8 values, then loads the data tensor and outputs the indices (starting at 0) of the first occurrence of each value in the data tensor, for each partition.

The data tensor can be up to 5-dimensional, while the vals tensor must be up to 3-dimensional. The data tensor must have between 8 and 16,384 elements per partition. The vals tensor must have exactly 8 elements per partition. The output will contain exactly 8 elements per partition and will be uint16 or uint32 type. Default output type is uint32.

Behavior is undefined if vals tensor contains values that are not in the data tensor.

If provided, a mask is applied only to the data tensor.

Parameters:
    

  * dst – a 2D tile containing indices (uint16 or uint32) of the 8 values in each partition with shape [par_dim, 8]

  * data – the data tensor to find indices from

  * vals – tensor containing the 8 values per partition whose indices will be found


---

### nc_match_replace8

`nc_match_replace8` | Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced tensor.  

---

### nki.isa.nc_match_replace8

nki.isa.nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None)
    

Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced tensor. If `dst_idx` tile is provided, the indices of the matched values are written to `dst_idx`.

Parameters:
    

  * dst – output tile with replaced values

  * data – the data tensor to search and replace in

  * vals – tensor containing the 8 values per partition to match

  * imm – the immediate float value to replace matched values with

  * dst_idx – optional tile to store indices of matched values

## Sequence Bounds

### sequence_bounds

`sequence_bounds` | Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.  

---

### nki.isa.sequence_bounds

nki.isa.sequence_bounds(dst, segment_ids, name=None)
    

Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.

Given a tile of segment IDs, this function identifies where each segment begins and ends. For each element, it returns a pair of values: [start_index, end_index] indicating the boundaries of the segment that element belongs to. All segment IDs must be non-negative integers. Padding elements (with segment ID of zero) receive special boundary values: a start index of n and an end index of (-1), where n is the length of `segment_ids`.

The output tile contains two values per input element: the start index (first column) and end index (second column) of each segment. The partition dimension must always be 1. For example, with input shape (1, 512), the output shape becomes (1, 2, 512), where the additional dimension holds the start and end indices for each element.

Both the input tile (`segment_ids`) and output tile (`dst`) must have data type `nl.float32` or `nl.int32`.

NumPy equivalent:

Parameters:
    

  * dst – tile containing the sequence bounds.

  * segment_ids – tile containing the segment IDs. Elements with ID=0 are treated as padding.


## Random Number Generation (ISA)

### rng

`rng` | Generate pseudo random numbers using the Vector or GpSimd Engine.  

---

### nki.isa.rng

nki.isa.rng(dst, engine=engine.unknown, name=None)
    

Generate pseudo random numbers using the Vector or GpSimd Engine.

This instruction generates 32 random bits per element and writes them to the destination tensor. Depending on the size of the dtype, the instruction truncates each 32-bit random value to the specified data type, taking the least significant bits.

Example use case: To generate random FP32 numbers between 0.0 and 1.0, follow the Rng instruction with a normalization instruction (e.g., write 16 random bits as UINT16, then divide by (2^16-1) to get a random FP32 number between 0.0 and 1.0).

Memory types.

The output `dst` tile can be in SBUF or PSUM.

Data types.

The output `dst` tile must be an integer type: int8, int16, int32, uint8, uint16, or uint32.

Tile size.

The partition dimension size of `dst` must not exceed 128. The number of elements per partition of `dst` must not exceed the physical size of each SBUF/PSUM partition.

Constraints.

  * Supported arch versions: NeuronCore-v2+.

  * Supported engines: NeuronCore-v2: Vector. NeuronCore-v3+: GpSimd, Vector.

  * Since GpSimd Engine cannot access PSUM, `dst` must be in SBUF when using GpSimd Engine.

Parameters:
    

  * dst – the destination tensor to write random values to

  * engine – specify which engine to use: `nki.isa.engine.vector`, `nki.isa.engine.gpsimd`, or `nki.isa.engine.unknown` (default, the best engine will be selected)


---

### rand2

`rand2` | Generate pseudo random numbers with uniform distribution using Vector Engine.  

---

### nki.isa.rand2

nki.isa.rand2(dst, min, max, name=None)
    

Generate pseudo random numbers with uniform distribution using Vector Engine.

Note

Available only on NeuronCore-v4 and newer.

This instruction generates pseudo random numbers and stores them into SBUF/PSUM. The generated values follow a uniform distribution within the specified [min, max] range.

Key features:

  * Uses XORWOW PRNG algorithm for high-quality random number generation

  * Generates FP32 random values with uniform distribution

  * Supports output conversion to various data types

Memory types.

The output `dst` tile can be in SBUF or PSUM.

Data types.

The output `dst` tile can be any of: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32, int8, int16, int32, uint8, uint16, or uint32.

Tile size.

The partition dimension size of `dst` must not exceed 128. The number of elements per partition of `dst` must not exceed the physical size of each SBUF/PSUM partition.

Constraints.

  * Supported arch versions: NeuronCore-v4+.

  * Supported engines: Vector.

  * min < max for valid range.

Parameters:
    

  * dst – the destination tensor to write random values to

  * min – minimum value for uniform distribution range (FP32), can be a scalar or vector value

  * max – maximum value for uniform distribution range (FP32), can be a scalar or vector value


---

### rand_set_state

`rand_set_state` | Seed the pseudo random number generator (PRNG) inside the engine.  

---

### nki.isa.rand_set_state

# nki.isa.rand_set_state
nki.isa.rand_set_state(src_seeds, engine=engine.unknown, name=None)
    

Seed the pseudo random number generator (PRNG) inside the engine.

This instruction initializes the PRNG state for future random number generation operations. Each partition in the source tensor seeds the PRNG states for the corresponding compute lane inside the engine.

The PRNG state is cached inside the engine as a persistent state during the rest of NEFF execution. However, the state cannot survive TPB resets or Runtime reload.

Memory types.

The input `src_seeds` tile must be in SBUF.

Data types.

The input `src_seeds` tile must be uint32.

Tile size.

  * src_seeds element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

Constraints.

  * Supported arch versions: NeuronCore-v3+.

  * Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.

  * `src_seeds` must be in SBUF.

Parameters:
    

  * src_seeds – the source tensor containing seed values for the PRNG; must be a 2D uint32 tensor with the partition dimension representing the compute lanes and the free dimension containing the seed values

  * engine – specify which engine to use: `nki.isa.engine.vector`, `nki.isa.engine.gpsimd`, or `nki.isa.engine.unknown` (default, the best engine will be selected)


---

### rand_get_state

`rand_get_state` | Store the current pseudo random number generator (PRNG) states from the engine.  

---

### nki.isa.rand_get_state

# nki.isa.rand_get_state
nki.isa.rand_get_state(dst, engine=engine.unknown, name=None)
    

Store the current pseudo random number generator (PRNG) states from the engine.

This instruction stores the current PRNG states cached inside the engine to SBUF/PSUM. Each partition in the output tensor holds the PRNG states for the corresponding compute lane inside the engine.

Memory types.

The output `dst` tile must be in SBUF (NeuronCore-v3) or SBUF/PSUM (NeuronCore-v4+).

Data types.

The output `dst` tile must be uint32.

Tile size.

  * dst element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

Constraints.

  * Supported arch versions: NeuronCore-v3+.

  * Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.

  * Since GpSimd Engine cannot access PSUM, `dst` must be in SBUF when using GpSimd Engine.

Parameters:
    

  * dst – the destination tensor to store PRNG state values; must be a 2D uint32 tensor

  * engine – specify which engine to use: `nki.isa.engine.vector`, `nki.isa.engine.gpsimd`, or `nki.isa.engine.unknown` (default, the best engine will be selected)


---

### set_rng_seed

`set_rng_seed` | Seed the pseudo random number generator (PRNG) inside the Vector Engine.  

---

### nki.isa.set_rng_seed

# nki.isa.set_rng_seed
nki.isa.set_rng_seed(src_seeds, name=None)
    

Seed the pseudo random number generator (PRNG) inside the Vector Engine.

The PRNG state is cached inside the engine as a persistent state during the rest of NEFF execution. However, the state cannot survive TPB resets or Runtime reload.

Using the same seed will generate the same sequence of random numbers when used together with the `nisa.rng()` on the Vector Engine.

Memory types.

The input `src_seeds` must be in SBUF or PSUM.

Data types.

The input `src_seeds` must be a 32-bit value.

Tile size.

The input `src_seeds` must be a [1,1] tensor.

Parameters:
    

src_seeds – a [1,1] tensor on SBUF or PSUM with a 32-bit value to be used as the seed


## Register Operations

### register_alloc

`register_alloc` | Allocate a virtual register and optionally initialize it with a value.  

### register_alloc

    def register_alloc(x: Optional[int]) -> register: ...

---

### nki.isa.register_alloc

nki.isa.register_alloc(x=None)
    

Allocate a virtual register and optionally initialize it with a value.

Each engine sequencer (Tensor/Scalar/Vector/GpSimd/Sync Engine) within a NeuronCore maintains its own set of physical registers for scalar operations (64x 32-bit registers per engine sequencer in NeuronCore v2-v4). This API conceptually allocates a register within a virtual register space. Users do not need to explicitly free a register through nisa APIs. The NKI compiler handles physical register allocation (and deallocation) across the appropriate engine sequencers based on the dynamic program flow.

NKI provides the following APIs to manipulate allocated registers:

  * `nisa.register_move`: Move a constant integer or another register’s value into a register

  * `nisa.register_load`: Load a scalar (32-bit) value from HBM/SBUF into a register

  * `nisa.register_store`: Store register contents to HBM/SBUF

In the current NKI release, these registers are primarily used to specify dynamic loop boundaries and while loop conditions. The NKI compiler compiles such dynamic looping constructs to branching instructions executed by engine sequencers. For additional details, see `nl.dynamic_range`. For more information on engine sequencer and its capabilities, see Trainium/Inferentia2 architecture guide.

Parameters:
    

x – 

optional initialization value. Can be one of:

  * `None` (default): allocate an uninitialized register

  * `int`: allocate a register initialized with this immediate integer value

Example:

Three ways to allocate a register initialized to zero:
    
    
    # Approach 1: Using an immediate value
    reg1 = nisa.register_alloc(0)
    
    # Approach 2: Two-step with register_load
    zero_tensor = nl.zeros([1, 1], dtype=nl.int32, buffer=nl.sbuf)
    reg2 = nisa.register_alloc(None)
    nisa.register_load(reg2, zero_tensor)
    

---

### register_move

`register_move` | Move a value into a virtual register.  

### register_move

    def register_move(dst: imm: int): ...

---

### nki.isa.register_move

# nki.isa.register_move
nki.isa.register_move(dst, src)
    

Move a value into a virtual register.

This instruction loads a value into the specified virtual register. The source can be either a compile-time constant integer or another virtual register.

The virtual register system allows the NKI compiler to allocate physical registers across different engine sequencers as needed. See `nisa.register_alloc` for more details on virtual register allocation.

This instruction operates on virtual registers only and does not access SBUF, PSUM, or HBM.

Parameters:
    

  * dst – the destination virtual register (allocated via `nisa.register_alloc`)

  * src – source value - either a compile-time constant integer or a VirtualRegister

Example:
    
    
    # Allocate a register and initialize it with a constant
    loop_count = nisa.register_alloc()
    nisa.register_move(loop_count, 10)  # Set register to 10
    
    # Copy from another register
    reg2 = nisa.register_alloc()
    nisa.register_move(reg2, loop_count)  # Copy value from loop_count
    

---

### register_load

`register_load` | Load a scalar value from memory (HBM or SBUF) into a virtual register.  

### register_load

    def register_load(dst: register, src: tensor): ...

---

### nki.isa.register_load

# nki.isa.register_load
nki.isa.register_load(dst, src)
    

Load a scalar value from memory (HBM or SBUF) into a virtual register.

This instruction reads a single scalar value (up to 32-bit) from a memory location (HBM or SBUF) and stores it in the specified virtual register. The source must be a NKI tensor with exactly one element (shape [1] or [1, 1]). This enables dynamic loading of values computed at runtime into registers for use in control flow operations.

The virtual register system allows the NKI compiler to allocate physical registers across different engine sequencers as needed. See `nisa.register_alloc` for more details on virtual register allocation.

Parameters:
    

  * dst – the destination virtual register (allocated via `nisa.register_alloc`)

  * src – the source tensor containing a single scalar value to load

Example:
    
    
    # Load a computed value into a register
    computed_bound = nl.ones([1], dtype=nl.int32, buffer=nl.sbuf)  # bound of 1 in SBUF
    loop_reg = nisa.register_alloc()
    nisa.register_load(loop_reg, computed_bound)
    

---

### register_store

`register_store` | Store the value from a virtual register into memory (HBM/SBUF).  

### register_store

    def register_store(dst: tensor, src: register): ...

---

### nki.isa.register_store

# nki.isa.register_store
nki.isa.register_store(dst, src)
    

Store the value from a virtual register into memory (HBM/SBUF).

This instruction writes the scalar value (up to 32-bit) stored in a virtual register to a memory location (HBM or SBUF). The destination must be a tensor with exactly one element (shape [1] or [1, 1]). This enables saving register values back to memory for later use or for output purposes.

The virtual register system allows the NKI compiler to allocate physical registers across different engine sequencers as needed. See `nisa.register_alloc` for more details on virtual register allocation.

Parameters:
    

  * dst – the destination tensor with a single element to store the register value

  * src – the source virtual register (allocated via `nisa.register_alloc`)

Example:
    
    
    # Store a register value back to memory
    counter_reg = nisa.register_alloc(0)
    # ... perform operations that modify counter_reg ...
    result_tensor = nl.ndarray([1], dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(result_tensor, counter_reg)
    

---

### nisa.register_store

      nisa.register_store(cond, reg)

---

### nisa.register_load

      nisa.register_load(reg, cond)

## Control Flow and Loop Iterators

### affine_range

`affine_range` | Create a sequence for fully unrolled loop iteration.  

---

### sequential_range

`sequential_range` | Create a sequence for fully unrolled loop iteration.  

---

### static_range

`static_range` | Create a sequence for fully unrolled loop iteration.  

---

### nki.language.affine_range

nki.language.affine_range(start, stop=None, step=1)
    

Create a sequence for fully unrolled loop iteration.

Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop. Prefer static_range instead.

Warning

This API is deprecated and will be removed in future releases.

Parameters:
    

  * start – start value (or stop if `stop` is None).

  * stop – stop value (exclusive).

  * step – step size.

Returns:
    

an iterator yielding integer values from start to stop.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.affine_range
    for i in nl.affine_range(input_tensor.shape[1] // 512):
        offset = i * 512
        tile = nl.load(input_tensor[0:128, offset:offset+512])
        result = nl.multiply(tile, tile)
        nl.store(out_tensor[0:128, offset:offset+512], result)
    

---

### nki.language.sequential_range

nki.language.sequential_range(start, stop=None, step=1)
    

Create a sequence for fully unrolled loop iteration.

Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop. Prefer static_range instead.

Warning

This API is deprecated and will be removed in future releases.

Parameters:
    

  * start – start value (or stop if `stop` is None).

  * stop – stop value (exclusive).

  * step – step size.

Returns:
    

an iterator yielding integer values from start to stop.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.sequential_range
    for i in nl.sequential_range(input_tensor.shape[1] // 512):
        offset = i * 512
        tile = nl.load(input_tensor[0:128, offset:offset+512])
        result = nl.multiply(tile, tile)
        nl.store(out_tensor[0:128, offset:offset+512], result)
    

### nki.language.sequential_range

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

  * `nki.language.ndarray` — `buffer` default changed from `None` to `nl.sbuf`

  * `nki.language.zeros` — `buffer` default changed from `None` to `nl.sbuf`

  * `nki.language.sequential_range` — `stop` and `step` now have default values (`None` and `1`)

---

### nki.language.static_range

nki.language.static_range(start, stop=None, step=1)
    

Create a sequence for fully unrolled loop iteration.

Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop. Prefer this method over affine_range and sequential_range

Parameters:
    

  * start – start value (or stop if `stop` is None).

  * stop – stop value (exclusive).

  * step – step size.

Returns:
    

an iterator yielding integer values from start to stop.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.static_range
    for i in nl.static_range(input_tensor.shape[1] // 512):
        offset = i * 512
        tile = nl.load(input_tensor[0:128, offset:offset+512])
        result = nl.multiply(tile, tile)
        nl.store(out_tensor[0:128, offset:offset+512], result)
    

---

### nki.language.dynamic_range

nki.language.dynamic_range(start, stop=None, step=1)
    

Create a sequence for dynamic loop iteration.

Create a sequence of numbers for use as dynamic loop iterators in NKI. The loop runs on device with dynamic bounds.

Parameters:
    

  * start – start value (or stop if `stop` is None), can be VirtualRegister.

  * stop – stop value (exclusive), can be VirtualRegister.

  * step – step size, must be a compile-time positive integer (not VirtualRegister).

Returns:
    

an iterator yielding integer values from start to stop.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.dynamic_range
    for _ in nl.dynamic_range(1):
        tile = nl.load(input_tensor[0:128, 0:512])
        result = nl.multiply(tile, tile)
        nl.store(out_tensor[0:128, 0:512], result)
    

### nki.language.dynamic_range

nki.language.dynamic_range(start, stop=None, step=1)
    

Create a sequence for dynamic loop iteration with runtime bounds.

Parameters:
    

  * start – Start value (inclusive), or stop if `stop` is `None`. Can be a `VirtualRegister`.

  * stop – Stop value (exclusive). Can be a `VirtualRegister`.

  * step – Step size. Must be a compile-time positive `int` (not a `VirtualRegister`).

Returns:
    

An iterator yielding integer values from start to stop.

---

### dynamic_range

The most basic dynamic loop is a `for` loop that uses a register value for the iteration value and another register for the upper bound. Developers can write this kind of loop using `dynamic_range`:
    
    
    # dynamic loop with dynamically computed upper bounds
    # upper_bound is a hardware register
    # the loop index, i, is also a hardware register
    upper_bound = register_alloc()
    register_load(upper_bound, tensor)
    for i in dynamic_range(5, upper_bound, 2):
      ...
    

### dynamic_range

`dynamic_range` | Create a sequence for dynamic loop iteration.  

---

### dynamic_range Parameter Constraints

## Parameter Constraints
`start` / `stop`
    

Can be Python `int` literals or `VirtualRegister` objects (runtime values computed on device). When only one positional argument is given it is treated as `stop` and `start` defaults to `0`, matching the Python `range()` convention.

`step`
    

Must be a compile-time positive integer. Passing a `VirtualRegister` raises an `AssertionError`: The step must be known at compile time because the hardware loop instruction encodes the step as an immediate operand.

---

### Range Iterators Comparison Table

## Comparison with Other Range Iterators
NKI provides four range iterators. The table below summarises their key differences:

Iterator | Bounds | Unrolled? | Generated Code | Primary Use Case  
---|---|---|---|---  
`static_range` | Compile-time `int` | Yes (at compile time) | Fully unrolled—no loop instruction | Default choice—supersedes `sequential_range` and `affine_range`.  
`sequential_range` | Compile-time `int` | Yes (at compile time) | Fully unrolled—no loop instruction | Deprecated, formerly for iterations with loop-carried dependencies. Prefer `static_range` instead.  
`affine_range` | Compile-time `int` | Yes (at compile time) | Fully unrolled—no loop instruction | Deprecated, formerly for parallel iterations with no loop-carried dependency. Prefer `static_range` instead.  
`dynamic_range` | Runtime `VirtualRegister` or `int` | No | Hardware loop instruction | Trip count unknown at compile time  
  
There are three key distinctions worth calling out:

  * `static_range`, `affine_range`, and `sequential_range` require all bounds to be compile-time integers. The compiler keeps them as loops internally but may unroll them in the backend. `dynamic_range` bounds can be runtime values and the loop is never unrolled.

  * `static_range`, `affine_range`, and `sequential_range` fully unrolls at compile time, which can dramatically increase compilation time, `dynamic_range` avoids this entirely.

---

### dynamic_range Hardware Lowering

## Hardware Lowering
The compiler lowers `dynamic_range` loops to hardware loop instructions on the NeuronCore. Because the loop exists as a single hardware instruction with a body:

  * The compiled artifact size does not grow with the trip count.

  * The loop variable is a device register, not a Python `int`. You cannot use it in host-side Python expressions (e.g., `if i == 0:`). Use NKI device-side operations for any conditional logic that depends on the loop variable.

---

### dynamic_range Register Allocation Implications

## Register Allocation Implications
Inside a `dynamic_range` loop the compiler must keep all live tensors in on-chip memory (SBUF/PSUM) for the entire duration of the loop, because the hardware re-executes the same body on each iteration. This means:

  * Tensors allocated inside the loop body are allocated once and reused across iterations.

  * Keeping the loop body small and limiting the number of live tiles reduces memory pressure.

In contrast, `static_range` unrolls each iteration independently, giving the compiler full freedom to schedule instructions across the flattened instruction stream. However, this does not solve the issue when the trip count is unknown at compile time—which is precisely when `dynamic_range` is needed.

---

### dynamic_range with no_reorder Interaction

## Interaction with `no_reorder`
`dynamic_range` loops inside a `nl.no_reorder()` block are not currently supported.
    
    
    # ✗ This is NOT supported and will error
    with nl.no_reorder():
        for i in nl.dynamic_range(n):
            ...
    

`affine_range`, `sequential_range`, and `static_range` are all permitted inside `no_reorder` blocks.

To work around this, place the `no_reorder` block inside the loop body:
    
    
    # ✓ no_reorder inside the dynamic loop body
    for i in nl.dynamic_range(n):
        with nl.no_reorder():
            ...
    

---

### VirtualRegister while loop alternative

## Using `while` with a `VirtualRegister`
As an alternative to `dynamic_range`, you can use a standard `while` loop with a `VirtualRegister` as the condition. The loop terminates when the register holds the value `0`.
    
    
    import nki.language as nl
    import nki.isa as nisa
    
    reg = nisa.register_alloc(1)
    while reg:
        # perform work ...
    
        # update condition from an SBUF tensor
        nisa.register_load(reg, cond_tensor)
    

---

### dynamic_range Usage Guidelines

## When to Use `dynamic_range`
Use `dynamic_range` when:

  * The number of iterations is not known at compile time—for example, it depends on a value loaded from a tensor or computed on device.

  * The trip count is large and unrolling (`static_range`, `affine_range`, or `sequential_range`) would cause excessive compilation time or code size.

Prefer other iterators when:

  * Bounds are compile-time constants and iterations are independent, contain loop-carried dependencies, or need full unrolling → `static_range`, `affine_range`, or `sequential_range`.

---

### dynamic_range Basic Usage Example

### Basic usage with a constant bound
    import nki.language as nl
    import nki.isa as nisa
    
    for _ in nl.dynamic_range(1):
        tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(src=input_tensor[0:128, 0:512], dst=tile)
        nisa.tensor_tensor(dst=result, data1=tile, data2=tile, op=nl.multiply)
        nisa.dma_copy(src=result, dst=out_tensor[0:128, 0:512])
    

Even with a constant bound, this generates a hardware loop instruction rather than unrolling.

---

### dynamic_range Runtime Trip Count Example

### Runtime trip count from a `VirtualRegister`
    import nki.language as nl
    import nki.isa as nisa
    
    start = nisa.register_alloc(0)
    stop = nisa.register_alloc(512)
    for i in nl.dynamic_range(start, stop, 128):
        tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(src=input_tensor.ap([[512, 128], [1, 512]], scalar_offset=i), dst=tile)
        nisa.tensor_scalar(dst=result, data=tile, op0=nl.add, operand0=2.0)
        nisa.dma_copy(src=result, dst=out_tensor.ap([[512, 128], [1, 512]], scalar_offset=i))
    

---

### dynamic_range Start Stop Step Example

### Specifying start, stop, and step
    import nki.language as nl
    import nki.isa as nisa
    
    # Loop from `begin` to `end` with step 2
    # begin and end are VirtualRegisters; step must be a compile-time int
    begin = nisa.register_alloc(0)
    end = nisa.register_alloc(4)
    for i in nl.dynamic_range(begin, end, 2):
        ...
    

## Scheduling and Synchronization

### no_reorder

        with nl.no_reorder():
            for i in range(len(in_tiles)):
                tile = in_tiles[i]
                out_tile = nl.ndarray(tile.shape, tile.dtype, buffer=nl.sbuf)
                nisa.activation(dst=out_tile, data=tile, op=nl.exp, name=f"act{i}")
                out_tiles.append(out_tile)
    
        outs = []
        for tile in out_tiles:
            out = nl.ndarray(tile.shape, tile.dtype, buffer=nl.hbm)
            nisa.dma_copy(dst=out, src=tile)
            outs.append(out)
    
        return tuple(outs)
    

The `no_reorder` block instructs the compiler to insert dependency edges between every instruction. Note, the `no_reorder` block is “dynamically scoped”, meaning it applies to all of the code that would execute under the block, not just the code that is syntactically under the block. For example, the following code is equivalent to the above.

### no_reorder

`no_reorder` | Prevent the scheduler from reordering operations in this region.  

---

### nki.language.no_reorder

# nki.language.no_reorder
nki.language.no_reorder()
    

Prevent the scheduler from reordering operations in this region.

Use as a context manager (`with nl.no_reorder():`) to guarantee that operations inside the block execute in program order. Without this directive, the compiler scheduler is free to reorder independent operations for better hardware utilization.

Dynamic loops (`nl.dynamic_range`) are not supported inside a `no_reorder` block. Static loops (`nl.affine_range`, `nl.sequential_range`, `nl.static_range`) are allowed because they are fully unrolled at compile time.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.no_reorder -- guarantee execution order
    with nl.no_reorder():
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.add(a, b)
    expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### with_schedule()

    scheduled = kernel.with_schedule([
        ("plus1", "recip")
    ])
    
    # The second component of each pair can be a single name or a list of names
    # This is equivalent to above. Using a list is convenient
    # for declaring multiple dependency edges.
    scheduled = kernel.with_schedule([
        ("plus1", ["recip"])
    ])

---

### core_barrier

`core_barrier` | Synchronize execution across multiple NeuronCores by implementing a barrier mechanism.  

---

### nki.isa.core_barrier

nki.isa.core_barrier(data, cores, engine=engine.gpsimd, name=None)
    

Synchronize execution across multiple NeuronCores by implementing a barrier mechanism.

Note

Available only on NeuronCore-v3 or newer.

This instruction creates a synchronization point where all specified NeuronCores must reach before any can proceed. The barrier is implemented using a semaphore-based protocol where each NeuronCore writes a semaphore to each other core (remote semaphore update) and then waits for the other cores’ semaphores before continuing execution (local semaphore wait).

The use case is when two NeuronCores both need to write to disjoint portions of a shared HBM tensor (`data`) and they both need to consume the tensor after both cores have finished writing into the tensor. In this case, both cores can perform the write to `data` in HBM using `nisa.dma_copy`, and then signal to each other when the write operation is complete using `nisa.core_barrier`.

This instruction is only allowed in NeuronCore-v3 or newer when LNC (Logical NeuronCore) is enabled. Currently only `cores=(0, 1)` is supported. This allows synchronization between exactly two NeuronCores that share the same HBM stack.

The `data` parameter represents the shared data that all cores need to synchronize on. This must be data in shared HBM that multiple cores are accessing.

The `engine` parameter allows specifying which engine inside the NeuronCores should execute the barrier instruction (that is, the remote semaphore update and local semaphore wait). The barrier will block execution on this engine, other engines will not be blocked.

Parameters:
    

  * data – the shared data that all cores need to synchronize on; must be data in shared HBM

  * cores – a tuple of core indices to synchronize; only `(0, 1)` is supported when LNC2 is enabled

  * engine – the engine to execute the barrier instruction on; defaults to GpSimd Engine

Example:
    
    
    # Synchronize between two cores after each core writes to half of shared tensor
    shared_tensor = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Each core writes to half of the tensor
    if core_id == 0:
        # Core 0 writes to first half
        core0_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=shared_tensor[:batch_size // 2, :], src=core0_data)
    else:
        # Core 1 writes to second half
        core1_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=shared_tensor[batch_size // 2:, :], src=core1_data)
    
    core_barrier(data=shared_tensor, cores=(0, 1))
    
    # Now both cores can safely read the complete tensor
    

### nki.isa.core_barrier

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

## Inter-Core Communication

### sendrecv

`sendrecv` | Perform point-to-point communication between NeuronCores by sending and receiving data simultaneously using DMA engines.  

---

### nki.isa.sendrecv

nki.isa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, dma_engine=dma_engine.dma, name=None)
    

Perform point-to-point communication between NeuronCores by sending and receiving data simultaneously using DMA engines.

Note

Available only on NeuronCore-v3 or newer.

This instruction enables bidirectional data exchange between two NeuronCores within a Logical NeuronCore (LNC) configuration. The current NeuronCore sends its `src` tile to the `dst` location of the target NeuronCore specified by `send_to_rank`, while simultaneously receiving data from `recv_from_rank` into its own `dst` tile.

The use case is when NeuronCores need to exchange data for distributed computation patterns, such as all-gather communication or other collective operations where cores need to coordinate their computations by exchanging tiles.

This instruction is only allowed in NeuronCore-v3 or newer when LNC (Logical NeuronCore) is enabled. The communication occurs between NeuronCores that share the same HBM stack within the LNC configuration. Therefore, `send_to_rank` and `recv_from_rank` must be either 0 or 1.

The `pipe_id` parameter provides synchronization control by grouping sendrecv operations. Operations with the same `pipe_id` form a logical group where all operations in the group must complete before any can proceed. Operations with different `pipe_id` values can progress independently without blocking each other.

The `dma_engine` parameter specifies which DMA transfer mechanism to use:

  * `nisa.dma_engine.dma` (default): Uses the standard DMA engine with CoreBarrier synchronization. Can be triggered from any engine.

  * `nisa.dma_engine.gpsimd_dma`: Uses the GPSIMD’s internal DMA engine for low-latency SB-to-SB swaps in LNC=2. Implies GPSIMD as the trigger engine. This mode has restrictions: the partition dimension size of `src`/`dst` must be a multiple of 16, and the data size per partition must not exceed 1024 bytes for 32-bit types, 512 bytes for 16-bit types, or 256 bytes for 8-bit types.

Memory types.

Both `src` and `dst` tiles must be in SBUF.

Data types.

`src` and `dst` must have the same data type, but they can be any supported data types in NKI.

Layout.

`src` and `dst` must have the same shape and layout.

Tile size.

`src` and `dst` must have the same partition dimension size and the same number of elements per partition.

Parameters:
    

  * src – the source tile on the current NeuronCore to be sent to the target NeuronCore

  * dst – the destination tile on the current NeuronCore where received data will be stored

  * send_to_rank – rank ID of the target NeuronCore to send data to

  * recv_from_rank – rank ID of the source NeuronCore to receive data from

  * pipe_id – synchronization identifier that groups sendrecv operations; operations with the same pipe_id are synchronized

  * dma_engine – the DMA transfer mode; defaults to `nisa.dma_engine.dma`

Example:
    
    
    # Exchange data between two cores in a ring pattern
    num_cores = 2
    current_rank = nl.program_id()
    next_rank = (current_rank + 1) % num_cores
    prev_rank = (current_rank - 1) % num_cores
    
    # Data to send and buffer to receive
    send_data = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    recv_buffer = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform bidirectional exchange
    sendrecv(
        src=send_data,
        dst=recv_buffer,
        send_to_rank=next_rank,
        recv_from_rank=prev_rank,
        pipe_id=0
    )
    
    # Now recv_buffer contains data from the previous core
    

## SPMD and Program Identity

### program_id

The program_id API will return the logical core id that the current instance is running on. In the case of LNC=2, this API will return either 0 or 1. When not using LNC, this API will return 0. This API can be used to programmatically divide work between multiple cores.

For example, suppose we have a tensor with shape 2x128x128 and we want to compute the reciprocal of all of the elements of this tensor. We can write a kernel function that is LNC-aware and can make use of extra cores when available.
    
    
    def lnc_test(input):
     # Check the first dimension is 2 for this example
     assert input.shape[0] == 2
    
     # create temporary storage on SBUF for comptation
     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    
     # create output tensor
     output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)
    
     if nl.num_programs() == 1:
       # Not using multiple cores, process two tiles
       for i in range(2):
         nisa.dma_copy(in_tile, input[i])
         nisa.reciprocal(out_tile, in_tile)
         nisa.dma_copy(output[i], out_tile)
     else:
       # Using multiple cores, process tiles in parallel, one per core
       i = nl.program_id(0)
       nisa.dma_copy(in_tile, input[i])
       nisa.reciprocal(out_tile, in_tile)
       nisa.dma_copy(output[i], out_tile)
     return output
    

The code above has two cases, one for when we are not using LNC (num_programs returns 1), and one for when we are using LNC=2 (num_programs returns 2). In the non-LNC case, there is a for loop that processes each input tiles one after the other. However, in the LNC=2 case, we can use the program_id API to query which core we are on. This API will return either 0 or 1. The code uses the program_id to have each core process one of the two tiles, in parallel.

### program_id

`program_id` | Index of the current SPMD program along the given axis in the launch grid.  

---

### num_programs

When writing a NKI kernel for multiple cores, there are two important APIs that can be used to tell how many cores are being used and which core the current instance is running on. These APIs are called num_programs and program_id.

The num_programs API will return the total number of cores the current kernel is running on. If LNC is not being used, this API will return 1. So, we can tell if we are running on multiple cores by inspecting the result of this variable:
    
    
    @nki.jit
    def lnc_test(input):
      if nl.num_programs() > 1:
        print("Running on multiple cores")
      else:
        print("Running on one core - no LNC")
    
    # Launch lnc_test on 1 core
    # prints "Running on one core - no LNC"
    lnc_test(input)
    
    # Launch lnc_test on 2 cores
    # prints "Running on multiple cores"
    lnc_test[2](input)
    

### num_programs

`num_programs` | Number of SPMD programs along the given axes in the launch grid.  

---

### program_ndim

`program_ndim` | Number of dimensions in the SPMD launch grid.  

---

### nki.language.program_id

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

### nki.language.program_id

nki.language.program_id(axis=0)
    

Index of the current SPMD program along the given axis in the launch grid.

Parameters:
    

axis – the axis of the launch grid.

Returns:
    

the program id along `axis`.

---

### nki.language.num_programs

### Parameter Default Value Updates
The following default values changed in NKI 0.3.0:

  * `nki.isa.iota` — `offset` is now optional with a default of `0`

  * `nki.isa.core_barrier` — `engine` default changed from `unknown` to `gpsimd` (no behavioral change)

  * `nki.language.num_programs` — `axes` default changed from `None` to `0`

  * `nki.language.program_id` — `axis` now has a default value of `0`

### nki.language.num_programs

nki.language.num_programs(axes=0)
    

Number of SPMD programs along the given axes in the launch grid.

Parameters:
    

axes – the axes of the launch grid. If not provided, returns the total number of programs along the entire launch grid.

Returns:
    

the number of SPMD programs along `axes` in the launch grid.

---

### nki.language.program_ndim

nki.language.program_ndim()
    

Number of dimensions in the SPMD launch grid.

Returns:
    

the number of dimensions in the launch grid, i.e. the number of axes. 0 if no grid.

## Hardware Engine and Version Info

### engine

`engine` | Neuron Device engines.  

---

### nki.isa.engine

class nki.isa.engine(value)
    

Neuron Device engines.

Attributes

`tensor` | Tensor Engine  
---|---  
`vector` | Vector Engine  
`scalar` | Scalar Engine  
`gpsimd` | GpSIMD Engine  
`dma` | DMA Engine  
`sync` | Sync Engine  
`unknown` | Unknown Engine  
  

---

### reduce_cmd

`reduce_cmd` | Engine register reduce commands.  

---

### nki.isa.reduce_cmd

class nki.isa.reduce_cmd(value)
    

Engine register reduce commands.

Attributes

`idle` | Not using the accumulator registers  
---|---  
`reset` | Resets the accumulator registers to its initial state  
`reduce` | Keeps accumulating over the current value of the accumulator registers  
`reset_reduce` | Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators  
`load_reduce` | Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators  
  

---

### nc_version

`nc_version` | NeuronCore version.  

---

### get_nc_version

`get_nc_version` | Returns the nc_version of the current target context.  

---

### nki.isa.get_nc_version

# nki.isa.get_nc_version
nki.isa.get_nc_version()
    

Returns the nc_version of the current target context.

This document is relevant for: `Trn2`, `Trn3`

## Debugging and Profiling

### device_print

`device_print` | Print a message with a string prefix followed by the value of a tile.  

### device_print

  * `device_print` is available to inspect tensor values

---

### nki.language.device_print

nki.language.device_print(print_prefix, tensor)
    

Print a message with a string prefix followed by the value of a tile.

During kernel execution on hardware, the Neuron Runtime (NRT) exports device-printed tensors via the NRT debug stream API. By default, setting the environment variable `NEURON_RT_DEBUG_OUTPUT_DIR` to a directory path enables the default stream consumer, which dumps tensor data to that directory. The output is organized as: `<output_dir>/<print_prefix>/core_<logical_core_id>/<iteration>/`.

In CPU simulation, this prints immediately to stdout.

Parameters:
    

  * print_prefix – prefix of the print message. Evaluated at trace time; must be a constant string.

  * tensor – tensor to print out. Can be in SBUF or HBM.


## Convenience Wrapper APIs

### nki.language APIs

### `nki.language` APIs
NKI 0.3.0 introduces `nki.language` APIs as convenience wrappers around `nki.isa` APIs. These include operations such as `nl.load`, `nl.store`, `nl.copy`, `nl.matmul`, `nl.transpose`, `nl.softmax`, and other high-level operations that map to one or more `nki.isa` calls.

Note

The `nki.language` convenience APIs are experimental in NKI 0.3.0.

---

### maximum

`maximum` | Maximum of the inputs, element-wise.  

---

### minimum

`minimum` | Minimum of the inputs, element-wise.  

---

### nki.language.maximum

nki.language.maximum(x, y, dtype=None)
    

Maximum of the inputs, element-wise.

((Similar to numpy.maximum))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has the maximum of each element from x and y.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.maximum -- max(3.0, 5.0) = 5.0
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.maximum(a, b)
    expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.maximum -- with a scalar operand
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.maximum(a, 5.0)
    expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    

---

### nki.language.minimum

nki.language.minimum(x, y, dtype=None)
    

Minimum of the inputs, element-wise.

((Similar to numpy.minimum))

Warning

This API is experimental and may change in future releases.

Parameters:
    

  * x – a tile or a scalar value.

  * y – a tile or a scalar value.

  * dtype – (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);

Returns:
    

a tile that has the minimum of each element from x and y.

Examples:
    
    
    import nki.language as nl
    
    # nki.language.minimum -- min(3.0, 5.0) = 3.0
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    b = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.minimum(a, b)
    expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    
    # nki.language.minimum -- with a scalar operand
    a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    c = nl.minimum(a, 5.0)
    expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
    assert nl.equal(c, expected)
    