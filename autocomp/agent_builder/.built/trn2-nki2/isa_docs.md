## Data Types and Constants

### nki.language.bool_

nki.language.bool_ = 'bool'#
    

Boolean (True or False) stored as a byte

---

### bool_

`bool_` | Boolean (True or False) stored as a byte  

---

### nki.language.int8

nki.language.int8 = 'int8'#
    

8-bit signed integer number

---

### int8

`int8` | 8-bit signed integer number  

---

### nki.language.int16

nki.language.int16 = 'int16'#
    

16-bit signed integer number

---

### int16

`int16` | 16-bit signed integer number  

---

### nki.language.int32

nki.language.int32 = 'int32'#
    

32-bit signed integer number

---

### int32

`int32` | 32-bit signed integer number  

---

### nki.language.uint8

nki.language.uint8 = 'uint8'#
    

8-bit unsigned integer number

---

### uint8

`uint8` | 8-bit unsigned integer number  

---

### nki.language.uint16

nki.language.uint16 = 'uint16'#
    

16-bit unsigned integer number

---

### uint16

`uint16` | 16-bit unsigned integer number  

---

### nki.language.uint32

nki.language.uint32 = 'uint32'#
    

32-bit unsigned integer number

---

### uint32

`uint32` | 32-bit unsigned integer number  

---

### nki.language.float16

nki.language.float16 = 'float16'#
    

16-bit floating-point number

---

### float16

`float16` | 16-bit floating-point number  

---

### nki.language.float32

nki.language.float32 = 'float32'#
    

32-bit floating-point number

---

### float32

`float32` | 32-bit floating-point number  

---

### nki.language.bfloat16

nki.language.bfloat16 = 'bfloat16'#
    

16-bit floating-point number (1S,8E,7M)


---

### bfloat16

`bfloat16` | 16-bit floating-point number (1S,8E,7M)  

---

### nki.language.tfloat32

nki.language.tfloat32 = 'tfloat32'#
    

32-bit floating-point number (1S,8E,10M)

---

### tfloat32

`tfloat32` | 32-bit floating-point number (1S,8E,10M)  

---

### nki.language.float8_e4m3

nki.language.float8_e4m3 = 'float8_e4m3'#
    

8-bit floating-point number (1S,4E,3M)

---

### float8_e4m3

`float8_e4m3` | 8-bit floating-point number (1S,4E,3M)  

---

### nki.language.float8_e5m2

nki.language.float8_e5m2 = 'float8_e5m2'#
    

8-bit floating-point number (1S,5E,2M)

---

### float8_e5m2

`float8_e5m2` | 8-bit floating-point number (1S,5E,2M)  

---

### nki.language.float8_e4m3fn

nki.language.float8_e4m3fn = 'float8_e4m3fn'#
    

no inf, NaN represented by 0bS111’1111

Type:
    

8-bit floating-point number (1S,4E,3M), Extended range

---

### float8_e4m3fn

`float8_e4m3fn` | no inf, NaN represented by 0bS111'1111  

---

### nki.language.float8_e5m2_x4

nki.language.float8_e5m2_x4 = 'float8_e5m2_x4'#
    

4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

---

### float8_e5m2_x4

`float8_e5m2_x4` | 4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### nki.language.float8_e4m3fn_x4

nki.language.float8_e4m3fn_x4 = 'float8_e4m3fn_x4'#
    

4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

---

### float8_e4m3fn_x4

`float8_e4m3fn_x4` | 4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### nki.language.float4_e2m1fn_x4

nki.language.float4_e2m1fn_x4 = 'float4_e2m1fn_x4'#
    

4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4

---

### float4_e2m1fn_x4

`float4_e2m1fn_x4` | 4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4  

---

### Supported Data Types by NKI

## Supported Data Types
Supported Data Types by NKI below lists all supported data types by NKI. Almost all of the NKI APIs accept a data type field, dtype, which must be a nki.language data type.

Table 11 Supported Data Types by NKI# | Data Type | Accepted `dtype` Field by NKI APIs  
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

---

### Supported Math Operators by NKI ISA

## Supported Math Operators for NKI ISA
Supported Math Operators by NKI ISA below lists all the mathematical operator primitives supported by NKI. Many nki.isa APIs (instructions) allow programmable operators through the `op` field. The supported operators fall into two categories: bitvec and arithmetic. In general, instructions using bitvec operators expect integer data types and treat input elements as bit patterns. On the other hand, instructions using arithmetic operators accept any valid NKI data type and convert input elements into float32 before performing the operators.

Table 12 Supported Math Operators by NKI ISA# | Operator | `op` | Legal Reduction `op`  
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

### Supported Activation Functions by NKI ISA

## Supported Activation Functions for NKI ISA
Supported Activation Functions by NKI ISA below lists all the activation function supported by the `nki.isa.activation` API. These activation functions are approximated with piece-wise polynomials on Scalar Engine. NOTE: if input values fall outside the supported Valid Input Range listed below, the Scalar Engine will generate invalid output results.

Table 13 Supported Activation Functions by NKI ISA# Function Name | Accepted `op` by Scalar Engine | Valid Input Range  
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
Natural Log | `nki.language.log` `[2^-64, 2^64]`  
Sine | `nki.language.sin` | `[-PI, PI]`  
Arctan | `nki.language.arctan` | `[-PI/2, PI/2]`  
Square Root | `nki.language.sqrt` | `[2^-116, 2^118]`  
Reverse Square Root | `nki.language.rsqrt` | `[2^-87, 2^97]`  
Reciprocal | `nki.language.reciprocal` | `±[2^-42, 2^42]`  
Sign | `nki.language.sign` | `[-inf, inf]`  
Absolute | `nki.language.abs` | `[-inf, inf]`  

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

## Kernel Launch and SPMD Primitives

### nki.jit

nki.jit(func=None, mode='auto', **kwargs)
    

This decorator compiles a top-level NKI function to run on NeuronDevices.

This decorator tries to automatically detect the current framework and compile the function as a custom operator. To bypass the framework detection logic, you can specify the `mode` parameter explicitly.

You might need to explicitly set the target platform using the `NEURON_PLATFORM_TARGET_OVERRIDE` environment variable. Supported values are “trn1”/”gen2”, “trn2”/”gen3”, and “trn3”/”gen4”.

Parameters:
    

  * func – Function that defines the custom operation.

  * mode – Compilation mode. Supported values are “jax”, “torchxla”, and “auto”. (Default: “auto”.)

Listing 11 Writing an addition kernel using `@nki.jit`#
    
    
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

With the NKI Compiler, we have chosen to define the NKI language as a subset of Python. This means that all NKI programs are valid Python programs, but not all Python programs are valid NKI programs. The delineation is the `nki.jit` decorator. Just as before, you mark your NKI kernels with the `nki.jit` decorator. However, unlike before, the functions under this decorator will be passed to the NKI Compiler and not be evaluated by the Python interpreter.
    
    
    def a_function(x,y,z):
      # this is Python code
    
    @nki.jit
    def kernel(x,y,z):
      # this is NKI code
    

---

### nki.language.program_id

nki.language.program_id(axis)
    

Index of the current SPMD program along the given axis in the launch grid.

Parameters:
    

axis – The axis of the ND launch grid.

Returns:
    

The program id along `axis` in the launch grid

---

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

### nki.language.num_programs

nki.language.num_programs(axes=None)
    

Number of SPMD programs along the given axes in the launch grid. If `axes` is not provided, returns the total number of programs.

Parameters:
    

axes – The axes of the ND launch grid. If not provided, returns the total number of programs along the entire launch grid.

Returns:
    

The number of SPMD(single process multiple data) programs along `axes` in the launch grid

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

### nki.language.program_ndim

nki.language.program_ndim()
    

Number of dimensions in the SPMD launch grid.

Returns:
    

The number of dimensions in the launch grid, i.e. the number of axes

---

### program_ndim

`program_ndim` | Number of dimensions in the SPMD launch grid.  

## Tensor Allocation and Buffers

### nki.language.ndarray

nki.language.ndarray(shape, dtype, buffer=None, name='')
    

Create a new tensor of given shape and dtype on the specified buffer.

((Similar to numpy.ndarray))

Parameters:
    

  * shape – the shape of the tensor.

  * dtype – the data type of the tensor (see Supported Data Types for more information).

  * buffer – the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.

  * name – the name of the tensor. The `name` parameter has to be unique for tensors on each Physical NeuronCore(PNC) within each Logical NeuronCore(LNC). It is optional for SRAM tensors, IO tensors, and any HBM tensors that are only visible to one Physical NeuronCore. For `shared_hbm` tensors that are not used as kernel inputs or outputs, `name` must be specified. In addition, the compiler uses the `name` to link non-IO `shared_hbm` tensors among PNCs. In other word, `shared_hbm` tensors will point to the same underlying memory as long as they have the same name, even if the tensors appear in different control flow.

Returns:
    

a new tensor allocated on the buffer.

---

### ndarray

`ndarray` | Create a new tensor of given shape and dtype on the specified buffer.  

---

### nl.ndarray

     in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
     out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)

### nl.ndarray

    a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result",
      address=(0, 128), buffer=nl.sbuf)

---

### nl.ndarray with address parameter

### Improved Allocation API
The manual allocation API has been simplified. In Beta 2 the there is a new argument to `nl.ndarray` that allows the offset of each tensor to be specified: (partition_offset, free_offset). Similar to the Beta 1, while the partition offset corresponds to a physical partition lane on the hardware, the free dimension offset is the element offset within each partition. The free dimension offset is translated into physical SBUF address in the compiler.
    
    
    # creates your buffer on parition 0, offset by 128 elements of your data type
    a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result",
      address=(0, 128), buffer=nl.sbuf)
    

The address space for PSUM is now also 2D to be consistent with the hardware. Recall that PSUM on NeuronCore v2/v3/v4 is organized into 128 partitions, each consisting of 16KB of memory. Each partition is further divided into 8 PSUM banks, with each bank holding up to 2048 bit worth of values. The allocation for PSUM tensors must start at the beginning of each bank - the compiler will throw an error otherwise.

For example, the following code will allocate a PSUM tensor on bank 3:
    
    
    bank_id = 3
    PSUM_BANK_SIZE = 2048
    psum_t = nl.ndarray(dtype=nl.bfloat16, shape=(128, 1024),
      address=(0, bank_id*PSUM_BANK_SIZE))
    

---

### nki.language.zeros

nki.language.zeros(shape, dtype, buffer=None, name='')
    

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

### zeros

`zeros` | Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.  

---

### nki.language.shared_constant

nki.language.shared_constant(constant, dtype=None)
    

Create a tensor filled with compile-time constant data.

This function creates a tensor that contains constant data specified by a trace-time tensor. The constant data is evaluated at compile time and shared across all instances where the same constant is used, making it memory efficient for frequently used constant values.

Parameters:
    

  * constant (nki.tensor.TraceTimeTensor) – A trace-time tensor containing the constant data to be filled into the output tensor. This can be created using functions from `nki.tensor` such as `nki.tensor.zeros()`, `nki.tensor.identity()`, or `nki.tensor.arange()`.

  * dtype (nki.language dtype) – The data type of the output tensor. Must be specified. Only types that can be serialized to npy files are supported. See Supported Data Types for supported data types.

Returns:
    

A tensor containing the constant data with the specified dtype.

Return type:
    

Tensor

Note

The constant tensor is shared across all uses of the same constant data and dtype, which helps reduce memory usage in the compiled kernel.

Examples:

Create a constant identity matrix:
    
    
    import nki.tensor as ntensor
    import nki.language as nl
    
    # Create a 128x128 identity matrix as a shared constant
    identity_matrix = nl.shared_constant(
        ntensor.identity(128, dtype=nl.int8),
        dtype=nl.float16
    )
    

Create a constant tensor with sequential values:
    
    
    # Create a constant tensor with values [0, 1, 2, ..., 31]
    sequential_values = nl.shared_constant(
        ntensor.arange(0, 32, 1, dtype=nl.int32),
        dtype=nl.float32
    )
    

Create a constant tensor with arithmetic operations:
    
    
    # Create a constant tensor filled with ones
    ones_tensor = nl.shared_constant(
        ntensor.zeros((64, 64), dtype=nl.int8) + 1,
        dtype=nl.int16
    )
    

---

### shared_constant

`shared_constant` | Create a tensor filled with compile-time constant data.  

---

### nki.language.psum

nki.language.psum = 'psum'#
    

PSUM - Only visible to each individual kernel instance in the SPMD grid

---

### psum

`psum` | PSUM - Only visible to each individual kernel instance in the SPMD grid  

---

### sbuf

`sbuf` | State Buffer - Only visible to each individual kernel instance in the SPMD grid  

---

### hbm

`hbm` | HBM - Alias of private_hbm  

---

### nki.language.private_hbm

nki.language.private_hbm = 'private_hbm'#
    

HBM - Only visible to each individual kernel instance in the SPMD grid


---

### private_hbm

`private_hbm` | HBM - Only visible to each individual kernel instance in the SPMD grid  

---

### nki.language.shared_hbm

nki.language.shared_hbm = 'shared_hbm'#
    

Shared HBM - Visible to all kernel instances in the SPMD grid

---

### shared_hbm

`shared_hbm` | Shared HBM - Visible to all kernel instances in the SPMD grid  

---

### SbufManager

### SbufManager
class nkilib.core.utils.allocator.SbufManager(sb_lower_bound, sb_upper_bound, logger=None, use_auto_alloc=False, default_stack_alloc=True)#
    

Stack-based SBUF memory manager with scope support.

Parameters:
    

  * sb_lower_bound (int) – Lower bound of available SBUF memory region.

  * sb_upper_bound (int) – Upper bound of available SBUF memory region.

  * logger (Logger, optional) – Optional logger instance for allocation tracking.

  * use_auto_alloc (bool) – If True, delegates address assignment to compiler. Default False.

  * default_stack_alloc (bool) – If True, `alloc()` uses stack; if False, uses heap. Default True.

open_scope(interleave_degree=1, name='')#
    

Opens a new allocation scope. Allocations within this scope are freed when the scope closes.

Parameters:
    

  * interleave_degree (int) – Number of buffer sections for multi-buffering. Default 1.

  * name (str) – Optional scope name for debugging.

Return type:
    

None

close_scope()#
    

Closes the current scope and frees all stack allocations made within it.

Return type:
    

None

increment_section()#
    

Advances to the next buffer section within a multi-buffer scope. When all sections are used, wraps back to the first section.

Return type:
    

None

alloc_stack(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)#
    

Allocates a tensor on the stack (freed when scope closes).

Parameters:
    

  * shape (tuple[int, ...]) – Shape of the tensor.

  * dtype (dtype) – Data type (e.g., `nl.bfloat16`, `nl.float32`).

  * buffer (buffer) – Buffer type. Only `nl.sbuf` supported.

  * name (str, optional) – Optional tensor name (must be unique).

  * base_partition (int) – Base partition for allocation. Default 0.

  * align (int, optional) – Alignment requirement in bytes.

Returns:
    

Allocated SBUF tensor.

Return type:
    

nl.ndarray

alloc_heap(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)#
    

Allocates a tensor on the heap (must be manually freed with `pop_heap()`).

Parameters are identical to `alloc_stack()`.

Return type:
    

nl.ndarray

alloc(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)#
    

Allocates a tensor on the stack or heap, depending on the `default_stack_alloc` setting.

Parameters are identical to `alloc_stack()`.

Return type:
    

nl.ndarray

pop_heap()#
    

Frees the most recently allocated heap tensor.

Return type:
    

None

get_total_space()#
    

Returns the total number of bytes in the managed region.

Return type:
    

int

get_free_space()#
    

Returns the number of free bytes between stack and heap.

Return type:
    

int

get_used_space()#
    

Returns the number of bytes currently used by stack and heap allocations.

Return type:
    

int

get_stack_curr_addr()#
    

Returns the current stack address. Not supported in auto-allocation mode.

Return type:
    

int

get_heap_curr_addr()#
    

Returns the current heap address. Not supported in auto-allocation mode.

Return type:
    

int

align_stack_curr_addr(align=32)#
    

Aligns the current stack address to the given alignment. Not supported in auto-allocation mode.

Parameters:
    

align (int) – Alignment in bytes. Default 32.

Return type:
    

None

set_name_prefix(prefix)#
    

Sets a prefix string prepended to all subsequent allocation names.

Parameters:
    

prefix (str) – Prefix string.

Return type:
    

None

get_name_prefix()#
    

Returns the current name prefix.

Return type:
    

str

flush_logs()#
    

Prints buffered allocation logs in tree format.

Return type:
    

None

---

### create_auto_alloc_manager

### create_auto_alloc_manager
nkilib.core.utils.allocator.create_auto_alloc_manager(logger=None)#
    

Creates an SbufManager that delegates address assignment to the compiler.

Parameters:
    

logger (Logger, optional) – Optional logger instance.

Returns:
    

Auto-allocation SbufManager instance.

Return type:
    

SbufManager

---

### TensorView

### TensorView
class nkilib.core.utils.tensor_view.TensorView(base_tensor)#
    

A view wrapper around NKI tensors supporting various operations without copying data.

Parameters:
    

base_tensor (nl.ndarray) – The underlying NKI tensor.

shape: tuple[int, ...]#
    

Current shape of the view.

strides: tuple[int, ...]#
    

Stride of each dimension in elements.

get_view()#
    

Generates the actual NKI tensor view using array pattern.

Returns:
    

NKI tensor with the view pattern applied.

Return type:
    

nl.ndarray

slice(dim, start, end, step=1)#
    

Creates a sliced view along a dimension.

Parameters:
    

  * dim (int) – Dimension to slice.

  * start (int) – Start index (inclusive).

  * end (int) – End index (exclusive).

  * step (int) – Step size. Default 1.

Returns:
    

New TensorView with sliced dimension.

Return type:
    

TensorView

permute(dims)#
    

Creates a permuted view by reordering dimensions.

Parameters:
    

dims (tuple[int, ...]) – New order of dimensions.

Returns:
    

New TensorView with permuted dimensions.

Return type:
    

TensorView

Note: For SBUF tensors, partition dimension (dim 0) must remain at position 0.

broadcast(dim, size)#
    

Expands a size-1 dimension to a larger size without copying.

Parameters:
    

  * dim (int) – Dimension to broadcast (must have size 1).

  * size (int) – New size for the dimension.

Returns:
    

New TensorView with broadcasted dimension.

Return type:
    

TensorView

reshape_dim(dim, shape)#
    

Reshapes a single dimension into multiple dimensions.

Parameters:
    

  * dim (int) – Dimension to reshape.

  * shape (tuple[int, ...]) – New sizes (can contain one -1 for inference).

Returns:
    

New TensorView with reshaped dimension.

Return type:
    

TensorView

flatten_dims(start_dim, end_dim)#
    

Flattens a range of contiguous dimensions into one.

Parameters:
    

  * start_dim (int) – First dimension to flatten (inclusive).

  * end_dim (int) – Last dimension to flatten (inclusive).

Returns:
    

New TensorView with flattened dimensions.

Return type:
    

TensorView

expand_dim(dim)#
    

Inserts a new dimension of size 1.

Parameters:
    

dim (int) – Position to insert the new dimension.

Returns:
    

New TensorView with added dimension.

Return type:
    

TensorView

squeeze_dim(dim)#
    

Removes a dimension of size 1.

Parameters:
    

dim (int) – Dimension to remove (must have size 1).

Returns:
    

New TensorView with removed dimension.

Return type:
    

TensorView

select(dim, index)#
    

Selects a single element along a dimension, reducing dimensionality.

Parameters:
    

  * dim (int) – Dimension to select from.

  * index (int | nl.ndarray) – Index to select (int for static, nl.ndarray for dynamic).

Returns:
    

New TensorView with one fewer dimension.

Return type:
    

TensorView

rearrange(src_pattern, dst_pattern, fixed_sizes=None)#
    

Rearranges dimensions using einops-style patterns.

Parameters:
    

  * src_pattern (tuple[str | tuple[str, ...], ...]) – Source dimension pattern with named dimensions.

  * dst_pattern (tuple[str | tuple[str, ...], ...]) – Destination dimension pattern.

  * fixed_sizes (dict[str, int], optional) – Dictionary mapping dimension names to sizes.

Returns:
    

New TensorView with rearranged dimensions.

Return type:
    

TensorView

reshape(new_shape)#
    

Reshapes the tensor to new dimensions.

Parameters:
    

new_shape (tuple[int, ...]) – New dimension shape.

Returns:
    

New TensorView with reshaped dimensions.

Return type:
    

TensorView

Note

General reshape is not yet implemented and will raise an error. Use `reshape_dim` for single-dimension reshaping.

has_dynamic_access()#
    

Checks if the tensor view uses dynamic indexing (via a prior `select` with an `nl.ndarray` index).

Returns:
    

True if the view has dynamic access, False otherwise.

Return type:
    

bool

### TensorView

TensorView | Zero-copy tensor view operations including slicing, permuting, reshaping, and broadcasting.  

## Indexing, Slicing, and Access Patterns

### nki.language.ds

nki.language.ds(start, size)
    

Construct a dynamic slice for simple tensor indexing.

---

### ds

`ds` | Construct a dynamic slice for simple tensor indexing.  

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

---

### Access Pattern Semantics

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

### Access Pattern Concrete Example

## A Concrete Example
Given a tensor `t` of size (16P, 16F), to iterate all the elements in `t[0:16, 8:16]` the access pattern can be written as:
    
    
    t = nl.ndarray((16, 16), dtype=nl.float32, buffer=nl.sbuf)
    access = t.ap(pattern=[[16, 16], [1, 8]], offset=8)
    
    
    # Semantics, the following is pseudo-code
    access = nl.ndarray((16, 8), dtype=nl.float32, buffer=nl.sbuf)
    # in loop form
    for w in range(16):
      for z in range(8):
        idx = 8 + (w * 16) + (1 * z)
        t_flatten = t.flatten()
        access[w, z] = t_flatten[idx]
    

---

### SBUF/PSUM Tensor Access Pattern Restrictions

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
    

## Loop Iterators and Control Flow

### nki.language.affine_range

nki.language.affine_range(start, stop=None, step=1)
    

Create a sequence of numbers for use as parallel loop iterators in NKI. `affine_range` should be the default loop iterator choice, when there is no loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. A concrete examplesof associative reduction is the set of :doc:`nisa.nc_matmul <nki.isa.nc_matmul>`calls accumulating into the same output buffer defined outside of this loop level (see code example #2 below).

When the above conditions are not met, we recommend using sequential_range instead.

Notes:

  * Using `affine_range` prevents Neuron compiler from unrolling the loops until entering compiler backend, which typically results in better compilation time compared to the fully unrolled iterator static_range.

  * Using `affine_range` also allows Neuron compiler to perform additional loop-level optimizations, such as loop vectorization in current release. The exact type of loop-level optimizations applied is subject to changes in future releases.

  * Since each kernel instance only runs on a single NeuronCore, affine_range does not parallelize different loop iterations across multiple NeuronCores. However, different iterations could be parallelized/pipelined on different compute engines within a NeuronCore depending on the invoked instructions (engines) and data dependency in the loop body.

    
    
     1import nki.language as nl
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

### affine_range

`affine_range` | Create a sequence of numbers for use as parallel loop iterators in NKI. `affine_range` should be the default loop iterator choice, when there is no loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. A concrete examplesof associative reduction is the set of :doc:`nisa.nc_matmul <nki.isa.nc_matmul>`calls accumulating into the same output buffer defined outside of this loop level (see code example #2 below).  

---

### nki.language.static_range

nki.language.static_range(start, stop=None, step=1)
    

Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop. Unlike affine_range or sequential_range, Neuron compiler will fully unroll the loop during NKI kernel tracing.

Notes:

  * Due to loop unrolling, compilation time may go up significantly compared to affine_range or sequential_range.

  * On-chip memory (SBUF) usage may also go up significantly compared to affine_range or sequential_range.

  * No loop-level optimizations will be performed in the compiler.

  * `static_range` should only be used as a fall-back option for debugging purposes when affine_range or sequential_range is giving functionally incorrect results or undesirable performance characteristics.

---

### static_range

`static_range` | Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop.  

---

### nki.language.sequential_range

nki.language.sequential_range(start, stop, step)
    

Create a sequence of numbers for use as sequential loop iterators in NKI. `sequential_range` should be used when there is a loop carried dependency. Note, associative reductions are not considered loop carried dependencies in this context. See affine_range for an example of such associative reduction.

Notes:

  * Inside a NKI kernel, any use of Python `range(...)` will be replaced with `sequential_range(...)` by Neuron compiler.

  * Using `sequential_range` prevents Neuron compiler from unrolling the loops until entering compiler backend, which typically results in better compilation time compared to the fully unrolled iterator static_range.

  * Using `sequential_range` informs Neuron compiler to respect inter-loop dependency and perform much more conservative loop-level optimizations compared to `affine_range`.

  * Using `affine_range` instead of `sequential_range` in case of loop carried dependency incorrectly is considered unsafe and could lead to numerical errors.

    
    
     1import nki.language as nl
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

### sequential_range

`sequential_range` | Create a sequence of numbers for use as sequential loop iterators in NKI.  

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
    

---

### TiledRange

TiledRange | Divides dimensions into tiles with automatic remainder handling.  

## Tile Size and Hardware Constants

### nki.language.tile_size

class nki.language.tile_size
    

Tile size constants.

Attributes

`bn_stats_fmax` | Maximum free dimension of BN_STATS  
---|---  
`gemm_moving_fmax` | Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine  
`gemm_stationary_fmax` | Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine  
`pmax` | Maximum partition dimension of a tile  
`psum_fmax` | Maximum free dimension of a tile on PSUM buffer  
`psum_min_align` | Minimum byte alignment requirement for PSUM free dimension address  
`sbuf_min_align` | Minimum byte alignment requirement for SBUF free dimension address  
`total_available_sbuf_size` | Total SBUF available size  

---

### tile_size

`tile_size` | Tile size constants.  

---

### nki.isa.nc_version

class nki.isa.nc_version(value)
    

NeuronCore version

__init__()#
    

Attributes

`gen2` | Trn1/Inf2 target  
---|---  
`gen3` | Trn2 target  
`gen4` | Trn3 target  

---

### nc_version

`nc_version` | NeuronCore version  

---

### nki.isa.get_nc_version

nki.isa.get_nc_version()
    

Returns the `nc_version` of the current target context.

---

### get_nc_version

`get_nc_version` | Returns the `nc_version` of the current target context.  

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
`dma` | DMA Engine  
`sync` | Sync Engine  
`unknown` | Unknown Engine  

---

### engine

`engine` | Neuron Device engines  

---

### nki.isa.reduce_cmd

class nki.isa.reduce_cmd(value)
    

Engine Register Reduce commands

Attributes

`idle` | Not using the accumulator registers  
---|---  
`reset` | Resets the accumulator registers to its initial state  
`reduce` | Keeps accumulating over the current value of the accumulator registers  
`reset_reduce` | Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators  
`load_reduce` | Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators  

---

### reduce_cmd

`reduce_cmd` | Engine Register Reduce commands  

---

### nki.isa.dge_mode

class nki.isa.dge_mode(value)
    

Neuron Descriptor Generation Engine Mode

Attributes

`unknown` | Unknown DGE mode, i.e., let compiler decide the DGE mode  
---|---  
`swdge` | Software DGE  
`hwdge` | Hardware DGE  
`none` | Not using DGE  

---

### dge_mode

`dge_mode` | Neuron Descriptor Generation Engine Mode  

---

### nki.isa.oob_mode

class nki.isa.oob_mode(value)
    

Neuron OOB Access Mode

Attributes

`error` | Raise an error when an out-of-bounds access is detected  
---|---  
`skip` | Skip the out-of-bounds access silently  

---

### oob_mode

`oob_mode` | Neuron OOB Access Mode  

## Register Operations

### nki.isa.register_alloc

nki.isa.register_alloc(x=None)
    

Allocate a virtual register and optionally initialize it with an integer value `x`.

Each engine sequencer (Tensor/Scalar/Vector/GpSimd/Sync Engine) within a NeuronCore maintains its own set of physical registers for scalar operations (64x 32-bit registers per engine sequencer in NeuronCore v2-v4). The `nisa.register_alloc` API conceptually allocates a register within a virtual register space. Users do not need to expliclity free a register through nisa APIs. The NKI compiler handles physical register allocation (and deallocation) across the appropriate engine sequencers based on the dynamic program flow.

NKI provides the following APIs to manipulate allocated registers:

  * `nisa.register_move`: Move a constant value into a register

  * `nisa.register_load`: Load a scalar (32-bit) value from HBM/SBUF into a register

  * `nisa.register_store`: Store register contents to HBM/SBUF

In the current NKI release, these registers are primarily used to specify dynamic loop boundaries and while loop conditions. The NKI compiler compiles such dynamic looping constructs to branching instructions executed by engine sequencers. For additional details, see `nl.dynamic_range`. For more information on engine sequencer and its capabilities, see Trainium/Inferentia2 architecture guide.

Parameters:
    

  * dst – a virtual register object

  * x – optional integer value to initialize the register with


---

### register_alloc

`register_alloc` | Allocate a virtual register and optionally initialize it with an integer value `x`.  

### register_alloc

    def register_alloc(x: Optional[int]) -> register: ...

---

### nki.isa.register_move

nki.isa.register_move(dst, imm)
    

Move a compile-time constant integer value into a virtual register.

This instruction loads an immediate (compile-time constant) integer value into the specified virtual register. The immediate value must be known at compile time and cannot be a runtime variable. This is typically used to initialize registers with known constants for loop bounds, counters, or other control flow operations.

The virtual register system allows the NKI compiler to allocate physical registers across different engine sequencers as needed. See `nisa.register_alloc` for more details on virtual register allocation.

This instruction operates on virtual registers only and does not access SBUF, PSUM, or HBM.

Parameters:
    

  * dst – the destination virtual register (allocated via `nisa.register_alloc`)

  * imm – a compile-time constant integer value to load into the register

Example:
    
    
    # Allocate a register and initialize it with a constant
    loop_count = nisa.register_alloc()
    nisa.register_move(loop_count, 10)  # Set register to 10
    

---

### register_move

`register_move` | Move a compile-time constant integer value into a virtual register.  

### register_move

    def register_move(dst: imm: int): ...

---

### nki.isa.register_load

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

### register_load

`register_load` | Load a scalar value from memory (HBM or SBUF) into a virtual register.  

### register_load

    def register_load(dst: register, src: tensor): ...

---

### nki.isa.register_store

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

### register_store

`register_store` | Store the value from a virtual register into memory (HBM/SBUF).  

### register_store

    def register_store(dst: tensor, src: register): ...

---

### nisa.register_store

      nisa.register_store(cond, reg)

---

### nisa.register_load

      nisa.register_load(reg, cond)

## Memory and DMA Operations

### nki.isa.dma_copy

nki.isa.dma_copy(dst, src, dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, unique_indices=True, name=None)
    

Copy data from `src` to `dst` using DMA engines with optional read-modify-write operations.

This instruction performs data movement between memory locations (SBUF or HBM) using DMA engines. The basic operation copies data from the source tensor to the destination tensor: `dst = src`. Optionally, a read-modify-write operation can be performed where the source data is combined with existing destination data using a specified operation: `dst = dst_rmw_op(dst, src)`.

Currently, only `nl.add` is supported for `dst_rmw_op` when performing read-modify-write operations. When `dst_rmw_op=None`, the source data directly overwrites the destination data.

`nisa.dma_copy` supports different modes of DMA descritpor generation (DGE):

  * `nisa.dge_mode.none`: Neuron Runtime generates DMA descriptors and stores them into HBM before NEFF execution.

  * `nisa.dge_mode.swdge`: Gpsimd Engine generates DMA descriptors as part of the `nisa.dma_copy` instruction during NEFF execution.

  * `nisa.dge_mode.hwdge`: Sync Engine or Scalar Engine sequencers invoke DGE hardware block to generate DMA descriptors as part of the `nisa.dma_copy` instruction during NEFF execution.

See Trainium2 arch guide and Introduction to DMA with NKI for more discussion.

When either `sw_dge` or `hw_dge` mode is used, the `src` and `dst` tensors can have a dynamic start address which depends on a variable that cannot be resolved at compile time. When `sw_dge` is selected, `nisa.dma_copy` can also perform a gather or scatter operation, using a list of unique dynamic indices from SBUF. In both of these dynamic modes, out-of-bound address checking is turned on automatically during execution. By default a runtime error is raised (`oob_mode=oob_mode.error` as default setting). Developers can disable this error and make the nisa.dma_copy instruction skips the DMA transfer for a given dynamic address or index when it is out of bound using `oob_mode=oob_mode.skip`. If `dst_rmw_op` is specified for these dynamic modes, only `oob_mode.error` is allowed. See Beta2 NKI kernel migration guide for the latest syntax to handle dynamic addresses or indices.

Memory types.

Both `src` and `dst` tiles can be in HBM or SBUF. However, if both tiles are in SBUF, consider using an alternative for better performance:

  * nisa.tensor_copy for direct copies

  * nisa.nc_n_gather to gather elements within each partition independently

  * nisa.local_gather to gather elements within groups of partitions

Data types.

Both `src` and `dst` tiles can be any supported NKI data types (see Supported Data Types for more information).

The DMA engines automatically handle data type conversion when `src` and `dst` have different data types. The conversion is performed through a two-step process: first casting from `src.dtype` to float32, then from float32 to `dst.dtype`.

If `dst_rmw_op` is used, the DMA engines automatically cast input data types to float32 before performing the read-modify-write computation, and the final float32 result is cast to the output data type in a pipelined fashion.

Layout.

If `dst_rmw_op` is used, the computation is done element-wise between `src` and dst.

Tile size.

The total number of data elements in `src` must match that of `dst`.

Parameters:
    

  * dst – the destination tensor to copy data into

  * src – the source tensor to copy data from

  * dst_rmw_op – optional read-modify-write operation (currently only `nl.add` is supported)

  * unique_indices – optional and is only used when dst_rmw_op is used. It indicates whether the scatter indices provided are unique.

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See Trainium2 arch guide for more information.

  * oob_mode – 

(optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

    * `oob_mode.error`: (Default) Raises an error when encountering out-of-bounds indices.

    * `oob_mode.skip`: Silently skips any operations involving out-of-bounds indices.

For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

---

### dma_copy

`dma_copy` | Copy data from `src` to `dst` using DMA engines with optional read-modify-write operations.  

---

### nisa.dma_copy

         nisa.dma_copy(in_tile, input[i])
         nisa.reciprocal(out_tile, in_tile)
         nisa.dma_copy(output[i], out_tile)
     else:
       # Using multiple cores, process tiles in parallel, one per core
       i = nl.program_id(0)
       nisa.dma_copy(in_tile, input[i])
       nisa.reciprocal(out_tile, in_tile)
       nisa.dma_copy(output[i], out_tile)

### nisa.dma_copy

  * `nl.load` and `nl.store` have been removed, use `nisa.dma_copy`

---

### nki.isa.dma_transpose

nki.isa.dma_transpose(dst, src, axes=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None)
    

Perform a transpose on input `src` using DMA Engine.

The permutation of transpose follow the rules described below:

  1. For 2-d input tile, the permutation will be [1, 0]

  2. For 3-d input tile, the permutation will be [2, 1, 0]

  3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

DMA Transpose Constraints

The only valid `dge_mode` s are `unknown` and `hwdge`. If `hwdge`, this instruction will be lowered to a Hardware DGE transpose. This has additional restrictions:

  1. `src.shape[0] == 16`

  2. `src.shape[-1] % 128 == 0`

  3. `dtype` is 2 bytes

DMA Indirect Transpose Constraints

The only valid `dge_mode` s are `unknown` and `swdge`. If `swdge`, this instruction will be lowered to a Software DGE transpose. This has additional restrictions:

  1. `src` is a 3-d tile

  2. `src.shape[-1] == 128`

  3. `src.dtype` is 2 bytes

  4. `indices.shape[1] == 1`

  5. `indices.shape[0] % 16 == 0`

  6. `indices.dtype` is np.uint32

Parameters:
    

  * src – the source of transpose, must be a tile in HBM or SBUF.

  * axes – transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source. Supported axes are `(1, 0)`, `(2, 1, 0)`, and `(3, 1, 2, 0)`.

  * dge_mode – (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: `nki.isa.dge_mode.none` (turn off DGE) or `nki.isa.dge_mode.swdge` (software DGE) or `nki.isa.dge_mode.hwdge` (hardware DGE) or `nki.isa.dge_mode.unknown` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See Trainium2 arch guide for more information.

  * oob_mode – 

(optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

    * `oob_mode.error`: (Default) Raises an error when encountering out-of-bounds indices.

    * `oob_mode.skip`: Silently skips any operations involving out-of-bounds indices.

For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

---

### dma_transpose

`dma_transpose` | Perform a transpose on input `src` using DMA Engine.  

---

### nki.isa.dma_compute

nki.isa.dma_compute(dst, srcs, scales, reduce_op, name=None)
    

Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.

This instruction leverages the compute capabilities within DMA engines to perform scaled element-wise operations followed by reduction across multiple source tensors. The computation follows the pattern: `dst = reduce_op(srcs[0] * scales[0], srcs[1] * scales[1], ...)`, where each source tensor is first multiplied by its corresponding scale factor, then all scaled results are combined using the specified reduction operation. Currently, only `nl.add` is supported for `reduce_op`, and all values in `scales` must be `1.0`.

The DMA engines perform all computations in float32 precision internally. Input tensors are automatically cast from their source data types to float32 before computation, and the final float32 result is cast to the output data type in a pipelined fashion.

Memory types.

Both input `srcs` tensors and output `dst` tensor can be in HBM or SBUF. Both `srcs` and `dst` tensors must have compile-time known addresses.

Data types.

All input `srcs` tensors and the output `dst` tensor can be any supported NKI data types (see Supported Data Types for more information). The DMA engines automatically cast input data types to float32 before performing the scaled reduction computation. The float32 computation results are then cast to the data type of `dst` in a pipelined fashion.

Layout.

The computation is performed element-wise across all tensors, with the reduction operation applied across the scaled source tensors at each element position.

Tile size.

The element count of each tensor in `srcs` and `dst` must match exactly. The max number of source tensors in `srcs` is 16.

Parameters:
    

  * dst – the output tensor to store the computed results

  * srcs – a list of input tensors to be scaled and reduced

  * scales – a list of scale factors corresponding to each tensor in `srcs` (must be [1.0, 1.0, …])

  * reduce_op – the reduction operation to apply (currently only `nl.add` is supported)

---

### dma_compute

`dma_compute` | Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.  

---

### nki.isa.memset

nki.isa.memset(dst, value, engine=engine.unknown, name=None)
    

Initialize `dst` by filling it with a compile-time constant `value`, using Vector or GpSimd Engine. The memset instruction supports all valid NKI dtypes (see Supported Data Types).

Parameters:
    

  * dst – destination tile to initialize.

  * value – the constant value to initialize with

  * engine – specify which engine to use for memset: `nki.isa.vector_engine` or `nki.isa.gpsimd_engine` ; `nki.isa.unknown_engine` by default, lets compiler select the best engine for the given input tile shape

---

### memset

`memset` | Initialize `dst` by filling it with a compile-time constant `value`, using Vector or GpSimd Engine.  

---

### nisa.memset

      nisa.memset(batch_idx, value=3*128)

---

### nki.isa.tensor_copy

nki.isa.tensor_copy(dst, src, engine=engine.unknown, name=None)
    

Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

The output tile has the same partition axis size and also the same number of elements per partition as the input tile `src`.

`tensor_copy` casting behavior depends on the input and output data types.

  1. When `src` and `dst` data types are the same: `tensor_copy` performs a bit-accurate copy.

  2. When `src` and `dst` data types differ: `tensor_copy` performs an intermediate FP32 cast.

In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or output tile is in PSUM (see NeuronCore-v2 Compute Engines for details).

On NeuronCore v2, `tensor_copy` is not supported on the Scalar Engine. Instead, use nisa.activation with `op=nl.copy`.

Constraints.

  * Supported engines:
    
    * NeuronCore v2: Vector, GpSimd

    * NeuronCore v3+: Vector, Scalar, GpSimd

  * Since GpSimd cannot access PSUM, `src` and `dst` must be in SBUF when using GpSimd Engine.

Parameters:
    

  * dst – a tile with the same content and partition axis size as the `src` tile.

  * src – the source of copy, must be a tile in SBUF or PSUM.

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, compiler selects best engine based on engine workload).


---

### tensor_copy

`tensor_copy` | Create a copy of `src` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.  

---

### nki.isa.tensor_copy_predicated

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

---

### tensor_copy_predicated

`tensor_copy_predicated` | Conditionally copy elements from the `src` tile to the destination tile on SBUF / PSUM based on a `predicate` using Vector Engine.  

## Matrix Multiplication

### nki.isa.nc_matmul

nki.isa.nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None)
    

Compute `dst = stationary.T @ moving` matrix multiplication using Tensor Engine.

The figure below illustrates how to map a matrix multiplication from a mathematical definition to `nisa.nc_matmul` on Tensor Engine. For more detailed discussion of Tensor Engine capabilities, see Trainium arch guide.

Fig. 16 MxKxN Matrix Multiplication Visualization.#

Performance mode.

On NeuronCore-v2, performance mode is not supported. On NeuronCore-v3 and NeuronCore-v4, Tensor Engine supports FP8 double performance mode, enabled by setting performance mode to `double_row`. See Trainium2 arch guide for more details. `double_row` performance mode cannot be combined with Tensor Engine column tiling mode (details below).

Tiling mode. NeuronCore Tensor Engine is built upon a systolic array with 128 rows and 128 columns of processing elements (PEs). Tensor Engine supports both row and column tiling modes, which allow multiple `nc_matmul` instructions with a stationary tile size smaller than [128, 128] to run in parallel to improve hardware utilization. Row tiling mode slices the 128 PE rows into 2x 64 row tiles (NeuronCore-v2 or newer), or 4x 32 row tiles (NeuronCore-v3 or newer). Column tiling mode slices the 128 PE columns in the same fashion. The row and column tile sizes can be set independently in the `tile_size` field as a tuple `(row_size, column_size)`. The stationary tile size must not exceed the chosen `tile_size`.

In addition, a given `nc_matmul` can also pick the exact row and column tile within the 128x128 systolic array, by specifying the starting row and starting column in `tile_position` as a tuple `(start_row, start_column)`. The `start_row` must be a multiple of `row_size` specified in `tile_size` and must not exceed 128. Similarly, the `start_column` must be a multiple of `column_size` and must not exceed 128.

For example, setting `tile_position` to (64, 0) and `tile_size` to (64, 128) means using the bottom half of the systolic array.

Note, `tile_position` and `tile_size` must both be set to enable tiling mode. If they are not set, the default is to use the full systolic array, which is equivalent to `tile_position=(0, 0)` and `tile_size=(128, 128)`. The values in `tile_position` and `tile_size` tuples can be integers or affine expressions.

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

ISA operand syntax.

When inspecting matmul instructions in profiler or debug output (e.g., neuron-profile), the operands are displayed in a compact ISA syntax:
    
    
    src=<dtype>@<address>[<strides>][<num_elem>] dst=<dtype>@<address>[<strides>][<num_elem>] <M>*<K> acc_flags=<flags>
    

Where:

  * `<dtype>`: data type (e.g., `bfloat16`, `fp8e4`, `fp8e5`)

  * `<address>`: hex memory address in SBUF (for src) or PSUM (for dst)

  * `[<strides>]`: element strides per dimension

  * `[<num_elem>]`: number of elements per dimension

  * `<M>*<K>`: matmul dimensions (M rows × K contraction)

  * `acc_flags`: accumulator control flags (e.g., `2` = reset accumulator)

Parameters:
    

  * dst – the matmul output

  * stationary – the stationary operand

  * moving – the moving operand

  * is_stationary_onezero – hints to the compiler whether the `stationary` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `stationary` tile is in float32; the field has no impact for non-float32 `stationary`

  * is_moving_onezero – hints to the compiler whether the `moving` operand is a tile with ones/zeros only; setting this field explicitly could lead to 2x better performance if `moving` tile is in float32; the field has no impact for non-float32 `moving`

  * is_transpose – controls Tensor Engine transpose mode on/off starting NeuronCore-v3

  * tile_position – a 2D tuple (start_row, start_column) to control starting row in Tensor Engine tiling mode; start_column must be 0

  * tile_size – a 2D tuple (row_size, column_size) to control row tile size in Tensor Engine tiling mode; column_size must be 128

  * perf_mode – controls Tensor Engine FP8 double performance mode on/off starting NeuronCore-v3: `matmul_perf_mode.none` (default) disables double FP8 mode; `matmul_perf_mode.double_row` enables double FP8 mode which achieves 2x matmul throughput by packing two FP8 weight/ifmap element pairs and computing two multiplications in parallel per cycle; cannot be combined with column tiling mode. See Trainium2 arch guide for more information.

---

### nc_matmul

`nc_matmul` | Compute `dst = stationary.T @ moving` matrix multiplication using Tensor Engine.  

---

### nki.isa.nc_matmul_mx

nki.isa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None, name=None)
    

Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.

Note

Available only on NeuronCore-v4 and newer.

The NeuronCore-v4 Tensor Engine supports matrix multiplication of MXFP8/MXFP4 quantized matrices as defined in the OCP Microscaling standard. This instruction performs matrix multiplication between quantized `stationary` and `moving` matrices while applying dequantization scales during computation. The micro-scaling group size is 32 elements in groupss of 8 partitions × 4 elements per partition of both `stationary` and `moving` tensors. See Trainium3 arch guide for more detailed discussion.

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

The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4/float4_e2m1fn_x4) pack multiple quantized values into single elements. These packed data types are required because 4 microscaling quantized data values share 1 scale value and must operate together as a compact group.

Layout.

The contraction dimension of the matrix multiplication is along the partition dimension of `stationary` and `moving` tensors and also the x4 dimension within each packed data type element (float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4).

The free dimension of the `stationary` tile matches the partition dimension of the output `dst` tile in size, while the free dimension of the `moving` tile matches the free dimension of the `dst` tile in size.

The scale tensors follow a special layout requirement. See more details in `nisa.quantize_mx` API doc.

Tile size

  * The partition dimension size of `stationary` and `moving` must be identical and be a multiple of 32, not exceeding 128.

  * The free dimension size of `stationary` must be even and not exceed 128.

  * The free dimension size of `moving` must not exceed 512 when `dst` is in float32 or 1024 when `dst` is in bfloat16.

  * The scale tensors have partition dimensions that depend on whether the data tensors span multiple quadrants. See more details in `nisa.quantize_mx` API doc.

ISA operand syntax.

When inspecting `nc_matmul_mx` instructions in profiler or debug output (e.g., neuron-profile), the operands use a compact ISA syntax. For MX matmul, the source operand includes scale information:
    
    
    src=<dtype>@$MX[<data_addr>,<scale_addr>,<start_scale_partition>]@[<step_elem>][<num_elem>]
    dst=<dtype>@<address>[<strides>][<num_elem>] <M>*<K> acc_flags=<flags> psum_zero=<val>
    

Where the MX source access pattern `$MX[...]` contains:

  * `<data_addr>`: hex address of the quantized data in SBUF

  * `<scale_addr>`: hex address of the dequantization scales in SBUF

  * `<start_scale_partition>`: starting partition index for scale lookup

  * `[<step_elem>]`: element step size (typically 1)

  * `[<num_elem>]`: total number of elements

Additional fields:

  * `psum_zero`: PSUM zero-initialization control value

Parameters:
    

  * dst – the matrix multiplication output (PSUM tile)

  * stationary – the stationary quantized matrix (SBUF tile)

  * moving – the moving quantized matrix (SBUF tile)

  * stationary_scale – the dequantization scales for stationary matrix (SBUF tile)

  * moving_scale – the dequantization scales for moving matrix (SBUF tile)

  * tile_position – a 2D tuple (start_row, start_column) to control starting row and column in Tensor Engine tiling mode

  * tile_size – a 2D tuple (row_size, column_size) to control row and column tile sizes in Tensor Engine tiling mode


---

### nc_matmul_mx

`nc_matmul_mx` | Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.  

## Tensor-Scalar and Tensor-Tensor Arithmetic

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

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.scalar_engine, nki.isa.gpsimd_engine (only allowed for rsqrt) or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

---

### tensor_scalar

`tensor_scalar` | Apply up to two math operators to the input `data` tile by broadcasting scalar/vector operands in the free dimension using Vector or Scalar or GpSimd Engine: `(data <op0> operand0) <op1> operand1`.  

---

### nisa.tensor_scalar

      nisa.tensor_scalar(cond, cond, nl.add, -1)

---

### nki.isa.tensor_tensor

nki.isa.tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None)
    

Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine. The two tiles must have the same partition axis size and the same number of elements per partition.

The element-wise operator is specified using the `op` field. Valid choices for `op`:

  1. Any supported binary operator that runs on the Vector Engine. (See Supported Math Operators for NKI ISA for details.)

  2. `nl.power`. (Which runs on the GpSimd engine.)

For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, there is no restriction on the input/output data types, but the engine automatically casts input data types to float32 and performs the element-wise operation in float32 math. The float32 computation results are cast to `dst.dtype` at no additional performance cost.

Since GpSimd Engine cannot access PSUM, the input/output tiles cannot be in PSUM if `op` is `nl.power`. (See NeuronCore-v2 Compute Engines for details.)

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

  * engine – (optional) the engine to use for the operation: nki.isa.vector_engine, nki.isa.gpsimd_engine or nki.isa.unknown_engine (default, let compiler select best engine based on the input tile shape).

---

### tensor_tensor

`tensor_tensor` | Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine.  

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

### scalar_tensor_tensor

`scalar_tensor_tensor` | Apply two math operators in sequence using Vector Engine: `(data <op0> operand0) <op1> operand1`.  

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

### tensor_scalar_reduce

`tensor_scalar_reduce` | Perform the same computation as `nisa.tensor_scalar` with one math operator and also a reduction along the free dimension of the `nisa.tensor_scalar` result using Vector Engine.  

---

### nki.isa.tensor_scalar_cumulative

nki.isa.tensor_scalar_cumulative(dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce)
    

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

### tensor_scalar_cumulative

`tensor_scalar_cumulative` | Perform tensor-scalar arithmetic operation with cumulative reduction using Vector Engine.  

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

---

### tensor_tensor_scan

`tensor_tensor_scan` | Perform a scan operation of two input tiles using Vector Engine.  

## Reduction Operations

### nki.isa.tensor_reduce

nki.isa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None)
    

Apply a reduction operation to the free axes of an input `data` tile using Vector Engine.

The reduction operator is specified in the `op` input field (see Supported Math Operators for NKI ISA for a list of supported reduction operators). `nisa.tensor_reduce` supports two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or) and 2) arithmetic operators (e.g., add, subtract, multiply).

The reduction axes are specified in the `axis` field using a list of integer(s) to indicate axis indices. The reduction axes can contain up to four free axes and must start at the most minor free axis. Since axis 0 is the partition axis in a tile, the reduction axes must contain axis 1 (most-minor). In addition, the reduction axes must be consecutive: e.g., [1, 2, 3, 4] is a legal `axis` field, but [1, 3, 4] is not.

When the reduction `op` is an arithmetic operator, the instruction can also multiply the output reduction results by `-1.0` before writing into the output tile, at no additional performance cost. This behavior is controlled by the `negate` input field.

Memory types.

Both the input `data` and `dst` tiles can be in SBUF or PSUM.

Data types.

For bitvec operators, the input/output data types must be integer types and Vector Engine treats all input elements as bit patterns without any data type casting. For arithmetic operators, the input/output data types can be any supported NKI data types, but the engine automatically casts input data types to float32 and performs the reduction operation in float32 math. The float32 reduction results are cast to the data type of `dst`.

Layout.

`nisa.tensor_reduce` only supports free axes reduction. Therefore, the partition dimension of the input `data` is considered the parallel compute dimension. To perform a partition axis reduction, we can either:

  1. invoke a `nisa.nc_transpose` instruction on the input tile and then this `nisa.tensor_reduce` on the transposed tile, or

  2. invoke `nki.isa.nc_matmul` instructions to multiply a `nl.ones([128, 1], dtype=data.dtype)` as a stationary tensor with the input tile as a moving tensor. See more discussion on Tensor Engine alternative usage in Trainium architecture guide.

Tile size.

The partition dimension size of input `data` and output `dst` tiles must be the same and must not exceed 128. The number of elements per partition of `data` must not exceed the physical size of each SBUF partition. The number of elements per partition in `dst` must be consistent with the `axis` field. For example, if `axis` indicates all free dimensions of `data` are reduced, the number of elements per partition in `dst` must be 1.

Parameters:
    

  * dst – output tile of the reduction result

  * op – the reduction operator (see Supported Math Operators for NKI ISA for supported reduction operators)

  * data – the input tile to be reduced

  * axis – int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: `[1], [1,2], [1,2,3], [1,2,3,4]`

  * negate – if True, reduction result is multiplied by `-1.0`; only applicable when op is an arithmetic operator

  * keepdims – If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.


---

### tensor_reduce

`tensor_reduce` | Apply a reduction operation to the free axes of an input `data` tile using Vector Engine.  

---

### nki.isa.tensor_partition_reduce

nki.isa.tensor_partition_reduce(dst, op, data, name=None)
    

Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine.

Parameters:
    

  * dst – output tile with reduced result

  * op – the reduction operator (add, max, bitwise_or, bitwise_and)

  * data – the input tile to be reduced

---

### tensor_partition_reduce

`tensor_partition_reduce` | Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine.  

## Activation and Element-wise Math

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

Both input `data` and output `dst` tiles can be in any valid NKI data type (see Supported Data Types for more information). The Scalar Engine always performs the math operations in float32 precision. Therefore, the engine automatically casts the input `data` tile to float32 before performing multiply/add/activate specified in the activation instruction. The engine is also capable of casting the float32 math results into another output data type in `dst` at no additional performance cost. The `scale` parameter must have a float32 data type, while the `bias` parameter can be float32/float16/bfloat16.

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

### activation

`activation` | Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation before the activation and an optional reduction operation after the activation in the same instruction.  

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

\\[\begin{split}output = f_{act}(data * scale + bias) \\\ reduce\\_res = reduce\\_op(output, axis=<FreeAxis>)\end{split}\\]

Parameters:
    

  * dst – output tile of the activation instruction; layout: same as input `data` tile

  * op – an activation function (see Supported Activation Functions for NKI ISA for supported functions)

  * data – the input tile; layout: (partition axis <= 128, free axis)

  * reduce_op – the reduce operation to perform on the free dimension of the activation result

  * reduce_res – a tile of shape `(data.shape[0], 1)`, where data.shape[0] is the partition axis size of the input `data` tile. The result of `sum(ReductionResult)` is written in-place into the tensor.

  * bias – a vector with the same partition axis size as `data` for broadcast add (after broadcast multiply with `scale`)

  * scale – a scalar or a vector with the same partition axis size as `data` for broadcast multiply

---

### activation_reduce

`activation_reduce` | Perform the same computation as `nisa.activation` and also a reduction along the free dimension of the `nisa.activation` result using Scalar Engine.  

---

### nki.isa.exponential

nki.isa.exponential(dst, src, max_value=0.0, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_init=0.0)
    

Apply exponential function to each element after subtracting a max_value using Vector Engine.

Note

Available only on NeuronCore-v4 and newer.

This instruction computes `exp(src - max_value)` for each element. The instruction can optionally maintain a running sum of the exponential values using shared internal reduction registers in the Vector Engine.

The exponential operation is performed as:
    
    
    dst[i] = exp(src[i] - max_value)
    

When accumulation is enabled through `reduce_cmd`, the instruction also computes:
    
    
    reduce_res[i] = sum(dst[i])
    

The Vector Engine performs the computation in float32 precision internally and can output results in various data types as specified by the `dst` dtype field.

Constraints

  * Supported engines: Vector.

  * `src`, `dst` must have the same number of elements in the partition dimension.

  * `src`, `dst` must have the same number of elements in the free dimensions.

  * `src`, `dst` can be up to 4D tensor.

  * `reduce_init` should be unset or set to `0.0` when `reduce_cmd` is not `load_reduce`.

Parameters:
    

  * dst – The output tile with exponential function applied. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32, int8, int16, int32, uint8, uint16.

  * src – The input tile to apply exponential function on. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, int8, int16, int32, uint8, uint16, uint32.

  * max_value – The maximum value to subtract from each element before applying exponential (for numerical stability). Can be a scalar or vector of shape `(src.shape[0], 1)`. Supported dtypes: float32.

  * reduce_res – Optional tile to store reduction results (sum of exponentials). Must have shape `(src.shape[0], 1)`. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32.

  * reduce_cmd – Control the state of reduction registers for accumulating exponential results. Supported: `idle`, `reset_reduce`, `reduce`, `load_reduce`.

  * reduce_init – Initial value for reduction when using `reduce_cmd.load_reduce`. Supported dtypes: float32.

Accumulator behavior:

The Vector Engine maintains internal accumulator registers that can be controlled via the `reduce_cmd` parameter:

  * `reduce_cmd.reset_reduce`: Reset accumulators to 0, then accumulate the current results.

  * `reduce_cmd.reduce`: Continue accumulating without resetting (useful for multi-step reductions).

  * `reduce_cmd.load_reduce`: Load the values from `reduce_init` into the accumulator, then accumulate the current result on top of it.

  * `reduce_cmd.idle`: (default) No accumulation performed, accumulator state unknown.

Note

Even when `reduce_cmd` is set to `idle`, the accumulator state may still be modified. Always use `reset_reduce` after any Vector Engine operation that ran with `idle` mode to ensure consistent behavior.

Note

The accumulator registers are shared for other Vector Engine accumulation instructions such nki.isa.range_select, nki.isa.select_reduce, nki.isa.tensor_scalar_cumulative,

Behavior
    
    
    # Initialize reduction if requested
    if reduce_cmd == reduce_cmd.reset_reduce:
        accumulator = 0
    elif reduce_cmd == reduce_cmd.load_reduce:
        accumulator = reduce_init
    elif reduce_cmd == reduce_cmd.idle:
        accumulator = undefined  # Not used
    
    # Process each element
    for i in range(num_elements):
        dst[i] = exp(src[i] - max_value)
    
        # Update reduction if active
        if reduce_cmd != reduce_cmd.idle:
            accumulator += dst[i]
    

---

### exponential

`exponential` | Apply exponential function to each element after subtracting a max_value using Vector Engine.  

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

### reciprocal

`reciprocal` | Compute element-wise reciprocal (1.0/x) of the input `data` tile using Vector Engine.  

## Transpose Operations

### nki.isa.nc_transpose

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

  * engine – specify which engine to use for transpose: `nki.isa.tensor_engine` or `nki.isa.vector_engine`; by default, the best engine will be selected for the given input tile shape

---

### nc_transpose

`nc_transpose` | Perform a 2D transpose between the partition axis and the free axis of input `data` using Tensor or Vector Engine.  

## Selection and Predicate Operations

### nki.isa.affine_select

nki.isa.affine_select(dst, pattern, offset, channel_multiplier, on_true_tile, on_false_value, cmp_op=<function equal>, name=None)
    

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

nki.isa.range_select(dst, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.reset_reduce, reduce_res=None, reduce_op=<function maximum>, range_start=0.0, on_false_value=-3.4028235e+38, name=None)
    

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

  * on_false_value – ignored. Always set to FP32_MIN (`-3.4028235e+38`) regardless of the value passed in.

  * comp_op0 – first comparison operator

  * comp_op1 – second comparison operator

  * bound0 – tile with one element per partition for first comparison

  * bound1 – tile with one element per partition for second comparison

  * reduce_op – reduction operator to apply on across the selected output. Currently only `nl.maximum` is supported.

  * reduce_cmd – ignored. Always set to `reduce_cmd.reset_reduce` regardless of the value passed in.

  * reduce_res – optional tile to store reduction results.

  * range_start – starting base offset for index array for the free dimension of `on_true_tile`. Defaults to 0, and must be a compile-time integer.

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

Contents

---

### select_reduce

`select_reduce` | Selectively copy elements from either `on_true` or `on_false` to the destination tile based on a `predicate` using Vector Engine, with optional reduction (max).  

## Gather, Scatter, and Shuffle

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

### local_gather

`local_gather` | Gather SBUF data in `src_buffer` using `index` on GpSimd Engine.  

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

---

### nc_n_gather

`nc_n_gather` | Gather elements from `data` according to `indices` using GpSimd Engine.  

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

Parameters:
    

  * dst – the destination tile

  * src – the source tile

  * shuffle_mask – a 32-element list that specifies the shuffle source and destination partition

---

### nc_stream_shuffle

`nc_stream_shuffle` | Apply cross-partition data movement within a quadrant of 32 partitions from source tile `src` to destination tile `dst` using Vector Engine.  

---

### stream_shuffle_broadcast

stream_shuffle_broadcast | Broadcasts a single partition across the partition dimension using hardware shuffle.  

## Search and Replace

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

### max8

`max8` | Find the 8 largest values in each partition of the source tile.  

---

### nki.isa.nc_find_index8

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

### nc_find_index8

`nc_find_index8` | Find indices of the 8 given vals in each partition of the data tensor.  

---

### nki.isa.nc_match_replace8

nki.isa.nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None)
    

Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced tensor. If `dst_idx` tile is provided, the indices of the matched values are written to `dst_idx`.

This instruction reads the input `data`, replaces the first occurrence of each of the given values (from `vals` tensor) with the specified immediate constant and, optionally, output indices of matched values to `dst_idx`. When performing the operation, the free dimensions of both `data` and `vals` are flattened. However, these dimensions are preserved in the replaced output tensor and in `dst_idx` respectively. The partition dimension defines the parallelization boundary. Match, replace, and index generation operations execute independently within each partition.

The `data` tensor can be up to 5-dimensional, while the `vals` tensor can be up to 3-dimensional. The `vals` tensor must have exactly 8 elements per partition. The data tensor must have no more than 16,384 elements per partition. The replaced output will have the same shape as the input data tensor. `data` and `vals` must have the same number of partitions. Both input tensors can come from SBUF or PSUM.

Behavior is undefined if vals tensor contains values that are not in the data tensor.

If provided, a mask is applied to the data tensor.

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
    

  * dst – the modified data tensor

  * data – the data tensor to modify

  * dst_idx – (optional) the destination tile to write flattened indices of matched values

  * vals – tensor containing the 8 values per partition to replace

  * imm – float32 constant to replace matched values with


---

### nc_match_replace8

`nc_match_replace8` | Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced tensor.  

---

### nki.isa.nonzero_with_count

nki.isa.nonzero_with_count(dst, src, index_offset=0, padding_val=-1)
    

Find indices of nonzero elements in an input tensor and their total count using GpSimd Engine.

Note

Available only on NeuronCore-v3 and newer.

NOTE: this instruction only operates on partitions [0, 16, 32, …, 112] of the input tile and writes to partitions [0, 16, 32, …, 112] of the destination tile. The data in other partitions of the destination tile are not modified, including the last ‘extra’ slot for count.

This behavior is due to the physical connectivity of GpSimd engine. Each of the eight GpSimd cores connects to 16 contiguous SBUF partitions (e.g., core[0] connects to partitions[0:16]). In nonzero_with_count, each GpSimd core reads from and writes to its 0-th partition only.

This instruction takes an input array and produces an output array containing the indices of all nonzero elements, followed by padding values, and ending with the count of nonzero elements found.

The output tensor has one more element in the free dimension than the input tensor:

  * First N elements: 0-indexed positions of nonzero elements, offset by `index_offset`

  * Next T-N elements: Filled with `padding_val`

  * Last element: Count `N` of nonzero elements found

The `index_offset` parameter is useful when processing arrays in tiles, allowing indices to be relative to the original array position rather than the tile.

Example for one partition of the tensor:
    
    
    Input array (T=8): [0, 1, 1, 0, 0, 1, 0, 0]
    index_offset = 16
    padding_val = -1
    
    Output (T+1=9): [17, 18, 21, -1, -1, -1, -1, -1, 3]
    
    Where:
    
    - 17, 18, 21 are the indices (1, 2, 5) plus offset 16
    - -1 is the padding value for unused slots
    - 3 is the count of nonzero elements
    

Constraints

  * Supported arch versions: NeuronCore-v3+.

  * Supported engines: GpSimd.

  * Parameters `src`, `dst` must have the same number of elements in the partition dimension.

  * Destination tensor must have exactly 1 more element than the source tensor in the free dimension.

  * Only accesses the 0-th partition for each GpSimd core (i.e., [0, 16, 32, …, 112]).

  * `src` must be in SBUF with dtype float32 or int32.

  * `dst` must be in SBUF with dtype int32.

  * `index_offset` and `padding_val` must be int32.

Parameters:
    

  * src – Input tensor to find nonzero indices from. Only partitions [0, 16, 32, …, 112] are read from. Supported buffers: SBUF. Supported dtypes: float32, int32.

  * dst – Output tensor containing nonzero indices, padding, and count. Only partitions [0, 16, 32, …, 112] are written to. It must have one extra element than src in the free dimension. Supported buffers: SBUF. Supported dtypes: int32.

  * index_offset – Offset to add to the found indices (useful for tiled processing). Supported dtypes: int32.

  * padding_val – Value to use for padding unused output elements. Supported dtypes: int32.

Behavior
    
    
    # Find all nonzero elements in input
    nonzero_indices = []
    for i in range(len(input_array)):
        if input_array[i] != 0:
            nonzero_indices.append(i + index_offset)
    
    # Build output array
    output = []
    # Add found indices
    for idx in nonzero_indices:
        output.append(idx)
    # Add padding for remaining slots
    for _ in range(len(input_array) - len(nonzero_indices)):
        output.append(padding_val)
    # Add count as last element
    output.append(len(nonzero_indices))
    

Example
    
    
    def nonzero_with_count_kernel(in_tensor):
        in_shape = in_tensor.shape
        assert len(in_tensor.shape) == 2, "expected 2D tensor"
    
        in_tile = nl.ndarray(in_shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=in_tile, src=in_tensor)
    
        out_tile = nl.ndarray((in_shape[0], in_shape[1] + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=out_tile, src=in_tile, index_offset=0, padding_val=-1)
    
        out_tensor = nl.ndarray(out_tile.shape, dtype=out_tile.dtype, buffer=nl.hbm)
        nisa.dma_copy(dst=out_tensor, src=out_tile)
    
        return out_tensor
    

---

### nonzero_with_count

`nonzero_with_count` | Find indices of nonzero elements in an input tensor and their total count using GpSimd Engine.  

## Quantization

### nki.isa.quantize_mx

nki.isa.quantize_mx(dst, src, dst_scale, name=None)
    

Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.

Note

Available only on NeuronCore-v4 and newer.

The resulting MXFP8 tensors, `dst` and `dst_scale` are as defined in the OCP Microscaling standard. This instruction calculates the required scales for each group of 32 values in `src`, divides them by the calculated scale, and casts to the target MXFP8 datatype. The output layout is suitable for direct consumption by the `nisa.nc_matmul_mx` API running on Tensor Engine.

Memory types.

All input `src` and output tiles (`dst` and `dst_scale`) must be in SBUF. The `dst` and `dst_scale` tiles must reside in the same SBUF quadrant(s) as each other.

Data types.

The input `src` tile must be float16 or bfloat16. The output `dst` tile must be float8_e5m2_x4 or float8_e4m3fn_x4 (4-packed FP8 data types). The `dst_scale` tile must be uint8.

The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4) are 32-bit data types that pack four 8-bit float8_e5m2/float8_e4m3fn values.

Layout.

The quantization operates on groups of 32 elements from the input `src` tile, where each group consists of 8 partitions × 4 elements per partition. For each 32-element group, the instruction produces:

  * Quantized FP8 data in `dst`

  * One shared scale value in `dst_scale` per group

Logically, `dst` should have the same shape as `src` if `dst` is interpreted as a pure FP8 data type. However, in NKI, `dst` uses a custom 4-packed data type that packs four contiguous FP8 elements into a single float8_e5m2_x4/float8_e4m3fn_x4 element. Therefore, `dst` has one quarter of the element count per partition compared to that of `src`.

Logically, `dst_scale` should have 1/32 the element count of `src` due to the microscaling group size of 32. Physically, the `dst_scale` tensor follows a special SBUF quadrant (32 partitions) distribution pattern where scale values are distributed across multiple SBUF quadrants while maintaining the same partition offset at each quadrant. Within each SBUF quadrant, a 32-partition slice of `src` tile produces 32//8 = 4 partitions worth of scale, where 8 is due to each group consisted of 8 partitions × 4 elements per partition. The number of scales per partition is 1/4 of the free dimension size of the `src` tile. Different SBUF quadrants of scales are produced in parallel, with the scales written to the first (or second) 8 partitions of each SBUF quadrant. In other words, the `dst_scale` must be placed in the first 16 partitions of each SBUF quadrant. The `dst_scale` tile declaration must always occupy a multiple 32 partitions, even though not all partitions can be filled with scale values by `nisa.quantize_mx`.

Tile size.

  * The partition dimension size of `src` must be a multiple of 32 and must not exceed 128.

  * The free dimension size of `src` must be a multiple of 4 and must not exceed the physical size of each SBUF partition.

  * The `dst` tile has the same partition dimension size as `src` but a free dimension size that is 1/4 of `src` free dimension size due to the special 4-packed FP8 data types.

  * The `dst_scale` tile partition dimension depends on whether `src` spans multiple SBUF quadrants.
    
    * If `src` occupies only 32 partitions, `dst_scale` will occupy 4 partitions.

    * Otherwise, `dst_scale` will occupy the same number of partitions as `src`.

Parameters:
    

  * dst – the quantized MXFP8 output tile

  * src – the input FP16/BF16 tile to be quantized

  * dst_scale – the output scale tile


---

### quantize_mx

`quantize_mx` | Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.  

## Batch Normalization Statistics

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

---

### bn_stats

`bn_stats` | Compute mean- and variance-related statistics for each partition of an input tile `data` in parallel using Vector Engine.  

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


---

### bn_aggr

`bn_aggr` | Aggregate one or multiple `bn_stats` outputs to generate a mean and variance per partition using Vector Engine.  

## Random Number Generation

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

  * engine – specify which engine to use: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`, or `nki.isa.unknown_engine` (default, the best engine will be selected)


---

### rng

`rng` | Generate pseudo random numbers using the Vector or GpSimd Engine.  

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

### rand2

`rand2` | Generate pseudo random numbers with uniform distribution using Vector Engine.  

---

### nki.isa.rand_set_state

# nki.isa.rand_set_state

##  Contents 

# nki.isa.rand_set_state
nki.isa.rand_set_state(src_seeds, engine=engine.unknown, name=None)
    

Seed the pseudo random number generator (PRNG) inside the engine.

This instruction initializes the PRNG state for future random number generation operations. Each partition in the source tensor seeds the PRNG states for the corresponding compute lane inside the engine.

The PRNG state is cached inside the engine as a persistent state during the rest of NEFF execution. However, the state cannot survive TPB resets or Runtime reload.

Memory types.

The input `src_seeds` tile must be in SBUF or PSUM.

Data types.

The input `src_seeds` tile must be uint32.

Tile size.

  * src_seeds element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

Constraints.

  * Supported arch versions: NeuronCore-v3+.

  * Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.

  * Since GpSimd Engine cannot access PSUM, `src_seeds` must be in SBUF when using GpSimd Engine.

Parameters:
    

  * src_seeds – the source tensor containing seed values for the PRNG; must be a 2D uint32 tensor with the partition dimension representing the compute lanes and the free dimension containing the seed values

  * engine – specify which engine to use: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`, or `nki.isa.unknown_engine` (default, the best engine will be selected)


---

### rand_set_state

`rand_set_state` | Seed the pseudo random number generator (PRNG) inside the engine.  

---

### rand_get_state

`rand_get_state` | Store the current pseudo random number generator (PRNG) states from the engine to SBUF.  

---

### nki.isa.set_rng_seed

# nki.isa.set_rng_seed

##  Contents 

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


---

### set_rng_seed

`set_rng_seed` | Seed the pseudo random number generator (PRNG) inside the Vector Engine.  

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


---

### dropout

`dropout` | Randomly replace some elements of the input tile `data` with zeros based on input probabilities using Vector Engine.  

## Pattern Generation

### nki.isa.iota

nki.isa.iota(dst, pattern, offset, channel_multiplier=0, name=None)
    

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

---

### iota

`iota` | Generate a constant literal pattern into SBUF using GpSimd Engine.  

---

### nisa.iota

      nisa.iota(dynamic_idx_legal, [[1, 1]], 0, 2)

## Sequence and Segment Operations

### sequence_bounds

`sequence_bounds` | Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.  

## Scheduling and Synchronization

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

### no_reorder

        with nl.no_reorder():
            for i in range(len(in_tiles)):
                tile = in_tiles[i]
                out_tile = nl.ndarray(tile.shape, tile.dtype, buffer=nl.sbuf)
                nisa.activation(dst=out_tile, data=tile, op=nl.exp, name=f"act{i}")
                out_tiles.append(out_tile)

---

### nki.isa.core_barrier

nki.isa.core_barrier(data, cores, engine=engine.unknown, name=None)
    

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

  * engine – the engine to execute the barrier instruction on; defaults to automatic selection

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
    

---

### nki.isa.sendrecv

nki.isa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, name=None)
    

Perform point-to-point communication between NeuronCores by sending and receiving data simultaneously using DMA engines.

Note

Available only on NeuronCore-v3 or newer.

This instruction enables bidirectional data exchange between two NeuronCores within a Logical NeuronCore (LNC) configuration. The current NeuronCore sends its `src` tile to the `dst` location of the target NeuronCore specified by `send_to_rank`, while simultaneously receiving data from `recv_from_rank` into its own `dst` tile.

The use case is when NeuronCores need to exchange data for distributed computation patterns, such as all-gather communication or other collective operations where cores need to coordinate their computations by exchanging tiles.

This instruction is only allowed in NeuronCore-v3 or newer when LNC (Logical NeuronCore) is enabled. The communication occurs between NeuronCores that share the same HBM stack within the LNC configuration. Therefore, `send_to_rank` and `recv_from_rank` must be either 0 or 1.

The `pipe_id` parameter provides synchronization control by grouping sendrecv operations. Operations with the same `pipe_id` form a logical group where all operations in the group must complete before any can proceed. Operations with different `pipe_id` values can progress independently without blocking each other.

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
    

## Debugging

### device_print

`device_print` | Print a message with a string `print_prefix` followed by the value of a tile `tensor`.  

### device_print

  * `device_print` is available to inspect tensor values