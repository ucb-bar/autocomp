## Memory Spaces and Constants

### pltpu.VMEM

            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[
            # Our output lives in VMEM
            pl.BlockSpec(memory_space=pltpu.VMEM),
            # Our double-buffer lives in HBM
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        grid=(num_devices,),
        scratch_shapes=(
            [pltpu.SemaphoreType.DMA] * 3
            + [pltpu.SemaphoreType.REGULAR]  # capacity_sem
            + [pltpu.VMEM((8, 128), jnp.float32)]  # receive_scratch

### pltpu.VMEM

`pltpu.VMEM` | VMEM | SRAM  
`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

### pltpu.VMEM

            *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM allocations

### pltpu.VMEM

            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],

---

### pltpu.SMEM

`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

### pltpu.SMEM

                       scratch_shapes=[pltpu.SMEM((1,), jnp.int32)])((x, index))

---

### pltpu.SEMAPHORE

`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.

  * `MemorySpace.SEMAPHORE` is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

### pltpu.SEMAPHORE

              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),

---

### pl.ANY

`pl.ANY` | HBM (usually) or VMEM | DRAM  
`pltpu.VMEM` | VMEM | SRAM  
`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.


### pl.ANY

              pl.BlockSpec(memory_space=pl.ANY),
          ],
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pl.ANY),

### pl.ANY

        # MemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),

---

### MemorySpace

`MemorySpace`(value[, names, module, ...]) |   

---

### MemorySpace.SMEM

TPUs distinguish between vector and scalar memory spaces and in this case the output must be placed in scalar memory (`MemorySpace.SMEM`) since `i` is a scalar. For more details read TPU and its memory spaces. To call the above kernel on TPU, run:
    
    
    # TPU version
    from jax.experimental.pallas import tpu as pltpu
    
    def iota(size: int):
      return pl.pallas_call(iota_kernel,
                            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                            out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                            grid=(size,))()
    iota(8)
    

---

### with_memory_space_constraint

`with_memory_space_constraint`(x, memory_space) | Constrains the memory space of an array.  

## References and Mutable State

### Ref

  1. Users now use reference types called `Ref`s in their JAX code. This gives users more precise control over memory access and layout in JAX will more closely resemble physical layout.

  2. Users write their JAX programs using a subset of JAX primitives, along with a set of Pallas-specific primitives.

  3. Users embed their Pallas kernels in an outer JAX program via a special `pallas_call` higher-order function, that executes the kernel in a map. It is analogous to `pmap` or `shard_map`, except with references to shared memory.

We’ll go over these three extensions one at a time, by example.

Note that these APIs are still experimental and subject to change.

### Reference types
Let’s look at an example Pallas program for adding two vectors:
    
    
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    
    def add_kernel(x_ref, y_ref, o_ref):
      # In this code, `x_ref`, `y_ref` and `o_ref` are (8,)-shaped `Ref`s
      x = x_ref[:]
      y = y_ref[:]
      o_ref[:] = x + y
    x, y = jnp.arange(8), jnp.arange(8, 16)
    add = pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.int32))
    add(x, y)
    

Unlike a regular JAX program, `add_kernel` does not receive immutable array arguments. Instead, it’s provided with references that can be read from and updated in-place using NumPy-like syntax. `Ref`s are not a Pallas-specific concept – they were introduced to JAX to represent stateful computations. However, we can leverage them when writing kernels that operate on mutable memory too.

Pallas kernels not only receive `Ref`s corresponding to the inputs to the kernel, but also receive `Ref`s for the outputs as well (specified in `pallas_call` via `out_shape`). `Ref`s are special types that cannot be passed into the usual set of JAX primitives without being read from first. When you read from a `Ref` you get a JAX `Array` type out, and you must write an `Array` into a `Ref`.

#### Reading from/writing into Refs
Reading from a `Ref` corresponds to loading an array into the lowest level of the memory hierarchy (L1-cache on GPU and vector registers on TPU). Writing into a `Ref` is analogous.
    
    
    def f(x_ref, o_ref):
      # Using vanilla Python indexing
      x = x_ref[0, 2:5, :]
      # Or via Numpy advanced int indexing
      o_ref[jnp.arange(3), :] = x
    
    # Note that in order to use NumPy advanced int indexing, you need to broadcast the indices against each other into the desired multidimensional shape:
    def f(x_ref):
      # Assume x_ref is (8, 4) and we want to read out a (2, 3) slice
      x = x_ref[jnp.arange(2)[..., None], jnp.arange(3)[None, ...]]
    

Writing to `Ref`s can be done via analogous `__setitem__` style indexing.

Other forms of indexing (for example, dynamic slicing) can be done via `pallas.load` and `pallas.store`, new JAX primitives designed to make loading from/storing into memory easier. We’ll discuss these new primitives later.

### Ref

`Ref` types

Let’s dissect this function a bit. Unlike most JAX functions you’ve probably written, it does not take in `jax.Array`s as inputs and doesn’t return any values. Instead, it takes in `Ref` objects as inputs, which represent mutable buffers in memory. Note that we also don’t have any outputs but we are given an `o_ref`, which corresponds to the desired output.

Reading from `Ref`s

In the body, we are first reading from `x_ref` and `y_ref`, indicated by the `[...]` (the ellipsis means we are reading the whole `Ref`; alternatively we also could have used `x_ref[:]`). Reading from a `Ref` like this returns a `jax.Array`.

Writing to `Ref`s

We then write `x + y` to `o_ref`. Mutation has not historically been supported in JAX – `jax.Array`s are immutable! `Ref`s are new (experimental) types that allow mutation under certain circumstances. We can interpret writing to a `Ref` as mutating its underlying buffer.

Indexing and Slicing `Ref`s with `.at`

In addition to accessing the entire underlying buffer through a reference, it is possible to also access only a slice by using the `.at` property. Using `x_ref.at[slice]` does not immediately read or write data; it creates a new `Ref` object that points to a slice of the original buffer. For example `ref.at[0:128]` creates a view of the first 128 elements; `ref.at[::2]` creates a strided view.

Once you have a new `Ref` that represents a slice you can read it or write to it with the usual syntax. Here is a simple example:
    
    
    def add_sliced_kernel(x_ref, y_ref, o_ref):
      small_mid = x_ref.shape[0] // 2
    
      x_left = x_ref.at[:small_mid]
      x_right = x_ref.at[small_mid:]
      y_left = y_ref.at[:small_mid]
      y_right = y_ref.at[small_mid:]
    
      # The output shape is (4*small_mid).
      large_mid = 2*small_mid
      o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
      o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
      o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
      o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]
    

Note that using `x_ref.at[slice][...]` is equivalent to `x_ref[slice]`. The `.at` is useful if you want to compose multiple slices (e.g. `x_ref.at[block_slice][thread_slice]`) or if need to pass a slice to a subkernel function that takes a `Ref`.

### Ref

`Ref`(aval, refs) | Mutable array reference.  

---

### AbstractRef

`AbstractRef`(inner_aval[, memory_space, kind]) | Abstract mutable array reference.  

---

### new_ref

`new_ref`(init_val, *[, memory_space, kind]) | Create a mutable array reference with initial value `init_val`.  

---

### get

`get`(ref[, idx]) | Read a value from an Ref.  

---

### set

`set`(ref, idx, value) | Set a value in an Ref in-place.  

---

### swap

`swap`(ref, idx, value[, _function_name]) | Update an array value inplace while returning the previous value.  

---

### addupdate

`addupdate`(ref, idx, x) | Add to an element in an Ref in-place.  

---

### freeze

`freeze`(ref) | Invalidate a given reference and return its final value.  

---

### get_global

`get_global`(what) | Returns a global reference that persists across all kernel invocations.  

## Memory Load and Store Operations

### pallas.load

#### `pallas.load` and `pallas.store`
`pallas.load` and `pallas.store` are primitives that allow loading from memory and storing into memory. Unlike `__getitem__` and `__setitem__` they are more flexible at the cost of being more verbose. Specifically, you can use the `pallas.dynamic_slice` (`pallas.ds` for short) construct (which should maybe be upstreamed into JAX to be used with Ref `__getitem__` and `__setitem__`).
    
    
    def f(x_ref, o_ref):
      # Reading from memory via pallas.load
      x = pl.load(x_ref, (0, slice(2, 5), slice(None)))
      # Using integer indexing automatically broadcasts
      x = pl.load(x_ref, (0, 2 + jnp.arange(3), slice(None)))
      # You can also use `pl.dynamic_slice` (`pl.ds` for short) objects as well
      pl.store(o_ref, (0, pl.ds(start=2, size=3), slice(None)), x)
    

`pallas.load` and `pallas.store` also support masking via the mask argument.
    
    
    def f(x_ref, o_ref):
      # Reading from memory via pallas.load
      idx = jnp.arange(8)
      mask = idx < 5
      x = pl.load(x_ref, (idx,), mask=mask, other=float('-inf'))
    

Masking is important when doing out-of-bounds loads/stores. The operational semantics of masking can be compiler-determined (if we understand the documentation properly, Triton avoids the read from/write to memory if it’s masked).

---

### load

`load`(ref, *[, mask]) | Loads an array from the given ref.  

---

### store

`store`(ref, val, *[, mask]) | Stores a value to the given ref.  

## Slicing and Indexing

### pl.ds

`pltpu.emit_pipeline` supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with `pl.BoundedSlice(max_size)` rather than a static integer size, where `max_size` is the maximum size of the block. In addition, the corresponding index returned by `index_map` should be a dynamic slice constructed via `pl.ds(start, size)` where both `start` and `size` are element indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:
    
    
    pl.BlockSpec(
       block_shape=(pl.BoundedSlice(32), 256),
       index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
    )
    
    
    
    # The following kernel copies `x` to the output in dynamic-sized chunks
    # passed in via `slices`.
    
    def dynamic_block_example_kernel(x_hbm, slices_hbm, o_hbm, slices_smem):
        pltpu.sync_copy(slices_hbm, slices_smem)  # Copy slices into SMEM.
        def pipeline_body(x_vmem, o_vmem):
            o_vmem[...] = x_vmem[...]
        def index_map(i):
            start = slices_smem[i, 0]
            size = slices_smem[i, 1] - slices_smem[i, 0]
            return (pl.ds(start, size), 0)

### pl.ds

      # Slices can be specified using pl.ds(start, size)
      left_copy_slice = pl.ds(0, block_size[0] // 2)
      right_copy_slice = pl.ds(block_size[0] // 2, block_size[0] // 2)
      current_phase_slice = pl.ds(phase * (block_size[0] // 2), block_size[0] // 2)

### pl.ds

      slc = pl.ds(core_index * slc_size, slc_size)

---

### dslice

`dslice`(start[, size, stride]) | Constructs a `Slice` from a start index and a size.  

---

### Slice

`Slice`(start, size[, stride]) | A slice with a start index and a size.  

---

### pl.BoundedSlice

`pltpu.emit_pipeline` supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with `pl.BoundedSlice(max_size)` rather than a static integer size, where `max_size` is the maximum size of the block. In addition, the corresponding index returned by `index_map` should be a dynamic slice constructed via `pl.ds(start, size)` where both `start` and `size` are element indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:
    
    
    pl.BlockSpec(
       block_shape=(pl.BoundedSlice(32), 256),
       index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
    )

### pl.BoundedSlice

              block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,

---

### pl.multiple_of

          pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8), j)

---

### multiple_of

`multiple_of`(x, values) | A compiler hint that asserts a value is a static multiple of another.  

---

### Blocked

The behavior documented above applies to the default “blocked” indexing mode. When integers are used in the `block_shape` tuple e.g. `(4, 8)`, it is equivalent to passing in a `pl.Blocked(block_size)` object instead, e.g. `(pl.Blocked(4), pl.Blocked(8))`. Blocked indexing mode means the indices returned by `index_map` are block indices. We can pass in objects other than `pl.Blocked` to change the semantics of `index_map`, most notably, `pl.Element(block_size)`.. When using the `pl.Element` indexing mode the values returned by the index map function are used directly as the array indices, without first scaling them by the block size. When using the `pl.Element` mode you can specify virtual padding of the array as a tuple of low-high paddings for the dimension: the behavior is as if the overall array is padded on input. No guarantees are made for the padding values in element mode, similarly to the padding values for the blocked indexing mode when the block shape does not divide the overall array shape.

## Math and Arithmetic

### jax.lax.dot_general

      acc_ref[...] += jax.lax.dot_general(
          x_ref[...], y_ref[...], dims, preferred_element_type=jnp.float32,
      )

---

### jnp.dot

      acc_ref[...] += jnp.dot(
          x_ref[...], y_ref[...], preferred_element_type=jnp.float32
      )

---

### dot

`dot`(a, b[, trans_a, trans_b, allow_tf32, ...]) | Computes the dot product of two arrays.  

---

### cdiv

`cdiv`() | Computes the ceiling division of a divided by b.  

## Array Creation and Manipulation

### empty

`empty`(shape, dtype, *[, out_sharding]) | Create an empty array of possibly uninitialized values.  

---

### empty_like

`empty_like`(x) | Create an empty PyTree of possibly uninitialized values.  

---

### broadcast_to

`broadcast_to`(a, shape) | Broadcasts an array to a new shape.  

## Random Number Generation

### jax.random.bits()

  * `jax.random.bits()`

---

### jax.random.uniform()

  * `jax.random.uniform()`

---

### jax.random.normal()

  * `jax.random.normal()`

---

### jax.random.key()

  * `jax.random.key()`

---

### jax.random.fold_in()

  * `jax.random.fold_in()`

---

### pltpu.prng_seed()

Using the Pallas PRNG in stateful mode is the most native and efficient method for generative random numbers. First, the PRNG seed should be set using `pltpu.prng_seed(N)`, where N is an integer seed.

---

### pltpu.stateful_uniform()

  * `pltpu.stateful_uniform`: the stateful equivalent to `jax.random.uniform()`

  * `pltpu.stateful_normal`: the stateful equivalent to `jax.random.normal()`

  * `pltpu.stateful_bernoulli`: the stateful equivalent to `jax.random.bernoulli()`

Generating any random number updates the internal state of the PRNG and subsequent calls will generate different numbers. Unlike in JAX, there is no need to `split` or `fold_in` keys and pass them into the sampling functions.

For example, the following kernel generates a set of uniform numbers from 0 to 1:
    
    
    from jax.experimental.pallas import tpu as pltpu
    
    def kernel_body(o_ref):
      pltpu.prng_seed(0)
      o_ref[...] = pltpu.stateful_uniform(shape=o_ref.shape, minval=0.0, maxval=1.0)

---

### pltpu.to_pallas_key()

Pallas offers an intermediate API between the stateless API described previously and the stateless `jax.random` API and allows you to use the hardware PRNG in a stateless manner. In order to do so, convert a JAX key into a special Pallas-typed key via `pltpu.to_pallas_key(key)` and pass this key into the kernel via SMEM. Once the key is dereferenced inside the kernel, it can be passed into supported sampling functions from `jax.random` to produce random numbers. Compared to the stateless API, there is an overhead of computing and setting a seed every time the random number generator is invoked.

For example, the following kernel draws uniform numbers using the hardware PRNG:
    
    
    def body(key_ref, o_ref):
      o_ref[...] = jax.random.uniform(
          key_ref[...], shape=o_ref[...].shape
      )
    
    rbg_key = jax_random.key(0, impl="threefry2x32")
    key = pltpu.to_pallas_key(rbg_key)
    o_shape = jax.ShapeDtypeStruct((8, 128), dtype)
    result = pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_shape=o_shape,
    )(key)

---

### pltpu.sample_block()

Next, call `pltpu.sample_block` with the following arguments:
    
    
    pltpu.sample_block(
      sampler_function,  # A JAX random function, such as `jax.random.uniform`.
      global_key,  # A global key shared across all blocks.
      block_size,  # The local block size to generate.
      tile_size,  # The tile size.
      total_size,  # The total shape of the generated array across all blocks.
      block_index,  # The block index into total_size. Usually this is the current program instance.
      **sampler_kwargs  # Keyword arguments to sampler_function
    )
    

For example, the following snippet generates identical numbers over a (16, 128) block shape, and a (32, 256) block shape with a transposed grid iteration order:
    
    
    def make_kernel_body(index_map):
      def body(key_ref, o_ref):
        key = key_ref[...]
        samples = pltpu.sample_block(
            jax.random.uniform,
            key,
            block_size=o_ref[...].shape,
            tile_size=(16, 128),
            total_size=(64, 512),
            block_index=index_map(pl.program_id(0), pl.program_id(1)),
            minval=0.0,
            maxval=1.0)
        o_ref[...] = samples
      return body
    
    global_key = pltpu.to_pallas_key(jax_random.key(0))
    o_shape = jnp.ones((64, 512), dtype=jnp.float32)
    key_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    out_spec = pl.BlockSpec((16, 128), lambda i, j: (i, j))
    result_16x128 = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (i, j)),
        out_shape=o_shape,
        in_specs=[key_spec],
        out_specs=out_spec,
        grid=(4, 4),
    )(global_key)
    
    out_spec = pl.BlockSpec((32, 256), lambda i, j: (j, i))
    result_32x256_transposed = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (j, i)),
        in_specs=[key_spec],
        out_shape=o_shape,
        out_specs=out_spec,
        grid=(2, 2),
    )(global_key)
    

---

### prng_seed

`prng_seed`(*seeds) | Sets the seed for PRNG.  

---

### sample_block

`sample_block`(sampler_fn, global_key, ...[, ...]) | Samples a block of random values with invariance guarantees.  

---

### stateful_bernoulli

`stateful_bernoulli`(*args, **kwargs) | Sample Bernoulli random values with given shape and mean.  

---

### stateful_bits

`stateful_bits`(*args, **kwargs) | Sample uniform bits in the form of unsigned integers.  

---

### stateful_normal

`stateful_normal`(*args, **kwargs) | Sample standard normal random values with given shape and float dtype.  

---

### stateful_uniform

`stateful_uniform`(*args, **kwargs) | Sample uniform random values in [minval, maxval) with given shape/dtype.  

---

### to_pallas_key

`to_pallas_key`(key) | Helper function for converting non-Pallas PRNG keys into Pallas keys.  

## Semaphores and Synchronization

### DMA Semaphores

### DMA Semaphores
`send_sem` and `recv_sem` are instances of a special type of semaphore reserved exclusively for use with DMAs. They must be allocated with the `tpu.SemaphoreType.DMA` type when specifying input specs to `pallas_call`.

Internally, DMA semaphores can be thought of as integer-valued progress trackers. On DMA start, the local device will begin to increment the value of `send_sem` and the receiver’s `recv_sem` asynchronously. Waiting on a semaphore will block until the value of the semaphore reaches the total bytes of data sent/received; when the value is reached, waiting threads are released and the semaphore’s value is decremented by the same amount. This means that either all data has been sent (for `send_sem`) or all data has been received (for `recv_sem`). The value of the semaphore can be read with `pl.semaphore_read`, but note that the underlying semantics of the value could change between hardware generations (e.g. the value may not represent exactly the number of bytes sent, although this is a useful mental model to have when reasoning about the behavior of the semaphore).


---

### SemaphoreType

`SemaphoreType`(value[, names, module, ...]) |   

---

### pltpu.SemaphoreType.REGULAR

#### Regular Semaphores
Regular semaphores are the standard tool used to synchronize across multiple devices. Semaphores are fundamentally counters - they can be incremented by any device after which a device can block until the value of the semaphore reaches a specific value (and then decrement the value).

The three main operations that can be used on regular semaphores are signal, wait, and read:
    
    
    def semaphore_signal(
        sem: Ref[SemaphoreType],
        inc: int,
        device_id: int | tuple[int, ...],
        device_id_type: DeviceIdType
    ) -> None:
      ... # Increments the semaphore `sem` on the target device `device_id` by `inc`.
    
    def semaphore_wait(
        semaphore: Ref[SemaphoreType],
        value: int,
    ) -> None:
      ... # Blocks until the locally allocated copy of `sem` reaches `value`, then decrement by `value` and proceed.
    
    def semaphore_read(
        sem: Ref[SemaphoreType],
    ) -> jax.Array:
      ...  # Returns the current value of `sem` as an `int32[]`.
    

In order to use regular semaphores, they can be allocated in the same way as a DMA semaphore, but by specifying `pltpu.SemaphoreType.REGULAR` rather than `pltpu.SemaphoreType.DMA`.

---

### pl.semaphore_signal

Before communicating between cores, it is good practice to perform a barrier (using `pl.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).
    
    
    # This runs on every core
    def swap_cores_kernel(in_hbm, out_hbm,
                          in_vmem, scratch_vmem, out_vmem,
                          sem, send_sem, recv_sem):
      core_index = jax.lax.axis_index('core')
      num_cores = jax.lax.axis_size('core')
      slc_size = in_hbm.shape[-1] // num_cores
      slc = pl.ds(core_index * slc_size, slc_size)
    
      # Copy in a core-dependent slice of the input
      pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()
    
      # A barrier to make sure all cores have entered run_scoped.
      # You won't need this if not doing inter-core communications.
      dst_core = (core_index + 1) % num_cores
      sem0 = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})

### pl.semaphore_signal

The three main operations that can be used on regular semaphores are signal, wait, and read:
    
    
    def semaphore_signal(
        sem: Ref[SemaphoreType],
        inc: int,
        device_id: int | tuple[int, ...],
        device_id_type: DeviceIdType
    ) -> None:
      ... # Increments the semaphore `sem` on the target device `device_id` by `inc`.
    
    def semaphore_wait(
        semaphore: Ref[SemaphoreType],
        value: int,
    ) -> None:
      ... # Blocks until the locally allocated copy of `sem` reaches `value`, then decrement by `value` and proceed.
    
    def semaphore_read(
        sem: Ref[SemaphoreType],
    ) -> jax.Array:
      ...  # Returns the current value of `sem` as an `int32[]`.
    

---

### pl.semaphore_wait

Before communicating between cores, it is good practice to perform a barrier (using `pl.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).
    
    
    # This runs on every core
    def swap_cores_kernel(in_hbm, out_hbm,
                          in_vmem, scratch_vmem, out_vmem,
                          sem, send_sem, recv_sem):
      core_index = jax.lax.axis_index('core')
      num_cores = jax.lax.axis_size('core')
      slc_size = in_hbm.shape[-1] // num_cores
      slc = pl.ds(core_index * slc_size, slc_size)
    
      # Copy in a core-dependent slice of the input
      pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()
    
      # A barrier to make sure all cores have entered run_scoped.
      # You won't need this if not doing inter-core communications.
      dst_core = (core_index + 1) % num_cores
      sem0 = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})
      pl.semaphore_wait(sem0, 1)

### pl.semaphore_wait

The three main operations that can be used on regular semaphores are signal, wait, and read:
    
    
    def semaphore_signal(
        sem: Ref[SemaphoreType],
        inc: int,
        device_id: int | tuple[int, ...],
        device_id_type: DeviceIdType
    ) -> None:
      ... # Increments the semaphore `sem` on the target device `device_id` by `inc`.
    
    def semaphore_wait(
        semaphore: Ref[SemaphoreType],
        value: int,
    ) -> None:
      ... # Blocks until the locally allocated copy of `sem` reaches `value`, then decrement by `value` and proceed.
    
    def semaphore_read(
        sem: Ref[SemaphoreType],
    ) -> jax.Array:
      ...  # Returns the current value of `sem` as an `int32[]`.
    

---

### semaphore_read

`semaphore_read`(sem_or_view) | Reads the value of a semaphore.  

---

### semaphore_signal

`semaphore_signal`(sem_or_view[, inc, ...]) | Increments the value of a semaphore.  

---

### semaphore_wait

`semaphore_wait`(sem_or_view[, value, decrement]) | Blocks execution of the current thread until a semaphore reaches a value.  

---

### pltpu.semaphore_signal

      pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)

---

### pltpu.semaphore_wait

      pltpu.semaphore_wait(barrier_sem, 1)

---

### pltpu.get_barrier_semaphore

#### Barrier Semaphores
Barrier semaphores are globally-allocated semaphores used to synchronize devices across an entire program and ensure that all devices have entered the Pallas kernel.

If a Pallas kernel is executed within the context of a larger XLA program, we need to ensure that all devices that communicate have entered the kernel. However, DMA and regular semaphores are both locally scoped - they are only understood by other devices that have entered the kernel. Barrier semaphores serve as a globally understood semaphore that can be used for synchronization no matter where in the XLA program the device is currently executing.

By default, if you do not specify a barrier semaphore, Pallas will automatically insert a barrier semaphore at the beginning of your program. However, it can be more efficient to write your own. Barrier semaphores are similar to regular semaphores in that they are counters that can be incremented via `semaphore_signal` and can be decremented via `semaphore_wait`. They are created by calling `get_barrier_semaphore()` within a kernel. Typically, we use barriers once at the beginning of a kernel to synchronize with all devices we are communicating with.
    
    
    from jax.experimental.pallas import tpu as pltpu
    
    def example_kernel(...):
      # Use barrier semaphores at the beginning of a kernel.
      # is_start_of_kernel = ...
      # right_neighbor = ...
      # ...
      @pl.when(is_start_of_kernel)
      def _():
        barrier_sem = pltpu.get_barrier_semaphore()
        # Increment the semaphore of your right neighbor.
        pl.semaphore_signal(
              barrier_sem,
              device_id=right_neighbor,
              device_id_type=pl.DeviceIdType.LOGICAL,
        )
        # Wait until your left neighbor has incremented your semaphore
        pl.semaphore_wait(barrier_sem, 1)
      # ...
    

### pltpu.get_barrier_semaphore

Before communicating between cores, it is good practice to perform a barrier (using `pl.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).
    
    
    # This runs on every core
    def swap_cores_kernel(in_hbm, out_hbm,
                          in_vmem, scratch_vmem, out_vmem,
                          sem, send_sem, recv_sem):
      core_index = jax.lax.axis_index('core')
      num_cores = jax.lax.axis_size('core')
      slc_size = in_hbm.shape[-1] // num_cores
      slc = pl.ds(core_index * slc_size, slc_size)
    
      # Copy in a core-dependent slice of the input
      pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()
    
      # A barrier to make sure all cores have entered run_scoped.
      # You won't need this if not doing inter-core communications.
      dst_core = (core_index + 1) % num_cores
      sem0 = pltpu.get_barrier_semaphore()

---

### get_barrier_semaphore

`get_barrier_semaphore`() | Returns a barrier semaphore.  

---

### core_barrier

`core_barrier`(sem, *, core_axis_name) | Synchronizes all cores in a given axis.  

---

### optimization_barrier

To force the `x + 1` to happen between the `ppermute` ops, we can use `optimization_barrier`, which is semantically the identity function (i.e. `lambda x: x`) but introduces an explicit data dependency between values. Specifically, if we make the `x` that is used in `x + 1` dependent on the `fut` returned by `ppermute_start`, it must happen after `ppermute_start`.

We also introduce a dependency that forces the output value `y` to depend on `z`.
    
    
    def f(x):
      fut = ppermute_start(x)
      x, fut = optimization_barrier((x, fut))  # x now depends on fut
      z = x + 1
      z, fut = optimization_barrier((z, fut)) # fut now depends on z
      y = ppermute_done(fut)
      return y, z

## DMA and Async Copy Operations

### make_async_remote_copy

### Async Remote Copy Operation
The `pltpu.make_async_remote_copy` function is used to create a remote DMA descriptor object which parameterizes both a “send” operation and a “receive” operation. Here’s its signature:
    
    
     def make_async_remote_copy(
         src_ref: Ref,
         dst_ref: Ref,
         send_sem: Ref[SemaphoreType],
         recv_sem: Ref[SemaphoreType],
         device_id: int | tuple[int, ...],
         device_id_type: DeviceIdType
     ) -> AsyncCopyDescriptor:
    

  * `src_ref` is the local `Ref` (in any memory space) containing the data you wish to send to `dst_ref` on another device.

  * `dst_ref` is the remote `Ref` (in any memory space) at which data will be copied to on the target device.

  * `send_sem` is a DMA semaphore used to block until all data has been sent from `src_ref`.

  * `recv_sem` is a DMA semaphore used to block until the expected number of bytes have been received at `dst_ref`. The sender of the DMA will write to the receiver’s `recv_sem`.

  * `device_id` is the device ID of the target device to send to.

  * `device_id_type` specifies the format of `device_id`, which can either be in LOGICAL format (integer device ID), or in MESH format (an ND-tuple index into the logical device mesh). The default mode is MESH.

`make_async_remote_copy` returns a descriptor object on which you use the `.start()` method to initiate the DMA, and the `.wait_send()` to block on `send_sem` and `.wait_recv()` to block on `recv_sem` (or `.wait()` to block on both). If a device is only expected to send data, it is sufficient to only call `.start()` and `.wait_send()`, and likewise if a device is only receiving it is sufficient to only call `.wait_recv()`. If using a SPMD pattern where all devices execute the DMA, each device will generally call both `.start()` and `.wait()`.
    
    
    dma_descriptor = make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id)
    dma_descriptor.start() # Initiate the DMA (non-blocking).
    # ... do other work
    dma_descriptor.wait_send() # Block until all data has been sent.
    dma_descriptor.wait_recv() # Block until all data has been received.
    

### make_async_remote_copy

`make_async_remote_copy`(src_ref, dst_ref, ...) | Creates a description of a remote copy operation.  

---

### pltpu.make_async_copy

      @pl.when(outer_step == 0)
      def _():
        local_copy_op = pltpu.make_async_copy(
          src_ref=input_ref,
          dst_ref=output_ref.at[my_id],
          sem=local_copy_sem,
        )
        local_copy_op.start()
        local_copy_op.wait()

### pltpu.make_async_copy

      pltpu.make_async_copy(ref, ref, send_sem).wait()
      pltpu.make_async_copy(ref, ref, recv_sem).wait()

---

### pltpu.make_async_remote_copy

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).
    
    
    # This runs on every core
    def swap_cores_kernel(in_hbm, out_hbm,
                          in_vmem, scratch_vmem, out_vmem,
                          sem, send_sem, recv_sem):
      core_index = jax.lax.axis_index('core')
      num_cores = jax.lax.axis_size('core')
      slc_size = in_hbm.shape[-1] // num_cores
      slc = pl.ds(core_index * slc_size, slc_size)
    
      # Copy in a core-dependent slice of the input
      pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()
    
      # A barrier to make sure all cores have entered run_scoped.
      # You won't need this if not doing inter-core communications.
      dst_core = (core_index + 1) % num_cores
      sem0 = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})
      pl.semaphore_wait(sem0, 1)
    
      # Swap data between core 0 and core 1
      the_copy = pltpu.make_async_remote_copy(
          in_vmem, scratch_vmem, send_sem, recv_sem, device_id={'core': dst_core},
      )

### pltpu.make_async_remote_copy

      descriptor = pltpu.make_async_remote_copy(x_ref, y_ref, send_sem, recv_sem, device_id=right_neighbor)

---

### async_copy

`async_copy`(src_ref, dst_ref, sem, *[, ...]) | Issues a DMA copying from src_ref to dst_ref.  

---

### async_remote_copy

`async_remote_copy`(src_ref, dst_ref, ...[, ...]) | Issues a remote DMA copying from src_ref to dst_ref.  

---

### make_async_copy

`make_async_copy`(src_ref, dst_ref, sem) | Creates a description of an asynchronous copy operation.  

---

### pltpu.sync_copy

This involves pre-allocating an SMEM buffer (via the `pl.run_scoped` call inside `kernel`) and populating the buffer using a `sync_copy` before the pipeline starts. Close over the dynamic index value inside the `index_map` to use it.

Manually delegate work amongst cores

The code example below also shows how `core_map` allows you to customize exactly how the work is split between cores, without relying on the automatic API shown above.

To achieve that, customize your `index_map` to use the core index to work on different slices on different cores.
    
    
    input_shape = (1024, 1024)
    in_spec = jax.P('device', None)
    output_shape = (1024, 512)
    
    def indexed_add_one_kernel(in_refs, out_refs, i_smem_ref):
      (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
      in_shape = x_hbm_ref.shape
      pltpu.sync_copy(i_hbm_ref, i_smem_ref)

### pltpu.sync_copy

      pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)

---

### sync_copy

`sync_copy`(src_ref, dst_ref, *[, add]) | Synchronously copies a PyTree of refs to another PyTree of refs.  

---

### Double-buffering technique

### Double-buffering
In order to avoid reading from a local `Ref` that is also being written into by another device and creating a race condition, a useful technique is the “double-buffered” strategy where we allocate a two `Ref`s for each destination value. On each iteration, one `Ref` will be designated as a “working” slot, and the other will be designated as a “receiving” slot. The device is free to use the working slot for computation, but will only copy data into its neighbor’s receiving slot. The working and receiving slots alternate every iteration, so that once a copy is finished, the old receiving slot becomes the new working slot, and vice versa. Using this scheme properly, data is never read from and written to the same buffer.

The following code skeleton demonstrates how double-buffering can be used. We keep a running iteration counter in the variable `iteration`, and the `working_slot` and `receiving_slot` alternate between 0 and 1 every iteration. `dst_ref` is allocated as a double-buffer and has the size `[2, ...]`. On each iteration, we read from the working slot using `dst_ref.at[working_slot, ...]` and use the value to perform computation. Simultaneously, we copy to our neighbor’s `dst_ref.at[receiving_slot]` to avoid overwriting their `working_slot` value. By structuring our communication in this fashion it is possible to overlap the communication latency of the remote DMA with local computation while minimizing the risk of race conditions.
    
    
    def kernel(...):
      # ...
      iteration = pl.program_id(0)
      working_slot = lax.rem(iteration, 2)
      receiving_slot = 1 - working_slot
      # ...
    
      local_copy_op = pltpu.make_async_copy(
        src_ref=dst_ref.at[working_slot, ...],
        dst_ref=local_scratch_ref,
        sem=local_copy_sem,
      )
      local_copy_op.start()
      remote_copy_op = pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=dst_ref.at[receiving_slot, ...],
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=target_device,
        device_id_type=pl.DeviceIdType.MESH,
      )
      remote_copy_op.start()
    
      local_copy_op.wait()
      # ... do work on local_scratch while waiting for async_copy_op to finish.
      remote_copy_op.wait()
    
    

In terms of synchronization, the double-buffered construction works if all devices are executing on the same iteration. If a sender manages to get one iteration ahead of its receiver, it’s `working_slot` and `receiving_slot` indices will be flipped compared to the receiver, meaning that it could be writing into the `working_slot` at the same time the receiver is reading from it. In order to avoid this, it may be necessary to use a semaphore to synchronize the sender with the receiver, or add additional buffering slots (“triple”, “quadruple”, or N-buffered) to allow additional run-ahead at the cost of more memory. In our previous `all_gather` example, note that the kernel contained a receiving buffer with N slots, which avoids race conditions altogether. In our next kernel, we will instead go through an example which uses a double-buffer with explicit synchronization.

---

### BufferedRef

`BufferedRef`(_spec, _buffer_type, ...[, ...]) | A helper class to automate VMEM double buffering in pallas pipelines.  

## Control Flow

### pl.when

As an example, let’s visualize a DMA where we consider 4 devices (indexed 0, 1, 2, 3). We consider a scheme where device 0 copies to device 1, and device 2 & 3 copy to each other. In practice, we can create such an asymmetric communication pattern by using `@pl.when` to branch on the device ID.

(1) Each device creates the DMA descriptor. Devices 0, 2, and 3 call `.start()` to initiate the DMA from `src_ref`. Device 1 is skips the `.start()` and does nothing, e.g. by using `pl.when`.

(2) As `.start()` is non-blocking, each device is free to do other computation while the DMA is in flight. Devices 0, 2, and 3 call `.wait_send()` to wait on `send_sem` which blocks until all data has been sent.

(3) Finally, devices 1, 2, and 3 will call `.wait_recv()` to wait on `recv_sem` until all data has arrived at `dst_ref`.

The above communication pattern can be written as follows:
    
    
    def example_kernel(input_ref, output_ref, send_sem, recv_sem):
        device_id = lax.axis_index('x')
        copy_0_to_1 = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=1,
        )
        copy_2_to_3 = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=3,
        )
        copy_3_to_2 = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=2,
        )
        @pl.when(device_id == 0)
        def _():
          copy_0_to_1.start()
          copy_0_to_1.wait_send()
        @pl.when(device_id == 1)
        def _():
          copy_0_to_1.wait_recv()
        @pl.when(device_id == 2)
        def _():
          copy_2_to_3.start()
          copy_2_to_3.wait_send()
          copy_3_to_2.wait_recv()
        @pl.when(device_id == 3)
        def _():
          copy_3_to_2.start()
          copy_3_to_2.wait_send()
          copy_2_to_3.wait_recv()
    

### pl.when

      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros_like(z_ref)

---

### when

`when`(condition, /) | Calls the decorated function when the condition is met.  

---

### pl.loop

      @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
      def _reg_loop_0(c0):
        @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
        def _reg_loop_1(c1):
          slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
          out_vmem[slc] = in_vmem[slc] + 1

---

### loop

`loop`() | Returns a decorator that calls the decorated function in a loop.  

## Grid, Program IDs, and Kernel Execution

### pl.program_id

#### Ring Communication Pattern
We will write our kernel assuming a ring topology. Rings are a natural fit for TPUs as slicing along any dimension of a torus produces a ring. When writing collectives, we often only need to think about 1D slices of our torus at a time because the different dimensions of the torus are reserved for different types of parallelism (data vs. model, for example).

The strategy we will use is to write a looped kernel, where on each iteration a device receives one slice of the sharded array from its left neighbor, and copies the previously received slice to its right neighbor. After `num_devices` iterations, each device will have a copy of the entire array in its local HBM.

We can re-purpose Pallas’s `grid` argument to implement the loop. Rather than iterating over tiles of an array as we have done in previous tutorials, we instead set the grid to `(num_devices,)` to indicate that we want to loop over the number of devices and use `pl.program_id` to obtain the loop iteration inside of the Pallas kernel. The following code snippet demonstrates how to implement this:

### pl.program_id

      @pl.when(pl.program_id(2) == 0)

---

### pallas.program_id

#### `pallas.program_id` and `pallas.num_programs`
As we’ll soon see, we’ll be executing the same Pallas kernels many times (either in parallel or in a pipeline depending on the backend). These new primitives tell us “where” we are in the execution of the kernel.

`pallas.program_id` takes in an axis argument, which tells us which index in an axis of a multidimensional grid this kernel is currently executing in (analogous to `threadId` from CUDA programming or `lax.axis_index` in `jax.pmap`). Note that we are currently borrowing the “program” terminology from Triton and in the future we might want to change it to something more familiar to JAX users.
    
    
    def f(x_ref, o_ref):
      i = pl.program_id(axis=0)  # execution index in the first axis of the grid
      o_ref[i] = jnp.exp(x_ref[i])
    

`pallas.num_programs` also takes in an axis and returns the grid size for that axis.

Note that while `program_id` and `num_programs` are Triton-specific terminology they are easily generalized to make sense on TPU as well.

---

### program_id

This generalizes to any tuple of integers (a length `d` grid will correspond to `d` nested loops). The kernel is executed as many times as `prod(grid)`. The default grid value `()` results in one kernel invocation. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `jax.experimental.pallas.program_id()`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`. You can also use `jax.experimental.pallas.num_programs()` to get the grid size for a given axis.

### program_id

When we provide a `grid` to `pallas_call`, the kernel is executed as many times as `prod(grid)`. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `program_id(axis=...)`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.

Here’s an example kernel that uses a `grid` and `program_id`.
    
    
    def iota_kernel(o_ref):
      i = pl.program_id(0)
      o_ref[i] = i

### program_id

`program_id`(axis) | Returns the kernel execution position along the given axis of the grid.  

---

### pl.num_programs

      last_iteration = outer_step == pl.num_programs(0) - 1

---

### num_programs

This generalizes to any tuple of integers (a length `d` grid will correspond to `d` nested loops). The kernel is executed as many times as `prod(grid)`. The default grid value `()` results in one kernel invocation. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `jax.experimental.pallas.program_id()`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`. You can also use `jax.experimental.pallas.num_programs()` to get the grid size for a given axis.

### num_programs

`num_programs`(axis) | Returns the size of the grid along the given axis.  

---

### lax.axis_index

### Example: Right Permute (`lax.ppermute`)
Let’s dive into a very basic example. We will implement a kernel that performs a right permutation, where each device sends its slice of the data to its right neighbor.

Suppose we had an array with 512 elements, which we shard into slices of size 128 across 4 devices. Each device will pass its slice to the next device, and the output will consist of the same data, but with the slices rotated by 1. This is identical to the `lax.ppermute` operation where the permutation is set to `(n, (n+1) % 4)`.

In order to call the kernel in distributed mode, we wrap the `pallas_call` in a `shard_map` transformation. From there, we can write the kernel the same way as you would write a normal single-device Pallas kernel, except we now have access to remote DMA instructions. JAX collective primitives such as `lax.axis_index` can be used to obtain a `device_id` that can be used to compute which target devices to copy to, by referencing the same named axes names passed into `shard_map`.
    

---

### grid

To automatically “carve” up the inputs and outputs, you provide a `grid` and `BlockSpec`s to `pallas_call`.

A `grid` is a tuple of integers (e.g. `()`, `(2, 3, 4)`, or `(8,)`) that specifies an iteration space. For example, a grid `(4, 5)` would have 20 elements: `(0, 0), (0, 1), ..., (0, 4), (1, 0), ..., (3, 4)`. We run the kernel function once for each element, a style of single-program multiple-data (SPMD) programming.

A 2D grid

When we provide a `grid` to `pallas_call`, the kernel is executed as many times as `prod(grid)`. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `program_id(axis=...)`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.


---

### dimension_semantics

The basic idea is that if we have embarrassingly parallel dimensions in our computation, we can split up those dimensions across the TensorCores. We can indicate which dimensions are parallelizable by providing an annotation to `pallas_call` called `dimension_semantics`.
    
    
    def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
      # Load x and y from VMEM into VREGs
      x_vregs = x_vmem_ref[:, :]
      y_vregs = y_vmem_ref[:, :]
      # Execute a vectorized add
      z_vregs = x_vregs + y_vregs
      # Store the output values in VREGs back into VMEM
      z_vmem_ref[:, :] = z_vregs
    
    def add_matrices_pipelined_megacore(x: jax.Array, y: jax.Array) -> jax.Array:
      block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
      return pl.pallas_call(
          add_matrices_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[block_spec, block_spec],
          out_specs=block_spec,
          grid=(2,),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",))
      )(x, y)
    
    x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
    add_matrices_pipelined_megacore(x, y)
    
    
    
    Array([[2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           ...,
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
    

`dimension_semantics` should be a tuple of same length as `grid` where each entry is either `"parallel"` or `"arbitrary"`. `"parallel"` indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. `"arbitrary"` indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

## Block and Grid Specifications

### pl.BlockSpec

      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
      scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),)
    )(x)
    
    np.testing.assert_allclose(out, x[0:1] + 1)
    

### Multiple Buffering
Multiple buffering can be specified on a per-argument basis to the pipeline via the `pipeline_mode` option on `pl.BlockSpec`. To do so, pass a `pl.Buffered` object to `pl.BlockSpec` specifying the number of buffers to allocate for this particular argument:
    
    
    pl.BlockSpec(
      pipeline_mode=pl.Buffered(buffer_count=buffer_count)
    )

### pl.BlockSpec

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=(
            # We allocate DMA semaphores in scratch memory.
            [pltpu.SemaphoreType.DMA] * 2
        ),
    )

### pl.BlockSpec

              pl.BlockSpec(memory_space=pl.ANY),
          ],
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pl.ANY),
          ),

### pl.BlockSpec

          in_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          out_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),

---

### BlockSpec

In conjunction with the `grid` argument, we need to provide Pallas the information on how to slice up the input for each invocation. Specifically, we need to provide a mapping between the iteration of the loop to which block of our inputs and outputs to be operated on. This is provided via `jax.experimental.pallas.BlockSpec` objects.

Before we get into the details of `BlockSpec`s, you may want to revisit Block specs by example in Pallas Quickstart.

`BlockSpec`s are provided to `pallas_call` via the `in_specs` and `out_specs`, one for each input and output respectively.

First, we discuss the semantics of `BlockSpec` when `indexing_mode == pl.Blocked()`.

Informally, the `index_map` of the `BlockSpec` takes as arguments the invocation indices (as many as the length of the `grid` tuple), and returns block indices (one block index for each axis of the overall array). Each block index is then multiplied by the corresponding axis size from `block_shape` to get the actual element index on the corresponding array axis.

Note

Not all block shapes are supported.

  * On TPU, only blocks with rank at least 1 are supported. Furthermore, the last two dimensions of your block shape must be equal to the respective dimension of the overall array, or be divisible by 8 and 128 respectively. For blocks of rank 1, the block dimension must be equal to the array dimension, or be a multiple of 1024, or be a power of 2 and at least `128 * (32 / bitwidth(dtype))`.

  * On GPU, when using the Mosaic GPU backend, the size of the blocks is unrestricted. However, due to hardware limitations, the size of the minormost array dimension must by such that it is a multiple of 16 bytes. For example, it must be a multiple of 8 if the input is `jnp.float16`.

  * On GPU, when using the Triton backend, the size of the blocks themselves is unrestricted, but each operation (including a load or store) must operate on arrays whose size is a power of 2.

If the block shape does not divide evenly the overall shape then the last iteration on each axis will still receive references to blocks of `block_shape` but the elements that are out-of-bounds are padded on input and discarded on output. The values of the padding are unspecified, and you should assume they are garbage. In the `interpret=True` mode, we pad with NaN for floating-point values, to give users a chance to spot accessing out-of-bounds elements, but this behavior should not be depended upon. Note that at least one of the elements in each block must be within bounds.

More precisely, the slices for each axis of the input `x` of shape `x_shape` are computed as in the function `slice_for_invocation` below:
    
    
    >>> import jax
    >>> from jax.experimental import pallas as pl
    >>> def slices_for_invocation(x_shape: tuple[int, ...],
    ...                           x_spec: pl.BlockSpec,
    ...                           grid: tuple[int, ...],
    ...                           invocation_indices: tuple[int, ...]) -> tuple[slice, ...]:
    ...   assert len(invocation_indices) == len(grid)
    ...   assert all(0 <= i < grid_size for i, grid_size in zip(invocation_indices, grid))
    ...   block_indices = x_spec.index_map(*invocation_indices)
    ...   assert len(x_shape) == len(x_spec.block_shape) == len(block_indices)
    ...   elem_indices = []
    ...   for x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):
    ...     start_idx = block_idx * block_size
    ...     # At least one element of the block must be within bounds
    ...     assert start_idx < x_size
    ...     elem_indices.append(slice(start_idx, start_idx + block_size))
    ...   return elem_indices
    
    

For example:
    
    
    >>> slices_for_invocation(x_shape=(100, 100),
    ...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
    ...                       grid = (10, 5),
    ...                       invocation_indices = (2, 4))
    [slice(20, 30, None), slice(80, 100, None)]
    
    >>> # Same shape of the array and blocks, but we iterate over each block 4 times
    >>> slices_for_invocation(x_shape=(100, 100),
    ...                       x_spec = pl.BlockSpec((10, 20), lambda i, j, k: (i, j)),
    ...                       grid = (10, 5, 4),
    ...                       invocation_indices = (2, 4, 0))
    [slice(20, 30, None), slice(80, 100, None)]
    
    >>> # An example when the block is partially out-of-bounds in the 2nd axis.
    >>> slices_for_invocation(x_shape=(100, 90),
    ...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
    ...                       grid = (10, 5),
    ...                       invocation_indices = (2, 4))
    [slice(20, 30, None), slice(80, 100, None)]
    
    

The function `show_program_ids` defined below uses Pallas to show the invocation indices. The `iota_2D_kernel` will fill each output block with a decimal number where the first digit represents the invocation index over the first axis, and the second the invocation index over the second axis:
    
    
    >>> def show_program_ids(x_shape, block_shape, grid,
    ...                      index_map=lambda i, j: (i, j)):
    ...   def program_ids_kernel(o_ref):  # Fill the output block with 10*program_id(1) + program_id(0)
    ...     axes = 0
    ...     for axis in range(len(grid)):
    ...       axes += pl.program_id(axis) * 10**(len(grid) - 1 - axis)
    ...     o_ref[...] = jnp.full(o_ref.shape, axes)
    ...   res = pl.pallas_call(program_ids_kernel,
    ...                        out_shape=jax.ShapeDtypeStruct(x_shape, dtype=np.int32),
    ...                        grid=grid,
    ...                        in_specs=[],
    ...                        out_specs=pl.BlockSpec(block_shape, index_map),
    ...                        interpret=True)()
    ...   print(res)
    
    

For example:
    
    
    >>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2),
    ...                  index_map=lambda i, j: (i, j))
    [[ 0  0  0  1  1  1]
     [ 0  0  0  1  1  1]
     [10 10 10 11 11 11]
     [10 10 10 11 11 11]
     [20 20 20 21 21 21]
     [20 20 20 21 21 21]
     [30 30 30 31 31 31]
     [30 30 30 31 31 31]]
    
    >>> # An example with out-of-bounds accesses
    >>> show_program_ids(x_shape=(7, 5), block_shape=(2, 3), grid=(4, 2),
    ...                  index_map=lambda i, j: (i, j))
    [[ 0  0  0  1  1]
     [ 0  0  0  1  1]
     [10 10 10 11 11]
     [10 10 10 11 11]
     [20 20 20 21 21]
     [20 20 20 21 21]
     [30 30 30 31 31]]
    
    >>> # It is allowed for the shape to be smaller than block_shape
    >>> show_program_ids(x_shape=(1, 2), block_shape=(2, 3), grid=(1, 1),
    ...                  index_map=lambda i, j: (i, j))
    [[0 0]]
    
    

When multiple invocations write to the same elements of the output array the result is platform dependent.

In the example below, we have a 3D grid with the last grid dimension not used in the block selection (`index_map=lambda i, j, k: (i, j)`). Hence, we iterate over the same output block 10 times. The output shown below was generated on CPU using `interpret=True` mode, which at the moment executes the invocation sequentially. On TPUs, programs are executed in a combination of parallel and sequential, and this function generates the output shown. See Noteworthy properties and restrictions.
    
    
    >>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2, 10),
    ...                  index_map=lambda i, j, k: (i, j))
    [[  9   9   9  19  19  19]
     [  9   9   9  19  19  19]
     [109 109 109 119 119 119]
     [109 109 109 119 119 119]
     [209 209 209 219 219 219]
     [209 209 209 219 219 219]
     [309 309 309 319 319 319]
     [309 309 309 319 319 319]]
    
    

A `None` value appearing as a dimension value in the `block_shape` behaves as the value `1`, except that the corresponding block axis is squeezed (you could also pass in `pl.Squeezed()` instead of `None`). In the example below, observe that the shape of the `o_ref` is (2,) when the block shape was specified as `(None, 2)` (the leading dimension was squeezed).
    
    
    >>> def kernel(o_ref):
    ...   assert o_ref.shape == (2,)
    ...   o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))
    >>> pl.pallas_call(kernel,
    ...                jax.ShapeDtypeStruct((3, 4), dtype=np.int32),
    ...                out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),
    ...                grid=(3, 2), interpret=True)()
    Array([[ 0,  0, 10, 10],
           [ 1,  1, 11, 11],
           [ 2,  2, 12, 12]], dtype=int32)
    
    

When we construct a `BlockSpec` we can use the value `None` for the `block_shape` parameter, in which case the shape of the overall array is used as `block_shape`. And if we use the value `None` for the `index_map` parameter then a default index map function that returns a tuple of zeros is used: `index_map=lambda *invocation_indices: (0,) * len(block_shape)`.

### BlockSpec

#### Transformation functions
The `in_specs` and `out_specs` arguments to `pallas_call` allow inputs and outputs to be transformed in some way. The two options that Pallas offers right now are an identity transformation (where inputs and outputs are left unchanged), and `BlockSpec`s, take fixed-size slices of `Ref`s determined by the loop index.

A `BlockSpec` takes an `index_map` function and a `block_shape`. Logically, it takes an array and slices it along each axis into `block_shape` sizes blocks. The `index_map` function takes loop indices (from the grid index set) and maps them to block indices. The transform function converts `Ref`s into logical views of the `Ref` at the corresponding block. When we specify `None` in an entry in block_shape, that corresponds to “mapping” over that dimension, removing it from the block within the kernel.
    
    
    class BlockSpec:
      index_map: Callable[[Tuple[Int, ...]], Tuple[Int, ...]]
      block_shape: Tuple[Optional[int], ...]
    
      def transform(self, ref, *loop_indices):
        block_indices = self.transform_function(loop_indices)
        # Returns a view of `ref` starting at `block_indices` of shape self.block_shape
        ...
    

We could also imagine other `Spec`s that are used with `pallas_call`, for example a `Spec` that corresponds to overlapping windows to, say, implement convolutions.

### BlockSpec

To express this pattern, we use `BlockSpec`s. A `BlockSpec` specifies a block shape for each input and output, and an “index map” function, that maps a set of program indices to a block index.

A visualization of a `BlockSpec`

For a concrete example, let’s say we’d like to multiply two `(1024, 1024)` matrices `x` and `y` together to produce `z`, and would like to parallelize the computation 4 ways. We split up `z` into 4 `(512, 512)` blocks where each block is computed with a `(512, 1024) x (1024, 512)` matrix multiplication. To express this, we’d first use a `(2, 2)` grid (one block for each program).

For `x`, we use `BlockSpec((512, 1024), lambda i, j: (i, 0))` – this carves `x` up into “row” blocks. To see this, see how both program instances `(1, 0)` and `(1, 1)` pick the `(1, 0)` block in `x`. For `y`, we use a transposed version `BlockSpec((1024, 512), lambda i, j: (0, j))`. Finally, for `z` we use `BlockSpec((512, 512), lambda i, j: (i, j))`.

These `BlockSpec`s are passed into `pallas_call` via `in_specs` and `out_specs`.

For more detail on `BlockSpec`s see BlockSpec, a.k.a. how to chunk up inputs.

### BlockSpec

          in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),

### BlockSpec

`BlockSpec`([block_shape, index_map, ...]) | Specifies how an array should be sliced for each invocation of a kernel.  

---

### GridSpec

`GridSpec`([grid, in_specs, out_specs, ...]) | Encodes the grid parameters for `jax.experimental.pallas.pallas_call()`.  

---

### pltpu.PrefetchScalarGridSpec

          grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
          ),

### pltpu.PrefetchScalarGridSpec

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=(
            # We allocate DMA semaphores in scratch memory.
            [pltpu.SemaphoreType.DMA] * 2
        ),
    )

---

### PrefetchScalarGridSpec

    class PrefetchScalarGridSpec:
      def __init__(self,
        num_scalar_prefetch: int,
        grid: tuple[int, ...],
        in_specs: PyTree[BlockSpec],
        out_specs: PyTree[BlockSpec],
        scratch_shapes: tuple[MemorySpace, ...]):
          ...

### PrefetchScalarGridSpec

`PrefetchScalarGridSpec`(num_scalar_prefetch) |   

---

### index_map signature with prefetch_refs

    def index_map(*grid_indices, *prefetch_refs):
        ...

---

### mask_index_map pattern

    def mask_index_map(prefetch_map, i, j, ...):
      next_nonzero_block = prefetch_map[i, j]
      return (next_nonzero_block, 0, 0)

## Kernel Entry Points and Invocation

### pl.pallas_call

      send_sem, recv_sem, out = pl.pallas_call(
          functools.partial(ppermute_start_kernel, axis_name=axis_name),
          out_shape=(
              pltpu.SemaphoreType.DMA(()),
              pltpu.SemaphoreType.DMA(()),
              jax.ShapeDtypeStruct(
                  x.shape,
                  dtype=x.dtype,
              ),
          ),
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
          ],
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pl.ANY),
          ),
      )(x)

### pl.pallas_call

    right_permute = pl.pallas_call(
        right_permute_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
    )

---

### pallas_call

### Executing Pallas kernels with `pallas_call`
Now that we’ve written our Pallas kernels (a.k.a. JAX with `Ref`s and the extra Pallas primitives), how do we execute them on a GPU or TPU? We use `pallas_call`, a higher order function (akin to `jax.jit` and `jax.pmap`) that executes the kernel.

The signature of `pallas_call` is as follows:
    
    
    def pallas_call(
        kernel: Callable,
        out_shape: Sequence[jax.ShapeDtypeStruct],
        *,
        in_specs: Sequence[Spec],
        out_specs: Sequence[Spec],
        grid: Optional[Tuple[int, ...]] = None) -> Callable:
      ...
    

When we provide a kernel to `pallas_call` we provide additional information. The first is `out_shape` which tells the kernel what the outputs look like (`pallas_call` will pass a `Ref` corresponding to these into the kernel to be written to). The rest of the information (`in_specs`, `out_specs`, and `grid`) are information about how the kernel will be scheduled on the accelerator.

The (rough) semantics for `pallas_call` are as follows:
    
    
    def pallas_call(kernel, out_shape, *, in_specs, out_specs, grid):
      def execute(*args):
        outputs = map(empty_ref, out_shape)
        grid_indices = map(range, grid)
        for indices in itertools.product(*grid_indices): # Could run in parallel!
          local_inputs = [in_spec.transform(arg, indices) for arg, in_spec in
                          zip(args, in_specs)]
          local_outputs = [out_spec.transform(arg, indices) for arg, out_spec  in
                           zip(outputs, out_specs)]
          kernel(*local_inputs, *local_outputs) # writes to outputs
      return execute
    

Specifically, `pallas_call` will “loop” over grid iteration space, applying a transformation to the inputs and outputs specified via the `in_specs` and `out_specs`. In each iteration, the kernel will be called on the transformed inputs and outputs. Note that the “loop” over the iteration space could be executed in parallel (e.g. on GPU). `pallas_call` also provides no guarantees on the order of loop iterations over the iteration space, just that every member of the iteration space will be looped over. Compilers like Triton and Mosaic will have more specific operational semantics associated with the grid.

### pallas_call

So we’ve written what we call a “kernel”, which we define as a program that will run as an atomic unit of execution on an accelerator, without any interaction with the host. How do we invoke it from a JAX computation? We use the `pallas_call` higher-order function.
    
    
    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x, y)
    add_vectors(jnp.arange(8), jnp.arange(8))
    
    
    
    Array([ 0,  2,  4,  6,  8, 10, 12, 14], dtype=int32)
    

`pallas_call` lifts the Pallas kernel function into an operation that can be called as part of a larger JAX program. But, to do so, it needs a few more details. Here we specify `out_shape`, an object that has a `.shape` and `.dtype` (or a list thereof). `out_shape` determines the shape/dtype of `o_ref` in our `add_vector_kernel`.

`pallas_call` returns a function that takes in and returns `jax.Array`s.

### pallas_call

      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
          in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          grid=(m // bm, n // bn, k // bk),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary")),
      )(x, y)

### pallas_call

When using `jax.experimental.pallas.pallas_call()` the kernel function is executed multiple times on different inputs, as specified via the `grid` argument to `pallas_call`. Conceptually:
    
    
    pl.pallas_call(some_kernel, grid=(n,))(...)
    

maps to
    
    
    for i in range(n):
      some_kernel(...)
    

### pallas_call

`pallas_call`(kernel, out_shape, *[, ...]) | Entry point for creating a Pallas kernel.  

---

### pallas_call with scalar prefetch

    kernel = pl.pallas_call(...)
    result = kernel(*prefetch_args, *input_args)

---

### kernel signature with prefetch_refs

    def kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs):
        ...

---

### pl.core_map

In this guide, we explore using `pl.core_map` to write Pallas kernels. Compared with `pallas_call`, `core_map` offers a few key characteristics:

  * Per-core level programming: You write code for an TPU/GPU core, not for a JAX device. This gives you full control over what runs on every core, or how cores communicate and distribute work among one another.

  * Collectives: `core_map` explicitly models physical cores, so inter-core communication can be expressed safely.

  * Platform generic: `core_map` programming model works for TPU (TensorCore and SparseCore) and GPU with minimal boilerplate changes.

This guide focuses on TPU. For how to use `core_map` on GPU to achieve higher thread flexibility, check out our Pallas GPU `core_map` tutorial.

## Environment setup
Modern accelerators often have multiple cores under a device. For recent TPU chips (v4, v5p), every JAX device may contains 2 TensorCores (aka. a Megacore). Some TPUs (v5p, v6e, 7x) also contain SparseCores, each of which consists of many subcores.

This guide was written on a v5p chip, which contains 4 devices (2 TensorCores each) and 4 SparseCores, each with 16 subcores.
    
    
    from functools import partial
    
    import jax
    from jax.sharding import NamedSharding
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc
    import jax.numpy as jnp
    import numpy as np
    
    
    num_devices = jax.local_device_count()
    assert num_devices > 1, "Please run this notebook with more than one device."
    
    tpu_info = pltpu.get_tpu_info()  # This notebook only runs on TPU.
    print(f"Running on {num_devices} TPU {tpu_info.chip_version} devices.")
    
    
    
    Running on 4 TPU v5p devices.
    

In addition to the typical TPU device mesh, you need to make a mesh of cores. Consider this as an addition dimension called `core`, with length 2, in addition to the 4-device mesh you work with. That is 8 cores in total.
    
    
    # Mesh of devices
    mesh = jax.make_mesh((jax.device_count(),), ('device',))
    print(mesh)
    
    # Mesh of cores, within a JAX device
    tc_mesh = pltpu.create_tensorcore_mesh('core')
    print(tc_mesh)
    
    num_devices = mesh.size
    num_cores = len(tc_mesh.devices)
    print(f"There are {num_devices} devices, and {num_cores} cores each.")
    
    
    
    Mesh('device': 4, axis_types=(Explicit,))
    TensorCoreMesh(devices=array([TensorCore(id=0), TensorCore(id=1)], dtype=object), axis_names=('core',))
    There are 4 devices, and 2 cores each.
    

## A simple per-core kernel
`pl.core_map` allows you to write per-core local code, just as `jax.shard_map` allows you to write per-device code.

In the example kernel below, each core has its own VMEM and semaphore allocations. As with normal kernel, you can initiate copies between HBM and VMEM refs using `pltpu.async_copy`.

Communication between cores

Before communicating between cores, it is good practice to perform a barrier (using `pl.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).
    
    
    # This runs on every core
    def swap_cores_kernel(in_hbm, out_hbm,
                          in_vmem, scratch_vmem, out_vmem,
                          sem, send_sem, recv_sem):
      core_index = jax.lax.axis_index('core')
      num_cores = jax.lax.axis_size('core')
      slc_size = in_hbm.shape[-1] // num_cores
      slc = pl.ds(core_index * slc_size, slc_size)
    
      # Copy in a core-dependent slice of the input
      pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()
    
      # A barrier to make sure all cores have entered run_scoped.
      # You won't need this if not doing inter-core communications.
      dst_core = (core_index + 1) % num_cores
      sem0 = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})
      pl.semaphore_wait(sem0, 1)
    
      # Swap data between core 0 and core 1
      the_copy = pltpu.make_async_remote_copy(
          in_vmem, scratch_vmem, send_sem, recv_sem, device_id={'core': dst_core},
      )
      the_copy.start()
      the_copy.wait()
    
      # Core-local compute
      out_vmem[...] = scratch_vmem[...] * 2
    
      # Copy out the output
      pltpu.async_copy(out_vmem, out_hbm.at[:, slc], sem).wait()
    

Once you have the local kernel:

  * Start your top-level JAX code with HBM refs, and allocate output refs if needed.

  * Use `pl.core_map`, which takes the TensorCore mesh, to start per-core programming.

    * You will need `collective_id` for the barrier semaphore.

  * Inside `pl.core_map`, invoke `pl.run_scoped` to allocate per-core scratch spaces (VMEM and semaphores) and run the local kernel.

---

### core_map

`core_map`(mesh, *[, compiler_params, ...]) | Runs a function on a mesh, mapping it over the devices in the mesh.  

---

### pl.kernel

You can use the `pl.kernel` decorator to wrap boilerplate such as `core_map`, `run_scoped`, and output buffer allocation.

Note that this should run inside any `jax.shard_map` you may have at the top level.
    
    
    @jax.jit
    @partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
    def swap_cores(x):
      scratch_shapes = [pltpu.VMEM(local_vmem_shape, x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 3
      return pl.kernel(swap_cores_kernel, out_shape=x, mesh=tc_mesh,
                       scratch_shapes=scratch_shapes,
                       compiler_params=pltpu.CompilerParams(collective_id=0))(x)

---

### pl.run_scoped

  * Inside `pl.core_map`, invoke `pl.run_scoped` to allocate per-core scratch spaces (VMEM and semaphores) and run the local kernel.

    
    
    input_shape = (32, 256)
    local_vmem_shape = (32 // num_devices, 256 // num_cores)
    in_spec = jax.P('device', None)
    sharding = NamedSharding(mesh, in_spec)
    
    @jax.jit
    @partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec,
             check_vma=False)
    def swap_cores(x):
      # Get buffers out of the input and output
      x_hbm_ref = jax.new_ref(x)
      o_hbm_ref = jax.new_ref(jax.lax.empty(x.shape, x.dtype))
    
      @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))
      def _():
        pl.run_scoped(
            partial(swap_cores_kernel, x_hbm_ref, o_hbm_ref),
            *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM allocations
            *([pltpu.SemaphoreType.DMA] * 3),                # semaphores
        )

### pl.run_scoped

        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped,
                           second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
          for neighbor in [left_neighbor, right_neighbor]:
            pl.semaphore_signal(
              second_barrier,
              inc=1,
              device_id=(neighbor,),
              device_id_type=pl.DeviceIdType.MESH,
            )
          pl.semaphore_wait(second_barrier, 2)

---

### run_scoped

`run_scoped`(f, *types[, collective_axes]) | Calls the function with allocated references and returns the result.  

---

### with_scoped

`with_scoped`(*types[, collective_axes]) | Returns a function decorator that runs a function with provided allocations.  

---

### input_output_aliases

          input_output_aliases={0:0}

---

### run_on_first_core

`run_on_first_core`(core_axis_name) | Runs a function on the first core in a given axis.  

## Pipelining

### pltpu.emit_pipeline

### Nested Remote and Local DMA Pipelines
A limitation of the previous all-reduce and reduce-scatter kernels that we wrote is that the blocks we copy via remote DMA must be small enough to fit in our working VMEM that we use for accumulation. For some kernels it may be advantageous to use larger block sizes to better utilize the TPU. For example, a matrix multiplication requires on the order of \\(O(N^3)\\) compute operations, but only \\(O(N^2)\\) memory transfers. Therefore, we want each block of work transferred between devices to be large enough such that the operation becomes compute bound and we can hide the communication cost using pipelining. For reference, the VMEM of a TPU (for generations v4/v5) is typically on the order of 10-100MB, whereas HBM ranges from 10-100GB.

To address this problem, we need to be able to write an “inner kernel” that handles local HBM-VMEM pipelining inside of the “outer kernel” that handles pipelining larger HBM-HBM transfers between devices. Pallas offers an API for constructing nested pipelines using the `emit_pipeline` function. See the TPU pipelining guide for a general overview on `emit_pipeline`. Because our outer kernel only involves remote HBM-HBM transfers, we are not using any of the built-in pipelining that `pallas_call` provides for HBM-VMEM transfers. The following code skeleton demonstrates what a typical program structure would look like using this pattern:
    
    
    def outer_kernel(...):
      # ... do work to pipeline remote HBM-HBM transfers (outer kernel)
    
      def inner_kernel(...):
        # ... do work (inner kernel)
      pltpu.emit_pipeline(
              inner_kernel,
              grid=inner_grid,
              in_specs=...,
              out_specs=...,
      )(inner_kernel_args)
      # ... do more work (outer kernel)
    
    pl.pallas_call(
      outer_kernel,
      grid=outer_grid,
      in_specs=...
      out_specs=...
      scratch=inner_kernel_allocs
    )
    

### pltpu.emit_pipeline

Note that the kernel above only does simple copies and compute, without automatic pipelining via Pallas `grid` and `BlockSpec`. To do pipelining inside `core_map`, use `pltpu.emit_pipeline` inside the core-local kernel.

Automatically parallelize work amongst cores

The simple way is to annotate a block axis as `pltpu.PARALLEL`, and Pallas will automatically parallelize work along this axis. Both `pl.pallas_call` and `pltpu.emit_pipeline` supports this, via arguments `core_axis` and `dimension_semantics`. The `pallas_call` example is in another guide, and the `emit_pipeline` case is shown below.

When the `PARALLEL` annotation is provided, the corresponding grid dimension will be logically split and executed on separate cores. (The exact semantics of which grid dimensions are executed on which core is guaranteed).

Scratch shapes allocation

Note that in the example below, the top level `pl.run_scoped` (wrapped inside `kernel`) did not allocate any VMEM scratch buffers. Instead, `pltpu.emit_pipeline` allocates its own scratch buffers in VMEM and use them for its multiple buffering.
    
    
    def add_one_body(in_vmem, out_vmem):
      out_vmem[...] = in_vmem[...] + 1
    
    input_shape = (1024, 1024)
    in_spec = jax.P('device', None)
    
    def add_one_kernel(x_hbm_ref, o_hbm_ref):
      in_shape = x_hbm_ref.shape
      pltpu.emit_pipeline(
          add_one_body,
          grid=(in_shape[0] // 8, in_shape[1] // 128),
          in_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          out_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          core_axis_name='core',
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
      )(x_hbm_ref, o_hbm_ref)

### pltpu.emit_pipeline

### pltpu.emit_pipeline
`pltpu.emit_pipeline` is a pipelining API implemented in Pallas that allows you to construct pipelines inside of a kernel rather than only on kernel entry. This several use-cases over using `pl.pallas_call`, such as:

  * For constructing nested pipelines. For example, an outer pipeline that communicates between chips, and an inner pipeline that performs HBM-VMEM pipelining.

  * For using `emit_pipeline` specific features such as lookahead prefetch and dynamic block shapes (covered below).

`pltpu.emit_pipeline` follows a similar signature to `pl.pallas_call` and requires you to specify a body `kernel`, a grid, and block specs for inputs and outputs:
    
    
    def emit_pipeline(
        kernel: Callable,
        grid: tuple[int],
        in_specs: PyTree[BlockSpec] = None,
        out_specs: PyTree[BlockSpec] = None,
        dimension_semantics: tuple[GridDimensionSemantics] = None,
        core_axis: int | None = None,
    ) -> Callable:
      ... # Returns a custom pipeline given an inner kernel and BlockSpecs.
    

The `dimension_semantics` and `core_axis` arguments are used for partitioning the kernel grid over Megacore (see below).

---

### emit_pipeline

`emit_pipeline`(body, *, grid[, in_specs, ...]) | Creates a function to emit a manual pallas pipeline.  

---

### emit_pipeline_with_allocations

`emit_pipeline_with_allocations`(body, *, grid) | Creates pallas pipeline and top-level allocation preparation functions.  

## Compiler Parameters and Hardware Info

### pltpu.CompilerParams

          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",))
      )(x, y)
    
    x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
    add_matrices_pipelined_megacore(x, y)
    
    
    
    Array([[2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           ...,
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.],
           [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
    

`dimension_semantics` should be a tuple of same length as `grid` where each entry is either `"parallel"` or `"arbitrary"`. `"parallel"` indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. `"arbitrary"` indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

By specifying `dimension_semantics`, we now execute the kernel simultaneously on each TensorCore. Pallas will handle splitting up the grid automatically.

> Note that Megacore is only currently available on TPU `v4` and TPU `v5p`. Supplying `dimension_semantics` annotations is a no-op on other platforms, but not specifying it will result in only one TensorCore being used (even if there are more than one available).

When using `pltpu.emit_pipeline`, `core_axis` should be passed into `emit_pipeline`. `core_axis` should be the index of a parallel grid axis to partition the grid on. For example, the following template can be used to partition the kernel over a leading parallel grid dimension:
    
    
    def kernel_body(...):
      def inner_pipeline_body(...):
        ...
      pltpu.emit_pipeline(inner_pipeline_body,
                          grid=(4, 4), 
                          core_axis=0,
                          dimension_semantics=("parallel", "sequential"))
    
    pl.pallas_call(
          kernel_body,
          grid=(num_cores,),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",))
      )

### pltpu.CompilerParams

If you are doing indexed retrieval at the beginning of a kernel, you could use the `indexed_by` and `indexed_dim` argument of `plsc.BlockSpec` on the top-level `pl.pallas_call` to refer to another input as the indices of this input on this axis.

This call will parallelize the DMA from HBM to VMEM and the gather operation that does the indexed lookup, resulting in 4 pipeline stages: indices copy-in, gather, kernel computation and output copy-out. This allows you to overlap gather and any further computation on gathered outputs.

Note that the `plsc.BlockSpec` is experimental and subject to change.
    
    
    @jax.jit
    def gather_add_one(x, indices):
      @partial(
          pl.pallas_call,
          out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype),
          grid=(num_indices // gather_window_size,),
          in_specs=(
              plsc.BlockSpec(
                  (gather_window_size, value_dim), indexed_by=1, indexed_dim=0
              ),
              pl.BlockSpec((gather_window_size,), lambda i: i),
          ),
          out_specs=pl.BlockSpec((gather_window_size, value_dim), lambda i: (i, 0)),
          compiler_params=pltpu.CompilerParams(
              kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
              dimension_semantics=(pltpu.PARALLEL,),
          ),
      )
      def kernel(gathered_ref, _, o_ref):
        # gathered_ref is the gathered content of x[indices]
        @pl.loop(0, gather_window_size)
        def _(c0):
          @pl.loop(0, o_ref.shape[1], step=16)
          def _(c1):
            slc = (pl.ds(c0, 1), pl.ds(c1, 16))
            o_ref.at[*slc][...] = gathered_ref.at[*slc][...] + 1
    
      return kernel(x, indices)
    
    
    out = gather_add_one(x, indices)
    np.testing.assert_array_equal(out, jnp.take(x, indices, axis=0) + 1)
    

### pltpu.CompilerParams

When using barrier semaphores, the `collective_id` compiler parameter must be passed to `pallas_call` to specify which barrier semaphore is being used. A TPU has a small, fixed number of barrier semaphores available (typically on the order of 20-30) and therefore they should be used sparingly. In order to ensure correctness, only kernels that share the same communication pattern should use the same `collective_id`. For example, if two kernels synchronize only with neighbors on the same mesh axis, they are allowed to share the same `collective_id`. However, if two kernels synchronize along different axes, they must have different `collective_id`s. Failure to do so may result in race conditions that are difficult to debug.
    
    
    kernel = pl.pallas_call(
          example_kernel,
          ...,
          compiler_params=pltpu.CompilerParams(collective_id=0),
    )

### pltpu.CompilerParams

          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary")),

### pltpu.CompilerParams

      @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))

---

### CompilerParams

`CompilerParams`([dimension_semantics, ...]) | Mosaic TPU compiler parameters.  

---

### pltpu.PARALLEL

## Pipelining in SparseCore kernels
You can `pltpu.emit_pipeline` to write pipelined SparseCore kernels. The `core_axis_name` and `dimension_semantics` arguments to `emit_pipeline` enable partitioning the pipeline across SparseCores/subcores.
    
    
    SC_REG_OP_SHAPE = (1, sc_info.num_lanes)
    dma_block = (8, 128)
    
    
    @jax.jit
    def sc_add_one(x):
      @pl.kernel(out_shape=x, mesh=vector_mesh, scratch_shapes=[])
      def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
        in_shape = x_hbm_ref.shape
    
        def sc_add_one_body(in_vmem, out_vmem):
          @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
          def _(c0):
            @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
            def _(c1):
              slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
              out_vmem.at[*slc][...] = in_vmem.at[*slc][...] + 1
    
        pltpu.emit_pipeline(
            sc_add_one_body,
            grid=(in_shape[0] // dma_block[0], in_shape[1] // dma_block[1]),
            in_specs=[
                pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))
            ],
            out_specs=[
                pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))
            ],
            core_axis_name=('core', 'subcore'),
            dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
        )(x_hbm_ref, o_hbm_ref)
    
      return sc_add_one_kernel(x)
    
    
    x = jax.random.randint(jax.random.key(0), (4096, 128), 0, 64, jnp.int32)
    y = sc_add_one(x)
    np.testing.assert_array_equal(y, x + 1)
    

### pltpu.PARALLEL

The simple way is to annotate a block axis as `pltpu.PARALLEL`, and Pallas will automatically parallelize work along this axis. Both `pl.pallas_call` and `pltpu.emit_pipeline` supports this, via arguments `core_axis` and `dimension_semantics`. The `pallas_call` example is in another guide, and the `emit_pipeline` case is shown below.

When the `PARALLEL` annotation is provided, the corresponding grid dimension will be logically split and executed on separate cores. (The exact semantics of which grid dimensions are executed on which core is guaranteed).

Scratch shapes allocation

Note that in the example below, the top level `pl.run_scoped` (wrapped inside `kernel`) did not allocate any VMEM scratch buffers. Instead, `pltpu.emit_pipeline` allocates its own scratch buffers in VMEM and use them for its multiple buffering.
    
    
    def add_one_body(in_vmem, out_vmem):
      out_vmem[...] = in_vmem[...] + 1
    
    input_shape = (1024, 1024)
    in_spec = jax.P('device', None)
    
    def add_one_kernel(x_hbm_ref, o_hbm_ref):
      in_shape = x_hbm_ref.shape
      pltpu.emit_pipeline(
          add_one_body,
          grid=(in_shape[0] // 8, in_shape[1] // 128),
          in_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          out_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          core_axis_name='core',
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),

---

### GridDimensionSemantics

`GridDimensionSemantics`(value[, names, ...]) |   

---

### ChipVersion

`ChipVersion`(value[, names, module, ...]) | TPU chip version.  

---

### TpuInfo

`TpuInfo`(*, chip_version, generation, ...[, ...]) | TPU hardware information.  

---

### get_tpu_info

`get_tpu_info`() | Returns the TPU hardware info for the current device.  

## Debugging

### debug_print

`debug_print`(fmt, *args) | Prints values from inside a Pallas kernel.  

## Other

### kernel

`kernel`([body, out_shape, scratch_shapes, ...]) | Entry point for creating a Pallas kernel.  