## API Reference

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

### Block specs by example
With `grid` and `program_id` in mind, Pallas provides an abstraction that takes care of some common indexing patterns seen in a lot of kernels. To build intuition, let’s try to implement a matrix multiplication.

A simple strategy for implementing a matrix multiplication in Pallas is to implement it recursively. We know our underlying hardware has support for small matrix multiplications (using GPU and TPU tensorcores), so we just express a big matrix multiplication in terms of smaller ones.

Suppose we have input matrices \\(X\\) and \\(Y\\) and are computing \\(Z = XY\\). We first express \\(X\\) and \\(Y\\) as block matrices. \\(X\\) will have “row” blocks and \\(Y\\) will have “column” blocks.

\\[\begin{split} \begin{align*} X = \begin{bmatrix} X_0 \\\ X_1 \end{bmatrix} \end{align*} \end{split}\\]

\\[ \begin{align*} Y = \begin{bmatrix} Y_0 & Y_1 \end{bmatrix} \end{align*} \\]

\\[\begin{split} \begin{align*} Z &= \begin{bmatrix} X_0 \\\ X_1 \end{bmatrix} \begin{matrix} \begin{bmatrix} Y_0 & Y_1 \end{bmatrix} \\\ ~ \end{matrix} \\\ &= \begin{bmatrix} X_0 Y_0 & X_0 Y_1 \\\ X_1 Y_0 & X_1 Y_1 \end{bmatrix} \end{align*} \end{split}\\]

Our strategy is that because \\(Z\\) is also a block matrix, we can assign each of the programs in our Pallas kernel one of the output blocks. Computing each output block corresponds to doing a smaller matrix multiply between a “row” block of \\(X\\) and a “column” block of \\(Y\\).

To express this pattern, we use `BlockSpec`s. A `BlockSpec` specifies a block shape for each input and output, and an “index map” function, that maps a set of program indices to a block index.

A visualization of a `BlockSpec`

For a concrete example, let’s say we’d like to multiply two `(1024, 1024)` matrices `x` and `y` together to produce `z`, and would like to parallelize the computation 4 ways. We split up `z` into 4 `(512, 512)` blocks where each block is computed with a `(512, 1024) x (1024, 512)` matrix multiplication. To express this, we’d first use a `(2, 2)` grid (one block for each program).

For `x`, we use `BlockSpec((512, 1024), lambda i, j: (i, 0))` – this carves `x` up into “row” blocks. To see this, see how both program instances `(1, 0)` and `(1, 1)` pick the `(1, 0)` block in `x`. For `y`, we use a transposed version `BlockSpec((1024, 512), lambda i, j: (0, j))`. Finally, for `z` we use `BlockSpec((512, 512), lambda i, j: (i, j))`.

These `BlockSpec`s are passed into `pallas_call` via `in_specs` and `out_specs`.

For more detail on `BlockSpec`s see BlockSpec, a.k.a. how to chunk up inputs.

Underneath the hood, `pallas_call` will automatically carve up your inputs and outputs into `Ref`s for each block that will be passed into the kernel.
    
    
    def matmul_kernel(x_ref, y_ref, z_ref):
      z_ref[...] = x_ref[...] @ y_ref[...]
    
    def matmul(x: jax.Array, y: jax.Array):
      return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec(
            (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j),
        )
      )(x, y)

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

          in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),

---

### pl.when

      @pl.when(is_start | changed_blocks)
      def _():
        accum_scratch[...] = jnp.zeros_like(accum_scratch)

### pl.when

      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros_like(z_ref)

---

### pl.program_id

      @pl.when(pl.program_id(2) == 0)

### pl.program_id

      blk_idx = pl.program_id(1)

---

### CompilerParams

          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary")),

### CompilerParams

`CompilerParams`([dimension_semantics, ...]) | Mosaic TPU compiler parameters.  

---

### PrefetchScalarGridSpec

class jax.experimental.pallas.tpu.PrefetchScalarGridSpec(num_scalar_prefetch: 'int', grid: 'pallas_core.Grid' = (), in_specs: 'pallas_core.BlockSpecTree' = NoBlockSpec, out_specs: 'pallas_core.BlockSpecTree' = NoBlockSpec, scratch_shapes: 'pallas_core.ScratchShapeTree' = ())
    

Parameters:
    

  * num_scalar_prefetch (int)

  * grid (TupleGrid)

  * in_specs (BlockSpecTree)

  * out_specs (BlockSpecTree)

  * scratch_shapes (ScratchShapeTree)

__init__(num_scalar_prefetch, grid=(), in_specs=NoBlockSpec, out_specs=NoBlockSpec, scratch_shapes=())
    

Parameters:
    

  * num_scalar_prefetch (int)

  * grid (pallas_core.Grid)

  * in_specs (pallas_core.BlockSpecTree)

  * out_specs (pallas_core.BlockSpecTree)

  * scratch_shapes (pallas_core.ScratchShapeTree)

### PrefetchScalarGridSpec

﻿jax.experimental.pallas.tpu.PrefetchScalarGridSpec ================================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: PrefetchScalarGridSpec .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~PrefetchScalarGridSpec.__init__ .. rubric:: Attributes .. autosummary:: ~PrefetchScalarGridSpec.scratch_shapes ~PrefetchScalarGridSpec.num_scalar_prefetch ~PrefetchScalarGridSpec.grid ~PrefetchScalarGridSpec.grid_names ~PrefetchScalarGridSpec.in_specs ~PrefetchScalarGridSpec.out_specs

### PrefetchScalarGridSpec

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

### pltpu.VMEM

    * In Pallas, the VMEM spaces are denoted as `pltpu.VMEM` and `pltpu.VMEM_SHARED`, and SMEM is denoted as `pltpu.SMEM`.

    * In other documentation, the shared VMEM is often called “SPMEM” and the per-subcore VMEM is called “TileSPMEM” or “local SPMEM”.

### pltpu.VMEM

`pltpu.VMEM` | VMEM | SRAM  
`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

### pltpu.VMEM

            *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM allocations

### pltpu.VMEM

        scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)]

### pltpu.VMEM

            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],

---

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

          in_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),
          )],
          out_specs=[pl.BlockSpec(
              block_shape=(8, 128), index_map=lambda i, j: (i, j),

### pl.BlockSpec

          in_specs=[pl.BlockSpec(
              sizes,
              lambda i, j, block_idx: (block_idx[0], block_idx[1]))],
          out_specs=pl.BlockSpec(sizes, lambda *_: (0, 0)),

---

### pl.pallas_call

      kernel = pl.pallas_call(
        dynamic_slice_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
      )

---

### pl.num_programs

        @pl.when(k == pl.num_programs(2) - 1)

---

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

---

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

Before communicating between cores, it is good practice to perform a barrier (using `pltpu.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

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
      pltpu.semaphore_signal(sem0, 1, device_id={'core': dst_core})
      pltpu.semaphore_wait(sem0, 1)
    
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

---

### pltpu.CompilerParams

          compiler_params=pltpu.CompilerParams(
              kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
              dimension_semantics=(pltpu.PARALLEL,),
          ),

### pltpu.CompilerParams

          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",))

### pltpu.CompilerParams

      @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))

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

### pltpu.PARALLEL

            dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
        )(x_hbm_ref, o_hbm_ref)

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

### pltpu.SMEM

`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

### pltpu.SMEM

    * In Pallas, the VMEM spaces are denoted as `pltpu.VMEM` and `pltpu.VMEM_SHARED`, and SMEM is denoted as `pltpu.SMEM`.

    * In other documentation, the shared VMEM is often called “SPMEM” and the per-subcore VMEM is called “TileSPMEM” or “local SPMEM”.

### pltpu.SMEM

                       scratch_shapes=[pltpu.SMEM((1,), jnp.int32)])((x, index))

---

### pl.loop

      @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
      def _reg_loop_0(c0):
        @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
        def _reg_loop_1(c1):
          slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
          out_vmem[slc] = in_vmem[slc] + 1

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
          )],
          out_specs=[pl.BlockSpec(
              block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,

---

### pl.multiple_of

          pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8), j)

---

### program_id

This generalizes to any tuple of integers (a length `d` grid will correspond to `d` nested loops). The kernel is executed as many times as `prod(grid)`. The default grid value `()` results in one kernel invocation. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `jax.experimental.pallas.program_id()`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`. You can also use `jax.experimental.pallas.num_programs()` to get the grid size for a given axis.

### program_id

When we provide a `grid` to `pallas_call`, the kernel is executed as many times as `prod(grid)`. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `program_id(axis=...)`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.

Here’s an example kernel that uses a `grid` and `program_id`.
    
    
    def iota_kernel(o_ref):
      i = pl.program_id(0)
      o_ref[i] = i
    

---

### grid

### Grids by example
To automatically “carve” up the inputs and outputs, you provide a `grid` and `BlockSpec`s to `pallas_call`.

A `grid` is a tuple of integers (e.g. `()`, `(2, 3, 4)`, or `(8,)`) that specifies an iteration space. For example, a grid `(4, 5)` would have 20 elements: `(0, 0), (0, 1), ..., (0, 4), (1, 0), ..., (3, 4)`. We run the kernel function once for each element, a style of single-program multiple-data (SPMD) programming.

A 2D grid

When we provide a `grid` to `pallas_call`, the kernel is executed as many times as `prod(grid)`. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `program_id(axis=...)`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.

Here’s an example kernel that uses a `grid` and `program_id`.
    
    
    def iota_kernel(o_ref):
      i = pl.program_id(0)
      o_ref[i] = i
    

We now execute it using `pallas_call` with an additional `grid` argument. On GPUs, we can call the kernel directly like so:
    
    
    # GPU version
    def iota(size: int):
      return pl.pallas_call(iota_kernel,
                            out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                            grid=(size,))()
    iota(8)
    
    
    
    Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
    

TPUs distinguish between vector and scalar memory spaces and in this case the output must be placed in scalar memory (`MemorySpace.SMEM`) since `i` is a scalar. For more details read TPU and its memory spaces. To call the above kernel on TPU, run:
    
    
    # TPU version
    from jax.experimental.pallas import tpu as pltpu
    
    def iota(size: int):
      return pl.pallas_call(iota_kernel,
                            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                            out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                            grid=(size,))()
    iota(8)
    

### Grid semantics
On GPUs, each program is executed in parallel on separate threads. Thus, we need to think about race conditions on writes to HBM. A reasonable approach is to write our kernels in such a way that different programs write to disjoint locations in HBM to avoid these parallel writes. On the other hand, parallelizing the computation is how we can execute operations like matrix multiplications really quickly.

In contrast, TPUs operate like a very wide SIMD machine. Some TPU models contain multiple cores, but in many cases a TPU can be treated as a single-threaded processor. The grid on a TPU can be specified in a combination of parallel and sequential dimensions, where sequential dimensions are guaranteed to run serially.

You can read more details at grid, a.k.a. kernels in a loop and Noteworthy properties and restrictions.

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
    

### MemorySpace.SMEM

﻿jax.experimental.pallas.tpu.MemorySpace ======================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: MemorySpace .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~MemorySpace.from_type .. rubric:: Attributes .. autosummary:: ~MemorySpace.VMEM ~MemorySpace.VMEM_SHARED ~MemorySpace.SMEM ~MemorySpace.CMEM ~MemorySpace.SEMAPHORE ~MemorySpace.HBM ~MemorySpace.HOST

---

### pltpu.async_copy()

## A basic SparseCore kernel
See below for a simple scalar subcore kernel that includes DMAs, per-core customizing and compute operations. Note that the scalar subcore can only do scalar operations.
    
    
    @jax.jit
    def cumsum(x):
      @pl.kernel(
          out_shape=x,
          mesh=scalar_mesh,
          scratch_shapes=[
              pltpu.SMEM((x.shape[1],), x.dtype),
              pltpu.SemaphoreType.DMA,
          ],
      )
      def kernel(x_ref, o_ref, tmp_ref, sem):
        idx = jax.lax.axis_index('core')
        pltpu.async_copy(x_ref.at[idx], tmp_ref, sem).wait()
    
        @pl.loop(1, x.shape[1])
        def _(i):
          tmp_ref[i] += tmp_ref[i - 1]
    
        pltpu.async_copy(tmp_ref, o_ref.at[idx], sem).wait()
    
      return kernel(x)
    
    
    x_shape = (sc_info.num_cores, sc_info.num_lanes)
    x = jax.random.randint(jax.random.key(0), x_shape, 0, 64, jnp.int32)
    np.testing.assert_array_equal(cumsum(x), jnp.cumsum(x, axis=1))
    

---

### pltpu.emit_pipeline()

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
    

---

### pltpu.sync_copy()

## Gather and scatter
SparseCore has specific optimized ops for indexed retrievals and updates. Given an input or output ref in HBM (named `data`) and an array of indices in VMEM (named `indices`), it can quickly read from (“gather”) or write to (“scatter”) `data[indices]`.

We can use these gather/scatter by indexing a Ref with an indices Ref as part of an `async_copy` or `sync_copy`. For example, `sync_copy(data_ref.at[indices_ref], target_ref)` will trigger a gather.

Below is a kernel that pipelines loading indices into a vector subcore’s VMEM. In the body, we execute a gather using those indices.
    
    
    batch_size = 4096
    value_dim = 128
    gather_window_size = 128
    num_steps = 1024
    sc_num_cores, sc_num_subcores = sc_info.num_cores, sc_info.num_subcores
    num_indices = gather_window_size * sc_num_cores * sc_num_subcores * num_steps
    x = jnp.arange(batch_size * value_dim).reshape(batch_size, value_dim)
    indices = jax.random.randint(
        jax.random.key(0), (num_indices,), 0, batch_size, jnp.int32
    )
    
    
    @jax.jit
    def gather(x, indices):
      indices = indices.reshape((1, num_indices))
    
      @pl.kernel(
          out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype),
          mesh=vector_mesh,
      )
      def kernel(x_hbm, i_hbm, o_hbm):
        def body(i_vmem, o_vmem):
          pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)  # The gather op
    
        pltpu.emit_pipeline(
            body,
            grid=(num_indices // gather_window_size,),
            in_specs=[
                pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i))
            ],
            out_specs=[
                pl.BlockSpec(
                    (gather_window_size, value_dim), index_map=lambda i: (i, 0)
                )
            ],
            core_axis_name='subcore',
            dimension_semantics=(pltpu.PARALLEL,),
        )(i_hbm, o_hbm)
    
      return kernel(x, indices)
    
    
    out = gather(x, indices)
    np.testing.assert_array_equal(out, jnp.take(x, indices, axis=0))
    

---

### pl.ANY

`pl.ANY` | HBM (usually) or VMEM | DRAM  
`pltpu.VMEM` | VMEM | SRAM  
`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.


---

### pltpu.SEMAPHORE

`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.

  * `MemorySpace.SEMAPHORE` is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

---

### pl.Buffered

Multiple buffering can be specified on a per-argument basis to the pipeline via the `pipeline_mode` option on `pl.BlockSpec`. To do so, pass a `pl.Buffered` object to `pl.BlockSpec` specifying the number of buffers to allocate for this particular argument:
    
    
    pl.BlockSpec(
      pipeline_mode=pl.Buffered(buffer_count=buffer_count)
    )
    

The default buffer count is 2 for all inputs and outputs.

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

### Lookahead Prefetch
Lookahead prefetch is a pipelining feature where the pipeline will attempt to prefetch the next input block as soon as a buffering slot is available, rather than the iteration directly before it would be used. For example, if the kernel had a grid of `(8,)` and the block indices to fetch on each iteration were `0, 0, 0, 0, 1, 1, 1, 1`, then lookahead prefetch will begin fetching both blocks `0` and `1` on iteration 0, whereas the standard pipeline schedule would fetch block `0` on iteration 0 but not begin fetching block `1` until iteration 3. There is a small amount of control flow overhead in performing lookahead so it is disabled by default.

Lookahead is primarily useful when there is a variable amount of compute work in each block, such as when some blocks contain skipped or a reduced amount of work. In these cases, there may not be enough compute work in the iteration immediately preceding the step when the block is needed to fully overlap with the memory transfer. Therefore, we would like to begin fetching blocks earlier in the pipeline.

Lookahead prefetch can be used in conjunction with multiple buffering and can likewise be enabled by passing `pl.Buffered` into the `pipeline_mode` argument:
    
    
    pl.BlockSpec(
      pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
    )
    

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

---

### Blocked

The behavior documented above applies to the default “blocked” indexing mode. When integers are used in the `block_shape` tuple e.g. `(4, 8)`, it is equivalent to passing in a `pl.Blocked(block_size)` object instead, e.g. `(pl.Blocked(4), pl.Blocked(8))`. Blocked indexing mode means the indices returned by `index_map` are block indices. We can pass in objects other than `pl.Blocked` to change the semantics of `index_map`, most notably, `pl.Element(block_size)`.. When using the `pl.Element` indexing mode the values returned by the index map function are used directly as the array indices, without first scaling them by the block size. When using the `pl.Element` mode you can specify virtual padding of the array as a tuple of low-high paddings for the dimension: the behavior is as if the overall array is padded on input. No guarantees are made for the padding values in element mode, similarly to the padding values for the blocked indexing mode when the block shape does not divide the overall array shape.

---

### jax.random.bits()

  * `jax.random.bits()`

---

### jax.random.uniform()

  * `jax.random.uniform()`

---

### jax.random.bernoulli()

  * `jax.random.bernoulli()`

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

### jax.random.wrap_key_data()

  * `jax.random.wrap_key_data()`

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

### BufferedRef

# jax.experimental.pallas.tpu.BufferedRef
class jax.experimental.pallas.tpu.BufferedRef(_spec, _buffer_type, window_ref, accum_ref, copy_in_slot, wait_in_slot, copy_out_slot, wait_out_slot, _copy_in_slot_reg, _wait_in_slot_reg, _copy_out_slot_reg, _wait_out_slot_reg, next_fetch_smem, next_fetch_sreg, sem_recvs, sem_sends, swap, tiling)
    

A helper class to automate VMEM double buffering in pallas pipelines.

Parameters:
    

  * _spec (pl.BlockSpec)

  * _buffer_type (BufferType)

  * window_ref (ArrayRef | None)

  * accum_ref (ArrayRef | None)

  * copy_in_slot (ArrayRef | None)

  * wait_in_slot (ArrayRef | None)

  * copy_out_slot (ArrayRef | None)

  * wait_out_slot (ArrayRef | None)

  * _copy_in_slot_reg (int | jax.Array | None)

  * _wait_in_slot_reg (int | jax.Array | None)

  * _copy_out_slot_reg (int | jax.Array | None)

  * _wait_out_slot_reg (int | jax.Array | None)

  * next_fetch_smem (Sequence[jax.Array] | None)

  * next_fetch_sreg (Sequence[jax.Array] | None)

  * sem_recvs (SemaphoreTuple | None)

  * sem_sends (SemaphoreTuple | None)

  * swap (ArrayRef | None)

  * tiling (tpu_info.Tiling | None)

spec
    

pallas blockspec.

buffer_type
    

enum indicating whether this is an input, output, or in/out accumulator buffered reference.

window_ref#
    

a multiple-buffer to hold the working and dirty buffers used to copy into and out of. In the case of a BufferedRef targeting a VMEM reference, this simply points to the existing ref.

Type:
    

ArrayRef | None

accum_ref#
    

accumulating buffer used by accumulator BufferedRefs.

Type:
    

ArrayRef | None

copy_in_slot#
    

current slot to copy in for the working buffer.

Type:
    

ArrayRef | None

copy_out_slot#
    

current slot to copy out for the working buffer.

Type:
    

ArrayRef | None

wait_in_slot#
    

current slot to wait in for the working buffer.

Type:
    

ArrayRef | None

wait_out_slot#
    

current slot to wait out for the working buffer.

Type:
    

ArrayRef | None

next_fetch_smem#
    

Holds the next grid indices to fetch for lookahead. This is the SMEM backing buffer used to persist state between pipeline invocations.

Type:
    

Sequence[jax.Array] | None

next_fetch_sreg#
    

Holds the next grid indices to fetch for lookahead. This is the register state used to track the indices within the pipeline loop.

Type:
    

Sequence[jax.Array] | None

sem_recvs#
    

Multiple buffered semaphores for input DMAs.

Type:
    

SemaphoreTuple | None

sem_sends#
    

Multiple buffered semaphores for output DMAs.

Type:
    

SemaphoreTuple | None

block_shape
    

passthrough property for the BlockSpec’s block_shape.

compute_index
    

passthrough property for the BlockSpec’s compute_index.

memory_space#
    

passthrough property for the BlockSpec’s memory_space.

current_ref
    

points to the current working slice of the double-buffer.

is_input
    

whether this BufferedRef acts as a pipeline input.

is_output
    

whether this BufferedRef acts as a pipeline output.

is_accumulator
    

whether this BufferedRef is an accumulator.

is_input_output
    

whether this BufferedRef is an input/output without automatic accumulation.

swap#
    

Tracks whether the BufferedRef slots need to be swapped before next copy.

Type:
    

ArrayRef | None

tiling#
    

The tiling to assume for the buffers.

Type:
    

tpu_info.Tiling | None

### BufferedRef

`BufferedRef`(_spec, _buffer_type, window_ref, ...) | A helper class to automate VMEM double buffering in pallas pipelines.  

---

### BufferedRef.__init__

__init__(_spec, _buffer_type, window_ref, accum_ref, copy_in_slot, wait_in_slot, copy_out_slot, wait_out_slot, _copy_in_slot_reg, _wait_in_slot_reg, _copy_out_slot_reg, _wait_out_slot_reg, next_fetch_smem, next_fetch_sreg, sem_recvs, sem_sends, swap, tiling)#
    

Parameters:
    

  * _spec (pl.BlockSpec)

  * _buffer_type (BufferType)

  * window_ref (ArrayRef | None)

  * accum_ref (ArrayRef | None)

  * copy_in_slot (ArrayRef | None)

  * wait_in_slot (ArrayRef | None)

  * copy_out_slot (ArrayRef | None)

  * wait_out_slot (ArrayRef | None)

  * _copy_in_slot_reg (int | jax.Array | None)

  * _wait_in_slot_reg (int | jax.Array | None)

  * _copy_out_slot_reg (int | jax.Array | None)

  * _wait_out_slot_reg (int | jax.Array | None)

  * next_fetch_smem (Sequence[jax.Array] | None)

  * next_fetch_sreg (Sequence[jax.Array] | None)

  * sem_recvs (SemaphoreTuple | None)

  * sem_sends (SemaphoreTuple | None)

  * swap (ArrayRef | None)

  * tiling (tpu_info.Tiling | None)

Return type:
    

None

---

### BufferedRef.accumulate

`accumulate`() | Add into the current slot.  

---

### BufferedRef.accumulator

`accumulator`(spec, dtype_or_type[, buffer_count]) |   

---

### BufferedRef.advance_copy_in_slot

`advance_copy_in_slot`([predicate]) | Switch to the next copy slot.  

---

### BufferedRef.advance_copy_out_slot

`advance_copy_out_slot`([predicate]) | Switch to the next copy slot.  

---

### BufferedRef.advance_wait_in_slot

`advance_wait_in_slot`([predicate]) | Switch to the next wait slot.  

---

### BufferedRef.advance_wait_out_slot

`advance_wait_out_slot`([predicate]) | Switch to the next wait slot.  

---

### BufferedRef.bind_existing_ref

`bind_existing_ref`(window_ref, indices) | For handling VMEM references, the pipeline aliases the existing ref.  

---

### BufferedRef.compute_slice

`compute_slice`(grid_indices) | Compute DMA slice from grid indices.  

---

### BufferedRef.copy_in

`copy_in`(src_ref, grid_indices) | Starts copy of HBM dma slice into the current slot.  

---

### BufferedRef.copy_out

`copy_out`(dst_ref, grid_indices) | Starts copy of HBM dma slice from the current slot.  

---

### BufferedRef.create

`create`(spec, dtype_or_type, buffer_type, ...) | Create a BufferedRef.  

---

### BufferedRef.get_dma_slice

`get_dma_slice`(src_ty, grid_indices) |   

---

### BufferedRef.init_slots

`init_slots`() | Initialize slot indices.  

---

### BufferedRef.input

`input`(spec, dtype_or_type[, buffer_count]) |   

---

### BufferedRef.input_output

`input_output`(spec, dtype_or_type[, buffer_count]) |   

---

### BufferedRef.load_slots

`load_slots`([predicate]) | Load slot information into registers.  

---

### BufferedRef.output

`output`(spec, dtype_or_type[, buffer_count]) |   

---

### BufferedRef.save_slots

`save_slots`([predicate]) | Save slot information from registers.  

---

### BufferedRef.set_accumulator

`set_accumulator`([init]) | Set accumulator or zero it out to initialize.  

---

### BufferedRef.wait_in

`wait_in`(src_ref, grid_indices) | Waits for input copy to finish.  

---

### BufferedRef.wait_out

`wait_out`(dst_ref, grid_indices) | Waits for output copy to finish.  

---

### BufferedRef.with_next_fetch

`with_next_fetch`([next_fetch]) |   

---

### BufferedRef.with_slot_index

`with_slot_index`([copy_in_slot, ...]) | Returns a new BufferedRef with the given slot index.  

---

### BufferedRef.with_spec

`with_spec`(spec) | Returns a new BufferedRef with the given block spec.  

---

### BufferedRef.buffer_count

`buffer_count` | Returns the number of buffers used for multiple buffering.  

---

### BufferedRef.current_copy_in_slot

`current_copy_in_slot` | Index in multiple buffer corresponding to the current slot.  

---

### BufferedRef.current_copy_out_slot

`current_copy_out_slot` | Index in multiple buffer corresponding to the current copy slot.  

---

### BufferedRef.current_wait_in_slot

`current_wait_in_slot` | Index in multiple buffer corresponding to the current wait slot.  

---

### BufferedRef.current_wait_out_slot

`current_wait_out_slot` | Index in multiple buffer corresponding to the current wait slot.  

---

### BufferedRef.is_buffered

`is_buffered` | Whether this buffer is multiple-buffered.  

---

### BufferedRef.next_fetch_indices

`next_fetch_indices` | Returns the next grid indices to fetch from if using lookahead.  

---

### BufferedRef.use_lookahead

`use_lookahead` | Whether this buffer allows lookahead for fetching blocks.  

---

### jax.experimental.pallas.tpu.CompilerParams

# jax.experimental.pallas.tpu.CompilerParams
class jax.experimental.pallas.tpu.CompilerParams(dimension_semantics=None, allow_input_fusion=None, vmem_limit_bytes=None, collective_id=None, has_side_effects=False, flags=None, internal_scratch_in_bytes=None, serialization_format=1, kernel_type=CoreType.TC, disable_bounds_checks=False, disable_semaphore_checks=False, skip_device_barrier=False, allow_collective_id_without_custom_barrier=False, shape_invariant_numerics=True, use_tc_tiling_on_sc=None)
    

Mosaic TPU compiler parameters.

Parameters:
    

  * dimension_semantics (tuple[DimensionSemantics, ...] | None)

  * allow_input_fusion (tuple[bool, ...] | None)

  * vmem_limit_bytes (int | None)

  * collective_id (int | None)

  * has_side_effects (bool | SideEffectType)

  * flags (dict[str, Any] | None)

  * internal_scratch_in_bytes (int | None)

  * serialization_format (int)

  * kernel_type (CoreType)

  * disable_bounds_checks (bool)

  * disable_semaphore_checks (bool)

  * skip_device_barrier (bool)

  * allow_collective_id_without_custom_barrier (bool)

  * shape_invariant_numerics (bool)

  * use_tc_tiling_on_sc (bool | None)

dimension_semantics#
    

A list of dimension semantics for each grid dimension of the kernel. Either “parallel” for dimensions that can execute in any order, or “arbitrary” for dimensions that must be executed sequentially.

Type:
    

tuple[DimensionSemantics, …] | None

allow_input_fusion#
    

A list of booleans indicating whether input fusion is allowed for each argument.

Type:
    

tuple[bool, …] | None

vmem_limit_bytes#
    

Overrides the default VMEM limit for a kernel. Note that this must be used in conjunction with the –xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes.

Type:
    

int | None

collective_id#
    

Indicates which barrier semaphore to use for the kernel. Note that using the same collective_id does not guarantee that the same barrier semaphore will be allocated between kernels.

Type:
    

int | None

has_side_effects#
    

Set to True to prevent kernel being CSEd by XLA.

Type:
    

bool | SideEffectType

flags#
    

A dictionary of command line flags for the kernel.

Type:
    

dict[str, Any] | None

internal_scratch_in_bytes#
    

The size of the internal scratch space used by Mosaic.

Type:
    

int | None

serialization_format#
    

The serialization format for the kernel body.

Type:
    

int

kernel_type#
    

Specify if the kernel is meant to run on TensorCore or one of the SparseCores

Type:
    

CoreType

disable_bounds_checks#
    

Disable bounds checks in the kernel.

Type:
    

bool

disable_semaphore_checks#
    

Disable semaphore checks in the kernel.

Type:
    

bool

skip_device_barrier#
    

Skip the default device barrier for the kernel.

Type:
    

bool

allow_collective_id_without_custom_barrier#
    

Allow the use of collective_id without a custom barrier.

Type:
    

bool

use_tc_tiling_on_sc#
    

Use TensorCore tiling for SparseCore. This flag is only used for `SC_*_SUBCORE` kernels.

Type:
    

bool | None

__init__(dimension_semantics=None, allow_input_fusion=None, vmem_limit_bytes=None, collective_id=None, has_side_effects=False, flags=None, internal_scratch_in_bytes=None, serialization_format=1, kernel_type=CoreType.TC, disable_bounds_checks=False, disable_semaphore_checks=False, skip_device_barrier=False, allow_collective_id_without_custom_barrier=False, shape_invariant_numerics=True, use_tc_tiling_on_sc=None)
    

Parameters:
    

  * dimension_semantics (Sequence[DimensionSemantics] | None)

  * allow_input_fusion (Sequence[bool] | None)

  * vmem_limit_bytes (int | None)

  * collective_id (int | None)

  * has_side_effects (bool | SideEffectType)

  * flags (Mapping[str, Any] | None)

  * internal_scratch_in_bytes (int | None)

  * serialization_format (int)

  * kernel_type (CoreType)

  * disable_bounds_checks (bool)

  * disable_semaphore_checks (bool)

  * skip_device_barrier (bool)

  * allow_collective_id_without_custom_barrier (bool)

  * shape_invariant_numerics (bool)

  * use_tc_tiling_on_sc (bool | None)

---

### CompilerParams.replace

`replace`(**changes) | Return a new object replacing specified fields with new values.  

---

### jax.experimental.pallas.pallas_call

jax.experimental.pallas.pallas_call(kernel, out_shape, *, grid_spec=None, grid=(), in_specs=NoBlockSpec, out_specs=NoBlockSpec, scratch_shapes=(), input_output_aliases={}, debug=False, interpret=False, name=None, compiler_params=None, cost_estimate=None, metadata=None)
    

Entry point for creating a Pallas kernel.

In contrast to `jax.experimental.pallas.kernel()`, this entry point assumes that the kernel will be executed over a `grid`.

See Pallas Quickstart.

Parameters:
    

  * kernel (Callable[..., None]) – the kernel function, that receives a Ref for each input and output. The shape of the Refs are given by the `block_shape` in the corresponding `in_specs` and `out_specs`.

  * out_shape (Any) – a PyTree of `jax.ShapeDtypeStruct` describing the shape and dtypes of the outputs.

  * grid_spec (pallas_core.GridSpec | None) – An alternative way to specify `grid`, `in_specs`, `out_specs` and `scratch_shapes`. If given, those other parameters must not be also given.

  * grid (pallas_core.TupleGrid) – the iteration space, as a tuple of integers. The kernel is executed as many times as `prod(grid)`. See details at grid, a.k.a. kernels in a loop.

  * in_specs (pallas_core.BlockSpecTree) – a PyTree of `jax.experimental.pallas.BlockSpec` with a structure matching that of the positional arguments. The default value for `in_specs` specifies the whole array for all inputs, e.g., as `pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)`. See details at BlockSpec, a.k.a. how to chunk up inputs.

  * out_specs (pallas_core.BlockSpecTree) – a PyTree of `jax.experimental.pallas.BlockSpec` with a structure matching that of the outputs. The default value for `out_specs` specifies the whole array, e.g., as `pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)`. See details at BlockSpec, a.k.a. how to chunk up inputs.

  * scratch_shapes (pallas_core.ScratchShapeTree) – a PyTree of backend-specific temporary objects required by the kernel, such as temporary buffers, synchronization primitives, etc.

  * input_output_aliases (Mapping[int, int]) – a dictionary mapping the index of some inputs to the index of the output that aliases them. These indices are in the flattened inputs and outputs (ignoring None values).

  * debug (bool) – if True, Pallas prints various intermediate forms of the kernel as it is being processed.

  * interpret (Any) – runs the `pallas_call` as a `jax.jit` of a scan over the grid whose body is the kernel lowered as a JAX function. This does not require a TPU or a GPU, and is the only way to run Pallas kernels on CPU. This is useful for debugging.

  * name (str | None) – if present, specifies the name to use for this kernel call in debugging and error messages. To this name we append the file and line where the kernel function is defined, .e.g: {name} for kernel function {kernel_name} at {file}:{line}. If missing, then we use {kernel_name} at {file}:{line}.

  * compiler_params (pallas_core.CompilerParams | None) – Optional compiler parameters. The value should be a backend-specific dataclass (`jax.experimental.pallas.tpu.CompilerParams`, `jax.experimental.pallas.triton.CompilerParams`, `jax.experimental.pallas.mosaic_gpu.CompilerParams`).

  * metadata (dict[str, str] | None) – Optional dictionary of information about the kernel that will be serialized as JSON in the HLO. Can be used for debugging and analysis.

  * cost_estimate (CostEstimate | None)

Returns:
    

A function that can be called on a number of positional array arguments to invoke the Pallas kernel.

Return type:
    

Callable[…, Any]

---

### ChipVersion

class jax.experimental.pallas.tpu.ChipVersion(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
    

TPU chip version.

The following table summarizes the differences between TPU versions:

Version | Physical TensorCores per chip | Lite chip | Megacore support  
---|---|---|---  
v2 | 2 | No | No  
v3 | 2 | No | No  
v4i | 1 | Yes | No  
v4 | 2 | No | Yes  
v5e | 1 | Yes | No  
v5p | 2 | No | Yes  
v6e | 1 | Yes | No  
7 | 2 | No | No  
7x | 2 | No | No  
  
__init__(*args, **kwds)#
    

Attributes

`num_physical_tensor_cores_per_chip` |   
---|---  
`supports_megacore` |   
`is_lite` |   
`TPU_V2` |   
`TPU_V3` |   
`TPU_V4I` |   
`TPU_V4` |   
`TPU_V5E` |   
`TPU_V5P` |   
`TPU_V6E` |   
`TPU_7` |   
`TPU_7X` |   

### ChipVersion

﻿jax.experimental.pallas.tpu.ChipVersion ======================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: ChipVersion .. automethod:: __init__ .. rubric:: Attributes .. autosummary:: ~ChipVersion.num_physical_tensor_cores_per_chip ~ChipVersion.supports_megacore ~ChipVersion.is_lite ~ChipVersion.TPU_V2 ~ChipVersion.TPU_V3 ~ChipVersion.TPU_V4I ~ChipVersion.TPU_V4 ~ChipVersion.TPU_V5E ~ChipVersion.TPU_V5P ~ChipVersion.TPU_V6E ~ChipVersion.TPU_7 ~ChipVersion.TPU_7X

### ChipVersion

`ChipVersion`(value[, names, module, ...]) | TPU chip version.  

---

### GridDimensionSemantics

﻿jax.experimental.pallas.tpu.GridDimensionSemantics ================================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: GridDimensionSemantics .. automethod:: __init__ .. rubric:: Attributes .. autosummary:: ~GridDimensionSemantics.PARALLEL ~GridDimensionSemantics.CORE_PARALLEL ~GridDimensionSemantics.SUBCORE_PARALLEL ~GridDimensionSemantics.ARBITRARY

### GridDimensionSemantics

class jax.experimental.pallas.tpu.GridDimensionSemantics(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
    

__init__(*args, **kwds)#
    

Attributes

`PARALLEL` |   
---|---  
`CORE_PARALLEL` |   
`SUBCORE_PARALLEL` |   
`ARBITRARY` |   
  

### GridDimensionSemantics

`GridDimensionSemantics`(value[, names, ...]) |   

---

### MemorySpace

﻿jax.experimental.pallas.tpu.MemorySpace ======================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: MemorySpace .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~MemorySpace.from_type .. rubric:: Attributes .. autosummary:: ~MemorySpace.VMEM ~MemorySpace.VMEM_SHARED ~MemorySpace.SMEM ~MemorySpace.CMEM ~MemorySpace.SEMAPHORE ~MemorySpace.HBM ~MemorySpace.HOST

### MemorySpace

class jax.experimental.pallas.tpu.MemorySpace(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
    

__init__(*args, **kwds)#
    

Methods

`from_type`(ty) |   
---|---  
  
Attributes

`VMEM` |   
---|---  
`VMEM_SHARED` |   
`SMEM` |   
`CMEM` |   
`SEMAPHORE` |   
`HBM` |   
`HOST` |   
  

### MemorySpace

`MemorySpace`(value[, names, module, ...]) |   

---

### SemaphoreType

﻿jax.experimental.pallas.tpu.SemaphoreType ========================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: SemaphoreType .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~SemaphoreType.get_array_aval ~SemaphoreType.get_ref_aval .. rubric:: Attributes .. autosummary:: ~SemaphoreType.REGULAR ~SemaphoreType.DMA ~SemaphoreType.BARRIER

### SemaphoreType

class jax.experimental.pallas.tpu.SemaphoreType(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
    

__init__(*args, **kwds)#
    

Methods

`get_array_aval`() |   
---|---  
`get_ref_aval`() |   
  
Attributes

`REGULAR` |   
---|---  
`DMA` |   
`BARRIER` |   

### SemaphoreType

`SemaphoreType`(value[, names, module, ...]) |   

---

### TpuInfo

# jax.experimental.pallas.tpu.TpuInfo
class jax.experimental.pallas.tpu.TpuInfo(*, chip_version, generation, num_cores, num_lanes, num_sublanes, mxu_column_size, vmem_capacity_bytes, cmem_capacity_bytes, smem_capacity_bytes, hbm_capacity_bytes, mem_bw_bytes_per_second, bf16_ops_per_second, int8_ops_per_second, fp8_ops_per_second, int4_ops_per_second, sparse_core=None)
    

TPU hardware information.

Note that all information is per-TensorCore so you would need to multiply by num_cores to obtain the total for the chip.

Parameters:
    

  * chip_version (ChipVersionBase)

  * generation (int)

  * num_cores (int)

  * num_lanes (int)

  * num_sublanes (int)

  * mxu_column_size (int)

  * vmem_capacity_bytes (int)

  * cmem_capacity_bytes (int)

  * smem_capacity_bytes (int)

  * hbm_capacity_bytes (int)

  * mem_bw_bytes_per_second (int)

  * bf16_ops_per_second (int)

  * int8_ops_per_second (int)

  * fp8_ops_per_second (int)

  * int4_ops_per_second (int)

  * sparse_core (SparseCoreInfo | None)

__init__(*, chip_version, generation, num_cores, num_lanes, num_sublanes, mxu_column_size, vmem_capacity_bytes, cmem_capacity_bytes, smem_capacity_bytes, hbm_capacity_bytes, mem_bw_bytes_per_second, bf16_ops_per_second, int8_ops_per_second, fp8_ops_per_second, int4_ops_per_second, sparse_core=None)#
    

Parameters:
    

  * chip_version (ChipVersionBase)

  * generation (int)

  * num_cores (int)

  * num_lanes (int)

  * num_sublanes (int)

  * mxu_column_size (int)

  * vmem_capacity_bytes (int)

  * cmem_capacity_bytes (int)

  * smem_capacity_bytes (int)

  * hbm_capacity_bytes (int)

  * mem_bw_bytes_per_second (int)

  * bf16_ops_per_second (int)

  * int8_ops_per_second (int)

  * fp8_ops_per_second (int)

  * int4_ops_per_second (int)

  * sparse_core (SparseCoreInfo | None)

Return type:
    

None

### TpuInfo

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

### TpuInfo

`TpuInfo`(*, chip_version, generation, ...[, ...]) | TPU hardware information.  

---

### load

`load`(ref, *[, mask]) | Loads an array from the given ref.  

---

### store

`store`(ref, val, *[, mask]) | Stores a value to the given ref.  

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

### make_async_remote_copy

`make_async_remote_copy`(src_ref, dst_ref, ...) | Creates a description of a remote copy operation.  

---

### sync_copy

`sync_copy`(src_ref, dst_ref, *[, add]) | Synchronously copies a PyTree of refs to another PyTree of refs.  

---

### emit_pipeline

`emit_pipeline`(body, *, grid[, in_specs, ...]) | Creates a function to emit a manual pallas pipeline.  

---

### emit_pipeline_with_allocations

`emit_pipeline_with_allocations`(body, *, grid) | Creates pallas pipeline and top-level allocation preparation functions.  

---

### get_pipeline_schedule

`get_pipeline_schedule`(schedule) | Retrieve a named pipeline schedule or pass through fully specified one.  

---

### make_pipeline_allocations

﻿jax.experimental.pallas.tpu.make\\_pipeline\\_allocations ======================================================= .. currentmodule:: jax.experimental.pallas.tpu .. autofunction:: make_pipeline_allocations

### make_pipeline_allocations

`make_pipeline_allocations`(*refs[, in_specs, ...]) | Create BufferedRefs for the pipeline.  

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

---

### core_barrier

`core_barrier`(sem, *, core_axis_name) | Synchronizes all cores in a given axis.  

---

### get_barrier_semaphore

`get_barrier_semaphore`() | Returns a barrier semaphore.  

---

### get_tpu_info

`get_tpu_info`() | Returns the TPU hardware info for the current device.  

---

### run_on_first_core

`run_on_first_core`(core_axis_name) | Runs a function on the first core in a given axis.  

---

### with_memory_space_constraint

﻿jax.experimental.pallas.tpu.with\\_memory\\_space\\_constraint =========================================================== .. currentmodule:: jax.experimental.pallas.tpu .. autofunction:: with_memory_space_constraint

### with_memory_space_constraint

`with_memory_space_constraint`(x, memory_space) | Constrains the memory space of an array.  

---

### TpuInfo.__init__

__init__(*, chip_version, generation, num_cores, num_lanes, num_sublanes, mxu_column_size, vmem_capacity_bytes, cmem_capacity_bytes, smem_capacity_bytes, hbm_capacity_bytes, mem_bw_bytes_per_second, bf16_ops_per_second, int8_ops_per_second, fp8_ops_per_second, int4_ops_per_second, sparse_core=None)#
    

Parameters:
    

  * chip_version (ChipVersionBase)

  * generation (int)

  * num_cores (int)

  * num_lanes (int)

  * num_sublanes (int)

  * mxu_column_size (int)

  * vmem_capacity_bytes (int)

  * cmem_capacity_bytes (int)

  * smem_capacity_bytes (int)

  * hbm_capacity_bytes (int)

  * mem_bw_bytes_per_second (int)

  * bf16_ops_per_second (int)

  * int8_ops_per_second (int)

  * fp8_ops_per_second (int)

  * int4_ops_per_second (int)

  * sparse_core (SparseCoreInfo | None)

Return type:
    

None

### TpuInfo.__init__

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

---

### TpuInfo.get_sublane_tiling

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

### TpuInfo.get_sublane_tiling

`get_sublane_tiling`(dtype) | Returns the sublane tiling for the given itemsize.  

---

### TpuInfo.is_matmul_supported

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

### TpuInfo.is_matmul_supported

`is_matmul_supported`(lhs_dtype, rhs_dtype) | Returns whether the chip natively supports matmul on the given input dtypes (no casting needed).  

---

### TpuInfo.is_megacore

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

### TpuInfo.is_megacore

`is_megacore` | Returns True if the chip is configured in Megacore mode.  

---

### TpuInfo.is_split_chip

﻿jax.experimental.pallas.tpu.TpuInfo =================================== .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: TpuInfo .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~TpuInfo.__init__ ~TpuInfo.get_sublane_tiling ~TpuInfo.is_matmul_supported .. rubric:: Attributes .. autosummary:: ~TpuInfo.is_lite ~TpuInfo.is_megacore ~TpuInfo.is_split_chip ~TpuInfo.sparse_core ~TpuInfo.chip_version ~TpuInfo.generation ~TpuInfo.num_cores ~TpuInfo.num_lanes ~TpuInfo.num_sublanes ~TpuInfo.mxu_column_size ~TpuInfo.vmem_capacity_bytes ~TpuInfo.cmem_capacity_bytes ~TpuInfo.smem_capacity_bytes ~TpuInfo.hbm_capacity_bytes ~TpuInfo.mem_bw_bytes_per_second ~TpuInfo.bf16_ops_per_second ~TpuInfo.int8_ops_per_second ~TpuInfo.fp8_ops_per_second ~TpuInfo.int4_ops_per_second

### TpuInfo.is_split_chip

`is_split_chip` | Returns True if the chip is a multi-core chip being used in single-core mode.  

---

### jax.experimental.pallas.tpu.emit_pipeline

jax.experimental.pallas.tpu.emit_pipeline(body, *, grid, in_specs=(), out_specs=(), tiling=None, should_accumulate_out=False, core_axis=None, core_axis_name=None, dimension_semantics=None, trace_scopes=True, no_pipelining=False, _explicit_indices=False)
    

Creates a function to emit a manual pallas pipeline.

This has the same semantics as pallas_call but is meant to be called inside pallas_call for nesting grids. This is useful when you need to have separate windowing strategies for communication and computation.

The new argument should_accumulate_out can be used to specify which outputs we should accumulate into automatically within and across pipeline invocations.

Parameters:
    

  * body – pallas kernel to set up pipeline for.

  * grid (tuple[int | jax.Array, ...]) – a pallas grid definition.

  * in_specs – input pallas block specs

  * out_specs – output pallas block specs

  * tiling (tpu_info.Tiling | None) – optional tiling to assume for the refs.

  * should_accumulate_out (bool) – booleans to indicate which outputs should be treated as accumulators.

  * core_axis (tuple[int, ...] | int | None) – optional int or tuple of int, indicates whether or not to partition the grid along the core axis.

  * core_axis_name (tuple[str, ...] | str | None) – optional str or tuple of str, indicates whether or not to partition the grid along the core axis.

  * dimension_semantics (tuple[GridDimensionSemantics, ...] | None) – optional tuple of GridDimensionSemantics (e.g. PARALLEL or ARBITRARY).

  * trace_scopes (bool) – optional bool, indicates whether to annotate each region in the pipeline using named_scope.

  * no_pipelining (bool) – If True, turns off pipelining and all copies will be made synchronous. This is useful for debugging multiple-buffering related bugs.

  * _explicit_indices (bool) – If True, the body will receive the iteration indices as its first argument. This parameter is meant for internal use only.

---

### jax.experimental.pallas.tpu.sample_block

jax.experimental.pallas.tpu.sample_block(sampler_fn, global_key, block_size, tile_size, total_size, block_index=None, **kwargs)
    

Samples a block of random values with invariance guarantees.

`sample_block` allows the sampling of identical blocks of random values across kernels with different block shapes and iteration orders. Each call to sample_block returns a block_size-shaped array of random samples corresponding to the block_index.

`tile_size` should be chosen such that it is a divisor to all block sizes one needs to be invariant to. The larger the `tile_size`, the more efficient the sampling process will be and therefore the best choice is typically the greatest common divisor between all possible block sizes.

Parameters:
    

  * sampler_fn (SampleFn) – A sampling function that consumes a key and returns random samples.

  * global_key (Array) – The global key to use for sampling.

  * block_size (tuple[int, ...]) – The shape of an individual block.

  * tile_size (tuple[int, ...]) – The shape of a `tile`, which is the smallest unit at which samples are generated. This should be selected to be a divisor of all block sizes one needs to be invariant to.

  * total_size (tuple[int, ...]) – The total size of the array to sample.

  * block_index (tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray, ...] | None) – The index denoting which block to generate keys for. Defaults to the program_id for each block axis.

  * **kwargs – Additional arguments to pass to the sampler_fn.

Returns:
    

A `block_size` shaped array of samples for the current block corresponding to `block_index`.

Return type:
    

Array

---

### jax.experimental.pallas.core_map

jax.experimental.pallas.core_map(mesh, *, compiler_params=None, interpret=False, debug=False, cost_estimate=None, name=None, metadata=None, scratch_shapes=())
    

Runs a function on a mesh, mapping it over the devices in the mesh.

The function should be stateful in that it takes in no inputs and returns no outputs but can mutate closed-over Refs, for example.

Parameters:
    

  * mesh – The mesh to run the function on.

  * compiler_params (Any | None) – The compiler parameters to pass to the backend.

  * interpret (bool) – Whether to run the function in interpret mode.

  * debug (bool) – Whether or not to out helpful debugging information.

  * cost_estimate (CostEstimate | None) – The cost estimate of the function.

  * name (str | None) – The (optional) name of the kernel.

  * metadata (dict[str, str] | None) – Optional dictionary of information about the kernel that will be serialized as JSON in the HLO. Can be used for debugging and analysis.

  * scratch_shapes (ScratchShapeTree) – The scratch arrays for the kernel. Supports both sequence and dict format. The space will be core-local unless the memory space type is specified to be shared (e.g., VMEM_SHARED).

---

### jax.experimental.pallas.tpu.make_async_remote_copy

jax.experimental.pallas.tpu.make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type=DeviceIdType.MESH)
    

Creates a description of a remote copy operation.

Copies data from src_ref on the current device to dst_ref on the device specified by device_id. Both semaphores should be waited on using the descriptor on both source and target devices.

Note that device_id can also refer to the current device.

Parameters:
    

  * src_ref – The source Reference.

  * dst_ref – The destination Reference.

  * send_sem – The semaphore on the source device.

  * recv_sem – The semaphore on the destination device.

  * device_id (MultiDimDeviceId | IntDeviceId | None) – The device id of the destination device. It could be a tuple, or a dictionary specifying the communication axis and destination index.

  * device_id_type (primitives.DeviceIdType) – The type of the device id.

Returns:
    

An AsyncCopyDescriptor.

Return type:
    

AsyncCopyDescriptor

---

### jax.experimental.pallas.tpu.get_barrier_semaphore

jax.experimental.pallas.tpu.get_barrier_semaphore()
    

Returns a barrier semaphore.

This function returns a barrier semaphore based on the collective_id of the current pallas kernel.

It’s very important that the semaphore is wait-ed back down to 0, or else the semaphores will become corrupted.

It’s also very important that the collective_id is different for each pallas kernel with communication. E.g. if you have two pallas kernels, one that syncs across the X axis of the device mesh and the second that syncs across the Y axis, they must have different collective_ids. However it is legal for two kernels that perform the same synchronization pattern (e.g. only communicating with neighbours on the same mesh axis) to share a collective_id. However, if in doubt, prefer not sharing collective_ids, as doing so incorrectly can lead to silent data corruption or crashes. Note that reusing the same collective_id doesn’t guarantee that the same semaphore is provided by XLA.

---

### jax.experimental.pallas.tpu.make_pipeline_allocations

jax.experimental.pallas.tpu.make_pipeline_allocations(*refs, in_specs=(), out_specs=(), tiling=None, should_accumulate_out=False, needs_swap_ref=True, grid=())
    

Create BufferedRefs for the pipeline.

This function creates buffered refs for an inner pipeline that can be created at the top-level of a pallas call such that they may be reused across multiple invocations of the inner pipeline.

Parameters:
    

  * in_specs – input pallas block specs

  * out_specs – output pallas block specs

  * should_accumulate_out – booleans to indicate which outputs should be treated as accumulators.

  * needs_swap_ref – whether a swap slots tracker needs to be allocated.

  * grid – grid to use for the pipeline.

  * tiling (tpu_info.Tiling | None)

Returns:
    

A list of BufferedRefs, one corresponding to each ref specified in the in_specs and out_specs.

---

### jax.experimental.pallas.tpu.emit_pipeline_with_allocations

jax.experimental.pallas.tpu.emit_pipeline_with_allocations(body, *, grid, in_specs=(), out_specs=(), should_accumulate_out=False)
    

Creates pallas pipeline and top-level allocation preparation functions.

Parameters:
    

  * body – pallas kernel to set up pipeline for.

  * grid – a pallas grid definition.

  * in_specs – input pallas block specs

  * out_specs – output pallas block specs

  * should_accumulate_out – booleans to indicate which outputs should be treated as accumulators.

Returns:
    

(emit_pipeline, make_allocations) function pair, where
    

  * emit_pipeline is the pallas pipeline function.

  * make_allocations is a function to create buffered refs for the inner pipeline that can be created at the top-level of a pallas call to be reused across multiple invocations of the inner pipeline.

---

### jax.experimental.pallas.tpu.with_memory_space_constraint

jax.experimental.pallas.tpu.with_memory_space_constraint(x, memory_space)
    

Constrains the memory space of an array.

This primitive does not change the value of `x`, but it constrains the memory space where it should be allocated. This is useful to force Pallas to allocate an array in a specific memory space.

As of now, this only operates on the inputs pallas_calls, as in you can apply this to the arguments of a pallas_call and it will constrain them, but other operations will not respect this constraint.

Parameters:
    

  * x (jax.Array) – The array to constrain.

  * memory_space (Any) – The memory space to constrain to.

Returns:
    

The array `x` with the memory space constraint.

Return type:
    

jax.Array

---

### jax.experimental.pallas.tpu.store

jax.experimental.pallas.tpu.store(ref, val, *, mask=None)
    

Stores a value to the given ref.

If `mask` is not specified, this function has the same semantics as `ref[idx] = val` in JAX.

Parameters:
    

  * ref (Ref) – The ref to store to.

  * val (jax.Array) – The value to store.

  * mask (jax.Array | None) – An optional boolean mask specifying which indices to store.

Return type:
    

None

---

### jax.experimental.pallas.tpu.load

jax.experimental.pallas.tpu.load(ref, *, mask=None)
    

Loads an array from the given ref.

If `mask` is not specified, this function has the same semantics as `ref[idx]` in JAX.

Parameters:
    

  * ref (Ref) – The ref to load from.

  * mask (jax.Array | None) – An optional boolean mask specifying which indices to load.

Returns:
    

The loaded array.

Return type:
    

jax.Array

---

### jax.experimental.pallas.tpu.make_async_copy

jax.experimental.pallas.tpu.make_async_copy(src_ref, dst_ref, sem)
    

Creates a description of an asynchronous copy operation.

Parameters:
    

  * src_ref – The source Reference.

  * dst_ref – The destination Reference.

  * sem – The semaphore used to track completion of the copy.

Returns:
    

An AsyncCopyDescriptor.

Return type:
    

AsyncCopyDescriptor

---

### MemorySpace.__init__

﻿jax.experimental.pallas.tpu.MemorySpace ======================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: MemorySpace .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~MemorySpace.from_type .. rubric:: Attributes .. autosummary:: ~MemorySpace.VMEM ~MemorySpace.VMEM_SHARED ~MemorySpace.SMEM ~MemorySpace.CMEM ~MemorySpace.SEMAPHORE ~MemorySpace.HBM ~MemorySpace.HOST

### MemorySpace.__init__

__init__(*args, **kwds)#
    

---

### MemorySpace.from_type

﻿jax.experimental.pallas.tpu.MemorySpace ======================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: MemorySpace .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~MemorySpace.from_type .. rubric:: Attributes .. autosummary:: ~MemorySpace.VMEM ~MemorySpace.VMEM_SHARED ~MemorySpace.SMEM ~MemorySpace.CMEM ~MemorySpace.SEMAPHORE ~MemorySpace.HBM ~MemorySpace.HOST

### MemorySpace.from_type

`from_type`(ty) |   
---|---  

---

### SemaphoreType.__init__

﻿jax.experimental.pallas.tpu.SemaphoreType ========================================= .. currentmodule:: jax.experimental.pallas.tpu .. autoclass:: SemaphoreType .. automethod:: __init__ .. rubric:: Methods .. autosummary:: ~SemaphoreType.get_array_aval ~SemaphoreType.get_ref_aval .. rubric:: Attributes .. autosummary:: ~SemaphoreType.REGULAR ~SemaphoreType.DMA ~SemaphoreType.BARRIER

### SemaphoreType.__init__

__init__(*args, **kwds)#
    

---

### jax.experimental.pallas.tpu.prng_seed

jax.experimental.pallas.tpu.prng_seed(*seeds)
    

Sets the seed for PRNG.

Parameters:
    

seeds (int | jax.Array) – One or more integer seeds for setting the PRNG seed. If more than one seed is passed in, the seed material will be mixed before setting the internal PRNG state.

Return type:
    

None

---

### jax.experimental.pallas.tpu.async_remote_copy

jax.experimental.pallas.tpu.async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type=DeviceIdType.MESH)
    

Issues a remote DMA copying from src_ref to dst_ref.

Parameters:
    

device_id_type (primitives.DeviceIdType)

Return type:
    

AsyncCopyDescriptor

---

### jax.experimental.pallas.tpu.get_tpu_info

jax.experimental.pallas.tpu.get_tpu_info()
    

Returns the TPU hardware info for the current device.

Note that all information is per-TensorCore so you would need to multiply by num_cores to obtain the total for the chip.

Return type:
    

TpuInfo

---

### jax.experimental.pallas.tpu.async_copy

jax.experimental.pallas.tpu.async_copy(src_ref, dst_ref, sem, *, priority=0, add=False)
    

Issues a DMA copying from src_ref to dst_ref.

Parameters:
    

  * priority (int)

  * add (bool)

Return type:
    

AsyncCopyDescriptor

---

### jax.experimental.pallas.tpu.sync_copy

jax.experimental.pallas.tpu.sync_copy(src_ref, dst_ref, *, add=False)
    

Synchronously copies a PyTree of refs to another PyTree of refs.

Parameters:
    

add (bool)

Return type:
    

None

---

### jax.experimental.pallas.tpu.to_pallas_key

jax.experimental.pallas.tpu.to_pallas_key(key)
    

Helper function for converting non-Pallas PRNG keys into Pallas keys.

Parameters:
    

key (Array)

Return type:
    

Array

---

### jax.experimental.pallas.tpu.get_pipeline_schedule

jax.experimental.pallas.tpu.get_pipeline_schedule(schedule)
    

Retrieve a named pipeline schedule or pass through fully specified one.

Return type:
    

Any

---

### jax.experimental.pallas.tpu.run_on_first_core

jax.experimental.pallas.tpu.run_on_first_core(core_axis_name)
    

Runs a function on the first core in a given axis.

Parameters:
    

core_axis_name (str)


---

### jax.experimental.pallas.tpu.core_barrier

jax.experimental.pallas.tpu.core_barrier(sem, *, core_axis_name)
    

Synchronizes all cores in a given axis.

Parameters:
    

core_axis_name (str)
