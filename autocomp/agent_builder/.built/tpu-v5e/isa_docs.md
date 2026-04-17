## Grid and Kernel Specification

### grid

### Grids by example
To automatically “carve” up the inputs and outputs, you provide a `grid` and `BlockSpec`s to `pallas_call`.

A `grid` is a tuple of integers (e.g. `()`, `(2, 3, 4)`, or `(8,)`) that specifies an iteration space. For example, a grid `(4, 5)` would have 20 elements: `(0, 0), (0, 1), ..., (0, 4), (1, 0), ..., (3, 4)`. We run the kernel function once for each element, a style of single-program multiple-data (SPMD) programming.

A 2D grid

When we provide a `grid` to `pallas_call`, the kernel is executed as many times as `prod(grid)`. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `program_id(axis=...)`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.


---

### GridSpec

`GridSpec`([grid, in_specs, out_specs, ...]) | Encodes the grid parameters for `jax.experimental.pallas.pallas_call()`.  

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
    
    
    >>> show_program_ids(x_shape=(4, 4), block_shape=None, grid=(2, 3),
    ...                  index_map=None)
    [[12 12 12 12]
     [12 12 12 12]
     [12 12 12 12]
     [12 12 12 12]]
    
    >>> show_program_ids(x_shape=(4, 4), block_shape=(4, 4), grid=(2, 3),
    ...                  index_map=None)
    [[12 12 12 12]
     [12 12 12 12]
     [12 12 12 12]
     [12 12 12 12]]
    
    

### BlockSpec

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

`BlockSpec`([block_shape, index_map, ...]) | Specifies how an array should be sliced for each invocation of a kernel.  

---

### pl.BlockSpec

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the `memory_space` argument on a `BlockSpec`. Note that pipelining is not allowed unless the `memory_space` is marked as `VMEM`. Memory spaces can also be used to specify scratch arguments to a kernel via the `scratch_shapes` argument on `pallas_call`. Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in `VMEM`, `SMEM`, or `SEMAPHORE`.

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer `x_hbm_ref` into a scratch VMEM buffer `scratch_vmem_ref` before using it for arithmetic and storing the result into an output VMEM buffer:
    
    
    def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
      pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
      out_vmem_ref[...] = scratch_vmem_ref[...] + 1
    
    x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
    out = pl.pallas_call(hbm_vmem_kernel,
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

### Blocked

### The “element” indexing mode
The behavior documented above applies to the default “blocked” indexing mode. When integers are used in the `block_shape` tuple e.g. `(4, 8)`, it is equivalent to passing in a `pl.Blocked(block_size)` object instead, e.g. `(pl.Blocked(4), pl.Blocked(8))`. Blocked indexing mode means the indices returned by `index_map` are block indices. We can pass in objects other than `pl.Blocked` to change the semantics of `index_map`, most notably, `pl.Element(block_size)`.. When using the `pl.Element` indexing mode the values returned by the index map function are used directly as the array indices, without first scaling them by the block size. When using the `pl.Element` mode you can specify virtual padding of the array as a tuple of low-high paddings for the dimension: the behavior is as if the overall array is padded on input. No guarantees are made for the padding values in element mode, similarly to the padding values for the blocked indexing mode when the block shape does not divide the overall array shape.

The `Element` mode is currently supported only on TPUs.

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

When using `jax.experimental.pallas.pallas_call()` the kernel function is executed multiple times on different inputs, as specified via the `grid` argument to `pallas_call`. Conceptually:
    
    
    pl.pallas_call(some_kernel, grid=(n,))(...)
    

maps to
    
    
    for i in range(n):
      some_kernel(...)
    

Grids can be generalized to be multi-dimensional, corresponding to nested loops. For example,
    
    
    pl.pallas_call(some_kernel, grid=(n, m))(...)
    

is equivalent to
    
    
    for i in range(n):
      for j in range(m):
        some_kernel(...)
    

This generalizes to any tuple of integers (a length `d` grid will correspond to `d` nested loops). The kernel is executed as many times as `prod(grid)`. The default grid value `()` results in one kernel invocation. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `jax.experimental.pallas.program_id()`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`. You can also use `jax.experimental.pallas.num_programs()` to get the grid size for a given axis.

See Grids by example for a simple kernel that uses this API.

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

`pallas_call`(kernel, out_shape, *[, ...]) | Entry point for creating a Pallas kernel.  

---

### pallas_call with scalar prefetch

    kernel = pl.pallas_call(...)
    result = kernel(*prefetch_args, *input_args)

---

### kernel

`kernel`([body, out_shape, scratch_shapes, ...]) | Entry point for creating a Pallas kernel.  

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

### num_programs

This generalizes to any tuple of integers (a length `d` grid will correspond to `d` nested loops). The kernel is executed as many times as `prod(grid)`. The default grid value `()` results in one kernel invocation. Each of these invocations is referred to as a “program”. To access which program (i.e. which element of the grid) the kernel is currently executing, we use `jax.experimental.pallas.program_id()`. For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`. You can also use `jax.experimental.pallas.num_programs()` to get the grid size for a given axis.

### num_programs

`num_programs`(axis) | Returns the size of the grid along the given axis.  

---

### core_map

`core_map`(mesh, *[, compiler_params, ...]) | Runs a function on a mesh, mapping it over the devices in the mesh.  

## Index Maps and Prefetch Patterns

### index_map signature for BlockSpec with prefetch

    def index_map(*grid_indices, *prefetch_refs):
        ...

---

### kernel signature with prefetch

    def kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs):
        ...

---

### mask_index_map pattern

    def mask_index_map(prefetch_map, i, j, ...):
      next_nonzero_block = prefetch_map[i, j]
      return (next_nonzero_block, 0, 0)

## Memory Spaces and Compiler Configuration

### MemorySpace

`MemorySpace`(value[, names, module, ...]) |   

---

### pl.ANY

`pl.ANY` | HBM (usually) or VMEM | DRAM  

---

### pltpu.VMEM

`pltpu.VMEM` | VMEM | SRAM  
`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

---

### pltpu.SMEM

`pltpu.SMEM` | SMEM | SRAM  
`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

---

### pltpu.SEMAPHORE

`pltpu.SEMAPHORE` | Semaphore | SRAM  
  
  * `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.

  * `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.

  * `MemorySpace.SEMAPHORE` is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

---

### CompilerParams

`CompilerParams`([dimension_semantics, ...]) | Mosaic TPU compiler parameters.  

---

### pltpu.CompilerParams

          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",))

---

### ChipVersion

`ChipVersion`(value[, names, module, ...]) | TPU chip version.  

---

### TpuInfo

`TpuInfo`(*, chip_version, generation, ...[, ...]) | TPU hardware information.  

---

### get_tpu_info

`get_tpu_info`() | Returns the TPU hardware info for the current device.  

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

## Memory Load and Store

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

## Asynchronous Copy and DMA

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

### pltpu.sync_copy

  * `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.

  * `MemorySpace.SEMAPHORE` is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details.

Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The default behavior for `pallas_call` on TPU is that arguments to `pallas_call` are assumed to live in HBM, and inputs to the user kernel body are stored in VMEM.

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the `memory_space` argument on a `BlockSpec`. Note that pipelining is not allowed unless the `memory_space` is marked as `VMEM`. Memory spaces can also be used to specify scratch arguments to a kernel via the `scratch_shapes` argument on `pallas_call`. Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in `VMEM`, `SMEM`, or `SEMAPHORE`.

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer `x_hbm_ref` into a scratch VMEM buffer `scratch_vmem_ref` before using it for arithmetic and storing the result into an output VMEM buffer:
    
    
    def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
      pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
      out_vmem_ref[...] = scratch_vmem_ref[...] + 1

---

### sync_copy

`sync_copy`(src_ref, dst_ref, *[, add]) | Synchronously copies a PyTree of refs to another PyTree of refs.  

## Pipeline and Buffering

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

### pl.BoundedSlice

`pltpu.emit_pipeline` supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with `pl.BoundedSlice(max_size)` rather than a static integer size, where `max_size` is the maximum size of the block. In addition, the corresponding index returned by `index_map` should be a dynamic slice constructed via `pl.ds(start, size)` where both `start` and `size` are element indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:
    
    
    pl.BlockSpec(
       block_shape=(pl.BoundedSlice(32), 256),
       index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
    )

---

### BufferedRef

`BufferedRef`(_spec, _buffer_type, ...[, ...]) | A helper class to automate VMEM double buffering in pallas pipelines.  

---

### BufferedRefBase

`BufferedRefBase`() | Abstract interface for BufferedRefs.  

---

### emit_pipeline

`emit_pipeline`(body, *, grid[, in_specs, ...]) | Creates a function to emit a manual pallas pipeline.  

---

### emit_pipeline_with_allocations

`emit_pipeline_with_allocations`(body, *, grid) | Creates pallas pipeline and top-level allocation preparation functions.  

## Synchronization and Semaphores

### SemaphoreType

`SemaphoreType`(value[, names, module, ...]) |   

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

### core_barrier

`core_barrier`(sem, *, core_axis_name) | Synchronizes all cores in a given axis.  

---

### get_barrier_semaphore

`get_barrier_semaphore`() | Returns a barrier semaphore.  

---

### run_on_first_core

`run_on_first_core`(core_axis_name) | Runs a function on the first core in a given axis.  

## Random Number Generation

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

Afterwards, you can call any number of stateful sampling functions which are equivalent to the corresponding JAX version but lack the `key` argument:

  * `pltpu.stateful_uniform`: the stateful equivalent to `jax.random.uniform()`

  * `pltpu.stateful_normal`: the stateful equivalent to `jax.random.normal()`

  * `pltpu.stateful_bernoulli`: the stateful equivalent to `jax.random.bernoulli()`

Generating any random number updates the internal state of the PRNG and subsequent calls will generate different numbers. Unlike in JAX, there is no need to `split` or `fold_in` keys and pass them into the sampling functions.

For example, the following kernel generates a set of uniform numbers from 0 to 1:
    
    
    from jax.experimental.pallas import tpu as pltpu
    
    def kernel_body(o_ref):
      pltpu.prng_seed(0)
      o_ref[...] = pltpu.stateful_uniform(shape=o_ref.shape, minval=0.0, maxval=1.0)
    
    pl.pallas_call(kernel_body,
                   out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32))
    

---

### prng_seed

`prng_seed`(*seeds) | Sets the seed for PRNG.  

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

### to_pallas_key

`to_pallas_key`(key) | Helper function for converting non-Pallas PRNG keys into Pallas keys.  

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

## Math, Arithmetic, and Tensor Operations

### dot

`dot`(a, b[, trans_a, trans_b, allow_tf32, ...]) | Computes the dot product of two arrays.  

---

### broadcast_to

`broadcast_to`(a, shape) | Broadcasts an array to a new shape.  

---

### cdiv

`cdiv`() | Computes the ceiling division of a divided by b.  

---

### multiple_of

`multiple_of`(x, values) | A compiler hint that asserts a value is a static multiple of another.  

## Control Flow

### when

`when`(condition, /) | Calls the decorated function when the condition is met.  

---

### loop

`loop`() | Returns a decorator that calls the decorated function in a loop.  

## Scoped Allocation and Execution

### run_scoped

`run_scoped`(f, *types[, collective_axes]) | Calls the function with allocated references and returns the result.  

---

### with_scoped

`with_scoped`(*types[, collective_axes]) | Returns a function decorator that runs a function with provided allocations.  

---

### empty

`empty`(shape, dtype, *[, out_sharding]) | Create an empty array of possibly uninitialized values.  

---

### empty_like

`empty_like`(x) | Create an empty PyTree of possibly uninitialized values.  

## Slicing and Indexing

### Slice

`Slice`(start, size[, stride]) | A slice with a start index and a size.  

---

### dslice

`dslice`(start[, size, stride]) | Constructs a `Slice` from a start index and a size.  

## Sparse Block Utilities

### generate_block_sparse_mat

    def generate_block_sparse_mat(key, M, N, blk_M, blk_N, p=0.2, dtype=jnp.float32):
      """Returns a sampled matrix and its block-sparse representation.
    
      Args:
        key: RNG Key.
        M: Major array dimension.
        N: Minor array dimension.
        blk_M: Block size along M dimension.
        blk_N: Block size along N dimension.
        p: Probability that a block will be non-zero.
        dtype: dtype of the sampled matrix.
    
      Returns:
        dense_mat: A (M, N) dense sampled array.
        block_data: A (num_blocks, blk_M, blk_N) array of data blocks representing
          the non-zero blocks of the matrix.
        indices_i: A (num_blocks,) array of block indices for the first axis.
        indices_j: A (num_blocks,) array of block indices for the second axis.
      """
      mask_key, blocks_key = jax.random.split(key)
      num_blocks = (M // blk_M, N // blk_N)
      # We first sample a block mask, denoting which blocks are nonzero.
      block_mask = jax.random.bernoulli(mask_key, p=p, shape=num_blocks)
      num_blocks = jnp.sum(block_mask)
      indices = jnp.where(block_mask)
      # For each non-zero block, we sample a block of random values.
      block_data = jax.random.uniform(blocks_key,
                                      shape=(num_blocks, blk_M, blk_N),
                                      dtype=dtype)
      # For checking purposes, create the dense version of the sparse matrix.
      dense_mat = jnp.zeros((M, N), dtype=dtype)
      for blk in range(num_blocks):
        idx_i = indices[0][blk]
        idx_j = indices[1][blk]
        slice_i = slice(idx_i * blk_M, (idx_i + 1) * blk_M)
        slice_j = slice(idx_j * blk_N, (idx_j + 1) * blk_N)
        dense_mat = dense_mat.at[slice_i, slice_j].set(block_data[blk])
      return dense_mat, block_data, indices[0], indices[1]

---

### sparsify_mask

    def sparsify_mask(mask: jax.Array,
                      block_shape: tuple[int, int]):
      """Preprocesses a mask into a sparse representation.
    
      Args:
        mask: A boolean array of shape [M, N]
        block_shape: The size of a single block.
    
      Returns:
        block_mask: A block_shape array of booleans indicating whether a block
          is all-zeros (0) or contains non-zero elements (1).
        prefetch_mask: A block_shape array of integers indicating the index of the
          next non-zero block.
        mask_data: A (num_blocks, block_shape) array containing
          the data for non-zero blocks of the mask.
      """
      M, N = mask.shape
      bm, bn = block_shape
    
      block_mask = jnp.zeros((M // bm, N // bn), dtype=mask.dtype)
      mask_types_finder = []
      mask_data = []
    
      next_mask_type_idx = 0
      prefetch_mask = jnp.zeros_like(block_mask)
      next_i = (M // bm) - 1
      next_j = (N // bn) - 1
      prefetch_i = jnp.zeros_like(block_mask)
      prefetch_j = jnp.zeros_like(block_mask)
      for i in range(M // bm, -1, -1):
        for j in range(N // bn, -1, -1):
          mask_block = mask[i * bm :(i + 1) * bm,
                            j * bn :(j + 1) * bn]
          is_nonzero = jnp.any(mask_block)
          if is_nonzero:
            try:
              type_index = mask_types_finder.index(str(mask_block))
            except ValueError:
              type_index = len(mask_types_finder)
              mask_types_finder.append(str(mask_block))
              mask_data.append(mask_block)
            next_mask_type_idx = type_index
            next_i = i
            next_j = j
          else:
            type_index = -1
          block_mask = block_mask.at[i, j].set(is_nonzero)
          prefetch_mask = prefetch_mask.at[i, j].set(next_mask_type_idx)
          prefetch_i = prefetch_i.at[i, j].set(next_i)
          prefetch_j = prefetch_j.at[i, j].set(next_j)
      return block_mask, prefetch_mask, prefetch_i, prefetch_j, jnp.stack(mask_data)

## Debugging and Diagnostics

### debug_print

`debug_print`(fmt, *args) | Prints values from inside a Pallas kernel.  

---

### debug_check

`debug_check`(condition, message) | Check the condition if `enable_debug_checks()` is set, otherwise do nothing.  