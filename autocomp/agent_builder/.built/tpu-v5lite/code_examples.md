## distributed.html

SUMMARY: This document covers distributed computing in Pallas for TPUs, demonstrating remote DMA operations, collective primitives (ppermute, all_gather, psum, psum_scatter), synchronization with semaphores, double-buffering, bi-directional communication, and nested pipelining techniques.

```python
import functools
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
```

```python
def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem):
    my_id = lax.axis_index('x')
    right_neighbor = lax.rem(my_id + 1, num_devices)
    remote_copy_op = pltpu.make_async_remote_copy(
        src_ref=input_ref,
        dst_ref=output_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    remote_copy_op.start()
    remote_copy_op.wait()
```

```python
def all_gather_kernel(input_ref,
                      output_ref,
                      local_copy_sem,
                      send_sem,
                      recv_sems):
    outer_step = pl.program_id(0)
    my_id = lax.axis_index('x')
    right_neighbor = lax.rem(my_id + 1, num_devices)
    copy_slot = my_id - outer_step
    copy_slot = lax.rem(copy_slot + num_devices, num_devices)

    @pl.when(outer_step == 0)
    def _():
        local_copy_op = pltpu.make_async_copy(
            src_ref=input_ref,
            dst_ref=output_ref.at[my_id],
            sem=local_copy_sem,
        )
        local_copy_op.start()
        local_copy_op.wait()

    remote_copy_op = pltpu.make_async_remote_copy(
        src_ref=output_ref.at[copy_slot],
        dst_ref=output_ref.at[copy_slot],
        send_sem=send_sem,
        recv_sem=recv_sems.at[outer_step],
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    remote_copy_op.start()
    remote_copy_op.wait()
```

```python
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
```

```python
def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
    """Performs a barrier with neighbors on the global barrier semaphore."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:
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
```

```python
def all_reduce_kernel(
    x_ref,
    o_ref,
    hbm_scratch,
    copy_sem,
    remote_recv_sem,
    remote_send_sem,
    capacity_sem,
    receive_scratch,
):
    outer_step = pl.program_id(0)
    working_slot = lax.rem(outer_step, 2)
    receiving_slot = 1 - working_slot

    my_id = lax.axis_index('x')
    right_neighbor = lax.rem(my_id + 1, num_devices)
    left_neighbor = lax.rem(my_id - 1 + num_devices, num_devices)

    @pl.when(outer_step == 0)
    def _():
        local_barrier(left_neighbor, right_neighbor)
        o_ref[...] = jnp.zeros_like(o_ref)
        receive_scratch[...] = jnp.zeros_like(receive_scratch)
        initial_copy = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=hbm_scratch.at[working_slot],
            send_sem=remote_send_sem,
            recv_sem=remote_recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        initial_copy.start()
        initial_copy.wait()

    pl.semaphore_signal(
        capacity_sem,
        inc=1,
        device_id=(left_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    local_copy = pltpu.make_async_copy(
        src_ref=hbm_scratch.at[working_slot],
        dst_ref=receive_scratch,
        sem=copy_sem,
    )
    local_copy.start()

    pl.semaphore_wait(capacity_sem, 1)
    remote_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot],
        dst_ref=hbm_scratch.at[receiving_slot],
        send_sem=remote_send_sem,
        recv_sem=remote_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    remote_copy.start()
    local_copy.wait()
    o_ref[...] += receive_scratch[...]
    remote_copy.wait()
```

```python
def reduce_scatter_kernel(
    x_ref,
    o_ref,
    hbm_scratch,
    local_copy_sem,
    left_recv_sem,
    left_send_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
    accum_scratch,
):
    outer_step = pl.program_id(0)
    phase = pl.program_id(1)
    is_start = jnp.logical_and(outer_step == 0, phase == 0)
    last_iteration = outer_step == pl.num_programs(0) - 1

    working_slot = lax.rem(outer_step, 2)
    receiving_slot = 1 - working_slot
    my_id = lax.axis_index('x')
    right_neighbor = mod(my_id + 1, num_devices)
    left_neighbor = mod(my_id - 1, num_devices)

    left_copy_device = mod(my_id + outer_step + 1, num_devices)
    right_copy_device = mod(my_id - outer_step - 1, num_devices)
    left_copy_slice = pl.ds(0, block_size[0] // 2)
    right_copy_slice = pl.ds(block_size[0] // 2, block_size[0] // 2)
    current_phase_slice = pl.ds(phase * (block_size[0] // 2), block_size[0] // 2)

    initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    right_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    @pl.when(is_start)
    def _():
        local_barrier(left_neighbor, right_neighbor)
        o_ref[...] = jnp.zeros_like(o_ref[...])
        accum_scratch[...] = jnp.zeros_like(accum_scratch[...])
        initial_left_copy.start()
        initial_left_copy.wait()
        initial_right_copy.start()
        signal(LEFT, right_capacity_sem)
        signal(RIGHT, left_capacity_sem)

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            pl.semaphore_wait(right_capacity_sem, 1)
            right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
            pl.semaphore_wait(left_capacity_sem, 1)
            left_copy.start()

    local_copy = pltpu.make_async_copy(
        src_ref=hbm_scratch.at[working_slot, current_phase_slice],
        dst_ref=accum_scratch,
        sem=local_copy_sem,
    )
    local_copy.start()
    local_copy.wait()

    @pl.when(~last_iteration)
    def _():
        @pl.when(phase == LEFT)
        def _():
            accum_scratch[...] += x_ref[left_copy_device, left_copy_slice]

        @pl.when(phase == RIGHT)
        def _():
            accum_scratch[...] += x_ref[right_copy_device, right_copy_slice]

    local_copy = pltpu.make_async_copy(
        src_ref=accum_scratch,
        dst_ref=hbm_scratch.at[working_slot, current_phase_slice],
        sem=local_copy_sem,
    )
    local_copy.start()
    local_copy.wait()

    @pl.when(is_start)
    def _():
        initial_right_copy.wait()

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            right_copy.wait()
            signal(LEFT, right_capacity_sem)

        @pl.when(phase == RIGHT)
        def _():
            left_copy.wait()
            signal(RIGHT, left_capacity_sem)

    @pl.when(last_iteration)
    def _():
        @pl.when(phase == LEFT)
        def _():
            o_ref[left_copy_slice, ...] = accum_scratch[...]
            pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
            o_ref[right_copy_slice, ...] = accum_scratch[...]
            pl.semaphore_wait(left_capacity_sem, 1)
```

```python
def inner_kernel(input_ref, accum_ref):
    @pl.when(pl.program_id(1) == 0)
    def _():
        accum_ref[...] = jnp.zeros_like(accum_ref)
    accum_ref[...] += input_ref[...]

accum_pipeline = pltpu.emit_pipeline(
    inner_kernel,
    in_specs=[inner_block_spec],
    out_specs=inner_block_spec,
    grid=inner_grid,
)

@pl.when(~last_iteration)
def _():
    @pl.when(phase == LEFT)
    def _():
        accum_pipeline(
            x_ref.at[left_copy_device, left_copy_slice],
            hbm_scratch.at[working_slot, left_copy_slice],
        )

    @pl.when(phase == RIGHT)
    def _():
        accum_pipeline(
            x_ref.at[right_copy_device, right_copy_slice],
            hbm_scratch.at[working_slot, right_copy_slice],
        )
```

## jax-ai-stack

SUMMARY: This document provides a comprehensive overview of the JAX AI stack for production ML on Cloud TPUs, covering the full ecosystem from core libraries (JAX, Flax, Optax, Orbax, Grain) through infrastructure (XLA, Pathways), advanced development tools (Pallas, Tokamax, Qwix), and application-layer solutions (MaxText, Tunix, vLLM). It demonstrates API usage patterns for optimization, quantization, and kernel authoring.

```python
# Optax implementation of a RMSProp optimizer with a custom learning rate
# schedule, gradient clipping and gradient accumulation.
optimizer = optax.chain(
  optax.clip_by_global_norm(GRADIENT_CLIP_VALUE),
  optax.rmsprop(learning_rate=optax.cosine_decay_schedule(init_value=lr,decay_steps=decay)),
  optax.apply_every(k=ACCUMULATION_STEPS)
)
```

```python
# Qwix quantization example: applying w4a4 (4-bit weight, 4-bit activation) 
# quantization to an LLM's MLP layers and w8 (8-bit weight) quantization to the embedder
fp_model = ModelWithoutQuantization(...)
rules = [
    qwix.QuantizationRule(
        module_path=r'embedder',
        weight_qtype='int8',
    ),
    qwix.QuantizationRule(
        module_path=r'layers_\d+/mlp',
        weight_qtype='int4',
        act_qtype='int4',
        tile_size=128,
        weight_calibration_method='rms,7',
    ),
]
quantized_model = qwix.quantize_model(fp_model, qwix.PtqProvider(rules))
```

## matmul.html

SUMMARY: This document covers writing efficient matrix multiplication kernels for TPU using JAX Pallas, demonstrating block matrix multiplication, pipelining, bfloat16 support, and kernel fusion techniques with BlockSpec and grid specifications.

```python
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
```

```python
def matmul_kernel(x_ref, y_ref, z_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    z_ref[...] = jnp.zeros_like(z_ref)

  z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
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
```

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)

@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk),
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
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...], y_ref[...], dims, preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)

@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'transpose_rhs'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk, transpose_rhs=transpose_rhs),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            y_block_spec,
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```python
def matmul_kernel(
    x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs, activation
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...],
      y_ref[...],
      dims,
      preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)

@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'activation'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(
          matmul_kernel,
          nsteps=k // bk,
          transpose_rhs=transpose_rhs,
          activation=activation,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              y_block_spec,
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

## pipelining.html

SUMMARY: This document covers software pipelining fundamentals for JAX Pallas kernels, explaining how to overlap communication and compute operations to hide memory latency. It demonstrates the Pallas API for writing pipelined kernels using grid, BlockSpec, and pallas_call, with examples of elementwise operations and reductions.

```python
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
```

```python
def add_matrices_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
  # Load x and y from SRAM into registers
  x_regs = x_sram_ref[:, :]
  y_regs = y_sram_ref[:, :]
  # Execute a vectorized add
  z_regs = x_regs + y_regs
  # Store the output values in registers back into SRAM
  z_sram_ref[:, :] = z_regs

def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  return z
```

```python
def add_matrices_pipelined_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

def add_matrices_pipelined(x: jax.Array, y: jax.Array):
  total_shape = (4096, 4096)
  block_shape = (512, 512)
  return pl.pallas_call(
    add_matrices_pipelined_kernel,
    grid=tuple(total // block for (total, block) in zip(total_shape, block_shape)),
    in_specs=[
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j))
    ],
    out_specs=pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
    out_shape=jax.ShapeDtypeStruct(total_shape, dtype=jnp.float32),
  )(x, y)
```

```python
def add_matrices_pipelined_param(
    x: jax.Array, y: jax.Array, *, bm: int = 256, bn: int = 256
) -> jax.Array:
  m, n = x.shape
  block_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
  return pl.pallas_call(
      add_matrices_pipelined_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(m // bm, n // bn),
  )(x, y)
```

```python
def correct_sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
  o_ref[...] += x_ref[...]

def correct_sum(x: jax.Array,
                block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
  return pl.pallas_call(
      correct_sum_kernel,
      grid=grid,
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)
```

## async_note.html

SUMMARY: This document describes how to implement decomposed async operations in Pallas on TPU, enabling overlapping computation and communication across multiple kernels using semaphores and stateful references to manage lifetimes and scheduling.

```python
import functools
import jax
import jax.lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Basic async ppermute decomposition with start/done kernels

def ppermute_start_kernel(
    in_ref, send_sem, recv_sem, out_ref, _, *, axis_name,
):
  axis_size = jax.lax.psum(1, axis_name)
  left_neighbor = jax.lax.rem(
      jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
  )
  right_neighbor = jax.lax.rem(jax.lax.axis_index(axis_name) + 1, axis_size)
  barrier_sem = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
  pltpu.semaphore_wait(barrier_sem, 1)
  pltpu.make_async_remote_copy(
      in_ref, out_ref, send_sem, recv_sem, device_id=right_neighbor
  ).start()

def ppermute_start(x, *, axis_name):
  send_sem, recv_sem, x, out = pl.pallas_call(
      functools.partial(ppermute_start_kernel, axis_name=axis_name),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
          jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
      ),
      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_specs=(
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
      ),
      input_output_aliases={0: 2}
  )(x)
  return send_sem, recv_sem, x, out

def ppermute_done_kernel(_, ref, send_sem, recv_sem, _):
  pltpu.make_async_copy(ref, ref, send_sem).wait()
  pltpu.make_async_copy(ref, ref, recv_sem).wait()

def ppermute_done(send_sem, recv_sem, x, out):
  out = pl.pallas_call(
      ppermute_done_kernel,
      out_shape=(jax.ShapeDtypeStruct(out.shape, dtype=out.dtype),),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      input_output_aliases={1: 0}
  )(x, out, send_sem, recv_sem)
  return out

# Usage with optimization barriers for scheduling
def f_with_barriers(x):
  fut = ppermute_start(x)
  x, fut = jax.lax.optimization_barrier((x, fut))
  z = x + 1
  z, fut = jax.lax.optimization_barrier((z, fut))
  y = ppermute_done(fut)
  return y, z

# Usage in a loop with unrolling to avoid defensive copies
def f_loop_unrolled(x):
  def body(i, x):
    fut = ppermute_start(x)
    y = ppermute_done(fut)
    return y
  return jax.lax.fori_loop(0, 8, body, x, unroll=2)

# Staggered loop passing futures across boundaries
def f_staggered_loop(x):
  fut = ppermute_start(x)
  def body(i, fut):
    x = ppermute_done(fut)
    fut = ppermute_start(x)
    return fut
  fut = jax.lax.fori_loop(0, 7, body, fut, unroll=2)
  return ppermute_done(fut)

# Loop with accumulation
def f_loop_accumulate(x):
  out = jnp.zeros_like(x)
  send_sem, recv_sem, x, out_buf = ppermute_start(x)
  out = out + x
  def body(i, carry):
    out, (send_sem, recv_sem, x, out_buf) = carry
    x = ppermute_done(send_sem, recv_sem, x, out_buf)
    send_sem, recv_sem, x, out_buf = ppermute_start(x)
    out = out + x
    return out, (send_sem, recv_sem, x, out_buf)
  out, (send_sem, recv_sem, x, out_buf) = jax.lax.fori_loop(
      0, 7, body, (out, (send_sem, recv_sem, x, out_buf)), unroll=2
  )
  return out, ppermute_done(send_sem, recv_sem, x, out_buf)
```

## sparse.html

SUMMARY: This document covers block-sparse computation in JAX Pallas on TPU, demonstrating how to use scalar prefetch to enable dynamic block indexing and implement sparse kernels including block-aligned dynamic slicing, sparse-dense matrix multiplication, and dense matrix multiplication with block-sparse output masks.

```python
import functools
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import checkify
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
```

```python
def dynamic_slice_kernel(indices, x_ref, o_ref):
  del indices
  o_ref[...] = x_ref[...]

@checkify.checkify
@functools.partial(jax.jit, static_argnums=(2,))
def block_dynamic_slice(x, starts, sizes):
  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(1, 1),
      in_specs=[pl.BlockSpec(
          sizes,
          lambda i, j, block_idx: (block_idx[0], block_idx[1]))],
      out_specs=pl.BlockSpec(sizes, lambda *_: (0, 0)),
  )

  kernel = pl.pallas_call(
    dynamic_slice_kernel,
    grid_spec=grid_spec,
    out_shape=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
  )
  checkify.check(starts[0] % sizes[0] == 0, "Starts must be divisible by size.")
  checkify.check(starts[1] % sizes[1] == 0, "Starts must be divisible by size.")
  block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]])
  return kernel(block_idx, x)
```

```python
def generate_block_sparse_mat(key, M, N, blk_M, blk_N, p=0.2, dtype=jnp.float32):
  """Returns a sampled matrix and its block-sparse representation."""
  mask_key, blocks_key = jax.random.split(key)
  num_blocks = (M // blk_M, N // blk_N)
  block_mask = jax.random.bernoulli(mask_key, p=p, shape=num_blocks)
  num_blocks = jnp.sum(block_mask)
  indices = jnp.where(block_mask)
  block_data = jax.random.uniform(blocks_key,
                                  shape=(num_blocks, blk_M, blk_N),
                                  dtype=dtype)
  dense_mat = jnp.zeros((M, N), dtype=dtype)
  for blk in range(num_blocks):
    idx_i = indices[0][blk]
    idx_j = indices[1][blk]
    slice_i = slice(idx_i * blk_M, (idx_i + 1) * blk_M)
    slice_j = slice(idx_j * blk_N, (idx_j + 1) * blk_N)
    dense_mat = dense_mat.at[slice_i, slice_j].set(block_data[blk])
  return dense_mat, block_data, indices[0], indices[1]
```

```python
def dsd_kernel(idxs_i_ref, idxs_k_ref,
               x_ref, y_ref, _, o_ref,
               accum_scratch,
               ):
  """A DSD (Dense = Sparse @ Dense) matmul kernel."""
  del idxs_k_ref
  blk_idx = pl.program_id(1)
  is_start = blk_idx == 0
  changed_blocks = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx-1, 0)])
  @pl.when(is_start | changed_blocks)
  def _():
    accum_scratch[...] = jnp.zeros_like(accum_scratch)
  accum_scratch[...] += jnp.dot(x_ref[0, :, :], y_ref[...], preferred_element_type=jnp.float32)

  next_block_change = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.minimum(blk_idx+1, num_blocks)])
  is_end = blk_idx == (num_blocks - 1)
  @pl.when(is_end | next_block_change)
  def _():
    o_ref[...] = accum_scratch[...].astype(o_ref.dtype)

def x_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del j, blk_idxs_i, blk_idxs_k
  return (blk_idx, 0, 0)

def y_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del blk_idxs_i
  return (blk_idxs_k[blk_idx], j)

def o_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del blk_idxs_k
  return (blk_idxs_i[blk_idx], j)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=2,
    grid=(N // blk_N, num_blocks),
    in_specs=[pl.BlockSpec((1, blk_M, blk_K), x_map),
              pl.BlockSpec((blk_K, blk_N), y_map),
              pl.BlockSpec((blk_M, blk_N), o_map),
              ],
    out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
    scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)]
)
kernel = pl.pallas_call(
  dsd_kernel,
  grid_spec=grid_spec,
  out_shape=out_shape,
  input_output_aliases={4: 0},
)
```

```python
def sparsify_mask(mask: jax.Array,
                  block_shape: tuple[int, int]):
  """Preprocesses a mask into a sparse representation."""
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
```

```python
def sparse_mask_matmul(
    block_mask_ref, prefetch_mask, prefetch_i, prefetch_j,
    x_ref, y_ref, mask_ref, o_ref,
    accum_scratch
    ):
  del prefetch_mask, prefetch_i, prefetch_j
  i, j, k = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  should_compute = block_mask_ref[i, j] != 0
  @pl.when(k == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
    accum_scratch[...] = jnp.zeros_like(accum_scratch[...])

  @pl.when(should_compute)
  def _():
    result = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    accum_scratch[...] += result
    @pl.when(k == pl.num_programs(2) - 1)
    def _():
      o_ref[...] = (mask_ref[0, ...] * accum_scratch[...]).astype(o_ref.dtype)

def x_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
  del prefetch_mask, prefetch_j
  k_fetch = (block_mask[i, j] != 0) * k
  return (prefetch_i[i, j], k_fetch)

def y_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
  del prefetch_mask, prefetch_i
  k_fetch = (block_mask[i, j] != 0) * k
  return (k_fetch, prefetch_j[i, j])

def mask_map(i, j, k, block_mask, prefetch_mask, *_):
  del k, block_mask
  return (prefetch_mask[i, j], 0, 0)

def o_map(i, j, k, *_):
  del k
  return (i, j)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=4,
    grid=(M // blk_M, N // blk_N, K // blk_K),
    in_specs=[pl.BlockSpec((blk_M, blk_K), x_map),
              pl.BlockSpec((blk_K, blk_N), y_map),
              pl.BlockSpec((1, blk_M, blk_N), mask_map)],
    out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
    scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)]
)
kernel = pl.pallas_call(
  sparse_mask_matmul,
  grid_spec=grid_spec,
  out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
)
```

## core_map.html

SUMMARY: This document covers Pallas `core_map` API for per-core TPU/GPU kernel programming, demonstrating inter-core communication, pipelining with automatic and manual work distribution, scalar prefetch with dynamic indexing, and SparseCore operations.

```python
from functools import partial
import jax
from jax.sharding import NamedSharding
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np
```

```python
# Simple per-core kernel with inter-core communication
def swap_cores_kernel(in_hbm, out_hbm,
                      in_vmem, scratch_vmem, out_vmem,
                      sem, send_sem, recv_sem):
  core_index = jax.lax.axis_index('core')
  num_cores = jax.lax.axis_size('core')
  slc_size = in_hbm.shape[-1] // num_cores
  slc = pl.ds(core_index * slc_size, slc_size)

  # Copy in a core-dependent slice of the input
  pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()

  # Barrier to synchronize cores
  dst_core = (core_index + 1) % num_cores
  sem0 = pltpu.get_barrier_semaphore()
  pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})
  pl.semaphore_wait(sem0, 1)

  # Swap data between cores
  the_copy = pltpu.make_async_remote_copy(
      in_vmem, scratch_vmem, send_sem, recv_sem, device_id={'core': dst_core},
  )
  the_copy.start()
  the_copy.wait()

  # Core-local compute
  out_vmem[...] = scratch_vmem[...] * 2

  # Copy out the output
  pltpu.async_copy(out_vmem, out_hbm.at[:, slc], sem).wait()
```

```python
# Top-level function using core_map with kernel decorator
@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def swap_cores(x):
  scratch_shapes = [pltpu.VMEM(local_vmem_shape, x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 3
  return pl.kernel(swap_cores_kernel, out_shape=x, mesh=tc_mesh,
                   scratch_shapes=scratch_shapes,
                   compiler_params=pltpu.CompilerParams(collective_id=0))(x)
```

```python
# Pipelining with automatic work parallelization
def add_one_body(in_vmem, out_vmem):
  out_vmem[...] = in_vmem[...] + 1

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

@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def add_one(x):
  return pl.kernel(add_one_kernel, out_shape=x, mesh=tc_mesh, scratch_shapes=[])(x)
```

```python
# Scalar prefetch with manual core work distribution
def indexed_add_one_kernel(in_refs, out_refs, i_smem_ref):
  (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
  in_shape = x_hbm_ref.shape
  pltpu.sync_copy(i_hbm_ref, i_smem_ref)

  core_idx = jax.lax.axis_index('core')
  core_slc_size = in_shape[0] // num_cores
  i_map = lambda i: core_idx * core_slc_size // 8 + i
  j_map = lambda j: i_smem_ref[0] // 128 + j

  pltpu.emit_pipeline(
      add_one_body,
      grid=(core_slc_size // 8, output_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j_map(j)),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j),
      )]
  )(x_hbm_ref, o_hbm_ref)

@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(in_spec, jax.P()), out_specs=in_spec, check_vma=False)
def indexed_add_one(x, index):
  out_shape = jax.ShapeDtypeStruct((x.shape[0], x.shape[1] // 2), x.dtype)
  return pl.kernel(indexed_add_one_kernel,
                   out_shape=out_shape, mesh=tc_mesh,
                   scratch_shapes=[pltpu.SMEM((1,), jnp.int32)])((x, index))
```

```python
# SparseCore kernel with nested loops for register operations
def sc_add_one_body(in_vmem, out_vmem):
  @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
  def _reg_loop_0(c0):
    @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
    def _reg_loop_1(c1):
      slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
      out_vmem[slc] = in_vmem[slc] + 1

def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
  in_shape = x_hbm_ref.shape
  core_idx = jax.lax.axis_index('core')
  subcore_idx = jax.lax.axis_index("subcore")
  cm_idx = core_idx * sc_num_subcores + subcore_idx
  slc_size = in_shape[0] // (sc_num_subcores * sc_num_cores)
  index_map = lambda i, j: (
      pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8), j)

  pltpu.emit_pipeline(
      sc_add_one_body,
      grid=(slc_size // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )]
  )(x_hbm_ref, o_hbm_ref)

@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def sc_add_one(x):
  return pl.kernel(sc_add_one_kernel, out_shape=x, mesh=sc_mesh, scratch_shapes=[])(x)
```

## quickstart.html

SUMMARY: This document covers Pallas, a JAX extension for writing custom GPU and TPU kernels, demonstrating core APIs like `pallas_call`, `Ref` types, grids, `program_id`, and `BlockSpec` for memory-aware kernel programming with examples ranging from simple vector addition to optimized matrix multiplication.

```python
from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
```

```python
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
```

```python
def add_sliced_kernel(x_ref, y_ref, o_ref):
  small_mid = x_ref.shape[0] // 2

  x_left = x_ref.at[:small_mid]
  x_right = x_ref.at[small_mid:]
  y_left = y_ref.at[:small_mid]
  y_right = y_ref.at[small_mid:]

  large_mid = 2*small_mid
  o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
  o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
  o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
  o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]
```

```python
def iota_kernel(o_ref):
  i = pl.program_id(0)
  o_ref[i] = i

def iota(size: int):
  return pl.pallas_call(iota_kernel,
                        out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                        grid=(size,))()
```

```python
from jax.experimental.pallas import tpu as pltpu

def iota(size: int):
  return pl.pallas_call(iota_kernel,
                        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                        out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                        grid=(size,))()
```

```python
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
```

```python
def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
  z_ref[...] = activation(x_ref[...] @ y_ref[...])

def matmul(x: jax.Array, y: jax.Array, *, activation):
  return pl.pallas_call(
    partial(matmul_kernel, activation=activation),
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j)
    ),
  )(x, y)
```

```python
z = jax.vmap(partial(matmul, activation=jax.nn.relu))(x, y)
```

## sparsecore.html

SUMMARY: This document covers SparseCore kernel writing in JAX Pallas, demonstrating how to write kernels targeting TPU SparseCores for sparse memory access operations, including basic kernels, pipelining, gather/scatter operations, and overlapping with TensorCore computations.

```python
from functools import partial
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

# Basic SparseCore kernel with DMAs and scalar operations
@jax.jit
def cumsum(x):
  scalar_mesh = plsc.ScalarSubcoreMesh(
      axis_name="core", num_cores=2
  )
  
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
```

```python
# Pipelined SparseCore kernel with vector operations
@jax.jit
def sc_add_one(x):
  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core", subcore_axis_name="subcore"
  )
  SC_REG_OP_SHAPE = (1, 16)
  dma_block = (8, 128)
  
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
```

```python
# Gather operation using indexed retrieval
@jax.jit
def gather(x, indices):
  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core", subcore_axis_name="subcore"
  )
  gather_window_size = 128
  value_dim = 128
  num_indices = indices.shape[0]
  indices = indices.reshape((1, num_indices))

  @pl.kernel(
      out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype),
      mesh=vector_mesh,
  )
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

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
```

```python
# Gather with indexed_by BlockSpec and computation
@jax.jit
def gather_add_one(x, indices):
  gather_window_size = 128
  value_dim = 128
  num_indices = indices.shape[0]
  
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
    @pl.loop(0, gather_window_size)
    def _(c0):
      @pl.loop(0, o_ref.shape[1], step=16)
      def _(c1):
        slc = (pl.ds(c0, 1), pl.ds(c1, 16))
        o_ref.at[*slc][...] = gathered_ref.at[*slc][...] + 1

  return kernel(x, indices)
```

```python
# Scatter operation using indexed write
@jax.jit
def scatter(x, indices):
  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core", subcore_axis_name="subcore"
  )
  gather_window_size = 128
  value_dim = 128
  num_indices = indices.shape[0]
  batch_size = 4096
  indices = indices.reshape((1, num_indices))

  @pl.kernel(
      out_shape=jax.ShapeDtypeStruct((batch_size, value_dim), x.dtype),
      mesh=vector_mesh,
      scratch_shapes=[],
  )
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(x_vmem, i_vmem):
      pltpu.sync_copy(x_vmem, o_hbm.at[i_vmem.at[0]])

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // gather_window_size,),
        in_specs=[
            pl.BlockSpec(
                (gather_window_size, value_dim), index_map=lambda i: (i, 0)
            ),
            pl.BlockSpec(
                (1, gather_window_size,),
                index_map=lambda i: (0, i),
            ),
        ],
        out_specs=[],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(x_hbm, i_hbm)

  return kernel(x, indices)
```

```python
# Overlapping TensorCore and SparseCore kernels
@jax.jit
def tc_add_one(x):
  return x + 1

@jax.jit
def two_add_ones(x):
  return sc_add_one(x), tc_add_one(x)
```

## vmapped_log_probs.html

SUMMARY: This document demonstrates autobatching for Bayesian inference using JAX's vmap function, showing how to write non-batched probabilistic models that automatically handle batched inputs, and includes a complete variational inference example with SGD optimization.

```python
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
```

```python
def log_joint(beta):
    result = 0.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.))
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta))))
    return result
```

```python
def batched_log_joint(beta):
    result = 0.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.),
                           axis=-1)
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta.T).T)),
                           axis=-1)
    return result
```

```python
vmap_batched_log_joint = jax.vmap(log_joint)
```

```python
@jax.jit
def log_joint(beta):
    result = 0.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=10.))
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta))))
    return result

batched_log_joint = jax.jit(jax.vmap(log_joint))
```

```python
def elbo(beta_loc, beta_log_scale, epsilon):
    beta_sample = beta_loc + jnp.exp(beta_log_scale) * epsilon
    return jnp.mean(batched_log_joint(beta_sample), 0) + jnp.sum(beta_log_scale - 0.5 * jnp.log(2*jnp.pi))

elbo = jax.jit(elbo)
elbo_val_and_grad = jax.jit(jax.value_and_grad(elbo, argnums=(0, 1)))
```

```python
def normal_sample(key, shape):
    """Convenience function for quasi-stateful RNG."""
    new_key, sub_key = random.split(key)
    return new_key, random.normal(sub_key, shape)

normal_sample = jax.jit(normal_sample, static_argnums=(1,))
```

## prng.html

SUMMARY: This document covers pseudo-random number generation APIs in JAX Pallas for TPU kernels, demonstrating three approaches: the portable jax.random API, the hardware PRNG with stateful and stateless modes, and block-invariant sampling for consistent random number generation across different block sizes.

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax
import jax.numpy as jnp
import jax.random as jax_random

# Example 1: Using jax.random API with key passed via VMEM
def body(key_ref, o_ref):
    key = key_ref[...]
    o_ref[...] = jax_random.uniform(
        key, shape=o_ref[...].shape, minval=0.0, maxval=1.0
    )

threefry_key = jax_random.key(0, impl="threefry2x32")

result = pl.pallas_call(
    body,
    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
    out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32)
)(threefry_key)
```

```python
# Example 2: Stateful PRNG with hardware seed
from jax.experimental.pallas import tpu as pltpu

def kernel_body(o_ref):
    pltpu.prng_seed(0)
    o_ref[...] = pltpu.stateful_uniform(shape=o_ref.shape, minval=0.0, maxval=1.0)

pl.pallas_call(kernel_body,
               out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32))()
```

```python
# Example 3: Stateless hardware PRNG with key passed via SMEM
def body(key_ref, o_ref):
    o_ref[...] = jax.random.uniform(
        key_ref[...], shape=o_ref[...].shape
    )

rbg_key = jax_random.key(0, impl="threefry2x32")
key = pltpu.to_pallas_key(rbg_key)
o_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
result = pl.pallas_call(
    body,
    in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
    out_shape=o_shape,
)(key)
```

```python
# Example 4: Block-invariant sampling across different block sizes
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
```