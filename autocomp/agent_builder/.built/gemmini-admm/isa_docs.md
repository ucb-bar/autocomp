## Configuration Instructions

### config_ex

```c
#define config_ex(dataflow, act, A_stride, A_transpose, B_transpose)
```

Configure the state of the accelerator.

- `dataflow` is `WEIGHT_STATIONARY` or `OUTPUT_STATIONARY`.
- `act` is the activation function. Options: `NO_ACTIVATION`, `RELU`, `LAYERNORM`, `IGELU`, `SOFTMAX`.
- `A_stride` is the stride with which rows of A in the scratchpad are loaded into the systolic array, during computes. If this stride is `1`, consecutive rows in the scratchpad starting from the starting address of A are fed into the systolic array as the A matrix. If the stride is `2`, every other row is fed in instead.
- `A_transpose` is a boolean value that represents whether the matrix A is transposed.
- `B_transpose` is a boolean value that represents whether the matrix B is transposed.

### config_ld

```c
#define config_ld(dram_stride, scale_factor, spad_block_stride, id)
```

Configure mvin instructions.

- `dram_stride` — stride in bytes, with which to load from DRAM.
- `scale_factor` — factor to multiply loaded values; can be negative.
- `spad_block_stride` — when more than `DIM` columns are loaded, the distance in rows between each block of `DIM` columns.
- `id` — id of the mvin instruction; `id = 0` for `mvin`, `1` for `mvin2`, `2` for `mvin3`.

### config_st

```c
#define config_st(dram_stride, scale_factor)
```

Configure the mvout instruction.

- `dram_stride` — stride in bytes, with which to store to DRAM.
- `scale_factor` — factor by which to multiply stored values; can be negative.

## Move-In Instructions

### mvin

```c
#define mvin(dram_addr, spad_acc_addr, cols, rows)
```

Move data from DRAM to scratchpad or accumulator. Configured by `config_ld(..., 0)`.

- `rows` must be less than or equal to `DIM`. If more than `DIM` rows are needed, multiple `mvin` instructions are required.
- `cols` must be less than or equal to `4 * DIM` (when `pe_dim == 16`) or `DIM` otherwise.
- If `dram_addr = 0`, then zeroes are moved into the scratchpad/accumulator (max size `DIM × DIM`).

### mvin2

```c
#define mvin2(dram_addr, spad_acc_addr, cols, rows)
```

Behavior identical to `mvin`, but configured by `config_ld(..., 1)`.

### mvin3

```c
#define mvin3(dram_addr, spad_acc_addr, cols, rows)
```

Behavior identical to `mvin`, but configured by `config_ld(..., 2)`.

## Compute Instructions

Conventions:

- `A` = input matrix, `B` = weight matrix, `C` = output matrix, `D` = optional bias matrix.
- Assume a weight-stationary dataflow.
- `preload`, `compute_preloaded`, and `compute_accumulated` are used to compute `DIM × DIM` matrix multiplications. If no bias, `C = A * B` is computed; with a bias, `C = A * B + D` is computed (note: the bias path can also be used for plain element-wise matrix addition).

### preload

```c
#define preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows)
```

Preload weights `B` onto the `DIM × DIM` systolic array.

- `B` must be preloaded before compute.
- `B` must have been moved into the scratchpad first.
- `B_cols`, `B_rows`, `C_cols`, `C_rows` must each be `≤ DIM`.
- Must run to change the output address to `C_acc_addr`.
- If `B_spad_addr` is unchanged from the previous `preload`, set `B_spad_addr = 0xffffffff`. It must be specified otherwise.

### compute_preloaded

```c
#define compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
```

Compute `A * B (+ D) = C` on the `DIM × DIM` systolic array, with optional bias `D` (which can be used for element-wise addition).

- `A` must have been moved into the scratchpad first.
- This is the first compute after a `preload` to the systolic array.
- Either overwrites or accumulates `C` depending on bit 30 of `C_acc_addr`.
- All size arguments must be `≤ DIM`.
- `bias_spad_addr = 0xffffffff` if there is no bias.
- If a bias is used, `bias_cols` and `bias_rows` are typically equal to `C_cols` and `C_rows` from the `preload` instruction.

### compute_accumulated

```c
#define compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
```

Compute `A * B (+ D) = C` on the `DIM × DIM` systolic array.

- `A` must have been moved into the scratchpad first.
- For weight-stationary dataflow, use this when `B_spad_addr` has not changed.
- Either overwrites or accumulates `C` depending on bit 30 of `C_acc_addr`.
- All size arguments must be `≤ DIM`.
- `bias_spad_addr = 0xffffffff` if there is no bias.
- If a bias is used, `bias_cols` and `bias_rows` are typically equal to `B_cols` and `B_rows` from the `preload` instruction.

## Move-Out and Synchronization

### mvout

```c
#define mvout(dram_addr, spad_acc_addr, cols, rows)
```

Move data from scratchpad or accumulator to DRAM. `cols` and `rows` must each be `≤ DIM`. The DRAM stride and per-element scale are set via `config_st(dram_stride, scale_factor)`.

### fence

```c
#define fence() asm volatile("fence")
```

Inserts a memory fence. Use to synchronize controllers when ordering across the LoadController, ExecuteController, and StoreController is required.
