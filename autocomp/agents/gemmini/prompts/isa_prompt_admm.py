def PROMPT(pe_dim):
    return f"""
```
// defined functions

#define config_ex(dataflow, act, A_stride, A_transpose, B_transpose)
// configure the state of the accelerator
// dataflow is WEIGHT_STATIONARY or OUTPUT_STATIONARY
// act is the activation function, options are NO_ACTIVATION, RELU, LAYERNORM, IGELU, SOFTMAX
// A_stride is the stride with which rows of A in the scratchpad are loaded into the systolic array, during computes. If this stride is 1, then we feed consecutive rows in the scratchpad, starting from the starting address of A, into the systolic array as the A matrix. If the stride is 2, then we feed every other row into the systolic array instead.
// A_transpose is a boolean value that represents whether the matrix A is transposed
// B_transpose is a boolean value that represents whether the matrix B is transposed

#define config_ld(dram_stride, scale_factor, spad_block_stride, id)
// configure mvin instructions
// dram_stride = stride in bytes, with which to load from DRAM
// scale_factor = factor to multiply loaded values; can be negative
// spad_block_stride = when more than DIM columns are loaded, the distance in rows between each block of DIM columns
// id = id of mvin instruction; id = 0 for mvin, 1 for mvin2, 2 for mvin3

#define mvin(dram_addr, spad_acc_addr, cols, rows)
// mvin from DRAM to scratchpad or accumulator
// mvin, configured by config_ld(..., 0)
// rows must be less than or equal to DIM. if more than DIM rows, multiple mvin instructions are needed
// cols must be less than or equal to {"4 * DIM" if pe_dim == 16 else "DIM"}.
// if dram_addr = 0, then zeroes are moved into scratchpad/accumulator, max size DIM x DIM

#define mvin2(dram_addr, spad_acc_addr, cols, rows)
// behavior identical to mvin, but configured by config_ld(..., 1)

#define mvin3(dram_addr, spad_acc_addr, cols, rows)
// behavior identical to mvin, but configured by config_ld(..., 2)

// A = input matrix
// B = weight matrix
// C = output matrix
// assume a weight-stationary dataflow

// preload, compute_preloaded, and compute_accumulated are used to compute DIM x DIM matrix multiplications.
// if no bias, C = A * B is computed; if there is a bias, C = A * B + bias is computed

#define preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows)
// preload weights, B, onto DIM by DIM systolic array
// B must be preloaded before compute
// B must have been moved in to the scratchpad first
// B_cols must be less than or equal to DIM, B_rows must be less than or equal to DIM, C_cols must be less than or equal to DIM, C_rows must be less than or equal to DIM
// must run to change the output address to C_acc_addr 
// if B_spad_addr unchaged from previous preload instruction, can set B_spad_addr = 0xffffffff; must be specified otherwise

#define compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
// compute A * B (+ D) = C on DIM by DIM systolic array, with optional bias D (can be used for element-wise addition)
// A must have been moved in to the scratchpad first
// first compute after preload to systolic array
// either overwrites or accumulates C depending on bit 30 of C_acc_addr
// A_cols must be less than or equal to DIM, A_rows must be less than or equal to DIM, bias_cols must be less than or equal to DIM, bias_rows must be less than or equal to DIM
// bias_spad_addr = 0xffffffff if no bias
// if there is a bias, bias_cols and bias_rows are probably equal to C_cols and C_rows from preload instruction

#define compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) 
// compute A * B (+ D) = C on DIM by DIM systolic array, with optional bias D (can be used for element-wise addition)
// A must have been moved in to the scratchpad first
// for weight stationary, use when B_spad_addr has not changed
// either overwrites or accumulates C depending on bit 30 of C_acc_addr
// A_cols must be less than or equal to DIM, A_rows must be less than or equal to DIM, bias_cols must be less than or equal to DIM, bias_rows must be less than or equal to DIM
// bias_spad_addr = 0xffffffff if no bias
// if there is a bias, bias_cols and bias_rows are probably equal to B_cols and B_rows from preload instruction

#define config_st(dram_stride, scale_factor)
// configure mvout instruction
// dram_stride = stride in bytes, with which to store to DRAM
// scale_factor = factor by which to multiply loaded values; can be negative

#define mvout(dram_addr, spad_acc_addr, cols, rows)
// mvout from scratchpad or accumulator to DRAM
// cols must be less than or equal to DIM
// rows must be less than or equal to DIM

#define fence() asm volatile("fence") 
// fence

'''
Gemmini's private memory is "row-addressed", where each row is DIM elements wide, where DIM is the number of PEs across the width of the systolic array. These elements will be of type inputType in the scratchpad, and of type accType in the accumulator.

Every private Gemmini memory address is 32 bits long. The three most signficant bits are reserved, and have special meanings:

    Bit 31 (the MSB) is 0 if we are addressing the scratchpad, and 1 if we are addressing the accumulator.
    Bit 30 is ignored if we are addressing the scratchpad, or if we are reading from the accumulator. If, instead, we are writing to the accumulator, then bit 30 is 0 if we want to overwrite the data at that address, and 1 if we want to accumulate on top of the data already at that address.
    Bit 29 is ignored if we are addressing the scratchpad, or if we are writing to the accumulator. If, instead, we are reading from the accumulator, then bit 29 is 0 if we want to read scaled-down inputType data from the accumulator, and 1 if we want to read accType data from the accumulator.
        If bit 29 is 1 for an accumulator read address, then we do not apply activation functions or scaling to the output of the accumulator.
'''

'''
Gemmini is a decoupled access/execute architecture, which means that "memory-access" and "execute" instructions happen concurrently, in different regions of the hardware.
It has an ExecuteController (for preload and compute instructions), LoadController (mvin), and StoreController (mvout).
Gemmini includes an ROB which is meant to detect hazards between instructions in different controllers. 
Each controller also handles its own dependencies and hazards internally.
'''
```
"""