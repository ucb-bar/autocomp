## Saturn Vector Unit Architecture

Saturn is a RISC-V Vector (RVV) implementation targeting DSP and domain-specialized workloads. It implements a "short-vector" SIMD-style microarchitecture with efficient dynamic scheduling and configurable datapath characteristics.

### Hardware Configuration

Saturn's key parameters are configurable:

- **VLEN**: Vector register length in bits (total bits per vector register)
- **DLEN**: Datapath width in bits (element-group size per cycle)
- **MLEN**: Memory interface width in bits
- **Chime length**: VLEN/DLEN cycles—the fundamental occupancy unit for vector instructions
- **Issue queue**: Unified, Shared, or Split (controls integer/FP parallelism)

Example: with VLEN=512 and DLEN=128, each vector instruction takes 4 cycles (chime).

### Vector Configuration (LMUL and SEW)

RVV grouping allows efficient use of consecutive vector registers:

- **LMUL**: Length multiplier—groups 1, 2, 4, or 8 consecutive vector registers as one logical register
- **SEW**: Selected Element Width—8, 16, 32, or 64 bits per element
- Higher LMUL increases effective chime time but amortizes loop overhead and enables better utilization
- LMUL=8 with SEW=32 uses all 32 registers, maximizing capacity but reducing instruction diversity

### Execution Units

Saturn includes:

- **Integer pipeline**: Add, subtract, shift, bitwise (configurable latency, typically 1–2 cycles)
- **Multiply pipeline**: Integer multiply (configurable latency, typically 3 cycles)
- **FMA pipeline**: Floating-point multiply-add (configurable latency, typically 4 cycles; requires LMUL ≥ 4 to saturate)
- **Memory unit**: Load/store with support for unit-stride, strided, indexed (gather/scatter), and segmented access
- **Iterative units**: Division and square root (element-wise, variable latency)

### Chaining

Saturn supports **vector chaining** at DLEN granularity:

- Dependent instructions can begin as soon as the first element group is written back
- Enables pipeline depth overlap between load, arithmetic, and store operations
- Critical for achieving full utilization with smaller vector lengths
- Requires interleaving instructions across different sequencers (load vs. execute)

### Memory System

- Unit-stride loads/stores saturate memory bandwidth (preferred)
- Strided and indexed operations: 1 address per cycle (memory-bound)
- Segmented loads/stores convert array-of-structs to struct-of-arrays in hardware
- Masked unit-stride loads ignore mask during memory access; apply at register writeback

### Optimization Principles

The biggest performance levers for Saturn are:

1. **Maximize LMUL** without causing register spilling
2. **Enable chaining** by interleaving loads, arithmetic, and stores
3. **Use unit-stride memory access** to saturate bandwidth
4. **Minimize vsetvl overhead** by keeping it outside inner loops
5. **Accumulate in vector** across iterations, reduce scalar once at the end
6. **Balance across sequencers** (int/FP mix depends on issue queue configuration)
