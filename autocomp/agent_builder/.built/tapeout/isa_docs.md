## Utility Functions

### _sign_extend

def _sign_extend(value: int, length: int):
    """Sign-extends a value of a given bit length to the Python integer width."""
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value

---

### _int_to_le_bytes

def _int_to_le_bytes(data, length: int) -> torch.Tensor:
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Length must be 1, 2, or 4 bytes.")
    return torch.tensor([data], dtype=type_map[length]).view(torch.uint8).clone()

---

### _le_bytes_to_int

def _le_bytes_to_int(tensor: torch.Tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")
    raw_val = tensor.contiguous().view(type_map[length]).item()
    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return int(raw_val) & masks[length]

## Scalar Memory Operations

### lb

| `lb`                     | `I`      | `0000011` | `000`                   |                  | `03/0`     | Load Byte                         | `x[rd] = {{24{VMEM[x[rs1] + imm][7]}}, VMEM[x[rs1] + imm]}` |

---

### lh

| `lh`                     | `I`      | `0000011` | `001`                   |                  | `03/1`     | Load Halfword                     | `x[rd] = {{16{VMEM[x[rs1] + imm][15]}}, VMEM[x[rs1] + imm]}` |

---

### lw

| `lw`                     | `I`      | `0000011` | `010`                   |                  | `03/2`     | Load Word                         | `x[rd] = VMEM[x[rs1] + imm]` |

---

### lbu

| `lbu`                    | `I`      | `0000011` | `100`                   |                  | `03/4`     | Load Byte Unsigned                | `x[rd] = {24'b0, VMEM[x[rs1] + imm]}` |

---

### lhu

| `lhu`                    | `I`      | `0000011` | `101`                   |                  | `03/5`     | Load Halfword Unsigned            | `x[rd] = {16'b0, VMEM[x[rs1] + imm]}` |

---

### sb

| `sb`                     | `S`      | `0100011` | `000`                   |                  | `23/0`     | Store Byte                        | `VMEM[x[rs1] + imm] = x[rs2][7:0]` |

---

### sh

| `sh`                     | `S`      | `0100011` | `001`                   |                  | `23/1`     | Store Halfword                    | `VMEM[x[rs1] + imm] = x[rs2][15:0]` |

---

### sw

| `sw`                     | `S`      | `0100011` | `010`                   |                  | `23/2`     | Store Word                        | `VMEM[x[rs1] + imm] = x[rs2]` |

## Tensor Memory Operations

### vload

| `vload`                  | `VLS`    | `0000111` | `00`                    |                  | `07/0`     | Tensor Load                       | `m[vd] = VMEM[x[rs1] + (imm12 << 5)];` |

---

### vstore

| `vstore`                 | `VLS`    | `0000111` | `01`                    |                  | `07/1`     | Tensor Store                      | `VMEM[x[rs1] + (imm12 << 5)] = m[vd];` |

---

### fence

| `fence`                  | `I`      | `0001111` | `000`                   | `000000000000`   | `0F/0/000` | Fence                             | `np-op` |

## DMA Operations

### dma.load.ch<N>

| `dma.load.ch<N>`         | `R`      | `1111011` | `000 ~ 111`             | `0000000`        | `7B/00`    | DMA Load                          | `issue_dma_load(channel=N, vmem_addr=x[rd], dram_addr={base, x[rs1]}, size=x[rs2]);` |

---

### dma.store.ch<N>

| `dma.store.ch<N>`        | `R`      | `1111011` | `000 ~ 111`             | `0000001`        | `7B/01`    | DMA Store                         | `issue_dma_store(channel=N, vmem_addr=x[rs1], dram_addr={base, x[rd]}, size=x[rs2]);` |

---

### dma.config.ch<N>

| `dma.config.ch<N>`       | `I`      | `1111111` | `000 ~ 111`             | `0000000`        | `7F/00`    | DMA Load                          | `dma.base = x[rs1]` |

---

### dma.wait.ch<N>

| `dma.wait.ch<N>`         | `I`      | `1111111` | `000 ~ 111`             | `0000001`        | `7F/01`    | DMA Wait                          | `wait_until_dma_channel_idle(channel=N);` |

## Scalar Arithmetic and Logic

### add

| `add`                    | `R`      | `0110011` | `000`                   | `0000000`        | `33/0/00`  | Add                               | `x[rd] = x[rs1] + x[rs2]` |

---

### sub

| `sub`                    | `R`      | `0110011` | `000`                   | `0100000`        | `33/0/20`  | Subtract                          | `x[rd] = x[rs1] - x[rs2]` |

---

### addi

| `addi`                   | `I`      | `0010011` | `000`                   |                  | `13/0`     | Add Immediate                     | `x[rd] = x[rs1] + imm` |

---

### sll

| `sll`                    | `R`      | `0110011` | `001`                   | `0000000`        | `33/1/00`  | Shift Left Logical                | `x[rd] = x[rs1] << x[rs2][4:0]` |

---

### slli

| `slli`                   | `I`      | `0010011` | `001`                   | `0000000_shamt`  | `13/1/00`  | Shift Left Logical Immediate      | `x[rd] = x[rs1] << shamt` |

---

### srl

| `srl`                    | `R`      | `0110011` | `101`                   | `0000000`        | `33/5/00`  | Shift Right Logical               | `x[rd] = x[rs1] >> x[rs2][4:0]` |

---

### srli

| `srli`                   | `I`      | `0010011` | `101`                   | `0000000_shamt`  | `13/5/00`  | Shift Right Logical Immediate     | `x[rd] = x[rs1] >> shamt` |

---

### sra

| `sra`                    | `R`      | `0110011` | `101`                   | `0100000`        | `33/5/20`  | Shift Right Arithmetic            | `x[rd] = $signed(x[rs1]) >>> x[rs2][4:0]` |

---

### srai

| `srai`                   | `I`      | `0010011` | `101`                   | `0100000_shamt`  | `13/5/20`  | Shift Right Arithmetic Immediate  | `x[rd] = $signed(x[rs1]) >>> shamt` |

---

### and

| `and`                    | `R`      | `0110011` | `111`                   | `0000000`        | `33/7/00`  | AND                               | `x[rd] = x[rs1] & x[rs2]` |

---

### andi

| `andi`                   | `I`      | `0010011` | `111`                   |                  | `13/7`     | AND Immediate                     | `x[rd] = x[rs1] & imm` |

---

### or

| `or`                     | `R`      | `0110011` | `110`                   | `0000000`        | `33/6/00`  | OR                                | `x[rd] = x[rs1] \| x[rs2]` |

---

### ori

| `ori`                    | `I`      | `0010011` | `110`                   |                  | `13/6`     | OR Immediate                      | `x[rd] = x[rs1] \| imm` |

---

### xor

| `xor`                    | `R`      | `0110011` | `100`                   | `0000000`        | `33/4/00`  | XOR                               | `x[rd] = x[rs1] ^ x[rs2]` |

---

### xori

| `xori`                   | `I`      | `0010011` | `100`                   |                  | `13/4`     | XOR Immediate                     | `x[rd] = x[rs1] ^ imm` |

---

### slt

| `slt`                    | `R`      | `0110011` | `010`                   | `0000000`        | `33/2/00`  | Set Less Than                     | `x[rd] = ($signed(x[rs1]) < $signed(x[rs2]))` |

---

### slti

| `slti`                   | `I`      | `0010011` | `010`                   |                  | `13/2`     | Set Less Than Immediate           | `x[rd] = ($signed(x[rs1]) < $signed(imm))` |

---

### sltu

| `sltu`                   | `R`      | `0110011` | `011`                   | `0000000`        | `33/3/00`  | Set Less Than Unsigned            | `x[rd] = (x[rs1] < x[rs2])` |

---

### sltiu

| `sltiu`                  | `I`      | `0010011` | `011`                   |                  | `13/3`     | Set Less Than Immediate Unsigned  | `x[rd] = (x[rs1] < $unsigned($signed(imm)))` |

---

### lui

| `lui`                    | `U`      | `0110111` |                         |                  | `37`       | Load Upper Immediate              | `x[rd] = {imm[31:12], 12'b0}` |

---

### auipc

| `auipc`                  | `U`      | `0010111` |                         |                  | `17`       | Add Upper Immediate to PC         | `x[rd] = pc + {imm[31:12], 12'b0}` |

## Scale Factor Operations

### seld

| `seld`                   | `I`      | `0000011` | `110`                   |                  | `03/6`     | Scale Factor Load                 | `e[rd] = VMEM[x[rs1] + imm];` |

---

### seli

| `seli`                   | `I`      | `0000011` | `111`                   |                  | `03/7`     | Scale Factor Load Immediate       | `e[rd] = imm;` |

## Control Flow

### beq

| `beq`                    | `B`      | `1100011` | `000`                   |                  | `63/0`     | Branch Equal                      | `if (x[rs1] == x[rs2]) pc = pc + imm after 2 delay slots` |

---

### bne

| `bne`                    | `B`      | `1100011` | `001`                   |                  | `63/1`     | Branch Not Equal                  | `if (x[rs1] != x[rs2]) pc = pc + imm after 2 delay slots` |

---

### blt

| `blt`                    | `B`      | `1100011` | `100`                   |                  | `63/4`     | Branch Less Than                  | `if ($signed(x[rs1]) < $signed(x[rs2])) pc = pc + imm after 2 delay slots` |

---

### bge

| `bge`                    | `B`      | `1100011` | `101`                   |                  | `63/5`     | Branch Greater Or Equal           | `if ($signed(x[rs1]) >= $signed(x[rs2])) pc = pc + imm after 2 delay slots` |

---

### bltu

| `bltu`                   | `B`      | `1100011` | `110`                   |                  | `63/6`     | Branch Less Than Unsigned         | `if (x[rs1] < x[rs2]) pc = pc + imm after 2 delay slots` |

---

### bgeu

| `bgeu`                   | `B`      | `1100011` | `111`                   |                  | `63/7`     | Branch Greater Or Equal Unsigned  | `if (x[rs1] >= x[rs2]) pc = pc + imm after 2 delay slots` |

---

### jal

| `jal`                    | `J`      | `1101111` |                         |                  | `6F`       | Jump And Link                     | `x[rd] = pc + 4; pc = pc + imm after 2 delay slots` |

---

### jalr

| `jalr`                   | `I`      | `1100111` | `000`                   |                  | `67/0`     | Jump And Link Register            | `next_pc = x[rs1] + imm; x[rd] = pc + 4; pc = next_pc after 2 delay slots` |

---

### delay

| `delay`                  | `I`      | `1100111` | `001`                   |                  | `67/1`     | Frontend Delay                    | `hold decode issue for imm cycles;` |

---

### ecall

| `ecall`                  | `I`      | `1110011` | `000`                   | `000000000000`   | `73/0/000` | Environment Call                  | `halt_reason = ECALL; halt = 1'b1;` |

---

### ebreak

| `ebreak`                 | `I`      | `1110011` | `000`                   | `000000000001`   | `73/0/001` | Breakpoint                        | `halt_reason = EBREAK; halt = 1'b1;` |

## Tensor Arithmetic

### vadd.bf16

| `vadd.bf16`              | `VR`     | `1010111` |                         | `0000000`        | `57/00`    | Vector Add                        | `m[vd] = m[vs1].view(bf16) + m[vs2].view(bf16);` |

---

### vsub.bf16

| `vsub.bf16`              | `VR`     | `1010111` |                         | `0000010`        | `57/02`    | Vector Subtract                   | `m[vd] = m[vs1].view(bf16) - m[vs2].view(bf16);` |

---

### vmul.bf16

| `vmul.bf16`              | `VR`     | `1010111` |                         | `0000011`        | `57/03`    | Vector Multiply                   | `m[vd] = m[vs1].view(bf16) * m[vs2].view(bf16);` |

---

### vminimum.bf16

| `vminimum.bf16`          | `VR`     | `1010111` |                         | `0000100`        | `57/04`    | Vector Minimum                    | `m[vd] = min(m[vs1].view(bf16), m[vs2].view(bf16));` |

---

### vmaximum.bf16

| `vmaximum.bf16`          | `VR`     | `1010111` |                         | `0000110`        | `57/06`    | Vector Maximum                    | `m[vd] = max(m[vs1].view(bf16), m[vs2].view(bf16));` |

## Tensor Reduction

### vredsum.bf16

| `vredsum.bf16`           | `VR`     | `1010111` |                         | `0000001`        | `57/01`    | Vector Sublane Reduction Sum      | `m[vd][i, j] = sum(m[vs1].view(bf16)[:, j]);` |

---

### vredmin.bf16

| `vredmin.bf16`           | `VR`     | `1010111` |                         | `0000101`        | `57/05`    | Vector Sublane Reduction Min      | `m[vd][i, j] = min(m[vs1].view(bf16)[:, j]);` |

---

### vredmax.bf16

| `vredmax.bf16`           | `VR`     | `1010111` |                         | `0000111`        | `57/07`    | Vector Sublane Reduction Max      | `m[vd][i, j] = max(m[vs1].view(bf16)[:, j]);` |

---

### vredsum.row.bf16

| `vredsum.row.bf16`       | `VR`     | `1010111` |                         | `0100001`        | `57/21`    | Vector Lane Reduction Sum         | `m[vd][i, j] = sum(m[vs1].view(bf16)[i, :]);` |

---

### vredmin.row.bf16

| `vredmin.row.bf16`       | `VR`     | `1010111` |                         | `0100100`        | `57/24`    | Vector Lane Reduction Min         | `m[vd][i, j] = min(m[vs1].view(bf16)[i, :]);` |

---

### vredmax.row.bf16

| `vredmax.row.bf16`       | `VR`     | `1010111` |                         | `0100110`        | `57/26`    | Vector Lane Reduction Max         | `m[vd][i, j] = max(m[vs1].view(bf16)[i, :]);` |

## Tensor Transcendental and Activation Functions

### vrecip.bf16

| `vrecip.bf16`            | `VR`     | `1010111` |                         | `1000001`        | `57/41`    | Vector Reciprocal                 | `m[vd] = 1.f / m[vs1];` |

---

### vexp.bf16

| `vexp.bf16`              | `VR`     | `1010111` |                         | `1000010`        | `57/42`    | Vector Exponential                | `m[vd] = bf16_exp(m[vs1]);` |

---

### vexp2.bf16

| `vexp2.bf16`             | `VR`     | `1010111` |                         | `1000011`        | `57/43`    | Vector Base 2 Exponential         | `m[vd] = bf16_exp2(m[vs1]);` |

---

### vrelu.bf16

| `vrelu.bf16`             | `VR`     | `1010111` |                         | `1001000`        | `57/48`    | Vector ReLU                       | `m[vd] = bf16_relu(m[vs1]);` |

---

### vsin.bf16

| `vsin.bf16`              | `VR`     | `1010111` |                         | `1001001`        | `57/49`    | Vector Sin                        | `m[vd] = bf16_sin(m[vs1]);` |

---

### vcos.bf16

| `vcos.bf16`              | `VR`     | `1010111` |                         | `1001010`        | `57/4A`    | Vector Cos                        | `m[vd] = bf16_cos(m[vs1]);` |

---

### vtanh.bf16

| `vtanh.bf16`             | `VR`     | `1010111` |                         | `1001011`        | `57/4B`    | Vector Tanh                       | `m[vd] = bf16_tanh(m[vs1]);` |

---

### vlog2.bf16

| `vlog2.bf16`             | `VR`     | `1010111` |                         | `1001100`        | `57/4C`    | Vector log2                       | `m[vd] = bf16_log2(m[vs1]);` |

---

### vsqrt.bf16

| `vsqrt.bf16`             | `VR`     | `1010111` |                         | `1001101`        | `57/4D`    | Vector sqrt                       | `m[vd] = bf16_sqrt(m[vs1]);` |

## Tensor Data Movement and Immediate Loading

### vmov

| `vmov`                   | `VR`     | `1010111` |                         | `1000000`        | `57/40`    | Vector Move                       | `m[vd] = m[vs1];` |

---

### vtrpose.xlu

| `vtrpose.xlu`            | `VR`     | `1101011` |                         | `0000000`        | `6B/00`    | Matrix Transpose                  | `m[vd] = m[vs1].T;` |

---

### vli.all

| `vli.all`                | `VI`     | `1011111` | `000`                   |                  | `5F/0`     | Vector Load Immediate             | `m[vd][:] = imm;` |

---

### vli.row

| `vli.row`                | `VI`     | `1011111` | `001`                   |                  | `5F/1`     | Vector Load Immediate             | `m[vd][0, :] = imm;` |

---

### vli.col

| `vli.col`                | `VI`     | `1011111` | `010`                   |                  | `5F/2`     | Vector Load Immediate             | `m[vd][:, 0] = imm;` |

---

### vli.one

| `vli.one`                | `VI`     | `1011111` | `011`                   |                  | `5F/3`     | Vector Load Immediate             | `m[vd][0, 0] = imm;` |

## Tensor Packing and Quantization

### vpack.bf16.fp8

| `vpack.bf16.fp8`         | `VR`     | `1010111` |                         | `1000100`        | `57/44`    | Vector Packing                    | `m[vd] = quantize(m[vs1], m[vs1+1], e[es1]);` |

---

### vunpack.fp8.bf16

| `vunpack.fp8.bf16`       | `VR`     | `1010111` |                         | `1000101`        | `57/45`    | Vector Unpacking                  | `m[vd], m[vd+1] = dequantize(m[vs1], e[es1]);` |

## Matrix Multiply Unit (MXU) Operations

### vmatpush.weight.mxu0

| `vmatpush.weight.mxu0`   | `VR`     | `1110111` |                         | `0000000`        | `77/00`    | Push Tensor To MXU0 Weight Slot   | `mxu0.w[vd] = m[vs];` |

---

### vmatpush.weight.mxu1

| `vmatpush.weight.mxu1`   | `VR`     | `1110111` |                         | `0000001`        | `77/01`    | Push Tensor To MXU1 Weight Slot   | `mxu1.w[vd] = m[vs];` |

---

### vmatpush.acc.fp8.mxu0

| `vmatpush.acc.fp8.mxu0`  | `VR`     | `1110111` |                         | `0000010`        | `77/02`    | Push Tensor To MXU0 Accumulator   | `mxu0.acc[vd[0]] = dequantize(m[vs]);` |

---

### vmatpush.acc.fp8.mxu1

| `vmatpush.acc.fp8.mxu1`  | `VR`     | `1110111` |                         | `0000011`        | `77/03`    | Push Tensor To MXU1 Accumulator   | `mxu1.acc[vd[0]] = dequantize(m[vs]);` |

---

### vmatpush.acc.bf16.mxu0

| `vmatpush.acc.bf16.mxu0` | `VR`     | `1110111` |                         | `0000100`        | `77/04`    | Push Tensor To MXU0 Accumulator   | `mxu0.acc[vd[0]] = {m[vs], m[vs+1]};` |

---

### vmatpush.acc.bf16.mxu1

| `vmatpush.acc.bf16.mxu1` | `VR`     | `1110111` |                         | `0000101`        | `77/05`    | Push Tensor To MXU1 Accumulator   | `mxu1.acc[vd[0]] = {m[vs], m[vs+1]};` |

---

### vmatpop.fp8.acc.mxu0

| `vmatpop.fp8.acc.mxu0`   | `VR`     | `1110111` |                         | `0000110`        | `77/06`    | Pop MXU0 FP8 Accumulator View     | `m[vd] = quantize_fp8(mxu0.acc[vs2[0]], e[es1]);` |

---

### vmatpop.fp8.acc.mxu1

| `vmatpop.fp8.acc.mxu1`   | `VR`     | `1110111` |                         | `0000111`        | `77/07`    | Pop MXU1 FP8 Accumulator View     | `m[vd] = quantize_fp8(mxu1.acc[vs2[0]], e[es1]);` |

---

### vmatpop.bf16.acc.mxu0

| `vmatpop.bf16.acc.mxu0`  | `VR`     | `1110111` |                         | `0001000`        | `77/08`    | Pop MXU0 BF16 Accumulator         | `{m[vd], m[vd + 1]} = mxu0.acc[vs2[0]];` |

---

### vmatpop.bf16.acc.mxu1

| `vmatpop.bf16.acc.mxu1`  | `VR`     | `1110111` |                         | `0001001`        | `77/09`    | Pop MXU1 BF16 Accumulator         | `{m[vd], m[vd + 1]} = mxu1.acc[vs2[0]];` |

---

### vmatmul.mxu0

| `vmatmul.mxu0`           | `VR`     | `1110111` |                         | `0001010`        | `77/10`    | MXU0 Matmul                       | `mxu0.acc[vd[0]] = m[vs1] @ mxu0.w[vs2[0]];` |

---

### vmatmul.mxu1

| `vmatmul.mxu1`           | `VR`     | `1110111` |                         | `0001011`        | `77/11`    | MXU1 Matmul                       | `mxu1.acc[vd[0]] = m[vs1] @ mxu1.w[vs2[0]];` |

---

### vmatmul.acc.mxu0

| `vmatmul.acc.mxu0`       | `VR`     | `1110111` |                         | `0001100`        | `77/12`    | MXU0 Matmul Accumulate            | `mxu0.acc[vd[0]] = mxu0.acc[vd[0]] + m[vs1] @ mxu0.w[vs2[0]];` |

---

### vmatmul.acc.mxu1

| `vmatmul.acc.mxu1`       | `VR`     | `1110111` |                         | `0001101`        | `77/13`    | MXU1 Matmul Accumulate            | `mxu1.acc[vd[0]] = mxu1.acc[vd[0]] + m[vs1] @ mxu1.w[vs2[0]];` |

## Example instruction call format and ISA behaviors

### SmolVLASiluProgram

```python
"""SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""

instructions: List[Instruction[Any]] = [
    # ── Scalar register setup ──
    Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=VMEM_INPUT_BASE)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=VMEM_OUTPUT_BASE)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_INPUT_BASE)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_OUTPUT_BASE)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=TILE_BYTES)),
    # ── DMA: DRAM → VMEM ──
    Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    # ── Load input to MRF + constants ──
    Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),   # v0 = x
    Instruction(mnemonic="vli.all", args=VectorArgs(vd=1, imm=-1)),          # v1 = -1.0
    Instruction(mnemonic="vli.all", args=VectorArgs(vd=2, imm=1)),           # v2 = +1.0
    # ── SiLU: x / (1 + exp(-x)) ──
    Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=3, vs1=0, vs2=1)),  # v3 = -x
    Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=4, vs1=3)),          # v4 = exp(-x)
    Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=5, vs1=4, vs2=2)),  # v5 = 1+exp(-x)
    Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=6, vs1=5)),        # v6 = sigmoid(x)
    Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=7, vs1=0, vs2=6)),  # v7 = silu(x)
    # ── Store: MRF → VMEM → DRAM ──
    Instruction(mnemonic="vstore", args=VectorArgs(vd=7, rs1=2, imm12=0)),
    Instruction(mnemonic="delay", args=ScalarArgs(imm=20)),
    Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=4, rs1=2, rs2=5, channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
]
```

---

### GemmaMlpProgram

```python
instructions: List[Instruction] = [
    # x1..x4: VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
    # 0x2000
    Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
    # 0x2400
    Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
    # 0x2800 = 0x3000 - 0x800
    Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
    # 0x2C00 = 0x3000 - 0x400
    Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x3)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-1024)),
    # x5..x8: DRAM bases
    Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=DRAM_GATE_WEIGHT_BASE)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=DRAM_UP_WEIGHT_BASE)),
    # DRAM_ACTIVATION_BASE = 0x0800 = 0x1000 - 0x800
    Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)),
    # DRAM_OUTPUT_BASE = 0x0C00 does not fit in signed 12-bit addi; use LUI/ADDI.
    Instruction(mnemonic="lui", args=ScalarArgs(rd=8, imm=0x1)),
    Instruction(mnemonic="addi", args=ScalarArgs(rd=8, rs1=8, imm=-1024)),
    # x9: byte length for fp8 tile (32*32*1 = 1024)
    Instruction(mnemonic="addi", args=ScalarArgs(rd=9, rs1=0, imm=1024)),
    # x10: byte length for bf16 tile (32*16*2 = 1024)
    Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=0, imm=1024)),

    # DRAM -> VMEM
    Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=5, rs2=9, channel=0)),
    Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=6, rs2=9, channel=1)),
    Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=7, rs2=9, channel=2)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=2)),

    # VMEM -> MRF (weights + activation)
    Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),  # gate W (fp8)
    Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),  # up W (fp8)
    Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=3, imm12=0)),  # act (fp8)

    # Push weights to MXU0 WB slots 0 and 1
    Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=0)),
    Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=1, vs1=1)),
    Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),

    # --- PHASE 3: Matrix Multiplications ---
    # Gate projection: activation @ gate_weight -> Acc/MRF
    # Note: Using MatrixArgs for matmul
    Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=0)),
    Instruction(mnemonic="delay", args=ScalarArgs(imm=33)),
    Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=4, vs1=0)),  # gate -> mrf4+5
    # Up projection: activation @ up_weight -> Acc/MRF
    Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
    Instruction(mnemonic="delay", args=ScalarArgs(imm=33)),
    Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=6, vs1=0)),  # up -> mrf6+7
    # --- PHASE 4: Element-wise Multiplication (GeGLU Simplified) ---
    Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=4, vs2=6)),
    Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=9, vs1=5, vs2=7)),
    # --- PHASE 5: Store Results ---
    Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=4, imm12=0)),
    Instruction(mnemonic="vstore", args=VectorArgs(vd=9, rs1=4, imm12=32)),
    Instruction(mnemonic="delay", args=ScalarArgs(imm=40)),

    # VMEM -> DRAM (two 1024B tiles)
    Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=4, imm=1024)),  # vmem+1024
    Instruction(mnemonic="addi", args=ScalarArgs(rd=12, rs1=8, imm=1024)),  # dram+1024
    Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=8, rs1=4, rs2=10, channel=0)),
    Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=12, rs1=11, rs2=10, channel=1)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
]
```

### ISA Implementation
```python
import torch

from typing import Any
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,
    ScalarArgs,
    VectorArgs,
    instr,
    InstructionType,
)
from npu_model.hardware.arch_state import ArchState


PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Helper Functions
# =============================================================================


def _sign_extend(value: int, length: int):
    """Sign-extends a value of a given bit length to the Python integer width."""
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def _int_to_le_bytes(data, length: int) -> torch.Tensor:
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Length must be 1, 2, or 4 bytes.")
    return torch.tensor([data], dtype=type_map[length]).view(torch.uint8).clone()


def _le_bytes_to_int(tensor: torch.Tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")
    raw_val = tensor.contiguous().view(type_map[length]).item()
    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return int(raw_val) & masks[length]


def _tensor_register_bytes(state: ArchState) -> int:
    return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize


def _vmatmul(state: ArchState, unit: str, args: MatrixArgs, accumulate: bool) -> None:
    activation_fp16 = state.read_mrf_fp8(args.vs1).to(torch.float16)
    weight_fp16 = state.read_wb_fp8(unit, args.vs2).to(torch.float16)
    result_fp16 = activation_fp16 @ weight_fp16
    if accumulate:
        result_fp16 = result_fp16 + state.read_acc_bf16(unit, args.vd).to(torch.float16)
    state.write_acc_bf16(unit, args.vd, result_fp16.to(torch.bfloat16))


@instr("lb", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b000)
def lb(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 1))
    state.write_xrf(args.rd, _sign_extend(value, 8))


@instr("lh", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b001)
def lh(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 2))
    state.write_xrf(args.rd, _sign_extend(value, 16))


@instr("lw", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b010)
def lw(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 4))
    state.write_xrf(args.rd, value)


@instr("lbu", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b100)
def lbu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 1))
    state.write_xrf(args.rd, value)


@instr("lhu", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b101)
def lhu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 2))
    state.write_xrf(args.rd, value)


@instr(
    "seld", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b110
)
def seld(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_erf(
        args.rd,
        int(state.read_vmem(state.read_xrf(args.rs1), imm, 1).view(torch.uint8)),
    )


@instr(
    "seli", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b111
)
def seli(state: ArchState, args: ScalarArgs):
    state.write_erf(args.rd, _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "vload", instruction_type=InstructionType.VECTOR.VLS, opcode=0b0000111, funct2=0b00
)
def vload(state: ArchState, args: VectorArgs) -> None:
    addr = state.read_xrf(args.rs1) + (args.imm12 << 5)
    data = state.read_vmem(addr, 0, _tensor_register_bytes(state)).view(torch.uint8)
    state.write_mrf_u8(args.vd, data)


@instr(
    "vstore", instruction_type=InstructionType.VECTOR.VLS, opcode=0b0000111, funct2=0b01
)
def vstore(state: ArchState, args: VectorArgs) -> None:
    addr = state.read_xrf(args.rs1) + (args.imm12 << 5)
    data = state.read_mrf_fp8(args.vd).view(torch.uint8)
    state.write_vmem(addr, 0, data)


@instr(
    "fence", instruction_type=InstructionType.SCALAR.I, opcode=0b0001111, funct3=0b000
)
def fence(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "addi", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b000
)
def addi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "slli", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b001
)
def slli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << (args.imm & 0x3F))


@instr(
    "slti", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b010
)
def slti(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < imm else 0)


@instr(
    "sltiu", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b011
)
def sltiu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = _sign_extend(args.imm & 0xFFF, 12) & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr(
    "xori", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b100
)
def xori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "srli", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b101
)
def srli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> (args.imm & 0x3F))


@instr(
    "srai", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b101
)
def srai(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> (args.imm & 0x3F))


@instr("ori", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b110)
def ori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "andi", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b111
)
def andi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & _sign_extend(args.imm & 0xFFF, 12))


@instr("auipc", instruction_type=InstructionType.SCALAR.U, opcode=0b0010111)
def auipc(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, ((args.imm << 12) & 0xFFFFFFFF) + state.pc)


@instr("sb", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b000)
def sb(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFF, 1),
    )


@instr("sh", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b001)
def sh(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFFFF, 2),
    )


@instr("sw", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b010)
def sw(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFFFFFFFF, 4),
    )


@instr(
    "add",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0000000,
)
def add(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + state.xrf[args.rs2])


@instr(
    "sub",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0100000,
)
def sub(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] - state.xrf[args.rs2])


@instr(
    "sll",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b001,
    funct7=0b0000000,
)
def sll(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << state.xrf[args.rs2])


@instr(
    "slt",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b010,
    funct7=0b0000000,
)
def slt(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < state.xrf[args.rs2] else 0)


@instr(
    "sltu",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b011,
    funct7=0b0000000,
)
def sltu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr(
    "xor",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b100,
    funct7=0b0000000,
)
def xor(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ state.xrf[args.rs2])


@instr(
    "srl",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0000000,
)
def srl(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr(
    "sra",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0100000,
)
def sra(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr(
    "or",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b110,
    funct7=0b0000000,
)
def or_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | state.xrf[args.rs2])


@instr(
    "and",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b111,
    funct7=0b0000000,
)
def and_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & state.xrf[args.rs2])


@instr(
    "lui", instruction_type=InstructionType.SCALAR.U, opcode=0b0110111, funct7=0b0000000
)
def lui(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, (args.imm << 12) & _MASK64)


@instr(
    "vadd.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000000,
)
def vadd_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, (a + b).to(torch.bfloat16))


@instr(
    "vredsum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000001,
)
def vredsum_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = x.sum(dim=0, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vsub.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000010,
)
def vsub_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, (a - b).to(torch.bfloat16))


@instr(
    "vmul.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000011,
)
def vmul_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    result = (a * b).to(torch.bfloat16)
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vminimum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000100,
)
def vminimum_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, torch.minimum(a, b).to(torch.bfloat16))


@instr(
    "vredmin.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000101,
)
def vredmin_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = (
        x.min(dim=0, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vmaximum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000110,
)
def vmaximum_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, torch.maximum(a, b).to(torch.bfloat16))


@instr(
    "vredmax.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000111,
)
def vredmax_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = (
        x.max(dim=0, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vredsum.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100001,
)
def vredsum_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = x.sum(dim=1, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vredmin.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100100,
)
def vredmin_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = (
        x.min(dim=1, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vredmax.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100110,
)
def vredmax_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    result = (
        x.max(dim=1, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    state.write_mrf_bf16(args.vd, result)


@instr(
    "vmov",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000000,
)
def vmov(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, state.read_mrf_bf16(args.vs1))


@instr(
    "vrecip.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000001,
)
def vrecip_bf16(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, 1.0 / state.read_mrf_bf16(args.vs1))


@instr(
    "vexp.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000010,
)
def vexp_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.exp(x).to(torch.bfloat16))


@instr(
    "vexp2.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000011,
)
def vexp2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.exp2(x).to(torch.bfloat16))


@instr(
    "vpack.bf16.fp8",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000100,
)
def vpack_bf16_fp8(state: ArchState, args: VectorArgs) -> None:
    assert args.vs1 != state.cfg.num_m_registers - 1
    scale = state.read_erf(args.es1)
    reg_low = state.read_mrf_bf16(args.vs1)
    reg_high = state.read_mrf_bf16(args.vs1 + 1)
    combined_bf16 = torch.cat([reg_low, reg_high], dim=1)
    quantized_fp8 = (combined_bf16 * scale).to(torch.float8_e4m3fn)
    state.write_mrf_fp8(args.vd, quantized_fp8)


@instr(
    "vunpack.fp8.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000101,
)
def vunpack_fp8_bf16(state: ArchState, args: VectorArgs) -> None:
    assert args.vd != state.cfg.num_m_registers - 1
    scale = state.read_erf(args.es1)
    source_fp8 = state.read_mrf_fp8(args.vs1)
    dequantized_bf16 = source_fp8.to(torch.bfloat16)
    scaled_bf16 = dequantized_bf16 / scale
    reg_low, reg_high = torch.chunk(scaled_bf16, chunks=2, dim=1)
    state.write_mrf_bf16(args.vd, reg_low)
    state.write_mrf_bf16(args.vd + 1, reg_high)


@instr(
    "vrelu.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001000,
)
def vrelu_bf16(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, torch.relu(state.read_mrf_bf16(args.vs1)))


@instr(
    "vsin.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001001,
)
def vsin_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.sin(x).to(torch.bfloat16))


@instr(
    "vcos.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001010,
)
def vcos_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.cos(x).to(torch.bfloat16))


@instr(
    "vtanh.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001011,
)
def vtanh_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.tanh(x).to(torch.bfloat16))


@instr(
    "vlog2.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001100,
)
def vlog2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.log2(x).to(torch.bfloat16))


@instr(
    "vsqrt.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001101,
)
def vsqrt_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.sqrt(x).to(torch.bfloat16))


@instr(
    "vli.all",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b000,
)
def vli_all(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    state.write_mrf_bf16(args.vd, torch.full(shape, args.imm, dtype=torch.bfloat16))


@instr(
    "vli.row",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b001,
)
def vli_row(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, :] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "vli.col",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b010,
)
def vli_col(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[:, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "vli.one",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b011,
)
def vli_one(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "beq", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b000
)
def beq(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] == state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bne", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b001
)
def bne(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] != state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "blt", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b100
)
def blt(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] < state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bge", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b101
)
def bge(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] >= state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bltu", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b110
)
def bltu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a < b:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bgeu", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b111
)
def bgeu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a >= b:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "jalr", instruction_type=InstructionType.SCALAR.I, opcode=0b1100111, funct3=0b000
)
def jalr(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(state.read_xrf(args.rs1) + imm - PIPELINE_LATENCY)


@instr(
    "delay", instruction_type=InstructionType.DELAY.I, opcode=0b1100111, funct3=0b001
)
def delay(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "vtrpose.xlu",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1101011,
    funct7=0b0000000,
)
def vtrpose_xlu(state: ArchState, args: VectorArgs) -> None:
    reg_in = state.read_mrf_fp8(args.vs1)
    transposed = reg_in.view(32, 32).t().contiguous().reshape(-1)
    state.write_mrf_fp8(args.vd, transposed)


@instr("jal", instruction_type=InstructionType.SCALAR.UJ, opcode=0b1101111)
def jal(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFFFF, 20)
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "csrrw", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b001
)
def csrrw(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrs", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b010
)
def csrrs(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old | state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrc", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b011
)
def csrrc(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old & ~state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrwi",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b101,
)
def csrrwi(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, args.rs1 & 0b11111)
    state.write_xrf(args.rd, old)


@instr(
    "csrrsi",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b110,
)
def csrrsi(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old | (args.rs1 & 0b11111))
    state.write_xrf(args.rd, old)


@instr(
    "csrrci",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b100,
)
def csrrci(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old & ~(args.rs1 & 0b11111))
    state.write_xrf(args.rd, old)


@instr(
    "ecall", instruction_type=InstructionType.SCALAR.I, opcode=0b1110011, funct3=0b000
)
def ecall(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "ebreak", instruction_type=InstructionType.SCALAR.I, opcode=0b1110011, funct3=0b000
)
def ebreak(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "vmatpush.weight.mxu0",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000000,
)
def vmatpush_weight_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_wb_u8("mxu0", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr(
    "vmatpush.weight.mxu1",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000001,
)
def vmatpush_weight_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_wb_u8("mxu1", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr(
    "vmatpush.acc.fp8.mxu0",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000010,
)
def vmatpush_acc_fp8_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu0", args.vd, state.read_mrf_fp8(args.vs1).to(torch.bfloat16)
    )


@instr(
    "vmatpush.acc.fp8.mxu1",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000011,
)
def vmatpush_acc_fp8_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu1", args.vd, state.read_mrf_fp8(args.vs1).to(torch.bfloat16)
    )


@instr(
    "vmatpush.acc.bf16.mxu0",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000100,
)
def vmatpush_acc_bf16_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16("mxu0", args.vd, state.read_mrf_bf16_tile(args.vs1))


@instr(
    "vmatpush.acc.bf16.mxu1",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000101,
)
def vmatpush_acc_bf16_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16("mxu1", args.vd, state.read_mrf_bf16_tile(args.vs1))


@instr(
    "vmatpop.fp8.acc.mxu0",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000110,
)
def vmatpop_fp8_acc_mxu0(state: ArchState, args: VectorArgs) -> None:
    quantized = state.read_acc_bf16("mxu0", args.vs1).to(torch.float8_e4m3fn)
    state.write_mrf_u8(args.vd, quantized.view(torch.uint8))


@instr(
    "vmatpop.fp8.acc.mxu1",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0000111,
)
def vmatpop_fp8_acc_mxu1(state: ArchState, args: VectorArgs) -> None:
    quantized = state.read_acc_bf16("mxu1", args.vs1).to(torch.float8_e4m3fn)
    state.write_mrf_fp8(args.vd, quantized.view(torch.uint8))


@instr(
    "vmatpop.bf16.acc.mxu0",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0001000,
)
def vmatpop_bf16_acc_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16_tile(args.vd, state.read_acc_bf16("mxu0", args.vs1))


@instr(
    "vmatpop.bf16.acc.mxu1",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1110111,
    funct7=0b0001001,
)
def vmatpop_bf16_acc_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16_tile(args.vd, state.read_acc_bf16("mxu1", args.vs1))


@instr(
    "vmatmul.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0001010,
)
def vmatmul_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=False)


@instr(
    "vmatmul.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0001011,
)
def vmatmul_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=False)


@instr(
    "vmatmul.acc.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0001100,
)
def vmatmul_acc_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=True)


@instr(
    "vmatmul.acc.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0001101,
)
def vmatmul_acc_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=True)


@instr(
    "dma.load.ch<N>",
    instruction_type=InstructionType.DMA.R,
    opcode=0b1111011,
    funct7=0b0000000,
)
def dma_load_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_dram(state.read_xrf(args.rs1), length)
    state.write_vmem(state.read_xrf(args.rd), 0, data)


@instr(
    "dma.store.ch<N>",
    instruction_type=InstructionType.DMA.R,
    opcode=0b1111011,
    funct7=0b0000001,
)
def dma_store_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_vmem(state.read_xrf(args.rs1), 0, length)
    state.write_dram(state.read_xrf(args.rd), data)


@instr(
    "dma.config.ch<N>",
    instruction_type=InstructionType.DMA.I,
    opcode=0b1111111,
    funct7=0b0000000,
)
def dma_config_ch_n(state: ArchState, args: DmaArgs) -> None:
    state.base = state.read_xrf(args.rs1)


@instr(
    "dma.wait.ch<N>",
    instruction_type=InstructionType.BARRIER.I,
    opcode=0b1111111,
    funct7=0b0000001,
)
def dma_wait_ch_n(state: ArchState, args: DmaArgs) -> None:
    pass
```