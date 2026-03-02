from autocomp.hw_config.hardware_config import HardwareConfig


class GemminiHardwareConfig(HardwareConfig):
    def __init__(self, pe_dim: int, spad_size_kb: int = 256, acc_size_kb: int = 64):
        self.pe_dim = pe_dim
        self.spad_size_kb = spad_size_kb
        self.acc_size_kb = acc_size_kb

    def get_hw_config_specific_rules(self) -> list[str]:
        rules = [
            f"The Gemmini systolic array is {self.pe_dim} by {self.pe_dim}.",
        ]
        if self.pe_dim == 4:
            rules.append("Each element is 4 bytes.")
        else:
            rules.append("Elements in the scratchpad are 1 byte, and elements in the accumulator are 4 bytes.")
        rules.append(f"The scratchpad size is {self.spad_size_kb}KB and the accumulator size is {self.acc_size_kb}KB.")
        return rules

    def get_hw_description(self) -> str:
        return f"Gemmini (pe_dim={self.pe_dim}, spad={self.spad_size_kb}KB, acc={self.acc_size_kb}KB)"
