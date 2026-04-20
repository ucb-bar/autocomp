from autocomp.hw_config.hardware_config import HardwareConfig


class MetalHardwareConfig(HardwareConfig):
    def __init__(self, gpu_name: str, metal_version: str, gpu_family: str, gpu_cores: int):
        self.gpu_name = gpu_name
        self.metal_version = metal_version
        self.gpu_family = gpu_family
        self.gpu_cores = gpu_cores

    def get_hw_config_specific_rules(self) -> list[str]:
        return [
            f"You will be running the code on an {self.gpu_name} GPU with Metal {self.metal_version}, {self.gpu_cores} GPU cores, GPU family {self.gpu_family}.",
        ]

    def get_hw_description(self) -> str:
        return f"{self.gpu_name} (Metal {self.metal_version}, {self.gpu_cores} cores, {self.gpu_family})"
