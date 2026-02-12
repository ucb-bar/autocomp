from autocomp.hw_config.hardware_config import HardwareConfig


class CudaHardwareConfig(HardwareConfig):
    def __init__(self, gpu_name: str, pytorch_version: str, cuda_version: str):
        self.gpu_name = gpu_name
        self.pytorch_version = pytorch_version
        self.cuda_version = cuda_version

    def get_hw_config_specific_rules(self) -> list[str]:
        return [
            f"You will be running the code on an {self.gpu_name} GPU with PyTorch {self.pytorch_version} and CUDA {self.cuda_version}",
        ]

    def get_hw_description(self) -> str:
        return f"{self.gpu_name} (PyTorch {self.pytorch_version}, CUDA {self.cuda_version})"
