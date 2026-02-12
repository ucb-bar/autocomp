from autocomp.hw_config.hardware_config import HardwareConfig


class TrnHardwareConfig(HardwareConfig):
    def __init__(self, instance_type: str):
        self.instance_type = instance_type

    def get_hw_config_specific_rules(self) -> list[str]:
        return [
            f"You are targeting a {self.instance_type} instance.",
        ]

    def get_hw_description(self) -> str:
        return f"Trainium ({self.instance_type})"
