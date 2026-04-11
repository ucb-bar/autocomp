from autocomp.hw_config.hardware_config import HardwareConfig

class TapeoutHardwareConfig(HardwareConfig):

    def get_hw_config_specific_rules(self) -> list[str]:
        """Return a list of hardware-config-specific rule strings for LLM prompts."""
        return [
            "You are targeting the latest tapeout_npu model.",
        ]

    def get_hw_description(self) -> str:
        return "The latest tapeout_npu model."
