class HardwareConfig:
    """Base class for hardware configurations."""

    def get_hw_config_specific_rules(self) -> list[str]:
        """Return a list of hardware-config-specific rule strings for LLM prompts."""
        return []

    def get_hw_description(self) -> str:
        """Return a short hardware description string for display/logging."""
        return "Unknown hardware"
