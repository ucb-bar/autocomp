"""
Hardware configuration for Saturn Vector Unit.

The default configuration matches GENV256D128ShuttleConfig:
- 256-bit VLEN, 128-bit DLEN (dual-issue)
- Separate floating-point and integer vector issue units (split)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SaturnConfig:
    """Hardware configuration for a Saturn vector unit instance.
    
    Default values match the recommended GENV256D128ShuttleConfig:
    dual-issue core with 256-bit VLEN, 128-bit wide SIMD datapath,
    and separate floating-point and integer vector issue units.
    """
    
    # Vector dimensions (bits)
    vlen: int = 256  # Vector register length
    dlen: int = 128  # Datapath width (VLEN/2 for dual-issue)
    mlen: int = 128  # Memory interface width (typically equals DLEN)
    
    # Issue queue configuration
    # "split" = separate fp and int issue units (GENV256D128ShuttleConfig default)
    issue_queue: Literal["unified", "shared", "split"] = "split"
    
    # Execution unit latencies (cycles)
    fma_latency: int = 4
    mul_latency: int = 3
    int_latency: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dlen > self.vlen:
            raise ValueError(f"DLEN ({self.dlen}) cannot exceed VLEN ({self.vlen})")
        if self.vlen % self.dlen != 0:
            raise ValueError(f"VLEN ({self.vlen}) must be divisible by DLEN ({self.dlen})")
    
    @property
    def chime_length(self) -> int:
        """Base chime length in cycles (VLEN/DLEN)."""
        return self.vlen // self.dlen
    
    @property
    def min_lmul_for_fma_saturation(self) -> int:
        """Minimum LMUL needed to hide FMA latency."""
        return max(1, self.fma_latency // self.chime_length)
