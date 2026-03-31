"""
Turbo Format Presets for KV Cache Compression.

Pre-defined compression formats matching turboquant_plus:
- turbo2: 2-bit (2.5 bits/val), 6.4x compression vs fp16
- turbo3: 3-bit (3.5 bits/val), 4.6x compression vs fp16
- turbo4: 4-bit (4.25 bits/val), 3.8x compression vs fp16

Reference:
- q8_0: 1.9-2.0x compression
- q4_0: 3.6-4.0x compression
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class TurboFormat:
    """Configuration for a TurboQuant cache format."""
    name: str
    sq_bits: int  # Scalar quantization bits (Stage 1)
    qjl_dim: int  # QJL projection dimension (Stage 2)
    compression_factor: float  # Expected compression vs fp16
    description: str
    
    @property
    def bits_per_dim(self) -> float:
        """Calculate effective bits per dimension."""
        # Total bits = sq_bits * dim + qjl_dim (for signs) + qjl_dim (for norms, 32-bit)
        # For large dim, qjl_dim is negligible, so ~sq_bits + overhead
        return self.sq_bits + (2 * self.qjl_dim) / 1024  # Approximate overhead


# Pre-defined format presets
TURBO2 = TurboFormat(
    name="turbo2",
    sq_bits=2,
    qjl_dim=64,
    compression_factor=6.4,
    description="2-bit format with 6.4x compression (2.5 bits/val effective)"
)

TURBO3 = TurboFormat(
    name="turbo3",
    sq_bits=3,
    qjl_dim=64,
    compression_factor=4.6,
    description="3-bit format with 4.6x compression (3.5 bits/val effective)"
)

TURBO4 = TurboFormat(
    name="turbo4",
    sq_bits=4,
    qjl_dim=64,
    compression_factor=3.8,
    description="4-bit format with 3.8x compression (4.25 bits/val effective)"
)

# Reference formats for comparison
Q8_0 = TurboFormat(
    name="q8_0",
    sq_bits=8,
    qjl_dim=0,  # No QJL for standard quantization
    compression_factor=2.0,
    description="Standard 8-bit quantization (1.9-2.0x compression)"
)

Q4_0 = TurboFormat(
    name="q4_0",
    sq_bits=4,
    qjl_dim=0,
    compression_factor=3.8,
    description="Standard 4-bit quantization (3.6-4.0x compression)"
)

# All available formats
FORMAT_PRESETS: Dict[str, TurboFormat] = {
    "turbo2": TURBO2,
    "turbo3": TURBO3,
    "turbo4": TURBO4,
    "q8_0": Q8_0,
    "q4_0": Q4_0,
}


def get_format(name: str) -> TurboFormat:
    """Get a format preset by name."""
    if name not in FORMAT_PRESETS:
        available = ", ".join(FORMAT_PRESETS.keys())
        raise ValueError(f"Unknown format '{name}'. Available: {available}")
    return FORMAT_PRESETS[name]


def list_formats() -> str:
    """Return a formatted string of all available formats."""
    lines = ["Available TurboQuant Formats:", ""]
    for fmt in FORMAT_PRESETS.values():
        lines.append(f"  {fmt.name:10s} | {fmt.compression_factor:4.1f}x compression | {fmt.sq_bits}-bit SQ | QJL dim: {fmt.qjl_dim}")
        lines.append(f"             | {fmt.description}")
        lines.append("")
    return "\n".join(lines)


def create_codec_from_format(
    format_name: str,
    dim: int,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> 'TurboQuantCodec':
    """
    Create a TurboQuantCodec from a format preset.
    
    Args:
        format_name: Name of the format (turbo2, turbo3, turbo4, etc.)
        dim: Input dimension
        device: Torch device
        seed: Random seed
    
    Returns:
        Configured TurboQuantCodec
    """
    from .codec import TurboQuantConfig, TurboQuantCodec
    
    fmt = get_format(format_name)
    
    config = TurboQuantConfig(
        num_bits=fmt.sq_bits,
        qjl_dim=fmt.qjl_dim,
        seed=seed,
        pack_bits=True,
        rotation_type="hadamard"
    )
    
    return TurboQuantCodec(dim, config=config, device=device)


def calculate_memory_usage(
    format_name: str,
    dim: int,
    num_keys: int,
    baseline: str = "fp16"
) -> Dict[str, Any]:
    """
    Calculate memory usage for a given format.
    
    Args:
        format_name: Format preset name
        dim: Vector dimension
        num_keys: Number of keys to store
        baseline: Baseline format for comparison (fp16, fp32)
    
    Returns:
        Dictionary with memory usage statistics
    """
    fmt = get_format(format_name)
    
    # Baseline memory (fp16 = 16 bits per dim)
    if baseline == "fp16":
        bits_per_dim_baseline = 16
    elif baseline == "fp32":
        bits_per_dim_baseline = 32
    else:
        bits_per_dim_baseline = 16
    
    baseline_bytes = num_keys * dim * bits_per_dim_baseline // 8
    
    # Compressed memory
    # SQ indices: sq_bits per dim
    # QJL signs: 1 bit per qjl_dim
    # QJL norms: 32 bits per qjl_dim
    # Scales: 32 bits per key
    sq_bits_total = num_keys * dim * fmt.sq_bits
    qjl_signs_bits = num_keys * fmt.qjl_dim
    qjl_norms_bits = num_keys * 32
    scales_bits = num_keys * 32
    
    compressed_bits = sq_bits_total + qjl_signs_bits + qjl_norms_bits + scales_bits
    compressed_bytes = compressed_bits // 8
    
    # Add padding for bit-packing alignment
    compressed_bytes = ((compressed_bytes + 31) // 32) * 32
    
    factor = baseline_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    ratio = compressed_bytes / baseline_bytes if baseline_bytes > 0 else 1.0
    
    return {
        "format": fmt.name,
        "baseline_bytes": baseline_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_factor": f"{factor:.1f}x",
        "compression_ratio": f"{ratio:.1%}",
        "savings_mb": (baseline_bytes - compressed_bytes) / (1024 * 1024),
        "bits_per_dim_effective": fmt.bits_per_dim
    }
