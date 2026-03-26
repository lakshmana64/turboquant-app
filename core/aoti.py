"""
TurboQuant AOTI Compilation & Export

Provides utilities for:
  - torch.compile() integration for core operations
  - AOTInductor export for production deployment
  - Optimized inference graph generation

Usage:
    from turboquant.core.aoti import compile_codec, export_aot_inductor
    
    codec = TurboQuantCodecOptimized(dim=128, device='cuda')
    compiled_codec = compile_codec(codec)
    
    # Export for deployment
    export_aot_inductor(codec, "turboquant_lib.so")
"""

import torch
import torch._inductor.config
from typing import Dict

from .optimized import TurboQuantCodecOptimized

# Inductor optimizations
torch._inductor.config.cpp_wrapper = True
torch._inductor.config.triton.unique_kernel_names = True

class CompiledTurboQuantCodec:
    """
    TurboQuant codec with torch.compile() optimization.
    
    Wraps an optimized codec and applies torch.compile() to its
    performance-critical methods.
    """
    
    def __init__(
        self,
        codec: TurboQuantCodecOptimized,
        mode: str = "reduce-overhead",
        fullgraph: bool = False
    ):
        """
        Initialize compiled codec.
        
        Args:
            codec: Base optimized codec
            mode: torch.compile() mode ('default', 'reduce-overhead', 'max-autotune')
            fullgraph: Whether to require a full graph (no graph breaks)
        """
        self.codec = codec
        self.mode = mode
        
        # Compile core methods
        self.encode_keys_batch = torch.compile(
            codec.encode_keys_batch_optimized,
            mode=mode,
            fullgraph=fullgraph
        )
        
        self.estimate_inner_products = torch.compile(
            codec.estimate_inner_products_vectorized,
            mode=mode,
            fullgraph=fullgraph
        )
        
        self.decode_keys = torch.compile(
            codec.decode_keys_vectorized,
            mode=mode,
            fullgraph=fullgraph
        )
        
    def __getattr__(self, name):
        """Delegate other attributes to the base codec."""
        return getattr(self.codec, name)


def compile_codec(
    codec: TurboQuantCodecOptimized,
    mode: str = "reduce-overhead"
) -> CompiledTurboQuantCodec:
    """
    Helper to compile a TurboQuant codec.
    """
    return CompiledTurboQuantCodec(codec, mode=mode)


def export_aot_inductor(
    codec: TurboQuantCodecOptimized,
    output_path: str,
    num_keys_example: int = 1024,
    num_queries_example: int = 32
) -> str:
    """
    Export TurboQuant operations using AOTInductor.
    
    This generates a shared library that can be loaded in C++ or other environments
    without a full Python runtime.
    
    Args:
        codec: Codec to export
        output_path: Path to save the .so library
        num_keys_example: Example number of keys for shape inference
        num_queries_example: Example number of queries for shape inference
        
    Returns:
        Path to the generated library
    """
    dim = codec.dim
    device = codec.device
    
    # Create example inputs
    keys = torch.randn(num_keys_example, dim, device=device)
    queries = torch.randn(num_queries_example, dim, device=device)
    
    # Define a module for export
    class TurboQuantModule(torch.nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.codec = codec
            
        def forward(self, keys, queries):
            # Combined encode + estimate
            encoded = self.codec.encode_keys_batch_optimized(keys)
            scores = self.codec.estimate_inner_products_vectorized(queries, encoded)
            return scores

    module = TurboQuantModule(codec)
    
    # AOTInductor export
    with torch.no_grad():
        so_path = torch._inductor.aot_compile(module, (keys, queries))
        
    # Copy to desired output path
    if output_path != so_path:
        import shutil
        shutil.copy(so_path, output_path)
        
    return output_path


def benchmark_compiled(
    dim: int = 4096,
    num_keys: int = 8192,
    num_queries: int = 32,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark eager vs. compiled vs. AOTI.
    """
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        
    codec = TurboQuantCodecOptimized(dim=dim, device=device)
    compiled_codec = compile_codec(codec)
    
    keys = torch.randn(num_keys, dim, device=device)
    queries = torch.randn(num_queries, dim, device=device)
    
    # Warmup
    for _ in range(3):
        _ = codec.encode_keys_batch_optimized(keys)
        _ = compiled_codec.encode_keys_batch(keys)
        
    # Benchmark Eager
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(10):
        encoded = codec.encode_keys_batch_optimized(keys)
        _ = codec.estimate_inner_products_vectorized(queries, encoded)
    end.record()
    torch.cuda.synchronize()
    eager_time = start.elapsed_time(end) / 10
    
    # Benchmark Compiled
    start.record()
    for _ in range(10):
        encoded = compiled_codec.encode_keys_batch(keys)
        _ = compiled_codec.estimate_inner_products(queries, encoded)
    end.record()
    torch.cuda.synchronize()
    compiled_time = start.elapsed_time(end) / 10
    
    return {
        "eager_ms": eager_time,
        "compiled_ms": compiled_time,
        "speedup": eager_time / compiled_time
    }
