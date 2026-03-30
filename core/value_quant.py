"""
Value Quantization for TurboQuant.

Specialized unbiased quantization for 'Value' vectors in KV-caches.
Unlike Keys (inner products), Values are used for weighted sums.
This module ensures that E[quant(V)] = V.
"""

import torch
from torch import Tensor
from typing import Tuple, Dict, Any, Optional
from .scalar_quant import quantize_scalar, dequantize_scalar

class TurboValueCodec:
    """
    Specialized codec for Value vectors (V) in KV-cache.
    Focuses on unbiased vector summation.
    """
    def __init__(self, dim: int, num_bits: int = 4, device: Optional[torch.device] = None):
        self.dim = dim
        self.num_bits = num_bits
        self.device = device or torch.device('cpu')
        
    def encode(self, v: Tensor) -> Dict[str, Any]:
        """
        Encode value vectors with bias correction for summation.
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
            
        # 1. Standard Scalar Quantization
        indices, scales, norms, _ = quantize_scalar(v, self.num_bits)
        v_hat = dequantize_scalar(indices, scales, self.num_bits)
        
        # 2. Summation Bias Correction
        # We ensure the mean of the quantized vector matches the original
        bias = (v - v_hat).mean(dim=-1, keepdim=True)
        
        return {
            'indices': indices,
            'scales': scales,
            'bias': bias,
            'dim': self.dim
        }
        
    def decode(self, encoded: Dict[str, Any]) -> Tensor:
        """
        Decode and apply bias correction.
        """
        v_hat = dequantize_scalar(
            encoded['indices'], 
            encoded['scales'], 
            self.num_bits
        )
        # Apply the mean-preserving bias
        return v_hat + encoded['bias']

def apply_value_quantization(v: Tensor, num_bits: int = 4) -> Tensor:
    """Convenience function for one-shot unbiased value quantization."""
    codec = TurboValueCodec(v.shape[-1], num_bits=num_bits, device=v.device)
    encoded = codec.encode(v)
    return codec.decode(encoded)
