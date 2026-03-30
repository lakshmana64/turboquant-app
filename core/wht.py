"""
Fast Walsh-Hadamard Transform (FWHT) for TurboQuant.

Provides an O(d log d) rotation instead of the O(d^2) random matrix.
This is the state-of-the-art approach used in the official TurboQuant paper.
"""

import torch
from torch import Tensor
import math

def fast_walsh_hadamard_transform(x: Tensor) -> Tensor:
    """
    Highly optimized, vectorized FWHT in PyTorch.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    
    # Pad to next power of 2
    if (d & (d - 1)) != 0:
        next_p2 = 2**(d - 1).bit_length()
        padding = torch.zeros((*orig_shape[:-1], next_p2 - d), device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-1)
        d = next_p2

    x = x.view(-1, d)
    
    # Vectorized iterative FWHT
    # The trick is to use reshape and sum/diff
    res = x.t()
    d_curr = d
    while d_curr > 1:
        res = res.view(2, d_curr // 2, -1)
        res = torch.stack([res[0] + res[1], res[0] - res[1]], dim=0)
        d_curr //= 2
    
    res = res.view(d, -1).t()
    return res.view(orig_shape[:-1] + (d,)) / math.sqrt(d)

def apply_random_wht(x: Tensor, seed: int = 42) -> Tensor:
    """
    Apply WHT with a random sign flip (Rademacher matrix) to simulate 
    a random orthogonal rotation.
    """
    # 1. Random sign flip
    torch.manual_seed(seed)
    signs = torch.randint(0, 2, (x.shape[-1],), device=x.device, dtype=x.dtype) * 2 - 1
    x = x * signs
    
    # 2. Fast WHT
    return fast_walsh_hadamard_transform(x)
