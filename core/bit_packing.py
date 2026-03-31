"""
Bit-packing utilities for TurboQuant.

Allows packing low-bit indices (1, 2, 4 bits) into uint8 tensors
to achieve theoretical memory savings.
"""

import torch
from torch import Tensor
import math


def pack_bits(x: Tensor, bits: int) -> Tensor:
    """
    Pack low-bit integer tensor into uint8.
    
    Args:
        x: Integer tensor of shape (..., d)
        bits: Number of bits per element (must be 1, 2, 4, or 8)
        
    Returns:
        Packed uint8 tensor of shape (..., ceil(d * bits / 8))
    """
    if bits == 8:
        return x.to(torch.uint8)
    
    if x.numel() == 0:
        return torch.empty((*x.shape[:-1], 0), dtype=torch.uint8, device=x.device)

    if bits not in [1, 2, 4]:
        raise ValueError(f"Unsupported bit width for packing: {bits}. Must be 1, 2, or 4.")
    
    orig_shape = x.shape
    d = orig_shape[-1]
    elements_per_byte = 8 // bits
    
    # Flatten to (N, d)
    x_flat = x.view(-1, d).to(torch.uint8)
    N = x_flat.shape[0]
    
    # Calculate padded length to be multiple of elements_per_byte
    padded_d = math.ceil(d / elements_per_byte) * elements_per_byte
    if padded_d > d:
        padding = torch.zeros((N, padded_d - d), dtype=torch.uint8, device=x.device)
        x_flat = torch.cat([x_flat, padding], dim=1)
    
    # Reshape to (N, num_bytes, elements_per_byte)
    num_bytes = padded_d // elements_per_byte
    x_reshaped = x_flat.view(N, num_bytes, elements_per_byte)
    
    # Create bitmask shifts
    shifts = torch.arange(0, 8, bits, device=x.device, dtype=torch.uint8)
    # shifts: [0, 2, 4, 6] for 2-bit
    
    # Pack
    packed = torch.zeros((N, num_bytes), dtype=torch.uint8, device=x.device)
    for i in range(elements_per_byte):
        packed |= x_reshaped[:, :, i] << shifts[i]
        
    # Restore batch dims if any
    new_shape = list(orig_shape[:-1]) + [num_bytes]
    return packed.view(*new_shape)


def unpack_bits(packed: Tensor, bits: int, original_dim: int) -> Tensor:
    """
    Unpack uint8 tensor back into low-bit integers.
    
    Args:
        packed: Packed uint8 tensor
        bits: Number of bits per element
        original_dim: The original dimension d before packing
        
    Returns:
        Unpacked integer tensor of shape (..., original_dim)
    """
    if bits == 8:
        return packed.to(torch.int64)
        
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], original_dim), dtype=torch.int64, device=packed.device)

    if bits not in [1, 2, 4]:
        raise ValueError(f"Unsupported bit width for unpacking: {bits}")
        
    orig_shape = packed.shape
    num_bytes = orig_shape[-1]
    elements_per_byte = 8 // bits
    
    # Flatten
    packed_flat = packed.view(-1, num_bytes)
    N = packed_flat.shape[0]
    
    # Unpack
    unpacked = torch.zeros((N, num_bytes, elements_per_byte), dtype=torch.uint8, device=packed.device)
    
    mask = (1 << bits) - 1
    shifts = torch.arange(0, 8, bits, device=packed.device, dtype=torch.uint8)
    
    for i in range(elements_per_byte):
        unpacked[:, :, i] = (packed_flat >> shifts[i]) & mask
        
    # Reshape and truncate padding
    unpacked_flat = unpacked.view(N, -1)
    unpacked_flat = unpacked_flat[:, :original_dim]
    
    # Restore shape
    new_shape = list(orig_shape[:-1]) + [original_dim]
    return unpacked_flat.view(*new_shape).to(torch.int64)


def pack_signs(signs: Tensor) -> Tensor:
    """
    Pack {-1, 1} signs into 1-bit packed uint8.
    
    Args:
        signs: Tensor of shape (..., m) containing -1 or 1
        
    Returns:
        Packed uint8 tensor
    """
    # Map {-1, 1} to {0, 1}
    binary = (signs > 0).to(torch.uint8)
    return pack_bits(binary, 1)


def unpack_signs(packed: Tensor, original_dim: int) -> Tensor:
    """
    Unpack 1-bit uint8 back into {-1, 1} signs.
    
    Args:
        packed: Packed uint8 tensor
        original_dim: Original dimension m
        
    Returns:
        Sign tensor of shape (..., original_dim) containing -1.0 or 1.0
    """
    binary = unpack_bits(packed, 1, original_dim)
    return (binary.to(torch.float32) * 2.0 - 1.0)
