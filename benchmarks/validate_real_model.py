#!/usr/bin/env python3
"""
Validate TurboQuant on Real Model Data.

Tests TurboQuant quantization on actual model weights and activations
from HuggingFace transformers.

Reference: turboquant_plus/benchmarks/validate_real_model.py
"""

import torch
from typing import Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Validation result for a model layer."""
    layer_name: str
    param_type: str  # 'weight' or 'activation'
    original_shape: List[int]
    mse: float
    cosine: float
    compression_factor: float


def extract_model_data(
    model_name: str = "facebook/opt-125m"
) -> Dict[str, torch.Tensor]:
    """
    Extract weights and activations from a real model.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Dictionary of tensors
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract weights
        data = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                data[name] = param.detach().cpu()
        
        # Generate some activations
        print("Generating activations...")
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract hidden states
        for i, hidden_state in enumerate(outputs.hidden_states):
            data[f"hidden_state_{i}"] = hidden_state[0].cpu()
        
        return data
        
    except ImportError:
        print("Transformers not installed, using synthetic data")
        return generate_synthetic_model_data()


def generate_synthetic_model_data() -> Dict[str, torch.Tensor]:
    """Generate synthetic model-like data."""
    data = {}
    
    # Simulate transformer layers
    for i in range(12):
        # Weights
        data[f"layer.{i}.self_attn.q_proj.weight"] = torch.randn(1024, 1024)
        data[f"layer.{i}.self_attn.k_proj.weight"] = torch.randn(1024, 1024)
        data[f"layer.{i}.self_attn.v_proj.weight"] = torch.randn(1024, 1024)
        data[f"layer.{i}.mlp.fc1.weight"] = torch.randn(4096, 1024)
        data[f"layer.{i}.mlp.fc2.weight"] = torch.randn(1024, 4096)
        
        # Activations
        data[f"layer.{i}.hidden_state"] = torch.randn(50, 1024)
    
    return data


def validate_tensor(
    name: str,
    tensor: torch.Tensor,
    num_bits: int = 4
) -> ValidationResult:
    """
    Validate quantization on a single tensor.
    
    Args:
        name: Tensor name
        tensor: Tensor to validate
        num_bits: Quantization bits
    
    Returns:
        ValidationResult
    """
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    # Flatten for validation
    if tensor.dim() > 2:
        tensor = tensor.view(-1, tensor.shape[-1])
    
    dim = tensor.shape[-1]
    
    # Create codec
    codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=num_bits))
    
    # Quantize and dequantize
    encoded = codec.encode_key(tensor)
    decoded = codec.decode_key(encoded)
    
    # Calculate metrics
    mse = ((tensor - decoded) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        tensor.view(-1, dim), decoded.view(-1, dim)
    ).mean().item()
    
    # Compression factor
    compression_factor = codec.compression_factor
    
    # Determine if weight or activation
    param_type = "activation" if "hidden" in name or "state" in name else "weight"
    
    return ValidationResult(
        layer_name=name,
        param_type=param_type,
        original_shape=list(tensor.shape),
        mse=mse,
        cosine=cosine,
        compression_factor=compression_factor
    )


def run_validation(
    model_name: str = "facebook/opt-125m",
    num_bits: int = 4
) -> List[ValidationResult]:
    """
    Run validation on model data.
    
    Args:
        model_name: Model to validate
        num_bits: Quantization bits
    
    Returns:
        List of ValidationResult
    """
    print("="*70)
    print("Real Model Validation")
    print("="*70)
    print()
    
    # Extract model data
    data = extract_model_data(model_name)
    
    print(f"Found {len(data)} tensors")
    print()
    
    # Validate each tensor
    results = []
    
    for name, tensor in list(data.items())[:20]:  # Limit to 20 tensors
        print(f"Validating {name}...")
        result = validate_tensor(name, tensor, num_bits)
        results.append(result)
        
        print(f"  Shape: {result.original_shape}")
        print(f"  MSE: {result.mse:.6f}")
        print(f"  Cosine: {result.cosine:.4f}")
        print(f"  Compression: {result.compression_factor:.1f}x")
        print()
    
    return results


def print_summary(results: List[ValidationResult]):
    """Print validation summary."""
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    # Separate weights and activations
    weights = [r for r in results if r.param_type == "weight"]
    activations = [r for r in results if r.param_type == "activation"]
    
    if weights:
        avg_mse_w = sum(r.mse for r in weights) / len(weights)
        avg_cosine_w = sum(r.cosine for r in weights) / len(weights)
        print(f"Weights ({len(weights)} tensors):")
        print(f"  Average MSE: {avg_mse_w:.6f}")
        print(f"  Average Cosine: {avg_cosine_w:.4f}")
        print()
    
    if activations:
        avg_mse_a = sum(r.mse for r in activations) / len(activations)
        avg_cosine_a = sum(r.cosine for r in activations) / len(activations)
        print(f"Activations ({len(activations)} tensors):")
        print(f"  Average MSE: {avg_mse_a:.6f}")
        print(f"  Average Cosine: {avg_cosine_a:.4f}")
        print()
    
    overall_avg_mse = sum(r.mse for r in results) / len(results)
    overall_avg_cosine = sum(r.cosine for r in results) / len(results)
    
    print(f"Overall ({len(results)} tensors):")
    print(f"  Average MSE: {overall_avg_mse:.6f}")
    print(f"  Average Cosine: {overall_avg_cosine:.4f}")
    print()
    print("="*70)


def main():
    """Run real model validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate TurboQuant on real models")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                       help="Model name to validate")
    parser.add_argument("--bits", type=int, default=4,
                       help="Quantization bits")
    
    args = parser.parse_args()
    
    results = run_validation(args.model, args.bits)
    print_summary(results)
    
    # Save results
    output = {
        "benchmark": "validate_real_model",
        "model": args.model,
        "num_bits": args.bits,
        "timestamp": time.time(),
        "results": [
            {
                "layer_name": r.layer_name,
                "param_type": r.param_type,
                "original_shape": r.original_shape,
                "mse": r.mse,
                "cosine": r.cosine,
                "compression_factor": r.compression_factor
            }
            for r in results
        ]
    }
    
    with open("benchmark_validate_real_model_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: benchmark_validate_real_model_results.json")


if __name__ == "__main__":
    main()
