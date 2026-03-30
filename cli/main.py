"""
TurboQuant CLI - Command Line Tool for Unbiased Quantization

Usage:
    python cli/main.py quantize input.pt --output encoded.pt --qjl_bits 64 --pack_bits
"""

import argparse
import torch
import sys
import time
from turboquant.sdk.optimize import optimize, TurboQuantizer

def get_memory_usage(obj):
    """Estimate memory usage of tensors in an object."""
    total_bytes = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            total_bytes += get_memory_usage(v)
    elif isinstance(obj, torch.Tensor):
        total_bytes = obj.element_size() * obj.nelement()
    return total_bytes

def main():
    parser = argparse.ArgumentParser(description="TurboQuant CLI - Unbiased Quantization")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a tensor")
    quant_parser.add_argument("input", type=str, help="Path to input tensor (.pt file)")
    quant_parser.add_argument("--output", type=str, default="encoded.pt", help="Path to output file")
    quant_parser.add_argument("--qjl_bits", type=int, default=64, help="Number of bits for Stage 2 (QJL)")
    quant_parser.add_argument("--sq_bits", type=int, default=4, help="Number of bits for Stage 1 (SQ)")
    quant_parser.add_argument("--no_packing", action="store_false", dest="pack_bits", help="Disable bit-packing")
    quant_parser.set_defaults(pack_bits=True)
    
    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate inner product")
    est_parser.add_argument("--query", type=str, required=True, help="Path to query tensor (.pt file)")
    est_parser.add_argument("--encoded", type=str, required=True, help="Path to encoded data (.pt file)")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run memory and accuracy benchmark")
    bench_parser.add_argument("--num_keys", type=int, default=1000, help="Number of keys")
    bench_parser.add_argument("--dim", type=int, default=4096, help="Vector dimension")
    bench_parser.add_argument("--sq_bits", type=int, default=4, help="SQ bits")
    bench_parser.add_argument("--qjl_bits", type=int, default=64, help="QJL bits")
    
    args = parser.parse_args()
    
    if args.command == "quantize":
        print(f"Loading input: {args.input}")
        x = torch.load(args.input)
        
        print(f"Quantizing with {args.sq_bits} SQ bits and {args.qjl_bits} QJL bits...")
        print(f"Bit-packing: {'Enabled' if args.pack_bits else 'Disabled'}")
        
        encoded, quantizer = optimize(
            x, 
            qjl_bits=args.qjl_bits, 
            sq_bits=args.sq_bits,
            pack_bits=args.pack_bits
        )
        
        # Save components
        save_data = {
            'encoded': encoded,
            'qjl_bits': args.qjl_bits,
            'sq_bits': args.sq_bits,
            'input_dim': x.shape[-1],
            'pack_bits': args.pack_bits
        }
        
        torch.save(save_data, args.output)
        
        orig_mem = x.nelement() * x.element_size()
        tq_mem = get_memory_usage(encoded)
        
        print(f"\nResults:")
        print(f"  Memory (Original): {orig_mem / 1024**2:.2f} MB")
        print(f"  Memory (Packed):   {tq_mem / 1024**2:.2f} MB")
        print(f"  Compression:       {orig_mem / tq_mem:.2f}x smaller")
        print(f"Successfully saved encoded data to: {args.output}")
        
    elif args.command == "estimate":
        print(f"Loading encoded data: {args.encoded}")
        data = torch.load(args.encoded)
        
        print(f"Loading query: {args.query}")
        q = torch.load(args.query)
        
        # Re-initialize quantizer
        quantizer = TurboQuantizer(
            data['input_dim'], 
            qjl_bits=data['qjl_bits'], 
            sq_bits=data['sq_bits'],
            pack_bits=data.get('pack_bits', True)
        )
        
        # Estimate
        estimate = quantizer.estimate_batch(q, data['encoded'])
        print(f"\nEstimate Result (Shape: {estimate.shape}):")
        print(estimate)

    elif args.command == "benchmark":
        print(f"Running benchmark: {args.num_keys} keys, {args.dim} dim, {args.sq_bits}-bit SQ")
        
        keys = torch.randn(args.num_keys, args.dim)
        queries = torch.randn(10, args.dim)
        
        start = time.time()
        encoded, quantizer = optimize(
            keys, 
            qjl_bits=args.qjl_bits, 
            sq_bits=args.sq_bits,
            pack_bits=True
        )
        duration = time.time() - start
        
        orig_mem = keys.nelement() * 2 # FP16 baseline
        tq_mem = get_memory_usage(encoded)
        
        print(f"\nBenchmark Results:")
        print(f"  Encoding Time:   {duration:.4f}s")
        print(f"  Memory (FP16):   {orig_mem / 1024**2:.2f} MB")
        print(f"  Memory (TQ):     {tq_mem / 1024**2:.2f} MB")
        print(f"  Savings Factor:  {orig_mem / tq_mem:.2f}x (vs FP16)")
        
        true_dots = queries @ keys.T
        est_dots = quantizer.estimate_batch(queries, encoded)
        mse = torch.mean((true_dots - est_dots)**2).item()
        print(f"  Estimation MSE:  {mse:.6f}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
