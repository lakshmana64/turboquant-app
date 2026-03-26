"""
TurboQuant CLI - Command Line Tool for Unbiased Quantization

Usage:
    python cli/main.py quantize input.pt --output encoded.pt --qjl_bits 64
"""

import argparse
import torch
from turboquant.sdk.optimize import optimize, TurboQuantizer

def main():
    parser = argparse.ArgumentParser(description="TurboQuant CLI - Unbiased Quantization")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a tensor")
    quant_parser.add_argument("input", type=str, help="Path to input tensor (.pt file)")
    quant_parser.add_argument("--output", type=str, default="encoded.pt", help="Path to output file")
    quant_parser.add_argument("--qjl_bits", type=int, default=64, help="Number of bits for Stage 2 (QJL)")
    quant_parser.add_argument("--sq_bits", type=int, default=2, help="Number of bits for Stage 1 (SQ)")
    
    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate inner product")
    est_parser.add_argument("--query", type=str, required=True, help="Path to query tensor (.pt file)")
    est_parser.add_argument("--encoded", type=str, required=True, help="Path to encoded data (.pt file)")
    
    args = parser.parse_args()
    
    if args.command == "quantize":
        print(f"Loading input: {args.input}")
        x = torch.load(args.input)
        
        print(f"Quantizing with {args.sq_bits} SQ bits and {args.qjl_bits} QJL bits...")
        encoded, quantizer = optimize(x, qjl_bits=args.qjl_bits, sq_bits=args.sq_bits)
        
        # Save components
        save_data = {
            'encoded': encoded,
            'qjl_bits': args.qjl_bits,
            'sq_bits': args.sq_bits,
            'input_dim': x.shape[-1]
        }
        
        torch.save(save_data, args.output)
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
            sq_bits=data['sq_bits']
        )
        
        # Estimate
        estimate = quantizer.estimate_batch(q, data['encoded'])
        print(f"\nEstimate Result (Shape: {estimate.shape}):")
        print(estimate)

if __name__ == "__main__":
    main()
