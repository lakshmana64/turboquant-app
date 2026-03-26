"""
Ollama Plugin CLI

Command-line interface for the Ollama plugin.

Usage:
    # Compress a single prompt
    python -m turboquant.integrations.plugins.ollama compress "Your prompt"
    
    # Query compressed embeddings
    python -m turboquant.integrations.plugins.ollama query "Your query" --prompts "p1" "p2"
    
    # Test connection
    python -m turboquant.integrations.plugins.ollama status
    
    # List available models
    python -m turboquant.integrations.plugins.ollama models
"""

import argparse
import sys


def cmd_status(args):
    """Check Ollama connection status."""
    from .ollama import OllamaPlugin
    
    plugin = OllamaPlugin(
        model=args.model,
        host=args.host,
        port=args.port
    )
    
    if plugin.connect():
        print(f"✓ Connected to Ollama at {plugin.config.base_url}")
        
        # Get available models
        try:
            import requests
            response = requests.get(f"{plugin.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"\nAvailable models ({len(models)}):")
                for m in models[:10]:
                    print(f"  - {m.get('name', 'unknown')}")
                if len(models) > 10:
                    print(f"  ... and {len(models) - 10} more")
        except Exception as e:
            print(f"Could not list models: {e}")
        
        return 0
    else:
        print(f"✗ Could not connect to Ollama at {plugin.config.base_url}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        return 1


def cmd_compress(args):
    """Compress a prompt's embedding."""
    from .ollama import OllamaPlugin
    
    plugin = OllamaPlugin(
        model=args.model,
        host=args.host,
        port=args.port,
        num_bits=args.bits,
        qjl_dim=args.qjl_dim
    )
    
    if not plugin.connect():
        print("Error: Could not connect to Ollama")
        return 1
    
    result = plugin.compress(args.prompt, validate=not args.no_validate)
    
    if result is None:
        print("Error: Failed to compress")
        return 1
    
    print(f"Prompt: {result.prompt}")
    print(f"Dimension: {result.original_dim}")
    print(
        "Compression: "
        f"{result.compression_ratio:.2%} of FP32 "
        f"({result.compression_factor:.2f}x smaller, {result.bits_per_dim:.2f} bits/dim)"
    )
    
    if result.mse is not None:
        print(f"MSE: {result.mse:.8f}")
        print(f"Correlation: {result.correlation:.6f}")
    
    if args.output:
        # Save encoded data
        import torch
        torch.save(result.encoded, args.output)
        print(f"\nSaved to: {args.output}")
    
    return 0


def cmd_query(args):
    """Query against compressed embeddings."""
    from .ollama import OllamaPlugin
    
    plugin = OllamaPlugin(
        model=args.model,
        host=args.host,
        port=args.port,
        num_bits=args.bits,
        qjl_dim=args.qjl_dim
    )
    
    if not plugin.connect():
        print("Error: Could not connect to Ollama")
        return 1
    
    # Compress all prompts
    print(f"Compressing {len(args.prompts)} prompts...")
    results = plugin.compress_batch(args.prompts)
    results = [r for r in results if r is not None]
    
    if not results:
        print("Error: Failed to compress any prompts")
        return 1
    
    # Query
    print(f"Querying: {args.query}")
    matches = plugin.query(
        args.query,
        results,
        top_k=args.top_k,
        scale=1.0 / (results[0].original_dim ** 0.5) if args.scale else None
    )
    
    print(f"\nTop {len(matches)} matches:")
    for match in matches:
        print(f"  {match['rank']}. [{match['score']:.4f}] {match['prompt']}")
    
    return 0


def cmd_models(args):
    """List available Ollama models."""
    import requests
    
    url = f"http://{args.host}:{args.port}/api/tags"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            
            if not models:
                print("No models found")
                return 0
            
            print(f"Available models ({len(models)}):\n")
            
            for m in models:
                name = m.get("name", "unknown")
                size = m.get("size", 0)
                modified = m.get("modified_at", "")[:10]
                
                size_str = f"{size / 1e9:.1f}GB" if size else "N/A"
                
                print(f"  {name}")
                print(f"    Size: {size_str}, Modified: {modified}")
            
            return 0
        else:
            print(f"Error: Status {response.status_code}")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        return 1


def cmd_benchmark(args):
    """Run benchmark on Ollama embeddings."""
    from .ollama import OllamaPlugin
    
    plugin = OllamaPlugin(
        model=args.model,
        host=args.host,
        port=args.port,
        num_bits=args.bits,
        qjl_dim=args.qjl_dim
    )
    
    if not plugin.connect():
        print("Error: Could not connect to Ollama")
        return 1
    
    # Test prompts
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?",
        "What is the capital of France?",
        "Write a haiku about programming",
    ]
    
    print(f"Benchmarking with {len(prompts)} prompts...")
    print(f"Model: {args.model}")
    print(f"Config: {args.bits}-bit + {args.qjl_dim}-bit QJL\n")
    
    results = []
    for prompt in prompts:
        result = plugin.compress(prompt, validate=True)
        if result:
            results.append(result)
            print(f"✓ {prompt[:40]}...")
        else:
            print(f"✗ Failed: {prompt[:40]}...")
    
    if not results:
        print("\nNo results!")
        return 1
    
    # Aggregate stats
    avg_ratio = sum(r.compression_ratio for r in results) / len(results)
    avg_factor = sum(r.compression_factor for r in results) / len(results)
    avg_mse = sum(r.mse or 0 for r in results) / len(results)
    avg_corr = sum(r.correlation or 0 for r in results) / len(results)
    
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Prompts processed: {len(results)}")
    print(f"Average compression: {avg_ratio:.2%} of FP32 ({avg_factor:.2f}x smaller)")
    print(f"Average MSE: {avg_mse:.8f}")
    print(f"Average correlation: {avg_corr:.6f}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Plugin CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument("--host", default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", default="llama3", help="Ollama model")
    parser.add_argument("--bits", type=int, default=4, help="Scalar bits")
    parser.add_argument("--qjl-dim", type=int, default=64, help="QJL dimension")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check connection")
    status_parser.set_defaults(func=cmd_status)
    
    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a prompt")
    compress_parser.add_argument("prompt", help="Text to compress")
    compress_parser.add_argument("-o", "--output", help="Output file")
    compress_parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    compress_parser.set_defaults(func=cmd_compress)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query compressed embeddings")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--prompts", nargs="+", required=True, help="Prompts to query")
    query_parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    query_parser.add_argument("--scale", action="store_true", help="Apply scaling")
    query_parser.set_defaults(func=cmd_query)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List models")
    models_parser.set_defaults(func=cmd_models)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
