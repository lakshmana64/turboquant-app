"""
Ollama Integration for TurboQuant

Tests TurboQuant's compression accuracy on real embeddings pulled from Ollama.
Requires: A local Ollama instance running (e.g., 'ollama serve').

Usage:
    python ollama_test.py --model llama3 --qjl 64 --sq 4
"""

import requests
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig


def get_ollama_embeddings(prompt: str, model: str = "llama3") -> torch.Tensor:
    """
    Fetch embeddings from local Ollama API.
    
    Args:
        prompt: Text to embed
        model: Ollama model name
        
    Returns:
        Embedding tensor (D,)
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": prompt}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding is None:
            raise ValueError("No 'embedding' field in response")
        return torch.tensor(embedding)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama at localhost:11434")
        print("Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return None


def run_ollama_test(
    model: str = "llama3",
    qjl_dim: int = 64,
    num_bits: int = 4
):
    """
    Run TurboQuant compression test on Ollama embeddings.
    
    Args:
        model: Ollama model name
        qjl_dim: QJL output dimension
        num_bits: Scalar quantization bits
    """
    print(f"--- TurboQuant vs Ollama ({model}) ---")
    print(f"Config: {num_bits}-bit scalar + {qjl_dim}-bit QJL\n")
    
    # 1. Collect real-world data from Ollama
    prompts = [
        "Quantum computing is a type of computing that uses quantum-mechanical phenomena.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the modern world.",
        "A long time ago in a galaxy far, far away...",
        "To be or not to be, that is the question.",
        "Machine learning is a subset of artificial intelligence.",
        "The capital of France is Paris.",
        "Python is a popular programming language for data science.",
    ]
    
    print(f"Fetching embeddings for {len(prompts)} prompts...")
    embeddings = []
    for p in prompts:
        vec = get_ollama_embeddings(p, model)
        if vec is not None:
            embeddings.append(vec)
    
    if not embeddings:
        print("Failed to fetch any embeddings. Exiting.")
        return
    
    # Stack into batch (N, D)
    keys = torch.stack(embeddings)
    queries = keys.clone()  # Self-attention style test
    N, D = keys.shape
    
    print(f"Data loaded: {N} vectors of dimension {D}")
    
    # 2. Create TurboQuant codec
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=42)
    codec = TurboQuantCodec(dim=D, config=config)
    
    # 3. Encode keys (compress KV cache simulation)
    print("\nEncoding with TurboQuant...")
    encoded = codec.encode_keys_batch(keys)
    
    # 4. Compute true and estimated inner products
    print("Computing inner products...")
    true_dots = queries @ keys.T  # (N, N)
    
    turbo_dots = torch.zeros(N, N)
    for i in range(N):
        turbo_dots[i] = codec.estimate_inner_products(queries[i], encoded)
    
    # 5. Compute metrics
    mse = torch.mean((true_dots - turbo_dots) ** 2).item()
    
    # Correlation
    correlation = torch.corrcoef(
        torch.stack([true_dots.view(-1), turbo_dots.view(-1)])
    )[0, 1].item()
    
    # Max absolute error
    max_error = (true_dots - turbo_dots).abs().max().item()
    
    # Mean absolute error
    mae = (true_dots - turbo_dots).abs().mean().item()
    
    # 6. Compression ratio
    original_bits = D * 32  # FP32
    compressed_bits = D * num_bits + qjl_dim
    compression_ratio = compressed_bits / original_bits
    compression_factor = original_bits / compressed_bits
    
    # 7. Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Dimension:         {D}")
    print(f"Num vectors:       {N}")
    print("Original bits:     32.0 per dim (FP32)")
    print(f"TurboQuant bits:   {num_bits}-bit scalar + {qjl_dim}-bit QJL")
    print(f"Bits per dim:      {compressed_bits / D:.2f}")
    print(f"Compression:       {compression_ratio:.2%} ({compression_factor:.1f}x smaller)")
    print()
    print(f"Mean Squared Error:    {mse:.8f}")
    print(f"Mean Absolute Error:   {mae:.6f}")
    print(f"Max Absolute Error:    {max_error:.6f}")
    print(f"Correlation:           {correlation:.6f}")
    print()
    
    # 8. Sample comparison
    print("Sample Inner Products (Top-Left 3x3):")
    print("\nTrue:")
    print(true_dots[:3, :3])
    print("\nTurboQuant:")
    print(turbo_dots[:3, :3])
    print("\nDifference:")
    print((true_dots[:3, :3] - turbo_dots[:3, :3]).abs())
    
    # 9. Attention comparison (softmax)
    scale = 1.0 / (D ** 0.5)
    true_attention = torch.softmax(true_dots * scale, dim=-1)
    turbo_attention = torch.softmax(turbo_dots * scale, dim=-1)
    
    attn_mse = ((true_attention - turbo_attention) ** 2).mean().item()
    attn_cosine = (
        (true_attention.view(-1) @ turbo_attention.view(-1)) /
        (true_attention.view(-1).norm() * turbo_attention.view(-1).norm())
    ).item()
    
    print("\n" + "=" * 50)
    print("ATTENTION COMPARISON (after softmax)")
    print("=" * 50)
    print(f"Attention MSE:       {attn_mse:.8f}")
    print(f"Attention Cosine:    {attn_cosine:.6f}")
    
    # Top-K token agreement
    k = min(3, N)
    agreements = []
    for i in range(N):
        true_topk = true_dots[i].topk(k).indices.sort().values
        turbo_topk = turbo_dots[i].topk(k).indices.sort().values
        agreement = (true_topk == turbo_topk).float().mean().item()
        agreements.append(agreement)
    
    print(f"Top-{k} Agreement:     {sum(agreements) / len(agreements):.2%}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test TurboQuant compression on Ollama embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name (default: llama3)"
    )
    parser.add_argument(
        "--qjl",
        type=int,
        default=64,
        help="QJL output dimension (default: 64)"
    )
    parser.add_argument(
        "--sq",
        type=int,
        default=4,
        help="Scalar quantization bits (default: 4)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Check if Ollama is available
    try:
        response = requests.get(f"{args.url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"Connected to Ollama at {args.url}")
            models = response.json().get("models", [])
            print(f"Available models: {[m['name'] for m in models[:5]]}")
        else:
            print(f"Warning: Ollama returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Warning: Could not connect to Ollama at {args.url}")
        print("Make sure Ollama is running: ollama serve")
        print("\nRunning with synthetic data instead...")
        
        # Fallback: synthetic test
        torch.manual_seed(42)
        D = 4096  # Typical embedding dimension
        N = 10
        keys = torch.randn(N, D)
        queries = keys.clone()
        
        config = TurboQuantConfig(num_bits=args.sq, qjl_dim=args.qjl, seed=42)
        codec = TurboQuantCodec(dim=D, config=config)
        encoded = codec.encode_keys_batch(keys)
        
        true_dots = queries @ keys.T
        turbo_dots = torch.stack([
            codec.estimate_inner_products(queries[i], encoded)
            for i in range(N)
        ])
        
        mse = (true_dots - turbo_dots).pow(2).mean().item()
        corr = torch.corrcoef(torch.stack([true_dots.view(-1), turbo_dots.view(-1)]))[0, 1].item()
        
        print(f"\nSynthetic test ({N}x{D}):")
        print(f"  MSE: {mse:.8f}")
        print(f"  Correlation: {corr:.6f}")
        return
    
    run_ollama_test(model=args.model, qjl_dim=args.qjl, num_bits=args.sq)


if __name__ == "__main__":
    main()
