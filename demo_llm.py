#!/usr/bin/env python3
"""
TurboQuant LLM Demo

Quick demonstration of TurboQuant compression with real LLM embeddings.

Requirements:
    ollama serve  # Ollama must be running
    
Usage:
    python demo_llm.py
    
Or with custom settings:
    python demo_llm.py --model llama3 --bits 4
"""

import torch
import requests
from turboquant.core.optimized import TurboQuantCodecOptimized


class Colors:
    """ANSI colors for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_result(name: str, value: float, threshold: float, passed: bool):
    """Print test result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"{status}  {name}")
    print(f"       Value: {value:.4f}, Threshold: {threshold:.4f}")


def check_ollama(host: str = "localhost", port: int = 11434) -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_embedding(text: str, model: str = "llama3", host: str = "localhost", port: int = 11434) -> list:
    """Get embedding from Ollama."""
    try:
        response = requests.post(
            f"http://{host}:{port}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        print(f"Error: {e}")
    return []


def demo_semantic_search(model: str = "llama3", bits: int = 4, qjl_dim: int = 64):
    """
    Demo: Semantic search with compressed embeddings.
    
    Shows how TurboQuant preserves semantic similarity rankings.
    """
    print_header("DEMO: Semantic Search")
    
    # Test data
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The cat is sleeping on the sofa.",
        "A feline is resting on the couch.",
        "Python is a popular programming language.",
        "Coding in Python is enjoyable.",
    ]
    
    queries = [
        "What is AI?",
        "Tell me about neural networks",
        "Where is the cat?",
        "I love programming",
    ]
    
    # Check Ollama
    if not check_ollama():
        print(f"{Colors.YELLOW}⚠ Ollama not running. Using synthetic data.{Colors.RESET}")
        # Use synthetic data
        torch.manual_seed(42)
        embeddings = [torch.randn(4096) for _ in documents]
    else:
        print(f"Fetching embeddings from Ollama ({model})...")
        embeddings = []
        for doc in documents:
            emb = get_embedding(doc, model)
            if emb:
                embeddings.append(torch.tensor(emb))
                print(f"  {Colors.GREEN}✓{Colors.RESET} {doc[:40]}...")
    
    if len(embeddings) < 4:
        print(f"{Colors.RED}✗ Not enough embeddings{Colors.RESET}")
        return
    
    dim = embeddings[0].shape[0]
    print(f"\nEmbedding dimension: {dim}")
    
    # Initialize codec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec = TurboQuantCodecOptimized(dim=dim, num_bits=bits, qjl_dim=qjl_dim, device=device)
    
    # Compress
    print(f"\nCompressing with TurboQuant ({bits}-bit + {qjl_dim}-bit QJL)...")
    embed_tensor = torch.stack(embeddings).to(device)
    encoded = codec.encode_keys_batch_optimized(embed_tensor)
    
    # Memory stats
    original_mb = embed_tensor.numel() * 4 / 1e6
    memory_usage = codec.get_memory_usage(len(documents))
    compressed_mb = memory_usage['compressed'] / 1e6
    
    print(f"  Original: {original_mb:.2f} MB")
    print(f"  Compressed: {compressed_mb:.2f} MB")
    print(f"  Compression: {memory_usage['ratio']:.2%} ({1/memory_usage['ratio']:.1f}x smaller)")
    
    # Query
    print(f"\n{Colors.BOLD}Query Results:{Colors.RESET}")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        
        if check_ollama():
            query_emb = get_embedding(query, model)
            if query_emb:
                query_tensor = torch.tensor(query_emb).unsqueeze(0).to(device)
            else:
                continue
        else:
            query_tensor = torch.randn(1, dim).to(device)
        
        # Get scores
        scores = codec.estimate_inner_products_vectorized(query_tensor, encoded)[0]
        
        # Rank
        _, indices = torch.topk(scores, min(3, len(scores)))
        
        for i, idx in enumerate(indices):
            score = scores[idx].item()
            print(f"  {i+1}. [{score:.4f}] {documents[idx][:50]}...")
    
    print()


def demo_code_search(model: str = "llama3", bits: int = 4, qjl_dim: int = 64):
    """
    Demo: Code search with compressed embeddings.
    
    Shows how TurboQuant preserves code semantics.
    """
    print_header("DEMO: Code Search")
    
    code_snippets = [
        "def add(a, b): return a + b",
        "function add(a, b) { return a + b; }",
        "const add = (a, b) => a + b",
        "print('Hello World')",
        "console.log('Hello World')",
        "echo 'Hello World'",
        "SELECT * FROM users WHERE id = 1",
        "db.users.find({id: 1})",
    ]
    
    queries = [
        "How do I add two numbers?",
        "Print hello world",
        "Query database for user",
    ]
    
    # Check Ollama
    if not check_ollama():
        print(f"{Colors.YELLOW}⚠ Ollama not running. Using synthetic data.{Colors.RESET}")
        torch.manual_seed(42)
        embeddings = [torch.randn(4096) for _ in code_snippets]
    else:
        print(f"Fetching code embeddings from Ollama ({model})...")
        embeddings = []
        for code in code_snippets:
            emb = get_embedding(code, model)
            if emb:
                embeddings.append(torch.tensor(emb))
                print(f"  {Colors.GREEN}✓{Colors.RESET} {code[:40]}...")
    
    if len(embeddings) < 4:
        print(f"{Colors.RED}✗ Not enough embeddings{Colors.RESET}")
        return
    
    dim = embeddings[0].shape[0]
    
    # Initialize codec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec = TurboQuantCodecOptimized(dim=dim, num_bits=bits, qjl_dim=qjl_dim, device=device)
    
    # Compress
    embed_tensor = torch.stack(embeddings).to(device)
    encoded = codec.encode_keys_batch_optimized(embed_tensor)
    
    # Query
    print(f"\n{Colors.BOLD}Code Search Results:{Colors.RESET}")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        
        if check_ollama():
            query_emb = get_embedding(query, model)
            if query_emb:
                query_tensor = torch.tensor(query_emb).unsqueeze(0).to(device)
            else:
                continue
        else:
            query_tensor = torch.randn(1, dim).to(device)
        
        scores = codec.estimate_inner_products_vectorized(query_tensor, encoded)[0]
        _, indices = torch.topk(scores, min(3, len(scores)))
        
        for i, idx in enumerate(indices):
            score = scores[idx].item()
            print(f"  {i+1}. [{score:.4f}] {code_snippets[idx]}")
    
    print()


def demo_attention_simulation(seq_len: int = 64, bits: int = 4, qjl_dim: int = 64):
    """
    Demo: Simulated transformer attention with compressed KV cache.
    
    Shows attention score preservation.
    """
    print_header("DEMO: Attention Simulation")
    
    d_model = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Device: {device}")
    
    # Generate random Q, K (simulating transformer)
    queries = torch.randn(seq_len, d_model, device=device)
    keys = torch.randn(seq_len, d_model, device=device)
    
    # Initialize codec
    codec = TurboQuantCodecOptimized(dim=d_model, num_bits=bits, qjl_dim=qjl_dim, device=device)
    
    # True attention
    scale = 1.0 / (d_model ** 0.5)
    true_scores = (queries @ keys.T) * scale
    true_attention = torch.softmax(true_scores, dim=-1)
    
    # Compress keys
    print("\nCompressing KV cache...")
    encoded = codec.encode_keys_batch_optimized(keys)
    
    memory_usage = codec.get_memory_usage(seq_len)
    print(f"  Compression: {memory_usage['ratio']:.2%} ({1/memory_usage['ratio']:.1f}x smaller)")
    
    # Estimated attention
    print("Computing compressed attention...")
    est_scores_list = []
    for i in range(seq_len):
        scores = codec.estimate_inner_products_vectorized(queries[i:i+1], encoded)[0] * scale
        est_scores_list.append(scores)
    
    est_scores = torch.stack(est_scores_list)
    est_attention = torch.softmax(est_scores, dim=-1)
    
    # Metrics
    cos_sim = torch.cosine_similarity(true_attention.view(-1), est_attention.view(-1), dim=0).item()
    attn_mse = ((true_attention - est_attention) ** 2).mean().item()
    
    # Top-K agreement
    k = min(5, seq_len)
    agreements = []
    for i in range(seq_len):
        true_topk = true_attention[i].topk(k).indices.sort().values
        est_topk = est_attention[i].topk(k).indices.sort().values
        agreement = (true_topk == est_topk).float().mean().item()
        agreements.append(agreement)
    
    avg_agreement = sum(agreements) / len(agreements)
    
    # Results
    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print("-" * 60)
    
    passed = cos_sim >= 0.9
    status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.YELLOW}⚠{Colors.RESET}"
    print(f"{status} Cosine Similarity: {cos_sim:.4f} (threshold: 0.9000)")
    
    passed = attn_mse < 0.01
    status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.YELLOW}⚠{Colors.RESET}"
    print(f"{status} Attention MSE: {attn_mse:.6f} (threshold: 0.0100)")
    
    passed = avg_agreement >= 0.8
    status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.YELLOW}⚠{Colors.RESET}"
    print(f"{status} Top-{k} Agreement: {avg_agreement:.4f} (threshold: 0.8000)")
    
    print()


def main():
    """Run all demos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TurboQuant LLM Demo")
    parser.add_argument("--model", default="llama3", help="Ollama model")
    parser.add_argument("--bits", type=int, default=4, help="Scalar bits")
    parser.add_argument("--qjl-dim", type=int, default=64, help="QJL dimension")
    parser.add_argument("--demo", choices=["all", "semantic", "code", "attention"], default="all")
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}TurboQuant LLM Demo{Colors.RESET}")
    print(f"Model: {args.model}")
    print(f"Config: {args.bits}-bit + {args.qjl_dim}-bit QJL")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Check Ollama
    if check_ollama():
        print(f"{Colors.GREEN}✓ Ollama connected{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}⚠ Ollama not running (using synthetic data){Colors.RESET}")
    
    if args.demo in ["all", "semantic"]:
        demo_semantic_search(model=args.model, bits=args.bits, qjl_dim=args.qjl_dim)
    
    if args.demo in ["all", "code"]:
        demo_code_search(model=args.model, bits=args.bits, qjl_dim=args.qjl_dim)
    
    if args.demo in ["all", "attention"]:
        demo_attention_simulation(bits=args.bits, qjl_dim=args.qjl_dim)
    
    print(f"\n{Colors.BOLD}Demo Complete!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
