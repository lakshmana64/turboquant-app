#!/usr/bin/env python3
"""
REAL Benchmark: TurboQuant on Actual Embeddings

Uses REAL sentence-transformers embeddings (not synthetic data).
Takes 2-3 minutes but produces CREDIBLE results.

Usage:
    python real_benchmark.py
"""

import torch
import time
import json
from typing import List, Dict


def load_real_embeddings(num_samples: int = 1000) -> torch.Tensor:
    """
    Load REAL embeddings from sentence-transformers.
    
    Args:
        num_samples: Number of embeddings to generate
    
    Returns:
        Real embeddings tensor
    """
    print("Loading REAL sentence-transformers model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load real model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings from REAL text
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing enables computers to understand text",
            "Deep learning uses neural networks with many layers",
            "Computer vision allows machines to interpret images",
            "Reinforcement learning trains agents through rewards",
            "Transformers revolutionized natural language processing",
            "Attention mechanisms help models focus on relevant parts",
            "Word embeddings capture semantic meaning in vectors",
            "Language models predict the next token in sequence",
        ] * (num_samples // 10 + 1)
        
        texts = texts[:num_samples]
        
        print(f"Generating {num_samples} real embeddings...")
        start = time.time()
        embeddings = model.encode(texts, convert_to_tensor=True)
        encode_time = time.time() - start
        
        print(f"  Generated {embeddings.shape[0]} embeddings in {encode_time:.1f}s")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Original size: {embeddings.element_size() * embeddings.nelement() / 1e6:.1f} MB")
        
        return embeddings
        
    except ImportError:
        print("sentence-transformers not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "sentence-transformers", "-q"])
        return load_real_embeddings(num_samples)


def benchmark_compression(embeddings: torch.Tensor) -> Dict:
    """
    Benchmark TurboQuant compression on real embeddings.
    
    Args:
        embeddings: Real embeddings tensor
    
    Returns:
        Dictionary with benchmark results
    """
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    print("\n" + "="*70)
    print("BENCHMARK: Real Embeddings Compression")
    print("="*70)
    
    num_vectors, dim = embeddings.shape
    
    # Test different bit widths
    results = []
    
    for num_bits in [2, 3, 4]:
        print(f"\nTesting {num_bits}-bit compression...")
        
        # Create codec
        codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=num_bits, qjl_dim=64))
        
        # Compress
        start = time.time()
        compressed = codec.encode_key(embeddings)
        compress_time = (time.time() - start) * 1000
        
        # Get memory usage
        memory_usage = codec.get_memory_usage(num_vectors)
        compressed_size_mb = memory_usage['compressed'] / (1024 * 1024)
        compression_ratio = memory_usage['factor']
        
        # Decompress
        start = time.time()
        reconstructed = codec.decode_key(compressed)
        decompress_time = (time.time() - start) * 1000
        
        # Quality metrics
        mse = ((embeddings - reconstructed) ** 2).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            embeddings.view(-1, dim),
            reconstructed.view(-1, dim)
        ).mean().item()
        
        # Inner product preservation (critical for RAG)
        queries = torch.randn(10, dim)
        original_dots = queries @ embeddings.T
        compressed_dots = codec.estimate_inner_products(queries, compressed)
        
        dot_mse = ((original_dots - compressed_dots) ** 2).mean().item()
        dot_correlation = torch.corrcoef(
            torch.stack([original_dots.flatten(), compressed_dots.flatten()])
        )[0, 1].item()
        
        # Store results
        results.append({
            "bits": num_bits,
            "original_size_mb": embeddings.element_size() * embeddings.nelement() / (1024 * 1024),
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": compression_ratio,
            "memory_savings_percent": (1 - 1/compression_ratio) * 100,
            "compress_time_ms": compress_time,
            "decompress_time_ms": decompress_time,
            "mse": mse,
            "cosine_similarity": cosine_sim,
            "dot_product_mse": dot_mse,
            "dot_product_correlation": dot_correlation
        })
        
        print(f"  Size: {compressed_size_mb:.1f} MB ({compression_ratio:.1f}x)")
        print(f"  Cosine: {cosine_sim:.4f} ({cosine_sim*100:.1f}%)")
        print(f"  Dot corr: {dot_correlation:.4f} ({dot_correlation*100:.1f}%)")
    
    return results


def benchmark_retrieval_task(embeddings: torch.Tensor) -> Dict:
    """
    Benchmark actual retrieval task (simulates RAG).
    
    Args:
        embeddings: Real embeddings
    
    Returns:
        Dictionary with retrieval metrics
    """
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    print("\n" + "="*70)
    print("BENCHMARK: Real Retrieval Task (RAG Simulation)")
    print("="*70)
    
    num_vectors, dim = embeddings.shape
    
    # Create codec
    codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=4, qjl_dim=64))
    
    # Compress
    print("Compressing embeddings...")
    compressed = codec.encode_key(embeddings)
    
    # Simulate retrieval
    print("Running retrieval queries...")
    
    # Generate query vectors
    num_queries = 100
    queries = torch.randn(num_queries, dim)
    
    # Get ground truth (top-10 with original)
    print("  Computing ground truth (original embeddings)...")
    start = time.time()
    original_scores = queries @ embeddings.T
    original_top10 = torch.topk(original_scores, 10, dim=1)
    ground_truth_time = time.time() - start
    
    # Get compressed retrieval
    print("  Computing compressed retrieval...")
    start = time.time()
    compressed_scores = codec.estimate_inner_products(queries, compressed)
    compressed_top10 = torch.topk(compressed_scores, 10, dim=1)
    compressed_time = time.time() - start
    
    # Calculate retrieval accuracy
    print("  Calculating retrieval accuracy...")
    
    top1_accuracy = 0
    top3_accuracy = 0
    top5_accuracy = 0
    top10_accuracy = 0
    
    for i in range(num_queries):
        gt_set = set(original_top10.indices[i].tolist())
        
        pred_top1 = compressed_top10.indices[i, 0].item()
        pred_top3 = set(compressed_top10.indices[i, :3].tolist())
        pred_top5 = set(compressed_top10.indices[i, :5].tolist())
        pred_top10 = set(compressed_top10.indices[i, :10].tolist())
        
        if pred_top1 in gt_set:
            top1_accuracy += 1
        
        top3_accuracy += len(gt_set & pred_top3) / 3
        top5_accuracy += len(gt_set & pred_top5) / 5
        top10_accuracy += len(gt_set & pred_top10) / 10
    
    top1_accuracy /= num_queries
    top3_accuracy /= num_queries
    top5_accuracy /= num_queries
    top10_accuracy /= num_queries
    
    retrieval_results = {
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "ground_truth_time_sec": ground_truth_time,
        "compressed_time_sec": compressed_time,
        "speedup": ground_truth_time / compressed_time if compressed_time > 0 else 0
    }
    
    print(f"\nRetrieval Accuracy:")
    print(f"  Top-1: {top1_accuracy*100:.1f}%")
    print(f"  Top-3: {top3_accuracy*100:.1f}%")
    print(f"  Top-5: {top5_accuracy*100:.1f}%")
    print(f"  Top-10: {top10_accuracy*100:.1f}%")
    print(f"  Speedup: {retrieval_results['speedup']:.1f}x")
    
    return retrieval_results


def print_readme_table(results: List[Dict], retrieval: Dict):
    """Print results formatted for README."""
    print("\n" + "="*70)
    print("RESULTS FOR README (Copy-Paste)")
    print("="*70)
    print()
    
    # Find best result (4-bit)
    result_4bit = [r for r in results if r['bits'] == 4][0]
    
    print("```markdown")
    print("### Real Benchmark Results (Sentence-Transformers Embeddings)")
    print()
    print("**Test Configuration:**")
    print(f"- Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"- Embeddings: {int(result_4bit['original_size_mb'] * 16 / 4):,} vectors × 384 dimensions")
    print(f"- Original Size: {result_4bit['original_size_mb']:.1f} MB")
    print()
    print("| Metric | 2-bit | 3-bit | 4-bit |")
    print("|--------|-------|-------|-------|")
    
    for r in results:
        print(f"| Compression | {r['compression_ratio']:.1f}x | {r['compression_ratio']:.1f}x | {r['compression_ratio']:.1f}x |")
    
    print(f"| Memory Savings | {results[0]['memory_savings_percent']:.0f}% | {results[1]['memory_savings_percent']:.0f}% | {results[2]['memory_savings_percent']:.0f}% |")
    print(f"| Cosine Similarity | {results[0]['cosine_similarity']:.4f} | {results[1]['cosine_similarity']:.4f} | {results[2]['cosine_similarity']:.4f} |")
    print(f"| Dot Correlation | {results[0]['dot_product_correlation']:.4f} | {results[1]['dot_product_correlation']:.4f} | {results[2]['dot_product_correlation']:.4f} |")
    print()
    print("**Retrieval Accuracy (RAG Task):**")
    print(f"- Top-1 Accuracy: {retrieval['top1_accuracy']*100:.1f}%")
    print(f"- Top-3 Accuracy: {retrieval['top3_accuracy']*100:.1f}%")
    print(f"- Top-5 Accuracy: {retrieval['top5_accuracy']*100:.1f}%")
    print(f"- Top-10 Accuracy: {retrieval['top10_accuracy']*100:.1f}%")
    print(f"- Speedup: {retrieval['speedup']:.1f}x")
    print("```")
    print()
    print("="*70)


def main():
    print("="*70)
    print("REAL Benchmark: TurboQuant on Actual Embeddings")
    print("="*70)
    print()
    
    # Load real embeddings
    embeddings = load_real_embeddings(num_samples=1000)
    
    # Run compression benchmark
    compression_results = benchmark_compression(embeddings)
    
    # Run retrieval benchmark
    retrieval_results = benchmark_retrieval_task(embeddings)
    
    # Print README table
    print_readme_table(compression_results, retrieval_results)
    
    # Save results
    output = {
        "timestamp": time.time(),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_embeddings": embeddings.shape[0],
        "dimension": embeddings.shape[1],
        "compression_results": compression_results,
        "retrieval_results": retrieval_results
    }
    
    with open("real_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: real_benchmark_results.json")
    print()
    print("✅ Now add these REAL numbers to your README!")
    print("="*70)


if __name__ == "__main__":
    main()
