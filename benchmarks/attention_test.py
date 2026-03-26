"""
Attention Fidelity Benchmark for TurboQuant

Validates that TurboQuant-compressed attention produces similar outputs
to full-precision attention.

Metrics:
  - MSE between compressed and full-precision attention scores
  - Cosine similarity between score vectors
  - Top-K token selection agreement
"""

import torch
from typing import Dict, Any
import math


def benchmark_attention(
    dim: int = 128,
    num_bits: int = 4,
    qjl_dim: int = 64,
    seq_len: int = 256,
    num_heads: int = 8,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark attention score fidelity.
    
    Compares softmax(Q @ K^T / sqrt(d)) with compressed version.
    
    Args:
        dim: Head dimension
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        seq_len: Sequence length
        num_heads: Number of attention heads
        device: Device to run on
        
    Returns:
        Dict with mse, cosine_similarity, topk_agreement
    """
    from ..core.codec import TurboQuantCodec, TurboQuantConfig
    
    device = torch.device(device)
    torch.manual_seed(42)
    
    # Generate random Q, K
    query = torch.randn(num_heads, dim, device=device)
    keys = torch.randn(seq_len, dim, device=device)
    
    # True attention scores (per head)
    scale = 1.0 / math.sqrt(dim)
    true_scores = torch.zeros(num_heads, seq_len, device=device)
    for h in range(num_heads):
        true_scores[h] = (query[h] @ keys.T) * scale
    
    true_attention = torch.softmax(true_scores, dim=-1)
    
    # Compressed attention
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=42)
    codec = TurboQuantCodec(dim, config, device)
    
    # Encode keys
    encoded_keys = codec.encode_keys_batch(keys)
    
    # Estimate scores
    est_scores = torch.zeros(num_heads, seq_len, device=device)
    for h in range(num_heads):
        est_scores[h] = codec.estimate_inner_products(query[h], encoded_keys) * scale
    
    est_attention = torch.softmax(est_scores, dim=-1)
    
    # Metrics
    # 1. MSE between attention distributions
    mse = ((true_attention - est_attention) ** 2).mean().item()
    
    # 2. Cosine similarity
    true_flat = true_attention.view(-1)
    est_flat = est_attention.view(-1)
    cosine_sim = (true_flat @ est_flat) / (true_flat.norm() * est_flat.norm() + 1e-8)
    cosine_sim = cosine_sim.item()
    
    # 3. Top-K token selection agreement
    topk = min(10, seq_len)
    
    # Average over heads
    agreements = []
    for h in range(num_heads):
        true_topk = true_scores[h].topk(topk).indices.sort().values
        est_topk = est_scores[h].topk(topk).indices.sort().values
        agreement = (true_topk == est_topk).float().mean().item()
        agreements.append(agreement)
    topk_agreement = sum(agreements) / len(agreements)
    
    # 4. Score correlation
    score_corr = torch.corrcoef(
        torch.stack([true_scores.view(-1), est_scores.view(-1)])
    )[0, 1].item()
    
    return {
        'mse': mse,
        'cosine_similarity': cosine_sim,
        'topk_agreement': topk_agreement,
        'score_correlation': score_corr,
        'seq_len': seq_len,
        'num_heads': num_heads
    }


def benchmark_attention_output(
    dim: int = 128,
    num_bits: int = 4,
    qjl_dim: int = 64,
    seq_len: int = 256,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark full attention output (scores @ values).
    
    Args:
        dim: Head dimension
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        seq_len: Sequence length
        device: Device to run on
        
    Returns:
        Dict with output mse, cosine similarity
    """
    from ..core.codec import TurboQuantCodec, TurboQuantConfig
    
    device = torch.device(device)
    torch.manual_seed(42)
    
    # Generate Q, K, V
    query = torch.randn(dim, device=device)
    keys = torch.randn(seq_len, dim, device=device)
    values = torch.randn(seq_len, dim, device=device)
    
    # True attention output
    scale = 1.0 / math.sqrt(dim)
    true_scores = (query @ keys.T) * scale
    true_attention = torch.softmax(true_scores, dim=-1)
    true_output = true_attention @ values
    
    # Compressed
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=42)
    codec = TurboQuantCodec(dim, config, device)
    
    encoded_keys = codec.encode_keys_batch(keys)
    est_scores = codec.estimate_inner_products(query, encoded_keys) * scale
    est_attention = torch.softmax(est_scores, dim=-1)
    est_output = est_attention @ values
    
    # Metrics
    output_mse = ((true_output - est_output) ** 2).mean().item()
    output_cosine = (true_output @ est_output) / (true_output.norm() * est_output.norm() + 1e-8)
    
    return {
        'output_mse': output_mse,
        'output_cosine': output_cosine.item(),
        'score_mse': ((true_scores - est_scores) ** 2).mean().item()
    }


def benchmark_needle_in_haystack(
    dim: int = 128,
    num_bits: int = 4,
    qjl_dim: int = 64,
    seq_len: int = 1024,
    num_trials: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Needle-in-haystack test for attention.
    
    Tests if the model can attend to a specific "needle" token
    among many distractors.
    
    Args:
        dim: Head dimension
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        seq_len: Sequence length (haystack size)
        num_trials: Number of random trials
        device: Device to run on
        
    Returns:
        Dict with needle_recall, needle_rank
    """
    from ..core.codec import TurboQuantCodec, TurboQuantConfig
    
    device = torch.device(device)
    
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=42)
    codec = TurboQuantCodec(dim, config, device)
    scale = 1.0 / math.sqrt(dim)
    
    needle_recalls = []
    needle_ranks = []
    
    for trial in range(num_trials):
        torch.manual_seed(trial)
        
        # Generate random keys
        keys = torch.randn(seq_len, dim, device=device)
        
        # Needle: make one token distinctive
        needle_idx = torch.randint(0, seq_len, (1,)).item()
        needle_vector = torch.randn(dim, device=device) * 3  # Stronger signal
        keys[needle_idx] = needle_vector
        
        # Query matches needle
        query = needle_vector + torch.randn(dim, device=device) * 0.5
        
        # True scores
        true_scores = (query @ keys.T) * scale
        true_rank = (true_scores >= true_scores[needle_idx]).sum().item()
        
        # Compressed
        encoded_keys = codec.encode_keys_batch(keys)
        est_scores = codec.estimate_inner_products(query, encoded_keys) * scale
        est_rank = (est_scores >= est_scores[needle_idx]).sum().item()
        
        # Recall: is needle in top-5?
        top5 = est_scores.topk(5).indices
        needle_recall = 1.0 if needle_idx in top5 else 0.0
        
        needle_recalls.append(needle_recall)
        needle_ranks.append(est_rank)
    
    return {
        'needle_recall': sum(needle_recalls) / len(needle_recalls),
        'needle_avg_rank': sum(needle_ranks) / len(needle_ranks),
        'true_avg_rank': seq_len / 2  # Expected random rank
    }


def run_attention_suite(
    dim: int = 128,
    seq_len: int = 256,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run comprehensive attention fidelity test suite.
    
    Args:
        dim: Head dimension
        seq_len: Sequence length
        device: Device to run on
        
    Returns:
        Dict with all benchmark results
    """
    results = {}
    
    configs = [
        (2, 32),
        (2, 64),
        (4, 64),
        (4, 128),
        (8, 64),
    ]
    
    print("Running attention fidelity benchmark suite...")
    print(f"  Dimension: {dim}, Seq len: {seq_len}")
    print(f"  Device: {device}")
    print()
    
    for num_bits, qjl_dim in configs:
        print(f"  Testing {num_bits}-bit + {qjl_dim}-bit QJL...")
        
        result = benchmark_attention(
            dim=dim,
            num_bits=num_bits,
            qjl_dim=qjl_dim,
            seq_len=seq_len,
            device=device
        )
        
        key = f"{num_bits}b_{qjl_dim}d"
        results[key] = result
        
        print(f"    MSE:              {result['mse']:.6f}")
        print(f"    Cosine sim:       {result['cosine_similarity']:.4f}")
        print(f"    Top-10 agreement: {result['topk_agreement']:.4f}")
        print(f"    Score corr:       {result['score_correlation']:.4f}")
        print()
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--qjl-dim', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    result = benchmark_attention(
        dim=args.dim,
        num_bits=args.bits,
        qjl_dim=args.qjl_dim,
        seq_len=args.seq_len,
        device=args.device
    )
    
    print("\nAttention Fidelity Benchmark Results")
    print("=" * 40)
    print(f"  MSE:              {result['mse']:.6f}")
    print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")
    print(f"  Top-K agreement:   {result['topk_agreement']:.4f}")
    print(f"  Score correlation: {result['score_correlation']:.4f}")
