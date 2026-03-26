"""
LLM Integration Tests for TurboQuant

Tests TurboQuant compression with real LLM workloads:
  - Ollama embedding compression
  - Attention fidelity in transformer context
  - KV cache compression during generation
  - RAG application quality

Requirements:
    pip install ollama requests

Usage:
    python -m turboquant.benchmarks.llm_tests
"""

import torch
import requests
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from turboquant.core.optimized import TurboQuantCodecOptimized
from turboquant.core.mixed_precision import MixedPrecisionCodec


# Test prompts for different scenarios
TEST_PROMPTS = {
    "semantic_similarity": [
        "The cat sat on the mat.",
        "A feline is resting on a rug.",
        "The dog barked at the mailman.",
        "Programming in Python is fun.",
        "Coding with Python is enjoyable.",
        "JavaScript is different from Java.",
    ],
    "rag_queries": [
        "What is machine learning?",
        "Explain neural networks",
        "How does photosynthesis work?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ],
    "code_semantics": [
        "def add(a, b): return a + b",
        "function add(a, b) { return a + b; }",
        "const add = (a, b) => a + b",
        "print('Hello World')",
        "console.log('Hello World')",
        "SELECT * FROM users WHERE id = 1",
    ],
}


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OllamaClient:
    """Simple Ollama client for testing."""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_embedding(self, text: str, model: str = "llama3") -> Optional[List[float]]:
        """Get embedding from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except Exception as e:
            print(f"Error getting embedding: {e}")
        return None
    
    def generate(self, prompt: str, model: str = "llama3") -> Optional[str]:
        """Generate text from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            print(f"Error generating: {e}")
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception:
            pass
        return []


class LLMTestSuite:
    """
    Comprehensive LLM-based test suite for TurboQuant.
    
    Tests:
      1. Semantic similarity preservation
      2. RAG retrieval quality
      3. Code semantics preservation
      4. Attention fidelity
      5. End-to-end generation quality
    """
    
    def __init__(
        self,
        dim: int = 4096,  # Typical LLM embedding dimension
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize test suite.
        
        Args:
            dim: Embedding dimension
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Target device
        """
        self.dim = dim
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize codecs
        self.codec = TurboQuantCodecOptimized(
            dim=dim,
            num_bits=num_bits,
            qjl_dim=qjl_dim,
            device=self.device
        )
        
        self.mp_codec = MixedPrecisionCodec(
            dim=dim,
            key_dtype='fp8',
            query_dtype='fp16',
            device=self.device
        )
        
        # Ollama client
        self.ollama = OllamaClient()
        
        # Results storage
        self.results: List[TestResult] = []
    
    def test_semantic_similarity(
        self,
        model: str = "llama3",
        threshold: float = 0.85
    ) -> TestResult:
        """
        Test: Compressed embeddings preserve semantic similarity.
        
        Method:
          1. Get embeddings for similar sentence pairs
          2. Compress with TurboQuant
          3. Compare similarity rankings
        """
        print("\n" + "="*60)
        print("TEST: Semantic Similarity Preservation")
        print("="*60)
        
        if not self.ollama.is_available():
            return TestResult(
                name="semantic_similarity",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Ollama not available"
            )
        
        # Get embeddings
        prompts = TEST_PROMPTS["semantic_similarity"]
        embeddings = []
        
        print(f"Fetching {len(prompts)} embeddings from Ollama...")
        for prompt in prompts:
            emb = self.ollama.get_embedding(prompt, model)
            if emb:
                embeddings.append(torch.tensor(emb, device=self.device))
                print(f"  ✓ {prompt[:40]}...")
        
        if len(embeddings) < 4:
            return TestResult(
                name="semantic_similarity",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Not enough embeddings fetched"
            )
        
        # Update codec dimension if needed
        if embeddings[0].shape[0] != self.dim:
            self.dim = embeddings[0].shape[0]
            self.codec = TurboQuantCodecOptimized(
                dim=self.dim,
                num_bits=self.num_bits,
                qjl_dim=self.qjl_dim,
                device=self.device
            )
        
        # Compute true similarities
        true_sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0)
                true_sims.append((i, j, sim.item()))
        
        # Compress embeddings
        embed_tensor = torch.stack(embeddings)
        encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
        
        # Compute compressed similarities
        compressed_sims = []
        for i in range(len(embeddings)):
            query = embeddings[i].unsqueeze(0)
            scores = self.codec.estimate_inner_products_vectorized(query, encoded)[0]
            # Normalize
            scores = scores / (embeddings[i].norm() * encoded['original_norms'].squeeze())
            for j in range(len(embeddings)):
                if i < j:
                    compressed_sims.append((i, j, scores[j].item()))
        
        # Compare rankings (Kendall tau correlation)
        from scipy.stats import kendalltau
        
        true_vals = [s[2] for s in sorted(true_sims)]
        compressed_vals = [s[2] for s in sorted(compressed_sims)]
        
        if len(true_vals) > 2:
            correlation, _ = kendalltau(true_vals, compressed_vals)
        else:
            correlation = 1.0
        
        passed = correlation >= threshold
        
        result = TestResult(
            name="semantic_similarity",
            passed=passed,
            metric_name="kendall_tau",
            metric_value=correlation,
            threshold=threshold,
            details=f"Correlation: {correlation:.4f}"
        )
        
        self.results.append(result)
        print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  Kendall Tau Correlation: {correlation:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        
        return result
    
    def test_rag_retrieval(
        self,
        model: str = "llama3",
        top_k: int = 3,
        threshold: float = 0.7
    ) -> TestResult:
        """
        Test: RAG retrieval quality with compressed embeddings.
        
        Method:
          1. Create document embeddings
          2. Compress with TurboQuant
          3. Query and compare top-k results
        """
        print("\n" + "="*60)
        print("TEST: RAG Retrieval Quality")
        print("="*60)
        
        if not self.ollama.is_available():
            return TestResult(
                name="rag_retrieval",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Ollama not available"
            )
        
        # Use queries as both documents and queries
        documents = TEST_PROMPTS["rag_queries"]
        queries = TEST_PROMPTS["rag_queries"]
        
        print(f"Indexing {len(documents)} documents...")
        
        # Get document embeddings
        doc_embeddings = []
        for doc in documents:
            emb = self.ollama.get_embedding(doc, model)
            if emb:
                doc_embeddings.append(torch.tensor(emb, device=self.device))
        
        if len(doc_embeddings) < 3:
            return TestResult(
                name="rag_retrieval",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Not enough documents indexed"
            )
        
        # Update codec dimension
        if doc_embeddings[0].shape[0] != self.dim:
            self.dim = doc_embeddings[0].shape[0]
            self.codec = TurboQuantCodecOptimized(
                dim=self.dim,
                num_bits=self.num_bits,
                qjl_dim=self.qjl_dim,
                device=self.device
            )
        
        # Compress documents
        doc_tensor = torch.stack(doc_embeddings)
        encoded = self.codec.encode_keys_batch_optimized(doc_tensor)
        
        # Query and evaluate
        print(f"Querying with {len(queries)} queries...")
        recall_scores = []
        
        for i, query_text in enumerate(queries):
            query_emb = self.ollama.get_embedding(query_text, model)
            if not query_emb:
                continue
            
            query_tensor = torch.tensor(query_emb, device=self.device)
            
            # Get compressed scores
            scores = self.codec.estimate_inner_products_vectorized(
                query_tensor.unsqueeze(0),
                encoded
            )[0]
            
            # Get top-k from compressed
            _, compressed_topk = torch.topk(scores, min(top_k, len(scores)))
            compressed_set = set(compressed_topk.tolist())
            
            # Get true top-k
            true_scores = query_tensor @ torch.stack(doc_embeddings).T
            _, true_topk = torch.topk(true_scores, min(top_k, len(true_scores)))
            true_set = set(true_topk.tolist())
            
            # Recall@K
            intersection = len(compressed_set & true_set)
            recall = intersection / min(top_k, len(true_set))
            recall_scores.append(recall)
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        passed = avg_recall >= threshold
        
        result = TestResult(
            name="rag_retrieval",
            passed=passed,
            metric_name="recall_at_k",
            metric_value=avg_recall,
            threshold=threshold,
            details=f"Avg Recall@{top_k}: {avg_recall:.4f}"
        )
        
        self.results.append(result)
        print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  Recall@{top_k}: {avg_recall:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        
        return result
    
    def test_code_semantics(
        self,
        model: str = "llama3",
        threshold: float = 0.8
    ) -> TestResult:
        """
        Test: Code semantics preservation.
        
        Method:
          1. Get embeddings for code snippets
          2. Verify similar code has similar embeddings
          3. Check compression preserves this
        """
        print("\n" + "="*60)
        print("TEST: Code Semantics Preservation")
        print("="*60)
        
        if not self.ollama.is_available():
            return TestResult(
                name="code_semantics",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Ollama not available"
            )
        
        prompts = TEST_PROMPTS["code_semantics"]
        embeddings = []
        
        print(f"Fetching {len(prompts)} code embeddings...")
        for prompt in prompts:
            emb = self.ollama.get_embedding(prompt, model)
            if emb:
                embeddings.append(torch.tensor(emb, device=self.device))
        
        if len(embeddings) < 4:
            return TestResult(
                name="code_semantics",
                passed=False,
                metric_name="error",
                metric_value=0,
                threshold=threshold,
                details="Not enough code embeddings"
            )
        
        # Update codec dimension
        if embeddings[0].shape[0] != self.dim:
            self.dim = embeddings[0].shape[0]
            self.codec = TurboQuantCodecOptimized(
                dim=self.dim,
                num_bits=self.num_bits,
                qjl_dim=self.qjl_dim,
                device=self.device
            )
        
        # Known similar pairs (same functionality, different syntax)
        similar_pairs = [(0, 1), (0, 2), (3, 4)]  # add functions, print statements
        dissimilar_pairs = [(0, 3), (0, 4), (0, 5)]  # add vs print vs sql
        
        # Compress
        embed_tensor = torch.stack(embeddings)
        encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
        
        # Check if similar pairs have higher similarity than dissimilar
        correct_orderings = 0
        total_comparisons = 0
        
        for sim_pair in similar_pairs:
            for dissim_pair in dissimilar_pairs:
                # True similarities
                true_sim = torch.cosine_similarity(
                    embeddings[sim_pair[0]],
                    embeddings[sim_pair[1]],
                    dim=0
                )
                true_dissim = torch.cosine_similarity(
                    embeddings[dissim_pair[0]],
                    embeddings[dissim_pair[1]],
                    dim=0
                )
                
                # Compressed similarities
                q1 = embeddings[sim_pair[0]].unsqueeze(0)
                q2 = embeddings[dissim_pair[0]].unsqueeze(0)
                
                comp_sim = self.codec.estimate_inner_products_vectorized(q1, encoded)[0][sim_pair[1]]
                comp_sim = comp_sim / (embeddings[sim_pair[0]].norm() * encoded['original_norms'][sim_pair[1]])
                
                comp_dissim = self.codec.estimate_inner_products_vectorized(q2, encoded)[0][dissim_pair[1]]
                comp_dissim = comp_dissim / (embeddings[dissim_pair[0]].norm() * encoded['original_norms'][dissim_pair[1]])
                
                # Check ordering preserved
                if (true_sim > true_dissim) == (comp_sim > comp_dissim):
                    correct_orderings += 1
                total_comparisons += 1
        
        accuracy = correct_orderings / max(total_comparisons, 1)
        passed = accuracy >= threshold
        
        result = TestResult(
            name="code_semantics",
            passed=passed,
            metric_name="ordering_accuracy",
            metric_value=accuracy,
            threshold=threshold,
            details=f"Correct orderings: {correct_orderings}/{total_comparisons}"
        )
        
        self.results.append(result)
        print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  Ordering Accuracy: {accuracy:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        
        return result
    
    def test_attention_fidelity(
        self,
        seq_len: int = 128,
        threshold: float = 0.9
    ) -> TestResult:
        """
        Test: Attention score fidelity with compressed KV cache.
        
        Method:
          1. Generate random Q, K matrices (simulating transformer attention)
          2. Compute true attention scores
          3. Compress K and estimate scores
          4. Compare score distributions
        """
        print("\n" + "="*60)
        print("TEST: Attention Fidelity")
        print("="*60)
        
        # Simulate transformer attention
        d_model = 512
        queries = torch.randn(seq_len, d_model, device=self.device)
        keys = torch.randn(seq_len, d_model, device=self.device)
        
        # Update codec
        if d_model != self.dim:
            self.dim = d_model
            self.codec = TurboQuantCodecOptimized(
                dim=d_model,
                num_bits=self.num_bits,
                qjl_dim=self.qjl_dim,
                device=self.device
            )
        
        # True attention scores
        scale = 1.0 / (d_model ** 0.5)
        true_scores = (queries @ keys.T) * scale
        true_attention = torch.softmax(true_scores, dim=-1)
        
        # Compress keys
        encoded = self.codec.encode_keys_batch_optimized(keys)
        
        # Estimated attention scores
        est_scores_list = []
        for i in range(seq_len):
            scores = self.codec.estimate_inner_products_vectorized(
                queries[i:i+1],
                encoded
            )[0] * scale
            est_scores_list.append(scores)
        
        est_scores = torch.stack(est_scores_list)
        est_attention = torch.softmax(est_scores, dim=-1)
        
        # Metrics
        # 1. MSE between attention distributions
        attn_mse = ((true_attention - est_attention) ** 2).mean().item()
        
        # 2. Cosine similarity
        cos_sim = torch.cosine_similarity(
            true_attention.view(-1),
            est_attention.view(-1),
            dim=0
        ).item()
        
        # 3. Top-K token agreement
        k = min(10, seq_len)
        agreements = []
        for i in range(seq_len):
            true_topk = true_attention[i].topk(k).indices.sort().values
            est_topk = est_attention[i].topk(k).indices.sort().values
            agreement = (true_topk == est_topk).float().mean().item()
            agreements.append(agreement)
        
        avg_agreement = sum(agreements) / len(agreements)
        
        passed = cos_sim >= threshold
        
        result = TestResult(
            name="attention_fidelity",
            passed=passed,
            metric_name="cosine_similarity",
            metric_value=cos_sim,
            threshold=threshold,
            details=f"Cosine: {cos_sim:.4f}, Top-{k} Agreement: {avg_agreement:.4f}"
        )
        
        self.results.append(result)
        print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        print(f"  Attention MSE: {attn_mse:.6f}")
        print(f"  Top-{k} Agreement: {avg_agreement:.4f}")
        
        return result
    
    def run_all_tests(
        self,
        model: str = "llama3"
    ) -> Dict[str, Any]:
        """
        Run all LLM tests.
        
        Args:
            model: Ollama model to use
            
        Returns:
            Summary dict with all results
        """
        print("\n" + "="*60)
        print("TURBOQUANT LLM TEST SUITE")
        print("="*60)
        print(f"Model: {model}")
        print(f"Device: {self.device}")
        print(f"Config: {self.num_bits}-bit + {self.qjl_dim}-bit QJL")
        
        # Check Ollama availability
        if self.ollama.is_available():
            print("Ollama: Connected")
            models = self.ollama.get_available_models()
            print(f"Available models: {models[:5]}")
        else:
            print("Ollama: Not available (some tests will be skipped)")
        
        # Run tests
        self.results = []
        
        self.test_attention_fidelity()  # Doesn't need Ollama
        
        if self.ollama.is_available():
            self.test_semantic_similarity(model=model)
            self.test_rag_retrieval(model=model)
            self.test_code_semantics(model=model)
        else:
            print("\n⚠ Skipping Ollama-dependent tests (Ollama not running)")
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Passed: {passed}/{total}")
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.name}: {result.metric_name}={result.metric_value:.4f}")
        
        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / max(total, 1),
            "results": [r.to_dict() for r in self.results],
        }


def main():
    """Run LLM test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TurboQuant LLM Tests")
    parser.add_argument("--model", default="llama3", help="Ollama model")
    parser.add_argument("--bits", type=int, default=4, help="Scalar bits")
    parser.add_argument("--qjl-dim", type=int, default=64, help="QJL dimension")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create test suite
    suite = LLMTestSuite(
        num_bits=args.bits,
        qjl_dim=args.qjl_dim
    )
    
    # Run tests
    summary = suite.run_all_tests(model=args.model)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit code
    sys.exit(0 if summary["pass_rate"] >= 0.7 else 1)


if __name__ == "__main__":
    main()
