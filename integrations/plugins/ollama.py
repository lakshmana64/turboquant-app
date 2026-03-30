"""
Ollama Plugin for TurboQuant

A plug-and-play plugin for compressing Ollama embeddings with TurboQuant.

Usage:
    # As a module
    from turboquant.integrations.plugins.ollama import OllamaPlugin
    
    plugin = OllamaPlugin(model="llama3")
    compressed = plugin.compress("Your prompt here")
    results = plugin.query("Your query here")
    
    # Or use CLI
    python -m turboquant.integrations.plugins.ollama --model llama3
"""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests
import torch

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig


@dataclass
class OllamaPluginConfig:
    """Configuration for Ollama plugin."""
    
    # Ollama settings
    model: str = "llama3"
    host: str = "localhost"
    port: int = 11434
    timeout: int = 30
    
    # TurboQuant settings
    num_bits: int = 4
    qjl_dim: int = 64
    seed: int = 42
    pack_bits: bool = True
    
    # Cache settings
    cache_enabled: bool = True
    cache_path: Optional[str] = None
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaPluginConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_env(cls) -> 'OllamaPluginConfig':
        """Load config from environment variables."""
        return cls(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            num_bits=int(os.getenv("TURBOQUANT_BITS", "4")),
            qjl_dim=int(os.getenv("TURBOQUANT_QJL_DIM", "64")),
        )


@dataclass
class CompressionResult:
    """Result of compression operation."""
    
    # Input
    prompt: str
    original_dim: int
    
    # Compression stats
    compression_ratio: float
    bits_per_dim: float
    
    # Quality metrics
    mse: Optional[float] = None
    correlation: Optional[float] = None
    
    # Encoded data (optional, can be excluded for memory efficiency)
    encoded: Optional[Dict[str, torch.Tensor]] = None

    @property
    def compression_factor(self) -> float:
        """Return original size / compressed size for x-style reporting."""
        return 1.0 / self.compression_ratio
    
    def to_dict(self, include_encoded: bool = False) -> Dict[str, Any]:
        data = {
            "prompt": self.prompt,
            "original_dim": self.original_dim,
            "compression_ratio": self.compression_ratio,
            "compression_factor": self.compression_factor,
            "bits_per_dim": self.bits_per_dim,
            "mse": self.mse,
            "correlation": self.correlation,
        }
        if include_encoded and self.encoded is not None:
            data["encoded"] = {k: v.shape for k, v in self.encoded.items()}
        return data


class OllamaPlugin:
    """
    Plug-and-play Ollama integration for TurboQuant.
    
    Features:
        - Automatic Ollama connection detection
        - Embedding compression with TurboQuant
        - Query against compressed embeddings
        - Optional caching
        - Batch processing
    
    Example:
        >>> plugin = OllamaPlugin(model="llama3")
        >>> plugin.connect()  # Verify connection
        
        >>> # Compress a single prompt
        >>> result = plugin.compress("Hello world")
        
        >>> # Compress multiple prompts
        >>> results = plugin.compress_batch(["Prompt 1", "Prompt 2"])
        
        >>> # Query compressed embeddings
        >>> matches = plugin.query("Find similar", top_k=5)
    """
    
    def __init__(self, config: Optional[OllamaPluginConfig] = None, **kwargs):
        """
        Initialize Ollama plugin.
        
        Args:
            config: Plugin configuration
            **kwargs: Override config options
        """
        if config is None:
            config = OllamaPluginConfig.from_env()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self._codec = None
        self._connected = False
        self._cache: Dict[str, CompressionResult] = {}
        
        # Lazy import to avoid dependency if not using Ollama
        self._requests = None
        self._torch = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Ollama."""
        return self._connected
    
    @property
    def codec(self):
        """Lazy load TurboQuant codec."""
        if self._codec is None:
            raise RuntimeError(
                "TurboQuant codec is not initialized yet. "
                "Call compress() or _ensure_codec(dim) first."
            )
        return self._codec

    def _ensure_codec(self, dim: int) -> TurboQuantCodec:
        """Initialize the codec once the embedding dimension is known."""
        if self._codec is None or self._codec.dim != dim:
            tq_config = TurboQuantConfig(
                num_bits=self.config.num_bits,
                qjl_dim=self.config.qjl_dim,
                seed=self.config.seed,
                pack_bits=self.config.pack_bits
            )
            self._codec = TurboQuantCodec(dim=dim, config=tq_config)
        return self._codec
    
    def connect(self, timeout: Optional[int] = None) -> bool:
        """
        Test connection to Ollama.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        timeout = timeout or self.config.timeout
        
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=timeout
            )
            if response.status_code == 200:
                self._connected = True
                self._requests = requests
                self._torch = torch
                return True
            return False
        except requests.exceptions.ConnectionError:
            self._connected = False
            return False
    
    def get_embedding(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Fetch embedding from Ollama.
        
        Args:
            prompt: Text to embed
            model: Override model name
            
        Returns:
            Embedding tensor or None if failed
        """
        model = model or self.config.model
        
        try:
            response = requests.post(
                f"{self.config.base_url}/api/embeddings",
                json={"model": model, "prompt": prompt},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if embedding is not None:
                return torch.tensor(embedding)
            return None
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            return None
    
    def compress(
        self,
        prompt: str,
        validate: bool = True,
        cache: bool = None
    ) -> Optional[CompressionResult]:
        """
        Compress a single prompt's embedding.
        
        Args:
            prompt: Text to compress
            validate: If True, compute quality metrics
            cache: Use caching (default from config)
            
        Returns:
            CompressionResult or None if failed
        """
        cache = cache if cache is not None else self.config.cache_enabled
        
        # Check cache
        if cache and prompt in self._cache:
            return self._cache[prompt]
        
        # Get embedding
        embedding = self.get_embedding(prompt)
        if embedding is None:
            return None
        
        # Initialize codec with correct dimension
        D = embedding.shape[0]
        codec = self._ensure_codec(D)
        
        # Compress
        embedding = embedding.unsqueeze(0)  # (1, D)
        encoded = codec.encode_keys_batch(embedding)
        
        # Validate quality if requested
        mse = correlation = None
        if validate:
            true_dot = (embedding @ embedding.T).item()
            est_dot = codec.estimate_inner_products(embedding[0], encoded).item()
            mse = (true_dot - est_dot) ** 2
            correlation = 1.0 - mse / (true_dot ** 2 + 1e-8)
        
        # Compute compression stats
        original_bits = D * 32
        compressed_bits = D * self.config.num_bits + self.config.qjl_dim
        
        result = CompressionResult(
            prompt=prompt,
            original_dim=D,
            compression_ratio=compressed_bits / original_bits,
            bits_per_dim=compressed_bits / D,
            mse=mse,
            correlation=correlation,
            encoded=encoded
        )
        
        # Cache result
        if cache:
            self._cache[prompt] = result
        
        return result
    
    def compress_batch(
        self,
        prompts: List[str],
        validate: bool = True
    ) -> List[Optional[CompressionResult]]:
        """
        Compress multiple prompts.
        
        Args:
            prompts: List of texts to compress
            validate: If True, compute quality metrics
            
        Returns:
            List of CompressionResult or None for failures
        """
        return [self.compress(p, validate=validate) for p in prompts]
    
    def query(
        self,
        query_text: str,
        compressed_results: List[CompressionResult],
        top_k: int = 5,
        scale: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query against compressed embeddings.
        
        Args:
            query_text: Query text
            compressed_results: List of CompressionResult from compress()
            top_k: Number of results to return
            scale: Optional scaling factor for scores
            
        Returns:
            List of {prompt, score, rank} dicts
        """
        # Get query embedding
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []
        
        # Collect encoded data
        encoded_list = []
        prompts = []
        for result in compressed_results:
            if result.encoded is not None:
                encoded_list.append(result.encoded)
                prompts.append(result.prompt)
        
        if not encoded_list:
            return []
        
        # Batch query
        # For simplicity, query each (can be optimized for batch)
        scores = []
        for encoded in encoded_list:
            D = query_embedding.shape[0]
            codec = self._ensure_codec(D)
            score = codec.estimate_inner_products(query_embedding, encoded)
            if scale:
                score = score * scale
            scores.append(score.item())
        
        # Rank results
        ranked = sorted(
            zip(prompts, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {"prompt": prompt, "score": score, "rank": i + 1}
            for i, (prompt, score) in enumerate(ranked)
        ]
    
    def clear_cache(self):
        """Clear the compression cache."""
        self._cache.clear()
    
    def save_cache(self, path: str):
        """Save cache to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
    
    def load_cache(self, path: str):
        """Load cache from disk."""
        import pickle
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._cache = pickle.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "connected": self._connected,
            "cache_size": len(self._cache),
            "config": self.config.to_dict(),
        }
    
    def __repr__(self) -> str:
        return (
            f"OllamaPlugin(model={self.config.model!r}, "
            f"connected={self._connected}, "
            f"cache_size={len(self._cache)})"
        )


# Convenience functions for quick usage

def compress(
    prompt: str,
    model: str = "llama3",
    num_bits: int = 4,
    qjl_dim: int = 64,
    pack_bits: bool = True
) -> Optional[CompressionResult]:
    """Quick compress a single prompt."""
    plugin = OllamaPlugin(
        model=model,
        num_bits=num_bits,
        qjl_dim=qjl_dim,
        pack_bits=pack_bits
    )
    if not plugin.connect():
        print("Warning: Could not connect to Ollama")
        return None
    return plugin.compress(prompt)


def query(
    query_text: str,
    prompts: List[str],
    model: str = "llama3",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Quick query against multiple prompts."""
    plugin = OllamaPlugin(model=model)
    if not plugin.connect():
        print("Warning: Could not connect to Ollama")
        return []
    
    # Compress all prompts
    results = plugin.compress_batch(prompts)
    results = [r for r in results if r is not None]
    
    # Query
    return plugin.query(query_text, results, top_k=top_k)
