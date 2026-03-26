"""
SentenceTransformers plugin for TurboQuant.

This plugin uses a local embedding model, which makes it handy for offline
benchmarks and local RAG pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig

from .ollama import CompressionResult

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - exercised by import behavior
    raise ImportError(
        "sentence-transformers is required for SentenceTransformersPlugin. "
        "Install it with `pip install sentence-transformers`."
    ) from exc


@dataclass
class SentenceTransformersPluginConfig:
    """Configuration for the SentenceTransformers plugin."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    num_bits: int = 4
    qjl_dim: int = 64
    seed: int = 42
    cache_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentenceTransformersPlugin:
    """Compress local SentenceTransformers embeddings with TurboQuant."""

    def __init__(
        self,
        config: Optional[SentenceTransformersPluginConfig] = None,
        **kwargs,
    ):
        self.config = config or SentenceTransformersPluginConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._model: Optional[SentenceTransformer] = None
        self._codec: Optional[TurboQuantCodec] = None
        self._cache: Dict[str, CompressionResult] = {}

    def connect(self) -> bool:
        """Load the embedding model lazily."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        return True

    def _ensure_codec(self, dim: int) -> TurboQuantCodec:
        if self._codec is None or self._codec.dim != dim:
            tq_config = TurboQuantConfig(
                num_bits=self.config.num_bits,
                qjl_dim=self.config.qjl_dim,
                seed=self.config.seed,
            )
            self._codec = TurboQuantCodec(dim=dim, config=tq_config)
        return self._codec

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Embed a single string with SentenceTransformers."""
        self.connect()
        assert self._model is not None
        embedding = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        return embedding.detach().cpu().to(dtype=torch.float32)

    def compress(
        self,
        text: str,
        validate: bool = True,
        cache: Optional[bool] = None,
    ) -> Optional[CompressionResult]:
        cache = self.config.cache_enabled if cache is None else cache
        if cache and text in self._cache:
            return self._cache[text]

        embedding = self.get_embedding(text)
        if embedding is None:
            return None

        dim = embedding.shape[0]
        codec = self._ensure_codec(dim)
        batch = embedding.unsqueeze(0)
        encoded = codec.encode_keys_batch(batch)

        mse = correlation = None
        if validate:
            true_dot = (batch @ batch.T).item()
            est_dot = codec.estimate_inner_products(batch[0], encoded).item()
            mse = (true_dot - est_dot) ** 2
            correlation = 1.0 - mse / (true_dot ** 2 + 1e-8)

        compressed_bits = dim * self.config.num_bits + self.config.qjl_dim
        result = CompressionResult(
            prompt=text,
            original_dim=dim,
            compression_ratio=compressed_bits / (dim * 32),
            bits_per_dim=compressed_bits / dim,
            mse=mse,
            correlation=correlation,
            encoded=encoded,
        )

        if cache:
            self._cache[text] = result
        return result

    def compress_batch(
        self,
        texts: List[str],
        validate: bool = True,
    ) -> List[Optional[CompressionResult]]:
        return [self.compress(text, validate=validate) for text in texts]

    def query(
        self,
        query_text: str,
        compressed_results: List[CompressionResult],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []

        dim = query_embedding.shape[0]
        codec = self._ensure_codec(dim)

        scored: List[Dict[str, Any]] = []
        for result in compressed_results:
            if result is None or result.encoded is None:
                continue
            score = codec.estimate_inner_products(query_embedding, result.encoded).item()
            scored.append({"prompt": result.prompt, "score": score})

        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        return ranked

    def clear_cache(self) -> None:
        self._cache.clear()
