"""
OpenAI embedding plugin for TurboQuant.

This adapter keeps the dependency footprint light by using ``requests`` rather
than the OpenAI SDK. Set ``OPENAI_API_KEY`` before connecting.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests
import torch

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig

from .ollama import CompressionResult


@dataclass
class OpenAIPluginConfig:
    """Configuration for the OpenAI embedding plugin."""

    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    timeout: int = 30
    num_bits: int = 4
    qjl_dim: int = 64
    seed: int = 42
    cache_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "api_key": None if self.api_key is None else "***",
        }

    @classmethod
    def from_env(cls) -> "OpenAIPluginConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            num_bits=int(os.getenv("TURBOQUANT_BITS", "4")),
            qjl_dim=int(os.getenv("TURBOQUANT_QJL_DIM", "64")),
        )


class OpenAIPlugin:
    """Compress OpenAI embeddings with TurboQuant."""

    def __init__(self, config: Optional[OpenAIPluginConfig] = None, **kwargs):
        self.config = config or OpenAIPluginConfig.from_env()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._codec: Optional[TurboQuantCodec] = None
        self._cache: Dict[str, CompressionResult] = {}
        self._connected = False

    @property
    def headers(self) -> Dict[str, str]:
        if not self.config.api_key:
            return {}

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        return headers

    def connect(self, validate: bool = False) -> bool:
        """Check local credentials and optionally validate them over HTTP."""
        if not self.config.api_key:
            self._connected = False
            return False

        if not validate:
            self._connected = True
            return True

        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                headers=self.headers,
                timeout=self.config.timeout,
            )
            self._connected = response.ok
            return self._connected
        except requests.RequestException:
            self._connected = False
            return False

    def _ensure_codec(self, dim: int) -> TurboQuantCodec:
        if self._codec is None or self._codec.dim != dim:
            tq_config = TurboQuantConfig(
                num_bits=self.config.num_bits,
                qjl_dim=self.config.qjl_dim,
                seed=self.config.seed,
            )
            self._codec = TurboQuantCodec(dim=dim, config=tq_config)
        return self._codec

    def get_embedding(self, text: str, model: Optional[str] = None) -> Optional[torch.Tensor]:
        """Fetch a single embedding from the OpenAI embeddings API."""
        if not self.connect():
            return None

        payload = {
            "model": model or self.config.model,
            "input": text,
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"][0]["embedding"]
            return torch.tensor(data, dtype=torch.float32)
        except (KeyError, IndexError, requests.RequestException, ValueError):
            return None

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
