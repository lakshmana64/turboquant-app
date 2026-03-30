"""
TurboQuant Haystack integration plugin.

This module provides both a compressed document store and an embedder that now
actually encodes embeddings before handing them back to the pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from turboquant.core.codec import TurboQuantConfig
from turboquant.core.optimized import TurboQuantCodecOptimized


def _move_encoded(encoded: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    return {key: value.to(device=device) for key, value in encoded.items()}


def _encoded_row(encoded: Dict[str, Tensor], index: int) -> Dict[str, Tensor]:
    return {key: value[index : index + 1].detach().cpu() for key, value in encoded.items()}


def _merge_encoded_batches(
    current: Optional[Dict[str, Tensor]],
    new_batch: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    if current is None:
        return {key: value.detach().clone() for key, value in new_batch.items()}

    for key, value in new_batch.items():
        current[key] = torch.cat([current[key], value.detach()], dim=0)
    return current


class TurboQuantDocumentStore:
    """
    Haystack document store with compressed vector storage.
    """

    def __init__(
        self,
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: str = "cpu",
        pack_bits: bool = True,
    ):
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.device = device
        self.pack_bits = pack_bits
        self.documents: List[Document] = []
        self.encoded_vectors: Optional[Dict[str, Tensor]] = None
        self.codecs: Dict[int, TurboQuantCodecOptimized] = {}

    def _get_codec(self, dim: int) -> TurboQuantCodecOptimized:
        if dim not in self.codecs:
            config = TurboQuantConfig(
                num_bits=self.num_bits, 
                qjl_dim=self.qjl_dim,
                pack_bits=self.pack_bits
            )
            self.codecs[dim] = TurboQuantCodecOptimized(
                dim=dim,
                config=config,
                device=self.device,
            )
        return self.codecs[dim]

    def _encode_document(self, document: Document) -> Optional[Dict[str, Tensor]]:
        encoded = getattr(document, "_turboquant_encoded", None)
        if encoded is not None:
            return _move_encoded(encoded, self.device)

        embedding = getattr(document, "embedding", None)
        if embedding is None:
            return None

        vector = torch.tensor(embedding, device=self.device).unsqueeze(0)
        codec = self._get_codec(vector.shape[-1])
        return codec.encode_keys_batch_optimized(vector)

    def write_documents(self, documents: List[Document]) -> int:
        """
        Store and compress documents with searchable embeddings.
        """
        written = 0
        for document in documents:
            encoded = self._encode_document(document)
            if encoded is None:
                continue

            self.documents.append(document)
            self.encoded_vectors = _merge_encoded_batches(self.encoded_vectors, encoded)
            written += 1

        return written

    def query_documents(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Document]:
        """
        Query compressed document vectors.
        """
        if self.encoded_vectors is None or not self.documents:
            return []

        q_tensor = torch.tensor(query_embedding, device=self.device)
        dim = q_tensor.shape[-1]
        codec = self._get_codec(dim)

        scores = codec.estimate_inner_products_vectorized(
            q_tensor.view(-1, dim),
            self.encoded_vectors,
        ).squeeze(0)

        top_scores, top_indices = torch.topk(scores, min(top_k, scores.shape[0]))

        results = []
        for position, doc_index in enumerate(top_indices.tolist()):
            document = self.documents[doc_index]
            document.score = float(top_scores[position].item())
            results.append(document)

        return results


class TurboQuantDocumentEmbedder(SentenceTransformersDocumentEmbedder):
    """
    Haystack document embedder with built-in TurboQuant compression.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: str = "cpu",
        pack_bits: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.device = device
        self.pack_bits = pack_bits
        self._codec: Optional[TurboQuantCodecOptimized] = None
        self.last_encoded: Optional[Dict[str, Tensor]] = None
        self._compressed_count = 0

    def _get_codec(self, dim: int) -> TurboQuantCodecOptimized:
        if self._codec is None or self._codec.dim != dim:
            config = TurboQuantConfig(
                num_bits=self.num_bits, 
                qjl_dim=self.qjl_dim,
                pack_bits=self.pack_bits
            )
            self._codec = TurboQuantCodecOptimized(
                dim=dim,
                config=config,
                device=self.device,
            )
        return self._codec

    def run(self, documents: List[Document]):
        """
        Embed and compress documents.
        """
        result = super().run(documents)
        result_documents = result.get("documents", documents) if isinstance(result, dict) else documents

        embedded_documents = [
            document
            for document in result_documents
            if getattr(document, "embedding", None) is not None
        ]
        if not embedded_documents:
            self.last_encoded = None
            return result

        batch_vectors = torch.tensor(
            [document.embedding for document in embedded_documents],
            device=self.device,
        )
        codec = self._get_codec(batch_vectors.shape[-1])
        encoded = codec.encode_keys_batch_optimized(batch_vectors)
        reconstructed = codec.decode_keys_vectorized(encoded).detach().cpu().tolist()
        self.last_encoded = {key: value.detach().cpu() for key, value in encoded.items()}
        self._compressed_count += len(embedded_documents)

        for index, document in enumerate(embedded_documents):
            document.embedding = reconstructed[index]
            meta = dict(getattr(document, "meta", {}) or {})
            meta["turboquant"] = {
                "compression_ratio": codec.get_memory_usage(1)["ratio"],
                "num_bits": self.num_bits,
                "qjl_dim": self.qjl_dim,
                "original_dim": len(reconstructed[index]),
            }
            document.meta = meta
            setattr(document, "_turboquant_encoded", _encoded_row(encoded, index))

        if isinstance(result, dict):
            result["documents"] = result_documents
            result["turboquant_encoded"] = self.last_encoded

        return result

    def get_compression_stats(self) -> Dict[str, float]:
        if self._codec is None:
            return {
                "compressed_count": 0,
                "compression_ratio": 1.0,
                "original_mb": 0.0,
                "compressed_mb": 0.0,
                "savings_mb": 0.0,
            }

        memory_usage = self._codec.get_memory_usage(self._compressed_count)
        return {
            "compressed_count": self._compressed_count,
            "compression_ratio": memory_usage["ratio"],
            "original_mb": memory_usage["original"] / 1e6,
            "compressed_mb": memory_usage["compressed"] / 1e6,
            "savings_mb": memory_usage["savings_mb"],
        }


__all__ = ["TurboQuantDocumentStore", "TurboQuantDocumentEmbedder"]

