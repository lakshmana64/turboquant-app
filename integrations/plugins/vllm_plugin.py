"""
TurboQuant VLLM integration helpers.

This module now provides a concrete adapter surface instead of a placeholder:
it can encode paged KV caches head-by-head and expose patch helpers on a
running engine instance.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from turboquant.core.codec import TurboQuantConfig
from turboquant.core.optimized import TurboQuantCodecOptimized


class TurboQuantVLLMAdapter:
    """
    Adapter to integrate TurboQuant with a VLLM-style paged KV cache.
    """

    def __init__(
        self,
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: str = "cuda",
    ):
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.device = device
        self.codecs: Dict[int, TurboQuantCodecOptimized] = {}

    def _get_codec(self, dim: int) -> TurboQuantCodecOptimized:
        if dim not in self.codecs:
            config = TurboQuantConfig(num_bits=self.num_bits, qjl_dim=self.qjl_dim)
            self.codecs[dim] = TurboQuantCodecOptimized(
                dim=dim,
                config=config,
                device=self.device,
            )
        return self.codecs[dim]

    def compress_kv_cache(
        self,
        key_cache: Tensor,
        value_cache: Tensor,
    ) -> Dict[str, Any]:
        """
        Compress VLLM KV cache blocks head-by-head.

        Expected layout:
            key_cache:   (num_blocks, num_heads, head_size, block_size)
            value_cache: (num_blocks, num_heads, head_size, block_size)
        """
        if key_cache.dim() != 4 or value_cache.dim() != 4:
            raise ValueError("VLLM caches must be rank-4 tensors")

        num_blocks, num_heads, head_size, block_size = key_cache.shape
        codec = self._get_codec(head_size)

        encoded_heads: List[Dict[str, Tensor]] = []
        values_by_head = value_cache.permute(1, 0, 3, 2).reshape(
            num_heads,
            num_blocks * block_size,
            head_size,
        )

        for head_index in range(num_heads):
            head_keys = key_cache[:, head_index].permute(0, 2, 1).reshape(-1, head_size)
            encoded_heads.append(codec.encode_keys_batch_optimized(head_keys))

        return {
            "encoded_heads": tuple(encoded_heads),
            "values_by_head": values_by_head,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "head_size": head_size,
            "block_size": block_size,
        }

    def compute_paged_attention(
        self,
        query: Tensor,
        encoded_keys: Dict[str, Any],
        value_cache: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        scale: float = 1.0,
    ) -> Tensor:
        """
        Compute attention against a compressed paged cache.
        """
        if query.dim() != 3:
            raise ValueError("query must have shape (num_queries, num_heads, head_size)")
        if "encoded_heads" not in encoded_keys:
            raise ValueError("encoded_keys must come from compress_kv_cache()")

        num_queries, num_heads, head_size = query.shape
        codec = self._get_codec(head_size)
        output = torch.empty_like(query)

        if value_cache is not None:
            values_by_head = value_cache.permute(1, 0, 3, 2).reshape(
                value_cache.shape[1],
                value_cache.shape[0] * value_cache.shape[3],
                value_cache.shape[2],
            )
        else:
            values_by_head = encoded_keys["values_by_head"]

        encoded_heads = encoded_keys["encoded_heads"]
        if len(encoded_heads) != num_heads:
            raise ValueError("query head count does not match encoded cache head count")

        for head_index in range(num_heads):
            head_scores = codec.estimate_inner_products_vectorized(
                query[:, head_index, :],
                encoded_heads[head_index],
            )
            if block_tables is not None:
                # The block table layout differs across VLLM releases, so this
                # adapter currently treats it as advisory metadata.
                pass
            attention = torch.softmax(head_scores * scale, dim=-1)
            output[:, head_index, :] = attention @ values_by_head[head_index]

        return output


def patch_vllm_with_turboquant(
    llm_engine: Any,
    num_bits: int = 4,
    qjl_dim: int = 64,
    device: str = "cuda",
):
    """
    Attach TurboQuant helpers to a VLLM engine instance.

    The patch is additive: it does not overwrite engine internals, but it
    exposes `compress_kv_cache()` and `compute_paged_attention()` methods that
    callers can use from serving code or custom execution hooks.
    """
    adapter = TurboQuantVLLMAdapter(
        num_bits=num_bits,
        qjl_dim=qjl_dim,
        device=device,
    )

    llm_engine.turboquant_adapter = adapter

    def _compress_kv_cache(self, key_cache: Tensor, value_cache: Tensor):
        return self.turboquant_adapter.compress_kv_cache(key_cache, value_cache)

    def _compute_paged_attention(
        self,
        query: Tensor,
        encoded_keys: Dict[str, Any],
        value_cache: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        scale: float = 1.0,
    ):
        return self.turboquant_adapter.compute_paged_attention(
            query=query,
            encoded_keys=encoded_keys,
            value_cache=value_cache,
            block_tables=block_tables,
            scale=scale,
        )

    llm_engine.compress_kv_cache = MethodType(_compress_kv_cache, llm_engine)
    llm_engine.compute_paged_attention = MethodType(
        _compute_paged_attention,
        llm_engine,
    )

    model_executor = getattr(llm_engine, "model_executor", None)
    if model_executor is not None:
        model_executor.turboquant_adapter = adapter

    return llm_engine


__all__ = ["TurboQuantVLLMAdapter", "patch_vllm_with_turboquant"]

