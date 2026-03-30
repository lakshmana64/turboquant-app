"""
Hugging Face integration helpers for TurboQuant.

The wrapper below keeps the original attention implementation intact while
adding concrete cache-compression hooks around it. This makes the integration
usable with attention modules that already know how to compute rotary
embeddings, grouped-query attention, masking, and model-specific quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig


@dataclass
class CompressedPastKeyValue:
    """Compressed key cache plus the original value cache."""

    encoded_keys: Tuple[Dict[str, torch.Tensor], ...]
    value_states: torch.Tensor
    batch_size: int
    num_key_value_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype


def _move_encoded(encoded: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move encoded tensors to the requested device."""
    return {key: value.to(device=device) for key, value in encoded.items()}


class TurboQuantAttentionWrapper(nn.Module):
    """
    Wrap a Hugging Face attention layer with cache compression hooks.

    The wrapper is intentionally conservative: it delegates the real attention
    math to the original layer and only intercepts the KV cache boundary.
    That keeps the integration compatible with model-specific attention logic.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        config: TurboQuantConfig = TurboQuantConfig(),
        return_compressed_cache: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.config = config
        self.return_compressed_cache = return_compressed_cache
        self.head_dim = getattr(original_layer, "head_dim", None)
        self.last_compressed_cache: Optional[CompressedPastKeyValue] = None
        self._codec: Optional[TurboQuantCodec] = None

    def _ensure_codec(self, head_dim: int, device: torch.device) -> TurboQuantCodec:
        if (
            self._codec is None
            or self._codec.dim != head_dim
            or self._codec.device != torch.device(device)
        ):
            self._codec = TurboQuantCodec(head_dim, config=self.config, device=device)
        return self._codec

    def compress_past_key_value(
        self,
        past_key_value: Optional[Any],
    ) -> Optional[Any]:
        """Compress a legacy `(key_states, value_states)` cache tuple."""
        if past_key_value is None or isinstance(past_key_value, CompressedPastKeyValue):
            return past_key_value

        if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 2:
            return past_key_value

        key_states, value_states = past_key_value
        if not isinstance(key_states, torch.Tensor) or not isinstance(value_states, torch.Tensor):
            return past_key_value
        if key_states.dim() != 4:
            return past_key_value

        batch_size, num_key_value_heads, seq_len, head_dim = key_states.shape
        codec = self._ensure_codec(head_dim, key_states.device)

        encoded_groups: List[Dict[str, torch.Tensor]] = []
        flat_keys = key_states.reshape(batch_size * num_key_value_heads, seq_len, head_dim)
        for group_keys in flat_keys:
            encoded = codec.encode_keys_batch(group_keys)
            encoded_groups.append(_move_encoded(encoded, torch.device("cpu")))

        return CompressedPastKeyValue(
            encoded_keys=tuple(encoded_groups),
            value_states=value_states.detach(),
            batch_size=batch_size,
            num_key_value_heads=num_key_value_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=key_states.dtype,
        )

    def decompress_past_key_value(
        self,
        past_key_value: Optional[Any],
        device: torch.device,
    ) -> Optional[Any]:
        """Restore a compressed cache object into the legacy tensor tuple."""
        if past_key_value is None or not isinstance(past_key_value, CompressedPastKeyValue):
            return past_key_value

        codec = self._ensure_codec(past_key_value.head_dim, device)
        decoded_groups: List[torch.Tensor] = []
        for encoded in past_key_value.encoded_keys:
            decoded = codec.decode_keys(_move_encoded(encoded, device))
            decoded_groups.append(decoded.to(device=device, dtype=past_key_value.dtype))

        key_states = torch.stack(decoded_groups, dim=0).reshape(
            past_key_value.batch_size,
            past_key_value.num_key_value_heads,
            past_key_value.seq_len,
            past_key_value.head_dim,
        )
        value_states = past_key_value.value_states.to(device=device)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        **kwargs,
    ):
        """
        Run the original attention layer with transparent cache conversion.
        """
        normalized_cache = self.decompress_past_key_value(
            past_key_value,
            hidden_states.device,
        )

        output = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=normalized_cache,
            **kwargs,
        )

        if not isinstance(output, tuple) or not output:
            return output

        present_key_value = output[-1]
        compressed_cache = self.compress_past_key_value(present_key_value)
        if isinstance(compressed_cache, CompressedPastKeyValue):
            self.last_compressed_cache = compressed_cache

        if self.return_compressed_cache and isinstance(compressed_cache, CompressedPastKeyValue):
            output_items = list(output)
            output_items[-1] = compressed_cache
            return tuple(output_items)

        return output


def _should_wrap_attention_module(name: str, module: nn.Module) -> bool:
    module_name = module.__class__.__name__.lower()
    child_name = name.lower()
    if isinstance(module, TurboQuantAttentionWrapper):
        return False
    if getattr(module, "head_dim", None) is None:
        return False
    return "attention" in module_name or "attn" in child_name


def _wrap_attention_modules(
    parent: nn.Module,
    config: TurboQuantConfig,
    return_compressed_cache: bool,
    prefix: str = "",
) -> List[str]:
    wrapped: List[str] = []

    for child_name, child in list(parent.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if _should_wrap_attention_module(child_name, child):
            setattr(
                parent,
                child_name,
                TurboQuantAttentionWrapper(
                    child,
                    config=config,
                    return_compressed_cache=return_compressed_cache,
                ),
            )
            wrapped.append(full_name)
            continue

        wrapped.extend(
            _wrap_attention_modules(
                child,
                config=config,
                return_compressed_cache=return_compressed_cache,
                prefix=full_name,
            )
        )

    return wrapped


def apply_turboquant_to_hf_model(
    model: nn.Module,
    sq_bits: int = 2,
    qjl_dim: int = 64,
    return_compressed_cache: bool = False,
    pack_bits: bool = True,
) -> nn.Module:
    """
    Recursively wrap attention modules with TurboQuant cache compression.

    The returned model keeps the same callable surface as the original model.
    If `return_compressed_cache=True`, wrapped attention layers emit
    `CompressedPastKeyValue` instances instead of raw `(k, v)` tuples.
    """
    config = TurboQuantConfig(num_bits=sq_bits, qjl_dim=qjl_dim, pack_bits=pack_bits)
    wrapped = _wrap_attention_modules(
        model,
        config=config,
        return_compressed_cache=return_compressed_cache,
    )
    model._turboquant_wrapped_modules = wrapped
    return model


__all__ = [
    "CompressedPastKeyValue",
    "TurboQuantAttentionWrapper",
    "apply_turboquant_to_hf_model",
]
