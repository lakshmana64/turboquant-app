"""
Layer-Adaptive Mode for TurboQuant.

Configures different quantization strategies per transformer layer.
Keeps the last N layers at higher precision (q8_0) while compressing
earlier layers more aggressively.

Strategy:
- Last 8 layers: q8_0 (higher quality, critical for output)
- Earlier layers: turbo2/turbo3/turbo4 (aggressive compression)
- Overall: ~3.5x compression with minimal quality loss

Rationale:
- Later layers have more impact on output quality
- Early layers can tolerate more compression
- Provides optimal quality/compression tradeoff
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from .turbo_formats import get_format, TurboFormat
from .codec import TurboQuantCodec, TurboQuantConfig
from .value_quant import TurboValueCodec


@dataclass
class LayerConfig:
    """Configuration for a single layer's quantization."""
    layer_idx: int
    k_format: str
    v_format: str
    k_codec: Optional[TurboQuantCodec] = None
    v_codec: Optional[TurboValueCodec] = None


class LayerAdaptiveConfig:
    """Configuration for layer-adaptive quantization."""
    
    def __init__(
        self,
        num_layers: int = 32,
        dim: int = 4096,
        keep_last_n: int = 8,
        default_k_format: str = "turbo4",
        default_v_format: str = "turbo4",
        protected_k_format: str = "q8_0",
        protected_v_format: str = "q8_0",
        enable_sparse_v: bool = True,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize layer-adaptive config.
        
        Args:
            num_layers: Total number of transformer layers
            dim: Hidden dimension
            keep_last_n: Number of final layers to keep at high precision
            default_k_format: Format for compressed Keys
            default_v_format: Format for compressed Values
            protected_k_format: Format for protected final layers (Keys)
            protected_v_format: Format for protected final layers (Values)
            enable_sparse_v: Enable sparse V decoding for compressed layers
            seed: Random seed
            device: Torch device
        """
        self.num_layers = num_layers
        self.dim = dim
        self.keep_last_n = keep_last_n
        self.default_k_format = default_k_format
        self.default_v_format = default_v_format
        self.protected_k_format = protected_k_format
        self.protected_v_format = protected_v_format
        self.enable_sparse_v = enable_sparse_v
        self.seed = seed
        self.device = device or torch.device('cpu')
        
        # Calculate split point
        self.split_layer = max(0, num_layers - keep_last_n)
        
        # Validate
        if keep_last_n > num_layers:
            raise ValueError(
                f"keep_last_n ({keep_last_n}) > num_layers ({num_layers})"
            )
            
    def is_layer_protected(self, layer_idx: int) -> bool:
        """Check if a layer should use high-precision storage."""
        return layer_idx >= self.split_layer

    def get_config_for_layer(self, layer_idx: int) -> Any:
        """Get the configuration for a specific layer."""
        from .asymmetric_kv import AsymmetricKVConfig
        k_fmt, v_fmt = self.get_layer_format(layer_idx)
        return AsymmetricKVConfig(
            dim=self.dim,
            k_format=k_fmt,
            v_format=v_fmt,
            enable_sparse_v=self.enable_sparse_v and not self.is_layer_protected(layer_idx),
            seed=self.seed + layer_idx
        )
    
    def get_layer_format(self, layer_idx: int) -> Tuple[str, str]:
        """
        Get K/V formats for a specific layer.
        
        Args:
            layer_idx: Layer index (0 = first, num_layers-1 = last)
        
        Returns:
            Tuple of (k_format, v_format)
        """
        # Last N layers get high precision
        is_high_precision = layer_idx >= (self.num_layers - self.keep_last_n)
        
        if is_high_precision:
            return self.protected_k_format, self.protected_v_format
        else:
            return self.default_k_format, self.default_v_format
    
    def get_compression_estimate(self) -> Dict[str, float]:
        """Estimate overall compression ratio."""
        high_prec_fmt = get_format(self.protected_k_format)
        comp_fmt = get_format(self.default_k_format)
        
        # Weighted average
        high_prec_ratio = self.keep_last_n / self.num_layers
        comp_ratio = 1 - high_prec_ratio
        
        avg_compression = (
            high_prec_ratio * high_prec_fmt.compression_factor +
            comp_ratio * comp_fmt.compression_factor
        )
        
        return {
            "high_precision_layers": self.keep_last_n,
            "compressed_layers": self.num_layers - self.keep_last_n,
            "high_precision_format": self.protected_k_format,
            "compressed_format": self.default_k_format,
            "high_precision_compression": high_prec_fmt.compression_factor,
            "compressed_compression": comp_fmt.compression_factor,
            "estimated_overall_compression": avg_compression
        }


class LayerAdaptiveKVCache:
    """
    KV Cache with layer-adaptive quantization.
    
    Uses different compression formats for different layers
    to optimize quality/compression tradeoff.
    """
    
    def __init__(self, config: LayerAdaptiveConfig):
        """
        Initialize layer-adaptive KV cache.
        
        Args:
            config: LayerAdaptiveConfig instance
        """
        self.config = config
        self.device = config.device
        self.dim = config.dim
        
        # Initialize codecs for each layer
        self.layer_codecs: Dict[int, LayerConfig] = {}
        
        for layer_idx in range(config.num_layers):
            k_format, v_format = config.get_layer_format(layer_idx)
            
            # Create codecs
            k_codec = self._create_k_codec(k_format)
            v_codec = self._create_v_codec(v_format)
            
            self.layer_codecs[layer_idx] = LayerConfig(
                layer_idx=layer_idx,
                k_format=k_format,
                v_format=v_format,
                k_codec=k_codec,
                v_codec=v_codec
            )
        
        # Storage for KV cache per layer
        self.cache: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
            layer_idx: {"keys": [], "values": []}
            for layer_idx in range(config.num_layers)
        }
        
        # Statistics
        self.seq_len_per_layer = {layer_idx: 0 for layer_idx in range(config.num_layers)}
    
    def _create_k_codec(self, format_name: str) -> TurboQuantCodec:
        """Create K codec for a format."""
        fmt = get_format(format_name)
        config = TurboQuantConfig(
            num_bits=fmt.sq_bits,
            qjl_dim=fmt.qjl_dim,
            seed=self.config.seed,
            pack_bits=True
        )
        return TurboQuantCodec(self.dim, config, self.device)
    
    def _create_v_codec(self, format_name: str) -> TurboValueCodec:
        """Create V codec for a format."""
        fmt = get_format(format_name)
        return TurboValueCodec(self.dim, fmt.sq_bits, self.device)
    
    def append(
        self,
        layer_idx: int,
        k: Tensor,
        v: Tensor
    ) -> Dict[str, Any]:
        """
        Append K, V to a specific layer's cache.
        
        Args:
            layer_idx: Layer index
            k: Key tensor [seq, dim] or [batch, seq, dim]
            v: Value tensor matching k shape
        
        Returns:
            Metadata about the append
        """
        if layer_idx not in self.layer_codecs:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer_config = self.layer_codecs[layer_idx]
        
        # Normalize shape
        if k.dim() == 3:
            k = k.view(-1, self.dim)
            v = v.view(-1, self.dim)
        elif k.dim() == 1:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        
        # Encode K
        encoded_k = layer_config.k_codec.encode_key(k)
        
        # Encode V
        encoded_v = layer_config.v_codec.encode(v)
        
        # Store
        self.cache[layer_idx]["keys"].append(encoded_k)
        self.cache[layer_idx]["values"].append(encoded_v)
        self.seq_len_per_layer[layer_idx] += k.shape[0]
        
        return {
            "layer_idx": layer_idx,
            "k_format": layer_config.k_format,
            "v_format": layer_config.v_format,
            "seq_added": k.shape[0],
            "total_seq": self.seq_len_per_layer[layer_idx]
        }
    
    def get_attention_output(
        self,
        layer_idx: int,
        q: Tensor,
        scale: float = 1.0
    ) -> Tensor:
        """
        Compute attention output for a specific layer.
        
        Args:
            layer_idx: Layer index
            q: Query tensor [batch, dim]
            scale: Scaling factor
        
        Returns:
            Attention output [batch, dim]
        """
        if layer_idx not in self.layer_codecs:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer_config = self.layer_codecs[layer_idx]
        cache = self.cache[layer_idx]
        
        if len(cache["keys"]) == 0:
            return torch.zeros_like(q)
        
        # Compute attention scores from K
        all_scores = []
        for encoded_k in cache["keys"]:
            scores = layer_config.k_codec.compute_attention_scores(q, encoded_k, scale)
            all_scores.append(scores)
        
        # Concatenate and softmax
        all_scores = torch.cat(all_scores, dim=-1)
        attn_weights = torch.softmax(all_scores, dim=-1)
        
        # Decode V
        all_v = []
        for encoded_v in cache["values"]:
            v_decoded = layer_config.v_codec.decode(encoded_v)
            all_v.append(v_decoded)
        
        v_tensor = torch.cat(all_v, dim=0).unsqueeze(0)  # [1, seq, dim]
        
        # Apply attention
        output = (attn_weights.unsqueeze(-1) * v_tensor).sum(dim=1)
        
        return output
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Calculate memory usage per layer and total."""
        memory_per_layer = {}
        total_compressed = 0
        total_original = 0
        
        for layer_idx, layer_config in self.layer_codecs.items():
            seq_len = self.seq_len_per_layer[layer_idx]
            
            # K memory
            k_usage = layer_config.k_codec.get_memory_usage(seq_len)
            
            # V memory
            v_fmt = get_format(layer_config.v_format)
            v_bytes = seq_len * (v_fmt.sq_bits * self.dim + 32) // 8
            
            layer_total = k_usage['compressed'] + v_bytes
            layer_original = k_usage['original'] * 2  # K + V
            
            memory_per_layer[layer_idx] = {
                "k_format": layer_config.k_format,
                "v_format": layer_config.v_format,
                "k_memory": k_usage['compressed'],
                "v_memory": v_bytes,
                "total_memory": layer_total,
                "compression_factor": layer_original / layer_total if layer_total > 0 else 1
            }
            
            total_compressed += layer_total
            total_original += layer_original
        
        return {
            "per_layer": memory_per_layer,
            "total_compressed_bytes": total_compressed,
            "total_original_bytes": total_original,
            "overall_compression_factor": total_original / total_compressed if total_compressed > 0 else 1
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        protected_count = 0
        for i in range(self.config.num_layers):
            if self.config.is_layer_protected(i):
                protected_count += 1
                
        return {
            "num_layers": self.config.num_layers,
            "protected_layers": protected_count,
            "compressed_layers": self.config.num_layers - protected_count,
            "config": {
                "num_layers": self.config.num_layers,
                "keep_last_n": self.config.keep_last_n,
                "protected_format": self.config.protected_k_format,
                "compressed_format": self.config.default_k_format
            },
            "seq_len_per_layer": self.seq_len_per_layer,
            "memory": self.get_memory_usage(),
            "compression_estimate": self.config.get_compression_estimate()
        }
    
    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear cache for specific or all layers.
        
        Args:
            layer_idx: Specific layer to clear, or None for all
        """
        if layer_idx is not None:
            self.cache[layer_idx]["keys"] = []
            self.cache[layer_idx]["values"] = []
            self.seq_len_per_layer[layer_idx] = 0
        else:
            for idx in self.cache:
                self.cache[idx]["keys"] = []
                self.cache[idx]["values"] = []
                self.seq_len_per_layer[idx] = 0


def create_layer_adaptive_cache(
    num_layers: int = 32,
    keep_last_n: int = 8,
    default_format: str = "turbo4",
    protected_format: str = "q8_0",
    dim: int = 4096,
    device: Optional[torch.device] = None
) -> LayerAdaptiveKVCache:
    """
    Factory function for layer-adaptive KV cache.
    
    Args:
        num_layers: Total number of transformer layers
        keep_last_n: Number of final layers to keep at high precision
        default_format: Format for compressed layers
        protected_format: Format for high-precision layers
        dim: Hidden dimension
        device: Torch device
    
    Returns:
        Configured LayerAdaptiveKVCache
    """
    config = LayerAdaptiveConfig(
        num_layers=num_layers,
        dim=dim,
        keep_last_n=keep_last_n,
        default_k_format=default_format,
        default_v_format=default_format,
        protected_k_format=protected_format,
        protected_v_format=protected_format,
        device=device
    )
    
    return LayerAdaptiveKVCache(config)


def recommend_layer_config(
    model_size: str = "7b",
    target_compression: float = 3.5
) -> LayerAdaptiveConfig:
    """
    Recommend layer-adaptive configuration based on model and requirements.
    
    Args:
        model_size: Model size (7b, 13b, 70b)
        target_compression: Target compression factor
    
    Returns:
        Recommended LayerAdaptiveConfig
    """
    # Default layer counts
    layer_counts = {
        "1b": 12,
        "3b": 24,
        "7b": 32,
        "13b": 40,
        "70b": 80
    }
    
    num_layers = layer_counts.get(model_size.lower(), 32)
    
    # Adjust high-precision layers based on target compression
    if target_compression >= 4.0:
        # Aggressive compression - fewer high-precision layers
        keep_last_n = max(4, num_layers // 8)
        default_format = "turbo3"
    elif target_compression >= 3.0:
        # Balanced
        keep_last_n = max(8, num_layers // 4)
        default_format = "turbo4"
    else:
        # Quality-focused
        keep_last_n = max(12, num_layers // 3)
        default_format = "turbo4"
    
    return LayerAdaptiveConfig(
        num_layers=num_layers,
        dim=4096,  # Default, should be adjusted per model
        keep_last_n=keep_last_n,
        default_k_format=default_format,
        default_v_format=default_format,
        protected_k_format="q8_0",
        protected_v_format="q8_0"
    )
