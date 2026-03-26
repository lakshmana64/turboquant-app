"""
TurboQuant Streaming Encoder

Memory-efficient streaming encoder for processing sequences incrementally.
Ideal for:
  - Long sequence generation
  - KV cache compression during inference
  - Memory-constrained environments

Usage:
    from turboquant.core.streaming import StreamingEncoder
    
    encoder = StreamingEncoder(dim=128, chunk_size=32)
    
    # Process tokens one at a time
    for token in sequence:
        encoder.append(token)
    
    # Or process in chunks
    encoder.append_batch(chunk)
    
    # Query at any time
    scores = encoder.query(query_vector)
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any, List, Iterator

from .optimized import TurboQuantCodecOptimized
from .codec import TurboQuantConfig


class StreamingEncoder:
    """
    Memory-efficient streaming encoder for incremental processing.
    
    Features:
      - Process sequences token by token
      - Configurable chunk size for batch efficiency
      - Constant memory footprint regardless of sequence length
      - Query at any point during encoding
    
    Memory Usage:
      - O(chunk_size * dim) instead of O(seq_len * dim)
      - Automatically flushes encoded data to CPU if on GPU
    """
    
    def __init__(
        self,
        dim: int,
        chunk_size: int = 32,
        config: Optional[TurboQuantConfig] = None,
        device: Optional[str] = None,
        offload_to_cpu: bool = True
    ):
        """
        Initialize streaming encoder.
        
        Args:
            dim: Input dimension
            chunk_size: Number of tokens to process in each batch
            config: Codec configuration
            device: Device for computation
            offload_to_cpu: Move encoded data to CPU to save GPU memory
        """
        self.dim = dim
        self.chunk_size = chunk_size
        self.config = config or TurboQuantConfig()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.offload_to_cpu = offload_to_cpu
        
        # Create codec for chunk processing
        self.codec = TurboQuantCodecOptimized(
            dim, config, device=self.device, dtype=torch.float32
        )
        
        # Buffer for current chunk
        self._buffer: List[Tensor] = []
        self._buffer_size = 0
        
        # Accumulated encoded data (on CPU for memory efficiency)
        self._encoded_data: Dict[str, List[Tensor]] = {
            'indices': [],
            'scales': [],
            'r_signs': [],
            'r_norms': [],
        }
        
        # Statistics
        self._total_tokens = 0
        self._total_chunks = 0
    
    def append(self, token: Tensor) -> bool:
        """
        Append a single token to the stream.
        
        Args:
            token: Token vector (dim,)
            
        Returns:
            True if a chunk was flushed
        """
        if token.dim() == 1:
            token = token.unsqueeze(0)
        
        self._buffer.append(token)
        self._buffer_size += 1
        self._total_tokens += 1
        
        # Flush if chunk is full
        if self._buffer_size >= self.chunk_size:
            self._flush_buffer()
            return True
        
        return False
    
    def append_batch(self, tokens: Tensor) -> int:
        """
        Append a batch of tokens.
        
        Args:
            tokens: Token tensor (n, dim)
            
        Returns:
            Number of chunks flushed
        """
        n = tokens.shape[0]
        chunks_flushed = 0
        
        for i in range(0, n, self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            
            # Add to buffer
            for j in range(chunk.shape[0]):
                self._buffer.append(chunk[j:j+1])
                self._buffer_size += 1
                self._total_tokens += 1
            
            # Flush if full
            if self._buffer_size >= self.chunk_size:
                self._flush_buffer()
                chunks_flushed += 1
        
        return chunks_flushed
    
    def _flush_buffer(self):
        """Flush current buffer to encoded storage."""
        if not self._buffer:
            return
        
        # Stack buffer
        chunk = torch.cat(self._buffer, dim=0)
        self._buffer = []
        self._buffer_size = 0
        
        # Encode chunk
        encoded = self.codec.encode_keys_batch_optimized(chunk, return_x_hat=False)
        
        # Move to CPU for memory efficiency
        if self.offload_to_cpu:
            for key in encoded:
                encoded[key] = encoded[key].cpu()
        
        # Accumulate
        for key in self._encoded_data:
            if key in encoded:
                self._encoded_data[key].append(encoded[key])
        
        self._total_chunks += 1
    
    def finalize(self):
        """Flush remaining buffer and prepare for querying."""
        self._flush_buffer()
        
        # Concatenate all chunks
        self._encoded_final: Dict[str, Tensor] = {}
        for key in self._encoded_data:
            if self._encoded_data[key]:
                self._encoded_final[key] = torch.cat(self._encoded_data[key], dim=0)
        
        # Move back to device for querying
        if self.offload_to_cpu:
            for key in self._encoded_final:
                self._encoded_final[key] = self._encoded_final[key].to(self.device)
        
        # Reconstruct x_hat for Stage 1
        self._encoded_final['x_hat'] = self.codec.decode_keys_vectorized(self._encoded_final)
        
        # Clear chunk data
        self._encoded_data = {}
    
    def query(
        self,
        query: Tensor,
        top_k: Optional[int] = None,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Query the encoded stream.
        
        Args:
            query: Query vector (dim,) or (n_q, dim)
            top_k: Return only top-k scores
            scale: Optional scaling factor
            
        Returns:
            Scores (n_k,) or (n_q, n_k)
        """
        # Ensure finalized
        if not hasattr(self, '_encoded_final'):
            self.finalize()
        
        # Estimate inner products
        scores = self.codec.estimate_inner_products_vectorized(query, self._encoded_final)
        
        # Apply scaling
        if scale is not None:
            scores = scores * scale
        
        # Top-k selection
        if top_k is not None:
            if scores.dim() == 1:
                scores, _ = torch.topk(scores, min(top_k, scores.shape[0]))
            else:
                scores, _ = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        
        return scores
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoding statistics."""
        return {
            'total_tokens': self._total_tokens,
            'total_chunks': self._total_chunks,
            'buffer_size': self._buffer_size,
            'encoded_seq_len': sum(t.shape[0] for t in self._encoded_data.get('indices', [])),
        }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage in bytes."""
        if hasattr(self, '_encoded_final'):
            # Final encoded data
            total = sum(t.numel() * t.element_size() for t in self._encoded_final.values())
        else:
            # Chunk data
            total = sum(
                sum(t.numel() * t.element_size() for t in tensors)
                for tensors in self._encoded_data.values()
            )
        
        return {
            'encoded_bytes': total,
            'encoded_mb': total / 1e6,
        }
    
    def clear(self):
        """Clear all encoded data."""
        self._buffer = []
        self._buffer_size = 0
        self._encoded_data = {
            'indices': [],
            'scales': [],
            'r_signs': [],
            'r_norms': [],
        }
        self._total_tokens = 0
        self._total_chunks = 0
        if hasattr(self, '_encoded_final'):
            del self._encoded_final


class KVCacheStreamer:
    """
    Specialized streaming encoder for transformer KV cache.
    
    Features:
      - Layer-wise caching
      - Attention score computation
      - Integration with transformer forward pass
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        chunk_size: int = 32,
        config: Optional[TurboQuantConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize KV cache streamer.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            chunk_size: Chunk size for encoding
            config: Codec configuration
            device: Target device
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Create per-layer, per-head encoders
        self.encoders: List[List[StreamingEncoder]] = []
        for _ in range(num_layers):
            layer_encoders = [
                StreamingEncoder(head_dim, chunk_size, config, device)
                for _ in range(num_heads)
            ]
            self.encoders.append(layer_encoders)
        
        self._finalized = False
    
    def append_keys(
        self,
        keys: Tensor,
        layer_idx: int
    ):
        """
        Append keys for a layer.
        
        Args:
            keys: Key tensor (batch, num_heads, seq_len, head_dim)
            layer_idx: Layer index
        """
        batch, num_heads, seq_len, head_dim = keys.shape
        
        for head_idx in range(num_heads):
            head_keys = keys[:, head_idx, :, :].view(-1, head_dim)
            for i in range(head_keys.shape[0]):
                self.encoders[layer_idx][head_idx].append(head_keys[i])
    
    def compute_attention(
        self,
        queries: Tensor,
        layer_idx: int,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Compute attention scores for a layer.
        
        Args:
            queries: Query tensor (batch, num_heads, seq_len, head_dim)
            layer_idx: Layer index
            scale: Scaling factor
            
        Returns:
            Attention scores (batch, num_heads, seq_len, cache_len)
        """
        if not self._finalized:
            self.finalize()
        
        batch, num_heads, seq_len, head_dim = queries.shape
        all_scores = []
        
        for head_idx in range(num_heads):
            head_queries = queries[:, head_idx, :, :].view(-1, head_dim)
            scores = self.encoders[layer_idx][head_idx].query(
                head_queries, scale=scale
            )
            all_scores.append(scores.view(batch, seq_len, -1))
        
        return torch.stack(all_scores, dim=1)  # (batch, num_heads, seq_len, cache_len)
    
    def finalize(self):
        """Finalize all encoders."""
        for layer in self.encoders:
            for encoder in layer:
                encoder.finalize()
        self._finalized = True
    
    def clear(self):
        """Clear all caches."""
        for layer in self.encoders:
            for encoder in layer:
                encoder.clear()
        self._finalized = False


def stream_encode(
    tokens: Iterator[Tensor],
    dim: int,
    chunk_size: int = 32,
    config: Optional[TurboQuantConfig] = None,
    device: Optional[str] = None
) -> StreamingEncoder:
    """
    Convenience function for streaming encoding.
    
    Args:
        tokens: Iterator of token vectors
        dim: Input dimension
        chunk_size: Chunk size
        config: Codec configuration
        device: Target device
        
    Returns:
        Finalized StreamingEncoder
    """
    encoder = StreamingEncoder(dim, chunk_size, config, device)
    
    for token in tokens:
        encoder.append(token)
    
    encoder.finalize()
    return encoder
