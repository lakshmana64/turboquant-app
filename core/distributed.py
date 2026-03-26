"""
TurboQuant Distributed Support

Provides utilities for:
  - Multi-GPU KV cache distribution
  - Distributed streaming encoding
  - Tensor-parallel quantization

Usage:
    from turboquant.core.distributed import DistributedStreamingEncoder
    
    # Run across multiple GPUs
    encoder = DistributedStreamingEncoder(dim=4096, world_size=4)
    encoder.append(token)
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List

from .streaming import StreamingEncoder
from .codec import TurboQuantConfig

class DistributedStreamingEncoder:
    """
    Distributed version of StreamingEncoder for multi-GPU systems.
    
    Distributes tokens across GPUs to utilize parallel processing power
    and collective VRAM.
    """
    
    def __init__(
        self,
        dim: int,
        world_size: int,
        rank: int = 0,
        chunk_size: int = 32,
        config: Optional[TurboQuantConfig] = None,
        offload_to_cpu: bool = True
    ):
        """
        Initialize distributed encoder.
        
        Args:
            dim: Input dimension
            world_size: Number of GPUs in the group
            rank: Rank of the current process
            chunk_size: Processing chunk size
            config: Codec configuration
            offload_to_cpu: Move encoded data to CPU
        """
        self.dim = dim
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Each rank manages a subset of the data
        self.local_encoder = StreamingEncoder(
            dim, chunk_size, config, str(self.device), offload_to_cpu
        )
        
        self._total_tokens = 0
        
    def append(self, token: Tensor):
        """
        Append a token (round-robin distribution).
        """
        # Determine which rank handles this token
        target_rank = self._total_tokens % self.world_size
        
        if self.rank == target_rank:
            self.local_encoder.append(token)
            
        self._total_tokens += 1
        
    def finalize(self):
        """Finalize all local encoders."""
        self.local_encoder.finalize()
        
    def query(
        self,
        query: Tensor,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Distributed query using collective operations.
        
        Args:
            query: Query vector (dim,)
            scale: Scaling factor
            
        Returns:
            Aggregated scores from all ranks
        """
        # 1. Local query
        local_scores = self.local_encoder.query(query, scale=scale)
        
        # 2. Gather scores from all ranks (All-Gather)
        # We need to ensure all ranks have the same number of scores 
        # for a standard all-gather, or use Gatherv.
        # For simplicity in this implementation, we'll assume balanced distribution.
        
        if not dist.is_initialized():
            return local_scores
            
        all_scores = [torch.zeros_like(local_scores) for _ in range(self.world_size)]
        dist.all_gather(all_scores, local_scores)
        
        # Interleave scores to maintain original sequence order
        # (Since we distributed round-robin)
        final_scores = torch.zeros(
            self._total_tokens, device=query.device, dtype=query.dtype
        )
        
        for r in range(self.world_size):
            final_scores[r::self.world_size] = all_scores[r]
            
        return final_scores


class DistributedKVCacheStreamer:
    """
    Multi-GPU KV cache streamer for large transformer models.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        world_size: int,
        rank: int = 0,
        chunk_size: int = 32,
        config: Optional[TurboQuantConfig] = None
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        
        # Distribute heads across GPUs
        self.heads_per_gpu = num_heads // world_size
        self.start_head = rank * self.heads_per_gpu
        self.end_head = (rank + 1) * self.heads_per_gpu
        
        # Local encoders for this rank's heads
        self.local_encoders: List[List[StreamingEncoder]] = []
        for _ in range(num_layers):
            layer_encoders = [
                StreamingEncoder(head_dim, chunk_size, config, f'cuda:{rank}')
                for _ in range(self.heads_per_gpu)
            ]
            self.local_encoders.append(layer_encoders)
            
    def append_keys(self, keys: Tensor, layer_idx: int):
        """
        Append keys for a subset of heads.
        keys: (batch, num_heads, seq_len, head_dim)
        """
        # Only take the heads belonging to this rank
        local_keys = keys[:, self.start_head:self.end_head, :, :]
        
        for h_idx in range(self.heads_per_gpu):
            head_keys = local_keys[:, h_idx, :, :].view(-1, self.head_dim)
            for i in range(head_keys.shape[0]):
                self.local_encoders[layer_idx][h_idx].append(head_keys[i])
                
    def compute_attention(
        self,
        queries: Tensor,
        layer_idx: int,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Distributed attention computation.
        """
        # 1. Compute local head scores
        batch, num_heads, seq_len, head_dim = queries.shape
        local_scores = []
        
        for h_idx in range(self.heads_per_gpu):
            head_queries = queries[:, self.start_head + h_idx, :, :].view(-1, head_dim)
            scores = self.local_encoders[layer_idx][h_idx].query(head_queries, scale=scale)
            local_scores.append(scores.view(batch, seq_len, -1))
            
        local_scores_tensor = torch.stack(local_scores, dim=1) # (batch, heads_per_gpu, seq, cache)
        
        # 2. All-Gather if using multiple GPUs
        if not dist.is_initialized():
            return local_scores_tensor
            
        all_scores = [torch.zeros_like(local_scores_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_scores, local_scores_tensor)
        
        return torch.cat(all_scores, dim=1) # (batch, num_heads, seq, cache)
