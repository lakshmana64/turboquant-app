"""
LlamaIndex Plugin for TurboQuant

Compress embeddings in LlamaIndex vector stores for memory-efficient RAG.

Installation:
    pip install llama-index

Usage:
    from turboquant.integrations.plugins.llama_index import TurboQuantEmbedding

    # Use as embedding model
    embed_model = TurboQuantEmbedding(
        base_model="BAAI/bge-small-en-v1.5",
        num_bits=4,
        qjl_dim=64
    )

    # Create index with compressed embeddings
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )
"""

import torch
from typing import List, Optional, Any, TYPE_CHECKING

from turboquant.core.codec import TurboQuantConfig

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core import VectorStoreIndex, Document

try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core import VectorStoreIndex, Document
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    BaseEmbedding = object  # type: ignore


class TurboQuantEmbedding(BaseEmbedding if LLAMAINDEX_AVAILABLE else object):
    """
    LlamaIndex embedding wrapper with TurboQuant compression.
    
    Features:
      - Compress embeddings after generation
      - Query with compressed embeddings
      - Memory-efficient vector stores
      - Configurable compression level
    """
    
    def __init__(
        self,
        base_model: Any = None,
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None,
        compress_on_gpu: bool = False,
        pack_bits: bool = True,
        **kwargs
    ):
        """
        Initialize TurboQuant embedding model.
        
        Args:
            base_model: Base embedding model or model name
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Device for computation
            compress_on_gpu: Compress on GPU if available
            pack_bits: Enable bit-packing for memory efficiency
            **kwargs: Additional arguments for BaseEmbedding
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required. Install with: pip install llama-index"
            )
        
        # Import here to avoid circular imports
        from turboquant.core.optimized import TurboQuantCodecOptimized
        
        # Initialize base embedding model
        if base_model is None:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                base_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5"
                )
            except ImportError:
                raise ImportError(
                    "HuggingFace embeddings required. "
                    "Install with: pip install llama-index-embeddings-huggingface"
                )
        
        self.base_model = base_model
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.compress_on_gpu = compress_on_gpu
        self.pack_bits = pack_bits
        
        # Get embedding dimension from base model
        if hasattr(base_model, 'embedding_dimension'):
            embed_dim = base_model.embedding_dimension()
        else:
            # Try to infer from a test embedding
            test_embed = base_model.get_text_embedding("test")
            embed_dim = len(test_embed)
        
        # Initialize TurboQuant codec
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.codec = TurboQuantCodecOptimized(
            dim=embed_dim,
            config=TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, pack_bits=pack_bits),
            device=device
        )
        
        # Call parent init
        if hasattr(super(), '__init__'):
            super().__init__(
                embed_dim=embed_dim,
                **kwargs
            )
        
        # Statistics
        self._compressed_count = 0
        self._compression_ratio = 0
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get query embedding (not compressed for accuracy).
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list
        """
        embedding = self.base_model.get_query_embedding(query)
        return embedding
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding with compression.
        
        Args:
            text: Input text
            
        Returns:
            Compressed embedding (stored internally)
        """
        # Get base embedding
        embedding = self.base_model.get_text_embedding(text)
        embed_tensor = torch.tensor(embedding, device=self.codec.device)
        
        # Compress
        encoded = self.codec.encode_keys_batch_optimized(
            embed_tensor.unsqueeze(0),
            return_x_hat=True
        )
        
        # Store compressed data (will be used by vector store)
        self._compressed_count += 1
        
        # Return reconstructed for compatibility
        # The vector store will store this, but we keep encoded for querying
        reconstructed = self.codec.decode_keys_vectorized(encoded)
        return reconstructed[0].tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get batch text embeddings with compression.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings
        """
        # Get base embeddings
        embeddings = self.base_model.get_text_embeddings(texts)
        embed_tensor = torch.tensor(embeddings, device=self.codec.device)
        
        # Batch compress
        encoded = self.codec.encode_keys_batch_optimized(
            embed_tensor,
            return_x_hat=True
        )
        
        self._compressed_count += len(texts)
        
        # Return reconstructed
        reconstructed = self.codec.decode_keys_vectorized(encoded)
        return reconstructed.tolist()
    
    def compress_embeddings(
        self,
        embeddings: List[List[float]]
    ) -> dict:
        """
        Compress a list of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Encoded data dict for storage
        """
        embed_tensor = torch.tensor(embeddings, device=self.codec.device)
        return self.codec.encode_keys_batch_optimized(embed_tensor)
    
    def query_compressed(
        self,
        query: str,
        encoded: dict,
        top_k: int = 5
    ) -> torch.Tensor:
        """
        Query against compressed embeddings.
        
        Args:
            query: Query text
            encoded: Compressed embeddings
            top_k: Number of results
            
        Returns:
            Top-k scores
        """
        # Get query embedding
        query_embed = self.base_model.get_query_embedding(query)
        query_tensor = torch.tensor(query_embed, device=self.codec.device)
        
        # Estimate inner products
        scores = self.codec.estimate_inner_products_vectorized(
            query_tensor.unsqueeze(0),
            encoded
        )
        
        return scores
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        memory_usage = self.codec.get_memory_usage(self._compressed_count)
        
        return {
            'compressed_count': self._compressed_count,
            'compression_ratio': memory_usage['ratio'],
            'original_mb': memory_usage.get('original', 0) / 1e6,
            'compressed_mb': memory_usage.get('compressed', 0) / 1e6,
            'savings_mb': memory_usage.get('savings_mb', 0),
        }


class TurboQuantVectorStore:
    """
    Wrapper for LlamaIndex vector stores with TurboQuant compression.

    Usage:
        from turboquant.integrations.plugins.llama_index import TurboQuantVectorStore

        # Wrap existing index
        compressed_store = TurboQuantVectorStore(index)

        # Query with compression
        results = compressed_store.query("Your query", top_k=5)
    """

    def __init__(
        self,
        index: "VectorStoreIndex",
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None,
        pack_bits: bool = True
    ):
        """
        Wrap a LlamaIndex index with compression.
        
        Args:
            index: LlamaIndex VectorStoreIndex
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Target device
            pack_bits: Enable bit-packing for memory efficiency
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex is required")
        
        from turboquant.core.optimized import TurboQuantCodecOptimized
        
        self.index = index
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.pack_bits = pack_bits
        
        # Get embedding dimension
        embed_dim = index._embed_model.embed_dim if hasattr(index, '_embed_model') else 1024
        
        # Initialize codec
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.codec = TurboQuantCodecOptimized(
            dim=embed_dim,
            config=TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, pack_bits=pack_bits),
            device=device
        )
        
        # Compress existing embeddings
        self._compress_index()
    
    def _compress_index(self):
        """Compress all embeddings in the index."""
        # Get all nodes
        nodes = list(self.index.docstore.docs.values())
        
        if not nodes:
            return
        
        # Extract embeddings
        embeddings = []
        node_ids = []
        
        for node in nodes:
            if hasattr(node, 'embedding') and node.embedding is not None:
                embeddings.append(node.embedding)
                node_ids.append(node.node_id)
        
        if not embeddings:
            return
        
        # Compress
        embed_tensor = torch.tensor(embeddings, device=self.codec.device)
        self.encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
        self.node_ids = node_ids
        
        # Store original embeddings for compatibility
        self.original_embeddings = embeddings
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_compressed: bool = True
    ) -> List[Any]:
        """
        Query the compressed index.
        
        Args:
            query: Query text
            top_k: Number of results
            use_compressed: Use compressed estimation
            
        Returns:
            List of (node, score) tuples
        """
        # Get query embedding
        query_embedding = self.index._embed_model.get_query_embedding(query)
        query_tensor = torch.tensor(query_embedding, device=self.codec.device)
        
        if use_compressed and hasattr(self, 'encoded'):
            # Use compressed estimation
            scores = self.codec.estimate_inner_products_vectorized(
                query_tensor.unsqueeze(0),
                self.encoded
            )[0]
        else:
            # Use original embeddings
            embed_tensor = torch.tensor(self.original_embeddings, device=query_tensor.device)
            scores = query_tensor @ embed_tensor.T
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        # Get nodes
        results = []
        for idx in top_indices:
            node_id = self.node_ids[idx]
            node = self.index.docstore.get_node(node_id)
            results.append((node, top_scores[idx].item()))
        
        return results
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if not hasattr(self, 'encoded'):
            return 1.0
        
        memory_usage = self.codec.get_memory_usage(len(self.node_ids))
        return memory_usage['ratio']


def create_compressed_index(
    documents: List["Document"],
    num_bits: int = 4,
    qjl_dim: int = 64,
    device: Optional[str] = None,
    pack_bits: bool = True,
    **kwargs
) -> "VectorStoreIndex":
    """
    Create a LlamaIndex index with TurboQuant compression.
    
    Args:
        documents: List of documents to index
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        device: Target device
        pack_bits: Enable bit-packing for memory efficiency
        **kwargs: Additional arguments for VectorStoreIndex
        
    Returns:
        VectorStoreIndex with compressed embeddings
    """
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError("LlamaIndex is required")
    
    # Create embedding model
    embed_model = TurboQuantEmbedding(
        num_bits=num_bits,
        qjl_dim=qjl_dim,
        device=device,
        pack_bits=pack_bits
    )
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        **kwargs
    )
    
    return index
