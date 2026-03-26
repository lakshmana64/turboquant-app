"""
LangChain Plugin for TurboQuant

Compress embeddings in LangChain vector stores for memory-efficient RAG.

Installation:
    pip install langchain langchain-community

Usage:
    from turboquant.integrations.plugins.langchain import TurboQuantEmbeddings
    
    # Use as embedding model
    embeddings = TurboQuantEmbeddings(
        base_embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        num_bits=4,
        qjl_dim=64
    )
    
    # Create vector store with compressed embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)
"""

import torch
from typing import List, Optional, Any

from turboquant.core.codec import TurboQuantConfig

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Embeddings = object


class TurboQuantEmbeddings(Embeddings if LANGCHAIN_AVAILABLE else object):
    """
    LangChain embeddings wrapper with TurboQuant compression.
    
    Features:
      - Compress document embeddings
      - Query with compressed estimation
      - Memory-efficient vector stores
      - Compatible with all LangChain vector stores
    """
    
    def __init__(
        self,
        base_embeddings: Optional[Any] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None,
        compress_on_gpu: bool = False,
    ):
        """
        Initialize TurboQuant embeddings.
        
        Args:
            base_embeddings: Base embedding model
            model_name: Model name if creating base embeddings
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Device for computation
            compress_on_gpu: Compress on GPU if available
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required. Install with: pip install langchain langchain-community"
            )
        
        # Import here
        from turboquant.core.optimized import TurboQuantCodecOptimized
        
        # Initialize base embeddings
        if base_embeddings is None:
            base_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        self.base_embeddings = base_embeddings
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.compress_on_gpu = compress_on_gpu
        
        # Get embedding dimension
        test_embed = base_embeddings.embed_query("test")
        embed_dim = len(test_embed)
        
        # Initialize codec
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.codec = TurboQuantCodecOptimized(
            dim=embed_dim,
            config=TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim),
            device=device
        )
        
        # Storage for compressed embeddings
        self._encoded_cache: Optional[dict] = None
        self._texts_cache: List[str] = []
        
        # Statistics
        self._compressed_count = 0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with compression.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Get base embeddings
        embeddings = self.base_embeddings.embed_documents(texts)
        embed_tensor = torch.tensor(embeddings, device=self.codec.device)
        
        # Compress
        encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
        
        # Store in cache
        self._texts_cache.extend(texts)
        if self._encoded_cache is None:
            self._encoded_cache = encoded
        else:
            for key in encoded:
                self._encoded_cache[key] = torch.cat(
                    [self._encoded_cache[key], encoded[key]],
                    dim=0
                )
        
        self._compressed_count += len(texts)
        
        # Return reconstructed for vector store compatibility
        reconstructed = self.codec.decode_keys_vectorized(encoded)
        return reconstructed.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query (not compressed for accuracy).
        
        Args:
            text: Query text
            
        Returns:
            Query embedding
        """
        return self.base_embeddings.embed_query(text)
    
    def embed_documents_compressed(
        self,
        texts: List[str],
        return_encoded: bool = False
    ) -> Any:
        """
        Embed documents and return compressed representation.
        
        Args:
            texts: List of texts
            return_encoded: Return encoded dict instead of reconstructed
            
        Returns:
            Encoded data or reconstructed embeddings
        """
        embeddings = self.base_embeddings.embed_documents(texts)
        embed_tensor = torch.tensor(embeddings, device=self.codec.device)
        
        encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
        
        if return_encoded:
            return encoded
        else:
            reconstructed = self.codec.decode_keys_vectorized(encoded)
            return reconstructed.tolist()
    
    def compress_embeddings(
        self,
        embeddings: List[List[float]]
    ) -> dict:
        """
        Compress pre-computed embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Encoded data dict
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
        query_embed = self.base_embeddings.embed_query(query)
        query_tensor = torch.tensor(query_embed, device=self.codec.device)
        
        scores = self.codec.estimate_inner_products_vectorized(
            query_tensor.unsqueeze(0),
            encoded
        )[0]
        
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


class TurboQuantFAISS:
    """
    FAISS vector store with TurboQuant compression.
    
    Usage:
        from turboquant.integrations.plugins.langchain import TurboQuantFAISS
        
        # Create from documents
        vectorstore = TurboQuantFAISS.from_documents(
            documents,
            model_name="all-MiniLM-L6-v2",
            num_bits=4
        )
        
        # Query
        results = vectorstore.similarity_search("query", k=5)
    """
    
    def __init__(
        self,
        vectorstore: FAISS,
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None
    ):
        """
        Wrap FAISS with compression.
        
        Args:
            vectorstore: FAISS vector store
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Target device
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required")
        
        from turboquant.core.optimized import TurboQuantCodecOptimized
        
        self.vectorstore = vectorstore
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        
        # Get embedding dimension
        if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "d"):
            embed_dim = int(vectorstore.index.d)
        else:
            embedding_backend = getattr(vectorstore, "embeddings", None)
            if embedding_backend is None:
                embedding_backend = getattr(vectorstore, "embedding_function", None)
            if embedding_backend is None:
                raise ValueError("Could not infer embedding dimension from LangChain vector store")
            embed_dim = len(embedding_backend.embed_query("turboquant-dimension-probe"))
        
        # Initialize codec
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.codec = TurboQuantCodecOptimized(
            dim=embed_dim,
            config=TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim),
            device=device
        )
        
        # Compress existing vectors
        self._compress_index()
    
    def _compress_index(self):
        """Compress FAISS index vectors."""
        if not hasattr(self.vectorstore, 'index') or self.vectorstore.index is None:
            return
        
        # Get vectors from FAISS
        num_vectors = self.vectorstore.index.ntotal
        if num_vectors == 0:
            return
        
        # Reconstruct vectors from FAISS
        vectors = []
        for i in range(num_vectors):
            vec = self.vectorstore.index.reconstruct(i)
            vectors.append(vec)
        
        embed_tensor = torch.tensor(vectors, device=self.codec.device)
        self.encoded = self.codec.encode_keys_batch_optimized(embed_tensor)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        use_compressed: bool = True
    ) -> List[Any]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results
            use_compressed: Use compressed estimation
            
        Returns:
            List of documents
        """
        results = self.similarity_search_with_score(query, k, use_compressed)
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        use_compressed: bool = True
    ) -> List[tuple]:
        """
        Search with scores.
        
        Args:
            query: Query text
            k: Number of results
            use_compressed: Use compressed estimation
            
        Returns:
            List of (document, score) tuples
        """
        # Get query embedding
        embedding_backend = getattr(self.vectorstore, "embeddings", None)
        if embedding_backend is None:
            embedding_backend = getattr(self.vectorstore, "embedding_function", None)
        if embedding_backend is None:
            raise ValueError("LangChain vector store does not expose an embedding backend")

        query_embedding = embedding_backend.embed_query(query)
        query_tensor = torch.tensor(query_embedding, device=self.codec.device)
        
        if use_compressed and hasattr(self, 'encoded'):
            # Use compressed estimation
            scores = self.codec.estimate_inner_products_vectorized(
                query_tensor.unsqueeze(0),
                self.encoded
            )[0]
        else:
            # Use FAISS search
            scores, indices = self.vectorstore.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )
        
        # Get top-k
        if hasattr(scores, 'topk'):
            top_scores, top_indices = torch.topk(scores, min(k, len(scores)))
        else:
            top_scores = torch.tensor(scores[0])
            top_indices = torch.tensor(indices[0])
        
        # Get documents
        results = []
        docs = list(self.vectorstore.docstore._dict.values())
        
        for position, idx in enumerate(top_indices.tolist()):
            if 0 <= idx < len(docs):
                results.append((docs[idx], float(top_scores[position].item())))
        
        return results
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Any],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_bits: int = 4,
        qjl_dim: int = 64,
        device: Optional[str] = None,
        **kwargs
    ) -> 'TurboQuantFAISS':
        """
        Create from documents.
        
        Args:
            documents: List of documents
            model_name: Embedding model name
            num_bits: Scalar quantization bits
            qjl_dim: QJL output dimension
            device: Target device
            **kwargs: Additional arguments for FAISS
            
        Returns:
            TurboQuantFAISS instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required")
        
        # Create embeddings
        embeddings = TurboQuantEmbeddings(
            model_name=model_name,
            num_bits=num_bits,
            qjl_dim=qjl_dim,
            device=device
        )
        
        # Create FAISS
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Wrap with compression
        return cls(vectorstore, num_bits, qjl_dim, device)
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if not hasattr(self, 'encoded'):
            return 1.0
        
        memory_usage = self.codec.get_memory_usage(self.encoded['indices'].shape[0])
        return memory_usage['ratio']


def create_compressed_vectorstore(
    documents: List[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_bits: int = 4,
    qjl_dim: int = 64,
    device: Optional[str] = None,
    **kwargs
) -> FAISS:
    """
    Create a LangChain vector store with TurboQuant compression.
    
    Args:
        documents: List of documents
        model_name: Embedding model name
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        device: Target device
        **kwargs: Additional arguments
        
    Returns:
        FAISS vector store with compressed embeddings
    """
    return TurboQuantFAISS.from_documents(
        documents,
        model_name=model_name,
        num_bits=num_bits,
        qjl_dim=qjl_dim,
        device=device,
        **kwargs
    )
