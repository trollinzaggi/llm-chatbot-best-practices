"""
Semantic retriever using embeddings and vector similarity.

This module implements semantic search using embeddings for memory retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict
import json
from ..retrieval.base_retriever import BaseRetriever, RetrievalResult, RetrievalQuery
from ..core.models import Message, MemoryFragment
from ..core.exceptions import RetrievalError


class SemanticRetriever(BaseRetriever):
    """
    Semantic retriever using embeddings and cosine similarity.
    
    This retriever uses vector embeddings to find semantically similar memories.
    """
    
    def _initialize(self) -> None:
        """Initialize semantic retriever components."""
        # Embedding cache
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.content_index: Dict[str, Union[Message, MemoryFragment]] = {}
        
        # Configuration
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.similarity_metric = self.config.get('similarity_metric', 'cosine')
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Embedding provider
        self.embedding_provider = self.config.get('embedding_provider', 'local')
        self.embedding_model = None
        
        # Initialize embedding model if needed
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model based on provider."""
        if self.embedding_provider == 'openai':
            try:
                import openai
                self.embedding_model = 'text-embedding-ada-002'
            except ImportError:
                print("OpenAI not available, falling back to local embeddings")
                self.embedding_provider = 'local'
        
        if self.embedding_provider == 'local':
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to simple embeddings
                self.embedding_model = None
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve semantically similar memories.
        
        Args:
            query: Retrieval query
            
        Returns:
            List of retrieval results
        """
        if not query.use_embeddings:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query.text)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        results = []
        for content_id, content_embedding in self.embeddings_cache.items():
            similarity = self._calculate_similarity(query_embedding, content_embedding)
            
            if similarity >= query.threshold:
                content = self.content_index.get(content_id)
                if content:
                    result = RetrievalResult(
                        content=content,
                        score=similarity,
                        metadata={'similarity_type': self.similarity_metric},
                        source='semantic'
                    )
                    results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply limit
        results = results[:query.limit]
        
        # Apply filters if provided
        if query.filters:
            results = self.filter_results(results, query.filters)
        
        return results
    
    def add_to_index(self, content: Union[Message, MemoryFragment], 
                     embedding: Optional[List[float]] = None) -> None:
        """
        Add content to semantic index.
        
        Args:
            content: Content to index
            embedding: Optional pre-computed embedding
        """
        # Get content ID
        content_id = self._get_content_id(content)
        
        # Get or compute embedding
        if embedding:
            content_embedding = np.array(embedding)
        else:
            text = self._get_content_text(content)
            content_embedding = self._get_embedding(text)
        
        if content_embedding is not None:
            # Add to cache
            self.embeddings_cache[content_id] = content_embedding
            self.content_index[content_id] = content
            
            # Manage cache size
            self._manage_cache_size()
    
    def update_index(self, content_id: str, 
                    embedding: Optional[List[float]] = None) -> None:
        """
        Update content embedding in index.
        
        Args:
            content_id: Content ID
            embedding: New embedding
        """
        if content_id in self.embeddings_cache:
            if embedding:
                self.embeddings_cache[content_id] = np.array(embedding)
            else:
                # Recompute embedding
                content = self.content_index.get(content_id)
                if content:
                    text = self._get_content_text(content)
                    new_embedding = self._get_embedding(text)
                    if new_embedding is not None:
                        self.embeddings_cache[content_id] = new_embedding
    
    def remove_from_index(self, content_id: str) -> None:
        """
        Remove content from semantic index.
        
        Args:
            content_id: Content ID to remove
        """
        if content_id in self.embeddings_cache:
            del self.embeddings_cache[content_id]
        if content_id in self.content_index:
            del self.content_index[content_id]
    
    def clear_index(self) -> None:
        """Clear the semantic index."""
        self.embeddings_cache.clear()
        self.content_index.clear()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        if self.embedding_provider == 'openai':
            return self._get_openai_embedding(text)
        elif self.embedding_provider == 'local':
            return self._get_local_embedding(text)
        else:
            return self._get_simple_embedding(text)
    
    def _get_openai_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get OpenAI embedding."""
        try:
            import openai
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            return self._get_simple_embedding(text)
    
    def _get_local_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get local model embedding."""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return np.array(embedding)
            except Exception as e:
                print(f"Error getting local embedding: {e}")
                return self._get_simple_embedding(text)
        else:
            return self._get_simple_embedding(text)
    
    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """
        Get simple embedding using character and word features.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple embedding vector
        """
        # Simple embedding based on character and word features
        embedding = np.zeros(self.embedding_dim)
        
        # Character-based features
        for i, char in enumerate(text[:self.embedding_dim // 2]):
            embedding[i] = ord(char) / 255.0
        
        # Word-based features
        words = text.lower().split()
        for i, word in enumerate(words[:self.embedding_dim // 2]):
            idx = self.embedding_dim // 2 + i
            if idx < self.embedding_dim:
                # Simple hash-based feature
                embedding[idx] = (hash(word) % 1000) / 1000.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, 
                            embedding2: np.ndarray) -> float:
        """
        Calculate similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        if self.similarity_metric == 'cosine':
            return self._cosine_similarity(embedding1, embedding2)
        elif self.similarity_metric == 'euclidean':
            return self._euclidean_similarity(embedding1, embedding2)
        elif self.similarity_metric == 'dot':
            return self._dot_product_similarity(embedding1, embedding2)
        else:
            return self._cosine_similarity(embedding1, embedding2)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)
    
    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean similarity (inverse of distance)."""
        distance = np.linalg.norm(a - b)
        # Convert distance to similarity (0 to 1)
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)
    
    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate dot product similarity."""
        return float(np.dot(a, b))
    
    def _get_content_id(self, content: Union[Message, MemoryFragment]) -> str:
        """Get unique ID for content."""
        if hasattr(content, 'id') and content.id:
            return str(content.id)
        else:
            # Generate ID from content
            return str(hash(self._get_content_text(content)))
    
    def _get_content_text(self, content: Union[Message, MemoryFragment]) -> str:
        """Extract text from content."""
        if isinstance(content, Message):
            return content.content
        elif isinstance(content, MemoryFragment):
            return content.content
        else:
            return str(content)
    
    def _manage_cache_size(self) -> None:
        """Manage embedding cache size."""
        if len(self.embeddings_cache) > self.cache_size:
            # Remove oldest entries (simple FIFO for now)
            items_to_remove = len(self.embeddings_cache) - self.cache_size
            for key in list(self.embeddings_cache.keys())[:items_to_remove]:
                del self.embeddings_cache[key]
                if key in self.content_index:
                    del self.content_index[key]
    
    def find_nearest_neighbors(self, embedding: np.ndarray, 
                              k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors to an embedding.
        
        Args:
            embedding: Query embedding
            k: Number of neighbors
            
        Returns:
            List of (content_id, similarity) tuples
        """
        neighbors = []
        
        for content_id, content_embedding in self.embeddings_cache.items():
            similarity = self._calculate_similarity(embedding, content_embedding)
            neighbors.append((content_id, similarity))
        
        # Sort by similarity
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return neighbors[:k]
    
    def cluster_embeddings(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster embeddings to find memory groups.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster ID to content IDs
        """
        if not self.embeddings_cache:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            
            # Prepare embeddings matrix
            content_ids = list(self.embeddings_cache.keys())
            embeddings = np.array([self.embeddings_cache[cid] for cid in content_ids])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(content_ids)))
            labels = kmeans.fit_predict(embeddings)
            
            # Group by cluster
            clusters = defaultdict(list)
            for content_id, label in zip(content_ids, labels):
                clusters[int(label)].append(content_id)
            
            return dict(clusters)
            
        except ImportError:
            # Fallback to simple grouping
            clusters = defaultdict(list)
            content_ids = list(self.embeddings_cache.keys())
            
            for i, content_id in enumerate(content_ids):
                cluster_id = i % n_clusters
                clusters[cluster_id].append(content_id)
            
            return dict(clusters)
