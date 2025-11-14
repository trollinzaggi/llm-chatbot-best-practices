"""
Retrieval manager for coordinating different retrieval strategies.

This module manages multiple retrievers and provides caching,
load balancing, and fallback strategies.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
from ..retrieval.base_retriever import BaseRetriever, RetrievalResult, RetrievalQuery
from ..retrieval.semantic_retriever import SemanticRetriever
from ..retrieval.keyword_retriever import KeywordRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..core.models import Message, MemoryFragment
from ..core.exceptions import RetrievalError


class RetrievalCache:
    """Cache for retrieval results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[List[RetrievalResult], datetime]] = OrderedDict()
    
    def get(self, key: str) -> Optional[List[RetrievalResult]]:
        """Get cached results."""
        if key in self.cache:
            results, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return results
        
        return None
    
    def set(self, key: str, results: List[RetrievalResult]) -> None:
        """Cache results."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (results, datetime.now())
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys
            
        Returns:
            Number of entries invalidated
        """
        if pattern:
            keys_to_remove = [k for k in self.cache if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
            return len(keys_to_remove)
        else:
            count = len(self.cache)
            self.cache.clear()
            return count


class RetrievalManager:
    """
    Manager for coordinating multiple retrieval strategies.
    
    This class manages different retrievers, provides caching,
    and implements fallback strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retrieval manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize retrievers
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._initialize_retrievers()
        
        # Initialize cache
        cache_config = self.config.get('cache', {})
        self.cache = RetrievalCache(
            max_size=cache_config.get('max_size', 100),
            ttl_seconds=cache_config.get('ttl_seconds', 300)
        )
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retriever_usage': {},
            'avg_response_time': 0
        }
    
    def _initialize_retrievers(self) -> None:
        """Initialize configured retrievers."""
        # Default retriever configuration
        default_config = self.config.get('default_retriever', 'hybrid')
        
        # Initialize semantic retriever
        if self.config.get('enable_semantic', True):
            semantic_config = self.config.get('semantic_config', {})
            self.retrievers['semantic'] = SemanticRetriever(semantic_config)
        
        # Initialize keyword retriever
        if self.config.get('enable_keyword', True):
            keyword_config = self.config.get('keyword_config', {})
            self.retrievers['keyword'] = KeywordRetriever(keyword_config)
        
        # Initialize hybrid retriever
        if self.config.get('enable_hybrid', True):
            hybrid_config = self.config.get('hybrid_config', {})
            self.retrievers['hybrid'] = HybridRetriever(hybrid_config)
        
        # Set default retriever
        self.default_retriever = default_config
    
    def retrieve(self, query: Union[str, RetrievalQuery], 
                retriever_name: Optional[str] = None,
                use_cache: bool = True) -> List[RetrievalResult]:
        """
        Retrieve memories using specified or default retriever.
        
        Args:
            query: Query string or RetrievalQuery object
            retriever_name: Optional retriever to use
            use_cache: Whether to use cache
            
        Returns:
            List of retrieval results
        """
        # Convert string to RetrievalQuery if needed
        if isinstance(query, str):
            query = RetrievalQuery(text=query)
        
        # Update statistics
        self.stats['total_queries'] += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, retriever_name)
        
        # Check cache
        if use_cache:
            cached_results = self.cache.get(cache_key)
            if cached_results is not None:
                self.stats['cache_hits'] += 1
                return cached_results
            else:
                self.stats['cache_misses'] += 1
        
        # Select retriever
        retriever_name = retriever_name or self.default_retriever
        retriever = self.retrievers.get(retriever_name)
        
        if not retriever:
            raise RetrievalError(f"Retriever '{retriever_name}' not found")
        
        # Track usage
        if retriever_name not in self.stats['retriever_usage']:
            self.stats['retriever_usage'][retriever_name] = 0
        self.stats['retriever_usage'][retriever_name] += 1
        
        # Perform retrieval with fallback
        try:
            results = retriever.retrieve(query)
        except Exception as e:
            # Try fallback retriever
            results = self._fallback_retrieval(query, exclude=retriever_name)
            if not results:
                raise RetrievalError(f"Retrieval failed: {str(e)}")
        
        # Cache results
        if use_cache:
            self.cache.set(cache_key, results)
        
        return results
    
    def add_to_index(self, content: Union[Message, MemoryFragment],
                    embedding: Optional[List[float]] = None) -> None:
        """
        Add content to all retrievers.
        
        Args:
            content: Content to index
            embedding: Optional pre-computed embedding
        """
        for retriever in self.retrievers.values():
            try:
                retriever.add_to_index(content, embedding)
            except Exception as e:
                print(f"Error adding to retriever: {e}")
        
        # Invalidate relevant cache entries
        self.cache.invalidate()
    
    def batch_add_to_index(self, contents: List[Union[Message, MemoryFragment]],
                          embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Batch add contents to index.
        
        Args:
            contents: List of contents to index
            embeddings: Optional list of embeddings
        """
        embeddings = embeddings or [None] * len(contents)
        
        for content, embedding in zip(contents, embeddings):
            self.add_to_index(content, embedding)
    
    def update_index(self, content_id: str,
                    embedding: Optional[List[float]] = None) -> None:
        """
        Update content in all retrievers.
        
        Args:
            content_id: Content ID
            embedding: New embedding
        """
        for retriever in self.retrievers.values():
            try:
                retriever.update_index(content_id, embedding)
            except Exception as e:
                print(f"Error updating retriever: {e}")
        
        # Invalidate cache
        self.cache.invalidate()
    
    def remove_from_index(self, content_id: str) -> None:
        """
        Remove content from all retrievers.
        
        Args:
            content_id: Content ID to remove
        """
        for retriever in self.retrievers.values():
            try:
                retriever.remove_from_index(content_id)
            except Exception as e:
                print(f"Error removing from retriever: {e}")
        
        # Invalidate cache
        self.cache.invalidate()
    
    def clear_index(self) -> None:
        """Clear all retriever indices."""
        for retriever in self.retrievers.values():
            retriever.clear_index()
        
        # Clear cache
        self.cache.clear()
    
    def multi_query_retrieve(self, queries: List[str],
                           retriever_name: Optional[str] = None,
                           aggregate: bool = True) -> List[RetrievalResult]:
        """
        Retrieve using multiple queries.
        
        Args:
            queries: List of query strings
            retriever_name: Retriever to use
            aggregate: Whether to aggregate results
            
        Returns:
            List of retrieval results
        """
        all_results = []
        
        for query in queries:
            results = self.retrieve(query, retriever_name, use_cache=True)
            all_results.extend(results)
        
        if aggregate:
            # Deduplicate and re-rank
            return self._aggregate_results(all_results)
        
        return all_results
    
    def get_retriever(self, name: str) -> Optional[BaseRetriever]:
        """
        Get a specific retriever.
        
        Args:
            name: Retriever name
            
        Returns:
            Retriever instance or None
        """
        return self.retrievers.get(name)
    
    def add_retriever(self, name: str, retriever: BaseRetriever) -> None:
        """
        Add a custom retriever.
        
        Args:
            name: Retriever name
            retriever: Retriever instance
        """
        self.retrievers[name] = retriever
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with statistics
        """
        cache_stats = {
            'size': len(self.cache.cache),
            'max_size': self.cache.max_size,
            'hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['total_queries'])
            )
        }
        
        return {
            'total_queries': self.stats['total_queries'],
            'cache_stats': cache_stats,
            'retriever_usage': self.stats['retriever_usage'],
            'available_retrievers': list(self.retrievers.keys()),
            'default_retriever': self.default_retriever
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimize retrieval performance based on statistics.
        
        Returns:
            Optimization report
        """
        report = {
            'recommendations': [],
            'adjustments': []
        }
        
        # Check cache hit rate
        if self.stats['total_queries'] > 10:
            hit_rate = self.stats['cache_hits'] / self.stats['total_queries']
            
            if hit_rate < 0.3:
                # Increase cache size
                old_size = self.cache.max_size
                self.cache.max_size = min(old_size * 2, 1000)
                report['adjustments'].append(
                    f"Increased cache size from {old_size} to {self.cache.max_size}"
                )
            
            if hit_rate > 0.8:
                # Cache is very effective, maybe increase TTL
                old_ttl = self.cache.ttl_seconds
                self.cache.ttl_seconds = min(old_ttl * 1.5, 3600)
                report['adjustments'].append(
                    f"Increased cache TTL from {old_ttl} to {self.cache.ttl_seconds}"
                )
        
        # Check retriever usage
        if self.stats['retriever_usage']:
            most_used = max(
                self.stats['retriever_usage'].items(),
                key=lambda x: x[1]
            )[0]
            
            if most_used != self.default_retriever:
                report['recommendations'].append(
                    f"Consider setting '{most_used}' as default retriever"
                )
        
        return report
    
    def _generate_cache_key(self, query: RetrievalQuery, 
                          retriever_name: Optional[str]) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.text,
            str(query.limit),
            str(query.threshold),
            retriever_name or self.default_retriever,
            json.dumps(query.filters or {}, sort_keys=True)
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _fallback_retrieval(self, query: RetrievalQuery,
                          exclude: Optional[str] = None) -> List[RetrievalResult]:
        """
        Attempt retrieval with fallback retrievers.
        
        Args:
            query: Retrieval query
            exclude: Retriever to exclude
            
        Returns:
            Retrieval results or empty list
        """
        for name, retriever in self.retrievers.items():
            if name != exclude:
                try:
                    results = retriever.retrieve(query)
                    if results:
                        return results
                except Exception:
                    continue
        
        return []
    
    def _aggregate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Aggregate and deduplicate results.
        
        Args:
            results: Results to aggregate
            
        Returns:
            Aggregated results
        """
        # Group by content
        content_map = {}
        
        for result in results:
            key = self._get_content_key(result.content)
            
            if key not in content_map:
                content_map[key] = result
            else:
                # Keep higher score
                if result.score > content_map[key].score:
                    content_map[key] = result
        
        # Sort by score
        aggregated = list(content_map.values())
        aggregated.sort(key=lambda x: x.score, reverse=True)
        
        return aggregated
    
    def _get_content_key(self, content: Union[Message, MemoryFragment]) -> str:
        """Get unique key for content."""
        if isinstance(content, (Message, MemoryFragment)):
            if hasattr(content, 'id') and content.id:
                return f"{type(content).__name__}_{content.id}"
            return str(hash(content.content))
        return str(hash(str(content)))
