"""
Base retriever interface for memory retrieval.

This module defines the abstract interface for different retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from ..core.models import Message, MemoryFragment, Conversation


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    content: Union[Message, MemoryFragment, str]
    score: float
    metadata: Dict[str, Any]
    source: str  # 'session', 'persistent', 'document'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': str(self.content),
            'score': self.score,
            'metadata': self.metadata,
            'source': self.source
        }


@dataclass
class RetrievalQuery:
    """Query for retrieval operations."""
    text: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    limit: int = 5
    threshold: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    use_embeddings: bool = True
    search_type: str = 'hybrid'  # 'keyword', 'semantic', 'hybrid'


class BaseRetriever(ABC):
    """Abstract base class for memory retrievers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize retriever components."""
        pass
    
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Retrieval query
            
        Returns:
            List of retrieval results
        """
        pass
    
    @abstractmethod
    def add_to_index(self, content: Union[Message, MemoryFragment], 
                     embedding: Optional[List[float]] = None) -> None:
        """
        Add content to the retrieval index.
        
        Args:
            content: Content to index
            embedding: Optional pre-computed embedding
        """
        pass
    
    @abstractmethod
    def update_index(self, content_id: str, 
                    embedding: Optional[List[float]] = None) -> None:
        """
        Update content in the index.
        
        Args:
            content_id: ID of content to update
            embedding: New embedding
        """
        pass
    
    @abstractmethod
    def remove_from_index(self, content_id: str) -> None:
        """
        Remove content from the index.
        
        Args:
            content_id: ID of content to remove
        """
        pass
    
    @abstractmethod
    def clear_index(self) -> None:
        """Clear the entire index."""
        pass
    
    def rerank(self, results: List[RetrievalResult], 
              query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Rerank retrieval results.
        
        Args:
            results: Initial retrieval results
            query: Original query
            
        Returns:
            Reranked results
        """
        # Default: return as-is (can be overridden)
        return results
    
    def filter_results(self, results: List[RetrievalResult], 
                      filters: Dict[str, Any]) -> List[RetrievalResult]:
        """
        Apply filters to retrieval results.
        
        Args:
            results: Retrieval results
            filters: Filter criteria
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            include = True
            
            # Apply filters
            if 'min_score' in filters and result.score < filters['min_score']:
                include = False
            
            if 'source' in filters and result.source != filters['source']:
                include = False
            
            if 'user_id' in filters:
                if hasattr(result.content, 'user_id'):
                    if result.content.user_id != filters['user_id']:
                        include = False
            
            if include:
                filtered.append(result)
        
        return filtered
    
    def combine_results(self, *result_lists: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine multiple result lists and deduplicate.
        
        Args:
            *result_lists: Multiple lists of results
            
        Returns:
            Combined and deduplicated results
        """
        combined = {}
        
        for results in result_lists:
            for result in results:
                # Use content as key for deduplication
                key = str(result.content)
                if key not in combined or result.score > combined[key].score:
                    combined[key] = result
        
        # Sort by score
        return sorted(combined.values(), key=lambda x: x.score, reverse=True)
