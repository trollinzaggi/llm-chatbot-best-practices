"""
Memory retrieval module.

This module provides various retrieval strategies for finding relevant
memories including semantic search, keyword search, and hybrid approaches.
"""

from .base_retriever import (
    BaseRetriever,
    RetrievalQuery,
    RetrievalResult
)

from .semantic_retriever import SemanticRetriever
from .keyword_retriever import KeywordRetriever
from .hybrid_retriever import HybridRetriever
from .retrieval_manager import RetrievalManager, RetrievalCache

__all__ = [
    # Base classes
    'BaseRetriever',
    'RetrievalQuery',
    'RetrievalResult',
    
    # Retrievers
    'SemanticRetriever',
    'KeywordRetriever',
    'HybridRetriever',
    
    # Manager
    'RetrievalManager',
    'RetrievalCache'
]


def create_retriever(retriever_type: str = 'hybrid', config: Dict = None):
    """
    Factory function to create a retriever.
    
    Args:
        retriever_type: Type of retriever ('semantic', 'keyword', 'hybrid')
        config: Configuration dictionary
        
    Returns:
        Retriever instance
    """
    config = config or {}
    
    if retriever_type == 'semantic':
        return SemanticRetriever(config)
    elif retriever_type == 'keyword':
        return KeywordRetriever(config)
    elif retriever_type == 'hybrid':
        return HybridRetriever(config)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_retrieval_manager(config: Dict = None):
    """
    Create a retrieval manager with configured retrievers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RetrievalManager instance
    """
    return RetrievalManager(config)
