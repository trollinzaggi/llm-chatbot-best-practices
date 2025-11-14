"""
Memory system adapters for different LLM frameworks.

This module provides framework-specific memory adapters that integrate
with the unified memory system.
"""

from .base_adapter import BaseFrameworkAdapter
from .agno_adapter import AgnoMemoryAdapter
from .langchain_adapter import LangChainMemoryAdapter
from .langgraph_adapter import LangGraphMemoryAdapter, GraphMemoryState
from .crewai_adapter import CrewAIMemoryAdapter
from .autogen_adapter import AutoGenMemoryAdapter
from .llama_index_adapter import LlamaIndexMemoryAdapter

__all__ = [
    'BaseFrameworkAdapter',
    'AgnoMemoryAdapter',
    'LangChainMemoryAdapter',
    'LangGraphMemoryAdapter',
    'GraphMemoryState',
    'CrewAIMemoryAdapter',
    'AutoGenMemoryAdapter',
    'LlamaIndexMemoryAdapter'
]

# Adapter factory function
def create_adapter(framework: str, **kwargs):
    """
    Factory function to create the appropriate memory adapter.
    
    Args:
        framework: Name of the framework ('agno', 'langchain', 'langgraph', 
                   'crewai', 'autogen', 'llama_index')
        **kwargs: Framework-specific arguments
        
    Returns:
        Appropriate memory adapter instance
        
    Raises:
        ValueError: If framework is not supported
    """
    adapters = {
        'agno': AgnoMemoryAdapter,
        'langchain': LangChainMemoryAdapter,
        'langgraph': LangGraphMemoryAdapter,
        'crewai': CrewAIMemoryAdapter,
        'autogen': AutoGenMemoryAdapter,
        'llama_index': LlamaIndexMemoryAdapter
    }
    
    framework_lower = framework.lower()
    if framework_lower not in adapters:
        raise ValueError(f"Unsupported framework: {framework}")
    
    return adapters[framework_lower](**kwargs)
