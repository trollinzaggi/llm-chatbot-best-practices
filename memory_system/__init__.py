"""
Unified Memory System for LLM Frameworks.

A comprehensive memory management system that provides both session and persistent
memory capabilities across multiple LLM frameworks.

Features:
- Session memory with automatic summarization and compression
- Persistent memory with semantic search and consolidation
- Framework adapters for Agno, LangChain, LangGraph, CrewAI, AutoGen, and LlamaIndex
- Configurable storage backends (SQLite, PostgreSQL, MongoDB)
- Memory retrieval with embeddings and similarity search
- Automatic memory extraction and learning

Quick Start:
    from memory_system import create_memory_adapter
    
    # Create adapter for your framework
    memory = create_memory_adapter('langchain', user_id='user123')
    
    # Use in conversation
    memory.add_user_message("Hello, my name is Alice")
    memory.add_assistant_message("Nice to meet you, Alice!")
    
    # Retrieve context
    context = memory.get_conversation_context()

Author: Memory System Team
Version: 1.0.0
License: MIT
"""

from .core import (
    # Models
    Message,
    Conversation,
    MemoryFragment,
    ConversationSummary,
    UserProfile,
    MessageRole,
    MemoryType,
    Framework,
    
    # Exceptions
    MemorySystemError,
    StorageError,
    RetrievalError,
    AdapterError,
    ConfigurationError,
    
    # Base classes
    BaseMemory,
    BaseSessionMemory,
    BasePersistentMemory,
    
    # Implementations
    SessionMemory,
    PersistentMemory
)

from .storage import (
    BaseStorage,
    SQLiteStorage,
    create_storage
)

from .adapters import (
    BaseFrameworkAdapter,
    AgnoMemoryAdapter,
    LangChainMemoryAdapter,
    LangGraphMemoryAdapter,
    CrewAIMemoryAdapter,
    AutoGenMemoryAdapter,
    LlamaIndexMemoryAdapter,
    create_adapter
)

from .config import (
    MemoryConfig,
    ConfigManager,
    ConfigValidator,
    load_config,
    get_default_config
)

__version__ = '1.0.0'

__all__ = [
    # Core Models
    'Message',
    'Conversation',
    'MemoryFragment',
    'ConversationSummary',
    'UserProfile',
    'MessageRole',
    'MemoryType',
    'Framework',
    
    # Exceptions
    'MemorySystemError',
    'StorageError',
    'RetrievalError',
    'AdapterError',
    'ConfigurationError',
    
    # Memory Classes
    'BaseMemory',
    'BaseSessionMemory',
    'BasePersistentMemory',
    'SessionMemory',
    'PersistentMemory',
    
    # Storage
    'BaseStorage',
    'SQLiteStorage',
    'create_storage',
    
    # Adapters
    'BaseFrameworkAdapter',
    'AgnoMemoryAdapter',
    'LangChainMemoryAdapter',
    'LangGraphMemoryAdapter',
    'CrewAIMemoryAdapter',
    'AutoGenMemoryAdapter',
    'LlamaIndexMemoryAdapter',
    'create_adapter',
    
    # Configuration
    'MemoryConfig',
    'ConfigManager',
    'ConfigValidator',
    'load_config',
    'get_default_config',
    
    # Factory functions
    'create_memory_adapter',
    'create_memory_system'
]


def create_memory_adapter(framework: str,
                         user_id: str = 'default',
                         config: MemoryConfig = None,
                         **kwargs):
    """
    Create a memory adapter for a specific framework.
    
    Args:
        framework: Framework name ('agno', 'langchain', 'langgraph', 
                   'crewai', 'autogen', 'llama_index')
        user_id: User identifier
        config: Memory configuration (uses default if None)
        **kwargs: Additional framework-specific arguments
        
    Returns:
        Framework-specific memory adapter
        
    Example:
        # Create LangChain adapter
        memory = create_memory_adapter(
            'langchain',
            user_id='user123',
            llm=my_llm_instance
        )
        
        # Create CrewAI adapter
        memory = create_memory_adapter(
            'crewai',
            user_id='user123',
            crew=my_crew_instance
        )
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Convert to dict for adapter
    config_dict = config.to_dict()
    config_dict['user_id'] = user_id
    
    # Create appropriate adapter
    return create_adapter(framework, config=config_dict, **kwargs)


def create_memory_system(storage_type: str = 'sqlite',
                        config: MemoryConfig = None,
                        **storage_kwargs):
    """
    Create a complete memory system with storage.
    
    Args:
        storage_type: Type of storage backend ('sqlite', 'postgresql', 'mongodb')
        config: Memory configuration (uses default if None)
        **storage_kwargs: Additional storage-specific arguments
        
    Returns:
        Tuple of (SessionMemory, PersistentMemory, Storage)
        
    Example:
        session_mem, persistent_mem, storage = create_memory_system(
            storage_type='sqlite',
            db_path='my_memory.db'
        )
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Create storage backend
    storage = create_storage(storage_type, **storage_kwargs)
    
    # Create session memory
    session_memory = SessionMemory(config.session.__dict__)
    
    # Create persistent memory if enabled
    persistent_memory = None
    if config.persistent.enabled:
        persistent_memory = PersistentMemory(
            storage_backend=storage,
            config=config.persistent.__dict__
        )
    
    return session_memory, persistent_memory, storage


# Version information
def get_version():
    """Get the version of the memory system."""
    return __version__


# Configuration helpers
def get_supported_frameworks():
    """Get list of supported frameworks."""
    return ['agno', 'langchain', 'langgraph', 'crewai', 'autogen', 'llama_index']


def get_supported_storage():
    """Get list of supported storage backends."""
    return ['sqlite', 'postgresql', 'mongodb']  # Note: Only SQLite is currently implemented


# Initialize default components on import
try:
    from .config.config_loader import ConfigLoader
    default_config = ConfigLoader.auto_load()
except Exception:
    # Fallback to basic defaults if auto-load fails
    default_config = get_default_config()
