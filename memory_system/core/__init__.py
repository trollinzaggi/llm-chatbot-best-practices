"""
Core memory system module.

This module provides the fundamental memory management components including
data models, base classes, and implementations for both session and persistent memory.
"""

from .models import (
    Message,
    Conversation,
    MemoryFragment,
    ConversationSummary,
    UserProfile,
    MessageRole,
    MemoryType,
    Framework
)

from .exceptions import (
    MemorySystemError,
    StorageError,
    RetrievalError,
    AdapterError,
    ConfigurationError,
    MemoryNotFoundError,
    ConversationNotFoundError,
    UserNotFoundError,
    MemoryLimitExceededError,
    SerializationError,
    EmbeddingError,
    SummarizationError,
    ExtractionError,
    ValidationError
)

from .base_memory import (
    BaseMemory,
    BaseSessionMemory,
    BasePersistentMemory
)

from .session_memory import SessionMemory
from .persistent_memory import PersistentMemory

__all__ = [
    # Models
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
    'MemoryNotFoundError',
    'ConversationNotFoundError',
    'UserNotFoundError',
    'MemoryLimitExceededError',
    'SerializationError',
    'EmbeddingError',
    'SummarizationError',
    'ExtractionError',
    'ValidationError',
    
    # Base classes
    'BaseMemory',
    'BaseSessionMemory',
    'BasePersistentMemory',
    
    # Implementations
    'SessionMemory',
    'PersistentMemory'
]
