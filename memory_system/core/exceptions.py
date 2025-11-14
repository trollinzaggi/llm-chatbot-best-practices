"""
Custom exceptions for the memory system.

This module defines specific exceptions for different error scenarios
in the memory system, enabling better error handling and debugging.
"""


class MemorySystemError(Exception):
    """Base exception for all memory system errors."""
    pass


class StorageError(MemorySystemError):
    """Raised when database storage operations fail."""
    pass


class RetrievalError(MemorySystemError):
    """Raised when memory retrieval operations fail."""
    pass


class AdapterError(MemorySystemError):
    """Raised when framework adapter operations fail."""
    pass


class ConfigurationError(MemorySystemError):
    """Raised when there are configuration issues."""
    pass


class MemoryNotFoundError(MemorySystemError):
    """Raised when requested memory cannot be found."""
    pass


class ConversationNotFoundError(MemorySystemError):
    """Raised when requested conversation cannot be found."""
    pass


class UserNotFoundError(MemorySystemError):
    """Raised when requested user profile cannot be found."""
    pass


class MemoryLimitExceededError(MemorySystemError):
    """Raised when memory limits are exceeded."""
    pass


class SerializationError(MemorySystemError):
    """Raised when data serialization/deserialization fails."""
    pass


class EmbeddingError(MemorySystemError):
    """Raised when embedding generation fails."""
    pass


class SummarizationError(MemorySystemError):
    """Raised when text summarization fails."""
    pass


class ExtractionError(MemorySystemError):
    """Raised when information extraction fails."""
    pass


class ValidationError(MemorySystemError):
    """Raised when data validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error for field '{field}': {message}")
