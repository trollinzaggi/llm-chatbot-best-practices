"""
Storage backend implementations for the memory system.

This module provides different storage backends for persisting memory data.
"""

from .base_storage import BaseStorage
from .sqlite_storage import SQLiteStorage

__all__ = [
    'BaseStorage',
    'SQLiteStorage'
]

# Storage factory function
def create_storage(storage_type: str = 'sqlite', **kwargs):
    """
    Factory function to create the appropriate storage backend.
    
    Args:
        storage_type: Type of storage ('sqlite', 'postgresql', 'mongodb')
        **kwargs: Storage-specific arguments
        
    Returns:
        Appropriate storage backend instance
        
    Raises:
        ValueError: If storage type is not supported
    """
    storage_types = {
        'sqlite': SQLiteStorage,
        # Future implementations:
        # 'postgresql': PostgreSQLStorage,
        # 'mongodb': MongoDBStorage,
        # 'redis': RedisStorage
    }
    
    if storage_type not in storage_types:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    return storage_types[storage_type](**kwargs)
