"""
Abstract base storage interface.

This module defines the interface that all storage backends must implement,
allowing for database-agnostic memory storage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from ..core.models import (
    Message, Conversation, MemoryFragment,
    ConversationSummary, UserProfile
)


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    
    This interface allows the memory system to work with different
    storage backends (SQLite, PostgreSQL, MongoDB, etc.) without
    changing the core logic.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend and create necessary schema."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close storage connections and clean up resources."""
        pass
    
    # Message operations
    
    @abstractmethod
    def store_message(self, message: Message) -> int:
        """
        Store a message.
        
        Args:
            message: Message to store
            
        Returns:
            Message ID
        """
        pass
    
    @abstractmethod
    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Retrieve a message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Message object or None if not found
        """
        pass
    
    @abstractmethod
    def get_messages(self, conversation_id: Optional[str] = None,
                    limit: Optional[int] = None,
                    offset: int = 0) -> List[Message]:
        """
        Retrieve messages.
        
        Args:
            conversation_id: Optional conversation filter
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of Message objects
        """
        pass
    
    @abstractmethod
    def search_messages(self, query: str, limit: int = 10) -> List[Tuple[Message, float]]:
        """
        Search messages.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (Message, relevance_score) tuples
        """
        pass
    
    # Conversation operations
    
    @abstractmethod
    def store_conversation(self, conversation: Conversation) -> str:
        """
        Store a conversation.
        
        Args:
            conversation: Conversation to store
            
        Returns:
            Conversation ID
        """
        pass
    
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation object or None if not found
        """
        pass
    
    @abstractmethod
    def get_user_conversations(self, user_id: str,
                              limit: Optional[int] = None,
                              offset: int = 0) -> List[Conversation]:
        """
        Get conversations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum conversations
            offset: Number to skip
            
        Returns:
            List of Conversation objects
        """
        pass
    
    @abstractmethod
    def update_conversation(self, conversation: Conversation) -> None:
        """
        Update a conversation.
        
        Args:
            conversation: Conversation with updated data
        """
        pass
    
    # Memory fragment operations
    
    @abstractmethod
    def store_memory(self, memory: MemoryFragment) -> int:
        """
        Store a memory fragment.
        
        Args:
            memory: Memory fragment to store
            
        Returns:
            Memory ID
        """
        pass
    
    @abstractmethod
    def get_memory(self, memory_id: int) -> Optional[MemoryFragment]:
        """
        Retrieve a memory fragment by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            MemoryFragment or None if not found
        """
        pass
    
    @abstractmethod
    def search_memories(self, query: str, limit: int = 10) -> List[Tuple[MemoryFragment, float]]:
        """
        Search memory fragments.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (MemoryFragment, relevance_score) tuples
        """
        pass
    
    @abstractmethod
    def search_user_memories(self, user_id: str, query: str,
                           limit: int = 10,
                           search_type: str = 'semantic') -> List[MemoryFragment]:
        """
        Search memories for a specific user.
        
        Args:
            user_id: User ID
            query: Search query
            limit: Maximum results
            search_type: Type of search ('semantic', 'keyword', 'hybrid')
            
        Returns:
            List of MemoryFragment objects
        """
        pass
    
    @abstractmethod
    def get_recent_memories(self, user_id: str, limit: int = 10) -> List[MemoryFragment]:
        """
        Get recent memories for a user.
        
        Args:
            user_id: User ID
            limit: Maximum memories
            
        Returns:
            List of recent MemoryFragment objects
        """
        pass
    
    @abstractmethod
    def get_important_memories(self, user_id: str, limit: int = 10) -> List[MemoryFragment]:
        """
        Get most important memories for a user.
        
        Args:
            user_id: User ID
            limit: Maximum memories
            
        Returns:
            List of important MemoryFragment objects
        """
        pass
    
    @abstractmethod
    def get_all_user_memories(self, user_id: str) -> List[MemoryFragment]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of all MemoryFragment objects for the user
        """
        pass
    
    @abstractmethod
    def get_user_memory_count(self, user_id: str) -> int:
        """
        Get count of memories for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of memories
        """
        pass
    
    @abstractmethod
    def update_memory_access(self, memory_id: int) -> None:
        """
        Update memory access timestamp and count.
        
        Args:
            memory_id: Memory ID
        """
        pass
    
    # Memory deletion operations
    
    @abstractmethod
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory fragment.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def delete_expired_memories(self, user_id: str) -> int:
        """
        Delete expired memories for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of memories deleted
        """
        pass
    
    @abstractmethod
    def delete_low_importance_memories(self, user_id: str, threshold: float) -> int:
        """
        Delete memories below importance threshold.
        
        Args:
            user_id: User ID
            threshold: Importance threshold
            
        Returns:
            Number of memories deleted
        """
        pass
    
    @abstractmethod
    def delete_old_memories(self, user_id: str, cutoff_date: datetime) -> int:
        """
        Delete memories older than cutoff date.
        
        Args:
            user_id: User ID
            cutoff_date: Cutoff datetime
            
        Returns:
            Number of memories deleted
        """
        pass
    
    @abstractmethod
    def delete_all_user_memories(self, user_id: str) -> int:
        """
        Delete all memories for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of memories deleted
        """
        pass
    
    # Summary operations
    
    @abstractmethod
    def store_summary(self, summary: ConversationSummary) -> int:
        """
        Store a conversation summary.
        
        Args:
            summary: Summary to store
            
        Returns:
            Summary ID
        """
        pass
    
    @abstractmethod
    def get_conversation_summaries(self, conversation_id: str) -> List[ConversationSummary]:
        """
        Get summaries for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of ConversationSummary objects
        """
        pass
    
    # User profile operations
    
    @abstractmethod
    def store_user_profile(self, profile: UserProfile) -> None:
        """
        Store or update a user profile.
        
        Args:
            profile: User profile to store
        """
        pass
    
    @abstractmethod
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile or None if not found
        """
        pass
    
    @abstractmethod
    def update_user_profile(self, profile: UserProfile) -> None:
        """
        Update a user profile.
        
        Args:
            profile: Updated user profile
        """
        pass
    
    # Utility operations
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all data from storage."""
        pass
    
    @abstractmethod
    def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the storage.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def restore(self, backup_path: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        pass
