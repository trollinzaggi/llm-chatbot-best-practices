"""
Abstract base class for memory implementations.

This module provides the abstract interface that all memory implementations
must follow, ensuring consistency across different memory types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from ..core.models import (
    Message, Conversation, MemoryFragment, 
    MessageRole, MemoryType
)


class BaseMemory(ABC):
    """
    Abstract base class for all memory implementations.
    
    This class defines the interface that must be implemented by
    both session and persistent memory systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory system.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the memory system with necessary setup."""
        pass
    
    @abstractmethod
    def add_message(self, role: MessageRole, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to memory.
        
        Args:
            role: Role of the message sender
            content: Message content
            metadata: Optional metadata for the message
            
        Returns:
            The created Message object
        """
        pass
    
    @abstractmethod
    def get_messages(self, limit: Optional[int] = None, 
                    offset: int = 0) -> List[Message]:
        """
        Retrieve messages from memory.
        
        Args:
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            
        Returns:
            List of Message objects
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Tuple[Any, float]]:
        """
        Search memory for relevant content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of tuples containing (memory_item, relevance_score)
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory content."""
        pass
    
    @abstractmethod
    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get memory context for LLM API calls.
        
        Args:
            max_tokens: Maximum token limit for context
            
        Returns:
            List of message dictionaries suitable for API calls
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        pass


class BaseSessionMemory(BaseMemory):
    """
    Abstract base class for session-based memory.
    
    Session memory is temporary and exists only for the duration
    of a single conversation session.
    """
    
    @abstractmethod
    def summarize(self) -> str:
        """
        Generate a summary of the current session.
        
        Returns:
            Summary text
        """
        pass
    
    @abstractmethod
    def compress(self) -> None:
        """Compress older messages to save memory."""
        pass
    
    @abstractmethod
    def extract_topics(self) -> List[str]:
        """
        Extract main topics from the conversation.
        
        Returns:
            List of topic strings
        """
        pass


class BasePersistentMemory(BaseMemory):
    """
    Abstract base class for persistent memory.
    
    Persistent memory survives across sessions and is stored
    in a database or other long-term storage.
    """
    
    @abstractmethod
    def save_conversation(self, conversation: Conversation) -> str:
        """
        Save a conversation to persistent storage.
        
        Args:
            conversation: Conversation object to save
            
        Returns:
            Conversation ID
        """
        pass
    
    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Conversation:
        """
        Load a conversation from persistent storage.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Conversation object
        """
        pass
    
    @abstractmethod
    def extract_memories(self, conversation: Conversation) -> List[MemoryFragment]:
        """
        Extract memory fragments from a conversation.
        
        Args:
            conversation: Conversation to extract memories from
            
        Returns:
            List of extracted MemoryFragment objects
        """
        pass
    
    @abstractmethod
    def store_memory(self, memory: MemoryFragment) -> int:
        """
        Store a memory fragment.
        
        Args:
            memory: MemoryFragment to store
            
        Returns:
            Memory ID
        """
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, user_id: str, 
                         limit: int = 5) -> List[MemoryFragment]:
        """
        Retrieve relevant memories for a user.
        
        Args:
            query: Search query
            user_id: User ID
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant MemoryFragment objects
        """
        pass
    
    @abstractmethod
    def consolidate_memories(self, user_id: str) -> None:
        """
        Consolidate and optimize memories for a user.
        
        Args:
            user_id: User ID
        """
        pass
    
    @abstractmethod
    def forget_memories(self, user_id: str, 
                       criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Remove memories based on criteria.
        
        Args:
            user_id: User ID
            criteria: Optional criteria for memory deletion
            
        Returns:
            Number of memories removed
        """
        pass
