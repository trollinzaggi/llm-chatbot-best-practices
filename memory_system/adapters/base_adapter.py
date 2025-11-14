"""
Base adapter class for framework integrations.

This module provides the abstract base class that all framework adapters
must implement to integrate with the memory system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from ..core.models import Message, Conversation, MemoryFragment, MessageRole
from ..core.session_memory import SessionMemory
from ..core.persistent_memory import PersistentMemory
from ..storage.sqlite_storage import SQLiteStorage


class BaseFrameworkAdapter(ABC):
    """
    Abstract base class for framework-specific memory adapters.
    
    Each framework adapter implements this interface to provide
    seamless integration between the framework and the memory system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the framework adapter.
        
        Args:
            config: Configuration dictionary with optional keys:
                - db_path: Path to SQLite database (default: "memory.db")
                - session_config: Configuration for session memory
                - persistent_config: Configuration for persistent memory
                - enable_persistent: Whether to enable persistent memory (default: True)
        """
        self.config = config or {}
        
        # Initialize session memory
        session_config = self.config.get('session_config', {})
        self.session_memory = SessionMemory(session_config)
        
        # Initialize persistent memory if enabled
        self.persistent_memory = None
        if self.config.get('enable_persistent', True):
            db_path = self.config.get('db_path', 'memory.db')
            storage_backend = SQLiteStorage(db_path)
            persistent_config = self.config.get('persistent_config', {})
            self.persistent_memory = PersistentMemory(
                storage_backend=storage_backend,
                config=persistent_config
            )
        
        self.conversation_id = None
        self.user_id = self.config.get('user_id', 'default')
        self._initialize_framework()
    
    @abstractmethod
    def _initialize_framework(self) -> None:
        """Initialize framework-specific components."""
        pass
    
    @abstractmethod
    def inject_memory_context(self, input_text: str, 
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context into the input.
        
        Args:
            input_text: Original input text
            max_context_messages: Maximum context messages to include
            
        Returns:
            Enhanced input with memory context
        """
        pass
    
    @abstractmethod
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process framework response and extract memories.
        
        Args:
            response: Framework response
            metadata: Optional metadata from the framework
            
        Returns:
            Processed response
        """
        pass
    
    @abstractmethod
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get framework-specific context data.
        
        Returns:
            Dictionary with framework-specific context
        """
        pass
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a user message to memory.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
        """
        # Add to session memory
        message = self.session_memory.add_message(
            role=MessageRole.USER,
            content=content,
            metadata=metadata
        )
        
        # Add to persistent memory if enabled
        if self.persistent_memory and self.conversation_id:
            metadata = metadata or {}
            metadata['conversation_id'] = self.conversation_id
            self.persistent_memory.add_message(
                role=MessageRole.USER,
                content=content,
                metadata=metadata
            )
        
        return message
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add an assistant message to memory.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
        """
        # Add to session memory
        message = self.session_memory.add_message(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata=metadata
        )
        
        # Add to persistent memory if enabled
        if self.persistent_memory and self.conversation_id:
            metadata = metadata or {}
            metadata['conversation_id'] = self.conversation_id
            self.persistent_memory.add_message(
                role=MessageRole.ASSISTANT,
                content=content,
                metadata=metadata
            )
        
        return message
    
    def get_conversation_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation context for the framework.
        
        Args:
            max_tokens: Maximum token limit
            
        Returns:
            List of message dictionaries
        """
        return self.session_memory.get_context(max_tokens)
    
    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryFragment]:
        """
        Retrieve relevant long-term memories.
        
        Args:
            query: Search query
            limit: Maximum memories to retrieve
            
        Returns:
            List of relevant MemoryFragment objects
        """
        if not self.persistent_memory:
            return []
        
        return self.persistent_memory.retrieve_memories(
            query=query,
            user_id=self.user_id,
            limit=limit
        )
    
    def start_conversation(self, conversation_id: Optional[str] = None, 
                         title: Optional[str] = None) -> str:
        """
        Start a new conversation.
        
        Args:
            conversation_id: Optional conversation ID
            title: Optional conversation title
            
        Returns:
            Conversation ID
        """
        # Create new conversation
        self.session_memory.conversation.user_id = self.user_id
        self.session_memory.conversation.title = title
        
        if conversation_id:
            self.session_memory.conversation.id = conversation_id
        
        self.conversation_id = self.session_memory.conversation.id
        
        return self.conversation_id
    
    def save_conversation(self) -> str:
        """
        Save the current conversation to persistent storage.
        
        Returns:
            Conversation ID
        """
        if not self.persistent_memory:
            return self.conversation_id
        
        # Save conversation
        conversation_id = self.persistent_memory.save_conversation(
            self.session_memory.conversation
        )
        
        return conversation_id
    
    def load_conversation(self, conversation_id: str) -> Conversation:
        """
        Load a conversation from persistent storage.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Loaded Conversation object
        """
        if not self.persistent_memory:
            raise ValueError("Persistent memory not enabled")
        
        conversation = self.persistent_memory.load_conversation(conversation_id)
        
        # Load into session memory
        self.session_memory.conversation = conversation
        self.session_memory.messages.clear()
        self.session_memory.messages.extend(conversation.messages)
        self.conversation_id = conversation_id
        
        return conversation
    
    def clear_session_memory(self) -> None:
        """Clear session memory."""
        self.session_memory.clear()
        self.conversation_id = None
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            'session': self.session_memory.get_statistics(),
            'framework': self.get_framework_specific_context()
        }
        
        if self.persistent_memory:
            stats['persistent'] = self.persistent_memory.get_statistics()
        
        return stats
    
    def consolidate_user_memories(self) -> None:
        """Consolidate memories for the current user."""
        if self.persistent_memory:
            self.persistent_memory.consolidate_memories(self.user_id)
    
    def build_enhanced_prompt(self, user_input: str, include_memories: bool = True) -> str:
        """
        Build an enhanced prompt with memory context.
        
        Args:
            user_input: Original user input
            include_memories: Whether to include long-term memories
            
        Returns:
            Enhanced prompt
        """
        prompt_parts = []
        
        # Add conversation context summary if available
        if self.session_memory.summaries:
            summary = self.session_memory.summaries[-1]
            prompt_parts.append(f"Previous conversation summary: {summary}")
        
        # Add relevant long-term memories if enabled
        if include_memories and self.persistent_memory:
            memories = self.retrieve_relevant_memories(user_input, limit=3)
            if memories:
                memory_context = "Relevant information from previous conversations:"
                for memory in memories:
                    memory_context += f"\n- {memory.content}"
                prompt_parts.append(memory_context)
        
        # Add current conversation topics
        topics = self.session_memory.extract_topics()
        if topics:
            prompt_parts.append(f"Current topics: {', '.join(topics[:5])}")
        
        # Add the user input
        if prompt_parts:
            prompt_parts.append(f"\nUser: {user_input}")
            return "\n\n".join(prompt_parts)
        else:
            return user_input
