"""
Data models for the memory system.

This module defines the core data structures used throughout the memory system.
All models are designed to be database-agnostic and serializable.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid
import json


class MessageRole(Enum):
    """Enumeration of message roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    FUNCTION = "function"


class MemoryType(Enum):
    """Types of memory fragments that can be stored."""
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    SKILL = "skill"
    EPISODE = "episode"
    CONTEXT = "context"


class Framework(Enum):
    """Supported LLM frameworks."""
    AGNO = "agno"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    LLAMA_INDEX = "llama_index"


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender
        content: The text content of the message
        timestamp: When the message was created
        id: Unique identifier for the message
        conversation_id: ID of the parent conversation
        metadata: Additional framework-specific data
        token_count: Number of tokens in the message
        embedding: Optional vector embedding for semantic search
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        data = asdict(self)
        data['role'] = self.role.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.embedding:
            data['embedding'] = json.dumps(self.embedding)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        data['role'] = MessageRole(data['role'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('embedding') and isinstance(data['embedding'], str):
            data['embedding'] = json.loads(data['embedding'])
        return cls(**data)


@dataclass
class Conversation:
    """
    Represents a conversation session.
    
    Attributes:
        id: Unique conversation identifier
        user_id: ID of the user
        framework: The LLM framework being used
        title: Auto-generated conversation title
        created_at: When the conversation started
        updated_at: Last update timestamp
        messages: List of messages in the conversation
        summary: Periodic summary of the conversation
        metadata: Additional conversation data
        is_active: Whether the conversation is currently active
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    framework: Optional[Framework] = None
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            conversation_id=self.id,
            **kwargs
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_context(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation context for API calls."""
        messages = self.messages[-max_messages:] if max_messages else self.messages
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for storage."""
        data = asdict(self)
        data['framework'] = self.framework.value if self.framework else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['messages'] = [msg.to_dict() for msg in self.messages]
        return data


@dataclass
class MemoryFragment:
    """
    Represents a piece of extracted memory.
    
    Attributes:
        id: Unique identifier
        user_id: Owner of the memory
        conversation_id: Source conversation (optional)
        fragment_type: Type of memory fragment
        content: The memory content
        importance_score: Importance rating (0-1)
        created_at: When the memory was created
        last_accessed: Last time the memory was retrieved
        access_count: Number of times accessed
        expiry_date: When the memory should be forgotten (optional)
        embedding: Vector embedding for semantic search
        metadata: Additional memory data
    """
    user_id: str
    fragment_type: MemoryType
    content: str
    importance_score: float = 0.5
    id: Optional[int] = None
    conversation_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    expiry_date: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.expiry_date:
            return datetime.now() > self.expiry_date
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory fragment to dictionary."""
        data = asdict(self)
        data['fragment_type'] = self.fragment_type.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        if self.expiry_date:
            data['expiry_date'] = self.expiry_date.isoformat()
        if self.embedding:
            data['embedding'] = json.dumps(self.embedding)
        return data


@dataclass
class ConversationSummary:
    """
    Represents a summary of a conversation segment.
    
    Attributes:
        id: Unique identifier
        conversation_id: Parent conversation ID
        summary: The summary text
        key_topics: Main topics discussed
        created_at: When the summary was created
        message_count_at_summary: Number of messages when summarized
        metadata: Additional summary data
    """
    conversation_id: str
    summary: str
    key_topics: List[str] = field(default_factory=list)
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    message_count_at_summary: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['key_topics'] = json.dumps(self.key_topics)
        return data


@dataclass
class UserProfile:
    """
    Represents a user's profile and preferences.
    
    Attributes:
        user_id: Unique user identifier
        preferences: User preferences
        known_entities: Entities associated with the user
        interaction_style: Preferred interaction style
        created_at: Profile creation timestamp
        updated_at: Last update timestamp
        metadata: Additional user data
    """
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    known_entities: Dict[str, Any] = field(default_factory=dict)
    interaction_style: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference."""
        self.preferences[key] = value
        self.updated_at = datetime.now()
    
    def add_entity(self, entity_type: str, entity_value: str) -> None:
        """Add a known entity for the user."""
        if entity_type not in self.known_entities:
            self.known_entities[entity_type] = []
        if entity_value not in self.known_entities[entity_type]:
            self.known_entities[entity_type].append(entity_value)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['preferences'] = json.dumps(self.preferences)
        data['known_entities'] = json.dumps(self.known_entities)
        return data
