"""
Persistent memory implementation for long-term memory storage.

This module provides persistent storage for conversations and memories
across sessions, with support for retrieval, consolidation, and forgetting.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import re
from ..core.base_memory import BasePersistentMemory
from ..core.models import (
    Message, Conversation, MemoryFragment, UserProfile,
    ConversationSummary, MessageRole, MemoryType
)
from ..core.exceptions import (
    StorageError, ConversationNotFoundError, 
    UserNotFoundError, MemoryNotFoundError
)


class PersistentMemory(BasePersistentMemory):
    """
    Implementation of persistent memory for long-term storage.
    
    This class manages persistent storage of conversations and memories,
    providing features like semantic retrieval, memory consolidation,
    and intelligent forgetting.
    """
    
    def __init__(self, storage_backend=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize persistent memory.
        
        Args:
            storage_backend: Storage backend instance
            config: Configuration dictionary with optional keys:
                - importance_threshold: Minimum importance for storage (default: 0.3)
                - max_memories_per_user: Maximum memories per user (default: 10000)
                - consolidation_threshold: Similarity threshold for consolidation (default: 0.8)
                - default_expiry_days: Default memory expiry in days (default: 90)
        """
        self.storage = storage_backend
        super().__init__(config)
    
    def _initialize(self) -> None:
        """Initialize persistent memory components."""
        self.importance_threshold = self.config.get('importance_threshold', 0.3)
        self.max_memories_per_user = self.config.get('max_memories_per_user', 10000)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 0.8)
        self.default_expiry_days = self.config.get('default_expiry_days', 90)
        
        # Cache for frequently accessed data
        self.user_profiles_cache: Dict[str, UserProfile] = {}
        self.recent_conversations_cache: Dict[str, Conversation] = {}
        
        # Ensure storage is initialized
        if self.storage:
            self.storage.initialize()
    
    def add_message(self, role: MessageRole, content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to persistent storage.
        
        Args:
            role: Message role
            content: Message content  
            metadata: Optional metadata including conversation_id
            
        Returns:
            Created Message object
        """
        metadata = metadata or {}
        conversation_id = metadata.get('conversation_id')
        
        if not conversation_id:
            raise StorageError("conversation_id required for persistent storage")
        
        message = Message(
            role=role,
            content=content,
            conversation_id=conversation_id,
            metadata=metadata
        )
        
        # Store message
        if self.storage:
            message_id = self.storage.store_message(message)
            message.id = message_id
        
        # Extract memories if it's an important message
        if self._is_important_message(message):
            self._extract_and_store_memories(message)
        
        return message
    
    def get_messages(self, limit: Optional[int] = None,
                    offset: int = 0) -> List[Message]:
        """
        Retrieve messages from persistent storage.
        
        Args:
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of Message objects
        """
        if not self.storage:
            return []
        
        return self.storage.get_messages(limit=limit, offset=offset)
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[Any, float]]:
        """
        Search persistent memory for relevant content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (memory_item, relevance_score) tuples
        """
        if not self.storage:
            return []
        
        # Search both messages and memory fragments
        message_results = self.storage.search_messages(query, limit=limit)
        memory_results = self.storage.search_memories(query, limit=limit)
        
        # Combine and sort results
        all_results = message_results + memory_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results[:limit]
    
    def clear(self) -> None:
        """Clear all persistent memory."""
        if self.storage:
            self.storage.clear_all()
        self.user_profiles_cache.clear()
        self.recent_conversations_cache.clear()
    
    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get memory context for API calls.
        
        Args:
            max_tokens: Maximum token limit
            
        Returns:
            List of message dictionaries
        """
        # This is typically handled by session memory
        # Persistent memory provides context through retrieve_memories
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get persistent memory statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.storage:
            return {}
        
        return self.storage.get_statistics()
    
    def save_conversation(self, conversation: Conversation) -> str:
        """
        Save a conversation to persistent storage.
        
        Args:
            conversation: Conversation to save
            
        Returns:
            Conversation ID
        """
        if not self.storage:
            raise StorageError("No storage backend configured")
        
        # Generate title if not set
        if not conversation.title:
            conversation.title = self._generate_conversation_title(conversation)
        
        # Save to storage
        conversation_id = self.storage.store_conversation(conversation)
        
        # Update cache
        self.recent_conversations_cache[conversation_id] = conversation
        
        # Extract and store memories
        memories = self.extract_memories(conversation)
        for memory in memories:
            self.store_memory(memory)
        
        # Create summary if conversation is long enough
        if len(conversation.messages) > 10:
            summary = self._create_conversation_summary(conversation)
            self.storage.store_summary(summary)
        
        return conversation_id
    
    def load_conversation(self, conversation_id: str) -> Conversation:
        """
        Load a conversation from persistent storage.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation object
            
        Raises:
            ConversationNotFoundError: If conversation not found
        """
        # Check cache first
        if conversation_id in self.recent_conversations_cache:
            return self.recent_conversations_cache[conversation_id]
        
        if not self.storage:
            raise StorageError("No storage backend configured")
        
        conversation = self.storage.get_conversation(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        
        # Update cache
        self.recent_conversations_cache[conversation_id] = conversation
        
        return conversation
    
    def extract_memories(self, conversation: Conversation) -> List[MemoryFragment]:
        """
        Extract memory fragments from a conversation.
        
        Args:
            conversation: Conversation to process
            
        Returns:
            List of extracted MemoryFragment objects
        """
        memories = []
        
        for message in conversation.messages:
            # Extract different types of memories
            facts = self._extract_facts(message.content)
            entities = self._extract_entities(message.content)
            preferences = self._extract_preferences(message.content)
            
            # Create memory fragments
            for fact in facts:
                memory = MemoryFragment(
                    user_id=conversation.user_id,
                    conversation_id=conversation.id,
                    fragment_type=MemoryType.FACT,
                    content=fact,
                    importance_score=self._calculate_importance(fact, message)
                )
                memories.append(memory)
            
            for entity in entities:
                memory = MemoryFragment(
                    user_id=conversation.user_id,
                    conversation_id=conversation.id,
                    fragment_type=MemoryType.ENTITY,
                    content=entity,
                    importance_score=0.6  # Entities are generally important
                )
                memories.append(memory)
            
            for preference in preferences:
                memory = MemoryFragment(
                    user_id=conversation.user_id,
                    conversation_id=conversation.id,
                    fragment_type=MemoryType.PREFERENCE,
                    content=preference,
                    importance_score=0.7  # Preferences are important
                )
                memories.append(memory)
        
        # Filter by importance threshold
        memories = [m for m in memories if m.importance_score >= self.importance_threshold]
        
        return memories
    
    def store_memory(self, memory: MemoryFragment) -> int:
        """
        Store a memory fragment.
        
        Args:
            memory: Memory fragment to store
            
        Returns:
            Memory ID
        """
        if not self.storage:
            raise StorageError("No storage backend configured")
        
        # Set expiry date if not set
        if not memory.expiry_date:
            memory.expiry_date = datetime.now() + timedelta(days=self.default_expiry_days)
        
        # Check user memory limit
        user_memory_count = self.storage.get_user_memory_count(memory.user_id)
        if user_memory_count >= self.max_memories_per_user:
            # Remove least important memories
            self._cleanup_user_memories(memory.user_id)
        
        # Store memory
        memory_id = self.storage.store_memory(memory)
        memory.id = memory_id
        
        return memory_id
    
    def retrieve_memories(self, query: str, user_id: str,
                         limit: int = 5) -> List[MemoryFragment]:
        """
        Retrieve relevant memories for a user.
        
        Args:
            query: Search query
            user_id: User ID
            limit: Maximum memories to retrieve
            
        Returns:
            List of relevant MemoryFragment objects
        """
        if not self.storage:
            return []
        
        # Get user profile for context
        user_profile = self._get_or_create_user_profile(user_id)
        
        # Retrieve memories with different strategies
        semantic_memories = self.storage.search_user_memories(
            user_id, query, limit=limit, search_type='semantic'
        )
        
        recent_memories = self.storage.get_recent_memories(
            user_id, limit=limit // 2
        )
        
        important_memories = self.storage.get_important_memories(
            user_id, limit=limit // 2
        )
        
        # Combine and deduplicate
        all_memories = {}
        for memory in semantic_memories + recent_memories + important_memories:
            if memory.id not in all_memories:
                all_memories[memory.id] = memory
                # Update access count
                memory.access()
        
        # Sort by relevance and importance
        sorted_memories = sorted(
            all_memories.values(),
            key=lambda m: (m.importance_score, m.access_count),
            reverse=True
        )
        
        return sorted_memories[:limit]
    
    def consolidate_memories(self, user_id: str) -> None:
        """
        Consolidate and optimize memories for a user.
        
        Args:
            user_id: User ID
        """
        if not self.storage:
            return
        
        # Get all user memories
        memories = self.storage.get_all_user_memories(user_id)
        
        # Group similar memories
        memory_groups = self._group_similar_memories(memories)
        
        # Consolidate each group
        for group in memory_groups:
            if len(group) > 1:
                consolidated = self._consolidate_memory_group(group)
                # Store consolidated memory
                self.store_memory(consolidated)
                # Mark original memories for deletion
                for memory in group:
                    memory.expiry_date = datetime.now()
        
        # Clean up expired memories
        self.forget_memories(user_id, {'expired': True})
    
    def forget_memories(self, user_id: str,
                       criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Remove memories based on criteria.
        
        Args:
            user_id: User ID
            criteria: Deletion criteria
            
        Returns:
            Number of memories removed
        """
        if not self.storage:
            return 0
        
        criteria = criteria or {}
        
        # Default criteria: expired memories
        if 'expired' in criteria:
            count = self.storage.delete_expired_memories(user_id)
        # Low importance memories
        elif 'low_importance' in criteria:
            threshold = criteria.get('threshold', 0.2)
            count = self.storage.delete_low_importance_memories(user_id, threshold)
        # Old memories
        elif 'older_than_days' in criteria:
            days = criteria['older_than_days']
            cutoff_date = datetime.now() - timedelta(days=days)
            count = self.storage.delete_old_memories(user_id, cutoff_date)
        # All memories
        elif 'all' in criteria:
            count = self.storage.delete_all_user_memories(user_id)
        else:
            count = 0
        
        return count
    
    def _is_important_message(self, message: Message) -> bool:
        """Check if a message is important enough to extract memories from."""
        # User messages are generally important
        if message.role == MessageRole.USER:
            return True
        
        # Long assistant messages might contain important information
        if message.role == MessageRole.ASSISTANT and len(message.content) > 200:
            return True
        
        # Check for important keywords
        important_keywords = ['remember', 'important', 'note', 'my name', 'I am', 'I like']
        content_lower = message.content.lower()
        return any(keyword in content_lower for keyword in important_keywords)
    
    def _extract_and_store_memories(self, message: Message) -> None:
        """Extract and store memories from a message."""
        conversation = Conversation(id=message.conversation_id, messages=[message])
        memories = self.extract_memories(conversation)
        for memory in memories:
            self.store_memory(memory)
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract factual statements from content."""
        facts = []
        
        # Look for statements with "is", "are", "was", "were"
        sentences = content.split('.')
        for sentence in sentences:
            if any(verb in sentence for verb in [' is ', ' are ', ' was ', ' were ']):
                facts.append(sentence.strip())
        
        return facts[:3]  # Limit to top 3 facts
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        entities = []
        
        # Simple approach: extract capitalized sequences
        pattern = r'\b[A-Z][a-z]+ ?[A-Z]?[a-z]*\b'
        matches = re.findall(pattern, content)
        
        # Filter common words
        common_words = {'The', 'This', 'That', 'When', 'Where', 'What', 'Why', 'How'}
        entities = [m for m in matches if m.split()[0] not in common_words]
        
        return entities[:5]  # Limit to top 5 entities
    
    def _extract_preferences(self, content: str) -> List[str]:
        """Extract user preferences from content."""
        preferences = []
        
        # Look for preference indicators
        preference_patterns = [
            r"I (?:like|love|enjoy|prefer) ([^.]+)",
            r"I (?:don't|do not|dislike|hate) ([^.]+)",
            r"My favorite ([^.]+)",
            r"I'm (?:a|an) ([^.]+)"
        ]
        
        for pattern in preference_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            preferences.extend(matches)
        
        return preferences[:3]  # Limit to top 3 preferences
    
    def _calculate_importance(self, content: str, message: Message) -> float:
        """
        Calculate importance score for a memory.
        
        Args:
            content: Memory content
            message: Source message
            
        Returns:
            Importance score (0-1)
        """
        score = 0.5  # Base score
        
        # User messages are more important
        if message.role == MessageRole.USER:
            score += 0.2
        
        # Longer content might be more important
        if len(content) > 100:
            score += 0.1
        
        # Check for importance indicators
        importance_indicators = ['important', 'remember', 'note', 'always', 'never']
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in importance_indicators):
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_conversation_title(self, conversation: Conversation) -> str:
        """Generate a title for a conversation."""
        if not conversation.messages:
            return "Empty Conversation"
        
        # Use first user message as basis for title
        for message in conversation.messages:
            if message.role == MessageRole.USER:
                # Take first 50 characters
                title = message.content[:50]
                if len(message.content) > 50:
                    title += "..."
                return title
        
        return "Conversation"
    
    def _create_conversation_summary(self, conversation: Conversation) -> ConversationSummary:
        """Create a summary for a conversation."""
        # Extract key topics
        topics = []
        for message in conversation.messages:
            entities = self._extract_entities(message.content)
            topics.extend(entities)
        
        # Remove duplicates
        topics = list(set(topics))[:5]
        
        # Create summary text
        summary_parts = []
        for i, message in enumerate(conversation.messages[:5]):
            if message.role == MessageRole.USER:
                summary_parts.append(f"User: {message.content[:100]}")
            elif message.role == MessageRole.ASSISTANT:
                summary_parts.append(f"Assistant: {message.content[:100]}")
        
        summary_text = ' '.join(summary_parts)
        
        return ConversationSummary(
            conversation_id=conversation.id,
            summary=summary_text,
            key_topics=topics,
            message_count_at_summary=len(conversation.messages)
        )
    
    def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile."""
        if user_id in self.user_profiles_cache:
            return self.user_profiles_cache[user_id]
        
        if self.storage:
            profile = self.storage.get_user_profile(user_id)
            if not profile:
                profile = UserProfile(user_id=user_id)
                self.storage.store_user_profile(profile)
        else:
            profile = UserProfile(user_id=user_id)
        
        self.user_profiles_cache[user_id] = profile
        return profile
    
    def _group_similar_memories(self, memories: List[MemoryFragment]) -> List[List[MemoryFragment]]:
        """Group similar memories for consolidation."""
        groups = []
        used = set()
        
        for i, memory1 in enumerate(memories):
            if i in used:
                continue
            
            group = [memory1]
            used.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                # Check similarity (simple approach using word overlap)
                similarity = self._calculate_similarity(memory1.content, memory2.content)
                if similarity >= self.consolidation_threshold:
                    group.append(memory2)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _consolidate_memory_group(self, memories: List[MemoryFragment]) -> MemoryFragment:
        """Consolidate a group of similar memories."""
        # Combine content
        combined_content = ' '.join(m.content for m in memories)
        
        # Use highest importance score
        max_importance = max(m.importance_score for m in memories)
        
        # Sum access counts
        total_access = sum(m.access_count for m in memories)
        
        # Create consolidated memory
        consolidated = MemoryFragment(
            user_id=memories[0].user_id,
            fragment_type=memories[0].fragment_type,
            content=combined_content[:500],  # Limit length
            importance_score=min(max_importance * 1.2, 1.0),  # Boost importance
            access_count=total_access,
            metadata={'consolidated_from': len(memories)}
        )
        
        return consolidated
    
    def _cleanup_user_memories(self, user_id: str) -> None:
        """Clean up memories when user reaches limit."""
        # Remove low importance memories
        self.forget_memories(user_id, {'low_importance': True, 'threshold': 0.3})
        
        # If still over limit, remove old memories
        count = self.storage.get_user_memory_count(user_id)
        if count >= self.max_memories_per_user:
            self.forget_memories(user_id, {'older_than_days': 60})
