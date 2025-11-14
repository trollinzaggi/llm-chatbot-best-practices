"""
Session memory implementation for in-conversation memory management.

This module provides temporary memory storage for active conversation sessions,
including context management, summarization, and topic extraction.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import deque
import re
from ..core.base_memory import BaseSessionMemory
from ..core.models import Message, MessageRole, Conversation
from ..core.exceptions import MemoryLimitExceededError


class SessionMemory(BaseSessionMemory):
    """
    Implementation of session-based memory for active conversations.
    
    This class manages temporary memory within a single conversation session,
    providing features like context window management, automatic summarization,
    and topic tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize session memory.
        
        Args:
            config: Configuration with the following optional keys:
                - max_messages: Maximum messages to keep (default: 50)
                - max_tokens: Maximum token count (default: 8000)
                - summarize_after: Messages before summarization (default: 20)
                - compression_ratio: Target compression ratio (default: 0.3)
        """
        super().__init__(config)
        
    def _initialize(self) -> None:
        """Initialize session memory components."""
        self.max_messages = self.config.get('max_messages', 50)
        self.max_tokens = self.config.get('max_tokens', 8000)
        self.summarize_after = self.config.get('summarize_after', 20)
        self.compression_ratio = self.config.get('compression_ratio', 0.3)
        
        self.messages: deque = deque(maxlen=self.max_messages)
        self.conversation = Conversation()
        self.summaries: List[str] = []
        self.topics: List[str] = []
        self.current_token_count = 0
        self.message_count = 0
        self.compressed_messages: List[Dict[str, Any]] = []
    
    def add_message(self, role: MessageRole, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to session memory.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
            
        Raises:
            MemoryLimitExceededError: If memory limits are exceeded
        """
        # Estimate token count (rough approximation: 1 token per 4 characters)
        estimated_tokens = len(content) // 4
        
        # Check if we need to compress or summarize
        if self.current_token_count + estimated_tokens > self.max_tokens:
            self._manage_memory_pressure()
        
        # Create message
        message = Message(
            role=role,
            content=content,
            conversation_id=self.conversation.id,
            metadata=metadata or {},
            token_count=estimated_tokens
        )
        
        # Add to memory
        self.messages.append(message)
        self.conversation.add_message(role, content, metadata=metadata)
        self.current_token_count += estimated_tokens
        self.message_count += 1
        
        # Check if summarization is needed
        if self.message_count % self.summarize_after == 0:
            self._create_summary()
        
        # Extract topics periodically
        if self.message_count % 10 == 0:
            self._extract_and_update_topics()
        
        return message
    
    def get_messages(self, limit: Optional[int] = None, 
                    offset: int = 0) -> List[Message]:
        """
        Retrieve messages from session memory.
        
        Args:
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of Message objects
        """
        messages_list = list(self.messages)
        
        if offset:
            messages_list = messages_list[offset:]
        
        if limit:
            messages_list = messages_list[:limit]
        
        return messages_list
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[Message, float]]:
        """
        Search session memory for relevant messages.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (Message, relevance_score) tuples
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for message in self.messages:
            content_lower = message.content.lower()
            content_words = set(content_lower.split())
            
            # Calculate simple relevance score based on word overlap
            common_words = query_words.intersection(content_words)
            if common_words:
                score = len(common_words) / len(query_words)
                results.append((message, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def clear(self) -> None:
        """Clear all session memory."""
        self.messages.clear()
        self.conversation.messages.clear()
        self.summaries.clear()
        self.topics.clear()
        self.compressed_messages.clear()
        self.current_token_count = 0
        self.message_count = 0
    
    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation context for API calls.
        
        Args:
            max_tokens: Maximum token limit
            
        Returns:
            List of message dictionaries
        """
        max_tokens = max_tokens or self.max_tokens
        context = []
        token_count = 0
        
        # Add summaries if available
        if self.summaries:
            summary_context = {
                "role": "system",
                "content": f"Previous conversation summary: {' '.join(self.summaries[-2:])}"
            }
            context.append(summary_context)
            token_count += len(summary_context["content"]) // 4
        
        # Add recent messages
        for message in reversed(list(self.messages)):
            msg_dict = {
                "role": message.role.value,
                "content": message.content
            }
            msg_tokens = message.token_count or (len(message.content) // 4)
            
            if token_count + msg_tokens > max_tokens:
                break
            
            context.insert(len(context) if self.summaries else 0, msg_dict)
            token_count += msg_tokens
        
        return context
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "message_count": self.message_count,
            "current_messages": len(self.messages),
            "token_count": self.current_token_count,
            "summary_count": len(self.summaries),
            "topic_count": len(self.topics),
            "compressed_messages": len(self.compressed_messages),
            "conversation_id": self.conversation.id,
            "memory_usage_ratio": self.current_token_count / self.max_tokens
        }
    
    def summarize(self) -> str:
        """
        Generate a summary of the current session.
        
        Returns:
            Summary text
        """
        if not self.messages:
            return "No messages to summarize."
        
        # Simple extractive summarization
        # In production, use LLM for better summarization
        important_messages = []
        
        for message in self.messages:
            if message.role == MessageRole.ASSISTANT:
                # Extract first and last sentences
                sentences = message.content.split('. ')
                if sentences:
                    important_messages.append(sentences[0])
                    if len(sentences) > 1:
                        important_messages.append(sentences[-1])
        
        summary = '. '.join(important_messages[:5])
        return summary if summary else "Conversation in progress."
    
    def compress(self) -> None:
        """Compress older messages to save memory."""
        if len(self.messages) < 10:
            return
        
        # Move older messages to compressed storage
        messages_to_compress = []
        while len(self.messages) > self.max_messages * 0.7:
            msg = self.messages.popleft()
            messages_to_compress.append(msg)
        
        if messages_to_compress:
            # Create compressed representation
            compressed = {
                "timestamp_range": (
                    messages_to_compress[0].timestamp,
                    messages_to_compress[-1].timestamp
                ),
                "message_count": len(messages_to_compress),
                "summary": self._summarize_messages(messages_to_compress),
                "key_points": self._extract_key_points(messages_to_compress)
            }
            self.compressed_messages.append(compressed)
            
            # Update token count
            removed_tokens = sum(m.token_count or 0 for m in messages_to_compress)
            self.current_token_count -= removed_tokens
    
    def extract_topics(self) -> List[str]:
        """
        Extract main topics from the conversation.
        
        Returns:
            List of topics
        """
        return self.topics.copy()
    
    def _manage_memory_pressure(self) -> None:
        """Handle memory pressure by compressing or summarizing."""
        # First try compression
        self.compress()
        
        # If still over limit, create summary and clear old messages
        if self.current_token_count > self.max_tokens * 0.9:
            self._create_summary()
            
            # Keep only recent messages
            keep_count = min(10, len(self.messages))
            recent_messages = list(self.messages)[-keep_count:]
            self.messages.clear()
            self.messages.extend(recent_messages)
            
            # Recalculate token count
            self.current_token_count = sum(
                m.token_count or 0 for m in self.messages
            )
    
    def _create_summary(self) -> None:
        """Create and store a summary of recent messages."""
        if len(self.messages) < 5:
            return
        
        summary = self.summarize()
        self.summaries.append(summary)
        
        # Keep only recent summaries
        if len(self.summaries) > 5:
            self.summaries = self.summaries[-5:]
    
    def _extract_and_update_topics(self) -> None:
        """Extract and update conversation topics."""
        # Simple keyword extraction
        # In production, use NLP libraries for better extraction
        text = ' '.join(m.content for m in list(self.messages)[-10:])
        
        # Extract capitalized words as potential topics
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter common words
        common_words = {'The', 'This', 'That', 'What', 'When', 'Where', 'Why', 'How'}
        topics = [w for w in words if w not in common_words]
        
        # Update topics list
        for topic in topics:
            if topic not in self.topics:
                self.topics.append(topic)
        
        # Keep only recent topics
        if len(self.topics) > 20:
            self.topics = self.topics[-20:]
    
    def _summarize_messages(self, messages: List[Message]) -> str:
        """
        Create a summary of a list of messages.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary text
        """
        # Simple approach: concatenate key parts
        key_parts = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                # Get first sentence of user messages
                first_sentence = msg.content.split('.')[0]
                key_parts.append(f"User asked: {first_sentence}")
            elif msg.role == MessageRole.ASSISTANT:
                # Get first sentence of assistant responses
                first_sentence = msg.content.split('.')[0]
                key_parts.append(f"Assistant: {first_sentence}")
        
        return '. '.join(key_parts[:3])
    
    def _extract_key_points(self, messages: List[Message]) -> List[str]:
        """
        Extract key points from messages.
        
        Args:
            messages: Messages to process
            
        Returns:
            List of key points
        """
        key_points = []
        
        for msg in messages:
            # Look for numbered lists or bullet points
            lines = msg.content.split('\n')
            for line in lines:
                if re.match(r'^[\d\-\*]\s*[\.\)]\s*', line):
                    key_points.append(line.strip())
        
        return key_points[:5]
