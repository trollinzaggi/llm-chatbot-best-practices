"""
Text summarization for memory compression.

This module provides various summarization strategies for compressing
conversation history and memory content.
"""

from typing import List, Dict, Optional, Any, Tuple
import re
from collections import Counter
from heapq import nlargest
from ..processing.base_processor import BaseProcessor
from ..core.models import Message, Conversation, MessageRole


class TextSummarizer(BaseProcessor):
    """
    Text summarizer for memory compression.
    
    Provides multiple summarization strategies including extractive
    and abstractive summarization.
    """
    
    def _initialize(self) -> None:
        """Initialize summarizer components."""
        self.method = self.config.get('method', 'extractive')
        self.summary_ratio = self.config.get('summary_ratio', 0.3)
        self.max_summary_length = self.config.get('max_summary_length', 500)
        self.min_sentence_length = self.config.get('min_sentence_length', 10)
        
        # For extractive summarization
        self.use_tfidf = self.config.get('use_tfidf', True)
        self.use_position_weight = self.config.get('use_position_weight', True)
        
        # LLM for abstractive summarization
        self.llm_provider = self.config.get('llm_provider', None)
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Dictionary with summary and metadata
        """
        if self.method == 'extractive':
            summary = self.extractive_summarize(text)
        elif self.method == 'abstractive':
            summary = self.abstractive_summarize(text)
        elif self.method == 'hybrid':
            summary = self.hybrid_summarize(text)
        else:
            summary = self.simple_summarize(text)
        
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / max(len(text), 1),
            'method': self.method
        }
    
    def extractive_summarize(self, text: str) -> str:
        """
        Extractive summarization using sentence ranking.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 2:
            return text
        
        # Calculate number of sentences to extract
        num_sentences = max(1, int(len(sentences) * self.summary_ratio))
        
        # Score sentences
        scores = self._score_sentences(sentences)
        
        # Get top sentences
        top_sentence_indices = nlargest(
            num_sentences,
            range(len(sentences)),
            key=lambda i: scores[i]
        )
        
        # Sort by original position
        top_sentence_indices.sort()
        
        # Build summary
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        summary = ' '.join(summary_sentences)
        
        # Truncate if too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length].rsplit(' ', 1)[0] + '...'
        
        return summary
    
    def abstractive_summarize(self, text: str) -> str:
        """
        Abstractive summarization using LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        if self.llm_provider:
            # This would call an LLM API in production
            prompt = f"Summarize the following text in {self.max_summary_length} characters or less:\n\n{text}"
            # For now, fallback to extractive
            return self.extractive_summarize(text)
        else:
            # Fallback to extractive if no LLM available
            return self.extractive_summarize(text)
    
    def hybrid_summarize(self, text: str) -> str:
        """
        Hybrid summarization combining extractive and abstractive.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        # First, extract key sentences
        extractive_summary = self.extractive_summarize(text)
        
        # Then, refine with abstractive if available
        if self.llm_provider:
            return self.abstractive_summarize(extractive_summary)
        
        return extractive_summary
    
    def simple_summarize(self, text: str) -> str:
        """
        Simple summarization by taking first and last parts.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        if len(text) <= self.max_summary_length:
            return text
        
        # Take first and last parts
        part_length = self.max_summary_length // 2 - 10
        first_part = text[:part_length].rsplit(' ', 1)[0]
        last_part = text[-part_length:].split(' ', 1)[-1] if len(text) > part_length else ""
        
        if last_part:
            return f"{first_part} ... {last_part}"
        return first_part + "..."
    
    def summarize_conversation(self, conversation: Conversation) -> str:
        """
        Summarize a conversation.
        
        Args:
            conversation: Conversation to summarize
            
        Returns:
            Conversation summary
        """
        # Extract key exchanges
        key_exchanges = []
        
        for i, message in enumerate(conversation.messages):
            # Focus on user questions and assistant answers
            if message.role == MessageRole.USER:
                # Get the next assistant message if exists
                if i + 1 < len(conversation.messages):
                    next_msg = conversation.messages[i + 1]
                    if next_msg.role == MessageRole.ASSISTANT:
                        # Summarize the exchange
                        user_summary = self._summarize_message(message.content)
                        assistant_summary = self._summarize_message(next_msg.content)
                        key_exchanges.append(f"User: {user_summary}")
                        key_exchanges.append(f"Assistant: {assistant_summary}")
        
        # Limit number of exchanges
        max_exchanges = 10
        if len(key_exchanges) > max_exchanges:
            # Take first few and last few
            key_exchanges = key_exchanges[:max_exchanges//2] + ['...'] + key_exchanges[-max_exchanges//2:]
        
        summary = '\n'.join(key_exchanges)
        
        # Add metadata
        metadata = f"Conversation with {len(conversation.messages)} messages"
        if conversation.title:
            metadata = f"{conversation.title} - {metadata}"
        
        return f"{metadata}\n\n{summary}"
    
    def progressive_summarize(self, messages: List[Message], 
                            chunk_size: int = 10) -> List[str]:
        """
        Progressive summarization for long conversations.
        
        Args:
            messages: List of messages
            chunk_size: Messages per chunk
            
        Returns:
            List of progressive summaries
        """
        summaries = []
        
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i + chunk_size]
            
            # Convert chunk to text
            chunk_text = '\n'.join([
                f"{msg.role.value}: {msg.content}"
                for msg in chunk
            ])
            
            # Summarize chunk
            summary = self.extractive_summarize(chunk_text)
            summaries.append(summary)
        
        return summaries
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and clean
        sentences = [
            s.strip() for s in sentences 
            if s.strip() and len(s.strip()) >= self.min_sentence_length
        ]
        
        return sentences
    
    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences for importance.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of scores
        """
        scores = [0.0] * len(sentences)
        
        # Calculate word frequencies
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_freq.update(words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences based on word frequencies
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            
            # TF-IDF-like scoring
            if self.use_tfidf:
                for word in words:
                    scores[i] += word_freq[word]
                scores[i] = scores[i] / max(len(words), 1)
            
            # Position weight (first and last sentences are important)
            if self.use_position_weight:
                if i == 0 or i == len(sentences) - 1:
                    scores[i] *= 1.5
                elif i < 3:
                    scores[i] *= 1.2
            
            # Length penalty (very short or very long sentences are less important)
            length_penalty = 1.0
            if len(words) < 5:
                length_penalty = 0.5
            elif len(words) > 30:
                length_penalty = 0.8
            scores[i] *= length_penalty
        
        return scores
    
    def _summarize_message(self, content: str, max_length: int = 100) -> str:
        """
        Summarize a single message.
        
        Args:
            content: Message content
            max_length: Maximum summary length
            
        Returns:
            Summarized message
        """
        if len(content) <= max_length:
            return content
        
        # Take first sentence or truncate
        first_sentence = content.split('.')[0]
        if len(first_sentence) <= max_length:
            return first_sentence + '.'
        
        return content[:max_length-3] + '...'
    
    def get_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Source text
            num_points: Number of key points
            
        Returns:
            List of key points
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_points:
            return sentences
        
        # Score and select top sentences
        scores = self._score_sentences(sentences)
        top_indices = nlargest(
            num_points,
            range(len(sentences)),
            key=lambda i: scores[i]
        )
        
        # Sort by original position
        top_indices.sort()
        
        return [sentences[i] for i in top_indices]
    
    def create_title(self, text: str, max_length: int = 50) -> str:
        """
        Create a title from text.
        
        Args:
            text: Source text
            max_length: Maximum title length
            
        Returns:
            Generated title
        """
        # Get first sentence
        first_sentence = text.split('.')[0].strip()
        
        if len(first_sentence) <= max_length:
            return first_sentence
        
        # Extract key words
        words = first_sentence.split()
        
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        key_words = [w for w in words if w.lower() not in stopwords]
        
        # Build title from key words
        title = ' '.join(key_words)
        
        if len(title) > max_length:
            title = title[:max_length-3] + '...'
        
        return title if title else first_sentence[:max_length]
