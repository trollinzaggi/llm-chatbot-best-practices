"""
Processing manager for coordinating memory processing tasks.

This module manages different processors and provides a unified interface
for text processing, summarization, extraction, and consolidation.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from ..processing.base_processor import BaseProcessor
from ..processing.text_summarizer import TextSummarizer
from ..processing.information_extractor import InformationExtractor
from ..processing.memory_consolidator import MemoryConsolidator
from ..core.models import Message, Conversation, MemoryFragment
from ..core.exceptions import ProcessingError


class ProcessingManager:
    """
    Manager for coordinating different processing tasks.
    
    This class manages various processors and provides a unified
    interface for all memory processing operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processing manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize processors
        self.processors: Dict[str, BaseProcessor] = {}
        self._initialize_processors()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'processor_usage': {},
            'processing_time': {},
            'errors': []
        }
    
    def _initialize_processors(self) -> None:
        """Initialize configured processors."""
        # Initialize summarizer
        if self.config.get('enable_summarization', True):
            summarizer_config = self.config.get('summarizer_config', {})
            self.processors['summarizer'] = TextSummarizer(summarizer_config)
        
        # Initialize extractor
        if self.config.get('enable_extraction', True):
            extractor_config = self.config.get('extractor_config', {})
            self.processors['extractor'] = InformationExtractor(extractor_config)
        
        # Initialize consolidator
        if self.config.get('enable_consolidation', True):
            consolidator_config = self.config.get('consolidator_config', {})
            self.processors['consolidator'] = MemoryConsolidator(consolidator_config)
    
    def process_text(self, text: str, processors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process text with specified processors.
        
        Args:
            text: Text to process
            processors: List of processor names to use (uses all if None)
            
        Returns:
            Dictionary with processing results
        """
        processors = processors or list(self.processors.keys())
        results = {}
        
        self.stats['total_processed'] += 1
        
        for processor_name in processors:
            if processor_name not in self.processors:
                continue
            
            try:
                start_time = datetime.now()
                
                processor = self.processors[processor_name]
                result = processor.process(text)
                results[processor_name] = result
                
                # Update statistics
                elapsed = (datetime.now() - start_time).total_seconds()
                self._update_stats(processor_name, elapsed, success=True)
                
            except Exception as e:
                error_msg = f"Error in {processor_name}: {str(e)}"
                self.stats['errors'].append({
                    'processor': processor_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                results[processor_name] = {'error': error_msg}
                self._update_stats(processor_name, 0, success=False)
        
        return results
    
    def summarize_text(self, text: str, **kwargs) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments for summarizer
            
        Returns:
            Summary text
        """
        summarizer = self.processors.get('summarizer')
        if not summarizer:
            raise ProcessingError("Summarizer not initialized")
        
        result = summarizer.process(text)
        return result.get('summary', '')
    
    def summarize_conversation(self, conversation: Conversation) -> str:
        """
        Summarize a conversation.
        
        Args:
            conversation: Conversation to summarize
            
        Returns:
            Conversation summary
        """
        summarizer = self.processors.get('summarizer')
        if not summarizer:
            raise ProcessingError("Summarizer not initialized")
        
        if isinstance(summarizer, TextSummarizer):
            return summarizer.summarize_conversation(conversation)
        
        # Fallback to text summarization
        text = '\n'.join([
            f"{msg.role.value}: {msg.content}"
            for msg in conversation.messages
        ])
        return self.summarize_text(text)
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extraction results
        """
        extractor = self.processors.get('extractor')
        if not extractor:
            raise ProcessingError("Extractor not initialized")
        
        return extractor.process(text)
    
    def extract_memories(self, conversation: Conversation,
                        user_id: str) -> List[MemoryFragment]:
        """
        Extract memory fragments from a conversation.
        
        Args:
            conversation: Conversation to extract from
            user_id: User ID for memories
            
        Returns:
            List of extracted memory fragments
        """
        extractor = self.processors.get('extractor')
        if not extractor or not isinstance(extractor, InformationExtractor):
            raise ProcessingError("Information extractor not initialized")
        
        all_fragments = []
        
        for message in conversation.messages:
            # Extract information from message
            extraction_results = extractor.process(message.content)
            
            # Create memory fragments
            fragments = extractor.create_memory_fragments(extraction_results, user_id)
            
            # Add conversation ID
            for fragment in fragments:
                fragment.conversation_id = conversation.id
            
            all_fragments.extend(fragments)
        
        return all_fragments
    
    def consolidate_memories(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Consolidate memory fragments.
        
        Args:
            memories: Memories to consolidate
            
        Returns:
            Consolidated memories
        """
        consolidator = self.processors.get('consolidator')
        if not consolidator or not isinstance(consolidator, MemoryConsolidator):
            raise ProcessingError("Memory consolidator not initialized")
        
        return consolidator.consolidate_memories(memories)
    
    def optimize_memories(self, memories: List[MemoryFragment],
                         max_size: int = 1000) -> List[MemoryFragment]:
        """
        Optimize memory storage.
        
        Args:
            memories: Memories to optimize
            max_size: Maximum number of memories
            
        Returns:
            Optimized memories
        """
        consolidator = self.processors.get('consolidator')
        if not consolidator or not isinstance(consolidator, MemoryConsolidator):
            raise ProcessingError("Memory consolidator not initialized")
        
        return consolidator.optimize_storage(memories, max_size)
    
    def batch_process(self, texts: List[str],
                     processors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Batch process multiple texts.
        
        Args:
            texts: List of texts to process
            processors: Processors to use
            
        Returns:
            List of processing results
        """
        return [self.process_text(text, processors) for text in texts]
    
    def process_messages(self, messages: List[Message],
                        processors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process a list of messages.
        
        Args:
            messages: Messages to process
            processors: Processors to use
            
        Returns:
            List of processing results
        """
        results = []
        
        for message in messages:
            result = self.process_text(message.content, processors)
            result['message_role'] = message.role.value
            result['message_timestamp'] = message.timestamp.isoformat()
            results.append(result)
        
        return results
    
    def create_conversation_summary(self, conversation: Conversation,
                                   progressive: bool = False) -> Union[str, List[str]]:
        """
        Create conversation summary.
        
        Args:
            conversation: Conversation to summarize
            progressive: Whether to create progressive summaries
            
        Returns:
            Summary string or list of progressive summaries
        """
        summarizer = self.processors.get('summarizer')
        if not summarizer or not isinstance(summarizer, TextSummarizer):
            raise ProcessingError("Text summarizer not initialized")
        
        if progressive:
            return summarizer.progressive_summarize(conversation.messages)
        else:
            return summarizer.summarize_conversation(conversation)
    
    def extract_key_information(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Extract key information from a conversation.
        
        Args:
            conversation: Conversation to analyze
            
        Returns:
            Dictionary with extracted information
        """
        results = {
            'entities': {},
            'facts': [],
            'preferences': [],
            'topics': [],
            'dates': [],
            'numbers': {}
        }
        
        extractor = self.processors.get('extractor')
        summarizer = self.processors.get('summarizer')
        
        if not extractor:
            return results
        
        # Extract from each message
        for message in conversation.messages:
            extraction = extractor.process(message.content)
            
            # Merge entities
            for entity_type, values in extraction.get('entities', {}).items():
                if entity_type not in results['entities']:
                    results['entities'][entity_type] = []
                results['entities'][entity_type].extend(values)
            
            # Add facts
            results['facts'].extend(extraction.get('facts', []))
            
            # Add preferences
            results['preferences'].extend(extraction.get('preferences', []))
            
            # Add dates
            results['dates'].extend(extraction.get('dates', []))
            
            # Merge numbers
            for num_type, values in extraction.get('numbers', {}).items():
                if num_type not in results['numbers']:
                    results['numbers'][num_type] = []
                results['numbers'][num_type].extend(values)
        
        # Extract topics if summarizer available
        if summarizer and isinstance(summarizer, TextSummarizer):
            full_text = ' '.join(m.content for m in conversation.messages)
            topics = summarizer.get_key_points(full_text, num_points=10)
            results['topics'] = topics
        
        # Deduplicate
        for entity_type in results['entities']:
            results['entities'][entity_type] = list(set(results['entities'][entity_type]))
        
        for num_type in results['numbers']:
            results['numbers'][num_type] = list(set(results['numbers'][num_type]))
        
        return results
    
    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """
        Get a specific processor.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance or None
        """
        return self.processors.get(name)
    
    def add_processor(self, name: str, processor: BaseProcessor) -> None:
        """
        Add a custom processor.
        
        Args:
            name: Processor name
            processor: Processor instance
        """
        self.processors[name] = processor
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_processed': self.stats['total_processed'],
            'processor_usage': self.stats['processor_usage'],
            'average_processing_time': self._calculate_average_times(),
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-10:],  # Last 10 errors
            'available_processors': list(self.processors.keys())
        }
    
    def _update_stats(self, processor_name: str, elapsed_time: float, success: bool) -> None:
        """Update processor statistics."""
        if processor_name not in self.stats['processor_usage']:
            self.stats['processor_usage'][processor_name] = {'success': 0, 'failure': 0}
            self.stats['processing_time'][processor_name] = []
        
        if success:
            self.stats['processor_usage'][processor_name]['success'] += 1
            self.stats['processing_time'][processor_name].append(elapsed_time)
            
            # Keep only recent times
            if len(self.stats['processing_time'][processor_name]) > 100:
                self.stats['processing_time'][processor_name] = \
                    self.stats['processing_time'][processor_name][-100:]
        else:
            self.stats['processor_usage'][processor_name]['failure'] += 1
    
    def _calculate_average_times(self) -> Dict[str, float]:
        """Calculate average processing times."""
        averages = {}
        
        for processor_name, times in self.stats['processing_time'].items():
            if times:
                averages[processor_name] = sum(times) / len(times)
            else:
                averages[processor_name] = 0.0
        
        return averages
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'processor_usage': {},
            'processing_time': {},
            'errors': []
        }
