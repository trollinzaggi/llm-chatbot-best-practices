"""
Memory processing module.

This module provides text processing capabilities including summarization,
information extraction, and memory consolidation.
"""

from .base_processor import BaseProcessor
from .text_summarizer import TextSummarizer
from .information_extractor import InformationExtractor
from .memory_consolidator import MemoryConsolidator
from .processing_manager import ProcessingManager

__all__ = [
    'BaseProcessor',
    'TextSummarizer',
    'InformationExtractor',
    'MemoryConsolidator',
    'ProcessingManager'
]


def create_processor(processor_type: str = 'summarizer', config: dict = None):
    """
    Factory function to create a processor.
    
    Args:
        processor_type: Type of processor
        config: Configuration dictionary
        
    Returns:
        Processor instance
    """
    config = config or {}
    
    if processor_type == 'summarizer':
        return TextSummarizer(config)
    elif processor_type == 'extractor':
        return InformationExtractor(config)
    elif processor_type == 'consolidator':
        return MemoryConsolidator(config)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
