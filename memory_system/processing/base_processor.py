"""
Base processor interface for memory processing.

This module defines the abstract interface for different processing strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from ..core.models import Message, MemoryFragment, Conversation


class BaseProcessor(ABC):
    """Abstract base class for memory processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize processor components."""
        pass
    
    @abstractmethod
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text and extract information.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with processed results
        """
        pass
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of processing results
        """
        return [self.process(text) for text in texts]
