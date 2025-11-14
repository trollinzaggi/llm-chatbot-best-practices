"""
Memory consolidation for optimization and deduplication.

This module provides consolidation strategies for merging similar memories
and optimizing memory storage.
"""

from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from ..processing.base_processor import BaseProcessor
from ..core.models import MemoryFragment, MemoryType


class MemoryConsolidator(BaseProcessor):
    """
    Memory consolidator for optimization and deduplication.
    
    Consolidates similar memories, removes redundancies, and optimizes
    memory storage for better retrieval performance.
    """
    
    def _initialize(self) -> None:
        """Initialize consolidator components."""
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.importance_boost = self.config.get('importance_boost', 0.1)
        self.max_consolidation_size = self.config.get('max_consolidation_size', 5)
        self.preserve_recent = self.config.get('preserve_recent_days', 7)
        self.consolidation_strategy = self.config.get('strategy', 'similarity')
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process is not applicable for consolidator.
        
        Args:
            text: Not used
            
        Returns:
            Empty dictionary
        """
        return {}
    
    def consolidate_memories(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Consolidate a list of memories.
        
        Args:
            memories: List of memories to consolidate
            
        Returns:
            Consolidated list of memories
        """
        if len(memories) <= 1:
            return memories
        
        # Group memories by type
        memories_by_type = self._group_by_type(memories)
        
        consolidated = []
        
        for memory_type, type_memories in memories_by_type.items():
            if self.consolidation_strategy == 'similarity':
                consolidated_type = self._consolidate_by_similarity(type_memories)
            elif self.consolidation_strategy == 'temporal':
                consolidated_type = self._consolidate_by_time(type_memories)
            elif self.consolidation_strategy == 'importance':
                consolidated_type = self._consolidate_by_importance(type_memories)
            else:
                consolidated_type = type_memories
            
            consolidated.extend(consolidated_type)
        
        return consolidated
    
    def _consolidate_by_similarity(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Consolidate memories based on content similarity.
        
        Args:
            memories: Memories to consolidate
            
        Returns:
            Consolidated memories
        """
        if len(memories) <= 1:
            return memories
        
        # Find similar memory groups
        groups = self._find_similar_groups(memories)
        
        consolidated = []
        
        for group in groups:
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Merge similar memories
                merged = self._merge_memory_group(group)
                consolidated.append(merged)
        
        return consolidated
    
    def _consolidate_by_time(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Consolidate memories based on temporal proximity.
        
        Args:
            memories: Memories to consolidate
            
        Returns:
            Consolidated memories
        """
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        # Group memories within time windows
        time_window = timedelta(hours=1)  # 1-hour window
        groups = []
        current_group = [sorted_memories[0]]
        
        for memory in sorted_memories[1:]:
            if memory.created_at - current_group[-1].created_at <= time_window:
                current_group.append(memory)
            else:
                groups.append(current_group)
                current_group = [memory]
        
        if current_group:
            groups.append(current_group)
        
        # Consolidate each temporal group
        consolidated = []
        for group in groups:
            if len(group) == 1:
                consolidated.append(group[0])
            elif len(group) <= self.max_consolidation_size:
                merged = self._merge_memory_group(group)
                consolidated.append(merged)
            else:
                # Group too large, keep most important
                group.sort(key=lambda m: m.importance_score, reverse=True)
                consolidated.extend(group[:self.max_consolidation_size])
        
        return consolidated
    
    def _consolidate_by_importance(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Consolidate by keeping only important memories.
        
        Args:
            memories: Memories to consolidate
            
        Returns:
            Important memories only
        """
        # Calculate importance scores
        for memory in memories:
            # Adjust importance based on access frequency and recency
            recency_score = self._calculate_recency_score(memory)
            access_score = min(memory.access_count / 10, 1.0)
            
            # Combined importance
            memory.importance_score = (
                memory.importance_score * 0.5 +
                recency_score * 0.3 +
                access_score * 0.2
            )
        
        # Sort by importance
        sorted_memories = sorted(memories, key=lambda m: m.importance_score, reverse=True)
        
        # Keep top memories and consolidate similar ones in the rest
        keep_count = max(len(memories) // 3, 5)  # Keep at least top third
        important = sorted_memories[:keep_count]
        less_important = sorted_memories[keep_count:]
        
        # Try to consolidate less important memories
        if less_important:
            consolidated_rest = self._consolidate_by_similarity(less_important)
            # Keep only the most important from consolidated
            consolidated_rest.sort(key=lambda m: m.importance_score, reverse=True)
            important.extend(consolidated_rest[:max(1, len(consolidated_rest) // 2)])
        
        return important
    
    def deduplicate_memories(self, memories: List[MemoryFragment]) -> List[MemoryFragment]:
        """
        Remove duplicate memories.
        
        Args:
            memories: Memories to deduplicate
            
        Returns:
            Deduplicated memories
        """
        seen_content = set()
        unique_memories = []
        
        for memory in memories:
            content_key = self._get_content_key(memory.content)
            
            if content_key not in seen_content:
                unique_memories.append(memory)
                seen_content.add(content_key)
            else:
                # If duplicate, keep the one with higher importance
                for i, existing in enumerate(unique_memories):
                    if self._get_content_key(existing.content) == content_key:
                        if memory.importance_score > existing.importance_score:
                            unique_memories[i] = memory
                        break
        
        return unique_memories
    
    def optimize_storage(self, memories: List[MemoryFragment],
                        max_size: int = 1000) -> List[MemoryFragment]:
        """
        Optimize memory storage to fit within size limits.
        
        Args:
            memories: Memories to optimize
            max_size: Maximum number of memories
            
        Returns:
            Optimized list of memories
        """
        if len(memories) <= max_size:
            return memories
        
        # First, remove expired memories
        current_time = datetime.now()
        non_expired = [
            m for m in memories 
            if not m.expiry_date or m.expiry_date > current_time
        ]
        
        if len(non_expired) <= max_size:
            return non_expired
        
        # Consolidate to reduce size
        consolidated = self.consolidate_memories(non_expired)
        
        if len(consolidated) <= max_size:
            return consolidated
        
        # If still too large, keep most important
        consolidated.sort(key=lambda m: (m.importance_score, m.access_count), reverse=True)
        return consolidated[:max_size]
    
    def _group_by_type(self, memories: List[MemoryFragment]) -> Dict[MemoryType, List[MemoryFragment]]:
        """Group memories by type."""
        groups = defaultdict(list)
        for memory in memories:
            groups[memory.fragment_type].append(memory)
        return dict(groups)
    
    def _find_similar_groups(self, memories: List[MemoryFragment]) -> List[List[MemoryFragment]]:
        """Find groups of similar memories."""
        if len(memories) <= 1:
            return [memories]
        
        groups = []
        used = set()
        
        for i, memory1 in enumerate(memories):
            if i in used:
                continue
            
            group = [memory1]
            used.add(i)
            
            # Don't consolidate very recent memories
            if self._is_recent(memory1):
                groups.append(group)
                continue
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                # Skip recent memories
                if self._is_recent(memory2):
                    continue
                
                similarity = self._calculate_similarity(memory1.content, memory2.content)
                
                if similarity >= self.similarity_threshold:
                    group.append(memory2)
                    used.add(j)
                    
                    if len(group) >= self.max_consolidation_size:
                        break
            
            groups.append(group)
        
        # Add any remaining memories as individual groups
        for i, memory in enumerate(memories):
            if i not in used:
                groups.append([memory])
        
        return groups
    
    def _merge_memory_group(self, memories: List[MemoryFragment]) -> MemoryFragment:
        """
        Merge a group of similar memories.
        
        Args:
            memories: Memories to merge
            
        Returns:
            Merged memory fragment
        """
        if len(memories) == 1:
            return memories[0]
        
        # Combine content
        combined_content = self._combine_content(memories)
        
        # Calculate merged importance
        max_importance = max(m.importance_score for m in memories)
        avg_importance = sum(m.importance_score for m in memories) / len(memories)
        merged_importance = min(1.0, max_importance + self.importance_boost * (len(memories) - 1))
        
        # Sum access counts
        total_access = sum(m.access_count for m in memories)
        
        # Use earliest creation date
        earliest_created = min(m.created_at for m in memories)
        
        # Use latest access date
        latest_accessed = max(m.last_accessed for m in memories)
        
        # Merge metadata
        merged_metadata = {
            'consolidated_from': len(memories),
            'original_importance': [m.importance_score for m in memories],
            'consolidation_date': datetime.now().isoformat()
        }
        
        # Create merged memory
        merged = MemoryFragment(
            user_id=memories[0].user_id,
            fragment_type=memories[0].fragment_type,
            content=combined_content,
            importance_score=merged_importance,
            created_at=earliest_created,
            last_accessed=latest_accessed,
            access_count=total_access,
            metadata=merged_metadata
        )
        
        # Set expiry date to furthest expiry
        expiry_dates = [m.expiry_date for m in memories if m.expiry_date]
        if expiry_dates:
            merged.expiry_date = max(expiry_dates)
        
        return merged
    
    def _combine_content(self, memories: List[MemoryFragment]) -> str:
        """
        Combine content from multiple memories.
        
        Args:
            memories: Memories to combine
            
        Returns:
            Combined content string
        """
        # Simple approach: concatenate unique parts
        contents = [m.content for m in memories]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_contents = []
        
        for content in contents:
            normalized = content.lower().strip()
            if normalized not in seen:
                unique_contents.append(content)
                seen.add(normalized)
        
        # Join with separator
        combined = "; ".join(unique_contents)
        
        # Truncate if too long
        max_length = 500
        if len(combined) > max_length:
            combined = combined[:max_length-3] + "..."
        
        return combined
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _calculate_recency_score(self, memory: MemoryFragment) -> float:
        """
        Calculate recency score for a memory.
        
        Args:
            memory: Memory fragment
            
        Returns:
            Recency score (0-1)
        """
        age_days = (datetime.now() - memory.created_at).days
        
        # Exponential decay
        # Recent memories (< 7 days) have high score
        # Older memories decay exponentially
        if age_days < 7:
            return 1.0
        elif age_days < 30:
            return 0.8
        elif age_days < 90:
            return 0.5
        else:
            return max(0.1, 1.0 / (1 + age_days / 30))
    
    def _is_recent(self, memory: MemoryFragment) -> bool:
        """Check if memory is recent and should be preserved."""
        age_days = (datetime.now() - memory.created_at).days
        return age_days < self.preserve_recent
    
    def _get_content_key(self, content: str) -> str:
        """Get normalized content key for deduplication."""
        # Remove extra whitespace and lowercase
        normalized = ' '.join(content.lower().split())
        # Remove punctuation for comparison
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def analyze_consolidation_impact(self, original: List[MemoryFragment],
                                    consolidated: List[MemoryFragment]) -> Dict[str, Any]:
        """
        Analyze the impact of consolidation.
        
        Args:
            original: Original memories
            consolidated: Consolidated memories
            
        Returns:
            Analysis report
        """
        return {
            'original_count': len(original),
            'consolidated_count': len(consolidated),
            'reduction_ratio': 1 - (len(consolidated) / max(len(original), 1)),
            'original_total_importance': sum(m.importance_score for m in original),
            'consolidated_total_importance': sum(m.importance_score for m in consolidated),
            'types_affected': {
                memory_type: {
                    'original': sum(1 for m in original if m.fragment_type == memory_type),
                    'consolidated': sum(1 for m in consolidated if m.fragment_type == memory_type)
                }
                for memory_type in MemoryType
            }
        }
