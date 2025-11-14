"""
Hybrid retriever combining semantic and keyword search.

This module implements a hybrid retrieval strategy that combines
the strengths of both semantic and keyword-based search.
"""

from typing import List, Dict, Optional, Any, Union
import numpy as np
from ..retrieval.base_retriever import BaseRetriever, RetrievalResult, RetrievalQuery
from ..retrieval.semantic_retriever import SemanticRetriever
from ..retrieval.keyword_retriever import KeywordRetriever
from ..core.models import Message, MemoryFragment


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining semantic and keyword search.
    
    This retriever uses both embedding-based semantic search and
    traditional keyword search, then combines and reranks results.
    """
    
    def _initialize(self) -> None:
        """Initialize hybrid retriever components."""
        # Initialize sub-retrievers
        self.semantic_retriever = SemanticRetriever(self.config.get('semantic_config', {}))
        self.keyword_retriever = KeywordRetriever(self.config.get('keyword_config', {}))
        
        # Combination strategy
        self.combination_method = self.config.get('combination_method', 'weighted_sum')
        self.semantic_weight = self.config.get('semantic_weight', 0.6)
        self.keyword_weight = self.config.get('keyword_weight', 0.4)
        
        # Reranking options
        self.use_reranking = self.config.get('use_reranking', True)
        self.reranking_method = self.config.get('reranking_method', 'reciprocal_rank_fusion')
        
        # Score normalization
        self.normalize_scores = self.config.get('normalize_scores', True)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve memories using hybrid search.
        
        Args:
            query: Retrieval query
            
        Returns:
            List of retrieval results
        """
        # Get results from both retrievers
        semantic_results = []
        keyword_results = []
        
        if query.search_type in ['semantic', 'hybrid']:
            semantic_query = RetrievalQuery(
                text=query.text,
                user_id=query.user_id,
                conversation_id=query.conversation_id,
                limit=query.limit * 2,  # Get more for merging
                threshold=0,  # We'll apply threshold after combining
                filters=query.filters,
                use_embeddings=True
            )
            semantic_results = self.semantic_retriever.retrieve(semantic_query)
        
        if query.search_type in ['keyword', 'hybrid']:
            keyword_query = RetrievalQuery(
                text=query.text,
                user_id=query.user_id,
                conversation_id=query.conversation_id,
                limit=query.limit * 2,  # Get more for merging
                threshold=0,  # We'll apply threshold after combining
                filters=query.filters,
                use_embeddings=False
            )
            keyword_results = self.keyword_retriever.retrieve(keyword_query)
        
        # Combine results
        if self.combination_method == 'weighted_sum':
            combined_results = self._weighted_sum_combination(
                semantic_results, 
                keyword_results
            )
        elif self.combination_method == 'reciprocal_rank_fusion':
            combined_results = self._reciprocal_rank_fusion(
                semantic_results,
                keyword_results
            )
        else:
            combined_results = self._simple_merge(
                semantic_results,
                keyword_results
            )
        
        # Apply threshold
        combined_results = [
            r for r in combined_results 
            if r.score >= query.threshold
        ]
        
        # Rerank if enabled
        if self.use_reranking:
            combined_results = self.rerank(combined_results, query)
        
        # Apply limit
        combined_results = combined_results[:query.limit]
        
        return combined_results
    
    def add_to_index(self, content: Union[Message, MemoryFragment], 
                     embedding: Optional[List[float]] = None) -> None:
        """
        Add content to both indices.
        
        Args:
            content: Content to index
            embedding: Optional pre-computed embedding
        """
        self.semantic_retriever.add_to_index(content, embedding)
        self.keyword_retriever.add_to_index(content)
    
    def update_index(self, content_id: str, 
                    embedding: Optional[List[float]] = None) -> None:
        """
        Update content in both indices.
        
        Args:
            content_id: Content ID
            embedding: New embedding
        """
        self.semantic_retriever.update_index(content_id, embedding)
        self.keyword_retriever.update_index(content_id)
    
    def remove_from_index(self, content_id: str) -> None:
        """
        Remove content from both indices.
        
        Args:
            content_id: Content ID to remove
        """
        self.semantic_retriever.remove_from_index(content_id)
        self.keyword_retriever.remove_from_index(content_id)
    
    def clear_index(self) -> None:
        """Clear both indices."""
        self.semantic_retriever.clear_index()
        self.keyword_retriever.clear_index()
    
    def _weighted_sum_combination(self, semantic_results: List[RetrievalResult],
                                 keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine results using weighted sum of scores.
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            
        Returns:
            Combined results
        """
        # Create score dictionaries
        semantic_scores = {}
        keyword_scores = {}
        content_map = {}
        
        # Normalize scores if needed
        if self.normalize_scores:
            semantic_results = self._normalize_result_scores(semantic_results)
            keyword_results = self._normalize_result_scores(keyword_results)
        
        # Collect semantic scores
        for result in semantic_results:
            key = self._get_result_key(result)
            semantic_scores[key] = result.score
            content_map[key] = result
        
        # Collect keyword scores
        for result in keyword_results:
            key = self._get_result_key(result)
            keyword_scores[key] = result.score
            if key not in content_map:
                content_map[key] = result
        
        # Calculate combined scores
        combined_results = []
        all_keys = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for key in all_keys:
            sem_score = semantic_scores.get(key, 0) * self.semantic_weight
            key_score = keyword_scores.get(key, 0) * self.keyword_weight
            combined_score = sem_score + key_score
            
            result = content_map[key]
            new_result = RetrievalResult(
                content=result.content,
                score=combined_score,
                metadata={
                    'semantic_score': semantic_scores.get(key, 0),
                    'keyword_score': keyword_scores.get(key, 0),
                    'combination_method': 'weighted_sum'
                },
                source='hybrid'
            )
            combined_results.append(new_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def _reciprocal_rank_fusion(self, semantic_results: List[RetrievalResult],
                               keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            
        Returns:
            Combined results using RRF
        """
        k = 60  # RRF constant
        
        # Calculate RRF scores
        rrf_scores = {}
        content_map = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic_results, 1):
            key = self._get_result_key(result)
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (k + rank))
            content_map[key] = result
        
        # Add keyword results
        for rank, result in enumerate(keyword_results, 1):
            key = self._get_result_key(result)
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (k + rank))
            if key not in content_map:
                content_map[key] = result
        
        # Create combined results
        combined_results = []
        for key, rrf_score in rrf_scores.items():
            result = content_map[key]
            new_result = RetrievalResult(
                content=result.content,
                score=rrf_score,
                metadata={
                    'combination_method': 'reciprocal_rank_fusion',
                    'original_score': result.score
                },
                source='hybrid'
            )
            combined_results.append(new_result)
        
        # Sort by RRF score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def _simple_merge(self, semantic_results: List[RetrievalResult],
                     keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Simple merge of results with deduplication.
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            
        Returns:
            Merged results
        """
        # Use parent class combine_results method
        return self.combine_results(semantic_results, keyword_results)
    
    def rerank(self, results: List[RetrievalResult], 
              query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Rerank combined results.
        
        Args:
            results: Combined results
            query: Original query
            
        Returns:
            Reranked results
        """
        if self.reranking_method == 'cross_encoder':
            return self._cross_encoder_rerank(results, query)
        elif self.reranking_method == 'diversity':
            return self._diversity_rerank(results, query)
        else:
            return results
    
    def _cross_encoder_rerank(self, results: List[RetrievalResult],
                             query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Rerank using a cross-encoder model.
        
        Args:
            results: Results to rerank
            query: Query for reranking
            
        Returns:
            Reranked results
        """
        # This would use a cross-encoder model in production
        # For now, we'll use a simple heuristic
        
        query_terms = set(query.text.lower().split())
        
        for result in results:
            # Get content text
            if isinstance(result.content, (Message, MemoryFragment)):
                text = result.content.content
            else:
                text = str(result.content)
            
            # Calculate overlap
            content_terms = set(text.lower().split())
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            
            # Boost score based on overlap
            result.score *= (1 + overlap * 0.5)
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _diversity_rerank(self, results: List[RetrievalResult],
                         query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Rerank for diversity using MMR (Maximal Marginal Relevance).
        
        Args:
            results: Results to rerank
            query: Query for reranking
            
        Returns:
            Diverse reranked results
        """
        if len(results) <= 1:
            return results
        
        lambda_param = 0.5  # Balance between relevance and diversity
        
        # Keep track of selected results
        selected = []
        remaining = results.copy()
        
        # Select first result (highest score)
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse results
        while remaining and len(selected) < len(results):
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate relevance score (already in candidate.score)
                relevance = candidate.score
                
                # Calculate maximum similarity to selected results
                max_similarity = 0
                for selected_result in selected:
                    similarity = self._calculate_content_similarity(
                        candidate.content,
                        selected_result.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # Calculate MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _normalize_result_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: Results with scores
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        # Find min and max scores
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            # Normalize to [0, 1]
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _get_result_key(self, result: RetrievalResult) -> str:
        """Get unique key for a result."""
        if isinstance(result.content, (Message, MemoryFragment)):
            if hasattr(result.content, 'id') and result.content.id:
                return str(result.content.id)
            return str(hash(result.content.content))
        return str(hash(str(result.content)))
    
    def _calculate_content_similarity(self, content1: Union[Message, MemoryFragment],
                                     content2: Union[Message, MemoryFragment]) -> float:
        """
        Calculate similarity between two pieces of content.
        
        Args:
            content1: First content
            content2: Second content
            
        Returns:
            Similarity score
        """
        # Get text from content
        text1 = content1.content if isinstance(content1, (Message, MemoryFragment)) else str(content1)
        text2 = content2.content if isinstance(content2, (Message, MemoryFragment)) else str(content2)
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        semantic_stats = {}
        keyword_stats = {}
        
        # Get stats from semantic retriever
        if hasattr(self.semantic_retriever, 'embeddings_cache'):
            semantic_stats = {
                'cached_embeddings': len(self.semantic_retriever.embeddings_cache),
                'indexed_documents': len(self.semantic_retriever.content_index)
            }
        
        # Get stats from keyword retriever
        if hasattr(self.keyword_retriever, 'get_term_statistics'):
            keyword_stats = self.keyword_retriever.get_term_statistics()
        
        return {
            'combination_method': self.combination_method,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'semantic_stats': semantic_stats,
            'keyword_stats': keyword_stats
        }
