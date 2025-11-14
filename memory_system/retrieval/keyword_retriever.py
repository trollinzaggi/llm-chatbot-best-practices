"""
Keyword-based retriever for text search.

This module implements traditional keyword and text-based search strategies.
"""

import re
from typing import List, Dict, Optional, Any, Union, Set
from collections import Counter, defaultdict
import math
from ..retrieval.base_retriever import BaseRetriever, RetrievalResult, RetrievalQuery
from ..core.models import Message, MemoryFragment


class KeywordRetriever(BaseRetriever):
    """
    Keyword-based retriever using TF-IDF and BM25 scoring.
    
    This retriever uses traditional information retrieval techniques
    for keyword-based memory search.
    """
    
    def _initialize(self) -> None:
        """Initialize keyword retriever components."""
        # Document index
        self.documents: Dict[str, str] = {}
        self.content_index: Dict[str, Union[Message, MemoryFragment]] = {}
        
        # TF-IDF components
        self.term_frequencies: Dict[str, Dict[str, float]] = {}
        self.inverse_document_frequencies: Dict[str, float] = {}
        self.document_lengths: Dict[str, int] = {}
        
        # BM25 parameters
        self.k1 = self.config.get('bm25_k1', 1.2)
        self.b = self.config.get('bm25_b', 0.75)
        
        # Preprocessing options
        self.use_stemming = self.config.get('use_stemming', False)
        self.remove_stopwords = self.config.get('remove_stopwords', True)
        self.ngram_range = self.config.get('ngram_range', (1, 2))
        
        # Stopwords
        self.stopwords = self._load_stopwords()
        
        # Statistics
        self.avg_document_length = 0
        self.total_documents = 0
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve memories using keyword search.
        
        Args:
            query: Retrieval query
            
        Returns:
            List of retrieval results
        """
        # Preprocess query
        query_terms = self._preprocess_text(query.text)
        
        if not query_terms:
            return []
        
        # Score documents
        scores = {}
        
        if self.config.get('scoring_method', 'bm25') == 'bm25':
            scores = self._bm25_score(query_terms)
        else:
            scores = self._tfidf_score(query_terms)
        
        # Create results
        results = []
        for doc_id, score in scores.items():
            if score >= query.threshold:
                content = self.content_index.get(doc_id)
                if content:
                    result = RetrievalResult(
                        content=content,
                        score=score,
                        metadata={'scoring_method': self.config.get('scoring_method', 'bm25')},
                        source='keyword'
                    )
                    results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply limit
        results = results[:query.limit]
        
        # Apply filters
        if query.filters:
            results = self.filter_results(results, query.filters)
        
        return results
    
    def add_to_index(self, content: Union[Message, MemoryFragment], 
                     embedding: Optional[List[float]] = None) -> None:
        """
        Add content to keyword index.
        
        Args:
            content: Content to index
            embedding: Not used for keyword retrieval
        """
        # Get content ID and text
        doc_id = self._get_content_id(content)
        text = self._get_content_text(content)
        
        # Store document
        self.documents[doc_id] = text
        self.content_index[doc_id] = content
        
        # Preprocess text
        terms = self._preprocess_text(text)
        
        # Calculate term frequencies
        term_freq = Counter(terms)
        doc_length = len(terms)
        
        # Normalize term frequencies
        for term in term_freq:
            term_freq[term] = term_freq[term] / doc_length
        
        self.term_frequencies[doc_id] = dict(term_freq)
        self.document_lengths[doc_id] = doc_length
        
        # Update statistics
        self.total_documents += 1
        self._update_idf()
        self._update_avg_doc_length()
    
    def update_index(self, content_id: str, 
                    embedding: Optional[List[float]] = None) -> None:
        """
        Update content in keyword index.
        
        Args:
            content_id: Content ID
            embedding: Not used
        """
        if content_id in self.documents:
            # Re-index the document
            content = self.content_index.get(content_id)
            if content:
                self.remove_from_index(content_id)
                self.add_to_index(content)
    
    def remove_from_index(self, content_id: str) -> None:
        """
        Remove content from keyword index.
        
        Args:
            content_id: Content ID to remove
        """
        if content_id in self.documents:
            del self.documents[content_id]
            del self.term_frequencies[content_id]
            del self.document_lengths[content_id]
            if content_id in self.content_index:
                del self.content_index[content_id]
            
            self.total_documents -= 1
            self._update_idf()
            self._update_avg_doc_length()
    
    def clear_index(self) -> None:
        """Clear the keyword index."""
        self.documents.clear()
        self.content_index.clear()
        self.term_frequencies.clear()
        self.inverse_document_frequencies.clear()
        self.document_lengths.clear()
        self.total_documents = 0
        self.avg_document_length = 0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for indexing/searching.
        
        Args:
            text: Raw text
            
        Returns:
            List of processed terms
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Apply stemming if enabled
        if self.use_stemming:
            tokens = [self._stem_word(t) for t in tokens]
        
        # Generate n-grams if configured
        if self.ngram_range[1] > 1:
            tokens = self._generate_ngrams(tokens)
        
        return tokens
    
    def _stem_word(self, word: str) -> str:
        """
        Simple stemming implementation.
        
        Args:
            word: Word to stem
            
        Returns:
            Stemmed word
        """
        # Simple suffix removal (Porter stemmer would be better)
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def _generate_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List including n-grams
        """
        ngrams = tokens.copy()
        
        for n in range(2, min(self.ngram_range[1] + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords list."""
        # Common English stopwords
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by',
            'for', 'from', 'has', 'had', 'have', 'he', 'in', 'is', 'it',
            'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'the', 'this', 'these', 'those', 'there', 'their', 'them',
            'they', 'we', 'our', 'us', 'you', 'your', 'i', 'me', 'my'
        }
    
    def _update_idf(self) -> None:
        """Update inverse document frequencies."""
        if self.total_documents == 0:
            return
        
        # Count document frequencies
        document_frequencies = defaultdict(int)
        
        for doc_id, terms in self.term_frequencies.items():
            for term in terms:
                document_frequencies[term] += 1
        
        # Calculate IDF
        self.inverse_document_frequencies.clear()
        for term, df in document_frequencies.items():
            # IDF = log((N + 1) / (df + 1)) + 1
            idf = math.log((self.total_documents + 1) / (df + 1)) + 1
            self.inverse_document_frequencies[term] = idf
    
    def _update_avg_doc_length(self) -> None:
        """Update average document length."""
        if self.total_documents == 0:
            self.avg_document_length = 0
        else:
            total_length = sum(self.document_lengths.values())
            self.avg_document_length = total_length / self.total_documents
    
    def _tfidf_score(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for documents.
        
        Args:
            query_terms: Preprocessed query terms
            
        Returns:
            Dictionary mapping document ID to score
        """
        scores = {}
        
        # Calculate query term frequencies
        query_tf = Counter(query_terms)
        query_length = len(query_terms)
        
        for term in query_tf:
            query_tf[term] = query_tf[term] / query_length
        
        # Score each document
        for doc_id, doc_terms in self.term_frequencies.items():
            score = 0.0
            
            for term in query_tf:
                if term in doc_terms:
                    tf = doc_terms[term]
                    idf = self.inverse_document_frequencies.get(term, 0)
                    score += tf * idf * query_tf[term]
            
            if score > 0:
                scores[doc_id] = score
        
        return scores
    
    def _bm25_score(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate BM25 scores for documents.
        
        Args:
            query_terms: Preprocessed query terms
            
        Returns:
            Dictionary mapping document ID to score
        """
        scores = {}
        
        # Count query term frequencies
        query_tf = Counter(query_terms)
        
        # Score each document
        for doc_id, doc_terms in self.term_frequencies.items():
            score = 0.0
            doc_length = self.document_lengths[doc_id]
            
            for term in query_tf:
                if term in doc_terms:
                    # Get term frequency in document
                    tf = doc_terms[term] * doc_length  # Convert back to raw count
                    
                    # Get IDF
                    idf = self.inverse_document_frequencies.get(term, 0)
                    
                    # Calculate BM25 component
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_length / self.avg_document_length)
                    )
                    
                    score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_id] = score
        
        return scores
    
    def _get_content_id(self, content: Union[Message, MemoryFragment]) -> str:
        """Get unique ID for content."""
        if hasattr(content, 'id') and content.id:
            return str(content.id)
        else:
            return str(hash(self._get_content_text(content)))
    
    def _get_content_text(self, content: Union[Message, MemoryFragment]) -> str:
        """Extract text from content."""
        if isinstance(content, Message):
            return content.content
        elif isinstance(content, MemoryFragment):
            return content.content
        else:
            return str(content)
    
    def get_term_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed terms.
        
        Returns:
            Dictionary with term statistics
        """
        all_terms = set()
        for terms in self.term_frequencies.values():
            all_terms.update(terms.keys())
        
        return {
            'total_documents': self.total_documents,
            'unique_terms': len(all_terms),
            'avg_document_length': self.avg_document_length,
            'most_common_terms': self._get_most_common_terms(10),
            'rarest_terms': self._get_rarest_terms(10)
        }
    
    def _get_most_common_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get most common terms across all documents."""
        term_counts = Counter()
        
        for doc_terms in self.term_frequencies.values():
            for term in doc_terms:
                term_counts[term] += 1
        
        return term_counts.most_common(n)
    
    def _get_rarest_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get rarest terms across all documents."""
        term_counts = Counter()
        
        for doc_terms in self.term_frequencies.values():
            for term in doc_terms:
                term_counts[term] += 1
        
        # Get terms with lowest counts
        return term_counts.most_common()[-n:] if len(term_counts) >= n else list(term_counts.items())
