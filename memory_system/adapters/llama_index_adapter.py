"""
LlamaIndex framework memory adapter.

This module provides memory integration for LlamaIndex's document-based
RAG framework with hierarchical memory organization.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import NodeWithScore
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework, MemoryFragment, MemoryType


class LlamaIndexMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for LlamaIndex framework.
    
    This adapter integrates memory management with LlamaIndex's
    document-based indexing and RAG capabilities.
    """
    
    def __init__(self, index: Optional[BaseIndex] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex memory adapter.
        
        Args:
            index: LlamaIndex index instance
            config: Configuration dictionary
        """
        self.index = index
        self.indices: Dict[str, BaseIndex] = {}
        self.documents: Dict[str, Document] = {}
        super().__init__(config)
        
        # RAG-specific tracking
        self.retrieval_history: List[Dict[str, Any]] = []
        self.query_performance: Dict[str, float] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
    
    def _initialize_framework(self) -> None:
        """Initialize LlamaIndex-specific components."""
        self.session_memory.conversation.framework = Framework.LLAMA_INDEX
        
        # Initialize document tracking
        self.document_sources: Dict[str, str] = {}
        self.document_importance: Dict[str, float] = {}
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'average_relevance_score': 0.0
        }
        
        # Initialize hierarchical indices
        self._initialize_hierarchical_indices()
    
    def _initialize_hierarchical_indices(self) -> None:
        """Initialize hierarchical memory indices."""
        # Create indices for different memory levels
        # These would be properly initialized with vector stores in production
        
        # Short-term memory index (session documents)
        self.indices['short_term'] = None
        
        # Long-term memory index (persistent documents)
        self.indices['long_term'] = None
        
        # Episodic memory index (conversation episodes)
        self.indices['episodic'] = None
    
    def inject_memory_context(self, input_text: str,
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context using document retrieval.
        
        Args:
            input_text: Original input
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input with retrieved context
        """
        context_parts = []
        
        # Retrieve relevant documents from index
        if self.index:
            retrieved_nodes = self._retrieve_relevant_nodes(input_text, top_k=3)
            if retrieved_nodes:
                context_parts.append("Relevant information from documents:")
                for node in retrieved_nodes:
                    context_parts.append(f"- {node.text[:200]}...")
        
        # Add recent retrieval results
        if self.retrieval_history:
            recent_retrievals = self.retrieval_history[-2:]
            context_parts.append("\nRecent retrieval context:")
            for retrieval in recent_retrievals:
                context_parts.append(f"- Query: {retrieval['query'][:50]}")
                context_parts.append(f"  Found: {retrieval['result'][:100]}...")
        
        # Add long-term memories
        if self.persistent_memory:
            memories = self.retrieve_relevant_memories(input_text, limit=3)
            if memories:
                context_parts.append("\nLong-term memory context:")
                for memory in memories:
                    context_parts.append(f"- {memory.content}")
        
        if context_parts:
            return f"{chr(10).join(context_parts)}\n\nCurrent query: {input_text}"
        
        return input_text
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process LlamaIndex response and update document metadata.
        
        Args:
            response: Query/chat response
            metadata: Optional metadata including retrieval information
            
        Returns:
            Processed response
        """
        # Track retrieval performance
        if metadata and 'retrieval_nodes' in metadata:
            nodes = metadata['retrieval_nodes']
            if nodes:
                # Calculate average relevance score
                avg_score = sum(node.score for node in nodes) / len(nodes)
                self.retrieval_stats['average_relevance_score'] = (
                    (self.retrieval_stats['average_relevance_score'] * 
                     self.retrieval_stats['successful_retrievals'] + avg_score) /
                    (self.retrieval_stats['successful_retrievals'] + 1)
                )
                self.retrieval_stats['successful_retrievals'] += 1
                
                # Store retrieval history
                self.retrieval_history.append({
                    'query': metadata.get('query', ''),
                    'result': response[:500],
                    'num_nodes': len(nodes),
                    'avg_score': avg_score,
                    'timestamp': datetime.now()
                })
        
        # Update query statistics
        if metadata and 'query' in metadata:
            self.retrieval_stats['total_queries'] += 1
            query_key = metadata['query'][:50]
            self.query_performance[query_key] = metadata.get('performance', 1.0)
        
        # Keep history limited
        if len(self.retrieval_history) > 50:
            self.retrieval_history = self.retrieval_history[-50:]
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get LlamaIndex-specific context data.
        
        Returns:
            Dictionary with LlamaIndex-specific context
        """
        return {
            'total_documents': len(self.documents),
            'active_indices': list(self.indices.keys()),
            'retrieval_stats': self.retrieval_stats,
            'recent_queries': [h['query'][:50] for h in self.retrieval_history[-5:]],
            'document_sources': list(self.document_sources.keys())[:10],
            'index_initialized': self.index is not None
        }
    
    def create_memory_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a document from memory content.
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            LlamaIndex Document
        """
        doc_id = f"memory_{datetime.now().timestamp()}"
        
        doc_metadata = {
            'source': 'memory',
            'created_at': datetime.now().isoformat(),
            'type': 'conversation_memory'
        }
        if metadata:
            doc_metadata.update(metadata)
        
        document = Document(
            text=content,
            doc_id=doc_id,
            metadata=doc_metadata
        )
        
        # Store document reference
        self.documents[doc_id] = document
        self.document_metadata[doc_id] = doc_metadata
        
        return document
    
    def add_conversation_to_index(self, conversation_text: str, 
                                 summary: Optional[str] = None) -> None:
        """
        Add conversation content to the index as a document.
        
        Args:
            conversation_text: Full conversation text
            summary: Optional conversation summary
        """
        # Create document from conversation
        doc_content = summary if summary else conversation_text
        metadata = {
            'type': 'conversation',
            'message_count': len(self.session_memory.messages),
            'topics': self.session_memory.extract_topics()
        }
        
        document = self.create_memory_document(doc_content, metadata)
        
        # Add to index if available
        if self.index:
            self.index.insert(document)
        
        # Store as episodic memory
        self._store_episodic_memory(document)
    
    def create_hierarchical_query_engine(self):
        """
        Create a query engine that searches across memory hierarchies.
        
        Returns:
            Query engine with hierarchical search
        """
        if not self.index:
            raise ValueError("No index configured")
        
        # Create query engine with custom retriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        
        # This would be properly implemented with actual retrievers
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        # Wrap with memory enhancement
        original_query = query_engine.query
        
        def enhanced_query(query_str: str):
            # Enhance query with memory context
            enhanced_str = self.inject_memory_context(query_str)
            
            # Execute query
            response = original_query(enhanced_str)
            
            # Track query
            self.retrieval_stats['total_queries'] += 1
            
            # Store in history
            self.retrieval_history.append({
                'query': query_str,
                'result': str(response),
                'timestamp': datetime.now()
            })
            
            return response
        
        query_engine.query = enhanced_query
        return query_engine
    
    def process_rag_interaction(self, query: str, chat_mode: bool = False) -> str:
        """
        Process a complete RAG interaction with memory.
        
        Args:
            query: User query
            chat_mode: Whether to use chat mode
            
        Returns:
            RAG response
        """
        # Add query to memory
        self.add_user_message(query)
        
        if not self.index:
            response = "No index configured for retrieval"
        else:
            # Retrieve relevant nodes
            retrieved_nodes = self._retrieve_relevant_nodes(query)
            
            # Generate response
            if chat_mode and hasattr(self.index, 'as_chat_engine'):
                engine = self.index.as_chat_engine()
                result = engine.chat(query)
                response = result.response
            else:
                engine = self.index.as_query_engine()
                result = engine.query(query)
                response = str(result)
            
            # Process with metadata
            metadata = {
                'query': query,
                'retrieval_nodes': retrieved_nodes,
                'mode': 'chat' if chat_mode else 'query'
            }
            response = self.process_response(response, metadata)
        
        # Add response to memory
        self.add_assistant_message(response)
        
        return response
    
    def update_document_importance(self, doc_id: str, importance_delta: float) -> None:
        """
        Update the importance score of a document.
        
        Args:
            doc_id: Document ID
            importance_delta: Change in importance score
        """
        if doc_id not in self.document_importance:
            self.document_importance[doc_id] = 0.5
        
        self.document_importance[doc_id] = min(
            1.0,
            max(0.0, self.document_importance[doc_id] + importance_delta)
        )
        
        # Update document metadata
        if doc_id in self.document_metadata:
            self.document_metadata[doc_id]['importance'] = self.document_importance[doc_id]
    
    def consolidate_documents(self, max_documents: int = 100) -> None:
        """
        Consolidate documents to manage index size.
        
        Args:
            max_documents: Maximum documents to keep
        """
        if len(self.documents) <= max_documents:
            return
        
        # Sort documents by importance and recency
        doc_scores = []
        for doc_id, doc in self.documents.items():
            importance = self.document_importance.get(doc_id, 0.5)
            metadata = self.document_metadata.get(doc_id, {})
            created_at = metadata.get('created_at', '')
            
            # Calculate composite score
            recency_score = 1.0  # Would calculate based on timestamp
            score = importance * 0.7 + recency_score * 0.3
            doc_scores.append((doc_id, score))
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top documents
        keep_ids = set(doc_id for doc_id, _ in doc_scores[:max_documents])
        
        # Remove low-scoring documents
        for doc_id in list(self.documents.keys()):
            if doc_id not in keep_ids:
                del self.documents[doc_id]
                if doc_id in self.document_metadata:
                    del self.document_metadata[doc_id]
                if doc_id in self.document_importance:
                    del self.document_importance[doc_id]
    
    def create_memory_index(self, memories: List[MemoryFragment]) -> VectorStoreIndex:
        """
        Create an index from memory fragments.
        
        Args:
            memories: List of memory fragments
            
        Returns:
            VectorStoreIndex of memories
        """
        # Convert memories to documents
        documents = []
        for memory in memories:
            doc = Document(
                text=memory.content,
                metadata={
                    'type': memory.fragment_type.value,
                    'importance': memory.importance_score,
                    'created_at': memory.created_at.isoformat(),
                    'user_id': memory.user_id
                }
            )
            documents.append(doc)
        
        # Create index
        index = VectorStoreIndex.from_documents(documents)
        
        return index
    
    def _retrieve_relevant_nodes(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes from the index.
        
        Args:
            query: Search query
            top_k: Number of nodes to retrieve
            
        Returns:
            List of relevant nodes with scores
        """
        if not self.index:
            return []
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        return nodes
    
    def _store_episodic_memory(self, document: Document) -> None:
        """
        Store document as episodic memory.
        
        Args:
            document: Document to store
        """
        if self.persistent_memory:
            # Create memory fragment from document
            memory = MemoryFragment(
                user_id=self.user_id,
                fragment_type=MemoryType.EPISODE,
                content=document.text[:500],  # Truncate if too long
                importance_score=0.6,
                metadata={
                    'doc_id': document.doc_id,
                    'source': 'llama_index'
                }
            )
            self.persistent_memory.store_memory(memory)
    
    def export_index_state(self) -> Dict[str, Any]:
        """
        Export index and document state.
        
        Returns:
            Dictionary with index state
        """
        return {
            'documents': {
                doc_id: {
                    'text': doc.text[:500],  # Truncate for export
                    'metadata': doc.metadata
                }
                for doc_id, doc in list(self.documents.items())[:20]
            },
            'document_importance': self.document_importance,
            'retrieval_history': self.retrieval_history[-20:],
            'retrieval_stats': self.retrieval_stats,
            'query_performance': dict(list(self.query_performance.items())[-20:])
        }
    
    def import_index_state(self, state: Dict[str, Any]) -> None:
        """
        Import index and document state.
        
        Args:
            state: Index state to import
        """
        # Import documents
        for doc_id, doc_data in state.get('documents', {}).items():
            document = Document(
                text=doc_data['text'],
                doc_id=doc_id,
                metadata=doc_data.get('metadata', {})
            )
            self.documents[doc_id] = document
            
            # Add to index if available
            if self.index:
                self.index.insert(document)
        
        self.document_importance = state.get('document_importance', {})
        self.retrieval_history = state.get('retrieval_history', [])
        self.retrieval_stats = state.get('retrieval_stats', self.retrieval_stats)
        self.query_performance = state.get('query_performance', {})
    
    def get_retrieval_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about retrieval performance.
        
        Returns:
            Dictionary with retrieval analytics
        """
        analytics = {
            'total_queries': self.retrieval_stats['total_queries'],
            'success_rate': (
                self.retrieval_stats['successful_retrievals'] / 
                max(1, self.retrieval_stats['total_queries'])
            ),
            'average_relevance': self.retrieval_stats['average_relevance_score'],
            'top_performing_queries': [],
            'low_performing_queries': []
        }
        
        # Sort queries by performance
        if self.query_performance:
            sorted_queries = sorted(
                self.query_performance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            analytics['top_performing_queries'] = sorted_queries[:5]
            analytics['low_performing_queries'] = sorted_queries[-5:]
        
        return analytics
