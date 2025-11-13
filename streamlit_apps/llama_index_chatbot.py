"""
LlamaIndex Chatbot with Azure OpenAI

Streamlit chatbot application using LlamaIndex with Azure OpenAI.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.llama_index.azure_llama_index_setup import (
    AzureLlamaIndexSetup,
    create_sample_documents,
    create_rag_pipeline
)
from llama_index.core import Document
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class LlamaIndexChatbot(BaseChatbot):
    """LlamaIndex-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="LlamaIndex Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates LlamaIndex integration with Azure OpenAI.
            Features document indexing, retrieval-augmented generation (RAG), and context-aware conversations.
            """
        )
        self.setup = None
        self.rag_pipeline = None
        self.initialize_index()
    
    def initialize_index(self):
        """Initialize LlamaIndex components"""
        if "llama_index_setup" not in st.session_state:
            try:
                st.session_state.llama_index_setup = AzureLlamaIndexSetup()
                st.session_state.rag_pipeline = create_rag_pipeline(st.session_state.llama_index_setup)
                st.session_state.custom_documents = []
                logger.info("Initialized LlamaIndex with sample documents")
            except Exception as e:
                logger.error(f"Failed to initialize LlamaIndex: {str(e)}")
                st.session_state.llama_index_setup = None
                st.session_state.rag_pipeline = None
        
        self.setup = st.session_state.llama_index_setup
        self.rag_pipeline = st.session_state.rag_pipeline
    
    def render_sidebar(self):
        """Render sidebar with LlamaIndex-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("LlamaIndex Settings")
            
            # Query mode selection
            query_mode = st.selectbox(
                "Query Mode",
                ["Chat", "Query", "Streaming"],
                help="Select how to interact with the index"
            )
            st.session_state.query_mode = query_mode
            
            # Retrieval settings
            st.session_state.similarity_top_k = st.slider(
                "Top K Similar Documents",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of similar documents to retrieve"
            )
            
            # Chat mode settings
            if query_mode == "Chat":
                chat_mode = st.selectbox(
                    "Chat Mode",
                    ["condense_question", "simple", "context"],
                    help="Type of chat engine to use"
                )
                st.session_state.chat_mode = chat_mode
            
            st.divider()
            
            # Document management
            st.subheader("Document Management")
            
            # Current documents info
            if self.rag_pipeline:
                num_docs = len(self.rag_pipeline.get("documents", []))
                st.caption(f"Current documents: {num_docs}")
            
            # Add custom document
            with st.expander("Add Custom Document"):
                doc_text = st.text_area(
                    "Document Text",
                    height=100,
                    placeholder="Enter document text..."
                )
                doc_title = st.text_input("Document Title (optional)")
                
                if st.button("Add Document"):
                    if doc_text:
                        self.add_custom_document(doc_text, doc_title)
            
            # Reset to sample documents
            if st.button("Reset to Sample Documents"):
                self.reset_documents()
    
    def add_custom_document(self, text: str, title: str = ""):
        """Add a custom document to the index"""
        try:
            # Create new document
            metadata = {"title": title} if title else {}
            new_doc = Document(text=text, metadata=metadata)
            
            # Add to custom documents list
            st.session_state.custom_documents.append(new_doc)
            
            # Recreate index with all documents
            all_docs = create_sample_documents() + st.session_state.custom_documents
            new_index = self.setup.create_index_from_documents(all_docs)
            
            # Update pipeline
            st.session_state.rag_pipeline["index"] = new_index
            st.session_state.rag_pipeline["query_engine"] = self.setup.create_query_engine(new_index)
            st.session_state.rag_pipeline["chat_engine"] = self.setup.create_chat_engine(new_index)
            st.session_state.rag_pipeline["documents"] = all_docs
            
            self.rag_pipeline = st.session_state.rag_pipeline
            
            st.success(f"Added document: {title if title else 'Untitled'}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to add document: {str(e)}")
    
    def reset_documents(self):
        """Reset to original sample documents"""
        try:
            st.session_state.custom_documents = []
            st.session_state.rag_pipeline = create_rag_pipeline(self.setup)
            self.rag_pipeline = st.session_state.rag_pipeline
            st.success("Reset to sample documents")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to reset documents: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from LlamaIndex
        
        Args:
            prompt: User prompt
        
        Returns:
            Index response
        """
        try:
            if self.rag_pipeline is None:
                return "Index not initialized. Please check your configuration."
            
            query_mode = st.session_state.get("query_mode", "Chat")
            
            if query_mode == "Chat":
                # Use chat engine
                chat_engine = self.rag_pipeline["chat_engine"]
                response = self.setup.chat(prompt, chat_engine)
            
            elif query_mode == "Query":
                # Use query engine
                query_engine = self.setup.create_query_engine(
                    self.rag_pipeline["index"],
                    similarity_top_k=st.session_state.get("similarity_top_k", 3)
                )
                response = self.setup.query(prompt, query_engine)
            
            else:  # Streaming
                # Use streaming query engine
                query_engine = self.setup.create_query_engine(
                    self.rag_pipeline["index"],
                    similarity_top_k=st.session_state.get("similarity_top_k", 3),
                    streaming=True
                )
                response = query_engine.query(prompt)
                
                # Stream the response
                response_text = ""
                response_placeholder = st.empty()
                for text in response.response_gen:
                    response_text += text
                    response_placeholder.markdown(response_text)
                
                return response_text
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries
        with st.expander("Example Queries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Knowledge Base Queries:**")
                st.code("What is artificial intelligence?")
                st.code("Explain machine learning")
                st.code("How does deep learning work?")
            
            with col2:
                st.markdown("**Comparison Queries:**")
                st.code("Compare NLP and computer vision")
                st.code("What's the difference between ML and DL?")
                st.code("How do these AI technologies relate?")
        
        # Show indexed documents
        with st.expander("Indexed Documents"):
            if self.rag_pipeline and "documents" in self.rag_pipeline:
                for i, doc in enumerate(self.rag_pipeline["documents"], 1):
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    title = metadata.get('topic', metadata.get('title', f'Document {i}'))
                    st.markdown(f"**{i}. {title}**")
                    
                    # Show preview of document
                    preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                    st.caption(preview)
        
        # RAG pipeline visualization
        with st.expander("RAG Pipeline"):
            st.mermaid("""
            graph LR
                A[User Query] --> B[Embedding]
                B --> C[Vector Search]
                C --> D[Retrieve Top K Documents]
                D --> E[Context + Query]
                E --> F[LLM Generation]
                F --> G[Response]
                
                H[(Document Index)] --> C
                
                style H fill:#f9f,stroke:#333,stroke-width:2px
                style F fill:#bbf,stroke:#333,stroke-width:2px
            """)


def main():
    """Main function to run the LlamaIndex chatbot"""
    chatbot = LlamaIndexChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
