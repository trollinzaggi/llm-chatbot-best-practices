"""
LlamaIndex Setup for Azure OpenAI

This module demonstrates how to set up and use LlamaIndex with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Optional, Any
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    ServiceContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine, CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from config import config
from utils import setup_logger, log_api_call

# Set up logging
logger = setup_logger(__name__)


class AzureLlamaIndexSetup:
    """Azure OpenAI integrated LlamaIndex setup"""
    
    def __init__(self):
        """Initialize LlamaIndex with Azure OpenAI"""
        self.llm = self._create_llm()
        self.embed_model = self._create_embedding_model()
        self._configure_settings()
        self.index = None
        logger.info("Initialized LlamaIndex with Azure OpenAI")
    
    def _create_llm(self) -> AzureOpenAI:
        """Create Azure OpenAI LLM for LlamaIndex"""
        llm = AzureOpenAI(
            deployment_name=config.deployment_name,
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        logger.info("Created Azure OpenAI LLM for LlamaIndex")
        return llm
    
    def _create_embedding_model(self) -> Optional[AzureOpenAIEmbedding]:
        """Create Azure OpenAI Embedding model"""
        if config.embedding_deployment:
            embed_model = AzureOpenAIEmbedding(
                deployment_name=config.embedding_deployment,
                api_key=config.api_key,
                azure_endpoint=config.endpoint,
                api_version=config.api_version
            )
            logger.info("Created Azure OpenAI Embedding model")
            return embed_model
        return None
    
    def _configure_settings(self):
        """Configure global LlamaIndex settings"""
        Settings.llm = self.llm
        if self.embed_model:
            Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        logger.info("Configured LlamaIndex settings")
    
    def create_index_from_documents(
        self,
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 20
    ) -> VectorStoreIndex:
        """
        Create an index from documents
        
        Args:
            documents: List of Document objects
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        
        Returns:
            VectorStoreIndex instance
        """
        # Create node parser
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Create index
        self.index = VectorStoreIndex(nodes)
        
        logger.info(f"Created index from {len(documents)} documents")
        return self.index
    
    def create_index_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> VectorStoreIndex:
        """
        Create an index from text strings
        
        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
        
        Returns:
            VectorStoreIndex instance
        """
        # Create documents
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)
        
        return self.create_index_from_documents(documents)
    
    def create_query_engine(
        self,
        index: Optional[VectorStoreIndex] = None,
        similarity_top_k: int = 3,
        streaming: bool = False
    ) -> RetrieverQueryEngine:
        """
        Create a query engine from an index
        
        Args:
            index: VectorStoreIndex (uses self.index if None)
            similarity_top_k: Number of similar documents to retrieve
            streaming: Whether to stream responses
        
        Returns:
            RetrieverQueryEngine instance
        """
        if index is None:
            index = self.index
        
        if index is None:
            raise ValueError("No index available. Create an index first.")
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            streaming=streaming
        )
        
        logger.info("Created query engine")
        return query_engine
    
    def create_chat_engine(
        self,
        index: Optional[VectorStoreIndex] = None,
        chat_mode: str = "condense_question",
        memory_limit: int = 3000
    ) -> Any:
        """
        Create a chat engine from an index
        
        Args:
            index: VectorStoreIndex (uses self.index if None)
            chat_mode: Chat mode ("simple", "condense_question", "context")
            memory_limit: Token limit for chat memory
        
        Returns:
            Chat engine instance
        """
        if index is None:
            index = self.index
        
        if index is None:
            raise ValueError("No index available. Create an index first.")
        
        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=memory_limit
        )
        
        # Create chat engine based on mode
        if chat_mode == "condense_question":
            chat_engine = index.as_chat_engine(
                chat_mode="condense_question",
                memory=memory,
                llm=self.llm,
                verbose=True
            )
        elif chat_mode == "simple":
            chat_engine = index.as_chat_engine(
                chat_mode="simple",
                memory=memory,
                llm=self.llm,
                verbose=True
            )
        else:  # context mode
            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                llm=self.llm,
                verbose=True
            )
        
        logger.info(f"Created {chat_mode} chat engine")
        return chat_engine
    
    def query(
        self,
        query_text: str,
        query_engine: Optional[RetrieverQueryEngine] = None
    ) -> str:
        """
        Query the index
        
        Args:
            query_text: Query string
            query_engine: Query engine to use
        
        Returns:
            Response string
        """
        if query_engine is None:
            query_engine = self.create_query_engine()
        
        log_api_call(logger, "LlamaIndex", "query", query=query_text)
        response = query_engine.query(query_text)
        
        return str(response)
    
    def chat(
        self,
        message: str,
        chat_engine: Any = None
    ) -> str:
        """
        Chat with the index
        
        Args:
            message: Chat message
            chat_engine: Chat engine to use
        
        Returns:
            Response string
        """
        if chat_engine is None:
            chat_engine = self.create_chat_engine()
        
        log_api_call(logger, "LlamaIndex", "chat", message=message)
        response = chat_engine.chat(message)
        
        return str(response)


# Example data creation functions
def create_sample_documents() -> List[Document]:
    """Create sample documents for testing"""
    documents = [
        Document(
            text="""Artificial Intelligence (AI) is the simulation of human intelligence in machines 
            that are programmed to think and learn. AI systems can perform tasks that typically 
            require human intelligence, such as visual perception, speech recognition, 
            decision-making, and language translation.""",
            metadata={"category": "AI", "topic": "Introduction"}
        ),
        Document(
            text="""Machine Learning is a subset of AI that provides systems the ability to 
            automatically learn and improve from experience without being explicitly programmed. 
            ML focuses on developing computer programs that can access data and use it to 
            learn for themselves.""",
            metadata={"category": "AI", "topic": "Machine Learning"}
        ),
        Document(
            text="""Deep Learning is a subset of machine learning that uses neural networks with 
            multiple layers. These networks attempt to simulate the behavior of the human brain, 
            allowing it to learn from large amounts of data. Deep learning drives many AI 
            applications that improve automation and analytical tasks.""",
            metadata={"category": "AI", "topic": "Deep Learning"}
        ),
        Document(
            text="""Natural Language Processing (NLP) is a branch of AI that helps computers 
            understand, interpret and manipulate human language. NLP draws from many disciplines, 
            including computer science and computational linguistics, to bridge the gap between 
            human communication and computer understanding.""",
            metadata={"category": "AI", "topic": "NLP"}
        ),
        Document(
            text="""Computer Vision is a field of AI that trains computers to interpret and 
            understand the visual world. Using digital images from cameras and videos and deep 
            learning models, machines can accurately identify and classify objects.""",
            metadata={"category": "AI", "topic": "Computer Vision"}
        )
    ]
    return documents


def create_rag_pipeline(setup: AzureLlamaIndexSetup) -> Dict[str, Any]:
    """Create a complete RAG pipeline"""
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create index
    index = setup.create_index_from_documents(documents)
    
    # Create query engine
    query_engine = setup.create_query_engine(index, similarity_top_k=2)
    
    # Create chat engine
    chat_engine = setup.create_chat_engine(index, chat_mode="condense_question")
    
    return {
        "index": index,
        "query_engine": query_engine,
        "chat_engine": chat_engine,
        "documents": documents
    }


def demonstrate_query_modes(setup: AzureLlamaIndexSetup, index: VectorStoreIndex):
    """Demonstrate different query modes"""
    
    # Standard query
    print("Standard Query:")
    query_engine = setup.create_query_engine(index, similarity_top_k=2)
    response = setup.query("What is machine learning?", query_engine)
    print(f"Response: {response}\n")
    
    # Query with more context
    print("Query with More Context:")
    query_engine = setup.create_query_engine(index, similarity_top_k=4)
    response = setup.query("How does deep learning relate to machine learning?", query_engine)
    print(f"Response: {response}\n")
    
    # Streaming query
    print("Streaming Query:")
    query_engine = setup.create_query_engine(index, streaming=True)
    response = query_engine.query("Explain the different types of AI")
    print("Response: ", end="")
    for text in response.response_gen:
        print(text, end="", flush=True)
    print("\n")


def demonstrate_chat_modes(setup: AzureLlamaIndexSetup, index: VectorStoreIndex):
    """Demonstrate different chat modes"""
    
    # Condense question mode
    print("Condense Question Mode:")
    chat_engine = setup.create_chat_engine(index, chat_mode="condense_question")
    response = setup.chat("Tell me about AI", chat_engine)
    print(f"Response: {response}")
    response = setup.chat("What are its applications?", chat_engine)  # Follow-up
    print(f"Follow-up: {response}\n")
    
    # Simple mode
    print("Simple Mode:")
    chat_engine = setup.create_chat_engine(index, chat_mode="simple")
    response = setup.chat("What is NLP?", chat_engine)
    print(f"Response: {response}\n")
    
    # Context mode
    print("Context Mode:")
    chat_engine = setup.create_chat_engine(index, chat_mode="context")
    response = setup.chat("Compare computer vision and NLP", chat_engine)
    print(f"Response: {response}\n")


if __name__ == "__main__":
    # Example usage
    setup = AzureLlamaIndexSetup()
    
    print("Creating RAG Pipeline...")
    rag_pipeline = create_rag_pipeline(setup)
    
    print("\n=== Query Examples ===")
    demonstrate_query_modes(setup, rag_pipeline["index"])
    
    print("\n=== Chat Examples ===")
    demonstrate_chat_modes(setup, rag_pipeline["index"])
    
    print("\n=== Interactive Chat ===")
    chat_engine = rag_pipeline["chat_engine"]
    
    # Simulate conversation
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What's the difference between ML and deep learning?",
        "Can you give me practical examples?"
    ]
    
    for question in questions:
        print(f"User: {question}")
        response = setup.chat(question, chat_engine)
        print(f"Assistant: {response}\n")
