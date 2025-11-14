"""
LangChain framework memory adapter.

This module provides integration between LangChain's built-in memory
and our unified memory system.
"""

from typing import List, Dict, Optional, Any, Union
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework, MemoryFragment


class LangChainMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for LangChain framework.
    
    This adapter bridges LangChain's built-in memory system with our
    unified memory system, providing synchronization and enhanced features.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LangChain memory adapter.
        
        Args:
            llm: LangChain LLM instance for memory operations
            config: Configuration dictionary with optional keys:
                - memory_type: Type of LangChain memory ('buffer', 'summary_buffer', 'vector')
                - memory_key: Key for memory in chains (default: 'history')
                - return_messages: Whether to return messages (default: True)
        """
        self.llm = llm
        self.langchain_memory = None
        self.memory_type = config.get('memory_type', 'buffer') if config else 'buffer'
        self.memory_key = config.get('memory_key', 'history') if config else 'history'
        self.return_messages = config.get('return_messages', True) if config else True
        
        super().__init__(config)
    
    def _initialize_framework(self) -> None:
        """Initialize LangChain-specific components."""
        self.session_memory.conversation.framework = Framework.LANGCHAIN
        
        # Create LangChain memory based on configuration
        if self.memory_type == 'buffer':
            self.langchain_memory = ConversationBufferMemory(
                memory_key=self.memory_key,
                return_messages=self.return_messages
            )
        elif self.memory_type == 'summary_buffer' and self.llm:
            self.langchain_memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key=self.memory_key,
                return_messages=self.return_messages,
                max_token_limit=2000
            )
        else:
            # Default to buffer memory
            self.langchain_memory = ConversationBufferMemory(
                memory_key=self.memory_key,
                return_messages=self.return_messages
            )
    
    def inject_memory_context(self, input_text: str,
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context into LangChain input.
        
        Args:
            input_text: Original input
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input (for LangChain, we typically return original
            as memory is handled by the chain)
        """
        # LangChain handles memory internally, but we can enhance the prompt
        # with long-term memories from our system
        
        if self.persistent_memory:
            memories = self.retrieve_relevant_memories(input_text, limit=3)
            if memories:
                memory_context = "\nRelevant information from long-term memory:\n"
                for memory in memories:
                    memory_context += f"- {memory.content}\n"
                return f"{memory_context}\n{input_text}"
        
        return input_text
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process LangChain response and sync with our memory.
        
        Args:
            response: Chain response
            metadata: Optional metadata from the chain
            
        Returns:
            Processed response
        """
        # Extract any intermediate steps or chain outputs
        if metadata:
            # Check for intermediate steps in sequential chains
            if 'intermediate_steps' in metadata:
                for step in metadata['intermediate_steps']:
                    # Store intermediate results as context
                    if isinstance(step, dict) and 'output' in step:
                        self.session_memory.conversation.metadata['intermediate'] = step['output']
            
            # Check for tool usage
            if 'tool_calls' in metadata:
                for tool_call in metadata['tool_calls']:
                    # Track tool usage similar to Agno
                    if not hasattr(self, 'tool_history'):
                        self.tool_history = []
                    self.tool_history.append({
                        'tool': tool_call.get('tool'),
                        'input': tool_call.get('input'),
                        'output': tool_call.get('output')
                    })
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get LangChain-specific context data.
        
        Returns:
            Dictionary with LangChain-specific context
        """
        context = {
            'memory_type': self.memory_type,
            'memory_key': self.memory_key
        }
        
        # Add LangChain memory buffer if available
        if self.langchain_memory:
            if hasattr(self.langchain_memory, 'buffer'):
                context['buffer_size'] = len(self.langchain_memory.buffer)
            if hasattr(self.langchain_memory, 'moving_summary_buffer'):
                context['has_summary'] = bool(self.langchain_memory.moving_summary_buffer)
        
        return context
    
    def sync_with_langchain_memory(self) -> None:
        """Synchronize our memory with LangChain's memory."""
        if not self.langchain_memory:
            return
        
        # Get messages from LangChain memory
        if hasattr(self.langchain_memory, 'chat_memory'):
            langchain_messages = self.langchain_memory.chat_memory.messages
            
            # Sync with our session memory
            for lc_msg in langchain_messages:
                # Check if message already exists in our memory
                exists = False
                for our_msg in self.session_memory.messages:
                    if our_msg.content == lc_msg.content:
                        exists = True
                        break
                
                if not exists:
                    # Add to our memory
                    role = self._langchain_message_to_role(lc_msg)
                    self.session_memory.add_message(role, lc_msg.content)
    
    def sync_to_langchain_memory(self) -> None:
        """Synchronize our memory to LangChain's memory."""
        if not self.langchain_memory:
            return
        
        # Clear LangChain memory
        self.langchain_memory.clear()
        
        # Add our messages to LangChain memory
        for message in self.session_memory.messages:
            if message.role == MessageRole.USER:
                lc_message = HumanMessage(content=message.content)
            elif message.role == MessageRole.ASSISTANT:
                lc_message = AIMessage(content=message.content)
            elif message.role == MessageRole.SYSTEM:
                lc_message = SystemMessage(content=message.content)
            else:
                continue
            
            # Add to LangChain memory
            if hasattr(self.langchain_memory, 'chat_memory'):
                self.langchain_memory.chat_memory.add_message(lc_message)
    
    def create_memory_chain(self, chain_class, **chain_kwargs):
        """
        Create a LangChain chain with integrated memory.
        
        Args:
            chain_class: LangChain chain class
            **chain_kwargs: Additional arguments for the chain
            
        Returns:
            Chain instance with memory
        """
        # Ensure memory is synced
        self.sync_to_langchain_memory()
        
        # Add memory to chain kwargs
        chain_kwargs['memory'] = self.langchain_memory
        
        # Create chain
        chain = chain_class(**chain_kwargs)
        
        return chain
    
    def process_chain_interaction(self, user_input: str, chain) -> str:
        """
        Process a complete chain interaction with memory management.
        
        Args:
            user_input: User input
            chain: LangChain chain instance
            
        Returns:
            Chain response
        """
        # Add user message to our memory
        self.add_user_message(user_input)
        
        # Enhance input with long-term memories
        enhanced_input = self.inject_memory_context(user_input)
        
        # Run chain
        if hasattr(chain, 'invoke'):
            # New LangChain interface
            result = chain.invoke({'input': enhanced_input})
        elif hasattr(chain, 'run'):
            # Legacy interface
            result = chain.run(enhanced_input)
        else:
            result = chain(enhanced_input)
        
        # Extract response
        if isinstance(result, dict):
            response = result.get('output', result.get('response', str(result)))
        else:
            response = str(result)
        
        # Process and store response
        processed_response = self.process_response(response, {'chain_output': result})
        
        # Add assistant message to our memory
        self.add_assistant_message(processed_response)
        
        # Sync back from LangChain memory
        self.sync_with_langchain_memory()
        
        return processed_response
    
    def get_langchain_messages(self) -> List[BaseMessage]:
        """
        Get messages in LangChain format.
        
        Returns:
            List of LangChain BaseMessage objects
        """
        langchain_messages = []
        
        for message in self.session_memory.messages:
            if message.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=message.content))
            elif message.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=message.content))
            elif message.role == MessageRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=message.content))
        
        return langchain_messages
    
    def create_vector_memory(self, embeddings, vector_store=None):
        """
        Create a vector-based memory for semantic search.
        
        Args:
            embeddings: LangChain embeddings instance
            vector_store: Optional vector store instance
            
        Returns:
            VectorStoreRetrieverMemory instance
        """
        if not vector_store:
            # Create a simple in-memory vector store
            from langchain.vectorstores import FAISS
            texts = ["Initial memory"]
            vector_store = FAISS.from_texts(texts, embeddings)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        vector_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="semantic_history"
        )
        
        return vector_memory
    
    def add_to_vector_memory(self, vector_memory, text: str) -> None:
        """
        Add text to vector memory.
        
        Args:
            vector_memory: VectorStoreRetrieverMemory instance
            text: Text to add
        """
        if hasattr(vector_memory.retriever, 'vectorstore'):
            vector_memory.retriever.vectorstore.add_texts([text])
    
    def _langchain_message_to_role(self, message: BaseMessage) -> MessageRole:
        """
        Convert LangChain message to our MessageRole.
        
        Args:
            message: LangChain message
            
        Returns:
            MessageRole enum value
        """
        if isinstance(message, HumanMessage):
            return MessageRole.USER
        elif isinstance(message, AIMessage):
            return MessageRole.ASSISTANT
        elif isinstance(message, SystemMessage):
            return MessageRole.SYSTEM
        else:
            return MessageRole.ASSISTANT  # Default
    
    def export_to_langchain_format(self) -> Dict[str, Any]:
        """
        Export memory in LangChain-compatible format.
        
        Returns:
            Dictionary with LangChain-formatted memory
        """
        return {
            'messages': [
                {
                    'type': self._role_to_langchain_type(msg.role),
                    'content': msg.content
                }
                for msg in self.session_memory.messages
            ],
            'summaries': self.session_memory.summaries,
            'metadata': self.session_memory.conversation.metadata
        }
    
    def _role_to_langchain_type(self, role: MessageRole) -> str:
        """
        Convert our MessageRole to LangChain message type string.
        
        Args:
            role: MessageRole enum
            
        Returns:
            LangChain message type string
        """
        role_map = {
            MessageRole.USER: 'human',
            MessageRole.ASSISTANT: 'ai',
            MessageRole.SYSTEM: 'system'
        }
        return role_map.get(role, 'ai')
    
    def import_from_langchain_format(self, data: Dict[str, Any]) -> None:
        """
        Import memory from LangChain format.
        
        Args:
            data: LangChain-formatted memory data
        """
        # Clear current memory
        self.clear_session_memory()
        
        # Import messages
        for msg_data in data.get('messages', []):
            role = MessageRole.USER if msg_data['type'] == 'human' else MessageRole.ASSISTANT
            self.session_memory.add_message(role, msg_data['content'])
        
        # Import summaries
        if 'summaries' in data:
            self.session_memory.summaries = data['summaries']
        
        # Sync to LangChain memory
        self.sync_to_langchain_memory()
