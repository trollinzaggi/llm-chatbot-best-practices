"""
LangChain Chatbot with Azure OpenAI and Enhanced Memory System

Streamlit chatbot application using LangChain with Azure OpenAI and
integrated unified memory system for session and persistent memory.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.langchain.azure_langchain_setup import AzureLangChainSetup
from utils import setup_logger

# Import memory system
from memory_system import (
    LangChainMemoryAdapter,
    load_config,
    ConfigManager
)
from memory_system.retrieval import RetrievalQuery
from datetime import datetime

# Set up logging
logger = setup_logger(__name__)


class LangChainChatbot(BaseChatbot):
    """LangChain-based chatbot with Azure OpenAI and enhanced memory"""
    
    def __init__(self):
        super().__init__(
            title="LangChain Chatbot with Enhanced Memory",
            description="""
            This chatbot demonstrates LangChain integration with Azure OpenAI.
            Features include enhanced memory system with persistent storage,
            semantic retrieval, and intelligent context management.
            """
        )
        self.setup = None
        self.chain = None
        self.memory_adapter = None
        self.config_manager = None
        self.initialize_chain()
        self.initialize_memory()
    
    def initialize_chain(self):
        """Initialize LangChain components"""
        if "langchain_setup" not in st.session_state:
            try:
                st.session_state.langchain_setup = AzureLangChainSetup()
                st.session_state.conversation_chain = st.session_state.langchain_setup.create_conversation_chain(
                    system_prompt="""You are a helpful AI assistant powered by LangChain and Azure OpenAI.
                    You maintain context across conversations and provide thoughtful, structured responses.
                    Use your memory to reference previous parts of our conversation when relevant."""
                )
                logger.info("Initialized LangChain with conversation memory")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {str(e)}")
                st.session_state.langchain_setup = None
                st.session_state.conversation_chain = None
        
        self.setup = st.session_state.langchain_setup
        self.chain = st.session_state.conversation_chain
    
    def initialize_memory(self):
        """Initialize the enhanced memory system"""
        if "memory_adapter" not in st.session_state:
            try:
                # Load memory configuration
                config = load_config()
                
                # Create configuration manager
                self.config_manager = ConfigManager(config)
                
                # Initialize LangChain memory adapter
                st.session_state.memory_adapter = LangChainMemoryAdapter(
                    llm=self.setup.llm if self.setup else None,
                    config={
                        'memory_type': 'summary_buffer',
                        'memory_key': 'history',
                        'return_messages': True,
                        'user_id': st.session_state.get('user_id', 'default'),
                        'enable_persistent': True,
                        'db_path': 'langchain_memory.db'
                    }
                )
                
                # Start conversation
                st.session_state.conversation_id = st.session_state.memory_adapter.start_conversation(
                    title=f"LangChain Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                
                logger.info("Initialized enhanced memory system")
                
                # Sync with LangChain's native memory if available
                if self.chain and hasattr(self.chain, 'memory'):
                    st.session_state.memory_adapter.langchain_memory = self.chain.memory
                    st.session_state.memory_adapter.sync_with_langchain_memory()
                    
            except Exception as e:
                logger.error(f"Failed to initialize memory system: {str(e)}")
                st.session_state.memory_adapter = None
        
        self.memory_adapter = st.session_state.memory_adapter
    
    def render_sidebar(self):
        """Render sidebar with LangChain and memory settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            
            # LangChain Settings
            st.subheader("🔗 LangChain Settings")
            
            # Chain type selection
            chain_type = st.selectbox(
                "Chain Type",
                ["Conversation", "Analysis", "Q&A", "Creative"],
                help="Select the type of chain to use"
            )
            
            if st.button("Switch Chain Type"):
                self.switch_chain_type(chain_type)
            
            # Memory System Settings
            st.divider()
            st.subheader("🧠 Memory System")
            
            # User ID for persistent memory
            user_id = st.text_input(
                "User ID",
                value=st.session_state.get('user_id', 'default'),
                help="Unique ID for persistent memory"
            )
            if user_id != st.session_state.get('user_id'):
                st.session_state.user_id = user_id
                if self.memory_adapter:
                    self.memory_adapter.user_id = user_id
            
            # Memory statistics
            if self.memory_adapter:
                stats = self.memory_adapter.get_memory_statistics()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Session Messages", 
                             stats.get('session', {}).get('message_count', 0))
                    st.metric("Memory Usage",
                             f"{stats.get('session', {}).get('memory_usage_ratio', 0):.1%}")
                
                with col2:
                    st.metric("Total Memories",
                             stats.get('persistent', {}).get('total_memories', 0) if 'persistent' in stats else 0)
                    st.metric("Topics",
                             len(stats.get('session', {}).get('topics', [])))
            
            # Memory actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Conversation"):
                    self.save_conversation()
            
            with col2:
                if st.button("🗑️ Clear Memory"):
                    self.clear_memory()
            
            # Memory search
            st.divider()
            search_query = st.text_input("🔍 Search memories", key="memory_search")
            if search_query and st.button("Search"):
                self.search_memories(search_query)
            
            # Sync with LangChain memory
            if st.button("🔄 Sync Memories"):
                self.sync_memories()
    
    def switch_chain_type(self, chain_type: str):
        """Switch between different chain types"""
        try:
            if chain_type == "Conversation":
                st.session_state.conversation_chain = self.setup.create_conversation_chain()
            elif chain_type == "Analysis":
                from libraries.langchain.azure_langchain_setup import create_analysis_chain
                st.session_state.conversation_chain = create_analysis_chain(self.setup)
            elif chain_type == "Q&A":
                from libraries.langchain.azure_langchain_setup import create_qa_chain
                st.session_state.conversation_chain = create_qa_chain(self.setup)
            elif chain_type == "Creative":
                from libraries.langchain.azure_langchain_setup import create_creative_chain
                st.session_state.conversation_chain = create_creative_chain(self.setup)
            
            self.chain = st.session_state.conversation_chain
            
            # Re-sync memory with new chain
            if self.memory_adapter and self.chain and hasattr(self.chain, 'memory'):
                self.memory_adapter.langchain_memory = self.chain.memory
                self.memory_adapter.sync_to_langchain_memory()
            
            st.success(f"Switched to {chain_type} chain!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to switch chain: {str(e)}")
    
    def save_conversation(self):
        """Save current conversation to persistent storage"""
        try:
            if self.memory_adapter:
                conversation_id = self.memory_adapter.save_conversation()
                st.success(f"Conversation saved! ID: {conversation_id}")
                
                # Consolidate memories if needed
                self.memory_adapter.consolidate_user_memories()
        except Exception as e:
            st.error(f"Failed to save conversation: {str(e)}")
    
    def clear_memory(self):
        """Clear session and LangChain memory"""
        try:
            if self.memory_adapter:
                self.memory_adapter.clear_session_memory()
            
            if self.chain and hasattr(self.chain, 'memory'):
                self.chain.memory.clear()
            
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear memory: {str(e)}")
    
    def search_memories(self, query: str):
        """Search through memories"""
        try:
            if self.memory_adapter:
                # Search in session memory
                session_results = self.memory_adapter.session_memory.search(query, limit=3)
                
                # Search in persistent memory if available
                persistent_results = []
                if self.memory_adapter.persistent_memory:
                    persistent_results = self.memory_adapter.retrieve_relevant_memories(query, limit=3)
                
                # Display results
                if session_results or persistent_results:
                    with st.expander("Search Results", expanded=True):
                        if session_results:
                            st.subheader("Recent Conversations")
                            for msg, score in session_results:
                                st.write(f"**Score:** {score:.2f}")
                                st.write(f"**Content:** {msg.content[:200]}...")
                                st.divider()
                        
                        if persistent_results:
                            st.subheader("Long-term Memories")
                            for memory in persistent_results:
                                st.write(f"**Type:** {memory.fragment_type.value}")
                                st.write(f"**Content:** {memory.content[:200]}...")
                                st.divider()
                else:
                    st.info("No memories found matching your search.")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
    
    def sync_memories(self):
        """Sync between LangChain memory and our memory system"""
        try:
            if self.memory_adapter and self.chain and hasattr(self.chain, 'memory'):
                # Sync from LangChain to our system
                self.memory_adapter.sync_with_langchain_memory()
                
                # Sync from our system to LangChain
                self.memory_adapter.sync_to_langchain_memory()
                
                st.success("Memory synchronized!")
        except Exception as e:
            st.error(f"Sync failed: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from LangChain with memory enhancement
        
        Args:
            prompt: User prompt
        
        Returns:
            Chain response
        """
        try:
            if self.chain is None:
                return "Chain not initialized. Please check your configuration."
            
            # Add user message to memory
            if self.memory_adapter:
                self.memory_adapter.add_user_message(prompt)
                
                # Get enhanced prompt with memory context
                enhanced_prompt = self.memory_adapter.inject_memory_context(prompt)
            else:
                enhanced_prompt = prompt
            
            # Process with chain
            if self.memory_adapter and hasattr(self.memory_adapter, 'process_chain_interaction'):
                # Use memory adapter's chain interaction method
                response = self.memory_adapter.process_chain_interaction(enhanced_prompt, self.chain)
            else:
                # Fallback to direct chain invocation
                if hasattr(self.chain, 'invoke'):
                    if hasattr(self.chain, 'memory'):
                        response = self.chain.invoke({"input": enhanced_prompt})
                        response = response.get("response", response.get("text", str(response)))
                    else:
                        response = self.chain.invoke({"input": enhanced_prompt})
                        if isinstance(response, dict):
                            for key in ["response", "text", "output", "answer", "result"]:
                                if key in response:
                                    response = response[key]
                                    break
                        response = str(response)
                else:
                    response = "Invalid chain configuration"
                
                # Add assistant response to memory
                if self.memory_adapter:
                    self.memory_adapter.add_assistant_message(response)
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with memory status"""
        super().render_header()
        
        # Show memory status
        if self.memory_adapter:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.caption("🧠 Memory System: Active")
            
            with col2:
                if st.session_state.get('conversation_id'):
                    st.caption(f"📝 Conversation ID: {st.session_state.conversation_id[:8]}...")
            
            with col3:
                st.caption(f"👤 User: {st.session_state.get('user_id', 'default')}")
        
        # Show example queries based on chain type
        with st.expander("Example Queries"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Memory-Aware Examples:**")
                st.code("What did we discuss earlier?")
                st.code("Remember my preferences")
                st.code("Summarize our conversation")
            
            with col2:
                st.markdown("**Conversation Examples:**")
                st.code("Tell me about yourself")
                st.code("Can you elaborate on that?")
                st.code("What are your thoughts on AI?")
            
            with col3:
                st.markdown("**Analysis Examples:**")
                st.code("Analyze the pros and cons")
                st.code("What are the key trends?")
                st.code("Compare these options")


def main():
    """Main function to run the enhanced LangChain chatbot"""
    chatbot = LangChainChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
