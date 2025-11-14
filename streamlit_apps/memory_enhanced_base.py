"""
Enhanced base chatbot with integrated memory system.

This module provides a base chatbot implementation with full memory support
including session memory, persistent storage, and retrieval capabilities.
"""

import streamlit as st
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_system import (
    create_memory_adapter,
    load_config,
    ConfigManager,
    MemoryConfig
)
from memory_system.retrieval import RetrievalManager, RetrievalQuery
from memory_system.processing import ProcessingManager


class MemoryEnhancedChatbot:
    """Base chatbot class with integrated memory system."""
    
    def __init__(self, framework: str, title: str = "Memory-Enhanced Chatbot"):
        """
        Initialize the memory-enhanced chatbot.
        
        Args:
            framework: Framework name for memory adapter
            title: Chatbot title
        """
        self.framework = framework
        self.title = title
        self.memory_adapter = None
        self.retrieval_manager = None
        self.processing_manager = None
        self.config_manager = None
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'memory_initialized' not in st.session_state:
            st.session_state.memory_initialized = False
        
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = 'default_user'
        
        if 'memory_stats' not in st.session_state:
            st.session_state.memory_stats = {}
    
    def initialize_memory_system(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory system components.
        
        Args:
            config: Optional memory configuration
        """
        if not st.session_state.memory_initialized:
            # Load or create configuration
            if config is None:
                config = load_config()
            
            # Create configuration manager
            self.config_manager = ConfigManager(config)
            
            # Create memory adapter
            self.memory_adapter = create_memory_adapter(
                self.framework,
                user_id=st.session_state.user_id,
                config=config
            )
            
            # Start conversation
            st.session_state.conversation_id = self.memory_adapter.start_conversation(
                title=f"{self.title} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Initialize retrieval manager
            retrieval_config = {
                'enable_semantic': True,
                'enable_keyword': True,
                'enable_hybrid': True,
                'default_retriever': 'hybrid'
            }
            self.retrieval_manager = RetrievalManager(retrieval_config)
            
            # Initialize processing manager
            processing_config = {
                'enable_summarization': True,
                'enable_extraction': True,
                'enable_consolidation': True
            }
            self.processing_manager = ProcessingManager(processing_config)
            
            st.session_state.memory_initialized = True
            
            # Load previous conversations if persistent memory is enabled
            if config.persistent.enabled:
                self._load_previous_context()
    
    def _load_previous_context(self):
        """Load relevant context from previous conversations."""
        try:
            # Retrieve recent memories
            memories = self.memory_adapter.retrieve_relevant_memories(
                "previous conversations",
                limit=5
            )
            
            if memories:
                context_message = "Relevant context from previous conversations:\n"
                for memory in memories:
                    context_message += f"- {memory.content}\n"
                
                # Add as system message
                st.session_state.messages.insert(0, {
                    "role": "system",
                    "content": context_message
                })
        except Exception as e:
            st.warning(f"Could not load previous context: {str(e)}")
    
    def render_sidebar(self):
        """Render the sidebar with memory controls and statistics."""
        with st.sidebar:
            st.title("🧠 Memory System")
            
            # User settings
            st.subheader("User Settings")
            st.session_state.user_id = st.text_input(
                "User ID",
                value=st.session_state.user_id,
                help="Unique identifier for memory persistence"
            )
            
            # Memory controls
            st.subheader("Memory Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Conversation"):
                    self.save_conversation()
            
            with col2:
                if st.button("🗑️ Clear Session"):
                    self.clear_session()
            
            # Memory search
            st.subheader("Memory Search")
            search_query = st.text_input("Search memories", key="memory_search")
            if search_query:
                self.search_memories(search_query)
            
            # Memory statistics
            st.subheader("Memory Statistics")
            if self.memory_adapter:
                stats = self.memory_adapter.get_memory_statistics()
                st.session_state.memory_stats = stats
                
                # Display statistics
                if 'session' in stats:
                    st.metric("Messages", stats['session'].get('message_count', 0))
                    st.metric("Topics", len(stats['session'].get('topics', [])))
                    
                    # Memory usage bar
                    usage = stats['session'].get('memory_usage_ratio', 0)
                    st.progress(usage, text=f"Memory Usage: {usage:.1%}")
            
            # Configuration
            with st.expander("⚙️ Configuration"):
                self.render_configuration_panel()
            
            # Previous conversations
            with st.expander("📚 Previous Conversations"):
                self.render_conversation_history()
    
    def render_configuration_panel(self):
        """Render configuration options."""
        if self.config_manager:
            config = self.config_manager.get_config()
            
            # Session memory settings
            st.subheader("Session Memory")
            max_messages = st.number_input(
                "Max Messages",
                min_value=10,
                max_value=200,
                value=config.session.max_messages
            )
            
            if st.button("Update Session Config"):
                self.config_manager.update_session_config(max_messages=max_messages)
                st.success("Configuration updated")
            
            # Features
            st.subheader("Features")
            
            col1, col2 = st.columns(2)
            with col1:
                semantic_search = st.checkbox(
                    "Semantic Search",
                    value=config.retrieval.semantic_search
                )
                auto_summarize = st.checkbox(
                    "Auto Summarize",
                    value=config.processing.auto_summarize
                )
            
            with col2:
                extract_entities = st.checkbox(
                    "Extract Entities",
                    value=config.processing.extract_entities
                )
                auto_consolidate = st.checkbox(
                    "Auto Consolidate",
                    value=config.processing.auto_consolidate
                )
            
            if st.button("Update Features"):
                updates = {
                    'retrieval': {'semantic_search': semantic_search},
                    'processing': {
                        'auto_summarize': auto_summarize,
                        'extract_entities': extract_entities,
                        'auto_consolidate': auto_consolidate
                    }
                }
                self.config_manager.update_config(updates)
                st.success("Features updated")
    
    def render_conversation_history(self):
        """Render previous conversation list."""
        # This would connect to persistent storage
        st.info("Conversation history will appear here")
        
        # Mock data for demonstration
        conversations = [
            "Product Discussion - 2024-01-15",
            "Technical Support - 2024-01-14",
            "Feature Request - 2024-01-13"
        ]
        
        for conv in conversations:
            if st.button(conv, key=f"conv_{conv}"):
                st.info(f"Loading conversation: {conv}")
    
    def search_memories(self, query: str):
        """
        Search through memories and display results.
        
        Args:
            query: Search query
        """
        if self.retrieval_manager and self.memory_adapter:
            # Create retrieval query
            retrieval_query = RetrievalQuery(
                text=query,
                user_id=st.session_state.user_id,
                limit=5
            )
            
            # Search
            results = self.memory_adapter.session_memory.search(query, limit=5)
            
            if results:
                st.success(f"Found {len(results)} relevant memories")
                for result, score in results:
                    with st.container():
                        st.write(f"**Score:** {score:.2f}")
                        st.write(f"**Content:** {result.content[:200]}...")
            else:
                st.info("No memories found")
    
    def save_conversation(self):
        """Save the current conversation to persistent storage."""
        try:
            if self.memory_adapter:
                # Save conversation
                conversation_id = self.memory_adapter.save_conversation()
                
                # Extract and store memories
                if self.processing_manager and hasattr(self.memory_adapter, 'persistent_memory'):
                    memories = self.processing_manager.extract_memories(
                        self.memory_adapter.session_memory.conversation,
                        st.session_state.user_id
                    )
                    
                    for memory in memories:
                        self.memory_adapter.persistent_memory.store_memory(memory)
                
                st.success(f"Conversation saved with ID: {conversation_id}")
                
                # Consolidate memories if needed
                if self.memory_adapter.persistent_memory:
                    self.memory_adapter.consolidate_user_memories()
        except Exception as e:
            st.error(f"Failed to save conversation: {str(e)}")
    
    def clear_session(self):
        """Clear the current session."""
        if self.memory_adapter:
            self.memory_adapter.clear_session_memory()
        
        st.session_state.messages = []
        st.session_state.memory_initialized = False
        st.success("Session cleared")
        st.rerun()
    
    def add_message_with_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to both chat and memory.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata
        """
        # Add to Streamlit messages
        st.session_state.messages.append({
            "role": role,
            "content": content
        })
        
        # Add to memory system
        if self.memory_adapter:
            if role == "user":
                self.memory_adapter.add_user_message(content, metadata)
            else:
                self.memory_adapter.add_assistant_message(content, metadata)
            
            # Extract information if processing manager is available
            if self.processing_manager:
                try:
                    extraction = self.processing_manager.extract_information(content)
                    
                    # Store important entities
                    if extraction.get('entities'):
                        for entity_type, values in extraction['entities'].items():
                            for value in values[:3]:  # Limit to top 3
                                if value:
                                    st.session_state.memory_stats[f'entity_{entity_type}'] = value
                except Exception as e:
                    print(f"Extraction error: {e}")
    
    def get_enhanced_context(self, user_input: str) -> str:
        """
        Get enhanced context for the user input.
        
        Args:
            user_input: User's input message
            
        Returns:
            Enhanced input with memory context
        """
        if self.memory_adapter:
            # Get conversation context
            context = self.memory_adapter.get_conversation_context()
            
            # Retrieve relevant memories
            memories = self.memory_adapter.retrieve_relevant_memories(user_input, limit=3)
            
            # Build enhanced prompt
            enhanced = self.memory_adapter.build_enhanced_prompt(user_input)
            
            return enhanced
        
        return user_input
    
    def display_memory_insights(self):
        """Display memory insights and analytics."""
        with st.expander("🔍 Memory Insights"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Memories", 
                         st.session_state.memory_stats.get('total_memories', 0))
            
            with col2:
                st.metric("Topics Discussed", 
                         len(st.session_state.memory_stats.get('topics', [])))
            
            with col3:
                st.metric("Entities Found", 
                         len([k for k in st.session_state.memory_stats.keys() 
                              if k.startswith('entity_')]))
            
            # Display topics
            topics = st.session_state.memory_stats.get('topics', [])
            if topics:
                st.subheader("Main Topics")
                st.write(", ".join(topics[:10]))
    
    def run(self):
        """Run the memory-enhanced chatbot."""
        st.title(self.title)
        
        # Initialize memory system
        self.initialize_memory_system()
        
        # Render sidebar
        self.render_sidebar()
        
        # Display memory insights
        self.display_memory_insights()
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] != "system":  # Don't display system messages
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What's on your mind?"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add to memory
            self.add_message_with_memory("user", prompt)
            
            # Get enhanced context
            enhanced_prompt = self.get_enhanced_context(prompt)
            
            # Generate response (placeholder - override in subclasses)
            response = self.generate_response(enhanced_prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add to memory
            self.add_message_with_memory("assistant", response)
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response to the user prompt.
        Override this in framework-specific implementations.
        
        Args:
            prompt: User prompt (potentially enhanced with memory)
            
        Returns:
            Generated response
        """
        return f"This is a placeholder response. Override this method in {self.framework} implementation."


def main():
    """Main function to run the base memory-enhanced chatbot."""
    chatbot = MemoryEnhancedChatbot("base", "Memory-Enhanced Base Chatbot")
    chatbot.run()


if __name__ == "__main__":
    main()
