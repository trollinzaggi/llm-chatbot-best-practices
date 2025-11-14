"""
Unified Memory System Demo Application.

This Streamlit app demonstrates the memory system integration across
all six LLM frameworks with a unified interface.
"""

import streamlit as st
import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_system import (
    create_memory_adapter,
    load_config,
    ConfigManager,
    MemoryConfig,
    get_supported_frameworks
)


class UnifiedMemoryDemo:
    """Unified demonstration of memory system across frameworks."""
    
    def __init__(self):
        """Initialize the unified demo."""
        self.current_framework = None
        self.memory_adapter = None
        self.config_manager = None
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'framework' not in st.session_state:
            st.session_state.framework = 'langchain'
        
        if 'messages' not in st.session_state:
            st.session_state.messages = {}
        
        if 'memory_adapters' not in st.session_state:
            st.session_state.memory_adapters = {}
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = 'demo_user'
        
        if 'conversation_ids' not in st.session_state:
            st.session_state.conversation_ids = {}
        
        if 'memory_config' not in st.session_state:
            st.session_state.memory_config = None
    
    def render_header(self):
        """Render the application header."""
        st.title("🧠 Unified Memory System Demo")
        st.markdown("""
        This demo showcases the unified memory system working across all six LLM frameworks.
        Switch between frameworks while maintaining conversation context and memories.
        """)
        
        # Framework selector
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            framework = st.selectbox(
                "Select Framework",
                get_supported_frameworks(),
                index=get_supported_frameworks().index(st.session_state.framework),
                help="Choose an LLM framework to demonstrate"
            )
            
            if framework != st.session_state.framework:
                st.session_state.framework = framework
                self.switch_framework(framework)
        
        with col2:
            user_id = st.text_input(
                "User ID",
                value=st.session_state.user_id,
                help="User identifier for memory persistence"
            )
            if user_id != st.session_state.user_id:
                st.session_state.user_id = user_id
        
        with col3:
            if st.button("🔄 Reset All"):
                self.reset_all()
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.title("Memory System Controls")
            
            # Current framework info
            st.subheader(f"Current: {st.session_state.framework.upper()}")
            
            # Memory operations
            st.subheader("Operations")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Memory"):
                    self.save_current_memory()
            
            with col2:
                if st.button("🔄 Sync Memories"):
                    self.sync_memories()
            
            if st.button("🗑️ Clear Current"):
                self.clear_current_framework()
            
            # Memory statistics
            st.subheader("Memory Statistics")
            self.display_memory_stats()
            
            # Configuration
            with st.expander("⚙️ Configuration"):
                self.render_configuration()
            
            # Cross-framework memories
            with st.expander("🔀 Cross-Framework Memories"):
                self.display_cross_framework_memories()
            
            # Export/Import
            with st.expander("📦 Export/Import"):
                self.render_export_import()
    
    def render_configuration(self):
        """Render configuration panel."""
        if not st.session_state.memory_config:
            st.session_state.memory_config = load_config()
        
        config = st.session_state.memory_config
        
        # Memory settings
        st.subheader("Memory Settings")
        
        max_messages = st.number_input(
            "Max Messages per Session",
            min_value=10,
            max_value=500,
            value=config.session.max_messages
        )
        
        importance_threshold = st.slider(
            "Importance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.persistent.importance_threshold,
            help="Minimum importance for memory storage"
        )
        
        # Feature toggles
        st.subheader("Features")
        
        semantic_search = st.checkbox(
            "Semantic Search",
            value=config.retrieval.semantic_search
        )
        
        auto_consolidate = st.checkbox(
            "Auto Consolidate",
            value=config.processing.auto_consolidate
        )
        
        if st.button("Apply Configuration"):
            # Update configuration
            config.session.max_messages = max_messages
            config.persistent.importance_threshold = importance_threshold
            config.retrieval.semantic_search = semantic_search
            config.processing.auto_consolidate = auto_consolidate
            
            st.session_state.memory_config = config
            st.success("Configuration updated")
    
    def display_memory_stats(self):
        """Display memory statistics for all frameworks."""
        stats = {}
        
        for framework in get_supported_frameworks():
            if framework in st.session_state.memory_adapters:
                adapter = st.session_state.memory_adapters[framework]
                framework_stats = adapter.get_memory_statistics()
                
                # Extract key metrics
                stats[framework] = {
                    'messages': framework_stats.get('session', {}).get('message_count', 0),
                    'memories': framework_stats.get('persistent', {}).get('total_memories', 0)
                }
        
        if stats:
            # Display as metrics
            for framework, framework_stats in stats.items():
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            f"{framework.upper()}",
                            framework_stats['messages'],
                            help="Messages"
                        )
                    with col2:
                        st.metric(
                            "Memories",
                            framework_stats['memories']
                        )
        else:
            st.info("No memory statistics available")
    
    def display_cross_framework_memories(self):
        """Display memories shared across frameworks."""
        st.subheader("Shared Memories")
        
        # Collect memories from all frameworks
        all_memories = []
        
        for framework, adapter in st.session_state.memory_adapters.items():
            if hasattr(adapter, 'session_memory'):
                # Get topics from session
                topics = adapter.session_memory.extract_topics()
                for topic in topics[:3]:
                    all_memories.append({
                        'framework': framework,
                        'type': 'topic',
                        'content': topic
                    })
        
        # Display memories
        if all_memories:
            for memory in all_memories:
                st.write(f"**{memory['framework']}**: {memory['content']}")
        else:
            st.info("No cross-framework memories yet")
    
    def render_export_import(self):
        """Render export/import functionality."""
        st.subheader("Export")
        
        if st.button("Export All Memories"):
            export_data = self.export_all_memories()
            
            # Create download button
            st.download_button(
                label="Download Export",
                data=json.dumps(export_data, indent=2),
                file_name=f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.subheader("Import")
        
        uploaded_file = st.file_uploader(
            "Choose a memory export file",
            type="json"
        )
        
        if uploaded_file is not None:
            if st.button("Import Memories"):
                try:
                    import_data = json.load(uploaded_file)
                    self.import_memories(import_data)
                    st.success("Memories imported successfully")
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")
    
    def switch_framework(self, framework: str):
        """
        Switch to a different framework.
        
        Args:
            framework: Framework to switch to
        """
        self.current_framework = framework
        
        # Get or create adapter for framework
        if framework not in st.session_state.memory_adapters:
            # Create new adapter
            adapter = create_memory_adapter(
                framework,
                user_id=st.session_state.user_id,
                config=st.session_state.memory_config
            )
            
            # Start conversation
            conversation_id = adapter.start_conversation(
                title=f"{framework} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            st.session_state.memory_adapters[framework] = adapter
            st.session_state.conversation_ids[framework] = conversation_id
            
            # Initialize messages for framework
            if framework not in st.session_state.messages:
                st.session_state.messages[framework] = []
        
        self.memory_adapter = st.session_state.memory_adapters[framework]
    
    def save_current_memory(self):
        """Save current framework's memory."""
        if self.memory_adapter:
            try:
                conversation_id = self.memory_adapter.save_conversation()
                st.success(f"Saved {self.current_framework} conversation: {conversation_id}")
            except Exception as e:
                st.error(f"Failed to save: {str(e)}")
    
    def sync_memories(self):
        """Synchronize memories across all frameworks."""
        st.info("Synchronizing memories across frameworks...")
        
        # Collect all memories
        all_memories = []
        
        for framework, adapter in st.session_state.memory_adapters.items():
            if hasattr(adapter, 'persistent_memory') and adapter.persistent_memory:
                memories = adapter.persistent_memory.retrieve_memories(
                    "all",
                    st.session_state.user_id,
                    limit=10
                )
                all_memories.extend(memories)
        
        # Share memories across frameworks
        for framework, adapter in st.session_state.memory_adapters.items():
            if hasattr(adapter, 'persistent_memory') and adapter.persistent_memory:
                for memory in all_memories:
                    try:
                        adapter.persistent_memory.store_memory(memory)
                    except:
                        pass
        
        st.success("Memory synchronization complete")
    
    def clear_current_framework(self):
        """Clear current framework's session."""
        if self.current_framework:
            if self.memory_adapter:
                self.memory_adapter.clear_session_memory()
            
            st.session_state.messages[self.current_framework] = []
            st.success(f"Cleared {self.current_framework} session")
            st.rerun()
    
    def reset_all(self):
        """Reset all frameworks and memories."""
        st.session_state.messages = {}
        st.session_state.memory_adapters = {}
        st.session_state.conversation_ids = {}
        st.success("All frameworks reset")
        st.rerun()
    
    def export_all_memories(self) -> Dict[str, Any]:
        """Export all memories from all frameworks."""
        export_data = {
            'user_id': st.session_state.user_id,
            'export_date': datetime.now().isoformat(),
            'frameworks': {}
        }
        
        for framework, adapter in st.session_state.memory_adapters.items():
            framework_data = {
                'messages': [],
                'memories': [],
                'statistics': {}
            }
            
            # Export messages
            if framework in st.session_state.messages:
                framework_data['messages'] = st.session_state.messages[framework]
            
            # Export statistics
            framework_data['statistics'] = adapter.get_memory_statistics()
            
            export_data['frameworks'][framework] = framework_data
        
        return export_data
    
    def import_memories(self, import_data: Dict[str, Any]):
        """Import memories from export data."""
        # Import for each framework
        for framework, framework_data in import_data.get('frameworks', {}).items():
            if framework in get_supported_frameworks():
                # Restore messages
                if 'messages' in framework_data:
                    st.session_state.messages[framework] = framework_data['messages']
    
    def render_chat_interface(self):
        """Render the chat interface for current framework."""
        st.subheader(f"💬 Chat with {self.current_framework.upper()}")
        
        # Get messages for current framework
        if self.current_framework not in st.session_state.messages:
            st.session_state.messages[self.current_framework] = []
        
        messages = st.session_state.messages[self.current_framework]
        
        # Display messages
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Message to {self.current_framework}..."):
            # Add user message
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add to memory
            if self.memory_adapter:
                self.memory_adapter.add_user_message(prompt)
                
                # Get enhanced prompt with memory context
                enhanced_prompt = self.memory_adapter.build_enhanced_prompt(prompt)
            else:
                enhanced_prompt = prompt
            
            # Generate response (simulated for demo)
            response = self.generate_demo_response(enhanced_prompt)
            
            # Add assistant message
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add to memory
            if self.memory_adapter:
                self.memory_adapter.add_assistant_message(response)
    
    def generate_demo_response(self, prompt: str) -> str:
        """
        Generate a demo response for the current framework.
        
        Args:
            prompt: User prompt (potentially enhanced)
            
        Returns:
            Demo response
        """
        framework = self.current_framework
        
        # Framework-specific demo responses
        responses = {
            'langchain': f"[LangChain Response] I'm processing your request using chains and memory: '{prompt[:50]}...'",
            'langgraph': f"[LangGraph Response] Executing graph workflow for: '{prompt[:50]}...'",
            'crewai': f"[CrewAI Response] Coordinating agents to handle: '{prompt[:50]}...'",
            'autogen': f"[AutoGen Response] Autonomous agents processing: '{prompt[:50]}...'",
            'llama_index': f"[LlamaIndex Response] Retrieving from indexed documents: '{prompt[:50]}...'",
            'agno': f"[Agno Response] Using tools to respond to: '{prompt[:50]}...'"
        }
        
        base_response = responses.get(framework, f"Processing with {framework}: '{prompt[:50]}...'")
        
        # Add memory context if available
        if self.memory_adapter:
            memories = self.memory_adapter.retrieve_relevant_memories(prompt, limit=2)
            if memories:
                base_response += "\n\n*Using memories:*\n"
                for memory in memories:
                    base_response += f"- {memory.content[:100]}...\n"
        
        return base_response
    
    def display_memory_visualization(self):
        """Display visualization of memory connections."""
        with st.expander("📊 Memory Visualization"):
            st.subheader("Memory Network")
            
            # Create a simple visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Active Frameworks", 
                         len(st.session_state.memory_adapters))
            
            with col2:
                total_messages = sum(
                    len(msgs) for msgs in st.session_state.messages.values()
                )
                st.metric("Total Messages", total_messages)
            
            with col3:
                st.metric("User Sessions", 
                         len(st.session_state.conversation_ids))
            
            # Memory flow diagram (simplified)
            st.markdown("""
            ```mermaid
            graph LR
                A[User Input] --> B[Session Memory]
                B --> C[Memory Extraction]
                C --> D[Persistent Storage]
                D --> E[Cross-Framework Sync]
                E --> F[Enhanced Context]
                F --> A
            ```
            """)
    
    def run(self):
        """Run the unified memory demo."""
        # Render header
        self.render_header()
        
        # Initialize current framework
        if not self.current_framework:
            self.switch_framework(st.session_state.framework)
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat interface
            self.render_chat_interface()
        
        with col2:
            # Memory visualization
            self.display_memory_visualization()
            
            # Framework comparison
            with st.expander("🔍 Framework Comparison"):
                self.display_framework_comparison()
    
    def display_framework_comparison(self):
        """Display comparison of frameworks."""
        st.subheader("Framework Features")
        
        features = {
            'langchain': ['Chains', 'Built-in Memory', 'Tools', 'Agents'],
            'langgraph': ['State Graphs', 'Cycles', 'Conditional Edges', 'Checkpoints'],
            'crewai': ['Multi-Agent', 'Task Delegation', 'Collaboration', 'Roles'],
            'autogen': ['Autonomous', 'Code Execution', 'Learning', 'Group Chat'],
            'llama_index': ['Indexing', 'RAG', 'Document Store', 'Query Engine'],
            'agno': ['Simple API', 'Tool Usage', 'Lightweight', 'Flexible']
        }
        
        for framework, framework_features in features.items():
            if framework == self.current_framework:
                st.success(f"**{framework.upper()}** (Active)")
            else:
                st.info(f"**{framework.upper()}**")
            
            st.write(", ".join(framework_features))


def main():
    """Main function to run the unified demo."""
    # Page configuration
    st.set_page_config(
        page_title="Unified Memory System Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create and run demo
    demo = UnifiedMemoryDemo()
    demo.run()


if __name__ == "__main__":
    main()
