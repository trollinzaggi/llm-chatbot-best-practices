"""
Memory Integration Helper

This module provides utilities to easily add memory capabilities
to existing chatbot implementations.
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_system import (
    create_memory_adapter,
    load_config,
    ConfigManager,
    MemoryConfig
)


class MemoryMixin:
    """
    Mixin class to add memory capabilities to existing chatbots.
    
    Usage:
        class YourChatbot(BaseChatbot, MemoryMixin):
            def __init__(self):
                super().__init__(...)
                self.setup_memory('your_framework')
    """
    
    def setup_memory(self, framework: str, config: Optional[Dict[str, Any]] = None):
        """
        Setup memory system for the chatbot.
        
        Args:
            framework: Framework name ('langchain', 'agno', etc.)
            config: Optional configuration overrides
        """
        # Initialize memory adapter if not already done
        if "memory_adapter" not in st.session_state:
            try:
                # Load configuration
                memory_config = load_config()
                
                # Override with custom config if provided
                if config:
                    self.config_manager = ConfigManager(memory_config)
                    self.config_manager.update_config(config, validate=False)
                else:
                    self.config_manager = ConfigManager(memory_config)
                
                # Create memory adapter
                st.session_state.memory_adapter = create_memory_adapter(
                    framework,
                    user_id=st.session_state.get('user_id', 'default'),
                    config=self.config_manager.get_config()
                )
                
                # Start conversation
                st.session_state.conversation_id = st.session_state.memory_adapter.start_conversation(
                    title=f"{framework.title()} Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                
            except Exception as e:
                st.error(f"Failed to setup memory: {str(e)}")
                st.session_state.memory_adapter = None
        
        self.memory_adapter = st.session_state.get('memory_adapter')
    
    def add_memory_to_sidebar(self):
        """Add memory controls to the sidebar."""
        if not hasattr(self, 'memory_adapter') or not self.memory_adapter:
            return
        
        with st.sidebar:
            st.divider()
            st.subheader("🧠 Memory System")
            
            # User ID
            user_id = st.text_input(
                "User ID",
                value=st.session_state.get('user_id', 'default'),
                help="Your unique identifier for memory"
            )
            if user_id != st.session_state.get('user_id'):
                st.session_state.user_id = user_id
                if self.memory_adapter:
                    self.memory_adapter.user_id = user_id
            
            # Memory stats
            if self.memory_adapter:
                stats = self.memory_adapter.get_memory_statistics()
                col1, col2 = st.columns(2)
                
                with col1:
                    messages = stats.get('session', {}).get('message_count', 0)
                    st.metric("Messages", messages)
                
                with col2:
                    topics = len(stats.get('session', {}).get('topics', []))
                    st.metric("Topics", topics)
                
                # Memory usage
                usage = stats.get('session', {}).get('memory_usage_ratio', 0)
                st.progress(usage, text=f"Memory: {usage:.0%}")
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save"):
                    self.save_memory()
            
            with col2:
                if st.button("🗑️ Clear"):
                    self.clear_memory()
    
    def save_memory(self):
        """Save conversation to persistent storage."""
        try:
            if self.memory_adapter:
                conv_id = self.memory_adapter.save_conversation()
                st.success(f"Saved! ID: {conv_id[:8]}...")
        except Exception as e:
            st.error(f"Save failed: {str(e)}")
    
    def clear_memory(self):
        """Clear session memory."""
        try:
            if self.memory_adapter:
                self.memory_adapter.clear_session_memory()
                st.session_state.messages = []
                st.success("Memory cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Clear failed: {str(e)}")
    
    def enhance_prompt_with_memory(self, prompt: str) -> str:
        """
        Enhance user prompt with memory context.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Enhanced prompt with memory context
        """
        if not self.memory_adapter:
            return prompt
        
        try:
            # Add to memory
            self.memory_adapter.add_user_message(prompt)
            
            # Get enhanced prompt
            enhanced = self.memory_adapter.build_enhanced_prompt(prompt)
            
            return enhanced
        except Exception:
            return prompt
    
    def store_response_in_memory(self, response: str, metadata: Optional[Dict] = None):
        """
        Store assistant response in memory.
        
        Args:
            response: Assistant response
            metadata: Optional metadata
        """
        if self.memory_adapter:
            try:
                self.memory_adapter.add_assistant_message(response, metadata)
            except Exception:
                pass  # Silently fail to not disrupt chat


def quick_memory_setup(chatbot_class):
    """
    Decorator to quickly add memory to a chatbot class.
    
    Usage:
        @quick_memory_setup
        class YourChatbot(BaseChatbot):
            ...
    """
    original_init = chatbot_class.__init__
    original_sidebar = chatbot_class.render_sidebar if hasattr(chatbot_class, 'render_sidebar') else None
    original_response = chatbot_class.get_response if hasattr(chatbot_class, 'get_response') else None
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Auto-detect framework from class name
        framework = self.__class__.__name__.lower().replace('chatbot', '')
        if 'langchain' in framework:
            framework = 'langchain'
        elif 'agno' in framework:
            framework = 'agno'
        elif 'crewai' in framework:
            framework = 'crewai'
        elif 'autogen' in framework:
            framework = 'autogen'
        elif 'llama' in framework:
            framework = 'llama_index'
        elif 'langgraph' in framework:
            framework = 'langgraph'
        else:
            framework = 'agno'  # Default
        
        # Setup memory
        mixin = MemoryMixin()
        mixin.setup_memory(framework)
        self.memory_adapter = mixin.memory_adapter
        self.memory_mixin = mixin
    
    def new_sidebar(self, *args, **kwargs):
        if original_sidebar:
            original_sidebar(self, *args, **kwargs)
        if hasattr(self, 'memory_mixin'):
            self.memory_mixin.add_memory_to_sidebar()
    
    def new_response(self, prompt: str, *args, **kwargs):
        # Enhance prompt with memory
        if hasattr(self, 'memory_mixin'):
            enhanced_prompt = self.memory_mixin.enhance_prompt_with_memory(prompt)
        else:
            enhanced_prompt = prompt
        
        # Get response
        if original_response:
            response = original_response(self, enhanced_prompt, *args, **kwargs)
        else:
            response = "Response not implemented"
        
        # Store in memory
        if hasattr(self, 'memory_mixin'):
            self.memory_mixin.store_response_in_memory(response)
        
        return response
    
    # Replace methods
    chatbot_class.__init__ = new_init
    if original_sidebar:
        chatbot_class.render_sidebar = new_sidebar
    if original_response:
        chatbot_class.get_response = new_response
    
    return chatbot_class


# Example integration functions for each framework

def integrate_memory_langchain(chain, memory_adapter):
    """
    Integrate memory with a LangChain chain.
    
    Args:
        chain: LangChain chain
        memory_adapter: LangChainMemoryAdapter instance
    """
    if hasattr(chain, 'memory'):
        # Sync with existing memory
        memory_adapter.langchain_memory = chain.memory
        memory_adapter.sync_with_langchain_memory()
    
    # Create memory-enhanced chain
    return memory_adapter.create_memory_chain(chain.__class__, **chain.__dict__)


def integrate_memory_agno(agent, memory_adapter):
    """
    Integrate memory with an Agno agent.
    
    Args:
        agent: Agno agent
        memory_adapter: AgnoMemoryAdapter instance
    """
    memory_adapter.set_agent(agent)
    
    # Wrap agent calls
    original_chat = agent.chat if hasattr(agent, 'chat') else None
    
    def chat_with_memory(prompt, **kwargs):
        enhanced_prompt = memory_adapter.inject_memory_context(prompt)
        response = original_chat(enhanced_prompt, **kwargs) if original_chat else str(agent(enhanced_prompt))
        memory_adapter.process_response(response, {'agent': agent.name if hasattr(agent, 'name') else 'agent'})
        return response
    
    if original_chat:
        agent.chat = chat_with_memory
    
    return agent


def integrate_memory_crewai(crew, memory_adapter):
    """
    Integrate memory with a CrewAI crew.
    
    Args:
        crew: CrewAI crew
        memory_adapter: CrewAIMemoryAdapter instance
    """
    # Track agents
    if hasattr(crew, 'agents'):
        for agent in crew.agents:
            agent_name = agent.role if hasattr(agent, 'role') else str(agent)
            memory_adapter.agent_memories[agent_name] = []
    
    # Wrap kickoff
    original_kickoff = crew.kickoff
    
    def kickoff_with_memory(inputs):
        # Enhance inputs with memory
        enhanced_inputs = inputs.copy()
        if 'task' in enhanced_inputs:
            enhanced_inputs['task'] = memory_adapter.inject_memory_context(enhanced_inputs['task'])
        
        # Execute
        result = original_kickoff(enhanced_inputs)
        
        # Process result
        memory_adapter.process_response(str(result), {'crew': 'execution'})
        
        return result
    
    crew.kickoff = kickoff_with_memory
    return crew


def integrate_memory_autogen(agents, memory_adapter):
    """
    Integrate memory with AutoGen agents.
    
    Args:
        agents: List of AutoGen agents
        memory_adapter: AutoGenMemoryAdapter instance
    """
    for agent in agents:
        # Track interactions
        original_send = agent.send if hasattr(agent, 'send') else None
        
        def send_with_memory(message, recipient, **kwargs):
            # Track interaction
            memory_adapter.track_agent_interaction(
                agent.name if hasattr(agent, 'name') else str(agent),
                recipient.name if hasattr(recipient, 'name') else str(recipient),
                message
            )
            
            # Send
            if original_send:
                return original_send(message, recipient, **kwargs)
        
        if original_send:
            agent.send = send_with_memory
    
    return agents


def integrate_memory_llamaindex(index, memory_adapter):
    """
    Integrate memory with a LlamaIndex index.
    
    Args:
        index: LlamaIndex index
        memory_adapter: LlamaIndexMemoryAdapter instance
    """
    # Create query engine with memory
    query_engine = memory_adapter.create_hierarchical_query_engine()
    
    # Wrap query method
    original_query = query_engine.query if hasattr(query_engine, 'query') else None
    
    def query_with_memory(query_str):
        # Process with memory
        response = memory_adapter.process_rag_interaction(query_str)
        return response
    
    if original_query:
        query_engine.query = query_with_memory
    
    return query_engine


def integrate_memory_langgraph(graph, memory_adapter):
    """
    Integrate memory with a LangGraph graph.
    
    Args:
        graph: LangGraph graph
        memory_adapter: LangGraphMemoryAdapter instance
    """
    # Enhance graph with memory nodes
    enhanced_graph = memory_adapter.enhance_graph_with_memory(graph)
    
    # Wrap execution
    original_invoke = enhanced_graph.invoke if hasattr(enhanced_graph, 'invoke') else None
    
    def invoke_with_memory(initial_input):
        # Process with memory
        return memory_adapter.process_graph_execution(enhanced_graph, initial_input)
    
    if original_invoke:
        enhanced_graph.invoke = invoke_with_memory
    
    return enhanced_graph
