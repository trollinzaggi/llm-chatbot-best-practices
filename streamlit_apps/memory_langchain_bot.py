"""
LangChain chatbot with integrated memory system.

This module provides a LangChain-based chatbot with full memory support,
combining LangChain's built-in memory with our enhanced memory system.
"""

import streamlit as st
import os
import sys
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_apps.memory_enhanced_base import MemoryEnhancedChatbot
from memory_system.adapters import LangChainMemoryAdapter

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationChain, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.callbacks import StreamlitCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.error("LangChain is not installed. Please install it with: pip install langchain")


class LangChainMemoryBot(MemoryEnhancedChatbot):
    """LangChain chatbot with enhanced memory capabilities."""
    
    def __init__(self):
        """Initialize LangChain memory bot."""
        super().__init__("langchain", "🔗 LangChain Memory Bot")
        self.llm = None
        self.chain = None
        self.langchain_adapter = None
    
    def initialize_langchain(self):
        """Initialize LangChain components."""
        if not LANGCHAIN_AVAILABLE:
            return
        
        # Get API key
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Model selection
            model_name = st.sidebar.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"],
                help="Select the model to use"
            )
            
            # Temperature setting
            temperature = st.sidebar.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Controls randomness in responses"
            )
            
            # Initialize LLM
            if "gpt" in model_name:
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=True
                )
            else:
                self.llm = OpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=True
                )
            
            # Create LangChain memory adapter
            self.langchain_adapter = LangChainMemoryAdapter(
                llm=self.llm,
                config={
                    'memory_type': 'conversation_summary_buffer',
                    'memory_key': 'history',
                    'return_messages': True
                }
            )
            
            # Create conversation chain with memory
            self.chain = self.langchain_adapter.create_memory_chain(
                ConversationChain,
                llm=self.llm,
                verbose=True
            )
            
            return True
        return False
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using LangChain.
        
        Args:
            prompt: User prompt (potentially enhanced)
            
        Returns:
            Generated response
        """
        if not self.chain:
            return "Please configure your OpenAI API key in the sidebar."
        
        try:
            # Use LangChain adapter for processing
            if self.langchain_adapter:
                response = self.langchain_adapter.process_chain_interaction(
                    prompt,
                    self.chain
                )
            else:
                # Fallback to direct chain call
                response = self.chain.run(prompt)
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def render_langchain_specific_controls(self):
        """Render LangChain-specific controls in sidebar."""
        with st.sidebar.expander("🔗 LangChain Settings"):
            # Memory type selection
            memory_type = st.selectbox(
                "Memory Type",
                ["buffer", "summary_buffer", "conversation_summary_buffer"],
                help="Select the LangChain memory type"
            )
            
            # Max token limit for summary buffer
            if "summary" in memory_type:
                max_token_limit = st.number_input(
                    "Max Token Limit",
                    min_value=100,
                    max_value=4000,
                    value=2000,
                    help="Maximum tokens before summarization"
                )
            
            # Chain type selection
            chain_type = st.selectbox(
                "Chain Type",
                ["conversation", "llm", "sequential"],
                help="Select the chain type to use"
            )
            
            # Verbose output
            verbose = st.checkbox(
                "Verbose Output",
                value=False,
                help="Show detailed chain execution"
            )
            
            if st.button("Apply LangChain Settings"):
                # Update adapter configuration
                if self.langchain_adapter:
                    self.langchain_adapter.config.update({
                        'memory_type': memory_type,
                        'verbose': verbose
                    })
                    st.success("Settings applied")
    
    def display_chain_insights(self):
        """Display LangChain-specific insights."""
        if self.chain and self.langchain_adapter:
            with st.expander("🔍 Chain Insights"):
                # Get chain memory buffer
                if hasattr(self.chain.memory, 'buffer'):
                    st.subheader("Memory Buffer")
                    st.text(self.chain.memory.buffer[-500:])  # Last 500 chars
                
                # Get framework-specific context
                context = self.langchain_adapter.get_framework_specific_context()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Memory Type", context.get('memory_type', 'N/A'))
                with col2:
                    st.metric("Buffer Size", context.get('buffer_size', 0))
    
    def run(self):
        """Run the LangChain memory bot."""
        st.title(self.title)
        
        # Initialize LangChain
        if not self.initialize_langchain():
            st.warning("Please configure LangChain in the sidebar")
            return
        
        # Initialize memory system
        self.initialize_memory_system()
        
        # Render sidebar with LangChain controls
        self.render_sidebar()
        self.render_langchain_specific_controls()
        
        # Display insights
        self.display_memory_insights()
        self.display_chain_insights()
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input with streaming
        if prompt := st.chat_input("Ask me anything..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add to memory
            self.add_message_with_memory("user", prompt)
            
            # Get enhanced context
            enhanced_prompt = self.get_enhanced_context(prompt)
            
            # Generate response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Use callback handler for streaming
                if self.llm and hasattr(self.llm, 'streaming') and self.llm.streaming:
                    callback = StreamlitCallbackHandler(st.container())
                    
                    # Generate with callback
                    response = self.chain.run(enhanced_prompt, callbacks=[callback])
                else:
                    # Generate without streaming
                    with st.spinner("Thinking..."):
                        response = self.generate_response(enhanced_prompt)
                
                message_placeholder.markdown(response)
            
            # Add to memory
            self.add_message_with_memory("assistant", response)


def main():
    """Main function to run LangChain memory bot."""
    # Page configuration
    st.set_page_config(
        page_title="LangChain Memory Bot",
        page_icon="🔗",
        layout="wide"
    )
    
    # Create and run chatbot
    chatbot = LangChainMemoryBot()
    chatbot.run()


if __name__ == "__main__":
    main()
