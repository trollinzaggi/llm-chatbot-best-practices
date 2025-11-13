"""
Base Streamlit Chatbot Class

This module provides a base class for all Streamlit chatbot implementations.
"""
import streamlit as st
from typing import List, Dict, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from utils import setup_logger, sanitize_input

# Set up logging
logger = setup_logger(__name__)


class BaseChatbot:
    """Base class for Streamlit chatbot implementations"""
    
    def __init__(self, title: str, description: str):
        """
        Initialize base chatbot
        
        Args:
            title: Chatbot title
            description: Chatbot description
        """
        self.title = title
        self.description = description
        self.setup_page()
        self.initialize_session_state()
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.title,
            page_icon="AI",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "settings" not in st.session_state:
            st.session_state.settings = {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stream": False
            }
    
    def render_header(self):
        """Render chatbot header"""
        st.title(self.title)
        st.markdown(self.description)
        st.divider()
    
    def render_sidebar(self):
        """Render sidebar with settings"""
        with st.sidebar:
            st.header("Settings")
            
            # Temperature slider
            st.session_state.settings["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings["temperature"],
                step=0.1,
                help="Controls randomness in responses"
            )
            
            # Max tokens slider
            st.session_state.settings["max_tokens"] = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=st.session_state.settings["max_tokens"],
                step=100,
                help="Maximum length of response"
            )
            
            # Streaming toggle
            st.session_state.settings["stream"] = st.checkbox(
                "Stream Responses",
                value=st.session_state.settings["stream"],
                help="Stream responses in real-time"
            )
            
            st.divider()
            
            # Clear chat button
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
            
            # Export chat button
            if st.button("Export Chat", use_container_width=True):
                self.export_chat()
            
            st.divider()
            
            # Connection status
            self.render_connection_status()
    
    def render_connection_status(self):
        """Render Azure OpenAI connection status"""
        st.subheader("Connection Status")
        
        try:
            # Check if configuration is valid
            if config.endpoint and config.api_key:
                st.success("Connected to Azure OpenAI")
                st.caption(f"Endpoint: {config.endpoint}")
                st.caption(f"Deployment: {config.deployment_name}")
            else:
                st.error("Not configured")
                st.caption("Please configure Azure OpenAI settings")
        except Exception as e:
            st.error(f"Configuration Error: {str(e)}")
    
    def render_chat_messages(self):
        """Render chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def handle_user_input(self):
        """Handle user input"""
        if prompt := st.chat_input("Type your message here..."):
            # Sanitize and validate input
            prompt = sanitize_input(prompt, max_length=2000)
            
            if not prompt:
                st.warning("Please enter a valid message")
                return
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.get_response(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from the chatbot (to be implemented by subclasses)
        
        Args:
            prompt: User prompt
        
        Returns:
            Response string
        """
        raise NotImplementedError("Subclasses must implement get_response method")
    
    def export_chat(self):
        """Export chat history to file"""
        if not st.session_state.messages:
            st.warning("No messages to export")
            return
        
        # Create export content
        export_content = f"# {self.title} - Chat Export\n\n"
        for msg in st.session_state.messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            export_content += f"**{role}:** {content}\n\n"
        
        # Download button
        st.download_button(
            label="Download Chat",
            data=export_content,
            file_name="chat_export.md",
            mime="text/markdown"
        )
    
    def run(self):
        """
        Run the chatbot application
        """
        self.render_header()
        self.render_sidebar()
        self.render_chat_messages()
        self.handle_user_input()


class ErrorHandler:
    """Handle errors in chatbot applications"""
    
    @staticmethod
    def handle_api_error(error: Exception) -> str:
        """
        Handle API errors gracefully
        
        Args:
            error: Exception that occurred
        
        Returns:
            User-friendly error message
        """
        error_msg = str(error)
        
        if "api_key" in error_msg.lower():
            return "API Key error. Please check your Azure OpenAI configuration."
        elif "quota" in error_msg.lower():
            return "Rate limit exceeded. Please wait a moment and try again."
        elif "deployment" in error_msg.lower():
            return "Deployment not found. Please check your deployment name."
        elif "timeout" in error_msg.lower():
            return "Request timed out. Please try again."
        else:
            logger.error(f"API Error: {error_msg}")
            return f"An error occurred: {error_msg[:200]}..."
