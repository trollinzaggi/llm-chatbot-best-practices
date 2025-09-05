"""
Agno Chatbot with Azure OpenAI

Streamlit chatbot application using Agno library with Azure OpenAI.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.agno.azure_agno_setup import AzureAgnoAgent, CalculatorTool, WeatherTool
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class AgnoChatbot(BaseChatbot):
    """Agno-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="🤖 Agno Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates Agno integration with Azure OpenAI.
            Features include tool usage (calculator and weather) and structured responses.
            """
        )
        self.agent = None
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize Agno agent"""
        if "agent" not in st.session_state:
            try:
                st.session_state.agent = AzureAgnoAgent(
                    name="Agno Assistant",
                    system_prompt="""You are a helpful AI assistant powered by Azure OpenAI and Agno.
                    You have access to calculator and weather tools. Use them when appropriate.
                    Be concise but informative in your responses."""
                )
                
                # Add tools
                st.session_state.agent.add_tool(CalculatorTool())
                st.session_state.agent.add_tool(WeatherTool())
                
                logger.info("Initialized Agno agent with tools")
            except Exception as e:
                logger.error(f"Failed to initialize Agno agent: {str(e)}")
                st.session_state.agent = None
        
        self.agent = st.session_state.agent
    
    def render_sidebar(self):
        """Render sidebar with Agno-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("🛠️ Agno Settings")
            
            # Tool usage toggle
            use_tools = st.checkbox(
                "Enable Tools",
                value=True,
                help="Allow the agent to use calculator and weather tools"
            )
            st.session_state.use_tools = use_tools
            
            # Available tools info
            st.caption("Available Tools:")
            st.caption("• Calculator - Mathematical operations")
            st.caption("• Weather - Get weather information")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from Agno agent
        
        Args:
            prompt: User prompt
        
        Returns:
            Agent response
        """
        try:
            if self.agent is None:
                return "❌ Agent not initialized. Please check your configuration."
            
            # Update temperature and max_tokens if changed
            self.agent.client = None  # Force recreation with new settings
            self.agent.__init__(
                name=self.agent.name,
                system_prompt=self.agent.system_prompt
            )
            
            # Get response with or without tools
            use_tools = st.session_state.get("use_tools", True)
            response = self.agent.chat(prompt, use_tools=use_tools)
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries
        with st.expander("💡 Example Queries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Calculator Examples:**")
                st.code("What's 15 * 23?")
                st.code("Calculate 2^10")
                st.code("What's the square root of 144?")
            
            with col2:
                st.markdown("**Weather Examples:**")
                st.code("What's the weather in New York?")
                st.code("How's the weather in London?")
                st.code("Check weather for Tokyo")


def main():
    """Main function to run the Agno chatbot"""
    chatbot = AgnoChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
