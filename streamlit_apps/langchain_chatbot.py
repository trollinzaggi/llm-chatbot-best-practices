"""
LangChain Chatbot with Azure OpenAI

Streamlit chatbot application using LangChain with Azure OpenAI.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.langchain.azure_langchain_setup import AzureLangChainSetup
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class LangChainChatbot(BaseChatbot):
    """LangChain-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="🔗 LangChain Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates LangChain integration with Azure OpenAI.
            Features include conversation memory, chain of prompts, and structured reasoning.
            """
        )
        self.setup = None
        self.chain = None
        self.initialize_chain()
    
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
    
    def render_sidebar(self):
        """Render sidebar with LangChain-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("🔗 LangChain Settings")
            
            # Chain type selection
            chain_type = st.selectbox(
                "Chain Type",
                ["Conversation", "Analysis", "Q&A", "Creative"],
                help="Select the type of chain to use"
            )
            
            if st.button("Switch Chain Type"):
                self.switch_chain_type(chain_type)
            
            # Memory settings
            st.caption("Memory Settings:")
            if self.chain and hasattr(self.chain, 'memory'):
                memory_buffer = self.chain.memory.buffer if hasattr(self.chain.memory, 'buffer') else []
                st.caption(f"Messages in memory: {len(memory_buffer)}")
                
                if st.button("Clear Memory"):
                    self.chain.memory.clear()
                    st.success("Memory cleared!")
                    st.rerun()
    
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
            st.success(f"Switched to {chain_type} chain!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to switch chain: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from LangChain
        
        Args:
            prompt: User prompt
        
        Returns:
            Chain response
        """
        try:
            if self.chain is None:
                return "❌ Chain not initialized. Please check your configuration."
            
            # Check if it's a conversation chain or other type
            if hasattr(self.chain, 'invoke'):
                # For conversation chains
                if hasattr(self.chain, 'memory'):
                    response = self.chain.invoke({"input": prompt})
                    return response.get("response", response.get("text", str(response)))
                else:
                    # For other chains, we might need different input keys
                    try:
                        response = self.chain.invoke({"input": prompt})
                    except:
                        try:
                            response = self.chain.invoke({"text": prompt})
                        except:
                            response = self.chain.invoke({"question": prompt, "context": ""})
                    
                    if isinstance(response, dict):
                        # Try to get the most relevant output
                        for key in ["response", "text", "output", "answer", "result"]:
                            if key in response:
                                return response[key]
                        # If no known key, return the string representation
                        return str(response)
                    return str(response)
            else:
                return "❌ Invalid chain configuration"
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries based on chain type
        with st.expander("💡 Example Queries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Conversation Examples:**")
                st.code("Tell me about yourself")
                st.code("What did we just discuss?")
                st.code("Can you elaborate on that?")
            
            with col2:
                st.markdown("**Analysis Examples:**")
                st.code("Analyze the pros and cons of remote work")
                st.code("What are the key trends in AI?")
                st.code("Summarize the impact of climate change")


def main():
    """Main function to run the LangChain chatbot"""
    chatbot = LangChainChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
