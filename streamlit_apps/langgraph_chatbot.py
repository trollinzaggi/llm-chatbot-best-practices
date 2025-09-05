"""
LangGraph Chatbot with Azure OpenAI

Streamlit chatbot application using LangGraph with Azure OpenAI.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.langgraph.azure_langgraph_setup import AzureLangGraphSetup, run_graph_example
from langchain.schema import HumanMessage
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class LangGraphChatbot(BaseChatbot):
    """LangGraph-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="🔀 LangGraph Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates LangGraph integration with Azure OpenAI.
            Features include graph-based conversation flows, conditional routing, and iterative processing.
            """
        )
        self.setup = None
        self.graph = None
        self.initialize_graph()
    
    def initialize_graph(self):
        """Initialize LangGraph components"""
        if "langgraph_setup" not in st.session_state:
            try:
                st.session_state.langgraph_setup = AzureLangGraphSetup()
                st.session_state.current_graph_type = "simple"
                st.session_state.current_graph = st.session_state.langgraph_setup.create_simple_graph()
                logger.info("Initialized LangGraph with simple graph")
            except Exception as e:
                logger.error(f"Failed to initialize LangGraph: {str(e)}")
                st.session_state.langgraph_setup = None
                st.session_state.current_graph = None
        
        self.setup = st.session_state.langgraph_setup
        self.graph = st.session_state.current_graph
    
    def render_sidebar(self):
        """Render sidebar with LangGraph-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("🔀 LangGraph Settings")
            
            # Graph type selection
            graph_type = st.selectbox(
                "Graph Type",
                ["simple", "conditional", "cyclic"],
                index=["simple", "conditional", "cyclic"].index(
                    st.session_state.get("current_graph_type", "simple")
                ),
                help="Select the type of graph to use"
            )
            
            if graph_type != st.session_state.get("current_graph_type", "simple"):
                self.switch_graph_type(graph_type)
            
            # Graph visualization
            st.caption("Graph Information:")
            if graph_type == "simple":
                st.caption("📊 Linear flow: Input → Process → Analyze → Finalize")
            elif graph_type == "conditional":
                st.caption("📊 Conditional routing based on input classification")
                st.caption("• Questions → Question handler")
                st.caption("• Tasks → Task handler")
                st.caption("• Conversation → Conversation handler")
            else:  # cyclic
                st.caption("📊 Iterative improvement with max 3 cycles")
                st.caption("Generate → Review → Improve (if needed) → Finalize")
            
            # Show graph state
            if st.checkbox("Show Graph State", value=False):
                if hasattr(self.graph, 'get_state'):
                    state = self.graph.get_state()
                    st.json(state)
    
    def switch_graph_type(self, graph_type: str):
        """Switch between different graph types"""
        try:
            if graph_type == "simple":
                st.session_state.current_graph = self.setup.create_simple_graph()
            elif graph_type == "conditional":
                st.session_state.current_graph = self.setup.create_conditional_graph()
            elif graph_type == "cyclic":
                st.session_state.current_graph = self.setup.create_cyclic_graph()
            
            st.session_state.current_graph_type = graph_type
            self.graph = st.session_state.current_graph
            logger.info(f"Switched to {graph_type} graph")
        except Exception as e:
            st.error(f"Failed to switch graph: {str(e)}")
            logger.error(f"Graph switch error: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from LangGraph
        
        Args:
            prompt: User prompt
        
        Returns:
            Graph response
        """
        try:
            if self.graph is None:
                return "❌ Graph not initialized. Please check your configuration."
            
            # Run the graph with the user input
            response = run_graph_example(self.graph, prompt)
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries based on graph type
        with st.expander("💡 Example Queries"):
            graph_type = st.session_state.get("current_graph_type", "simple")
            
            if graph_type == "simple":
                st.markdown("**Simple Graph Examples:**")
                st.code("Explain the concept of machine learning")
                st.code("What are the benefits of cloud computing?")
                st.code("How does blockchain technology work?")
            
            elif graph_type == "conditional":
                st.markdown("**Conditional Graph Examples:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("*Questions:*")
                    st.code("What is quantum computing?")
                    st.code("How does AI work?")
                
                with col2:
                    st.markdown("*Tasks:*")
                    st.code("Help me write a Python script")
                    st.code("Create a marketing plan")
                
                with col3:
                    st.markdown("*Conversation:*")
                    st.code("Hello, how are you?")
                    st.code("Tell me a joke")
            
            else:  # cyclic
                st.markdown("**Cyclic Graph Examples (Iterative Improvement):**")
                st.code("Explain quantum entanglement")
                st.code("Write a comprehensive guide on REST APIs")
                st.code("Describe the process of photosynthesis in detail")
        
        # Graph flow visualization
        with st.expander("🔀 Graph Flow Visualization"):
            graph_type = st.session_state.get("current_graph_type", "simple")
            
            if graph_type == "simple":
                st.mermaid("""
                graph LR
                    A[User Input] --> B[Process]
                    B --> C[Analyze]
                    C --> D[Finalize]
                    D --> E[Response]
                """)
            
            elif graph_type == "conditional":
                st.mermaid("""
                graph TD
                    A[User Input] --> B[Classify]
                    B -->|Question| C[Question Handler]
                    B -->|Task| D[Task Handler]
                    B -->|Conversation| E[Conversation Handler]
                    C --> F[Response]
                    D --> F
                    E --> F
                """)
            
            else:  # cyclic
                st.mermaid("""
                graph TD
                    A[User Input] --> B[Generate]
                    B --> C[Review]
                    C -->|Needs Improvement| D[Improve]
                    D --> C
                    C -->|Satisfactory| E[Finalize]
                    E --> F[Response]
                """)


def main():
    """Main function to run the LangGraph chatbot"""
    chatbot = LangGraphChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
