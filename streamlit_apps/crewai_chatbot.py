"""
CrewAI Chatbot with Azure OpenAI

Streamlit chatbot application using CrewAI with Azure OpenAI.
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.crewai.azure_crewai_setup import (
    AzureCrewAISetup,
    create_research_crew,
    create_development_crew,
    create_marketing_crew
)
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class CrewAIChatbot(BaseChatbot):
    """CrewAI-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="👥 CrewAI Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates CrewAI integration with Azure OpenAI.
            Features multi-agent collaboration with specialized roles working together to solve complex tasks.
            """
        )
        self.setup = None
        self.crew = None
        self.initialize_crew()
    
    def initialize_crew(self):
        """Initialize CrewAI components"""
        if "crewai_setup" not in st.session_state:
            try:
                st.session_state.crewai_setup = AzureCrewAISetup()
                st.session_state.current_crew_type = "research"
                st.session_state.current_crew = create_research_crew(st.session_state.crewai_setup)
                logger.info("Initialized CrewAI with research crew")
            except Exception as e:
                logger.error(f"Failed to initialize CrewAI: {str(e)}")
                st.session_state.crewai_setup = None
                st.session_state.current_crew = None
        
        self.setup = st.session_state.crewai_setup
        self.crew = st.session_state.current_crew
    
    def render_sidebar(self):
        """Render sidebar with CrewAI-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("👥 CrewAI Settings")
            
            # Crew type selection
            crew_type = st.selectbox(
                "Crew Type",
                ["research", "development", "marketing"],
                index=["research", "development", "marketing"].index(
                    st.session_state.get("current_crew_type", "research")
                ),
                help="Select the type of crew to use"
            )
            
            if crew_type != st.session_state.get("current_crew_type", "research"):
                self.switch_crew_type(crew_type)
            
            # Crew information
            st.caption("Current Crew Composition:")
            if self.crew:
                for agent in self.crew.agents:
                    st.caption(f"• {agent.role}")
            
            # Task information
            st.caption("\nCrew Tasks:")
            if self.crew:
                for i, task in enumerate(self.crew.tasks, 1):
                    st.caption(f"{i}. {task.description[:50]}...")
            
            # Process type
            st.caption(f"\nProcess: {self.crew.process if self.crew else 'N/A'}")
    
    def switch_crew_type(self, crew_type: str):
        """Switch between different crew types"""
        try:
            if crew_type == "research":
                st.session_state.current_crew = create_research_crew(self.setup)
            elif crew_type == "development":
                st.session_state.current_crew = create_development_crew(self.setup)
            elif crew_type == "marketing":
                st.session_state.current_crew = create_marketing_crew(self.setup)
            
            st.session_state.current_crew_type = crew_type
            self.crew = st.session_state.current_crew
            logger.info(f"Switched to {crew_type} crew")
        except Exception as e:
            st.error(f"Failed to switch crew: {str(e)}")
            logger.error(f"Crew switch error: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from CrewAI crew
        
        Args:
            prompt: User prompt
        
        Returns:
            Crew response
        """
        try:
            if self.crew is None:
                return "❌ Crew not initialized. Please check your configuration."
            
            # Prepare inputs based on crew type
            crew_type = st.session_state.get("current_crew_type", "research")
            
            if crew_type == "research":
                inputs = {"topic": prompt}
            elif crew_type == "development":
                inputs = {"project_type": prompt}
            elif crew_type == "marketing":
                inputs = {"product": prompt}
            else:
                inputs = {"input": prompt}
            
            # Show progress
            with st.spinner("Crew working on your request..."):
                # Run the crew
                result = self.crew.kickoff(inputs=inputs)
            
            # Format the result
            if hasattr(result, 'raw_output'):
                return result.raw_output
            else:
                return str(result)
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries based on crew type
        with st.expander("💡 Example Queries"):
            crew_type = st.session_state.get("current_crew_type", "research")
            
            if crew_type == "research":
                st.markdown("**Research Crew Examples:**")
                st.code("Artificial Intelligence in Healthcare")
                st.code("Climate Change Solutions")
                st.code("Future of Remote Work")
                st.caption("The research crew will research, write, and edit content about your topic.")
            
            elif crew_type == "development":
                st.markdown("**Development Crew Examples:**")
                st.code("E-commerce Platform")
                st.code("REST API for Task Management")
                st.code("Real-time Chat Application")
                st.caption("The development crew will design, implement, and test your project.")
            
            else:  # marketing
                st.markdown("**Marketing Crew Examples:**")
                st.code("AI-powered Writing Assistant")
                st.code("Sustainable Coffee Brand")
                st.code("Online Learning Platform")
                st.caption("The marketing crew will analyze the market, create strategy, and develop content.")
        
        # Crew workflow visualization
        with st.expander("👥 Crew Workflow"):
            crew_type = st.session_state.get("current_crew_type", "research")
            
            if crew_type == "research":
                st.mermaid("""
                graph TD
                    A[User Input] --> B[Research Analyst]
                    B --> C[Content Writer]
                    C --> D[Content Editor]
                    D --> E[Final Output]
                    
                    B -.->|Research Findings| C
                    C -.->|Draft Content| D
                """)
            
            elif crew_type == "development":
                st.mermaid("""
                graph TD
                    A[User Input] --> B[Software Architect]
                    B --> C[Software Developer]
                    C --> D[QA Engineer]
                    D --> E[Final Output]
                    
                    B -.->|Architecture Design| C
                    C -.->|Implementation| D
                """)
            
            else:  # marketing
                st.mermaid("""
                graph TD
                    A[User Input] --> B[Market Analyst]
                    B --> C[Marketing Strategist]
                    C --> D[Content Creator]
                    D --> E[Final Output]
                    
                    B -.->|Market Analysis| C
                    C -.->|Strategy| D
                """)


def main():
    """Main function to run the CrewAI chatbot"""
    chatbot = CrewAIChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
