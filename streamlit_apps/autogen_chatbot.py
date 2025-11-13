"""
AutoGen Chatbot with Azure OpenAI

Streamlit chatbot application using AutoGen with Azure OpenAI.
"""
import streamlit as st
import sys
import os
import io
import contextlib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.autogen.azure_autogen_setup import (
    AzureAutoGenSetup,
    create_coding_team,
    create_research_team,
    create_brainstorming_team
)
from utils import setup_logger

# Set up logging
logger = setup_logger(__name__)


class AutoGenChatbot(BaseChatbot):
    """AutoGen-based chatbot with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(
            title="AutoGen Chatbot with Azure OpenAI",
            description="""
            This chatbot demonstrates AutoGen integration with Azure OpenAI.
            Features automated multi-agent conversations with code execution capabilities.
            """
        )
        self.setup = None
        self.team = None
        self.initialize_team()
    
    def initialize_team(self):
        """Initialize AutoGen components"""
        if "autogen_setup" not in st.session_state:
            try:
                st.session_state.autogen_setup = AzureAutoGenSetup()
                st.session_state.current_team_type = "coding"
                st.session_state.current_team = create_coding_team(st.session_state.autogen_setup)
                logger.info("Initialized AutoGen with coding team")
            except Exception as e:
                logger.error(f"Failed to initialize AutoGen: {str(e)}")
                st.session_state.autogen_setup = None
                st.session_state.current_team = None
        
        self.setup = st.session_state.autogen_setup
        self.team = st.session_state.current_team
    
    def render_sidebar(self):
        """Render sidebar with AutoGen-specific settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("AutoGen Settings")
            
            # Team type selection
            team_type = st.selectbox(
                "Team Type",
                ["coding", "research", "brainstorming"],
                index=["coding", "research", "brainstorming"].index(
                    st.session_state.get("current_team_type", "coding")
                ),
                help="Select the type of team to use"
            )
            
            if team_type != st.session_state.get("current_team_type", "coding"):
                self.switch_team_type(team_type)
            
            # Team information
            st.caption("Current Team Composition:")
            if self.team and "agents" in self.team:
                for agent in self.team["agents"]:
                    st.caption(f"- {agent.name}")
            
            # Max rounds setting
            st.session_state.max_rounds = st.number_input(
                "Max Conversation Rounds",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum rounds of agent conversation"
            )
            
            # Code execution toggle
            st.session_state.allow_code_execution = st.checkbox(
                "Allow Code Execution",
                value=False,
                help="Allow agents to execute code (use with caution)"
            )
    
    def switch_team_type(self, team_type: str):
        """Switch between different team types"""
        try:
            if team_type == "coding":
                st.session_state.current_team = create_coding_team(self.setup)
            elif team_type == "research":
                st.session_state.current_team = create_research_team(self.setup)
            elif team_type == "brainstorming":
                st.session_state.current_team = create_brainstorming_team(self.setup)
            
            st.session_state.current_team_type = team_type
            self.team = st.session_state.current_team
            logger.info(f"Switched to {team_type} team")
        except Exception as e:
            st.error(f"Failed to switch team: {str(e)}")
            logger.error(f"Team switch error: {str(e)}")
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from AutoGen team
        
        Args:
            prompt: User prompt
        
        Returns:
            Team response
        """
        try:
            if self.team is None:
                return "Team not initialized. Please check your configuration."
            
            # Capture the conversation output
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                # Get the appropriate agents for conversation
                team_type = st.session_state.get("current_team_type", "coding")
                
                if team_type == "coding":
                    # Initiate chat between coder and executor
                    initiator = self.team["executor"]
                    recipient = self.team["coder"]
                elif team_type == "research":
                    # Initiate chat between coordinator and researcher
                    initiator = self.team["coordinator"]
                    recipient = self.team["researcher"]
                else:  # brainstorming
                    # Initiate chat between facilitator and creative
                    initiator = self.team["facilitator"]
                    recipient = self.team["creative"]
                
                # Start the conversation
                initiator.initiate_chat(
                    recipient,
                    message=prompt,
                    max_rounds=st.session_state.get("max_rounds", 5)
                )
            
            # Get the output
            conversation_output = output_buffer.getvalue()
            
            # Format and return the output
            if conversation_output:
                # Clean up the output
                lines = conversation_output.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Skip system messages and empty lines
                    if line.strip() and not line.startswith(">>>>>"):
                        cleaned_lines.append(line)
                
                return '\n'.join(cleaned_lines) if cleaned_lines else "No output generated"
            else:
                return "No response generated from the team"
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with examples"""
        super().render_header()
        
        # Show example queries based on team type
        with st.expander("Example Queries"):
            team_type = st.session_state.get("current_team_type", "coding")
            
            if team_type == "coding":
                st.markdown("**Coding Team Examples:**")
                st.code("Create a Python function to find prime numbers")
                st.code("Write a script to process CSV files")
                st.code("Implement a binary search algorithm")
                st.caption("The coding team will write, review, test, and execute code.")
            
            elif team_type == "research":
                st.markdown("**Research Team Examples:**")
                st.code("Research the impact of AI on education")
                st.code("Analyze trends in renewable energy")
                st.code("Investigate best practices for remote work")
                st.caption("The research team will gather, analyze, and document findings.")
            
            else:  # brainstorming
                st.markdown("**Brainstorming Team Examples:**")
                st.code("Ideas for a sustainable startup")
                st.code("Ways to improve team productivity")
                st.code("Creative marketing strategies for a new product")
                st.caption("The brainstorming team will generate, critique, and synthesize ideas.")
        
        # Team interaction visualization
        with st.expander("Team Interaction Flow"):
            team_type = st.session_state.get("current_team_type", "coding")
            
            if team_type == "coding":
                st.mermaid("""
                graph TB
                    A[User Input] --> B[Executor]
                    B <--> C[Coder]
                    C <--> D[Reviewer]
                    C <--> E[Tester]
                    B --> F[Final Output]
                    
                    style B fill:#f9f,stroke:#333,stroke-width:2px
                    style C fill:#bbf,stroke:#333,stroke-width:2px
                """)
            
            elif team_type == "research":
                st.mermaid("""
                graph TB
                    A[User Input] --> B[Coordinator]
                    B <--> C[Researcher]
                    C <--> D[Analyst]
                    D <--> E[Writer]
                    B --> F[Final Report]
                    
                    style B fill:#f9f,stroke:#333,stroke-width:2px
                    style C fill:#bbf,stroke:#333,stroke-width:2px
                """)
            
            else:  # brainstorming
                st.mermaid("""
                graph TB
                    A[User Input] --> B[Facilitator]
                    B <--> C[Creative]
                    C <--> D[Critic]
                    D <--> E[Synthesizer]
                    B --> F[Final Ideas]
                    
                    style B fill:#f9f,stroke:#333,stroke-width:2px
                    style C fill:#bbf,stroke:#333,stroke-width:2px
                """)


def main():
    """Main function to run the AutoGen chatbot"""
    chatbot = AutoGenChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
