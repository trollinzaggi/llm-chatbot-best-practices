"""
AutoGen Setup for Azure OpenAI

This module demonstrates how to set up and use AutoGen with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Optional, Any, Union
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import config
from utils import setup_logger, log_api_call

# Set up logging
logger = setup_logger(__name__)


class AzureAutoGenSetup:
    """Azure OpenAI integrated AutoGen setup"""
    
    def __init__(self):
        """Initialize AutoGen with Azure OpenAI"""
        self.llm_config = self._create_llm_config()
        self.agents = []
        logger.info("Initialized AutoGen with Azure OpenAI")
    
    def _create_llm_config(self) -> Dict[str, Any]:
        """Create LLM configuration for AutoGen with Azure OpenAI"""
        llm_config = {
            "config_list": [
                {
                    "model": config.deployment_name,
                    "api_type": "azure",
                    "api_key": config.api_key,
                    "api_base": config.endpoint,
                    "api_version": config.api_version,
                }
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout_seconds,
            "cache_seed": 42  # For reproducibility
        }
        
        logger.info("Created LLM configuration for AutoGen")
        return llm_config
    
    def create_assistant_agent(
        self,
        name: str,
        system_message: str,
        max_consecutive_auto_reply: int = 10,
        human_input_mode: str = "NEVER",
        code_execution_config: Optional[Dict] = None
    ) -> AssistantAgent:
        """
        Create an AutoGen assistant agent
        
        Args:
            name: Agent name
            system_message: System message defining agent behavior
            max_consecutive_auto_reply: Maximum consecutive automated replies
            human_input_mode: When to request human input
            code_execution_config: Configuration for code execution
        
        Returns:
            AssistantAgent instance
        """
        agent = AssistantAgent(
            name=name,
            llm_config=self.llm_config,
            system_message=system_message,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config or False
        )
        
        self.agents.append(agent)
        logger.info(f"Created assistant agent: {name}")
        return agent
    
    def create_user_proxy_agent(
        self,
        name: str,
        system_message: str = "",
        max_consecutive_auto_reply: int = 10,
        human_input_mode: str = "TERMINATE",
        code_execution_config: Optional[Dict] = None,
        is_termination_msg: Optional[Any] = None
    ) -> UserProxyAgent:
        """
        Create an AutoGen user proxy agent
        
        Args:
            name: Agent name
            system_message: System message
            max_consecutive_auto_reply: Maximum consecutive automated replies
            human_input_mode: When to request human input
            code_execution_config: Configuration for code execution
            is_termination_msg: Function to determine if conversation should terminate
        
        Returns:
            UserProxyAgent instance
        """
        # Default termination function
        if is_termination_msg is None:
            is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
        
        agent = UserProxyAgent(
            name=name,
            system_message=system_message,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config or {
                "work_dir": "coding",
                "use_docker": False
            },
            is_termination_msg=is_termination_msg
        )
        
        self.agents.append(agent)
        logger.info(f"Created user proxy agent: {name}")
        return agent
    
    def create_group_chat(
        self,
        agents: List,
        max_round: int = 10,
        admin_name: str = "Admin"
    ) -> GroupChatManager:
        """
        Create a group chat with multiple agents
        
        Args:
            agents: List of agents to participate in group chat
            max_round: Maximum number of conversation rounds
            admin_name: Name of the group chat manager
        
        Returns:
            GroupChatManager instance
        """
        # Create group chat
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round
        )
        
        # Create manager
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
            name=admin_name
        )
        
        logger.info(f"Created group chat with {len(agents)} agents")
        return manager


def create_coding_team(setup: AzureAutoGenSetup) -> Dict[str, Any]:
    """Create a coding team with AutoGen"""
    
    # Create coder agent
    coder = setup.create_assistant_agent(
        name="Coder",
        system_message="""You are a skilled Python programmer. You write clean, efficient code.
        When asked to solve a problem:
        1. Understand the requirements
        2. Write Python code to solve it
        3. Include proper error handling
        4. Add comments for clarity
        Reply 'TERMINATE' when the task is complete."""
    )
    
    # Create reviewer agent
    reviewer = setup.create_assistant_agent(
        name="Reviewer",
        system_message="""You are a code reviewer. You:
        1. Review code for bugs and improvements
        2. Check for best practices
        3. Suggest optimizations
        4. Ensure code is readable and maintainable
        Provide constructive feedback."""
    )
    
    # Create tester agent
    tester = setup.create_assistant_agent(
        name="Tester",
        system_message="""You are a QA engineer. You:
        1. Write test cases for the code
        2. Identify edge cases
        3. Verify the code works as expected
        4. Report any issues found"""
    )
    
    # Create user proxy for execution
    executor = setup.create_user_proxy_agent(
        name="Executor",
        system_message="Execute the code and report results.",
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    return {
        "coder": coder,
        "reviewer": reviewer,
        "tester": tester,
        "executor": executor,
        "agents": [coder, reviewer, tester, executor]
    }


def create_research_team(setup: AzureAutoGenSetup) -> Dict[str, Any]:
    """Create a research team with AutoGen"""
    
    # Create researcher agent
    researcher = setup.create_assistant_agent(
        name="Researcher",
        system_message="""You are a research specialist. You:
        1. Gather information on topics
        2. Analyze data and trends
        3. Identify key insights
        4. Provide evidence-based findings
        Be thorough and cite sources when possible."""
    )
    
    # Create analyst agent
    analyst = setup.create_assistant_agent(
        name="Analyst",
        system_message="""You are a data analyst. You:
        1. Analyze research findings
        2. Identify patterns and correlations
        3. Create summaries and reports
        4. Provide actionable recommendations"""
    )
    
    # Create writer agent
    writer = setup.create_assistant_agent(
        name="Writer",
        system_message="""You are a technical writer. You:
        1. Transform research into clear documentation
        2. Create well-structured reports
        3. Ensure accuracy and clarity
        4. Make complex topics accessible
        Reply 'TERMINATE' when the document is complete."""
    )
    
    # Create coordinator
    coordinator = setup.create_user_proxy_agent(
        name="Coordinator",
        system_message="Coordinate the research team's efforts.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )
    
    return {
        "researcher": researcher,
        "analyst": analyst,
        "writer": writer,
        "coordinator": coordinator,
        "agents": [researcher, analyst, writer, coordinator]
    }


def create_brainstorming_team(setup: AzureAutoGenSetup) -> Dict[str, Any]:
    """Create a brainstorming team with AutoGen"""
    
    # Create creative agent
    creative = setup.create_assistant_agent(
        name="Creative",
        system_message="""You are a creative thinker. You:
        1. Generate innovative ideas
        2. Think outside the box
        3. Propose unique solutions
        4. Build on others' ideas creatively
        Be bold and imaginative."""
    )
    
    # Create critic agent
    critic = setup.create_assistant_agent(
        name="Critic",
        system_message="""You are a constructive critic. You:
        1. Evaluate ideas objectively
        2. Identify potential challenges
        3. Suggest improvements
        4. Ensure feasibility
        Be constructive and helpful."""
    )
    
    # Create synthesizer agent
    synthesizer = setup.create_assistant_agent(
        name="Synthesizer",
        system_message="""You are a synthesizer. You:
        1. Combine the best ideas
        2. Create cohesive solutions
        3. Resolve conflicts between ideas
        4. Produce final recommendations
        Reply 'TERMINATE' when synthesis is complete."""
    )
    
    # Create facilitator
    facilitator = setup.create_user_proxy_agent(
        name="Facilitator",
        system_message="Facilitate the brainstorming session.",
        human_input_mode="NEVER"
    )
    
    return {
        "creative": creative,
        "critic": critic,
        "synthesizer": synthesizer,
        "facilitator": facilitator,
        "agents": [creative, critic, synthesizer, facilitator]
    }


def run_two_agent_chat(agent1: AssistantAgent, agent2: UserProxyAgent, message: str):
    """Run a conversation between two agents"""
    logger.info(f"Starting conversation between {agent1.name} and {agent2.name}")
    agent2.initiate_chat(agent1, message=message)


def run_group_chat(manager: GroupChatManager, initial_message: str):
    """Run a group chat"""
    logger.info("Starting group chat")
    # Initiate with the first agent in the group
    first_agent = manager.groupchat.agents[0]
    first_agent.initiate_chat(manager, message=initial_message)


if __name__ == "__main__":
    # Example usage
    setup = AzureAutoGenSetup()
    
    # Example 1: Simple two-agent conversation
    print("=== Two-Agent Conversation ===")
    assistant = setup.create_assistant_agent(
        name="Assistant",
        system_message="You are a helpful AI assistant. Help the user with their requests."
    )
    
    user = setup.create_user_proxy_agent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )
    
    run_two_agent_chat(
        assistant,
        user,
        "Write a Python function to calculate fibonacci numbers"
    )
    
    print("\n=== Coding Team ===")
    # Example 2: Coding team
    coding_team = create_coding_team(setup)
    manager = setup.create_group_chat(coding_team["agents"])
    run_group_chat(manager, "Create a Python function to find prime numbers up to n")
    
    print("\n=== Research Team ===")
    # Example 3: Research team
    research_team = create_research_team(setup)
    manager = setup.create_group_chat(research_team["agents"])
    run_group_chat(manager, "Research the impact of AI on healthcare")
