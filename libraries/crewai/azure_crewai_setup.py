"""
CrewAI Setup for Azure OpenAI

This module demonstrates how to set up and use CrewAI with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Optional, Any
from crewai import Agent, Task, Crew, Process
from crewai.agent import Agent
from crewai.task import Task
from langchain_openai import AzureChatOpenAI
from config import config
from utils import setup_logger, log_api_call

# Set up logging
logger = setup_logger(__name__)


class AzureCrewAISetup:
    """Azure OpenAI integrated CrewAI setup"""
    
    def __init__(self):
        """Initialize CrewAI with Azure OpenAI"""
        self.llm = self._create_llm()
        self.agents = []
        self.tasks = []
        logger.info("Initialized CrewAI with Azure OpenAI")
    
    def _create_llm(self) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance for CrewAI"""
        return AzureChatOpenAI(
            azure_deployment=config.deployment_name,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries
        )
    
    def create_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List] = None,
        verbose: bool = True,
        allow_delegation: bool = False
    ) -> Agent:
        """
        Create a CrewAI agent with Azure OpenAI
        
        Args:
            role: Agent's role
            goal: Agent's goal
            backstory: Agent's backstory
            tools: Optional list of tools for the agent
            verbose: Whether to log agent actions
            allow_delegation: Whether agent can delegate tasks
        
        Returns:
            CrewAI Agent instance
        """
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            tools=tools or [],
            verbose=verbose,
            allow_delegation=allow_delegation
        )
        
        self.agents.append(agent)
        logger.info(f"Created agent: {role}")
        return agent
    
    def create_task(
        self,
        description: str,
        agent: Agent,
        expected_output: str,
        tools: Optional[List] = None,
        context: Optional[List[Task]] = None
    ) -> Task:
        """
        Create a CrewAI task
        
        Args:
            description: Task description
            agent: Agent to perform the task
            expected_output: Expected output description
            tools: Optional tools for the task
            context: Optional context tasks
        
        Returns:
            CrewAI Task instance
        """
        task = Task(
            description=description,
            agent=agent,
            expected_output=expected_output,
            tools=tools or [],
            context=context or []
        )
        
        self.tasks.append(task)
        logger.info(f"Created task for agent: {agent.role}")
        return task
    
    def create_crew(
        self,
        agents: Optional[List[Agent]] = None,
        tasks: Optional[List[Task]] = None,
        process: Process = Process.sequential,
        verbose: bool = True
    ) -> Crew:
        """
        Create a CrewAI crew
        
        Args:
            agents: List of agents (uses self.agents if None)
            tasks: List of tasks (uses self.tasks if None)
            process: Process type (sequential or hierarchical)
            verbose: Whether to log crew actions
        
        Returns:
            CrewAI Crew instance
        """
        crew = Crew(
            agents=agents or self.agents,
            tasks=tasks or self.tasks,
            process=process,
            verbose=verbose
        )
        
        logger.info(f"Created crew with {len(crew.agents)} agents and {len(crew.tasks)} tasks")
        return crew


def create_research_crew(setup: AzureCrewAISetup) -> Crew:
    """Create an example research crew"""
    
    # Create researcher agent
    researcher = setup.create_agent(
        role="Research Analyst",
        goal="Conduct thorough research on the given topic",
        backstory="""You are an experienced research analyst with expertise in 
        gathering and analyzing information from various sources. You excel at 
        finding relevant data and identifying key insights.""",
        allow_delegation=False
    )
    
    # Create writer agent
    writer = setup.create_agent(
        role="Content Writer",
        goal="Create compelling content based on research findings",
        backstory="""You are a skilled content writer who can transform research 
        findings into engaging and informative content. You have a talent for 
        making complex topics accessible to a general audience.""",
        allow_delegation=False
    )
    
    # Create editor agent
    editor = setup.create_agent(
        role="Content Editor",
        goal="Review and refine content for clarity and accuracy",
        backstory="""You are a meticulous editor with an eye for detail. You ensure 
        that all content is accurate, well-structured, and free of errors. You also 
        verify that the content meets the intended objectives.""",
        allow_delegation=True
    )
    
    # Create tasks
    research_task = setup.create_task(
        description="Research the topic of '{topic}' and identify key facts, trends, and insights",
        agent=researcher,
        expected_output="A comprehensive research summary with key findings and data points"
    )
    
    writing_task = setup.create_task(
        description="Write an informative article based on the research findings about '{topic}'",
        agent=writer,
        expected_output="A well-written article that is informative and engaging",
        context=[research_task]
    )
    
    editing_task = setup.create_task(
        description="Review and edit the article for clarity, accuracy, and engagement",
        agent=editor,
        expected_output="A polished, publication-ready article",
        context=[writing_task]
    )
    
    # Create and return crew
    return setup.create_crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential
    )


def create_development_crew(setup: AzureCrewAISetup) -> Crew:
    """Create an example software development crew"""
    
    # Create architect agent
    architect = setup.create_agent(
        role="Software Architect",
        goal="Design robust and scalable software architecture",
        backstory="""You are a senior software architect with years of experience 
        designing enterprise-level systems. You focus on creating scalable, 
        maintainable, and efficient architectures.""",
        allow_delegation=True
    )
    
    # Create developer agent
    developer = setup.create_agent(
        role="Software Developer",
        goal="Implement software solutions based on architectural designs",
        backstory="""You are a skilled full-stack developer proficient in multiple 
        programming languages and frameworks. You write clean, efficient code 
        following best practices.""",
        allow_delegation=False
    )
    
    # Create tester agent
    tester = setup.create_agent(
        role="QA Engineer",
        goal="Ensure software quality through comprehensive testing",
        backstory="""You are a detail-oriented QA engineer who specializes in 
        finding bugs and ensuring software reliability. You design comprehensive 
        test cases and validate functionality.""",
        allow_delegation=False
    )
    
    # Create tasks
    design_task = setup.create_task(
        description="Design the architecture for a {project_type} application",
        agent=architect,
        expected_output="A detailed architectural design document with components, interfaces, and data flow"
    )
    
    implementation_task = setup.create_task(
        description="Implement the core functionality based on the architectural design",
        agent=developer,
        expected_output="Working code implementation with documentation",
        context=[design_task]
    )
    
    testing_task = setup.create_task(
        description="Create and execute test cases for the implemented functionality",
        agent=tester,
        expected_output="Test report with results and any identified issues",
        context=[implementation_task]
    )
    
    # Create and return crew
    return setup.create_crew(
        agents=[architect, developer, tester],
        tasks=[design_task, implementation_task, testing_task],
        process=Process.sequential
    )


def create_marketing_crew(setup: AzureCrewAISetup) -> Crew:
    """Create an example marketing crew"""
    
    # Create market analyst agent
    analyst = setup.create_agent(
        role="Market Analyst",
        goal="Analyze market trends and competitive landscape",
        backstory="""You are a data-driven market analyst with expertise in 
        identifying market opportunities and analyzing competitor strategies. 
        You provide insights that drive strategic decisions.""",
        allow_delegation=False
    )
    
    # Create strategist agent
    strategist = setup.create_agent(
        role="Marketing Strategist",
        goal="Develop comprehensive marketing strategies",
        backstory="""You are a creative marketing strategist who develops 
        innovative campaigns that resonate with target audiences. You blend 
        creativity with data-driven insights.""",
        allow_delegation=True
    )
    
    # Create content creator agent
    content_creator = setup.create_agent(
        role="Content Creator",
        goal="Create engaging marketing content",
        backstory="""You are a versatile content creator who produces compelling 
        content across various formats and channels. You understand how to craft 
        messages that engage and convert.""",
        allow_delegation=False
    )
    
    # Create tasks
    analysis_task = setup.create_task(
        description="Analyze the market for {product} and identify target audience and opportunities",
        agent=analyst,
        expected_output="Market analysis report with target audience insights and opportunities"
    )
    
    strategy_task = setup.create_task(
        description="Develop a marketing strategy based on the market analysis",
        agent=strategist,
        expected_output="Comprehensive marketing strategy document",
        context=[analysis_task]
    )
    
    content_task = setup.create_task(
        description="Create marketing content based on the strategy",
        agent=content_creator,
        expected_output="Marketing content package including copy and content plan",
        context=[strategy_task]
    )
    
    # Create and return crew
    return setup.create_crew(
        agents=[analyst, strategist, content_creator],
        tasks=[analysis_task, strategy_task, content_task],
        process=Process.sequential
    )


if __name__ == "__main__":
    # Example usage
    setup = AzureCrewAISetup()
    
    # Create and run research crew
    print("Creating Research Crew...")
    research_crew = create_research_crew(setup)
    
    # Kickoff the crew with input
    result = research_crew.kickoff(inputs={"topic": "Artificial Intelligence in Healthcare"})
    print(f"Research Crew Result:\n{result}\n")
    
    # Create and run development crew
    print("Creating Development Crew...")
    dev_crew = create_development_crew(setup)
    
    result = dev_crew.kickoff(inputs={"project_type": "REST API"})
    print(f"Development Crew Result:\n{result}\n")
