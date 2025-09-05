"""
LangGraph Setup for Azure OpenAI

This module demonstrates how to set up and use LangGraph with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Optional, TypedDict, Annotated, Literal
from langchain_openai import AzureChatOpenAI
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver
import operator
from config import config
from utils import setup_logger, log_api_call

# Set up logging
logger = setup_logger(__name__)


# State type definitions
class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    context: Dict
    final_answer: Optional[str]


class AzureLangGraphSetup:
    """Azure OpenAI integrated LangGraph setup"""
    
    def __init__(self):
        """Initialize LangGraph with Azure OpenAI"""
        self.llm = self._create_llm()
        self.checkpointer = MemorySaver()
        logger.info("Initialized LangGraph with Azure OpenAI")
    
    def _create_llm(self) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance"""
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
    
    def create_simple_graph(self) -> StateGraph:
        """
        Create a simple linear graph
        
        Returns:
            StateGraph instance
        """
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Define nodes
        def process_input(state: AgentState) -> AgentState:
            """Process initial input"""
            logger.info("Processing input")
            messages = state["messages"]
            
            # Add system message if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content="You are a helpful AI assistant.")] + messages
            
            response = self.llm.invoke(messages)
            return {
                "messages": [response],
                "current_step": "analyze",
                "context": {"processed": True}
            }
        
        def analyze(state: AgentState) -> AgentState:
            """Analyze the response"""
            logger.info("Analyzing response")
            messages = state["messages"]
            
            analysis_prompt = HumanMessage(
                content="Provide a brief analysis of the conversation so far."
            )
            messages.append(analysis_prompt)
            
            response = self.llm.invoke(messages)
            return {
                "messages": [response],
                "current_step": "complete",
                "context": state["context"]
            }
        
        def finalize(state: AgentState) -> AgentState:
            """Finalize the response"""
            logger.info("Finalizing response")
            messages = state["messages"]
            
            # Get the final answer
            final_answer = messages[-1].content if messages else "No response generated"
            
            return {
                "messages": [],
                "current_step": "done",
                "final_answer": final_answer,
                "context": state["context"]
            }
        
        # Add nodes to graph
        workflow.add_node("process", process_input)
        workflow.add_node("analyze", analyze)
        workflow.add_node("finalize", finalize)
        
        # Define edges
        workflow.set_entry_point("process")
        workflow.add_edge("process", "analyze")
        workflow.add_edge("analyze", "finalize")
        workflow.add_edge("finalize", END)
        
        logger.info("Created simple graph")
        return workflow.compile(checkpointer=self.checkpointer)
    
    def create_conditional_graph(self) -> StateGraph:
        """
        Create a graph with conditional routing
        
        Returns:
            StateGraph instance with conditional logic
        """
        workflow = StateGraph(AgentState)
        
        def classify_input(state: AgentState) -> AgentState:
            """Classify the input type"""
            logger.info("Classifying input")
            messages = state["messages"]
            
            classification_prompt = [
                SystemMessage(content="Classify the user input as either 'question', 'task', or 'conversation'."),
                *messages
            ]
            
            response = self.llm.invoke(classification_prompt)
            classification = response.content.lower()
            
            if "question" in classification:
                input_type = "question"
            elif "task" in classification:
                input_type = "task"
            else:
                input_type = "conversation"
            
            return {
                "messages": [response],
                "current_step": input_type,
                "context": {"input_type": input_type}
            }
        
        def handle_question(state: AgentState) -> AgentState:
            """Handle question inputs"""
            logger.info("Handling question")
            messages = state["messages"]
            
            question_prompt = [
                SystemMessage(content="You are answering a question. Provide a clear, informative response."),
                *messages
            ]
            
            response = self.llm.invoke(question_prompt)
            return {
                "messages": [response],
                "current_step": "complete",
                "final_answer": response.content,
                "context": state["context"]
            }
        
        def handle_task(state: AgentState) -> AgentState:
            """Handle task inputs"""
            logger.info("Handling task")
            messages = state["messages"]
            
            task_prompt = [
                SystemMessage(content="You are helping with a task. Break it down into steps and provide guidance."),
                *messages
            ]
            
            response = self.llm.invoke(task_prompt)
            return {
                "messages": [response],
                "current_step": "complete",
                "final_answer": response.content,
                "context": state["context"]
            }
        
        def handle_conversation(state: AgentState) -> AgentState:
            """Handle conversational inputs"""
            logger.info("Handling conversation")
            messages = state["messages"]
            
            conv_prompt = [
                SystemMessage(content="You are having a friendly conversation. Respond naturally and engagingly."),
                *messages
            ]
            
            response = self.llm.invoke(conv_prompt)
            return {
                "messages": [response],
                "current_step": "complete",
                "final_answer": response.content,
                "context": state["context"]
            }
        
        def route_based_on_classification(state: AgentState) -> str:
            """Route to appropriate handler based on classification"""
            input_type = state["context"].get("input_type", "conversation")
            logger.info(f"Routing to {input_type} handler")
            return input_type
        
        # Add nodes
        workflow.add_node("classify", classify_input)
        workflow.add_node("question", handle_question)
        workflow.add_node("task", handle_task)
        workflow.add_node("conversation", handle_conversation)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify",
            route_based_on_classification,
            {
                "question": "question",
                "task": "task",
                "conversation": "conversation"
            }
        )
        
        # Add edges to END
        workflow.add_edge("question", END)
        workflow.add_edge("task", END)
        workflow.add_edge("conversation", END)
        
        logger.info("Created conditional graph")
        return workflow.compile(checkpointer=self.checkpointer)
    
    def create_cyclic_graph(self) -> StateGraph:
        """
        Create a graph with cycles for iterative processing
        
        Returns:
            StateGraph instance with cycles
        """
        workflow = StateGraph(AgentState)
        
        def generate_response(state: AgentState) -> AgentState:
            """Generate initial response"""
            logger.info("Generating response")
            messages = state["messages"]
            
            response = self.llm.invoke(messages)
            
            return {
                "messages": [response],
                "current_step": "review",
                "context": {"iteration": state["context"].get("iteration", 0) + 1}
            }
        
        def review_response(state: AgentState) -> AgentState:
            """Review and potentially improve response"""
            logger.info("Reviewing response")
            messages = state["messages"]
            iteration = state["context"].get("iteration", 1)
            
            review_prompt = HumanMessage(
                content=f"Review this response (iteration {iteration}). Is it complete and accurate? Answer 'yes' or 'needs improvement'."
            )
            messages.append(review_prompt)
            
            response = self.llm.invoke(messages)
            
            needs_improvement = "improvement" in response.content.lower()
            
            return {
                "messages": [response],
                "current_step": "improve" if needs_improvement else "finalize",
                "context": {**state["context"], "needs_improvement": needs_improvement}
            }
        
        def improve_response(state: AgentState) -> AgentState:
            """Improve the response"""
            logger.info("Improving response")
            messages = state["messages"]
            
            improve_prompt = HumanMessage(
                content="Please improve the previous response with more detail or clarity."
            )
            messages.append(improve_prompt)
            
            response = self.llm.invoke(messages)
            
            return {
                "messages": [response],
                "current_step": "review",
                "context": state["context"]
            }
        
        def finalize_response(state: AgentState) -> AgentState:
            """Finalize the response"""
            logger.info("Finalizing response")
            final_answer = state["messages"][-1].content if state["messages"] else ""
            
            return {
                "messages": [],
                "current_step": "done",
                "final_answer": final_answer,
                "context": state["context"]
            }
        
        def should_continue(state: AgentState) -> str:
            """Decide whether to continue improving"""
            iteration = state["context"].get("iteration", 0)
            needs_improvement = state["context"].get("needs_improvement", False)
            
            # Max 3 iterations
            if iteration >= 3:
                return "finalize"
            
            return "improve" if needs_improvement else "finalize"
        
        # Add nodes
        workflow.add_node("generate", generate_response)
        workflow.add_node("review", review_response)
        workflow.add_node("improve", improve_response)
        workflow.add_node("finalize", finalize_response)
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        # Add edges
        workflow.add_edge("generate", "review")
        workflow.add_conditional_edges(
            "review",
            should_continue,
            {
                "improve": "improve",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("improve", "review")  # Cycle back to review
        workflow.add_edge("finalize", END)
        
        logger.info("Created cyclic graph")
        return workflow.compile(checkpointer=self.checkpointer)


def run_graph_example(graph: StateGraph, user_input: str) -> str:
    """
    Run a graph with user input
    
    Args:
        graph: Compiled StateGraph
        user_input: User message
    
    Returns:
        Final response
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_step": "start",
        "context": {},
        "final_answer": None
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result.get("final_answer", "No response generated")


if __name__ == "__main__":
    # Example usage
    setup = AzureLangGraphSetup()
    
    # Test simple graph
    print("Testing Simple Graph:")
    simple_graph = setup.create_simple_graph()
    result = run_graph_example(simple_graph, "Tell me about the benefits of cloud computing")
    print(f"Simple Graph Result: {result}\n")
    
    # Test conditional graph
    print("Testing Conditional Graph:")
    conditional_graph = setup.create_conditional_graph()
    result = run_graph_example(conditional_graph, "What is machine learning?")
    print(f"Conditional Graph Result: {result}\n")
    
    # Test cyclic graph
    print("Testing Cyclic Graph:")
    cyclic_graph = setup.create_cyclic_graph()
    result = run_graph_example(cyclic_graph, "Explain quantum computing")
    print(f"Cyclic Graph Result: {result}")
