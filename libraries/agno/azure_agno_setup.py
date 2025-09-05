"""
Agno Library Setup for Azure OpenAI

This module demonstrates how to set up and use Agno with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Any, Optional
from agno import Agent, Tool, Message
from openai import AzureOpenAI
from config import config
from utils import setup_logger, log_api_call, retry_with_exponential_backoff

# Set up logging
logger = setup_logger(__name__)


class AzureAgnoAgent:
    """Azure OpenAI integrated Agno Agent"""
    
    def __init__(self, name: str = "Azure Agent", system_prompt: Optional[str] = None):
        """
        Initialize Agno agent with Azure OpenAI
        
        Args:
            name: Agent name
            system_prompt: Optional system prompt for the agent
        """
        self.name = name
        self.system_prompt = system_prompt or "You are a helpful AI assistant powered by Azure OpenAI."
        self.client = config.get_client()
        self.deployment = config.deployment_name
        self.tools = []
        
        logger.info(f"Initialized Agno agent: {name}")
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        logger.info(f"Added tool: {tool.name}")
    
    @retry_with_exponential_backoff(max_retries=3)
    def chat(self, message: str, use_tools: bool = True) -> str:
        """
        Send a message to the agent and get response
        
        Args:
            message: User message
            use_tools: Whether to use tools if available
        
        Returns:
            Agent response
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Prepare tools if available and requested
            tools_param = None
            if use_tools and self.tools:
                tools_param = [self._tool_to_function(tool) for tool in self.tools]
            
            log_api_call(
                logger,
                "Azure OpenAI",
                "chat.completions.create",
                deployment=self.deployment,
                with_tools=bool(tools_param)
            )
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools_param if tools_param else None,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Handle tool calls if present
            if response.choices[0].message.tool_calls and use_tools:
                return self._handle_tool_calls(
                    response.choices[0].message,
                    messages
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise
    
    def _tool_to_function(self, tool: Tool) -> Dict:
        """Convert Agno tool to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters if hasattr(tool, 'parameters') else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def _handle_tool_calls(self, message: Any, messages: List[Dict]) -> str:
        """Handle tool calls from the model"""
        tool_results = []
        
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    # Execute tool
                    result = tool.execute(tool_call.function.arguments)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result)
                    })
                    logger.info(f"Executed tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: {str(e)}"
                    })
        
        # Get final response with tool results
        messages.append(message.model_dump())
        messages.extend(tool_results)
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        return response.choices[0].message.content


# Example tools
class CalculatorTool(Tool):
    """Example calculator tool"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, args: str) -> str:
        """Execute the calculator tool"""
        import json
        try:
            params = json.loads(args)
            expression = params.get("expression", "")
            # Safety check - only allow basic math operations
            allowed_chars = "0123456789+-*/()., "
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"The result of {expression} is {result}"
            else:
                return "Invalid expression. Only basic math operations are allowed."
        except Exception as e:
            return f"Error calculating: {str(e)}"


class WeatherTool(Tool):
    """Example weather tool (mock implementation)"""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a location"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }
    
    def execute(self, args: str) -> str:
        """Execute the weather tool (mock implementation)"""
        import json
        import random
        try:
            params = json.loads(args)
            location = params.get("location", "Unknown")
            # Mock weather data
            temp = random.randint(60, 85)
            conditions = random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"])
            return f"Weather in {location}: {temp}°F, {conditions}"
        except Exception as e:
            return f"Error getting weather: {str(e)}"


def create_agent_with_tools() -> AzureAgnoAgent:
    """Create an example agent with tools"""
    agent = AzureAgnoAgent(
        name="Assistant with Tools",
        system_prompt="You are a helpful assistant with access to calculator and weather tools."
    )
    
    # Add tools
    agent.add_tool(CalculatorTool())
    agent.add_tool(WeatherTool())
    
    return agent


if __name__ == "__main__":
    # Example usage
    agent = create_agent_with_tools()
    
    # Test without tools
    response = agent.chat("Hello! What can you do?", use_tools=False)
    print(f"Response: {response}\n")
    
    # Test with tools
    response = agent.chat("What's 15 * 23?", use_tools=True)
    print(f"Calculator response: {response}\n")
    
    response = agent.chat("What's the weather in New York?", use_tools=True)
    print(f"Weather response: {response}")
