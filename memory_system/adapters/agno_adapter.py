"""
Agno framework memory adapter.

This module provides memory integration for the Agno framework,
which doesn't have built-in memory support.
"""

from typing import List, Dict, Optional, Any
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework


class AgnoMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for Agno framework.
    
    Since Agno doesn't have built-in memory, this adapter provides
    complete memory management by wrapping Agno agent interactions.
    """
    
    def __init__(self, agent=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Agno memory adapter.
        
        Args:
            agent: Agno agent instance
            config: Configuration dictionary
        """
        self.agent = agent
        super().__init__(config)
        
        # Track tool usage
        self.tool_history = []
        self.context_variables = {}
    
    def _initialize_framework(self) -> None:
        """Initialize Agno-specific components."""
        self.session_memory.conversation.framework = Framework.AGNO
        
        # Initialize Agno-specific tracking
        self.tool_usage_stats = {}
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0
    
    def inject_memory_context(self, input_text: str, 
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context into Agno agent input.
        
        Args:
            input_text: Original input text
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input with memory context
        """
        # Build context from session memory
        context_messages = self.session_memory.get_messages(limit=max_context_messages)
        
        if not context_messages:
            return input_text
        
        # Format context for Agno
        context_parts = ["Previous conversation context:"]
        
        for msg in context_messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
        
        # Add tool usage history if available
        if self.tool_history:
            context_parts.append("\nRecent tool usage:")
            for tool_use in self.tool_history[-3:]:
                context_parts.append(f"- {tool_use['tool']}: {tool_use['result'][:100]}")
        
        # Add known context variables
        if self.context_variables:
            context_parts.append("\nKnown information:")
            for key, value in list(self.context_variables.items())[:5]:
                context_parts.append(f"- {key}: {value}")
        
        # Combine context with input
        full_context = "\n".join(context_parts)
        return f"{full_context}\n\nCurrent request: {input_text}"
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process Agno response and extract relevant information.
        
        Args:
            response: Agent response
            metadata: Optional metadata including tool usage
            
        Returns:
            Processed response
        """
        # Extract tool usage from metadata
        if metadata and 'tools_used' in metadata:
            for tool in metadata['tools_used']:
                self.tool_history.append({
                    'tool': tool.get('name', 'unknown'),
                    'args': tool.get('arguments', {}),
                    'result': tool.get('result', ''),
                    'timestamp': self.session_memory.conversation.updated_at
                })
                
                # Update tool usage statistics
                tool_name = tool.get('name', 'unknown')
                if tool_name not in self.tool_usage_stats:
                    self.tool_usage_stats[tool_name] = 0
                self.tool_usage_stats[tool_name] += 1
                
                # Track success/failure
                if tool.get('success', True):
                    self.successful_tool_calls += 1
                else:
                    self.failed_tool_calls += 1
        
        # Extract and store any mentioned entities or facts
        self._extract_context_from_response(response)
        
        # Keep tool history limited
        if len(self.tool_history) > 20:
            self.tool_history = self.tool_history[-20:]
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get Agno-specific context data.
        
        Returns:
            Dictionary with Agno-specific context
        """
        return {
            'tool_usage_stats': self.tool_usage_stats,
            'successful_tool_calls': self.successful_tool_calls,
            'failed_tool_calls': self.failed_tool_calls,
            'recent_tools': [t['tool'] for t in self.tool_history[-5:]],
            'context_variables': self.context_variables,
            'agent_name': self.agent.name if self.agent and hasattr(self.agent, 'name') else None
        }
    
    def wrap_agent_call(self, user_input: str, use_tools: bool = True) -> str:
        """
        Wrap Agno agent call with memory management.
        
        Args:
            user_input: User input
            use_tools: Whether to allow tool usage
            
        Returns:
            Agent response
        """
        if not self.agent:
            raise ValueError("No agent configured")
        
        # Add user message to memory
        self.add_user_message(user_input)
        
        # Inject memory context
        enhanced_input = self.inject_memory_context(user_input)
        
        # Call agent
        metadata = {'use_tools': use_tools}
        
        # Get response from agent
        if hasattr(self.agent, 'chat'):
            response = self.agent.chat(enhanced_input, use_tools=use_tools)
        else:
            # Fallback for different agent interfaces
            response = str(self.agent(enhanced_input))
        
        # Check if tools were used and extract metadata
        if hasattr(self.agent, 'last_tool_calls'):
            metadata['tools_used'] = self.agent.last_tool_calls
        
        # Process response
        processed_response = self.process_response(response, metadata)
        
        # Add assistant message to memory
        self.add_assistant_message(processed_response, metadata)
        
        return processed_response
    
    def set_agent(self, agent) -> None:
        """
        Set or update the Agno agent.
        
        Args:
            agent: Agno agent instance
        """
        self.agent = agent
    
    def add_context_variable(self, key: str, value: Any) -> None:
        """
        Add a context variable for future reference.
        
        Args:
            key: Variable name
            value: Variable value
        """
        self.context_variables[key] = value
        
        # Limit context variables
        if len(self.context_variables) > 20:
            # Remove oldest entries
            items = list(self.context_variables.items())
            self.context_variables = dict(items[-20:])
    
    def get_tool_recommendations(self, user_input: str) -> List[str]:
        """
        Get tool recommendations based on input and history.
        
        Args:
            user_input: User input
            
        Returns:
            List of recommended tool names
        """
        recommendations = []
        
        # Check for calculator-related keywords
        if any(word in user_input.lower() for word in ['calculate', 'compute', 'math', 'sum', 'multiply']):
            recommendations.append('calculator')
        
        # Check for weather-related keywords
        if any(word in user_input.lower() for word in ['weather', 'temperature', 'forecast', 'rain']):
            recommendations.append('weather')
        
        # Check for frequently successful tools
        if self.tool_usage_stats:
            # Sort by usage frequency
            sorted_tools = sorted(self.tool_usage_stats.items(), 
                                key=lambda x: x[1], reverse=True)
            # Add top used tools
            for tool, count in sorted_tools[:2]:
                if tool not in recommendations:
                    recommendations.append(tool)
        
        return recommendations
    
    def create_tool_aware_prompt(self, user_input: str) -> str:
        """
        Create a prompt that includes tool recommendations.
        
        Args:
            user_input: User input
            
        Returns:
            Enhanced prompt with tool suggestions
        """
        enhanced_prompt = self.build_enhanced_prompt(user_input)
        
        # Get tool recommendations
        recommended_tools = self.get_tool_recommendations(user_input)
        
        if recommended_tools:
            tool_prompt = f"\nAvailable tools that might be helpful: {', '.join(recommended_tools)}"
            enhanced_prompt += tool_prompt
        
        return enhanced_prompt
    
    def _extract_context_from_response(self, response: str) -> None:
        """
        Extract context variables from response.
        
        Args:
            response: Agent response
        """
        import re
        
        # Look for named entities (simple approach)
        # Extract capitalized words that might be names
        name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        names = re.findall(name_pattern, response)
        for name in names[:3]:  # Limit to 3 names
            self.add_context_variable(f"mentioned_name", name)
        
        # Extract numbers that might be important
        number_pattern = r'\b(\d+\.?\d*)\b'
        numbers = re.findall(number_pattern, response)
        if numbers:
            # Store the last mentioned number as it might be a result
            self.add_context_variable("last_number", numbers[-1])
        
        # Extract dates if mentioned
        date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b'
        dates = re.findall(date_pattern, response)
        for date in dates[:2]:  # Limit to 2 dates
            self.add_context_variable("mentioned_date", date)
    
    def get_agent_memory_state(self) -> Dict[str, Any]:
        """
        Get the complete memory state for the agent.
        
        Returns:
            Dictionary with complete memory state
        """
        return {
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'message_count': len(self.session_memory.messages),
            'tool_history': self.tool_history,
            'context_variables': self.context_variables,
            'tool_usage_stats': self.tool_usage_stats,
            'topics': self.session_memory.extract_topics(),
            'summaries': self.session_memory.summaries
        }
    
    def restore_agent_memory_state(self, state: Dict[str, Any]) -> None:
        """
        Restore agent memory from a saved state.
        
        Args:
            state: Memory state dictionary
        """
        self.conversation_id = state.get('conversation_id')
        self.user_id = state.get('user_id', 'default')
        self.tool_history = state.get('tool_history', [])
        self.context_variables = state.get('context_variables', {})
        self.tool_usage_stats = state.get('tool_usage_stats', {})
        
        # Note: Full conversation restoration would need to load from persistent storage
