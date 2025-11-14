"""
Agno Chatbot with Memory Integration - Best Practices Implementation

This implementation shows the optimal way to integrate memory with Agno's
tool-based architecture, tracking tool usage patterns and results.
"""
import streamlit as st
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.agno.azure_agno_setup import AzureAgnoSetup
from utils import setup_logger

# Memory system imports
from memory_system import create_memory_adapter, AgnoMemoryAdapter
from memory_system.core.models import MemoryFragment, FragmentType, MessageRole

# Set up logging
logger = setup_logger(__name__)


class AgnoMemoryIntegratedChatbot(BaseChatbot):
    """
    Agno chatbot with properly integrated memory system.
    
    Best Practices Demonstrated:
    1. Tool usage tracking in memory
    2. Tool result caching for efficiency
    3. User preference learning for tool selection
    4. Context-aware tool parameter adjustment
    """
    
    def __init__(self):
        super().__init__(
            title="Agno Chatbot with Memory Integration",
            description="""
            Agno with integrated memory system that tracks tool usage patterns,
            caches tool results, and learns user preferences for optimal tool selection.
            """
        )
        self.setup = None
        self.agent = None
        self.memory_adapter = None
        self.tool_memory = {}  # Specialized memory for tool usage
        self.initialize_agno()
        self.initialize_memory()
    
    def initialize_agno(self):
        """Initialize Agno components"""
        if "agno_setup" not in st.session_state:
            try:
                st.session_state.agno_setup = AzureAgnoSetup()
                st.session_state.agno_agent = st.session_state.agno_setup.create_agent_with_tools()
                logger.info("Initialized Agno with tools")
            except Exception as e:
                logger.error(f"Failed to initialize Agno: {str(e)}")
                st.session_state.agno_setup = None
                st.session_state.agno_agent = None
        
        self.setup = st.session_state.agno_setup
        self.agent = st.session_state.agno_agent
    
    def initialize_memory(self):
        """Initialize memory system optimized for Agno's tool usage"""
        if "agno_memory_adapter" not in st.session_state:
            try:
                # Create Agno-specific memory adapter
                st.session_state.agno_memory_adapter = AgnoMemoryAdapter(
                    user_id=st.session_state.get('user_id', 'default'),
                    config={
                        'track_tool_usage': True,
                        'cache_tool_results': True,
                        'learn_tool_preferences': True,
                        'tool_result_ttl': 3600,  # Cache for 1 hour
                        'max_tool_history': 100
                    }
                )
                
                # Initialize tool-specific memory structures
                st.session_state.tool_usage_patterns = {
                    'frequency': {},  # How often each tool is used
                    'success_rate': {},  # Success rate per tool
                    'avg_execution_time': {},  # Performance metrics
                    'user_preferences': {},  # Learned preferences
                    'context_patterns': {}  # When tools are typically used
                }
                
                # Start conversation with tool tracking
                st.session_state.conversation_id = st.session_state.agno_memory_adapter.start_conversation(
                    title=f"Agno Tool Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    metadata={'tools_available': self.get_available_tools()}
                )
                
                logger.info("Initialized Agno memory with tool tracking")
                
            except Exception as e:
                logger.error(f"Failed to initialize memory: {str(e)}")
                st.session_state.agno_memory_adapter = None
        
        self.memory_adapter = st.session_state.agno_memory_adapter
        self.tool_memory = st.session_state.get('tool_usage_patterns', {})
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        if self.agent and hasattr(self.agent, 'tools'):
            return [tool.name for tool in self.agent.tools]
        return ['calculator', 'weather', 'search']  # Default tools
    
    def track_tool_usage(self, tool_name: str, input_params: Dict, result: Any, execution_time: float):
        """
        Track tool usage in memory for pattern learning.
        
        Best Practice: Comprehensive tool tracking for optimization
        """
        if not self.memory_adapter:
            return
        
        # Update frequency tracking
        if tool_name not in self.tool_memory['frequency']:
            self.tool_memory['frequency'][tool_name] = 0
        self.tool_memory['frequency'][tool_name] += 1
        
        # Track execution time
        if tool_name not in self.tool_memory['avg_execution_time']:
            self.tool_memory['avg_execution_time'][tool_name] = []
        self.tool_memory['avg_execution_time'][tool_name].append(execution_time)
        
        # Store in memory as tool interaction
        tool_fragment = MemoryFragment(
            user_id=st.session_state.get('user_id', 'default'),
            conversation_id=st.session_state.get('conversation_id'),
            content=f"Tool: {tool_name}, Input: {json.dumps(input_params)}, Result: {str(result)[:200]}",
            fragment_type=FragmentType.TOOL_USAGE,
            metadata={
                'tool_name': tool_name,
                'input_params': input_params,
                'execution_time': execution_time,
                'success': result is not None,
                'timestamp': datetime.now().isoformat()
            },
            importance=0.7  # Tool usage is generally important
        )
        
        # Add to memory adapter
        self.memory_adapter.add_tool_interaction(tool_name, input_params, result)
        
        # Cache result for potential reuse
        cache_key = f"{tool_name}:{json.dumps(input_params, sort_keys=True)}"
        self.memory_adapter.tool_result_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'ttl': 3600
        }
    
    def get_tool_recommendations(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on memory and patterns.
        
        Best Practice: Use historical data to suggest optimal tools
        """
        recommendations = []
        
        if not self.memory_adapter:
            return recommendations
        
        # Analyze user input for tool indicators
        input_lower = user_input.lower()
        
        # Check tool usage patterns from memory
        relevant_memories = self.memory_adapter.retrieve_relevant_memories(
            user_input,
            memory_types=[FragmentType.TOOL_USAGE],
            limit=5
        )
        
        # Extract tool suggestions from memories
        for memory in relevant_memories:
            if memory.metadata and 'tool_name' in memory.metadata:
                tool_name = memory.metadata['tool_name']
                confidence = memory.metadata.get('success_rate', 0.5)
                
                recommendations.append({
                    'tool': tool_name,
                    'confidence': confidence,
                    'reason': f"Previously used for similar query",
                    'cached': self.check_cache(tool_name, user_input)
                })
        
        # Add pattern-based recommendations
        if any(word in input_lower for word in ['calculate', 'compute', 'add', 'subtract', 'multiply', 'divide']):
            recommendations.append({
                'tool': 'calculator',
                'confidence': 0.9,
                'reason': 'Mathematical operation detected',
                'cached': False
            })
        
        if any(word in input_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
            recommendations.append({
                'tool': 'weather',
                'confidence': 0.9,
                'reason': 'Weather-related query detected',
                'cached': False
            })
        
        return recommendations
    
    def check_cache(self, tool_name: str, input_params: Any) -> bool:
        """Check if we have a cached result for this tool call"""
        if not self.memory_adapter:
            return False
        
        cache_key = f"{tool_name}:{json.dumps(input_params, sort_keys=True)}"
        if cache_key in self.memory_adapter.tool_result_cache:
            cached = self.memory_adapter.tool_result_cache[cache_key]
            # Check if cache is still valid
            age = (datetime.now() - cached['timestamp']).seconds
            return age < cached.get('ttl', 3600)
        return False
    
    def get_cached_result(self, tool_name: str, input_params: Any) -> Optional[Any]:
        """Retrieve cached tool result if available"""
        if not self.memory_adapter:
            return None
        
        cache_key = f"{tool_name}:{json.dumps(input_params, sort_keys=True)}"
        if cache_key in self.memory_adapter.tool_result_cache:
            cached = self.memory_adapter.tool_result_cache[cache_key]
            age = (datetime.now() - cached['timestamp']).seconds
            if age < cached.get('ttl', 3600):
                return cached['result']
        return None
    
    def render_sidebar(self):
        """Render sidebar with Agno-specific memory settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("🛠️ Tool Memory Settings")
            
            # Tool usage statistics
            if self.tool_memory and 'frequency' in self.tool_memory:
                st.caption("Tool Usage Statistics:")
                for tool, count in self.tool_memory['frequency'].items():
                    avg_time = 0
                    if tool in self.tool_memory.get('avg_execution_time', {}):
                        times = self.tool_memory['avg_execution_time'][tool]
                        avg_time = sum(times) / len(times) if times else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{tool}", count, help="Usage count")
                    with col2:
                        st.metric("Avg Time", f"{avg_time:.2f}s", help="Average execution time")
            
            # Cache settings
            st.caption("Cache Settings:")
            cache_ttl = st.slider(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=7200,
                value=3600,
                help="How long to cache tool results"
            )
            
            if st.button("Clear Tool Cache"):
                if self.memory_adapter:
                    self.memory_adapter.tool_result_cache.clear()
                    st.success("Tool cache cleared!")
            
            # Memory operations
            st.divider()
            st.subheader("🧠 Memory Operations")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Session"):
                    self.save_tool_session()
            
            with col2:
                if st.button("📊 Export Patterns"):
                    self.export_tool_patterns()
            
            # Tool recommendations
            if st.checkbox("Show Tool Recommendations"):
                self.show_tool_recommendations = True
            else:
                self.show_tool_recommendations = False
    
    def save_tool_session(self):
        """Save current session with tool usage patterns"""
        try:
            if self.memory_adapter:
                # Save conversation with tool metadata
                metadata = {
                    'tool_usage': self.tool_memory,
                    'cache_stats': {
                        'size': len(self.memory_adapter.tool_result_cache),
                        'hits': getattr(self.memory_adapter, 'cache_hits', 0),
                        'misses': getattr(self.memory_adapter, 'cache_misses', 0)
                    }
                }
                
                conversation_id = self.memory_adapter.save_conversation(metadata=metadata)
                
                # Extract and store tool patterns
                self.memory_adapter.extract_tool_patterns()
                
                st.success(f"Session saved with tool patterns! ID: {conversation_id}")
        except Exception as e:
            st.error(f"Failed to save session: {str(e)}")
    
    def export_tool_patterns(self):
        """Export learned tool usage patterns"""
        if self.tool_memory:
            export_data = {
                'user_id': st.session_state.get('user_id', 'default'),
                'export_date': datetime.now().isoformat(),
                'tool_patterns': self.tool_memory,
                'recommendations': []
            }
            
            # Add recommendation history if available
            if self.memory_adapter:
                memories = self.memory_adapter.retrieve_relevant_memories(
                    "tool usage",
                    memory_types=[FragmentType.TOOL_USAGE],
                    limit=20
                )
                for memory in memories:
                    if memory.metadata:
                        export_data['recommendations'].append({
                            'tool': memory.metadata.get('tool_name'),
                            'context': memory.content[:100],
                            'success': memory.metadata.get('success'),
                            'timestamp': memory.metadata.get('timestamp')
                        })
            
            # Create download button
            st.download_button(
                label="Download Tool Patterns",
                data=json.dumps(export_data, indent=2),
                file_name=f"agno_tool_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from Agno with memory-enhanced tool usage.
        
        Best Practice: Use memory to optimize tool selection and caching
        """
        try:
            if self.agent is None:
                return "Agent not initialized. Please check your configuration."
            
            # Add to memory
            if self.memory_adapter:
                self.memory_adapter.add_user_message(prompt)
            
            # Get tool recommendations based on memory
            tool_recommendations = self.get_tool_recommendations(prompt)
            
            # Display recommendations if enabled
            if hasattr(self, 'show_tool_recommendations') and self.show_tool_recommendations and tool_recommendations:
                with st.expander("🔧 Tool Recommendations", expanded=True):
                    for rec in tool_recommendations:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Tool:** {rec['tool']}")
                        with col2:
                            st.write(f"**Confidence:** {rec['confidence']:.1%}")
                        with col3:
                            if rec.get('cached'):
                                st.write("✅ Cached")
            
            # Check for cached results first
            for rec in tool_recommendations:
                if rec.get('cached'):
                    cached_result = self.get_cached_result(rec['tool'], prompt)
                    if cached_result:
                        response = f"[Using cached result] {cached_result}"
                        if self.memory_adapter:
                            self.memory_adapter.add_assistant_message(
                                response,
                                metadata={'used_cache': True, 'tool': rec['tool']}
                            )
                        return response
            
            # Enhance prompt with memory context
            if self.memory_adapter:
                # Get relevant tool usage history
                tool_context = self.memory_adapter.get_tool_context(prompt)
                if tool_context:
                    enhanced_prompt = f"{prompt}\n\n[Previous tool usage context: {tool_context}]"
                else:
                    enhanced_prompt = prompt
            else:
                enhanced_prompt = prompt
            
            # Execute with tool tracking
            import time
            start_time = time.time()
            
            # Process with agent
            response = self.agent.run(enhanced_prompt)
            
            execution_time = time.time() - start_time
            
            # Track tool usage if tools were called
            if hasattr(self.agent, 'last_tool_used'):
                self.track_tool_usage(
                    self.agent.last_tool_used,
                    {'prompt': prompt},
                    response,
                    execution_time
                )
            
            # Add response to memory
            if self.memory_adapter:
                self.memory_adapter.add_assistant_message(
                    response,
                    metadata={
                        'execution_time': execution_time,
                        'tools_available': len(self.get_available_tools())
                    }
                )
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error getting response: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with tool status"""
        super().render_header()
        
        # Show tool memory status
        if self.memory_adapter:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cache_size = len(getattr(self.memory_adapter, 'tool_result_cache', {}))
                st.caption(f"📦 Cache: {cache_size} items")
            
            with col2:
                tool_count = len(self.tool_memory.get('frequency', {}))
                st.caption(f"🛠️ Tools Used: {tool_count}")
            
            with col3:
                total_calls = sum(self.tool_memory.get('frequency', {}).values())
                st.caption(f"📊 Total Calls: {total_calls}")
            
            with col4:
                st.caption(f"👤 User: {st.session_state.get('user_id', 'default')}")
        
        # Show example queries
        with st.expander("Example Queries with Tools"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Calculator Tool:**")
                st.code("What's 15% of 250?")
                st.code("Calculate 2^10")
                st.code("Add 45.5 and 32.7")
            
            with col2:
                st.markdown("**Weather Tool:**")
                st.code("What's the weather today?")
                st.code("Will it rain tomorrow?")
                st.code("Temperature in New York")
            
            with col3:
                st.markdown("**Memory-Aware:**")
                st.code("Use the same tool as before")
                st.code("What tool did we use last?")
                st.code("Show my tool history")


def main():
    """Main function to run the Agno chatbot with memory"""
    chatbot = AgnoMemoryIntegratedChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
