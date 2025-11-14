"""
LangGraph framework memory adapter.

This module provides memory integration for LangGraph's state-based
graph workflows.
"""

from typing import List, Dict, Optional, Any, TypedDict, Annotated
from datetime import datetime
import operator
from langgraph.graph import StateGraph
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework, MemoryFragment


class GraphMemoryState(TypedDict):
    """
    State structure for memory-enhanced graphs.
    
    This state includes both standard conversation elements
    and memory-specific components.
    """
    # Conversation elements
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    
    # Memory elements
    session_memory: Dict[str, Any]
    long_term_memories: List[Dict[str, Any]]
    working_memory: Dict[str, Any]
    
    # Metadata
    user_id: str
    conversation_id: str
    timestamp: datetime


class LangGraphMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for LangGraph framework.
    
    This adapter integrates memory management into LangGraph's
    state-based graph workflows, providing memory as a state component.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LangGraph memory adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Graph-specific attributes
        self.current_graph = None
        self.state_history = []
        self.node_execution_history = []
        self.graph_metadata = {}
    
    def _initialize_framework(self) -> None:
        """Initialize LangGraph-specific components."""
        self.session_memory.conversation.framework = Framework.LANGGRAPH
        
        # Initialize graph-specific tracking
        self.state_transitions = []
        self.decision_points = []
        self.cycle_count = {}
    
    def inject_memory_context(self, input_text: str,
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context for graph processing.
        
        Args:
            input_text: Original input
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input (usually handled via state)
        """
        # For LangGraph, memory is typically handled through state
        # rather than prompt enhancement
        return input_text
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process graph output and extract state information.
        
        Args:
            response: Graph output
            metadata: Optional metadata including graph state
            
        Returns:
            Processed response
        """
        # Track state transitions
        if metadata and 'state' in metadata:
            self.state_history.append({
                'timestamp': datetime.now(),
                'state': metadata['state'],
                'response': response
            })
        
        # Track node execution
        if metadata and 'node' in metadata:
            self.node_execution_history.append({
                'node': metadata['node'],
                'timestamp': datetime.now(),
                'output': response[:200]  # Store truncated output
            })
        
        # Keep history limited
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
        if len(self.node_execution_history) > 100:
            self.node_execution_history = self.node_execution_history[-100:]
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get LangGraph-specific context data.
        
        Returns:
            Dictionary with LangGraph-specific context
        """
        return {
            'current_graph': self.current_graph.name if self.current_graph and hasattr(self.current_graph, 'name') else None,
            'state_history_length': len(self.state_history),
            'node_execution_count': len(self.node_execution_history),
            'recent_nodes': [n['node'] for n in self.node_execution_history[-5:]],
            'state_transitions': len(self.state_transitions),
            'cycle_counts': self.cycle_count,
            'graph_metadata': self.graph_metadata
        }
    
    def create_memory_enhanced_state(self, initial_input: str) -> GraphMemoryState:
        """
        Create an initial state with memory components.
        
        Args:
            initial_input: Initial user input
            
        Returns:
            GraphMemoryState with memory components initialized
        """
        # Get session context
        session_context = self.session_memory.get_context()
        
        # Get relevant long-term memories
        long_term_memories = []
        if self.persistent_memory:
            memories = self.retrieve_relevant_memories(initial_input, limit=5)
            long_term_memories = [
                {
                    'content': mem.content,
                    'type': mem.fragment_type.value,
                    'importance': mem.importance_score
                }
                for mem in memories
            ]
        
        # Create initial state
        state: GraphMemoryState = {
            'messages': [HumanMessage(content=initial_input)],
            'current_step': 'start',
            'session_memory': {
                'context': session_context,
                'topics': self.session_memory.extract_topics(),
                'summaries': self.session_memory.summaries
            },
            'long_term_memories': long_term_memories,
            'working_memory': {},
            'user_id': self.user_id,
            'conversation_id': self.conversation_id or self.start_conversation(),
            'timestamp': datetime.now()
        }
        
        return state
    
    def create_memory_node(self, node_type: str = 'retrieve'):
        """
        Create a memory node for graph workflows.
        
        Args:
            node_type: Type of memory node ('retrieve', 'store', 'consolidate')
            
        Returns:
            Node function for the graph
        """
        if node_type == 'retrieve':
            def retrieve_memory_node(state: GraphMemoryState) -> GraphMemoryState:
                """Retrieve relevant memories based on current context."""
                # Get the latest message
                if state['messages']:
                    latest_msg = state['messages'][-1].content
                    
                    # Retrieve memories
                    if self.persistent_memory:
                        memories = self.retrieve_relevant_memories(latest_msg, limit=3)
                        
                        # Add to working memory
                        state['working_memory']['retrieved_memories'] = [
                            {'content': mem.content, 'importance': mem.importance_score}
                            for mem in memories
                        ]
                
                return state
            return retrieve_memory_node
        
        elif node_type == 'store':
            def store_memory_node(state: GraphMemoryState) -> GraphMemoryState:
                """Store important information in memory."""
                # Extract information from the conversation
                if len(state['messages']) >= 2:
                    # Get last exchange
                    user_msg = state['messages'][-2].content if len(state['messages']) > 1 else ""
                    assistant_msg = state['messages'][-1].content
                    
                    # Store in session memory
                    self.add_user_message(user_msg)
                    self.add_assistant_message(assistant_msg)
                    
                    # Update working memory
                    state['working_memory']['stored'] = True
                
                return state
            return store_memory_node
        
        elif node_type == 'consolidate':
            def consolidate_memory_node(state: GraphMemoryState) -> GraphMemoryState:
                """Consolidate and summarize memories."""
                # Create summary if enough messages
                if len(state['messages']) > 10:
                    summary = self.session_memory.summarize()
                    state['session_memory']['summaries'].append(summary)
                    
                    # Update working memory
                    state['working_memory']['consolidated'] = True
                
                return state
            return consolidate_memory_node
        
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    def enhance_graph_with_memory(self, graph: StateGraph) -> StateGraph:
        """
        Enhance an existing graph with memory nodes.
        
        Args:
            graph: LangGraph StateGraph instance
            
        Returns:
            Enhanced graph with memory capabilities
        """
        # Add memory nodes
        graph.add_node("retrieve_memory", self.create_memory_node("retrieve"))
        graph.add_node("store_memory", self.create_memory_node("store"))
        graph.add_node("consolidate_memory", self.create_memory_node("consolidate"))
        
        # Store reference to current graph
        self.current_graph = graph
        
        return graph
    
    def process_graph_execution(self, graph, initial_input: str) -> str:
        """
        Process a complete graph execution with memory management.
        
        Args:
            graph: Compiled graph
            initial_input: User input
            
        Returns:
            Graph output
        """
        # Create initial state with memory
        initial_state = self.create_memory_enhanced_state(initial_input)
        
        # Add user message to our memory
        self.add_user_message(initial_input)
        
        # Execute graph
        final_state = graph.invoke(initial_state)
        
        # Extract response
        response = self._extract_response_from_state(final_state)
        
        # Process and store response
        metadata = {
            'state': final_state,
            'node': final_state.get('current_step', 'unknown')
        }
        processed_response = self.process_response(response, metadata)
        
        # Add assistant message to our memory
        self.add_assistant_message(processed_response)
        
        # Track state transition
        self.state_transitions.append({
            'from': initial_state.get('current_step'),
            'to': final_state.get('current_step'),
            'timestamp': datetime.now()
        })
        
        return processed_response
    
    def track_cycle_execution(self, node_name: str) -> int:
        """
        Track execution cycles for nodes.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Current cycle count for the node
        """
        if node_name not in self.cycle_count:
            self.cycle_count[node_name] = 0
        
        self.cycle_count[node_name] += 1
        return self.cycle_count[node_name]
    
    def create_conditional_memory_router(self):
        """
        Create a conditional router based on memory state.
        
        Returns:
            Router function for conditional edges
        """
        def memory_router(state: GraphMemoryState) -> str:
            """Route based on memory conditions."""
            # Check if we need to retrieve memories
            if not state['working_memory'].get('retrieved_memories'):
                return 'retrieve_memory'
            
            # Check if we need to consolidate
            message_count = len(state['messages'])
            if message_count > 20 and not state['working_memory'].get('consolidated'):
                return 'consolidate_memory'
            
            # Check if we have enough context
            if state['long_term_memories'] and len(state['long_term_memories']) > 3:
                return 'process_with_context'
            
            # Default route
            return 'process_standard'
        
        return memory_router
    
    def extract_graph_insights(self) -> Dict[str, Any]:
        """
        Extract insights from graph execution history.
        
        Returns:
            Dictionary with graph insights
        """
        insights = {
            'most_visited_nodes': [],
            'average_cycles': 0,
            'decision_patterns': [],
            'bottleneck_nodes': []
        }
        
        # Analyze node execution frequency
        node_freq = {}
        for execution in self.node_execution_history:
            node = execution['node']
            node_freq[node] = node_freq.get(node, 0) + 1
        
        # Most visited nodes
        if node_freq:
            sorted_nodes = sorted(node_freq.items(), key=lambda x: x[1], reverse=True)
            insights['most_visited_nodes'] = sorted_nodes[:5]
        
        # Average cycles
        if self.cycle_count:
            insights['average_cycles'] = sum(self.cycle_count.values()) / len(self.cycle_count)
        
        # Decision patterns
        decision_freq = {}
        for transition in self.state_transitions[-20:]:  # Last 20 transitions
            decision = f"{transition['from']}->{transition['to']}"
            decision_freq[decision] = decision_freq.get(decision, 0) + 1
        insights['decision_patterns'] = list(decision_freq.keys())[:5]
        
        # Identify potential bottlenecks (nodes with high execution time)
        # This would need actual timing data in production
        
        return insights
    
    def _extract_response_from_state(self, state: GraphMemoryState) -> str:
        """
        Extract the response from the final graph state.
        
        Args:
            state: Final graph state
            
        Returns:
            Extracted response string
        """
        # Look for final answer in state
        if 'final_answer' in state:
            return state['final_answer']
        
        # Look for last AI message
        if state['messages']:
            for msg in reversed(state['messages']):
                if isinstance(msg, AIMessage):
                    return msg.content
        
        # Check working memory for output
        if 'output' in state.get('working_memory', {}):
            return state['working_memory']['output']
        
        return "Graph execution completed without explicit output."
    
    def save_graph_state(self, state: GraphMemoryState, filename: str) -> None:
        """
        Save graph state for debugging or replay.
        
        Args:
            state: Graph state to save
            filename: Filename for saving
        """
        import json
        
        # Convert state to serializable format
        serializable_state = {
            'messages': [
                {'type': type(msg).__name__, 'content': msg.content}
                for msg in state['messages']
            ],
            'current_step': state['current_step'],
            'session_memory': state['session_memory'],
            'long_term_memories': state['long_term_memories'],
            'working_memory': state['working_memory'],
            'user_id': state['user_id'],
            'conversation_id': state['conversation_id'],
            'timestamp': state['timestamp'].isoformat()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_state, f, indent=2)
    
    def load_graph_state(self, filename: str) -> GraphMemoryState:
        """
        Load graph state from file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded GraphMemoryState
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct messages
        messages = []
        for msg_data in data['messages']:
            if msg_data['type'] == 'HumanMessage':
                messages.append(HumanMessage(content=msg_data['content']))
            elif msg_data['type'] == 'AIMessage':
                messages.append(AIMessage(content=msg_data['content']))
        
        # Reconstruct state
        state: GraphMemoryState = {
            'messages': messages,
            'current_step': data['current_step'],
            'session_memory': data['session_memory'],
            'long_term_memories': data['long_term_memories'],
            'working_memory': data['working_memory'],
            'user_id': data['user_id'],
            'conversation_id': data['conversation_id'],
            'timestamp': datetime.fromisoformat(data['timestamp'])
        }
        
        return state
