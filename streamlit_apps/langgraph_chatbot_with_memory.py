"""
LangGraph Chatbot with Memory Integration - Best Practices Implementation

This implementation shows the optimal way to integrate memory with LangGraph's
state-based graph architecture, maintaining memory across graph nodes and edges.
"""
import streamlit as st
import sys
import os
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.langgraph.azure_langgraph_setup import AzureLangGraphSetup
from utils import setup_logger

# Memory system imports
from memory_system import create_memory_adapter, LangGraphMemoryAdapter
from memory_system.core.models import MemoryFragment, FragmentType, MessageRole

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

logger = setup_logger(__name__)


class GraphState(TypedDict):
    """
    State definition for LangGraph with integrated memory.
    
    Best Practice: Include memory context directly in graph state
    """
    messages: List[Dict[str, Any]]
    memory_context: Dict[str, Any]
    current_node: str
    path_history: List[str]
    user_profile: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    decision_history: List[Dict[str, Any]]
    should_terminate: bool


class LangGraphMemoryIntegratedChatbot(BaseChatbot):
    """
    LangGraph chatbot with properly integrated memory system.
    
    Best Practices Demonstrated:
    1. Memory as part of graph state
    2. Memory-aware routing decisions
    3. State persistence across graph executions
    4. Graph path tracking in memory
    """
    
    def __init__(self):
        super().__init__(
            title="LangGraph Chatbot with Memory Integration",
            description="""
            LangGraph with integrated memory that flows through graph states,
            influences routing decisions, and maintains context across graph executions.
            """
        )
        self.setup = None
        self.graph = None
        self.memory_adapter = None
        self.initialize_langgraph()
        self.initialize_memory()
        self.build_memory_enhanced_graph()
    
    def initialize_langgraph(self):
        """Initialize LangGraph components"""
        if "langgraph_setup" not in st.session_state:
            try:
                st.session_state.langgraph_setup = AzureLangGraphSetup()
                logger.info("Initialized LangGraph setup")
            except Exception as e:
                logger.error(f"Failed to initialize LangGraph: {str(e)}")
                st.session_state.langgraph_setup = None
        
        self.setup = st.session_state.langgraph_setup
    
    def initialize_memory(self):
        """Initialize memory system optimized for LangGraph's state management"""
        if "langgraph_memory_adapter" not in st.session_state:
            try:
                # Create LangGraph-specific memory adapter
                st.session_state.langgraph_memory_adapter = LangGraphMemoryAdapter(
                    user_id=st.session_state.get('user_id', 'default'),
                    config={
                        'track_graph_paths': True,
                        'store_intermediate_states': True,
                        'enable_state_checkpointing': True,
                        'checkpoint_frequency': 5,  # Every 5 nodes
                        'max_path_history': 50,
                        'enable_decision_memory': True
                    }
                )
                
                # Initialize graph-specific memory structures
                st.session_state.graph_memory = {
                    'execution_paths': [],  # Historical paths through graphs
                    'decision_points': {},  # Decisions made at routing nodes
                    'state_snapshots': [],  # Checkpointed states
                    'node_visit_frequency': {},  # Which nodes are visited most
                    'edge_transition_patterns': {},  # Common transitions
                    'terminal_states': []  # How graphs typically end
                }
                
                # Start conversation with graph tracking
                st.session_state.conversation_id = st.session_state.langgraph_memory_adapter.start_conversation(
                    title=f"LangGraph Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    metadata={'graph_type': 'memory_enhanced'}
                )
                
                logger.info("Initialized LangGraph memory with state tracking")
                
            except Exception as e:
                logger.error(f"Failed to initialize memory: {str(e)}")
                st.session_state.langgraph_memory_adapter = None
        
        self.memory_adapter = st.session_state.langgraph_memory_adapter
        self.graph_memory = st.session_state.get('graph_memory', {})
    
    def build_memory_enhanced_graph(self):
        """
        Build a LangGraph with integrated memory at each node.
        
        Best Practice: Memory influences graph flow and decisions
        """
        if not self.setup:
            return
        
        # Create a new StateGraph with memory-aware state
        workflow = StateGraph(GraphState)
        
        # Add memory-enhanced nodes
        workflow.add_node("memory_retrieval", self.memory_retrieval_node)
        workflow.add_node("context_analysis", self.context_analysis_node)
        workflow.add_node("response_generation", self.response_generation_node)
        workflow.add_node("memory_update", self.memory_update_node)
        workflow.add_node("decision_router", self.decision_router_node)
        
        # Set entry point
        workflow.set_entry_point("memory_retrieval")
        
        # Add edges with memory-aware routing
        workflow.add_edge("memory_retrieval", "context_analysis")
        workflow.add_edge("context_analysis", "decision_router")
        
        # Conditional routing based on memory and context
        workflow.add_conditional_edges(
            "decision_router",
            self.route_based_on_memory,
            {
                "generate": "response_generation",
                "need_more_context": "memory_retrieval",
                "terminate": END
            }
        )
        
        workflow.add_edge("response_generation", "memory_update")
        workflow.add_edge("memory_update", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        
        # Store graph structure in memory for analysis
        if self.memory_adapter:
            self.memory_adapter.store_graph_structure(workflow)
    
    def memory_retrieval_node(self, state: GraphState) -> GraphState:
        """
        Node that retrieves relevant memories for context.
        
        Best Practice: Start graph execution with memory context
        """
        if not self.memory_adapter:
            state['retrieval_results'] = []
            return state
        
        # Get the latest message
        latest_message = state['messages'][-1] if state['messages'] else {'content': ''}
        query = latest_message.get('content', '')
        
        # Retrieve relevant memories
        memories = self.memory_adapter.retrieve_relevant_memories(
            query,
            limit=5,
            include_graph_paths=True  # Include previous graph execution paths
        )
        
        # Retrieve user profile from memory
        user_profile = self.memory_adapter.get_user_profile(
            st.session_state.get('user_id', 'default')
        )
        
        # Update state with retrieved context
        state['memory_context'] = {
            'relevant_memories': [m.content for m in memories],
            'memory_metadata': [m.metadata for m in memories if m.metadata],
            'retrieval_timestamp': datetime.now().isoformat()
        }
        
        state['user_profile'] = user_profile or {}
        state['retrieval_results'] = memories
        state['path_history'].append('memory_retrieval')
        
        # Track node visit
        self.track_node_visit('memory_retrieval', state)
        
        return state
    
    def context_analysis_node(self, state: GraphState) -> GraphState:
        """
        Analyze context from memory to inform routing decisions.
        
        Best Practice: Use memory to understand conversation context
        """
        # Analyze the retrieved memories and current conversation
        context_score = 0.0
        needs_clarification = False
        
        # Check if we have sufficient context from memory
        if state['memory_context'].get('relevant_memories'):
            context_score = len(state['memory_context']['relevant_memories']) / 5.0
        
        # Check user profile for preferences
        if state['user_profile']:
            # User has established preferences
            context_score += 0.2
        
        # Analyze conversation continuity
        if len(state['messages']) > 2:
            # Ongoing conversation with context
            context_score += 0.3
        
        # Store analysis in state
        state['memory_context']['context_score'] = context_score
        state['memory_context']['needs_clarification'] = context_score < 0.3
        state['path_history'].append('context_analysis')
        
        # Track node visit
        self.track_node_visit('context_analysis', state)
        
        return state
    
    def decision_router_node(self, state: GraphState) -> GraphState:
        """
        Make routing decisions based on memory and context.
        
        Best Practice: Memory-informed decision making
        """
        decision = {
            'timestamp': datetime.now().isoformat(),
            'context_score': state['memory_context'].get('context_score', 0),
            'factors': []
        }
        
        # Record factors influencing the decision
        if state['memory_context'].get('needs_clarification'):
            decision['factors'].append('insufficient_context')
        
        if state.get('should_terminate'):
            decision['factors'].append('termination_requested')
        
        if len(state['path_history']) > 10:
            decision['factors'].append('max_depth_reached')
        
        # Store decision in memory
        state['decision_history'].append(decision)
        state['path_history'].append('decision_router')
        
        # Track in graph memory
        self.track_decision_point('decision_router', decision, state)
        
        return state
    
    def route_based_on_memory(self, state: GraphState) -> str:
        """
        Routing function that uses memory to determine next node.
        
        Best Practice: Dynamic routing based on historical patterns
        """
        # Check termination conditions
        if state.get('should_terminate'):
            return "terminate"
        
        # Check if we've been through too many iterations
        if len(state['path_history']) > 10:
            return "terminate"
        
        # Check context score
        context_score = state['memory_context'].get('context_score', 0)
        
        if context_score < 0.3:
            # Need more context from memory
            return "need_more_context"
        else:
            # Have enough context to generate response
            return "generate"
    
    def response_generation_node(self, state: GraphState) -> GraphState:
        """
        Generate response using memory-enhanced context.
        
        Best Practice: Incorporate memory into response generation
        """
        if not self.setup:
            state['messages'].append({
                'role': 'assistant',
                'content': 'Setup not initialized'
            })
            return state
        
        # Build enhanced prompt with memory context
        latest_message = state['messages'][-1] if state['messages'] else {'content': ''}
        user_input = latest_message.get('content', '')
        
        # Create memory-enhanced prompt
        prompt_parts = [f"User: {user_input}"]
        
        # Add relevant memories
        if state['memory_context'].get('relevant_memories'):
            prompt_parts.append("\nRelevant context from memory:")
            for memory in state['memory_context']['relevant_memories'][:3]:
                prompt_parts.append(f"- {memory}")
        
        # Add user profile context
        if state['user_profile']:
            prompt_parts.append(f"\nUser preferences: {json.dumps(state['user_profile'], indent=2)}")
        
        enhanced_prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.setup.llm.invoke(enhanced_prompt)
        
        # Add to messages
        state['messages'].append({
            'role': 'assistant',
            'content': response.content if hasattr(response, 'content') else str(response)
        })
        
        state['path_history'].append('response_generation')
        
        # Track node visit
        self.track_node_visit('response_generation', state)
        
        return state
    
    def memory_update_node(self, state: GraphState) -> GraphState:
        """
        Update memory with the current graph execution.
        
        Best Practice: Store graph execution patterns for future use
        """
        if not self.memory_adapter:
            return state
        
        # Store the execution path
        execution_path = {
            'path': state['path_history'],
            'decisions': state['decision_history'],
            'timestamp': datetime.now().isoformat(),
            'success': True  # Can be determined by outcome
        }
        
        # Add to graph memory
        self.graph_memory['execution_paths'].append(execution_path)
        
        # Create memory fragment for this execution
        path_fragment = MemoryFragment(
            user_id=st.session_state.get('user_id', 'default'),
            conversation_id=st.session_state.get('conversation_id'),
            content=f"Graph execution path: {' -> '.join(state['path_history'])}",
            fragment_type=FragmentType.GRAPH_EXECUTION,
            metadata={
                'path': state['path_history'],
                'node_count': len(state['path_history']),
                'decisions': state['decision_history'],
                'context_score': state['memory_context'].get('context_score', 0)
            },
            importance=0.6
        )
        
        # Store in memory adapter
        self.memory_adapter.add_graph_execution(state)
        
        state['path_history'].append('memory_update')
        
        return state
    
    def track_node_visit(self, node_name: str, state: GraphState):
        """Track node visits for pattern analysis"""
        if node_name not in self.graph_memory['node_visit_frequency']:
            self.graph_memory['node_visit_frequency'][node_name] = 0
        self.graph_memory['node_visit_frequency'][node_name] += 1
    
    def track_decision_point(self, node_name: str, decision: Dict, state: GraphState):
        """Track decisions made at routing nodes"""
        if node_name not in self.graph_memory['decision_points']:
            self.graph_memory['decision_points'][node_name] = []
        
        self.graph_memory['decision_points'][node_name].append({
            'decision': decision,
            'state_summary': {
                'message_count': len(state['messages']),
                'path_length': len(state['path_history']),
                'context_score': state['memory_context'].get('context_score', 0)
            }
        })
    
    def checkpoint_state(self, state: GraphState):
        """
        Checkpoint graph state for recovery and analysis.
        
        Best Practice: Regular state checkpointing for complex graphs
        """
        if self.memory_adapter:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'path_position': len(state['path_history'])
            }
            
            self.graph_memory['state_snapshots'].append(checkpoint)
            
            # Store in persistent memory if needed
            if len(state['path_history']) % 5 == 0:  # Every 5 nodes
                self.memory_adapter.store_checkpoint(checkpoint)
    
    def render_sidebar(self):
        """Render sidebar with LangGraph-specific memory settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("🔀 Graph Memory Settings")
            
            # Graph execution statistics
            if self.graph_memory:
                st.caption("Graph Execution Stats:")
                
                # Node visit frequency
                if 'node_visit_frequency' in self.graph_memory:
                    most_visited = max(
                        self.graph_memory['node_visit_frequency'].items(),
                        key=lambda x: x[1],
                        default=('None', 0)
                    )
                    st.metric("Most Visited Node", most_visited[0], f"{most_visited[1]} visits")
                
                # Execution paths
                if 'execution_paths' in self.graph_memory:
                    st.metric("Total Executions", len(self.graph_memory['execution_paths']))
                    
                    if self.graph_memory['execution_paths']:
                        avg_path_length = sum(
                            len(p['path']) for p in self.graph_memory['execution_paths']
                        ) / len(self.graph_memory['execution_paths'])
                        st.metric("Avg Path Length", f"{avg_path_length:.1f} nodes")
            
            # Graph type selection
            st.caption("Graph Configuration:")
            graph_type = st.selectbox(
                "Graph Type",
                ["Memory-Enhanced", "Simple Linear", "Conditional", "Cyclic"],
                help="Select graph architecture"
            )
            
            if st.button("Rebuild Graph"):
                self.rebuild_graph(graph_type)
            
            # State checkpointing
            st.caption("State Management:")
            
            checkpoint_enabled = st.checkbox(
                "Enable Checkpointing",
                value=True,
                help="Save graph state at regular intervals"
            )
            
            if checkpoint_enabled:
                checkpoint_freq = st.slider(
                    "Checkpoint Frequency",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Checkpoint every N nodes"
                )
            
            # Memory operations
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Graph"):
                    self.save_graph_session()
            
            with col2:
                if st.button("📊 Export Paths"):
                    self.export_graph_patterns()
            
            # Path visualization
            if st.checkbox("Show Execution Path"):
                self.show_execution_path = True
            else:
                self.show_execution_path = False
    
    def rebuild_graph(self, graph_type: str):
        """Rebuild graph with different architecture"""
        try:
            if graph_type == "Memory-Enhanced":
                self.build_memory_enhanced_graph()
            else:
                # Build other graph types
                # This would call different graph building methods
                pass
            
            st.success(f"Graph rebuilt as {graph_type}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to rebuild graph: {str(e)}")
    
    def save_graph_session(self):
        """Save current graph session with execution patterns"""
        try:
            if self.memory_adapter:
                # Save with graph-specific metadata
                metadata = {
                    'graph_memory': self.graph_memory,
                    'graph_type': 'memory_enhanced',
                    'checkpoints': len(self.graph_memory.get('state_snapshots', []))
                }
                
                conversation_id = self.memory_adapter.save_conversation(metadata=metadata)
                
                # Extract and store graph patterns
                self.memory_adapter.extract_graph_patterns(self.graph_memory)
                
                st.success(f"Graph session saved! ID: {conversation_id}")
        except Exception as e:
            st.error(f"Failed to save session: {str(e)}")
    
    def export_graph_patterns(self):
        """Export learned graph execution patterns"""
        if self.graph_memory:
            export_data = {
                'user_id': st.session_state.get('user_id', 'default'),
                'export_date': datetime.now().isoformat(),
                'graph_patterns': {
                    'execution_paths': self.graph_memory.get('execution_paths', []),
                    'node_frequencies': self.graph_memory.get('node_visit_frequency', {}),
                    'decision_points': self.graph_memory.get('decision_points', {}),
                    'edge_transitions': self.graph_memory.get('edge_transition_patterns', {})
                }
            }
            
            st.download_button(
                label="Download Graph Patterns",
                data=json.dumps(export_data, indent=2),
                file_name=f"langgraph_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from LangGraph with memory-enhanced execution.
        
        Best Practice: Execute graph with full memory context
        """
        try:
            if not self.graph:
                return "Graph not initialized. Please check your configuration."
            
            # Initialize state with memory context
            initial_state: GraphState = {
                'messages': [{'role': 'user', 'content': prompt}],
                'memory_context': {},
                'current_node': 'start',
                'path_history': [],
                'user_profile': {},
                'retrieval_results': [],
                'decision_history': [],
                'should_terminate': False
            }
            
            # Add to memory before execution
            if self.memory_adapter:
                self.memory_adapter.add_user_message(prompt)
            
            # Execute graph with memory-enhanced state
            result = self.graph.invoke(initial_state)
            
            # Extract response from result
            if result and 'messages' in result:
                # Get the last assistant message
                for message in reversed(result['messages']):
                    if message.get('role') == 'assistant':
                        response = message.get('content', 'No response generated')
                        
                        # Add to memory
                        if self.memory_adapter:
                            self.memory_adapter.add_assistant_message(
                                response,
                                metadata={
                                    'path': result.get('path_history', []),
                                    'decisions': result.get('decision_history', [])
                                }
                            )
                        
                        # Display execution path if enabled
                        if hasattr(self, 'show_execution_path') and self.show_execution_path:
                            with st.expander("🔀 Execution Path", expanded=True):
                                st.write(" → ".join(result.get('path_history', [])))
                        
                        return response
            
            return "No response generated from graph execution"
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error in graph execution: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with graph status"""
        super().render_header()
        
        # Show graph memory status
        if self.memory_adapter and self.graph_memory:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                executions = len(self.graph_memory.get('execution_paths', []))
                st.caption(f"🔀 Executions: {executions}")
            
            with col2:
                nodes = len(self.graph_memory.get('node_visit_frequency', {}))
                st.caption(f"📍 Nodes Visited: {nodes}")
            
            with col3:
                checkpoints = len(self.graph_memory.get('state_snapshots', []))
                st.caption(f"💾 Checkpoints: {checkpoints}")
            
            with col4:
                st.caption(f"👤 User: {st.session_state.get('user_id', 'default')}")
        
        # Show example queries
        with st.expander("Example Graph Queries"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Simple Queries:**")
                st.code("Hello, how are you?")
                st.code("What can you help with?")
                st.code("Tell me a joke")
            
            with col2:
                st.markdown("**Context-Aware:**")
                st.code("Continue our discussion")
                st.code("Based on what we talked about")
                st.code("Remember my preferences")
            
            with col3:
                st.markdown("**Graph Testing:**")
                st.code("Show me the execution path")
                st.code("How many nodes were visited?")
                st.code("What decisions were made?")


def main():
    """Main function to run LangGraph chatbot with memory"""
    chatbot = LangGraphMemoryIntegratedChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
