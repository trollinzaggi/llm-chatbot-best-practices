"""
CrewAI Chatbot with Memory Integration - Best Practices Implementation

This implementation shows the optimal way to integrate memory with CrewAI's
multi-agent system, enabling memory sharing between agents and task persistence.
"""
import streamlit as st
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_chatbot import BaseChatbot, ErrorHandler
from libraries.crewai.azure_crewai_setup import AzureCrewAISetup
from utils import setup_logger

# Memory system imports  
from memory_system import create_memory_adapter, CrewAIMemoryAdapter
from memory_system.core.models import MemoryFragment, FragmentType, MessageRole

logger = setup_logger(__name__)


class CrewAIMemoryIntegratedChatbot(BaseChatbot):
    """
    CrewAI chatbot with properly integrated memory system.
    
    Best Practices Demonstrated:
    1. Shared memory pool accessible by all agents
    2. Agent-specific memory tracks for specialization
    3. Task execution history and outcomes
    4. Inter-agent communication memory
    5. Crew learning and adaptation over time
    """
    
    def __init__(self):
        super().__init__(
            title="CrewAI Chatbot with Memory Integration", 
            description="""
            CrewAI with integrated memory that enables agents to share knowledge,
            learn from past tasks, and improve crew coordination over time.
            """
        )
        self.setup = None
        self.crew = None
        self.memory_adapter = None
        self.agent_memories = {}
        self.crew_memory = {}
        self.initialize_crewai()
        self.initialize_memory()
    
    def initialize_crewai(self):
        """Initialize CrewAI components"""
        if "crewai_setup" not in st.session_state:
            try:
                st.session_state.crewai_setup = AzureCrewAISetup()
                logger.info("Initialized CrewAI setup")
            except Exception as e:
                logger.error(f"Failed to initialize CrewAI: {str(e)}")
                st.session_state.crewai_setup = None
        
        self.setup = st.session_state.crewai_setup
    
    def initialize_memory(self):
        """Initialize memory system optimized for multi-agent collaboration"""
        if "crewai_memory_adapter" not in st.session_state:
            try:
                # Create CrewAI-specific memory adapter
                st.session_state.crewai_memory_adapter = CrewAIMemoryAdapter(
                    user_id=st.session_state.get('user_id', 'default'),
                    config={
                        'enable_shared_memory': True,
                        'enable_agent_memory': True,
                        'enable_task_memory': True,
                        'enable_communication_tracking': True,
                        'memory_sync_interval': 10,  # Sync every 10 interactions
                        'agent_memory_limit': 100,  # Per agent
                        'shared_memory_limit': 500,
                        'enable_learning': True
                    }
                )
                
                # Initialize crew-specific memory structures
                st.session_state.crew_memory = {
                    'shared_knowledge': {},  # Knowledge accessible to all agents
                    'agent_specializations': {},  # What each agent is good at
                    'task_outcomes': [],  # Historical task results
                    'communication_log': [],  # Inter-agent communications
                    'crew_performance': {},  # Overall crew metrics
                    'learned_patterns': {},  # Patterns learned from tasks
                    'delegation_history': []  # How tasks were delegated
                }
                
                # Initialize per-agent memory tracks
                st.session_state.agent_memories = {}
                
                # Start conversation with crew tracking
                st.session_state.conversation_id = st.session_state.crewai_memory_adapter.start_conversation(
                    title=f"CrewAI Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    metadata={'crew_type': 'adaptive'}
                )
                
                logger.info("Initialized CrewAI memory with multi-agent support")
                
            except Exception as e:
                logger.error(f"Failed to initialize memory: {str(e)}")
                st.session_state.crewai_memory_adapter = None
        
        self.memory_adapter = st.session_state.crewai_memory_adapter
        self.crew_memory = st.session_state.get('crew_memory', {})
        self.agent_memories = st.session_state.get('agent_memories', {})
    
    def create_memory_enhanced_crew(self, crew_type: str):
        """
        Create a crew with memory-enhanced agents.
        
        Best Practice: Each agent has access to shared memory and maintains personal memory
        """
        if not self.setup:
            return None
        
        crew = None
        
        if crew_type == "research":
            crew = self.create_research_crew_with_memory()
        elif crew_type == "development":
            crew = self.create_development_crew_with_memory()
        elif crew_type == "marketing":
            crew = self.create_marketing_crew_with_memory()
        
        # Initialize agent memories
        if crew and hasattr(crew, 'agents'):
            for agent in crew.agents:
                agent_name = agent.role if hasattr(agent, 'role') else str(agent)
                if agent_name not in self.agent_memories:
                    self.agent_memories[agent_name] = {
                        'tasks_completed': 0,
                        'specialization_areas': [],
                        'success_rate': 1.0,
                        'interaction_history': [],
                        'learned_skills': []
                    }
                
                # Inject memory access into agent
                self.inject_memory_into_agent(agent)
        
        return crew
    
    def inject_memory_into_agent(self, agent):
        """
        Inject memory capabilities into an agent.
        
        Best Practice: Agents can read shared memory and write to both shared and personal memory
        """
        agent_name = agent.role if hasattr(agent, 'role') else str(agent)
        
        # Store original execute method
        original_execute = agent.execute if hasattr(agent, 'execute') else None
        
        def execute_with_memory(task):
            # Access shared memory for context
            shared_context = self.get_shared_memory_context(task)
            
            # Access agent's personal memory
            personal_context = self.agent_memories.get(agent_name, {})
            
            # Enhance task with memory context
            enhanced_task = self.enhance_task_with_memory(task, shared_context, personal_context)
            
            # Execute original task
            result = original_execute(enhanced_task) if original_execute else str(agent(enhanced_task))
            
            # Update memories
            self.update_agent_memory(agent_name, task, result)
            self.update_shared_memory(agent_name, task, result)
            
            return result
        
        # Replace execute method
        if original_execute:
            agent.execute = execute_with_memory
    
    def create_research_crew_with_memory(self):
        """Create research crew with memory-enhanced agents"""
        if not self.setup:
            return None
        
        crew = self.setup.create_research_crew()
        
        # Enhance with specific memory capabilities
        for agent in crew.agents:
            role = agent.role if hasattr(agent, 'role') else 'researcher'
            
            # Add role-specific memory enhancements
            if 'researcher' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['information_gathering', 'source_validation']
            elif 'writer' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['content_creation', 'summarization']
            elif 'editor' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['quality_control', 'fact_checking']
        
        return crew
    
    def create_development_crew_with_memory(self):
        """Create development crew with memory-enhanced agents"""
        if not self.setup:
            return None
        
        crew = self.setup.create_development_crew()
        
        # Enhance with specific memory capabilities
        for agent in crew.agents:
            role = agent.role if hasattr(agent, 'role') else 'developer'
            
            # Add role-specific memory enhancements
            if 'architect' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['system_design', 'architecture_patterns']
            elif 'developer' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['coding', 'implementation']
            elif 'tester' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['testing', 'quality_assurance']
        
        return crew
    
    def create_marketing_crew_with_memory(self):
        """Create marketing crew with memory-enhanced agents"""
        if not self.setup:
            return None
        
        crew = self.setup.create_marketing_crew()
        
        # Enhance with specific memory capabilities
        for agent in crew.agents:
            role = agent.role if hasattr(agent, 'role') else 'marketer'
            
            # Add role-specific memory enhancements
            if 'analyst' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['market_analysis', 'trend_identification']
            elif 'strategist' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['strategy_development', 'planning']
            elif 'content' in role.lower():
                self.agent_memories[role]['specialization_areas'] = ['content_creation', 'messaging']
        
        return crew
    
    def get_shared_memory_context(self, task: Any) -> Dict[str, Any]:
        """
        Get relevant shared memory context for a task.
        
        Best Practice: Agents access shared knowledge base
        """
        context = {
            'shared_knowledge': self.crew_memory.get('shared_knowledge', {}),
            'recent_outcomes': self.crew_memory.get('task_outcomes', [])[-5:],
            'learned_patterns': self.crew_memory.get('learned_patterns', {})
        }
        
        # Retrieve relevant memories from adapter
        if self.memory_adapter:
            task_str = str(task)
            memories = self.memory_adapter.retrieve_relevant_memories(
                task_str,
                memory_types=[FragmentType.TASK_RESULT, FragmentType.AGENT_LEARNING],
                limit=3
            )
            context['relevant_memories'] = [m.content for m in memories]
        
        return context
    
    def enhance_task_with_memory(self, task: Any, shared_context: Dict, personal_context: Dict) -> Any:
        """
        Enhance task with memory context.
        
        Best Practice: Tasks include historical context and learned patterns
        """
        # Convert task to string if needed
        task_str = str(task)
        
        # Add memory context
        enhanced = f"{task_str}\n\n"
        
        if shared_context.get('relevant_memories'):
            enhanced += "Relevant past experiences:\n"
            for memory in shared_context['relevant_memories']:
                enhanced += f"- {memory}\n"
        
        if personal_context.get('learned_skills'):
            enhanced += f"\nAgent capabilities: {', '.join(personal_context['learned_skills'])}\n"
        
        return enhanced
    
    def update_agent_memory(self, agent_name: str, task: Any, result: Any):
        """
        Update agent's personal memory.
        
        Best Practice: Track agent performance and learning
        """
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = {}
        
        agent_memory = self.agent_memories[agent_name]
        
        # Update task counter
        agent_memory['tasks_completed'] = agent_memory.get('tasks_completed', 0) + 1
        
        # Track interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'task': str(task)[:200],
            'result': str(result)[:200],
            'success': bool(result)
        }
        
        if 'interaction_history' not in agent_memory:
            agent_memory['interaction_history'] = []
        agent_memory['interaction_history'].append(interaction)
        
        # Update success rate
        history = agent_memory['interaction_history']
        successes = sum(1 for i in history if i.get('success', False))
        agent_memory['success_rate'] = successes / len(history) if history else 1.0
        
        # Store in memory adapter
        if self.memory_adapter:
            self.memory_adapter.track_agent_interaction(agent_name, agent_name, str(result))
    
    def update_shared_memory(self, agent_name: str, task: Any, result: Any):
        """
        Update shared crew memory.
        
        Best Practice: Build collective knowledge base
        """
        # Add to task outcomes
        outcome = {
            'agent': agent_name,
            'task': str(task)[:200],
            'result': str(result)[:200],
            'timestamp': datetime.now().isoformat(),
            'success': bool(result)
        }
        self.crew_memory['task_outcomes'].append(outcome)
        
        # Extract and store knowledge
        if result:
            # Simple knowledge extraction (would be more sophisticated in practice)
            knowledge_key = f"{agent_name}_{len(self.crew_memory['task_outcomes'])}"
            self.crew_memory['shared_knowledge'][knowledge_key] = str(result)[:500]
        
        # Update crew performance metrics
        self.update_crew_performance_metrics()
    
    def update_crew_performance_metrics(self):
        """Track overall crew performance"""
        if not self.crew_memory.get('task_outcomes'):
            return
        
        outcomes = self.crew_memory['task_outcomes']
        total_tasks = len(outcomes)
        successful_tasks = sum(1 for o in outcomes if o.get('success', False))
        
        self.crew_memory['crew_performance'] = {
            'total_tasks': total_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 1.0,
            'active_agents': len(self.agent_memories),
            'knowledge_items': len(self.crew_memory.get('shared_knowledge', {}))
        }
    
    def track_agent_communication(self, from_agent: str, to_agent: str, message: str):
        """
        Track inter-agent communication.
        
        Best Practice: Monitor and learn from agent interactions
        """
        communication = {
            'from': from_agent,
            'to': to_agent,
            'message': message[:500],
            'timestamp': datetime.now().isoformat()
        }
        
        self.crew_memory['communication_log'].append(communication)
        
        # Store in memory adapter
        if self.memory_adapter:
            self.memory_adapter.track_agent_interaction(from_agent, to_agent, message)
    
    def delegate_task_with_memory(self, task: str) -> Dict[str, Any]:
        """
        Delegate task to best agent based on memory.
        
        Best Practice: Use historical performance to optimize delegation
        """
        best_agent = None
        best_score = 0.0
        
        for agent_name, agent_memory in self.agent_memories.items():
            # Calculate suitability score
            score = agent_memory.get('success_rate', 0.5)
            
            # Check specialization match
            task_lower = task.lower()
            for specialization in agent_memory.get('specialization_areas', []):
                if specialization.lower() in task_lower:
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_agent = agent_name
        
        delegation = {
            'task': task,
            'delegated_to': best_agent,
            'confidence': best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.crew_memory['delegation_history'].append(delegation)
        
        return delegation
    
    def render_sidebar(self):
        """Render sidebar with CrewAI-specific memory settings"""
        super().render_sidebar()
        
        with st.sidebar:
            st.divider()
            st.subheader("👥 Crew Memory Settings")
            
            # Crew selection
            crew_type = st.selectbox(
                "Crew Type",
                ["research", "development", "marketing"],
                help="Select crew configuration"
            )
            
            if st.button("Initialize Crew"):
                self.crew = self.create_memory_enhanced_crew(crew_type)
                st.success(f"Initialized {crew_type} crew with memory")
            
            # Agent statistics
            if self.agent_memories:
                st.caption("Agent Performance:")
                for agent_name, agent_memory in self.agent_memories.items():
                    with st.expander(f"📊 {agent_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tasks", agent_memory.get('tasks_completed', 0))
                        with col2:
                            success_rate = agent_memory.get('success_rate', 1.0)
                            st.metric("Success Rate", f"{success_rate:.1%}")
                        
                        if agent_memory.get('specialization_areas'):
                            st.caption("Specializations:")
                            for spec in agent_memory['specialization_areas']:
                                st.caption(f"• {spec}")
            
            # Crew performance
            if self.crew_memory.get('crew_performance'):
                st.caption("Crew Performance:")
                perf = self.crew_memory['crew_performance']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tasks", perf.get('total_tasks', 0))
                    st.metric("Active Agents", perf.get('active_agents', 0))
                
                with col2:
                    st.metric("Success Rate", f"{perf.get('success_rate', 1.0):.1%}")
                    st.metric("Knowledge Base", perf.get('knowledge_items', 0))
            
            # Memory sharing settings
            st.divider()
            st.caption("Memory Sharing:")
            
            share_memory = st.checkbox(
                "Enable Memory Sharing",
                value=True,
                help="Allow agents to share knowledge"
            )
            
            sync_interval = st.slider(
                "Sync Interval",
                min_value=1,
                max_value=20,
                value=10,
                help="Sync memory every N interactions"
            )
            
            # Memory operations
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Crew"):
                    self.save_crew_session()
            
            with col2:
                if st.button("📊 Export Knowledge"):
                    self.export_crew_knowledge()
            
            # Communication log
            if st.checkbox("Show Communications"):
                self.show_communications = True
            else:
                self.show_communications = False
    
    def save_crew_session(self):
        """Save crew session with all agent memories"""
        try:
            if self.memory_adapter:
                # Save with crew-specific metadata
                metadata = {
                    'crew_memory': self.crew_memory,
                    'agent_memories': self.agent_memories,
                    'crew_type': st.session_state.get('crew_type', 'research')
                }
                
                conversation_id = self.memory_adapter.save_conversation(metadata=metadata)
                
                # Store crew learning
                for pattern_key, pattern_value in self.crew_memory.get('learned_patterns', {}).items():
                    self.memory_adapter.store_crew_learning(pattern_key, pattern_value)
                
                st.success(f"Crew session saved! ID: {conversation_id}")
        except Exception as e:
            st.error(f"Failed to save session: {str(e)}")
    
    def export_crew_knowledge(self):
        """Export crew's collective knowledge"""
        if self.crew_memory:
            export_data = {
                'user_id': st.session_state.get('user_id', 'default'),
                'export_date': datetime.now().isoformat(),
                'shared_knowledge': self.crew_memory.get('shared_knowledge', {}),
                'agent_specializations': {
                    agent: memory.get('specialization_areas', [])
                    for agent, memory in self.agent_memories.items()
                },
                'task_outcomes': self.crew_memory.get('task_outcomes', [])[-20:],
                'crew_performance': self.crew_memory.get('crew_performance', {}),
                'learned_patterns': self.crew_memory.get('learned_patterns', {})
            }
            
            st.download_button(
                label="Download Crew Knowledge",
                data=json.dumps(export_data, indent=2),
                file_name=f"crew_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def get_response(self, prompt: str) -> str:
        """
        Get response from CrewAI with memory-enhanced execution.
        
        Best Practice: Execute crew tasks with full memory context and learning
        """
        try:
            if not self.crew:
                # Initialize default crew if not set
                self.crew = self.create_memory_enhanced_crew("research")
                
            if self.crew is None:
                return "Crew not initialized. Please check your configuration."
            
            # Add to memory
            if self.memory_adapter:
                self.memory_adapter.add_user_message(prompt)
            
            # Check if we should delegate based on history
            delegation = self.delegate_task_with_memory(prompt)
            
            if delegation and delegation['confidence'] > 0.7:
                st.info(f"Delegating to {delegation['delegated_to']} (confidence: {delegation['confidence']:.1%})")
            
            # Enhance prompt with crew memory
            if self.memory_adapter:
                # Get relevant crew memories
                crew_memories = self.memory_adapter.retrieve_relevant_memories(
                    prompt,
                    limit=3
                )
                
                enhanced_prompt = prompt
                if crew_memories:
                    enhanced_prompt += "\n\nRelevant crew knowledge:\n"
                    for memory in crew_memories:
                        enhanced_prompt += f"- {memory.content}\n"
            else:
                enhanced_prompt = prompt
            
            # Execute with crew
            result = self.crew.kickoff(inputs={'task': enhanced_prompt})
            
            # Process result
            response = str(result)
            
            # Update crew memories
            self.update_shared_memory('crew', enhanced_prompt, response)
            
            # Display communications if enabled
            if hasattr(self, 'show_communications') and self.show_communications:
                recent_comms = self.crew_memory.get('communication_log', [])[-5:]
                if recent_comms:
                    with st.expander("💬 Agent Communications", expanded=True):
                        for comm in recent_comms:
                            st.write(f"**{comm['from']} → {comm['to']}**: {comm['message'][:100]}...")
            
            # Add response to memory
            if self.memory_adapter:
                self.memory_adapter.add_assistant_message(
                    response,
                    metadata={
                        'crew_size': len(self.agent_memories),
                        'delegation': delegation
                    }
                )
            
            return response
            
        except Exception as e:
            error_msg = ErrorHandler.handle_api_error(e)
            logger.error(f"Error in crew execution: {str(e)}")
            return error_msg
    
    def render_header(self):
        """Render enhanced header with crew status"""
        super().render_header()
        
        # Show crew memory status
        if self.crew_memory:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                agents = len(self.agent_memories)
                st.caption(f"👥 Agents: {agents}")
            
            with col2:
                tasks = len(self.crew_memory.get('task_outcomes', []))
                st.caption(f"📋 Tasks: {tasks}")
            
            with col3:
                knowledge = len(self.crew_memory.get('shared_knowledge', {}))
                st.caption(f"🧠 Knowledge: {knowledge}")
            
            with col4:
                st.caption(f"👤 User: {st.session_state.get('user_id', 'default')}")
        
        # Show example queries
        with st.expander("Example Crew Tasks"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Research Tasks:**")
                st.code("Research AI trends")
                st.code("Analyze market data")
                st.code("Summarize findings")
            
            with col2:
                st.markdown("**Development Tasks:**")
                st.code("Design system architecture")
                st.code("Implement feature X")
                st.code("Test the solution")
            
            with col3:
                st.markdown("**Marketing Tasks:**")
                st.code("Create campaign strategy")
                st.code("Write content")
                st.code("Analyze performance")


def main():
    """Main function to run CrewAI chatbot with memory"""
    chatbot = CrewAIMemoryIntegratedChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
