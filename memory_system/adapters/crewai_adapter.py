"""
CrewAI framework memory adapter.

This module provides memory integration for CrewAI's multi-agent
collaboration framework with shared memory management.
"""

from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework, MemoryFragment, MemoryType


class CrewAIMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for CrewAI framework.
    
    This adapter provides shared memory management for multiple agents
    in a crew, enabling knowledge sharing and collaborative memory.
    """
    
    def __init__(self, crew=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CrewAI memory adapter.
        
        Args:
            crew: CrewAI crew instance
            config: Configuration dictionary
        """
        self.crew = crew
        super().__init__(config)
        
        # Agent-specific memory stores
        self.agent_memories: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_skills: Dict[str, Set[str]] = {}
        self.shared_knowledge: List[Dict[str, Any]] = []
        
        # Task tracking
        self.task_history: List[Dict[str, Any]] = []
        self.task_results: Dict[str, Any] = {}
        
        # Inter-agent communication tracking
        self.agent_interactions: List[Dict[str, Any]] = []
    
    def _initialize_framework(self) -> None:
        """Initialize CrewAI-specific components."""
        self.session_memory.conversation.framework = Framework.CREWAI
        
        # Initialize crew-specific tracking
        self.crew_performance = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_completion_time': 0,
            'agent_contributions': {}
        }
        
        # Initialize agent profiles
        if self.crew and hasattr(self.crew, 'agents'):
            for agent in self.crew.agents:
                agent_name = agent.role if hasattr(agent, 'role') else str(agent)
                self.agent_memories[agent_name] = []
                self.agent_skills[agent_name] = set()
                self.crew_performance['agent_contributions'][agent_name] = 0
    
    def inject_memory_context(self, input_text: str,
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context for crew processing.
        
        Args:
            input_text: Original input
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input with crew memory context
        """
        context_parts = []
        
        # Add shared knowledge
        if self.shared_knowledge:
            context_parts.append("Shared crew knowledge:")
            for knowledge in self.shared_knowledge[-5:]:
                context_parts.append(f"- {knowledge['content']}")
        
        # Add recent task results
        if self.task_results:
            context_parts.append("\nRecent task results:")
            recent_tasks = list(self.task_results.items())[-3:]
            for task_name, result in recent_tasks:
                context_parts.append(f"- {task_name}: {str(result)[:100]}")
        
        # Add relevant long-term memories
        if self.persistent_memory:
            memories = self.retrieve_relevant_memories(input_text, limit=3)
            if memories:
                context_parts.append("\nRelevant past information:")
                for memory in memories:
                    context_parts.append(f"- {memory.content}")
        
        if context_parts:
            return f"{chr(10).join(context_parts)}\n\nCurrent request: {input_text}"
        
        return input_text
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process crew response and extract agent interactions.
        
        Args:
            response: Crew output
            metadata: Optional metadata including agent interactions
            
        Returns:
            Processed response
        """
        # Track task completion
        if metadata and 'task' in metadata:
            task_info = {
                'task': metadata['task'],
                'result': response,
                'timestamp': datetime.now(),
                'agents_involved': metadata.get('agents', [])
            }
            self.task_history.append(task_info)
            
            # Store task result
            task_name = metadata['task'].get('name', 'unnamed_task')
            self.task_results[task_name] = response
            
            # Update performance metrics
            if metadata.get('success', True):
                self.crew_performance['tasks_completed'] += 1
            else:
                self.crew_performance['tasks_failed'] += 1
        
        # Track agent interactions
        if metadata and 'interactions' in metadata:
            for interaction in metadata['interactions']:
                self.agent_interactions.append({
                    'from_agent': interaction.get('from'),
                    'to_agent': interaction.get('to'),
                    'message': interaction.get('message', ''),
                    'timestamp': datetime.now()
                })
        
        # Extract and store shared knowledge
        self._extract_shared_knowledge(response, metadata)
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get CrewAI-specific context data.
        
        Returns:
            Dictionary with CrewAI-specific context
        """
        return {
            'crew_size': len(self.agent_memories),
            'shared_knowledge_count': len(self.shared_knowledge),
            'tasks_completed': self.crew_performance['tasks_completed'],
            'tasks_failed': self.crew_performance['tasks_failed'],
            'agent_interactions_count': len(self.agent_interactions),
            'active_agents': list(self.agent_memories.keys()),
            'agent_skills': {
                agent: list(skills)[:5]  # Top 5 skills per agent
                for agent, skills in self.agent_skills.items()
            }
        }
    
    def add_agent_memory(self, agent_name: str, memory_content: str,
                        memory_type: str = 'observation') -> None:
        """
        Add a memory for a specific agent.
        
        Args:
            agent_name: Name/role of the agent
            memory_content: Content of the memory
            memory_type: Type of memory (observation, learning, skill)
        """
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = []
            self.agent_skills[agent_name] = set()
        
        memory = {
            'content': memory_content,
            'type': memory_type,
            'timestamp': datetime.now()
        }
        
        self.agent_memories[agent_name].append(memory)
        
        # If it's a skill, add to agent skills
        if memory_type == 'skill':
            self.agent_skills[agent_name].add(memory_content)
        
        # Keep memory limited
        if len(self.agent_memories[agent_name]) > 50:
            self.agent_memories[agent_name] = self.agent_memories[agent_name][-50:]
    
    def get_agent_memory(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories for a specific agent.
        
        Args:
            agent_name: Name/role of the agent
            limit: Maximum memories to return
            
        Returns:
            List of agent memories
        """
        if agent_name not in self.agent_memories:
            return []
        
        return self.agent_memories[agent_name][-limit:]
    
    def share_memory_between_agents(self, from_agent: str, to_agent: str,
                                   memory_content: str) -> None:
        """
        Share a memory from one agent to another.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            memory_content: Memory to share
        """
        # Add to target agent's memory
        self.add_agent_memory(
            to_agent,
            f"Shared from {from_agent}: {memory_content}",
            'shared'
        )
        
        # Track the interaction
        self.agent_interactions.append({
            'from_agent': from_agent,
            'to_agent': to_agent,
            'message': memory_content,
            'type': 'memory_share',
            'timestamp': datetime.now()
        })
    
    def broadcast_to_crew(self, content: str, source: str = 'system') -> None:
        """
        Broadcast information to all agents in the crew.
        
        Args:
            content: Content to broadcast
            source: Source of the broadcast
        """
        for agent_name in self.agent_memories.keys():
            self.add_agent_memory(
                agent_name,
                f"Broadcast from {source}: {content}",
                'broadcast'
            )
        
        # Add to shared knowledge
        self.shared_knowledge.append({
            'content': content,
            'source': source,
            'timestamp': datetime.now(),
            'type': 'broadcast'
        })
    
    def process_crew_execution(self, task_description: str, 
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a complete crew execution with memory management.
        
        Args:
            task_description: Description of the task
            context: Optional context for the task
            
        Returns:
            Crew execution result
        """
        # Add task to memory
        self.add_user_message(task_description, {'task_context': context})
        
        # Inject memory context
        enhanced_task = self.inject_memory_context(task_description)
        
        # Execute crew (if crew instance is available)
        if self.crew:
            # Prepare inputs for crew
            inputs = {'task': enhanced_task}
            if context:
                inputs.update(context)
            
            # Execute crew
            result = self.crew.kickoff(inputs=inputs)
            
            # Process result
            response = str(result.raw_output) if hasattr(result, 'raw_output') else str(result)
            
            # Extract metadata
            metadata = {
                'task': {'name': task_description[:50]},
                'success': True,
                'agents': [agent.role for agent in self.crew.agents] if hasattr(self.crew, 'agents') else []
            }
        else:
            response = "Crew not configured for execution"
            metadata = {'task': {'name': task_description[:50]}, 'success': False}
        
        # Process response
        processed_response = self.process_response(response, metadata)
        
        # Add to conversation memory
        self.add_assistant_message(processed_response, metadata)
        
        # Update agent contributions
        for agent in metadata.get('agents', []):
            if agent in self.crew_performance['agent_contributions']:
                self.crew_performance['agent_contributions'][agent] += 1
        
        return processed_response
    
    def get_agent_collaboration_insights(self) -> Dict[str, Any]:
        """
        Get insights about agent collaboration patterns.
        
        Returns:
            Dictionary with collaboration insights
        """
        insights = {
            'most_active_agent': None,
            'collaboration_frequency': {},
            'knowledge_sharing_rate': 0,
            'task_distribution': {}
        }
        
        # Find most active agent
        if self.crew_performance['agent_contributions']:
            most_active = max(
                self.crew_performance['agent_contributions'].items(),
                key=lambda x: x[1]
            )
            insights['most_active_agent'] = most_active[0]
        
        # Analyze collaboration frequency
        collab_pairs = {}
        for interaction in self.agent_interactions:
            pair = f"{interaction['from_agent']}->{interaction['to_agent']}"
            collab_pairs[pair] = collab_pairs.get(pair, 0) + 1
        insights['collaboration_frequency'] = collab_pairs
        
        # Calculate knowledge sharing rate
        total_memories = sum(len(memories) for memories in self.agent_memories.values())
        shared_memories = sum(
            1 for memories in self.agent_memories.values()
            for m in memories if m.get('type') == 'shared'
        )
        if total_memories > 0:
            insights['knowledge_sharing_rate'] = shared_memories / total_memories
        
        # Task distribution
        for task in self.task_history:
            for agent in task.get('agents_involved', []):
                insights['task_distribution'][agent] = insights['task_distribution'].get(agent, 0) + 1
        
        return insights
    
    def consolidate_crew_knowledge(self) -> None:
        """Consolidate knowledge across all agents in the crew."""
        # Collect all agent memories
        all_memories = []
        for agent_name, memories in self.agent_memories.items():
            for memory in memories:
                all_memories.append({
                    'agent': agent_name,
                    'content': memory['content'],
                    'type': memory['type'],
                    'timestamp': memory['timestamp']
                })
        
        # Identify common patterns
        common_topics = {}
        for memory in all_memories:
            # Simple word frequency analysis
            words = memory['content'].lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    common_topics[word] = common_topics.get(word, 0) + 1
        
        # Create consolidated knowledge entries
        top_topics = sorted(common_topics.items(), key=lambda x: x[1], reverse=True)[:10]
        for topic, frequency in top_topics:
            if frequency > 3:  # Topic mentioned by multiple agents
                self.shared_knowledge.append({
                    'content': f"Common topic: {topic} (mentioned {frequency} times)",
                    'source': 'consolidation',
                    'timestamp': datetime.now(),
                    'type': 'pattern'
                })
        
        # Store in persistent memory if available
        if self.persistent_memory:
            for knowledge in self.shared_knowledge[-5:]:
                memory_fragment = MemoryFragment(
                    user_id=self.user_id,
                    fragment_type=MemoryType.CONTEXT,
                    content=knowledge['content'],
                    importance_score=0.7,
                    metadata={'source': 'crew_consolidation'}
                )
                self.persistent_memory.store_memory(memory_fragment)
    
    def _extract_shared_knowledge(self, response: str, metadata: Optional[Dict[str, Any]]) -> None:
        """
        Extract shared knowledge from crew execution.
        
        Args:
            response: Crew response
            metadata: Execution metadata
        """
        # Look for key findings or conclusions
        import re
        
        # Extract numbered points
        numbered_points = re.findall(r'\d+\.\s+([^\n]+)', response)
        for point in numbered_points[:5]:
            self.shared_knowledge.append({
                'content': point.strip(),
                'source': 'extraction',
                'timestamp': datetime.now(),
                'type': 'finding'
            })
        
        # Extract conclusions
        conclusion_patterns = [
            r'In conclusion[,:]?\s+([^.]+)',
            r'Therefore[,:]?\s+([^.]+)',
            r'The result is[,:]?\s+([^.]+)'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches[:2]:
                self.shared_knowledge.append({
                    'content': match.strip(),
                    'source': 'conclusion',
                    'timestamp': datetime.now(),
                    'type': 'conclusion'
                })
        
        # Keep shared knowledge limited
        if len(self.shared_knowledge) > 100:
            self.shared_knowledge = self.shared_knowledge[-100:]
    
    def export_crew_memory(self) -> Dict[str, Any]:
        """
        Export complete crew memory state.
        
        Returns:
            Dictionary with crew memory state
        """
        return {
            'agent_memories': self.agent_memories,
            'agent_skills': {agent: list(skills) for agent, skills in self.agent_skills.items()},
            'shared_knowledge': self.shared_knowledge,
            'task_history': self.task_history[-20:],  # Last 20 tasks
            'task_results': self.task_results,
            'agent_interactions': self.agent_interactions[-50:],  # Last 50 interactions
            'crew_performance': self.crew_performance
        }
    
    def import_crew_memory(self, memory_state: Dict[str, Any]) -> None:
        """
        Import crew memory state.
        
        Args:
            memory_state: Crew memory state to import
        """
        self.agent_memories = memory_state.get('agent_memories', {})
        self.agent_skills = {
            agent: set(skills)
            for agent, skills in memory_state.get('agent_skills', {}).items()
        }
        self.shared_knowledge = memory_state.get('shared_knowledge', [])
        self.task_history = memory_state.get('task_history', [])
        self.task_results = memory_state.get('task_results', {})
        self.agent_interactions = memory_state.get('agent_interactions', [])
        self.crew_performance = memory_state.get('crew_performance', self.crew_performance)
