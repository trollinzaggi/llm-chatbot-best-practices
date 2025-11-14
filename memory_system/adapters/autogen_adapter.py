"""
AutoGen framework memory adapter.

This module provides memory integration for AutoGen's autonomous
multi-agent conversation framework with learning capabilities.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import re
from ..adapters.base_adapter import BaseFrameworkAdapter
from ..core.models import Message, MessageRole, Framework, MemoryFragment, MemoryType


class AutoGenMemoryAdapter(BaseFrameworkAdapter):
    """
    Memory adapter for AutoGen framework.
    
    This adapter provides memory management for autonomous agents
    with pattern learning and skill accumulation capabilities.
    """
    
    def __init__(self, agents: Optional[List] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AutoGen memory adapter.
        
        Args:
            agents: List of AutoGen agents
            config: Configuration dictionary
        """
        self.agents = agents or []
        super().__init__(config)
        
        # Agent-specific learning
        self.agent_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_skills: Dict[str, Dict[str, Any]] = {}
        self.conversation_patterns: List[Dict[str, Any]] = []
        
        # Code execution tracking
        self.code_snippets: List[Dict[str, Any]] = []
        self.execution_results: Dict[str, Any] = {}
        
        # Learning metrics
        self.learning_progress: Dict[str, float] = {}
        self.successful_patterns: List[str] = []
        self.failed_patterns: List[str] = []
    
    def _initialize_framework(self) -> None:
        """Initialize AutoGen-specific components."""
        self.session_memory.conversation.framework = Framework.AUTOGEN
        
        # Initialize agent profiles
        for agent in self.agents:
            agent_name = agent.name if hasattr(agent, 'name') else str(agent)
            self.agent_patterns[agent_name] = []
            self.agent_skills[agent_name] = {}
            self.learning_progress[agent_name] = 0.0
    
    def inject_memory_context(self, input_text: str,
                            max_context_messages: int = 10) -> str:
        """
        Inject memory context for AutoGen agents.
        
        Args:
            input_text: Original input
            max_context_messages: Maximum context messages
            
        Returns:
            Enhanced input with learned patterns
        """
        context_parts = []
        
        # Add successful patterns
        if self.successful_patterns:
            context_parts.append("Previously successful approaches:")
            for pattern in self.successful_patterns[-3:]:
                context_parts.append(f"- {pattern}")
        
        # Add relevant code snippets
        relevant_code = self._find_relevant_code(input_text)
        if relevant_code:
            context_parts.append("\nRelevant code from previous conversations:")
            for snippet in relevant_code[:2]:
                context_parts.append(f"```{snippet['language']}\n{snippet['code']}\n```")
        
        # Add learned skills
        all_skills = []
        for agent_skills in self.agent_skills.values():
            all_skills.extend(agent_skills.keys())
        
        if all_skills:
            context_parts.append("\nAvailable skills:")
            for skill in all_skills[:5]:
                context_parts.append(f"- {skill}")
        
        # Add long-term memories
        if self.persistent_memory:
            memories = self.retrieve_relevant_memories(input_text, limit=3)
            if memories:
                context_parts.append("\nRelevant past learnings:")
                for memory in memories:
                    context_parts.append(f"- {memory.content}")
        
        if context_parts:
            return f"{chr(10).join(context_parts)}\n\nCurrent task: {input_text}"
        
        return input_text
    
    def process_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process AutoGen response and extract learning patterns.
        
        Args:
            response: Agent response
            metadata: Optional metadata including agent interactions
            
        Returns:
            Processed response
        """
        # Extract code snippets
        code_blocks = self._extract_code_blocks(response)
        for code_block in code_blocks:
            self.code_snippets.append({
                'code': code_block['code'],
                'language': code_block.get('language', 'python'),
                'timestamp': datetime.now(),
                'agent': metadata.get('agent', 'unknown') if metadata else 'unknown'
            })
        
        # Track execution results
        if metadata and 'execution_result' in metadata:
            exec_result = metadata['execution_result']
            self.execution_results[datetime.now().isoformat()] = {
                'code': metadata.get('code', ''),
                'result': exec_result,
                'success': metadata.get('success', True),
                'agent': metadata.get('agent', 'unknown')
            }
            
            # Update patterns based on success/failure
            if metadata.get('success', True):
                pattern = self._extract_pattern(metadata.get('code', ''))
                if pattern:
                    self.successful_patterns.append(pattern)
            else:
                pattern = self._extract_pattern(metadata.get('code', ''))
                if pattern:
                    self.failed_patterns.append(pattern)
        
        # Extract conversation patterns
        self._analyze_conversation_pattern(response, metadata)
        
        # Extract and learn skills
        self._extract_and_learn_skills(response, metadata)
        
        return response
    
    def get_framework_specific_context(self) -> Dict[str, Any]:
        """
        Get AutoGen-specific context data.
        
        Returns:
            Dictionary with AutoGen-specific context
        """
        return {
            'agent_count': len(self.agents),
            'total_patterns_learned': sum(len(patterns) for patterns in self.agent_patterns.values()),
            'total_skills': sum(len(skills) for skills in self.agent_skills.values()),
            'code_snippets_count': len(self.code_snippets),
            'successful_patterns_count': len(self.successful_patterns),
            'failed_patterns_count': len(self.failed_patterns),
            'learning_progress': self.learning_progress,
            'execution_success_rate': self._calculate_success_rate()
        }
    
    def track_agent_interaction(self, sender_name: str, receiver_name: str,
                              message: str, message_type: str = 'chat') -> None:
        """
        Track interaction between agents.
        
        Args:
            sender_name: Sender agent name
            receiver_name: Receiver agent name
            message: Message content
            message_type: Type of message (chat, code, result)
        """
        interaction = {
            'sender': sender_name,
            'receiver': receiver_name,
            'message': message,
            'type': message_type,
            'timestamp': datetime.now()
        }
        
        # Add to conversation patterns
        self.conversation_patterns.append(interaction)
        
        # Learn from interaction
        self._learn_from_interaction(sender_name, receiver_name, message, message_type)
        
        # Keep patterns limited
        if len(self.conversation_patterns) > 100:
            self.conversation_patterns = self.conversation_patterns[-100:]
    
    def process_group_chat(self, initial_message: str, max_rounds: int = 10) -> str:
        """
        Process a group chat with memory management.
        
        Args:
            initial_message: Initial message to start the chat
            max_rounds: Maximum rounds of conversation
            
        Returns:
            Final result from the group chat
        """
        # Add initial message to memory
        self.add_user_message(initial_message)
        
        # Enhance with learned context
        enhanced_message = self.inject_memory_context(initial_message)
        
        # Track the conversation flow
        conversation_flow = []
        
        # Simulate group chat execution (actual implementation would use AutoGen's GroupChat)
        result = f"Group chat result for: {enhanced_message}"
        
        # Process result
        metadata = {
            'type': 'group_chat',
            'rounds': max_rounds,
            'agents': [agent.name if hasattr(agent, 'name') else str(agent) for agent in self.agents]
        }
        processed_result = self.process_response(result, metadata)
        
        # Add result to memory
        self.add_assistant_message(processed_result, metadata)
        
        return processed_result
    
    def learn_from_code_execution(self, code: str, result: Any, success: bool) -> None:
        """
        Learn from code execution results.
        
        Args:
            code: Executed code
            result: Execution result
            success: Whether execution was successful
        """
        # Extract pattern from code
        pattern = self._extract_pattern(code)
        
        if success:
            # Add to successful patterns
            if pattern and pattern not in self.successful_patterns:
                self.successful_patterns.append(pattern)
            
            # Extract skill if applicable
            skill = self._identify_skill_from_code(code)
            if skill:
                for agent in self.agents:
                    agent_name = agent.name if hasattr(agent, 'name') else str(agent)
                    if agent_name not in self.agent_skills:
                        self.agent_skills[agent_name] = {}
                    self.agent_skills[agent_name][skill['name']] = skill
        else:
            # Add to failed patterns
            if pattern and pattern not in self.failed_patterns:
                self.failed_patterns.append(pattern)
        
        # Store execution result
        self.execution_results[datetime.now().isoformat()] = {
            'code': code,
            'result': str(result),
            'success': success,
            'pattern': pattern
        }
    
    def get_agent_expertise(self, agent_name: str) -> Dict[str, Any]:
        """
        Get expertise profile for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary with agent expertise
        """
        return {
            'skills': self.agent_skills.get(agent_name, {}),
            'patterns': self.agent_patterns.get(agent_name, []),
            'learning_progress': self.learning_progress.get(agent_name, 0.0),
            'successful_interactions': self._count_successful_interactions(agent_name),
            'specialization': self._identify_specialization(agent_name)
        }
    
    def suggest_agent_for_task(self, task_description: str) -> Optional[str]:
        """
        Suggest the best agent for a given task based on learned expertise.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Name of the suggested agent or None
        """
        best_agent = None
        best_score = 0
        
        for agent_name in self.agent_skills.keys():
            score = 0
            
            # Check skills match
            for skill_name, skill_data in self.agent_skills[agent_name].items():
                if any(keyword in task_description.lower() for keyword in skill_data.get('keywords', [])):
                    score += 2
            
            # Check pattern history
            for pattern in self.agent_patterns.get(agent_name, []):
                if pattern.get('type') == 'success':
                    score += 1
            
            # Consider learning progress
            score += self.learning_progress.get(agent_name, 0.0)
            
            if score > best_score:
                best_score = score
                best_agent = agent_name
        
        return best_agent
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks from text."""
        code_blocks = []
        
        # Match code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            code_blocks.append({
                'language': language or 'python',
                'code': code.strip()
            })
        
        return code_blocks
    
    def _find_relevant_code(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant code snippets for a query."""
        relevant = []
        query_lower = query.lower()
        
        for snippet in self.code_snippets:
            code_lower = snippet['code'].lower()
            # Simple relevance check
            if any(word in code_lower for word in query_lower.split() if len(word) > 3):
                relevant.append(snippet)
        
        return relevant[:3]  # Return top 3 relevant snippets
    
    def _extract_pattern(self, code: str) -> Optional[str]:
        """Extract pattern from code."""
        # Simple pattern extraction based on structure
        lines = code.split('\n')
        
        # Look for function definitions
        for line in lines:
            if line.strip().startswith('def '):
                return f"function: {line.strip()}"
            elif line.strip().startswith('class '):
                return f"class: {line.strip()}"
            elif 'import ' in line:
                return f"import: {line.strip()}"
        
        # Look for common patterns
        if 'for ' in code:
            return "iteration pattern"
        elif 'if ' in code:
            return "conditional pattern"
        elif 'try:' in code:
            return "error handling pattern"
        
        return None
    
    def _identify_skill_from_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Identify skill from code."""
        skill = None
        
        # Check for data processing
        if any(lib in code for lib in ['pandas', 'numpy', 'csv']):
            skill = {
                'name': 'data_processing',
                'keywords': ['data', 'pandas', 'numpy', 'csv', 'dataframe'],
                'confidence': 0.8
            }
        # Check for web scraping
        elif any(lib in code for lib in ['requests', 'beautifulsoup', 'selenium']):
            skill = {
                'name': 'web_scraping',
                'keywords': ['web', 'scraping', 'requests', 'html', 'api'],
                'confidence': 0.8
            }
        # Check for file operations
        elif any(op in code for op in ['open(', 'read(', 'write(', 'with open']):
            skill = {
                'name': 'file_operations',
                'keywords': ['file', 'read', 'write', 'io'],
                'confidence': 0.7
            }
        
        return skill
    
    def _analyze_conversation_pattern(self, response: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Analyze and store conversation patterns."""
        pattern = {
            'response_length': len(response),
            'has_code': bool(self._extract_code_blocks(response)),
            'timestamp': datetime.now()
        }
        
        if metadata:
            pattern['agent'] = metadata.get('agent', 'unknown')
            pattern['interaction_type'] = metadata.get('type', 'standard')
        
        # Add to agent-specific patterns
        if pattern.get('agent') in self.agent_patterns:
            self.agent_patterns[pattern['agent']].append(pattern)
    
    def _extract_and_learn_skills(self, response: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Extract and learn skills from response."""
        # Look for skill indicators
        skill_indicators = [
            r'I can (.*?)[\.\,]',
            r'I know how to (.*?)[\.\,]',
            r'I\'ve learned (.*?)[\.\,]',
            r'Successfully (.*?)[\.\,]'
        ]
        
        for pattern in skill_indicators:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                skill_description = match.strip()
                if metadata and 'agent' in metadata:
                    agent_name = metadata['agent']
                    if agent_name in self.agent_skills:
                        skill_key = skill_description[:30]  # Truncate for key
                        self.agent_skills[agent_name][skill_key] = {
                            'description': skill_description,
                            'learned_at': datetime.now(),
                            'confidence': 0.6
                        }
    
    def _learn_from_interaction(self, sender: str, receiver: str, 
                               message: str, message_type: str) -> None:
        """Learn from agent interaction."""
        # Update learning progress
        if sender in self.learning_progress:
            self.learning_progress[sender] += 0.1
        
        # Store interaction pattern
        if sender in self.agent_patterns:
            self.agent_patterns[sender].append({
                'type': 'interaction',
                'receiver': receiver,
                'message_type': message_type,
                'timestamp': datetime.now()
            })
    
    def _calculate_success_rate(self) -> float:
        """Calculate code execution success rate."""
        if not self.execution_results:
            return 0.0
        
        successful = sum(1 for result in self.execution_results.values() if result.get('success', False))
        total = len(self.execution_results)
        
        return successful / total if total > 0 else 0.0
    
    def _count_successful_interactions(self, agent_name: str) -> int:
        """Count successful interactions for an agent."""
        count = 0
        for pattern in self.agent_patterns.get(agent_name, []):
            if pattern.get('type') == 'success':
                count += 1
        return count
    
    def _identify_specialization(self, agent_name: str) -> Optional[str]:
        """Identify agent specialization based on skills."""
        skills = self.agent_skills.get(agent_name, {})
        
        if not skills:
            return None
        
        # Count skill categories
        categories = {}
        for skill_name in skills.keys():
            if 'data' in skill_name.lower():
                categories['data_science'] = categories.get('data_science', 0) + 1
            elif 'code' in skill_name.lower() or 'function' in skill_name.lower():
                categories['programming'] = categories.get('programming', 0) + 1
            elif 'web' in skill_name.lower() or 'api' in skill_name.lower():
                categories['web_development'] = categories.get('web_development', 0) + 1
        
        if categories:
            return max(categories, key=categories.get)
        
        return None
    
    def export_learning_state(self) -> Dict[str, Any]:
        """Export complete learning state."""
        return {
            'agent_patterns': self.agent_patterns,
            'agent_skills': self.agent_skills,
            'conversation_patterns': self.conversation_patterns[-50:],
            'code_snippets': self.code_snippets[-20:],
            'execution_results': dict(list(self.execution_results.items())[-20:]),
            'learning_progress': self.learning_progress,
            'successful_patterns': self.successful_patterns,
            'failed_patterns': self.failed_patterns
        }
    
    def import_learning_state(self, state: Dict[str, Any]) -> None:
        """Import learning state."""
        self.agent_patterns = state.get('agent_patterns', {})
        self.agent_skills = state.get('agent_skills', {})
        self.conversation_patterns = state.get('conversation_patterns', [])
        self.code_snippets = state.get('code_snippets', [])
        self.execution_results = state.get('execution_results', {})
        self.learning_progress = state.get('learning_progress', {})
        self.successful_patterns = state.get('successful_patterns', [])
        self.failed_patterns = state.get('failed_patterns', [])
