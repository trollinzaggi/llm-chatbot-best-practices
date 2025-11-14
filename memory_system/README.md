# Unified Memory System for LLM Frameworks

A comprehensive memory management system that provides both session and persistent memory capabilities across multiple LLM frameworks.

## Features

- **Session Memory**: In-conversation memory with automatic summarization and compression
- **Persistent Memory**: Long-term storage with semantic search and consolidation
- **Framework Support**: Adapters for Agno, LangChain, LangGraph, CrewAI, AutoGen, and LlamaIndex
- **Flexible Storage**: SQLite built-in, with support for PostgreSQL and MongoDB
- **Smart Retrieval**: Embedding-based similarity search and context injection
- **Automatic Learning**: Extract entities, facts, preferences, and patterns

## Installation

```bash
# Install core memory system
pip install -r memory_system/requirements.txt

# Install framework-specific dependencies as needed
pip install langchain  # For LangChain support
pip install langgraph  # For LangGraph support
pip install crewai     # For CrewAI support
pip install pyautogen  # For AutoGen support
pip install llama-index # For LlamaIndex support
```

## Quick Start

### Basic Usage

```python
from memory_system import create_memory_adapter

# Create a memory adapter for your framework
memory = create_memory_adapter('langchain', user_id='user123')

# Start a conversation
memory.start_conversation(title="Product Discussion")

# Add messages
memory.add_user_message("I'm looking for a laptop under $1000")
memory.add_assistant_message("I can help you find laptops in your budget. What's your primary use case?")

# Get context for the LLM
context = memory.get_conversation_context()

# Save conversation when done
memory.save_conversation()
```

### Framework-Specific Examples

#### LangChain Integration

```python
from memory_system import LangChainMemoryAdapter
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# Create memory adapter
memory_adapter = LangChainMemoryAdapter(
    llm=OpenAI(),
    config={'memory_type': 'summary_buffer'}
)

# Create chain with memory
chain = memory_adapter.create_memory_chain(
    ConversationChain,
    llm=OpenAI(),
    verbose=True
)

# Process interaction
response = memory_adapter.process_chain_interaction(
    "Tell me about machine learning",
    chain
)
```

#### CrewAI Integration

```python
from memory_system import CrewAIMemoryAdapter
from crewai import Agent, Task, Crew

# Create memory adapter
memory_adapter = CrewAIMemoryAdapter(crew=my_crew)

# Process crew execution with memory
result = memory_adapter.process_crew_execution(
    "Research the latest AI trends",
    context={'focus': 'enterprise applications'}
)

# Get collaboration insights
insights = memory_adapter.get_agent_collaboration_insights()
```

#### AutoGen Integration

```python
from memory_system import AutoGenMemoryAdapter

# Create memory adapter
memory_adapter = AutoGenMemoryAdapter(agents=[agent1, agent2])

# Track agent interaction
memory_adapter.track_agent_interaction(
    "coder_agent", "reviewer_agent",
    "Here's my solution to the problem",
    message_type="code"
)

# Get agent expertise
expertise = memory_adapter.get_agent_expertise("coder_agent")

# Suggest best agent for task
best_agent = memory_adapter.suggest_agent_for_task("debug this Python code")
```

## Configuration

### Using Configuration Files

Create a `memory_config.json` file:

```json
{
  "session": {
    "max_messages": 100,
    "max_tokens": 16000
  },
  "persistent": {
    "enabled": true,
    "database_url": "my_memory.db"
  },
  "retrieval": {
    "top_k": 10,
    "use_embeddings": true
  }
}
```

Load configuration:

```python
from memory_system import load_config, create_memory_adapter

config = load_config('memory_config.json')
memory = create_memory_adapter('langchain', config=config)
```

### Using Environment Variables

Set environment variables in `.env`:

```bash
MEMORY_MAX_MESSAGES=100
MEMORY_DATABASE_URL=postgresql://localhost/memory_db
MEMORY_USE_EMBEDDINGS=true
```

### Dynamic Configuration

```python
from memory_system import ConfigManager

# Create configuration manager
config_manager = ConfigManager()

# Update configuration dynamically
config_manager.update_session_config(max_messages=200)
config_manager.enable_feature('semantic_search')

# Create snapshots
config_manager.create_snapshot('before_production')

# Restore if needed
config_manager.restore_snapshot('before_production')
```

## Advanced Features

### Memory Consolidation

```python
# Consolidate memories for a user
memory.consolidate_user_memories()

# Forget old or unimportant memories
memory.forget_memories(user_id='user123', criteria={'older_than_days': 30})
```

### Semantic Search

```python
# Search across all memories
results = memory.search("product recommendations", limit=5)

# Retrieve relevant memories for context
memories = memory.retrieve_relevant_memories("laptop specifications", limit=3)
```

### Multi-Agent Memory Sharing (CrewAI)

```python
# Share memory between agents
memory.share_memory_between_agents(
    "researcher", "writer",
    "Key finding: 73% increase in AI adoption"
)

# Broadcast to all agents
memory.broadcast_to_crew("Project deadline moved to Friday")
```

### Learning and Pattern Recognition (AutoGen)

```python
# Learn from code execution
memory.learn_from_code_execution(
    code="df = pd.read_csv('data.csv')",
    result="DataFrame with 1000 rows",
    success=True
)

# Get learned patterns
patterns = memory.successful_patterns
```

## Storage Backends

### SQLite (Default)

```python
from memory_system import create_memory_system

session, persistent, storage = create_memory_system(
    storage_type='sqlite',
    db_path='memory.db'
)
```

### PostgreSQL

```python
session, persistent, storage = create_memory_system(
    storage_type='postgresql',
    connection_string='postgresql://user:pass@localhost/db'
)
```

### MongoDB

```python
session, persistent, storage = create_memory_system(
    storage_type='mongodb',
    connection_string='mongodb://localhost:27017/',
    database='memory_db'
)
```

## Memory Types

- **Session Memory**: Temporary, conversation-specific
- **Episodic Memory**: Conversation episodes and summaries
- **Semantic Memory**: Facts, entities, and knowledge
- **Procedural Memory**: Learned patterns and skills
- **Working Memory**: Current context and active information

## Best Practices

1. **Regular Consolidation**: Consolidate memories periodically to maintain performance
2. **Importance Scoring**: Set appropriate importance thresholds for memory storage
3. **Memory Limits**: Configure reasonable limits to prevent unbounded growth
4. **Expiry Policies**: Set expiry dates for temporary information
5. **Privacy**: Implement proper user isolation and data protection

## API Reference

### Core Classes

- `SessionMemory`: Manages in-conversation memory
- `PersistentMemory`: Handles long-term memory storage
- `MemoryConfig`: Configuration management
- `BaseFrameworkAdapter`: Base class for framework adapters

### Key Methods

- `add_message()`: Add a message to memory
- `get_context()`: Get memory context for LLM
- `search()`: Search memory content
- `save_conversation()`: Persist conversation
- `retrieve_memories()`: Get relevant memories
- `consolidate_memories()`: Optimize memory storage

## Troubleshooting

### Memory Not Persisting

Check that persistent memory is enabled:
```python
config.persistent.enabled = True
```

### Search Not Working

Ensure embeddings are configured:
```python
config.retrieval.use_embeddings = True
```

### Framework Not Recognized

Check supported frameworks:
```python
from memory_system import get_supported_frameworks
print(get_supported_frameworks())
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub or contact the development team.
