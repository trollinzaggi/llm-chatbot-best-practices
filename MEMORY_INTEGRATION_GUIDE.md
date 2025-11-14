# Memory System Integration Guide

This guide explains how the unified memory system is integrated with the Streamlit chatbots, providing persistent memory, context retrieval, and cross-framework memory sharing capabilities.

## 🚀 Quick Start

### 1. Installation

```bash
# Install all requirements
pip install -r requirements.txt

# Or install core dependencies only
pip install streamlit python-dotenv dataclasses-json pyyaml numpy
```

### 2. Launch the System

```bash
# Start the launcher
streamlit run streamlit_apps/launcher.py

# Or run specific apps directly
streamlit run streamlit_apps/unified_memory_demo.py
streamlit run streamlit_apps/memory_langchain_bot.py
```

## 📁 Project Structure

```
llm-chatbot-best-practices/
├── memory_system/               # Core memory system
│   ├── core/                   # Core components
│   ├── storage/                # Storage backends
│   ├── adapters/               # Framework adapters
│   ├── retrieval/              # Memory retrieval
│   ├── processing/             # Memory processing
│   └── config/                 # Configuration
│
├── streamlit_apps/             # Streamlit applications
│   ├── launcher.py            # Main launcher
│   ├── unified_memory_demo.py # Unified demo
│   ├── memory_enhanced_base.py # Base implementation
│   └── memory_langchain_bot.py # LangChain example
│
└── config/                     # Configuration files
    └── memory_config.json     # Default configuration
```

## 🧠 Memory System Features

### Core Components

1. **Session Memory**
   - Temporary storage for active conversations
   - Automatic summarization and compression
   - Context window management

2. **Persistent Memory**
   - Long-term storage across sessions
   - SQLite database backend
   - Memory consolidation and optimization

3. **Memory Retrieval**
   - Semantic search using embeddings
   - Keyword-based search (TF-IDF, BM25)
   - Hybrid retrieval strategies

4. **Memory Processing**
   - Information extraction (entities, facts, preferences)
   - Text summarization (extractive and abstractive)
   - Memory consolidation and deduplication

### Framework Integration

Each framework has a dedicated adapter that provides:
- Framework-specific memory management
- Seamless integration with native memory systems
- Context enhancement with retrieved memories
- Cross-framework memory sharing

## 🎯 Usage Examples

### Basic Memory-Enhanced Chatbot

```python
from memory_system import create_memory_adapter

# Create adapter for LangChain
adapter = create_memory_adapter('langchain', user_id='user123')

# Start conversation
adapter.start_conversation(title="Product Discussion")

# Add messages
adapter.add_user_message("I need a laptop under $1000")
adapter.add_assistant_message("I can help you find budget laptops.")

# Retrieve relevant memories
memories = adapter.retrieve_relevant_memories("laptop specifications")

# Save conversation
adapter.save_conversation()
```

### Using the Unified Demo

The unified demo (`unified_memory_demo.py`) showcases:
- Switching between frameworks while maintaining context
- Cross-framework memory synchronization
- Memory visualization and analytics
- Import/export functionality

### Configuring Memory

Create a `memory_config.json`:

```json
{
  "session": {
    "max_messages": 100,
    "max_tokens": 8000
  },
  "persistent": {
    "enabled": true,
    "database_url": "conversations.db"
  },
  "retrieval": {
    "semantic_search": true,
    "top_k": 5
  }
}
```

Or use environment variables (`.env`):

```bash
MEMORY_MAX_MESSAGES=100
MEMORY_DATABASE_URL=conversations.db
MEMORY_USE_EMBEDDINGS=true
```

## 🔧 Advanced Features

### Memory Consolidation

The system automatically consolidates similar memories to optimize storage:

```python
from memory_system.processing import MemoryConsolidator

consolidator = MemoryConsolidator()
consolidated = consolidator.consolidate_memories(memories)
```

### Custom Retrieval Strategies

```python
from memory_system.retrieval import HybridRetriever

retriever = HybridRetriever({
    'semantic_weight': 0.6,
    'keyword_weight': 0.4
})
results = retriever.retrieve(query)
```

### Cross-Framework Context

Share memories between different frameworks:

```python
# In LangChain
langchain_adapter.add_user_message("My budget is $1000")

# In CrewAI - retrieves the budget context
crewai_adapter.retrieve_relevant_memories("budget")
```

## 📊 Memory Analytics

The system provides comprehensive analytics:
- Message and memory counts
- Topic extraction and tracking
- Memory importance scoring
- Access patterns and frequency

## 🛠️ Customization

### Adding New Storage Backends

```python
from memory_system.storage import BaseStorage

class MongoDBStorage(BaseStorage):
    def initialize(self):
        # MongoDB initialization
        pass
    
    def store_message(self, message):
        # Store in MongoDB
        pass
```

### Custom Framework Adapters

```python
from memory_system.adapters import BaseFrameworkAdapter

class CustomAdapter(BaseFrameworkAdapter):
    def _initialize_framework(self):
        # Framework-specific setup
        pass
    
    def inject_memory_context(self, input_text):
        # Add memory context
        pass
```

## 🐛 Troubleshooting

### Memory Not Persisting
- Check `persistent.enabled` in configuration
- Verify database file permissions
- Ensure `save_conversation()` is called

### Search Not Working
- Install sentence-transformers: `pip install sentence-transformers`
- Check embedding model availability
- Verify search configuration

### Framework Integration Issues
- Install framework-specific dependencies
- Check adapter initialization
- Verify API keys if required

## 📚 API Reference

### Core Classes

- `SessionMemory`: In-conversation memory management
- `PersistentMemory`: Long-term memory storage
- `BaseFrameworkAdapter`: Framework integration interface
- `RetrievalManager`: Memory retrieval coordination
- `ProcessingManager`: Memory processing pipeline

### Key Methods

- `add_message()`: Add message to memory
- `retrieve_relevant_memories()`: Get relevant memories
- `save_conversation()`: Persist conversation
- `consolidate_memories()`: Optimize memory storage
- `extract_information()`: Extract structured data

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional storage backends (PostgreSQL, MongoDB)
- More framework adapters
- Enhanced retrieval strategies
- Better visualization tools

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues or questions:
- Check the documentation in `/memory_system/README.md`
- Review examples in `/streamlit_apps/`
- Open an issue on GitHub

## 🎉 Features Highlights

✅ **Unified Memory System** - Single memory system for all frameworks  
✅ **Persistent Storage** - Conversations survive across sessions  
✅ **Semantic Search** - Find relevant memories using embeddings  
✅ **Auto Consolidation** - Automatic memory optimization  
✅ **Cross-Framework** - Share context between different LLM frameworks  
✅ **Easy Integration** - Simple API for adding memory to any chatbot  
✅ **Configurable** - Extensive configuration options  
✅ **Analytics** - Memory usage statistics and insights  

## 🗺️ Roadmap

- [ ] Vector database integration (Pinecone, Weaviate)
- [ ] Advanced embedding models
- [ ] Memory compression algorithms
- [ ] Multi-user support with isolation
- [ ] Memory encryption
- [ ] Cloud storage backends
- [ ] Real-time collaboration features
- [ ] Memory export formats (PDF, Markdown)
