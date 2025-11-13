# Azure OpenAI LLM POC Standards and Best Practices

This repository provides standardized setup and best practices for building LLM-based POCs using Azure OpenAI with various popular LLM libraries.

## Project Structure

```
azure-llm-poc-standards/
├── config/                  # Configuration files and environment setup
├── libraries/               # Library-specific implementations
│   ├── agno/               # Agno setup and examples
│   ├── langchain/          # LangChain setup and examples
│   ├── langgraph/          # LangGraph setup and examples
│   ├── crewai/             # CrewAI setup and examples
│   ├── autogen/            # AutoGen setup and examples
│   └── llama_index/        # LlamaIndex setup and examples
├── streamlit_apps/         # Streamlit chatbot applications
└── utils/                  # Shared utilities and helpers
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI Credentials

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your values:
```
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-01
```

### 3. Run Streamlit Applications

Each library has a corresponding Streamlit chatbot application:

```bash
# Agno chatbot
streamlit run streamlit_apps/agno_chatbot.py

# LangChain chatbot
streamlit run streamlit_apps/langchain_chatbot.py

# LangGraph chatbot
streamlit run streamlit_apps/langgraph_chatbot.py

# CrewAI chatbot
streamlit run streamlit_apps/crewai_chatbot.py

# AutoGen chatbot
streamlit run streamlit_apps/autogen_chatbot.py

# LlamaIndex chatbot
streamlit run streamlit_apps/llama_index_chatbot.py
```

## Library Implementations

### Agno
- Basic Azure OpenAI setup with Agno
- Tool integration examples
- Structured output handling

### LangChain
- Chain of prompts implementation
- Memory management
- Document loaders and retrievers

### LangGraph
- Graph-based conversation flow
- State management
- Conditional branching

### CrewAI
- Multi-agent system setup
- Task delegation
- Role-based interactions

### AutoGen
- Automated agent conversations
- Code execution capabilities
- Group chat scenarios

### LlamaIndex
- Document indexing and retrieval
- Query engines
- Chat engines with context

## Best Practices

1. **Environment Variables**: Always use environment variables for sensitive information
2. **Error Handling**: Implement proper error handling for API calls
3. **Logging**: Use structured logging for debugging
4. **Rate Limiting**: Implement rate limiting to avoid API throttling
5. **Testing**: Write unit tests for core functionality
6. **Documentation**: Document all custom implementations

## Requirements

See `requirements.txt` for full list of dependencies.

## Contributing

1. Create a feature branch
2. Follow the existing code structure
3. Add tests for new functionality
4. Update documentation
5. Submit a merge request

## License

[Your License Here]

## Support

For issues and questions, please create an issue in the GitLab repository.
