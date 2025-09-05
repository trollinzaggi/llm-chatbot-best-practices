#!/bin/bash

# Azure OpenAI LLM POC Standards - Chatbot Launcher
# This script helps launch different Streamlit chatbot applications

echo "=================================="
echo "Azure OpenAI LLM POC Chatbot Launcher"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Dependencies not installed. Installing..."
    pip install -r requirements.txt
    echo "Dependencies installed."
fi

# Check if .env file exists
if [ ! -f "config/.env" ]; then
    echo ""
    echo "⚠️  Warning: config/.env file not found!"
    echo "Please copy config/.env.example to config/.env and configure your Azure OpenAI settings."
    echo ""
    cp config/.env.example config/.env
    echo "Created config/.env from template. Please edit it with your Azure OpenAI credentials."
    echo ""
fi

# Display menu
echo "Select a chatbot to launch:"
echo ""
echo "1) Agno Chatbot - Tool usage and structured outputs"
echo "2) LangChain Chatbot - Chain of prompts and memory"
echo "3) LangGraph Chatbot - Graph-based conversation flows"
echo "4) CrewAI Chatbot - Multi-agent collaboration"
echo "5) AutoGen Chatbot - Automated agent conversations"
echo "6) LlamaIndex Chatbot - RAG with document indexing"
echo "7) Exit"
echo ""

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Launching Agno Chatbot..."
        streamlit run streamlit_apps/agno_chatbot.py
        ;;
    2)
        echo "Launching LangChain Chatbot..."
        streamlit run streamlit_apps/langchain_chatbot.py
        ;;
    3)
        echo "Launching LangGraph Chatbot..."
        streamlit run streamlit_apps/langgraph_chatbot.py
        ;;
    4)
        echo "Launching CrewAI Chatbot..."
        streamlit run streamlit_apps/crewai_chatbot.py
        ;;
    5)
        echo "Launching AutoGen Chatbot..."
        streamlit run streamlit_apps/autogen_chatbot.py
        ;;
    6)
        echo "Launching LlamaIndex Chatbot..."
        streamlit run streamlit_apps/llama_index_chatbot.py
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
