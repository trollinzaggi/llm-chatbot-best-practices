@echo off
REM Azure OpenAI LLM POC Standards - Chatbot Launcher (Windows)
REM This script helps launch different Streamlit chatbot applications

echo ==================================
echo Azure OpenAI LLM POC Chatbot Launcher
echo ==================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Dependencies not installed. Installing...
    pip install -r requirements.txt
    echo Dependencies installed.
)

REM Check if .env file exists
if not exist "config\.env" (
    echo.
    echo Warning: config\.env file not found!
    echo Please copy config\.env.example to config\.env and configure your Azure OpenAI settings.
    echo.
    copy config\.env.example config\.env
    echo Created config\.env from template. Please edit it with your Azure OpenAI credentials.
    echo.
)

REM Display menu
echo Select a chatbot to launch:
echo.
echo 1) Agno Chatbot - Tool usage and structured outputs
echo 2) LangChain Chatbot - Chain of prompts and memory
echo 3) LangGraph Chatbot - Graph-based conversation flows
echo 4) CrewAI Chatbot - Multi-agent collaboration
echo 5) AutoGen Chatbot - Automated agent conversations
echo 6) LlamaIndex Chatbot - RAG with document indexing
echo 7) Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo Launching Agno Chatbot...
    streamlit run streamlit_apps/agno_chatbot.py
) else if "%choice%"=="2" (
    echo Launching LangChain Chatbot...
    streamlit run streamlit_apps/langchain_chatbot.py
) else if "%choice%"=="3" (
    echo Launching LangGraph Chatbot...
    streamlit run streamlit_apps/langgraph_chatbot.py
) else if "%choice%"=="4" (
    echo Launching CrewAI Chatbot...
    streamlit run streamlit_apps/crewai_chatbot.py
) else if "%choice%"=="5" (
    echo Launching AutoGen Chatbot...
    streamlit run streamlit_apps/autogen_chatbot.py
) else if "%choice%"=="6" (
    echo Launching LlamaIndex Chatbot...
    streamlit run streamlit_apps/llama_index_chatbot.py
) else if "%choice%"=="7" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
    exit /b 1
)
