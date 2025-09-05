#!/usr/bin/env python3
"""
Setup script for Azure OpenAI LLM POC Standards

This script helps with initial setup and configuration validation.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("Azure OpenAI LLM POC Standards - Setup Script")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is suitable"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print("\nChecking virtual environment...")
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip")
    else:
        pip_path = Path("venv/bin/pip")
    
    if not pip_path.exists():
        print("❌ Virtual environment pip not found. Please activate venv first.")
        return False
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], capture_output=True)
    
    # Install requirements
    print("Installing requirements (this may take a few minutes)...")
    result = subprocess.run(
        [str(pip_path), "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    
    print("✅ Dependencies installed successfully")
    return True


def setup_environment_file():
    """Setup .env file from template"""
    print("\nSetting up environment configuration...")
    
    env_path = Path("config/.env")
    env_example_path = Path("config/.env.example")
    
    if not env_path.exists():
        if env_example_path.exists():
            print("Creating .env file from template...")
            env_path.write_text(env_example_path.read_text())
            print("✅ Created config/.env from template")
            print("\n⚠️  IMPORTANT: Please edit config/.env with your Azure OpenAI credentials:")
            print("   - AZURE_OPENAI_API_KEY")
            print("   - AZURE_OPENAI_ENDPOINT")
            print("   - AZURE_OPENAI_DEPLOYMENT_NAME")
            return False
        else:
            print("❌ Template file config/.env.example not found")
            return False
    else:
        print("✅ Environment file already exists")
        # Check if it's configured
        content = env_path.read_text()
        if "your-api-key-here" in content:
            print("\n⚠️  WARNING: config/.env contains placeholder values")
            print("   Please update it with your actual Azure OpenAI credentials")
            return False
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating necessary directories...")
    
    directories = [
        "logs",
        "coding",  # For AutoGen code execution
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✅ Created {directory}/ directory")
    
    return True


def validate_azure_config():
    """Validate Azure OpenAI configuration"""
    print("\nValidating Azure OpenAI configuration...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from config import config
        
        if not config.api_key or config.api_key == "your-api-key-here":
            print("❌ Azure OpenAI API key not configured")
            return False
        
        if not config.endpoint or "your-resource-name" in config.endpoint:
            print("❌ Azure OpenAI endpoint not configured")
            return False
        
        if not config.deployment_name or config.deployment_name == "your-deployment-name":
            print("❌ Azure OpenAI deployment name not configured")
            return False
        
        print("✅ Azure OpenAI configuration appears valid")
        print(f"   Endpoint: {config.endpoint}")
        print(f"   Deployment: {config.deployment_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to validate configuration: {str(e)}")
        return False


def test_imports():
    """Test if all major libraries can be imported"""
    print("\nTesting library imports...")
    
    libraries = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("crewai", "CrewAI"),
        ("autogen", "AutoGen"),
        ("llama_index", "LlamaIndex"),
        ("openai", "OpenAI SDK"),
    ]
    
    all_good = True
    for module, name in libraries:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {str(e)}")
            all_good = False
    
    return all_good


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("Setup Complete! Next Steps:")
    print("=" * 60)
    print()
    print("1. Edit config/.env with your Azure OpenAI credentials")
    print()
    print("2. Run a chatbot using one of these methods:")
    print("   - Use the launcher script: ./run_chatbot.sh (or run_chatbot.bat on Windows)")
    print("   - Run directly: streamlit run streamlit_apps/agno_chatbot.py")
    print()
    print("3. Access the chatbot at http://localhost:8501")
    print()
    print("Available Chatbots:")
    print("   • Agno - Tool usage and structured outputs")
    print("   • LangChain - Chain of prompts and memory")
    print("   • LangGraph - Graph-based conversation flows")
    print("   • CrewAI - Multi-agent collaboration")
    print("   • AutoGen - Automated agent conversations")
    print("   • LlamaIndex - RAG with document indexing")
    print()
    print("For more information, see README.md")


def main():
    """Main setup function"""
    print_header()
    
    # Track if everything is successful
    all_good = True
    config_ready = False
    
    # Run checks and setup
    if not check_python_version():
        all_good = False
    
    if not create_virtual_environment():
        all_good = False
    
    if not install_dependencies():
        all_good = False
    
    if not create_directories():
        all_good = False
    
    env_configured = setup_environment_file()
    if env_configured:
        config_ready = validate_azure_config()
    
    # Only test imports if dependencies were installed
    if all_good:
        test_imports()
    
    # Print summary
    print("\n" + "=" * 60)
    if all_good and config_ready:
        print("✅ Setup completed successfully!")
        print_next_steps()
    elif all_good and not config_ready:
        print("⚠️  Setup partially complete - Azure OpenAI configuration needed")
        print_next_steps()
    else:
        print("❌ Setup encountered errors. Please fix them and run again.")
    print("=" * 60)


if __name__ == "__main__":
    main()
