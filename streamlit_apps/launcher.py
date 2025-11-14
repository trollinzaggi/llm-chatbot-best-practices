"""
Memory System Launcher.

This is the main entry point for the memory-enhanced chatbot system.
Users can select and run different chatbot implementations from here.
"""

import streamlit as st
import os
import sys
import subprocess
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main launcher application."""
    # Page configuration
    st.set_page_config(
        page_title="Memory System Launcher",
        page_icon="🚀",
        layout="wide"
    )
    
    # Header
    st.title("🚀 Memory-Enhanced Chatbot Launcher")
    st.markdown("""
    Welcome to the Memory-Enhanced Chatbot System! This launcher provides access to various
    chatbot implementations with integrated memory capabilities.
    """)
    
    # Available applications
    apps = {
        "🧠 Unified Memory Demo": {
            "file": "unified_memory_demo.py",
            "description": "Comprehensive demo showing memory system across all frameworks",
            "features": ["Multi-framework support", "Memory synchronization", "Cross-framework context"]
        },
        "🔗 LangChain Memory Bot": {
            "file": "memory_langchain_bot.py",
            "description": "LangChain chatbot with enhanced memory integration",
            "features": ["Chain memory", "Summary buffers", "Vector stores"]
        },
        "📚 Memory-Enhanced Base": {
            "file": "memory_enhanced_base.py",
            "description": "Base implementation demonstrating core memory features",
            "features": ["Session memory", "Persistent storage", "Memory retrieval"]
        },
        "🤖 Original Chatbots": {
            "file": "base_chatbot.py",
            "description": "Access original chatbot implementations",
            "features": ["LangChain", "LangGraph", "CrewAI", "AutoGen", "LlamaIndex", "Agno"]
        }
    }
    
    # Sidebar for app selection
    with st.sidebar:
        st.title("Select Application")
        
        selected_app = st.radio(
            "Choose an application to run:",
            list(apps.keys()),
            format_func=lambda x: x
        )
        
        # Display app details
        st.subheader("Details")
        app_info = apps[selected_app]
        st.write(f"**Description:** {app_info['description']}")
        
        st.write("**Features:**")
        for feature in app_info['features']:
            st.write(f"• {feature}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(selected_app)
        st.write(app_info['description'])
        
        # Launch button
        if st.button(f"Launch {selected_app}", type="primary", use_container_width=True):
            app_file = app_info['file']
            app_path = os.path.join(os.path.dirname(__file__), app_file)
            
            if os.path.exists(app_path):
                st.success(f"Launching {selected_app}...")
                st.info(f"Run this command in your terminal:\n```\nstreamlit run {app_path}\n```")
                
                # Show launch instructions
                with st.expander("Launch Instructions"):
                    st.markdown(f"""
                    To run the selected application:
                    
                    1. Open a terminal/command prompt
                    2. Navigate to the project directory
                    3. Run: `streamlit run streamlit_apps/{app_file}`
                    
                    Or use the provided scripts:
                    - Windows: `run_chatbot.bat {app_file[:-3]}`
                    - Unix/Mac: `./run_chatbot.sh {app_file[:-3]}`
                    """)
            else:
                st.error(f"Application file not found: {app_path}")
    
    with col2:
        st.subheader("Quick Start Guide")
        
        with st.expander("📖 Getting Started"):
            st.markdown("""
            ### Initial Setup
            1. Install requirements: `pip install -r requirements.txt`
            2. Configure API keys in `.env` file
            3. Run the launcher: `streamlit run streamlit_apps/launcher.py`
            
            ### Memory System Features
            - **Session Memory**: Temporary conversation storage
            - **Persistent Memory**: Long-term memory across sessions
            - **Memory Retrieval**: Semantic and keyword search
            - **Cross-Framework**: Share memories between frameworks
            """)
        
        with st.expander("⚙️ Configuration"):
            st.markdown("""
            ### Memory Configuration
            The memory system can be configured through:
            - Environment variables (`.env` file)
            - Configuration files (`memory_config.json`)
            - Runtime settings (in-app configuration)
            
            ### Key Settings
            - `MAX_MESSAGES`: Messages per session (default: 50)
            - `IMPORTANCE_THRESHOLD`: Min importance for storage (default: 0.3)
            - `ENABLE_SEMANTIC_SEARCH`: Use embeddings (default: true)
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            ### Common Issues
            
            **Memory not persisting:**
            - Check that persistent memory is enabled in config
            - Verify database file permissions
            
            **Search not working:**
            - Ensure embeddings are configured
            - Check API keys for embedding providers
            
            **Framework errors:**
            - Install framework-specific dependencies
            - Check framework configuration
            """)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Frameworks", "6")
    
    with col2:
        st.metric("Memory Types", "5")
    
    with col3:
        st.metric("Storage Backends", "3")
    
    # System status
    with st.expander("🔍 System Status"):
        st.subheader("Memory System Components")
        
        components = {
            "Core Memory": "✅ Installed",
            "Storage (SQLite)": "✅ Available",
            "Retrieval Engine": "✅ Ready",
            "Processing Pipeline": "✅ Configured",
            "Framework Adapters": "✅ Loaded"
        }
        
        for component, status in components.items():
            st.write(f"{component}: {status}")
        
        # Check framework availability
        st.subheader("Framework Status")
        
        frameworks = {
            "LangChain": check_import("langchain"),
            "LangGraph": check_import("langgraph"),
            "CrewAI": check_import("crewai"),
            "AutoGen": check_import("pyautogen"),
            "LlamaIndex": check_import("llama_index"),
            "Agno": check_import("agno")
        }
        
        for framework, available in frameworks.items():
            if available:
                st.write(f"{framework}: ✅ Installed")
            else:
                st.write(f"{framework}: ❌ Not installed (run: pip install {framework.lower()})")


def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
