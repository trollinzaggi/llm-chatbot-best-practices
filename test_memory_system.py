"""
Memory System Test Script.

This script tests the installation and basic functionality of the memory system.
Run this to verify everything is working correctly.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from memory_system import (
            create_memory_adapter,
            SessionMemory,
            PersistentMemory,
            MemoryConfig
        )
        print("✅ Core memory system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core memory system: {e}")
        return False
    
    try:
        from memory_system.adapters import (
            LangChainMemoryAdapter,
            LangGraphMemoryAdapter,
            CrewAIMemoryAdapter
        )
        print("✅ Framework adapters imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import adapters: {e}")
        return False
    
    try:
        from memory_system.retrieval import (
            SemanticRetriever,
            KeywordRetriever,
            HybridRetriever
        )
        print("✅ Retrieval modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import retrieval modules: {e}")
        return False
    
    try:
        from memory_system.processing import (
            TextSummarizer,
            InformationExtractor,
            MemoryConsolidator
        )
        print("✅ Processing modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import processing modules: {e}")
        return False
    
    return True


def test_session_memory():
    """Test session memory functionality."""
    print("\nTesting session memory...")
    
    try:
        from memory_system import SessionMemory
        from memory_system.core.models import MessageRole
        
        # Create session memory
        session = SessionMemory({'max_messages': 10})
        
        # Add messages
        session.add_message(MessageRole.USER, "Hello, my name is Alice")
        session.add_message(MessageRole.ASSISTANT, "Nice to meet you, Alice!")
        
        # Get context
        context = session.get_context()
        
        # Search
        results = session.search("Alice")
        
        print(f"✅ Session memory working - {len(context)} messages in context")
        print(f"   Found {len(results)} search results for 'Alice'")
        
        return True
    except Exception as e:
        print(f"❌ Session memory test failed: {e}")
        return False


def test_persistent_storage():
    """Test persistent storage functionality."""
    print("\nTesting persistent storage...")
    
    try:
        from memory_system import PersistentMemory, SQLiteStorage
        from memory_system.core.models import Conversation, Message, MessageRole
        
        # Create storage
        storage = SQLiteStorage("test_memory.db")
        
        # Create persistent memory
        persistent = PersistentMemory(storage_backend=storage)
        
        # Create and save conversation
        conv = Conversation(user_id="test_user")
        conv.add_message(MessageRole.USER, "Test message")
        
        conv_id = persistent.save_conversation(conv)
        
        print(f"✅ Persistent storage working - saved conversation {conv_id}")
        
        # Clean up
        import os
        if os.path.exists("test_memory.db"):
            os.remove("test_memory.db")
        
        return True
    except Exception as e:
        print(f"❌ Persistent storage test failed: {e}")
        return False


def test_retrieval():
    """Test retrieval functionality."""
    print("\nTesting retrieval...")
    
    try:
        from memory_system.retrieval import KeywordRetriever, RetrievalQuery
        from memory_system.core.models import Message, MessageRole
        
        # Create retriever
        retriever = KeywordRetriever()
        
        # Add test documents
        msg1 = Message(role=MessageRole.USER, content="I love Python programming")
        msg2 = Message(role=MessageRole.USER, content="Machine learning is fascinating")
        
        retriever.add_to_index(msg1)
        retriever.add_to_index(msg2)
        
        # Search
        query = RetrievalQuery(text="Python", limit=5)
        results = retriever.retrieve(query)
        
        print(f"✅ Retrieval working - found {len(results)} results")
        
        return True
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def test_processing():
    """Test processing functionality."""
    print("\nTesting processing...")
    
    try:
        from memory_system.processing import TextSummarizer, InformationExtractor
        
        # Test summarizer
        summarizer = TextSummarizer()
        text = "This is a long text that needs to be summarized. " * 20
        result = summarizer.process(text)
        
        print(f"✅ Summarizer working - compressed {result['original_length']} to {result['summary_length']} chars")
        
        # Test extractor
        extractor = InformationExtractor()
        test_text = "My name is John. I like pizza. My email is john@example.com"
        extraction = extractor.process(test_text)
        
        entities_found = sum(len(v) for v in extraction['entities'].values())
        print(f"✅ Extractor working - found {entities_found} entities")
        
        return True
    except Exception as e:
        print(f"❌ Processing test failed: {e}")
        return False


def test_framework_adapters():
    """Test framework adapters."""
    print("\nTesting framework adapters...")
    
    frameworks_tested = []
    
    # Test each framework adapter
    frameworks = ['agno', 'langchain', 'langgraph', 'crewai', 'autogen', 'llama_index']
    
    for framework in frameworks:
        try:
            from memory_system import create_memory_adapter
            
            adapter = create_memory_adapter(framework, user_id='test_user')
            adapter.start_conversation(title=f"Test {framework}")
            adapter.add_user_message("Test message")
            
            frameworks_tested.append(framework)
            print(f"✅ {framework.upper()} adapter working")
        except Exception as e:
            print(f"⚠️  {framework.upper()} adapter not available: {e}")
    
    if frameworks_tested:
        print(f"   Successfully tested {len(frameworks_tested)}/{len(frameworks)} adapters")
        return True
    else:
        print("❌ No framework adapters could be tested")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from memory_system import MemoryConfig, ConfigManager, load_config
        
        # Create config
        config = MemoryConfig()
        
        # Create manager
        manager = ConfigManager(config)
        
        # Update config
        manager.update_session_config(max_messages=100)
        
        # Create snapshot
        manager.create_snapshot("test")
        
        print("✅ Configuration system working")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MEMORY SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Session Memory", test_session_memory),
        ("Persistent Storage", test_persistent_storage),
        ("Retrieval", test_retrieval),
        ("Processing", test_processing),
        ("Framework Adapters", test_framework_adapters),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The memory system is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
