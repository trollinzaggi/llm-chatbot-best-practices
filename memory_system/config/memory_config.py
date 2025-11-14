"""
Configuration settings for the memory system.

This module provides configuration management for all memory system components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SessionMemoryConfig:
    """Configuration for session memory."""
    max_messages: int = 50
    max_tokens: int = 8000
    summarize_after: int = 20
    compression_ratio: float = 0.3


@dataclass
class PersistentMemoryConfig:
    """Configuration for persistent memory."""
    enabled: bool = True
    database_url: str = "memory.db"
    importance_threshold: float = 0.3
    max_memories_per_user: int = 10000
    consolidation_threshold: float = 0.8
    default_expiry_days: int = 90


@dataclass
class RetrievalConfig:
    """Configuration for memory retrieval."""
    semantic_search: bool = True
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_embeddings: bool = True
    embedding_cache_size: int = 1000


@dataclass
class ProcessingConfig:
    """Configuration for memory processing."""
    extract_entities: bool = True
    extract_facts: bool = True
    extract_preferences: bool = True
    importance_threshold: float = 0.5
    auto_summarize: bool = True
    auto_consolidate: bool = True


@dataclass
class FrameworkConfig:
    """Framework-specific configuration."""
    langchain: Dict[str, Any] = field(default_factory=lambda: {
        'memory_type': 'conversation_summary_buffer',
        'memory_key': 'history',
        'return_messages': True
    })
    
    langgraph: Dict[str, Any] = field(default_factory=lambda: {
        'enable_memory_nodes': True,
        'track_state_history': True,
        'max_cycles': 3
    })
    
    crewai: Dict[str, Any] = field(default_factory=lambda: {
        'share_memories': True,
        'broadcast_important': True,
        'track_agent_expertise': True
    })
    
    autogen: Dict[str, Any] = field(default_factory=lambda: {
        'learn_patterns': True,
        'track_code_execution': True,
        'build_skill_library': True
    })
    
    llama_index: Dict[str, Any] = field(default_factory=lambda: {
        'index_type': 'vector_store',
        'hierarchical_indexing': True,
        'document_consolidation': True
    })
    
    agno: Dict[str, Any] = field(default_factory=lambda: {
        'track_tool_usage': True,
        'extract_context_variables': True
    })


@dataclass
class MemoryConfig:
    """Complete memory system configuration."""
    session: SessionMemoryConfig = field(default_factory=SessionMemoryConfig)
    persistent: PersistentMemoryConfig = field(default_factory=PersistentMemoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    frameworks: FrameworkConfig = field(default_factory=FrameworkConfig)
    
    # Global settings
    debug_mode: bool = False
    enable_telemetry: bool = False
    auto_save_interval: int = 10  # messages
    memory_sync_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'session': {
                'max_messages': self.session.max_messages,
                'max_tokens': self.session.max_tokens,
                'summarize_after': self.session.summarize_after,
                'compression_ratio': self.session.compression_ratio
            },
            'persistent': {
                'enabled': self.persistent.enabled,
                'database_url': self.persistent.database_url,
                'importance_threshold': self.persistent.importance_threshold,
                'max_memories_per_user': self.persistent.max_memories_per_user,
                'consolidation_threshold': self.persistent.consolidation_threshold,
                'default_expiry_days': self.persistent.default_expiry_days
            },
            'retrieval': {
                'semantic_search': self.retrieval.semantic_search,
                'top_k': self.retrieval.top_k,
                'similarity_threshold': self.retrieval.similarity_threshold,
                'use_embeddings': self.retrieval.use_embeddings,
                'embedding_cache_size': self.retrieval.embedding_cache_size
            },
            'processing': {
                'extract_entities': self.processing.extract_entities,
                'extract_facts': self.processing.extract_facts,
                'extract_preferences': self.processing.extract_preferences,
                'importance_threshold': self.processing.importance_threshold,
                'auto_summarize': self.processing.auto_summarize,
                'auto_consolidate': self.processing.auto_consolidate
            },
            'frameworks': {
                'langchain': self.frameworks.langchain,
                'langgraph': self.frameworks.langgraph,
                'crewai': self.frameworks.crewai,
                'autogen': self.frameworks.autogen,
                'llama_index': self.frameworks.llama_index,
                'agno': self.frameworks.agno
            },
            'debug_mode': self.debug_mode,
            'enable_telemetry': self.enable_telemetry,
            'auto_save_interval': self.auto_save_interval,
            'memory_sync_enabled': self.memory_sync_enabled
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update session config
        if 'session' in config_dict:
            for key, value in config_dict['session'].items():
                setattr(config.session, key, value)
        
        # Update persistent config
        if 'persistent' in config_dict:
            for key, value in config_dict['persistent'].items():
                setattr(config.persistent, key, value)
        
        # Update retrieval config
        if 'retrieval' in config_dict:
            for key, value in config_dict['retrieval'].items():
                setattr(config.retrieval, key, value)
        
        # Update processing config
        if 'processing' in config_dict:
            for key, value in config_dict['processing'].items():
                setattr(config.processing, key, value)
        
        # Update framework configs
        if 'frameworks' in config_dict:
            config.frameworks = FrameworkConfig(**config_dict['frameworks'])
        
        # Update global settings
        for key in ['debug_mode', 'enable_telemetry', 'auto_save_interval', 'memory_sync_enabled']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MemoryConfig':
        """Load configuration from JSON or YAML file."""
        import json
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON or YAML file."""
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                try:
                    import yaml
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML configuration files")
            else:
                json.dump(config_dict, f, indent=2)


# Default configuration instance
default_config = MemoryConfig()


def get_default_config() -> MemoryConfig:
    """Get the default memory configuration."""
    return default_config


def create_config_from_env() -> MemoryConfig:
    """Create configuration from environment variables."""
    import os
    
    config = MemoryConfig()
    
    # Session memory from env
    if 'MEMORY_MAX_MESSAGES' in os.environ:
        config.session.max_messages = int(os.environ['MEMORY_MAX_MESSAGES'])
    if 'MEMORY_MAX_TOKENS' in os.environ:
        config.session.max_tokens = int(os.environ['MEMORY_MAX_TOKENS'])
    
    # Persistent memory from env
    if 'MEMORY_DATABASE_URL' in os.environ:
        config.persistent.database_url = os.environ['MEMORY_DATABASE_URL']
    if 'MEMORY_ENABLED' in os.environ:
        config.persistent.enabled = os.environ['MEMORY_ENABLED'].lower() == 'true'
    
    # Retrieval from env
    if 'MEMORY_TOP_K' in os.environ:
        config.retrieval.top_k = int(os.environ['MEMORY_TOP_K'])
    if 'MEMORY_USE_EMBEDDINGS' in os.environ:
        config.retrieval.use_embeddings = os.environ['MEMORY_USE_EMBEDDINGS'].lower() == 'true'
    
    # Debug mode from env
    if 'MEMORY_DEBUG' in os.environ:
        config.debug_mode = os.environ['MEMORY_DEBUG'].lower() == 'true'
    
    return config
