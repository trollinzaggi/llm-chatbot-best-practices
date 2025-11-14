"""
Configuration loader for the memory system.

This module provides utilities to load configuration from various sources
with proper precedence and merging logic.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from .memory_config import MemoryConfig
from .config_validator import validate_config_dict
from ..core.exceptions import ConfigurationError


class ConfigLoader:
    """
    Loader for memory system configuration from multiple sources.
    
    Precedence (highest to lowest):
    1. Environment variables
    2. User configuration file
    3. Project configuration file
    4. Default configuration
    """
    
    @staticmethod
    def load(config_paths: Optional[List[str]] = None,
             use_env: bool = True,
             use_defaults: bool = True,
             validate: bool = True) -> MemoryConfig:
        """
        Load configuration from multiple sources.
        
        Args:
            config_paths: List of configuration file paths to try
            use_env: Whether to use environment variables
            use_defaults: Whether to use default configuration
            validate: Whether to validate the final configuration
            
        Returns:
            Loaded MemoryConfig instance
        """
        # Start with defaults if enabled
        if use_defaults:
            config_dict = ConfigLoader.load_defaults()
        else:
            config_dict = {}
        
        # Load from files
        if config_paths:
            for path in config_paths:
                if os.path.exists(path):
                    file_config = ConfigLoader.load_from_file(path)
                    config_dict = ConfigLoader.merge_configs(config_dict, file_config)
        
        # Override with environment variables if enabled
        if use_env:
            env_config = ConfigLoader.load_from_env()
            config_dict = ConfigLoader.merge_configs(config_dict, env_config)
        
        # Validate if requested
        if validate:
            is_valid, errors = validate_config_dict(config_dict)
            if not is_valid:
                raise ConfigurationError("config", f"Invalid configuration: {errors}")
        
        # Create MemoryConfig instance
        return MemoryConfig.from_dict(config_dict)
    
    @staticmethod
    def load_defaults() -> Dict[str, Any]:
        """
        Load default configuration.
        
        Returns:
            Default configuration dictionary
        """
        # Try to load from default_config.json in the config directory
        config_dir = Path(__file__).parent
        default_file = config_dir / "default_config.json"
        
        if default_file.exists():
            with open(default_file, 'r') as f:
                return json.load(f)
        
        # Fallback to hardcoded defaults
        return MemoryConfig().to_dict()
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(filepath):
            raise ConfigurationError("filepath", f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                return json.load(f)
    
    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment
        """
        config = {}
        
        # Session configuration
        session_config = {}
        if 'MEMORY_MAX_MESSAGES' in os.environ:
            session_config['max_messages'] = int(os.environ['MEMORY_MAX_MESSAGES'])
        if 'MEMORY_MAX_TOKENS' in os.environ:
            session_config['max_tokens'] = int(os.environ['MEMORY_MAX_TOKENS'])
        if 'MEMORY_SUMMARIZE_AFTER' in os.environ:
            session_config['summarize_after'] = int(os.environ['MEMORY_SUMMARIZE_AFTER'])
        if 'MEMORY_COMPRESSION_RATIO' in os.environ:
            session_config['compression_ratio'] = float(os.environ['MEMORY_COMPRESSION_RATIO'])
        
        if session_config:
            config['session'] = session_config
        
        # Persistent configuration
        persistent_config = {}
        if 'MEMORY_ENABLED' in os.environ:
            persistent_config['enabled'] = os.environ['MEMORY_ENABLED'].lower() == 'true'
        if 'MEMORY_DATABASE_URL' in os.environ:
            persistent_config['database_url'] = os.environ['MEMORY_DATABASE_URL']
        if 'MEMORY_IMPORTANCE_THRESHOLD' in os.environ:
            persistent_config['importance_threshold'] = float(os.environ['MEMORY_IMPORTANCE_THRESHOLD'])
        if 'MEMORY_MAX_MEMORIES_PER_USER' in os.environ:
            persistent_config['max_memories_per_user'] = int(os.environ['MEMORY_MAX_MEMORIES_PER_USER'])
        if 'MEMORY_CONSOLIDATION_THRESHOLD' in os.environ:
            persistent_config['consolidation_threshold'] = float(os.environ['MEMORY_CONSOLIDATION_THRESHOLD'])
        if 'MEMORY_DEFAULT_EXPIRY_DAYS' in os.environ:
            persistent_config['default_expiry_days'] = int(os.environ['MEMORY_DEFAULT_EXPIRY_DAYS'])
        
        if persistent_config:
            config['persistent'] = persistent_config
        
        # Retrieval configuration
        retrieval_config = {}
        if 'MEMORY_SEMANTIC_SEARCH' in os.environ:
            retrieval_config['semantic_search'] = os.environ['MEMORY_SEMANTIC_SEARCH'].lower() == 'true'
        if 'MEMORY_TOP_K' in os.environ:
            retrieval_config['top_k'] = int(os.environ['MEMORY_TOP_K'])
        if 'MEMORY_SIMILARITY_THRESHOLD' in os.environ:
            retrieval_config['similarity_threshold'] = float(os.environ['MEMORY_SIMILARITY_THRESHOLD'])
        if 'MEMORY_USE_EMBEDDINGS' in os.environ:
            retrieval_config['use_embeddings'] = os.environ['MEMORY_USE_EMBEDDINGS'].lower() == 'true'
        if 'MEMORY_EMBEDDING_CACHE_SIZE' in os.environ:
            retrieval_config['embedding_cache_size'] = int(os.environ['MEMORY_EMBEDDING_CACHE_SIZE'])
        
        if retrieval_config:
            config['retrieval'] = retrieval_config
        
        # Processing configuration
        processing_config = {}
        if 'MEMORY_EXTRACT_ENTITIES' in os.environ:
            processing_config['extract_entities'] = os.environ['MEMORY_EXTRACT_ENTITIES'].lower() == 'true'
        if 'MEMORY_EXTRACT_FACTS' in os.environ:
            processing_config['extract_facts'] = os.environ['MEMORY_EXTRACT_FACTS'].lower() == 'true'
        if 'MEMORY_EXTRACT_PREFERENCES' in os.environ:
            processing_config['extract_preferences'] = os.environ['MEMORY_EXTRACT_PREFERENCES'].lower() == 'true'
        if 'MEMORY_PROCESSING_IMPORTANCE_THRESHOLD' in os.environ:
            processing_config['importance_threshold'] = float(os.environ['MEMORY_PROCESSING_IMPORTANCE_THRESHOLD'])
        if 'MEMORY_AUTO_SUMMARIZE' in os.environ:
            processing_config['auto_summarize'] = os.environ['MEMORY_AUTO_SUMMARIZE'].lower() == 'true'
        if 'MEMORY_AUTO_CONSOLIDATE' in os.environ:
            processing_config['auto_consolidate'] = os.environ['MEMORY_AUTO_CONSOLIDATE'].lower() == 'true'
        
        if processing_config:
            config['processing'] = processing_config
        
        # Global settings
        if 'MEMORY_DEBUG' in os.environ:
            config['debug_mode'] = os.environ['MEMORY_DEBUG'].lower() == 'true'
        if 'MEMORY_TELEMETRY' in os.environ:
            config['enable_telemetry'] = os.environ['MEMORY_TELEMETRY'].lower() == 'true'
        if 'MEMORY_AUTO_SAVE_INTERVAL' in os.environ:
            config['auto_save_interval'] = int(os.environ['MEMORY_AUTO_SAVE_INTERVAL'])
        if 'MEMORY_SYNC_ENABLED' in os.environ:
            config['memory_sync_enabled'] = os.environ['MEMORY_SYNC_ENABLED'].lower() == 'true'
        
        return config
    
    @staticmethod
    def load_from_dotenv(dotenv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a .env file.
        
        Args:
            dotenv_path: Path to .env file (searches for .env in current directory if None)
            
        Returns:
            Configuration dictionary
        """
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise ImportError("python-dotenv is required to load .env files")
        
        # Load .env file
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()  # Searches for .env in current directory
        
        # Now load from environment
        return ConfigLoader.load_from_env()
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to override with
            
        Returns:
            Merged configuration
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    @staticmethod
    def find_config_files() -> List[str]:
        """
        Find configuration files in standard locations.
        
        Returns:
            List of found configuration file paths
        """
        config_files = []
        
        # Check current directory
        for filename in ['memory_config.json', 'memory_config.yaml', '.memory_config.json']:
            if os.path.exists(filename):
                config_files.append(filename)
        
        # Check user home directory
        home = Path.home()
        for filename in ['.memory_config.json', '.memory_config.yaml']:
            path = home / filename
            if path.exists():
                config_files.append(str(path))
        
        # Check project directory
        project_config = Path('config') / 'memory_config.json'
        if project_config.exists():
            config_files.append(str(project_config))
        
        return config_files
    
    @staticmethod
    def auto_load() -> MemoryConfig:
        """
        Automatically load configuration from all available sources.
        
        Returns:
            Loaded MemoryConfig instance
        """
        # Find all available configuration files
        config_files = ConfigLoader.find_config_files()
        
        # Load with all sources
        return ConfigLoader.load(
            config_paths=config_files,
            use_env=True,
            use_defaults=True,
            validate=True
        )


def load_config(config_path: Optional[str] = None,
                use_env: bool = True,
                use_defaults: bool = True) -> MemoryConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Optional path to configuration file
        use_env: Whether to use environment variables
        use_defaults: Whether to use default configuration
        
    Returns:
        Loaded MemoryConfig instance
    """
    config_paths = [config_path] if config_path else ConfigLoader.find_config_files()
    
    return ConfigLoader.load(
        config_paths=config_paths,
        use_env=use_env,
        use_defaults=use_defaults,
        validate=True
    )
