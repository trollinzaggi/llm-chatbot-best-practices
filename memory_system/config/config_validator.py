"""
Configuration validation for the memory system.

This module provides validation logic to ensure configuration values
are within acceptable ranges and properly formatted.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import fields
from ..core.exceptions import ConfigurationError


class ConfigValidator:
    """Validator for memory system configuration."""
    
    @staticmethod
    def validate_session_config(config) -> List[str]:
        """
        Validate session memory configuration.
        
        Args:
            config: SessionMemoryConfig instance
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate max_messages
        if config.max_messages < 1:
            errors.append("max_messages must be at least 1")
        elif config.max_messages > 10000:
            errors.append("max_messages cannot exceed 10000")
        
        # Validate max_tokens
        if config.max_tokens < 100:
            errors.append("max_tokens must be at least 100")
        elif config.max_tokens > 128000:
            errors.append("max_tokens cannot exceed 128000")
        
        # Validate summarize_after
        if config.summarize_after < 5:
            errors.append("summarize_after must be at least 5")
        elif config.summarize_after > config.max_messages:
            errors.append("summarize_after cannot exceed max_messages")
        
        # Validate compression_ratio
        if not 0.1 <= config.compression_ratio <= 1.0:
            errors.append("compression_ratio must be between 0.1 and 1.0")
        
        return errors
    
    @staticmethod
    def validate_persistent_config(config) -> List[str]:
        """
        Validate persistent memory configuration.
        
        Args:
            config: PersistentMemoryConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate database_url
        if not config.database_url:
            errors.append("database_url cannot be empty")
        
        # Validate importance_threshold
        if not 0.0 <= config.importance_threshold <= 1.0:
            errors.append("importance_threshold must be between 0.0 and 1.0")
        
        # Validate max_memories_per_user
        if config.max_memories_per_user < 100:
            errors.append("max_memories_per_user must be at least 100")
        elif config.max_memories_per_user > 1000000:
            errors.append("max_memories_per_user cannot exceed 1000000")
        
        # Validate consolidation_threshold
        if not 0.0 <= config.consolidation_threshold <= 1.0:
            errors.append("consolidation_threshold must be between 0.0 and 1.0")
        
        # Validate default_expiry_days
        if config.default_expiry_days < 1:
            errors.append("default_expiry_days must be at least 1")
        elif config.default_expiry_days > 3650:  # 10 years
            errors.append("default_expiry_days cannot exceed 3650")
        
        return errors
    
    @staticmethod
    def validate_retrieval_config(config) -> List[str]:
        """
        Validate retrieval configuration.
        
        Args:
            config: RetrievalConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate top_k
        if config.top_k < 1:
            errors.append("top_k must be at least 1")
        elif config.top_k > 100:
            errors.append("top_k cannot exceed 100")
        
        # Validate similarity_threshold
        if not 0.0 <= config.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")
        
        # Validate embedding_cache_size
        if config.embedding_cache_size < 0:
            errors.append("embedding_cache_size cannot be negative")
        elif config.embedding_cache_size > 100000:
            errors.append("embedding_cache_size cannot exceed 100000")
        
        return errors
    
    @staticmethod
    def validate_processing_config(config) -> List[str]:
        """
        Validate processing configuration.
        
        Args:
            config: ProcessingConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate importance_threshold
        if not 0.0 <= config.importance_threshold <= 1.0:
            errors.append("importance_threshold must be between 0.0 and 1.0")
        
        # Check for logical consistency
        if not any([config.extract_entities, config.extract_facts, config.extract_preferences]):
            errors.append("At least one extraction type must be enabled")
        
        return errors
    
    @staticmethod
    def validate_framework_config(config) -> List[str]:
        """
        Validate framework-specific configuration.
        
        Args:
            config: FrameworkConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate LangChain config
        if 'memory_type' in config.langchain:
            valid_types = ['buffer', 'summary_buffer', 'vector', 'conversation_summary_buffer']
            if config.langchain['memory_type'] not in valid_types:
                errors.append(f"Invalid LangChain memory_type: {config.langchain['memory_type']}")
        
        # Validate LangGraph config
        if 'max_cycles' in config.langgraph:
            if config.langgraph['max_cycles'] < 1 or config.langgraph['max_cycles'] > 100:
                errors.append("LangGraph max_cycles must be between 1 and 100")
        
        # Validate LlamaIndex config
        if 'index_type' in config.llama_index:
            valid_types = ['vector_store', 'list', 'tree', 'keyword']
            if config.llama_index['index_type'] not in valid_types:
                errors.append(f"Invalid LlamaIndex index_type: {config.llama_index['index_type']}")
        
        return errors
    
    @staticmethod
    def validate_memory_config(config) -> Tuple[bool, List[str]]:
        """
        Validate complete memory configuration.
        
        Args:
            config: MemoryConfig instance
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Validate session config
        errors = ConfigValidator.validate_session_config(config.session)
        if errors:
            all_errors.extend([f"Session: {e}" for e in errors])
        
        # Validate persistent config
        errors = ConfigValidator.validate_persistent_config(config.persistent)
        if errors:
            all_errors.extend([f"Persistent: {e}" for e in errors])
        
        # Validate retrieval config
        errors = ConfigValidator.validate_retrieval_config(config.retrieval)
        if errors:
            all_errors.extend([f"Retrieval: {e}" for e in errors])
        
        # Validate processing config
        errors = ConfigValidator.validate_processing_config(config.processing)
        if errors:
            all_errors.extend([f"Processing: {e}" for e in errors])
        
        # Validate framework config
        errors = ConfigValidator.validate_framework_config(config.frameworks)
        if errors:
            all_errors.extend([f"Framework: {e}" for e in errors])
        
        # Validate global settings
        if config.auto_save_interval < 1:
            all_errors.append("auto_save_interval must be at least 1")
        elif config.auto_save_interval > 1000:
            all_errors.append("auto_save_interval cannot exceed 1000")
        
        return len(all_errors) == 0, all_errors
    
    @staticmethod
    def suggest_fixes(errors: List[str]) -> List[str]:
        """
        Suggest fixes for validation errors.
        
        Args:
            errors: List of validation errors
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        for error in errors:
            if "must be at least" in error:
                field = error.split(":")[1].split()[0] if ":" in error else error.split()[0]
                suggestions.append(f"Increase {field} to meet minimum requirement")
            elif "cannot exceed" in error:
                field = error.split(":")[1].split()[0] if ":" in error else error.split()[0]
                suggestions.append(f"Decrease {field} to stay within maximum limit")
            elif "must be between" in error:
                field = error.split(":")[1].split()[0] if ":" in error else error.split()[0]
                suggestions.append(f"Adjust {field} to be within the specified range")
            elif "cannot be empty" in error:
                field = error.split(":")[1].split()[0] if ":" in error else error.split()[0]
                suggestions.append(f"Provide a value for {field}")
            elif "Invalid" in error:
                suggestions.append("Use one of the valid options specified in the error message")
            else:
                suggestions.append(f"Review and correct: {error}")
        
        return suggestions


def validate_config(config, raise_on_error: bool = False) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a configuration.
    
    Args:
        config: MemoryConfig instance
        raise_on_error: Whether to raise exception on validation error
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Raises:
        ConfigurationError: If raise_on_error is True and validation fails
    """
    is_valid, errors = ConfigValidator.validate_memory_config(config)
    
    if not is_valid and raise_on_error:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        suggestions = ConfigValidator.suggest_fixes(errors)
        if suggestions:
            error_msg += "\n\nSuggested fixes:\n" + "\n".join(suggestions)
        raise ConfigurationError("config", error_msg)
    
    return is_valid, errors


def validate_config_dict(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration dictionary before creating a MemoryConfig.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required sections
    required_sections = ['session', 'persistent', 'retrieval', 'processing']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")
    
    # Check data types
    if 'session' in config_dict and not isinstance(config_dict['session'], dict):
        errors.append("session must be a dictionary")
    
    if 'persistent' in config_dict and not isinstance(config_dict['persistent'], dict):
        errors.append("persistent must be a dictionary")
    
    # Check for unknown keys
    known_sections = ['session', 'persistent', 'retrieval', 'processing', 'frameworks',
                     'debug_mode', 'enable_telemetry', 'auto_save_interval', 'memory_sync_enabled']
    unknown_keys = set(config_dict.keys()) - set(known_sections)
    if unknown_keys:
        errors.append(f"Unknown configuration keys: {unknown_keys}")
    
    return len(errors) == 0, errors
