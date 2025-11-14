"""
Configuration manager for dynamic configuration updates.

This module provides a manager for handling configuration updates,
persistence, and runtime modifications.
"""

import json
import os
from typing import Dict, Any, Optional, Callable, List
from threading import Lock
from datetime import datetime
from pathlib import Path
from .memory_config import MemoryConfig
from .config_validator import validate_config, validate_config_dict
from ..core.exceptions import ConfigurationError


class ConfigManager:
    """
    Manager for memory system configuration with support for
    dynamic updates, persistence, and change tracking.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None,
                 config_file: Optional[str] = None,
                 auto_save: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial configuration (uses default if None)
            config_file: Path to configuration file for persistence
            auto_save: Whether to automatically save on changes
        """
        self.config = config or MemoryConfig()
        self.config_file = config_file
        self.auto_save = auto_save
        
        # Thread safety
        self._lock = Lock()
        
        # Change tracking
        self._change_history: List[Dict[str, Any]] = []
        self._change_callbacks: List[Callable] = []
        
        # Configuration snapshots
        self._snapshots: Dict[str, MemoryConfig] = {}
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def get_config(self) -> MemoryConfig:
        """
        Get current configuration.
        
        Returns:
            Current MemoryConfig instance
        """
        with self._lock:
            return self.config
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
            validate: Whether to validate changes before applying
            
        Returns:
            True if update was successful
            
        Raises:
            ConfigurationError: If validation fails
        """
        with self._lock:
            # Create a copy for rollback
            original_config = self._create_config_copy()
            
            try:
                # Apply updates
                self._apply_updates(updates)
                
                # Validate if requested
                if validate:
                    is_valid, errors = validate_config(self.config)
                    if not is_valid:
                        # Rollback changes
                        self.config = original_config
                        raise ConfigurationError("config", f"Validation failed: {errors}")
                
                # Track change
                self._track_change(updates)
                
                # Notify callbacks
                self._notify_callbacks(updates)
                
                # Auto-save if enabled
                if self.auto_save and self.config_file:
                    self.save_to_file(self.config_file)
                
                return True
                
            except Exception as e:
                # Rollback on any error
                self.config = original_config
                raise e
    
    def update_session_config(self, **kwargs) -> bool:
        """
        Update session memory configuration.
        
        Args:
            **kwargs: Session configuration parameters
            
        Returns:
            True if update was successful
        """
        updates = {'session': kwargs}
        return self.update_config(updates)
    
    def update_persistent_config(self, **kwargs) -> bool:
        """
        Update persistent memory configuration.
        
        Args:
            **kwargs: Persistent configuration parameters
            
        Returns:
            True if update was successful
        """
        updates = {'persistent': kwargs}
        return self.update_config(updates)
    
    def update_framework_config(self, framework: str, **kwargs) -> bool:
        """
        Update framework-specific configuration.
        
        Args:
            framework: Framework name
            **kwargs: Framework configuration parameters
            
        Returns:
            True if update was successful
        """
        updates = {'frameworks': {framework: kwargs}}
        return self.update_config(updates)
    
    def enable_feature(self, feature: str) -> bool:
        """
        Enable a specific feature.
        
        Args:
            feature: Feature name to enable
            
        Returns:
            True if feature was enabled
        """
        feature_map = {
            'persistent_memory': ('persistent', 'enabled', True),
            'semantic_search': ('retrieval', 'semantic_search', True),
            'embeddings': ('retrieval', 'use_embeddings', True),
            'entity_extraction': ('processing', 'extract_entities', True),
            'fact_extraction': ('processing', 'extract_facts', True),
            'preference_extraction': ('processing', 'extract_preferences', True),
            'auto_summarize': ('processing', 'auto_summarize', True),
            'auto_consolidate': ('processing', 'auto_consolidate', True),
            'debug_mode': ('debug_mode', None, True),
            'telemetry': ('enable_telemetry', None, True),
            'memory_sync': ('memory_sync_enabled', None, True)
        }
        
        if feature not in feature_map:
            raise ConfigurationError("feature", f"Unknown feature: {feature}")
        
        section, key, value = feature_map[feature]
        
        if key:
            updates = {section: {key: value}}
        else:
            updates = {section: value}
        
        return self.update_config(updates)
    
    def disable_feature(self, feature: str) -> bool:
        """
        Disable a specific feature.
        
        Args:
            feature: Feature name to disable
            
        Returns:
            True if feature was disabled
        """
        feature_map = {
            'persistent_memory': ('persistent', 'enabled', False),
            'semantic_search': ('retrieval', 'semantic_search', False),
            'embeddings': ('retrieval', 'use_embeddings', False),
            'entity_extraction': ('processing', 'extract_entities', False),
            'fact_extraction': ('processing', 'extract_facts', False),
            'preference_extraction': ('processing', 'extract_preferences', False),
            'auto_summarize': ('processing', 'auto_summarize', False),
            'auto_consolidate': ('processing', 'auto_consolidate', False),
            'debug_mode': ('debug_mode', None, False),
            'telemetry': ('enable_telemetry', None, False),
            'memory_sync': ('memory_sync_enabled', None, False)
        }
        
        if feature not in feature_map:
            raise ConfigurationError("feature", f"Unknown feature: {feature}")
        
        section, key, value = feature_map[feature]
        
        if key:
            updates = {section: {key: value}}
        else:
            updates = {section: value}
        
        return self.update_config(updates)
    
    def create_snapshot(self, name: str) -> None:
        """
        Create a named snapshot of current configuration.
        
        Args:
            name: Snapshot name
        """
        with self._lock:
            self._snapshots[name] = self._create_config_copy()
    
    def restore_snapshot(self, name: str) -> bool:
        """
        Restore configuration from a snapshot.
        
        Args:
            name: Snapshot name
            
        Returns:
            True if restoration was successful
        """
        with self._lock:
            if name not in self._snapshots:
                raise ConfigurationError("snapshot", f"Snapshot not found: {name}")
            
            self.config = self._create_config_copy(self._snapshots[name])
            
            # Track restoration
            self._track_change({'action': 'restore_snapshot', 'snapshot': name})
            
            # Auto-save if enabled
            if self.auto_save and self.config_file:
                self.save_to_file(self.config_file)
            
            return True
    
    def list_snapshots(self) -> List[str]:
        """
        List available snapshots.
        
        Returns:
            List of snapshot names
        """
        return list(self._snapshots.keys())
    
    def delete_snapshot(self, name: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            name: Snapshot name
            
        Returns:
            True if deletion was successful
        """
        with self._lock:
            if name in self._snapshots:
                del self._snapshots[name]
                return True
            return False
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for configuration changes.
        
        Args:
            callback: Function to call on configuration changes
        """
        self._change_callbacks.append(callback)
    
    def get_change_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get configuration change history.
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of change records
        """
        with self._lock:
            if limit:
                return self._change_history[-limit:]
            return self._change_history.copy()
    
    def clear_change_history(self) -> None:
        """Clear configuration change history."""
        with self._lock:
            self._change_history.clear()
    
    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: File path (uses default if None)
        """
        filepath = filepath or self.config_file
        if not filepath:
            raise ConfigurationError("filepath", "No filepath specified for saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save_to_file(filepath)
        
        # Save metadata
        metadata_file = filepath.replace('.json', '_metadata.json')
        metadata = {
            'last_saved': datetime.now().isoformat(),
            'snapshots': list(self._snapshots.keys()),
            'change_count': len(self._change_history)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_from_file(self, filepath: Optional[str] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            filepath: File path (uses default if None)
        """
        filepath = filepath or self.config_file
        if not filepath:
            raise ConfigurationError("filepath", "No filepath specified for loading")
        
        if not os.path.exists(filepath):
            raise ConfigurationError("filepath", f"Configuration file not found: {filepath}")
        
        # Load configuration
        self.config = MemoryConfig.load_from_file(filepath)
        
        # Load metadata if available
        metadata_file = filepath.replace('.json', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                # Could restore snapshots here if needed
    
    def export_config(self, include_defaults: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Args:
            include_defaults: Whether to include default values
            
        Returns:
            Configuration dictionary
        """
        config_dict = self.config.to_dict()
        
        if not include_defaults:
            # Remove default values
            default = MemoryConfig()
            default_dict = default.to_dict()
            config_dict = self._remove_defaults(config_dict, default_dict)
        
        return config_dict
    
    def reset_to_defaults(self, section: Optional[str] = None) -> bool:
        """
        Reset configuration to defaults.
        
        Args:
            section: Optional section to reset (resets all if None)
            
        Returns:
            True if reset was successful
        """
        with self._lock:
            if section:
                # Reset specific section
                default = MemoryConfig()
                if section == 'session':
                    self.config.session = default.session
                elif section == 'persistent':
                    self.config.persistent = default.persistent
                elif section == 'retrieval':
                    self.config.retrieval = default.retrieval
                elif section == 'processing':
                    self.config.processing = default.processing
                elif section == 'frameworks':
                    self.config.frameworks = default.frameworks
                else:
                    raise ConfigurationError("section", f"Unknown section: {section}")
            else:
                # Reset all
                self.config = MemoryConfig()
            
            # Track reset
            self._track_change({'action': 'reset', 'section': section})
            
            return True
    
    def _apply_updates(self, updates: Dict[str, Any]) -> None:
        """Apply updates to configuration."""
        for key, value in updates.items():
            if key == 'session' and isinstance(value, dict):
                for k, v in value.items():
                    setattr(self.config.session, k, v)
            elif key == 'persistent' and isinstance(value, dict):
                for k, v in value.items():
                    setattr(self.config.persistent, k, v)
            elif key == 'retrieval' and isinstance(value, dict):
                for k, v in value.items():
                    setattr(self.config.retrieval, k, v)
            elif key == 'processing' and isinstance(value, dict):
                for k, v in value.items():
                    setattr(self.config.processing, k, v)
            elif key == 'frameworks' and isinstance(value, dict):
                for framework, framework_config in value.items():
                    if hasattr(self.config.frameworks, framework):
                        current = getattr(self.config.frameworks, framework)
                        current.update(framework_config)
            else:
                # Top-level attribute
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def _create_config_copy(self, config: Optional[MemoryConfig] = None) -> MemoryConfig:
        """Create a deep copy of configuration."""
        config = config or self.config
        config_dict = config.to_dict()
        return MemoryConfig.from_dict(config_dict)
    
    def _track_change(self, change: Dict[str, Any]) -> None:
        """Track a configuration change."""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'change': change
        }
        self._change_history.append(change_record)
        
        # Limit history size
        if len(self._change_history) > 1000:
            self._change_history = self._change_history[-1000:]
    
    def _notify_callbacks(self, change: Dict[str, Any]) -> None:
        """Notify registered callbacks of configuration change."""
        for callback in self._change_callbacks:
            try:
                callback(change)
            except Exception as e:
                # Log error but don't fail the update
                print(f"Error in configuration change callback: {e}")
    
    def _remove_defaults(self, config_dict: Dict[str, Any], 
                        default_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove default values from configuration dictionary."""
        result = {}
        
        for key, value in config_dict.items():
            if key not in default_dict:
                result[key] = value
            elif isinstance(value, dict) and isinstance(default_dict[key], dict):
                nested = self._remove_defaults(value, default_dict[key])
                if nested:
                    result[key] = nested
            elif value != default_dict[key]:
                result[key] = value
        
        return result
