"""
⚡ ASCII Forge Configuration System ⚡

Quantum-precise configuration management with zero overhead.
This module provides hyper-optimized access to user and system settings 
with intelligent fallbacks and state persistence.

Key features:
- Layered configuration hierarchy (system → user → runtime)
- Zero-latency access patterns
- Atomic setting operations
- Self-healing state management
- Deterministic configuration resolution
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Set
import threading
from enum import Enum

logger = logging.getLogger(__name__)

# Type definitions for configuration values
ConfigValue = Union[str, int, float, bool, List[str], Dict[str, Any]]
ConfigStore = Dict[str, Dict[str, ConfigValue]]


class ConfigScope(Enum):
    """Configuration scopes with precise access semantics."""
    SYSTEM = "system"   # System-wide, read-only default settings
    USER = "user"       # User-specific persistent settings
    RUNTIME = "runtime" # Session-only ephemeral settings


class ConfigManager:
    """
    Eidosian configuration manager with zero-compromise precision.
    
    Manages layered configuration with surgical access patterns
    and atomic state transitions. Configuration resolves through
    cascading layers with perfect determinism.
    
    Attributes:
        config (ConfigStore): Primary configuration state
        _lock (threading.RLock): Thread synchronization mechanism
        _config_paths (Dict[ConfigScope, Path]): Configuration file paths
        _dirty_scopes (Set[ConfigScope]): Scopes requiring persistence
    """
    
    # Default configuration values - core system baseline
    DEFAULT_CONFIG: ConfigStore = {
        "banner": {
            "default_font": "slant",
            "default_width": 80,
            "default_style": "minimal",
            "cache_enabled": True,
            "cache_size": 100,
            "cache_ttl": 3600,  # seconds
            "unicode_enabled": True
        },
        "image": {
            "default_charset": "general",
            "default_width": 100,
            "max_width": 500,
            "brightness": 1.0,
            "contrast": 1.0,
            "dithering": False,
            "parallel_processing": True,
            "max_threads": 4
        },
        "io": {
            "output_format": "text",
            "auto_detect_terminal": True,
            "color_output": True,
            "backup_files": True,
            "temp_directory": ""
        },
        "performance": {
            "optimization_level": 3,  # 1-5 scale
            "cache_enabled": True,
            "lazy_loading": True,
            "debug_mode": False
        }
    }
    
    def __init__(self):
        """Initialize configuration manager with precise state hierarchy."""
        self.config: ConfigStore = {}
        self._lock = threading.RLock()
        self._dirty_scopes: Set[ConfigScope] = set()
        
        # Determine configuration paths with system-appropriate locations
        self._config_paths = self._initialize_paths()
        
        # Load configuration layers with precise fallback cascade
        self._load_system_defaults()
        self._load_user_config()
        
        logger.debug("ConfigManager initialized with %d sections", len(self.config))
    
    def _initialize_paths(self) -> Dict[ConfigScope, Path]:
        """Initialize configuration paths with system-aware locations."""
        config_paths = {}
        
        # System config path
        system_config_dir = Path("/etc/ascii_forge")
        if not system_config_dir.exists():
            # Fallback to package directory
            package_dir = Path(__file__).parent.parent
            system_config_dir = package_dir / "config"
        config_paths[ConfigScope.SYSTEM] = system_config_dir / "system_config.json"
        
        # User config path - platform aware
        if os.name == 'nt':  # Windows
            user_config_dir = Path(os.environ.get('APPDATA', '')) / "ASCII_Forge"
        else:  # Unix/Linux/Mac
            xdg_config_home = os.environ.get('XDG_CONFIG_HOME', '')
            if xdg_config_home:
                user_config_dir = Path(xdg_config_home) / "ascii_forge"
            else:
                user_config_dir = Path.home() / ".config" / "ascii_forge"
                
        # Ensure user config directory exists
        os.makedirs(user_config_dir, exist_ok=True)
        config_paths[ConfigScope.USER] = user_config_dir / "user_config.json"
        
        # Runtime config exists only in memory
        config_paths[ConfigScope.RUNTIME] = None
        
        return config_paths
    
    def _load_system_defaults(self) -> None:
        """Load system default configuration with fallback hierarchy."""
        with self._lock:
            # Start with hardcoded defaults
            self.config = self._deep_copy_config(self.DEFAULT_CONFIG)
            
            # Try to load system configuration file if it exists
            system_path = self._config_paths[ConfigScope.SYSTEM]
            if system_path and system_path.exists():
                try:
                    with open(system_path, 'r') as f:
                        system_config = json.load(f)
                    
                    # Merge system config with defaults
                    self._merge_configs(self.config, system_config)
                    logger.debug("Loaded system configuration from %s", system_path)
                except Exception as e:
                    logger.warning("Failed to load system config: %s", str(e))
    
    def _load_user_config(self) -> None:
        """Load user configuration with atomic state update."""
        with self._lock:
            user_path = self._config_paths[ConfigScope.USER]
            if user_path and user_path.exists():
                try:
                    with open(user_path, 'r') as f:
                        user_config = json.load(f)
                    
                    # Merge user config with current state
                    self._merge_configs(self.config, user_config)
                    logger.debug("Loaded user configuration from %s", user_path)
                except Exception as e:
                    logger.warning("Failed to load user config: %s", str(e))
    
    def _merge_configs(self, base: ConfigStore, overlay: Dict[str, Any]) -> None:
        """Merge configuration dictionaries with precise overlay semantics."""
        for section, values in overlay.items():
            if section not in base:
                base[section] = {}
                
            if isinstance(values, dict):
                for key, value in values.items():
                    base[section][key] = value
    
    def _deep_copy_config(self, config: ConfigStore) -> ConfigStore:
        """Create a deep copy of configuration with zero shared references."""
        result: ConfigStore = {}
        for section, values in config.items():
            result[section] = {k: v for k, v in values.items()}
        return result
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value with deterministic resolution.
        
        Args:
            section: Configuration section name
            key: Setting key within section
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            try:
                return self.config[section][key]
            except KeyError:
                return default
    
    def set(self, section: str, key: str, value: Any, 
            scope: ConfigScope = ConfigScope.USER) -> None:
        """
        Set configuration value with atomic state update.
        
        Args:
            section: Configuration section name
            key: Setting key within section
            value: Value to set
            scope: Configuration scope (default: USER)
        """
        with self._lock:
            # Create section if it doesn't exist
            if section not in self.config:
                self.config[section] = {}
            
            # Set value and mark scope as dirty for persistence
            self.config[section][key] = value
            
            # Mark for saving if in persistent scope
            if scope != ConfigScope.RUNTIME:
                self._dirty_scopes.add(scope)
                self._save_config(scope)
                
            logger.debug("Set config [%s.%s] = %s", section, key, value)
    
    def _save_config(self, scope: ConfigScope) -> None:
        """Save configuration to persistent storage with atomic file write."""
        if scope == ConfigScope.RUNTIME:
            return  # Runtime config is not persisted
            
        config_path = self._config_paths.get(scope)
        if not config_path:
            return
            
        # Extract configuration for this scope
        config_to_save: ConfigStore = {}
        for section, values in self.config.items():
            config_to_save[section] = {}
            for key, value in values.items():
                # Skip None values and complex objects
                if value is not None and isinstance(value, (str, int, float, bool, list, dict)):
                    config_to_save[section][key] = value
        
        # Create parent directory if needed
        os.makedirs(config_path.parent, exist_ok=True)
        
        # Write with atomic replace strategy
        temp_path = config_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            # Atomic replacement (works on both POSIX and Windows)
            if os.path.exists(config_path):
                os.replace(temp_path, config_path)
            else:
                os.rename(temp_path, config_path)
                
            # Remove from dirty scopes
            self._dirty_scopes.discard(scope)
            logger.debug("Saved configuration to %s", config_path)
        except Exception as e:
            logger.error("Failed to save configuration: %s", str(e))
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """
        Reset configuration to system defaults.
        
        Args:
            section: Section to reset (or all if None)
        """
        with self._lock:
            if section:
                # Reset specific section
                if section in self.DEFAULT_CONFIG:
                    self.config[section] = self._deep_copy_config({section: self.DEFAULT_CONFIG[section]})[section]
            else:
                # Reset all sections
                self.config = self._deep_copy_config(self.DEFAULT_CONFIG)
                
            # Mark all persistent scopes as dirty
            self._dirty_scopes.update([ConfigScope.USER, ConfigScope.SYSTEM])
            self._save_config(ConfigScope.USER)
            
            logger.info("Reset configuration to defaults%s", f" for section '{section}'" if section else "")
    
    def get_sections(self) -> List[str]:
        """Get list of all configuration sections."""
        with self._lock:
            return list(self.config.keys())
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get all settings in a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary of settings or empty dict if section not found
        """
        with self._lock:
            return self.config.get(section, {}).copy()


# Singleton configuration manager
_config_instance: Optional[ConfigManager] = None
_config_lock = threading.Lock()

def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance with zero redundant initialization.
    
    Returns:
        ConfigManager singleton instance
    """
    global _config_instance
    
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ConfigManager()
                
    return _config_instance