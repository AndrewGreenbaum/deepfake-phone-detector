"""
Configuration management for Phone Deepfake Detector.

Supports:
- YAML config files
- CLI argument overrides
- Nested config access
- Config merging
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """
    Configuration container with dot-notation access.

    Usage:
        config = Config.load('config/default.yaml')
        print(config.model.name)
        print(config.training.learning_rate)
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def to_dict(self) -> dict:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def load(cls, path: str, overrides: Optional[dict] = None) -> 'Config':
        """
        Load config from YAML file with optional overrides.

        Args:
            path: Path to YAML config file
            overrides: Dictionary of values to override

        Returns:
            Config object
        """
        data = load_yaml(path)
        if overrides:
            data = deep_merge(data, overrides)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split('.')
        value = self
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        return value


def get_default_config() -> Config:
    """Load default configuration."""
    default_path = Path(__file__).parent / 'default.yaml'
    return Config.load(str(default_path))


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None
) -> Config:
    """
    Load configuration with CLI overrides.

    Priority (highest to lowest):
    1. CLI arguments
    2. Custom config file
    3. Default config

    Args:
        config_path: Path to custom config file (optional)
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        Merged Config object
    """
    # Start with defaults
    default_path = Path(__file__).parent / 'default.yaml'
    config_data = load_yaml(str(default_path))

    # Merge custom config if provided
    if config_path:
        custom_data = load_yaml(config_path)
        config_data = deep_merge(config_data, custom_data)

    # Apply CLI overrides
    if cli_overrides:
        config_data = deep_merge(config_data, cli_overrides)

    return Config(config_data)


def cli_to_config_overrides(args) -> dict:
    """
    Convert argparse namespace to config override dict.

    Maps common CLI args to config paths:
        --lr 0.001 -> {'training': {'learning_rate': 0.001}}
        --model mobile -> {'model': {'name': 'mobile'}}
    """
    overrides = {}

    # Map CLI args to config paths
    mappings = {
        'lr': ('training', 'learning_rate'),
        'learning_rate': ('training', 'learning_rate'),
        'batch_size': ('training', 'batch_size'),
        'epochs': ('training', 'epochs'),
        'weight_decay': ('training', 'weight_decay'),
        'model': ('model', 'name'),
        'data': ('data', 'path'),
        'save_dir': ('output', 'checkpoint_dir'),
        'log_dir': ('output', 'log_dir'),
        'threshold': ('evaluation', 'threshold'),
    }

    args_dict = vars(args) if hasattr(args, '__dict__') else args

    for arg_name, config_path in mappings.items():
        if arg_name in args_dict and args_dict[arg_name] is not None:
            # Build nested dict
            current = overrides
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[config_path[-1]] = args_dict[arg_name]

    return overrides


# Quick access to default config values
DEFAULT_CONFIG = None

def get_config() -> Config:
    """Get cached default config."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = get_default_config()
    return DEFAULT_CONFIG


if __name__ == "__main__":
    # Test config loading
    config = get_default_config()

    print("Default Configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Sample Rate: {config.audio.sample_rate}")
    print(f"  Degradation Enabled: {config.degradation.enabled}")

    print("\nUsing dot notation:")
    print(f"  config.get('model.name'): {config.get('model.name')}")
    print(f"  config.get('training.epochs'): {config.get('training.epochs')}")

    print("\nConfig as dict:")
    import json
    print(json.dumps(config.to_dict(), indent=2))
