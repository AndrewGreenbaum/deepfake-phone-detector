"""
Unit tests for configuration management.

Tests:
- Config loading from YAML
- Config merging
- Dot notation access
- CLI argument mapping
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add config path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from config import (
    Config,
    load_yaml,
    deep_merge,
    load_config,
    get_default_config,
    cli_to_config_overrides
)


class TestDeepMerge:
    """Test deep dictionary merging."""

    def test_simple_merge(self):
        """Test simple key merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {
            "model": {"name": "mobile", "dropout": 0.3},
            "training": {"epochs": 50}
        }
        override = {
            "model": {"dropout": 0.5},
            "training": {"lr": 0.001}
        }

        result = deep_merge(base, override)

        assert result["model"]["name"] == "mobile"
        assert result["model"]["dropout"] == 0.5
        assert result["training"]["epochs"] == 50
        assert result["training"]["lr"] == 0.001

    def test_original_not_modified(self):
        """Test that original dict is not modified."""
        base = {"a": 1}
        override = {"a": 2}

        deep_merge(base, override)

        assert base["a"] == 1, "Base should not be modified"


class TestConfig:
    """Test Config class."""

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model": {"name": "mobile", "dropout": 0.3},
            "training": {"epochs": 50, "batch_size": 32}
        }

        config = Config(data)

        assert config.model.name == "mobile"
        assert config.model.dropout == 0.3
        assert config.training.epochs == 50

    def test_config_to_dict(self):
        """Test converting config back to dictionary."""
        data = {
            "model": {"name": "mobile"},
            "training": {"epochs": 50}
        }

        config = Config(data)
        result = config.to_dict()

        assert result == data

    def test_config_get_method(self):
        """Test dot notation get method."""
        data = {
            "model": {"name": "mobile"},
            "training": {"epochs": 50}
        }

        config = Config(data)

        assert config.get("model.name") == "mobile"
        assert config.get("training.epochs") == 50

    def test_config_get_default(self):
        """Test get method with default value."""
        config = Config({"a": 1})

        assert config.get("nonexistent", "default") == "default"
        assert config.get("a.b.c", 42) == 42


class TestLoadConfig:
    """Test config loading functions."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = get_default_config()

        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'audio')

    def test_default_config_values(self):
        """Test specific default config values."""
        config = get_default_config()

        # Model defaults
        assert config.model.name == "mobile"
        assert config.model.dropout == 0.3

        # Audio defaults
        assert config.audio.sample_rate == 8000
        assert config.audio.n_mels == 64

        # Training defaults
        assert config.training.epochs == 50
        assert config.training.batch_size == 32

    def test_load_yaml_file(self):
        """Test loading YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("model:\n  name: test_model\n  dropout: 0.5\n")
            f.flush()

            data = load_yaml(f.name)

            assert data["model"]["name"] == "test_model"
            assert data["model"]["dropout"] == 0.5

    def test_load_config_with_overrides(self):
        """Test load_config with CLI overrides."""
        overrides = {
            "training": {"learning_rate": 0.01}
        }

        config = load_config(cli_overrides=overrides)

        assert config.training.learning_rate == 0.01


class TestCLIOverrides:
    """Test CLI argument to config mapping."""

    def test_cli_to_config_basic(self):
        """Test basic CLI argument mapping."""
        from argparse import Namespace

        args = Namespace(
            lr=0.001,
            batch_size=64,
            epochs=100
        )

        overrides = cli_to_config_overrides(args)

        assert overrides["training"]["learning_rate"] == 0.001
        assert overrides["training"]["batch_size"] == 64
        assert overrides["training"]["epochs"] == 100

    def test_cli_to_config_model(self):
        """Test CLI model argument mapping."""
        from argparse import Namespace

        args = Namespace(model="attention")

        overrides = cli_to_config_overrides(args)

        assert overrides["model"]["name"] == "attention"

    def test_cli_to_config_none_ignored(self):
        """Test that None values are ignored."""
        from argparse import Namespace

        args = Namespace(lr=None, batch_size=32)

        overrides = cli_to_config_overrides(args)

        assert "learning_rate" not in overrides.get("training", {})
        assert overrides["training"]["batch_size"] == 32

    def test_cli_to_config_dict_input(self):
        """Test CLI mapping with dict input."""
        args = {"lr": 0.001, "model": "mobile"}

        overrides = cli_to_config_overrides(args)

        assert overrides["training"]["learning_rate"] == 0.001
        assert overrides["model"]["name"] == "mobile"


class TestConfigValidation:
    """Test config validation and edge cases."""

    def test_empty_config(self):
        """Test empty config."""
        config = Config({})

        assert config.to_dict() == {}

    def test_deeply_nested_config(self):
        """Test deeply nested configuration."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    }
                }
            }
        }

        config = Config(data)

        assert config.level1.level2.level3.value == 42

    def test_config_with_list(self):
        """Test config with list values."""
        data = {
            "items": [1, 2, 3],
            "nested": {"list": ["a", "b"]}
        }

        config = Config(data)

        assert config.items == [1, 2, 3]
        assert config.nested.list == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
