import os
import pytest
from opensentinel.config.settings import SentinelSettings, YamlConfigSource


def test_flat_config_path_env_var():
    """Verify that OSNTL_POLICY__ENGINE__CONFIG_PATH sets the config_path correctly."""
    # Set the simplified environment variable
    os.environ["OSNTL_POLICY__ENGINE__CONFIG_PATH"] = "/tmp/flat/config.yml"

    # Reload settings
    settings = SentinelSettings()

    # Check if config_path was populated in the model
    assert settings.policy.engine.config_path == "/tmp/flat/config.yml"

    # Check if get_policy_config() merges it into the config dict
    policy_config = settings.get_policy_config()
    assert policy_config["config"]["config_path"] == "/tmp/flat/config.yml"

    # Clean up
    del os.environ["OSNTL_POLICY__ENGINE__CONFIG_PATH"]


class TestInlinePolicyMapping:
    """Tests for inline policy handling in YamlConfigSource._map_to_settings()."""

    def _build_source(self, yaml_data):
        """Create a YamlConfigSource with injected YAML data."""
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = None
        return source

    def test_policy_list_maps_to_inline_policy(self):
        """A list of rules should map to policy.engine.config.inline_policy."""
        source = self._build_source({
            "engine": "judge",
            "policy": ["No PII", "Be helpful"],
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config"]["inline_policy"] == ["No PII", "Be helpful"]
        assert "config_path" not in result["policy"]["engine"]

    def test_policy_dict_maps_to_inline_policy(self):
        """A dict policy should map to policy.engine.config.inline_policy."""
        source = self._build_source({
            "engine": "judge",
            "policy": {"rules": ["No PII"]},
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config"]["inline_policy"] == {"rules": ["No PII"]}

    def test_policy_string_maps_to_config_path(self):
        """A string policy should still map to config_path (backward compat)."""
        source = self._build_source({
            "engine": "judge",
            "policy": "./policy.yaml",
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config_path"] == "./policy.yaml"
        assert "config" not in result["policy"]["engine"] or \
               "inline_policy" not in result["policy"]["engine"].get("config", {})


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

