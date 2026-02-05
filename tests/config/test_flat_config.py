import os
import pytest
from panoptes.config.settings import PanoptesSettings

def test_flat_config_path_env_var():
    """Verify that PANOPTES_POLICY__ENGINE__CONFIG_PATH sets the config_path correctly."""
    # Set the simplified environment variable
    os.environ["PANOPTES_POLICY__ENGINE__CONFIG_PATH"] = "/tmp/flat/config.yml"
    
    # Reload settings
    settings = PanoptesSettings()
    
    # Check if config_path was populated in the model
    assert settings.policy.engine.config_path == "/tmp/flat/config.yml"
    
    # Check if get_policy_config() merges it into the config dict
    policy_config = settings.get_policy_config()
    assert policy_config["config"]["config_path"] == "/tmp/flat/config.yml"
    
    # Clean up
    del os.environ["PANOPTES_POLICY__ENGINE__CONFIG_PATH"]

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
