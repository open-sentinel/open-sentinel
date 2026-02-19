
import os
import pytest
from opensentinel.config.settings import SentinelSettings

class TestModelDefaults:
    
    def test_no_keys_and_no_model_raises_error(self, monkeypatch):
        """Test that initializing Settings with no keys and no explicit model raises ValueError."""
        # Ensure environment is clean
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        
        # Initialize settings without .env file interference
        settings = SentinelSettings(_env_file=None)
        
        # Validation should fail
        with pytest.raises(ValueError, match="No LLM API keys detected"):
            settings.validate()

    def test_gemini_key_autodetects_model(self, monkeypatch):
        """Test that GEMINI_API_KEY triggers auto-detection of gemini model."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        
        settings = SentinelSettings(_env_file=None)
        
        # Should detect gemini
        assert "gemini" in settings.proxy.default_model
        
        # Validation should pass
        settings.validate()

    def test_anthropic_key_autodetects_model(self, monkeypatch):
        """Test that ANTHROPIC_API_KEY triggers auto-detection of claude model."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key")
        
        settings = SentinelSettings(_env_file=None)
        
        # Should detect claude
        assert "anthropic" in settings.proxy.default_model
        
        # Validation should pass
        settings.validate()

    def test_explicit_model_overrides_autodetect(self, monkeypatch):
        """Test that explicit model configuration takes precedence over auto-detection."""
        # Set OpenAI key which would normally default to gpt-4o-mini
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
        
        # Explicitly configure a different model via init
        settings = SentinelSettings(
            proxy={"default_model": "custom/model"},
            _env_file=None
        )
        
        assert settings.proxy.default_model == "custom/model"
        # Validate might fail if it strictly checks key matching for custom model, 
        # but here we just check precedence.
        
    def test_mixed_keys_priority(self, monkeypatch):
        """Test that OpenAI key takes priority if multiple keys are present."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        
        settings = SentinelSettings(_env_file=None)
        
        # OpenAI is first check in detect_available_model
        assert "gpt" in settings.proxy.default_model
