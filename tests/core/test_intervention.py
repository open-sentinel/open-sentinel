"""Tests for intervention system."""

import pytest

from panoptes.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    HardBlockStrategy,
    WorkflowViolationError,
    STRATEGY_REGISTRY,
)
from panoptes.policy.engines.fsm.injector import PromptInjector


class TestInterventionStrategies:
    """Tests for intervention strategies."""

    @pytest.fixture
    def sample_data(self):
        """Sample LLM request data."""
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Help me with my account."},
            ],
        }

    @pytest.fixture
    def config(self):
        """Sample intervention config."""
        return InterventionConfig(
            strategy_type=StrategyType.SYSTEM_PROMPT_APPEND,
            message_template="Please verify identity first. Current state: {current_state}",
        )

    def test_system_prompt_append(self, sample_data, config):
        """Test system prompt append strategy."""
        strategy = SystemPromptAppendStrategy()
        context = {"current_state": "greeting"}

        result = strategy.apply(sample_data, config, context)

        # Should have modified system message
        system_msg = result["messages"][0]
        assert "[WORKFLOW GUIDANCE]" in system_msg["content"]
        assert "verify identity" in system_msg["content"]
        assert "greeting" in system_msg["content"]

    def test_system_prompt_append_no_existing_system(self, config):
        """Test system prompt append when no system message exists."""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        strategy = SystemPromptAppendStrategy()
        context = {"current_state": "test"}

        result = strategy.apply(data, config, context)

        # Should have inserted system message
        assert result["messages"][0]["role"] == "system"
        assert "[WORKFLOW GUIDANCE]" in result["messages"][0]["content"]

    def test_user_message_inject(self, sample_data):
        """Test user message inject strategy."""
        config = InterventionConfig(
            strategy_type=StrategyType.USER_MESSAGE_INJECT,
            message_template="Please check this first.",
        )
        strategy = UserMessageInjectStrategy()
        context = {}

        result = strategy.apply(sample_data, config, context)

        # Should have injected a user message
        messages = result["messages"]
        # Find the injected message
        injected = [m for m in messages if "[System Note]" in m.get("content", "")]
        assert len(injected) == 1
        assert injected[0]["role"] == "user"

    def test_hard_block(self, sample_data):
        """Test hard block strategy raises exception."""
        config = InterventionConfig(
            strategy_type=StrategyType.HARD_BLOCK,
            message_template="Action not allowed: {reason}",
        )
        strategy = HardBlockStrategy()
        context = {"reason": "missing verification"}

        with pytest.raises(WorkflowViolationError) as exc_info:
            strategy.apply(sample_data, config, context)

        assert "missing verification" in str(exc_info.value)

    def test_strategy_registry(self):
        """Test that all strategy types are in registry."""
        for strategy_type in StrategyType:
            assert strategy_type in STRATEGY_REGISTRY

    def test_message_format_with_missing_key(self):
        """Test message formatting handles missing context keys."""
        strategy = SystemPromptAppendStrategy()
        template = "Value: {missing_key}"

        # Should not raise, just leave placeholder
        result = strategy.format_message(template, {})
        assert template == result  # Returns original if key missing


class TestPromptInjector:
    """Tests for PromptInjector."""

    @pytest.fixture
    def workflow(self, simple_workflow):
        """Use simple workflow fixture."""
        return simple_workflow

    @pytest.fixture
    def injector(self, workflow):
        """Create PromptInjector."""
        return PromptInjector(workflow)

    def test_inject_known_intervention(self, injector):
        """Test injecting a known intervention."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }

        result = injector.inject(data, "prompt_search", context={})

        # Should have modified messages
        assert len(result["messages"]) > 1 or "[WORKFLOW GUIDANCE]" in str(result)

    def test_inject_unknown_intervention(self, injector):
        """Test injecting unknown intervention returns unchanged data."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }

        result = injector.inject(data, "nonexistent", context={})

        # Should return unchanged
        assert result == data

    def test_list_interventions(self, injector):
        """Test listing available interventions."""
        interventions = injector.list_interventions()

        assert "prompt_search" in interventions

    def test_get_intervention_info(self, injector):
        """Test getting intervention info."""
        info = injector.get_intervention_info("prompt_search")

        assert info is not None
        assert info["name"] == "prompt_search"
        assert "strategy" in info
        assert "template" in info

    def test_application_tracking(self, injector):
        """Test that applications are tracked per session."""
        data = {"messages": [{"role": "user", "content": "test"}]}

        # Apply multiple times
        injector.inject(data, "prompt_search", session_id="session1")
        injector.inject(data, "prompt_search", session_id="session1")

        count = injector._get_application_count("session1", "prompt_search")
        assert count == 2

    def test_reset_session_counts(self, injector):
        """Test resetting session application counts."""
        data = {"messages": [{"role": "user", "content": "test"}]}

        injector.inject(data, "prompt_search", session_id="session1")
        injector.reset_session_counts("session1")

        count = injector._get_application_count("session1", "prompt_search")
        assert count == 0
