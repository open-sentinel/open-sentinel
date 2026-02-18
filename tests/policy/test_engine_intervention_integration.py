"""
Tests for engine intervention integration.

Verifies get_intervention_handler() and resolve_intervention() across
all engine types following the new PolicyEngine protocol.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from opensentinel.policy.protocols import (
    PolicyEngine,
    InterventionHandlerProtocol,
)
from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine
from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
from opensentinel.policy.engines.llm.engine import LLMPolicyEngine
from opensentinel.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from opensentinel.policy.engines.composite.engine import CompositePolicyEngine
from opensentinel.core.intervention.strategies import InterventionConfig, StrategyType


# ---------------------------------------------------------------------------
# PolicyEngine base defaults
# ---------------------------------------------------------------------------


class TestPolicyEngineDefaults:
    """Default implementations on the PolicyEngine ABC."""

    def test_get_compiler_default_is_none(self):
        """Uninitialized engines default to None compiler."""
        engine = JudgePolicyEngine()
        # Judge *does* override, but LLM doesn't
        llm = LLMPolicyEngine()
        assert llm.get_compiler() is None

    def test_get_intervention_handler_default_is_none(self):
        engine = JudgePolicyEngine()
        assert engine.get_intervention_handler() is None

    def test_resolve_intervention_default_is_none(self):
        engine = JudgePolicyEngine()
        assert engine.resolve_intervention("anything") is None


# ---------------------------------------------------------------------------
# FSM engine
# ---------------------------------------------------------------------------


class TestFSMInterventionHandler:
    """FSMPolicyEngine intervention handler integration."""

    @pytest.fixture
    def initialized_fsm(self):
        engine = FSMPolicyEngine()
        with patch("opensentinel.policy.engines.fsm.engine.WorkflowParser") as mock_parser, \
             patch("opensentinel.policy.engines.fsm.engine.WorkflowStateMachine"), \
             patch("opensentinel.policy.engines.fsm.engine.StateClassifier"), \
             patch("opensentinel.policy.engines.fsm.engine.ConstraintEvaluator"):
            mock_workflow = MagicMock(
                name="test_workflow",
                states=[],
                constraints=[],
                interventions={"warn_user": "Please stay on topic."},
            )
            mock_parser().parse_dict.return_value = mock_workflow
            mock_parser.parse_file.return_value = mock_workflow

            import asyncio
            asyncio.get_event_loop().run_until_complete(
                engine.initialize({"workflow": {}})
            )
        return engine

    def test_get_intervention_handler_returns_handler(self, initialized_fsm):
        handler = initialized_fsm.get_intervention_handler()
        assert handler is not None

    def test_handler_satisfies_protocol(self, initialized_fsm):
        handler = initialized_fsm.get_intervention_handler()
        assert isinstance(handler, InterventionHandlerProtocol)

    def test_handler_has_get_config(self, initialized_fsm):
        handler = initialized_fsm.get_intervention_handler()
        assert callable(getattr(handler, "get_config", None))

    def test_handler_has_list_interventions(self, initialized_fsm):
        handler = initialized_fsm.get_intervention_handler()
        assert callable(getattr(handler, "list_interventions", None))

    def test_resolve_intervention_delegates_to_handler(self, initialized_fsm):
        """resolve_intervention should delegate to handler.get_config."""
        handler = initialized_fsm.get_intervention_handler()
        # Mock the handler's get_config to return a known value
        mock_config = InterventionConfig(
            strategy_type=StrategyType.SYSTEM_PROMPT_APPEND,
            message_template="test",
        )
        handler.get_config = MagicMock(return_value=mock_config)

        result = initialized_fsm.resolve_intervention("warn_user")
        handler.get_config.assert_called_once_with("warn_user")
        assert result is mock_config

    def test_uninitialised_fsm_handler_is_none(self):
        engine = FSMPolicyEngine()
        assert engine.get_intervention_handler() is None


# ---------------------------------------------------------------------------
# LLM engine
# ---------------------------------------------------------------------------


class TestLLMInterventionHandler:
    """LLMPolicyEngine intervention handler integration."""

    def test_uninitialised_handler_is_none(self):
        engine = LLMPolicyEngine()
        assert engine.get_intervention_handler() is None

    def test_handler_satisfies_protocol(self):
        """After adding get_config/list_interventions, the LLM handler conforms."""
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"remind": "Stay focused."},
        )
        handler = InterventionHandler(workflow)
        assert isinstance(handler, InterventionHandlerProtocol)

    def test_get_config_returns_config(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"remind": "Stay focused."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("remind")
        assert config is not None
        assert isinstance(config, InterventionConfig)
        assert config.message_template == "Stay focused."
        assert config.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_get_config_with_block_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"hard_stop": "block: This action is blocked."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("hard_stop")
        assert config is not None
        assert config.strategy_type == StrategyType.HARD_BLOCK
        assert config.message_template == "This action is blocked."

    def test_get_config_with_inject_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"inject_msg": "inject: Please clarify your request."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("inject_msg")
        assert config is not None
        assert config.strategy_type == StrategyType.USER_MESSAGE_INJECT
        assert config.message_template == "Please clarify your request."

    def test_get_config_with_remind_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"reminder": "remind: Remember the policy."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("reminder")
        assert config is not None
        assert config.strategy_type == StrategyType.CONTEXT_REMINDER

    def test_get_config_unknown_returns_none(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[],
            interventions={},
        )
        handler = InterventionHandler(workflow)
        assert handler.get_config("nonexistent") is None

    def test_list_interventions(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[],
            interventions={"a": "msg a", "b": "msg b"},
        )
        handler = InterventionHandler(workflow)
        names = handler.list_interventions()
        assert set(names) == {"a", "b"}


# ---------------------------------------------------------------------------
# NeMo engine
# ---------------------------------------------------------------------------


class TestNemoResolveIntervention:
    """NemoGuardrailsPolicyEngine.resolve_intervention() tests."""

    def test_resolves_output_blocked(self):
        engine = NemoGuardrailsPolicyEngine()
        config = engine.resolve_intervention("nemo_output_blocked")
        assert config is not None
        assert isinstance(config, InterventionConfig)
        assert config.strategy_type == StrategyType.HARD_BLOCK
        assert "blocked" in config.message_template.lower()

    def test_resolves_input_blocked(self):
        engine = NemoGuardrailsPolicyEngine()
        config = engine.resolve_intervention("nemo_input_blocked")
        assert config is not None
        assert config.strategy_type == StrategyType.HARD_BLOCK

    def test_output_blocked_uses_context_message(self):
        engine = NemoGuardrailsPolicyEngine()
        config = engine.resolve_intervention(
            "nemo_output_blocked",
            context={"nemo_response": "Custom blocked message"},
        )
        assert config is not None
        assert config.message_template == "Custom blocked message"

    def test_unknown_intervention_returns_none(self):
        engine = NemoGuardrailsPolicyEngine()
        assert engine.resolve_intervention("unknown_thing") is None

    def test_get_intervention_handler_is_none(self):
        """NeMo has no handler, only resolve_intervention."""
        engine = NemoGuardrailsPolicyEngine()
        assert engine.get_intervention_handler() is None


# ---------------------------------------------------------------------------
# Judge engine
# ---------------------------------------------------------------------------


class TestJudgeIntervention:
    """JudgePolicyEngine has no intervention handler."""

    def test_get_intervention_handler_is_none(self):
        engine = JudgePolicyEngine()
        assert engine.get_intervention_handler() is None

    def test_resolve_intervention_is_none(self):
        engine = JudgePolicyEngine()
        assert engine.resolve_intervention("system_prompt_append") is None


# ---------------------------------------------------------------------------
# Composite engine
# ---------------------------------------------------------------------------


class TestCompositeResolveIntervention:
    """CompositePolicyEngine delegates resolve_intervention to children."""

    def test_delegates_to_child(self):
        engine = CompositePolicyEngine()
        child = MagicMock(spec=PolicyEngine)
        expected = InterventionConfig(
            strategy_type=StrategyType.HARD_BLOCK,
            message_template="blocked",
        )
        child.resolve_intervention.return_value = expected
        engine._engines = [child]

        result = engine.resolve_intervention("test_intervention")
        child.resolve_intervention.assert_called_once_with("test_intervention", None)
        assert result is expected

    def test_returns_first_non_none(self):
        engine = CompositePolicyEngine()
        child1 = MagicMock(spec=PolicyEngine)
        child1.resolve_intervention.return_value = None
        child2 = MagicMock(spec=PolicyEngine)
        expected = InterventionConfig(
            strategy_type=StrategyType.SYSTEM_PROMPT_APPEND,
            message_template="gentle nudge",
        )
        child2.resolve_intervention.return_value = expected
        engine._engines = [child1, child2]

        result = engine.resolve_intervention("test")
        assert result is expected

    def test_returns_none_when_no_child_resolves(self):
        engine = CompositePolicyEngine()
        child = MagicMock(spec=PolicyEngine)
        child.resolve_intervention.return_value = None
        engine._engines = [child]

        assert engine.resolve_intervention("unknown") is None

    def test_returns_none_with_no_children(self):
        engine = CompositePolicyEngine()
        engine._engines = []
        assert engine.resolve_intervention("anything") is None


# ---------------------------------------------------------------------------
# InterventionHandlerProtocol
# ---------------------------------------------------------------------------


class TestInterventionHandlerProtocol:
    """Test that the protocol is runtime-checkable."""

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(InterventionHandlerProtocol, "__protocol_attrs__") or True
        # Protocol should be usable with isinstance
        class FakeHandler:
            def get_config(self, intervention_name: str):
                return None

            def list_interventions(self):
                return []

        assert isinstance(FakeHandler(), InterventionHandlerProtocol)

    def test_non_conforming_object_fails(self):
        class BadHandler:
            pass

        assert not isinstance(BadHandler(), InterventionHandlerProtocol)
