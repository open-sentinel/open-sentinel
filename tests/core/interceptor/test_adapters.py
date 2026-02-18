"""
Tests for PolicyEngineChecker adapter.

Verifies that the adapter correctly maps PolicyEngine evaluate_request/
evaluate_response calls and decision types to Checker/CheckResult types.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opensentinel.core.interceptor import (
    CheckPhase,
    CheckerMode,
    CheckerContext,
)
from opensentinel.core.interceptor.adapters import PolicyEngineChecker
from opensentinel.policy.protocols import PolicyDecision, PolicyEvaluationResult, PolicyViolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine(name: str = "test_engine") -> MagicMock:
    """Create a mock PolicyEngine with a given name."""
    engine = MagicMock()
    engine.name = name
    engine.evaluate_request = AsyncMock()
    engine.evaluate_response = AsyncMock()
    return engine


def _context(**overrides) -> CheckerContext:
    defaults = {
        "session_id": "sess-1",
        "user_request_id": "req-1",
        "request_data": {"messages": [{"role": "user", "content": "hi"}]},
        "response_data": None,
    }
    defaults.update(overrides)
    return CheckerContext(**defaults)


# ===========================================================================
# Phase routing tests
# ===========================================================================


class TestPhaseRouting:

    async def test_pre_call_calls_evaluate_request(self):
        """PRE_CALL phase calls engine.evaluate_request."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW
        )

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        ctx = _context()
        await checker.check(ctx)

        engine.evaluate_request.assert_called_once_with(
            session_id="sess-1",
            request_data=ctx.request_data,
            context={"user_request_id": "req-1"},
        )
        engine.evaluate_response.assert_not_called()

    async def test_post_call_calls_evaluate_response(self):
        """POST_CALL phase calls engine.evaluate_response."""
        engine = _mock_engine()
        engine.evaluate_response.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW
        )
        response = {"answer": "hello"}

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.POST_CALL)
        ctx = _context(response_data=response)
        await checker.check(ctx)

        engine.evaluate_response.assert_called_once_with(
            session_id="sess-1",
            response_data=response,
            request_data=ctx.request_data,
            context={"user_request_id": "req-1"},
        )
        engine.evaluate_request.assert_not_called()


# ===========================================================================
# Decision mapping tests
# ===========================================================================


class TestDecisionMapping:

    @pytest.mark.parametrize(
        "policy_decision, expected_decision",
        [
            (PolicyDecision.ALLOW, PolicyDecision.ALLOW),
            (PolicyDecision.DENY, PolicyDecision.DENY),
            (PolicyDecision.MODIFY, PolicyDecision.DENY),  # SYNC coerces MODIFY→DENY
            (PolicyDecision.WARN, PolicyDecision.ALLOW),   # SYNC coerces WARN→ALLOW
        ],
    )
    async def test_sync_decision_mapping(self, policy_decision, expected_decision):
        """SYNC mode: ALLOW/DENY pass through, MODIFY→DENY, WARN→ALLOW."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=policy_decision
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.SYNC
        )
        result = await checker.check(_context())

        assert result.decision == expected_decision

    @pytest.mark.parametrize(
        "policy_decision",
        [PolicyDecision.ALLOW, PolicyDecision.DENY, PolicyDecision.MODIFY, PolicyDecision.WARN],
    )
    async def test_async_decision_passthrough(self, policy_decision):
        """ASYNC mode: all decisions pass through unchanged."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=policy_decision
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.decision == policy_decision

    async def test_sync_modify_strips_modified_data(self):
        """SYNC mode: MODIFY coercion also strips modified_data."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.MODIFY,
            modified_request={"system_prompt_append": "Be safe"},
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.SYNC
        )
        result = await checker.check(_context())

        assert result.decision == PolicyDecision.DENY
        assert result.modified_data is None

    async def test_async_modify_keeps_modified_data(self):
        """ASYNC mode: MODIFY keeps modified_data."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.MODIFY,
            modified_request={"system_prompt_append": "Be safe"},
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.decision == PolicyDecision.MODIFY
        assert result.modified_data is not None
        assert result.modified_data["system_prompt_append"] == "Be safe"


# ===========================================================================
# Data forwarding tests
# ===========================================================================


class TestDataForwarding:

    async def test_modified_request_forwarded(self):
        """modified_request from engine maps to CheckResult.modified_data (ASYNC mode)."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.MODIFY,
            modified_request={"messages": [{"role": "system", "content": "injected"}]},
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.modified_data is not None
        assert result.modified_data["messages"][0]["content"] == "injected"

    async def test_violations_forwarded(self):
        """Violation list is preserved in CheckResult."""
        violation = PolicyViolation(
            name="no_pii",
            message="PII detected",
            severity="high",
        )
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            violations=[violation],
        )

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        result = await checker.check(_context())

        assert len(result.violations) == 1
        assert result.violations[0].name == "no_pii"
        assert result.violations[0].message == "PII detected"

    async def test_violation_messages_joined(self):
        """Multiple violation messages are joined with semicolons."""
        v1 = PolicyViolation(name="c1", message="bad thing", severity="high")
        v2 = PolicyViolation(name="c2", message="worse thing", severity="high")
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            violations=[v1, v2],
        )

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        result = await checker.check(_context())

        assert "bad thing" in result.message
        assert "worse thing" in result.message
        assert ";" in result.message

    async def test_intervention_as_modified_data(self):
        """When engine sets intervention_needed with metadata, it maps to strategy-type key (ASYNC)."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.WARN,
            intervention_needed="system_prompt_append",
            metadata={"message": "Do X instead"},
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.modified_data is not None
        assert result.modified_data["system_prompt_append"] == "Do X instead"

    async def test_intervention_fallback_without_message_key(self):
        """Fallback uses str(metadata) when no 'message' key present."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.WARN,
            intervention_needed="user_message_inject",
            metadata={"correction": "Do X instead"},
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.modified_data is not None
        assert "user_message_inject" in result.modified_data
        assert "correction" in result.modified_data["user_message_inject"]

    async def test_intervention_without_metadata(self):
        """Fallback produces empty message when no metadata present."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = PolicyEvaluationResult(
            decision=PolicyDecision.WARN,
            intervention_needed="context_reminder",
        )

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        result = await checker.check(_context())

        assert result.modified_data is not None
        assert result.modified_data["context_reminder"] == ""


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:

    async def test_engine_exception_returns_fail(self):
        """Engine errors produce FAIL with error message."""
        engine = _mock_engine()
        engine.evaluate_request.side_effect = RuntimeError("engine exploded")

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        result = await checker.check(_context())

        assert result.decision == PolicyDecision.DENY
        assert "engine exploded" in result.message


# ===========================================================================
# Naming
# ===========================================================================


class TestNaming:

    async def test_name_includes_engine_and_phase(self):
        """Checker name is {engine.name}_{phase}."""
        engine = _mock_engine(name="my_engine")
        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)

        assert checker.name == "my_engine_pre_call"

    async def test_name_with_suffix(self):
        """name_suffix is appended."""
        engine = _mock_engine(name="my_engine")
        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.POST_CALL, name_suffix="v2"
        )

        assert checker.name == "my_engine_post_call_v2"

    async def test_mode_property(self):
        """Mode property reflects what was passed in."""
        engine = _mock_engine()
        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )

        assert checker.mode == CheckerMode.ASYNC
