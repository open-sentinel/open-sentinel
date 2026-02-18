"""
Integration test: async POST_CALL checker returns MODIFY with a strategy key
→ next PRE_CALL applies it → verify the modifications appear in the result.

Tests the full deferred intervention flow end-to-end through the Interceptor.
"""

import asyncio
from typing import Dict, Any

from opensentinel.core.interceptor import (
    Interceptor,
    Checker,
    CheckPhase,
    CheckerMode,
    PolicyDecision,
    CheckResult,
    CheckerContext,
)


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class _FakeAsyncChecker(Checker):
    """Async POST_CALL checker that returns a MODIFY with strategy-type key."""

    def __init__(self, modified_data: Dict[str, Any]):
        self._modified_data = modified_data

    @property
    def name(self) -> str:
        return "fake_async_intervention"

    @property
    def phase(self) -> CheckPhase:
        return CheckPhase.POST_CALL

    @property
    def mode(self) -> CheckerMode:
        return CheckerMode.ASYNC

    async def check(self, context: CheckerContext) -> CheckResult:
        return CheckResult(
            decision=PolicyDecision.MODIFY,
            checker_name=self.name,
            modified_data=self._modified_data,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "integration-session"


def _request(content: str = "hello") -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
        "model": "gpt-4",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeferredInterventionIntegration:

    async def test_system_prompt_append_collected(self):
        """Async checker returns system_prompt_append → result collected with modify data."""
        checker = _FakeAsyncChecker(
            modified_data={"system_prompt_append": "Always verify identity first."}
        )
        interceptor = Interceptor([checker])

        # Request 1: POST_CALL fires the async checker
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        # Request 2: PRE_CALL collects the deferred result
        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        # The async MODIFY result should be collected
        mod_results = [
            r for r in result.results if r.checker_name == "fake_async_intervention"
        ]
        assert len(mod_results) == 1
        assert mod_results[0].decision == PolicyDecision.MODIFY
        assert mod_results[0].modified_data["system_prompt_append"] == (
            "Always verify identity first."
        )

    async def test_user_message_inject_collected(self):
        """Async checker returns user_message_inject → result collected with modify data."""
        checker = _FakeAsyncChecker(
            modified_data={"user_message_inject": "Please verify identity."}
        )
        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        mod_results = [
            r for r in result.results if r.checker_name == "fake_async_intervention"
        ]
        assert len(mod_results) == 1
        assert mod_results[0].modified_data["user_message_inject"] == (
            "Please verify identity."
        )

    async def test_context_reminder_collected(self):
        """Async checker returns context_reminder → result collected with modify data."""
        checker = _FakeAsyncChecker(
            modified_data={"context_reminder": "I must check credentials."}
        )
        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        mod_results = [
            r for r in result.results if r.checker_name == "fake_async_intervention"
        ]
        assert len(mod_results) == 1
        assert mod_results[0].modified_data["context_reminder"] == (
            "I must check credentials."
        )

    async def test_merge_adds_new_top_level_key(self):
        """When modification adds a new top-level key, modified_data is set on result."""
        checker = _FakeAsyncChecker(
            modified_data={
                "system_prompt_append": "Be safe.",
                "sentinel_applied": True,
            }
        )
        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next"), "req-002")

        assert result.allowed is True
        # The new key ensures modified_data != request_data
        assert result.modified_data is not None
        assert result.modified_data["sentinel_applied"] is True
        # System prompt was appended
        system_msg = result.modified_data["messages"][0]
        assert "Be safe." in system_msg["content"]
