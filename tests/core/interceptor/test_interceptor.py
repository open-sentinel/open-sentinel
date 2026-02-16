"""
Comprehensive tests for the Interceptor orchestrator.

Covers:
- Sync PRE_CALL checker flow (pass, fail, warn/modify, chaining, short-circuit)
- Sync POST_CALL checker flow (pass, fail, warn/modify)
- Async checker lifecycle (fire, collect, cross-request handoff)
- Async edge cases (task failure, still-running, cleanup, shutdown)
- Modification merging (messages append, system_prompt_append, key replace)
- Response modification (full replacement, dict merge, immutable response)
"""

import asyncio
import pytest
from typing import Optional, Dict, Any

from opensentinel.core.interceptor import (
    Interceptor,
    Checker,
    CheckPhase,
    CheckerMode,
    CheckDecision,
    CheckResult,
    CheckerContext,
)


# ---------------------------------------------------------------------------
# FakeChecker: configurable test double implementing the Checker ABC
# ---------------------------------------------------------------------------


class FakeChecker(Checker):
    """
    Test double for Checker.

    Returns a preconfigured CheckResult, optionally with a delay
    to simulate slow async work. Tracks how many times check() was called.
    """

    def __init__(
        self,
        *,
        checker_name: str = "fake",
        phase: CheckPhase = CheckPhase.PRE_CALL,
        mode: CheckerMode = CheckerMode.SYNC,
        decision: CheckDecision = CheckDecision.PASS,
        modified_data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        delay: float = 0,
        raise_on_check: Optional[Exception] = None,
    ):
        self._name = checker_name
        self._phase = phase
        self._mode = mode
        self._decision = decision
        self._modified_data = modified_data
        self._message = message
        self._delay = delay
        self._raise_on_check = raise_on_check
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def phase(self) -> CheckPhase:
        return self._phase

    @property
    def mode(self) -> CheckerMode:
        return self._mode

    async def check(self, context: CheckerContext) -> CheckResult:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise_on_check:
            raise self._raise_on_check
        return CheckResult(
            decision=self._decision,
            checker_name=self._name,
            modified_data=self._modified_data,
            message=self._message,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "test-session"
REQUEST_ID = "req-001"

def _request(content: str = "hello") -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], "model": "gpt-4"}


# ===========================================================================
# Sync PRE_CALL tests
# ===========================================================================


class TestSyncPreCall:

    async def test_pass_unchanged(self):
        """Single PASS checker — request goes through unchanged."""
        checker = FakeChecker(phase=CheckPhase.PRE_CALL)
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None
        assert len(result.results) == 1
        assert result.results[0].decision == CheckDecision.PASS

    async def test_fail_blocks(self):
        """FAIL checker blocks the request."""
        checker = FakeChecker(
            phase=CheckPhase.PRE_CALL,
            decision=CheckDecision.FAIL,
            message="forbidden",
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.results[0].decision == CheckDecision.FAIL
        assert result.results[0].message == "forbidden"

    async def test_warn_merges_modifications(self):
        """WARN checker with modified_data merges into request."""
        mods = {"temperature": 0.5}
        checker = FakeChecker(
            phase=CheckPhase.PRE_CALL,
            decision=CheckDecision.WARN,
            modified_data=mods,
        )
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["temperature"] == 0.5
        # Original messages still present
        assert len(result.modified_data["messages"]) == 1

    async def test_multiple_checkers_chain(self):
        """Two WARN checkers — modifications chain correctly."""
        c1 = FakeChecker(
            checker_name="c1",
            phase=CheckPhase.PRE_CALL,
            decision=CheckDecision.WARN,
            modified_data={"key_a": 1},
        )
        c2 = FakeChecker(
            checker_name="c2",
            phase=CheckPhase.PRE_CALL,
            decision=CheckDecision.WARN,
            modified_data={"key_b": 2},
        )
        interceptor = Interceptor([c1, c2])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data["key_a"] == 1
        assert result.modified_data["key_b"] == 2
        assert c1.call_count == 1
        assert c2.call_count == 1

    async def test_fail_short_circuits(self):
        """First checker FAILs — second checker never runs."""
        c1 = FakeChecker(
            checker_name="blocker",
            phase=CheckPhase.PRE_CALL,
            decision=CheckDecision.FAIL,
        )
        c2 = FakeChecker(checker_name="skipped", phase=CheckPhase.PRE_CALL)
        interceptor = Interceptor([c1, c2])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert c1.call_count == 1
        assert c2.call_count == 0  # Never reached

    async def test_checker_exception_becomes_fail(self):
        """Exception in a sync checker produces FAIL result."""
        checker = FakeChecker(
            phase=CheckPhase.PRE_CALL,
            raise_on_check=RuntimeError("kaboom"),
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert "kaboom" in result.results[0].message


# ===========================================================================
# Sync POST_CALL tests
# ===========================================================================


class TestSyncPostCall:

    async def test_pass_unchanged(self):
        """PASS checker — response goes through unchanged."""
        checker = FakeChecker(phase=CheckPhase.POST_CALL)
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_post_call(SESSION, req, {"answer": "hi"}, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_fail_blocks(self):
        """FAIL checker — response is blocked."""
        checker = FakeChecker(
            phase=CheckPhase.POST_CALL,
            decision=CheckDecision.FAIL,
            message="toxic content",
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.results[0].message == "toxic content"

    async def test_warn_merges_response(self):
        """WARN checker — modifications applied to dict response."""
        checker = FakeChecker(
            phase=CheckPhase.POST_CALL,
            decision=CheckDecision.WARN,
            modified_data={"filtered": True},
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "ok"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["filtered"] is True
        assert result.modified_data["answer"] == "ok"


# ===========================================================================
# Async checker lifecycle tests
# ===========================================================================


class TestAsyncCheckerLifecycle:

    async def test_async_checker_started_during_post_call(self):
        """After run_post_call, async task is stored in _running_tasks."""
        async_checker = FakeChecker(
            checker_name="async_c",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=0.5,  # Slow enough to still be running
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert SESSION in interceptor._running_tasks
        assert len(interceptor._running_tasks[SESSION]) == 1
        # Cleanup
        await interceptor.shutdown()

    async def test_cross_request_handoff(self):
        """
        Full lifecycle: async checker runs during POST_CALL of req 1,
        its result is collected during PRE_CALL of req 2.
        """
        async_checker = FakeChecker(
            checker_name="async_monitor",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=CheckDecision.PASS,
            delay=0.01,  # Fast enough to complete before next call
        )
        interceptor = Interceptor([async_checker])

        # Request 1: POST_CALL fires the async checker
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        # Wait a bit for the async task to complete
        await asyncio.sleep(0.05)

        # Request 2: PRE_CALL collects the async result
        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        # The async result should be in the results
        assert len(result.results) >= 1
        async_results = [r for r in result.results if r.checker_name == "async_monitor"]
        assert len(async_results) == 1
        assert async_results[0].decision == CheckDecision.PASS

    async def test_async_fail_blocks_next_request(self):
        """Async checker returns FAIL — next PRE_CALL blocks."""
        async_checker = FakeChecker(
            checker_name="async_blocker",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=CheckDecision.FAIL,
            message="violation detected async",
            delay=0.01,
        )
        interceptor = Interceptor([async_checker])

        # Req 1 POST_CALL
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Req 2 PRE_CALL — should be blocked
        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is False
        fail_results = [r for r in result.results if r.decision == CheckDecision.FAIL]
        assert len(fail_results) == 1
        assert "violation detected async" in fail_results[0].message

    async def test_async_warn_merges_into_next_request(self):
        """Async checker returns WARN with modifications — applied on next PRE_CALL."""
        async_checker = FakeChecker(
            checker_name="async_modifier",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=CheckDecision.WARN,
            modified_data={"extra_key": "injected_value"},
            delay=0.01,
        )
        interceptor = Interceptor([async_checker])

        # Req 1 POST_CALL
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Req 2 PRE_CALL — should have collected and applied modifications
        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        # The async WARN result should be collected
        warn_results = [r for r in result.results if r.checker_name == "async_modifier"]
        assert len(warn_results) == 1
        assert warn_results[0].decision == CheckDecision.WARN
        # modified_data should contain the injected key
        assert result.modified_data is not None
        assert result.modified_data["extra_key"] == "injected_value"
        # Original messages should still be present
        assert "messages" in result.modified_data


# ===========================================================================
# Async edge cases
# ===========================================================================


class TestAsyncEdgeCases:

    async def test_async_exception_becomes_fail(self):
        """Async checker that raises — produces FAIL result on next collection."""
        async_checker = FakeChecker(
            checker_name="async_crasher",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            raise_on_check=RuntimeError("async boom"),
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Should still be allowed=False because async errors produce FAIL
        assert result.allowed is False
        fail_results = [r for r in result.results if r.decision == CheckDecision.FAIL]
        assert len(fail_results) == 1
        assert "async boom" in fail_results[0].message

    async def test_still_running_not_collected(self):
        """Async task that isn't done yet stays in _running_tasks."""
        slow_checker = FakeChecker(
            checker_name="slow_async",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,  # Very slow, won't finish
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        # Don't wait — collect immediately
        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Not collected yet, so no async results in the output
        async_results = [r for r in result.results if r.checker_name == "slow_async"]
        assert len(async_results) == 0

        # Task should still be running
        assert SESSION in interceptor._running_tasks
        assert len(interceptor._running_tasks[SESSION]) == 1

        # Cleanup
        await interceptor.shutdown()

    async def test_cleanup_session_cancels_tasks(self):
        """cleanup_session cancels running tasks and clears pending results."""
        slow_checker = FakeChecker(
            checker_name="cleanup_target",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._running_tasks

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._running_tasks
        assert SESSION not in interceptor._pending_async

    async def test_shutdown_cleans_all_sessions(self):
        """shutdown cancels tasks across all sessions."""
        slow_checker = FakeChecker(
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call("session-a", _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call("session-b", _request(), {"r": 2}, REQUEST_ID)

        assert len(interceptor._running_tasks) == 2

        await interceptor.shutdown()

        assert len(interceptor._running_tasks) == 0

    async def test_no_pending_async_on_first_request(self):
        """First PRE_CALL with no prior async results works cleanly."""
        interceptor = Interceptor([])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert len(result.results) == 0


# ===========================================================================
# Modification merging tests
# ===========================================================================


class TestModificationMerging:

    async def test_messages_appended(self):
        """Messages lists are appended, not replaced."""
        interceptor = Interceptor([])
        base = {"messages": [{"role": "user", "content": "hi"}]}
        mods = {"messages": [{"role": "assistant", "content": "hello"}]}

        result = interceptor._merge_modifications(base, mods)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "hi"
        assert result["messages"][1]["content"] == "hello"

    async def test_system_prompt_append_to_existing(self):
        """system_prompt_append appends to existing system message."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        }
        mods = {"system_prompt_append": "Always be safe."}

        result = interceptor._merge_modifications(base, mods)

        assert "You are helpful." in result["messages"][0]["content"]
        assert "Always be safe." in result["messages"][0]["content"]

    async def test_system_prompt_append_creates_new(self):
        """system_prompt_append creates system message if none exists."""
        interceptor = Interceptor([])
        base = {"messages": [{"role": "user", "content": "hi"}]}
        mods = {"system_prompt_append": "Be safe."}

        result = interceptor._merge_modifications(base, mods)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be safe."
        assert result["messages"][1]["role"] == "user"

    async def test_other_keys_replaced(self):
        """Non-special keys are replaced directly."""
        interceptor = Interceptor([])
        base = {"model": "gpt-4", "temperature": 0.7}
        mods = {"temperature": 0.2, "max_tokens": 100}

        result = interceptor._merge_modifications(base, mods)

        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 100
        assert result["model"] == "gpt-4"


# ===========================================================================
# Response modification tests
# ===========================================================================


class TestResponseModification:

    async def test_full_replacement_via_response_key(self):
        """{"response": ...} replaces entire response."""
        interceptor = Interceptor([])
        original = {"answer": "old"}
        mods = {"response": {"answer": "new", "extra": True}}

        result = interceptor._merge_response_modifications(original, mods)

        assert result == {"answer": "new", "extra": True}

    async def test_dict_response_merge(self):
        """Dict responses get fields merged."""
        interceptor = Interceptor([])
        original = {"answer": "old", "keep": True}
        mods = {"answer": "new"}

        result = interceptor._merge_response_modifications(original, mods)

        assert result["answer"] == "new"
        assert result["keep"] is True

    async def test_immutable_response_ignored(self):
        """Non-dict response with modifications returns unchanged."""
        interceptor = Interceptor([])

        class ImmutableResponse:
            pass

        original = ImmutableResponse()
        mods = {"filtered": True}

        result = interceptor._merge_response_modifications(original, mods)

        # Should return the original object unchanged
        assert result is original


# ===========================================================================
# Interceptor init categorization
# ===========================================================================


class TestInterceptorInit:

    async def test_categorizes_checkers_correctly(self):
        """Checkers are bucketed into sync_pre, sync_post, and async."""
        sync_pre = FakeChecker(checker_name="sp", phase=CheckPhase.PRE_CALL, mode=CheckerMode.SYNC)
        sync_post = FakeChecker(checker_name="spo", phase=CheckPhase.POST_CALL, mode=CheckerMode.SYNC)
        async_pre = FakeChecker(checker_name="ap", phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC)
        async_post = FakeChecker(checker_name="apo", phase=CheckPhase.POST_CALL, mode=CheckerMode.ASYNC)

        interceptor = Interceptor([sync_pre, sync_post, async_pre, async_post])

        assert len(interceptor._sync_pre_call) == 1
        assert len(interceptor._sync_post_call) == 1
        assert len(interceptor._async_checkers) == 2

    async def test_empty_checkers_list(self):
        """Interceptor with no checkers still works."""
        interceptor = Interceptor([])

        pre = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        post = await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert pre.allowed is True
        assert post.allowed is True
