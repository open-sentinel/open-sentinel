"""
Tests for panoptes.proxy.middleware — session extraction, workflow context,
and response transformation.

Covers the critical scenario where HTTP headers arrive embedded inside
the LiteLLM data dict rather than as a separate parameter.
"""

import uuid

import pytest

from panoptes.proxy.middleware import (
    SessionExtractor,
    WorkflowContextExtractor,
    ResponseTransformer,
    _get_header,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------
def _litellm_proxy_data(
    *,
    headers: dict | None = None,
    metadata: dict | None = None,
    user: str | None = None,
    thread_id: str | None = None,
    messages: list | None = None,
) -> dict:
    """Build a data dict that looks like what LiteLLM proxy passes to callbacks."""
    data: dict = {}
    if messages is not None:
        data["messages"] = messages
    if user is not None:
        data["user"] = user
    if thread_id is not None:
        data["thread_id"] = thread_id
    if metadata is not None:
        data["metadata"] = metadata
    # Simulate LiteLLM's proxy_server_request injection
    if headers is not None:
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/chat/completions",
            "method": "POST",
            "headers": headers,
            "body": {},
        }
    return data


# ===========================================================================
# _get_header — case-insensitive lookup
# ===========================================================================
class TestGetHeader:
    def test_exact_match(self):
        assert _get_header({"x-session-id": "abc"}, "x-session-id") == "abc"

    def test_case_insensitive(self):
        assert _get_header({"X-Session-Id": "abc"}, "x-session-id") == "abc"

    def test_missing_header(self):
        assert _get_header({"other": "val"}, "x-session-id") is None

    def test_empty_value_returns_none(self):
        assert _get_header({"x-session-id": ""}, "x-session-id") is None

    def test_empty_dict(self):
        assert _get_header({}, "x-session-id") is None


# ===========================================================================
# SessionExtractor._resolve_headers
# ===========================================================================
class TestResolveHeaders:
    def test_explicit_headers_take_priority(self):
        explicit = {"x-panoptes-session-id": "explicit"}
        data = _litellm_proxy_data(headers={"x-panoptes-session-id": "embedded"})
        resolved = SessionExtractor._resolve_headers(data, explicit)
        assert resolved is explicit

    def test_proxy_server_request_headers(self):
        data = _litellm_proxy_data(headers={"x-panoptes-session-id": "from-proxy"})
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-panoptes-session-id": "from-proxy"}

    def test_metadata_headers_fallback(self):
        data = {"metadata": {"headers": {"x-panoptes-session-id": "from-meta"}}}
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-panoptes-session-id": "from-meta"}

    def test_litellm_params_metadata_headers(self):
        data = {
            "litellm_params": {
                "metadata": {
                    "headers": {"x-panoptes-session-id": "from-lp"}
                }
            }
        }
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-panoptes-session-id": "from-lp"}

    def test_no_headers_returns_none(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        assert SessionExtractor._resolve_headers(data, None) is None

    def test_empty_proxy_server_request_headers(self):
        data = {"proxy_server_request": {"headers": {}}}
        assert SessionExtractor._resolve_headers(data, None) is None

    def test_non_dict_proxy_server_request(self):
        data = {"proxy_server_request": "not-a-dict"}
        assert SessionExtractor._resolve_headers(data, None) is None


# ===========================================================================
# SessionExtractor.extract_session_id — header-based extraction
# ===========================================================================
class TestExtractSessionIdFromHeaders:
    def test_explicit_header(self):
        data = {"messages": []}
        headers = {"x-panoptes-session-id": "sess-123"}
        assert SessionExtractor.extract_session_id(data, headers) == "sess-123"

    def test_x_session_id_header(self):
        data = {"messages": []}
        headers = {"x-session-id": "sess-456"}
        assert SessionExtractor.extract_session_id(data, headers) == "sess-456"

    def test_panoptes_header_takes_priority_over_x_session_id(self):
        data = {"messages": []}
        headers = {
            "x-panoptes-session-id": "panoptes",
            "x-session-id": "generic",
        }
        assert SessionExtractor.extract_session_id(data, headers) == "panoptes"

    def test_case_insensitive_header(self):
        data = {"messages": []}
        headers = {"X-Panoptes-Session-Id": "mixed-case"}
        assert SessionExtractor.extract_session_id(data, headers) == "mixed-case"

    def test_litellm_embedded_headers(self):
        """Core OpenClaw scenario: headers embedded by LiteLLM proxy."""
        data = _litellm_proxy_data(
            headers={"x-panoptes-session-id": "openclaw-session-42"}
        )
        # No explicit headers param — must pick from data dict
        assert SessionExtractor.extract_session_id(data) == "openclaw-session-42"

    def test_litellm_metadata_headers(self):
        """Fallback: headers in data["metadata"]["headers"]."""
        data = {"metadata": {"headers": {"x-session-id": "meta-sess"}}}
        assert SessionExtractor.extract_session_id(data) == "meta-sess"


# ===========================================================================
# SessionExtractor.extract_session_id — metadata-based extraction
# ===========================================================================
class TestExtractSessionIdFromMetadata:
    def test_session_id_in_metadata(self):
        data = {"metadata": {"session_id": "meta-123"}}
        assert SessionExtractor.extract_session_id(data) == "meta-123"

    def test_panoptes_session_id_in_metadata(self):
        data = {"metadata": {"panoptes_session_id": "pan-456"}}
        assert SessionExtractor.extract_session_id(data) == "pan-456"

    def test_run_id_in_metadata(self):
        data = {"metadata": {"run_id": "langchain-run-789"}}
        assert SessionExtractor.extract_session_id(data) == "langchain-run-789"

    def test_session_id_takes_priority_over_run_id(self):
        data = {"metadata": {"session_id": "sess", "run_id": "run"}}
        assert SessionExtractor.extract_session_id(data) == "sess"

    def test_header_takes_priority_over_metadata(self):
        data = _litellm_proxy_data(
            headers={"x-panoptes-session-id": "from-header"},
            metadata={"session_id": "from-meta"},
        )
        assert SessionExtractor.extract_session_id(data) == "from-header"


# ===========================================================================
# SessionExtractor.extract_session_id — body field extraction
# ===========================================================================
class TestExtractSessionIdFromBodyFields:
    def test_user_field(self):
        data = {"user": "alice"}
        assert SessionExtractor.extract_session_id(data) == "user_alice"

    def test_thread_id_field(self):
        data = {"thread_id": "thread_abc"}
        assert SessionExtractor.extract_session_id(data) == "thread_abc"

    def test_user_takes_priority_over_thread_id(self):
        data = {"user": "bob", "thread_id": "thread_xyz"}
        assert SessionExtractor.extract_session_id(data) == "user_bob"

    def test_metadata_takes_priority_over_user(self):
        data = {"user": "bob", "metadata": {"session_id": "explicit"}}
        assert SessionExtractor.extract_session_id(data) == "explicit"


# ===========================================================================
# SessionExtractor.extract_session_id — fallback UUID
# ===========================================================================
class TestExtractSessionIdFallback:
    def test_generates_uuid_when_nothing_available(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        result = SessionExtractor.extract_session_id(data)
        # Should be a valid UUID
        uuid.UUID(result)  # raises ValueError if not valid

    def test_different_calls_produce_different_uuids(self):
        data = {"messages": []}
        id1 = SessionExtractor.extract_session_id(data)
        id2 = SessionExtractor.extract_session_id(data)
        assert id1 != id2

    def test_uuid_fallback_logs_warning(self, caplog):
        """Ensure a warning is logged when falling back to UUID."""
        import logging

        data = {"messages": []}
        with caplog.at_level(logging.WARNING, logger="panoptes.proxy.middleware"):
            SessionExtractor.extract_session_id(data)
        assert "No session ID found" in caplog.text
        assert "x-panoptes-session-id" in caplog.text


# ===========================================================================
# SessionExtractor — multi-agent isolation scenarios
# ===========================================================================
class TestMultiAgentIsolation:
    """
    Simulate multiple concurrent agent sessions to verify that session IDs
    are correctly isolated per agent.
    """

    def test_different_headers_produce_different_sessions(self):
        agent_a = _litellm_proxy_data(
            headers={"x-panoptes-session-id": "agent-A-session"},
            messages=[{"role": "user", "content": "Hello from A"}],
        )
        agent_b = _litellm_proxy_data(
            headers={"x-panoptes-session-id": "agent-B-session"},
            messages=[{"role": "user", "content": "Hello from B"}],
        )
        assert SessionExtractor.extract_session_id(agent_a) == "agent-A-session"
        assert SessionExtractor.extract_session_id(agent_b) == "agent-B-session"

    def test_different_metadata_produce_different_sessions(self):
        agent_a = {"metadata": {"session_id": "meta-A"}}
        agent_b = {"metadata": {"session_id": "meta-B"}}
        assert SessionExtractor.extract_session_id(agent_a) == "meta-A"
        assert SessionExtractor.extract_session_id(agent_b) == "meta-B"

    def test_mixed_sources_still_isolate(self):
        """Agent A uses header, Agent B uses metadata."""
        agent_a = _litellm_proxy_data(
            headers={"x-panoptes-session-id": "header-sess"},
        )
        agent_b = {"metadata": {"session_id": "meta-sess"}}
        assert SessionExtractor.extract_session_id(agent_a) == "header-sess"
        assert SessionExtractor.extract_session_id(agent_b) == "meta-sess"


# ===========================================================================
# WorkflowContextExtractor
# ===========================================================================
class TestWorkflowContextExtractor:
    def test_extract_from_explicit_headers(self):
        data = {"messages": []}
        headers = {
            "x-panoptes-workflow": "customer-support",
            "x-panoptes-expected-state": "greeting",
            "x-panoptes-disable-intervention": "true",
        }
        context = WorkflowContextExtractor.extract_context(data, headers)
        assert context["workflow_name"] == "customer-support"
        assert context["expected_state"] == "greeting"
        assert context["disable_intervention"] is True

    def test_extract_from_embedded_headers(self):
        """LiteLLM proxy mode: headers in data dict."""
        data = _litellm_proxy_data(
            headers={
                "x-panoptes-workflow": "order-flow",
                "x-panoptes-expected-state": "cart",
            }
        )
        context = WorkflowContextExtractor.extract_context(data)
        assert context["workflow_name"] == "order-flow"
        assert context["expected_state"] == "cart"

    def test_extract_from_metadata(self):
        data = {
            "metadata": {
                "panoptes_workflow": "onboarding",
                "panoptes_expected_state": "verify_email",
                "panoptes_disable_intervention": True,
            }
        }
        context = WorkflowContextExtractor.extract_context(data)
        assert context["workflow_name"] == "onboarding"
        assert context["expected_state"] == "verify_email"
        assert context["disable_intervention"] is True

    def test_custom_metadata_collected(self):
        data = {
            "metadata": {
                "panoptes_custom_key": "custom_value",
                "panoptes_another": 42,
            }
        }
        context = WorkflowContextExtractor.extract_context(data)
        assert context["custom_metadata"] == {"custom_key": "custom_value", "another": 42}

    def test_empty_data_returns_empty_context(self):
        assert WorkflowContextExtractor.extract_context({}) == {}

    def test_case_insensitive_headers(self):
        data = {}
        headers = {"X-Panoptes-Workflow": "test-workflow"}
        context = WorkflowContextExtractor.extract_context(data, headers)
        assert context["workflow_name"] == "test-workflow"


# ===========================================================================
# ResponseTransformer
# ===========================================================================
class TestResponseTransformer:
    def test_add_workflow_state(self):
        result = ResponseTransformer.add_panoptes_headers(
            {}, workflow_state="verification"
        )
        assert result["x-panoptes-workflow-state"] == "verification"

    def test_add_intervention(self):
        result = ResponseTransformer.add_panoptes_headers(
            {}, intervention_applied="redirect_to_verification"
        )
        assert result["x-panoptes-intervention"] == "redirect_to_verification"

    def test_add_violations(self):
        result = ResponseTransformer.add_panoptes_headers(
            {}, violations=["skip_auth", "missing_context"]
        )
        assert result["x-panoptes-violations"] == "skip_auth,missing_context"

    def test_preserves_existing_headers(self):
        existing = {"content-type": "application/json"}
        result = ResponseTransformer.add_panoptes_headers(
            existing, workflow_state="done"
        )
        assert result["content-type"] == "application/json"
        assert result["x-panoptes-workflow-state"] == "done"

    def test_no_panoptes_headers_when_none(self):
        result = ResponseTransformer.add_panoptes_headers({})
        assert "x-panoptes-workflow-state" not in result
        assert "x-panoptes-intervention" not in result
        assert "x-panoptes-violations" not in result
