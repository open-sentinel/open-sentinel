"""
Middleware for extracting and propagating context through Panoptes.

Key responsibilities:
- Session ID extraction from various sources (headers, metadata, body fields)
- Workflow context extraction
- Request/response transformation

Session extraction is designed to work with:
- Direct HTTP headers (when called from FastAPI middleware)
- LiteLLM proxy callbacks (where HTTP headers are embedded in
  ``data["proxy_server_request"]["headers"]`` and ``data["metadata"]["headers"]``)
- OpenClaw and other agent frameworks that pass custom headers or metadata
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header names checked for session identity, in priority order.
# Case-insensitive lookup is performed by _get_header().
# ---------------------------------------------------------------------------
_SESSION_HEADER_NAMES: List[str] = [
    "x-panoptes-session-id",
    "x-session-id",
]


def _get_header(
    headers: Dict[str, str],
    name: str,
) -> Optional[str]:
    """Case-insensitive header lookup.

    HTTP headers are case-insensitive per RFC 7230.  LiteLLM sometimes
    stores them lower-cased, sometimes not — this helper normalises.
    """
    # Fast path: exact match (common when LiteLLM already lower-cased them)
    val = headers.get(name)
    if val:
        return val
    # Slow path: case-insensitive scan
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower and v:
            return v
    return None


class SessionExtractor:
    """
    Extract session ID from LLM request data.

    Session IDs are used to:
    - Group related LLM calls together
    - Maintain workflow state across calls
    - Correlate traces in observability backends (Langfuse, Jaeger, etc.)

    Extraction priority (first match wins):
    1. Explicit ``headers`` parameter (direct HTTP header access)
    2. HTTP headers embedded by LiteLLM in the data dict:
       a. ``data["proxy_server_request"]["headers"]``
       b. ``data["metadata"]["headers"]``
    3. ``metadata.session_id`` / ``metadata.panoptes_session_id``
    4. ``metadata.run_id`` (LangChain convention)
    5. ``user`` field (OpenAI convention)
    6. ``thread_id`` field (OpenAI Assistants convention)
    7. Random UUID (last resort — logged as warning)
    """

    @staticmethod
    def _resolve_headers(
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Resolve the best available HTTP headers dict.

        Priority:
        1. Explicitly passed ``headers`` (caller already has them).
        2. ``data["proxy_server_request"]["headers"]`` — set by LiteLLM proxy
           for every request through ``add_litellm_data_to_request()``.
        3. ``data["metadata"]["headers"]`` — also set by LiteLLM proxy
           (duplicate of #2 for guardrails access).

        Returns ``None`` if no headers can be found.
        """
        if headers:
            return headers

        # LiteLLM proxy embeds the original HTTP headers here:
        psr = data.get("proxy_server_request")
        if isinstance(psr, dict):
            psr_headers = psr.get("headers")
            if isinstance(psr_headers, dict) and psr_headers:
                return psr_headers

        # Fallback: metadata.headers (also set by LiteLLM)
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            meta_headers = metadata.get("headers")
            if isinstance(meta_headers, dict) and meta_headers:
                return meta_headers

        # LiteLLM library mode: litellm_params.metadata.headers
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_meta = litellm_params.get("metadata")
            if isinstance(lp_meta, dict):
                lp_headers = lp_meta.get("headers")
                if isinstance(lp_headers, dict) and lp_headers:
                    return lp_headers

        return None

    @staticmethod
    def extract_session_id(
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Extract session ID from request data and/or headers.

        Works in all deployment modes:
        - **LiteLLM Proxy mode**: HTTP headers are automatically extracted
          from ``data["proxy_server_request"]["headers"]``.
        - **LiteLLM Library/Router mode**: Headers from
          ``data["litellm_params"]["metadata"]["headers"]``.
        - **Direct call**: Pass ``headers`` explicitly.
        - **OpenClaw / other frameworks**: Send
          ``x-panoptes-session-id`` header or
          ``metadata.session_id`` in the request body.

        Args:
            data: Request data dict (messages, metadata, etc.)
            headers: Optional explicit HTTP headers (takes top priority)

        Returns:
            A deterministic session ID string.
        """
        resolved_headers = SessionExtractor._resolve_headers(data, headers)

        # 1. Check HTTP headers (case-insensitive)
        if resolved_headers:
            for header_name in _SESSION_HEADER_NAMES:
                session_id = _get_header(resolved_headers, header_name)
                if session_id:
                    return session_id

        # 2. Check metadata fields
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            if session_id := metadata.get("session_id"):
                return str(session_id)
            if session_id := metadata.get("panoptes_session_id"):
                return str(session_id)
            # LangChain often uses run_id
            if run_id := metadata.get("run_id"):
                return str(run_id)

        # 3. Check user field (OpenAI pattern)
        if user := data.get("user"):
            return f"user_{user}"

        # 4. Check for thread_id (OpenAI Assistants)
        if thread_id := data.get("thread_id"):
            return str(thread_id)

        # 5. Last resort: random UUID
        generated = str(uuid.uuid4())
        logger.warning(
            "No session ID found in request headers or metadata. "
            "Generated fallback UUID: %s. "
            "Set 'x-panoptes-session-id' header or 'metadata.session_id' "
            "in the request body for reliable session tracking.",
            generated,
        )
        return generated


class WorkflowContextExtractor:
    """
    Extract workflow-specific context from requests.

    Allows customers to specify:
    - Which workflow to use
    - Expected state hints
    - Custom intervention preferences
    """

    @staticmethod
    def extract_context(
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract workflow context from request.

        Args:
            data: Request data
            headers: Optional HTTP headers

        Returns:
            Dict with workflow context:
            - workflow_name: Name of workflow to use
            - expected_state: Hint for expected state
            - disable_intervention: Whether to skip intervention
            - custom_metadata: Any additional metadata
        """
        context: Dict[str, Any] = {}

        # Resolve headers using the same logic as SessionExtractor
        resolved_headers = SessionExtractor._resolve_headers(data, headers)

        # Check headers
        if resolved_headers:
            if workflow := _get_header(resolved_headers, "x-panoptes-workflow"):
                context["workflow_name"] = workflow
            if state := _get_header(resolved_headers, "x-panoptes-expected-state"):
                context["expected_state"] = state
            disable = _get_header(
                resolved_headers, "x-panoptes-disable-intervention"
            )
            if disable and disable.lower() == "true":
                context["disable_intervention"] = True

        # Check metadata
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            if workflow := metadata.get("panoptes_workflow"):
                context["workflow_name"] = workflow
            if state := metadata.get("panoptes_expected_state"):
                context["expected_state"] = state
            if metadata.get("panoptes_disable_intervention"):
                context["disable_intervention"] = True

            # Collect any panoptes_ prefixed metadata
            custom = {}
            for key, value in metadata.items():
                if key.startswith("panoptes_") and key not in [
                    "panoptes_workflow",
                    "panoptes_expected_state",
                    "panoptes_disable_intervention",
                    "panoptes_session_id",
                ]:
                    custom[key[9:]] = value  # Strip prefix

            if custom:
                context["custom_metadata"] = custom

        return context


class ResponseTransformer:
    """
    Transform responses to include Panoptes metadata.

    Optionally adds headers or metadata about:
    - Current workflow state
    - Intervention status
    - Constraint violations
    """

    @staticmethod
    def add_panoptes_headers(
        response_headers: Dict[str, str],
        workflow_state: Optional[str] = None,
        intervention_applied: Optional[str] = None,
        violations: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Add Panoptes headers to response.

        Args:
            response_headers: Existing response headers
            workflow_state: Current workflow state
            intervention_applied: Name of intervention if applied
            violations: List of constraint violations

        Returns:
            Updated headers dict
        """
        headers = dict(response_headers)

        if workflow_state:
            headers["x-panoptes-workflow-state"] = workflow_state

        if intervention_applied:
            headers["x-panoptes-intervention"] = intervention_applied

        if violations:
            headers["x-panoptes-violations"] = ",".join(violations)

        return headers
