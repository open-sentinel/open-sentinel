"""
Middleware for extracting and propagating context through Panoptes.

Key responsibilities:
- Session ID extraction from various sources
- Workflow context extraction
- Request/response transformation
"""

import uuid
from typing import Optional, Tuple, Dict, Any


class SessionExtractor:
    """
    Extract session ID from LLM request data.

    Session IDs are used to:
    - Group related LLM calls together
    - Maintain workflow state across calls
    - Correlate traces in Langfuse

    Extraction priority:
    1. Custom header: x-panoptes-session-id (or x-session-id)
    2. Metadata field: metadata.session_id
    3. OpenAI user field
    4. Random UUID
    """

    @staticmethod
    def extract_session_id(
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Extract session ID from request data and headers.

        Args:
            data: Request data dict (messages, metadata, etc.)
            headers: Optional HTTP headers

        Returns:
            Session ID string
        """
        # 1. Check headers
        if headers:
            for header_name in ["x-panoptes-session-id", "x-session-id"]:
                if session_id := headers.get(header_name):
                    return session_id

        # 2. Check metadata
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

        # 5. Generate random UUID
        return str(uuid.uuid4())


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

        # Check headers
        if headers:
            if workflow := headers.get("x-panoptes-workflow"):
                context["workflow_name"] = workflow
            if state := headers.get("x-panoptes-expected-state"):
                context["expected_state"] = state
            if headers.get("x-panoptes-disable-intervention", "").lower() == "true":
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
