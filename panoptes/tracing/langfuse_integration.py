"""
Simple HTTP-based Langfuse tracer for Panoptes events.

Uses Langfuse's public HTTP API directly to avoid Pydantic v1 + Python 3.13 issues.
LLM call tracing is handled by LiteLLM's langfuse_otel callback.
"""

import base64
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests

from panoptes.config.settings import LangfuseConfig

logger = logging.getLogger(__name__)


class PanoptesTracer:
    """
    Simple HTTP-based Langfuse tracer for Panoptes workflow events.

    This tracer uses Langfuse's public API directly via HTTP requests,
    bypassing the SDK and its Pydantic v1 compatibility issues.

    LLM call tracing is handled separately by LiteLLM's langfuse_otel callback.
    """

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self._session_traces: Dict[str, str] = {}  # session_id -> trace_id

        # Build base URL for API
        host = config.host.rstrip("/")
        self.api_url = f"{host}/api/public"

        # Basic auth header
        if config.public_key and config.secret_key:
            auth_str = f"{config.public_key}:{config.secret_key}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            self.headers = {
                "Authorization": f"Basic {auth_bytes}",
                "Content-Type": "application/json",
            }
            self._enabled = True
            logger.info(f"PanoptesTracer initialized (HTTP mode, host={host})")
        else:
            self.headers = {}
            self._enabled = False
            logger.warning("PanoptesTracer disabled: missing Langfuse credentials")

    def _get_or_create_trace(self, session_id: str) -> str:
        """Get existing trace ID for session or create new one."""
        if session_id not in self._session_traces:
            trace_id = str(uuid.uuid4())
            self._session_traces[session_id] = trace_id

            # Create the trace via API
            if self._enabled:
                try:
                    payload = {
                        "id": trace_id,
                        "name": "panoptes-session",
                        "sessionId": session_id,
                        "metadata": {
                            "panoptes_version": "0.1.0",
                            "session_id": session_id,
                        },
                    }
                    resp = requests.post(
                        f"{self.api_url}/traces",
                        headers=self.headers,
                        json=payload,
                        timeout=5,
                    )
                    if resp.status_code in (200, 201):
                        logger.debug(f"Created trace {trace_id} for session {session_id}")
                    else:
                        logger.warning(f"Failed to create trace: {resp.status_code} {resp.text}")
                except Exception as e:
                    logger.error(f"Error creating trace: {e}")

        return self._session_traces[session_id]

    def log_event(
        self,
        session_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a Panoptes event (span) to Langfuse.

        Use for workflow deviations, interventions, state transitions.
        """
        if not self._enabled:
            return

        trace_id = self._get_or_create_trace(session_id)
        span_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        try:
            payload = {
                "id": span_id,
                "traceId": trace_id,
                "name": name,
                "startTime": now,
                "endTime": now,
                "input": input_data or {"event_type": name},
                "output": output_data or {"logged": True},
                "metadata": {
                    "session_id": session_id,
                    "timestamp": now,
                    **(metadata or {}),
                },
            }
            resp = requests.post(
                f"{self.api_url}/spans",
                headers=self.headers,
                json=payload,
                timeout=5,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Logged event '{name}' for session {session_id} to Langfuse")
            else:
                logger.warning(f"Failed to log event: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error logging event: {e}")

    def log_state_transition(
        self,
        session_id: str,
        previous_state: str,
        new_state: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a workflow state transition."""
        self.log_event(
            session_id=session_id,
            name="state_transition",
            input_data={
                "previous_state": previous_state,
                "new_state": new_state,
            },
            output_data={
                "confidence": confidence,
            },
            metadata={
                "transition": f"{previous_state} -> {new_state}",
                **(metadata or {}),
            },
        )

    def log_intervention(
        self,
        session_id: str,
        intervention_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an intervention being applied."""
        self.log_event(
            session_id=session_id,
            name="intervention_applied",
            input_data={"intervention": intervention_name},
            metadata={
                "intervention_name": intervention_name,
                **(context or {}),
            },
        )

    def log_deviation(
        self,
        session_id: str,
        constraint_name: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a workflow deviation/constraint violation."""
        self.log_event(
            session_id=session_id,
            name="workflow_deviation",
            input_data={
                "constraint": constraint_name,
                "severity": severity,
            },
            metadata={
                "constraint_name": constraint_name,
                "severity": severity,
                **(details or {}),
            },
        )

    def log_llm_call(
        self,
        session_id: str,
        model: str,
        messages: list,
        response_content: Optional[str] = None,
        response_model: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Log an LLM call as a Langfuse Generation via the batch ingestion endpoint.

        Uses the /ingestion endpoint with generation-create event type which is
        the recommended way to log data and properly displays input/output in the UI.
        """
        if not self._enabled:
            return

        trace_id = self._get_or_create_trace(session_id)
        generation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        try:
            # Format messages for display - convert to proper format if needed
            formatted_input = messages if messages else []
            
            # Debug logging
            logger.info(f"log_llm_call: session={session_id}, model={model}")
            logger.info(f"  messages count: {len(formatted_input)}")
            logger.info(f"  response_content: {response_content[:100] if response_content else 'None'}...")
            logger.info(f"  usage: {usage}")

            # Build usage dict for Langfuse (using OpenAI format for compatibility)
            usage_data = None
            if usage:
                usage_data = {
                    "promptTokens": usage.get("prompt_tokens", 0),
                    "completionTokens": usage.get("completion_tokens", 0),
                    "totalTokens": usage.get("total_tokens", 0),
                }

            # Build the generation body for the ingestion event
            generation_body = {
                "id": generation_id,
                "traceId": trace_id,
                "name": "llm-call",
                "model": response_model or model,
                "modelParameters": {},
                "input": formatted_input,
                "output": response_content,
                "startTime": now,
                "endTime": now,
                "metadata": {
                    "session_id": session_id,
                    "requested_model": model,
                    "panoptes_traced": True,
                    **(metadata or {}),
                },
            }

            # Add usage if available
            if usage_data:
                generation_body["usage"] = usage_data

            # Wrap in batch ingestion format
            batch_payload = {
                "batch": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "generation-create",
                        "timestamp": now,
                        "body": generation_body,
                    }
                ]
            }

            logger.debug(f"Sending batch payload: {batch_payload}")

            resp = requests.post(
                f"{self.api_url}/ingestion",
                headers=self.headers,
                json=batch_payload,
                timeout=5,
            )
            logger.info(f"Langfuse /ingestion response: status={resp.status_code}")
            if resp.status_code in (200, 201, 207):
                logger.info(f"Logged LLM call for session {session_id} to Langfuse (model={model})")
                logger.debug(f"Langfuse response body: {resp.text[:500]}")
            else:
                logger.warning(f"Failed to log LLM call: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error logging LLM call: {e}")

    def end_trace(self, session_id: str) -> None:
        """Mark a session trace as complete."""
        if session_id in self._session_traces:
            trace_id = self._session_traces.pop(session_id)
            # Optionally update trace with completion status
            if self._enabled:
                try:
                    payload = {
                        "metadata": {"status": "completed"},
                    }
                    requests.patch(
                        f"{self.api_url}/traces/{trace_id}",
                        headers=self.headers,
                        json=payload,
                        timeout=5,
                    )
                    logger.debug(f"Ended trace for session {session_id}")
                except Exception as e:
                    logger.error(f"Error ending trace: {e}")

    def flush(self) -> None:
        """No-op for HTTP-based tracer (requests are synchronous)."""
        pass

    def shutdown(self) -> None:
        """Clean up any remaining traces."""
        for session_id in list(self._session_traces.keys()):
            self.end_trace(session_id)
        logger.info("PanoptesTracer shut down")
