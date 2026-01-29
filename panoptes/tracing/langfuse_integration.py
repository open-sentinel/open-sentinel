"""
Langfuse integration for Panoptes tracing.

Provides session-aware tracing with workflow state annotations.

Key concepts:
- Trace: Groups all observations for a session
- Generation: Individual LLM call within a trace
- Event: Point-in-time occurrence (deviation, intervention, etc.)

Usage:
    tracer = PanoptesTracer(langfuse_config)
    tracer.start_generation(session_id, "llm-call", input_data)
    # ... LLM call happens ...
    tracer.end_generation(session_id, output_data)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from panoptes.config.settings import LangfuseConfig

logger = logging.getLogger(__name__)


class PanoptesTracer:
    """
    Manages Langfuse tracing with Panoptes-specific semantics.

    Provides a simplified interface for:
    - Starting/ending traces for sessions
    - Recording LLM generations
    - Logging workflow events (deviations, interventions)

    Thread-safe for concurrent session handling.
    """

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self._client = None
        self._traces: Dict[str, Any] = {}  # session_id -> trace
        self._generations: Dict[str, Any] = {}  # session_id -> current generation

    @property
    def client(self):
        """Lazy-load Langfuse client."""
        if self._client is None:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.config.public_key,
                    secret_key=self.config.secret_key,
                    host=self.config.host,
                    debug=self.config.debug,
                )
                logger.info(f"Langfuse client initialized (host={self.config.host})")
            except ImportError:
                logger.warning("Langfuse not installed, tracing disabled")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                return None

        return self._client

    def _get_or_create_trace(self, session_id: str) -> Optional[Any]:
        """Get existing trace for session or create new one."""
        if not self.client:
            return None

        if session_id not in self._traces:
            trace = self.client.trace(
                name="panoptes-session",
                session_id=session_id,
                metadata={
                    "panoptes_version": "0.1.0",
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            self._traces[session_id] = trace
            logger.debug(f"Created new trace for session {session_id}")

        return self._traces[session_id]

    def start_generation(
        self,
        session_id: str,
        name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Start a new generation (LLM call) observation.

        Args:
            session_id: Session identifier
            name: Name for the generation (e.g., "llm-completion")
            input_data: Input to the LLM (messages, etc.)
            metadata: Additional metadata (workflow state, model, etc.)

        Returns:
            Generation ID if successful, None otherwise
        """
        trace = self._get_or_create_trace(session_id)
        if not trace:
            return None

        try:
            generation = trace.generation(
                name=name,
                input=self._sanitize_input(input_data),
                metadata={
                    "session_id": session_id,
                    "started_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            )
            self._generations[session_id] = generation
            logger.debug(f"Started generation for session {session_id}")
            return generation.id
        except Exception as e:
            logger.error(f"Failed to start generation: {e}")
            return None

    def end_generation(
        self,
        session_id: str,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        End a generation with output data.

        Args:
            session_id: Session identifier
            output: LLM output/response
            metadata: Additional metadata (state transitions, etc.)
            usage: Token usage (prompt_tokens, completion_tokens, total_tokens)
        """
        generation = self._generations.pop(session_id, None)
        if not generation:
            logger.debug(f"No active generation for session {session_id}")
            return

        try:
            end_data = {
                "output": self._sanitize_output(output),
                "metadata": {
                    "ended_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            }

            if usage:
                end_data["usage"] = usage

            generation.end(**end_data)
            logger.debug(f"Ended generation for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to end generation: {e}")

    def log_event(
        self,
        session_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a point-in-time event.

        Use for:
        - Workflow deviations
        - Interventions applied
        - State transitions
        - Errors

        Args:
            session_id: Session identifier
            name: Event name (e.g., "workflow_deviation", "intervention_applied")
            metadata: Event details
        """
        trace = self._get_or_create_trace(session_id)
        if not trace:
            return

        try:
            trace.event(
                name=name,
                metadata={
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            )
            logger.debug(f"Logged event '{name}' for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def update_trace_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update metadata on the session trace.

        Args:
            session_id: Session identifier
            metadata: Metadata to add/update
        """
        trace = self._traces.get(session_id)
        if not trace:
            return

        try:
            trace.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to update trace metadata: {e}")

    def end_trace(self, session_id: str) -> None:
        """
        End a trace for a session.

        Call this when a session is complete (e.g., terminal state reached).
        """
        trace = self._traces.pop(session_id, None)
        if trace:
            try:
                trace.update(
                    metadata={"ended_at": datetime.utcnow().isoformat()}
                )
                logger.debug(f"Ended trace for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to end trace: {e}")

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self.client:
            try:
                self.client.flush()
                logger.debug("Flushed traces to Langfuse")
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")

    def shutdown(self) -> None:
        """Graceful shutdown with flush."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
                logger.info("Langfuse client shut down")
            except Exception as e:
                logger.error(f"Failed to shutdown Langfuse client: {e}")

    def _sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data for logging (remove sensitive info)."""
        sanitized = dict(data)

        # Remove API keys if present
        for key in ["api_key", "authorization", "auth"]:
            sanitized.pop(key, None)

        # Truncate very long messages
        if "messages" in sanitized:
            messages = sanitized["messages"]
            for msg in messages:
                if isinstance(msg.get("content"), str) and len(msg["content"]) > 10000:
                    msg["content"] = msg["content"][:10000] + "... [truncated]"

        return sanitized

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize output data for logging."""
        if isinstance(output, dict):
            # Handle OpenAI-style response
            if "choices" in output:
                return {
                    "choices": output["choices"],
                    "model": output.get("model"),
                    "usage": output.get("usage"),
                }
            return output

        if hasattr(output, "model_dump"):
            return output.model_dump()

        return str(output)[:5000]  # Truncate if string
