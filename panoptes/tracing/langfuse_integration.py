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
import os
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

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
        self._initialized = False
        self._traces: Dict[str, Any] = {}  # session_id -> trace observation
        self._generations: Dict[str, Any] = {}  # session_id -> current generation

    def _ensure_env_configured(self):
        """Configure Langfuse environment variables from config."""
        # Use direct assignment instead of setdefault to ensure config values take precedence
        if self.config.public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.public_key
        if self.config.secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = self.config.secret_key
        if self.config.host:
            os.environ["LANGFUSE_HOST"] = self.config.host

        logger.debug(
            f"Langfuse env configured: host={self.config.host}, "
            f"public_key={'set' if self.config.public_key else 'missing'}, "
            f"secret_key={'set' if self.config.secret_key else 'missing'}"
        )

    @property
    def client(self):
        """Lazy-load Langfuse client using new get_client() API."""
        if self._client is None:
            try:
                self._ensure_env_configured()
                from langfuse import get_client

                # Use temp variable first - only assign to self._client on success
                client = get_client()

                # Verify authentication
                if client.auth_check():
                    self._client = client  # Only cache on successful auth
                    self._initialized = True
                    logger.info(f"Langfuse client initialized and authenticated (host={self.config.host})")
                else:
                    logger.error("Langfuse authentication failed. Check credentials and host.")
                    return None  # Don't cache failed client

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
            # Create a root span for the session using start_as_current_observation
            trace = self.client.start_as_current_observation(
                as_type="span",
                name="panoptes-session",
                input={"session_id": session_id},
                metadata={
                    "panoptes_version": "0.1.0",
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            # Enter the context to start the observation
            trace_ctx = trace.__enter__()
            self._traces[session_id] = {"observation": trace_ctx, "context_manager": trace}
            logger.debug(f"Created new trace for session {session_id}")

        return self._traces[session_id]

    def start_generation(
        self,
        session_id: str,
        name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start a new generation (LLM call) observation.

        Args:
            session_id: Session identifier
            name: Name for the generation (e.g., "llm-completion")
            input_data: Input to the LLM (messages, etc.)
            metadata: Additional metadata (workflow state, model, etc.)
            model: Model name (e.g., "gpt-4o", "claude-3-opus")

        Returns:
            Generation ID if successful, None otherwise
        """
        trace = self._get_or_create_trace(session_id)
        if not trace:
            return None

        try:
            # Use start_as_current_observation with as_type="generation"
            generation_cm = self.client.start_as_current_observation(
                as_type="generation",
                name=name,
                input=self._sanitize_input(input_data),
                model=model,
                metadata={
                    "session_id": session_id,
                    "started_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            )
            # Enter the context to start the generation
            generation = generation_cm.__enter__()
            self._generations[session_id] = {"observation": generation, "context_manager": generation_cm}
            logger.debug(f"Started generation for session {session_id}")
            return getattr(generation, 'id', None)
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
        generation_data = self._generations.pop(session_id, None)
        if not generation_data:
            logger.debug(f"No active generation for session {session_id}")
            return

        try:
            observation = generation_data["observation"]
            context_manager = generation_data["context_manager"]

            # Update the observation with output and metadata before closing
            update_kwargs = {
                "output": self._sanitize_output(output),
                "metadata": {
                    "ended_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            }

            if usage:
                update_kwargs["usage"] = usage

            observation.update(**update_kwargs)

            # Exit the context manager to close the generation
            context_manager.__exit__(None, None, None)
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
            # Events are point-in-time spans that immediately close
            if self.client:
                with self.client.start_as_current_observation(
                    as_type="span",
                    name=name,
                    input={"event_type": name},
                    metadata={
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        **(metadata or {}),
                    },
                ) as event:
                    event.update(output={"logged": True})
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
        trace_data = self._traces.get(session_id)
        if not trace_data:
            return

        try:
            observation = trace_data["observation"]
            observation.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to update trace metadata: {e}")

    def end_trace(self, session_id: str) -> None:
        """
        End a trace for a session.

        Call this when a session is complete (e.g., terminal state reached).
        """
        trace_data = self._traces.pop(session_id, None)
        if trace_data:
            try:
                observation = trace_data["observation"]
                context_manager = trace_data["context_manager"]

                observation.update(
                    output={"status": "completed"},
                    metadata={"ended_at": datetime.utcnow().isoformat()}
                )
                # Exit the context manager to close the span
                context_manager.__exit__(None, None, None)
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

    @contextmanager
    def trace_generation(
        self,
        session_id: str,
        name: str,
        input_data: Dict[str, Any],
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing a generation (LLM call).

        Usage:
            with tracer.trace_generation(session_id, "llm-call", input_data, model="gpt-4o") as gen:
                # Your LLM call here
                result = call_llm(input_data)
                gen.update(output=result, usage={"prompt_tokens": 100, "completion_tokens": 50})

        Args:
            session_id: Session identifier
            name: Name for the generation
            input_data: Input to the LLM
            model: Model name
            metadata: Additional metadata
        """
        trace = self._get_or_create_trace(session_id)
        if not trace or not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_observation(
                as_type="generation",
                name=name,
                input=self._sanitize_input(input_data),
                model=model,
                metadata={
                    "session_id": session_id,
                    "started_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            ) as generation:
                yield generation
        except Exception as e:
            logger.error(f"Error in trace_generation: {e}")
            yield None

    @contextmanager
    def trace_span(
        self,
        session_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing a span (non-LLM operation).

        Usage:
            with tracer.trace_span(session_id, "process-data", {"data": data}) as span:
                # Your processing logic here
                result = process(data)
                span.update(output=result)

        Args:
            session_id: Session identifier
            name: Name for the span
            input_data: Input data
            metadata: Additional metadata
        """
        trace = self._get_or_create_trace(session_id)
        if not trace or not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata={
                    "session_id": session_id,
                    "started_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            ) as span:
                yield span
        except Exception as e:
            logger.error(f"Error in trace_span: {e}")
            yield None

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
