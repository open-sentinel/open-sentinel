"""
OpenTelemetry-based tracer for Panoptes events.

Uses the OpenTelemetry SDK for vendor-agnostic distributed tracing.
Traces can be exported to any OTLP-compatible backend including:
- Jaeger, Zipkin, or other OTLP backends
- Langfuse (via their OTLP endpoint)
"""

import base64
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.trace import Status, StatusCode

from panoptes.config.settings import OTelConfig

logger = logging.getLogger(__name__)


class PanoptesTracer:
    """
    OpenTelemetry-based tracer for Panoptes workflow events.

    This tracer uses the OpenTelemetry SDK for distributed tracing,
    allowing traces to be exported to any OTLP-compatible backend
    including Langfuse.
    """

    def __init__(self, config: OTelConfig):
        self.config = config
        self._session_spans: Dict[str, trace.Span] = {}  # session_id -> root span
        self._enabled = config.enabled

        if not self._enabled or config.exporter_type == "none":
            self._enabled = False
            self._tracer = None
            logger.info("PanoptesTracer disabled")
            return

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: config.service_name})

        # Create and set tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter based on type
        if config.exporter_type == "console":
            exporter = ConsoleSpanExporter()
            logger.info("PanoptesTracer using console exporter")
        elif config.exporter_type == "langfuse":
            # Langfuse OTLP endpoint with HTTP and Basic Auth
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                logger.warning("PanoptesTracer disabled: missing Langfuse credentials")
                self._enabled = False
                self._tracer = None
                return
            
            # Build Langfuse OTLP endpoint
            langfuse_host = config.langfuse_host.rstrip("/")
            langfuse_endpoint = f"{langfuse_host}/api/public/otel/v1/traces"
            
            # Create Basic Auth header
            auth_str = f"{config.langfuse_public_key}:{config.langfuse_secret_key}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            headers = {"Authorization": f"Basic {auth_bytes}"}
            
            exporter = OTLPSpanExporterHTTP(
                endpoint=langfuse_endpoint,
                headers=headers,
            )
            logger.info(f"PanoptesTracer using Langfuse OTLP exporter (host={langfuse_host})")
        else:  # otlp (gRPC)
            exporter = OTLPSpanExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
            )
            logger.info(f"PanoptesTracer using OTLP gRPC exporter (endpoint={config.endpoint})")

        # Add batch processor for efficient export
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global provider
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("panoptes", "0.1.0")
        self._provider = provider

        logger.info(f"PanoptesTracer initialized (exporter={config.exporter_type})")

    def _get_or_create_session_span(self, session_id: str) -> trace.Span:
        """Get existing session span or create new one."""
        if session_id not in self._session_spans:
            if self._tracer:
                span = self._tracer.start_span(
                    "panoptes-session",
                    attributes={
                        "panoptes.session_id": session_id,
                        "panoptes.version": "0.1.0",
                    },
                )
                self._session_spans[session_id] = span
                logger.debug(f"Created session span for {session_id}")

        return self._session_spans.get(session_id)

    def log_event(
        self,
        session_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a Panoptes event as an OTEL span.

        Use for workflow deviations, interventions, state transitions.
        """
        if not self._enabled or not self._tracer:
            return

        parent_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        with self._tracer.start_as_current_span(
            name,
            context=parent_ctx,
            attributes={
                "panoptes.session_id": session_id,
                "panoptes.event_type": name,
            },
        ) as span:
            # Add input data as attributes
            if input_data:
                for key, value in input_data.items():
                    span.set_attribute(f"panoptes.input.{key}", str(value))

            # Add output data as attributes
            if output_data:
                for key, value in output_data.items():
                    span.set_attribute(f"panoptes.output.{key}", str(value))

            # Add metadata as attributes
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"panoptes.metadata.{key}", str(value))

            logger.info(f"Logged event '{name}' for session {session_id}")

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
        Log an LLM call as an OTEL span.
        """
        if not self._enabled or not self._tracer:
            return

        parent_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        with self._tracer.start_as_current_span(
            "llm-call",
            context=parent_ctx,
            attributes={
                "panoptes.session_id": session_id,
                "llm.model": response_model or model,
                "llm.requested_model": model,
                "llm.message_count": len(messages) if messages else 0,
            },
        ) as span:
            # Add response content (truncated for large responses)
            if response_content:
                truncated = response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                span.set_attribute("llm.response_preview", truncated)

            # Add usage info
            if usage:
                span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))
                span.set_attribute("llm.total_tokens", usage.get("total_tokens", 0))

            # Add latency
            if latency_ms:
                span.set_attribute("llm.latency_ms", latency_ms)

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"panoptes.metadata.{key}", str(value))

            logger.info(f"Logged LLM call for session {session_id} (model={model})")

    def end_trace(self, session_id: str) -> None:
        """Mark a session trace as complete."""
        if session_id in self._session_spans:
            span = self._session_spans.pop(session_id)
            span.set_status(Status(StatusCode.OK))
            span.end()
            logger.debug(f"Ended trace for session {session_id}")

    def flush(self) -> None:
        """Force flush any pending spans."""
        if hasattr(self, "_provider") and self._provider:
            self._provider.force_flush()

    def shutdown(self) -> None:
        """Clean up any remaining traces."""
        for session_id in list(self._session_spans.keys()):
            self.end_trace(session_id)

        if hasattr(self, "_provider") and self._provider:
            self._provider.shutdown()

        logger.info("PanoptesTracer shut down")
