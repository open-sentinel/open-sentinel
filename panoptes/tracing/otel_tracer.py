"""
OpenTelemetry-based tracer for Panoptes events.

Uses the OpenTelemetry SDK for vendor-agnostic distributed tracing.
Traces can be exported to any OTLP-compatible backend including:
- Jaeger, Zipkin, or other OTLP backends
- Langfuse (via their OTLP endpoint)
"""

import base64
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import contextmanager


from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.trace import Status, StatusCode

from panoptes.config.settings import OTelConfig

logger = logging.getLogger(__name__)


class SpanEventManager(logging.Handler):
    """
    Logging handler that attaches log records as events to the current OTEL span.
    """
    def emit(self, record):
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                attributes = {
                    "log.level": record.levelname,
                    "log.logger": record.name,
                    "code.filepath": record.pathname,
                    "code.lineno": record.lineno,
                }
                current_span.add_event(record.getMessage(), attributes=attributes)
                
                # Debug print for verification
                # print(f"DEBUG: Captured log as event: {record.getMessage()}")  
        except Exception:
            self.handleError(record)


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

        # Attach span event manager to capture NeMo Guardrails logs
        span_handler = SpanEventManager()
        # Ensure we don't duplicate handlers
        nemo_logger = logging.getLogger("nemoguardrails")
        if not any(isinstance(h, SpanEventManager) for h in nemo_logger.handlers):
            nemo_logger.addHandler(span_handler)

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

    def _safe_json(self, obj: Any) -> str:
        """Safely serialize object to JSON string for span attributes."""
        try:
            return json.dumps(obj, default=str, ensure_ascii=False)
        except Exception:
            return str(obj)

    @contextmanager
    def trace_block(
        self,
        name: str,
        session_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager to trace a block of code.
        Useful for capturing logs emitted during execution as span events.
        
        Args:
            name: Span name
            session_id: Session identifier
            attributes: Additional span attributes
            input_data: Input data to record (will be JSON serialized)
            output_data: Output data to record (will be JSON serialized)
            metadata: Additional metadata for the span
        """
        if not self._enabled or not self._tracer:
            yield None
            return

        parent_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        span_attrs = {
            "panoptes.session_id": session_id,
            **(attributes or {}),
        }

        with self._tracer.start_as_current_span(
            name,
            context=parent_ctx,
            attributes=span_attrs,
        ) as span:
            # Set input data using Langfuse-compatible attributes
            if input_data is not None:
                input_json = self._safe_json(input_data)
                span.set_attribute("input.value", input_json)
                span.set_attribute("langfuse.span.input", input_json)
            
            # Set metadata if provided
            if metadata:
                span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
                for key, value in metadata.items():
                    span.set_attribute(f"panoptes.metadata.{key}", str(value))
            
            yield span
            
            # Set output data after the block executes (allows for dynamic output)
            if output_data is not None:
                output_json = self._safe_json(output_data)
                span.set_attribute("output.value", output_json)
                span.set_attribute("langfuse.span.output", output_json)

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

    def log_policy_evaluation(
        self,
        session_id: str,
        engine_name: str,
        decision: str,
        hook_type: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        violations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a policy evaluation as an OTEL span with rich attributes.
        
        Args:
            session_id: Session identifier
            engine_name: Name of the policy engine (e.g., "nemo:guardrails")
            decision: Policy decision (ALLOW, DENY, MODIFY, WARN)
            hook_type: Which hook triggered this (pre_call, post_call, moderation)
            input_data: Request/messages being evaluated
            output_data: Policy evaluation result
            violations: List of violation dicts with name, severity, message
            metadata: Additional metadata
        """
        if not self._enabled or not self._tracer:
            return

        parent_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        span_name = f"policy_evaluation_{hook_type}" if hook_type else "policy_evaluation"
        
        span_attrs = {
            "panoptes.session_id": session_id,
            "panoptes.policy.engine": engine_name,
            "panoptes.policy.decision": decision,
            "panoptes.policy.hook": hook_type,
        }

        with self._tracer.start_as_current_span(
            span_name,
            context=parent_ctx,
            attributes=span_attrs,
        ) as span:
            # Set input data (what was evaluated)
            if input_data is not None:
                input_json = self._safe_json(input_data)
                span.set_attribute("input.value", input_json)
                span.set_attribute("langfuse.span.input", input_json)
            
            # Set output data (evaluation result)
            if output_data is not None:
                output_json = self._safe_json(output_data)
                span.set_attribute("output.value", output_json)
                span.set_attribute("langfuse.span.output", output_json)
            
            # Add violations as structured data
            if violations:
                span.set_attribute("panoptes.policy.violation_count", len(violations))
                violation_names = [v.get("name", "unknown") for v in violations]
                span.set_attribute("panoptes.policy.violations", self._safe_json(violation_names))
                
                # Add each violation as an event
                for violation in violations:
                    span.add_event(
                        f"violation:{violation.get('name', 'unknown')}",
                        attributes={
                            "severity": violation.get("severity", "unknown"),
                            "message": violation.get("message", ""),
                        }
                    )
            
            # Add metadata
            if metadata:
                span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
                for key, value in metadata.items():
                    span.set_attribute(f"panoptes.metadata.{key}", str(value))

            logger.debug(f"Logged policy evaluation for session {session_id} (decision={decision})")

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
        parent_span: Optional[Any] = None,
    ) -> None:
        """
        Log an LLM call as an OTEL span with GenAI semantic conventions.
        
        Uses OpenTelemetry GenAI semantic conventions for Langfuse compatibility:
        - gen_ai.* attributes for model/usage info
        - input.value / output.value for content
        - langfuse.span.* for explicit Langfuse mapping
        """
        if not self._enabled or not self._tracer:
            return

        # Use provided parent or get session span
        if parent_span is None:
            parent_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None

        # Build span attributes with GenAI semantic conventions
        span_attrs = {
            "panoptes.session_id": session_id,
            # GenAI semantic conventions
            "gen_ai.system": "openai",  # or derive from model
            "gen_ai.request.model": model,
            "gen_ai.response.model": response_model or model,
            # Legacy attributes for backward compatibility
            "llm.model": response_model or model,
            "llm.requested_model": model,
            "llm.message_count": len(messages) if messages else 0,
        }

        with self._tracer.start_as_current_span(
            "llm-call",
            context=parent_ctx,
            attributes=span_attrs,
        ) as span:
            # Set input (messages) using multiple attribute formats for compatibility
            if messages:
                messages_json = self._safe_json(messages)
                span.set_attribute("input.value", messages_json)
                span.set_attribute("langfuse.span.input", messages_json)
                span.set_attribute("gen_ai.content.prompt", messages_json)
            
            # Set output (response) using multiple attribute formats
            if response_content:
                span.set_attribute("output.value", response_content)
                span.set_attribute("langfuse.span.output", response_content)
                span.set_attribute("gen_ai.content.completion", response_content)
                # Also keep truncated preview for quick viewing
                truncated = response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                span.set_attribute("llm.response_preview", truncated)

            # Add usage info with GenAI semantic conventions
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # GenAI semantic conventions
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                # Legacy attributes
                span.set_attribute("llm.prompt_tokens", prompt_tokens)
                span.set_attribute("llm.completion_tokens", completion_tokens)
                span.set_attribute("llm.total_tokens", total_tokens)

            # Add latency
            if latency_ms:
                span.set_attribute("llm.latency_ms", latency_ms)

            # Add metadata
            if metadata:
                span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
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
