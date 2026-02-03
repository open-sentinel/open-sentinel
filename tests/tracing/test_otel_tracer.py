
import pytest
from unittest.mock import MagicMock, patch

from panoptes.tracing.otel_tracer import PanoptesTracer
from panoptes.config.settings import OTelConfig

@pytest.fixture
def mock_otel():
    with patch("panoptes.tracing.otel_tracer.trace") as mock_trace, \
         patch("panoptes.tracing.otel_tracer.TracerProvider") as mock_provider, \
         patch("panoptes.tracing.otel_tracer.OTLPSpanExporter") as mock_exporter, \
         patch("panoptes.tracing.otel_tracer.BatchSpanProcessor") as mock_processor, \
         patch("panoptes.tracing.otel_tracer.Resource") as mock_resource:
        
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        yield {
            "trace": mock_trace,
            "provider": mock_provider,
            "exporter": mock_exporter,
            "tracer": mock_tracer,
        }

def test_tracer_initialization(mock_otel):
    config = OTelConfig(
        enabled=True,
        service_name="test-service",
        endpoint="localhost:4317",
        exporter_type="otlp"
    )
    
    tracer = PanoptesTracer(config)
    
    assert tracer._enabled is True
    mock_otel["trace"].set_tracer_provider.assert_called_once()
    mock_otel["exporter"].assert_called_with(endpoint="localhost:4317", insecure=True)

def test_tracer_disabled(mock_otel):
    config = OTelConfig(enabled=False)
    
    tracer = PanoptesTracer(config)
    
    assert tracer._enabled is False
    mock_otel["trace"].set_tracer_provider.assert_not_called()

def test_log_event(mock_otel):
    config = OTelConfig(enabled=True)
    tracer = PanoptesTracer(config)
    
    # Mock span context manager
    mock_span = MagicMock()
    mock_otel["tracer"].start_as_current_span.return_value.__enter__.return_value = mock_span
    
    tracer.log_event(
        session_id="session-1",
        name="test_event",
        input_data={"k": "v"},
        output_data={"res": "ok"}
    )
    
    mock_otel["tracer"].start_as_current_span.assert_called()
    # Check attributes were set
    # panoptes.session_id, panoptes.event_type, panoptes.input.k, panoptes.output.res
    calls = mock_span.set_attribute.call_args_list
    assert any(call.args[0] == "panoptes.input.k" for call in calls)
    assert any(call.args[0] == "panoptes.output.res" for call in calls)

def test_log_llm_call(mock_otel):
    config = OTelConfig(enabled=True)
    tracer = PanoptesTracer(config)
    
    mock_span = MagicMock()
    mock_otel["tracer"].start_as_current_span.return_value.__enter__.return_value = mock_span
    
    tracer.log_llm_call(
        session_id="session-1",
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        response_content="hello",
        usage={"total_tokens": 100}
    )
    
    # Verify the span was created with correct name
    mock_otel["tracer"].start_as_current_span.assert_called()
    call_args = mock_otel["tracer"].start_as_current_span.call_args
    assert call_args[0][0] == "llm-call"  # First positional arg is the name
    
    # Check that required attributes are present (we added GenAI semantic conventions)
    attrs = call_args[1]["attributes"]
    assert attrs["panoptes.session_id"] == "session-1"
    assert attrs["llm.model"] == "gpt-4"
    assert attrs["llm.requested_model"] == "gpt-4"
    assert attrs["llm.message_count"] == 1
    # New GenAI semantic convention attributes
    assert attrs["gen_ai.request.model"] == "gpt-4"
    assert attrs["gen_ai.response.model"] == "gpt-4"
    
    # Verify span attributes were set for response/usage
    mock_span.set_attribute.assert_any_call("llm.total_tokens", 100)

def test_shutdown(mock_otel):
    config = OTelConfig(enabled=True)
    tracer = PanoptesTracer(config)
    
    tracer.shutdown()
    
    # We can't easily check internal provider shutdown call if we mocked class instantiation returning a mock
    # But checking if method calls proceed without error is good start.
    # Actually mock_provider is the class, mock_provider() is the instance.
    mock_otel["provider"].return_value.shutdown.assert_called_once()
