# Panoptes SDK

> Reliability layer for AI agents - monitors workflow adherence and intervenes when agents deviate.

## Overview

Panoptes is a **transparent proxy** that sits between your application and LLM providers. It intercepts all LLM calls to:

- **Monitor** - Classify LLM responses to determine workflow state
- **Enforce** - Evaluate temporal constraints (LTL-lite) against execution history
- **Intervene** - Inject correction prompts when deviations are detected
- **Observe** - Full tracing via OpenTelemetry for debugging and analysis

```
┌──────────────┐       ┌─────────────────────────────────────────┐       ┌──────────────┐
│              │       │              PANOPTES                   │       │              │
│   Your App   │──────▶│                                         │──────▶│ LLM Provider │
│              │       │  ┌───────┐  ┌────────┐  ┌──────────┐    │       │              │
│              │◀──────│  │ Hooks │─▶│ Policy │─▶│ Injector │    │◀──────│              │
└──────────────┘       │  └───────┘  └────────┘  └──────────┘    │       └──────────────┘
                       │       │          │            │         │
                       │       ▼          ▼            ▼         │
                       │  ┌─────────────────────────────────┐    │
                       │  │       OpenTelemetry Tracing     │    │
                       │  └─────────────────────────────────┘    │
                       └─────────────────────────────────────────┘
```

## Installation

```bash
pip install panoptes-sdk
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

1. **Define your workflow** in YAML:

```yaml
name: customer_support
version: "1.0"

states:
  - name: greeting
    is_initial: true
    classification:
      patterns: ["hello", "hi", "welcome"]
  
  - name: identify_issue
    classification:
      patterns: ["how can I help", "what.*issue"]
  
  - name: resolution
    is_terminal: true
    classification:
      patterns: ["resolved", "fixed", "solved"]

transitions:
  - from_state: greeting
    to_state: identify_issue
  - from_state: identify_issue
    to_state: resolution

constraints:
  - type: eventually
    target: resolution
    severity: warning
```

2. **Start the Panoptes proxy**:

For the FSM engine (workflow-based), specify the engine type explicitly:

```bash
export PANOPTES_POLICY__ENGINE__TYPE=fsm
export PANOPTES_WORKFLOW_PATH=./workflow.yaml
panoptes serve
```

For NeMo Guardrails (default):

```bash
export PANOPTES_POLICY__ENGINE__CONFIG__CONFIG_PATH=./nemo_config/
panoptes serve
```

3. **Point your LLM client at Panoptes**:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",  # Panoptes proxy
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Key Features

### Zero Code Changes
Customers only change `base_url` in their LLM client to point at Panoptes. No SDK integration required.

### Non-Blocking Monitoring
State classification and constraint evaluation run in parallel with LLM calls, adding **zero latency** to the critical path.

### LTL-Lite Constraints
Simplified temporal logic for practical workflow constraints (FSM engine):

| Type | Meaning | Example |
|------|---------|---------|
| `eventually` | Must eventually reach target | "Must reach resolution" |
| `always` | Condition always holds | "Always maintain session" |
| `never` | Target must never occur | "Never share credentials" |
| `precedence` | B cannot occur before A | "Verify identity before account actions" |
| `response` | If A occurs, B must follow | "If escalate, must resolve" |

### NeMo Guardrails Integration
Native support for NVIDIA NeMo Guardrails to enforce:
- **Input Rails**: Check for jailbreaks, PII, and toxicity before processing inputs.
- **Output Rails**: Validate LLM responses against safety policies and fact-checking.
- **Dialog Rails**: Control conversation flow using Colang scripts.

### Multiple Intervention Strategies

| Strategy | Use Case |
|----------|----------|
| `SYSTEM_PROMPT_APPEND` | Gentle guidance |
| `USER_MESSAGE_INJECT` | Important corrections |
| `CONTEXT_REMINDER` | Complex multi-step workflows |
| `HARD_BLOCK` | Critical violations |

## Configuration

Environment variables (prefix: `PANOPTES_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `PANOPTES_POLICY__ENGINE__TYPE` | Engine: `nemo`, `fsm`, `composite` | nemo |
| `PANOPTES_POLICY__ENGINE__CONFIG__CONFIG_PATH` | Path to NeMo config directory | - |
| `PANOPTES_WORKFLOW_PATH` | Path to workflow YAML (for FSM) | - |
| `PANOPTES_PROXY__PORT` | Server port | 4000 |
| `PANOPTES_OTEL__EXPORTER_TYPE` | Exporter: `otlp`, `langfuse`, `console`, `none` | otlp |
| `PANOPTES_OTEL__ENDPOINT` | OTLP endpoint (for `otlp` exporter) | http://localhost:4317 |
| `PANOPTES_OTEL__LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `PANOPTES_OTEL__LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `PANOPTES_OTEL__LANGFUSE_HOST` | Langfuse host (e.g. US region) | https://cloud.langfuse.com |
| `PANOPTES_CLASSIFIER__MODEL_NAME` | Embedding model | all-MiniLM-L6-v2 |

### OpenTelemetry Setup (Optional)

**Option 1: Export to Langfuse (via OTLP)**

Panoptes uses OpenTelemetry GenAI semantic conventions to provide rich traces in Langfuse, including model usage, costs, and policy evaluation events.

```bash
export PANOPTES_OTEL__EXPORTER_TYPE=langfuse
export PANOPTES_OTEL__LANGFUSE_PUBLIC_KEY=pk-lf-...
export PANOPTES_OTEL__LANGFUSE_SECRET_KEY=sk-lf-...
# For US region: export PANOPTES_OTEL__LANGFUSE_HOST=https://us.cloud.langfuse.com
```

**Option 2: Export to Standard OTLP Backend (Jaeger, Zipkin, etc.)**

```bash
export PANOPTES_OTEL__EXPORTER_TYPE=otlp
export PANOPTES_OTEL__ENDPOINT=http://localhost:4317
export PANOPTES_OTEL__SERVICE_NAME=panoptes

# Start Jaeger for local development
docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
```

Traces will appear in your tracing backend's UI grouped by session.

## Documentation

- [Architecture Guide](ARCHITECTURE.md) - Detailed system design
- [Developer Guide](DEVELOPER.md) - Contributing and development setup

## License

MIT
