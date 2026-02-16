# Panoptes

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
│              │       │  ┌───────┐  ┌─────────────┐  ┌───────┐  │       │              │
│              │◀──────│  │ Hooks │─▶│ Interceptor │─▶│Checker│  │◀──────│              │
└──────────────┘       │  └───────┘  └─────────────┘  └───────┘  │       └──────────────┘
                       │       │            │             │      │
                       │       ▼            ▼             ▼      │
                       │  ┌─────────────────────────────────┐    │
                       │  │       OpenTelemetry Tracing     │    │
                       │  └─────────────────────────────────┘    │
                       └─────────────────────────────────────────┘
```

## Installation

```bash
pip install panoptes
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize Configuration

Generates `panoptes.yaml` and `policy.yaml` with sensible defaults.

```bash
panoptes init
# or non-interactive:
# panoptes init --non-interactive
```

### 2. Configure Your Policy

Edit `policy.yaml` to define your guardrails. For the Judge engine (default), this means defining rubrics:

```yaml
rubrics:
  - name: safety_policy
    criteria:
      - name: no_pii
        description: "Response must not contain PII"
        scale: binary
        fail_threshold: 0.5
```

### 3. Start the Proxy

```bash
export OPENAI_API_KEY=sk-...
panoptes serve
```

### 4. Point Your Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",  # Panoptes proxy
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Policy Engines

Panoptes supports **pluggable policy engines**:

| Engine | Type | Best For |
|--------|------|----------|
| **Judge**| `judge` | **Recommended**. Soft constraint evaluation using LLM-as-judge reasoning |
| **FSM** | `fsm` | Deterministic workflow enforcement via state machines |
| **NeMo** | `nemo` | Content safety, jailbreak detection (NVIDIA NeMo Guardrails) |
| **Composite** | `composite` | Running multiple engines in parallel |

## Key Features

### Zero Code Changes
Customers only change `base_url` in their LLM client to point at Panoptes. No SDK integration required.

### Fail-Open Hardening
All hooks are wrapped with timeout and exception handling. Issues in Panoptes won't block your application unless explicitly configured to.

## Configuration

The primary configuration is `panoptes.yaml`.

```yaml
engine: judge
policy: ./policy.yaml
port: 4000

judge:
  model: gpt-4o-mini
  mode: balanced  # safe, balanced, aggressive

tracing:
  type: none      # none, console, otlp, langfuse
```

### Environment Variables

Environment variables can override `panoptes.yaml` settings (prefix `PANOPTES_`):

- `PANOPTES_POLICY__ENGINE__TYPE` -> `engine`
- `PANOPTES_POLICY__ENGINE__CONFIG_PATH` -> `policy`
- `PANOPTES_PROXY__PORT` -> `port`
- `PANOPTES_OTEL__EXPORTER_TYPE` -> `tracing.type`

## CLI Commands

| Command | Description |
|---------|-------------|
| `panoptes init` | Initialize configuration files |
| `panoptes serve` | Start the proxy server |
| `panoptes compile` | Compile natural language to policy |
| `panoptes validate` | Validate policy files |

## Documentation

- [Architecture Guide](ARCHITECTURE.md)
- [Developer Guide](DEVELOPER.md)
- [Examples](examples/)

## License

MIT
