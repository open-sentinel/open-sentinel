# Panoptes SDK

> Reliability layer for AI agents - monitors workflow adherence and intervenes when agents deviate.

## Overview

Panoptes is a **transparent proxy** that sits between your application and LLM providers. It intercepts all LLM calls to:

- **Monitor** - Classify LLM responses to determine workflow state
- **Enforce** - Evaluate temporal constraints (LTL-lite) against execution history
- **Intervene** - Inject correction prompts when deviations are detected
- **Observe** - Full tracing via Langfuse for debugging and analysis

```
┌─────────────────┐      ┌──────────────────────────────────────────────┐      ┌─────────────────┐
│                 │      │                  PANOPTES                    │      │                 │
│  Your App       │─────▶│  ┌──────────┐  ┌──────────┐  ┌───────────┐   │─────▶│  LLM Provider   │
│  (LLM Client)   │      │  │  Hooks   │─▶│ Tracker  │─▶│ Injector  │   │      │  (OpenAI, etc.) │
│                 │◀─────│  └──────────┘  └──────────┘  └───────────┘   │◀─────│                 │
└─────────────────┘      │       │             │              │         │      └─────────────────┘
                         │       ▼             ▼              ▼         │
                         │  ┌──────────────────────────────────────┐    │
                         │  │           Langfuse Tracing           │    │
                         │  └──────────────────────────────────────┘    │
                         └──────────────────────────────────────────────┘
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

```bash
export PANOPTES_WORKFLOW_PATH=./workflow.yaml
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
Simplified temporal logic for practical workflow constraints:

| Type | Meaning | Example |
|------|---------|---------|
| `eventually` | Must eventually reach target | "Must reach resolution" |
| `always` | Condition always holds | "Always maintain session" |
| `never` | Target must never occur | "Never share credentials" |
| `precedence` | B cannot occur before A | "Verify identity before account actions" |
| `response` | If A occurs, B must follow | "If escalate, must resolve" |

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
| `PANOPTES_WORKFLOW_PATH` | Path to workflow YAML | - |
| `PANOPTES_PROXY__PORT` | Server port | 4000 |
| `PANOPTES_LANGFUSE__PUBLIC_KEY` | Langfuse public key | - |
| `PANOPTES_LANGFUSE__SECRET_KEY` | Langfuse secret key | - |
| `PANOPTES_CLASSIFIER__MODEL_NAME` | Embedding model | all-MiniLM-L6-v2 |

### Langfuse Setup (Optional)

To enable tracing, get API keys from [cloud.langfuse.com](https://cloud.langfuse.com) and set:

```bash
export PANOPTES_LANGFUSE__PUBLIC_KEY=pk-lf-...
export PANOPTES_LANGFUSE__SECRET_KEY=sk-lf-...
```

For self-hosted Langfuse, also set `PANOPTES_LANGFUSE__HOST`. Traces will appear automatically in your Langfuse dashboard grouped by session.

## Documentation

- [Architecture Guide](ARCHITECTURE.md) - Detailed system design
- [Developer Guide](DEVELOPER.md) - Contributing and development setup

## License

MIT
