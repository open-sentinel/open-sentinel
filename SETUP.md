# Open Sentinel Setup Guide

## Quick Start (3 commands)

```bash
osentinel init --non-interactive
export OPENAI_API_KEY=sk-...
osentinel serve
```

This creates `osentinel.yaml` and `policy.yaml` with sensible defaults (judge engine, balanced mode, gpt-4o-mini). Point your LLM client to `http://localhost:4000/v1`.

## Interactive Setup

```bash
osentinel init
```

You'll be prompted for:
- **Engine type**: `judge` (LLM-as-a-judge evaluation) or `fsm` (finite state machine workflow enforcement)
- **Judge model**: Which LLM evaluates responses (default: `gpt-4o-mini`)
- **Reliability mode**: `safe`, `balanced`, or `aggressive`
- **Tracing**: `none`, `console`, `otlp`, or `langfuse`

## Manual Setup

If you prefer to configure manually instead of using `osentinel init`:

1. Create `osentinel.yaml` in your project root:

```yaml
engine: judge
policy: ./policy.yaml
port: 4000

judge:
  model: gpt-4o-mini
  mode: balanced

tracing:
  type: none
```

2. Create `policy.yaml` with your rubric:

```yaml
rubrics:
  - name: my_policy
    description: "Custom evaluation policy"
    scope: turn
    evaluation_type: pointwise
    pass_threshold: 0.6
    fail_action: warn
    criteria:
      - name: professional_tone
        description: "Is the response professional and appropriate?"
        scale: likert_5
        weight: 1.0
      - name: no_pii
        description: "Does the response avoid sharing PII?"
        scale: binary
        weight: 1.0
        fail_threshold: 0.5
```

3. Start the proxy:

```bash
export OPENAI_API_KEY=sk-...
osentinel serve
```

## Compile Policies from Natural Language

Instead of writing rubric YAML by hand, compile from a description:

```bash
# Auto-detects engine type
osentinel compile "be professional, never leak PII, always cite sources"

# Explicit judge engine
osentinel compile "never share internal info" --engine judge -o policy.yaml

# FSM workflow
osentinel compile "verify identity before processing refunds" --engine fsm
```

## Config Reference

### osentinel.yaml keys

- `engine` - Policy engine type: `judge`, `fsm`, `nemo`, `composite`
- `policy` - Path to policy file (rubric YAML for judge, workflow YAML for fsm)
- `port` - Proxy server port (default: 4000)
- `host` - Proxy server host (default: 0.0.0.0)
- `debug` - Enable debug logging (default: false)
- `judge.model` - LLM model for judge evaluation (default: gpt-4o-mini)
- `judge.mode` - Reliability mode: `safe`, `balanced`, `aggressive`
- `tracing.type` - Tracing exporter: `none`, `console`, `otlp`, `langfuse`

### Environment variables

All settings can also be set via environment variables with `OSNTL_` prefix. `osentinel.yaml` takes priority over env vars.

- `OSNTL_POLICY__ENGINE__TYPE` - Engine type
- `OSNTL_POLICY__ENGINE__CONFIG_PATH` - Policy file path
- `OSNTL_PROXY__PORT` - Proxy port
- `OSNTL_OTEL__EXPORTER_TYPE` - Tracing exporter
- `OSNTL_CONFIG` - Path to osentinel.yaml (if not in current directory)

Only API keys need to be env vars:
- `OPENAI_API_KEY` - Required for OpenAI models
- `GOOGLE_API_KEY` / `GEMINI_API_KEY` - Required for Gemini models

## Engine Types

### Judge (recommended for most use cases)

Uses an LLM to evaluate every response against a rubric. Good for:
- Content quality enforcement (tone, accuracy, helpfulness)
- Safety screening (PII, harmful content)
- Policy compliance (brand guidelines, regulatory requirements)

### FSM (for workflow enforcement)

Tracks conversation state through a finite state machine. Good for:
- Multi-step workflows (onboarding, support tickets)
- Ordering constraints (verify identity before refund)
- State-dependent interventions

### Composite

Combines multiple engines. For example, FSM for workflow + judge for quality.

## Reliability Modes (Judge Engine)

- **safe**: Stricter thresholds, pre-call safety screening enabled, ensemble evaluation when multiple models configured. Best for high-stakes applications.
- **balanced** (default): Moderate thresholds, post-call evaluation only. Good balance of safety and latency.
- **aggressive**: Looser thresholds, fewer interventions. Best for low-risk applications where you want minimal overhead.

## For LLM Agents Setting Up Open Sentinel

Copy-paste these commands to set up Open Sentinel programmatically:

```bash
# Install
pip install open-sentinel

# Initialize with defaults (no prompts)
osentinel init --non-interactive

# Set your API key
export OPENAI_API_KEY=sk-...

# Start the proxy
osentinel serve
```

To compile a custom policy from a description:

```bash
osentinel compile "your policy description here" -o policy.yaml
```

To use a custom config file path:

```bash
osentinel serve --config /path/to/osentinel.yaml
```

The proxy listens on `http://localhost:4000/v1`. Set your LLM client's `base_url` to this address.
