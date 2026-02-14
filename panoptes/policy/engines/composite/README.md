# Composite Policy Engine

> Combines multiple policy engines with configurable merge strategies for defense-in-depth.

## Overview

The Composite engine is Panoptes' **orchestration** policy engine. It doesn't evaluate policies itself — instead, it runs multiple child engines in parallel (or sequentially), collects their results, and merges them using a **most-restrictive-wins** strategy. This enables layered policy enforcement where different engines handle different concerns.

**Key characteristics:**
- 🔗 **Multi-engine** — Combine any registered engines (FSM + NeMo, LLM + NeMo, etc.)
- ⚡ **Parallel execution** — Engines run concurrently by default via `asyncio.gather`
- 🏆 **Most restrictive wins** — `DENY > MODIFY > WARN > ALLOW`
- 🛡️ **Fault tolerant** — Engine failures are captured as warnings, not propagated
- 🧩 **Recursive** — Composite engines can contain other composite engines

```
                    CompositePolicyEngine
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
      FSM Engine    NeMo Engine    LLM Engine
           │             │              │
           ▼             ▼              ▼
      ALLOW + 0v    DENY + 1v     WARN + 2v
           │             │              │
           └─────────────┼──────────────┘
                         ▼
                   Merged Result:
                   DENY + 3 violations
                   (most restrictive wins)
```

## Architecture

### Module Map

```
composite/
├── __init__.py    # Package exports
└── engine.py      # CompositePolicyEngine — single-file implementation
```

### Decision Priority

Results are merged using a strict priority hierarchy:

| Decision | Priority | Wins Over |
|----------|----------|-----------|
| `DENY` | 4 (highest) | Everything |
| `MODIFY` | 3 | WARN, ALLOW |
| `WARN` | 2 | ALLOW |
| `ALLOW` | 1 (lowest) | Nothing |

---

## Configuration

### Initialization

```python
from panoptes.policy.engines.composite import CompositePolicyEngine

engine = CompositePolicyEngine()
await engine.initialize({
    "engines": [
        {
            "type": "fsm",
            "config": {"config_path": "./workflow.yaml"}
        },
        {
            "type": "nemo",
            "config": {"config_path": "./nemo_config/"}
        }
    ],
    "strategy": "all",       # Run all engines (default)
    "parallel": True,        # Run concurrently (default)
})
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engines` | `list[dict]` | — | **Required.** List of engine configurations |
| `engines[].type` | `str` | — | Engine type (must be registered: `"fsm"`, `"nemo"`, `"llm"`, `"composite"`) |
| `engines[].config` | `dict` | `{}` | Engine-specific configuration (passed to `engine.initialize()`) |
| `strategy` | `str` | `"all"` | Merge strategy: `"all"` or `"first_deny"` |
| `parallel` | `bool` | `True` | Run engines concurrently or sequentially |

### Strategies

| Strategy | Behavior |
|----------|----------|
| `all` | Run all engines, merge all results |
| `first_deny` | Stop processing after first `DENY` result (optimization) |

---

## Merge Behavior

When results from multiple engines are collected, the Composite engine merges them as follows:

### Decision
**Most restrictive wins.** If any engine returns `DENY`, the final result is `DENY`, regardless of what other engines returned.

### Violations
**Collected from all engines.** Every violation from every engine is included in the final result.

### Intervention
**First one wins.** The first engine that requests an intervention has its intervention used. Subsequent interventions from other engines are ignored.

### Modified Request
**First one wins.** If multiple engines return `MODIFY` with different modified requests, only the first is used.

### Metadata
**Merged by engine name.** Each engine's metadata is stored under `metadata.engines[engine_name]`.

### Example Merge

```python
# Engine 1 (FSM): ALLOW, no violations
# Engine 2 (NeMo): DENY, 1 violation (jailbreak detected)
# Engine 3 (LLM): WARN, 2 violations (drift + constraint)

# Merged result:
PolicyEvaluationResult(
    decision=PolicyDecision.DENY,          # Most restrictive
    violations=[...],                       # All 3 violations
    intervention_needed=None,               # None requested
    metadata={
        "engines": {
            "fsm:customer-support": {...},
            "nemo:guardrails": {...},
            "llm:customer-support": {...},
        }
    }
)
```

---

## Evaluation Flow

### `evaluate_request(session_id, request_data, context)`

1. **Parallel mode**: Run all engines' `evaluate_request()` via `asyncio.gather`
   - Exceptions → converted to `WARN` violations (engine name included)
2. **Sequential mode**: Run engines one at a time
   - With `first_deny`, stop after first `DENY`
3. Merge and return combined result

### `evaluate_response(session_id, response_data, request_data, context)`

Same flow as request evaluation, but calling `evaluate_response()` on each child engine.

---

## Error Handling

Engine failures are **not propagated**. If a child engine throws an exception:

1. The exception is logged as an error
2. A `WARN` result is created with a violation named `{engine_type}_error`
3. Other engines continue running
4. The error violation is included in the merged result

```python
# If NeMo engine fails:
PolicyViolation(
    name="nemo_error",
    severity="warning",
    message="Connection timeout to NeMo service",
)
# This becomes a WARN, not a DENY — other engines still run
```

---

## Session Management

Session operations are delegated to **all child engines**:

### `get_session_state(session_id)`
Returns a dict with each engine's state:
```python
{
    "fsm:customer-support": {"current_state": "verify_identity", ...},
    "nemo:guardrails": {"session_context": {...}},
}
```

### `reset_session(session_id)`
Resets session in **all** child engines concurrently.

### `shutdown()`
Shuts down **all** child engines concurrently, then clears the engine list.

---

## Public API

### `CompositePolicyEngine` (extends `PolicyEngine`)

| Method | Description |
|--------|-------------|
| `initialize(config)` | Create and initialize all child engines |
| `evaluate_request(session_id, request_data, context)` | Run all engines, merge results |
| `evaluate_response(session_id, response_data, request_data, context)` | Run all engines, merge results |
| `get_session_state(session_id)` | Collect state from all engines |
| `reset_session(session_id)` | Reset all engines |
| `shutdown()` | Shutdown all engines |
| `get_engines()` | Get list of child engines (for debugging) |
| `get_engine_by_type(engine_type)` | Get a specific child engine by type |

### Key Properties

| Property | Value |
|----------|-------|
| `name` | `"composite:[engine1,engine2,...]"` (joined child names) |
| `engine_type` | `"composite"` |

---

## Common Patterns

### FSM + NeMo (Workflow + Safety)

```python
await engine.initialize({
    "engines": [
        {"type": "fsm", "config": {"config_path": "workflow.yaml"}},
        {"type": "nemo", "config": {"config_path": "nemo_config/"}},
    ]
})
# FSM enforces workflow state transitions and temporal constraints
# NeMo catches jailbreaks, PII, and unsafe content
```

### LLM + NeMo (Intelligence + Safety)

```python
await engine.initialize({
    "engines": [
        {"type": "llm", "config": {
            "config_path": "workflow.yaml",
            "llm_model": "gpt-4o-mini",
        }},
        {"type": "nemo", "config": {"config_path": "nemo_config/"}},
    ]
})
# LLM handles nuanced state classification and drift detection
# NeMo provides hard safety guardrails
```

### First-Deny Optimization

```python
await engine.initialize({
    "engines": [
        {"type": "nemo", "config": {"config_path": "nemo_config/"}},
        {"type": "fsm", "config": {"config_path": "workflow.yaml"}},
    ],
    "strategy": "first_deny",
    "parallel": False,  # Sequential for early exit
})
# NeMo runs first (fast safety check)
# FSM only runs if NeMo didn't deny
```

---

## Introspection

The Composite engine provides methods for runtime introspection:

```python
# List all child engines
engines = engine.get_engines()
for e in engines:
    print(f"{e.engine_type}: {e.name}")

# Get specific engine
nemo = engine.get_engine_by_type("nemo")
if nemo:
    state = await nemo.get_session_state("session-123")
```
