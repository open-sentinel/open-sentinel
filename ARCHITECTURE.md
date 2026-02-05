# Panoptes Architecture

> Reliability layer for AI agents - monitors workflow adherence and intervenes when agents deviate.

## Overview

Panoptes is a **transparent proxy** that sits between your application and LLM providers. It intercepts all LLM calls to:

1. **Monitor** - Classify LLM responses to determine workflow state
2. **Enforce** - Evaluate temporal constraints (LTL-lite) against execution history
3. **Intervene** - Inject correction prompts when deviations are detected
4. **Observe** - Full tracing via OpenTelemetry for debugging and analysis

```
┌─────────────────┐      ┌──────────────────────────────────────────────┐      ┌─────────────────┐
│                 │      │                  PANOPTES                    │      │                 │
│  Your App       │─────▶│  ┌──────────┐  ┌──────────┐  ┌───────────┐   │─────▶│  LLM Provider   │
│  (LLM Client)   │      │  │  Hooks   │─▶│ Tracker  │─▶│ Injector  │   │      │  (OpenAI, etc.) │
│                 │◀─────│  └──────────┘  └──────────┘  └───────────┘   │◀─────│                 │
└─────────────────┘      │       │             │              │         │      └─────────────────┘
                         │       ▼             ▼              ▼         │
                         │  ┌──────────────────────────────────────┐    │
                         │  │         OpenTelemetry Tracing        │    │
                         │  └──────────────────────────────────────┘    │
                         └──────────────────────────────────────────────┘
                         └──────────────────────────────────────────────┘
```

## Design Principles

### 1. Zero Code Changes for Customers
Customers only change `base_url` in their LLM client to point at Panoptes. No SDK integration, no code changes beyond configuration.

### 2. Non-Blocking Monitoring
State classification and constraint evaluation run in parallel with LLM calls (via `async_moderation_hook`), adding **zero latency** to the critical path.

### 3. Deferred Intervention
When a deviation is detected, the intervention is **scheduled for the next call**, not applied immediately. This preserves the current response while ensuring correction.

### 4. LTL-Lite Constraints
Simplified temporal logic for practical workflow constraints without full model-checking complexity. Runtime verification instead of static analysis.

---

## Core Components

### 1. Proxy Layer (`panoptes/proxy/`)

The proxy wraps LiteLLM to intercept all LLM traffic.

#### `server.py` - PanoptesProxy
Main entry point. Wraps LiteLLM Router with Panoptes callbacks.

```python
# Key class
class PanoptesProxy:
    def __init__(self, settings: PanoptesSettings)
    async def start(self)           # Start uvicorn server
    async def completion(self, **kwargs)  # Programmatic access
```

**Important**: The proxy uses LiteLLM's Router for model routing and load balancing. Configuration is in `PanoptesSettings.proxy.model_list`.

#### `hooks.py` - PanoptesCallback
Implements LiteLLM's `CustomLogger` interface with four key hooks:

| Hook | Timing | Purpose | Blocking? |
|------|--------|---------|-----------|
| `async_pre_call_hook` | Before LLM call | Apply pending interventions, run PRE_CALL checkers, start trace | Yes |
| `async_moderation_hook` | Parallel with LLM | **(Deprecated/Unused)** Logic moved to Interceptor | No |
| `async_post_call_success_hook` | After LLM response | Run POST_CALL checkers, complete trace | Yes |
| `async_post_call_failure_hook` | After LLM error | Log failure | Yes |

**Session State Management**: The callback maintains:
- `_sessions`: Dict mapping session_id → session state
- `_pending_interventions`: Dict mapping session_id → intervention to apply next call

#### `middleware.py` - Context Extraction
Extracts session IDs and workflow context from requests.

**Session ID Extraction Priority**:
1. Header: `x-panoptes-session-id` or `x-session-id`
2. Metadata: `metadata.session_id` or `metadata.panoptes_session_id`
3. Metadata: `metadata.run_id` (LangChain)
4. Field: `user` (OpenAI pattern)
5. Field: `thread_id` (OpenAI Assistants)
6. Hash of first user message content
7. Random UUID (fallback)

---

### 2. Policy Layer (`panoptes/policy/`)

Panoptes supports pluggable policy engines. The default distribution includes:

#### FSM Engine (`panoptes/policy/engines/fsm/`)

Wraps the workflow state machine, monitor, and injector for workflow-based enforcement.

**Components**:
- **Workflow**: Schema, Parser, StateMachine, Constraints (formerly top-level `workflow` module)
- **Monitor**: StateClassifier, WorkflowTracker (formerly top-level `monitor` module)
- **Injector**: PromptInjector for FSM-based interventions

#### `schema.py` - Workflow Data Models
Located in `panoptes/policy/engines/fsm/workflow/schema.py`. Pydantic models for workflow definitions:

```python
WorkflowDefinition
├── name: str
├── version: str
├── states: List[State]
│   ├── name: str
│   ├── is_initial: bool
│   ├── is_terminal: bool
│   └── classification: ClassificationHint
│       ├── tool_calls: List[str]    # Exact match (highest priority)
│       ├── patterns: List[str]      # Regex patterns
│       └── exemplars: List[str]     # Semantic similarity examples
├── transitions: List[Transition]
│   ├── from_state: str
│   ├── to_state: str
│   └── guard: Optional[TransitionGuard]
├── constraints: List[Constraint]
│   ├── type: ConstraintType (eventually|always|never|precedence|response|...)
│   ├── trigger: Optional[str]
│   ├── target: Optional[str]
│   ├── severity: Literal["warning", "error", "critical"]
│   └── intervention: Optional[str]
└── interventions: Dict[str, str]  # name → prompt template
```

#### NeMo Guardrails Engine (`panoptes/policy/engines/nemo/`)

Integrates NVIDIA's NeMo Guardrails for comprehensive content safety and dialog management.

**Key Features**:
- **Input Rails**: Pre-processing checks for jailbreaks, PII, and toxicity.
- **Output Rails**: Post-processing verification against safety policies and hallucination checks.
- **Dialog Rails**: Programmable conversation flow using Colang.

**Panoptes Bridge**:
- Registers custom actions (`panoptes_log_violation`, `panoptes_request_intervention`) to allow Colang scripts to interact with Panoptes' tracing and intervention systems.
- Adapts NeMo's `generate_async` output to `PolicyEvaluationResult`, translating blocked responses into policy violations.

---

### 3. Policy Compiler (`panoptes/policy/compiler/`)

Converts natural language policy descriptions into engine-specific configurations.

**Architecture**:
- **PolicyCompiler Protocol**: Abstract base class defining the compiler interface
- **LLMPolicyCompiler**: Base class with shared LLM interaction utilities
- **FSMCompiler**: Converts natural language → WorkflowDefinition YAML

```python
# Example usage
from panoptes.policy.compiler import PolicyCompilerRegistry

compiler = PolicyCompilerRegistry.create("fsm")
result = await compiler.compile(
    "Agent must verify identity before processing refunds. "
    "Never share internal system information."
)

if result.success:
    compiler.export(result, Path("workflow.yaml"))
```

**CompilationResult**:
- `success`: Whether compilation succeeded
- `config`: Engine-specific config (WorkflowDefinition for FSM)
- `warnings`: Non-fatal issues
- `errors`: Fatal issues that prevented compilation
- `metadata`: Token usage, state/constraint counts

**Extension Point**: To add a compiler for a new engine (e.g., NeMo Colang):
1. Create a class extending `LLMPolicyCompiler`
2. Implement `_build_compilation_prompt()` for engine-specific prompting
3. Implement `_parse_compilation_response()` for response parsing
4. Register with `@register_compiler("engine_name")`

---

### 4. Core Layer (`panoptes/core/`)

Shared components used across different policy engines.

#### `intervention` - Strategies
Located in `panoptes/core/intervention/`. Defines the base strategies for modifying LLM requests.

| Strategy | How It Works | Use Case |
|----------|--------------|----------|
| `SYSTEM_PROMPT_APPEND` | Appends `[WORKFLOW GUIDANCE]` to system message | Gentle guidance, least disruptive |
| `USER_MESSAGE_INJECT` | Inserts `[System Note]` as user message | Important corrections |
| `CONTEXT_REMINDER` | Inserts assistant message `[Context reminder]` | Complex multi-step workflows |
| `HARD_BLOCK` | Raises `WorkflowViolationError` | Critical violations, block request |

---

### 5. Intervention Execution

Once a violation is detected by the FSM policy engine, it triggers an intervention.

#### PromptInjector
Located in `panoptes/policy/engines/fsm/injector.py`. Applies interventions to request data using the strategies defined in the Core Layer.

```python
class PromptInjector:
    def inject(data, intervention_name, context, session_id) -> modified_data
```

**Escalation**: Tracks application counts per session. If an intervention is applied more than `max_applications` times, escalates to a stricter strategy (e.g. from `SYSTEM_PROMPT_APPEND` to `HARD_BLOCK`).

---

### 6. Tracing Layer (`panoptes/tracing/`)

#### `otel_tracer.py` - PanoptesTracer
Provides session-aware tracing via OpenTelemetry with special support for **Langfuse**.

**Key Components**:
- **PanoptesTracer**: Main class for creating spans and events.
- **SpanEventManager**: Logging handler that captures application logs as events on the active span (crucial for debugging NeMo internals).
- **GenAI Semantics**: Uses OpenTelemetry GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.prompt_tokens`) for rich observability.

```python
class PanoptesTracer:
    def log_policy_evaluation(session_id, decision, violations, ...) # Tracks policy results
    def log_llm_call(session_id, model, messages, response, ...)     # Tracks LLM usage/cost
    
    @contextmanager
    def trace_block(name, session_id): ...  # Scopes logs to specific spans
```

**Trace Hierarchy**:
- **Session Span**: Root span per session
- **Policy Evaluation Spans**: Captures inputs, decisions, and any violations
- **LLM Call Spans**: Detailed view of the raw LLM interaction
- **Log Events**: Granular logs from the application (e.g., NeMo's "activated rail") attached to relevant spans

---

### 7. Configuration (`panoptes/config/`)

#### `settings.py` - PanoptesSettings
Pydantic Settings with env var support:

```python
class PanoptesSettings(BaseSettings):
    # Prefix: PANOPTES_
    # Nested delimiter: __ (e.g., PANOPTES_OTEL__ENDPOINT)
    
    debug: bool
    # workflow_path - REMOVED (use policy.engine configuration)
    
    otel: OTelConfig
    proxy: ProxyConfig
    classifier: ClassifierConfig
    intervention: InterventionConfig
```

**Key Configuration Options**:
- `PANOPTES_POLICY__ENGINE__CONFIG_PATH`: Path to workflow YAML or NeMo config directory
- `PANOPTES_PROXY__PORT`: Server port (default 4000)
- `PANOPTES_OTEL__ENDPOINT`: OpenTelemetry OTLP endpoint

---

## Data Flow

### Normal Flow (No Violation)

```
1. Client sends LLM request
   │
2. async_pre_call_hook
   ├─ Extract session ID
   ├─ Check for pending intervention (none)
   └─ Start OTEL trace span
   │
3. LLM call executes + async_moderation_hook (parallel)
   │                    ├─ Get last assistant message
   │                    ├─ Classify to workflow state
   │                    ├─ Evaluate constraints
   │                    └─ Record intervention if needed
   │
4. async_post_call_success_hook
   ├─ Classify current response
   ├─ Update state machine
   └─ Complete OTEL trace span
   │
5. Response returned to client
```

### Violation Flow (With Intervention)

```
1. Call N: Constraint violation detected in moderation_hook
   └─ Pending intervention recorded for session
   │
2. Call N+1: async_pre_call_hook
   ├─ Detect pending intervention
   ├─ Apply intervention (modify messages)
   └─ Clear pending intervention
   │
3. Modified request sent to LLM
   │
4. LLM responds with corrected behavior
```

---

## File Structure

```
panoptes/
├── __init__.py          # Public API exports
├── cli.py               # CLI commands (serve, compile, validate, info)
│
├── config/
│   └── settings.py      # PanoptesSettings (pydantic-settings)
│
├── core/
│   └── intervention/
│       └── strategies.py    # Shared intervention strategies
│
├── policy/
│   ├── compiler/        # NLP Policy Compiler
│   │   ├── protocol.py  # PolicyCompiler interface
│   │   ├── base.py      # LLMPolicyCompiler base class
│   │   └── registry.py  # Compiler registry
│   │
│   ├── engines/
│   │   ├── fsm/         # FSM Engine
│   │   │   ├── workflow/# Schema, Parser, StateMachine
│   │   │   ├── classifier.py
│   │   │   ├── tracker.py
│   │   │   ├── injector.py
│   │   │   ├── compiler.py  # FSMCompiler (NL → YAML)
│   │   │   └── engine.py
│   │   │
│   │   └── nemo/        # NeMo Guardrails Engine
│   │       ├── __init__.py
│   │       └── engine.py
│   │
│   └── registry.py      # Policy engine registry
│
├── proxy/
│   ├── server.py        # PanoptesProxy, start_proxy()
│   ├── hooks.py         # PanoptesCallback (LiteLLM hooks)
│   └── middleware.py    # Session/context extraction
│
└── tracing/
    └── otel_tracer.py   # PanoptesTracer (OpenTelemetry)
```

---

## Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `litellm[proxy]` | LLM proxy and routing | >=1.50.0 |
| `nemoguardrails` | NeMo policy engine | >=0.9.0 |
| `opentelemetry-api` | Tracing API | >=1.20.0 |
| `opentelemetry-sdk` | Tracing SDK | >=1.20.0 |
| `opentelemetry-exporter-otlp` | OTLP exporter | >=1.20.0 |
| `pydantic` | Data validation | >=2.0.0 |
| `pydantic-settings` | Configuration management | >=2.0.0 |
| `sentence-transformers` | Embedding classification | >=2.2.0 |
| `pyyaml` | Workflow parsing | >=6.0 |

---

## Extension Points

### Adding New Constraint Types
1. Add to `ConstraintType` enum in `workflow/schema.py`
2. Add validation in `Constraint.validate_constraint_params()`
3. Implement evaluation in `ConstraintEvaluator._evaluate_constraint()`
4. Add message formatting in `_format_violation_message()`

### Adding New Content

### Adding New Intervention Strategies
1. Add to `StrategyType` enum in `panoptes/core/intervention/strategies.py`
2. Create new strategy class extending `InterventionStrategy`
3. Register in `STRATEGY_REGISTRY`

### Custom Classification Methods
Extend `StateClassifier` (in `panoptes/policy/engines/fsm/classifier.py`) and override/add methods to the cascade.

### Custom Session ID Extraction
Modify `SessionExtractor.extract_session_id()` in `panoptes/proxy/middleware.py`.
