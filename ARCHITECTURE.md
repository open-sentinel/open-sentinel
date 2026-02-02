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
| `async_pre_call_hook` | Before LLM call | Apply pending interventions, start trace | Yes |
| `async_moderation_hook` | Parallel with LLM | Classify previous response, check constraints | No |
| `async_post_call_success_hook` | After LLM response | Update state machine, complete trace | Yes |
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

### 2. Workflow Layer (`panoptes/workflow/`)

Defines and manages workflow state machines.

#### `schema.py` - Data Models
Pydantic models for workflow definitions:

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

#### `parser.py` - WorkflowParser
Parses YAML/JSON workflow files into validated `WorkflowDefinition` objects.

```python
workflow = WorkflowParser.parse_file("workflow.yaml")
workflow = WorkflowParser.parse_string(yaml_content)
workflow = WorkflowParser.parse_dict(data_dict)
```

Also includes `WorkflowRegistry` for multi-workflow scenarios (assign different workflows to different sessions).

#### `state_machine.py` - WorkflowStateMachine
Manages workflow execution state per session:

```python
class WorkflowStateMachine:
    async def get_or_create_session(session_id) -> SessionState
    async def transition(session_id, target_state, ...) -> (TransitionResult, error)
    async def get_valid_transitions(session_id) -> Set[str]
    async def get_state_history(session_id) -> List[str]
```

**SessionState** tracks:
- `current_state`: Current workflow state name
- `history`: List of `StateHistoryEntry` (for constraint evaluation)
- `pending_intervention`: Intervention to apply next call
- `constraint_violations`: List of violated constraint names

#### `constraints.py` - ConstraintEvaluator
Evaluates LTL-lite constraints against execution history.

**Constraint Types**:

| Type | Syntax | Meaning | Example |
|------|--------|---------|---------|
| `eventually` | F(target) | Must eventually reach target | "Must reach resolution" |
| `always` | G(condition) | Condition always holds | "Always maintain session" |
| `never` | G(!target) | Target must never occur | "Never share credentials" |
| `precedence` | !B U A | B cannot occur before A | "Verify identity before account actions" |
| `response` | G(A → F(B)) | If A occurs, B must follow | "If escalate, must resolve" |
| `until` | A U B | Stay in A until B | "Stay in greeting until issue identified" |
| `next` | X(target) | Next state must be target | "Next must be verification" |

**Evaluation Results**: `SATISFIED`, `VIOLATED`, or `PENDING` (can't determine yet)

---

### 3. Monitor Layer (`panoptes/monitor/`)

Classifies LLM outputs and orchestrates tracking.

#### `classifier.py` - StateClassifier
Classifies LLM responses to workflow states using a cascade:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Classification Cascade                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. Tool Calls (instant)                                            │
│     - Exact match on function names                                 │
│     - Confidence: 1.0                                               │
│     - Example: update_account → account_action state                │
├─────────────────────────────────────────────────────────────────────┤
│  2. Patterns (regex, ~1ms)                                          │
│     - Regex patterns in response content                            │
│     - Confidence: 0.9                                               │
│     - Example: "let me.*search" → lookup_info state                 │
├─────────────────────────────────────────────────────────────────────┤
│  3. Embeddings (semantic, ~50ms)                                    │
│     - Cosine similarity to state exemplars                          │
│     - Model: all-MiniLM-L6-v2 (22MB, fast)                          │
│     - Confidence: similarity score                                  │
│     - Example: "I'll look into that for you" → lookup_info state    │
├─────────────────────────────────────────────────────────────────────┤
│  4. Fallback                                                        │
│     - Stay in current state                                         │
│     - Confidence: 0.0                                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Performance Target**: <50ms total classification time

#### `tracker.py` - WorkflowTracker
Main orchestration layer. Coordinates classifier, state machine, and constraint evaluator.

```python
class WorkflowTracker:
    async def process_response(session_id, response, context) -> TrackingResult
```

**TrackingResult** contains:
- `classified_state`: Detected workflow state
- `classification_confidence`: 0.0-1.0
- `classification_method`: "tool_call", "pattern", "embedding", "fallback"
- `transition_result`: SUCCESS, INVALID_TRANSITION, etc.
- `constraint_violations`: List of violations
- `intervention_needed`: Intervention name if any

---

### 4. Intervention Layer (`panoptes/intervention/`)

Modifies LLM requests to guide agents back on track.

#### `strategies.py` - Intervention Strategies
Four strategies for injecting corrections:

| Strategy | How It Works | Use Case |
|----------|--------------|----------|
| `SYSTEM_PROMPT_APPEND` | Appends `[WORKFLOW GUIDANCE]` to system message | Gentle guidance, least disruptive |
| `USER_MESSAGE_INJECT` | Inserts `[System Note]` as user message | Important corrections |
| `CONTEXT_REMINDER` | Inserts assistant message `[Context reminder]` | Complex multi-step workflows |
| `HARD_BLOCK` | Raises `WorkflowViolationError` | Critical violations, block request |

#### `prompt_injector.py` - PromptInjector
Applies interventions to request data:

```python
class PromptInjector:
    def inject(data, intervention_name, context, session_id) -> modified_data
```

**Escalation**: Tracks application counts per session. If an intervention is applied more than `max_applications` times, escalates to a stricter strategy.

---

### 5. Tracing Layer (`panoptes/tracing/`)

#### `otel_tracer.py` - PanoptesTracer
Provides session-aware tracing via OpenTelemetry:

```python
class PanoptesTracer:
    def log_event(session_id, name, metadata)  # interventions, deviations
    def log_state_transition(session_id, previous_state, new_state, confidence)
    def log_llm_call(session_id, model, messages, response_content, usage)
```

**Trace Hierarchy**:
- **Session Span**: Root span per session
- **Event Spans**: Child spans for each event (LLM calls, state transitions, etc.)

---

### 6. Configuration (`panoptes/config/`)

#### `settings.py` - PanoptesSettings
Pydantic Settings with env var support:

```python
class PanoptesSettings(BaseSettings):
    # Prefix: PANOPTES_
    # Nested delimiter: __ (e.g., PANOPTES_OTEL__ENDPOINT)
    
    debug: bool
    workflow_path: Optional[str]
    
    otel: OTelConfig
    proxy: ProxyConfig
    classifier: ClassifierConfig
    intervention: InterventionConfig
```

**Key Configuration Options**:
- `PANOPTES_WORKFLOW_PATH`: Path to workflow YAML
- `PANOPTES_PROXY__PORT`: Server port (default 4000)
- `PANOPTES_OTEL__ENDPOINT`: OpenTelemetry OTLP endpoint
- `PANOPTES_CLASSIFIER__MODEL_NAME`: Embedding model (default "all-MiniLM-L6-v2")

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
├── cli.py               # CLI commands (serve, validate, info)
│
├── config/
│   └── settings.py      # PanoptesSettings (pydantic-settings)
│
├── proxy/
│   ├── server.py        # PanoptesProxy, start_proxy()
│   ├── hooks.py         # PanoptesCallback (LiteLLM hooks)
│   └── middleware.py    # Session/context extraction
│
├── workflow/
│   ├── schema.py        # Pydantic models (State, Transition, Constraint, etc.)
│   ├── parser.py        # WorkflowParser, WorkflowRegistry
│   ├── state_machine.py # WorkflowStateMachine, SessionState
│   └── constraints.py   # ConstraintEvaluator, LTL-lite evaluation
│
├── monitor/
│   ├── classifier.py    # StateClassifier (tool/pattern/embedding)
│   └── tracker.py       # WorkflowTracker (main orchestrator)
│
├── intervention/
│   ├── strategies.py    # InterventionStrategy implementations
│   └── prompt_injector.py # PromptInjector
│
└── tracing/
    └── otel_tracer.py       # PanoptesTracer (OpenTelemetry)
```

---

## Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `litellm[proxy]` | LLM proxy and routing | >=1.50.0 |
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

### Adding New Intervention Strategies
1. Add to `StrategyType` enum in `intervention/strategies.py`
2. Create new strategy class extending `InterventionStrategy`
3. Register in `STRATEGY_REGISTRY`

### Custom Classification Methods
Extend `StateClassifier` and override/add methods to the cascade.

### Custom Session ID Extraction
Modify `SessionExtractor.extract_session_id()` in `proxy/middleware.py`.
