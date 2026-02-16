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
│  Your App       │─────▶│  ┌──────────┐  ┌─────────────┐               │─────▶│  LLM Provider   │
│  (LLM Client)   │      │  │  Hooks   │─▶│ Interceptor │               │      │  (OpenAI, etc.) │
│                 │◀─────│  │safe_hook │  │ ┌─────────┐ │               │◀─────│                 │
└─────────────────┘      │  └──────────┘  │ │Checkers │ │               │      └─────────────────┘
                         │       │        │ └─────────┘ │               │
                         │       ▼        └─────────────┘               │
                         │  ┌────────────────────────────────────┐      │
                         │  │         Policy Engines             │      │
                         │  │  ┌─────┐  ┌─────┐  ┌──────┐        │      │
                         │  │  │ FSM │  │ LLM │  │ NeMo │  ...   │      │
                         │  │  └─────┘  └─────┘  └──────┘        │      │
                         │  └────────────────────────────────────┘      │ 
                         │       │                                      │
                         │       ▼                                      │
                         │  ┌────────────────────────────────────┐      │
                         │  │       OpenTelemetry Tracing        │      │
                         │  └────────────────────────────────────┘      │
                         └──────────────────────────────────────────────┘
```

## Design Principles

### 1. Zero Code Changes for Customers
Customers only change `base_url` in their LLM client to point at Panoptes. No SDK integration, no code changes beyond configuration.

### 2. Non-Blocking Monitoring
State classification and constraint evaluation can run asynchronously in parallel with LLM calls (via `ASYNC` mode checkers), adding **zero latency** to the critical path.

### 3. Deferred Intervention
When a deviation is detected, the intervention is **scheduled for the next call**, not applied immediately. This preserves the current response while ensuring correction.

### 4. Fail-Open Hardening
All hooks are wrapped with `safe_hook()` which provides timeout and exception handling. If a hook fails or times out, the request passes through unmodified. Only `WorkflowViolationError` (intentional blocks) propagates.

### 5. Pluggable Engines
Policy engines are registered via `@register_engine("type")` and loaded dynamically. The `Interceptor` wraps them as `Checker` instances so the hook layer doesn't need to know engine internals.

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
Implements LiteLLM's `CustomLogger` interface. All hooks are wrapped with `safe_hook()` for fail-open semantics.

| Hook | Timing | Purpose | Blocking? |
|------|--------|---------|-----------|
| `async_pre_call_hook` | Before LLM call | Apply pending async results, run PRE_CALL checkers via Interceptor, start trace | Yes |
| `async_moderation_hook` | Parallel with LLM | Reserved (currently unused) | No |
| `async_post_call_success_hook` | After LLM response | Run POST_CALL checkers via Interceptor, start async checkers, complete trace | Yes |
| `async_post_call_failure_hook` | After LLM error | Log failure | Yes |

**Key Design**: The callback delegates all policy evaluation to the `Interceptor`. It lazily initializes the policy engine and wraps it as `PolicyEngineChecker` instances for PRE_CALL (sync) and POST_CALL (sync + async) phases.

**Fail-Open**: Each hook's implementation is in a private `_*_impl` method. The public hook wraps it with `safe_hook()`:
```python
async def async_pre_call_hook(self, ...):
    return await safe_hook(
        self._pre_call_impl, ...,
        timeout=self._hook_timeout,
        fallback=data,
        hook_name="pre_call",
    )
```

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

### 2. Interceptor Framework (`panoptes/core/interceptor/`)

The Interceptor is the orchestration layer between hooks and policy engines. It provides a general-purpose checker system for running checks at different phases with different execution modes.

#### `types.py` - Core Types

| Type | Values | Purpose |
|------|--------|---------|
| `CheckPhase` | `PRE_CALL`, `POST_CALL` | When a checker runs |
| `CheckerMode` | `SYNC`, `ASYNC` | How a checker executes |
| `CheckDecision` | `PASS`, `WARN`, `FAIL` | Result of a check |
| `CheckResult` | — | Decision + optional modified_data + violations |
| `CheckerContext` | — | Session ID, request/response data passed to checkers |
| `InterceptionResult` | — | Aggregated result from running checkers |

#### `checker.py` - Checker ABC
Abstract base class all checkers implement:

```python
class Checker(ABC):
    @property
    def name(self) -> str: ...      # Unique name
    @property
    def phase(self) -> CheckPhase: ...  # PRE_CALL or POST_CALL
    @property
    def mode(self) -> CheckerMode: ...  # SYNC or ASYNC

    async def check(self, context: CheckerContext) -> CheckResult: ...
```

#### `adapters.py` - PolicyEngineChecker
Wraps any `PolicyEngine` as a `Checker`:

| PolicyDecision | → CheckDecision |
|----------------|-----------------|
| `ALLOW` | `PASS` |
| `DENY` | `FAIL` |
| `MODIFY` | `WARN` (with modified_data) |
| `WARN` | `WARN` |

For `PRE_CALL`, calls `engine.evaluate_request()`. For `POST_CALL`, calls `engine.evaluate_response()`.

#### `interceptor.py` - Interceptor Orchestrator
Manages checker execution across the request lifecycle:

**`run_pre_call(session_id, request_data)`**:
1. Collect completed async results from previous request
   - `FAIL` → block this request
   - `WARN` → merge `modified_data` into request
2. Run sync PRE_CALL checkers
   - `FAIL` → reject request
   - `WARN` → merge modifications
3. Start async PRE_CALL checkers in background
4. Return `InterceptionResult`

**`run_post_call(session_id, request_data, response_data)`**:
1. Run sync POST_CALL checkers
   - `FAIL` → reject response
   - `WARN` → merge modifications
2. Start async POST_CALL checkers in background (results applied next request)
3. Return `InterceptionResult`

---

### 3. Policy Layer (`panoptes/policy/`)

Panoptes supports pluggable policy engines. All engines implement the `PolicyEngine` ABC.

#### `protocols.py` - Engine Contracts

```python
class PolicyEngine(ABC):
    name: str                    # Unique instance name
    engine_type: str             # "fsm", "nemo", "llm", "composite"

    async def initialize(config)
    async def evaluate_request(session_id, request_data, context) -> PolicyEvaluationResult
    async def evaluate_response(session_id, response_data, request_data, context) -> PolicyEvaluationResult
    async def get_session_state(session_id) -> Optional[Dict]
    async def reset_session(session_id)
    async def shutdown()

class StatefulPolicyEngine(PolicyEngine):
    # Adds state classification capabilities
    async def classify_response(session_id, response_data, current_state) -> StateClassificationResult
    async def get_current_state(session_id) -> str
    async def get_state_history(session_id) -> List[str]
    async def get_valid_next_states(session_id) -> List[str]
```

#### `registry.py` - Engine Registry
Dynamic registration and lookup of policy engines:

```python
@register_engine("fsm")
class FSMPolicyEngine(StatefulPolicyEngine): ...

# Usage
engine = PolicyEngineRegistry.create("fsm")
await engine.initialize(config)
```

---

### 4. FSM Engine (`panoptes/policy/engines/fsm/`)

Wraps the workflow state machine, monitor, and injector for deterministic workflow enforcement.

**Components**:
- **Workflow**: Schema, Parser, StateMachine, Constraints (in `workflow/` subpackage)
- **Monitor**: StateClassifier
- **Injector**: PromptInjector for FSM-based interventions

#### `workflow/schema.py` - Workflow Data Models

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

---

### 5. LLM Engine (`panoptes/policy/engines/llm/`)

Uses a lightweight LLM (e.g. `gpt-4o-mini`) as a reasoning backbone for all policy evaluation tasks.

**Components**:
- **`engine.py`** — `LLMPolicyEngine`: Main orchestrator registered as `@register_engine("llm")`
- **`state_classifier.py`** — `LLMStateClassifier`: LLM-based classification with confidence tiers and skip violation detection
- **`drift_detector.py`** — `DriftDetector`: Temporal + semantic drift computation
- **`constraint_evaluator.py`** — `LLMConstraintEvaluator`: Soft constraint evaluation with evidence memory
- **`intervention.py`** — `InterventionHandler`: Maps violations/drift to intervention strategies with cooldown
- **`llm_client.py`** — `LLMClient`: Thin wrapper around litellm for structured JSON responses
- **`models.py`** — `SessionContext`, `DriftScores`, `LLMClassificationResult`, etc.
- **`prompts.py`**, **`templates.py`** — System/user prompt templates

**Evaluation Flow** (`evaluate_response`):
1. Extract assistant message and tool calls from response
2. Add turn to session sliding window
3. **Classify state** via LLM (confidence tiers: CONFIDENT, UNCERTAIN, LOST)
4. **Compute drift** (temporal + semantic, weighted combination)
5. **Evaluate constraints** via LLM (batched, with evidence memory)
6. **Decide intervention** (severity → strategy mapping, cooldown, self-correction detection)
7. Record state transition

**Key Concepts**:
- **Confidence Tiers**: `CONFIDENT` (>0.8), `UNCERTAIN` (0.5-0.8), `LOST` (<0.5)
- **Drift Levels**: `NOMINAL` (<0.3), `WARNING` (0.3-0.6), `INTERVENTION` (0.6-0.85), `CRITICAL` (>0.85)
- **Structural Drift**: 3 consecutive UNCERTAIN classifications triggers structural drift flag
- **Cooldown**: Prevents intervention spam (default 2 turns between interventions)
- **Self-Correction**: Detects decreasing drift and cancels pending intervention

---

### 6. NeMo Guardrails Engine (`panoptes/policy/engines/nemo/`)

Integrates NVIDIA's NeMo Guardrails for comprehensive content safety and dialog management.

**Key Features**:
- **Input Rails**: Pre-processing checks for jailbreaks, PII, and toxicity.
- **Output Rails**: Post-processing verification against safety policies and hallucination checks.
- **Dialog Rails**: Programmable conversation flow using Colang.

**Panoptes Bridge**:
- Registers custom actions (`panoptes_log_violation`, `panoptes_request_intervention`) to allow Colang scripts to interact with Panoptes' tracing and intervention systems.
- Adapts NeMo's `generate_async` output to `PolicyEvaluationResult`, translating blocked responses into policy violations.

---

### 7. Composite Engine (`panoptes/policy/engines/composite/`)

Combines multiple policy engines to run in parallel, merging their results with configurable strategies.

**Configuration**:
```yaml
# Environment variable approach
PANOPTES_POLICY__ENGINE__TYPE=composite

# Or in code
config = {
    "engines": [
        {"type": "nemo", "config": {"config_path": "./nemo_config/"}},
        {"type": "fsm", "config": {"config_path": "./workflow.yaml"}},
    ],
    "strategy": "all",     # "all" or "first_deny"
    "parallel": True,      # Run engines in parallel
}
```

**Merge Strategy**:
- **Decision**: Most restrictive wins (DENY > MODIFY > WARN > ALLOW)
- **Violations**: Collected from all engines
- **Intervention**: First one found
- **Modified request**: First one found
- **Metadata**: Merged from all engines

---

### 8. Policy Compiler (`panoptes/policy/compiler/`)

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

**Extension Point**: To add a compiler for a new engine:
1. Create a class extending `LLMPolicyCompiler`
2. Implement `_build_compilation_prompt()` for engine-specific prompting
3. Implement `_parse_compilation_response()` for response parsing
4. Register with `@register_compiler("engine_name")`

---

### 9. Core Layer (`panoptes/core/`)

Shared components used across different policy engines.

#### `intervention/` - Strategies
Defines the base strategies for modifying LLM requests.

| Strategy | How It Works | Use Case |
|----------|--------------|----------|
| `SYSTEM_PROMPT_APPEND` | Appends `[WORKFLOW GUIDANCE]` to system message | Gentle guidance, least disruptive |
| `USER_MESSAGE_INJECT` | Inserts `[System Note]` as user message | Important corrections |
| `CONTEXT_REMINDER` | Inserts assistant message `[Context reminder]` | Complex multi-step workflows |
| `HARD_BLOCK` | Raises `WorkflowViolationError` | Critical violations, block request |

#### `interceptor/` - Checker Framework
See [Interceptor Framework](#2-interceptor-framework-panoptescoreinterceptor) above.

---

### 10. Tracing Layer (`panoptes/tracing/`)

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

### 11. Configuration (`panoptes/config/`)

#### `settings.py` - PanoptesSettings
Pydantic Settings with env var support:

```python
class PanoptesSettings(BaseSettings):
    # Prefix: PANOPTES_
    # Nested delimiter: __ (e.g., PANOPTES_OTEL__ENDPOINT)
    
    debug: bool
    
    otel: OTelConfig
    proxy: ProxyConfig
    classifier: ClassifierConfig
    intervention: InterventionConfig
    policy: PolicyConfig

class PolicyConfig(BaseModel):
    engine: PolicyEngineConfig     # Primary engine configuration
    engines: List[PolicyEngineConfig]  # For composite engine
    fail_open: bool = True         # Pass-through on hook failures
    hook_timeout_seconds: float = 30.0  # Max hook execution time

class PolicyEngineConfig(BaseModel):
    type: str = "nemo"             # "fsm", "llm", "nemo", "composite"
    enabled: bool = True
    config_path: Optional[str] = None
    config: Dict[str, Any] = {}    # Engine-specific config
```

**Key Configuration Options**:
- `PANOPTES_POLICY__ENGINE__TYPE`: Engine type (fsm, llm, nemo, composite)
- `PANOPTES_POLICY__ENGINE__CONFIG_PATH`: Path to workflow YAML or NeMo config directory
- `PANOPTES_POLICY__FAIL_OPEN`: Enable fail-open mode (default: true)
- `PANOPTES_POLICY__HOOK_TIMEOUT_SECONDS`: Hook timeout (default: 30s)
- `PANOPTES_PROXY__PORT`: Server port (default 4000)
- `PANOPTES_OTEL__ENDPOINT`: OpenTelemetry OTLP endpoint

---

## Data Flow

### Normal Flow (No Violation)

```
1. Client sends LLM request
   │
2. async_pre_call_hook (wrapped in safe_hook)
   ├─ Extract session ID
   ├─ Interceptor.run_pre_call():
   │   ├─ Collect completed async results (none on first call)
   │   └─ Run sync PRE_CALL checkers (PolicyEngineChecker → engine.evaluate_request)
   └─ Start OTEL trace span
   │
3. LLM call executes
   │
4. async_post_call_success_hook (wrapped in safe_hook)
   ├─ Interceptor.run_post_call():
   │   ├─ Run sync POST_CALL checkers (PolicyEngineChecker → engine.evaluate_response)
   │   └─ Start async POST_CALL checkers in background
   └─ Complete OTEL trace span
   │
5. Response returned to client
```

### Violation Flow (With Intervention)

```
1. Call N: POST_CALL checker detects violation
   ├─ Sync checker → WARN with modified_data (intervention info)
   └─ Or async checker → result stored for next request

2. Call N+1: async_pre_call_hook
   ├─ Interceptor.run_pre_call():
   │   ├─ Collect async results from Call N
   │   │   ├─ FAIL → block request (WorkflowViolationError)
   │   │   └─ WARN → merge modifications (intervention) into request
   │   └─ Run sync PRE_CALL checkers (engine applies pending intervention)
   └─ Modified request sent to LLM

3. LLM responds with corrected behavior
```

### Fail-Open Flow

```
1. Hook implementation throws unexpected exception or times out
   │
2. safe_hook catches exception
   ├─ WorkflowViolationError → re-raised (intentional block)
   └─ Any other error:
       ├─ Log warning with hook name and error
       ├─ Increment _fail_open_counter[hook_name]
       └─ Return fallback value (original data/response)
   │
3. Request/response passes through unmodified
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
│   ├── interceptor/     # Checker orchestration framework
│   │   ├── types.py     # CheckPhase, CheckerMode, CheckDecision, CheckResult
│   │   ├── checker.py   # Checker ABC
│   │   ├── adapters.py  # PolicyEngineChecker (wraps engines as checkers)
│   │   └── interceptor.py  # Interceptor orchestrator
│   │
│   └── intervention/
│       └── strategies.py    # Shared intervention strategies
│
├── policy/
│   ├── protocols.py     # PolicyEngine, StatefulPolicyEngine ABCs
│   ├── registry.py      # PolicyEngineRegistry + @register_engine
│   │
│   ├── compiler/        # NLP Policy Compiler
│   │   ├── protocol.py  # PolicyCompiler interface
│   │   ├── base.py      # LLMPolicyCompiler base class
│   │   └── registry.py  # Compiler registry
│   │
│   └── engines/
│       ├── fsm/         # FSM Engine
│       │   ├── workflow/ # Schema, Parser, StateMachine, Constraints
│       │   ├── classifier.py

│       │   ├── injector.py
│       │   ├── compiler.py  # FSMCompiler (NL → YAML)
│       │   └── engine.py
│       │
│       ├── llm/         # LLM Engine
│       │   ├── engine.py           # LLMPolicyEngine orchestrator
│       │   ├── state_classifier.py # LLM-based state classification
│       │   ├── drift_detector.py   # Temporal + semantic drift
│       │   ├── constraint_evaluator.py  # LLM-based constraint eval
│       │   ├── intervention.py     # Intervention decision engine
│       │   ├── llm_client.py       # LiteLLM wrapper
│       │   ├── models.py           # SessionContext, DriftScores, etc.
│       │   ├── prompts.py          # LLM prompt templates
│       │   └── templates.py        # Intervention message templates
│       │
│       ├── nemo/        # NeMo Guardrails Engine
│       │   ├── __init__.py
│       │   └── engine.py
│       │
│       └── composite/   # Composite Engine (multi-engine)
│           ├── __init__.py
│           └── engine.py
│
├── proxy/
│   ├── server.py        # PanoptesProxy, start_proxy()
│   ├── hooks.py         # PanoptesCallback (LiteLLM hooks + safe_hook)
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

### Adding a New Policy Engine

1. Create a new engine class extending `PolicyEngine` (or `StatefulPolicyEngine` for stateful engines)
2. Implement all abstract methods (`initialize`, `evaluate_request`, `evaluate_response`, etc.)
3. Register with `@register_engine("my_engine")`
4. The `Interceptor` will automatically wrap it via `PolicyEngineChecker`

### Adding a New Checker

1. Create a class extending `Checker`
2. Implement `name`, `phase`, `mode`, and `check()` method
3. Register by adding it to the checker list in `PanoptesCallback._get_interceptor()`

### Adding New Constraint Types
1. Add to `ConstraintType` enum in `workflow/schema.py`
2. Add validation in `Constraint.validate_constraint_params()`
3. Implement evaluation in `ConstraintEvaluator._evaluate_constraint()`
4. Add message formatting in `_format_violation_message()`

### Adding New Intervention Strategies
1. Add to `StrategyType` enum in `panoptes/core/intervention/strategies.py`
2. Create new strategy class extending `InterventionStrategy`
3. Register in `STRATEGY_REGISTRY`

### Custom Classification Methods
Extend `StateClassifier` (in `panoptes/policy/engines/fsm/classifier.py`) and override/add methods to the cascade.

### Custom Session ID Extraction
Modify `SessionExtractor.extract_session_id()` in `panoptes/proxy/middleware.py`.
