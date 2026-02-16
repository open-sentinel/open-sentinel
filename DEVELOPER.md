# Developer Documentation

This document provides guidance for developers (humans and AI agents) working on the Open Sentinel codebase.

## Quick Start

### Setup

```bash
# Clone and install
cd argus
pip install -e ".[dev]"

# Run tests
pytest

# Initialize configuration (creates osentinel.yaml and policy.yaml)
osentinel init

# Start proxy (uses osentinel.yaml)
osentinel serve
```

### Alternative: Manual Configuration

You can also configure via environment variables (useful for testing different engines quickly):

```bash
# Start proxy with FSM workflow
export OSNTL_POLICY__ENGINE__TYPE=fsm
export OSNTL_POLICY__ENGINE__CONFIG_PATH=workflow.yaml
osentinel serve --port 4000

# Start proxy with LLM engine
export OSNTL_POLICY__ENGINE__TYPE=llm
export OSNTL_POLICY__ENGINE__CONFIG_PATH=workflow.yaml
osentinel serve --port 4000

# Start proxy with NeMo Guardrails (requires examples/nemo_guardrails/config/)
export OSNTL_POLICY__ENGINE__CONFIG_PATH=examples/nemo_guardrails/config/
osentinel serve --port 4000
```

### Point Your LLM Client at Open Sentinel

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",  # Point to Open Sentinel
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Enable OpenTelemetry Tracing (Optional)

```bash
export OSNTL_OTEL__ENDPOINT=http://localhost:4317
export OSNTL_OTEL__SERVICE_NAME=opensentinel
# Start Jaeger for local development: docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
```

Traces appear in your OTLP-compatible backend (Jaeger, Zipkin, etc.) grouped by session.

---

## Codebase Conventions

### File Organization

- **One class per file** for major components
- **`__init__.py`** exports public API
- **Type hints required** on all function signatures
- **Docstrings** on all public classes and methods

### Naming Conventions

| Entity | Convention | Example |
|--------|------------|---------|
| Files | `snake_case.py` | `state_machine.py` |
| Classes | `PascalCase` | `PolicyEngine` |
| Functions/Methods | `snake_case` | `process_response` |
| Constants | `UPPER_SNAKE_CASE` | `STRATEGY_REGISTRY` |
| Private | `_prefix` | `_sessions`, `_extract_content` |

### Import Order

```python
# 1. Standard library
import logging
from typing import Optional, Dict, Any

# 2. Third-party
from pydantic import BaseModel
import litellm

# 3. Local imports
from opensentinel.policy.protocols import PolicyEngine
from opensentinel.config.settings import SentinelSettings
```

---

## Module Guide

### Working with Workflows

#### Creating a Workflow Programmatically

```python
from opensentinel.policy.engines.fsm import (
    WorkflowDefinition,
    State,
    Transition,
    Constraint,
    ConstraintType,
    ClassificationHint,
)

workflow = WorkflowDefinition(
    name="my-workflow",
    version="1.0",
    states=[
        State(
            name="greeting",
            is_initial=True,
            classification=ClassificationHint(
                patterns=["hello", "hi", "hey"]
            )
        ),
        State(
            name="action",
            classification=ClassificationHint(
                tool_calls=["do_action"]
            )
        ),
        State(
            name="done",
            is_terminal=True,
        ),
    ],
    transitions=[
        Transition(from_state="greeting", to_state="action"),
        Transition(from_state="action", to_state="done"),
    ],
    constraints=[
        Constraint(
            name="must_greet_first",
            type=ConstraintType.PRECEDENCE,
            trigger="greeting",
            target="action",
            severity="error",
            intervention="prompt_greet",
        )
    ],
    interventions={
        "prompt_greet": "Please greet the user before taking action."
    }
)
```

#### Loading from YAML

```python
from opensentinel.policy.engines.fsm import WorkflowParser

# From file
workflow = WorkflowParser.parse_file("workflow.yaml")

# From string
workflow = WorkflowParser.parse_string("""
name: test
states:
  - name: start
    is_initial: true
""")

# Validate without loading
is_valid, message = WorkflowParser.validate_file("workflow.yaml")
```

### Working with the Policy Compiler

The Policy Compiler converts natural language policies to FSM workflow configuration:

```python
from opensentinel.policy.compiler import PolicyCompilerRegistry
from pathlib import Path
import asyncio

async def compile_policy():
    # Create FSM compiler
    compiler = PolicyCompilerRegistry.create("fsm")
    
    # Compile natural language policy
    result = await compiler.compile(
        "Agent must verify identity before processing refunds. "
        "Never share internal system information.",
        context={"domain": "customer support"}  # Optional hints
    )
    
    if result.success:
        # Export to YAML file
        compiler.export(result, Path("workflow.yaml"))
        
        # Access the compiled workflow
        workflow = result.config
        print(f"Generated {len(workflow.states)} states")
        print(f"Generated {len(workflow.constraints)} constraints")
    else:
        print("Errors:", result.errors)
        
    # Check warnings
    if result.warnings:
        print("Warnings:", result.warnings)

asyncio.run(compile_policy())
```

**CLI Usage:**
```bash
# Basic compilation
osentinel compile "verify identity before refunds"

# With options
osentinel compile "never share internal info" \\
    -o my_workflow.yaml \\
    -d "customer support" \\
    -m gpt-4o
```

### Working with Policy Engines

#### Using the Engine Registry

```python
from opensentinel.policy import PolicyEngineRegistry

# Create and initialize any registered engine
engine = await PolicyEngineRegistry.create_and_initialize(
    "fsm",
    {"config_path": "./workflow.yaml"}
)

# Evaluate a request
result = await engine.evaluate_request(
    session_id="session-123",
    request_data={"messages": [{"role": "user", "content": "Hello!"}]},
)

if result.decision == PolicyDecision.DENY:
    print("Blocked:", result.violations)

# List available engines
print(PolicyEngineRegistry.list_engines())  # ["fsm", "llm", "nemo", "composite"]
```

#### Choosing an Engine

| Engine | When to Use |
|--------|-------------|
| `fsm` | Deterministic workflows with known states, tools, and transitions |
| `llm` | Soft constraints requiring reasoning, drift detection, or fuzzy classification |
| `nemo` | Content safety, jailbreak detection, PII filtering, Colang-scripted dialog flows |
| `composite` | Combining multiple engines (e.g., `nemo` for safety + `fsm` for workflow) |

### Working with the LLM Engine

The LLM engine uses a lightweight LLM for all policy evaluation tasks.

```python
engine = await PolicyEngineRegistry.create_and_initialize(
    "llm",
    {
        "config_path": "./workflow.yaml",
        "model": "gpt-4o-mini",            # LLM used for classification/evaluation
        "temporal_weight": 0.55,            # Weight for temporal drift (vs semantic)
        "cooldown_turns": 2,                # Turns between interventions
    }
)

# Evaluate a response
result = await engine.evaluate_response(
    session_id="session-123",
    response_data=llm_response,
    request_data={"messages": [...]},
)

# Access session state (rich with LLM engine info)
state = await engine.get_session_state("session-123")
print(state["current_state"])
print(state["drift_score"])
print(state["is_structurally_drifting"])
```

**Key Concepts**:
- **Confidence Tiers**: CONFIDENT (>0.8), UNCERTAIN (0.5–0.8), LOST (<0.5)
- **Drift Scores**: Composite of temporal (state sequence divergence) and semantic (content drift) scores
- **Cooldown**: Prevents intervention spam — default 2 turns between interventions
- **Self-Correction**: If drift decreases between turns, pending intervention is cancelled

### Working with NeMo Guardrails

#### Configuration Structure
NeMo Guardrails requires a configuration directory containing:
- `config.yml`: Main configuration (models, rails instructions)
- `prompts.yml`: (Optional) Custom prompts
- `actions.py`: (Optional) Custom python actions (if not using Open Sentinel registration)

Example `config.yml`:
```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o

rails:
  input:
    flows:
      - check input security
  output:
    flows:
      - check output safety
```

#### Registering Custom Actions
Open Sentinel automatically registers bridge actions, but you can add your own by registering them with `custom_actions` in the engine config or via your application bootstrapping.

```python
async def my_custom_action(context={}):
    return {"status": "ok"}
```

#### Debugging NeMo
Use `trace_block` to capture internal NeMo steps in your traces:

```python
with tracer.trace_block("nemo_step", session_id) as span:
    # NeMo logic here
    pass
```

### Working with the Interceptor

The Interceptor is the core orchestration layer. Policy engines are wrapped as `Checker` instances via the `PolicyEngineChecker` adapter.

#### Creating a Custom Checker

```python
from opensentinel.core.interceptor import Checker, CheckPhase, CheckerMode, CheckResult, CheckDecision, CheckerContext

class MyChecker(Checker):
    @property
    def name(self) -> str:
        return "my_custom_checker"
    
    @property
    def phase(self) -> CheckPhase:
        return CheckPhase.POST_CALL  # Run after LLM response
    
    @property
    def mode(self) -> CheckerMode:
        return CheckerMode.SYNC  # Block until complete
    
    async def check(self, context: CheckerContext) -> CheckResult:
        # Access request/response data
        response = context.response_data
        
        # Your custom logic here
        if some_condition:
            return CheckResult(
                decision=CheckDecision.FAIL,
                checker_name=self.name,
                message="Custom check failed",
            )
        
        return CheckResult(
            decision=CheckDecision.PASS,
            checker_name=self.name,
        )
```

#### Using the Interceptor Directly

```python
from opensentinel.core.interceptor import Interceptor

interceptor = Interceptor(checkers=[my_checker, another_checker])

# Pre-call phase
result = await interceptor.run_pre_call(
    session_id="session-123",
    request_data={"messages": [...]},
)

if not result.allowed:
    raise Exception("Request blocked")

# Post-call phase  
result = await interceptor.run_post_call(
    session_id="session-123",
    request_data={"messages": [...]},
    response_data=llm_response,
)
```

### Working with the State Machine (FSM Engine)

```python
from opensentinel.policy.engines.fsm import WorkflowStateMachine

machine = WorkflowStateMachine(workflow)

# Create/get session
session = await machine.get_or_create_session("session-123")

# Transition
result, error = await machine.transition(
    "session-123",
    "new_state",
    context={"key": "value"},
    confidence=0.95,
    method="tool_call"
)

# Query state
history = await machine.get_state_history("session-123")
valid_next = await machine.get_valid_transitions("session-123")
is_done = await machine.is_in_terminal_state("session-123")

# Set intervention for next call
await machine.set_pending_intervention("session-123", "intervention_name")
```

### Working with the Classifier (FSM Engine)

```python
from opensentinel.policy.engines.fsm import StateClassifier

classifier = StateClassifier(workflow.states)

# Classify an LLM response
result = classifier.classify(
    response={
        "choices": [{
            "message": {
                "content": "Let me search that for you",
                "tool_calls": []
            }
        }]
    },
    current_state="greeting"
)

print(result.state_name)    # "lookup_info"
print(result.confidence)    # 0.9
print(result.method)        # "pattern"

# Quick classification from tool call only
result = classifier.classify_from_tool_call("search_kb")
```

### Working with Constraints (FSM Engine)

```python
from opensentinel.policy.engines.fsm import ConstraintEvaluator

evaluator = ConstraintEvaluator(workflow.constraints)

# Evaluate all constraints
violations = evaluator.evaluate_all(session_state)

# Check a proposed transition
violations = evaluator.evaluate_all(
    session_state,
    proposed_state="account_action"
)

for v in violations:
    print(f"{v.constraint_name}: {v.message}")
    print(f"  Intervention: {v.intervention}")
```

### Working with Interventions (FSM Engine)

```python
from opensentinel.policy.engines.fsm import PromptInjector

injector = PromptInjector(workflow)

# Apply an intervention
modified_data = injector.inject(
    data={"messages": [...]},
    intervention_name="prompt_verify_identity",
    context={"current_state": "greeting"},
    session_id="session-123"
)

# List available interventions
names = injector.list_interventions()

# Get intervention info
info = injector.get_intervention_info("prompt_verify_identity")
```



---

## Adding New Features

### Adding a New Policy Engine

**Step 1**: Create the engine class:

```python
# opensentinel/policy/engines/my_engine/engine.py
from opensentinel.policy.protocols import PolicyEngine, PolicyEvaluationResult, PolicyDecision
from opensentinel.policy.registry import register_engine

@register_engine("my_engine")
class MyPolicyEngine(PolicyEngine):
    @property
    def name(self) -> str:
        return "my_engine"
    
    @property
    def engine_type(self) -> str:
        return "my_engine"
    
    async def initialize(self, config):
        # Load configuration
        ...
    
    async def evaluate_request(self, session_id, request_data, context=None):
        # Evaluate incoming request
        return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)
    
    async def evaluate_response(self, session_id, response_data, request_data, context=None):
        # Evaluate LLM response
        return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)
    
    async def get_session_state(self, session_id):
        return None
    
    async def reset_session(self, session_id):
        pass
```

**Step 2**: Import in `__init__.py` to trigger registration:

```python
# opensentinel/policy/engines/my_engine/__init__.py
from .engine import MyPolicyEngine
```

The engine will be automatically wrapped as a `PolicyEngineChecker` by the hook system.

### Adding a New Checker

For custom checks that don't fit the `PolicyEngine` pattern:

```python
# opensentinel/core/interceptor/my_checker.py
from opensentinel.core.interceptor import Checker, CheckPhase, CheckerMode, CheckResult, CheckDecision, CheckerContext

class MyChecker(Checker):
    @property
    def name(self) -> str:
        return "my_checker"
    
    @property
    def phase(self) -> CheckPhase:
        return CheckPhase.POST_CALL
    
    @property
    def mode(self) -> CheckerMode:
        return CheckerMode.ASYNC  # Runs in background
    
    async def check(self, context: CheckerContext) -> CheckResult:
        # Your logic here
        return CheckResult(decision=CheckDecision.PASS, checker_name=self.name)
```

Then register it in `SentinelCallback._get_interceptor()`.

### Adding a New Constraint Type

**Step 1**: Add to enum in `opensentinel/policy/engines/fsm/workflow/schema.py`:

```python
class ConstraintType(str, Enum):
    # ... existing types ...
    MY_NEW_TYPE = "my_new_type"
```

**Step 2**: Add validation in `Constraint.validate_constraint_params()`:

```python
@model_validator(mode="after")
def validate_constraint_params(self):
    # ... existing validation ...
    
    if t == ConstraintType.MY_NEW_TYPE and not self.target:
        raise ValueError("MY_NEW_TYPE constraint requires 'target'")
    
    return self
```

**Step 3**: Implement evaluation in `opensentinel/policy/engines/fsm/workflow/constraints.py`:

```python
def _evaluate_constraint(self, constraint, history, session):
    match constraint.type:
        # ... existing cases ...
        
        case ConstraintType.MY_NEW_TYPE:
            return self._eval_my_new_type(constraint.target, history)

def _eval_my_new_type(self, target, history):
    # Implement logic
    # Return EvaluationResult.SATISFIED, VIOLATED, or PENDING
    pass
```

**Step 4**: Add message formatting in `_format_violation_message()`:

```python
case ConstraintType.MY_NEW_TYPE:
    return f"Constraint '{constraint.name}': My new type message. History: {recent_history}"
```

### Adding a New Intervention Strategy

**Step 1**: Add to enum in `opensentinel/core/intervention/strategies.py`:

```python
class StrategyType(Enum):
    # ... existing types ...
    MY_STRATEGY = "my_strategy"
```

**Step 2**: Create strategy class:

```python
class MyStrategy(InterventionStrategy):
    """My new intervention strategy."""
    
    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        messages = list(data.get("messages", []))
        correction = self.format_message(config.message_template, context)
        
        # Modify messages as needed
        # ...
        
        data["messages"] = messages
        logger.debug("Applied my_strategy intervention")
        return data
```

**Step 3**: Register in `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY: Dict[StrategyType, InterventionStrategy] = {
    # ... existing entries ...
    StrategyType.MY_STRATEGY: MyStrategy(),
}
```

### Adding a New Classification Method (FSM Engine)

Modify `StateClassifier` in `opensentinel/policy/engines/fsm/classifier.py`:

```python
def classify(self, response, current_state):
    # ... existing cascade ...
    
    # Add new method in the cascade
    if content:
        result = self._classify_by_my_method(content)
        if result:
            return result
    
    # ... fallback ...

def _classify_by_my_method(self, content: str) -> Optional[ClassificationResult]:
    # Implement classification logic
    # Return ClassificationResult or None
    pass
```

### Adding a New Policy Compiler

To add a compiler for a new policy engine:

**Step 1**: Create a compiler class in the engine's directory:

```python
# opensentinel/policy/engines/my_engine/compiler.py
from opensentinel.policy.compiler.base import LLMPolicyCompiler
from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import register_compiler

@register_compiler("my_engine")
class MyEngineCompiler(LLMPolicyCompiler):
    @property
    def engine_type(self) -> str:
        return "my_engine"
    
    def _build_compilation_prompt(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Build prompt with engine-specific schema
        return f"""Convert to my engine format:
        {natural_language}
        
        Output JSON with: ..."""
    
    def _parse_compilation_response(
        self,
        response: Dict[str, Any],
        natural_language: str,
    ) -> CompilationResult:
        # Parse JSON into engine-specific config
        config = MyEngineConfig(**response)
        return CompilationResult(
            success=True,
            config=config,
            metadata={"source": natural_language[:100]}
        )
    
    def export(self, result: CompilationResult, output_path: Path) -> None:
        # Write config to file(s)
        with open(output_path, "w") as f:
            yaml.dump(result.config.to_dict(), f)
```

**Step 2**: Import the compiler in the engine's `__init__.py` to trigger registration:

```python
# opensentinel/policy/engines/my_engine/__init__.py
try:
    from .compiler import MyEngineCompiler
except ImportError:
    pass  # Handle missing dependencies gracefully
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py               # Shared fixtures
├── tracing/
│   └── test_otel_tracer.py   # Tracing tests
├── core/
│   ├── interceptor/
│   │   ├── test_interceptor.py   # Interceptor orchestration tests
│   │   └── test_adapters.py      # PolicyEngineChecker adapter tests
│   └── test_intervention.py      # Strategy tests
├── proxy/
│   ├── test_hooks.py         # Open SentinelCallback tests
│   └── test_middleware.py    # Session extraction tests
└── policy/
    ├── compiler/             # Compiler tests
    └── engines/
        ├── fsm/
        │   ├── test_classifier.py    # StateClassifier tests
        │   ├── test_constraints.py   # ConstraintEvaluator tests
        │   ├── test_state_machine.py # WorkflowStateMachine tests
        │   └── test_workflow_parser.py # Parser tests
        ├── llm/              # LLM engine tests
        ├── nemo/             # NeMo engine tests
        └── composite/        # Composite engine tests
```

### Key Fixtures (in `conftest.py`)

```python
@pytest.fixture
def sample_workflow():
    """Full customer support workflow from examples/"""

@pytest.fixture  
def simple_workflow():
    """Minimal 3-state workflow for basic tests"""

@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses"""
    
@pytest.fixture
def mock_tool_call():
    """Factory for creating mock tool calls"""
```

### Writing Tests

```python
import pytest
from opensentinel.policy.engines.fsm import WorkflowStateMachine, TransitionResult

class TestMyFeature:
    @pytest.fixture
    def machine(self, simple_workflow):
        return WorkflowStateMachine(simple_workflow)
    
    @pytest.mark.asyncio
    async def test_something(self, machine):
        session = await machine.get_or_create_session("test")
        result, error = await machine.transition("test", "middle")
        
        assert result == TransitionResult.SUCCESS
        assert error is None
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/policy/engines/fsm/test_state_machine.py

# Specific test
pytest tests/policy/engines/fsm/test_state_machine.py::TestWorkflowStateMachine::test_create_session

# With coverage
pytest --cov=opensentinel

# Verbose output
pytest -v
```

---

## Common Tasks

### Adding a New CLI Command

Edit `opensentinel/cli.py`:

```python
@main.command()
@click.option("--option", "-o", help="Description")
@click.argument("arg")
def mycommand(option: str, arg: str):
    """Short description.
    
    Longer description with examples.
    
    Example:
        osentinel mycommand --option value argument
    """
    # Implementation
    click.echo(f"Running with {option} and {arg}")
```

### Adding Configuration Options

Edit `opensentinel/config/settings.py`:

```python
class MyComponentConfig(BaseModel):
    """My component configuration."""
    
    option_a: str = "default"
    option_b: int = 42
    option_c: bool = False

class SentinelSettings(BaseSettings):
    # ... existing fields ...
    
    my_component: MyComponentConfig = Field(default_factory=MyComponentConfig)
```

Environment variables: `OSNTL_MY_COMPONENT__OPTION_A=value`

### Debugging

**Enable debug logging:**

```bash
OSNTL_DEBUG=true osentinel serve
```

**Or in code:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check specific loggers:**

```python
logging.getLogger("opensentinel.policy.engines.fsm.classifier").setLevel(logging.DEBUG)
logging.getLogger("opensentinel.policy.engines.llm.engine").setLevel(logging.DEBUG)
logging.getLogger("opensentinel.core.interceptor").setLevel(logging.DEBUG)
```

**Monitor fail-open activations:**

```python
from opensentinel.proxy.hooks import get_fail_open_counts
counts = get_fail_open_counts()
# {"pre_call": 0, "post_call": 1, ...}
```

---

## Troubleshooting

### Common Issues

**"No workflow configured - running in pass-through mode"**
- Set `OSNTL_POLICY__ENGINE__CONFIG_PATH` environment variable

**"Failed to load embedding model"**
- Ensure `sentence-transformers` is installed
- Check disk space (model downloads ~100MB first time)

**"Unknown intervention: ..."**
- Intervention name in constraint must match key in `interventions` dict

**"Workflow has no initial state"**
- At least one state must have `is_initial: true`

**"Constraint references unknown state"**
- Check that `trigger` and `target` in constraints match state names exactly

**"Unknown policy engine type: '...'"**
- Check `OSNTL_POLICY__ENGINE__TYPE` is one of: `fsm`, `llm`, `nemo`, `composite`
- Ensure the engine module is imported (check `opensentinel/policy/engines/__init__.py`)

### NeMo Guardrails Issues

**"API key not valid (400)"**
- Ensure `OSNTL_POLICY__ENGINE__TYPE` is set to `nemo` (default)
- Check that your model configuration maps to a valid API key in your `.env` file
- Verify `config.yml` refers to properly configured models

**"No workflow_path configured - running in pass-through mode"**
- This is a WARN log from the FSM engine. If using the `nemo` engine, you can safely ignore it.

**"TypeError: 'function' object is not subscriptable"**
- Conflict between Pydantic V1 (used by LangChain) and V2 (used by Open Sentinel). Ensure compatible versions.

### LLM Engine Issues

**LLM classification returning low confidence**
- Add more descriptive state descriptions and exemplars in the workflow
- Try a more capable model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
- Check the `turn_window` — the LLM needs conversation context

**Excessive interventions**
- Increase `cooldown_turns` in the LLM engine config
- Check `self_correction_margin` — a lower value makes self-correction detection more sensitive

**High drift scores**
- Adjust `temporal_weight` to balance temporal vs semantic drift weighting
- Ensure workflow states and transitions accurately reflect the expected agent behavior

### Interceptor Issues

**Async checker results not being applied**
- Async results are collected at the start of the *next* `run_pre_call`. Ensure the session ID is consistent across requests.
- Check `interceptor._pending_results` for accumulated results

**Checker errors not propagating**
- By default, checker errors result in a `FAIL` decision. If using `safe_hook`, only `WorkflowViolationError` propagates — all other errors result in pass-through.

### Session ID Issues

If workflows aren't being tracked properly:

1. Check if session ID is being extracted correctly (add debug logging to `middleware.py`)
2. Ensure you're sending consistent session identifiers across calls
3. Use `x-sentinel-session-id` header for explicit control

### Classification Issues (FSM Engine)

If states aren't being classified correctly:

1. Check classification hints in workflow definition
2. Add more specific patterns or tool_calls for deterministic matching
3. Add exemplars for semantic matching
4. Lower `min_similarity` threshold if embeddings are close but not matching

---

## Performance Considerations

### Classification Performance

| Method | Latency | When Used |
|--------|---------|-----------|
| Tool calls | ~0ms | When response has tool_calls |
| Patterns | ~1ms | When patterns defined and no tool match |
| Embeddings | ~50ms | When exemplars defined and no pattern match |
| LLM (LLM engine) | ~200-500ms | LLM-based classification |

**Optimization**: Define `tool_calls` and `patterns` for states to avoid embedding inference. For the LLM engine, use a fast model like `gpt-4o-mini`.

### Memory

- Embedding model: ~100MB RAM
- Compiled regex patterns: Cached per workflow
- State embeddings: Cached after first computation
- Session state: ~1KB per active session (FSM), ~5KB (LLM engine due to turn window)

### Concurrent Sessions

- State machine uses `asyncio.Lock` for session operations
- Sessions are stored in memory (not persistent)
- For high concurrency, consider external state storage

---

## API Reference Quick Links

### Public Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `SentinelSettings` | `config.settings` | Configuration |
| `SentinelProxy` | `proxy.server` | Main proxy server |
| `WorkflowDefinition` | `policy.engines.fsm` | Workflow data model |
| `State` | `policy.engines.fsm` | State definition |
| `Transition` | `policy.engines.fsm` | Transition definition |
| `Constraint` | `policy.engines.fsm` | Constraint definition |
| `ConstraintType` | `policy.engines.fsm` | Constraint type enum |
| `WorkflowParser` | `policy.engines.fsm` | Parse workflow files |
| `WorkflowStateMachine` | `policy.engines.fsm` | State management |
| `ConstraintEvaluator` | `policy.engines.fsm` | Constraint checking |
| `StateClassifier` | `policy.engines.fsm` | Response classification |
| `WorkflowTracker` | `policy.engines.fsm` | Main orchestrator |
| `PromptInjector` | `policy.engines.fsm` | Apply interventions |
| `PolicyEngine` | `policy.protocols` | Engine base class |
| `PolicyEngineRegistry` | `policy.registry` | Engine factory |
| `PolicyDecision` | `policy.protocols` | Decision enum |
| `PolicyEvaluationResult` | `policy.protocols` | Evaluation result |
| `PolicyViolation` | `policy.protocols` | Violation details |
| `PolicyCompiler` | `policy.compiler` | Compiler protocol |
| `CompilationResult` | `policy.compiler` | Compilation output |
| `PolicyCompilerRegistry` | `policy.compiler` | Compiler factory |
| `FSMCompiler` | `policy.engines.fsm.compiler` | NL → FSM workflow |
| `Interceptor` | `core.interceptor` | Checker orchestrator |
| `Checker` | `core.interceptor` | Checker base class |
| `PolicyEngineChecker` | `core.interceptor` | Engine-to-checker adapter |
| `StrategyType` | `core.intervention.strategies` | Intervention strategy enum |
| `LLMPolicyEngine` | `policy.engines.llm` | LLM-based engine |
| `NemoGuardrailsPolicyEngine` | `policy.engines.nemo` | NeMo engine |
| `CompositePolicyEngine` | `policy.engines.composite` | Multi-engine combiner |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `start_proxy()` | `proxy.server` | Start proxy (blocking) |
| `get_strategy()` | `core.intervention.strategies` | Get strategy by type |
| `safe_hook()` | `proxy.hooks` | Fail-open hook wrapper |
| `get_fail_open_counts()` | `proxy.hooks` | Monitor fail-open activations |

---

## Workflow YAML Reference

```yaml
# Required
name: workflow-name
version: "1.0"

# Optional
description: |
  Multi-line description of the workflow.

# Required: At least one state with is_initial: true
states:
  - name: state_name              # Required, alphanumeric with _ or -
    description: "State desc"     # Optional
    is_initial: true              # One state must be initial
    is_terminal: false            # Terminal states end the workflow
    is_error: false               # Error states for failure tracking
    max_duration_seconds: 60      # Optional, for temporal constraints
    
    classification:               # How to detect this state
      tool_calls:                 # Exact match on function names (priority 1)
        - function_name
      patterns:                   # Regex patterns (priority 2)
        - "regex.*pattern"
      exemplars:                  # Semantic similarity (priority 3)
        - "Example phrase"
      min_similarity: 0.7         # Threshold for embedding match

# Optional: If not specified, any transition is allowed
transitions:
  - from_state: state_a
    to_state: state_b
    description: "Optional description"
    priority: 0                   # Higher = preferred
    guard:                        # Optional conditions
      expression: "confidence > 0.8"
      required_metadata:
        key: value

# Optional: Temporal constraints
constraints:
  - name: constraint_name
    type: precedence              # eventually|always|never|precedence|response|until|next
    trigger: trigger_state        # For precedence/response/until
    target: target_state          # For most constraint types
    condition: "expression"       # For always constraints
    severity: error               # warning|error|critical
    intervention: intervention_name
    description: "Optional description"

# Optional: Intervention templates
interventions:
  intervention_name: |
    Multi-line prompt template.
    Can use {placeholders} from context.
```
