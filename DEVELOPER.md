# Developer Documentation

This document provides guidance for developers (humans and AI agents) working on the Panoptes codebase.

## Quick Start

### Setup

```bash
# Clone and install
cd argus
pip install -e ".[dev]"

# Run tests
pytest

# Start proxy with a workflow
panoptes serve --workflow examples/customer_support.yaml --port 4000
```

### Point Your LLM Client at Panoptes

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",  # Point to Panoptes
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

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
| Classes | `PascalCase` | `WorkflowTracker` |
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
from panoptes.workflow.schema import WorkflowDefinition
from panoptes.config.settings import PanoptesSettings
```

---

## Module Guide

### Working with Workflows

#### Creating a Workflow Programmatically

```python
from panoptes.workflow.schema import (
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
from panoptes.workflow.parser import WorkflowParser

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

### Working with the State Machine

```python
from panoptes.workflow.state_machine import WorkflowStateMachine

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

### Working with the Classifier

```python
from panoptes.monitor.classifier import StateClassifier

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

### Working with Constraints

```python
from panoptes.workflow.constraints import ConstraintEvaluator

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

### Working with Interventions

```python
from panoptes.intervention.prompt_injector import PromptInjector

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

### Working with the Tracker

```python
from panoptes.monitor.tracker import WorkflowTracker

tracker = WorkflowTracker(workflow)

# Process an LLM response
result = await tracker.process_response(
    session_id="session-123",
    response=llm_response,
    context={"messages": messages}
)

if result.intervention_needed:
    print(f"Intervention: {result.intervention_needed}")
    
# Get workflow info
info = tracker.get_workflow_info()
```

---

## Adding New Features

### Adding a New Constraint Type

**Step 1**: Add to enum in `panoptes/workflow/schema.py`:

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

**Step 3**: Implement evaluation in `panoptes/workflow/constraints.py`:

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

**Step 1**: Add to enum in `panoptes/intervention/strategies.py`:

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

### Adding a New Classification Method

Modify `StateClassifier` in `panoptes/monitor/classifier.py`:

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

---

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_classifier.py    # StateClassifier tests
├── test_constraints.py   # ConstraintEvaluator tests
├── test_intervention.py  # Intervention tests
├── test_state_machine.py # WorkflowStateMachine tests
└── test_workflow_parser.py # Parser tests
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
from panoptes.workflow.state_machine import WorkflowStateMachine, TransitionResult

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
pytest tests/test_state_machine.py

# Specific test
pytest tests/test_state_machine.py::TestWorkflowStateMachine::test_create_session

# With coverage
pytest --cov=panoptes

# Verbose output
pytest -v
```

---

## Common Tasks

### Adding a New CLI Command

Edit `panoptes/cli.py`:

```python
@main.command()
@click.option("--option", "-o", help="Description")
@click.argument("arg")
def mycommand(option: str, arg: str):
    """Short description.
    
    Longer description with examples.
    
    Example:
        panoptes mycommand --option value argument
    """
    # Implementation
    click.echo(f"Running with {option} and {arg}")
```

### Adding Configuration Options

Edit `panoptes/config/settings.py`:

```python
class MyComponentConfig(BaseModel):
    """My component configuration."""
    
    option_a: str = "default"
    option_b: int = 42
    option_c: bool = False

class PanoptesSettings(BaseSettings):
    # ... existing fields ...
    
    my_component: MyComponentConfig = Field(default_factory=MyComponentConfig)
```

Environment variables: `PANOPTES_MY_COMPONENT__OPTION_A=value`

### Debugging

**Enable debug logging:**

```bash
PANOPTES_DEBUG=true panoptes serve --workflow workflow.yaml
```

**Or in code:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check specific loggers:**

```python
logging.getLogger("panoptes.monitor.classifier").setLevel(logging.DEBUG)
logging.getLogger("panoptes.workflow.constraints").setLevel(logging.DEBUG)
```

---

## Troubleshooting

### Common Issues

**"No workflow configured - running in pass-through mode"**
- Set `PANOPTES_WORKFLOW_PATH` or pass `--workflow` to CLI

**"Failed to load embedding model"**
- Ensure `sentence-transformers` is installed
- Check disk space (model downloads ~100MB first time)

**"Unknown intervention: ..."**
- Intervention name in constraint must match key in `interventions` dict

**"Workflow has no initial state"**
- At least one state must have `is_initial: true`

**"Constraint references unknown state"**
- Check that `trigger` and `target` in constraints match state names exactly

### Session ID Issues

If workflows aren't being tracked properly:

1. Check if session ID is being extracted correctly (add debug logging to `middleware.py`)
2. Ensure you're sending consistent session identifiers across calls
3. Use `x-panoptes-session-id` header for explicit control

### Classification Issues

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

**Optimization**: Define `tool_calls` and `patterns` for states to avoid embedding inference.

### Memory

- Embedding model: ~100MB RAM
- Compiled regex patterns: Cached per workflow
- State embeddings: Cached after first computation
- Session state: ~1KB per active session

### Concurrent Sessions

- State machine uses `asyncio.Lock` for session operations
- Sessions are stored in memory (not persistent)
- For high concurrency, consider external state storage

---

## API Reference Quick Links

### Public Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `PanoptesSettings` | `config.settings` | Configuration |
| `PanoptesProxy` | `proxy.server` | Main proxy server |
| `WorkflowDefinition` | `workflow.schema` | Workflow data model |
| `State` | `workflow.schema` | State definition |
| `Transition` | `workflow.schema` | Transition definition |
| `Constraint` | `workflow.schema` | Constraint definition |
| `ConstraintType` | `workflow.schema` | Constraint type enum |
| `WorkflowParser` | `workflow.parser` | Parse workflow files |
| `WorkflowStateMachine` | `workflow.state_machine` | State management |
| `ConstraintEvaluator` | `workflow.constraints` | Constraint checking |
| `StateClassifier` | `monitor.classifier` | Response classification |
| `WorkflowTracker` | `monitor.tracker` | Main orchestrator |
| `PromptInjector` | `intervention.prompt_injector` | Apply interventions |
| `StrategyType` | `intervention.strategies` | Intervention strategy enum |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `start_proxy()` | `proxy.server` | Start proxy (blocking) |
| `get_strategy()` | `intervention.strategies` | Get strategy by type |

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
