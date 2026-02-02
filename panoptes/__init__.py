"""
Panoptes SDK - Reliability layer for AI agents.

Monitors workflow adherence and intervenes when agents deviate from expected behavior.
Uses LiteLLM proxy approach - customers only need to change their base_url.

Supports multiple policy engines:
- FSM: Finite State Machine workflow enforcement
- NeMo: NVIDIA NeMo Guardrails integration
- Composite: Combine multiple engines together

Quick Start:
    ```python
    from panoptes import PanoptesProxy, PanoptesSettings

    # Configure and start proxy with FSM workflow
    settings = PanoptesSettings(workflow_path="workflow.yaml")
    proxy = PanoptesProxy(settings)
    await proxy.start()
    ```

    Then point your LLM client to http://localhost:4000/v1

Using Policy Engines Directly:
    ```python
    from panoptes.policy import PolicyEngineRegistry, PolicyDecision

    # Create and initialize an FSM engine
    engine = PolicyEngineRegistry.create("fsm")
    await engine.initialize({"workflow_path": "./workflow.yaml"})

    # Evaluate a request
    result = await engine.evaluate_request(
        session_id="session-123",
        request_data={"messages": [...]},
    )

    if result.decision == PolicyDecision.DENY:
        print("Request blocked:", result.violations)
    ```
"""

__version__ = "0.1.0"

# Core configuration
from panoptes.config.settings import PanoptesSettings

# Proxy server
from panoptes.proxy.server import PanoptesProxy, start_proxy

# Workflow components (from FSM engine)
from panoptes.policy.engines.fsm import (
    WorkflowDefinition,
    State,
    Transition,
    Constraint,
    ConstraintType,
    WorkflowParser,
    WorkflowStateMachine,
    WorkflowTracker,
    StateClassifier,
)

# Policy engines
from panoptes.policy import (
    PolicyEngine,
    PolicyEngineRegistry,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyViolation,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "PanoptesSettings",
    # Proxy
    "PanoptesProxy",
    "start_proxy",
    # Workflow
    "WorkflowDefinition",
    "State",
    "Transition",
    "Constraint",
    "ConstraintType",
    "WorkflowParser",
    "WorkflowStateMachine",
    # Monitoring
    "WorkflowTracker",
    "StateClassifier",
    # Policy engines
    "PolicyEngine",
    "PolicyEngineRegistry",
    "PolicyDecision",
    "PolicyEvaluationResult",
    "PolicyViolation",
]
