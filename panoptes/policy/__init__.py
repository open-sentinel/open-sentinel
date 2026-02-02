"""
Panoptes policy engine system.

This module provides a pluggable infrastructure for policy evaluation,
supporting multiple policy mechanisms including:

- FSM (Finite State Machine): Workflow enforcement using states and transitions
- NeMo Guardrails: NVIDIA's guardrails for input/output filtering
- Composite: Combine multiple engines together

Usage:
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

For composite engines:
    ```python
    engine = PolicyEngineRegistry.create("composite")
    await engine.initialize({
        "engines": [
            {"type": "fsm", "config": {"workflow_path": "..."}},
            {"type": "nemo", "config": {"config_path": "..."}}
        ]
    })
    ```
"""

from panoptes.policy.protocols import (
    PolicyEngine,
    StatefulPolicyEngine,
    InterventionProvider,
    PolicyEvaluationResult,
    PolicyDecision,
    PolicyViolation,
    StateClassificationResult,
)
from panoptes.policy.registry import PolicyEngineRegistry, register_engine

# Import engines to trigger auto-registration
# Note: We use try/except to handle optional dependencies gracefully
try:
    from panoptes.policy.engines.fsm import FSMPolicyEngine
except ImportError:
    FSMPolicyEngine = None  # type: ignore

try:
    from panoptes.policy.engines.nemo import NemoGuardrailsEngine
except ImportError:
    # NeMo is optional
    NemoGuardrailsEngine = None  # type: ignore

try:
    from panoptes.policy.engines.composite import CompositePolicyEngine
except ImportError:
    CompositePolicyEngine = None  # type: ignore

__all__ = [
    # Core protocols
    "PolicyEngine",
    "StatefulPolicyEngine",
    "InterventionProvider",
    # Result types
    "PolicyEvaluationResult",
    "PolicyDecision",
    "PolicyViolation",
    "StateClassificationResult",
    # Registry
    "PolicyEngineRegistry",
    "register_engine",
    # Engines (may be None if not available)
    "FSMPolicyEngine",
    "NemoGuardrailsEngine",
    "CompositePolicyEngine",
]
