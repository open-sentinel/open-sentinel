"""
Policy engine implementations.

This package contains all available policy engine implementations:

- fsm: Finite State Machine based workflow enforcement
- nemo: NVIDIA NeMo Guardrails integration
- composite: Combine multiple engines together

Engines are auto-registered when imported.
"""

# Import engines to trigger registration
# These imports populate the PolicyEngineRegistry
from opensentinel.policy.engines import fsm
from opensentinel.policy.engines import nemo
from opensentinel.policy.engines import composite
from opensentinel.policy.engines import llm

__all__ = ["fsm", "nemo", "composite", "llm"]

