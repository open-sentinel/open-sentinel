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
from panoptes.policy.engines import fsm
from panoptes.policy.engines import nemo
from panoptes.policy.engines import composite

__all__ = ["fsm", "nemo", "composite"]
