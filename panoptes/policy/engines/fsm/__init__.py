"""
Finite State Machine policy engine.

Wraps Panoptes' existing workflow/state machine implementation
as a PolicyEngine for use with the pluggable policy infrastructure.
"""

from panoptes.policy.engines.fsm.engine import FSMPolicyEngine

__all__ = ["FSMPolicyEngine"]
