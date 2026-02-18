"""
Interceptor module for Open Sentinel.

Provides a clean abstraction for running checkers at different phases
of the LLM request lifecycle.
"""

from .adapters import PolicyEngineChecker
from .checker import Checker
from .interceptor import Interceptor
from .types import (
    CheckerContext,
    CheckerMode,
    CheckPhase,
    CheckResult,
    InterceptionResult,
    PolicyViolation,
)

# Re-export PolicyDecision for convenience (replaces old CheckDecision)
from opensentinel.policy.protocols import PolicyDecision

__all__ = [
    # Types
    "CheckPhase",
    "CheckerMode",
    "PolicyDecision",
    "CheckResult",
    "CheckerContext",
    "InterceptionResult",
    "PolicyViolation",
    # Classes
    "Checker",
    "Interceptor",
    "PolicyEngineChecker",
]
