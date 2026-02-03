"""
Interceptor module for Panoptes.

Provides a clean abstraction for running checkers at different phases
of the LLM request lifecycle.
"""

from .adapters import PolicyEngineChecker
from .checker import Checker
from .interceptor import Interceptor
from .types import (
    CheckDecision,
    CheckerContext,
    CheckerMode,
    CheckPhase,
    CheckResult,
    InterceptionResult,
)

__all__ = [
    # Types
    "CheckPhase",
    "CheckerMode",
    "CheckDecision",
    "CheckResult",
    "CheckerContext",
    "InterceptionResult",
    # Classes
    "Checker",
    "Interceptor",
    "PolicyEngineChecker",
]
