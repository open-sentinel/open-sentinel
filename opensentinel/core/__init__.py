"""
Open Sentinel core module.

Contains shared abstractions used across all policy engines.
"""

from opensentinel.core.intervention import (
    InterventionStrategy,
    StrategyType,
    InterventionConfig,
    WorkflowViolationError,
    get_strategy,
    STRATEGY_REGISTRY,
)

__all__ = [
    "InterventionStrategy",
    "StrategyType",
    "InterventionConfig",
    "WorkflowViolationError",
    "get_strategy",
    "STRATEGY_REGISTRY",
]
