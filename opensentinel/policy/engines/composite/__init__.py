"""
Composite policy engine.

Combines multiple policy engines to run in parallel,
merging their results with configurable strategies.
"""

from opensentinel.policy.engines.composite.engine import CompositePolicyEngine

__all__ = ["CompositePolicyEngine"]
