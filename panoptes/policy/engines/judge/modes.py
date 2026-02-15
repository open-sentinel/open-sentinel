"""
Reliability modes for the Judge Policy Engine.

These presets provide "one-switch" configurations that trade off strictness
and operational overhead. They are intentionally separate from core engine
logic and only map to existing judge config fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ReliabilityMode:
    """Named reliability preset with a config payload."""

    name: str
    config: Dict[str, Any]


_MODES: Dict[str, ReliabilityMode] = {
    "safe": ReliabilityMode(
        name="safe",
        config={
            # Stricter thresholds -> more WARN/INTERVENE/BLOCK
            "warn_threshold": 0.7,
            "block_threshold": 0.45,
            # Prefer safety checks up front
            "pre_call_enabled": True,
            "pre_call_rubric": "safety",
            # Evaluate conversation more frequently
            "conversation_eval_interval": 2,
            # Prefer ensemble with conservative aggregation when available
            "ensemble_enabled": True,
            "aggregation_strategy": "conservative",
            "min_agreement": 0.75,
            # Flag more low-confidence verdicts
            "confidence_threshold": 0.6,
            # Keep default rubrics unless overridden
            "default_rubric": "agent_behavior",
            "conversation_rubric": "conversation_policy",
        },
    ),
    "balanced": ReliabilityMode(
        name="balanced",
        config={
            "warn_threshold": 0.5,
            "block_threshold": 0.3,
            "pre_call_enabled": False,
            "conversation_eval_interval": 5,
            "ensemble_enabled": False,
            "confidence_threshold": 0.5,
            "default_rubric": "agent_behavior",
            "conversation_rubric": "conversation_policy",
        },
    ),
    "aggressive": ReliabilityMode(
        name="aggressive",
        config={
            # Looser thresholds -> fewer interventions
            "warn_threshold": 0.35,
            "block_threshold": 0.15,
            "pre_call_enabled": False,
            "conversation_eval_interval": 8,
            "ensemble_enabled": False,
            "confidence_threshold": 0.45,
            "default_rubric": "agent_behavior",
            "conversation_rubric": "conversation_policy",
        },
    ),
}


def list_reliability_modes() -> Dict[str, ReliabilityMode]:
    """Return available reliability modes."""
    return dict(_MODES)


def get_reliability_mode(mode: str) -> ReliabilityMode:
    """Get a reliability mode by name."""
    key = mode.strip().lower()
    if key not in _MODES:
        raise ValueError(
            f"Unknown reliability mode: '{mode}'. "
            f"Expected one of: {sorted(_MODES.keys())}"
        )
    return _MODES[key]


def build_mode_config(
    mode: str,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a judge config by applying a reliability mode.

    Args:
        mode: One of "safe", "balanced", "aggressive".
        base_config: Optional user config. Any keys here override mode defaults.

    Returns:
        Merged config dict ready for JudgePolicyEngine.initialize().
    """
    selected = get_reliability_mode(mode)
    merged = dict(selected.config)
    if base_config:
        merged.update(base_config)
    return merged

