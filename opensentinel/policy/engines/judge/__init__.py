"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rubrics using LLM judges.

Usage:
    The engine is auto-registered as "judge" when this package is imported.
    Use PolicyEngineRegistry.create("judge") to instantiate.
"""

# Import engine to trigger @register_engine("judge")
from opensentinel.policy.engines.judge.engine import JudgePolicyEngine

# Re-export model types
from opensentinel.policy.engines.judge.models import (
    EvaluationType,
    EvaluationScope,
    ScoreScale,
    VerdictAction,
    RubricCriterion,
    Rubric,
    JudgeScore,
    JudgeVerdict,
    EnsembleVerdict,
    EvaluationRequest,
    JudgeSessionContext,
)

# Re-export components
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.rubrics import RubricRegistry
from opensentinel.policy.engines.judge.bias import randomize_positions
from opensentinel.policy.engines.judge.ensemble import JudgeEnsemble, AggregationStrategy
from opensentinel.policy.engines.judge.modes import (
    ReliabilityMode,
    list_reliability_modes,
    get_reliability_mode,
    build_mode_config,
)

__all__ = [
    "JudgePolicyEngine",
    "EvaluationType",
    "EvaluationScope",
    "ScoreScale",
    "VerdictAction",
    "RubricCriterion",
    "Rubric",
    "JudgeScore",
    "JudgeVerdict",
    "EnsembleVerdict",
    "EvaluationRequest",
    "JudgeSessionContext",
    "JudgeClient",
    "JudgeEvaluator",
    "JudgeEnsemble",
    "AggregationStrategy",
    "RubricRegistry",
    "randomize_positions",
    "ReliabilityMode",
    "list_reliability_modes",
    "get_reliability_mode",
    "build_mode_config",
]
