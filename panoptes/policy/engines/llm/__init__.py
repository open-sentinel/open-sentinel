"""
LLM Policy Engine package.

Provides an LLM-based policy engine that uses a lightweight LLM
(e.g. gpt-4o-mini) for state classification, drift detection,
and soft constraint evaluation.

Usage:
    from panoptes.policy.engines.llm import LLMPolicyEngine
    
    engine = LLMPolicyEngine()
    await engine.initialize({
        "config_path": "workflow.yaml",
        "llm_model": "gpt-4o-mini",
    })
"""

# Import engine to trigger @register_engine("llm")
from panoptes.policy.engines.llm.engine import LLMPolicyEngine

# Re-export model types
from panoptes.policy.engines.llm.models import (
    ConfidenceTier,
    DriftLevel,
    LLMStateCandidate,
    LLMClassificationResult,
    StateTransition,
    DriftScores,
    ConstraintEvaluation,
    SessionContext,
)

# Re-export components
from panoptes.policy.engines.llm.llm_client import LLMClient, LLMClientError
from panoptes.policy.engines.llm.state_classifier import LLMStateClassifier
from panoptes.policy.engines.llm.drift_detector import DriftDetector
from panoptes.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from panoptes.policy.engines.llm.intervention import InterventionDecisionEngine

__all__ = [
    # Main engine
    "LLMPolicyEngine",
    # Model types
    "ConfidenceTier",
    "DriftLevel",
    "LLMStateCandidate",
    "LLMClassificationResult",
    "StateTransition",
    "DriftScores",
    "ConstraintEvaluation",
    "SessionContext",
    # Components
    "LLMClient",
    "LLMClientError",
    "LLMStateClassifier",
    "DriftDetector",
    "LLMConstraintEvaluator",
    "InterventionDecisionEngine",
]
