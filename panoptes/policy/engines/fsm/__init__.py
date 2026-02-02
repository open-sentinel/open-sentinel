"""
Finite State Machine policy engine.

Wraps Panoptes' existing workflow/state machine implementation
as a PolicyEngine for use with the pluggable policy infrastructure.

This module now contains all FSM-specific components:
- Workflow: Schema, Parser, StateMachine, Constraints
- Monitor: StateClassifier, WorkflowTracker
- Intervention: PromptInjector, InterventionBuilder
"""

from panoptes.policy.engines.fsm.engine import FSMPolicyEngine

# Workflow components
from panoptes.policy.engines.fsm.workflow import (
    # Schema
    ClassificationHint,
    State,
    TransitionGuard,
    Transition,
    ConstraintType,
    Constraint,
    WorkflowDefinition,
    # Parser
    WorkflowParser,
    WorkflowRegistry,
    # State machine
    TransitionResult,
    StateHistoryEntry,
    SessionState,
    WorkflowStateMachine,
    # Constraints
    EvaluationResult,
    ConstraintViolation,
    ConstraintEvaluator,
)

# Monitor components
from panoptes.policy.engines.fsm.classifier import (
    ClassificationResult,
    StateClassifier,
)
from panoptes.policy.engines.fsm.tracker import (
    TrackingResult,
    WorkflowTracker,
)

# Intervention components
from panoptes.policy.engines.fsm.injector import (
    PromptInjector,
    InterventionBuilder,
)

__all__ = [
    # Engine
    "FSMPolicyEngine",
    # Workflow - Schema
    "ClassificationHint",
    "State",
    "TransitionGuard",
    "Transition",
    "ConstraintType",
    "Constraint",
    "WorkflowDefinition",
    # Workflow - Parser
    "WorkflowParser",
    "WorkflowRegistry",
    # Workflow - State machine
    "TransitionResult",
    "StateHistoryEntry",
    "SessionState",
    "WorkflowStateMachine",
    # Workflow - Constraints
    "EvaluationResult",
    "ConstraintViolation",
    "ConstraintEvaluator",
    # Monitor
    "ClassificationResult",
    "StateClassifier",
    "TrackingResult",
    "WorkflowTracker",
    # Intervention
    "PromptInjector",
    "InterventionBuilder",
]

