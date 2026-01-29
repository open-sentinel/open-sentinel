"""
LTL-lite constraint evaluator.

Implements runtime verification for temporal constraints:
- EVENTUALLY (F): Must reach target state
- ALWAYS (G): Condition must always hold
- NEVER (G!): Condition must never hold
- UNTIL (U): Stay in state until condition
- NEXT (X): Immediate next state requirement
- RESPONSE: If trigger, then eventually response
- PRECEDENCE: Target before trigger

Based on runtime verification semantics where constraints can be:
- SATISFIED: Constraint is met
- VIOLATED: Constraint is broken
- PENDING: Cannot yet determine (waiting for more states)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from panoptes.workflow.schema import Constraint, ConstraintType
from panoptes.workflow.state_machine import SessionState

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """Constraint evaluation outcome."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    PENDING = "pending"  # Not yet determinable


@dataclass
class ConstraintViolation:
    """Details of a constraint violation."""

    constraint_name: str
    constraint_type: ConstraintType
    severity: str
    message: str
    intervention: Optional[str]
    details: dict


class ConstraintEvaluator:
    """
    Evaluates LTL-lite constraints against workflow execution.

    This implements runtime verification where we evaluate constraints
    as the workflow progresses, rather than model-checking all possible
    paths upfront.

    Example:
        ```python
        from panoptes.workflow import ConstraintEvaluator

        evaluator = ConstraintEvaluator(workflow.constraints)

        # Check constraints for a session
        violations = evaluator.evaluate_all(session_state)

        # Check what would happen if we transition
        violations = evaluator.evaluate_all(session_state, proposed_state="resolution")
        ```
    """

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        logger.debug(f"ConstraintEvaluator initialized with {len(constraints)} constraints")

    def evaluate_all(
        self,
        session: SessionState,
        proposed_state: Optional[str] = None,
    ) -> List[ConstraintViolation]:
        """
        Evaluate all constraints for a session.

        Args:
            session: Current session state
            proposed_state: If given, evaluates as if that transition happened

        Returns:
            List of constraint violations (empty if all satisfied)
        """
        violations = []
        history = session.get_state_sequence()

        if proposed_state:
            history = history + [proposed_state]

        for constraint in self.constraints:
            result = self._evaluate_constraint(constraint, history, session)

            if result == EvaluationResult.VIOLATED:
                violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.type,
                    severity=constraint.severity,
                    message=self._format_violation_message(constraint, history),
                    intervention=constraint.intervention,
                    details={
                        "history": history[-5:],  # Last 5 states
                        "current_state": session.current_state,
                        "proposed_state": proposed_state,
                    },
                )
                violations.append(violation)
                logger.info(
                    f"Constraint violated: {constraint.name} ({constraint.type.value})"
                )

        return violations

    def evaluate_transition(
        self,
        session: SessionState,
        from_state: str,
        to_state: str,
    ) -> List[ConstraintViolation]:
        """
        Evaluate constraints for a specific transition.

        More efficient than evaluate_all when only checking one transition.
        """
        return self.evaluate_all(session, proposed_state=to_state)

    def _evaluate_constraint(
        self,
        constraint: Constraint,
        history: List[str],
        session: SessionState,
    ) -> EvaluationResult:
        """Evaluate a single constraint."""
        match constraint.type:
            case ConstraintType.EVENTUALLY:
                return self._eval_eventually(constraint.target, history)

            case ConstraintType.ALWAYS:
                return self._eval_always(constraint.condition, history, session)

            case ConstraintType.NEVER:
                return self._eval_never(constraint.target, history)

            case ConstraintType.UNTIL:
                return self._eval_until(constraint.trigger, constraint.target, history)

            case ConstraintType.NEXT:
                return self._eval_next(constraint.target, history)

            case ConstraintType.RESPONSE:
                return self._eval_response(constraint.trigger, constraint.target, history)

            case ConstraintType.PRECEDENCE:
                return self._eval_precedence(constraint.trigger, constraint.target, history)

        return EvaluationResult.PENDING

    def _eval_eventually(
        self, target: Optional[str], history: List[str]
    ) -> EvaluationResult:
        """
        F(target): Must eventually reach target state.

        This can only be VIOLATED when we know we can't reach target
        (e.g., session ended). Otherwise it's PENDING or SATISFIED.
        """
        if not target:
            return EvaluationResult.PENDING

        if target in history:
            return EvaluationResult.SATISFIED

        # Can't definitively say violated until workflow ends
        return EvaluationResult.PENDING

    def _eval_always(
        self,
        condition: Optional[str],
        history: List[str],
        session: SessionState,
    ) -> EvaluationResult:
        """
        G(condition): Condition must hold in all states.

        For simplicity, condition can be:
        - A state name that must always be present (matches if current)
        - "*" (always true)
        - "!state_name" (state must never occur)
        """
        if not condition:
            return EvaluationResult.PENDING

        if condition == "*":
            return EvaluationResult.SATISFIED

        # Handle negation
        if condition.startswith("!"):
            forbidden = condition[1:]
            if forbidden in history:
                return EvaluationResult.VIOLATED
            return EvaluationResult.SATISFIED

        # For positive condition, check if it's always true
        # This is a simplification - real G(p) would need p to hold at every step
        # Here we just check if the condition state has been reached
        if condition not in history and session.current_state != condition:
            return EvaluationResult.PENDING

        return EvaluationResult.SATISFIED

    def _eval_never(
        self, target: Optional[str], history: List[str]
    ) -> EvaluationResult:
        """
        G(!target): Target state must never occur.
        """
        if not target:
            return EvaluationResult.PENDING

        if target in history:
            return EvaluationResult.VIOLATED

        return EvaluationResult.SATISFIED

    def _eval_until(
        self,
        trigger: Optional[str],
        target: Optional[str],
        history: List[str],
    ) -> EvaluationResult:
        """
        trigger U target: Stay in trigger until target reached.

        All states before target must be trigger.
        """
        if not trigger or not target:
            return EvaluationResult.PENDING

        target_idx = None
        for i, state in enumerate(history):
            if state == target:
                target_idx = i
                break
            # Before target, must be in trigger state
            if state != trigger:
                return EvaluationResult.VIOLATED

        if target_idx is not None:
            return EvaluationResult.SATISFIED

        return EvaluationResult.PENDING

    def _eval_next(
        self, target: Optional[str], history: List[str]
    ) -> EvaluationResult:
        """
        X(target): Next state must be target.

        Only evaluates after at least one transition.
        """
        if not target:
            return EvaluationResult.PENDING

        if len(history) < 2:
            return EvaluationResult.PENDING

        # Check if second state is target
        if history[1] == target:
            return EvaluationResult.SATISFIED

        return EvaluationResult.VIOLATED

    def _eval_response(
        self,
        trigger: Optional[str],
        target: Optional[str],
        history: List[str],
    ) -> EvaluationResult:
        """
        G(trigger -> F(target)): If trigger occurs, target must eventually follow.

        This is the "response" pattern - whenever we see trigger,
        we must eventually see target.
        """
        if not trigger or not target:
            return EvaluationResult.PENDING

        # Find all occurrences of trigger
        trigger_indices = [i for i, s in enumerate(history) if s == trigger]

        if not trigger_indices:
            # Trigger never occurred - constraint vacuously satisfied
            return EvaluationResult.SATISFIED

        # For each trigger, check if target eventually follows
        for trigger_idx in trigger_indices:
            # Look for target after this trigger
            found_target = False
            for j in range(trigger_idx + 1, len(history)):
                if history[j] == target:
                    found_target = True
                    break

            if not found_target:
                # Haven't seen target yet after this trigger
                # Could still happen - PENDING, not VIOLATED
                return EvaluationResult.PENDING

        return EvaluationResult.SATISFIED

    def _eval_precedence(
        self,
        trigger: Optional[str],
        target: Optional[str],
        history: List[str],
    ) -> EvaluationResult:
        """
        !target U trigger OR G(!target): Target cannot occur before trigger.

        This is the "precedence" pattern - target requires trigger first.

        Example: "identity_verified must precede account_action"
        means account_action cannot happen before identity_verified.
        """
        if not trigger or not target:
            return EvaluationResult.PENDING

        trigger_seen = False
        for state in history:
            if state == trigger:
                trigger_seen = True
            if state == target and not trigger_seen:
                # Target occurred before trigger - VIOLATED
                return EvaluationResult.VIOLATED

        return EvaluationResult.SATISFIED

    def _format_violation_message(
        self,
        constraint: Constraint,
        history: List[str],
    ) -> str:
        """Format a human-readable violation message."""
        recent_history = " -> ".join(history[-5:]) if history else "(empty)"

        match constraint.type:
            case ConstraintType.EVENTUALLY:
                return (
                    f"Constraint '{constraint.name}': Must eventually reach "
                    f"'{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.ALWAYS:
                return (
                    f"Constraint '{constraint.name}': Condition '{constraint.condition}' "
                    f"must always hold. History: {recent_history}"
                )

            case ConstraintType.NEVER:
                return (
                    f"Constraint '{constraint.name}': State '{constraint.target}' "
                    f"must never occur. History: {recent_history}"
                )

            case ConstraintType.UNTIL:
                return (
                    f"Constraint '{constraint.name}': Must stay in '{constraint.trigger}' "
                    f"until '{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.NEXT:
                return (
                    f"Constraint '{constraint.name}': Next state must be "
                    f"'{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.RESPONSE:
                return (
                    f"Constraint '{constraint.name}': After '{constraint.trigger}', "
                    f"must eventually reach '{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.PRECEDENCE:
                return (
                    f"Constraint '{constraint.name}': '{constraint.target}' cannot occur "
                    f"before '{constraint.trigger}'. History: {recent_history}"
                )

        return f"Constraint '{constraint.name}' violated. History: {recent_history}"
