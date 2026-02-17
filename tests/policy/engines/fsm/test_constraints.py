"""Tests for constraint evaluation."""

import pytest
from datetime import datetime, timezone

from opensentinel.policy.engines.fsm.workflow.constraints import (
    ConstraintEvaluator,
    EvaluationResult,
    ConstraintViolation,
)
from opensentinel.policy.engines.fsm.workflow.schema import Constraint, ConstraintType
from opensentinel.policy.engines.fsm.workflow.state_machine import SessionState, StateHistoryEntry


def make_session(states: list[str]) -> SessionState:
    """Helper to create a session with given state history."""
    history = [
        StateHistoryEntry(state_name=s, entered_at=datetime.now(timezone.utc)) for s in states
    ]
    return SessionState(
        session_id="test",
        workflow_name="test",
        current_state=states[-1] if states else "unknown",
        history=history,
    )


class TestConstraintEvaluator:
    """Tests for ConstraintEvaluator."""

    def test_eventually_satisfied(self):
        """Test EVENTUALLY constraint when target is reached."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.EVENTUALLY,
                target="goal",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "goal"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_eventually_pending(self):
        """Test EVENTUALLY constraint when target not yet reached."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.EVENTUALLY,
                target="goal",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle"])

        violations = evaluator.evaluate_all(session)

        # EVENTUALLY is PENDING, not VIOLATED, when target not reached
        assert len(violations) == 0

    def test_never_satisfied(self):
        """Test NEVER constraint when forbidden state not reached."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_never_violated(self):
        """Test NEVER constraint when forbidden state is reached."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
                intervention="fix_it",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "forbidden", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 1
        assert violations[0].constraint_name == "test"
        assert violations[0].intervention == "fix_it"

    def test_precedence_satisfied(self):
        """Test PRECEDENCE constraint when order is correct."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        # verify comes before action
        session = make_session(["start", "verify", "action", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_precedence_violated(self):
        """Test PRECEDENCE constraint when order is wrong."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
                intervention="must_verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        # action comes before verify - violation!
        session = make_session(["start", "action", "verify", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 1
        assert violations[0].constraint_name == "test"

    def test_precedence_with_proposed_state(self):
        """Test PRECEDENCE with proposed transition."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
                intervention="must_verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        # Proposing to go to action without verify
        violations = evaluator.evaluate_all(session, proposed_state="action")

        assert len(violations) == 1

    def test_response_satisfied(self):
        """Test RESPONSE constraint when response follows trigger."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.RESPONSE,
                trigger="request",
                target="acknowledge",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "request", "acknowledge", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_response_no_trigger(self):
        """Test RESPONSE constraint when trigger never occurs."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.RESPONSE,
                trigger="request",
                target="acknowledge",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        violations = evaluator.evaluate_all(session)

        # Vacuously satisfied - trigger never occurred
        assert len(violations) == 0

    def test_until_satisfied(self):
        """Test UNTIL constraint when satisfied."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.UNTIL,
                trigger="waiting",
                target="done",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["waiting", "waiting", "done"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_until_violated(self):
        """Test UNTIL constraint when violated."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.UNTIL,
                trigger="waiting",
                target="done",
                intervention="keep_waiting",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        # Interrupted by "other" before "done"
        session = make_session(["waiting", "other", "done"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 1

    def test_next_satisfied(self):
        """Test NEXT constraint when satisfied."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEXT,
                target="second",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["first", "second", "third"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_next_violated(self):
        """Test NEXT constraint when violated."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEXT,
                target="second",
                intervention="wrong_order",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["first", "wrong", "second"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 1

    def test_multiple_constraints(self):
        """Test evaluating multiple constraints."""
        constraints = [
            Constraint(
                name="never_bad",
                type=ConstraintType.NEVER,
                target="bad",
            ),
            Constraint(
                name="verify_first",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            ),
        ]
        evaluator = ConstraintEvaluator(constraints)
        # Both constraints satisfied
        session = make_session(["start", "verify", "action", "end"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 0

    def test_violation_details(self):
        """Test that violations include useful details."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
                severity="critical",
                intervention="fix_it",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "forbidden"])

        violations = evaluator.evaluate_all(session)

        assert len(violations) == 1
        v = violations[0]
        assert v.constraint_name == "test"
        assert v.constraint_type == ConstraintType.NEVER
        assert v.severity == "critical"
        assert v.intervention == "fix_it"
        assert "forbidden" in v.message
