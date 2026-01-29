"""
Workflow execution tracker.

Coordinates:
- State classification
- State machine updates
- Constraint evaluation
- Intervention triggering

This is the main orchestration layer that ties together
the classifier, state machine, and constraint evaluator.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from panoptes.workflow.schema import WorkflowDefinition
from panoptes.workflow.state_machine import WorkflowStateMachine, TransitionResult
from panoptes.workflow.constraints import ConstraintEvaluator, ConstraintViolation
from panoptes.monitor.classifier import StateClassifier, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result of processing an LLM response."""

    # Classification results
    classified_state: str
    classification_confidence: float
    classification_method: str

    # Transition results
    transition_result: TransitionResult
    previous_state: str

    # Constraint results
    constraint_violations: List[ConstraintViolation]

    # Intervention
    intervention_needed: Optional[str]


class WorkflowTracker:
    """
    Tracks workflow execution across sessions.

    Main entry point for monitoring LLM responses. Coordinates:
    1. State classification from LLM output
    2. State machine transition
    3. Constraint evaluation
    4. Intervention triggering

    Example:
        ```python
        from panoptes.monitor import WorkflowTracker
        from panoptes.workflow import WorkflowParser

        workflow = WorkflowParser.parse_file("workflow.yaml")
        tracker = WorkflowTracker(workflow)

        # Process an LLM response
        result = await tracker.process_response(
            session_id="session-123",
            response=llm_response,
        )

        if result.intervention_needed:
            print(f"Intervention required: {result.intervention_needed}")
        ```
    """

    def __init__(self, workflow: WorkflowDefinition):
        self.workflow = workflow
        self.state_machine = WorkflowStateMachine(workflow)
        self.classifier = StateClassifier(workflow.states)
        self.constraint_evaluator = ConstraintEvaluator(workflow.constraints)

        logger.info(f"WorkflowTracker initialized for workflow '{workflow.name}'")

    async def process_response(
        self,
        session_id: str,
        response: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> TrackingResult:
        """
        Process an LLM response and update tracking.

        This is the main entry point called by the hooks.

        Steps:
        1. Get current session state
        2. Classify the response to determine new state
        3. Check constraints BEFORE transition
        4. Attempt state transition
        5. Determine if intervention needed

        Args:
            session_id: Session identifier
            response: LLM response (dict or object)
            context: Additional context (messages, request data, etc.)

        Returns:
            TrackingResult with classification, transition, and intervention info
        """
        # Get current session state
        session = await self.state_machine.get_or_create_session(session_id)
        previous_state = session.current_state

        # Classify the response
        classification = self.classifier.classify(response, previous_state)

        logger.debug(
            f"Session {session_id}: Classified as '{classification.state_name}' "
            f"(confidence={classification.confidence:.2f}, method={classification.method})"
        )

        # Check constraints BEFORE transition
        violations = self.constraint_evaluator.evaluate_all(
            session,
            proposed_state=classification.state_name,
        )

        # Attempt transition
        transition_result, error = await self.state_machine.transition(
            session_id,
            classification.state_name,
            context,
            confidence=classification.confidence,
            method=classification.method,
        )

        if error:
            logger.warning(
                f"Session {session_id}: Transition failed: {error}"
            )

        # Determine intervention
        intervention = None
        if violations:
            # Use first violation's intervention (prioritized by order in workflow)
            for v in violations:
                if v.intervention:
                    intervention = v.intervention
                    await self.state_machine.set_pending_intervention(
                        session_id,
                        intervention,
                    )
                    logger.info(
                        f"Session {session_id}: Intervention scheduled: {intervention}"
                    )
                    break

        return TrackingResult(
            classified_state=classification.state_name,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            transition_result=transition_result,
            previous_state=previous_state,
            constraint_violations=violations,
            intervention_needed=intervention,
        )

    async def get_current_state(self, session_id: str) -> str:
        """Get current state for a session."""
        session = await self.state_machine.get_or_create_session(session_id)
        return session.current_state

    async def get_pending_intervention(self, session_id: str) -> Optional[str]:
        """Get pending intervention for a session."""
        return await self.state_machine.get_pending_intervention(session_id)

    async def get_state_history(self, session_id: str) -> List[str]:
        """Get state history for a session."""
        return await self.state_machine.get_state_history(session_id)

    async def is_session_complete(self, session_id: str) -> bool:
        """Check if session has reached a terminal state."""
        return await self.state_machine.is_in_terminal_state(session_id)

    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """Get list of valid next states from current state."""
        valid = await self.state_machine.get_valid_transitions(session_id)
        return list(valid)

    async def force_state(
        self,
        session_id: str,
        state_name: str,
        reason: str = "manual override",
    ) -> TransitionResult:
        """
        Force a session into a specific state (bypass normal transition rules).

        Use with caution - this is for recovery scenarios.
        """
        result, _ = await self.state_machine.transition(
            session_id,
            state_name,
            context={"forced": True, "reason": reason},
            confidence=1.0,
            method="forced",
        )
        logger.warning(
            f"Session {session_id}: Forced to state '{state_name}' ({reason})"
        )
        return result

    async def reset_session(self, session_id: str) -> None:
        """Reset a session to initial state."""
        await self.state_machine.reset_session(session_id)
        logger.info(f"Session {session_id}: Reset to initial state")

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow."""
        return {
            "name": self.workflow.name,
            "version": self.workflow.version,
            "states": [s.name for s in self.workflow.states],
            "initial_states": [s.name for s in self.workflow.get_initial_states()],
            "terminal_states": [s.name for s in self.workflow.get_terminal_states()],
            "constraints": [c.name for c in self.workflow.constraints],
            "interventions": list(self.workflow.interventions.keys()),
        }
