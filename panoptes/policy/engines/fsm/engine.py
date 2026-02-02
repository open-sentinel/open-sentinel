"""
Finite State Machine policy engine implementation.

Wraps the existing workflow/state machine implementation as a PolicyEngine,
enabling it to be used alongside other policy mechanisms.
"""

from typing import Optional, Dict, Any, List
import logging

from panoptes.policy.protocols import (
    StatefulPolicyEngine,
    PolicyEvaluationResult,
    PolicyDecision,
    PolicyViolation,
    StateClassificationResult,
)
from panoptes.policy.registry import register_engine
from panoptes.policy.engines.fsm.workflow.schema import WorkflowDefinition
from panoptes.policy.engines.fsm.workflow.parser import WorkflowParser
from panoptes.policy.engines.fsm.workflow.state_machine import WorkflowStateMachine, TransitionResult
from panoptes.policy.engines.fsm.workflow.constraints import ConstraintEvaluator
from panoptes.policy.engines.fsm.classifier import StateClassifier

logger = logging.getLogger(__name__)


@register_engine("fsm")
class FSMPolicyEngine(StatefulPolicyEngine):
    """
    Finite State Machine based policy engine.

    Uses workflow definitions with states, transitions, and LTL-lite
    constraints to enforce policies. This wraps the existing Panoptes
    workflow implementation as a PolicyEngine.

    Configuration:
        - workflow_path: str - Path to workflow YAML/JSON file
        OR
        - workflow: dict - Workflow definition as dictionary

    Example:
        ```python
        engine = FSMPolicyEngine()
        await engine.initialize({
            "workflow_path": "./examples/customer_support.yaml"
        })

        result = await engine.evaluate_response(
            session_id="session-123",
            response_data=llm_response,
            request_data=original_request,
        )
        ```
    """

    def __init__(self):
        self._workflow: Optional[WorkflowDefinition] = None
        self._state_machine: Optional[WorkflowStateMachine] = None
        self._classifier: Optional[StateClassifier] = None
        self._constraint_evaluator: Optional[ConstraintEvaluator] = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        if self._workflow:
            return f"fsm:{self._workflow.name}"
        return "fsm:uninitialized"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "fsm"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with workflow configuration.

        Args:
            config: Configuration dict with either:
                - workflow_path: str - Path to workflow YAML/JSON
                - workflow: dict - Workflow definition as dict

        Raises:
            ValueError: If neither workflow_path nor workflow provided
        """
        if "workflow_path" in config:
            self._workflow = WorkflowParser.parse_file(config["workflow_path"])
        elif "workflow" in config:
            workflow_data = config["workflow"]
            if isinstance(workflow_data, dict):
                self._workflow = WorkflowParser().parse_dict(workflow_data)
            else:
                self._workflow = workflow_data
        else:
            raise ValueError(
                "FSM engine requires 'workflow_path' or 'workflow' in config"
            )

        self._state_machine = WorkflowStateMachine(self._workflow)
        self._classifier = StateClassifier(self._workflow.states)
        self._constraint_evaluator = ConstraintEvaluator(self._workflow.constraints)
        self._initialized = True

        logger.info(
            f"FSMPolicyEngine initialized with workflow '{self._workflow.name}' "
            f"({len(self._workflow.states)} states, "
            f"{len(self._workflow.constraints)} constraints)"
        )

    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate request - check if there are pending interventions.

        For FSM, most evaluation happens after response (in evaluate_response).
        Pre-call evaluation checks for pending interventions from previous violations.

        Args:
            session_id: Unique session identifier
            request_data: The LLM request data
            context: Additional context

        Returns:
            PolicyEvaluationResult with decision
        """
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        session = await self._state_machine.get_or_create_session(session_id)

        # Check for pending intervention from previous violation
        pending = session.pending_intervention
        if pending:
            logger.debug(
                f"Session {session_id}: Found pending intervention '{pending}'"
            )
            return PolicyEvaluationResult(
                decision=PolicyDecision.MODIFY,
                violations=[],
                intervention_needed=pending,
                metadata={
                    "current_state": session.current_state,
                    "workflow": self._workflow.name,
                },
            )

        return PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            violations=[],
            metadata={
                "current_state": session.current_state,
                "workflow": self._workflow.name,
            },
        )

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate response - classify state, check constraints, transition.

        This is the main evaluation method that:
        1. Classifies the response to determine what workflow state it represents
        2. Checks constraints BEFORE transitioning
        3. Performs the state transition
        4. Records any violations and pending interventions

        Args:
            session_id: Unique session identifier
            response_data: The LLM response
            request_data: Original request data
            context: Additional context

        Returns:
            PolicyEvaluationResult with decision and any violations
        """
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        session = await self._state_machine.get_or_create_session(session_id)
        previous_state = session.current_state

        # Classify response to determine workflow state
        classification = self._classifier.classify(response_data, previous_state)

        logger.debug(
            f"Session {session_id}: Classified response as '{classification.state_name}' "
            f"(confidence={classification.confidence:.2f}, method={classification.method})"
        )

        # Check constraints BEFORE transition
        constraint_violations = self._constraint_evaluator.evaluate_all(
            session,
            proposed_state=classification.state_name,
        )

        # Convert to PolicyViolation format
        violations = [
            PolicyViolation(
                name=cv.constraint_name,
                severity=cv.severity,
                message=cv.message,
                intervention=cv.intervention,
                metadata={
                    "constraint_type": cv.constraint_type.value,
                    **cv.details,
                },
            )
            for cv in constraint_violations
        ]

        # Attempt state transition
        transition_result, error = await self._state_machine.transition(
            session_id,
            classification.state_name,
            context,
            confidence=classification.confidence,
            method=classification.method,
        )

        # Determine intervention and decision
        intervention = None
        decision = PolicyDecision.ALLOW

        if violations:
            # Find first violation with intervention
            for v in violations:
                if v.intervention:
                    intervention = v.intervention
                    await self._state_machine.set_pending_intervention(
                        session_id, intervention
                    )
                    decision = PolicyDecision.WARN
                    break

            # Critical violations should deny
            if any(v.severity == "critical" for v in violations):
                decision = PolicyDecision.DENY

        return PolicyEvaluationResult(
            decision=decision,
            violations=violations,
            intervention_needed=intervention,
            metadata={
                "previous_state": previous_state,
                "current_state": classification.state_name,
                "classification_confidence": classification.confidence,
                "classification_method": classification.method,
                "transition_result": transition_result.value,
                "transition_error": error,
                "workflow": self._workflow.name,
            },
        )

    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: Optional[str] = None,
    ) -> StateClassificationResult:
        """
        Classify response to workflow state.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response to classify
            current_state: Current state (looked up if not provided)

        Returns:
            StateClassificationResult with detected state
        """
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        if current_state is None:
            session = await self._state_machine.get_or_create_session(session_id)
            current_state = session.current_state

        result = self._classifier.classify(response_data, current_state)

        return StateClassificationResult(
            state_name=result.state_name,
            confidence=result.confidence,
            method=result.method,
            details=result.details,
        )

    async def get_current_state(self, session_id: str) -> str:
        """Get current state name for session."""
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        session = await self._state_machine.get_or_create_session(session_id)
        return session.current_state

    async def get_state_history(self, session_id: str) -> List[str]:
        """Get state transition history."""
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        return await self._state_machine.get_state_history(session_id)

    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """Get valid next states from current state."""
        if not self._initialized:
            raise RuntimeError("FSMPolicyEngine not initialized. Call initialize() first.")

        valid = await self._state_machine.get_valid_transitions(session_id)
        return list(valid)

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state for debugging/tracing."""
        if not self._initialized:
            return None

        session = await self._state_machine.get_session(session_id)
        if not session:
            return None

        return {
            "current_state": session.current_state,
            "history": session.get_state_sequence(),
            "pending_intervention": session.pending_intervention,
            "constraint_violations": session.constraint_violations,
            "workflow": self._workflow.name,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
        }

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        if not self._initialized:
            return

        await self._state_machine.reset_session(session_id)
        logger.debug(f"Session {session_id} reset")

    async def shutdown(self) -> None:
        """Cleanup resources."""
        # FSM engine doesn't need explicit cleanup
        self._initialized = False
        logger.info("FSMPolicyEngine shutdown")
