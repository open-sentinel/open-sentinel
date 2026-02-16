"""
LLM Policy Engine implementation.

The main orchestrator that uses an LLM for state classification,
drift detection, and soft constraint evaluation. Registered via
@register_engine("llm") for use with PolicyEngineRegistry.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from panoptes.policy.registry import register_engine
from panoptes.policy.protocols import (
    StatefulPolicyEngine,
    PolicyEvaluationResult,
    PolicyDecision,
    PolicyViolation,
    StateClassificationResult,
    require_initialized,
)
from panoptes.policy.engines.llm.models import (
    SessionContext,
    ConfidenceTier,
    DriftLevel,
)
from panoptes.policy.engines.llm.llm_client import LLMClient
from panoptes.policy.engines.llm.state_classifier import LLMStateClassifier
from panoptes.policy.engines.llm.drift_detector import DriftDetector
from panoptes.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from panoptes.policy.engines.llm.intervention import InterventionHandler
from panoptes.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


@register_engine("llm")
class LLMPolicyEngine(StatefulPolicyEngine):
    """LLM-based policy engine.
    
    Uses a lightweight LLM (e.g. gpt-4o-mini) as a reasoning backbone for:
    - State classification with confidence scoring
    - Drift detection (temporal + semantic)
    - Soft constraint evaluation
    
    Reuses the same WorkflowDefinition schema as FSM engine, so users
    can swap engines without rewriting policies.
    
    Example:
        engine = LLMPolicyEngine()
        await engine.initialize({
            "config_path": "workflow.yaml",
            "llm_model": "gpt-4o-mini",
            "temporal_weight": 0.55,
        })
        
        result = await engine.evaluate_response(
            session_id="abc123",
            response_data=llm_response,
            request_data=request,
        )
    """

    def __init__(self):
        self._workflow: Optional[WorkflowDefinition] = None
        self._llm_client: Optional[LLMClient] = None
        self._state_classifier: Optional[LLMStateClassifier] = None
        self._drift_detector: Optional[DriftDetector] = None
        self._constraint_evaluator: Optional[LLMConstraintEvaluator] = None
        self._intervention_engine: Optional[InterventionHandler] = None
        self._sessions: Dict[str, SessionContext] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        if self._workflow:
            return f"llm:{self._workflow.name}"
        return "llm:uninitialized"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "llm"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the engine with configuration.
        
        Args:
            config: Configuration dict with:
                - config_path: Path to workflow YAML/JSON
                - workflow: Workflow definition as dict (alternative)
                - llm_model: LLM model to use (default: gpt-4o-mini)
                - temperature: LLM temperature (default: 0.0)
                - temporal_weight: Weight for temporal drift (default: 0.55)
                - cooldown_turns: Intervention cooldown (default: 2)
        """
        import yaml
        from pathlib import Path
        
        # Load workflow
        workflow_path = config.get("config_path")
        workflow_dict = config.get("workflow")
        
        if workflow_path:
            path = Path(workflow_path)
            if not path.exists():
                raise ValueError(f"Workflow file not found: {workflow_path}")
            
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    workflow_dict = yaml.safe_load(f)
                else:
                    workflow_dict = json.load(f)
        
        if not workflow_dict:
            raise ValueError("Either config_path or workflow must be provided")
        
        self._workflow = WorkflowDefinition(**workflow_dict)
        
        # Create LLM client
        # Priority: explicit config > default from settings > autodetection
        model = config.get("llm_model") or config.get("default_model")
        if not model:
            from panoptes.config.settings import detect_available_model
            model, provider, _ = detect_available_model()
            logger.info(f"No llm_model or default configured, using detected model: {model}")
        else:
            logger.info(f"Using model for LLM engine: {model}")
        self._llm_client = LLMClient(
            model=model,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
            timeout=config.get("timeout", 10.0),
        )
        
        # Create components
        self._state_classifier = LLMStateClassifier(
            self._llm_client,
            self._workflow,
            confident_threshold=config.get("confident_threshold", 0.8),
            uncertain_threshold=config.get("uncertain_threshold", 0.5),
        )
        
        self._drift_detector = DriftDetector(
            self._workflow,
            temporal_weight=config.get("temporal_weight", 0.55),
        )
        
        self._constraint_evaluator = LLMConstraintEvaluator(
            self._llm_client,
            self._workflow,
            max_constraints_per_batch=config.get("max_constraints_per_batch", 5),
        )
        
        self._intervention_engine = InterventionHandler(
            self._workflow,
            cooldown_turns=config.get("cooldown_turns", 2),
        )
        
        self._initialized = True
        logger.info(f"LLMPolicyEngine initialized: {self.name}")

    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate incoming request - apply pending interventions.
        
        If there's a pending intervention from previous response evaluation,
        apply it to the request.
        """
        if not self._initialized:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)
        
        session = self._get_or_create_session(session_id)
        
        # Apply pending intervention if any
        if session.pending_intervention:
            try:
                # Parse stored intervention config
                intervention = session.pending_intervention
                session.pending_intervention = None
                
                # Apply intervention
                if self._intervention_engine and isinstance(intervention, dict):
                    from panoptes.core.intervention.strategies import (
                        InterventionConfig,
                        StrategyType,
                    )
                    config = InterventionConfig(
                        strategy_type=StrategyType(intervention["strategy_type"]),
                        message_template=intervention["message_template"],
                        priority=intervention.get("priority", 0),
                    )
                    modified = self._intervention_engine.apply_intervention(
                        request_data, config, session
                    )
                    
                    return PolicyEvaluationResult(
                        decision=PolicyDecision.MODIFY,
                        modified_request=modified,
                        metadata={"intervention_applied": True},
                    )
            except Exception as e:
                logger.error(f"Failed to apply pending intervention: {e}")
        
        return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate LLM response - classify, detect drift, check constraints.
        
        This is the main evaluation flow:
        1. Extract assistant message and tool calls
        2. Add turn to session
        3. Classify state
        4. Compute drift
        5. Evaluate constraints
        6. Decide intervention
        7. Record transition
        """
        if not self._initialized:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)
        
        session = self._get_or_create_session(session_id)
        
        # Extract content from response
        message = self._extract_content(response_data)
        tool_calls = self._extract_tool_calls(response_data)
        
        # Add turn to session
        session.add_turn({
            "role": "assistant",
            "message": message,
            "tool_calls": tool_calls,
        })
        
        violations: List[PolicyViolation] = []
        
        try:
            # 1. Classify state
            classification = await self._state_classifier.classify(
                session, message, tool_calls
            )
            
            # Update confidence buffer
            session.add_confidence(classification.best_confidence)
            
            # Check for structural drift
            if session.is_structurally_drifting():
                violations.append(PolicyViolation(
                    name="structural_drift",
                    severity="warning",
                    message="Multiple consecutive uncertain classifications",
                ))
            
            # Check for skip violations
            for skipped in classification.skip_violations:
                violations.append(PolicyViolation(
                    name="skip_violation",
                    severity="error",
                    message=f"Skipped required state: {skipped}",
                    metadata={"skipped_state": skipped},
                ))
            
            # 2. Compute drift
            expected_tools = self._get_expected_tools(classification.best_state)
            drift = self._drift_detector.compute_drift(
                session, message, tool_calls, expected_tools
            )
            
            # Add anomaly violations
            if drift.anomaly_flags.get("unexpected_tool_call"):
                violations.append(PolicyViolation(
                    name="unexpected_tool_call",
                    severity="warning",
                    message="Unexpected tool call for current state",
                ))
            
            if drift.anomaly_flags.get("missing_expected_tool_call"):
                violations.append(PolicyViolation(
                    name="missing_expected_tool_call",
                    severity="warning",
                    message="Expected tool call not made",
                ))
            
            # 3. Evaluate constraints
            constraint_evals = await self._constraint_evaluator.evaluate(
                session, message, tool_calls
            )
            
            # Add constraint violations
            for cv in constraint_evals:
                if cv.violated:
                    violations.append(PolicyViolation(
                        name=cv.constraint_id,
                        severity=cv.severity,
                        message=cv.evidence,
                        metadata={
                            "confidence": cv.confidence,
                            "constraint_id": cv.constraint_id,
                        },
                    ))
            
            # 4. Decide intervention
            intervention_config = None
            if self._intervention_engine:
                intervention_config = self._intervention_engine.decide(
                    session, constraint_evals, drift
                )
            
            # 5. Record transition
            prev_state = session.current_state
            session.record_transition(
                from_state=prev_state,
                to_state=classification.best_state,
                confidence=classification.best_confidence,
                tier=classification.tier,
                drift_score=drift.composite,
                metadata={
                    "method": "llm",
                    "candidates": len(classification.candidates),
                },
            )
            
            # 6. Store pending intervention
            if intervention_config:
                session.pending_intervention = {
                    "strategy_type": intervention_config.strategy_type.value,
                    "message_template": intervention_config.message_template,
                    "priority": intervention_config.priority,
                }
            
            # Determine decision
            decision = PolicyDecision.ALLOW
            if intervention_config:
                from panoptes.core.intervention.strategies import StrategyType
                if intervention_config.strategy_type == StrategyType.HARD_BLOCK:
                    decision = PolicyDecision.DENY
                else:
                    decision = PolicyDecision.WARN
            
            return PolicyEvaluationResult(
                decision=decision,
                violations=violations,
                intervention_needed=intervention_config.strategy_type.value if intervention_config else None,
                metadata={
                    "state": classification.best_state,
                    "confidence": classification.best_confidence,
                    "tier": classification.tier.value,
                    "drift": drift.composite,
                    "drift_level": drift.level.value,
                    "transition_legal": classification.transition_legal,
                },
            )
            
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return PolicyEvaluationResult(
                decision=PolicyDecision.ALLOW,
                metadata={"error": str(e)},
            )

    @require_initialized
    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: Optional[str] = None,
    ) -> StateClassificationResult:
        """Classify a response to a workflow state."""
        
        session = self._get_or_create_session(session_id)
        
        message = self._extract_content(response_data)
        tool_calls = self._extract_tool_calls(response_data)
        
        result = await self._state_classifier.classify(session, message, tool_calls)
        
        return StateClassificationResult(
            state_name=result.best_state,
            confidence=result.best_confidence,
            method="llm",
            details={
                "tier": result.tier.value,
                "candidates": len(result.candidates),
                "transition_legal": result.transition_legal,
            },
        )

    async def get_current_state(self, session_id: str) -> str:
        """Get current state name for session."""
        session = self._sessions.get(session_id)
        if session:
            return session.current_state
        
        # Return initial state
        if self._workflow:
            initial = self._workflow.get_initial_states()
            if initial:
                return initial[0].name
        return "unknown"

    async def get_state_history(self, session_id: str) -> List[str]:
        """Get state transition history."""
        session = self._sessions.get(session_id)
        if session:
            return session.get_state_sequence()
        return []

    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """Get valid next states from current state."""
        current = await self.get_current_state(session_id)
        if self._workflow:
            transitions = self._workflow.get_transitions_from(current)
            return [t.to_state for t in transitions]
        return []

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state for debugging/tracing."""
        session = self._sessions.get(session_id)
        if session:
            return session.to_dict()
        return None

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Reset session: {session_id}")

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        logger.info("LLMPolicyEngine shutdown complete")

    def _get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            # Get initial state
            initial_state = "unknown"
            if self._workflow:
                initial = self._workflow.get_initial_states()
                if initial:
                    initial_state = initial[0].name
            
            self._sessions[session_id] = SessionContext(
                session_id=session_id,
                workflow_name=self._workflow.name if self._workflow else "unknown",
                current_state=initial_state,
            )
            logger.debug(f"Created session: {session_id}")
        
        return self._sessions[session_id]

    def _get_expected_tools(self, state_name: str) -> List[str]:
        """Get expected tool calls for a state."""
        if self._workflow:
            state = self._workflow.get_state(state_name)
            if state and state.classification.tool_calls:
                return state.classification.tool_calls
        return []

    def _extract_content(self, response: Any) -> str:
        """Extract text content from response."""
        if isinstance(response, dict):
            # OpenAI format
            if "choices" in response:
                message = response["choices"][0].get("message", {})
                return message.get("content", "") or ""
            if "content" in response:
                return response.get("content", "") or ""
        
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                return getattr(choice.message, "content", "") or ""
        
        return str(response) if response else ""

    def _extract_tool_calls(self, response: Any) -> List[str]:
        """Extract tool call names from response."""
        tool_names = []
        
        if isinstance(response, dict):
            # OpenAI dict format
            if "choices" in response:
                message = response["choices"][0].get("message", {})
                for tc in message.get("tool_calls", []):
                    if func := tc.get("function", {}).get("name"):
                        tool_names.append(func)
            elif "tool_calls" in response:
                for tc in response.get("tool_calls", []):
                    if isinstance(tc, dict):
                        if func := tc.get("function", {}).get("name"):
                            tool_names.append(func)
                        elif name := tc.get("name"):
                            tool_names.append(name)
        
        elif hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                tool_calls = getattr(choice.message, "tool_calls", None) or []
                for tc in tool_calls:
                    if hasattr(tc, "function") and tc.function:
                        name = getattr(tc.function, "name", None)
                        if name:
                            tool_names.append(name)
        
        return tool_names
