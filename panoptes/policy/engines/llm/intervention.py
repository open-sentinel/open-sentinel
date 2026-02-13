"""
Intervention decision engine for the LLM Policy Engine.

Maps violations and drift scores to intervention strategies with
cooldown management and self-correction detection.
"""

import logging
from typing import Optional, Dict, Any, List

from panoptes.policy.engines.llm.models import (
    ConstraintEvaluation,
    DriftScores,
    DriftLevel,
    SessionContext,
)
from panoptes.policy.engines.llm.templates import DEFAULT_TEMPLATES, format_template
from panoptes.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
    get_strategy,
)
from panoptes.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


class InterventionHandler:
    """Decides when and how to intervene based on violations and drift.
    
    Maps constraint violations and drift levels to intervention strategies,
    with cooldown to prevent intervention spam and self-correction detection.
    
    Example:
        engine = InterventionHandler(workflow, cooldown_turns=2)
        intervention = engine.decide(session, violations, drift)
        if intervention:
            modified_request = engine.apply_intervention(
                request_data, intervention, session
            )
    """

    # Severity to strategy mapping
    SEVERITY_STRATEGY_MAP = {
        "low": StrategyType.SYSTEM_PROMPT_APPEND,
        "warning": StrategyType.SYSTEM_PROMPT_APPEND,
        "medium": StrategyType.USER_MESSAGE_INJECT,
        "error": StrategyType.USER_MESSAGE_INJECT,
        "high": StrategyType.CONTEXT_REMINDER,
        "critical": StrategyType.HARD_BLOCK,
    }

    # Drift level to strategy mapping
    DRIFT_STRATEGY_MAP = {
        DriftLevel.NOMINAL: None,
        DriftLevel.WARNING: StrategyType.SYSTEM_PROMPT_APPEND,
        DriftLevel.INTERVENTION: StrategyType.CONTEXT_REMINDER,
        DriftLevel.CRITICAL: StrategyType.HARD_BLOCK,
    }

    def __init__(
        self,
        workflow: WorkflowDefinition,
        cooldown_turns: int = 2,
        self_correction_margin: float = 0.1,
    ):
        self.workflow = workflow
        self.cooldown_turns = cooldown_turns
        self.self_correction_margin = self_correction_margin

    def decide(
        self,
        session: SessionContext,
        violations: List[ConstraintEvaluation],
        drift: DriftScores,
    ) -> Optional[InterventionConfig]:
        """Decide if intervention is needed and which strategy to use.
        
        Args:
            session: Current session context
            violations: Constraint violations detected
            drift: Drift scores computed
            
        Returns:
            InterventionConfig if intervention needed, None otherwise
        """
        # Check for critical violations that bypass cooldown
        has_critical = any(
            v.severity == "critical" and v.violated
            for v in violations
        )
        
        # Cooldown check (skip if critical)
        if not has_critical:
            turns_since_intervention = session.turn_count - session.last_intervention_turn
            if turns_since_intervention < self.cooldown_turns:
                logger.debug(
                    f"Cooldown active: {turns_since_intervention}/{self.cooldown_turns} turns"
                )
                return None
        
        # Self-correction check
        if session.pending_intervention:
            # Check if drift is decreasing (agent self-correcting)
            if drift.composite < session.drift_score - self.self_correction_margin:
                logger.info(
                    f"Self-correction detected: drift {session.drift_score:.3f} → "
                    f"{drift.composite:.3f}"
                )
                session.pending_intervention = None
                return None
        
        # Get violation-based strategy
        violation_strategy = None
        violation_severity = None
        first_violation = None
        
        for v in violations:
            if v.violated:
                first_violation = v
                strategy = self.SEVERITY_STRATEGY_MAP.get(
                    v.severity, StrategyType.SYSTEM_PROMPT_APPEND
                )
                # Take the most severe
                if violation_strategy is None or self._compare_strategies(
                    strategy, violation_strategy
                ) > 0:
                    violation_strategy = strategy
                    violation_severity = v.severity
        
        # Get drift-based strategy
        drift_strategy = self.DRIFT_STRATEGY_MAP.get(drift.level)
        
        # No intervention needed
        if not violation_strategy and not drift_strategy:
            return None
        
        # Take most severe strategy
        final_strategy = None
        if violation_strategy and drift_strategy:
            if self._compare_strategies(violation_strategy, drift_strategy) >= 0:
                final_strategy = violation_strategy
            else:
                final_strategy = drift_strategy
        else:
            final_strategy = violation_strategy or drift_strategy
        
        if final_strategy is None:
            return None
        
        # Select template
        template = self._select_template(
            first_violation,
            drift,
            session,
        )
        
        # Determine priority based on strategy severity
        priority = self._get_strategy_priority(final_strategy)
        
        return InterventionConfig(
            strategy_type=final_strategy,
            message_template=template,
            priority=priority,
        )

    def apply_intervention(
        self,
        data: Dict[str, Any],
        config: InterventionConfig,
        session: SessionContext,
    ) -> Dict[str, Any]:
        """Apply an intervention to request data.
        
        Args:
            data: LLM request data
            config: Intervention configuration
            session: Session context
            
        Returns:
            Modified request data
        """
        # Build context for template formatting
        context = self._build_context(session)
        
        # Get strategy
        strategy = get_strategy(config.strategy_type)
        
        # Apply intervention
        modified_data = strategy.apply(data, config, context)
        
        # Update session
        session.last_intervention_turn = session.turn_count
        
        logger.info(
            f"Applied {config.strategy_type.value} intervention for "
            f"session {session.session_id}"
        )
        
        return modified_data

    def should_escalate(self, drift: DriftScores) -> bool:
        """Check if situation requires escalation (human review).
        
        Args:
            drift: Current drift scores
            
        Returns:
            True if escalation is warranted
        """
        return drift.level == DriftLevel.CRITICAL

    def _compare_strategies(
        self,
        a: StrategyType,
        b: StrategyType,
    ) -> int:
        """Compare strategy severity. Returns positive if a > b."""
        order = [
            StrategyType.SYSTEM_PROMPT_APPEND,
            StrategyType.USER_MESSAGE_INJECT,
            StrategyType.CONTEXT_REMINDER,
            StrategyType.HARD_BLOCK,
        ]
        return order.index(a) - order.index(b)

    def _get_strategy_priority(self, strategy: StrategyType) -> int:
        """Get priority value for a strategy type."""
        priority_map = {
            StrategyType.SYSTEM_PROMPT_APPEND: 1,
            StrategyType.USER_MESSAGE_INJECT: 2,
            StrategyType.CONTEXT_REMINDER: 3,
            StrategyType.HARD_BLOCK: 4,
        }
        return priority_map.get(strategy, 0)

    def _select_template(
        self,
        violation: Optional[ConstraintEvaluation],
        drift: DriftScores,
        session: SessionContext,
    ) -> str:
        """Select appropriate template for intervention."""
        # Check workflow-defined interventions first
        if violation and violation.constraint_id:
            # Look up constraint to get intervention name
            for c in self.workflow.constraints:
                if c.name == violation.constraint_id and c.intervention:
                    if c.intervention in self.workflow.interventions:
                        return self.workflow.interventions[c.intervention]
        
        # Fall back to default templates
        if violation and violation.violated:
            return DEFAULT_TEMPLATES.get(
                "constraint_violation",
                DEFAULT_TEMPLATES["policy_violation"]
            )
        
        if drift.level == DriftLevel.CRITICAL:
            return DEFAULT_TEMPLATES["drift_critical"]
        elif drift.level == DriftLevel.INTERVENTION:
            return DEFAULT_TEMPLATES["drift_intervention"]
        elif drift.level == DriftLevel.WARNING:
            return DEFAULT_TEMPLATES["drift_warning"]
        
        # Structural drift (multiple uncertain classifications)
        if session.is_structurally_drifting():
            return DEFAULT_TEMPLATES["structural_drift"]
        
        return DEFAULT_TEMPLATES["policy_violation"]

    def _build_context(self, session: SessionContext) -> Dict[str, Any]:
        """Build context dict for template formatting."""
        # Get current state info
        state = self.workflow.get_state(session.current_state)
        state_description = state.description if state else ""
        
        # Get active constraints
        active_constraints = ", ".join(
            c.name for c in self.workflow.constraints[:5]
        ) or "none"
        
        return {
            "session_id": session.session_id,
            "workflow_name": self.workflow.name,
            "current_state": session.current_state,
            "state_description": state_description,
            "drift_score": session.drift_score,
            "active_constraints": active_constraints,
            "turn_count": session.turn_count,
        }
