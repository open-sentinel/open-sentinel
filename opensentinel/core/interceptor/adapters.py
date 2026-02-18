"""
Adapters to wrap existing components as Checkers.

Provides PolicyEngineChecker to wrap PolicyEngine instances.
"""

import logging
from typing import Any, Dict, Optional

from opensentinel.policy.protocols import PolicyDecision, PolicyEngine, PolicyEvaluationResult

from .checker import Checker
from .types import CheckerContext, CheckerMode, CheckPhase, CheckResult

logger = logging.getLogger(__name__)


class PolicyEngineChecker(Checker):
    """
    Wraps an existing PolicyEngine as a Checker.

    Maps PolicyEngine.evaluate_request() for PRE_CALL phase
    and PolicyEngine.evaluate_response() for POST_CALL phase.

    Mode-aware decision coercion:
    - SYNC mode: MODIFY → DENY, WARN → ALLOW (sync checkers are gates only)
    - ASYNC mode: pass through all decisions as-is
    """

    def __init__(
        self,
        engine: PolicyEngine,
        phase: CheckPhase,
        mode: CheckerMode = CheckerMode.SYNC,
        name_suffix: str = "",
    ):
        """
        Initialize the adapter.

        Args:
            engine: The PolicyEngine to wrap
            phase: When to run (PRE_CALL or POST_CALL)
            mode: How to run (SYNC or ASYNC)
            name_suffix: Optional suffix for checker name
        """
        self._engine = engine
        self._phase = phase
        self._mode = mode
        self._name_suffix = name_suffix

    @property
    def name(self) -> str:
        """Unique name of this checker."""
        suffix = f"_{self._name_suffix}" if self._name_suffix else ""
        return f"{self._engine.name}_{self._phase.value}{suffix}"

    @property
    def phase(self) -> CheckPhase:
        """When this checker runs."""
        return self._phase

    @property
    def mode(self) -> CheckerMode:
        """How this checker executes."""
        return self._mode

    async def check(self, context: CheckerContext) -> CheckResult:
        """
        Execute the policy engine evaluation.

        Args:
            context: CheckerContext with request/response data

        Returns:
            CheckResult mapped from PolicyEvaluationResult
        """
        try:
            if self._phase == CheckPhase.PRE_CALL:
                result = await self._engine.evaluate_request(
                    session_id=context.session_id,
                    request_data=context.request_data,
                    context={"user_request_id": context.user_request_id},
                )
            else:  # POST_CALL
                result = await self._engine.evaluate_response(
                    session_id=context.session_id,
                    response_data=context.response_data,
                    request_data=context.request_data,
                    context={"user_request_id": context.user_request_id},
                )

            return self._map_result(result)

        except Exception as e:
            logger.error(f"PolicyEngine '{self._engine.name}' evaluation failed: {e}")
            return CheckResult(
                decision=PolicyDecision.DENY,
                checker_name=self.name,
                message=f"Policy engine error: {e}",
            )

    def _map_result(self, result: PolicyEvaluationResult) -> CheckResult:
        """
        Map PolicyEvaluationResult to CheckResult.

        Mode-aware coercion:
        - SYNC: MODIFY → DENY (log warning), WARN → ALLOW. Strip modified_data.
        - ASYNC: Pass through as-is. Keep modified_data.
        """
        decision = result.decision

        # Build modified_data from various sources
        modified_data: Optional[Dict[str, Any]] = None

        if result.modified_request:
            modified_data = result.modified_request
        elif result.intervention_needed and result.metadata:
            modified_data = {
                "intervention_name": result.intervention_needed,
                "intervention_context": result.metadata,
            }

        # Mode-aware coercion
        if self._mode == CheckerMode.SYNC:
            if decision == PolicyDecision.MODIFY:
                logger.warning(
                    f"Sync checker '{self.name}' returned MODIFY, coercing to DENY"
                )
                decision = PolicyDecision.DENY
                modified_data = None
            elif decision == PolicyDecision.WARN:
                decision = PolicyDecision.ALLOW
                modified_data = None

        # Pass through violations directly as PolicyViolation instances
        violations = list(result.violations)

        # Build message from violations
        message: Optional[str] = None
        if violations:
            messages = [v.message for v in violations if v.message]
            if messages:
                message = "; ".join(messages)

        return CheckResult(
            decision=decision,
            checker_name=self.name,
            modified_data=modified_data,
            violations=violations,
            message=message,
        )
