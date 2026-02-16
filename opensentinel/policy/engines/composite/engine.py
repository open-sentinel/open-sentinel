"""
Composite policy engine implementation.

Combines multiple policy engines to run in parallel,
merging their results with configurable strategies.

This enables using multiple policy mechanisms together,
e.g., FSM workflow enforcement + NeMo content moderation.
"""

from typing import Optional, Dict, Any, List
import logging
import asyncio

from opensentinel.policy.protocols import (
    PolicyEngine,
    PolicyEvaluationResult,
    PolicyDecision,
    PolicyViolation,
    require_initialized,
)
from opensentinel.policy.registry import register_engine, PolicyEngineRegistry

logger = logging.getLogger(__name__)

# Decision priority for merging (higher = more restrictive)
DECISION_PRIORITY = {
    PolicyDecision.DENY: 4,
    PolicyDecision.MODIFY: 3,
    PolicyDecision.WARN: 2,
    PolicyDecision.ALLOW: 1,
}


@register_engine("composite")
class CompositePolicyEngine(PolicyEngine):
    """
    Combines multiple policy engines.

    Evaluation strategy:
    - Request: All engines evaluate in parallel, most restrictive decision wins
    - Response: All engines evaluate in parallel, collect all violations
    - Interventions: First engine with intervention wins

    Configuration:
        - engines: list - List of engine configurations
          Each entry: {"type": "fsm|nemo|...", "config": {...}}
        - strategy: str - Merge strategy: "all" (run all) or "first_deny" (stop on first deny)
        - parallel: bool - Run engines in parallel (default: True)

    Example:
        ```python
        engine = CompositePolicyEngine()
        await engine.initialize({
            "engines": [
                {"type": "fsm", "config": {"workflow_path": "./workflow.yaml"}},
                {"type": "nemo", "config": {"config_path": "./nemo_config/"}}
            ],
            "strategy": "all"
        })
        ```
    """

    def __init__(self):
        self._engines: List[PolicyEngine] = []
        self._engine_configs: List[Dict[str, Any]] = []
        self._strategy = "all"
        self._parallel = True
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name showing all combined engines."""
        if not self._engines:
            return "composite:empty"
        names = [e.name for e in self._engines]
        return f"composite:[{','.join(names)}]"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "composite"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with list of engine configurations.

        Args:
            config: Configuration dict with:
                - engines: List of {"type": str, "config": dict}
                - strategy: "all" or "first_deny" (optional)
                - parallel: bool (optional, default True)

        Raises:
            ValueError: If engines list is empty or invalid
        """
        engine_configs = config.get("engines", [])
        if not engine_configs:
            raise ValueError("Composite engine requires at least one engine in 'engines' list")

        self._strategy = config.get("strategy", "all")
        self._parallel = config.get("parallel", True)
        self._engine_configs = engine_configs

        # Create and initialize all engines
        for engine_config in engine_configs:
            engine_type = engine_config.get("type")
            if not engine_type:
                raise ValueError("Each engine config must have a 'type' field")

            engine = PolicyEngineRegistry.create(engine_type)
            await engine.initialize(engine_config.get("config", {}))
            self._engines.append(engine)

            logger.debug(f"Composite: Added engine '{engine.name}'")

        self._initialized = True
        logger.info(
            f"CompositePolicyEngine initialized with {len(self._engines)} engines: "
            f"{[e.engine_type for e in self._engines]}"
        )

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate request through all engines.

        Returns the most restrictive decision from all engines.
        If any engine returns MODIFY with modified_request, uses the first one.

        Args:
            session_id: Unique session identifier
            request_data: The LLM request data
            context: Additional context

        Returns:
            Merged PolicyEvaluationResult
        """

        if self._parallel:
            results = await asyncio.gather(*[
                engine.evaluate_request(session_id, request_data, context)
                for engine in self._engines
            ], return_exceptions=True)

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Engine {self._engines[i].name} request evaluation failed: {result}"
                    )
                    # Add error as violation but continue
                    valid_results.append(PolicyEvaluationResult(
                        decision=PolicyDecision.WARN,
                        violations=[
                            PolicyViolation(
                                name=f"{self._engines[i].engine_type}_error",
                                severity="warning",
                                message=str(result),
                            )
                        ],
                    ))
                else:
                    valid_results.append(result)

            # Truncate at first DENY when strategy is "first_deny"
            if self._strategy == "first_deny":
                truncated_results = []
                for result in valid_results:
                    truncated_results.append(result)
                    if result.decision == PolicyDecision.DENY:
                        break
                valid_results = truncated_results

            return self._merge_results(valid_results)
        else:
            # Sequential evaluation
            results = []
            for engine in self._engines:
                try:
                    result = await engine.evaluate_request(
                        session_id, request_data, context
                    )
                    results.append(result)

                    # Early exit on first deny if strategy is "first_deny"
                    if self._strategy == "first_deny" and result.decision == PolicyDecision.DENY:
                        break

                except Exception as e:
                    logger.error(f"Engine {engine.name} request evaluation failed: {e}")
                    results.append(PolicyEvaluationResult(
                        decision=PolicyDecision.WARN,
                        violations=[
                            PolicyViolation(
                                name=f"{engine.engine_type}_error",
                                severity="warning",
                                message=str(e),
                            )
                        ],
                    ))

            return self._merge_results(results)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate response through all engines.

        Collects violations from all engines and returns most restrictive decision.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response
            request_data: Original request data
            context: Additional context

        Returns:
            Merged PolicyEvaluationResult
        """

        if self._parallel:
            results = await asyncio.gather(*[
                engine.evaluate_response(session_id, response_data, request_data, context)
                for engine in self._engines
            ], return_exceptions=True)

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Engine {self._engines[i].name} response evaluation failed: {result}"
                    )
                    valid_results.append(PolicyEvaluationResult(
                        decision=PolicyDecision.WARN,
                        violations=[
                            PolicyViolation(
                                name=f"{self._engines[i].engine_type}_error",
                                severity="warning",
                                message=str(result),
                            )
                        ],
                    ))
                else:
                    valid_results.append(result)

            # Truncate at first DENY when strategy is "first_deny"
            if self._strategy == "first_deny":
                truncated_results = []
                for result in valid_results:
                    truncated_results.append(result)
                    if result.decision == PolicyDecision.DENY:
                        break
                valid_results = truncated_results

            return self._merge_results(valid_results)
        else:
            # Sequential evaluation
            results = []
            for engine in self._engines:
                try:
                    result = await engine.evaluate_response(
                        session_id, response_data, request_data, context
                    )
                    results.append(result)

                    if self._strategy == "first_deny" and result.decision == PolicyDecision.DENY:
                        break

                except Exception as e:
                    logger.error(f"Engine {engine.name} response evaluation failed: {e}")
                    results.append(PolicyEvaluationResult(
                        decision=PolicyDecision.WARN,
                        violations=[
                            PolicyViolation(
                                name=f"{engine.engine_type}_error",
                                severity="warning",
                                message=str(e),
                            )
                        ],
                    ))

            return self._merge_results(results)

    def _merge_results(
        self,
        results: List[PolicyEvaluationResult],
    ) -> PolicyEvaluationResult:
        """
        Merge results from multiple engines.

        Strategy:
        - Decision: Most restrictive wins (DENY > MODIFY > WARN > ALLOW)
        - Violations: Collect all
        - Intervention: First one found
        - Modified request: First one found
        - Metadata: Merge all
        """
        if not results:
            return PolicyEvaluationResult(
                decision=PolicyDecision.ALLOW,
                violations=[],
            )

        final_decision = PolicyDecision.ALLOW
        all_violations: List[PolicyViolation] = []
        intervention = None
        modified_request = None
        metadata: Dict[str, Any] = {"engines": {}}

        for i, result in enumerate(results):
            engine_name = self._engines[i].name if i < len(self._engines) else f"engine_{i}"

            # Most restrictive decision wins
            if DECISION_PRIORITY[result.decision] > DECISION_PRIORITY[final_decision]:
                final_decision = result.decision

            # Collect all violations
            all_violations.extend(result.violations)

            # First intervention wins
            if result.intervention_needed and not intervention:
                intervention = result.intervention_needed

            # First modified request wins
            if result.modified_request and not modified_request:
                modified_request = result.modified_request

            # Merge metadata
            if result.metadata:
                metadata["engines"][engine_name] = result.metadata

        return PolicyEvaluationResult(
            decision=final_decision,
            violations=all_violations,
            intervention_needed=intervention,
            modified_request=modified_request,
            metadata=metadata,
        )

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state from all engines."""
        if not self._initialized:
            return None

        states: Dict[str, Any] = {}
        for engine in self._engines:
            state = await engine.get_session_state(session_id)
            if state:
                states[engine.name] = state

        return states if states else None

    async def reset_session(self, session_id: str) -> None:
        """Reset session state in all engines."""
        if not self._initialized:
            return

        await asyncio.gather(*[
            engine.reset_session(session_id)
            for engine in self._engines
        ])
        logger.debug(f"Composite: Reset session {session_id} in all engines")

    async def shutdown(self) -> None:
        """Cleanup all engines."""
        await asyncio.gather(*[
            engine.shutdown()
            for engine in self._engines
        ])
        self._engines.clear()
        self._initialized = False
        logger.info("CompositePolicyEngine shutdown")

    def get_engines(self) -> List[PolicyEngine]:
        """Get list of child engines (for debugging)."""
        return list(self._engines)

    def get_engine_by_type(self, engine_type: str) -> Optional[PolicyEngine]:
        """Get a specific engine by type."""
        for engine in self._engines:
            if engine.engine_type == engine_type:
                return engine
        return None
