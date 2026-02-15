"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rubrics using LLM judges. Integrates with the Panoptes
policy engine infrastructure via PolicyEngine ABC.
"""

import logging
from typing import Dict, Any, Optional, List

from panoptes.policy.protocols import (
    PolicyEngine,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyViolation,
)
from panoptes.policy.registry import register_engine
from panoptes.policy.engines.judge.models import (
    JudgeVerdict,
    JudgeSessionContext,
    VerdictAction,
    EvaluationScope,
)
from panoptes.policy.engines.judge.client import JudgeClient
from panoptes.policy.engines.judge.evaluator import JudgeEvaluator
from panoptes.policy.engines.judge.rubrics import RubricRegistry

logger = logging.getLogger(__name__)

# Mapping from VerdictAction to (PolicyDecision, severity, intervention)
_VERDICT_MAP: Dict[VerdictAction, tuple] = {
    VerdictAction.PASS: (PolicyDecision.ALLOW, None, None),
    VerdictAction.WARN: (PolicyDecision.WARN, "warning", None),
    VerdictAction.INTERVENE: (PolicyDecision.MODIFY, "error", "system_prompt_append"),
    VerdictAction.BLOCK: (PolicyDecision.DENY, "critical", "hard_block"),
    VerdictAction.ESCALATE: (PolicyDecision.WARN, "warning", None),
}


@register_engine("judge")
class JudgePolicyEngine(PolicyEngine):
    """Policy engine that uses LLM judges to evaluate agent behavior.

    Supports turn-level and conversation-level evaluation against
    configurable rubrics. Works with single or multiple judge models.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._client: Optional[JudgeClient] = None
        self._evaluator: Optional[JudgeEvaluator] = None
        self._sessions: Dict[str, JudgeSessionContext] = {}

        # Config
        self._default_rubric: str = "agent_behavior"
        self._conversation_rubric: Optional[str] = "conversation_policy"
        self._pre_call_enabled: bool = False
        self._pre_call_rubric: str = "safety"
        self._conversation_eval_interval: int = 5

    @property
    def name(self) -> str:
        return f"judge:{self._default_rubric}"

    @property
    def engine_type(self) -> str:
        return "judge"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the judge engine with configuration.

        Args:
            config: Configuration dict with:
                - models: List of judge model configs [{name, model, temperature, ...}]
                - default_rubric: Name of default turn-scope rubric
                - conversation_rubric: Name of conversation-scope rubric (or null to disable)
                - pre_call_enabled: Whether to evaluate requests (default: false)
                - pre_call_rubric: Rubric for pre-call evaluation
                - pass_threshold: Score threshold for PASS (default: 0.6)
                - warn_threshold: Score threshold for WARN (default: 0.4)
                - block_threshold: Score threshold for BLOCK (default: 0.2)
                - conversation_eval_interval: Run conversation eval every N turns (default: 5)
                - custom_rubrics_path: Path to custom rubric YAML files
                - checker_mode: "async" or "sync" (used by interceptor, not engine)
        """
        # Build client with judge models
        self._client = JudgeClient()
        models = config.get("models", [])
        if not models:
            raise ValueError("Judge engine requires at least one model in 'models' list")

        for model_config in models:
            self._client.add_model(
                name=model_config.get("name", "primary"),
                model=model_config["model"],
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 2048),
                timeout=model_config.get("timeout", 15.0),
            )

        # Build evaluator
        self._evaluator = JudgeEvaluator(
            client=self._client,
            pass_threshold=config.get("pass_threshold", 0.6),
            warn_threshold=config.get("warn_threshold", 0.4),
            block_threshold=config.get("block_threshold", 0.2),
        )

        # Config
        self._default_rubric = config.get("default_rubric", "agent_behavior")
        self._conversation_rubric = config.get("conversation_rubric", "conversation_policy")
        self._pre_call_enabled = config.get("pre_call_enabled", False)
        self._pre_call_rubric = config.get("pre_call_rubric", "safety")
        self._conversation_eval_interval = config.get("conversation_eval_interval", 5)

        # Load custom rubrics if configured
        custom_path = config.get("custom_rubrics_path")
        if custom_path:
            RubricRegistry.load_from_yaml(custom_path)

        self._initialized = True
        logger.info(f"JudgePolicyEngine initialized: {self.name}")

    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate an incoming request (PRE_CALL).

        Default: ALLOW (most judgment happens post-call).
        If pre_call_enabled and a pending intervention exists from a
        previous evaluation, apply it here.

        Args:
            session_id: Unique session identifier.
            request_data: The LLM request data.
            context: Additional context.

        Returns:
            PolicyEvaluationResult.
        """
        if not self._initialized:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        session = self._get_or_create_session(session_id)

        # Apply pending intervention from previous turn's evaluation
        if session.pending_intervention:
            intervention = session.pending_intervention
            session.pending_intervention = None
            return PolicyEvaluationResult(
                decision=PolicyDecision.MODIFY,
                violations=[PolicyViolation(
                    name="judge_deferred_intervention",
                    severity="warning",
                    message="Judge intervention from previous turn evaluation.",
                    intervention="system_prompt_append",
                )],
                intervention_needed="system_prompt_append",
                modified_request=self._apply_system_prompt_guidance(
                    request_data, intervention,
                ),
                metadata={"judge_deferred": True},
            )

        # Optional pre-call screening
        if self._pre_call_enabled:
            return await self._evaluate_pre_call(session_id, request_data, context)

        return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate an LLM response (POST_CALL).

        Main evaluation path:
        1. Always run turn-scope rubric on latest response
        2. Run conversation-scope rubric on interval or when triggered
        3. Merge verdicts (most restrictive action wins)
        4. Map to PolicyEvaluationResult

        Args:
            session_id: Unique session identifier.
            response_data: The LLM response.
            request_data: The original request data.
            context: Additional context.

        Returns:
            PolicyEvaluationResult with decision and any violations.
        """
        if not self._initialized:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        session = self._get_or_create_session(session_id)
        response_content = self._extract_response_content(response_data)
        conversation = self._extract_conversation(request_data)
        metadata = (context or {}).get("metadata", {})

        primary_model = self._client.primary_model
        if not primary_model:
            logger.error("No judge models configured")
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        verdicts: List[JudgeVerdict] = []

        # 1. Turn-scope evaluation (always runs)
        turn_rubric = RubricRegistry.get(self._default_rubric)
        if turn_rubric:
            try:
                turn_verdict = await self._evaluator.evaluate_turn(
                    model_name=primary_model,
                    rubric=turn_rubric,
                    response_content=response_content,
                    conversation=conversation,
                    metadata=metadata,
                )
                verdicts.append(turn_verdict)
                session.record_verdict(turn_verdict)
            except Exception as e:
                logger.error(f"Turn evaluation failed: {e}")
        else:
            logger.warning(f"Default rubric not found: {self._default_rubric}")

        # 2. Conversation-scope evaluation (on interval or trigger)
        if self._should_run_conversation_eval(session, verdicts):
            conv_rubric = RubricRegistry.get(self._conversation_rubric)
            if conv_rubric:
                try:
                    conv_verdict = await self._evaluator.evaluate_conversation(
                        model_name=primary_model,
                        rubric=conv_rubric,
                        full_conversation=conversation,
                        metadata=metadata,
                    )
                    verdicts.append(conv_verdict)
                    session.record_verdict(conv_verdict)
                except Exception as e:
                    logger.error(f"Conversation evaluation failed: {e}")

        # 3. Merge verdicts and build result
        if not verdicts:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        return self._build_result(verdicts, session)

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state for debugging/tracing."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.to_dict()

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        logger.info("JudgePolicyEngine shut down")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _get_or_create_session(self, session_id: str) -> JudgeSessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = JudgeSessionContext(session_id=session_id)
        return self._sessions[session_id]

    def _should_run_conversation_eval(
        self,
        session: JudgeSessionContext,
        turn_verdicts: List[JudgeVerdict],
    ) -> bool:
        """Determine if conversation-scope evaluation should run."""
        if not self._conversation_rubric:
            return False

        # Run on interval
        if (
            session.turn_count > 0
            and session.turn_count % self._conversation_eval_interval == 0
        ):
            return True

        # Run when a turn verdict is warn or worse
        for v in turn_verdicts:
            if v.action in (VerdictAction.WARN, VerdictAction.INTERVENE, VerdictAction.BLOCK):
                return True

        return False

    async def _evaluate_pre_call(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """Run pre-call safety screening on user message."""
        rubric = RubricRegistry.get(self._pre_call_rubric)
        if not rubric:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        messages = request_data.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        primary_model = self._client.primary_model
        if not primary_model:
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

        try:
            verdict = await self._evaluator.evaluate_turn(
                model_name=primary_model,
                rubric=rubric,
                response_content=user_message,
                conversation=messages,
                metadata=(context or {}).get("metadata", {}),
            )
            return self._build_result([verdict], self._get_or_create_session(session_id))
        except Exception as e:
            logger.error(f"Pre-call evaluation failed: {e}")
            return PolicyEvaluationResult(decision=PolicyDecision.ALLOW)

    def _build_result(
        self,
        verdicts: List[JudgeVerdict],
        session: JudgeSessionContext,
    ) -> PolicyEvaluationResult:
        """Build PolicyEvaluationResult from judge verdicts.

        Takes the most restrictive action across all verdicts.
        """
        # Find most restrictive verdict
        action_priority = {
            VerdictAction.PASS: 0,
            VerdictAction.WARN: 1,
            VerdictAction.ESCALATE: 2,
            VerdictAction.INTERVENE: 3,
            VerdictAction.BLOCK: 4,
        }

        worst_verdict = max(verdicts, key=lambda v: action_priority.get(v.action, 0))
        decision, severity, intervention = _VERDICT_MAP[worst_verdict.action]

        violations = []
        if severity:
            for verdict in verdicts:
                if verdict.action != VerdictAction.PASS:
                    v_decision, v_severity, v_intervention = _VERDICT_MAP[verdict.action]
                    violations.append(PolicyViolation(
                        name=f"judge_{verdict.scope.value}_{verdict.action.value}",
                        severity=v_severity or "warning",
                        message=verdict.summary,
                        intervention=v_intervention,
                        metadata={
                            "composite_score": verdict.composite_score,
                            "judge_model": verdict.judge_model,
                            "scope": verdict.scope.value,
                        },
                    ))

        # For INTERVENE in async mode, store as pending for next request
        intervention_needed = None
        modified_request = None
        if worst_verdict.action == VerdictAction.INTERVENE:
            session.pending_intervention = worst_verdict.summary
            intervention_needed = "system_prompt_append"

        metadata = {
            "judge": {
                "verdicts": [v.to_dict() for v in verdicts],
                "session_turn": session.turn_count,
            }
        }

        if worst_verdict.action == VerdictAction.ESCALATE:
            metadata["escalate"] = True

        return PolicyEvaluationResult(
            decision=decision,
            violations=violations,
            intervention_needed=intervention_needed,
            modified_request=modified_request,
            metadata=metadata,
        )

    def _extract_response_content(self, response_data: Any) -> str:
        """Extract text content from response data."""
        if isinstance(response_data, str):
            return response_data
        if isinstance(response_data, dict):
            # OpenAI-style response
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
            return response_data.get("content", "")
        return str(response_data)

    def _extract_conversation(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation messages from request data."""
        return request_data.get("messages", [])

    def _apply_system_prompt_guidance(
        self,
        request_data: Dict[str, Any],
        guidance: str,
    ) -> Dict[str, Any]:
        """Apply judge guidance to the system prompt."""
        data = dict(request_data)
        messages = list(data.get("messages", []))

        guidance_text = f"\n\n[JUDGE GUIDANCE]: {guidance}"

        # Find and append to system message
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = dict(msg)
                messages[i]["content"] = msg.get("content", "") + guidance_text
                data["messages"] = messages
                return data

        # No system message, insert one
        messages.insert(0, {"role": "system", "content": f"[JUDGE GUIDANCE]: {guidance}"})
        data["messages"] = messages
        return data
