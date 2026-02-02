"""
Panoptes LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Evaluates request through policy engine
   - Injects correction prompts when needed

2. async_moderation_hook: Runs IN PARALLEL with LLM call (non-blocking)
   - Evaluates previous response through policy engine
   - Records pending interventions for next call
   - Does NOT block or add latency to the critical path

3. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Evaluates response through policy engine
   - Updates state machine (for stateful engines)

The hook system now supports pluggable policy engines:
- FSM: Finite State Machine workflow enforcement
- NeMo: NVIDIA NeMo Guardrails
- Composite: Combine multiple engines

Tracing is handled via OpenTelemetry (see tracing/otel_tracer.py).

Based on LiteLLM's CustomLogger API:
https://docs.litellm.ai/docs/observability/custom_callback
"""

import logging
from typing import Optional, Union, Dict, Any, Literal
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

from panoptes.config.settings import PanoptesSettings
from panoptes.policy.protocols import PolicyEngine, PolicyDecision

logger = logging.getLogger(__name__)

# Type alias for call types
CallType = Literal[
    "completion",
    "text_completion",
    "embeddings",
    "image_generation",
    "moderation",
    "audio_transcription",
    "acompletion",
    "atext_completion",
    "aembeddings",
]


class WorkflowViolationError(Exception):
    """Raised when a policy engine blocks a request."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class PanoptesCallback(CustomLogger):
    """
    Main Panoptes callback for LiteLLM.

    Implements policy enforcement through LiteLLM's hook system.
    This is registered as a callback when the proxy starts.

    Supports pluggable policy engines:
    - FSM: Finite State Machine workflow enforcement
    - NeMo: NVIDIA NeMo Guardrails
    - Composite: Combine multiple engines

    The callback maintains state per session:
    - Policy engine instance
    - Pending interventions
    - Trace context

    Thread-safety is ensured through asyncio locks for session state.
    """

    def __init__(self, settings: Optional[PanoptesSettings] = None):
        self.settings = settings or PanoptesSettings()

        # Session state storage
        # Maps session_id -> session state dict
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Pending interventions to apply on next call
        # Maps session_id -> intervention config
        self._pending_interventions: Dict[str, Dict[str, Any]] = {}

        # Policy engine (lazy initialized)
        self._policy_engine: Optional[PolicyEngine] = None
        self._policy_engine_initialized = False

        # Workflow tracker (lazy initialized) - kept for backward compatibility
        self._tracker = None

        # Prompt injector (lazy initialized)
        self._injector = None

        # OpenTelemetry tracer for Panoptes events (lazy initialized)
        self._tracer = None

        logger.info("PanoptesCallback initialized")

    async def _get_policy_engine(self) -> Optional[PolicyEngine]:
        """Lazy-load policy engine based on configuration."""
        if self._policy_engine_initialized:
            return self._policy_engine

        try:
            from panoptes.policy.registry import PolicyEngineRegistry

            policy_config = self.settings.get_policy_config()
            engine_type = policy_config.get("type", "nemo")
            engine_config = policy_config.get("config", {})

            # Only initialize if we have configuration
            if engine_type == "fsm" and not engine_config.get("workflow_path"):
                logger.debug("No workflow_path configured, skipping policy engine")
                self._policy_engine_initialized = True
                return None

            if engine_type == "nemo" and not engine_config.get("config_path"):
                logger.debug("No NeMo config_path configured, skipping policy engine")
                self._policy_engine_initialized = True
                return None

            if engine_type == "composite" and not engine_config.get("engines"):
                logger.debug("No engines configured for composite, skipping")
                self._policy_engine_initialized = True
                return None

            logger.info(f"Initializing policy engine: {engine_type}")
            self._policy_engine = await PolicyEngineRegistry.create_and_initialize(
                engine_type, engine_config
            )
            self._policy_engine_initialized = True
            logger.info(f"Policy engine initialized: {self._policy_engine.name}")

        except Exception as e:
            logger.error(f"Failed to initialize policy engine: {e}")
            self._policy_engine_initialized = True
            self._policy_engine = None

        return self._policy_engine

    @property
    def tracker(self):
        """Lazy-load tracker to avoid import issues (backward compatibility)."""
        if self._tracker is None and self.settings.workflow_path:
            from panoptes.policy.engines.fsm import WorkflowParser, WorkflowTracker

            workflow = WorkflowParser.parse_file(self.settings.workflow_path)
            self._tracker = WorkflowTracker(workflow)
        return self._tracker

    @property
    def injector(self):
        """Lazy-load injector to avoid import issues."""
        if self._injector is None and self.settings.workflow_path:
            from panoptes.policy.engines.fsm import WorkflowParser, PromptInjector

            workflow = WorkflowParser.parse_file(self.settings.workflow_path)
            self._injector = PromptInjector(workflow)
        return self._injector

    @property
    def tracer(self):
        """Lazy-load tracer for Panoptes event logging via OpenTelemetry."""
        if self._tracer is None:
            logger.debug(f"Tracer check: otel.enabled={self.settings.otel.enabled}")
            if self.settings.otel.enabled:
                from panoptes.tracing.otel_tracer import PanoptesTracer
                self._tracer = PanoptesTracer(self.settings.otel)
                logger.info(f"PanoptesTracer initialized: {self._tracer}")
        return self._tracer

    def _extract_session_id(self, data: dict) -> str:
        """
        Extract session ID from request data.

        Priority:
        1. x-session-id in metadata
        2. metadata.session_id
        3. user field
        4. Hash of first message
        """
        from panoptes.proxy.middleware import SessionExtractor

        return SessionExtractor.extract_session_id(data)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallType,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Execute BEFORE LLM call.

        This hook:
        1. Evaluates request through policy engine
        2. Checks for pending interventions from previous call
        3. If intervention needed, modifies request to inject correction
        4. Starts OTEL trace span

        Args:
            user_api_key_dict: User API key information
            cache: LiteLLM cache instance
            data: Request data (messages, model, etc.)
            call_type: Type of LLM call

        Returns:
            Modified data dict, or None to proceed unchanged
        """
        session_id = self._extract_session_id(data)

        logger.debug(f"pre_call_hook: session={session_id}, call_type={call_type}")

        # Evaluate request through policy engine
        policy_engine = await self._get_policy_engine()
        if policy_engine:
            try:
                result = await policy_engine.evaluate_request(
                    session_id=session_id,
                    request_data=data,
                    context={"call_type": call_type},
                )

                # Handle policy decision
                if result.decision == PolicyDecision.DENY:
                    logger.warning(
                        f"Request blocked by policy engine for session {session_id}: "
                        f"{[v.name for v in result.violations]}"
                    )
                    raise WorkflowViolationError(
                        "Request blocked by policy engine",
                        context={
                            "session_id": session_id,
                            "violations": [v.name for v in result.violations],
                        },
                    )

                if result.decision == PolicyDecision.MODIFY:
                    # Apply intervention if needed
                    if result.intervention_needed and self.injector:
                        logger.info(
                            f"Applying intervention from policy engine for session {session_id}: "
                            f"{result.intervention_needed}"
                        )
                        data = self.injector.inject(
                            data,
                            result.intervention_needed,
                            context=result.metadata or {},
                        )

                        # Log intervention via OTEL
                        if self.tracer:
                            self.tracer.log_intervention(
                                session_id=session_id,
                                intervention_name=result.intervention_needed,
                                context=result.metadata or {},
                            )

                    # Use modified request if provided
                    elif result.modified_request:
                        data = result.modified_request

            except WorkflowViolationError:
                raise
            except Exception as e:
                logger.error(f"Policy engine evaluation failed: {e}")
                if not self.settings.policy.fail_open:
                    raise WorkflowViolationError(
                        f"Policy engine evaluation failed: {e}",
                        context={"session_id": session_id},
                    )

        # Fallback: Check for pending intervention (backward compatibility)
        if session_id in self._pending_interventions:
            intervention = self._pending_interventions.pop(session_id)
            logger.info(
                f"Applying pending intervention for session {session_id}: {intervention.get('name')}"
            )

            if self.injector:
                data = self.injector.inject(
                    data,
                    intervention.get("name", "default"),
                    context=intervention.get("context", {}),
                )

            # Log intervention via OTEL
            if self.tracer:
                self.tracer.log_intervention(
                    session_id=session_id,
                    intervention_name=intervention.get("name"),
                    context=intervention.get("context", {}),
                )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallType,
    ) -> None:
        """
        Execute IN PARALLEL with LLM call.

        This hook runs concurrently with the LLM request, so it adds
        no latency to the critical path. It:

        1. Analyzes the PREVIOUS response (from message history)
        2. Evaluates through policy engine
        3. Records any pending intervention for the NEXT call

        Note: This hook can only REJECT requests (raise exception),
        not modify them. For corrections, we record the intervention
        and apply it in the next async_pre_call_hook.

        Args:
            data: Request data
            user_api_key_dict: User API key information
            call_type: Type of LLM call
        """
        session_id = self._extract_session_id(data)
        logger.debug(f"moderation_hook: session={session_id}")

        # Get the last assistant message (previous LLM response)
        messages = data.get("messages", [])
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            return  # No previous response to analyze

        # Try policy engine first
        policy_engine = await self._get_policy_engine()
        if policy_engine:
            try:
                result = await policy_engine.evaluate_response(
                    session_id=session_id,
                    response_data=last_assistant_msg,
                    request_data=data,
                    context={"messages": messages, "call_type": call_type},
                )

                # Log any violations via OTEL
                if result.violations and self.tracer:
                    for violation in result.violations:
                        self.tracer.log_deviation(
                            session_id=session_id,
                            constraint_name=violation.name,
                            severity=violation.severity,
                        )

                # If intervention needed, record for next call
                if result.intervention_needed:
                    self._pending_interventions[session_id] = {
                        "name": result.intervention_needed,
                        "context": result.metadata or {},
                        "strategy": self.settings.intervention.default_strategy,
                    }
                    logger.info(
                        f"Intervention scheduled for session {session_id}: "
                        f"{result.intervention_needed}"
                    )

                return  # Policy engine handled it

            except Exception as e:
                logger.error(f"Policy engine moderation failed: {e}")
                # Fall through to legacy tracker

        # Fallback to legacy tracker (backward compatibility)
        if not self.tracker:
            return

        result = await self.tracker.process_response(
            session_id=session_id,
            response=last_assistant_msg,
            context={"messages": messages},
        )

        # Log any constraint violations via OTEL
        if result.constraint_violations and self.tracer:
            for violation in result.constraint_violations:
                self.tracer.log_deviation(
                    session_id=session_id,
                    constraint_name=violation.constraint_name,
                    severity=getattr(violation, "severity", "warning"),
                )

        # If intervention needed, record for next call
        if result.intervention_needed:
            self._pending_interventions[session_id] = {
                "name": result.intervention_needed,
                "context": {
                    "current_state": result.classified_state,
                    "previous_state": result.previous_state,
                    "violations": [v.constraint_name for v in result.constraint_violations],
                },
                "strategy": self.settings.intervention.default_strategy,
            }

            logger.info(
                f"Intervention scheduled for session {session_id}: {result.intervention_needed}"
            )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Execute AFTER successful LLM response.

        This hook:
        1. Evaluates response through policy engine
        2. Updates state machine (for stateful engines)
        3. Completes the OTEL trace span

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response

        Returns:
            Response (potentially modified, but we don't modify here)
        """
        session_id = self._extract_session_id(data)

        policy_engine = await self._get_policy_engine()
        logger.info(
            f"post_call_success_hook: session={session_id}, "
            f"has_policy_engine={policy_engine is not None}, "
            f"has_tracker={self.tracker is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        # Evaluate response through policy engine
        policy_result = None
        if policy_engine:
            try:
                policy_result = await policy_engine.evaluate_response(
                    session_id=session_id,
                    response_data=response,
                    request_data=data,
                    context={"hook": "post_call_success"},
                )

                # Log state transition for stateful engines
                if self.tracer and policy_result.metadata:
                    prev_state = policy_result.metadata.get("previous_state")
                    curr_state = policy_result.metadata.get("current_state")
                    confidence = policy_result.metadata.get("classification_confidence", 0.0)

                    if prev_state and curr_state:
                        self.tracer.log_state_transition(
                            session_id=session_id,
                            previous_state=prev_state,
                            new_state=curr_state,
                            confidence=confidence,
                        )

                # Log violations
                if policy_result.violations and self.tracer:
                    for violation in policy_result.violations:
                        self.tracer.log_deviation(
                            session_id=session_id,
                            constraint_name=violation.name,
                            severity=violation.severity,
                        )

                # Schedule intervention for next call if needed
                if policy_result.intervention_needed:
                    self._pending_interventions[session_id] = {
                        "name": policy_result.intervention_needed,
                        "context": policy_result.metadata or {},
                        "strategy": self.settings.intervention.default_strategy,
                    }

            except Exception as e:
                logger.error(f"Policy engine post-call evaluation failed: {e}")

        # Fallback to legacy tracker (backward compatibility)
        tracking_result = None
        if not policy_engine and self.tracker:
            tracking_result = await self.tracker.process_response(
                session_id=session_id,
                response=response,
                context={"request_data": data},
            )

            logger.debug(
                f"State transition: {tracking_result.previous_state} -> "
                f"{tracking_result.classified_state} "
                f"(confidence={tracking_result.classification_confidence:.2f})"
            )

            # Log state transition via OTEL
            if self.tracer:
                self.tracer.log_state_transition(
                    session_id=session_id,
                    previous_state=tracking_result.previous_state or "unknown",
                    new_state=tracking_result.classified_state or "unknown",
                    confidence=tracking_result.classification_confidence,
                )

        # Log LLM call if no workflow tracking
        if self.tracer and not policy_result and not tracking_result:
            response_content = None
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            usage_info = None
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            self.tracer.log_llm_call(
                session_id=session_id,
                model=data.get("model", "unknown"),
                messages=data.get("messages", []),
                response_content=response_content,
                response_model=getattr(response, "model", None),
                usage=usage_info,
                metadata={"has_workflow": False, "hook": "post_call_success"},
            )

        return response

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
        **kwargs: Any,
    ) -> None:
        """
        Execute AFTER failed LLM call.

        Logs the failure.
        """
        session_id = self._extract_session_id(request_data)

        logger.warning(f"LLM call failed for session {session_id}: {original_exception}")

    # Synchronous hooks (for logging/metrics)

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        """Log before API call (sync)."""
        logger.debug(f"API call starting: model={model}")

    def log_post_api_call(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log after API call (sync)."""
        if end_time and start_time:
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"API call completed: duration={duration:.2f}s")
        else:
            logger.debug("API call completed")

    def log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log successful completion (sync)."""
        pass  # Handled in async hook

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log failed completion (sync)."""
        logger.error(f"LLM call failed: {response_obj}")

    # Async logging hooks (called by LiteLLM Router in library mode)

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Called AFTER successful LLM response (library/router mode).

        Uses policy engine for evaluation, falls back to legacy tracker.
        """
        session_id = self._extract_session_id(kwargs)

        policy_engine = await self._get_policy_engine()
        logger.info(
            f"async_log_success_event: session={session_id}, "
            f"has_policy_engine={policy_engine is not None}, "
            f"has_tracker={self.tracker is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        # Evaluate through policy engine
        policy_result = None
        if policy_engine:
            try:
                policy_result = await policy_engine.evaluate_response(
                    session_id=session_id,
                    response_data=response_obj,
                    request_data=kwargs,
                    context={"hook": "async_log_success"},
                )

                # Log state transition for stateful engines
                if self.tracer and policy_result.metadata:
                    prev_state = policy_result.metadata.get("previous_state")
                    curr_state = policy_result.metadata.get("current_state")
                    confidence = policy_result.metadata.get("classification_confidence", 0.0)

                    if prev_state and curr_state:
                        self.tracer.log_state_transition(
                            session_id=session_id,
                            previous_state=prev_state,
                            new_state=curr_state,
                            confidence=confidence,
                        )

                # Schedule intervention if needed
                if policy_result.intervention_needed:
                    self._pending_interventions[session_id] = {
                        "name": policy_result.intervention_needed,
                        "context": policy_result.metadata or {},
                        "strategy": self.settings.intervention.default_strategy,
                    }

            except Exception as e:
                logger.error(f"Policy engine async_log_success evaluation failed: {e}")

        # Fallback to legacy tracker
        tracking_result = None
        if not policy_engine and self.tracker:
            tracking_result = await self.tracker.process_response(
                session_id=session_id,
                response=response_obj,
                context={"request_data": kwargs},
            )

            logger.debug(
                f"State transition: {tracking_result.previous_state} -> "
                f"{tracking_result.classified_state} "
                f"(confidence={tracking_result.classification_confidence:.2f})"
            )

            if self.tracer:
                self.tracer.log_state_transition(
                    session_id=session_id,
                    previous_state=tracking_result.previous_state or "unknown",
                    new_state=tracking_result.classified_state or "unknown",
                    confidence=tracking_result.classification_confidence,
                )

        # Log LLM call if no workflow tracking
        if self.tracer and not policy_result and not tracking_result:
            response_content = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                first_choice = response_obj.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            usage_info = None
            if hasattr(response_obj, "usage") and response_obj.usage:
                usage_info = {
                    "prompt_tokens": getattr(response_obj.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response_obj.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response_obj.usage, "total_tokens", 0),
                }

            self.tracer.log_llm_call(
                session_id=session_id,
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                response_content=response_content,
                response_model=getattr(response_obj, "model", None),
                usage=usage_info,
                metadata={"has_workflow": False, "hook": "async_log_success"},
            )

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Called AFTER failed LLM call (library/router mode).
        """
        session_id = self._extract_session_id(kwargs)

        logger.warning(f"LLM call failed for session {session_id}: {response_obj}")
