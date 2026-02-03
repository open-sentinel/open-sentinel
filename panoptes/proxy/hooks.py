"""
Panoptes LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Applies pending async checker results from previous call
   - Runs sync PRE_CALL checkers
   - Modifies request if WARN with modified_data

2. async_moderation_hook: Runs IN PARALLEL with LLM call (non-blocking)
   - Legacy hook for backward compatibility
   - Records pending interventions for next call

3. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Runs sync POST_CALL checkers (can modify response on WARN)
   - Starts async checkers in background
   - Completes tracing

The hook system uses the Interceptor for clean checker orchestration:
- Sync checkers block and can modify request/response
- Async checkers run in background, results applied next request

Based on LiteLLM's CustomLogger API:
https://docs.litellm.ai/docs/observability/custom_callback
"""

import logging
from typing import Optional, Union, Dict, Any, Literal, List
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

from panoptes.config.settings import PanoptesSettings
from panoptes.policy.protocols import PolicyEngine
from panoptes.core.intervention.strategies import (
    WorkflowViolationError as CoreWorkflowViolationError,
)
from panoptes.core.interceptor import (
    Interceptor,
    Checker,
    CheckPhase,
    CheckerMode,
    PolicyEngineChecker,
    CheckDecision,
)

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

    Uses the Interceptor to orchestrate checkers:
    - Sync PRE_CALL checkers run before LLM call
    - Sync POST_CALL checkers run after LLM call (can modify response)
    - Async checkers run in background, results applied next request

    The callback maintains:
    - Interceptor instance with configured checkers
    - OpenTelemetry tracer for observability

    Thread-safety is ensured through asyncio locks for session state.
    """

    def __init__(self, settings: Optional[PanoptesSettings] = None):
        self.settings = settings or PanoptesSettings()

        # Interceptor (lazy initialized)
        self._interceptor: Optional[Interceptor] = None
        self._interceptor_initialized = False

        # Policy engine (lazy initialized) - kept for direct access if needed
        self._policy_engine: Optional[PolicyEngine] = None
        self._policy_engine_initialized = False

        # Workflow tracker (lazy initialized) - kept for backward compatibility
        self._tracker = None

        # Prompt injector (lazy initialized)
        self._injector = None

        # OpenTelemetry tracer for Panoptes events (lazy initialized)
        self._tracer = None

        logger.info("PanoptesCallback initialized")

    async def _get_interceptor(self) -> Optional[Interceptor]:
        """Lazy-load interceptor with configured checkers."""
        if self._interceptor_initialized:
            return self._interceptor

        try:
            policy_engine = await self._get_policy_engine()
            if not policy_engine:
                self._interceptor_initialized = True
                return None

            # Create checkers from policy engine
            checkers: List[Checker] = []

            # Sync PRE_CALL checker for request evaluation
            checkers.append(
                PolicyEngineChecker(
                    engine=policy_engine,
                    phase=CheckPhase.PRE_CALL,
                    mode=CheckerMode.SYNC,
                )
            )

            # Sync POST_CALL checker for response evaluation
            checkers.append(
                PolicyEngineChecker(
                    engine=policy_engine,
                    phase=CheckPhase.POST_CALL,
                    mode=CheckerMode.SYNC,
                )
            )

            self._interceptor = Interceptor(checkers)
            self._interceptor_initialized = True
            logger.info(f"Interceptor initialized with {len(checkers)} checkers")

        except Exception as e:
            logger.error(f"Failed to initialize interceptor: {e}")
            self._interceptor_initialized = True
            self._interceptor = None

        return self._interceptor

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

    def _get_workflow_path_for_fsm(self) -> Optional[str]:
        """Resolve workflow_path from settings or policy engine config."""
        if self.settings.workflow_path:
            return self.settings.workflow_path

        policy_config = self.settings.get_policy_config()
        engine_type = policy_config.get("type")
        engine_config = policy_config.get("config", {})

        if engine_type == "fsm":
            return engine_config.get("workflow_path")

        if engine_type == "composite":
            for engine in engine_config.get("engines", []):
                if engine.get("type") == "fsm":
                    return (engine.get("config") or {}).get("workflow_path")

        return None

    @property
    def tracker(self) -> Any:
        """Lazy-load tracker to avoid import issues (backward compatibility)."""
        workflow_path = self._get_workflow_path_for_fsm()
        if self._tracker is None and workflow_path:
            from panoptes.policy.engines.fsm import WorkflowParser, WorkflowTracker

            workflow = WorkflowParser.parse_file(workflow_path)
            self._tracker = WorkflowTracker(workflow)
        return self._tracker

    @property
    def injector(self) -> Any:
        """Lazy-load injector to avoid import issues."""
        workflow_path = self._get_workflow_path_for_fsm()
        if self._injector is None and workflow_path:
            from panoptes.policy.engines.fsm import WorkflowParser, PromptInjector

            workflow = WorkflowParser.parse_file(workflow_path)
            self._injector = PromptInjector(workflow)
        return self._injector

    @property
    def tracer(self) -> Any:
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
        1. Runs interceptor.run_pre_call() which:
           - Applies pending async results from previous call
           - Runs sync PRE_CALL checkers
        2. If FAIL, raises WorkflowViolationError
        3. If WARN with modified_data, applies modifications
        4. Logs via OTEL tracer

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

        interceptor = await self._get_interceptor()
        if interceptor:
            try:
                # Wrap in a trace block
                if self.tracer:
                    cm = self.tracer.trace_block(
                        "interceptor_pre_call",
                        session_id,
                        attributes={"hook": "pre_call"},
                        input_data=data.get("messages", []),
                        metadata={
                            "call_type": call_type,
                            "model": data.get("model", "unknown"),
                        },
                    )
                else:
                    from contextlib import nullcontext

                    cm = nullcontext()

                with cm as span:
                    result = await interceptor.run_pre_call(
                        session_id=session_id,
                        request_data=data,
                        user_request_id=str(id(data)),
                    )

                    # Set output on span
                    if span is not None:
                        import json

                        output_data = {
                            "allowed": result.allowed,
                            "num_results": len(result.results),
                            "has_modifications": result.modified_data is not None,
                        }
                        output_json = json.dumps(output_data, default=str)
                        span.set_attribute("output.value", output_json)
                        span.set_attribute("langfuse.span.output", output_json)

                # Handle result
                if not result.allowed:
                    # Find the failing result for context
                    fail_results = [
                        r for r in result.results if r.decision == CheckDecision.FAIL
                    ]
                    violations = []
                    message = "Request blocked by checker"
                    if fail_results:
                        for r in fail_results:
                            violations.extend(v.get("name", "unknown") for v in r.violations)
                        message = fail_results[0].message or message

                    logger.warning(
                        f"Request blocked for session {session_id}: {violations}"
                    )
                    raise WorkflowViolationError(
                        message,
                        context={
                            "session_id": session_id,
                            "violations": violations,
                        },
                    )

                # Apply modifications if any
                if result.modified_data:
                    data = result.modified_data

                    # Log intervention via OTEL
                    if self.tracer:
                        self.tracer.log_intervention(
                            session_id=session_id,
                            intervention_name="pre_call_modification",
                            context={"num_checkers": len(result.results)},
                        )

            except (WorkflowViolationError, CoreWorkflowViolationError):
                raise
            except Exception as e:
                logger.error(f"Interceptor pre_call failed: {e}")
                if not self.settings.policy.fail_open:
                    raise WorkflowViolationError(
                        f"Interceptor evaluation failed: {e}",
                        context={"session_id": session_id},
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

        This hook is kept for backward compatibility but most logic
        is now handled by async checkers in the Interceptor.

        Args:
            data: Request data
            user_api_key_dict: User API key information
            call_type: Type of LLM call
        """
        session_id = self._extract_session_id(data)
        logger.debug(f"moderation_hook: session={session_id}")

        # Moderation logic is now handled by async checkers in the interceptor
        # This hook is kept for backward compatibility with legacy tracker
        if not self._interceptor and self.tracker:
            messages = data.get("messages", [])
            last_assistant_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg
                    break

            if not last_assistant_msg:
                return

            # Legacy tracker handling would go here
            # But since we're using the interceptor, this is just a passthrough

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Execute AFTER successful LLM response.

        This hook:
        1. Runs interceptor.run_post_call() which:
           - Runs sync POST_CALL checkers (can modify response on WARN)
           - Starts async checkers in background
        2. If FAIL, raises WorkflowViolationError
        3. If WARN with modified_data, applies modifications to response
        4. Logs via OTEL tracer

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response

        Returns:
            Response (potentially modified)
        """
        session_id = self._extract_session_id(data)

        interceptor = await self._get_interceptor()
        policy_engine = await self._get_policy_engine()

        logger.info(
            f"post_call_success_hook: session={session_id}, "
            f"has_interceptor={interceptor is not None}, "
            f"has_tracker={self.tracker is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        if interceptor:
            try:
                # Extract response content for tracing
                response_content_for_trace = None
                if hasattr(response, "choices") and response.choices:
                    first_choice = response.choices[0]
                    if hasattr(first_choice, "message") and first_choice.message:
                        response_content_for_trace = first_choice.message.content
                    elif hasattr(first_choice, "text"):
                        response_content_for_trace = first_choice.text

                if self.tracer:
                    cm = self.tracer.trace_block(
                        "interceptor_post_call",
                        session_id,
                        attributes={"hook": "post_call_success"},
                        input_data={
                            "response": response_content_for_trace,
                            "messages": data.get("messages", []),
                        },
                        metadata={
                            "model": data.get("model", "unknown"),
                        },
                    )
                else:
                    from contextlib import nullcontext

                    cm = nullcontext()

                with cm as span:
                    result = await interceptor.run_post_call(
                        session_id=session_id,
                        request_data=data,
                        response_data=response,
                        user_request_id=str(id(data)),
                    )

                    # Set output on span
                    if span is not None:
                        import json

                        output_data = {
                            "allowed": result.allowed,
                            "num_results": len(result.results),
                            "has_modifications": result.modified_data is not None,
                        }
                        output_json = json.dumps(output_data, default=str)
                        span.set_attribute("output.value", output_json)
                        span.set_attribute("langfuse.span.output", output_json)

                # Handle result
                if not result.allowed:
                    fail_results = [
                        r for r in result.results if r.decision == CheckDecision.FAIL
                    ]
                    violations = []
                    message = "Response blocked by checker"
                    if fail_results:
                        for r in fail_results:
                            violations.extend(v.get("name", "unknown") for v in r.violations)
                        message = fail_results[0].message or message

                    logger.warning(
                        f"Response blocked for session {session_id}: {violations}"
                    )
                    raise WorkflowViolationError(
                        message,
                        context={
                            "session_id": session_id,
                            "violations": violations,
                        },
                    )

                # Apply modifications to response if any
                if result.modified_data:
                    # For response modifications, we may need to handle the response object
                    # The interceptor returns modifications as a dict
                    if isinstance(result.modified_data, dict):
                        # If it's a full response replacement
                        if "response" in result.modified_data:
                            response = result.modified_data["response"]
                        # Otherwise log that modifications were requested
                        else:
                            logger.info(
                                f"POST_CALL modifications requested for session {session_id}"
                            )

            except (WorkflowViolationError, CoreWorkflowViolationError):
                raise
            except Exception as e:
                logger.error(f"Interceptor post_call failed: {e}")

        # Fallback to legacy tracker (backward compatibility)
        elif self.tracker:
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

        # Log LLM call via OTEL
        if self.tracer:
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
                metadata={
                    "has_interceptor": interceptor is not None,
                    "hook": "post_call_success",
                },
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

        Uses interceptor for evaluation, falls back to legacy tracker.
        """
        session_id = self._extract_session_id(kwargs)

        interceptor = await self._get_interceptor()
        logger.info(
            f"async_log_success_event: session={session_id}, "
            f"has_interceptor={interceptor is not None}, "
            f"has_tracker={self.tracker is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        # NOTE: We skip interceptor evaluation here to avoid TimeoutErrors in the logging worker.
        # The logging worker has a short timeout and policy evaluation can take longer.
        # Evaluation is handled in `async_post_call_success_hook` which runs in the main flow.

        # Fallback to legacy tracker
        if not interceptor and self.tracker:
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

        # Log LLM call via OTEL
        if self.tracer:
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
                    "completion_tokens": getattr(
                        response_obj.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response_obj.usage, "total_tokens", 0),
                }

            self.tracer.log_llm_call(
                session_id=session_id,
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                response_content=response_content,
                response_model=getattr(response_obj, "model", None),
                usage=usage_info,
                metadata={
                    "has_interceptor": interceptor is not None,
                    "hook": "async_log_success",
                },
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
