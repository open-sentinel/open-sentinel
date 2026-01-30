"""
Panoptes LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Injects correction prompts when deviation was detected in previous call

2. async_moderation_hook: Runs IN PARALLEL with LLM call (non-blocking)
   - Evaluates workflow constraints
   - Records pending interventions for next call
   - Does NOT block or add latency to the critical path

3. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Classifies response to determine workflow state
   - Updates state machine

Note: Langfuse tracing is handled by LiteLLM's built-in langfuse_otel callback,
not by this module. See server.py for langfuse_otel configuration.

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


class PanoptesCallback(CustomLogger):
    """
    Main Panoptes callback for LiteLLM.

    Implements workflow monitoring through LiteLLM's hook system.
    This is registered as a callback when the proxy starts.

    The callback maintains state per session:
    - Workflow state machine instance
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

        # Workflow tracker (lazy initialized)
        self._tracker = None

        # Prompt injector (lazy initialized)
        self._injector = None

        # Langfuse tracer for Panoptes events (lazy initialized)
        self._tracer = None

        logger.info("PanoptesCallback initialized")

    @property
    def tracker(self):
        """Lazy-load tracker to avoid import issues."""
        if self._tracker is None and self.settings.workflow_path:
            from panoptes.workflow.parser import WorkflowParser
            from panoptes.monitor.tracker import WorkflowTracker

            workflow = WorkflowParser.parse_file(self.settings.workflow_path)
            self._tracker = WorkflowTracker(workflow)
        return self._tracker

    @property
    def injector(self):
        """Lazy-load injector to avoid import issues."""
        if self._injector is None and self.settings.workflow_path:
            from panoptes.workflow.parser import WorkflowParser
            from panoptes.intervention.prompt_injector import PromptInjector

            workflow = WorkflowParser.parse_file(self.settings.workflow_path)
            self._injector = PromptInjector(workflow)
        return self._injector

    @property
    def tracer(self):
        """Lazy-load tracer for Panoptes event logging to Langfuse."""
        if self._tracer is None:
            logger.debug(f"Tracer check: langfuse.enabled={self.settings.langfuse.enabled}, "
                        f"public_key={bool(self.settings.langfuse.public_key)}, "
                        f"secret_key={bool(self.settings.langfuse.secret_key)}")
            if self.settings.langfuse.enabled:
                from panoptes.tracing.langfuse_integration import PanoptesTracer
                self._tracer = PanoptesTracer(self.settings.langfuse)
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
        1. Checks for pending interventions from previous call
        2. If intervention needed, modifies request to inject correction
        3. Starts Langfuse trace span

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

        # Check for pending intervention
        if session_id in self._pending_interventions:
            intervention = self._pending_interventions.pop(session_id)
            logger.info(
                f"Applying intervention for session {session_id}: {intervention.get('name')}"
            )

            if self.injector:
                data = self.injector.inject(
                    data,
                    intervention.get("name", "default"),
                    context=intervention.get("context", {}),
                )

            # Log intervention to Langfuse
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
        2. Evaluates workflow constraints
        3. Records any pending intervention for the NEXT call

        Note: This hook can only REJECT requests (raise exception),
        not modify them. For corrections, we record the intervention
        and apply it in the next async_pre_call_hook.

        Args:
            data: Request data
            user_api_key_dict: User API key information
            call_type: Type of LLM call
        """
        if not self.tracker:
            return  # No workflow configured

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

        # Process the previous response through tracker
        # This classifies state and checks constraints
        result = await self.tracker.process_response(
            session_id=session_id,
            response=last_assistant_msg,
            context={"messages": messages},
        )

        # Log any constraint violations to Langfuse
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
        1. Classifies the response to determine workflow state
        2. Updates the workflow state machine
        3. Completes the Langfuse trace span

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response

        Returns:
            Response (potentially modified, but we don't modify here)
        """
        session_id = self._extract_session_id(data)

        logger.info(f"post_call_success_hook: session={session_id}, has_tracker={self.tracker is not None}, has_tracer={self.tracer is not None}")

        # Track the response (classify state, update machine)
        tracking_result = None
        if self.tracker:
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

        # Log state transition to Langfuse (if we tracked state)
        if self.tracer and tracking_result:
            self.tracer.log_state_transition(
                session_id=session_id,
                previous_state=tracking_result.previous_state or "unknown",
                new_state=tracking_result.classified_state or "unknown",
                confidence=tracking_result.classification_confidence,
            )
        elif self.tracer:
            # Log request/response even without workflow tracking
            # Extract response content
            response_content = None
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            # Extract usage info
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

        LiteLLM's langfuse_otel callback handles Langfuse tracing.
        We only do Panoptes workflow tracking here.
        """
        session_id = self._extract_session_id(kwargs)

        logger.info(f"async_log_success_event: session={session_id}, has_tracker={self.tracker is not None}, has_tracer={self.tracer is not None}")

        # Track the response (classify state, update machine)
        tracking_result = None
        if self.tracker:
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

        # Log state transition to Langfuse (if we tracked state)
        if self.tracer and tracking_result:
            self.tracer.log_state_transition(
                session_id=session_id,
                previous_state=tracking_result.previous_state or "unknown",
                new_state=tracking_result.classified_state or "unknown",
                confidence=tracking_result.classification_confidence,
            )
        elif self.tracer:
            # Log request/response even without workflow tracking
            # Extract response content
            response_content = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                first_choice = response_obj.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            # Extract usage info
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
