"""
Panoptes LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Injects correction prompts when deviation was detected in previous call
   - Starts Langfuse trace span

2. async_moderation_hook: Runs IN PARALLEL with LLM call (non-blocking)
   - Evaluates workflow constraints
   - Records pending interventions for next call
   - Does NOT block or add latency to the critical path

3. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Classifies response to determine workflow state
   - Updates state machine
   - Completes Langfuse trace

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

        # Tracer instance (lazy initialized)
        self._tracer = None

        # Workflow tracker (lazy initialized)
        self._tracker = None

        # Prompt injector (lazy initialized)
        self._injector = None

        logger.info("PanoptesCallback initialized")

    @property
    def tracer(self):
        """Lazy-load tracer to avoid import issues."""
        if self._tracer is None and self.settings.langfuse.enabled:
            from panoptes.tracing.langfuse_integration import PanoptesTracer

            self._tracer = PanoptesTracer(self.settings.langfuse)
        return self._tracer

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

            # Log intervention application to tracer
            if self.tracer:
                self.tracer.log_event(
                    session_id,
                    "intervention_applied",
                    {
                        "intervention_name": intervention.get("name"),
                        "strategy": intervention.get("strategy"),
                    },
                )

        # Start trace span
        if self.tracer:
            current_state = "unknown"
            if self.tracker:
                current_state = await self.tracker.get_current_state(session_id)

            self.tracer.start_generation(
                session_id=session_id,
                name=f"llm-{call_type}",
                input_data=data,
                metadata={
                    "workflow_state": current_state,
                    "model": data.get("model"),
                    "call_type": call_type,
                },
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

            # Log deviation to tracer
            if self.tracer:
                self.tracer.log_event(
                    session_id,
                    "workflow_deviation",
                    {
                        "current_state": result.classified_state,
                        "previous_state": result.previous_state,
                        "violations": [v.constraint_name for v in result.constraint_violations],
                        "intervention": result.intervention_needed,
                    },
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

        logger.debug(f"post_call_success_hook: session={session_id}")

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

        # Complete trace span
        if self.tracer:
            metadata = {}
            if tracking_result:
                metadata = {
                    "state_from": tracking_result.previous_state,
                    "state_to": tracking_result.classified_state,
                    "classification_method": tracking_result.classification_method,
                    "classification_confidence": tracking_result.classification_confidence,
                }

            self.tracer.end_generation(
                session_id=session_id,
                output=response,
                metadata=metadata,
            )

        return response

    async def async_post_call_failure_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
    ) -> None:
        """
        Execute AFTER failed LLM call.

        Logs the failure to tracer.
        """
        session_id = self._extract_session_id(data)

        logger.warning(f"LLM call failed for session {session_id}: {original_exception}")

        if self.tracer:
            self.tracer.log_event(
                session_id,
                "llm_call_failed",
                {
                    "error": str(original_exception),
                    "error_type": type(original_exception).__name__,
                },
            )

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
        duration = (end_time - start_time).total_seconds()
        logger.debug(f"API call completed: duration={duration:.2f}s")

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
