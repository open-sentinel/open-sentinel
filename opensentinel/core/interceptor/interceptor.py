"""
Interceptor orchestrator.

Manages the execution of checkers at PRE_CALL and POST_CALL phases,
handles async checker task management, and applies modifications.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .checker import Checker
from opensentinel.policy.protocols import PolicyDecision

from .types import (
    CheckerContext,
    CheckerMode,
    CheckPhase,
    CheckResult,
    InterceptionResult,
)

logger = logging.getLogger(__name__)


class Interceptor:
    """
    Orchestrator for running checkers during LLM request lifecycle.

    Manages:
    - Sync checkers that block and must complete (ALLOW or DENY only)
    - Async checkers that run in background with results applied next request
    - Pending async results per session
    - Modification merging for deferred MODIFY decisions
    """

    def __init__(self, checkers: List[Checker]):
        """
        Initialize interceptor with a list of checkers.

        Args:
            checkers: List of Checker instances to run
        """
        # Categorize checkers into 4 buckets by phase and mode
        self._sync_pre_call: List[Checker] = []
        self._sync_post_call: List[Checker] = []
        self._async_pre_call: List[Checker] = []
        self._async_post_call: List[Checker] = []

        for checker in checkers:
            if checker.mode == CheckerMode.ASYNC:
                if checker.phase == CheckPhase.PRE_CALL:
                    self._async_pre_call.append(checker)
                else:
                    self._async_post_call.append(checker)
            elif checker.phase == CheckPhase.PRE_CALL:
                self._sync_pre_call.append(checker)
            else:  # POST_CALL + SYNC
                self._sync_post_call.append(checker)

        # session_id -> pending async results
        self._pending_async: Dict[str, List[CheckResult]] = {}

        # session_id -> running async tasks
        self._running_tasks: Dict[str, List[asyncio.Task[CheckResult]]] = {}

        logger.info(
            f"Interceptor initialized: {len(self._sync_pre_call)} sync pre-call, "
            f"{len(self._sync_post_call)} sync post-call, "
            f"{len(self._async_pre_call)} async pre-call, "
            f"{len(self._async_post_call)} async post-call"
        )

    async def run_pre_call(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        user_request_id: str = "",
    ) -> InterceptionResult:
        """
        Run PRE_CALL phase.

        1. Apply pending async results from previous request
           - DENY -> block this request
           - MODIFY -> merge modified_data into request
        2. Run sync PRE_CALL checkers (gate only: ALLOW or DENY)
        3. Start async PRE_CALL checkers in background
        4. Return result with possibly modified request_data

        Args:
            session_id: Session identifier
            request_data: The LLM request data
            user_request_id: Optional request ID for tracing

        Returns:
            InterceptionResult with allowed flag and modified_data
        """
        all_results: List[CheckResult] = []
        modified_data = dict(request_data)

        # Step 1: Apply pending async results
        pending_results = self._collect_completed_async(session_id)
        for result in pending_results:
            all_results.append(result)

            if result.decision == PolicyDecision.DENY:
                logger.warning(
                    f"Request blocked by async checker '{result.checker_name}': "
                    f"{result.message}"
                )
                return InterceptionResult(
                    allowed=False,
                    modified_data=None,
                    results=all_results,
                )

            if result.decision == PolicyDecision.MODIFY and result.modified_data:
                logger.info(
                    f"Applying async modification from '{result.checker_name}'"
                )
                modified_data = self._merge_modifications(
                    modified_data, result.modified_data
                )

        # Step 2: Run sync PRE_CALL checkers (gate only: ALLOW or DENY)
        context = CheckerContext(
            session_id=session_id,
            user_request_id=user_request_id,
            request_data=modified_data,
            response_data=None,
        )

        for checker in self._sync_pre_call:
            try:
                result = await checker.check(context)
                all_results.append(result)

                if result.decision == PolicyDecision.DENY:
                    logger.warning(
                        f"Request blocked by sync checker '{result.checker_name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        modified_data=None,
                        results=all_results,
                    )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                error_result = CheckResult(
                    decision=PolicyDecision.DENY,
                    checker_name=checker.name,
                    message=f"Checker error: {e}",
                )
                all_results.append(error_result)
                return InterceptionResult(
                    allowed=False,
                    modified_data=None,
                    results=all_results,
                )

        # Step 3: Start async PRE_CALL checkers in background
        if self._async_pre_call:
            async_context = CheckerContext(
                session_id=session_id,
                user_request_id=user_request_id,
                request_data=modified_data,
                response_data=None,
            )
            for checker in self._async_pre_call:
                self._start_async_checker(checker, async_context)

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data if modified_data != request_data else None,
            results=all_results,
        )

    async def run_post_call(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        response_data: Any,
        user_request_id: str = "",
    ) -> InterceptionResult:
        """
        Run POST_CALL phase.

        1. Run sync POST_CALL checkers (gate only: ALLOW or DENY)
        2. Start async POST_CALL checkers in background (don't wait)
        3. Return result

        Args:
            session_id: Session identifier
            request_data: The original LLM request data
            response_data: The LLM response
            user_request_id: Optional request ID for tracing

        Returns:
            InterceptionResult with allowed flag
        """
        all_results: List[CheckResult] = []

        # Step 1: Run sync POST_CALL checkers (gate only: ALLOW or DENY)
        context = CheckerContext(
            session_id=session_id,
            user_request_id=user_request_id,
            request_data=request_data,
            response_data=response_data,
        )

        for checker in self._sync_post_call:
            try:
                result = await checker.check(context)
                all_results.append(result)

                if result.decision == PolicyDecision.DENY:
                    logger.warning(
                        f"Response blocked by sync checker '{result.checker_name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        modified_data=None,
                        results=all_results,
                    )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                error_result = CheckResult(
                    decision=PolicyDecision.DENY,
                    checker_name=checker.name,
                    message=f"Checker error: {e}",
                )
                all_results.append(error_result)
                return InterceptionResult(
                    allowed=False,
                    modified_data=None,
                    results=all_results,
                )

        # Step 2: Start async POST_CALL checkers in background
        if self._async_post_call:
            async_context = CheckerContext(
                session_id=session_id,
                user_request_id=user_request_id,
                request_data=request_data,
                response_data=response_data,
            )
            for checker in self._async_post_call:
                self._start_async_checker(checker, async_context)

        return InterceptionResult(
            allowed=True,
            modified_data=None,
            results=all_results,
        )

    def _collect_completed_async(self, session_id: str) -> List[CheckResult]:
        """
        Collect results from completed async tasks for a session.

        Removes completed tasks and returns their results.
        """
        results: List[CheckResult] = []

        # Get pending results stored from previous collection
        if session_id in self._pending_async:
            results.extend(self._pending_async.pop(session_id))

        # Check running tasks
        if session_id in self._running_tasks:
            tasks = self._running_tasks[session_id]
            still_running: List[asyncio.Task[CheckResult]] = []

            for task in tasks:
                if task.done():
                    try:
                        result = task.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Async checker task failed: {e}")
                        results.append(
                            CheckResult(
                                decision=PolicyDecision.DENY,
                                checker_name="async_task_error",
                                message=f"Async task error: {e}",
                            )
                        )
                else:
                    still_running.append(task)

            if still_running:
                self._running_tasks[session_id] = still_running
            else:
                del self._running_tasks[session_id]

        return results

    def _start_async_checker(
        self, checker: Checker, context: CheckerContext
    ) -> None:
        """
        Start an async checker task in the background.

        The task will be collected in the next run_pre_call.
        """
        session_id = context.session_id

        async def run_checker() -> CheckResult:
            try:
                return await checker.check(context)
            except Exception as e:
                logger.error(f"Async checker '{checker.name}' failed: {e}")
                return CheckResult(
                    decision=PolicyDecision.DENY,
                    checker_name=checker.name,
                    message=f"Async checker error: {e}",
                )

        task = asyncio.create_task(run_checker())

        if session_id not in self._running_tasks:
            self._running_tasks[session_id] = []
        self._running_tasks[session_id].append(task)

        logger.debug(f"Started async checker '{checker.name}' for session {session_id}")

    def _merge_modifications(
        self, base: Dict[str, Any], modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge modifications into base request data.

        Handles special cases like messages list appending.
        """
        result = dict(base)

        for key, value in modifications.items():
            if key == "messages" and isinstance(value, list):
                # Append new messages to existing
                existing = result.get("messages", [])
                result["messages"] = existing + value
            elif key == "system_prompt_append" and isinstance(value, str):
                # Append to system message if exists, or add new one
                messages = result.get("messages", [])
                system_msg_idx = None
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        system_msg_idx = i
                        break

                if system_msg_idx is not None:
                    messages[system_msg_idx]["content"] += "\n" + value
                else:
                    messages.insert(0, {"role": "system", "content": value})
                result["messages"] = messages
            else:
                result[key] = value

        return result

    async def cleanup_session(self, session_id: str) -> None:
        """
        Clean up resources for a session.

        Cancels any running async tasks and clears pending results.
        """
        if session_id in self._running_tasks:
            for task in self._running_tasks[session_id]:
                if not task.done():
                    task.cancel()
            del self._running_tasks[session_id]

        if session_id in self._pending_async:
            del self._pending_async[session_id]

        logger.debug(f"Cleaned up session {session_id}")

    async def shutdown(self) -> None:
        """
        Shutdown the interceptor.

        Cancels all running async tasks.
        """
        for session_id in list(self._running_tasks.keys()):
            await self.cleanup_session(session_id)

        logger.info("Interceptor shutdown complete")
