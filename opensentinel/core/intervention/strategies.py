"""
Intervention strategies for workflow correction.

Strategies define HOW to modify LLM requests when deviation is detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message
   - Least disruptive, preserves conversation flow
   - Best for gentle guidance

2. USER_MESSAGE_INJECT: Add user message with guidance
   - More visible to the model
   - Good for important corrections

3. CONTEXT_REMINDER: Add assistant context reminder
   - Simulates the assistant "remembering" something
   - Useful for complex multi-step workflows

4. HARD_BLOCK: Reject request entirely
   - Most severe, use for critical violations
   - Returns error to client
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of intervention strategies."""

    SYSTEM_PROMPT_APPEND = "system_prompt_append"
    USER_MESSAGE_INJECT = "user_message_inject"
    CONTEXT_REMINDER = "context_reminder"
    HARD_BLOCK = "hard_block"


@dataclass
class InterventionConfig:
    """Configuration for an intervention."""

    strategy_type: StrategyType
    message_template: str
    priority: int = 0  # Higher = more important
    max_applications: int = 3  # Max times to apply before escalating
    escalation_strategy: Optional[StrategyType] = None


class InterventionStrategy(ABC):
    """
    Base class for intervention strategies.

    Strategies modify LLM request data to guide the agent
    back to the expected workflow path.
    """

    @abstractmethod
    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        """
        Apply intervention to request data.

        Args:
            data: LLM request data (messages, model, etc.)
            config: Intervention configuration
            context: Additional context (states, violations, etc.)

        Returns:
            Modified request data
        """
        pass

    @staticmethod
    @abstractmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        """
        Merge an intervention value into a messages list.

        Low-level operation used when applying deferred interventions
        from async checkers. Unlike apply(), takes a pre-formatted string
        and operates directly on messages.

        Args:
            messages: The messages list (will be copied, not mutated).
            value: The pre-formatted intervention text.

        Returns:
            New messages list with the intervention applied.
        """
        pass

    @staticmethod
    def format_message(template: str, context: Dict[str, Any]) -> str:
        """Format message template with context."""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key in template: {e}")
            # Return template with unfilled placeholders rather than failing
            return template


class SystemPromptAppendStrategy(InterventionStrategy):
    """
    Append correction guidance to system message.

    This is the least disruptive strategy - it adds guidance
    to the system message without altering the conversation flow.

    Example output:
    ```
    System: You are a helpful assistant.

    [WORKFLOW GUIDANCE]: You must verify the customer's identity
    before performing any account actions.
    ```
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        messages = [dict(m) for m in messages]

        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        guidance = f"\n\n[WORKFLOW GUIDANCE]: {value}"

        if system_idx is not None:
            messages[system_idx]["content"] = (
                messages[system_idx].get("content", "") + guidance
            )
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"[WORKFLOW GUIDANCE]: {value}",
            })

        return messages

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        data = dict(data)
        correction = self.format_message(config.message_template, context)
        data["messages"] = self.merge(data.get("messages", []), correction)
        logger.debug("Applied system_prompt_append intervention")
        return data


class UserMessageInjectStrategy(InterventionStrategy):
    """
    Inject a user message with guidance.

    More visible than system prompt, appears as if the user
    is providing additional instructions.

    Example output:
    ```
    User: [System Note] Before proceeding, please verify
    the customer's identity as required by the workflow.
    ```
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        messages = list(messages)
        guidance = {"role": "user", "content": f"[System Note]: {value}"}

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            messages.insert(last_user_idx, guidance)
        else:
            messages.append(guidance)

        return messages

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        data = dict(data)
        correction = self.format_message(config.message_template, context)
        data["messages"] = self.merge(data.get("messages", []), correction)
        logger.debug("Applied user_message_inject intervention")
        return data


class ContextReminderStrategy(InterventionStrategy):
    """
    Add an assistant context reminder.

    Simulates the assistant "remembering" important context.
    Useful for maintaining workflow state awareness.

    Example output:
    ```
    Assistant: [Context reminder: I need to verify the customer's
    identity before I can help with account changes.]
    ```
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        messages = list(messages)
        reminder = {"role": "assistant", "content": f"[Context reminder: {value}]"}

        if messages:
            messages.insert(-1, reminder)
        else:
            messages.append(reminder)

        return messages

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        data = dict(data)
        correction = self.format_message(config.message_template, context)
        data["messages"] = self.merge(data.get("messages", []), correction)
        logger.debug("Applied context_reminder intervention")
        return data


class HardBlockStrategy(InterventionStrategy):
    """
    Block the request entirely.

    Most severe intervention - raises an exception that
    prevents the LLM call from proceeding.

    Use for critical violations where continuing would be harmful.
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        raise WorkflowViolationError(
            f"Workflow violation blocked: {value}",
            context={},
        )

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        correction = self.format_message(config.message_template, context)

        logger.warning(f"Hard block intervention: {correction}")

        # Raise exception to block the request
        raise WorkflowViolationError(
            f"Workflow violation blocked: {correction}",
            context=context,
        )


class WorkflowViolationError(Exception):
    """Exception raised when a workflow violation blocks a request."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


# Strategy registry
STRATEGY_REGISTRY: Dict[StrategyType, InterventionStrategy] = {
    StrategyType.SYSTEM_PROMPT_APPEND: SystemPromptAppendStrategy(),
    StrategyType.USER_MESSAGE_INJECT: UserMessageInjectStrategy(),
    StrategyType.CONTEXT_REMINDER: ContextReminderStrategy(),
    StrategyType.HARD_BLOCK: HardBlockStrategy(),
}


def get_strategy(strategy_type: StrategyType) -> InterventionStrategy:
    """Get strategy instance by type."""
    strategy = STRATEGY_REGISTRY.get(strategy_type)
    if not strategy:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    return strategy


def merge_by_strategy(
    strategy_key: str, messages: List[Dict[str, Any]], value: str
) -> List[Dict[str, Any]]:
    """
    Look up a strategy by its string key and apply its merge operation.

    Args:
        strategy_key: Strategy type value string (e.g., "system_prompt_append").
        messages: Messages list to modify.
        value: Pre-formatted intervention text.

    Returns:
        New messages list with intervention applied.

    Raises:
        ValueError: If strategy_key is not recognized.
        WorkflowViolationError: If strategy is hard_block.
    """
    strategy_type = StrategyType(strategy_key)
    return STRATEGY_REGISTRY[strategy_type].merge(messages, value)
