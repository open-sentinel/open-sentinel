"""
Generic intervention strategies for policy engines.

This module provides the intervention infrastructure that can be used
by any policy engine (FSM, NeMo, custom engines, etc.).

Strategies define HOW to modify LLM requests when policy violations
are detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message
2. USER_MESSAGE_INJECT: Add user message with guidance
3. CONTEXT_REMINDER: Add assistant context reminder
4. HARD_BLOCK: Reject request entirely
"""

from opensentinel.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
    InterventionStrategy,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    ContextReminderStrategy,
    HardBlockStrategy,
    WorkflowViolationError,
    STRATEGY_REGISTRY,
    get_strategy,
    merge_by_strategy,
)

__all__ = [
    "StrategyType",
    "InterventionConfig",
    "InterventionStrategy",
    "SystemPromptAppendStrategy",
    "UserMessageInjectStrategy",
    "ContextReminderStrategy",
    "HardBlockStrategy",
    "WorkflowViolationError",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "merge_by_strategy",
]
