"""
Checker abstract base class.

Defines the interface that all checkers must implement.
"""

from abc import ABC, abstractmethod

from .types import CheckPhase, CheckerMode, CheckResult, CheckerContext


class Checker(ABC):
    """
    Base class for all checkers.

    Checkers evaluate requests or responses and return a CheckResult.
    They can be configured to run at different phases (PRE_CALL, POST_CALL)
    and in different modes (SYNC, ASYNC).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this checker."""
        ...

    @property
    @abstractmethod
    def phase(self) -> CheckPhase:
        """When this checker runs (PRE_CALL or POST_CALL)."""
        ...

    @property
    @abstractmethod
    def mode(self) -> CheckerMode:
        """How this checker executes (SYNC or ASYNC)."""
        ...

    @abstractmethod
    async def check(self, context: CheckerContext) -> CheckResult:
        """
        Execute the check.

        Args:
            context: CheckerContext with session_id, request_data, and optionally response_data

        Returns:
            CheckResult with decision (PASS, WARN, FAIL) and optional modified_data
        """
        ...
