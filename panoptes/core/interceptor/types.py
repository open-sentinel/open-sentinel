"""
Core types for the interceptor system.

Defines the enums and dataclasses used throughout the interceptor module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from panoptes.policy.protocols import PolicyViolation


class CheckPhase(Enum):
    """When a checker runs in the request lifecycle."""

    PRE_CALL = "pre_call"    # Before LLM call
    POST_CALL = "post_call"  # After LLM call


class CheckerMode(Enum):
    """How a checker executes."""

    SYNC = "sync"    # Blocking, must complete before proceeding
    ASYNC = "async"  # Background, results applied on next request


class CheckDecision(Enum):
    """Result of a check operation."""

    PASS = "pass"  # Allowed, no changes
    WARN = "warn"  # Allowed, may have modified_data
    FAIL = "fail"  # Blocked


@dataclass
class CheckResult:
    """Result returned by a checker."""

    decision: CheckDecision
    checker_name: str
    modified_data: Optional[Dict[str, Any]] = None  # For WARN passthrough
    violations: List[PolicyViolation] = field(default_factory=list)
    message: Optional[str] = None


@dataclass
class CheckerContext:
    """Context passed to checkers."""

    session_id: str
    user_request_id: str  # Single ID for tracing
    request_data: Dict[str, Any]
    response_data: Optional[Any] = None


@dataclass
class InterceptionResult:
    """Result of running interceptor pre_call or post_call."""

    allowed: bool
    modified_data: Optional[Dict[str, Any]] = None
    results: List[CheckResult] = field(default_factory=list)
