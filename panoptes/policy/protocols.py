"""
Core protocol definitions for policy engines.

These protocols define the contract that all policy engines must implement,
enabling pluggable policy evaluation while maintaining a consistent API.
"""

from typing import Protocol, Optional, Dict, Any, List, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class PolicyDecision(Enum):
    """Result of policy evaluation."""

    ALLOW = "allow"      # Action is permitted
    DENY = "deny"        # Action is blocked
    MODIFY = "modify"    # Action allowed but request should be modified
    WARN = "warn"        # Action allowed but logged as warning


@dataclass
class PolicyViolation:
    """Details of a policy violation."""

    name: str                                          # Violation identifier
    severity: str                                      # "warning", "error", "critical"
    message: str                                       # Human-readable description
    intervention: Optional[str] = None                 # Suggested intervention name
    metadata: Optional[Dict[str, Any]] = None          # Additional context

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating a request/response against policies."""

    decision: PolicyDecision
    violations: List[PolicyViolation] = field(default_factory=list)
    intervention_needed: Optional[str] = None
    modified_request: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StateClassificationResult:
    """Result of classifying a response to a state/intent."""

    state_name: str
    confidence: float
    method: str
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PolicyEngine(ABC):
    """
    Base class for all policy engines.

    Policy engines evaluate requests/responses against configured policies
    and determine what interventions (if any) are needed.

    Implementations should be registered using the @register_engine decorator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        ...

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Type identifier (e.g., 'fsm', 'nemo', 'composite')."""
        ...

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the engine with configuration.

        Args:
            config: Engine-specific configuration dictionary
        """
        ...

    @abstractmethod
    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate an incoming request against policies.

        Called BEFORE the LLM call. Can modify, allow, or block the request.

        Args:
            session_id: Unique session identifier
            request_data: The LLM request data (messages, model, etc.)
            context: Additional context for evaluation

        Returns:
            PolicyEvaluationResult with decision and any violations
        """
        ...

    @abstractmethod
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate an LLM response against policies.

        Called AFTER the LLM call. Records violations for potential
        intervention on next call.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response
            request_data: The original request data
            context: Additional context for evaluation

        Returns:
            PolicyEvaluationResult with decision and any violations
        """
        ...

    @abstractmethod
    async def get_session_state(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get current session state for debugging/tracing.

        Args:
            session_id: Unique session identifier

        Returns:
            Session state dictionary or None if session doesn't exist
        """
        ...

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        """
        Reset session state.

        Args:
            session_id: Unique session identifier
        """
        ...

    async def shutdown(self) -> None:
        """
        Cleanup resources.

        Override in subclasses that need cleanup.
        """
        pass


class StatefulPolicyEngine(PolicyEngine):
    """
    Policy engine that tracks state across requests.

    Extends PolicyEngine with state classification capabilities.
    Used by FSM and similar state-machine-based engines.
    """

    @abstractmethod
    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: Optional[str] = None,
    ) -> StateClassificationResult:
        """
        Classify a response to a state.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response to classify
            current_state: Current state (optional, will be looked up if not provided)

        Returns:
            StateClassificationResult with detected state and confidence
        """
        ...

    @abstractmethod
    async def get_current_state(self, session_id: str) -> str:
        """
        Get current state name for session.

        Args:
            session_id: Unique session identifier

        Returns:
            Current state name
        """
        ...

    @abstractmethod
    async def get_state_history(self, session_id: str) -> List[str]:
        """
        Get state transition history.

        Args:
            session_id: Unique session identifier

        Returns:
            List of state names in chronological order
        """
        ...

    @abstractmethod
    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """
        Get valid next states from current state.

        Args:
            session_id: Unique session identifier

        Returns:
            List of valid state names that can be transitioned to
        """
        ...


@runtime_checkable
class InterventionProvider(Protocol):
    """
    Protocol for providing interventions when policies are violated.

    Separated from policy evaluation to allow flexible intervention
    strategies across different policy engines.
    """

    def get_intervention(
        self,
        violation: PolicyViolation,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Get intervention configuration for a violation.

        Args:
            violation: The policy violation
            context: Additional context

        Returns:
            Dict with intervention config:
            - name: Intervention name
            - strategy: How to apply (system_prompt, user_message, etc.)
            - content: Intervention content/template

            Or None if no intervention available
        """
        ...

    def apply_intervention(
        self,
        request_data: Dict[str, Any],
        intervention: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply an intervention to request data.

        Args:
            request_data: The request to modify
            intervention: Intervention configuration
            context: Additional context

        Returns:
            Modified request data
        """
        ...
