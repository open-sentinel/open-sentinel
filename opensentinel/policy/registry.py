"""
Policy engine registry for dynamic engine loading.

Provides a central registry for policy engine implementations,
enabling dynamic selection and instantiation of engines.
"""

from typing import Dict, Type, Optional, Any, List, Callable
import logging

from opensentinel.policy.protocols import PolicyEngine

logger = logging.getLogger(__name__)


class PolicyEngineRegistry:
    """
    Registry for policy engine implementations.

    Allows dynamic registration and lookup of policy engines.
    Engines are registered by type identifier (e.g., 'fsm', 'nemo').

    Usage:
        # Registration (usually via decorator)
        PolicyEngineRegistry.register("fsm", FSMPolicyEngine)

        # Lookup
        engine_class = PolicyEngineRegistry.get("fsm")

        # Factory
        engine = PolicyEngineRegistry.create("fsm", config)
        await engine.initialize(config)
    """

    _engines: Dict[str, Type[PolicyEngine]] = {}

    @classmethod
    def register(cls, engine_type: str, engine_class: Type[PolicyEngine]) -> None:
        """
        Register a policy engine class.

        Args:
            engine_type: Type identifier (e.g., 'fsm', 'nemo')
            engine_class: The engine class to register
        """
        if engine_type in cls._engines:
            logger.warning(
                f"Overwriting existing policy engine registration: {engine_type}"
            )
        cls._engines[engine_type] = engine_class
        logger.debug(f"Registered policy engine: {engine_type}")

    @classmethod
    def get(cls, engine_type: str) -> Optional[Type[PolicyEngine]]:
        """
        Get a policy engine class by type.

        Args:
            engine_type: Type identifier

        Returns:
            Engine class or None if not found
        """
        return cls._engines.get(engine_type)

    @classmethod
    def create(cls, engine_type: str) -> PolicyEngine:
        """
        Create a policy engine instance.

        Note: The engine must be initialized separately by calling
        `await engine.initialize(config)`.

        Args:
            engine_type: Type identifier

        Returns:
            Uninitialized engine instance

        Raises:
            ValueError: If engine type is not registered
        """
        engine_class = cls.get(engine_type)
        if not engine_class:
            available = ", ".join(cls._engines.keys()) or "none"
            raise ValueError(
                f"Unknown policy engine type: '{engine_type}'. "
                f"Available engines: {available}"
            )

        return engine_class()

    @classmethod
    async def create_and_initialize(
        cls,
        engine_type: str,
        config: Dict[str, Any],
    ) -> PolicyEngine:
        """
        Create and initialize a policy engine instance.

        Convenience method that combines create() and initialize().

        Args:
            engine_type: Type identifier
            config: Engine configuration

        Returns:
            Initialized engine instance
        """
        engine = cls.create(engine_type)
        await engine.initialize(config)
        return engine

    @classmethod
    def list_engines(cls) -> List[str]:
        """
        List all registered engine types.

        Returns:
            List of registered engine type identifiers
        """
        return list(cls._engines.keys())

    @classmethod
    def is_registered(cls, engine_type: str) -> bool:
        """
        Check if an engine type is registered.

        Args:
            engine_type: Type identifier

        Returns:
            True if registered, False otherwise
        """
        return engine_type in cls._engines

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations.

        Mainly useful for testing.
        """
        cls._engines.clear()


def register_engine(engine_type: str) -> Callable[[Type[PolicyEngine]], Type[PolicyEngine]]:
    """
    Decorator to auto-register a policy engine class.

    Usage:
        @register_engine("fsm")
        class FSMPolicyEngine(PolicyEngine):
            ...

    Args:
        engine_type: Type identifier for the engine

    Returns:
        Decorator function
    """

    def decorator(cls: Type[PolicyEngine]) -> Type[PolicyEngine]:
        PolicyEngineRegistry.register(engine_type, cls)
        return cls

    return decorator
