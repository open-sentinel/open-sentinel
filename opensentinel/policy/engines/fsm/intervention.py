"""
Prompt injection for workflow corrections.

Applies intervention strategies to modify LLM requests.

This module is the main interface for applying interventions:
1. Looks up intervention configuration from workflow
2. Selects appropriate strategy
3. Applies modification to request data
"""

import logging
from typing import Optional, Dict, Any

from opensentinel.core.intervention.strategies import (
    InterventionConfig,
    StrategyType,
    STRATEGY_REGISTRY,
    get_strategy,
)
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


class InterventionHandler:
    """
    Injects correction prompts into LLM requests.

    Uses workflow-defined intervention strategies to modify
    requests when constraint violations are detected.

    Example:
        ```python
        from opensentinel.policy.engines.fsm import InterventionHandler, WorkflowParser

        workflow = WorkflowParser.parse_file("workflow.yaml")
        injector = InterventionHandler(workflow)

        # Apply intervention to request data
        modified_data = injector.inject(
            data=request_data,
            intervention_name="prompt_verify_identity",
            context={"current_state": "greeting"}
        )
        ```
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        default_strategy: StrategyType = StrategyType.SYSTEM_PROMPT_APPEND,
    ):
        self.workflow = workflow
        self.default_strategy = default_strategy
        self._intervention_configs = self._load_configs()

        # Track intervention applications per session
        self._application_counts: Dict[str, Dict[str, int]] = {}

        logger.debug(
            f"InterventionHandler initialized with {len(self._intervention_configs)} interventions"
        )

    def _load_configs(self) -> Dict[str, InterventionConfig]:
        """Load intervention configs from workflow."""
        configs = {}

        for name, template in self.workflow.interventions.items():
            # Parse strategy from template if specified
            # Format: "strategy:template" or just "template"
            strategy = self.default_strategy
            message = template

            if template.startswith("block:"):
                strategy = StrategyType.HARD_BLOCK
                message = template[6:].strip()
            elif template.startswith("inject:"):
                strategy = StrategyType.USER_MESSAGE_INJECT
                message = template[7:].strip()
            elif template.startswith("remind:"):
                strategy = StrategyType.CONTEXT_REMINDER
                message = template[7:].strip()

            configs[name] = InterventionConfig(
                strategy_type=strategy,
                message_template=message,
            )

        return configs

    def inject(
        self,
        data: dict,
        intervention_name: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Apply an intervention to request data.

        Args:
            data: The LLM request data (messages, model, etc.)
            intervention_name: Name of intervention from workflow
            context: Additional context for template formatting
            session_id: Session ID for tracking application counts

        Returns:
            Modified request data with intervention applied.
        """
        config = self._intervention_configs.get(intervention_name)

        if not config:
            logger.warning(f"Unknown intervention: {intervention_name}")
            return data

        # Track application count
        if session_id:
            self._track_application(session_id, intervention_name)
            count = self._get_application_count(session_id, intervention_name)

            # Check if we should escalate
            if count > config.max_applications:
                logger.warning(
                    f"Intervention '{intervention_name}' exceeded max applications "
                    f"({count} > {config.max_applications})"
                )
                if config.escalation_strategy:
                    config = InterventionConfig(
                        strategy_type=config.escalation_strategy,
                        message_template=config.message_template,
                    )

        # Get strategy
        strategy = get_strategy(config.strategy_type)

        # Build full context
        full_context = self._build_context(intervention_name, context)

        # Apply intervention
        try:
            modified_data = strategy.apply(data, config, full_context)
            logger.info(
                f"Applied intervention '{intervention_name}' "
                f"(strategy={config.strategy_type.value})"
            )
            return modified_data

        except Exception as e:
            logger.error(f"Failed to apply intervention '{intervention_name}': {e}")
            raise

    def _build_context(
        self,
        intervention_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build full context for template formatting."""
        full_context = {
            "workflow_name": self.workflow.name,
            "intervention_name": intervention_name,
        }

        if context:
            full_context.update(context)

        return full_context

    def _track_application(self, session_id: str, intervention_name: str) -> None:
        """Track intervention application."""
        if session_id not in self._application_counts:
            self._application_counts[session_id] = {}

        counts = self._application_counts[session_id]
        counts[intervention_name] = counts.get(intervention_name, 0) + 1

    def _get_application_count(self, session_id: str, intervention_name: str) -> int:
        """Get number of times an intervention has been applied."""
        return self._application_counts.get(session_id, {}).get(intervention_name, 0)

    def reset_session_counts(self, session_id: str) -> None:
        """Reset application counts for a session."""
        if session_id in self._application_counts:
            del self._application_counts[session_id]

    def get_intervention_info(self, intervention_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an intervention."""
        config = self._intervention_configs.get(intervention_name)
        if not config:
            return None

        return {
            "name": intervention_name,
            "strategy": config.strategy_type.value,
            "template": config.message_template,
            "max_applications": config.max_applications,
        }

    def list_interventions(self) -> list[str]:
        """List all available intervention names."""
        return list(self._intervention_configs.keys())


class InterventionBuilder:
    """
    Builder for creating intervention configurations programmatically.

    Useful when you want to create interventions without YAML.
    """

    def __init__(self):
        self._configs: Dict[str, InterventionConfig] = {}

    def add(
        self,
        name: str,
        template: str,
        strategy: StrategyType = StrategyType.SYSTEM_PROMPT_APPEND,
        max_applications: int = 3,
        escalation: Optional[StrategyType] = None,
    ) -> "InterventionBuilder":
        """Add an intervention configuration."""
        self._configs[name] = InterventionConfig(
            strategy_type=strategy,
            message_template=template,
            max_applications=max_applications,
            escalation_strategy=escalation,
        )
        return self

    def build(self) -> Dict[str, InterventionConfig]:
        """Build the intervention configurations."""
        return dict(self._configs)
