"""
Panoptes SDK - Reliability layer for AI agents.

Monitors workflow adherence and intervenes when agents deviate from expected behavior.
Uses LiteLLM proxy approach - customers only need to change their base_url.

Quick Start:
    ```python
    from panoptes import PanoptesProxy, PanoptesSettings

    # Configure and start proxy
    settings = PanoptesSettings(workflow_path="workflow.yaml")
    proxy = PanoptesProxy(settings)
    await proxy.start()
    ```

    Then point your LLM client to http://localhost:4000/v1
"""

__version__ = "0.1.0"

# Core configuration
from panoptes.config.settings import PanoptesSettings

# Proxy server
from panoptes.proxy.server import PanoptesProxy, start_proxy

# Workflow components
from panoptes.workflow.schema import (
    WorkflowDefinition,
    State,
    Transition,
    Constraint,
    ConstraintType,
)
from panoptes.workflow.parser import WorkflowParser
from panoptes.workflow.state_machine import WorkflowStateMachine

# Monitoring
from panoptes.monitor.tracker import WorkflowTracker
from panoptes.monitor.classifier import StateClassifier

__all__ = [
    # Version
    "__version__",
    # Configuration
    "PanoptesSettings",
    # Proxy
    "PanoptesProxy",
    "start_proxy",
    # Workflow
    "WorkflowDefinition",
    "State",
    "Transition",
    "Constraint",
    "ConstraintType",
    "WorkflowParser",
    "WorkflowStateMachine",
    # Monitoring
    "WorkflowTracker",
    "StateClassifier",
]
