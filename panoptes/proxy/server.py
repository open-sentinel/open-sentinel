"""
Panoptes proxy server - wraps LiteLLM proxy with custom hooks.

This module provides two approaches for running the proxy:
1. Programmatic: Use PanoptesProxy class directly in Python
2. CLI: Use `panoptes serve` command which calls start_proxy()

The proxy intercepts all LLM calls, enabling:
- Workflow state tracking
- Constraint evaluation
- Automatic intervention when deviations detected
- Full observability via Langfuse
"""

import asyncio
import atexit
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Any

import yaml
import litellm
from litellm import Router

from panoptes.config.settings import PanoptesSettings

logger = logging.getLogger(__name__)


class PanoptesProxy:
    """
    Main Panoptes proxy class.

    Wraps LiteLLM's Router with Panoptes hooks for workflow monitoring.

    Example:
        ```python
        from panoptes import PanoptesSettings
        from panoptes.proxy import PanoptesProxy

        settings = PanoptesSettings(workflow_path="workflow.yaml")
        proxy = PanoptesProxy(settings)
        await proxy.start()
        ```
    """

    def __init__(self, settings: Optional[PanoptesSettings] = None):
        self.settings = settings or PanoptesSettings()
        self.router: Optional[Router] = None
        self._hooks_registered = False
        self._workflow = None
        self._callback = None  # Store reference to callback for shutdown

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.settings.log_level)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        if self.settings.debug:
            litellm.set_verbose = True

    def _register_hooks(self) -> None:
        """Set up environment variables and LiteLLM's Langfuse OTEL callback."""
        if self._hooks_registered:
            return

        # Register shutdown handler
        atexit.register(self._shutdown_tracer)

        self._hooks_registered = True

        if self.settings.langfuse.enabled and self.settings.langfuse.public_key:
            logger.info("Langfuse tracing enabled via LiteLLM OTEL callback")

            # Set environment variables required by Langfuse
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.settings.langfuse.public_key
            if self.settings.langfuse.secret_key:
                os.environ["LANGFUSE_SECRET_KEY"] = self.settings.langfuse.secret_key
            os.environ["LANGFUSE_HOST"] = self.settings.langfuse.host

            # Add langfuse_otel to LiteLLM's success callback (NOT "langfuse")
            # The legacy "langfuse" callback is incompatible with Langfuse SDK v3
            if "langfuse_otel" not in litellm.success_callback:
                litellm.success_callback.append("langfuse_otel")

            logger.info(f"LiteLLM success_callback: {litellm.success_callback}")

    def _shutdown_tracer(self) -> None:
        """Shutdown callback and flush any pending data."""
        # LiteLLM's langfuse_otel callback handles cleanup automatically
        logger.info("Panoptes proxy shutting down...")

    def _load_workflow(self) -> Optional[Any]:
        """Load workflow definition if configured."""
        if not self.settings.workflow_path:
            logger.warning("No workflow_path configured - running in pass-through mode")
            return None

        from panoptes.workflow.parser import WorkflowParser

        workflow_path = Path(self.settings.workflow_path)
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        workflow = WorkflowParser.parse_file(workflow_path)
        logger.info(f"Loaded workflow: {workflow.name} v{workflow.version}")
        return workflow

    def _create_router(self) -> Router:
        """Create LiteLLM Router with model configuration and callbacks."""
        model_list = self.settings.get_model_list()

        # Import here to avoid circular imports
        from panoptes.proxy.hooks import PanoptesCallback

        # Create callback instance with settings
        callback = PanoptesCallback(self.settings)
        self._callback = callback  # Store reference for shutdown

        # Register callback globally with litellm BEFORE creating router
        # This ensures callbacks are picked up by the proxy server
        if litellm.callbacks is None:
            litellm.callbacks = []
        litellm.callbacks.append(callback)

        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            set_verbose=self.settings.debug,
        )

        logger.info(f"Created LiteLLM router with {len(model_list)} models and PanoptesCallback")
        return router

    def generate_litellm_config(self) -> str:
        """
        Generate LiteLLM config.yaml content for CLI usage.

        Returns:
            YAML string for LiteLLM proxy configuration.
        """
        config = {
            "model_list": self.settings.get_model_list(),
            "litellm_settings": {
                "callbacks": ["panoptes.proxy.hooks.PanoptesCallback"],
                "set_verbose": self.settings.debug,
            },
            "general_settings": {
                "master_key": self.settings.proxy.master_key or "sk-panoptes-dev",
            },
        }

        # Add Langfuse if configured (use langfuse_otel for Langfuse v3 compatibility)
        if self.settings.langfuse.enabled and self.settings.langfuse.public_key:
            config["litellm_settings"]["success_callback"] = ["langfuse_otel"]
            config["environment_variables"] = {
                "LANGFUSE_PUBLIC_KEY": self.settings.langfuse.public_key,
                "LANGFUSE_SECRET_KEY": self.settings.langfuse.secret_key or "",
                "LANGFUSE_HOST": self.settings.langfuse.host,
            }

        return yaml.dump(config, default_flow_style=False)

    async def initialize(self) -> None:
        """Initialize the proxy (load workflow, register hooks)."""
        self._setup_logging()
        self._workflow = self._load_workflow()
        self._register_hooks()
        self.router = self._create_router()

    async def start(self) -> None:
        """
        Start the Panoptes proxy server.

        This runs the LiteLLM proxy with Panoptes hooks enabled.
        """
        await self.initialize()

        # For programmatic usage, we use uvicorn directly
        import uvicorn
        import litellm.proxy.proxy_server
        from litellm.proxy.proxy_server import app

        # Inject our configured router into LiteLLM
        litellm.proxy.proxy_server.llm_router = self.router

        config = uvicorn.Config(
            app=app,
            host=self.settings.proxy.host,
            port=self.settings.proxy.port,
            workers=self.settings.proxy.workers,
            log_level=self.settings.log_level.lower(),
        )

        server = uvicorn.Server(config)
        logger.info(
            f"Starting Panoptes proxy on {self.settings.proxy.host}:{self.settings.proxy.port}"
        )
        await server.serve()

    async def completion(self, **kwargs) -> Any:
        """
        Make a completion request through the proxy.

        This is useful for testing or programmatic usage without HTTP.
        """
        if self.router is None:
            await self.initialize()

        return await self.router.acompletion(**kwargs)


def start_proxy(settings: Optional[PanoptesSettings] = None) -> None:
    """
    Start the Panoptes proxy server (blocking).

    This is the main entry point for the CLI.

    Args:
        settings: Optional PanoptesSettings. If not provided, will be loaded
                 from environment variables.
    """
    proxy = PanoptesProxy(settings)
    asyncio.run(proxy.start())


def start_proxy_cli(
    workflow_path: Optional[str] = None,
    port: int = 4000,
    host: str = "0.0.0.0",
    debug: bool = False,
) -> None:
    """
    Start proxy with CLI-friendly arguments.

    This is called by the CLI to translate CLI args to settings.
    """
    settings = PanoptesSettings(
        workflow_path=workflow_path,
        debug=debug,
        proxy={"host": host, "port": port},
    )
    start_proxy(settings)
