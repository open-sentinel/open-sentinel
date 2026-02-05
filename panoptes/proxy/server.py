"""
Panoptes proxy server - wraps LiteLLM proxy with custom hooks.

This module provides two approaches for running the proxy:
1. Programmatic: Use PanoptesProxy class directly in Python
2. CLI: Use `panoptes serve` command which calls start_proxy()

The proxy intercepts all LLM calls, enabling:
- Workflow state tracking
- Constraint evaluation
- Automatic intervention when deviations detected
- Full observability via OpenTelemetry
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



class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to logs."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class PanoptesProxy:
    """
    Main Panoptes proxy class.

    Wraps LiteLLM's Router with Panoptes hooks for workflow monitoring.

    Example:
        ```python
        from panoptes import PanoptesSettings
        from panoptes.proxy import PanoptesProxy

        settings = PanoptesSettings()
        proxy = PanoptesProxy(settings)
        await proxy.start()
        ```
    """

    def __init__(self, settings: Optional[PanoptesSettings] = None):
        self.settings = settings or PanoptesSettings()
        self.router: Optional[Router] = None
        self._hooks_registered = False
        self._callback = None  # Store reference to callback for shutdown

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.settings.log_level)
        
        # Create console handler with custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter())
        
        # Get root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        if root_logger.handlers:
            root_logger.handlers.clear()
            
        root_logger.addHandler(handler)

        if self.settings.debug:
            litellm.set_verbose = True


    def _register_hooks(self) -> None:
        """Register hooks and set up cleanup handlers."""
        if self._hooks_registered:
            return

        # Register shutdown handler
        atexit.register(self._shutdown_tracer)

        self._hooks_registered = True
        logger.info("Panoptes hooks registered")

    def _shutdown_tracer(self) -> None:
        """Shutdown callback and flush any pending data."""
        logger.info("Panoptes proxy shutting down...")

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

        # Also register for async success callbacks (required for async_log_success_event)
        # This is what LiteLLM uses for async callbacks in proxy mode
        litellm.logging_callback_manager.add_litellm_async_success_callback(callback)
        logger.info(f"Registered PanoptesCallback for async success callbacks")

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

        return yaml.dump(config, default_flow_style=False)

    async def initialize(self) -> None:
        """Initialize the proxy (register hooks)."""
        self._setup_logging()
        self._log_policy_config()
        self._register_hooks()
        self.router = self._create_router()

    def _log_policy_config(self) -> None:
        """Log active policy engine configuration."""
        policy_config = self.settings.get_policy_config()
        engine_type = policy_config.get("type")
        engine_conf = policy_config.get("config", {})

        if engine_type == "nemo" and engine_conf.get("config_path"):
            logger.info("Running with NeMo Guardrails engine")
        elif engine_type == "fsm" and engine_conf.get("workflow_path"):
            logger.info(f"Running with FSM engine: {engine_conf.get('workflow_path')}")
        elif engine_type == "composite" and engine_conf.get("engines"):
            logger.info("Running with Composite policy engine")
        else:
            logger.warning("No policy engine configured - running in pass-through mode")

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
    port: int = 4000,
    host: str = "0.0.0.0",
    debug: bool = False,
) -> None:
    """
    Start proxy with CLI-friendly arguments.

    This is called by the CLI to translate CLI args to settings.
    """
    settings = PanoptesSettings(
        debug=debug,
        proxy={"host": host, "port": port},
    )
    start_proxy(settings)
