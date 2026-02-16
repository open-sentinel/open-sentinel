"""
Panoptes configuration management using Pydantic Settings.

Configuration can be provided via:
1. Environment variables (prefixed with PANOPTES_)
2. panoptes.yaml config file
3. .env file
4. Direct instantiation

Priority (highest wins): env vars > panoptes.yaml > defaults

The simplified panoptes.yaml format:
    engine: judge
    policy: ./policy.yaml
    port: 4000
    judge:
      model: gpt-4o-mini
      mode: balanced
    tracing:
      type: none

Environment variable examples:
    PANOPTES_DEBUG=true
    PANOPTES_OTEL__ENDPOINT=http://localhost:4317
    PANOPTES_OTEL__SERVICE_NAME=panoptes
    PANOPTES_PROXY__PORT=4000

Policy engine examples:
    PANOPTES_POLICY__ENGINE__TYPE=nemo
    PANOPTES_POLICY__ENGINE__CONFIG_PATH=/path/to/nemo_config/
    PANOPTES_POLICY__ENGINE__TYPE=fsm
    PANOPTES_POLICY__ENGINE__CONFIG_PATH=/path/to/workflow.yaml
    PANOPTES_POLICY__ENGINE__TYPE=composite
"""

import logging
import os
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any, Tuple, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

logger = logging.getLogger(__name__)


class OTelConfig(BaseModel):
    """OpenTelemetry tracing configuration.

    Supports multiple exporters:
    - otlp: Standard OTLP endpoint (Jaeger, Zipkin, etc.)
    - langfuse: Langfuse's OTLP endpoint (requires public_key/secret_key)
    - console: Print traces to console (for debugging)
    - none: Disable tracing
    """

    enabled: bool = True
    endpoint: str = "http://localhost:4317"
    service_name: str = "panoptes"
    exporter_type: Literal["otlp", "langfuse", "console", "none"] = "otlp"
    insecure: bool = True  # Use insecure connection (no TLS) for local dev

    # Langfuse-specific settings (used when exporter_type="langfuse")
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = (
        "https://cloud.langfuse.com"  # EU region, use https://us.cloud.langfuse.com for US
    )


class ProxyConfig(BaseModel):
    """LiteLLM proxy server configuration."""

    host: str = "0.0.0.0"
    port: int = 4000
    workers: int = 1
    timeout: int = 600
    master_key: Optional[str] = None
    # Model routing
    default_model: str = "gpt-4"
    model_list: List[dict] = Field(default_factory=list)


class ClassifierConfig(BaseModel):
    """State classifier configuration."""

    # Model for semantic similarity
    model_name: str = "all-MiniLM-L6-v2"
    # Use ONNX backend for faster inference (<50ms)
    backend: Literal["pytorch", "onnx"] = "pytorch"
    # Minimum similarity score to consider a match
    similarity_threshold: float = 0.7
    # Cache embeddings for workflow states
    cache_embeddings: bool = True
    # Device for inference
    device: str = "cpu"


class InterventionConfig(BaseModel):
    """Intervention system configuration."""

    # Default strategy when not specified in workflow
    default_strategy: Literal[
        "system_prompt_append", "user_message_inject", "hard_block"
    ] = "system_prompt_append"
    # Maximum times to apply same intervention before escalating
    max_intervention_attempts: int = 3
    # Include intervention metadata in response headers
    include_headers: bool = True


class PolicyEngineConfig(BaseModel):
    """Configuration for a single policy engine.

    The 'type' field accepts any engine registered via @register_engine
    in the PolicyEngineRegistry (see panoptes/policy/registry.py).

    Built-in engine types include: fsm, nemo, llm, composite.
    Custom engines are automatically supported once registered.

    The 'config' field contains engine-specific configuration.
    """

    # Accepts any registered engine type — not hard-coded to a fixed set.
    # See PolicyEngineRegistry.list_engines() for available types at runtime.
    type: str = "nemo"
    enabled: bool = True
    # Unified configuration path (can be set via PANOPTES_POLICY__ENGINE__CONFIG_PATH)
    config_path: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs):
        """Custom dump to merge config_path into config dict for engines."""
        data = super().model_dump(**kwargs)
        # If config_path is set, use it to populate/override config dict
        if data.get("config_path"):
            data["config"]["config_path"] = data["config_path"]
        return data


class PolicyConfig(BaseModel):
    """Policy system configuration.

    Supports multiple policy engines with different mechanisms:
    - FSM for workflow enforcement
    - NeMo Guardrails for content moderation
    - Composite to combine multiple engines

    Examples:
        # NeMo only (default)
        policy:
          engine:
            type: nemo

        # FSM only
        policy:
          engine:
            type: fsm
            config:
              workflow_path: ./workflow.yaml

        # Combined FSM + NeMo
        policy:
          engine:
            type: composite
          engines:
            - type: fsm
              config:
                workflow_path: ./workflow.yaml
            - type: nemo
              config:
                config_path: ./nemo_config/
    """

    # Primary engine configuration
    engine: PolicyEngineConfig = Field(default_factory=PolicyEngineConfig)

    # For composite engine: list of child engines
    engines: List[PolicyEngineConfig] = Field(default_factory=list)

    # Fallback behavior when engine evaluation fails
    # True = allow request on error (fail open)
    # False = deny request on error (fail closed)
    fail_open: bool = True

    # Maximum time (seconds) any hook is allowed to run before fail-open timeout.
    # Set generously since interceptor checks may involve LLM calls.
    hook_timeout_seconds: float = 30.0


class YamlConfigSource(PydanticBaseSettingsSource):
    """Custom settings source that reads from a panoptes.yaml config file.

    Discovers config at:
    1. Explicit path passed via _config_path init kwarg
    2. $PANOPTES_CONFIG env var
    3. ./panoptes.yaml
    4. ./panoptes.yml

    Maps simplified YAML keys to the nested PanoptesSettings structure.
    """

    def __init__(
        self, settings_cls: Type[BaseSettings], config_path: Optional[str] = None
    ):
        super().__init__(settings_cls)
        self._config_path = config_path
        self._yaml_data: Optional[Dict[str, Any]] = None
        self._load()

    def _discover_config_file(self) -> Optional[Path]:
        """Find the config file to load."""
        if self._config_path:
            p = Path(self._config_path)
            return p if p.is_file() else None

        env_path = os.environ.get("PANOPTES_CONFIG")
        if env_path:
            p = Path(env_path)
            return p if p.is_file() else None

        for name in ("panoptes.yaml", "panoptes.yml"):
            p = Path(name)
            if p.is_file():
                return p

        return None

    def _load(self) -> None:
        """Load and parse the YAML config file."""
        path = self._discover_config_file()
        if path is None:
            self._yaml_data = {}
            return

        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
            self._yaml_data = data if isinstance(data, dict) else {}
            logger.debug(f"Loaded config from {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            self._yaml_data = {}

    def _map_to_settings(self) -> Dict[str, Any]:
        """Map simplified YAML keys to nested PanoptesSettings structure."""
        if not self._yaml_data:
            return {}

        data = self._yaml_data
        result: Dict[str, Any] = {}

        # engine -> policy.engine.type
        if "engine" in data:
            result.setdefault("policy", {}).setdefault("engine", {})["type"] = data[
                "engine"
            ]

        # policy -> policy.engine.config_path
        if "policy" in data and isinstance(data["policy"], str):
            result.setdefault("policy", {}).setdefault("engine", {})["config_path"] = (
                data["policy"]
            )

        # port -> proxy.port
        if "port" in data:
            result.setdefault("proxy", {})["port"] = data["port"]

        # host -> proxy.host
        if "host" in data:
            result.setdefault("proxy", {})["host"] = data["host"]

        # debug
        if "debug" in data:
            result["debug"] = data["debug"]

        # log_level
        if "log_level" in data:
            result["log_level"] = data["log_level"]

        # judge.* -> policy.engine.config.*
        judge_cfg = data.get("judge", {})
        if isinstance(judge_cfg, dict) and judge_cfg:
            engine_config = (
                result.setdefault("policy", {})
                .setdefault("engine", {})
                .setdefault("config", {})
            )

            if "model" in judge_cfg:
                engine_config["models"] = [
                    {"name": "primary", "model": judge_cfg["model"]}
                ]

            if "mode" in judge_cfg:
                try:
                    from panoptes.policy.engines.judge.modes import build_mode_config

                    mode_config = build_mode_config(judge_cfg["mode"])
                    for k, v in mode_config.items():
                        engine_config.setdefault(k, v)
                except (ImportError, ValueError) as e:
                    logger.warning(
                        f"Failed to apply judge mode '{judge_cfg['mode']}': {e}"
                    )

        # tracing.* -> otel.*
        tracing_cfg = data.get("tracing", {})
        if isinstance(tracing_cfg, dict) and tracing_cfg:
            otel = result.setdefault("otel", {})
            if "type" in tracing_cfg:
                tracing_type = tracing_cfg["type"]
                otel["exporter_type"] = tracing_type
                if tracing_type == "none":
                    otel["enabled"] = False
                else:
                    otel["enabled"] = True
            if "endpoint" in tracing_cfg:
                otel["endpoint"] = tracing_cfg["endpoint"]
            if "service_name" in tracing_cfg:
                otel["service_name"] = tracing_cfg["service_name"]

        return result

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        mapped = self._map_to_settings()
        value = mapped.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> Dict[str, Any]:
        return self._map_to_settings()


class PanoptesSettings(BaseSettings):
    """
    Main Panoptes configuration.

    All settings can be overridden via environment variables with PANOPTES_ prefix.
    Nested settings use double underscore: PANOPTES_OTEL__ENDPOINT

    A panoptes.yaml config file is also supported (env vars take priority).
    Pass _config_path to override the config file location.
    """

    model_config = SettingsConfigDict(
        env_prefix="PANOPTES_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Path to panoptes.yaml (set via _config_path kwarg, not a real setting field)
    _config_path: Optional[str] = None

    # General settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Component configurations
    otel: OTelConfig = Field(default_factory=OTelConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)

    # Policy engine configuration
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    def __init__(self, _config_path: Optional[str] = None, **kwargs: Any):
        self.__class__._config_path = _config_path
        super().__init__(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Insert YAML config source between env vars and defaults.

        Priority (highest first): init > env > dotenv > yaml > file_secret
        """
        yaml_source = YamlConfigSource(settings_cls, config_path=cls._config_path)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )

    def get_policy_config(self) -> Dict[str, Any]:
        """
        Get policy engine configuration.

        Returns:
            Configuration dict ready for PolicyEngineRegistry.create_and_initialize()
        """
        engine_config = self.policy.engine.model_dump()

        # Handle composite engine
        if engine_config["type"] == "composite":
            # Use engines list from policy config
            if self.policy.engines:
                engine_config["config"]["engines"] = [
                    e.model_dump() for e in self.policy.engines
                ]

        return engine_config

    def get_model_list(self) -> List[dict]:
        """Get model list for LiteLLM router."""
        if self.proxy.model_list:
            return self.proxy.model_list

        # Default model configuration
        return [
            # OpenAI
            {
                "model_name": "openai-gpt-5.2",
                "litellm_params": {"model": "gpt-5.2"},
            },
            {
                "model_name": "openai-gpt-5-pro",
                "litellm_params": {"model": "gpt-5-pro"},
            },
            {
                "model_name": "openai-gpt-4o",
                "litellm_params": {"model": "gpt-4o"},
            },
            {
                "model_name": "openai-gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            # Anthropic Claude
            {
                "model_name": "anthropic-claude-opus-4.1",
                "litellm_params": {"model": "anthropic/claude-opus-4-1"},
            },
            {
                "model_name": "anthropic-claude-sonnet-4.5",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5"},
            },
            {
                "model_name": "anthropic-claude-3.7",
                "litellm_params": {"model": "anthropic/claude-3-7-sonnet"},
            },
            {
                "model_name": "anthropic-claude-instant-1.2",
                "litellm_params": {"model": "anthropic/claude-instant-1.2"},
            },
            # Google Gemini (text/chat)
            {
                "model_name": "gemini-1.5-pro",
                "litellm_params": {"model": "gemini/1.5-pro"},
            },
            {
                "model_name": "gemini-1.5-flash",
                "litellm_params": {"model": "gemini/1.5-flash"},
            },
            # TogetherAI (LLaMA & Falcon via Together)
            {
                "model_name": "together-llama-2-70b-chat",
                "litellm_params": {
                    "model": "together_ai/togethercomputer/llama-2-70b-chat"
                },
            },
            {
                "model_name": "together-falcon-40b-instruct",
                "litellm_params": {
                    "model": "together_ai/togethercomputer/falcon-40b-instruct"
                },
            },
            # Replicate / HuggingFace Example (if you have keys)
            {
                "model_name": "huggingface-llama-3-13b",
                "litellm_params": {"model": "huggingface/llama-3-13b"},
            },
            {
                "model_name": "replicate-mistral-7b",
                "litellm_params": {"model": "replicate/mistral-7b"},
            },
            {
                "model_name": "openrouter/openai/gpt-3.5-turbo",
                "litellm_params": {"model": "openrouter/openai/gpt-3.5-turbo"},
            },
            {
                "model_name": "gemini/gemini-2.5-flash",
                "litellm_params": {"model": "gemini/gemini-2.5-flash"},
            },
        ]
