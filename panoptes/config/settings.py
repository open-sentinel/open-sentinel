"""
Panoptes configuration management using Pydantic Settings.

Configuration can be provided via:
1. Environment variables (prefixed with PANOPTES_)
2. .env file
3. Direct instantiation

Environment variable examples:
    PANOPTES_DEBUG=true
    PANOPTES_OTEL__ENDPOINT=http://localhost:4317
    PANOPTES_OTEL__SERVICE_NAME=panoptes
    PANOPTES_PROXY__PORT=4000

Policy engine examples:
    # Use NeMo Guardrails engine (default)
    PANOPTES_POLICY__ENGINE__TYPE=nemo
    PANOPTES_POLICY__ENGINE__CONFIG_PATH=/path/to/nemo_config/

    # Use FSM engine
    # Use FSM engine
    PANOPTES_POLICY__ENGINE__TYPE=fsm
    PANOPTES_POLICY__ENGINE__CONFIG_PATH=/path/to/workflow.yaml

    # Use composite engine (combine multiple)
    PANOPTES_POLICY__ENGINE__TYPE=composite
"""

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    langfuse_host: str = "https://cloud.langfuse.com"  # EU region, use https://us.cloud.langfuse.com for US


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

    Supported engine types:
    - fsm: Finite State Machine (workflow-based)
    - nemo: NeMo Guardrails
    - composite: Combine multiple engines

    The 'config' field contains engine-specific configuration:
    - fsm: {"config_path": "..."} or {"workflow": {...}}
    - nemo: {"config_path": "..."} or {"config": {...}}
    - composite: {"engines": [...], "strategy": "all"|"first_deny"}
    """

    type: Literal["fsm", "nemo", "composite", "llm"] = "nemo"
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


class PanoptesSettings(BaseSettings):
    """
    Main Panoptes configuration.

    All settings can be overridden via environment variables with PANOPTES_ prefix.
    Nested settings use double underscore: PANOPTES_OTEL__ENDPOINT
    """

    model_config = SettingsConfigDict(
        env_prefix="PANOPTES_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
            }
        ]
