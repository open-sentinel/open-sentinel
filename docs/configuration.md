# Configuration Reference

OpenSentinel reads configuration from three sources, applied in this order (highest priority wins):

1. `osentinel.yaml` (or `osentinel.yml`) in the working directory
2. Environment variables with `OSNTL_` prefix
3. Built-in defaults

API keys are always read from environment variables or `.env` files. Never put keys in YAML.

## Config File Discovery

OpenSentinel looks for the config file in this order:

1. Explicit path via `osentinel serve --config path/to/config.yaml`
2. `$OSNTL_CONFIG` environment variable
3. `./osentinel.yaml` in the current directory
4. `./osentinel.yml` in the current directory

If none are found, all settings use defaults.

## Minimal Config

```yaml
engine: judge
policy:
  - "No financial advice"
  - "Be professional"
```

This uses the judge engine with inline rules, auto-detected model, default port 4000.

## Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `engine` | string | `judge` | Policy engine type: `judge`, `fsm`, `llm`, `nemo`, `composite` |
| `model` | string | auto-detected | Default LLM model. Auto-detected from whichever API key is present. Engines can override in their own section. |
| `port` | int | `4000` | Proxy server port |
| `host` | string | `0.0.0.0` | Proxy server bind address |
| `debug` | bool | `false` | Enable debug logging |
| `log_level` | string | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Model auto-detection priority: `OPENAI_API_KEY` -> `gpt-4o-mini`, `GOOGLE_API_KEY`/`GEMINI_API_KEY` -> `gemini/gemini-2.5-flash`, `ANTHROPIC_API_KEY` -> `anthropic/claude-sonnet-4-5`.

## Policy

The `policy` key accepts three forms:

**File path** (string) -- passed as `config_path` to the engine:
```yaml
policy: ./customer_support.yaml
```

**Inline rules** (list) -- judge engine only:
```yaml
policy:
  - "No financial advice"
  - "Be professional"
```

**Inline rubrics** (dict) -- judge engine only:
```yaml
policy:
  rules: ["No financial advice"]
  rubrics:
    - name: my_rubric
      description: Custom rubric
      criteria:
        - name: tone
          description: Professional tone
          scale: binary
```

## Judge Engine

Set `engine: judge` at the top level. Configure under the `judge:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `judge.model` | string | global `model` | LLM model for evaluation. Overrides the global model setting. |
| `judge.mode` | string | `balanced` | Reliability preset: `safe`, `balanced`, `aggressive`. Individual keys below override the preset. |
| `judge.pass_threshold` | float | `0.6` | Score above which a response passes (0.0-1.0) |
| `judge.warn_threshold` | float | `0.4` | Score below pass but above this triggers a warning |
| `judge.block_threshold` | float | `0.2` | Score below this blocks the response |
| `judge.confidence_threshold` | float | `0.5` | Minimum confidence for the judge's own assessment |
| `judge.pre_call_enabled` | bool | `false` | Evaluate requests before forwarding to the LLM |
| `judge.pre_call_rubric` | string | `safety` | Which rubric to use for pre-call evaluation |
| `judge.default_rubric` | string | `agent_behavior` | Default rubric for per-turn evaluation |
| `judge.conversation_rubric` | string | `conversation_policy` | Rubric for multi-turn conversation evaluation |
| `judge.custom_rubrics_path` | string | -- | Path to directory containing custom rubric YAML files |
| `judge.conversation_eval_interval` | int | `5` | Run conversation-level evaluation every N turns |
| `judge.ensemble_enabled` | bool | `false` | Use multiple models for judging |
| `judge.aggregation_strategy` | string | `mean_score` | How to combine ensemble results: `mean_score`, `conservative` |
| `judge.min_agreement` | float | `0.6` | Minimum agreement ratio across ensemble models |

### Reliability Modes

The `mode` key sets sensible defaults for thresholds. Any individual key you set overrides the preset.

**safe** -- Stricter thresholds, pre-call safety screening enabled, ensemble when multiple models configured. For high-stakes applications.

**balanced** (default) -- Moderate thresholds, post-call evaluation only. Reasonable tradeoff between safety and latency.

**aggressive** -- Looser thresholds, fewer interventions. For low-risk applications where overhead matters more than coverage.

### Multi-Model Ensemble

For explicit multi-model configuration, use `judge.models` instead of `judge.model`:

```yaml
judge:
  ensemble_enabled: true
  aggregation_strategy: mean_score
  models:
    - name: primary
      model: anthropic/claude-sonnet-4-5
      temperature: 0.0
      max_tokens: 2048
      timeout: 15.0
    - name: secondary
      model: gpt-4o-mini
      temperature: 0.0
```

## LLM Engine

Set `engine: llm` at the top level. Requires `policy:` pointing to a workflow YAML file. Configure under the `llm:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | global `model` | LLM model for state classification |
| `llm.temperature` | float | `0.0` | LLM temperature |
| `llm.max_tokens` | int | `1024` | Maximum tokens per LLM call |
| `llm.timeout` | float | `10.0` | Request timeout in seconds |
| `llm.confident_threshold` | float | `0.8` | Confidence above which a state classification is accepted |
| `llm.uncertain_threshold` | float | `0.5` | Below this, classification is rejected |
| `llm.temporal_weight` | float | `0.55` | Weight for temporal signals in drift detection |
| `llm.cooldown_turns` | int | `2` | Minimum turns between constraint re-evaluations |
| `llm.max_constraints_per_batch` | int | `5` | Maximum constraints evaluated per batch |

### LLM Engine Intervention Settings

```yaml
llm:
  intervention:
    default_strategy: user_message_inject   # system_prompt_append | user_message_inject | hard_block
    max_intervention_attempts: 3
    include_headers: true
```

## FSM Engine

Set `engine: fsm` at the top level. Requires `policy:` pointing to a workflow YAML file. Configure under the `fsm:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fsm.classifier.model_name` | string | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding-based state classification |
| `fsm.classifier.backend` | string | `pytorch` | Inference backend: `pytorch` or `onnx` |
| `fsm.classifier.similarity_threshold` | float | `0.7` | Minimum cosine similarity for a state match |
| `fsm.classifier.cache_embeddings` | bool | `true` | Cache computed embeddings for workflow states |
| `fsm.classifier.device` | string | `cpu` | Inference device: `cpu` or `cuda` |

### FSM Engine Intervention Settings

```yaml
fsm:
  intervention:
    default_strategy: system_prompt_append   # system_prompt_append | user_message_inject | hard_block
    max_intervention_attempts: 3
    include_headers: true
```

## NeMo Guardrails Engine

Set `engine: nemo` at the top level. Requires `policy:` pointing to a NeMo Guardrails config directory. Configure under the `nemo:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nemo.fail_closed` | bool | `false` | If true, block on NeMo evaluation errors. If false (default), warn and allow. |
| `nemo.rails` | list | all configured | Which rails to enable. Omit to use all rails from NeMo config. |

## Composite Engine

Set `engine: composite` at the top level. Combines multiple engines. Configure under the `composite:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `composite.strategy` | string | `all` | Evaluation strategy: `all` (run all engines) or `first_deny` (stop at first denial) |
| `composite.parallel` | bool | `true` | Run engines concurrently |
| `composite.engines` | list | -- | List of sub-engine configurations |

Example:

```yaml
engine: composite
composite:
  strategy: all
  parallel: true
  engines:
    - type: judge
      config:
        models:
          - name: primary
            model: gpt-4o-mini
        inline_policy:
          - "No financial advice"
    - type: fsm
      config:
        config_path: ./workflow.yaml
```

## Tracing

Configure under the `tracing:` section. Tracing uses OpenTelemetry spans.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.type` | string | `otlp` | Exporter type: `otlp`, `langfuse`, `console`, `none` |
| `tracing.endpoint` | string | `http://localhost:4317` | OTLP endpoint URL |
| `tracing.service_name` | string | `opensentinel` | Service name in traces |

### Langfuse

When `tracing.type: langfuse`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.langfuse_public_key` | string | -- | Langfuse public key |
| `tracing.langfuse_secret_key` | string | -- | Langfuse secret key |
| `tracing.langfuse_host` | string | `https://cloud.langfuse.com` | Langfuse host URL |

## Environment Variables

All settings can be overridden via environment variables with the `OSNTL_` prefix. Nested keys use `__` (double underscore) as delimiter.

### Mapping Rules

| YAML Path | Environment Variable |
|-----------|---------------------|
| `debug` | `OSNTL_DEBUG` |
| `log_level` | `OSNTL_LOG_LEVEL` |
| (proxy port) | `OSNTL_PROXY__PORT` |
| (proxy host) | `OSNTL_PROXY__HOST` |
| (engine type) | `OSNTL_POLICY__ENGINE__TYPE` |
| (engine config path) | `OSNTL_POLICY__ENGINE__CONFIG_PATH` |
| (tracing exporter) | `OSNTL_OTEL__EXPORTER_TYPE` |
| (tracing endpoint) | `OSNTL_OTEL__ENDPOINT` |
| (config file path) | `OSNTL_CONFIG` |

### API Keys

API keys bypass the `OSNTL_` prefix. Set them directly:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `GEMINI_API_KEY` | Google (Gemini, alternative) |
| `GROQ_API_KEY` | Groq |
| `TOGETHERAI_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |

If multiple keys are present, the auto-detected model uses the first one found in the order above.

### Langfuse via Environment

```bash
OSNTL_OTEL__EXPORTER_TYPE=langfuse
OSNTL_OTEL__LANGFUSE_PUBLIC_KEY=pk-lf-...
OSNTL_OTEL__LANGFUSE_SECRET_KEY=sk-lf-...
OSNTL_OTEL__LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## .env File

OpenSentinel reads `.env` files automatically. API keys found in `.env` are synced to `os.environ` so downstream libraries (LiteLLM, etc.) can use them without explicit `load_dotenv()` calls.

See `.env.example` in the repository root for a template.

## Config Validation

The `osentinel serve` command validates configuration at startup:

- Checks that referenced policy files exist on disk
- Verifies that the required API key is present for the configured model
- Applies reliability mode defaults before engine-specific overrides

If validation fails, the server prints the error and exits with code 1. Use `--debug` for a full traceback.

## Full Example

```yaml
engine: judge
model: gemini/gemini-2.5-flash
port: 4000
debug: false

policy: ./policy.yaml

judge:
  model: anthropic/claude-sonnet-4-5
  mode: balanced
  pass_threshold: 0.6
  pre_call_enabled: false
  ensemble_enabled: false

tracing:
  type: none
```
