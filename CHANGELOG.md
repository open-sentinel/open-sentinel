# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.0 (alpha)

Initial release.

### Added

- Transparent LLM proxy built on LiteLLM. Point any OpenAI-compatible client at it with a one-line `base_url` change.
- **Judge engine**: scores responses against plain-English rubrics using a sidecar LLM. Three reliability modes (safe/balanced/aggressive). Async by default -- zero latency on the critical path.
- **FSM engine**: enforces agent behavior as a finite state machine. Three-tier classification cascade (tool call matching, regex, embedding similarity). LTL-lite temporal constraint evaluation.
- **LLM engine**: classifies conversation state and detects drift using LLM-based reasoning.
- **NeMo engine**: integrates NVIDIA NeMo Guardrails for content safety and dialog rails.
- **Composite engine**: runs multiple engines in parallel, merges results (most restrictive wins).
- **Policy compiler**: translates natural language policies to engine-specific YAML via `osentinel compile`.
- **CLI**: `osentinel init`, `osentinel serve`, `osentinel compile`, `osentinel validate`, `osentinel info`.
- **OpenTelemetry tracing**: spans for every proxy call, policy evaluation, and intervention. Console, OTLP, and Langfuse backends.
- Fail-open design: hook failures pass the request through unmodified. Only intentional blocks propagate.
- Deferred intervention: violations detected async are applied as prompt injections on the next turn.
- Session ID extraction from headers, query params, or request body.

### Known Limitations

- Session state is in-memory only. Not persistent across restarts.
- No dashboard UI.
- No pre-built policy library.
- No rate limiting.
