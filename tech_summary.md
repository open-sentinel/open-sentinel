# Panoptes (Argus) Technical Summary

## 1. Project Overview
Panoptes is a **reliability layer for AI agents** that acts as a transparent proxy between an application and LLM providers. It monitors interactions, enforces workflow policies, and intervenes in real-time to correct deviations.

**Core Value Proposition:**
- **Zero-code integration**: Works by just changing the `base_url`.
- **Latency-neutral monitoring**: Asynchronous analysis of conversations.
- **Deterministic enforcement**: Uses FSMs or NeMo Guardrails for implementation.

## 2. Architecture

### 2.1 Proxy Layer (`panoptes/proxy`)
- Built on **LiteLLM** to handle routing and protocol standardization.
- **Server**: FastAPI/Uvicorn-based proxy (`server.py`).
- **Hooks**: Intercepts requests/responses (`hooks.py`) for:
  - `async_pre_call_hook`: Applying pending interventions.
  - `async_post_call_success_hook`: Updating state and detecting violations.
- **Middleware**: Extracts session IDs from headers, metadata, or content.

### 2.2 Policy Layer (`panoptes/policy`)
Pluggable engine architecture to control agent behavior.

#### Engines:
1.  **FSM Engine** (`panoptes/policy/engines/fsm`): 
    -   **Declarative**: Defined in YAML (`workflow.yaml`).
    -   **Components**: States, Transitions, Constraints (LTL-lite), Interventions.
    -   **Enforcement**: Tracks state transitions and evaluates temporal constraints (e.g., "Verification must happen before Refund").
    -   **Monitoring**: Uses semantic search (embeddings) or regex to classify agent outputs into states.

2.  **NeMo Engine** (`panoptes/policy/engines/nemo`):
    -   **Integration**: Wraps NVIDIA NeMo Guardrails.
    -   **Capabilities**: Input/Output rails, Colang-based dialog flows.

3.  **Composite Engine**: logic for combining multiple engines (structure exists, details TBD).

#### Compiler (`panoptes/policy/compiler`):
-   **Function**: Converts natural language policies into structured engine configurations.
-   **CLI**: `panoptes compile "check ID before refund"` -> `workflow.yaml`.
-   **Protocol**: Extensible design to support different target engines.

### 2.3 Core Logic (`panoptes/core`)
-   **Intervention**: Strategies for correcting agent behavior.
    -   `SYSTEM_PROMPT_APPEND`: Passive guidance.
    -   `USER_MESSAGE_INJECT`: Active correction.
    -   `HARD_BLOCK`: Stopping dangerous requests.
-   **Tracing**: OpenTelemetry integration to export traces to Langfuse or Jaeger.

## 3. Key Workflows

### The "Loop"
1.  **Request**: App sends request to Proxy.
2.  **Pre-Call**: Proxy checks for *pending* interventions from previous turns and injects them.
3.  **Call**: Request goes to LLM Provider.
4.  **Post-Call (Async)**: 
    -   Response sent back to App immediately.
    -   **Monitor**: Analyzes response to update FSM state.
    -   **Evaluate**: Checks constraints against history.
    -   **Schedule**: If violation, schedules intervention for *next* turn.

## 4. Configuration & CLI
-   **Configuration**: Pydantic settings (`settings.py`) loaded from Environment Variables (e.g., `PANOPTES_POLICY__ENGINE__TYPE`).
-   **CLI Tools**:
    -   `panoptes serve`: Starts the proxy.
    -   `panoptes compile`: NLP -> YAML.
    -   `panoptes validate`: Checks YAML correctness.
    -   `panoptes info`: Inspects compiled workflows.
    -   `panoptes debug`: (Implicit in logs/tracing).

## 5. Technology Stack
-   **Language**: Python 3.10+
-   **Web Framework**: FastAPI / Uvicorn
-   **LLM Interface**: LiteLLM
-   **Validation**: Pydantic
-   **Observability**: OpenTelemetry
-   **Design Pattern**: Strategy Pattern (Engines), Observer Pattern (Hooks).


