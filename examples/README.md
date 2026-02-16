# Panoptes Examples

This directory contains examples for different policy engines.

## 1. Judge Engine (Recommended)
Located in `examples/judge/`.

Uses an LLM to evaluate responses against a rubric.

### How to run:

1. Start Panoptes:
   ```bash
   cd examples/judge
   export OPENAI_API_KEY=sk-...
   panoptes serve
   ```
   *This uses the local `panoptes.yaml` configuration.*

2. Run the agent client:
   ```bash
   python judge_agent.py
   ```

## 2. Gemini + FSM Engine
Located in `examples/gemini_fsm/`.

Demonstrates deterministic workflow enforcement.

### How to run:

1. Start Panoptes:
   ```bash
   cd examples/gemini_fsm
   export GOOGLE_API_KEY=AIza...
   panoptes serve
   ```

2. Run the agent script:
   ```bash
   python gemini_agent.py
   ```

## 3. NeMo Guardrails
Located in `examples/nemo_guardrails/`.

Demonstrates NeMo Guardrails integration.

### How to run:

1. Start Panoptes:
   ```bash
   cd examples/nemo_guardrails
   export OPENAI_API_KEY=sk-...  # or GOOGLE_API_KEY depending on config
   panoptes serve
   ```

2. Run the agent script:
   ```bash
   python nemo_agent.py
   ```

## 4. NLP Policy Compiler (Experimental)

Compile natural language policies into executable workflows.

### How to use:

1. Compile a policy:
   ```bash
   panoptes compile "verify identity before processing refunds" -o refund_policy.yaml
   ```

2. Run with generated policy:
   ```bash
   # Create a panoptes.yaml or run with flags
   panoptes serve --policy refund_policy.yaml --engine fsm
   ```
