# Open Sentinel Examples

This directory contains examples for different policy engines.

## 1. Judge Engine (Recommended)
Located in `examples/judge/`.

Uses an LLM to evaluate responses against a rubric.

### How to run:

1. Start Open Sentinel:
   ```bash
   cd examples/judge
   export GOOGLE_API_KEY=AIza...
   osentinel serve
   ```
   *This uses the local `osentinel.yaml` configuration (Gemini model). If you switch to an OpenAI model in the config, use `OPENAI_API_KEY` instead.*

2. Run the agent client:
   ```bash
   python llm_judge.py
   ```

## 2. Gemini + FSM Engine
Located in `examples/gemini_fsm/`.

Demonstrates deterministic workflow enforcement.

### How to run:

1. Start Open Sentinel:
   ```bash
   cd examples/gemini_fsm
   export GOOGLE_API_KEY=AIza...
   osentinel serve
   ```

2. Run the agent script:
   ```bash
   python gemini_agent.py
   ```

## 3. NeMo Guardrails
Located in `examples/nemo_guardrails/`.

Demonstrates NeMo Guardrails integration.

### How to run:

1. Start Open Sentinel:
   ```bash
   cd examples/nemo_guardrails
   export OPENAI_API_KEY=sk-...  # or GOOGLE_API_KEY depending on config
   osentinel serve
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
   osentinel compile "verify identity before processing refunds" -o refund_policy.yaml
   ```

2. Run with generated policy:
   ```bash
   # Create a osentinel.yaml or run with flags
   osentinel serve --policy refund_policy.yaml --engine fsm
   ```
