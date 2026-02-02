# Panoptes Examples

This directory contains examples of how to use Panoptes with different policy engines.

## 1. Gemini + FSM Engine
Located in `examples/gemini_fsm/`.

This example demonstrates using the Finite State Machine (FSM) engine to enforce a specific workflow for a customer support agent.

### How to run:

1. Start Panoptes with FSM engine (default if not specified, but good to be explicit):
   ```bash
   export PANOPTES_POLICY__ENGINE__TYPE=fsm
   export PANOPTES_WORKFLOW_PATH=./examples/gemini_fsm/customer_support.yaml
   export GOOGLE_API_KEY=your_key_here
   panoptes serve
   ```

2. Run the agent script:
   ```bash
   python examples/gemini_fsm/gemini_agent.py
   ```

## 2. NeMo Guardrails
Located in `examples/nemo_guardrails/`.

This example demonstrates using NVIDIA NeMo Guardrails to filter input/output, specifically blocking financial advice.

### File Structure:
- `config/config.yml`: NeMo configuration defining the model and rails.
- `config/rails.co`: Colang definitions for the guardrails.
- `nemo_agent.py`: Client script.

### How to run:

1. Start Panoptes with NeMo engine:
   ```bash
   export PANOPTES_POLICY__ENGINE__TYPE=nemo
   export PANOPTES_POLICY__ENGINE__CONFIG__CONFIG_PATH=$(pwd)/examples/nemo_guardrails/config/
   export GOOGLE_API_KEY=your_key_here
   panoptes serve
   ```
   *Note: Use absolute path for config path if possible, or ensure it resolves correctly.*

2. Run the agent script:
   ```bash
   python examples/nemo_guardrails/nemo_agent.py
   ```
