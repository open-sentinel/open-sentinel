"""
Panoptes init command - interactive project bootstrapping.

Generates panoptes.yaml and a starter policy.yaml so users can
get running with minimal configuration.
"""

from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

PANOPTES_YAML_JUDGE = """\
engine: judge
policy: ./policy.yaml
port: 4000

judge:
  model: {model}
  mode: {mode}

tracing:
  type: {tracing}
"""

PANOPTES_YAML_FSM = """\
engine: fsm
policy: ./workflow.yaml
port: 4000

tracing:
  type: {tracing}
"""

STARTER_POLICY_YAML = """\
rubrics:
  - name: custom_policy
    description: "Custom evaluation policy"
    scope: turn
    evaluation_type: pointwise
    pass_threshold: 0.6
    fail_action: warn
    criteria:
      - name: professional_tone
        description: "Is the response professional and appropriate?"
        scale: likert_5
        weight: 1.0
      - name: no_pii
        description: "Does the response avoid sharing PII?"
        scale: binary
        weight: 1.0
        fail_threshold: 0.5
"""

STARTER_WORKFLOW_YAML = """\
name: starter-workflow
version: "1.0"
description: "Starter FSM workflow - customize states and transitions"

states:
  - name: greeting
    description: "Initial greeting and intent detection"
    is_initial: true
    classification:
      exemplars:
        - "Hello, how can I help you?"
        - "Welcome! What can I do for you today?"

  - name: processing
    description: "Processing the user's request"
    classification:
      exemplars:
        - "Let me look into that for you"
        - "I'm working on your request"

  - name: completed
    description: "Request has been fulfilled"
    is_terminal: true
    classification:
      exemplars:
        - "Is there anything else I can help with?"
        - "Your request has been completed"

transitions:
  - from_state: greeting
    to_state: processing
    description: "User states their request"
  - from_state: processing
    to_state: completed
    description: "Request is fulfilled"
  - from_state: completed
    to_state: greeting
    description: "User has a new request"

constraints:
  - name: must_greet_first
    type: precedence
    trigger: processing
    target: greeting
    severity: warning
    intervention: remind_greeting
    description: "Agent should greet before processing"

interventions:
  remind_greeting: "Please greet the user before proceeding with their request."
"""


def run_init(
    engine: Optional[str] = None,
    non_interactive: bool = False,
) -> None:
    """Run the init flow, writing panoptes.yaml and a starter policy file.

    Args:
        engine: Engine type override (skips prompt if set).
        non_interactive: Use all defaults without prompting.
    """
    # 1. Engine type
    if engine:
        engine_type = engine
    elif non_interactive:
        engine_type = "judge"
    else:
        engine_type = click.prompt(
            "Policy engine type",
            type=click.Choice(["judge", "fsm"]),
            default="judge",
        )

    # Collect judge-specific options
    model = "gpt-4o-mini"
    mode = "balanced"
    tracing = "none"

    if engine_type == "judge":
        if not non_interactive:
            model = click.prompt("Judge model", default="gpt-4o-mini")
            mode = click.prompt(
                "Reliability mode",
                type=click.Choice(["safe", "balanced", "aggressive"]),
                default="balanced",
            )

    # Tracing
    if not non_interactive:
        tracing = click.prompt(
            "Tracing exporter",
            type=click.Choice(["none", "console", "otlp", "langfuse"]),
            default="none",
        )

    # 2. Generate files
    config_path = Path("panoptes.yaml")
    if config_path.exists() and not non_interactive:
        if not click.confirm(
            f"{config_path} already exists. Overwrite?", default=False
        ):
            click.echo("Aborted.")
            return

    if engine_type == "judge":
        config_content = PANOPTES_YAML_JUDGE.format(
            model=model,
            mode=mode,
            tracing=tracing,
        )
        policy_path = Path("policy.yaml")
        policy_content = STARTER_POLICY_YAML
    else:
        config_content = PANOPTES_YAML_FSM.format(tracing=tracing)
        policy_path = Path("workflow.yaml")
        policy_content = STARTER_WORKFLOW_YAML

    # Write config
    config_path.write_text(config_content)
    click.echo(click.style(f"  Created {config_path}", fg="green"))

    # Write starter policy (only if it doesn't exist)
    if policy_path.exists() and not non_interactive:
        if click.confirm(f"{policy_path} already exists. Overwrite?", default=False):
            policy_path.write_text(policy_content)
            click.echo(click.style(f"  Created {policy_path}", fg="green"))
        else:
            click.echo(f"  Kept existing {policy_path}")
    else:
        policy_path.write_text(policy_content)
        click.echo(click.style(f"  Created {policy_path}", fg="green"))

    # 3. Print next steps
    click.echo("")
    click.echo(click.style("Setup complete!", bold=True))
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Export your API key:")
    click.echo(f"     export OPENAI_API_KEY=sk-...")
    click.echo(f"  2. Review and customize {policy_path}")
    click.echo(f"  3. Start the proxy:")
    click.echo(f"     panoptes serve")
    click.echo("")
    click.echo(f"Your LLM client should point to: http://localhost:4000/v1")
