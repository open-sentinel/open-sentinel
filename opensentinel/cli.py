"""
Open Sentinel CLI entry point.

Commands:
- osentinel init: Initialize a new Open Sentinel project
- osentinel serve: Start the proxy server
- osentinel compile: Compile natural language policy to engine config
- osentinel validate: Validate a workflow file
- osentinel info: Show workflow information
"""

import logging
import sys
from pathlib import Path

import click

from opensentinel import __version__


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.version_option(version=__version__, prog_name="osentinel")
def main():
    """Open Sentinel - Reliability layer for AI agents.

    Monitor workflow adherence and intervene when agents deviate.
    """
    pass


@main.command()
@click.option(
    "--port",
    "-p",
    type=int,
    default=4000,
    help="Proxy server port (default: 4000)",
)
@click.option(
    "--host",
    "-h",
    type=str,
    default="0.0.0.0",
    help="Proxy server host (default: 0.0.0.0)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to osentinel.yaml config file",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def serve(port: int, host: str, config: Path, debug: bool):
    """Start the Open Sentinel proxy server.

    The proxy intercepts LLM calls and monitors workflow adherence.
    Point your LLM client's base_url to http://HOST:PORT/v1

    Configure via osentinel.yaml or environment variables:
        osentinel serve -c osentinel.yaml
        osentinel serve --port 4000
    """
    setup_logging(debug)

    from opensentinel.config.settings import SentinelSettings
    from opensentinel.proxy.server import start_proxy

    click.echo(f"Starting Open Sentinel proxy on {host}:{port}")

    try:
        settings = SentinelSettings(
            _config_path=str(config) if config else None,
            debug=debug,
        )
        settings.proxy.host = host
        settings.proxy.port = port
        settings.validate()
        
    except Exception as e:
        click.echo(click.style(f"Configuration Error: {e}", fg="red"), err=True)
        # In debug mode, show the full traceback for context
        if debug:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)

    try:
        start_proxy(settings)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option(
    "--from",
    "compile_from",
    type=str,
    default=None,
    help="Natural language policy description to compile into rules",
)
def init(compile_from: str):
    """Initialize a new Open Sentinel project.

    Creates a minimal osentinel.yaml in the current directory.

    Examples:

        # Default setup (3 starter rules)
        osentinel init

        # Compile from natural language
        osentinel init --from "customer support bot, never share internal pricing"
    """
    from opensentinel.cli_init import run_init

    click.echo("")
    run_init(compile_from=compile_from)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
def validate(config_path: Path):
    """Validate a workflow definition file.

    Checks that the workflow YAML is valid and all references are correct.

    Example:
        osentinel validate workflow.yaml
    """
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        click.echo(click.style("✓ Valid workflow", fg="green"))
        click.echo(f"  Name: {workflow.name}")
        click.echo(f"  Version: {workflow.version}")
        click.echo(f"  States: {len(workflow.states)}")
        click.echo(f"  Transitions: {len(workflow.transitions)}")
        click.echo(f"  Constraints: {len(workflow.constraints)}")
        click.echo(f"  Interventions: {len(workflow.interventions)}")

    except FileNotFoundError:
        click.echo(click.style(f"✗ File not found: {config_path}", fg="red"), err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(click.style(f"✗ Validation error: {e}", fg="red"), err=True)
        raise SystemExit(1)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information",
)
def info(config_path: Path, verbose: bool):
    """Show detailed workflow information.

    Displays states, transitions, constraints, and interventions.

    Example:
        osentinel info workflow.yaml --verbose
    """
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        click.echo(f"\n{click.style(workflow.name, bold=True)} v{workflow.version}")
        if workflow.description:
            click.echo(f"  {workflow.description}")

        # States
        click.echo(f"\n{click.style('States:', bold=True)} ({len(workflow.states)})")
        for state in workflow.states:
            flags = []
            if state.is_initial:
                flags.append(click.style("initial", fg="green"))
            if state.is_terminal:
                flags.append(click.style("terminal", fg="blue"))
            if state.is_error:
                flags.append(click.style("error", fg="red"))

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            click.echo(f"  • {state.name}{flag_str}")

            if verbose and state.description:
                click.echo(f"    {state.description}")

            if verbose:
                hint = state.classification
                if hint.tool_calls:
                    click.echo(f"    Tools: {', '.join(hint.tool_calls)}")
                if hint.patterns:
                    click.echo(f"    Patterns: {len(hint.patterns)} patterns")
                if hint.exemplars:
                    click.echo(f"    Exemplars: {len(hint.exemplars)} examples")

        # Transitions
        if workflow.transitions:
            click.echo(
                f"\n{click.style('Transitions:', bold=True)} ({len(workflow.transitions)})"
            )
            for t in workflow.transitions:
                click.echo(f"  • {t.from_state} → {t.to_state}")

        # Constraints
        if workflow.constraints:
            click.echo(
                f"\n{click.style('Constraints:', bold=True)} ({len(workflow.constraints)})"
            )
            for c in workflow.constraints:
                severity_color = {
                    "warning": "yellow",
                    "error": "red",
                    "critical": "magenta",
                }.get(c.severity, "white")

                click.echo(
                    f"  • {c.name} "
                    f"[{click.style(c.type.value, fg='cyan')}] "
                    f"[{click.style(c.severity, fg=severity_color)}]"
                )
                if verbose and c.description:
                    click.echo(f"    {c.description}")

        # Interventions
        if workflow.interventions:
            click.echo(
                f"\n{click.style('Interventions:', bold=True)} ({len(workflow.interventions)})"
            )
            for name, template in workflow.interventions.items():
                click.echo(f"  • {name}")
                if verbose:
                    # Truncate long templates
                    preview = template[:80] + "..." if len(template) > 80 else template
                    click.echo(f"    {preview}")

        click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)


@main.command()
def version():
    """Show version information."""
    click.echo(f"Open Sentinel v{__version__}")


def _detect_engine_type(policy_text: str) -> str:
    """Auto-detect the best engine type from the policy text.

    Uses keyword heuristics:
    - Temporal/workflow keywords -> fsm
    - Quality/behavioral keywords -> judge
    - Default to judge if ambiguous
    """
    text_lower = policy_text.lower()

    fsm_keywords = {
        "before",
        "after",
        "first",
        "then",
        "state",
        "step",
        "workflow",
        "sequence",
        "transition",
        "phase",
        "stage",
        "proceed",
        "next",
        "previous",
        "order",
    }
    judge_keywords = {
        "ensure",
        "always",
        "never",
        "professional",
        "safe",
        "appropriate",
        "tone",
        "quality",
        "helpful",
        "accurate",
        "polite",
        "respectful",
        "pii",
        "harmful",
        "block",
        "warn",
        "evaluate",
        "score",
        "rubric",
    }

    fsm_score = sum(1 for kw in fsm_keywords if kw in text_lower)
    judge_score = sum(1 for kw in judge_keywords if kw in text_lower)

    if fsm_score > judge_score:
        return "fsm"
    return "judge"


@main.command()
@click.argument("policy", type=str)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["fsm", "judge", "auto"]),
    default="auto",
    help="Target engine type (default: auto)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: auto-selected based on engine)",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="gpt-4o-mini",
    help="LLM model for compilation (default: gpt-4o-mini)",
)
@click.option(
    "--domain",
    "-d",
    type=str,
    help="Application domain hint (e.g., 'customer support')",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate generated output (default: True)",
)
@click.option(
    "--base-url",
    "-b",
    type=str,
    help="Base URL for LLM API (e.g., http://localhost:4000/v1 for Open Sentinel proxy)",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="API key for LLM provider (uses OPENAI_API_KEY or GOOGLE_API_KEY env var if not set)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def compile(
    policy: str,
    engine: str,
    output: Path,
    model: str,
    domain: str,
    validate: bool,
    base_url: str,
    api_key: str,
    debug: bool,
):
    """Compile natural language policy to engine configuration.

    POLICY is the natural language policy description.

    Examples:

        # Auto-detect engine type
        osentinel compile "be professional, never leak PII"

        # Explicit judge engine
        osentinel compile "never share internal info" --engine judge

        # FSM workflow
        osentinel compile "verify identity before refunds" --engine fsm

        # With domain hint
        osentinel compile "verify before refunds" -d "customer support"

        # Use a different model
        osentinel compile "verify identity" -m gpt-4o
    """
    import asyncio

    setup_logging(debug)

    # Handle 'auto' engine selection
    if engine == "auto":
        engine = _detect_engine_type(policy)
        click.echo(f"Auto-detected engine: {engine}")

    # Determine output path based on engine type
    if output is None:
        output = Path("policy.yaml") if engine == "judge" else Path("workflow.yaml")

    # Build context
    context = {}
    if domain:
        context["domain"] = domain

    async def run_compile():
        import os
        from opensentinel.policy.compiler import PolicyCompilerRegistry

        try:
            # Create compiler
            compiler = PolicyCompilerRegistry.create(engine)

            # Override model if specified
            if hasattr(compiler, "model"):
                compiler.model = model

            # Set base_url if specified
            if base_url and hasattr(compiler, "_base_url"):
                compiler._base_url = base_url

            # Set api_key - auto-detect from env if using Gemini
            resolved_api_key = api_key
            if not resolved_api_key and model.startswith("gemini"):
                resolved_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv(
                    "GEMINI_API_KEY"
                )
            if resolved_api_key and hasattr(compiler, "_api_key"):
                compiler._api_key = resolved_api_key

            click.echo(f"Compiling policy with {engine} compiler...")
            click.echo(f"Model: {model}")
            if base_url:
                click.echo(f"Base URL: {base_url}")

            # Compile
            result = await compiler.compile(policy, context if context else None)

            if not result.success:
                click.echo(click.style("✗ Compilation failed", fg="red"), err=True)
                for error in result.errors:
                    click.echo(f"  Error: {error}", err=True)
                raise SystemExit(1)

            # Show warnings
            for warning in result.warnings:
                click.echo(click.style(f"  Warning: {warning}", fg="yellow"))

            # Validate if requested
            if validate:
                validation_errors = compiler.validate_result(result)
                if validation_errors:
                    click.echo(click.style("✗ Validation errors:", fg="red"), err=True)
                    for error in validation_errors:
                        click.echo(f"  {error}", err=True)
                    raise SystemExit(1)

            # Export
            compiler.export(result, output)

            click.echo(click.style("✓ Compiled successfully", fg="green"))
            click.echo(f"  Output: {output}")

            # Show summary
            if result.metadata:
                if "state_count" in result.metadata:
                    click.echo(f"  States: {result.metadata['state_count']}")
                if "constraint_count" in result.metadata:
                    click.echo(f"  Constraints: {result.metadata['constraint_count']}")
                if "rubric_count" in result.metadata:
                    click.echo(f"  Rubrics: {result.metadata['rubric_count']}")
                if "criteria_count" in result.metadata:
                    click.echo(f"  Criteria: {result.metadata['criteria_count']}")

            # Suggest next steps based on engine type
            click.echo(f"\nNext steps:")
            if engine == "judge":
                click.echo(f"  1. Review the generated rubric: {output}")
                click.echo(f"  2. Set policy path in osentinel.yaml or env var:")
                click.echo(f"     export OSNTL_POLICY__ENGINE__TYPE=judge")
                click.echo(f"     export OSNTL_POLICY__ENGINE__CONFIG_PATH={output}")
                click.echo(f"  3. Start the proxy: osentinel serve")
            else:
                click.echo(f"  osentinel validate {output}")
                click.echo(f"  export OSNTL_POLICY__ENGINE__TYPE=fsm")
                click.echo(f"  export OSNTL_POLICY__ENGINE__CONFIG_PATH={output}")
                click.echo(f"  osentinel serve")

        except ValueError as e:
            click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
            raise SystemExit(1)
        except ImportError as e:
            click.echo(
                click.style(f"✗ Missing dependency: {e}", fg="red"),
                err=True,
            )
            click.echo("Install with: pip install openai", err=True)
            raise SystemExit(1)

    asyncio.run(run_compile())


if __name__ == "__main__":
    main()
