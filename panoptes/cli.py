"""
Panoptes CLI entry point.

Commands:
- panoptes serve: Start the proxy server
- panoptes validate: Validate a workflow file
- panoptes info: Show workflow information
"""

import logging
import sys
from pathlib import Path

import click

from panoptes import __version__


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.version_option(version=__version__, prog_name="panoptes")
def main():
    """Panoptes - Reliability layer for AI agents.

    Monitor workflow adherence and intervene when agents deviate.
    """
    pass


@main.command()
@click.option(
    "--workflow", "-w",
    type=click.Path(exists=True, path_type=Path),
    help="Path to workflow YAML file",
)
@click.option(
    "--port", "-p",
    type=int,
    default=4000,
    help="Proxy server port (default: 4000)",
)
@click.option(
    "--host", "-h",
    type=str,
    default="0.0.0.0",
    help="Proxy server host (default: 0.0.0.0)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def serve(workflow: Path, port: int, host: str, debug: bool):
    """Start the Panoptes proxy server.

    The proxy intercepts LLM calls and monitors workflow adherence.
    Point your LLM client's base_url to http://HOST:PORT/v1

    Example:
        panoptes serve --workflow workflow.yaml --port 4000
    """
    setup_logging(debug)

    from panoptes.config.settings import PanoptesSettings
    from panoptes.proxy.server import start_proxy

    click.echo(f"Starting Panoptes proxy on {host}:{port}")

    if workflow:
        click.echo(f"Using workflow: {workflow}")

    settings = PanoptesSettings(
        workflow_path=str(workflow) if workflow else None,
        debug=debug,
    )
    settings.proxy.host = host
    settings.proxy.port = port

    try:
        start_proxy(settings)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.argument(
    "workflow_path",
    type=click.Path(exists=True, path_type=Path),
)
def validate(workflow_path: Path):
    """Validate a workflow definition file.

    Checks that the workflow YAML is valid and all references are correct.

    Example:
        panoptes validate workflow.yaml
    """
    from panoptes.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(workflow_path)

        click.echo(click.style("✓ Valid workflow", fg="green"))
        click.echo(f"  Name: {workflow.name}")
        click.echo(f"  Version: {workflow.version}")
        click.echo(f"  States: {len(workflow.states)}")
        click.echo(f"  Transitions: {len(workflow.transitions)}")
        click.echo(f"  Constraints: {len(workflow.constraints)}")
        click.echo(f"  Interventions: {len(workflow.interventions)}")

    except FileNotFoundError:
        click.echo(click.style(f"✗ File not found: {workflow_path}", fg="red"), err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(click.style(f"✗ Validation error: {e}", fg="red"), err=True)
        raise SystemExit(1)


@main.command()
@click.argument(
    "workflow_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed information",
)
def info(workflow_path: Path, verbose: bool):
    """Show detailed workflow information.

    Displays states, transitions, constraints, and interventions.

    Example:
        panoptes info workflow.yaml --verbose
    """
    from panoptes.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(workflow_path)

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
            click.echo(f"\n{click.style('Transitions:', bold=True)} ({len(workflow.transitions)})")
            for t in workflow.transitions:
                click.echo(f"  • {t.from_state} → {t.to_state}")

        # Constraints
        if workflow.constraints:
            click.echo(f"\n{click.style('Constraints:', bold=True)} ({len(workflow.constraints)})")
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
            click.echo(f"\n{click.style('Interventions:', bold=True)} ({len(workflow.interventions)})")
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
    click.echo(f"Panoptes SDK v{__version__}")


if __name__ == "__main__":
    main()
