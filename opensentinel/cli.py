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
from opensentinel.cli_ui import (
    config_panel,
    console,
    dim,
    error,
    key_value,
    make_table,
    next_steps,
    spinner,
    success,
    warning,
    yaml_preview,
)
from rich.text import Text


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
def main() -> None:
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
def serve(port: int, host: str, config: Path, debug: bool) -> None:
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

    try:
        with spinner("Loading configuration..."):
            settings = SentinelSettings(
                _config_path=str(config) if config else None,
                debug=debug,
            )
            settings.proxy.host = host
            settings.proxy.port = port
            settings.validate()

        # Show config summary
        engine = getattr(settings, "engine", "judge")
        model = getattr(settings, "model", "default")
        fail_open = getattr(settings.policy, "fail_open", True) if hasattr(settings, "policy") else True
        config_panel(
            "Open Sentinel Proxy",
            {
                "Engine": str(engine),
                "Model": str(model),
                "Host": f"{host}:{port}",
                "Fail Open": str(fail_open),
            },
        )
        console.print(
            Text.assemble(
                ("  Listening on ", ""),
                (f"http://{host}:{port}/v1", "bold cyan underline"),
            )
        )
        console.print()

    except Exception as e:
        error(str(e), hint="Check your osentinel.yaml or run: osentinel init")
        if debug:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)

    try:
        start_proxy(settings)
    except KeyboardInterrupt:
        dim("\nShutting down...")
    except Exception as e:
        error(str(e))
        raise SystemExit(1)


@main.command()
@click.option(
    "--from",
    "compile_from",
    type=str,
    default=None,
    help="Natural language policy description to compile into rules",
)
def init(compile_from: str) -> None:
    """Initialize a new Open Sentinel project.

    Creates an osentinel.yaml in the current directory.
    Without --from, runs an interactive wizard with arrow-key selection.

    Examples:

        # Interactive setup (default)
        osentinel init

        # Compile from natural language
        osentinel init --from "customer support bot, never share internal pricing"
    """
    from opensentinel.cli_init import run_init

    run_init(compile_from=compile_from)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
def validate(config_path: Path) -> None:
    """Validate a workflow definition file.

    Checks that the workflow YAML is valid and all references are correct.

    Example:
        osentinel validate workflow.yaml
    """
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        config_panel(
            "\u2713 Valid Workflow",
            {
                "Name": workflow.name,
                "Version": workflow.version,
                "States": str(len(workflow.states)),
                "Transitions": str(len(workflow.transitions)),
                "Constraints": str(len(workflow.constraints)),
                "Interventions": str(len(workflow.interventions)),
            },
        )

    except FileNotFoundError:
        error(f"File not found: {config_path}")
        raise SystemExit(1)
    except Exception as e:
        error(f"Validation error: {e}")
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
def info(config_path: Path, verbose: bool) -> None:
    """Show detailed workflow information.

    Displays states, transitions, constraints, and interventions.

    Example:
        osentinel info workflow.yaml --verbose
    """
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        console.print()
        console.print(
            Text.assemble(
                (workflow.name, "bold"),
                (f"  v{workflow.version}", "dim"),
            )
        )
        if workflow.description:
            dim(workflow.description)

        # States table
        state_rows: list[list[str]] = []
        for state in workflow.states:
            flags = []
            if state.is_initial:
                flags.append("[green]initial[/]")
            if state.is_terminal:
                flags.append("[blue]terminal[/]")
            if state.is_error:
                flags.append("[red]error[/]")
            flag_str = ", ".join(flags) if flags else "-"

            desc = ""
            if verbose and state.description:
                desc = state.description
            state_rows.append([state.name, flag_str, desc])

        columns = ["Name", "Type", "Description"] if verbose else ["Name", "Type"]
        rows = [r if verbose else r[:2] for r in state_rows]
        make_table("States", columns, rows)

        # Transitions table
        if workflow.transitions:
            t_rows = [[t.from_state, f"\u2192 {t.to_state}"] for t in workflow.transitions]
            make_table("Transitions", ["From", "To"], t_rows)

        # Constraints table
        if workflow.constraints:
            severity_markup = {
                "warning": "[yellow]warning[/]",
                "error": "[red]error[/]",
                "critical": "[magenta]critical[/]",
            }
            c_rows = []
            for c in workflow.constraints:
                sev = severity_markup.get(c.severity, c.severity)
                row = [c.name, f"[cyan]{c.type.value}[/]", sev]
                if verbose and c.description:
                    row.append(c.description)
                elif verbose:
                    row.append("")
                c_rows.append(row)

            cols = ["Name", "Type", "Severity"]
            if verbose:
                cols.append("Description")
            make_table("Constraints", cols, c_rows)

        # Interventions
        if workflow.interventions:
            console.print()
            console.print("[bold]Interventions:[/]")
            for name, template in workflow.interventions.items():
                if verbose:
                    preview = template[:80] + "..." if len(template) > 80 else template
                    key_value(name, f"[dim]{preview}[/]", indent=2)
                else:
                    console.print(f"  {name}")

        console.print()

    except Exception as e:
        error(str(e))
        raise SystemExit(1)


@main.command()
def version() -> None:
    """Show version information."""
    console.print(
        Text.assemble(
            ("Open Sentinel", "bold"),
            (f" v{__version__}", "dim"),
        )
    )


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
    help="Base URL for LLM API",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="API key for LLM provider",
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
) -> None:
    """Compile natural language policy to engine configuration.

    POLICY is the natural language policy description.

    Examples:

        # Auto-detect engine type
        osentinel compile "be professional, never leak PII"

        # Explicit judge engine
        osentinel compile "never share internal info" --engine judge

        # FSM workflow
        osentinel compile "verify identity before refunds" --engine fsm
    """
    import asyncio

    setup_logging(debug)

    # Handle 'auto' engine selection
    if engine == "auto":
        engine = _detect_engine_type(policy)
        dim(f"Auto-detected engine: {engine}")

    # Determine output path based on engine type
    if output is None:
        output = Path("policy.yaml") if engine == "judge" else Path("workflow.yaml")

    # Build context
    context = {}
    if domain:
        context["domain"] = domain

    async def run_compile() -> None:
        import os
        from opensentinel.policy.compiler import PolicyCompilerRegistry

        try:
            compiler = PolicyCompilerRegistry.create(engine)

            if hasattr(compiler, "model"):
                compiler.model = model

            if base_url and hasattr(compiler, "_base_url"):
                compiler._base_url = base_url

            resolved_api_key = api_key
            if not resolved_api_key and model.startswith("gemini"):
                resolved_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv(
                    "GEMINI_API_KEY"
                )
            if resolved_api_key and hasattr(compiler, "_api_key"):
                compiler._api_key = resolved_api_key

            key_value("Engine", engine)
            key_value("Model", model)
            if base_url:
                key_value("Base URL", base_url)

            with spinner("Compiling policy..."):
                result = await compiler.compile(policy, context if context else None)

            if not result.success:
                error("Compilation failed")
                for err in result.errors:
                    console.print(f"    [dim]{err}[/]")
                raise SystemExit(1)

            for w in result.warnings:
                warning(w)

            if validate:
                validation_errors = compiler.validate_result(result)
                if validation_errors:
                    error("Validation errors:")
                    for err in validation_errors:
                        console.print(f"    [dim]{err}[/]")
                    raise SystemExit(1)

            compiler.export(result, output)
            success("Compiled successfully")

            # Show summary panel
            summary: dict[str, str] = {"Output": str(output)}
            if result.metadata:
                if "state_count" in result.metadata:
                    summary["States"] = str(result.metadata["state_count"])
                if "constraint_count" in result.metadata:
                    summary["Constraints"] = str(result.metadata["constraint_count"])
                if "rubric_count" in result.metadata:
                    summary["Rubrics"] = str(result.metadata["rubric_count"])
                if "criteria_count" in result.metadata:
                    summary["Criteria"] = str(result.metadata["criteria_count"])
            config_panel("Compilation Result", summary)

            # Show yaml preview
            try:
                generated = output.read_text()
                yaml_preview(generated, title=str(output))
            except OSError:
                pass

            # Next steps
            if engine == "judge":
                next_steps(
                    [
                        f"Review the generated rubric: {output}",
                        f"Update osentinel.yaml:  policy: {output}",
                        "Start the proxy: osentinel serve",
                    ]
                )
            else:
                next_steps(
                    [
                        f"Validate: osentinel validate {output}",
                        f"Update osentinel.yaml:  engine: fsm / policy: {output}",
                        "Start the proxy: osentinel serve",
                    ]
                )

        except ValueError as e:
            error(str(e))
            raise SystemExit(1)
        except ImportError as e:
            error(f"Missing dependency: {e}", hint="Install with: pip install openai")
            raise SystemExit(1)

    asyncio.run(run_compile())


if __name__ == "__main__":
    main()
