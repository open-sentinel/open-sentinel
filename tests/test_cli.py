"""Tests for opensentinel.cli commands."""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock
from pathlib import Path

from click.testing import CliRunner

from opensentinel.cli import main


def _invoke(args):
    """Invoke CLI and capture both Click output and Rich stdout."""
    runner = CliRunner()
    # Rich writes to sys.stdout; CliRunner captures click.echo output.
    # We need to capture both.
    buf = StringIO()
    from opensentinel.cli_ui import console

    old_file = console.file
    console.file = buf
    try:
        result = runner.invoke(main, args)
    finally:
        console.file = old_file
    combined = result.output + buf.getvalue()
    return result, combined


class TestVersionCommand:
    def test_version_output(self):
        result, output = _invoke(["version"])
        assert result.exit_code == 0
        assert "Open Sentinel" in output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "osentinel" in result.output


class TestValidateCommand:
    def test_validate_missing_file(self):
        result, _ = _invoke(["validate", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_validate_valid_workflow(self):
        """Test validate with a mock workflow parser."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.yaml").write_text("name: test\nversion: '1.0'")

            mock_workflow = MagicMock()
            mock_workflow.name = "Test"
            mock_workflow.version = "1.0"
            mock_workflow.states = []
            mock_workflow.transitions = []
            mock_workflow.constraints = []
            mock_workflow.interventions = {}

            buf = StringIO()
            from opensentinel.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                with patch(
                    "opensentinel.policy.engines.fsm.workflow.parser.WorkflowParser.parse_file",
                    return_value=mock_workflow,
                ):
                    result = runner.invoke(main, ["validate", "test.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "Valid Workflow" in combined


class TestInitCommand:
    def test_init_no_interactive_flag(self):
        """Verify -i flag was removed."""
        result, _ = _invoke(["init", "-i"])
        assert result.exit_code != 0

    def test_init_quick_flag_exists(self):
        """Verify --quick flag works."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Mock run_init since it might try to write files or check env
            with patch("opensentinel.cli_init.run_init") as mock_run:
                result = runner.invoke(main, ["init", "--quick"])
                assert result.exit_code == 0
                mock_run.assert_called_once_with(quick=True)

    def test_init_non_tty_without_from(self):
        """Without --from and without TTY, should show error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("opensentinel.cli_init.is_interactive", return_value=False):
                result = runner.invoke(main, ["init"])
                assert result.exit_code != 0


class TestServeCommand:
    def test_serve_missing_config(self):
        result, _ = _invoke(["serve", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0


class TestCompileCommand:
    def test_compile_help(self):
        result, _ = _invoke(["compile", "--help"])
        assert result.exit_code == 0
        assert "POLICY" in result.output


class TestHelpOutput:
    def test_main_help(self):
        result, _ = _invoke(["--help"])
        assert result.exit_code == 0
        assert "Open Sentinel" in result.output
        assert "init" in result.output
        assert "serve" in result.output
        assert "compile" in result.output
        assert "validate" in result.output
        assert "info" in result.output
        assert "version" in result.output
