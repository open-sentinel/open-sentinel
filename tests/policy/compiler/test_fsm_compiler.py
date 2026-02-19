"""Tests for FSM policy compiler."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import PolicyCompilerRegistry
from opensentinel.policy.engines.fsm.compiler import FSMCompiler
from opensentinel.policy.engines.fsm.workflow.schema import (
    WorkflowDefinition,
    ConstraintType,
)


class TestFSMCompilerRegistration:
    """Test FSM compiler is properly registered."""

    def test_fsm_compiler_registered(self):
        """Test that FSMCompiler is auto-registered."""
        assert PolicyCompilerRegistry.is_registered("fsm")

    def test_create_fsm_compiler(self):
        """Test creating FSMCompiler via engine.get_compiler()."""
        from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine

        engine = FSMPolicyEngine()
        compiler = engine.get_compiler()

        assert isinstance(compiler, FSMCompiler)
        assert compiler.engine_type == "fsm"


class TestFSMCompilerPromptBuilding:
    """Tests for prompt construction."""

    @pytest.fixture
    def compiler(self):
        """Create FSM compiler."""
        return FSMCompiler()

    def test_build_basic_prompt(self, compiler):
        """Test building prompt without context."""
        prompt = compiler._build_compilation_prompt(
            "verify identity before refunds"
        )

        assert "verify identity before refunds" in prompt
        assert "precedence" in prompt.lower()
        assert "states" in prompt.lower()

    def test_build_prompt_with_context(self, compiler):
        """Test building prompt with context hints."""
        context = {
            "domain": "customer support",
            "tool_names": ["verify_identity", "process_refund"],
            "existing_states": ["greeting", "resolution"],
        }

        prompt = compiler._build_compilation_prompt(
            "verify identity before refunds",
            context=context,
        )

        assert "customer support" in prompt
        assert "verify_identity" in prompt
        assert "greeting" in prompt


class TestFSMCompilerResponseParsing:
    """Tests for LLM response parsing."""

    @pytest.fixture
    def compiler(self):
        """Create FSM compiler."""
        return FSMCompiler()

    def test_parse_valid_response(self, compiler):
        """Test parsing a valid LLM response."""
        response = {
            "name": "test-workflow",
            "description": "Test workflow",
            "states": [
                {
                    "name": "identity_verification",
                    "description": "Verify customer identity",
                    "is_initial": True,
                    "classification": {
                        "tool_calls": ["verify_identity"],
                        "patterns": ["verify.*identity"],
                    },
                },
                {
                    "name": "process_refund",
                    "description": "Process the refund",
                    "is_terminal": True,
                    "classification": {
                        "tool_calls": ["process_refund"],
                    },
                },
            ],
            "transitions": [
                {
                    "from_state": "identity_verification",
                    "to_state": "process_refund",
                    "description": "After verification, process refund",
                },
            ],
            "constraints": [
                {
                    "name": "verify_before_refund",
                    "description": "Must verify identity before refund",
                    "type": "precedence",
                    "trigger": "process_refund",
                    "target": "identity_verification",
                    "severity": "error",
                    "intervention": "prompt_verification",
                },
            ],
            "interventions": {
                "prompt_verification": "Please verify the customer's identity first.",
            },
        }

        result = compiler._parse_compilation_response(response, "test policy")

        assert result.success is True
        assert isinstance(result.config, WorkflowDefinition)

        workflow: WorkflowDefinition = result.config
        assert workflow.name == "test-workflow"
        assert len(workflow.states) == 2
        assert len(workflow.transitions) == 1
        assert len(workflow.constraints) == 1

        # Check constraint
        constraint = workflow.constraints[0]
        assert constraint.type == ConstraintType.PRECEDENCE
        assert constraint.trigger == "process_refund"
        assert constraint.target == "identity_verification"

    def test_parse_response_auto_sets_initial_state(self, compiler):
        """Test that parser auto-sets initial state if missing."""
        response = {
            "name": "test",
            "states": [
                {"name": "state_a"},
                {"name": "state_b"},
            ],
        }

        result = compiler._parse_compilation_response(response, "test")

        assert result.success is True
        workflow: WorkflowDefinition = result.config
        assert workflow.states[0].is_initial is True
        assert any("No initial state" in w for w in result.warnings)

    def test_parse_response_auto_generates_intervention(self, compiler):
        """Test auto-generation of missing interventions."""
        response = {
            "name": "test",
            "states": [
                {"name": "state_a", "is_initial": True},
                {"name": "state_b"},
            ],
            "constraints": [
                {
                    "name": "test_constraint",
                    "type": "never",
                    "target": "forbidden_state",
                    "intervention": "missing_intervention",
                },
            ],
        }

        result = compiler._parse_compilation_response(response, "test")

        assert result.success is True
        workflow: WorkflowDefinition = result.config
        assert "missing_intervention" in workflow.interventions
        assert any("Auto-generated" in w for w in result.warnings)

    def test_parse_response_handles_unknown_constraint_type(self, compiler):
        """Test handling of unknown constraint types."""
        response = {
            "name": "test",
            "states": [{"name": "state_a", "is_initial": True}],
            "constraints": [
                {
                    "name": "bad_constraint",
                    "type": "invalid_type",
                    "target": "state_a",
                },
            ],
        }

        result = compiler._parse_compilation_response(response, "test")

        assert result.success is True
        workflow: WorkflowDefinition = result.config
        assert len(workflow.constraints) == 0
        assert any("Unknown constraint type" in w for w in result.warnings)

    def test_parse_response_no_states_fails(self, compiler):
        """Test that response with no states fails."""
        response = {"name": "test", "states": []}

        result = compiler._parse_compilation_response(response, "test")

        assert result.success is False
        assert any("No states" in e for e in result.errors)


class TestFSMCompilerValidation:
    """Tests for validation."""

    @pytest.fixture
    def compiler(self):
        """Create FSM compiler."""
        return FSMCompiler()

    def test_validate_valid_workflow(self, compiler):
        """Test validation of valid workflow."""
        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "state_a", "is_initial": True},
                {"name": "state_b"},
            ],
            transitions=[
                {"from_state": "state_a", "to_state": "state_b"},
            ],
        )
        result = CompilationResult(success=True, config=workflow)

        errors = compiler.validate_result(result)

        assert errors == []

    def test_validate_invalid_constraint_reference(self, compiler):
        """Test validation catches invalid constraint trigger references."""
        # Create a workflow with a constraint referencing a non-existent trigger
        # Note: WorkflowDefinition validates transitions at creation time,
        # but our compiler validation adds an extra layer for edge cases
        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "state_a", "is_initial": True},
                {"name": "state_b"},
            ],
            transitions=[
                {"from_state": "state_a", "to_state": "state_b"},
            ],
        )

        # Manually add an invalid constraint to test validation
        # (This simulates a case where parsing created an invalid state)
        from opensentinel.policy.engines.fsm.workflow.schema import Constraint, ConstraintType
        workflow.constraints.append(
            Constraint(
                name="bad_constraint",
                type=ConstraintType.PRECEDENCE,
                trigger="nonexistent_trigger",
                target="state_a",
            )
        )

        result = CompilationResult(success=True, config=workflow)
        errors = compiler.validate_result(result)

        assert len(errors) > 0
        assert any("unknown trigger" in e.lower() for e in errors)


class TestFSMCompilerExport:
    """Tests for YAML export."""

    @pytest.fixture
    def compiler(self):
        """Create FSM compiler."""
        return FSMCompiler()

    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for export."""
        return WorkflowDefinition(
            name="test-export",
            version="1.0",
            description="Test workflow for export",
            states=[
                {
                    "name": "greeting",
                    "is_initial": True,
                    "classification": {"patterns": ["hello", "hi"]},
                },
                {
                    "name": "resolution",
                    "is_terminal": True,
                },
            ],
            transitions=[
                {"from_state": "greeting", "to_state": "resolution"},
            ],
            constraints=[
                {
                    "name": "test_constraint",
                    "type": "eventually",
                    "target": "resolution",
                    "intervention": "remind_resolution",
                },
            ],
            interventions={
                "remind_resolution": "Please work toward resolution.",
            },
        )

    def test_export_creates_yaml_file(self, compiler, sample_workflow, tmp_path):
        """Test that export creates a YAML file."""
        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "workflow.yaml"

        compiler.export(result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "test-export" in content
        assert "greeting" in content
        assert "resolution" in content

    def test_export_creates_parent_directories(self, compiler, sample_workflow, tmp_path):
        """Test that export creates parent directories."""
        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "nested" / "dir" / "workflow.yaml"

        compiler.export(result, output_path)

        assert output_path.exists()

    def test_export_failed_result_raises(self, compiler, tmp_path):
        """Test that exporting a failed result raises ValueError."""
        result = CompilationResult.failure(["error"])
        output_path = tmp_path / "workflow.yaml"

        with pytest.raises(ValueError, match="Cannot export failed"):
            compiler.export(result, output_path)

    def test_exported_yaml_is_valid_workflow(self, compiler, sample_workflow, tmp_path):
        """Test that exported YAML can be parsed back."""
        from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "workflow.yaml"

        compiler.export(result, output_path)

        # Parse the exported file
        parsed = WorkflowParser.parse_file(output_path)
        assert parsed.name == "test-export"
        assert len(parsed.states) == 2
        assert len(parsed.constraints) == 1


class TestFSMCompilerIntegration:
    """Integration tests for FSM compiler (requires mocked LLM)."""

    @pytest.fixture
    def compiler(self):
        """Create FSM compiler."""
        return FSMCompiler(model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_compile_with_mocked_llm(self, compiler):
        """Test full compilation flow with mocked LLM."""
        mock_response = {
            "name": "customer-support",
            "description": "Customer support workflow",
            "states": [
                {
                    "name": "identity_verification",
                    "is_initial": True,
                    "classification": {"tool_calls": ["verify_identity"]},
                },
                {
                    "name": "process_refund",
                    "is_terminal": True,
                    "classification": {"tool_calls": ["process_refund"]},
                },
            ],
            "constraints": [
                {
                    "name": "verify_before_refund",
                    "type": "precedence",
                    "trigger": "process_refund",
                    "target": "identity_verification",
                    "intervention": "prompt_verify",
                },
            ],
            "interventions": {
                "prompt_verify": "Please verify customer identity first.",
            },
        }

        # Mock the LLM call
        with patch.object(compiler, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)

            result = await compiler.compile(
                "Agent must verify identity before processing refunds."
            )

            assert result.success is True
            assert isinstance(result.config, WorkflowDefinition)

            workflow: WorkflowDefinition = result.config
            assert workflow.name == "customer-support"
            assert len(workflow.states) == 2
            assert len(workflow.constraints) == 1

    @pytest.mark.asyncio
    async def test_compile_handles_llm_error(self, compiler):
        """Test compilation handles LLM errors gracefully."""
        with patch.object(compiler, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")

            result = await compiler.compile("test policy")

            assert result.success is False
            assert any("API error" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_compile_handles_invalid_json(self, compiler):
        """Test compilation handles invalid JSON response."""
        with patch.object(compiler, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "not valid json {"

            result = await compiler.compile("test policy")

            assert result.success is False
            assert any("JSON" in e for e in result.errors)
