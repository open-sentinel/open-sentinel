"""Tests for workflow parsing and validation."""

import pytest
from pathlib import Path

from panoptes.policy.engines.fsm.workflow.parser import WorkflowParser
from panoptes.policy.engines.fsm.workflow.schema import WorkflowDefinition, ConstraintType


class TestWorkflowParser:
    """Tests for WorkflowParser."""

    def test_parse_file_yaml(self, sample_workflow_path: Path):
        """Test parsing a YAML workflow file."""
        workflow = WorkflowParser.parse_file(sample_workflow_path)

        assert workflow.name == "customer-support-agent"
        assert workflow.version == "1.0"
        assert len(workflow.states) > 0

    def test_parse_string_yaml(self):
        """Test parsing YAML from string."""
        yaml_content = """
name: test-workflow
version: "1.0"
states:
  - name: start
    is_initial: true
"""
        workflow = WorkflowParser.parse_string(yaml_content)

        assert workflow.name == "test-workflow"
        assert len(workflow.states) == 1
        assert workflow.states[0].is_initial is True

    def test_parse_dict(self, simple_workflow_dict):
        """Test parsing from dictionary."""
        workflow = WorkflowParser.parse_dict(simple_workflow_dict)

        assert workflow.name == "test-workflow"
        assert len(workflow.states) == 3

    def test_parse_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            WorkflowParser.parse_file("/nonexistent/path.yaml")

    def test_validate_file(self, sample_workflow_path: Path):
        """Test workflow validation."""
        is_valid, message = WorkflowParser.validate_file(sample_workflow_path)

        assert is_valid is True
        assert "Valid workflow" in message


class TestWorkflowSchema:
    """Tests for workflow schema validation."""

    def test_requires_initial_state(self):
        """Test that workflow must have an initial state."""
        with pytest.raises(ValueError, match="initial state"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "not_initial"}],
                }
            )

    def test_transition_references_valid_states(self):
        """Test that transitions must reference valid states."""
        with pytest.raises(ValueError, match="unknown state"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "transitions": [{"from_state": "start", "to_state": "nonexistent"}],
                }
            )

    def test_constraint_references_valid_states(self):
        """Test that constraints must reference valid states."""
        # Use 'eventually' type since 'never' constraints are allowed to reference
        # conceptual forbidden states that don't exist in the workflow
        with pytest.raises(ValueError, match="unknown"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "constraints": [
                        {
                            "name": "test",
                            "type": "eventually",
                            "target": "nonexistent",
                        }
                    ],
                }
            )

    def test_constraint_requires_parameters(self):
        """Test that constraints require appropriate parameters."""
        with pytest.raises(ValueError, match="requires"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "constraints": [
                        {
                            "name": "test",
                            "type": "precedence",
                            # Missing trigger and target
                        }
                    ],
                }
            )

    def test_valid_constraint_types(self, simple_workflow):
        """Test all constraint types are valid."""
        for ctype in ConstraintType:
            assert ctype.value in [
                "eventually",
                "always",
                "never",
                "until",
                "next",
                "response",
                "precedence",
            ]

    def test_get_state(self, simple_workflow):
        """Test getting state by name."""
        state = simple_workflow.get_state("start")
        assert state is not None
        assert state.name == "start"

        assert simple_workflow.get_state("nonexistent") is None

    def test_get_initial_states(self, simple_workflow):
        """Test getting initial states."""
        initial = simple_workflow.get_initial_states()
        assert len(initial) == 1
        assert initial[0].name == "start"

    def test_get_terminal_states(self, simple_workflow):
        """Test getting terminal states."""
        terminal = simple_workflow.get_terminal_states()
        assert len(terminal) == 1
        assert terminal[0].name == "end"
