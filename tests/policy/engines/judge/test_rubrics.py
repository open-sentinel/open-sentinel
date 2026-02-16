"""
Tests for rubric registry and built-in rubrics.
"""

import pytest
from panoptes.policy.engines.judge.models import (
    EvaluationType,
    EvaluationScope,
    ScoreScale,
    VerdictAction,
)
from panoptes.policy.engines.judge.rubrics import RubricRegistry


class TestRubricRegistry:
    def test_builtin_rubrics_registered(self):
        """All built-in rubrics should be registered at import time."""
        names = RubricRegistry.list_rubrics()
        assert "general_quality" in names
        assert "instruction_following" in names
        assert "safety" in names
        assert "agent_behavior" in names
        assert "conversation_policy" in names
        assert "comparison" in names

    def test_get_existing_rubric(self):
        rubric = RubricRegistry.get("agent_behavior")
        assert rubric is not None
        assert rubric.name == "agent_behavior"
        assert len(rubric.criteria) == 4

    def test_get_nonexistent_rubric(self):
        assert RubricRegistry.get("nonexistent") is None

    def test_agent_behavior_rubric(self):
        rubric = RubricRegistry.get("agent_behavior")
        assert rubric.evaluation_type == EvaluationType.POINTWISE
        assert rubric.scope == EvaluationScope.TURN
        criteria_names = [c.name for c in rubric.criteria]
        assert "instruction_following" in criteria_names
        assert "tool_use_safety" in criteria_names
        assert "no_hallucination" in criteria_names
        assert "task_completion" in criteria_names

    def test_safety_rubric(self):
        rubric = RubricRegistry.get("safety")
        assert rubric.fail_action == VerdictAction.BLOCK
        assert rubric.pass_threshold == 0.8
        for criterion in rubric.criteria:
            assert criterion.scale == ScoreScale.BINARY

    def test_conversation_policy_rubric(self):
        rubric = RubricRegistry.get("conversation_policy")
        assert rubric.scope == EvaluationScope.CONVERSATION
        assert rubric.fail_action == VerdictAction.INTERVENE

    def test_comparison_rubric(self):
        rubric = RubricRegistry.get("comparison")
        assert rubric.evaluation_type == EvaluationType.PAIRWISE


class TestCreateRulesRubric:
    def test_creates_binary_rubric(self):
        from panoptes.policy.engines.judge.rubrics import create_rules_rubric
        rules = ["No financial advice", "Be professional"]
        rubric = create_rules_rubric(rules)

        assert rubric.name == "inline_policy"
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "policy_compliance"
        assert rubric.criteria[0].scale == ScoreScale.BINARY
        assert rubric.fail_action == VerdictAction.BLOCK
        assert "additional_instructions" in rubric.prompt_overrides
        assert "No financial advice" in rubric.prompt_overrides["additional_instructions"]
        assert "Be professional" in rubric.prompt_overrides["additional_instructions"]

    def test_custom_name(self):
        from panoptes.policy.engines.judge.rubrics import create_rules_rubric
        rubric = create_rules_rubric(["rule1"], name="my_policy")
        assert rubric.name == "my_policy"

