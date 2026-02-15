"""
Tests for the core judge evaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from panoptes.policy.engines.judge.evaluator import JudgeEvaluator
from panoptes.policy.engines.judge.client import JudgeClient
from panoptes.policy.engines.judge.models import (
    VerdictAction,
    EvaluationScope,
    ScoreScale,
    RubricCriterion,
    Rubric,
    EvaluationType,
)


@pytest.fixture
def mock_client():
    """Create a JudgeClient with a mocked call_judge method."""
    client = JudgeClient()
    client.add_model("primary", "gpt-4o-mini")
    client.call_judge = AsyncMock()
    client.get_model_id = MagicMock(return_value="gpt-4o-mini")
    client.get_tokens_for_model = MagicMock(return_value=100)
    return client


@pytest.fixture
def evaluator(mock_client):
    return JudgeEvaluator(
        client=mock_client,
        pass_threshold=0.6,
        warn_threshold=0.4,
        block_threshold=0.2,
    )


@pytest.fixture
def simple_rubric():
    return Rubric(
        name="test_rubric",
        description="Test rubric",
        criteria=[
            RubricCriterion(
                name="quality",
                description="Response quality",
                scale=ScoreScale.LIKERT_5,
                weight=1.0,
            ),
            RubricCriterion(
                name="safety",
                description="Response safety",
                scale=ScoreScale.LIKERT_5,
                weight=1.0,
            ),
        ],
        pass_threshold=0.6,
        fail_action=VerdictAction.WARN,
    )


@pytest.fixture
def conversation():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]


class TestEvaluateTurn:
    @pytest.mark.asyncio
    async def test_passing_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": ["clear"], "confidence": 0.9},
                {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.95},
            ],
            "summary": "Good response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="Python is a programming language.",
            conversation=conversation,
        )

        assert verdict.action == VerdictAction.PASS
        assert verdict.composite_score == 1.0
        assert verdict.scope == EvaluationScope.TURN
        assert verdict.judge_model == "gpt-4o-mini"
        assert len(verdict.scores) == 2

    @pytest.mark.asyncio
    async def test_failing_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 1, "reasoning": "Bad", "evidence": [], "confidence": 0.9},
                {"criterion": "safety", "score": 1, "reasoning": "Unsafe", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Poor response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="bad response",
            conversation=conversation,
        )

        assert verdict.action == VerdictAction.BLOCK
        assert verdict.composite_score == 0.0

    @pytest.mark.asyncio
    async def test_warning_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.8},
                {"criterion": "safety", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.8},
            ],
            "summary": "Average response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="ok response",
            conversation=conversation,
        )

        # 3/5 -> normalized 0.5, below pass_threshold 0.6, above warn_threshold 0.4
        assert verdict.action == VerdictAction.WARN

    @pytest.mark.asyncio
    async def test_missing_criterion_filled(self, evaluator, mock_client, simple_rubric, conversation):
        """Missing criteria should be filled with min score."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                # "safety" is missing
            ],
            "summary": "Partial response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        assert len(verdict.scores) == 2
        safety_score = next(s for s in verdict.scores if s.criterion == "safety")
        assert safety_score.score == 1  # min_score for LIKERT_5
        assert safety_score.confidence == 0.0


class TestEvaluateConversation:
    @pytest.mark.asyncio
    async def test_conversation_scope(self, evaluator, mock_client, conversation):
        conv_rubric = Rubric(
            name="conv_test",
            description="Test",
            criteria=[
                RubricCriterion(name="consistency", description="Consistent?", scale=ScoreScale.LIKERT_5),
            ],
            scope=EvaluationType.POINTWISE,
            pass_threshold=0.6,
        )

        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "consistency", "score": 4, "reasoning": "Good", "evidence": ["turn 1"], "confidence": 0.9},
            ],
            "summary": "Consistent conversation",
        }

        verdict = await evaluator.evaluate_conversation(
            model_name="primary",
            rubric=conv_rubric,
            full_conversation=conversation,
        )

        assert verdict.scope == EvaluationScope.CONVERSATION
        assert verdict.action == VerdictAction.PASS


class TestEvaluatePairwise:
    @pytest.mark.asyncio
    async def test_pairwise_evaluation(self, evaluator, mock_client, conversation):
        pairwise_rubric = Rubric(
            name="pair_test",
            description="Test",
            criteria=[
                RubricCriterion(name="overall_preference", description="Which is better?", scale=ScoreScale.LIKERT_5),
            ],
            evaluation_type=EvaluationType.PAIRWISE,
            pass_threshold=0.5,
        )

        mock_client.call_judge.return_value = {
            "scores": [
                {
                    "criterion": "overall_preference",
                    "score_a": 4, "score_b": 2,
                    "winner": "a",
                    "reasoning": "A is better",
                    "evidence": ["clearer"],
                    "confidence": 0.85,
                },
            ],
            "overall_winner": "a",
            "summary": "Response A is better",
        }

        verdict = await evaluator.evaluate_pairwise(
            model_name="primary",
            rubric=pairwise_rubric,
            response_a="Good answer",
            response_b="Bad answer",
            conversation=conversation,
        )

        assert verdict.metadata.get("pairwise") is True
        assert len(verdict.scores) == 1


class TestCompositeScoring:
    def test_weighted_composite(self, evaluator):
        from panoptes.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok"),  # normalized 1.0
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="ok"),  # normalized 0.0
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        composite = evaluator._compute_composite(scores, criteria)
        assert composite == 0.5

    def test_weighted_composite_unequal_weights(self, evaluator):
        from panoptes.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok"),  # normalized 1.0
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="ok"),  # normalized 0.0
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=3.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        composite = evaluator._compute_composite(scores, criteria)
        assert composite == 0.75

    def test_empty_scores(self, evaluator):
        assert evaluator._compute_composite([], []) == 0.0


class TestActionMapping:
    def test_pass(self, evaluator, simple_rubric):
        assert evaluator._map_action(0.8, simple_rubric) == VerdictAction.PASS

    def test_warn(self, evaluator, simple_rubric):
        assert evaluator._map_action(0.5, simple_rubric) == VerdictAction.WARN

    def test_fail_action(self, evaluator, simple_rubric):
        # Between block_threshold and warn_threshold -> rubric's fail_action
        assert evaluator._map_action(0.3, simple_rubric) == VerdictAction.WARN

    def test_block(self, evaluator, simple_rubric):
        assert evaluator._map_action(0.1, simple_rubric) == VerdictAction.BLOCK
