"""
Tests for confidence scoring and uncertainty estimation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.models import (
    Rubric,
    RubricCriterion,
    JudgeScore,
    JudgeVerdict,
    VerdictAction,
    ScoreScale,
)


@pytest.fixture
def mock_client():
    client = JudgeClient()
    client.add_model("primary", "gpt-4o-mini")
    client.call_judge = AsyncMock()
    client.get_model_id = MagicMock(return_value="gpt-4o-mini")
    client.get_tokens_for_model = MagicMock(return_value=100)
    return client


@pytest.fixture
def simple_rubric():
    return Rubric(
        name="test_rubric",
        description="Test",
        criteria=[
            RubricCriterion(name="quality", description="Quality", scale=ScoreScale.LIKERT_5, weight=1.0),
            RubricCriterion(name="safety", description="Safety", scale=ScoreScale.LIKERT_5, weight=1.0),
        ],
        pass_threshold=0.6,
    )


@pytest.fixture
def conversation():
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


class TestConfidenceComputation:
    def test_high_confidence(self, mock_client):
        evaluator = JudgeEvaluator(client=mock_client)
        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok", confidence=0.9),
            JudgeScore(criterion="b", score=4, max_score=5, reasoning="ok", confidence=0.95),
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        confidence = evaluator._compute_confidence(scores, criteria)
        assert confidence == pytest.approx(0.925)

    def test_low_confidence(self, mock_client):
        evaluator = JudgeEvaluator(client=mock_client)
        scores = [
            JudgeScore(criterion="a", score=3, max_score=5, reasoning="ok", confidence=0.3),
            JudgeScore(criterion="b", score=3, max_score=5, reasoning="ok", confidence=0.2),
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        confidence = evaluator._compute_confidence(scores, criteria)
        assert confidence == pytest.approx(0.25)

    def test_weighted_confidence(self, mock_client):
        evaluator = JudgeEvaluator(client=mock_client)
        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok", confidence=1.0),
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="ok", confidence=0.0),
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=3.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        confidence = evaluator._compute_confidence(scores, criteria)
        assert confidence == pytest.approx(0.75)

    def test_empty_scores(self, mock_client):
        evaluator = JudgeEvaluator(client=mock_client)
        assert evaluator._compute_confidence([], []) == 0.0


class TestConfidenceInVerdict:
    @pytest.mark.asyncio
    async def test_high_confidence_verdict(self, mock_client, simple_rubric, conversation):
        evaluator = JudgeEvaluator(client=mock_client, confidence_threshold=0.5)
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.95},
            ],
            "summary": "Good response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        assert verdict.overall_confidence == pytest.approx(0.925)
        assert verdict.low_confidence is False

    @pytest.mark.asyncio
    async def test_low_confidence_verdict(self, mock_client, simple_rubric, conversation):
        evaluator = JudgeEvaluator(client=mock_client, confidence_threshold=0.5)
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 3, "reasoning": "Unsure", "evidence": [], "confidence": 0.3},
                {"criterion": "safety", "score": 3, "reasoning": "Maybe", "evidence": [], "confidence": 0.2},
            ],
            "summary": "Uncertain",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        assert verdict.overall_confidence < 0.5
        assert verdict.low_confidence is True

    @pytest.mark.asyncio
    async def test_confidence_in_to_dict(self, mock_client, simple_rubric, conversation):
        evaluator = JudgeEvaluator(client=mock_client)
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Good",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        d = verdict.to_dict()
        assert "overall_confidence" in d
        assert "low_confidence" in d

    @pytest.mark.asyncio
    async def test_custom_confidence_threshold(self, mock_client, simple_rubric, conversation):
        """Custom threshold should affect low_confidence flag."""
        evaluator = JudgeEvaluator(client=mock_client, confidence_threshold=0.95)
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Good",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        # 0.9 < 0.95 threshold, so low_confidence should be True
        assert verdict.low_confidence is True


class TestConfidenceInEngine:
    @pytest.mark.asyncio
    async def test_low_confidence_in_metadata(self):
        from opensentinel.policy.engines.judge.engine import JudgePolicyEngine

        engine = JudgePolicyEngine()
        await engine.initialize({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "confidence_threshold": 0.5,
        })
        engine._client.call_judge = AsyncMock(return_value={
            "scores": [
                {"criterion": "instruction_following", "score": 3, "reasoning": "ok", "evidence": [], "confidence": 0.3},
                {"criterion": "tool_use_safety", "score": 3, "reasoning": "ok", "evidence": [], "confidence": 0.2},
                {"criterion": "no_hallucination", "score": 3, "reasoning": "ok", "evidence": [], "confidence": 0.3},
                {"criterion": "task_completion", "score": 3, "reasoning": "ok", "evidence": [], "confidence": 0.2},
            ],
            "summary": "Uncertain evaluation",
        })

        request = {"messages": [{"role": "user", "content": "Hello"}]}
        response = {"choices": [{"message": {"content": "Hi!"}}]}

        result = await engine.evaluate_response("s1", response, request)
        assert result.metadata["judge"]["low_confidence"] is True
        assert "confidence_warning" in result.metadata["judge"]
