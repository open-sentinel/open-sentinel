"""
Tests for multi-judge ensemble aggregation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opensentinel.policy.engines.judge.ensemble import (
    JudgeEnsemble,
    AggregationStrategy,
)
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.models import (
    Rubric,
    RubricCriterion,
    JudgeScore,
    JudgeVerdict,
    EnsembleVerdict,
    VerdictAction,
    EvaluationScope,
    ScoreScale,
)


@pytest.fixture
def mock_client():
    client = JudgeClient()
    client.add_model("primary", "gpt-4o-mini")
    client.add_model("secondary", "gpt-4o")
    client.call_judge = AsyncMock()
    client.get_model_id = MagicMock(side_effect=lambda n: f"model-{n}")
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
        description="Test",
        criteria=[
            RubricCriterion(name="quality", description="Quality", scale=ScoreScale.LIKERT_5),
            RubricCriterion(name="safety", description="Safety", scale=ScoreScale.LIKERT_5),
        ],
        pass_threshold=0.6,
    )


@pytest.fixture
def conversation():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


def _good_judge_response():
    return {
        "scores": [
            {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
            {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.95},
        ],
        "summary": "Good response",
    }


def _mediocre_judge_response():
    return {
        "scores": [
            {"criterion": "quality", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.7},
            {"criterion": "safety", "score": 4, "reasoning": "Mostly safe", "evidence": [], "confidence": 0.8},
        ],
        "summary": "Average response",
    }


def _bad_judge_response():
    return {
        "scores": [
            {"criterion": "quality", "score": 1, "reasoning": "Bad", "evidence": [], "confidence": 0.9},
            {"criterion": "safety", "score": 1, "reasoning": "Unsafe", "evidence": [], "confidence": 0.85},
        ],
        "summary": "Poor response",
    }


class TestAggregationStrategy:
    def test_valid_strategies(self):
        assert "mean_score" in AggregationStrategy.ALL
        assert "majority_vote" in AggregationStrategy.ALL
        assert "median_score" in AggregationStrategy.ALL
        assert "conservative" in AggregationStrategy.ALL

    def test_invalid_strategy_raises(self, evaluator):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            JudgeEnsemble(evaluator=evaluator, strategy="invalid")


class TestEnsembleEvaluateTurn:
    @pytest.mark.asyncio
    async def test_all_judges_agree_pass(self, evaluator, mock_client, simple_rubric, conversation):
        """When all judges agree, result should be PASS with high agreement."""
        mock_client.call_judge.return_value = _good_judge_response()

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="mean_score")
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        assert isinstance(result, EnsembleVerdict)
        assert result.final_action == VerdictAction.PASS
        assert result.agreement_rate == 1.0
        assert len(result.individual_verdicts) == 2
        assert result.aggregation_strategy == "mean_score"

    @pytest.mark.asyncio
    async def test_judges_disagree(self, evaluator, mock_client, simple_rubric, conversation):
        """When judges disagree, agreement rate should be < 1.0."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            _bad_judge_response(),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="mean_score")
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        assert result.agreement_rate < 1.0
        assert len(result.individual_verdicts) == 2

    @pytest.mark.asyncio
    async def test_low_agreement_escalates(self, evaluator, mock_client, simple_rubric, conversation):
        """Low agreement should trigger escalation."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            _bad_judge_response(),
        ]

        ensemble = JudgeEnsemble(
            evaluator=evaluator,
            strategy="mean_score",
            min_agreement=0.9,  # High threshold
        )
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        assert result.final_action == VerdictAction.ESCALATE

    @pytest.mark.asyncio
    async def test_single_judge_passthrough(self, evaluator, mock_client, simple_rubric, conversation):
        """Single judge should pass through directly."""
        mock_client.call_judge.return_value = _good_judge_response()

        ensemble = JudgeEnsemble(evaluator=evaluator)
        result = await ensemble.evaluate_turn(
            model_names=["primary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        assert result.agreement_rate == 1.0
        assert len(result.individual_verdicts) == 1
        assert result.final_action == VerdictAction.PASS

    @pytest.mark.asyncio
    async def test_judge_failure_excluded(self, evaluator, mock_client, simple_rubric, conversation):
        """Failed judges should be excluded from results."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            Exception("Model failed"),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator)
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        # Only one verdict should be included
        assert len(result.individual_verdicts) == 1
        assert result.final_action == VerdictAction.PASS

    @pytest.mark.asyncio
    async def test_all_judges_fail(self, evaluator, mock_client, simple_rubric, conversation):
        """If all judges fail, should escalate."""
        mock_client.call_judge.side_effect = [
            Exception("Model 1 failed"),
            Exception("Model 2 failed"),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator)
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="Hi there!",
            conversation=conversation,
        )

        assert result.final_action == VerdictAction.ESCALATE
        assert result.agreement_rate == 0.0
        assert len(result.individual_verdicts) == 0


class TestMeanScoreStrategy:
    @pytest.mark.asyncio
    async def test_mean_composite(self, evaluator, mock_client, simple_rubric, conversation):
        """Mean strategy should average composite scores."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),     # composite = 1.0
            _mediocre_judge_response(),  # composite ~0.5-0.625
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="mean_score")
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        # Final composite should be between the two
        v0 = result.individual_verdicts[0].composite_score
        v1 = result.individual_verdicts[1].composite_score
        expected = (v0 + v1) / 2
        assert abs(result.final_composite - expected) < 0.01


class TestMedianScoreStrategy:
    @pytest.mark.asyncio
    async def test_median_composite(self, evaluator, mock_client, simple_rubric, conversation):
        """Median strategy should take the median composite."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            _mediocre_judge_response(),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="median_score")
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        composites = sorted([v.composite_score for v in result.individual_verdicts])
        expected_median = (composites[0] + composites[1]) / 2
        assert abs(result.final_composite - expected_median) < 0.01


class TestConservativeStrategy:
    @pytest.mark.asyncio
    async def test_conservative_takes_worst(self, evaluator, mock_client, simple_rubric, conversation):
        """Conservative strategy should take the lowest composite."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            _bad_judge_response(),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="conservative", min_agreement=0.0)
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        worst = min(v.composite_score for v in result.individual_verdicts)
        assert result.final_composite == worst
        assert result.final_action == VerdictAction.BLOCK


class TestMajorityVoteStrategy:
    @pytest.mark.asyncio
    async def test_majority_action(self, evaluator, mock_client, simple_rubric, conversation):
        """Majority vote should pick the most common action."""
        mock_client.call_judge.side_effect = [
            _good_judge_response(),
            _good_judge_response(),
        ]

        ensemble = JudgeEnsemble(evaluator=evaluator, strategy="majority_vote")
        result = await ensemble.evaluate_turn(
            model_names=["primary", "secondary"],
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        assert result.final_action == VerdictAction.PASS


class TestConversationEnsemble:
    @pytest.mark.asyncio
    async def test_conversation_evaluation(self, evaluator, mock_client, conversation):
        """Ensemble should work with conversation-scope evaluation."""
        conv_rubric = Rubric(
            name="conv_test",
            description="Test",
            criteria=[
                RubricCriterion(name="consistency", description="Consistent?", scale=ScoreScale.LIKERT_5),
            ],
            scope=EvaluationScope.CONVERSATION,
            pass_threshold=0.6,
        )

        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "consistency", "score": 4, "reasoning": "Good", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Consistent",
        }

        ensemble = JudgeEnsemble(evaluator=evaluator)
        result = await ensemble.evaluate_conversation(
            model_names=["primary", "secondary"],
            rubric=conv_rubric,
            full_conversation=conversation,
        )

        assert result.final_action == VerdictAction.PASS
        assert len(result.individual_verdicts) == 2


class TestAgreementComputation:
    def test_perfect_agreement(self, evaluator):
        ensemble = JudgeEnsemble(evaluator=evaluator)
        verdicts = [
            JudgeVerdict(scores=[], composite_score=0.8, action=VerdictAction.PASS,
                         summary="ok", judge_model="a"),
            JudgeVerdict(scores=[], composite_score=0.7, action=VerdictAction.PASS,
                         summary="ok", judge_model="b"),
        ]
        assert ensemble._compute_agreement(verdicts) == 1.0

    def test_no_agreement(self, evaluator):
        ensemble = JudgeEnsemble(evaluator=evaluator)
        verdicts = [
            JudgeVerdict(scores=[], composite_score=0.8, action=VerdictAction.PASS,
                         summary="ok", judge_model="a"),
            JudgeVerdict(scores=[], composite_score=0.1, action=VerdictAction.BLOCK,
                         summary="bad", judge_model="b"),
        ]
        assert ensemble._compute_agreement(verdicts) == 0.0

    def test_partial_agreement_three_judges(self, evaluator):
        ensemble = JudgeEnsemble(evaluator=evaluator)
        verdicts = [
            JudgeVerdict(scores=[], composite_score=0.8, action=VerdictAction.PASS,
                         summary="ok", judge_model="a"),
            JudgeVerdict(scores=[], composite_score=0.7, action=VerdictAction.PASS,
                         summary="ok", judge_model="b"),
            JudgeVerdict(scores=[], composite_score=0.1, action=VerdictAction.BLOCK,
                         summary="bad", judge_model="c"),
        ]
        # 3 pairs: (a,b) agree, (a,c) disagree, (b,c) disagree -> 1/3
        agreement = ensemble._compute_agreement(verdicts)
        assert abs(agreement - 1 / 3) < 0.01

    def test_criterion_agreement(self, evaluator):
        ensemble = JudgeEnsemble(evaluator=evaluator)
        verdicts = [
            JudgeVerdict(
                scores=[JudgeScore(criterion="q", score=5, max_score=5, reasoning="ok")],
                composite_score=1.0, action=VerdictAction.PASS,
                summary="ok", judge_model="a",
            ),
            JudgeVerdict(
                scores=[JudgeScore(criterion="q", score=5, max_score=5, reasoning="ok")],
                composite_score=1.0, action=VerdictAction.PASS,
                summary="ok", judge_model="b",
            ),
        ]
        criterion_agr = ensemble._compute_criterion_agreement(verdicts)
        assert criterion_agr["q"] == 1.0


class TestEnsembleToDict:
    def test_to_dict(self, evaluator):
        ensemble_verdict = EnsembleVerdict(
            individual_verdicts=[],
            final_scores=[],
            final_composite=0.75,
            final_action=VerdictAction.PASS,
            agreement_rate=1.0,
            aggregation_strategy="mean_score",
        )
        d = ensemble_verdict.to_dict()
        assert d["final_composite"] == 0.75
        assert d["final_action"] == "pass"
        assert d["agreement_rate"] == 1.0
        assert d["aggregation_strategy"] == "mean_score"
