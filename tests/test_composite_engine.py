
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from panoptes.policy.engines.composite.engine import CompositePolicyEngine
from panoptes.policy.protocols import (
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyViolation,
)

@pytest.fixture
def engine():
    return CompositePolicyEngine()

@pytest.fixture
def mock_registry():
    with patch("panoptes.policy.engines.composite.engine.PolicyEngineRegistry") as mock:
        yield mock

@pytest.mark.asyncio
async def test_initialization(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.configure_mock(name="mock1", engine_type="type1")
    # Explicitly make initialize return an awaitable
    mock_engine1.initialize = AsyncMock()

    mock_engine2 = MagicMock()
    mock_engine2.configure_mock(name="mock2", engine_type="type2")
    mock_engine2.initialize = AsyncMock()

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    config = {
        "engines": [
            {"type": "type1", "config": {"a": 1}},
            {"type": "type2", "config": {"b": 2}},
        ]
    }
    
    await engine.initialize(config)
    
    assert engine._initialized
    assert len(engine._engines) == 2
    mock_registry.create.assert_any_call("type1")
    mock_registry.create.assert_any_call("type2")
    mock_engine1.initialize.assert_awaited_with({"a": 1})

@pytest.mark.asyncio
async def test_evaluate_request_all_allow(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.ALLOW, violations=[]
    ))
    
    mock_engine2 = MagicMock()
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.ALLOW, violations=[]
    ))
    
    mock_registry.create.side_effect = [mock_engine1, mock_engine2]
    
    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })
    
    result = await engine.evaluate_request("sid", {}, {})
    
    assert result.decision == PolicyDecision.ALLOW

@pytest.mark.asyncio
async def test_evaluate_request_one_deny(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.ALLOW, violations=[]
    ))
    
    mock_engine2 = MagicMock()
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.DENY, violations=[PolicyViolation("v1", "critical", "msg")]
    ))
    
    mock_registry.create.side_effect = [mock_engine1, mock_engine2]
    
    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })
    
    result = await engine.evaluate_request("sid", {}, {})
    
    assert result.decision == PolicyDecision.DENY
    assert len(result.violations) == 1

@pytest.mark.asyncio
async def test_evaluate_response_parallel_execution(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_response = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.WARN, violations=[PolicyViolation("v1", "warning", "msg")]
    ))
    
    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_response = AsyncMock(return_value=PolicyEvaluationResult(
        decision=PolicyDecision.MODIFY, 
        violations=[],
        modified_request={"new": "req"}
    ))
    
    mock_registry.create.side_effect = [mock_engine1, mock_engine2]
    
    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}],
        "parallel": True
    })
    
    result = await engine.evaluate_response("sid", "data", {})
    
    assert result.decision == PolicyDecision.MODIFY
    assert len(result.violations) == 1
    assert result.modified_request == {"new": "req"}

@pytest.mark.asyncio
async def test_evaluate_request_engine_error(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(side_effect=Exception("Engine failure"))
    
    mock_registry.create.return_value = mock_engine1
    
    await engine.initialize({
        "engines": [{"type": "t1"}]
    })
    
    result = await engine.evaluate_request("sid", {}, {})
    
    assert result.decision == PolicyDecision.WARN
    assert result.violations[0].message == "Engine failure"
