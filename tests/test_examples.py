"""Smoke tests for example scripts: ensure they are importable and expose expected entrypoints."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load_example_module(relative_path: str):
    """Load an example script as a module by file path."""
    path = EXAMPLES_DIR / relative_path
    if not path.exists():
        pytest.skip(f"Example not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_judge_example_import_and_entrypoint():
    """Judge example module imports and exposes run_judge_conversation."""
    mod = _load_example_module("judge/llm_judge.py")
    assert hasattr(mod, "run_judge_conversation"), "llm_judge should define run_judge_conversation"
    assert callable(mod.run_judge_conversation)


def test_gemini_fsm_example_import_and_entrypoint():
    """Gemini FSM example module imports and exposes run_support_conversation."""
    mod = _load_example_module("gemini_fsm/gemini_agent.py")
    assert hasattr(mod, "run_support_conversation"), "gemini_agent should define run_support_conversation"
    assert callable(mod.run_support_conversation)


def test_nemo_example_import_and_entrypoint():
    """NeMo Guardrails example module imports and exposes run_nemo_conversation."""
    mod = _load_example_module("nemo_guardrails/nemo_agent.py")
    assert hasattr(mod, "run_nemo_conversation"), "nemo_agent should define run_nemo_conversation"
    assert callable(mod.run_nemo_conversation)
