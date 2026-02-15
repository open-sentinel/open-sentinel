"""
Tests for judge reliability modes.
"""

import pytest

from panoptes.policy.engines.judge.modes import (
    list_reliability_modes,
    get_reliability_mode,
    build_mode_config,
)


def test_list_reliability_modes_contains_expected():
    modes = list_reliability_modes()
    assert "safe" in modes
    assert "balanced" in modes
    assert "aggressive" in modes


def test_get_reliability_mode_invalid():
    with pytest.raises(ValueError, match="Unknown reliability mode"):
        get_reliability_mode("unknown")


def test_build_mode_config_merges_base_config():
    config = build_mode_config(
        "balanced",
        base_config={
            "warn_threshold": 0.9,
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        },
    )
    assert config["warn_threshold"] == 0.9
    assert config["models"][0]["model"] == "gpt-4o-mini"

