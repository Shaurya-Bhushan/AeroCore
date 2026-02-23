from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config_validation import ConfigValidationError, validate_config


def _load_cfg() -> dict:
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())


def test_default_config_is_valid() -> None:
    cfg = _load_cfg()
    validate_config(cfg)


def test_spacecraft_mission_days_must_be_7() -> None:
    cfg = _load_cfg()
    cfg["spacecraft"]["mission_days"] = 5
    try:
        validate_config(cfg)
    except ConfigValidationError:
        return
    raise AssertionError("Expected ConfigValidationError for mission_days != 7")


def test_planner_strategy_must_be_supported() -> None:
    cfg = _load_cfg()
    cfg["planner_strategy"] = "random_walk"
    try:
        validate_config(cfg)
    except ConfigValidationError:
        return
    raise AssertionError("Expected ConfigValidationError for unsupported planner_strategy")
