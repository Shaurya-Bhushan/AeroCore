from __future__ import annotations

from pathlib import Path

import yaml

from src.aircraft.mission import build_aircraft_problem
from src.core import UnifiedPlanner
from src.spacecraft.mission import build_spacecraft_problem


def _load_cfg() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    return yaml.safe_load(cfg_path.read_text())


def test_aircraft_problem_builds_and_solves_partial() -> None:
    cfg = _load_cfg()["aircraft"]
    problem = build_aircraft_problem(
        cfg,
        planner_overrides={"beam_width": 24, "max_expansions": 6000, "candidate_limit": 8},
    )
    result = UnifiedPlanner().solve(problem, strategy="beam")
    assert len(result.sequence) >= 1
    assert result.final_state.get("time_s", 0.0) >= 0.0


def test_spacecraft_problem_builds_and_solves_partial() -> None:
    cfg = _load_cfg()["spacecraft"]
    problem = build_spacecraft_problem(
        cfg,
        planner_overrides={"beam_width": 24, "max_expansions": 10000, "candidate_limit": 10, "max_depth": 120},
    )
    result = UnifiedPlanner().solve(problem, strategy="beam")
    assert len(result.sequence) >= 1
    assert result.final_state.get("time_s", 0.0) >= 0.0
