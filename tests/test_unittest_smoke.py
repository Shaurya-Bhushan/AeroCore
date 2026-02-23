from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from run_all import _spacecraft_horizon_covered
from src.aircraft.mission import build_aircraft_problem
from src.core import UnifiedHybridOCPEngine, UnifiedPlanner, validate_config
from src.spacecraft.mission import build_spacecraft_problem


def _load_default_cfg() -> dict:
    import yaml

    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())


class SmokeTests(unittest.TestCase):
    def test_default_config_valid(self) -> None:
        cfg = _load_default_cfg()
        validate_config(cfg)

    def test_aircraft_and_spacecraft_build(self) -> None:
        cfg = _load_default_cfg()
        aircraft_problem = build_aircraft_problem(
            cfg["aircraft"],
            planner_overrides={"beam_width": 8, "max_expansions": 600, "candidate_limit": 6, "max_depth": 16},
        )
        spacecraft_problem = build_spacecraft_problem(
            cfg["spacecraft"],
            planner_overrides={"beam_width": 8, "max_expansions": 600, "candidate_limit": 6, "max_depth": 48},
        )

        planner = UnifiedPlanner()
        a_result = planner.solve(aircraft_problem, strategy="beam")
        s_result = planner.solve(spacecraft_problem, strategy="beam")

        self.assertGreaterEqual(len(a_result.sequence), 1)
        self.assertGreaterEqual(float(a_result.final_state.get("time_s", 0.0)), 0.0)
        self.assertGreaterEqual(len(s_result.sequence), 1)
        self.assertGreaterEqual(float(s_result.final_state.get("time_s", 0.0)), 0.0)

    def test_hybrid_engine_returns_valid_plans(self) -> None:
        cfg = _load_default_cfg()
        planner = UnifiedPlanner()
        hybrid = UnifiedHybridOCPEngine(
            {
                "control_iterations": 1,
                "max_sequence_candidates": 4,
                "max_mutations_per_seed": 6,
                "max_control_nodes_aircraft": 8,
                "max_control_nodes_spacecraft": 10,
            }
        )

        aircraft_problem = build_aircraft_problem(
            cfg["aircraft"],
            planner_overrides={"beam_width": 10, "max_expansions": 900, "candidate_limit": 6, "max_depth": 18},
        )
        spacecraft_problem = build_spacecraft_problem(
            cfg["spacecraft"],
            planner_overrides={"beam_width": 10, "max_expansions": 1200, "candidate_limit": 8, "max_depth": 60},
        )

        a_result = hybrid.solve(aircraft_problem, planner=planner)
        s_result = hybrid.solve(spacecraft_problem, planner=planner)

        self.assertGreaterEqual(len(a_result.sequence), 1)
        self.assertGreaterEqual(len(s_result.sequence), 1)
        self.assertEqual(a_result.sequence[0], aircraft_problem.start_task_id)
        self.assertEqual(s_result.sequence[0], spacecraft_problem.start_task_id)

    def test_spacecraft_horizon_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schedule_csv = Path(tmp) / "schedule.csv"
            with schedule_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["action_type", "task_end_s"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"action_type": "observation", "task_end_s": 100.0},
                        {"action_type": "end", "task_end_s": 604800.0},
                    ]
                )
            self.assertTrue(_spacecraft_horizon_covered(schedule_csv, 604800.0))


if __name__ == "__main__":
    unittest.main()
