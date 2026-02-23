from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.validation.independent_checks import verify_aircraft_outputs, verify_spacecraft_outputs


class IndependentChecksTests(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_aircraft_independent_check_detects_geofence_crossing(self) -> None:
        cfg = {
            "altitude_min_m": 0.0,
            "altitude_max_m": 1000.0,
            "reserve_energy_wh": 10.0,
            "battery_capacity_wh": 100.0,
            "max_turn_rate_deg_s": 45.0,
            "no_fly_zones": [[[4.0, -1.0], [6.0, -1.0], [6.0, 1.0], [4.0, 1.0]]],
        }

        with tempfile.TemporaryDirectory() as tmp:
            t_csv = Path(tmp) / "traj.csv"
            s_csv = Path(tmp) / "seg.csv"

            self._write_csv(
                t_csv,
                ["time_s", "x_m", "y_m", "alt_m", "heading_rad", "energy_wh"],
                [
                    {"time_s": 0, "x_m": 0, "y_m": 0, "alt_m": 100, "heading_rad": 0.0, "energy_wh": 80},
                    {"time_s": 1, "x_m": 10, "y_m": 0, "alt_m": 100, "heading_rad": 0.0, "energy_wh": 70},
                ],
            )
            self._write_csv(
                s_csv,
                ["start_time_s", "geofence_clear"],
                [{"start_time_s": 0, "geofence_clear": True}],
            )

            report = verify_aircraft_outputs(cfg, t_csv, s_csv)
            self.assertFalse(report["passed"])
            self.assertGreater(report["checks"]["geofence_violations"], 0)

    def test_spacecraft_independent_check_passes_valid_schedule(self) -> None:
        cfg = {
            "mission_days": 7,
            "battery_capacity_wh": 200.0,
            "battery_min_wh": 50.0,
            "data_buffer_capacity_mb": 1000.0,
            "max_ops_per_orbit": 4,
            "min_operation_gap_s": 30.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            sch_csv = Path(tmp) / "schedule.csv"
            vis_csv = Path(tmp) / "visibility.csv"

            self._write_csv(
                vis_csv,
                ["task_id", "action_type", "window_start_s", "window_end_s"],
                [
                    {"task_id": "OBS_A_0", "action_type": "observation", "window_start_s": 100, "window_end_s": 400},
                    {"task_id": "DL_GS_0", "action_type": "downlink", "window_start_s": 500, "window_end_s": 900},
                ],
            )
            self._write_csv(
                sch_csv,
                [
                    "to_task",
                    "action_type",
                    "window_start_s",
                    "window_end_s",
                    "task_start_s",
                    "task_end_s",
                    "battery_wh",
                    "data_buffer_mb",
                    "ops_this_orbit",
                ],
                [
                    {
                        "to_task": "OBS_A_0",
                        "action_type": "observation",
                        "window_start_s": 100,
                        "window_end_s": 400,
                        "task_start_s": 120,
                        "task_end_s": 240,
                        "battery_wh": 120,
                        "data_buffer_mb": 150,
                        "ops_this_orbit": 1,
                    },
                    {
                        "to_task": "DL_GS_0",
                        "action_type": "downlink",
                        "window_start_s": 500,
                        "window_end_s": 900,
                        "task_start_s": 560,
                        "task_end_s": 860,
                        "battery_wh": 110,
                        "data_buffer_mb": 20,
                        "ops_this_orbit": 2,
                    },
                    {
                        "to_task": "SC_END",
                        "action_type": "end",
                        "window_start_s": 604800,
                        "window_end_s": 604800,
                        "task_start_s": 604800,
                        "task_end_s": 604800,
                        "battery_wh": 100,
                        "data_buffer_mb": 0,
                        "ops_this_orbit": 0,
                    },
                ],
            )

            report = verify_spacecraft_outputs(cfg, sch_csv, vis_csv)
            self.assertTrue(report["passed"])
            self.assertEqual(report["hard_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
