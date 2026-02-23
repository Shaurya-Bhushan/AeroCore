from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.aircraft.mission import _build_aircraft_constraint_certification
from src.spacecraft.mission import _build_spacecraft_constraint_certification


class ConstraintCertificationTests(unittest.TestCase):
    def test_aircraft_constraint_certification_shapes(self) -> None:
        aircraft_cfg = {
            "altitude_min_m": 400.0,
            "altitude_max_m": 1200.0,
            "reserve_energy_wh": 220.0,
            "battery_capacity_wh": 1500.0,
            "mission_horizon_s": 18000.0,
            "max_turn_rate_deg_s": 4.0,
            "geofence_margin_m": 450.0,
            "no_fly_zones": [[[47000.0, 32000.0], [76000.0, 32000.0], [76000.0, 69000.0], [47000.0, 69000.0]]],
        }
        result = SimpleNamespace(
            steps=[
                SimpleNamespace(
                    metadata={
                        "turn_rate_used_rad_s": 0.03,
                        "samples": [
                            {"x_m": 10000.0, "y_m": 10000.0, "alt_m": 500.0, "energy_wh": 1480.0},
                            {"x_m": 30000.0, "y_m": 25000.0, "alt_m": 520.0, "energy_wh": 1400.0},
                        ],
                    }
                ),
                SimpleNamespace(
                    metadata={
                        "turn_rate_used_rad_s": 0.04,
                        "samples": [
                            {"x_m": 45000.0, "y_m": 25000.0, "alt_m": 530.0, "energy_wh": 1280.0},
                            {"x_m": 90000.0, "y_m": 20000.0, "alt_m": 510.0, "energy_wh": 1210.0},
                        ],
                    }
                ),
            ],
            final_state={"time_s": 7400.0},
        )
        problem = SimpleNamespace(
            initial_state={"time_s": 0.0, "x_m": 10000.0, "y_m": 10000.0, "alt_m": 500.0, "energy_wh": 1500.0}
        )

        rows = _build_aircraft_constraint_certification(result, problem, aircraft_cfg)
        self.assertEqual(len(rows), 7)
        names = {str(r["constraint"]) for r in rows}
        self.assertIn("energy_bounds", names)
        self.assertIn("turn_rate", names)
        self.assertIn("stall_margin", names)
        self.assertIn("propulsion_power_margin", names)
        self.assertIn("geofence_margin", names)

    def test_spacecraft_constraint_certification_shapes(self) -> None:
        spacecraft_cfg = {
            "battery_min_wh": 65.0,
            "battery_capacity_wh": 320.0,
            "data_buffer_capacity_mb": 3200.0,
            "max_ops_per_orbit": 4,
            "min_downlink_elevation_deg": 10.0,
            "max_observation_off_nadir_deg": 55.0,
            "slew_rate_deg_s": 1.2,
            "battery_initial_wh": 220.0,
        }
        result = SimpleNamespace(
            steps=[
                SimpleNamespace(
                    to_task_id="OBS_T1_0",
                    metadata={
                        "transition_time_s": 180.0,
                        "slew_time_s": 140.0,
                        "visibility_off_nadir_deg": 32.0,
                    },
                ),
                SimpleNamespace(
                    to_task_id="DL_GS1_1",
                    metadata={
                        "transition_time_s": 220.0,
                        "slew_time_s": 150.0,
                        "visibility_elevation_deg": 22.0,
                    },
                ),
            ]
        )
        schedule_rows = [
            {"battery_wh": 210.0, "data_buffer_mb": 120.0, "ops_this_orbit": 1},
            {"battery_wh": 198.0, "data_buffer_mb": 340.0, "ops_this_orbit": 2},
        ]

        rows = _build_spacecraft_constraint_certification(result, schedule_rows, spacecraft_cfg)
        self.assertEqual(len(rows), 6)
        names = {str(r["constraint"]) for r in rows}
        self.assertIn("battery_bounds", names)
        self.assertIn("slew_feasibility", names)
        self.assertIn("downlink_elevation", names)


if __name__ == "__main__":
    unittest.main()
