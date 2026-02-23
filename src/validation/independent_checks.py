from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from shapely.geometry import LineString, Polygon


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wrap_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _sort_float_rows(rows: Sequence[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)))


def verify_aircraft_outputs(
    aircraft_cfg: Dict[str, Any],
    trajectory_csv: Path,
    segments_csv: Path,
) -> Dict[str, Any]:
    trajectory = _sort_float_rows(_read_csv_rows(trajectory_csv), "time_s")
    segments = _sort_float_rows(_read_csv_rows(segments_csv), "start_time_s")

    violations: List[str] = []
    checks: Dict[str, int] = {
        "time_monotonic_violations": 0,
        "altitude_violations": 0,
        "energy_violations": 0,
        "turn_rate_violations": 0,
        "geofence_violations": 0,
    }

    altitude_min_m = float(aircraft_cfg.get("altitude_min_m", -1e9))
    altitude_max_m = float(aircraft_cfg.get("altitude_max_m", 1e9))
    reserve_energy_wh = float(aircraft_cfg.get("reserve_energy_wh", 0.0))
    capacity_energy_wh = float(aircraft_cfg.get("battery_capacity_wh", 1e12))
    max_turn_rate_rad_s = math.radians(float(aircraft_cfg.get("max_turn_rate_deg_s", 360.0)))
    allow_energy_increase = bool(aircraft_cfg.get("allow_energy_regen", False)) or float(
        aircraft_cfg.get("solar_charge_w", 0.0)
    ) > 0.0
    geofence_margin_m = float(aircraft_cfg.get("geofence_margin_m", 0.0))

    no_fly_polys = [Polygon(p) for p in aircraft_cfg.get("no_fly_zones", []) if isinstance(p, list)]

    if not trajectory:
        violations.append("aircraft: trajectory CSV is empty")

    prev_t = -1e30
    prev_heading = None
    prev_time = None
    prev_energy = None
    prev_xy = None
    for idx, row in enumerate(trajectory):
        t = float(row.get("time_s", 0.0))
        x = float(row.get("x_m", 0.0))
        y = float(row.get("y_m", 0.0))
        alt = float(row.get("alt_m", 0.0))
        energy = float(row.get("energy_wh", 0.0))
        heading = float(row.get("heading_rad", 0.0))

        if t + 1e-9 < prev_t:
            checks["time_monotonic_violations"] += 1
            violations.append(f"aircraft: non-monotonic time at trajectory row {idx}")
        prev_t = max(prev_t, t)

        if alt < altitude_min_m - 1e-6 or alt > altitude_max_m + 1e-6:
            checks["altitude_violations"] += 1
            violations.append(f"aircraft: altitude out of bounds at row {idx}: {alt}")

        if energy < reserve_energy_wh - 1e-6 or energy > capacity_energy_wh + 1e-6:
            checks["energy_violations"] += 1
            violations.append(f"aircraft: energy out of bounds at row {idx}: {energy}")

        if (not allow_energy_increase) and prev_energy is not None and energy > prev_energy + 1.0:
            checks["energy_violations"] += 1
            violations.append(f"aircraft: energy increased significantly between rows {idx-1} and {idx}")
        prev_energy = energy

        if prev_heading is not None and prev_time is not None:
            dt = max(1e-9, t - prev_time)
            turn_rate = abs(_wrap_angle_rad(heading - prev_heading)) / dt
            if turn_rate > max_turn_rate_rad_s + 1e-4:
                checks["turn_rate_violations"] += 1
                violations.append(
                    f"aircraft: turn-rate violation rows {idx-1}->{idx} (used={turn_rate:.4f}, limit={max_turn_rate_rad_s:.4f})"
                )
        prev_heading = heading
        prev_time = t

        if prev_xy is not None:
            seg = LineString([prev_xy, (x, y)])
            for poly_idx, poly in enumerate(no_fly_polys):
                if seg.crosses(poly) or seg.within(poly) or seg.distance(poly) < geofence_margin_m - 1e-6:
                    checks["geofence_violations"] += 1
                    violations.append(
                        f"aircraft: no-fly crossing or margin violation between trajectory rows {idx-1}->{idx} (polygon {poly_idx})"
                    )
                    break
        prev_xy = (x, y)

    # Secondary consistency check from segment table if available.
    for idx, row in enumerate(segments):
        clear = str(row.get("geofence_clear", "True")).lower() in {"true", "1", "yes"}
        if not clear:
            checks["geofence_violations"] += 1
            violations.append(f"aircraft: segment table reports geofence_clear=False at segment row {idx}")

    violation_count = sum(checks.values())
    return {
        "domain": "aircraft",
        "passed": violation_count == 0,
        "hard_violation_count": violation_count,
        "checks": checks,
        "violations": violations[:200],
        "inputs": {
            "trajectory_csv": str(trajectory_csv),
            "segments_csv": str(segments_csv),
            "trajectory_rows": len(trajectory),
            "segment_rows": len(segments),
        },
    }


def verify_spacecraft_outputs(
    spacecraft_cfg: Dict[str, Any],
    schedule_csv: Path,
    visibility_csv: Path,
) -> Dict[str, Any]:
    schedule = _sort_float_rows(_read_csv_rows(schedule_csv), "task_start_s")
    windows = _read_csv_rows(visibility_csv)
    valid_task_ids = {str(r.get("task_id", "")) for r in windows if r.get("task_id")}

    violations: List[str] = []
    checks: Dict[str, int] = {
        "time_monotonic_violations": 0,
        "window_violations": 0,
        "battery_violations": 0,
        "buffer_violations": 0,
        "ops_per_orbit_violations": 0,
        "visibility_reference_violations": 0,
        "terminal_violations": 0,
        "gap_violations": 0,
    }

    capacity_wh = float(spacecraft_cfg.get("battery_capacity_wh", 1e12))
    min_wh = float(spacecraft_cfg.get("battery_min_wh", -1e12))
    buffer_cap = float(spacecraft_cfg.get("data_buffer_capacity_mb", 1e12))
    max_ops = int(spacecraft_cfg.get("max_ops_per_orbit", 10**9))
    horizon_s = float(int(spacecraft_cfg.get("mission_days", 7)) * 24 * 3600)
    min_gap_s = float(spacecraft_cfg.get("min_operation_gap_s", 0.0))

    if not schedule:
        violations.append("spacecraft: schedule CSV is empty")

    prev_end = -1e30
    last_op_end = -1e30
    end_rows = 0
    for idx, row in enumerate(schedule):
        task_id = str(row.get("to_task", ""))
        action_type = str(row.get("action_type", "")).lower()
        task_start = float(row.get("task_start_s", 0.0))
        task_end = float(row.get("task_end_s", 0.0))
        window_start = float(row.get("window_start_s", 0.0))
        window_end = float(row.get("window_end_s", 0.0))
        battery = float(row.get("battery_wh", 0.0))
        buffer_mb = float(row.get("data_buffer_mb", 0.0))
        ops = int(float(row.get("ops_this_orbit", 0)))

        if task_start + 1e-9 < prev_end:
            checks["time_monotonic_violations"] += 1
            violations.append(f"spacecraft: overlapping/non-monotonic schedule at row {idx}")
        prev_end = max(prev_end, task_end)

        if task_start < window_start - 1e-6 or task_end > window_end + 1e-6:
            checks["window_violations"] += 1
            violations.append(f"spacecraft: task outside window at row {idx}")

        if battery < min_wh - 1e-6 or battery > capacity_wh + 1e-6:
            checks["battery_violations"] += 1
            violations.append(f"spacecraft: battery out of bounds at row {idx}: {battery}")

        if buffer_mb < -1e-6 or buffer_mb > buffer_cap + 1e-6:
            checks["buffer_violations"] += 1
            violations.append(f"spacecraft: data buffer out of bounds at row {idx}: {buffer_mb}")

        if ops > max_ops:
            checks["ops_per_orbit_violations"] += 1
            violations.append(f"spacecraft: ops_this_orbit exceeds max at row {idx}: {ops}>{max_ops}")

        if action_type in {"observation", "downlink"} and task_id and task_id not in valid_task_ids:
            checks["visibility_reference_violations"] += 1
            violations.append(f"spacecraft: scheduled task not present in visibility windows at row {idx}: {task_id}")

        if action_type in {"observation", "downlink"}:
            if task_start - last_op_end < min_gap_s - 1e-6 and last_op_end > -1e20:
                checks["gap_violations"] += 1
                violations.append(f"spacecraft: operation gap below min at row {idx}")
            last_op_end = task_end

        if action_type == "end":
            end_rows += 1
            if abs(task_end - horizon_s) > 1.0:
                checks["terminal_violations"] += 1
                violations.append(f"spacecraft: end task does not close horizon (row {idx}, end={task_end}, expected={horizon_s})")

    if end_rows == 0:
        checks["terminal_violations"] += 1
        violations.append("spacecraft: missing terminal end action")

    violation_count = sum(checks.values())
    return {
        "domain": "spacecraft",
        "passed": violation_count == 0,
        "hard_violation_count": violation_count,
        "checks": checks,
        "violations": violations[:200],
        "inputs": {
            "schedule_csv": str(schedule_csv),
            "visibility_csv": str(visibility_csv),
            "schedule_rows": len(schedule),
            "visibility_rows": len(windows),
        },
    }


def write_check_report(path: Path, report: Dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2))
