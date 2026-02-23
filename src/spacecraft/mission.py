from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import numpy as np

from src.core import (
    CallableConstraint,
    MaxOpsPerOrbitConstraint,
    MissionProblem,
    MissionTask,
    PlannerSettings,
    RangeConstraint,
    TaskWindowConstraint,
    Transition,
    UnifiedHybridOCPEngine,
    UnifiedPlanner,
    WeightedObjective,
    write_certification_report,
    write_csv,
)
from src.spacecraft.model import (
    SpacecraftResourceConfig,
    build_initial_state,
    build_spacecraft_transitions,
    clear_spacecraft_model_caches,
    generate_opportunity_tasks,
    orbit_from_config,
    simulate_spacecraft_step,
)


def _accept_hybrid_result(hybrid_result, baseline_result) -> bool:
    if not UnifiedHybridOCPEngine.is_better(hybrid_result, baseline_result):
        return False
    base_delivered = float(baseline_result.final_state.get("delivered_science", 0.0))
    hyb_delivered = float(hybrid_result.final_state.get("delivered_science", 0.0))
    # Guardrail: hybrid must not reduce delivered science.
    if hyb_delivered + 1e-6 < base_delivered:
        return False
    return True


def _best_plan_result(candidates: Sequence[tuple[str, Any]]) -> tuple[str, Any]:
    best_label, best_result = candidates[0]
    for label, candidate in candidates[1:]:
        if UnifiedHybridOCPEngine.is_better(candidate, best_result):
            best_label = label
            best_result = candidate
    return best_label, best_result


def _spacecraft_candidate_fn(
    problem: MissionProblem,
    state: Dict[str, Any],
    current_task_id: str,
    visited: Sequence[str],
) -> List[str]:
    visited_set = set(visited)
    now = float(state.get("time_s", 0.0))
    buffer_mb = float(state.get("data_buffer_mb", 0.0))
    battery_wh = float(state.get("battery_wh", 0.0))
    horizon = float(problem.tasks[problem.end_task_id].window_end_s) if problem.end_task_id else now

    scored: List[tuple[int, float, float, str]] = []
    for task_id, task in problem.tasks.items():
        if task_id in visited_set or task_id == problem.start_task_id:
            continue
        if task.task_type == "end":
            continue
        if task.window_end_s < now:
            continue
        if task.window_start_s > now + problem.settings.lookahead_s:
            continue

        transition = problem.transitions.get((current_task_id, task_id))
        if transition is None or not transition.feasible:
            continue

        urgency = task.window_end_s - max(now, task.window_start_s)
        if task.task_type == "downlink" and buffer_mb > 0.25 * float(state.get("data_buffer_capacity_mb", 1.0)):
            priority = 0
        elif task.task_type == "downlink" and buffer_mb > 10.0:
            priority = 1
        elif task.task_type == "observation":
            priority = 2
        elif task.task_type == "downlink":
            priority = 3
        else:
            priority = 4

        battery_bias = 0.0 if battery_wh > float(state.get("battery_min_wh", 0.0)) + 20.0 else 1.0
        repeat_penalty = 0
        if task.task_type == "observation":
            tid = str(task.metadata.get("target_id", task.task_id))
            repeat_penalty = int(state.get("target_visits", {}).get(tid, 0))
        scored.append((priority, repeat_penalty, battery_bias, urgency, -task.value, task.window_start_s, task_id))

    scored.sort()
    candidate_ids = [item[-1] for item in scored[: problem.settings.candidate_limit]]

    if problem.end_task_id and problem.end_task_id not in visited_set:
        if (not candidate_ids) or now > 0.985 * horizon:
            candidate_ids.append(problem.end_task_id)
        else:
            candidate_ids = candidate_ids + [problem.end_task_id]

    return candidate_ids


def build_spacecraft_problem(
    spacecraft_cfg: Dict[str, Any],
    planner_overrides: Dict[str, Any] | None = None,
) -> MissionProblem:
    if bool(spacecraft_cfg.get("clear_model_caches", False)):
        clear_spacecraft_model_caches()
    tasks = generate_opportunity_tasks(spacecraft_cfg)

    mission_days = int(spacecraft_cfg.get("mission_days", 7))
    horizon_s = float(mission_days * 24 * 3600)

    orbit = orbit_from_config(spacecraft_cfg, horizon_s=horizon_s)

    resources = SpacecraftResourceConfig(
        battery_capacity_wh=float(spacecraft_cfg.get("battery_capacity_wh", 240.0)),
        battery_min_wh=float(spacecraft_cfg.get("battery_min_wh", 40.0)),
        battery_initial_wh=float(spacecraft_cfg.get("battery_initial_wh", 190.0)),
        data_buffer_capacity_mb=float(spacecraft_cfg.get("data_buffer_capacity_mb", 2000.0)),
        slew_rate_deg_s=float(spacecraft_cfg.get("slew_rate_deg_s", 1.2)),
        obs_power_w=float(spacecraft_cfg.get("obs_power_w", 32.0)),
        downlink_power_w=float(spacecraft_cfg.get("downlink_power_w", 28.0)),
        housekeeping_power_w=float(spacecraft_cfg.get("housekeeping_power_w", 10.0)),
        solar_charge_w=float(spacecraft_cfg.get("solar_charge_w", 38.0)),
        solar_capture_efficiency=float(spacecraft_cfg.get("solar_capture_efficiency", 0.62)),
        downlink_rate_mb_s=float(spacecraft_cfg.get("downlink_rate_mb_s", 2.0)),
        max_ops_per_orbit=int(spacecraft_cfg.get("max_ops_per_orbit", 4)),
        min_downlink_elevation_deg=float(spacecraft_cfg.get("min_downlink_elevation_deg", 10.0)),
        max_observation_off_nadir_deg=float(spacecraft_cfg.get("max_observation_off_nadir_deg", 30.0)),
        min_operation_gap_s=float(spacecraft_cfg.get("min_operation_gap_s", 60.0)),
    )

    transitions = build_spacecraft_transitions(tasks, resources, orbit=orbit)

    planner_cfg = spacecraft_cfg.get("planner", {})
    settings = PlannerSettings(
        beam_width=int(planner_cfg.get("beam_width", 80)),
        max_expansions=int(planner_cfg.get("max_expansions", 90000)),
        max_depth=int(planner_cfg.get("max_depth", 260)),
        candidate_limit=int(planner_cfg.get("candidate_limit", 24)),
        lookahead_s=float(planner_cfg.get("lookahead_s", 6 * 3600)),
        soft_penalty_weight=float(planner_cfg.get("soft_penalty_weight", 2.0)),
        allow_revisit=bool(planner_cfg.get("allow_revisit", False)),
        multistart_runs=int(planner_cfg.get("multistart_runs", 1)),
        prune_time_bucket_s=float(planner_cfg.get("prune_time_bucket_s", 300.0)),
        prune_energy_bucket_wh=float(planner_cfg.get("prune_energy_bucket_wh", 25.0)),
        prune_battery_bucket_wh=float(planner_cfg.get("prune_battery_bucket_wh", 10.0)),
        prune_data_buffer_bucket_mb=float(planner_cfg.get("prune_data_buffer_bucket_mb", 100.0)),
        prune_science_bucket=float(planner_cfg.get("prune_science_bucket", 10.0)),
    )
    if planner_overrides:
        valid_keys = set(settings.__dict__.keys())
        unknown = sorted(str(k) for k in planner_overrides.keys() if k not in valid_keys)
        if unknown:
            raise ValueError(f"Unknown spacecraft planner_overrides keys: {', '.join(unknown)}")
        for key, value in planner_overrides.items():
            setattr(settings, key, value)

    def slew_constraint(
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> tuple[bool, str, float]:
        slew_required = float(step_meta.get("slew_time_s", transition.travel_time_s))
        available = float(step_meta.get("transition_time_s", transition.travel_time_s))
        ok = slew_required <= available + 1e-6
        return ok, "insufficient transition for slew" if not ok else "", max(0.0, slew_required - available)

    def simulate(
        state: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
    ):
        return simulate_spacecraft_step(
            state=state,
            from_task=from_task,
            to_task=to_task,
            transition=transition,
            orbit=orbit,
            resources=resources,
        )

    constraints = [
        TaskWindowConstraint(name="window", hard=True),
        RangeConstraint(
            key="battery_wh",
            min_value=resources.battery_min_wh,
            max_value=resources.battery_capacity_wh,
            name="battery_bounds",
            hard=True,
        ),
        RangeConstraint(
            key="data_buffer_mb",
            min_value=0.0,
            max_value=resources.data_buffer_capacity_mb,
            name="data_buffer_bounds",
            hard=True,
        ),
        MaxOpsPerOrbitConstraint(max_ops=resources.max_ops_per_orbit, name="ops_per_orbit", hard=True),
        CallableConstraint(fn=slew_constraint, name="slew", hard=True),
    ]

    objective_cfg = spacecraft_cfg.get("objective", {})
    objective = WeightedObjective(
        value_weight=float(objective_cfg.get("value_weight", 0.0)),
        delivered_weight=float(objective_cfg.get("delivered_weight", 18.0)),
        time_weight=float(objective_cfg.get("time_weight", 0.0005)),
        energy_weight=float(objective_cfg.get("energy_weight", 0.03)),
        required_completion_bonus=float(objective_cfg.get("required_completion_bonus", 120.0)),
        unfinished_required_penalty=float(objective_cfg.get("unfinished_required_penalty", 500.0)),
        unsolved_penalty=float(objective_cfg.get("unsolved_penalty", 300.0)),
    )

    return MissionProblem(
        name="spacecraft_mission",
        domain="spacecraft",
        tasks=tasks,
        transitions=transitions,
        start_task_id="SC_START",
        end_task_id="SC_END",
        required_task_ids=[],
        initial_state={
            **build_initial_state(spacecraft_cfg),
            "data_buffer_capacity_mb": resources.data_buffer_capacity_mb,
            "battery_min_wh": resources.battery_min_wh,
        },
        simulate_step=simulate,
        constraints=constraints,
        objective=objective,
        settings=settings,
        candidate_fn=_spacecraft_candidate_fn,
    )


def _schedule_rows(result, tasks: Dict[str, MissionTask]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in result.steps:
        task = tasks[step.to_task_id]
        rows.append(
            {
                "from_task": step.from_task_id,
                "to_task": step.to_task_id,
                "action_type": task.task_type,
                "window_start_s": task.window_start_s,
                "window_end_s": task.window_end_s,
                "task_start_s": step.start_time_s,
                "task_end_s": step.end_time_s,
                "duration_s": step.task_duration_s,
                "value": task.value,
                "target_id": task.metadata.get("target_id", ""),
                "station_id": task.metadata.get("station_id", ""),
                "lat_deg": task.metadata.get("lat_deg", None),
                "lon_deg": task.metadata.get("lon_deg", None),
                "min_elevation_deg": task.metadata.get("min_elevation_deg", None),
                "max_elevation_deg": task.metadata.get("max_elevation_deg", None),
                "min_off_nadir_deg": task.metadata.get("min_off_nadir_deg", None),
                "max_off_nadir_deg": task.metadata.get("max_off_nadir_deg", None),
                "slew_angle_deg": step.metadata.get("slew_angle_deg", 0.0),
                "downlinked_mb": step.metadata.get("downlinked_mb", 0.0),
                "battery_wh": step.state_after.get("battery_wh", None),
                "data_buffer_mb": step.state_after.get("data_buffer_mb", None),
                "delivered_science": step.state_after.get("delivered_science", None),
                "ops_this_orbit": step.state_after.get("ops_this_orbit", None),
            }
        )
    return rows


def _opportunity_rows(tasks: Dict[str, MissionTask]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_id, task in tasks.items():
        if task.task_type in {"start", "end"}:
            continue
        rows.append(
            {
                "task_id": task_id,
                "action_type": task.task_type,
                "window_start_s": task.window_start_s,
                "window_end_s": task.window_end_s,
                "duration_s": task.duration_s,
                "value": task.value,
                "target_id": task.metadata.get("target_id", ""),
                "station_id": task.metadata.get("station_id", ""),
                "lat_deg": task.metadata.get("lat_deg", None),
                "lon_deg": task.metadata.get("lon_deg", None),
                "min_elevation_deg": task.metadata.get("min_elevation_deg", None),
                "max_elevation_deg": task.metadata.get("max_elevation_deg", None),
                "min_off_nadir_deg": task.metadata.get("min_off_nadir_deg", None),
                "max_off_nadir_deg": task.metadata.get("max_off_nadir_deg", None),
            }
        )
    rows.sort(key=lambda r: (float(r["window_start_s"]), str(r["action_type"])))
    return rows


def _pseudo_multiplier(max_violation: float, min_slack: float, active_tol: float) -> float:
    if max_violation > 0.0:
        return float(1000.0 + 1000.0 * max_violation)
    if min_slack <= active_tol:
        return float(1.0 / max(1e-3, min_slack + 1e-3))
    return 0.0


def _build_spacecraft_constraint_certification(
    result: Any,
    schedule_rows: Sequence[Dict[str, Any]],
    spacecraft_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    battery_min = float(spacecraft_cfg.get("battery_min_wh", 0.0))
    battery_cap = float(spacecraft_cfg.get("battery_capacity_wh", battery_min + 1.0))
    data_cap = float(spacecraft_cfg.get("data_buffer_capacity_mb", 0.0))
    max_ops = int(spacecraft_cfg.get("max_ops_per_orbit", 0))
    min_elev = float(spacecraft_cfg.get("min_downlink_elevation_deg", 10.0))
    max_off_nadir = float(spacecraft_cfg.get("max_observation_off_nadir_deg", 90.0))
    slew_rate = float(spacecraft_cfg.get("slew_rate_deg_s", 1.0))

    if schedule_rows:
        battery_vals = np.array([float(r.get("battery_wh", np.nan)) for r in schedule_rows], dtype=float)
        data_vals = np.array([float(r.get("data_buffer_mb", np.nan)) for r in schedule_rows], dtype=float)
        ops_vals = np.array([float(r.get("ops_this_orbit", np.nan)) for r in schedule_rows], dtype=float)
    else:
        battery_vals = np.array([float(spacecraft_cfg.get("battery_initial_wh", 0.0))], dtype=float)
        data_vals = np.array([0.0], dtype=float)
        ops_vals = np.array([0.0], dtype=float)

    battery_vals = battery_vals[np.isfinite(battery_vals)]
    data_vals = data_vals[np.isfinite(data_vals)]
    ops_vals = ops_vals[np.isfinite(ops_vals)]

    min_batt = float(np.min(battery_vals)) if battery_vals.size else float(spacecraft_cfg.get("battery_initial_wh", 0.0))
    max_buffer = float(np.max(data_vals)) if data_vals.size else 0.0
    max_ops_seen = float(np.max(ops_vals)) if ops_vals.size else 0.0

    battery_violation = max(0.0, battery_min - min_batt)
    battery_slack = min_batt - battery_min
    battery_active = float(np.mean(battery_vals <= (battery_min + max(4.0, 0.03 * battery_cap)))) if battery_vals.size else 0.0

    buffer_violation = max(0.0, max_buffer - data_cap)
    buffer_slack = data_cap - max_buffer
    buffer_active = float(np.mean(data_vals >= (data_cap - max(15.0, 0.05 * max(1.0, data_cap))))) if data_vals.size else 0.0

    ops_violation = max(0.0, max_ops_seen - float(max_ops))
    ops_slack = float(max_ops) - max_ops_seen
    ops_active = float(np.mean(ops_vals >= max(0.0, float(max_ops) - 0.5))) if ops_vals.size else 0.0

    slew_slacks: List[float] = []
    elev_slacks: List[float] = []
    off_nadir_slacks: List[float] = []
    for step in result.steps:
        trans_time = float(step.metadata.get("transition_time_s", 0.0))
        slew_time = float(step.metadata.get("slew_time_s", 0.0))
        slew_slacks.append(trans_time - slew_time)

        action = str(step.metadata.get("action_type", "")).lower()
        # action_type not always in metadata; derive from to_task_id prefix fallback.
        if not action:
            tid = str(step.to_task_id)
            if tid.startswith("DL_"):
                action = "downlink"
            elif tid.startswith("OBS_"):
                action = "observation"

        if action == "downlink":
            elev = float(step.metadata.get("visibility_elevation_deg", np.nan))
            if np.isfinite(elev):
                elev_slacks.append(elev - min_elev)
        if action == "observation":
            off_nadir = float(step.metadata.get("visibility_off_nadir_deg", np.nan))
            if np.isfinite(off_nadir):
                off_nadir_slacks.append(max_off_nadir - off_nadir)

    slew_slacks_arr = np.array(slew_slacks, dtype=float) if slew_slacks else np.array([0.0], dtype=float)
    slew_min_slack = float(np.min(slew_slacks_arr))
    slew_violation = max(0.0, -slew_min_slack)
    slew_active = float(np.mean(slew_slacks_arr <= max(2.0, 0.05 * max(1e-6, 1.0 / max(1e-6, slew_rate))))) if slew_slacks_arr.size else 0.0

    if elev_slacks:
        elev_arr = np.array(elev_slacks, dtype=float)
        elev_min_slack = float(np.min(elev_arr))
        elev_violation = max(0.0, -elev_min_slack)
        elev_active = float(np.mean(elev_arr <= 2.0))
        elev_observed = min_elev + elev_min_slack
    else:
        elev_min_slack = float("inf")
        elev_violation = 0.0
        elev_active = 0.0
        elev_observed = float("nan")

    if off_nadir_slacks:
        off_arr = np.array(off_nadir_slacks, dtype=float)
        off_min_slack = float(np.min(off_arr))
        off_violation = max(0.0, -off_min_slack)
        off_active = float(np.mean(off_arr <= 3.0))
        off_observed = max_off_nadir - off_min_slack
    else:
        off_min_slack = float("inf")
        off_violation = 0.0
        off_active = 0.0
        off_observed = float("nan")

    return [
        {
            "constraint": "battery_bounds",
            "max_violation": battery_violation,
            "slack_margin": battery_slack,
            "active_fraction": battery_active,
            "pseudo_multiplier": _pseudo_multiplier(battery_violation, battery_slack, 4.0),
            "limit": battery_min,
            "observed": min_batt,
            "units": "Wh",
        },
        {
            "constraint": "data_buffer_bounds",
            "max_violation": buffer_violation,
            "slack_margin": buffer_slack,
            "active_fraction": buffer_active,
            "pseudo_multiplier": _pseudo_multiplier(buffer_violation, buffer_slack, max(15.0, 0.05 * max(1.0, data_cap))),
            "limit": data_cap,
            "observed": max_buffer,
            "units": "MB",
        },
        {
            "constraint": "ops_per_orbit",
            "max_violation": ops_violation,
            "slack_margin": ops_slack,
            "active_fraction": ops_active,
            "pseudo_multiplier": _pseudo_multiplier(ops_violation, ops_slack, 0.5),
            "limit": max_ops,
            "observed": max_ops_seen,
            "units": "count",
        },
        {
            "constraint": "slew_feasibility",
            "max_violation": slew_violation,
            "slack_margin": slew_min_slack,
            "active_fraction": slew_active,
            "pseudo_multiplier": _pseudo_multiplier(slew_violation, slew_min_slack, 2.0),
            "limit": "transition_time >= slew_time",
            "observed": slew_min_slack,
            "units": "s",
        },
        {
            "constraint": "downlink_elevation",
            "max_violation": elev_violation,
            "slack_margin": elev_min_slack,
            "active_fraction": elev_active,
            "pseudo_multiplier": _pseudo_multiplier(elev_violation, elev_min_slack, 2.0),
            "limit": min_elev,
            "observed": elev_observed,
            "units": "deg",
        },
        {
            "constraint": "observation_off_nadir",
            "max_violation": off_violation,
            "slack_margin": off_min_slack,
            "active_fraction": off_active,
            "pseudo_multiplier": _pseudo_multiplier(off_violation, off_min_slack, 3.0),
            "limit": max_off_nadir,
            "observed": off_observed,
            "units": "deg",
        },
    ]


def plot_spacecraft_gantt(schedule_rows: Sequence[Dict[str, Any]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    color_map = {"observation": "tab:green", "downlink": "tab:blue", "end": "tab:gray"}
    y_map = {"observation": 20, "downlink": 10, "end": 0}

    for row in schedule_rows:
        action = row["action_type"]
        start_h = float(row["task_start_s"]) / 3600.0
        width_h = max(0.01, (float(row["task_end_s"]) - float(row["task_start_s"])) / 3600.0)
        y = y_map.get(action, 0)
        ax.broken_barh([(start_h, width_h)], (y, 8), facecolors=color_map.get(action, "tab:orange"), alpha=0.8)

    ax.set_yticks([4, 14, 24])
    ax.set_yticklabels(["End", "Downlink", "Observation"])
    ax.set_xlabel("Mission Time [hours]")
    ax.set_title("Spacecraft 7-Day Schedule")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_spacecraft_resources(schedule_rows: Sequence[Dict[str, Any]], initial_battery_wh: float, save_path: Path) -> None:
    if not schedule_rows:
        return

    times = [0.0] + [float(r["task_end_s"]) for r in schedule_rows]
    battery = [initial_battery_wh] + [float(r["battery_wh"]) for r in schedule_rows]
    data_buffer = [0.0] + [float(r["data_buffer_mb"]) for r in schedule_rows]
    delivered = [0.0] + [float(r["delivered_science"]) for r in schedule_rows]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot([t / 3600.0 for t in times], battery, color="tab:purple")
    ax[0].set_ylabel("Battery [Wh]")
    ax[0].grid(alpha=0.3)

    ax[1].plot([t / 3600.0 for t in times], data_buffer, color="tab:orange")
    ax[1].set_ylabel("Data Buffer [MB]")
    ax[1].grid(alpha=0.3)

    ax[2].plot([t / 3600.0 for t in times], delivered, color="tab:green")
    ax[2].set_ylabel("Delivered Science")
    ax[2].set_xlabel("Mission Time [hours]")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Spacecraft Resource Profiles")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _mission_epoch_utc(spacecraft_cfg: Dict[str, Any]) -> datetime:
    year = int(spacecraft_cfg.get("epoch_year", 2026))
    day_of_year = float(spacecraft_cfg.get("epoch_day_of_year", 1.0))
    utc_hour = float(spacecraft_cfg.get("epoch_utc_hour", 0.0))
    day_index = max(1, int(day_of_year)) - 1
    day_frac = max(0.0, day_of_year - int(day_of_year))
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(days=day_index + day_frac, hours=utc_hour)


def _to_iso_utc(epoch: datetime, time_s: float) -> str:
    dt = epoch + timedelta(seconds=float(time_s))
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_spacecraft_schedule_kml(
    schedule_rows: Sequence[Dict[str, Any]],
    opportunities: Sequence[Dict[str, Any]],
    spacecraft_cfg: Dict[str, Any],
    save_path: Path,
) -> None:
    epoch = _mission_epoch_utc(spacecraft_cfg)
    placemarks: List[str] = []

    seen_targets: set[str] = set()
    seen_stations: set[str] = set()
    for row in opportunities:
        action_type = str(row.get("action_type", ""))
        lat = row.get("lat_deg")
        lon = row.get("lon_deg")
        if lat is None or lon is None:
            continue
        if action_type == "observation":
            target_id = str(row.get("target_id", "target"))
            if target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            placemarks.append(
                (
                    "<Placemark>"
                    f"<name>{xml_escape(target_id)}</name>"
                    "<styleUrl>#targetStyle</styleUrl>"
                    "<Point>"
                    f"<coordinates>{float(lon):.8f},{float(lat):.8f},0.0</coordinates>"
                    "</Point>"
                    "</Placemark>"
                )
            )
        elif action_type == "downlink":
            station_id = str(row.get("station_id", "station"))
            if station_id in seen_stations:
                continue
            seen_stations.add(station_id)
            placemarks.append(
                (
                    "<Placemark>"
                    f"<name>{xml_escape(station_id)}</name>"
                    "<styleUrl>#stationStyle</styleUrl>"
                    "<Point>"
                    f"<coordinates>{float(lon):.8f},{float(lat):.8f},0.0</coordinates>"
                    "</Point>"
                    "</Placemark>"
                )
            )

    for idx, row in enumerate(schedule_rows):
        action = str(row.get("action_type", ""))
        if action not in {"observation", "downlink"}:
            continue
        lat = row.get("lat_deg")
        lon = row.get("lon_deg")
        if lat is None or lon is None:
            continue
        start_iso = _to_iso_utc(epoch, float(row.get("task_start_s", 0.0)))
        end_iso = _to_iso_utc(epoch, float(row.get("task_end_s", 0.0)))
        tag = str(row.get("target_id" if action == "observation" else "station_id", ""))
        style = "#obsStyle" if action == "observation" else "#dlStyle"
        placemarks.append(
            (
                "<Placemark>"
                f"<name>{xml_escape(action.upper())} {xml_escape(tag)} {idx}</name>"
                "<TimeSpan>"
                f"<begin>{start_iso}</begin><end>{end_iso}</end>"
                "</TimeSpan>"
                f"<styleUrl>{style}</styleUrl>"
                f"<description>value={float(row.get('value', 0.0)):.2f}</description>"
                "<Point>"
                f"<coordinates>{float(lon):.8f},{float(lat):.8f},0.0</coordinates>"
                "</Point>"
                "</Placemark>"
            )
        )

    kml_text = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<kml xmlns=\"http://www.opengis.net/kml/2.2\">"
        "<Document><name>AeroHack Spacecraft Schedule</name>"
        "<Style id=\"targetStyle\"><IconStyle><color>ff00ff00</color></IconStyle></Style>"
        "<Style id=\"stationStyle\"><IconStyle><color>ffffaa00</color></IconStyle></Style>"
        "<Style id=\"obsStyle\"><IconStyle><color>ff00ff00</color></IconStyle></Style>"
        "<Style id=\"dlStyle\"><IconStyle><color>ffff6600</color></IconStyle></Style>"
        + "".join(placemarks)
        + "</Document></kml>"
    )
    save_path.write_text(kml_text, encoding="utf-8")


def run_spacecraft(
    spacecraft_cfg: Dict[str, Any],
    output_dir: Path,
    strategy: str = "beam",
    hybrid_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_spacecraft_problem(spacecraft_cfg)
    planner = UnifiedPlanner()
    baseline_result = planner.solve(problem, strategy="beam")
    result = baseline_result
    chosen_solver = "beam"

    if strategy == "auto_best":
        greedy_result = planner.solve(problem, strategy="greedy")
        best_nonhybrid_label, best_nonhybrid = _best_plan_result(
            [
                ("beam", baseline_result),
                ("greedy", greedy_result),
            ]
        )
        result = best_nonhybrid
        chosen_solver = f"{best_nonhybrid_label}(auto_best)"

        hybrid_result = UnifiedHybridOCPEngine(hybrid_cfg).solve(problem, planner=planner)
        if _accept_hybrid_result(hybrid_result, best_nonhybrid):
            result = hybrid_result
            chosen_solver = "hybrid_ocp(auto_best)"
    elif strategy == "hybrid_ocp":
        greedy_result = planner.solve(problem, strategy="greedy")
        best_nonhybrid_label, best_nonhybrid = _best_plan_result(
            [
                ("beam", baseline_result),
                ("greedy", greedy_result),
            ]
        )
        hybrid_result = UnifiedHybridOCPEngine(hybrid_cfg).solve(problem, planner=planner)
        if _accept_hybrid_result(hybrid_result, best_nonhybrid):
            result = hybrid_result
            chosen_solver = "hybrid_ocp"
        else:
            result = best_nonhybrid
            chosen_solver = f"{best_nonhybrid_label}(fallback_from_hybrid)"
    elif strategy == "beam":
        result = baseline_result
        chosen_solver = "beam"
    elif strategy == "greedy":
        result = planner.solve(problem, strategy="greedy")
        chosen_solver = "greedy"
    elif strategy == "multistart":
        result = planner.solve(problem, strategy="multistart")
        chosen_solver = "multistart"
    else:
        raise ValueError(f"Unsupported spacecraft strategy: {strategy}")

    schedule_rows = _schedule_rows(result, problem.tasks)
    opportunities = _opportunity_rows(problem.tasks)

    schedule_csv = output_dir / "spacecraft_7day_schedule.csv"
    visibility_csv = output_dir / "visibility_windows.csv"
    write_csv(schedule_csv, schedule_rows)
    write_csv(visibility_csv, opportunities)

    gantt_path = output_dir / "spacecraft_gantt.png"
    resources_path = output_dir / "spacecraft_resources.png"
    plot_spacecraft_gantt(schedule_rows, gantt_path)
    plot_spacecraft_resources(schedule_rows, float(spacecraft_cfg.get("battery_initial_wh", 0.0)), resources_path)
    kml_path = output_dir / "spacecraft_schedule.kml"
    write_spacecraft_schedule_kml(schedule_rows, opportunities, spacecraft_cfg, kml_path)

    hard_violations = len(result.hard_violations)
    if schedule_rows:
        battery_values = [float(r["battery_wh"]) for r in schedule_rows]
        data_values = [float(r["data_buffer_mb"]) for r in schedule_rows]
        battery_min = min(battery_values)
        battery_max = max(battery_values)
        data_max = max(data_values)
    else:
        battery_min = float(spacecraft_cfg.get("battery_initial_wh", 0.0))
        battery_max = float(spacecraft_cfg.get("battery_initial_wh", 0.0))
        data_max = 0.0

    metrics = {
        "solver_strategy": chosen_solver,
        "baseline_beam_objective_score": float(baseline_result.objective_score),
        "solved": bool(result.solved),
        "status": result.status,
        "objective_score": float(result.objective_score),
        "sequence_length": len(result.sequence),
        "executed_observations": int(result.final_state.get("executed_observations", 0)),
        "executed_downlinks": int(result.final_state.get("executed_downlinks", 0)),
        "delivered_science": float(result.final_state.get("delivered_science", 0.0)),
        "final_data_buffer_mb": float(result.final_state.get("data_buffer_mb", 0.0)),
        "battery_min_wh": battery_min,
        "battery_max_wh": battery_max,
        "data_buffer_max_mb": data_max,
        "final_time_s": float(result.final_state.get("time_s", 0.0)),
        "configured_horizon_s": float(spacecraft_cfg.get("mission_days", 7) * 24 * 3600),
        "hard_constraint_violations": hard_violations,
    }

    constraints_summary = {
        "hard_violations": result.hard_violations,
        "hard_violation_count": hard_violations,
        "battery_below_min": battery_min < float(spacecraft_cfg.get("battery_min_wh", 0.0)),
        "buffer_overflow": data_max > float(spacecraft_cfg.get("data_buffer_capacity_mb", 0.0)),
        "max_ops_per_orbit": int(spacecraft_cfg.get("max_ops_per_orbit", 0)),
    }
    certification_rows = _build_spacecraft_constraint_certification(result, schedule_rows, spacecraft_cfg)
    certification_csv = output_dir / "spacecraft_constraint_certification.csv"
    certification_json = output_dir / "spacecraft_constraint_certification.json"

    (output_dir / "spacecraft_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "spacecraft_constraint_summary.json").write_text(json.dumps(constraints_summary, indent=2))
    write_certification_report(certification_csv, certification_json, certification_rows)

    return {
        "problem": problem,
        "result": result,
        "metrics": metrics,
        "schedule_csv": str(schedule_csv),
        "visibility_csv": str(visibility_csv),
        "gantt_plot": str(gantt_path),
        "resources_plot": str(resources_path),
        "kml_path": str(kml_path),
        "constraint_summary": constraints_summary,
        "constraint_certification_csv": str(certification_csv),
        "constraint_certification_json": str(certification_json),
    }
