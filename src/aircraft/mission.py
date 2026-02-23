from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from src.aircraft.model import (
    AircraftDynamicsConfig,
    WindModel,
    build_aircraft_tasks,
    build_aircraft_transitions,
    build_initial_state,
    build_polygons,
    clear_aircraft_transition_cache,
    simulate_aircraft_step,
)
from src.core import (
    CallableConstraint,
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

EARTH_RADIUS_M = 6378137.0


def _accept_hybrid_result(hybrid_result, baseline_result) -> bool:
    if not UnifiedHybridOCPEngine.is_better(hybrid_result, baseline_result):
        return False
    base_time = float(baseline_result.final_state.get("time_s", 0.0))
    hyb_time = float(hybrid_result.final_state.get("time_s", 0.0))
    # Guardrail: reject hybrid if it sacrifices excessive mission time for marginal objective gain.
    if base_time > 0.0 and hyb_time > 1.05 * base_time:
        return False
    return True


def _best_plan_result(candidates: Sequence[tuple[str, Any]]) -> tuple[str, Any]:
    best_label, best_result = candidates[0]
    for label, candidate in candidates[1:]:
        if UnifiedHybridOCPEngine.is_better(candidate, best_result):
            best_label = label
            best_result = candidate
    return best_label, best_result


def _aircraft_candidate_fn(
    problem: MissionProblem,
    state: Dict[str, Any],
    current_task_id: str,
    visited: Sequence[str],
) -> List[str]:
    visited_set = set(visited)
    required_ids = [tid for tid in problem.required_task_ids if tid not in visited_set]
    now_s = float(state.get("time_s", 0.0))

    def transition_cost(task_id: str) -> tuple[float, float]:
        trans = problem.transitions.get((current_task_id, task_id))
        if trans is None or not trans.feasible:
            return float("inf"), float("inf")
        distance = float(trans.metadata.get("distance_m", trans.travel_time_s))
        earliest = max(now_s + float(trans.travel_time_s), float(problem.tasks[task_id].window_start_s))
        return distance, earliest

    if required_ids:
        feasible_required = [tid for tid in required_ids if transition_cost(tid)[0] < float("inf")]
        ordered = sorted(feasible_required, key=lambda tid: transition_cost(tid))
        return ordered[: problem.settings.candidate_limit]

    if problem.end_task_id and problem.end_task_id not in visited_set:
        return [problem.end_task_id]
    return []


def build_aircraft_problem(
    aircraft_cfg: Dict[str, Any],
    seed: int = 42,
    wind_scale: float = 1.0,
    battery_scale: float = 1.0,
    planner_overrides: Dict[str, Any] | None = None,
) -> MissionProblem:
    if bool(aircraft_cfg.get("clear_model_caches", False)):
        clear_aircraft_transition_cache()
    rng = np.random.default_rng(seed)

    tasks = build_aircraft_tasks(aircraft_cfg)
    geofences = build_polygons(aircraft_cfg.get("no_fly_zones", []))
    aero_cfg = aircraft_cfg.get("aerodynamics", {})

    dynamics = AircraftDynamicsConfig(
        cruise_speed_mps=float(aircraft_cfg["cruise_speed_mps"]),
        min_ground_speed_mps=float(aircraft_cfg.get("min_ground_speed_mps", 10.0)),
        cruise_power_w=float(aircraft_cfg["cruise_power_w"]),
        turn_power_w=float(aircraft_cfg.get("turn_power_w", aircraft_cfg["cruise_power_w"] * 1.1)),
        loiter_power_w=float(aircraft_cfg.get("loiter_power_w", aircraft_cfg["cruise_power_w"] * 0.8)),
        climb_power_w_per_mps=float(aircraft_cfg.get("climb_power_w_per_mps", 160.0)),
        max_turn_rate_deg_s=float(aircraft_cfg["max_turn_rate_deg_s"]),
        max_bank_angle_deg=float(aircraft_cfg.get("max_bank_angle_deg", 30.0)),
        max_climb_rate_mps=float(aircraft_cfg.get("max_climb_rate_mps", 3.0)),
        battery_capacity_wh=float(aircraft_cfg["battery_capacity_wh"]) * battery_scale,
        reserve_energy_wh=float(aircraft_cfg.get("reserve_energy_wh", 0.0)),
        altitude_min_m=float(aircraft_cfg.get("altitude_min_m", 0.0)),
        altitude_max_m=float(aircraft_cfg.get("altitude_max_m", 5000.0)),
        mission_horizon_s=float(aircraft_cfg["mission_horizon_s"]),
        geofence_margin_m=float(aircraft_cfg.get("geofence_margin_m", 400.0)),
        path_sample_spacing_m=float(aircraft_cfg.get("path_sample_spacing_m", 500.0)),
        mass_kg=float(aero_cfg.get("mass_kg", 7.5)),
        wing_area_m2=float(aero_cfg.get("wing_area_m2", 0.62)),
        cd0=float(aero_cfg.get("cd0", 0.028)),
        aspect_ratio=float(aero_cfg.get("aspect_ratio", 12.5)),
        oswald_efficiency=float(aero_cfg.get("oswald_efficiency", 0.82)),
        propulsive_efficiency=float(aero_cfg.get("propulsive_efficiency", 0.78)),
        air_density_kg_m3=float(aero_cfg.get("air_density_kg_m3", 1.225)),
        auxiliary_power_w=float(aero_cfg.get("auxiliary_power_w", 70.0)),
        propulsion_scale=float(aero_cfg.get("propulsion_scale", 0.58)),
        cl_max=float(aero_cfg.get("cl_max", 1.45)),
        stall_margin=float(aero_cfg.get("stall_margin", 1.10)),
        max_propulsion_power_w=float(aero_cfg.get("max_propulsion_power_w", 3200.0)),
        prop_power_altitude_scale_m=float(aero_cfg.get("prop_power_altitude_scale_m", 9000.0)),
        temp_offset_c=float(aero_cfg.get("temp_offset_c", 0.0)),
    )

    wind_cfg = aircraft_cfg["wind"]
    phase_u_rad = float(wind_cfg.get("phase_u_rad", rng.uniform(0.0, 2.0 * math.pi)))
    phase_v_rad = float(wind_cfg.get("phase_v_rad", rng.uniform(0.0, 2.0 * math.pi)))
    wind_model = WindModel(
        base_u_mps=float(wind_cfg.get("base_u_mps", 0.0)),
        base_v_mps=float(wind_cfg.get("base_v_mps", 0.0)),
        gust_mps=float(wind_cfg.get("gust_mps", 0.0)),
        spatial_scale_m=float(wind_cfg.get("spatial_scale_m", 10000.0)),
        temporal_period_s=float(wind_cfg.get("temporal_period_s", 3600.0)),
        phase_u_rad=phase_u_rad,
        phase_v_rad=phase_v_rad,
        harmonic_ratio=float(wind_cfg.get("harmonic_ratio", 0.35)),
    )

    transitions = build_aircraft_transitions(tasks, dynamics, geofences, wind_model=wind_model)

    required_task_ids = [t.task_id for t in tasks.values() if t.required]

    planner_cfg = aircraft_cfg.get("planner", {})
    settings = PlannerSettings(
        beam_width=int(planner_cfg.get("beam_width", 96)),
        max_expansions=int(planner_cfg.get("max_expansions", 40000)),
        max_depth=int(planner_cfg.get("max_depth", 32)),
        candidate_limit=int(planner_cfg.get("candidate_limit", 16)),
        lookahead_s=float(planner_cfg.get("lookahead_s", aircraft_cfg["mission_horizon_s"])),
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
            raise ValueError(f"Unknown aircraft planner_overrides keys: {', '.join(unknown)}")
        for key, value in planner_overrides.items():
            setattr(settings, key, value)

    def geofence_constraint(
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> tuple[bool, str, float]:
        ok = bool(step_meta.get("geofence_clear", True))
        return ok, "segment intersects no-fly zone" if not ok else "", 1.0 if not ok else 0.0

    def turn_constraint(
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> tuple[bool, str, float]:
        rate = float(step_meta.get("turn_rate_used_rad_s", 0.0))
        ok = rate <= dynamics.max_turn_rate_rad_s + 1e-6
        return ok, f"turn rate {rate:.4f} rad/s exceeds limit" if not ok else "", max(0.0, rate - dynamics.max_turn_rate_rad_s)

    def aerodynamic_constraint(
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> tuple[bool, str, float]:
        feasible = bool(step_meta.get("aero_feasible", True))
        stall_margin = float(step_meta.get("min_stall_margin_ratio", float("inf")))
        power_margin_w = float(step_meta.get("min_power_margin_w", float("inf")))
        slack = min(stall_margin - 1.0, power_margin_w / max(1.0, dynamics.max_propulsion_power_w))
        if feasible:
            return True, "", max(0.0, -slack)
        return (
            False,
            f"aerodynamic feasibility failed (stall_margin={stall_margin:.3f}, power_margin_w={power_margin_w:.1f})",
            max(0.0, -slack),
        )

    def simulate(
        state: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
    ):
        return simulate_aircraft_step(
            state=state,
            from_task=from_task,
            to_task=to_task,
            transition=transition,
            dynamics=dynamics,
            wind_model=wind_model,
            geofences=geofences,
        )

    constraints = [
        TaskWindowConstraint(name="window", hard=True),
        RangeConstraint(key="time_s", min_value=0.0, max_value=dynamics.mission_horizon_s, name="horizon", hard=True),
        RangeConstraint(
            key="energy_wh",
            min_value=dynamics.reserve_energy_wh,
            max_value=dynamics.battery_capacity_wh,
            name="energy_bounds",
            hard=True,
        ),
        RangeConstraint(
            key="alt_m",
            min_value=dynamics.altitude_min_m,
            max_value=dynamics.altitude_max_m,
            name="altitude_bounds",
            hard=True,
        ),
        CallableConstraint(fn=geofence_constraint, name="geofence", hard=True),
        CallableConstraint(fn=turn_constraint, name="turn_rate", hard=True),
        CallableConstraint(fn=aerodynamic_constraint, name="aerodynamic_feasibility", hard=True),
    ]

    objective_cfg = aircraft_cfg.get("objective", {})
    objective = WeightedObjective(
        value_weight=float(objective_cfg.get("value_weight", 1.0)),
        delivered_weight=float(objective_cfg.get("delivered_weight", 0.0)),
        time_weight=float(objective_cfg.get("time_weight", 0.02)),
        energy_weight=float(objective_cfg.get("energy_weight", 0.1)),
        required_completion_bonus=float(objective_cfg.get("required_completion_bonus", 500.0)),
        unfinished_required_penalty=float(objective_cfg.get("unfinished_required_penalty", 5000.0)),
    )

    initial_state = build_initial_state(aircraft_cfg, battery_scale=battery_scale)
    initial_state["wind_scale"] = float(wind_scale)
    initial_state["wind_phase_u_rad"] = float(wind_cfg.get("runtime_phase_u_rad", 0.0))
    initial_state["wind_phase_v_rad"] = float(wind_cfg.get("runtime_phase_v_rad", 0.0))

    return MissionProblem(
        name="aircraft_mission",
        domain="aircraft",
        tasks=tasks,
        transitions=transitions,
        start_task_id="BASE_START",
        end_task_id="BASE_END",
        required_task_ids=required_task_ids,
        initial_state=initial_state,
        simulate_step=simulate,
        constraints=constraints,
        objective=objective,
        settings=settings,
        candidate_fn=_aircraft_candidate_fn,
    )


def _collect_trajectory_rows(result) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in result.steps:
        samples = step.metadata.get("samples", [])
        for sample in samples:
            rows.append(
                {
                    "time_s": sample["time_s"],
                    "x_m": sample["x_m"],
                    "y_m": sample["y_m"],
                    "alt_m": sample["alt_m"],
                    "heading_rad": sample["heading_rad"],
                    "energy_wh": sample["energy_wh"],
                    "from_task": step.from_task_id,
                    "to_task": step.to_task_id,
                }
            )

    rows.sort(key=lambda r: float(r["time_s"]))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (round(float(row["time_s"]), 3), round(float(row["x_m"]), 3), round(float(row["y_m"]), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _collect_segment_rows(result) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in result.steps:
        rows.append(
            {
                "from_task": step.from_task_id,
                "to_task": step.to_task_id,
                "start_time_s": step.start_time_s,
                "end_time_s": step.end_time_s,
                "transition_time_s": step.transition_time_s,
                "task_duration_s": step.task_duration_s,
                "distance_m": step.metadata.get("distance_m", 0.0),
                "ground_speed_mps": step.metadata.get("ground_speed_mps", 0.0),
                "wind_u_mps": step.metadata.get("wind_u_mps", 0.0),
                "wind_v_mps": step.metadata.get("wind_v_mps", 0.0),
                "turn_rate_used_rad_s": step.metadata.get("turn_rate_used_rad_s", 0.0),
                "min_stall_margin_ratio": step.metadata.get("min_stall_margin_ratio", float("inf")),
                "min_power_margin_w": step.metadata.get("min_power_margin_w", float("inf")),
                "avg_density_kg_m3": step.metadata.get("avg_density_kg_m3", float("nan")),
                "aero_feasible": step.metadata.get("aero_feasible", True),
                "geofence_clear": step.metadata.get("geofence_clear", True),
            }
        )
    return rows


def _pseudo_multiplier(max_violation: float, min_slack: float, active_tol: float) -> float:
    if max_violation > 0.0:
        return float(1000.0 + 1000.0 * max_violation)
    if min_slack <= active_tol:
        return float(1.0 / max(1e-3, min_slack + 1e-3))
    return 0.0


def _build_aircraft_constraint_certification(
    result: Any,
    problem: MissionProblem,
    aircraft_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for step in result.steps:
        for s in step.metadata.get("samples", []):
            samples.append(s)

    if not samples:
        samples = [
            {
                "time_s": float(problem.initial_state.get("time_s", 0.0)),
                "x_m": float(problem.initial_state.get("x_m", 0.0)),
                "y_m": float(problem.initial_state.get("y_m", 0.0)),
                "alt_m": float(problem.initial_state.get("alt_m", 0.0)),
                "energy_wh": float(problem.initial_state.get("energy_wh", 0.0)),
            }
        ]

    altitudes = np.array([float(s.get("alt_m", np.nan)) for s in samples], dtype=float)
    energies = np.array([float(s.get("energy_wh", np.nan)) for s in samples], dtype=float)
    turn_rates = np.array([float(step.metadata.get("turn_rate_used_rad_s", 0.0)) for step in result.steps], dtype=float)
    turn_rates = turn_rates[np.isfinite(turn_rates)]
    stall_margins = np.array([float(step.metadata.get("min_stall_margin_ratio", np.nan)) for step in result.steps], dtype=float)
    stall_margins = stall_margins[np.isfinite(stall_margins)]
    power_margins = np.array([float(step.metadata.get("min_power_margin_w", np.nan)) for step in result.steps], dtype=float)
    power_margins = power_margins[np.isfinite(power_margins)]

    min_alt = float(aircraft_cfg.get("altitude_min_m", -1e9))
    max_alt = float(aircraft_cfg.get("altitude_max_m", 1e9))
    reserve_wh = float(aircraft_cfg.get("reserve_energy_wh", 0.0))
    capacity_wh = float(aircraft_cfg.get("battery_capacity_wh", max(reserve_wh + 1.0, 1.0)))
    horizon_s = float(aircraft_cfg.get("mission_horizon_s", 0.0))
    max_turn_rate_rad_s = math.radians(float(aircraft_cfg.get("max_turn_rate_deg_s", 0.0)))
    geofence_margin_m = float(aircraft_cfg.get("geofence_margin_m", 0.0))
    geofences = build_polygons(aircraft_cfg.get("no_fly_zones", []))

    finite_alt = altitudes[np.isfinite(altitudes)]
    finite_energy = energies[np.isfinite(energies)]
    min_alt_seen = float(np.min(finite_alt)) if finite_alt.size else float(problem.initial_state.get("alt_m", 0.0))
    max_alt_seen = float(np.max(finite_alt)) if finite_alt.size else float(problem.initial_state.get("alt_m", 0.0))
    min_energy_seen = float(np.min(finite_energy)) if finite_energy.size else float(problem.initial_state.get("energy_wh", 0.0))

    alt_violation = max(0.0, min_alt - min_alt_seen, max_alt_seen - max_alt)
    alt_slack = min(min_alt_seen - min_alt, max_alt - max_alt_seen)
    alt_active = (
        float(np.mean([(abs(float(v) - min_alt) <= 5.0) or (abs(max_alt - float(v)) <= 5.0) for v in finite_alt]))
        if finite_alt.size
        else 0.0
    )

    energy_violation = max(0.0, reserve_wh - min_energy_seen)
    energy_slack = min_energy_seen - reserve_wh
    energy_active_threshold = reserve_wh + max(5.0, 0.03 * capacity_wh)
    energy_active = float(np.mean([float(v) <= energy_active_threshold for v in finite_energy])) if finite_energy.size else 0.0

    if turn_rates.size:
        turn_eps = 1e-9
        turn_violation = float(np.max(np.maximum(turn_rates - max_turn_rate_rad_s - turn_eps, 0.0)))
        turn_slack = float(np.min(max_turn_rate_rad_s - turn_rates))
        if abs(turn_slack) <= turn_eps:
            turn_slack = 0.0
        turn_active = float(np.mean(turn_rates >= 0.95 * max_turn_rate_rad_s))
        turn_peak = float(np.max(turn_rates))
    else:
        turn_violation = 0.0
        turn_slack = max_turn_rate_rad_s
        turn_active = 0.0
        turn_peak = 0.0

    if stall_margins.size:
        stall_violation = float(np.max(np.maximum(1.0 - stall_margins, 0.0)))
        stall_slack = float(np.min(stall_margins - 1.0))
        stall_active = float(np.mean(stall_margins <= 1.08))
        min_stall_margin = float(np.min(stall_margins))
    else:
        stall_violation = 0.0
        stall_slack = float("inf")
        stall_active = 0.0
        min_stall_margin = float("inf")

    if power_margins.size:
        power_violation = float(np.max(np.maximum(-power_margins, 0.0)))
        power_slack = float(np.min(power_margins))
        power_active = float(np.mean(power_margins <= max(100.0, 0.05 * float(aircraft_cfg.get("cruise_power_w", 100.0)))))
        min_power_margin = float(np.min(power_margins))
    else:
        power_violation = 0.0
        power_slack = float("inf")
        power_active = 0.0
        min_power_margin = float("inf")

    geofence_slacks: List[float] = []
    for s in samples:
        p = Point(float(s["x_m"]), float(s["y_m"]))
        if not geofences:
            geofence_slacks.append(float("inf"))
            continue
        local_slacks: List[float] = []
        for poly in geofences:
            dist = float(poly.distance(p))
            if poly.contains(p):
                local_slacks.append(-geofence_margin_m)
            else:
                local_slacks.append(dist - geofence_margin_m)
        geofence_slacks.append(min(local_slacks))

    if geofence_slacks:
        min_geofence_slack = float(np.min(np.array(geofence_slacks, dtype=float)))
        geofence_violation = max(0.0, -min_geofence_slack)
        geofence_active = float(np.mean([s <= max(10.0, 0.1 * geofence_margin_m) for s in geofence_slacks]))
    else:
        min_geofence_slack = float("inf")
        geofence_violation = 0.0
        geofence_active = 0.0

    final_time_s = float(result.final_state.get("time_s", 0.0))
    horizon_violation = max(0.0, final_time_s - horizon_s)
    horizon_slack = horizon_s - final_time_s
    horizon_active = 1.0 if horizon_slack <= max(60.0, 0.02 * max(1.0, horizon_s)) else 0.0

    return [
        {
            "constraint": "energy_bounds",
            "max_violation": energy_violation,
            "slack_margin": energy_slack,
            "active_fraction": energy_active,
            "pseudo_multiplier": _pseudo_multiplier(energy_violation, energy_slack, 5.0),
            "limit": reserve_wh,
            "observed": min_energy_seen,
            "units": "Wh",
        },
        {
            "constraint": "turn_rate",
            "max_violation": turn_violation,
            "slack_margin": turn_slack,
            "active_fraction": turn_active,
            "pseudo_multiplier": _pseudo_multiplier(turn_violation, turn_slack, 0.02 * max_turn_rate_rad_s),
            "limit": max_turn_rate_rad_s,
            "observed": turn_peak,
            "units": "rad/s",
        },
        {
            "constraint": "stall_margin",
            "max_violation": stall_violation,
            "slack_margin": stall_slack,
            "active_fraction": stall_active,
            "pseudo_multiplier": _pseudo_multiplier(stall_violation, stall_slack, 0.08),
            "limit": 1.0,
            "observed": min_stall_margin,
            "units": "ratio",
        },
        {
            "constraint": "propulsion_power_margin",
            "max_violation": power_violation,
            "slack_margin": power_slack,
            "active_fraction": power_active,
            "pseudo_multiplier": _pseudo_multiplier(power_violation, power_slack, 100.0),
            "limit": 0.0,
            "observed": min_power_margin,
            "units": "W",
        },
        {
            "constraint": "altitude_bounds",
            "max_violation": alt_violation,
            "slack_margin": alt_slack,
            "active_fraction": alt_active,
            "pseudo_multiplier": _pseudo_multiplier(alt_violation, alt_slack, 5.0),
            "limit": f"[{min_alt},{max_alt}]",
            "observed": f"[{min_alt_seen},{max_alt_seen}]",
            "units": "m",
        },
        {
            "constraint": "geofence_margin",
            "max_violation": geofence_violation,
            "slack_margin": min_geofence_slack,
            "active_fraction": geofence_active,
            "pseudo_multiplier": _pseudo_multiplier(
                geofence_violation,
                min_geofence_slack,
                max(10.0, 0.1 * geofence_margin_m),
            ),
            "limit": geofence_margin_m,
            "observed": min_geofence_slack + geofence_margin_m,
            "units": "m",
        },
        {
            "constraint": "mission_horizon",
            "max_violation": horizon_violation,
            "slack_margin": horizon_slack,
            "active_fraction": horizon_active,
            "pseudo_multiplier": _pseudo_multiplier(
                horizon_violation,
                horizon_slack,
                max(60.0, 0.02 * max(1.0, horizon_s)),
            ),
            "limit": horizon_s,
            "observed": final_time_s,
            "units": "s",
        },
    ]


def plot_aircraft_path(
    trajectory_rows: Sequence[Dict[str, Any]],
    tasks: Dict[str, MissionTask],
    no_fly_zones: Sequence[Sequence[Sequence[float]]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for polygon in no_fly_zones:
        poly = np.asarray(polygon)
        ax.fill(poly[:, 0], poly[:, 1], color="tab:red", alpha=0.25, edgecolor="tab:red", linewidth=1.5)

    if trajectory_rows:
        xs = [float(r["x_m"]) for r in trajectory_rows]
        ys = [float(r["y_m"]) for r in trajectory_rows]
        ax.plot(xs, ys, color="tab:blue", linewidth=2.0, label="planned path")

    for task_id, task in tasks.items():
        x = task.metadata["x_m"]
        y = task.metadata["y_m"]
        if task.task_type == "waypoint":
            ax.scatter([x], [y], color="tab:green", s=45)
            ax.text(x + 250, y + 250, task_id, fontsize=8)
        elif task.task_type in {"start", "end"}:
            ax.scatter([x], [y], color="black", s=60, marker="s")

    ax.set_title("Aircraft Mission Path with Geofences")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_aircraft_energy_profile(
    trajectory_rows: Sequence[Dict[str, Any]],
    save_path: Path,
) -> None:
    if not trajectory_rows:
        return
    times = [float(r["time_s"]) / 3600.0 for r in trajectory_rows]
    energies = [float(r["energy_wh"]) for r in trajectory_rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, energies, color="tab:orange", linewidth=2.0)
    ax.set_title("Aircraft Energy vs Time")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Energy [Wh]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _xy_to_latlon(x_m: float, y_m: float, ref_lat_deg: float, ref_lon_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(ref_lat_deg)
    dlat = (float(y_m) / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (float(x_m) / max(1e-6, EARTH_RADIUS_M * math.cos(lat_rad))) * (180.0 / math.pi)
    return ref_lat_deg + dlat, ref_lon_deg + dlon


def _latlonalt_to_ecef_km(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r_km = (EARTH_RADIUS_M + float(alt_m)) / 1000.0
    x = r_km * math.cos(lat) * math.cos(lon)
    y = r_km * math.cos(lat) * math.sin(lon)
    z = r_km * math.sin(lat)
    return x, y, z


def write_aircraft_3d_html(
    trajectory_rows: Sequence[Dict[str, Any]],
    save_path: Path,
    reference_lat_deg: float,
    reference_lon_deg: float,
) -> None:
    if not trajectory_rows:
        save_path.write_text("<html><body>No trajectory samples available.</body></html>", encoding="utf-8")
        return

    latlons = [
        _xy_to_latlon(float(r["x_m"]), float(r["y_m"]), reference_lat_deg, reference_lon_deg)
        for r in trajectory_rows
    ]
    alts = [float(r.get("alt_m", 0.0)) for r in trajectory_rows]
    path_xyz = [_latlonalt_to_ecef_km(lat, lon, alt) for (lat, lon), alt in zip(latlons, alts)]
    px = [p[0] for p in path_xyz]
    py = [p[1] for p in path_xyz]
    pz = [p[2] for p in path_xyz]

    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception:
        rows = []
        for i, ((lat, lon), alt) in enumerate(zip(latlons, alts)):
            rows.append(f"<tr><td>{i}</td><td>{lat:.6f}</td><td>{lon:.6f}</td><td>{alt:.2f}</td></tr>")
        html = (
            "<html><body><h2>Aircraft 3D Trajectory</h2>"
            "<p>Plotly not installed; table fallback generated.</p>"
            "<table border='1'><tr><th>idx</th><th>lat</th><th>lon</th><th>alt_m</th></tr>"
            + "".join(rows)
            + "</table></body></html>"
        )
        save_path.write_text(html, encoding="utf-8")
        return

    u = np.linspace(0.0, 2.0 * math.pi, 50)
    v = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 28)
    uu, vv = np.meshgrid(u, v)
    r_earth_km = EARTH_RADIUS_M / 1000.0
    sx = r_earth_km * np.cos(vv) * np.cos(uu)
    sy = r_earth_km * np.cos(vv) * np.sin(uu)
    sz = r_earth_km * np.sin(vv)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=sx,
            y=sy,
            z=sz,
            colorscale="Blues",
            opacity=0.55,
            showscale=False,
            name="Earth",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode="lines",
            line=dict(color="red", width=6),
            name="UAV Path",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[px[0], px[-1]],
            y=[py[0], py[-1]],
            z=[pz[0], pz[-1]],
            mode="markers",
            marker=dict(size=6, color=["green", "black"]),
            name="Start/End",
        )
    )
    fig.update_layout(
        title="Aircraft 3D Trajectory over Earth",
        scene=dict(
            xaxis_title="X [km]",
            yaxis_title="Y [km]",
            zaxis_title="Z [km]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")


def write_aircraft_kml(
    trajectory_rows: Sequence[Dict[str, Any]],
    tasks: Dict[str, MissionTask],
    no_fly_zones: Sequence[Sequence[Sequence[float]]],
    save_path: Path,
    reference_lat_deg: float,
    reference_lon_deg: float,
) -> None:
    placemarks: List[str] = []

    if trajectory_rows:
        path_coords: List[str] = []
        for row in trajectory_rows:
            lat, lon = _xy_to_latlon(
                float(row["x_m"]),
                float(row["y_m"]),
                reference_lat_deg,
                reference_lon_deg,
            )
            alt = float(row.get("alt_m", 0.0))
            path_coords.append(f"{lon:.8f},{lat:.8f},{alt:.2f}")
        placemarks.append(
            (
                "<Placemark><name>UAV Planned Path</name><styleUrl>#pathStyle</styleUrl>"
                "<LineString><altitudeMode>absolute</altitudeMode><coordinates>"
                + " ".join(path_coords)
                + "</coordinates></LineString></Placemark>"
            )
        )

    for task_id, task in tasks.items():
        x = float(task.metadata.get("x_m", 0.0))
        y = float(task.metadata.get("y_m", 0.0))
        z = float(task.metadata.get("alt_m", 0.0))
        lat, lon = _xy_to_latlon(x, y, reference_lat_deg, reference_lon_deg)
        style = "#startEndStyle" if task.task_type in {"start", "end"} else "#waypointStyle"
        placemarks.append(
            (
                "<Placemark>"
                f"<name>{xml_escape(task_id)}</name>"
                f"<description>type={xml_escape(task.task_type)}</description>"
                f"<styleUrl>{style}</styleUrl>"
                "<Point><altitudeMode>absolute</altitudeMode>"
                f"<coordinates>{lon:.8f},{lat:.8f},{z:.2f}</coordinates>"
                "</Point></Placemark>"
            )
        )

    for idx, polygon in enumerate(no_fly_zones):
        if not polygon:
            continue
        coords: List[str] = []
        for x_m, y_m in polygon:
            lat, lon = _xy_to_latlon(float(x_m), float(y_m), reference_lat_deg, reference_lon_deg)
            coords.append(f"{lon:.8f},{lat:.8f},0.0")
        first_x, first_y = polygon[0]
        first_lat, first_lon = _xy_to_latlon(float(first_x), float(first_y), reference_lat_deg, reference_lon_deg)
        coords.append(f"{first_lon:.8f},{first_lat:.8f},0.0")
        placemarks.append(
            (
                "<Placemark>"
                f"<name>No-Fly Zone {idx + 1}</name><styleUrl>#nfzStyle</styleUrl>"
                "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                + " ".join(coords)
                + "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                "</Placemark>"
            )
        )

    kml_text = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<kml xmlns=\"http://www.opengis.net/kml/2.2\">"
        "<Document><name>AeroHack Aircraft Mission</name>"
        "<Style id=\"pathStyle\"><LineStyle><color>ff2a6bff</color><width>3</width></LineStyle></Style>"
        "<Style id=\"waypointStyle\"><IconStyle><color>ff00ff00</color></IconStyle></Style>"
        "<Style id=\"startEndStyle\"><IconStyle><color>ff000000</color></IconStyle></Style>"
        "<Style id=\"nfzStyle\"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>"
        "<PolyStyle><color>550000ff</color></PolyStyle></Style>"
        + "".join(placemarks)
        + "</Document></kml>"
    )
    save_path.write_text(kml_text, encoding="utf-8")


def run_aircraft(
    aircraft_cfg: Dict[str, Any],
    output_dir: Path,
    strategy: str = "beam",
    seed: int = 42,
    wind_scale: float = 1.0,
    battery_scale: float = 1.0,
    hybrid_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_aircraft_problem(
        aircraft_cfg,
        seed=seed,
        wind_scale=wind_scale,
        battery_scale=battery_scale,
    )
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
        raise ValueError(f"Unsupported aircraft strategy: {strategy}")

    traj_rows = _collect_trajectory_rows(result)
    seg_rows = _collect_segment_rows(result)

    traj_csv = output_dir / "uav_flight_plan.csv"
    seg_csv = output_dir / "uav_route_segments.csv"
    write_csv(traj_csv, traj_rows)
    write_csv(seg_csv, seg_rows)

    path_plot = output_dir / "uav_path.png"
    plot_aircraft_path(traj_rows, problem.tasks, aircraft_cfg.get("no_fly_zones", []), path_plot)
    energy_plot = output_dir / "uav_energy_profile.png"
    plot_aircraft_energy_profile(traj_rows, energy_plot)
    kml_path = output_dir / "uav_path.kml"
    map_ref = aircraft_cfg.get("map_reference", {})
    write_aircraft_kml(
        trajectory_rows=traj_rows,
        tasks=problem.tasks,
        no_fly_zones=aircraft_cfg.get("no_fly_zones", []),
        save_path=kml_path,
        reference_lat_deg=float(map_ref.get("lat_deg", 37.4275)),
        reference_lon_deg=float(map_ref.get("lon_deg", -122.1697)),
    )
    html_3d_path = output_dir / "uav_trajectory_3d.html"
    write_aircraft_3d_html(
        trajectory_rows=traj_rows,
        save_path=html_3d_path,
        reference_lat_deg=float(map_ref.get("lat_deg", 37.4275)),
        reference_lon_deg=float(map_ref.get("lon_deg", -122.1697)),
    )

    hard_violations = len(result.hard_violations)
    geofence_violations = sum(0 if bool(row.get("geofence_clear", True)) else 1 for row in seg_rows)
    turn_rate_limit = math.radians(float(aircraft_cfg["max_turn_rate_deg_s"])) + 1e-9
    turn_violations = sum(1 for row in seg_rows if float(row.get("turn_rate_used_rad_s", 0.0)) > turn_rate_limit)

    metrics = {
        "solver_strategy": chosen_solver,
        "baseline_beam_objective_score": float(baseline_result.objective_score),
        "solved": bool(result.solved),
        "status": result.status,
        "objective_score": float(result.objective_score),
        "visited_required": int(result.visited_required),
        "required_total": int(result.required_total),
        "sequence": result.sequence,
        "total_time_s": float(result.final_state.get("time_s", 0.0)),
        "total_distance_m": float(result.final_state.get("total_distance_m", 0.0)),
        "total_energy_used_wh": float(result.final_state.get("total_energy_used_wh", 0.0)),
        "energy_remaining_wh": float(result.final_state.get("energy_wh", 0.0)),
        "hard_constraint_violations": hard_violations,
        "geofence_violations": geofence_violations,
        "turn_rate_violations": turn_violations,
    }

    constraints_summary = {
        "hard_violations": result.hard_violations,
        "hard_violation_count": hard_violations,
        "geofence_violations": geofence_violations,
        "turn_rate_violations": turn_violations,
        "energy_violation": metrics["energy_remaining_wh"] < float(aircraft_cfg.get("reserve_energy_wh", 0.0)),
    }
    certification_rows = _build_aircraft_constraint_certification(result, problem, aircraft_cfg)
    certification_csv = output_dir / "uav_constraint_certification.csv"
    certification_json = output_dir / "uav_constraint_certification.json"

    (output_dir / "uav_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "uav_constraint_summary.json").write_text(json.dumps(constraints_summary, indent=2))
    write_certification_report(certification_csv, certification_json, certification_rows)

    return {
        "problem": problem,
        "result": result,
        "metrics": metrics,
        "trajectory_csv": str(traj_csv),
        "segments_csv": str(seg_csv),
        "plot_path": str(path_plot),
        "energy_plot": str(energy_plot),
        "kml_path": str(kml_path),
        "trajectory_3d_html": str(html_3d_path),
        "constraint_summary": constraints_summary,
        "constraint_certification_csv": str(certification_csv),
        "constraint_certification_json": str(certification_json),
    }
