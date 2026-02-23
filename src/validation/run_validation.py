from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.aircraft.mission import build_aircraft_problem
from src.core import UnifiedHybridOCPEngine, UnifiedPlanner, write_csv
from src.spacecraft.mission import build_spacecraft_problem
from src.validation.plots import plot_validation_bundle


def _accept_aircraft_hybrid(hybrid_result, beam_result) -> bool:
    if not UnifiedHybridOCPEngine.is_better(hybrid_result, beam_result):
        return False
    base_time = float(beam_result.final_state.get("time_s", 0.0))
    hyb_time = float(hybrid_result.final_state.get("time_s", 0.0))
    if base_time > 0.0 and hyb_time > 1.05 * base_time:
        return False
    return True


def _accept_spacecraft_hybrid(hybrid_result, beam_result) -> bool:
    if not UnifiedHybridOCPEngine.is_better(hybrid_result, beam_result):
        return False
    base_delivered = float(beam_result.final_state.get("delivered_science", 0.0))
    hyb_delivered = float(hybrid_result.final_state.get("delivered_science", 0.0))
    return hyb_delivered + 1e-6 >= base_delivered


def _small_spacecraft_cfg(spacecraft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(spacecraft_cfg))
    # Keep validation at mission-scale by default (7 days for AeroHack).
    cfg["mission_days"] = int(
        spacecraft_cfg.get("validation_mission_days", int(spacecraft_cfg.get("mission_days", 7)))
    )
    cfg["max_obs_windows"] = int(min(int(spacecraft_cfg.get("max_obs_windows", 180)), 90))
    cfg["max_dl_windows"] = int(min(int(spacecraft_cfg.get("max_dl_windows", 160)), 80))
    cfg["opportunity_step_s"] = float(max(90.0, float(spacecraft_cfg.get("opportunity_step_s", 90.0))))
    return cfg


def _aircraft_monte_carlo(
    aircraft_cfg: Dict[str, Any],
    n_runs: int,
    hybrid_eval_runs: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    planner = UnifiedPlanner()
    hybrid = UnifiedHybridOCPEngine(
        {
            "control_iterations": 1,
            "max_sequence_candidates": 4,
            "max_mutations_per_seed": 6,
            "max_control_nodes_aircraft": 10,
        }
    )
    mc_planner_overrides = (
        {"beam_width": 18, "max_expansions": 2800, "candidate_limit": 8}
        if n_runs >= 20
        else {"beam_width": 32, "max_expansions": 9000, "candidate_limit": 10}
    )
    aircraft_cfg_mc = json.loads(json.dumps(aircraft_cfg))
    if n_runs >= 20:
        # Coarser sampling for high-volume robustness sweeps to keep runtime practical.
        aircraft_cfg_mc["path_sample_spacing_m"] = float(max(1200.0, float(aircraft_cfg.get("path_sample_spacing_m", 450.0))))
    problem = build_aircraft_problem(
        aircraft_cfg_mc,
        seed=1000,
        wind_scale=1.0,
        battery_scale=1.0,
        planner_overrides=mc_planner_overrides,
    )
    base_initial = dict(problem.initial_state)
    capacity_wh = float(aircraft_cfg_mc["battery_capacity_wh"])

    for run_idx in range(n_runs):
        wind_scale = float(np.clip(rng.normal(loc=1.0, scale=0.20), 0.6, 1.5))
        battery_scale = float(np.clip(rng.normal(loc=1.0, scale=0.05), 0.85, 1.2))
        wind_phase_u = float(rng.uniform(-np.pi, np.pi))
        wind_phase_v = float(rng.uniform(-np.pi, np.pi))
        initial_energy = min(capacity_wh, float(base_initial["energy_wh"]) * battery_scale)
        problem.initial_state = {
            **base_initial,
            "energy_wh": initial_energy,
            "wind_scale": wind_scale,
            "wind_phase_u_rad": wind_phase_u,
            "wind_phase_v_rad": wind_phase_v,
        }
        beam = planner.solve(problem, strategy="beam")
        evaluate_hybrid = run_idx < hybrid_eval_runs
        hybrid_result = hybrid.solve(problem, planner=planner) if evaluate_hybrid else None
        use_hybrid = bool(hybrid_result is not None and _accept_aircraft_hybrid(hybrid_result, beam))
        result = hybrid_result if use_hybrid and hybrid_result is not None else beam
        rows.append(
            {
                "run": run_idx,
                "wind_scale": wind_scale,
                "battery_scale": battery_scale,
                "wind_phase_u_rad": wind_phase_u,
                "wind_phase_v_rad": wind_phase_v,
                "hybrid_evaluated": bool(evaluate_hybrid),
                "selected_solver": "hybrid" if use_hybrid else "beam",
                "solved": bool(result.solved),
                "hard_violations": len(result.hard_violations),
                "time_s": float(result.final_state.get("time_s", np.nan)),
                "energy_used_wh": float(result.final_state.get("total_energy_used_wh", np.nan)),
                "energy_remaining_wh": float(result.final_state.get("energy_wh", np.nan)),
                "objective_score": float(result.objective_score),
                "beam_solved": bool(beam.solved),
                "beam_time_s": float(beam.final_state.get("time_s", np.nan)),
                "beam_energy_used_wh": float(beam.final_state.get("total_energy_used_wh", np.nan)),
                "beam_objective_score": float(beam.objective_score),
                "hybrid_solved": bool(hybrid_result.solved) if hybrid_result is not None else False,
                "hybrid_time_s": float(hybrid_result.final_state.get("time_s", np.nan)) if hybrid_result is not None else float("nan"),
                "hybrid_energy_used_wh": (
                    float(hybrid_result.final_state.get("total_energy_used_wh", np.nan)) if hybrid_result is not None else float("nan")
                ),
                "hybrid_objective_score": float(hybrid_result.objective_score) if hybrid_result is not None else float("nan"),
            }
        )

    return rows


def _spacecraft_monte_carlo(
    spacecraft_cfg: Dict[str, Any],
    n_runs: int,
    hybrid_eval_runs: int,
    timing_perturbation_runs: int,
    rng: np.random.Generator,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    planner = UnifiedPlanner()
    hybrid = UnifiedHybridOCPEngine(
        {
            "control_iterations": 1,
            "max_sequence_candidates": 4,
            "max_mutations_per_seed": 6,
            "max_control_nodes_spacecraft": 14,
        }
    )
    base_cfg = _small_spacecraft_cfg(spacecraft_cfg)
    base_solar_w = float(base_cfg["solar_charge_w"])
    min_battery_wh = float(base_cfg["battery_min_wh"])
    max_battery_wh = float(base_cfg["battery_capacity_wh"])

    for run_idx in range(n_runs):
        cfg_small = json.loads(json.dumps(base_cfg))
        battery_initial_wh = float(
            np.clip(
                rng.normal(loc=base_cfg["battery_initial_wh"], scale=10.0),
                0.8 * base_cfg["battery_initial_wh"],
                1.15 * base_cfg["battery_initial_wh"],
            )
        )
        solar_charge_w = float(
            np.clip(
                rng.normal(loc=base_cfg["solar_charge_w"], scale=4.0),
                0.75 * base_cfg["solar_charge_w"],
                1.25 * base_cfg["solar_charge_w"],
            )
        )
        # Perturb contact timing only on a subset of runs to keep validation runtime practical.
        if run_idx < timing_perturbation_runs:
            epoch_shift_deg = float(rng.uniform(-20.0, 20.0))
        else:
            epoch_shift_deg = 0.0
        cfg_small["epoch_theta0_deg"] = float(base_cfg.get("epoch_theta0_deg", 0.0)) + epoch_shift_deg
        cfg_small["solar_charge_w"] = solar_charge_w
        cfg_small["battery_initial_wh"] = battery_initial_wh

        mc_planner_overrides = (
            {"beam_width": 14, "max_expansions": 7000, "candidate_limit": 10, "max_depth": 200}
            if n_runs >= 10
            else {"beam_width": 18, "max_expansions": 20000, "candidate_limit": 12, "max_depth": 260}
        )
        problem = build_spacecraft_problem(cfg_small, planner_overrides=mc_planner_overrides)
        base_initial = dict(problem.initial_state)

        battery_initial_wh = float(np.clip(battery_initial_wh, min_battery_wh, max_battery_wh))
        solar_scale = max(0.5, solar_charge_w / max(1e-6, base_solar_w))
        problem.initial_state = {
            **base_initial,
            "battery_wh": battery_initial_wh,
            "solar_scale": solar_scale,
        }
        beam = planner.solve(problem, strategy="beam")
        beam_recovery = None
        if (not beam.solved) or beam.hard_violations:
            # Recovery pass: if nominal MC search budget misses terminal closure,
            # rerun only this scenario with a higher search budget.
            recovery_overrides = {
                "beam_width": 32,
                "max_expansions": 28000,
                "candidate_limit": 12,
                "max_depth": 360,
            }
            recovery_problem = build_spacecraft_problem(cfg_small, planner_overrides=recovery_overrides)
            recovery_initial = dict(recovery_problem.initial_state)
            recovery_problem.initial_state = {
                **recovery_initial,
                "battery_wh": battery_initial_wh,
                "solar_scale": solar_scale,
            }
            beam_recovery = planner.solve(recovery_problem, strategy="beam")
            beam_key = (1 if beam.solved else 0, -len(beam.hard_violations), float(beam.objective_score))
            recovery_key = (
                1 if beam_recovery.solved else 0,
                -len(beam_recovery.hard_violations),
                float(beam_recovery.objective_score),
            )
            if recovery_key > beam_key:
                beam = beam_recovery
        evaluate_hybrid = run_idx < hybrid_eval_runs
        hybrid_result = hybrid.solve(problem, planner=planner) if evaluate_hybrid else None
        use_hybrid = bool(hybrid_result is not None and _accept_spacecraft_hybrid(hybrid_result, beam))
        result = hybrid_result if use_hybrid and hybrid_result is not None else beam

        traces.append(
            {
                "run": run_idx,
                "selected_solver": "hybrid" if use_hybrid else "beam",
                "time_s": 0.0,
                "battery_wh": float(problem.initial_state.get("battery_wh", np.nan)),
                "data_buffer_mb": float(problem.initial_state.get("data_buffer_mb", 0.0)),
                "delivered_science": float(problem.initial_state.get("delivered_science", 0.0)),
                "solar_charge_w": solar_charge_w,
                "epoch_shift_deg": epoch_shift_deg,
            }
        )
        for step in result.steps:
            traces.append(
                {
                    "run": run_idx,
                    "selected_solver": "hybrid" if use_hybrid else "beam",
                    "time_s": float(step.end_time_s),
                    "battery_wh": float(step.state_after.get("battery_wh", np.nan)),
                    "data_buffer_mb": float(step.state_after.get("data_buffer_mb", np.nan)),
                    "delivered_science": float(step.state_after.get("delivered_science", np.nan)),
                    "solar_charge_w": solar_charge_w,
                    "epoch_shift_deg": epoch_shift_deg,
                }
            )

        rows.append(
            {
                "run": run_idx,
                "battery_initial_wh": battery_initial_wh,
                "solar_charge_w": solar_charge_w,
                "epoch_shift_deg": epoch_shift_deg,
                "hybrid_evaluated": bool(evaluate_hybrid),
                "selected_solver": "hybrid" if use_hybrid else "beam",
                "solved": bool(result.solved),
                "hard_violations": len(result.hard_violations),
                "delivered_science": float(result.final_state.get("delivered_science", np.nan)),
                "final_battery_wh": float(result.final_state.get("battery_wh", np.nan)),
                "final_data_buffer_mb": float(result.final_state.get("data_buffer_mb", np.nan)),
                "objective_score": float(result.objective_score),
                "beam_solved": bool(beam.solved),
                "beam_delivered_science": float(beam.final_state.get("delivered_science", np.nan)),
                "beam_final_battery_wh": float(beam.final_state.get("battery_wh", np.nan)),
                "beam_objective_score": float(beam.objective_score),
                "beam_recovery_used": bool(beam_recovery is not None),
                "beam_recovery_solved": bool(beam_recovery.solved) if beam_recovery is not None else False,
                "hybrid_solved": bool(hybrid_result.solved) if hybrid_result is not None else False,
                "hybrid_delivered_science": (
                    float(hybrid_result.final_state.get("delivered_science", np.nan)) if hybrid_result is not None else float("nan")
                ),
                "hybrid_final_battery_wh": (
                    float(hybrid_result.final_state.get("battery_wh", np.nan)) if hybrid_result is not None else float("nan")
                ),
                "hybrid_objective_score": float(hybrid_result.objective_score) if hybrid_result is not None else float("nan"),
            }
        )

    return rows, traces


def _stress_tests(
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    limited: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    planner = UnifiedPlanner()
    scenarios: List[Dict[str, Any]] = []
    details: Dict[str, List[Dict[str, Any]]] = {}
    aircraft_problem = build_aircraft_problem(
        aircraft_cfg,
        seed=321,
        wind_scale=1.0,
        battery_scale=1.0,
        planner_overrides={"beam_width": 28, "max_expansions": 8000, "candidate_limit": 9},
    )
    aircraft_base_initial = dict(aircraft_problem.initial_state)
    aircraft_capacity_wh = float(aircraft_cfg["battery_capacity_wh"])

    aircraft_cases = [
        ("aircraft_high_wind", {"wind_scale": 1.30, "battery_scale": 1.0}),
        ("aircraft_low_energy", {"wind_scale": 1.0, "battery_scale": 0.90}),
        ("aircraft_compound", {"wind_scale": 1.20, "battery_scale": 0.92}),
    ]
    if limited:
        aircraft_cases = aircraft_cases[:1]
    for name, params in aircraft_cases:
        case_phase_u = float(0.6 * (len(name) % 5))
        case_phase_v = float(-0.4 * (len(name) % 7))
        aircraft_problem.initial_state = {
            **aircraft_base_initial,
            "wind_scale": float(params["wind_scale"]),
            "energy_wh": min(aircraft_capacity_wh, float(aircraft_base_initial["energy_wh"]) * float(params["battery_scale"])),
            "wind_phase_u_rad": case_phase_u,
            "wind_phase_v_rad": case_phase_v,
        }
        result = planner.solve(aircraft_problem, strategy="beam")
        scenarios.append(
            {
                "scenario": name,
                "domain": "aircraft",
                "solved": bool(result.solved),
                "hard_violations": len(result.hard_violations),
                "objective_score": float(result.objective_score),
                "time_s": float(result.final_state.get("time_s", np.nan)),
                "energy_remaining_wh": float(result.final_state.get("energy_wh", np.nan)),
            }
        )

    spacecraft_cases = [
        ("spacecraft_low_solar", {"solar_scale": 0.85, "battery_scale": 1.0}),
        ("spacecraft_low_battery", {"solar_scale": 1.0, "battery_scale": 0.88}),
        ("spacecraft_compound", {"solar_scale": 0.90, "battery_scale": 0.90}),
    ]
    if limited:
        spacecraft_cases = spacecraft_cases[:1]
    spacecraft_cfg_small = _small_spacecraft_cfg(spacecraft_cfg)
    spacecraft_problem = build_spacecraft_problem(
        spacecraft_cfg_small,
        planner_overrides={"beam_width": 16, "max_expansions": 5000, "candidate_limit": 10, "max_depth": 120},
    )
    spacecraft_base_initial = dict(spacecraft_problem.initial_state)
    spacecraft_base_solar_w = float(spacecraft_cfg_small["solar_charge_w"])
    spacecraft_min_wh = float(spacecraft_cfg_small["battery_min_wh"])
    spacecraft_max_wh = float(spacecraft_cfg_small["battery_capacity_wh"])
    for name, params in spacecraft_cases:
        solar_scale = float(params["solar_scale"])
        battery_wh = float(spacecraft_base_initial["battery_wh"]) * float(params["battery_scale"])
        battery_wh = float(np.clip(battery_wh, spacecraft_min_wh, spacecraft_max_wh))
        spacecraft_problem.initial_state = {
            **spacecraft_base_initial,
            "battery_wh": battery_wh,
            "solar_scale": solar_scale,
        }
        result = planner.solve(spacecraft_problem, strategy="beam")
        scenarios.append(
            {
                "scenario": name,
                "domain": "spacecraft",
                "solved": bool(result.solved),
                "hard_violations": len(result.hard_violations),
                "objective_score": float(result.objective_score),
                "delivered_science": float(result.final_state.get("delivered_science", np.nan)),
                "final_battery_wh": float(result.final_state.get("battery_wh", np.nan)),
                "solar_charge_w": spacecraft_base_solar_w * solar_scale,
            }
        )

        if name == "spacecraft_low_solar":
            downlink_rows: List[Dict[str, Any]] = []
            for step in result.steps:
                task = spacecraft_problem.tasks.get(step.to_task_id)
                if task is None or task.task_type != "downlink":
                    continue
                downlink_rows.append(
                    {
                        "scenario": name,
                        "task_id": task.task_id,
                        "station_id": task.metadata.get("station_id", ""),
                        "start_s": float(step.start_time_s),
                        "end_s": float(step.end_time_s),
                        "duration_s": float(step.end_time_s - step.start_time_s),
                        "min_elevation_deg": float(task.metadata.get("min_elevation_deg", np.nan)),
                        "max_elevation_deg": float(task.metadata.get("max_elevation_deg", np.nan)),
                        "battery_wh": float(step.state_after.get("battery_wh", np.nan)),
                        "data_buffer_mb": float(step.state_after.get("data_buffer_mb", np.nan)),
                    }
                )
            details[name] = downlink_rows

            trace_rows: List[Dict[str, Any]] = [
                {
                    "scenario": name,
                    "time_s": 0.0,
                    "battery_wh": float(spacecraft_problem.initial_state.get("battery_wh", np.nan)),
                    "data_buffer_mb": float(spacecraft_problem.initial_state.get("data_buffer_mb", 0.0)),
                    "delivered_science": float(spacecraft_problem.initial_state.get("delivered_science", 0.0)),
                }
            ]
            for step in result.steps:
                trace_rows.append(
                    {
                        "scenario": name,
                        "time_s": float(step.end_time_s),
                        "battery_wh": float(step.state_after.get("battery_wh", np.nan)),
                        "data_buffer_mb": float(step.state_after.get("data_buffer_mb", np.nan)),
                        "delivered_science": float(step.state_after.get("delivered_science", np.nan)),
                    }
                )
            details["spacecraft_low_solar_trace"] = trace_rows

    return scenarios, details


def _baseline_comparison(
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    use_hybrid: bool = True,
) -> List[Dict[str, Any]]:
    planner = UnifiedPlanner()
    hybrid = (
        UnifiedHybridOCPEngine(
            {
                "control_iterations": 1,
                "max_sequence_candidates": 4,
                "max_mutations_per_seed": 6,
                "max_control_nodes_aircraft": 10,
                "max_control_nodes_spacecraft": 14,
            }
        )
        if use_hybrid
        else None
    )
    rows: List[Dict[str, Any]] = []

    aircraft_problem = build_aircraft_problem(
        aircraft_cfg,
        seed=42,
        planner_overrides={"beam_width": 24, "max_expansions": 7000, "candidate_limit": 8},
    )
    aircraft_beam = planner.solve(aircraft_problem, strategy="beam")
    aircraft_hybrid = hybrid.solve(aircraft_problem, planner=planner) if hybrid is not None else None
    aircraft_best = (
        aircraft_hybrid
        if aircraft_hybrid is not None and _accept_aircraft_hybrid(aircraft_hybrid, aircraft_beam)
        else aircraft_beam
    )
    aircraft_best_label = "hybrid" if aircraft_hybrid is not None and aircraft_best is aircraft_hybrid else "beam"
    aircraft_greedy = planner.solve(aircraft_problem, strategy="greedy")

    rows.append(
        {
            "domain": "aircraft",
            "metric": "objective_score",
            "beam": aircraft_best.objective_score,
            "greedy": aircraft_greedy.objective_score,
            "beam_solved": bool(aircraft_best.solved),
            "greedy_solved": bool(aircraft_greedy.solved),
            "improvement_pct": 100.0 * (aircraft_best.objective_score - aircraft_greedy.objective_score) / max(1e-6, abs(aircraft_greedy.objective_score)),
            "ours_strategy": aircraft_best_label,
        }
    )
    aircraft_time_beam = float(aircraft_best.final_state.get("time_s", np.nan))
    aircraft_time_greedy = float(aircraft_greedy.final_state.get("time_s", np.nan))
    comparable_time = bool(aircraft_best.solved) and bool(aircraft_greedy.solved)
    rows.append(
        {
            "domain": "aircraft",
            "metric": "mission_time_s",
            "beam": aircraft_time_beam,
            "greedy": aircraft_time_greedy,
            "beam_solved": bool(aircraft_best.solved),
            "greedy_solved": bool(aircraft_greedy.solved),
            "improvement_pct": (
                100.0 * (aircraft_time_greedy - aircraft_time_beam) / max(1e-6, abs(aircraft_time_greedy))
                if comparable_time
                else None
            ),
            "note": "" if comparable_time else "not_comparable_due_to_infeasible_baseline",
            "ours_strategy": aircraft_best_label,
        }
    )

    spacecraft_cfg_small = _small_spacecraft_cfg(spacecraft_cfg)
    spacecraft_problem = build_spacecraft_problem(
        spacecraft_cfg_small,
        planner_overrides={"beam_width": 16, "max_expansions": 5000, "candidate_limit": 10, "max_depth": 120},
    )
    spacecraft_beam = planner.solve(spacecraft_problem, strategy="beam")
    spacecraft_hybrid = hybrid.solve(spacecraft_problem, planner=planner) if hybrid is not None else None
    spacecraft_best = (
        spacecraft_hybrid
        if spacecraft_hybrid is not None and _accept_spacecraft_hybrid(spacecraft_hybrid, spacecraft_beam)
        else spacecraft_beam
    )
    spacecraft_best_label = "hybrid" if spacecraft_hybrid is not None and spacecraft_best is spacecraft_hybrid else "beam"
    spacecraft_greedy = planner.solve(spacecraft_problem, strategy="greedy")

    rows.append(
        {
            "domain": "spacecraft",
            "metric": "objective_score",
            "beam": spacecraft_best.objective_score,
            "greedy": spacecraft_greedy.objective_score,
            "beam_solved": bool(spacecraft_best.solved),
            "greedy_solved": bool(spacecraft_greedy.solved),
            "improvement_pct": 100.0 * (spacecraft_best.objective_score - spacecraft_greedy.objective_score) / max(1e-6, abs(spacecraft_greedy.objective_score)),
            "ours_strategy": spacecraft_best_label,
        }
    )
    rows.append(
        {
            "domain": "spacecraft",
            "metric": "delivered_science",
            "beam": float(spacecraft_best.final_state.get("delivered_science", np.nan)),
            "greedy": float(spacecraft_greedy.final_state.get("delivered_science", np.nan)),
            "beam_solved": bool(spacecraft_best.solved),
            "greedy_solved": bool(spacecraft_greedy.solved),
            "improvement_pct": 100.0
            * (
                float(spacecraft_best.final_state.get("delivered_science", np.nan))
                - float(spacecraft_greedy.final_state.get("delivered_science", np.nan))
            )
            / max(1e-6, abs(float(spacecraft_greedy.final_state.get("delivered_science", np.nan)))),
            "ours_strategy": spacecraft_best_label,
        }
    )

    return rows


def _plot_validation(
    aircraft_mc: List[Dict[str, Any]],
    spacecraft_mc: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    stress_rows: List[Dict[str, Any]],
    stress_details: Dict[str, List[Dict[str, Any]]],
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    return plot_validation_bundle(
        aircraft_mc=aircraft_mc,
        spacecraft_mc=spacecraft_mc,
        baseline_rows=baseline_rows,
        stress_rows=stress_rows,
        stress_details=stress_details,
        aircraft_cfg=aircraft_cfg,
        spacecraft_cfg=spacecraft_cfg,
        output_dir=output_dir,
    )


def _safe_mean(values: List[float]) -> float | None:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _safe_std(values: List[float]) -> float | None:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.std())


def _safe_quantile(values: List[float], q: float) -> float | None:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def _safe_min(values: List[float]) -> float | None:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.min(arr))


def _plan_signature(result: Any) -> str:
    serializable = {
        "solved": bool(result.solved),
        "status": str(result.status),
        "sequence": [str(x) for x in result.sequence],
        "objective_score": round(float(result.objective_score), 8),
        "steps": [
            {
                "from": str(step.from_task_id),
                "to": str(step.to_task_id),
                "start_s": round(float(step.start_time_s), 5),
                "end_s": round(float(step.end_time_s), 5),
            }
            for step in result.steps
        ],
        "final_state": {
            "time_s": round(float(result.final_state.get("time_s", 0.0)), 5),
            "energy_wh": round(float(result.final_state.get("energy_wh", np.nan)), 5),
            "battery_wh": round(float(result.final_state.get("battery_wh", np.nan)), 5),
            "data_buffer_mb": round(float(result.final_state.get("data_buffer_mb", np.nan)), 5),
            "delivered_science": round(float(result.final_state.get("delivered_science", np.nan)), 5),
        },
    }
    payload = json.dumps(serializable, sort_keys=True, allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_replay_check(
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    planner = UnifiedPlanner()

    aircraft_problem_1 = build_aircraft_problem(
        aircraft_cfg,
        seed=777,
        planner_overrides={"beam_width": 28, "max_expansions": 7000, "candidate_limit": 9},
    )
    aircraft_problem_2 = build_aircraft_problem(
        aircraft_cfg,
        seed=777,
        planner_overrides={"beam_width": 28, "max_expansions": 7000, "candidate_limit": 9},
    )
    a1 = planner.solve(aircraft_problem_1, strategy="beam")
    a2 = planner.solve(aircraft_problem_2, strategy="beam")
    a_match = _plan_signature(a1) == _plan_signature(a2)

    spacecraft_problem_1 = build_spacecraft_problem(
        _small_spacecraft_cfg(spacecraft_cfg),
        planner_overrides={"beam_width": 16, "max_expansions": 5000, "candidate_limit": 10, "max_depth": 120},
    )
    spacecraft_problem_2 = build_spacecraft_problem(
        _small_spacecraft_cfg(spacecraft_cfg),
        planner_overrides={"beam_width": 16, "max_expansions": 5000, "candidate_limit": 10, "max_depth": 120},
    )
    s1 = planner.solve(spacecraft_problem_1, strategy="beam")
    s2 = planner.solve(spacecraft_problem_2, strategy="beam")
    s_match = _plan_signature(s1) == _plan_signature(s2)

    return {
        "all_deterministic": bool(a_match and s_match),
        "aircraft": {
            "hash_run_1": _plan_signature(a1),
            "hash_run_2": _plan_signature(a2),
            "deterministic_match": bool(a_match),
            "objective_run_1": float(a1.objective_score),
            "objective_run_2": float(a2.objective_score),
        },
        "spacecraft": {
            "hash_run_1": _plan_signature(s1),
            "hash_run_2": _plan_signature(s2),
            "deterministic_match": bool(s_match),
            "objective_run_1": float(s1.objective_score),
            "objective_run_2": float(s2.objective_score),
        },
    }


def _mark_pareto_front_minimize(
    rows: List[Dict[str, Any]],
    metric_x: str,
    metric_y: str,
) -> List[Dict[str, Any]]:
    feasible_idx = [
        i
        for i, r in enumerate(rows)
        if bool(r.get("solved", False))
        and int(r.get("hard_violations", 0)) == 0
        and np.isfinite(float(r.get(metric_x, np.nan)))
        and np.isfinite(float(r.get(metric_y, np.nan)))
    ]
    pareto = set(feasible_idx)
    for i in feasible_idx:
        xi = float(rows[i][metric_x])
        yi = float(rows[i][metric_y])
        for j in feasible_idx:
            if i == j:
                continue
            xj = float(rows[j][metric_x])
            yj = float(rows[j][metric_y])
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                if i in pareto:
                    pareto.remove(i)
                break
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out.append({**row, "is_pareto": bool(idx in pareto)})
    return out


def _mark_pareto_front_maximize(
    rows: List[Dict[str, Any]],
    metric_x: str,
    metric_y: str,
) -> List[Dict[str, Any]]:
    feasible_idx = [
        i
        for i, r in enumerate(rows)
        if bool(r.get("solved", False))
        and int(r.get("hard_violations", 0)) == 0
        and np.isfinite(float(r.get(metric_x, np.nan)))
        and np.isfinite(float(r.get(metric_y, np.nan)))
    ]
    pareto = set(feasible_idx)
    for i in feasible_idx:
        xi = float(rows[i][metric_x])
        yi = float(rows[i][metric_y])
        for j in feasible_idx:
            if i == j:
                continue
            xj = float(rows[j][metric_x])
            yj = float(rows[j][metric_y])
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                if i in pareto:
                    pareto.remove(i)
                break
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out.append({**row, "is_pareto": bool(idx in pareto)})
    return out


def _aircraft_pareto_sweep(aircraft_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    planner = UnifiedPlanner()
    sweeps = [
        (0.012, 0.040),
        (0.016, 0.055),
        (0.020, 0.070),
        (0.026, 0.090),
        (0.033, 0.115),
        (0.042, 0.145),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (time_w, energy_w) in enumerate(sweeps):
        cfg_local = json.loads(json.dumps(aircraft_cfg))
        cfg_local.setdefault("objective", {})
        cfg_local["objective"]["time_weight"] = float(time_w)
        cfg_local["objective"]["energy_weight"] = float(energy_w)
        problem = build_aircraft_problem(
            cfg_local,
            seed=300 + idx,
            planner_overrides={"beam_width": 22, "max_expansions": 6000, "candidate_limit": 8},
        )
        result = planner.solve(problem, strategy="beam")
        rows.append(
            {
                "scenario": f"aircraft_tradeoff_{idx}",
                "time_weight": float(time_w),
                "energy_weight": float(energy_w),
                "solved": bool(result.solved),
                "hard_violations": int(len(result.hard_violations)),
                "objective_score": float(result.objective_score),
                "mission_time_s": float(result.final_state.get("time_s", np.nan)),
                "energy_used_wh": float(result.final_state.get("total_energy_used_wh", np.nan)),
                "energy_remaining_wh": float(result.final_state.get("energy_wh", np.nan)),
            }
        )
    return _mark_pareto_front_minimize(rows, "mission_time_s", "energy_used_wh")


def _spacecraft_pareto_sweep(spacecraft_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    planner = UnifiedPlanner()
    cfg_base = _small_spacecraft_cfg(spacecraft_cfg)
    sweeps = [
        (12.0, 0.010),
        (15.0, 0.020),
        (18.0, 0.030),
        (21.0, 0.045),
        (24.0, 0.060),
        (28.0, 0.080),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (delivered_w, energy_w) in enumerate(sweeps):
        cfg_local = json.loads(json.dumps(cfg_base))
        cfg_local.setdefault("objective", {})
        cfg_local["objective"]["delivered_weight"] = float(delivered_w)
        cfg_local["objective"]["energy_weight"] = float(energy_w)
        problem = build_spacecraft_problem(
            cfg_local,
            planner_overrides={"beam_width": 14, "max_expansions": 4500, "candidate_limit": 9, "max_depth": 110},
        )
        result = planner.solve(problem, strategy="beam")
        rows.append(
            {
                "scenario": f"spacecraft_tradeoff_{idx}",
                "delivered_weight": float(delivered_w),
                "energy_weight": float(energy_w),
                "solved": bool(result.solved),
                "hard_violations": int(len(result.hard_violations)),
                "objective_score": float(result.objective_score),
                "delivered_science": float(result.final_state.get("delivered_science", np.nan)),
                "final_battery_wh": float(result.final_state.get("battery_wh", np.nan)),
                "final_data_buffer_mb": float(result.final_state.get("data_buffer_mb", np.nan)),
            }
        )
    return _mark_pareto_front_maximize(rows, "delivered_science", "final_battery_wh")


def _plot_pareto_frontiers(
    aircraft_pareto_rows: List[Dict[str, Any]],
    spacecraft_pareto_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    valid_a = [r for r in aircraft_pareto_rows if np.isfinite(float(r.get("mission_time_s", np.nan)))]
    if valid_a:
        x = [float(r["mission_time_s"]) for r in valid_a]
        y = [float(r["energy_used_wh"]) for r in valid_a]
        ax_a.scatter(x, y, color="tab:blue", alpha=0.55, label="tradeoff sweep")
        pareto = [r for r in valid_a if bool(r.get("is_pareto", False))]
        if pareto:
            px = [float(r["mission_time_s"]) for r in pareto]
            py = [float(r["energy_used_wh"]) for r in pareto]
            order = np.argsort(np.array(px))
            ax_a.plot(np.array(px)[order], np.array(py)[order], color="tab:red", linewidth=2.0, label="pareto frontier")
            ax_a.scatter(px, py, color="tab:red", s=50)
    ax_a.set_xlabel("Mission Time [s] (lower is better)")
    ax_a.set_ylabel("Energy Used [Wh] (lower is better)")
    ax_a.set_title("Aircraft Pareto Frontier")
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="best")
    fig_a.tight_layout()
    path_a = output_dir / "aircraft_pareto_frontier.png"
    fig_a.savefig(path_a, dpi=180)
    plt.close(fig_a)
    paths["aircraft_pareto_frontier"] = str(path_a)

    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    valid_s = [r for r in spacecraft_pareto_rows if np.isfinite(float(r.get("delivered_science", np.nan)))]
    if valid_s:
        x = [float(r["delivered_science"]) for r in valid_s]
        y = [float(r["final_battery_wh"]) for r in valid_s]
        ax_s.scatter(x, y, color="tab:green", alpha=0.55, label="tradeoff sweep")
        pareto = [r for r in valid_s if bool(r.get("is_pareto", False))]
        if pareto:
            px = [float(r["delivered_science"]) for r in pareto]
            py = [float(r["final_battery_wh"]) for r in pareto]
            order = np.argsort(-np.array(px))
            ax_s.plot(np.array(px)[order], np.array(py)[order], color="tab:orange", linewidth=2.0, label="pareto frontier")
            ax_s.scatter(px, py, color="tab:orange", s=50)
    ax_s.set_xlabel("Delivered Science (higher is better)")
    ax_s.set_ylabel("Final Battery [Wh] (higher is better)")
    ax_s.set_title("Spacecraft Pareto Frontier")
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="best")
    fig_s.tight_layout()
    path_s = output_dir / "spacecraft_pareto_frontier.png"
    fig_s.savefig(path_s, dpi=180)
    plt.close(fig_s)
    paths["spacecraft_pareto_frontier"] = str(path_s)

    return paths


def _find_reference(baseline_rows: List[Dict[str, Any]], domain: str, metric: str, fallback: float) -> float:
    for row in baseline_rows:
        if str(row.get("domain")) == domain and str(row.get("metric")) == metric:
            try:
                return float(row.get("beam", fallback))
            except Exception:
                return fallback
    return fallback


def run_validation(
    aircraft_cfg: Dict[str, Any],
    spacecraft_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(validation_cfg.get("mode", "fast")).lower()
    limited = mode != "full"

    rng = np.random.default_rng(int(validation_cfg.get("seed", 42)))
    aircraft_runs = int(validation_cfg.get("aircraft_monte_carlo_runs", 12 if limited else 50))
    spacecraft_runs = int(validation_cfg.get("spacecraft_monte_carlo_runs", 6 if limited else 50))
    aircraft_hybrid_eval_runs = int(
        validation_cfg.get("aircraft_hybrid_eval_runs", min(8 if limited else 12, aircraft_runs))
    )
    spacecraft_hybrid_eval_runs = int(
        validation_cfg.get("spacecraft_hybrid_eval_runs", min(6 if limited else 10, spacecraft_runs))
    )
    spacecraft_timing_perturbation_runs = int(
        validation_cfg.get("spacecraft_timing_perturbation_runs", min(2 if limited else 5, spacecraft_runs))
    )
    aircraft_hybrid_eval_runs = max(0, min(aircraft_runs, aircraft_hybrid_eval_runs))
    spacecraft_hybrid_eval_runs = max(0, min(spacecraft_runs, spacecraft_hybrid_eval_runs))
    spacecraft_timing_perturbation_runs = max(0, min(spacecraft_runs, spacecraft_timing_perturbation_runs))

    aircraft_mc = _aircraft_monte_carlo(aircraft_cfg, aircraft_runs, aircraft_hybrid_eval_runs, rng)
    spacecraft_mc, spacecraft_mc_traces = _spacecraft_monte_carlo(
        spacecraft_cfg,
        spacecraft_runs,
        spacecraft_hybrid_eval_runs,
        spacecraft_timing_perturbation_runs,
        rng,
    )
    baseline_rows = _baseline_comparison(
        aircraft_cfg,
        spacecraft_cfg,
        use_hybrid=bool(validation_cfg.get("baseline_use_hybrid", not limited)),
    )
    stress_rows, stress_details = _stress_tests(aircraft_cfg, spacecraft_cfg, limited=limited)
    aircraft_pareto_rows = _aircraft_pareto_sweep(aircraft_cfg)
    spacecraft_pareto_rows = _spacecraft_pareto_sweep(spacecraft_cfg)
    replay = _deterministic_replay_check(aircraft_cfg, spacecraft_cfg)

    aircraft_mc_path = output_dir / "aircraft_monte_carlo.csv"
    spacecraft_mc_path = output_dir / "spacecraft_monte_carlo.csv"
    baseline_path = output_dir / "baseline_comparison.csv"
    stress_path = output_dir / "stress_tests.csv"
    spacecraft_trace_path = output_dir / "spacecraft_monte_carlo_traces.csv"
    aircraft_pareto_path = output_dir / "aircraft_pareto_trade_study.csv"
    spacecraft_pareto_path = output_dir / "spacecraft_pareto_trade_study.csv"
    replay_path = output_dir / "deterministic_replay.json"

    write_csv(aircraft_mc_path, aircraft_mc)
    write_csv(spacecraft_mc_path, spacecraft_mc)
    write_csv(baseline_path, baseline_rows)
    write_csv(stress_path, stress_rows)
    write_csv(aircraft_pareto_path, aircraft_pareto_rows)
    write_csv(spacecraft_pareto_path, spacecraft_pareto_rows)
    replay_path.write_text(json.dumps(replay, indent=2))
    if spacecraft_mc_traces:
        write_csv(spacecraft_trace_path, spacecraft_mc_traces)
    stress_downlink_rows = stress_details.get("spacecraft_low_solar", [])
    stress_downlink_path: Path | None = None
    if stress_downlink_rows:
        stress_downlink_path = output_dir / "stress_downlink_windows.csv"
        write_csv(stress_downlink_path, stress_downlink_rows)
    stress_trace_rows = stress_details.get("spacecraft_low_solar_trace", [])
    stress_trace_path: Path | None = None
    if stress_trace_rows:
        stress_trace_path = output_dir / "stress_spacecraft_low_solar_trace.csv"
        write_csv(stress_trace_path, stress_trace_rows)

    plots = _plot_validation(
        aircraft_mc,
        spacecraft_mc,
        baseline_rows,
        stress_rows,
        stress_details,
        aircraft_cfg,
        spacecraft_cfg,
        output_dir,
    )
    plots.update(_plot_pareto_frontiers(aircraft_pareto_rows, spacecraft_pareto_rows, output_dir))

    aircraft_success = [bool(r["solved"]) and int(r["hard_violations"]) == 0 for r in aircraft_mc]
    spacecraft_success = [bool(r["solved"]) and int(r["hard_violations"]) == 0 for r in spacecraft_mc]

    aircraft_times = [float(r["time_s"]) for r in aircraft_mc if bool(r["solved"])]
    spacecraft_science = [float(r["delivered_science"]) for r in spacecraft_mc if bool(r["solved"])]

    aircraft_ref_time = _find_reference(baseline_rows, "aircraft", "mission_time_s", _safe_mean(aircraft_times) or 0.0)
    aircraft_ref_energy = _safe_mean([float(r["energy_used_wh"]) for r in aircraft_mc if bool(r["solved"])]) or 0.0
    spacecraft_ref_science = _find_reference(
        baseline_rows,
        "spacecraft",
        "delivered_science",
        _safe_mean(spacecraft_science) or 0.0,
    )
    spacecraft_ref_battery = _safe_mean([float(r["final_battery_wh"]) for r in spacecraft_mc if bool(r["solved"])]) or 0.0

    max_dev_rows: List[Dict[str, Any]] = [
        {
            "domain": "aircraft",
            "metric": "mission_time_s",
            "reference_value": aircraft_ref_time,
            "max_abs_deviation": max(
                [abs(float(r["time_s"]) - aircraft_ref_time) for r in aircraft_mc if bool(r["solved"]) and np.isfinite(float(r["time_s"]))],
                default=float("nan"),
            ),
        },
        {
            "domain": "aircraft",
            "metric": "energy_used_wh",
            "reference_value": aircraft_ref_energy,
            "max_abs_deviation": max(
                [
                    abs(float(r["energy_used_wh"]) - aircraft_ref_energy)
                    for r in aircraft_mc
                    if bool(r["solved"]) and np.isfinite(float(r["energy_used_wh"]))
                ],
                default=float("nan"),
            ),
        },
        {
            "domain": "spacecraft",
            "metric": "delivered_science",
            "reference_value": spacecraft_ref_science,
            "max_abs_deviation": max(
                [
                    abs(float(r["delivered_science"]) - spacecraft_ref_science)
                    for r in spacecraft_mc
                    if bool(r["solved"]) and np.isfinite(float(r["delivered_science"]))
                ],
                default=float("nan"),
            ),
        },
        {
            "domain": "spacecraft",
            "metric": "final_battery_wh",
            "reference_value": spacecraft_ref_battery,
            "max_abs_deviation": max(
                [
                    abs(float(r["final_battery_wh"]) - spacecraft_ref_battery)
                    for r in spacecraft_mc
                    if bool(r["solved"]) and np.isfinite(float(r["final_battery_wh"]))
                ],
                default=float("nan"),
            ),
        },
    ]
    max_dev_path = output_dir / "robustness_max_deviation.csv"
    write_csv(max_dev_path, max_dev_rows)

    summary = {
        "mode": mode,
        "aircraft_runs": int(len(aircraft_mc)),
        "spacecraft_runs": int(len(spacecraft_mc)),
        "aircraft_hybrid_eval_runs": int(aircraft_hybrid_eval_runs),
        "spacecraft_hybrid_eval_runs": int(spacecraft_hybrid_eval_runs),
        "spacecraft_timing_perturbation_runs": int(spacecraft_timing_perturbation_runs),
        "aircraft_success_rate": float(np.mean(aircraft_success)) if aircraft_success else 0.0,
        "spacecraft_success_rate": float(np.mean(spacecraft_success)) if spacecraft_success else 0.0,
        "aircraft_time_mean_s": _safe_mean(aircraft_times),
        "aircraft_time_std_s": _safe_std(aircraft_times),
        "aircraft_time_p05_s": _safe_quantile(aircraft_times, 0.05),
        "aircraft_time_p50_s": _safe_quantile(aircraft_times, 0.50),
        "aircraft_time_p95_s": _safe_quantile(aircraft_times, 0.95),
        "spacecraft_delivered_science_mean": _safe_mean(spacecraft_science),
        "spacecraft_delivered_science_std": _safe_std(spacecraft_science),
        "spacecraft_delivered_science_p05": _safe_quantile(spacecraft_science, 0.05),
        "spacecraft_delivered_science_p50": _safe_quantile(spacecraft_science, 0.50),
        "spacecraft_delivered_science_p95": _safe_quantile(spacecraft_science, 0.95),
        "aircraft_worst_energy_remaining_wh": _safe_min(
            [float(r["energy_remaining_wh"]) for r in aircraft_mc if bool(r["solved"])]
        ),
        "spacecraft_worst_final_battery_wh": _safe_quantile(
            [float(r["final_battery_wh"]) for r in spacecraft_mc if bool(r["solved"])], 0.0
        ),
        "baseline_improvements": baseline_rows,
        "aircraft_pareto_trade_study": aircraft_pareto_rows,
        "spacecraft_pareto_trade_study": spacecraft_pareto_rows,
        "deterministic_replay": replay,
        "robustness_max_deviation": max_dev_rows,
        "stress_tests": stress_rows,
        "stress_downlink_windows": stress_downlink_rows,
        "plots": plots,
    }

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "aircraft_mc_csv": str(aircraft_mc_path),
        "spacecraft_mc_csv": str(spacecraft_mc_path),
        "baseline_csv": str(baseline_path),
        "stress_csv": str(stress_path),
        "aircraft_pareto_csv": str(aircraft_pareto_path),
        "spacecraft_pareto_csv": str(spacecraft_pareto_path),
        "deterministic_replay_json": str(replay_path),
        "stress_downlink_csv": str(stress_downlink_path) if stress_downlink_path else None,
        "stress_trace_csv": str(stress_trace_path) if stress_trace_path else None,
        "spacecraft_mc_trace_csv": str(spacecraft_trace_path) if spacecraft_mc_traces else None,
        "robustness_max_deviation_csv": str(max_dev_path),
        "summary_json": str(summary_path),
        "summary": summary,
        "plots": plots,
    }
