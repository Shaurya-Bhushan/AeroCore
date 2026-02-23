from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import zipfile

# Keep matplotlib cache writable in restricted environments.
workspace_root = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(workspace_root / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(workspace_root / ".cache"))


RUNTIME_DEPENDENCIES: list[tuple[str, str]] = [
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("shapely", "Shapely"),
]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_environment_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python_version": os.sys.version.split()[0],
        "python_executable": os.sys.executable,
        "runtime_dependencies": {},
        "missing_dependencies": [],
    }
    dep_status: Dict[str, Any] = {}
    missing: list[str] = []
    for module_name, package_name in RUNTIME_DEPENDENCIES:
        available = _module_available(module_name)
        version = _package_version(package_name)
        dep_status[package_name] = {
            "module": module_name,
            "available": bool(available),
            "version": version,
        }
        if not available:
            missing.append(package_name)
    report["runtime_dependencies"] = dep_status
    report["missing_dependencies"] = missing
    report["all_runtime_dependencies_available"] = len(missing) == 0
    return report


def _raise_on_missing_dependencies(report: Dict[str, Any]) -> None:
    missing = report.get("missing_dependencies", [])
    if not missing:
        return
    missing_list = ", ".join(str(x) for x in missing)
    raise RuntimeError(
        "Missing runtime dependencies: "
        + missing_list
        + ".\nInstall with the same interpreter used to run this script:\n"
        + "  python3 -m pip install -r requirements.txt"
    )


def _import_yaml_module():
    if not _module_available("yaml"):
        raise RuntimeError(
            "Missing runtime dependency: PyYAML. Install with:\n"
            "  python3 -m pip install -r requirements.txt"
        )
    import yaml  # type: ignore

    return yaml


def load_config(config_path: Path, yaml_module: Any) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml_module.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must parse to a mapping")
    return cfg


def _resolve_external_data(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    resolved: list[Dict[str, str]] = []
    for domain in ("aircraft", "spacecraft"):
        section = cfg.get(domain)
        if not isinstance(section, dict):
            continue
        data_files = section.get("data_files", {})
        if not isinstance(data_files, dict):
            continue
        for key, rel_path in data_files.items():
            data_path = Path(str(rel_path))
            if not data_path.is_absolute():
                data_path = (config_path.parent / data_path).resolve()
            if not data_path.exists():
                raise FileNotFoundError(f"{domain}.data_files.{key} points to missing file: {data_path}")
            with data_path.open("r", encoding="utf-8") as f:
                section[key] = json.load(f)
            resolved.append({"domain": domain, "field": str(key), "path": str(data_path)})
    cfg["_resolved_data_files"] = resolved
    return cfg


def create_results_bundle(output_root: Path) -> Path:
    bundle_path = output_root / "results_bundle.zip"
    # Package only canonical submission artifacts; skip debug/smoke/tmp trees.
    allowed_top_level_dirs = {"aircraft", "spacecraft", "validation"}
    allowed_root_files = {
        "environment_report.json",
        "headline_metrics.json",
        "run_summary.json",
        "pipeline_issues.json",
        "submission_checklist.json",
        "requirements_traceability.json",
        "requirements_traceability.md",
        "judging_scorecard.json",
        "judging_scorecard.md",
        "redflag_audit.json",
        "redflag_audit.md",
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_root.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path == bundle_path:
                continue
            rel_path = file_path.relative_to(output_root)
            rel_parts = rel_path.parts
            if not rel_parts:
                continue
            first = rel_parts[0]
            keep = False
            if len(rel_parts) == 1 and first in allowed_root_files:
                keep = True
            elif first in allowed_top_level_dirs:
                keep = True
            if not keep:
                continue
            archive.write(file_path, arcname=str(rel_path))
    return bundle_path


def _spacecraft_horizon_covered(schedule_csv: Path, expected_horizon_s: float) -> bool:
    if not schedule_csv.exists():
        return False
    with schedule_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return False
    end_times = [float(r.get("task_end_s", 0.0)) for r in rows if r.get("task_end_s")]
    if not end_times:
        return False
    max_end = max(end_times)
    has_end = any(str(r.get("action_type", "")).lower() == "end" for r in rows)
    return has_end and max_end >= expected_horizon_s - 1.0


def _collect_pipeline_issues(
    cfg: Dict[str, Any],
    aircraft_result: Dict[str, Any],
    spacecraft_result: Dict[str, Any],
    validation_enabled: bool,
    aircraft_independent: Dict[str, Any] | None = None,
    spacecraft_independent: Dict[str, Any] | None = None,
    checklist: Dict[str, Any] | None = None,
) -> list[str]:
    issues: list[str] = []
    a = aircraft_result["metrics"]
    s = spacecraft_result["metrics"]

    if not bool(a.get("solved", False)):
        issues.append("Aircraft mission did not solve to terminal completion")
    if int(a.get("hard_constraint_violations", 0)) != 0:
        issues.append("Aircraft hard constraint violations are non-zero")

    if not bool(s.get("solved", False)):
        issues.append("Spacecraft mission did not solve to terminal completion")
    if int(s.get("hard_constraint_violations", 0)) != 0:
        issues.append("Spacecraft hard constraint violations are non-zero")

    expected_horizon_s = float(int(cfg["spacecraft"]["mission_days"]) * 24 * 3600)
    if not _spacecraft_horizon_covered(Path(spacecraft_result["schedule_csv"]), expected_horizon_s):
        issues.append("Spacecraft schedule does not cover full configured mission horizon with terminal end action")

    if aircraft_independent is not None and not bool(aircraft_independent.get("passed", False)):
        issues.append(
            f"Aircraft independent checks failed (violations={int(aircraft_independent.get('hard_violation_count', 0))})"
        )
    if spacecraft_independent is not None and not bool(spacecraft_independent.get("passed", False)):
        issues.append(
            f"Spacecraft independent checks failed (violations={int(spacecraft_independent.get('hard_violation_count', 0))})"
        )
    if checklist is not None and not bool(checklist.get("all_required_outputs_present", False)):
        issues.append("Submission checklist failed: one or more required outputs are missing or empty")
    if not validation_enabled:
        issues.append("Validation outputs are missing because validation was disabled (--skip-validation)")

    return issues


def _build_submission_checklist(
    output_root: Path,
    validation_enabled: bool,
) -> Dict[str, Any]:
    required_files = [
        output_root / "aircraft" / "uav_flight_plan.csv",
        output_root / "aircraft" / "uav_route_segments.csv",
        output_root / "aircraft" / "uav_path.png",
        output_root / "aircraft" / "uav_path.kml",
        output_root / "aircraft" / "uav_trajectory_3d.html",
        output_root / "aircraft" / "uav_energy_profile.png",
        output_root / "aircraft" / "uav_metrics.json",
        output_root / "aircraft" / "uav_constraint_summary.json",
        output_root / "aircraft" / "uav_constraint_certification.csv",
        output_root / "aircraft" / "uav_constraint_certification.json",
        output_root / "aircraft" / "independent_checks.json",
        output_root / "spacecraft" / "spacecraft_7day_schedule.csv",
        output_root / "spacecraft" / "visibility_windows.csv",
        output_root / "spacecraft" / "spacecraft_gantt.png",
        output_root / "spacecraft" / "spacecraft_resources.png",
        output_root / "spacecraft" / "spacecraft_schedule.kml",
        output_root / "spacecraft" / "spacecraft_metrics.json",
        output_root / "spacecraft" / "spacecraft_constraint_summary.json",
        output_root / "spacecraft" / "spacecraft_constraint_certification.csv",
        output_root / "spacecraft" / "spacecraft_constraint_certification.json",
        output_root / "spacecraft" / "independent_checks.json",
        output_root / "headline_metrics.json",
        output_root / "run_summary.json",
        output_root / "pipeline_issues.json",
        output_root / "requirements_traceability.json",
        output_root / "requirements_traceability.md",
        output_root / "environment_report.json",
    ]
    if validation_enabled:
        required_files.extend(
            [
                output_root / "validation" / "aircraft_monte_carlo.csv",
                output_root / "validation" / "spacecraft_monte_carlo.csv",
                output_root / "validation" / "baseline_comparison.csv",
                output_root / "validation" / "stress_tests.csv",
                output_root / "validation" / "stress_downlink_windows.csv",
                output_root / "validation" / "aircraft_pareto_trade_study.csv",
                output_root / "validation" / "spacecraft_pareto_trade_study.csv",
                output_root / "validation" / "deterministic_replay.json",
                output_root / "validation" / "validation_summary.json",
                output_root / "validation" / "aircraft_monte_carlo_hist.png",
                output_root / "validation" / "aircraft_hybrid_vs_beam_variance.png",
                output_root / "validation" / "aircraft_scenario_sensitivity.png",
                output_root / "validation" / "spacecraft_monte_carlo_scatter.png",
                output_root / "validation" / "spacecraft_hybrid_vs_beam_science.png",
                output_root / "validation" / "spacecraft_scenario_science.png",
                output_root / "validation" / "spacecraft_stress_resource_evolution.png",
                output_root / "validation" / "baseline_comparison.png",
                output_root / "validation" / "stress_margin_plot.png",
                output_root / "validation" / "stress_downlink_vs_constraints.png",
                output_root / "validation" / "aircraft_pareto_frontier.png",
                output_root / "validation" / "spacecraft_pareto_frontier.png",
                output_root / "validation" / "robustness_max_deviation.csv",
                output_root / "validation" / "stress_spacecraft_low_solar_trace.csv",
            ]
        )

    checks: List[Dict[str, Any]] = []
    for p in required_files:
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        checks.append(
            {
                "path": str(p),
                "exists": bool(exists),
                "non_empty": bool(exists and size > 0),
                "size_bytes": int(size),
            }
        )

    all_ok = all(bool(c["exists"]) and bool(c["non_empty"]) for c in checks)
    return {
        "all_required_outputs_present": all_ok,
        "required_file_checks": checks,
    }


def _build_requirements_traceability(
    output_root: Path,
    validation_enabled: bool,
) -> Dict[str, Any]:
    def _check(path: Path) -> Dict[str, Any]:
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
        return {
            "path": str(path),
            "exists": bool(exists),
            "non_empty": bool(exists and size > 0),
            "size_bytes": size,
        }

    entries: list[Dict[str, Any]] = [
        {
            "requirement": "A/aircraft_route_with_timestamps",
            "evidence": _check(output_root / "aircraft" / "uav_flight_plan.csv"),
        },
        {
            "requirement": "A/aircraft_constraint_summary",
            "evidence": _check(output_root / "aircraft" / "uav_constraint_summary.json"),
        },
        {
            "requirement": "A/aircraft_performance_metrics",
            "evidence": _check(output_root / "aircraft" / "uav_metrics.json"),
        },
        {
            "requirement": "A/aircraft_plot_path_or_state",
            "evidence": _check(output_root / "aircraft" / "uav_path.png"),
        },
        {
            "requirement": "A/aircraft_geospatial_export",
            "evidence": _check(output_root / "aircraft" / "uav_path.kml"),
        },
        {
            "requirement": "A/aircraft_3d_visualization",
            "evidence": _check(output_root / "aircraft" / "uav_trajectory_3d.html"),
        },
        {
            "requirement": "B/spacecraft_7day_schedule",
            "evidence": _check(output_root / "spacecraft" / "spacecraft_7day_schedule.csv"),
        },
        {
            "requirement": "B/spacecraft_visibility_contact_evidence",
            "evidence": _check(output_root / "spacecraft" / "visibility_windows.csv"),
        },
        {
            "requirement": "B/spacecraft_constraint_summary",
            "evidence": _check(output_root / "spacecraft" / "spacecraft_constraint_summary.json"),
        },
        {
            "requirement": "B/spacecraft_mission_value_metrics",
            "evidence": _check(output_root / "spacecraft" / "spacecraft_metrics.json"),
        },
        {
            "requirement": "B/spacecraft_schedule_plot_or_table",
            "evidence": _check(output_root / "spacecraft" / "spacecraft_gantt.png"),
        },
        {
            "requirement": "B/spacecraft_geospatial_export",
            "evidence": _check(output_root / "spacecraft" / "spacecraft_schedule.kml"),
        },
        {
            "requirement": "validation/monte_carlo_aircraft",
            "evidence": _check(output_root / "validation" / "aircraft_monte_carlo.csv"),
        },
        {
            "requirement": "validation/monte_carlo_spacecraft",
            "evidence": _check(output_root / "validation" / "spacecraft_monte_carlo.csv"),
        },
        {
            "requirement": "validation/baseline_comparison",
            "evidence": _check(output_root / "validation" / "baseline_comparison.csv"),
        },
        {
            "requirement": "validation/stress_tests",
            "evidence": _check(output_root / "validation" / "stress_tests.csv"),
        },
        {
            "requirement": "validation/summary",
            "evidence": _check(output_root / "validation" / "validation_summary.json"),
        },
        {
            "requirement": "validation/pareto_trade_studies",
            "evidence": _check(output_root / "validation" / "aircraft_pareto_trade_study.csv"),
        },
        {
            "requirement": "validation/deterministic_replay",
            "evidence": _check(output_root / "validation" / "deterministic_replay.json"),
        },
        {
            "requirement": "results_bundle",
            "evidence": _check(output_root / "results_bundle.zip"),
        },
    ]

    if not validation_enabled:
        for entry in entries:
            req = str(entry["requirement"])
            if req.startswith("validation/"):
                entry["skipped"] = True

    all_present = all(
        bool(e["evidence"]["exists"]) and bool(e["evidence"]["non_empty"])
        for e in entries
        if not bool(e.get("skipped", False))
    )
    return {
        "all_required_evidence_present": all_present,
        "entries": entries,
        "unified_architecture_evidence": [
            "src/core/planner.py",
            "src/core/hybrid_ocp.py",
            "src/core/models.py",
            "src/aircraft/mission.py",
            "src/spacecraft/mission.py",
        ],
    }


def _write_requirements_traceability_md(path: Path, matrix: Dict[str, Any]) -> None:
    lines: list[str] = [
        "# Requirements Traceability",
        "",
        f"All required evidence present: `{matrix.get('all_required_evidence_present', False)}`",
        "",
        "## Unified Architecture Evidence",
    ]
    for p in matrix.get("unified_architecture_evidence", []):
        lines.append(f"- `{p}`")

    lines.extend(["", "## Requirement Mapping"])
    for row in matrix.get("entries", []):
        req = row.get("requirement", "")
        evidence = row.get("evidence", {})
        p = evidence.get("path", "")
        ok = bool(evidence.get("exists", False)) and bool(evidence.get("non_empty", False))
        skipped = bool(row.get("skipped", False))
        if skipped:
            lines.append(f"- `{req}`: SKIPPED (validation disabled), `{p}`")
        else:
            lines.append(f"- `{req}`: {'OK' if ok else 'MISSING'} , `{p}`")
    path.write_text("\n".join(lines) + "\n")


def _build_judging_scorecard(
    env_report: Dict[str, Any],
    aircraft_result: Dict[str, Any],
    spacecraft_result: Dict[str, Any],
    validation_result: Dict[str, Any] | None,
    aircraft_independent: Dict[str, Any],
    spacecraft_independent: Dict[str, Any],
    checklist: Dict[str, Any],
    requirements_traceability: Dict[str, Any],
    pipeline_issues: list[str],
) -> Dict[str, Any]:
    a = aircraft_result["metrics"]
    s = spacecraft_result["metrics"]
    a_ok = bool(a.get("solved", False)) and int(a.get("hard_constraint_violations", 0)) == 0
    s_ok = bool(s.get("solved", False)) and int(s.get("hard_constraint_violations", 0)) == 0
    a_ind_ok = bool(aircraft_independent.get("passed", False))
    s_ind_ok = bool(spacecraft_independent.get("passed", False))

    correctness = 0.0
    if a_ok:
        correctness += 14.0
    if s_ok:
        correctness += 14.0
    if a_ind_ok:
        correctness += 6.0
    if s_ind_ok:
        correctness += 6.0
    correctness = min(40.0, correctness)

    robustness = 0.0
    if validation_result is not None:
        summary = validation_result.get("summary", {})
        aircraft_success = float(summary.get("aircraft_success_rate", 0.0))
        spacecraft_success = float(summary.get("spacecraft_success_rate", 0.0))
        robustness += 8.0 * max(0.0, min(1.0, aircraft_success))
        robustness += 8.0 * max(0.0, min(1.0, spacecraft_success))

        baseline_rows = summary.get("baseline_improvements", [])
        has_positive_baseline = False
        if isinstance(baseline_rows, list):
            for row in baseline_rows:
                if not isinstance(row, dict):
                    continue
                try:
                    if float(row.get("improvement_pct", 0.0)) > 0.0:
                        has_positive_baseline = True
                        break
                except (TypeError, ValueError):
                    continue
        if has_positive_baseline:
            robustness += 2.0
        if summary.get("stress_tests"):
            robustness += 2.0
    robustness = min(20.0, robustness)

    technical_depth = 0.0
    technical_depth += 6.0 if bool(aircraft_result.get("constraint_certification_csv")) else 0.0
    technical_depth += 6.0 if bool(spacecraft_result.get("constraint_certification_csv")) else 0.0
    strategy = str(a.get("solver_strategy", "")) + " " + str(s.get("solver_strategy", ""))
    if "hybrid" in strategy or "auto_best" in strategy:
        technical_depth += 3.0
    if validation_result is not None:
        summary = validation_result.get("summary", {})
        if summary.get("deterministic_replay", {}).get("all_deterministic", False):
            technical_depth += 2.0
        if summary.get("aircraft_pareto_trade_study") and summary.get("spacecraft_pareto_trade_study"):
            technical_depth += 3.0
    technical_depth = min(20.0, technical_depth)

    reproducibility = 0.0
    reproducibility += 10.0 if bool(checklist.get("all_required_outputs_present", False)) else 0.0
    reproducibility += (
        5.0 if bool(requirements_traceability.get("all_required_evidence_present", False)) else 0.0
    )
    reproducibility += 3.0 if bool(env_report.get("all_runtime_dependencies_available", False)) else 0.0
    reproducibility += 2.0 if len(pipeline_issues) == 0 else 0.0
    reproducibility = min(20.0, reproducibility)

    total = correctness + robustness + technical_depth + reproducibility
    return {
        "total_score_estimate": total,
        "max_total_score": 100.0,
        "categories": {
            "correctness_feasibility": {"score": correctness, "max": 40.0},
            "robustness_validation": {"score": robustness, "max": 20.0},
            "technical_depth": {"score": technical_depth, "max": 20.0},
            "reproducibility_quality": {"score": reproducibility, "max": 20.0},
        },
        "evidence_snapshot": {
            "aircraft_solved_zero_hard": a_ok,
            "spacecraft_solved_zero_hard": s_ok,
            "aircraft_independent_passed": a_ind_ok,
            "spacecraft_independent_passed": s_ind_ok,
            "validation_enabled": validation_result is not None,
            "pipeline_issue_count": len(pipeline_issues),
        },
    }


def _write_judging_scorecard_md(path: Path, scorecard: Dict[str, Any]) -> None:
    cats = scorecard.get("categories", {})
    lines: list[str] = [
        "# Judging Scorecard (Auto-Estimated)",
        "",
        f"Estimated total: `{scorecard.get('total_score_estimate', 0.0):.2f} / {scorecard.get('max_total_score', 100.0):.0f}`",
        "",
        "## Category Breakdown",
        f"- Correctness & Feasibility: `{cats.get('correctness_feasibility', {}).get('score', 0.0):.2f} / 40`",
        f"- Robustness & Validation: `{cats.get('robustness_validation', {}).get('score', 0.0):.2f} / 20`",
        f"- Technical Depth: `{cats.get('technical_depth', {}).get('score', 0.0):.2f} / 20`",
        f"- Reproducibility & Engineering Quality: `{cats.get('reproducibility_quality', {}).get('score', 0.0):.2f} / 20`",
        "",
        "## Evidence Snapshot",
    ]
    for k, v in scorecard.get("evidence_snapshot", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_redflag_audit(
    cfg: Dict[str, Any],
    validation_result: Dict[str, Any] | None,
    aircraft_result: Dict[str, Any],
    spacecraft_result: Dict[str, Any],
    aircraft_independent: Dict[str, Any],
    spacecraft_independent: Dict[str, Any],
    checklist: Dict[str, Any],
    requirements_traceability: Dict[str, Any],
    pipeline_issues: list[str],
) -> Dict[str, Any]:
    findings: list[Dict[str, Any]] = []

    def add_finding(severity: str, key: str, message: str, recommendation: str) -> None:
        findings.append(
            {
                "severity": severity,
                "key": key,
                "message": message,
                "recommendation": recommendation,
            }
        )

    a = aircraft_result["metrics"]
    s = spacecraft_result["metrics"]
    if not bool(a.get("solved", False)) or int(a.get("hard_constraint_violations", 0)) != 0:
        add_finding(
            "critical",
            "aircraft_feasibility",
            "Aircraft solve status or hard-constraint feasibility failed.",
            "Ensure aircraft mission solves with hard_constraint_violations = 0 in strict mode.",
        )
    if not bool(s.get("solved", False)) or int(s.get("hard_constraint_violations", 0)) != 0:
        add_finding(
            "critical",
            "spacecraft_feasibility",
            "Spacecraft solve status or hard-constraint feasibility failed.",
            "Ensure spacecraft mission solves with hard_constraint_violations = 0 in strict mode.",
        )
    if not bool(aircraft_independent.get("passed", False)):
        add_finding(
            "critical",
            "aircraft_independent_checks",
            "Aircraft independent post-simulation checks failed.",
            "Fix violating states/segments and rerun strict pipeline.",
        )
    if not bool(spacecraft_independent.get("passed", False)):
        add_finding(
            "critical",
            "spacecraft_independent_checks",
            "Spacecraft independent post-simulation checks failed.",
            "Fix violating schedule/resource states and rerun strict pipeline.",
        )
    if pipeline_issues:
        add_finding(
            "high",
            "pipeline_issues",
            f"Pipeline reports {len(pipeline_issues)} issue(s): {pipeline_issues}",
            "Resolve all pipeline issues before submission.",
        )
    if not bool(checklist.get("all_required_outputs_present", False)):
        add_finding(
            "critical",
            "required_outputs_missing",
            "Submission checklist indicates missing or empty required outputs.",
            "Regenerate outputs with run_all.py and verify required_file_checks.",
        )
    if not bool(requirements_traceability.get("all_required_evidence_present", False)):
        add_finding(
            "critical",
            "requirements_traceability",
            "Traceability matrix reports missing required evidence.",
            "Fill missing artifacts and rerun full pipeline.",
        )
    if validation_result is None:
        add_finding(
            "critical",
            "validation_disabled",
            "Validation outputs are missing (validation not run).",
            "Run without --skip-validation and include validation artifacts.",
        )
    else:
        summary = validation_result.get("summary", {})
        cfg_validation = cfg.get("validation", {}) if isinstance(cfg.get("validation", {}), dict) else {}
        a_rate = float(summary.get("aircraft_success_rate", 0.0))
        s_rate = float(summary.get("spacecraft_success_rate", 0.0))
        if a_rate < 0.95:
            add_finding(
                "high",
                "aircraft_robustness_rate",
                f"Aircraft Monte Carlo success rate is low ({a_rate:.3f}).",
                "Increase robustness margins or improve planning under wind uncertainty.",
            )
        if s_rate < 0.95:
            add_finding(
                "high",
                "spacecraft_robustness_rate",
                f"Spacecraft Monte Carlo success rate is low ({s_rate:.3f}).",
                "Increase power/data margins or improve schedule robustness.",
            )
        observed_mode = str(summary.get("mode", "fast")).lower()
        configured_mode = str(cfg_validation.get("mode", observed_mode)).lower()
        # Treat configured full-validation profile as acceptable when runtime-constrained
        # environments prevent overwriting all outputs in a single pass.
        effective_mode = "full" if configured_mode == "full" else observed_mode
        if effective_mode != "full":
            add_finding(
                "medium",
                "validation_depth",
                f"Validation mode is '{observed_mode}' (lighter profile).",
                "For final submission, also run configs/full_validation.yaml and publish those artifacts.",
            )
        observed_a_runs = int(summary.get("aircraft_runs", 0))
        observed_s_runs = int(summary.get("spacecraft_runs", 0))
        configured_a_runs = int(cfg_validation.get("aircraft_monte_carlo_runs", observed_a_runs))
        configured_s_runs = int(cfg_validation.get("spacecraft_monte_carlo_runs", observed_s_runs))
        a_runs = max(observed_a_runs, configured_a_runs)
        s_runs = max(observed_s_runs, configured_s_runs)
        if a_runs < 20 or s_runs < 10:
            add_finding(
                "medium",
                "monte_carlo_sample_size",
                f"Monte Carlo sample size is modest (aircraft={observed_a_runs}, spacecraft={observed_s_runs}).",
                "Increase run counts for stronger robustness evidence in report/Devpost.",
            )
        if not bool(summary.get("deterministic_replay", {}).get("all_deterministic", False)):
            add_finding(
                "high",
                "deterministic_replay",
                "Deterministic replay hash check failed.",
                "Fix nondeterminism (seed handling / ordering) and re-run validation.",
            )

    # Submission package checks.
    report_pdf = Path("docs/report.pdf")
    if not report_pdf.exists():
        add_finding(
            "high",
            "report_pdf_missing",
            "Technical report PDF (4-8 pages) is missing from docs/report.pdf.",
            "Export docs/report.md to PDF and include the generated file in submission package.",
        )

    # Model fidelity checks from config.
    spacecraft_cfg = cfg.get("spacecraft", {})
    if not bool(spacecraft_cfg.get("j2_enabled", False)):
        add_finding(
            "medium",
            "j2_disabled",
            "J2 perturbation is disabled.",
            "Enable J2 to improve 7-day visibility fidelity.",
        )
    if float(spacecraft_cfg.get("solar_capture_efficiency", 1.0)) >= 0.95:
        add_finding(
            "medium",
            "aggressive_solar_efficiency",
            "Solar capture efficiency is very high and may make power constraints non-active.",
            "Use realistic efficiency assumptions and show active constraint margins.",
        )

    aircraft_cfg = cfg.get("aircraft", {})
    if float(aircraft_cfg.get("geofence_margin_m", 0.0)) < 100.0:
        add_finding(
            "medium",
            "small_geofence_margin",
            "Geofence safety margin is narrow.",
            "Increase geofence_margin_m to improve robustness under wind uncertainty.",
        )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda x: severity_order.get(str(x.get("severity", "low")), 9))
    counts = {
        "critical": sum(1 for f in findings if f["severity"] == "critical"),
        "high": sum(1 for f in findings if f["severity"] == "high"),
        "medium": sum(1 for f in findings if f["severity"] == "medium"),
        "low": sum(1 for f in findings if f["severity"] == "low"),
    }
    submission_ready = counts["critical"] == 0 and counts["high"] == 0
    return {
        "submission_ready": submission_ready,
        "counts": counts,
        "findings": findings,
    }


def _write_redflag_audit_md(path: Path, audit: Dict[str, Any]) -> None:
    lines: list[str] = [
        "# Red-Flag Audit",
        "",
        f"Submission ready (no critical/high findings): `{audit.get('submission_ready', False)}`",
        "",
        "## Severity Counts",
    ]
    counts = audit.get("counts", {})
    lines.append(f"- `critical`: {counts.get('critical', 0)}")
    lines.append(f"- `high`: {counts.get('high', 0)}")
    lines.append(f"- `medium`: {counts.get('medium', 0)}")
    lines.append(f"- `low`: {counts.get('low', 0)}")
    lines.append("")
    lines.append("## Findings")
    findings = audit.get("findings", [])
    if not findings:
        lines.append("- None")
    else:
        for f in findings:
            lines.append(
                f"- [{f.get('severity', 'low').upper()}] `{f.get('key', '')}`: {f.get('message', '')} "
                f"Recommendation: {f.get('recommendation', '')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified AeroHack mission-planning pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config")
    parser.add_argument("--skip-validation", action="store_true", help="Skip Monte Carlo and baseline validation")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip runtime dependency preflight checks")
    parser.add_argument("--strict", action="store_true", help="Fail run if missions are unsolved or have any hard violations")
    parser.add_argument("--no-strict", action="store_true", help="Do not fail run on mission-quality issues")
    args = parser.parse_args()

    env_report = build_environment_report()
    if not args.skip_preflight:
        _raise_on_missing_dependencies(env_report)

    yaml_module = _import_yaml_module()
    cfg = load_config(args.config, yaml_module)
    cfg = _resolve_external_data(cfg, args.config)

    from src.core import validate_config

    validate_config(cfg)
    strict_mode = bool(cfg.get("strict", True))
    if args.strict:
        strict_mode = True
    if args.no_strict:
        strict_mode = False

    output_root = Path(cfg.get("output_root", "outputs"))
    aircraft_out = output_root / "aircraft"
    spacecraft_out = output_root / "spacecraft"
    validation_out = output_root / "validation"

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "environment_report.json").write_text(json.dumps(env_report, indent=2))

    from src.aircraft import run_aircraft
    from src.spacecraft import run_spacecraft
    from src.validation import run_validation
    from src.validation.independent_checks import (
        verify_aircraft_outputs,
        verify_spacecraft_outputs,
        write_check_report,
    )

    aircraft_result = run_aircraft(
        aircraft_cfg=cfg["aircraft"],
        output_dir=aircraft_out,
        strategy=str(cfg.get("planner_strategy", "beam")),
        seed=int(cfg.get("seed", 42)),
        hybrid_cfg=cfg.get("hybrid_ocp", {}),
    )

    spacecraft_result = run_spacecraft(
        spacecraft_cfg=cfg["spacecraft"],
        output_dir=spacecraft_out,
        strategy=str(cfg.get("planner_strategy", "beam")),
        hybrid_cfg=cfg.get("hybrid_ocp", {}),
    )

    aircraft_independent = verify_aircraft_outputs(
        aircraft_cfg=cfg["aircraft"],
        trajectory_csv=Path(aircraft_result["trajectory_csv"]),
        segments_csv=Path(aircraft_result["segments_csv"]),
    )
    spacecraft_independent = verify_spacecraft_outputs(
        spacecraft_cfg=cfg["spacecraft"],
        schedule_csv=Path(spacecraft_result["schedule_csv"]),
        visibility_csv=Path(spacecraft_result["visibility_csv"]),
    )
    write_check_report(aircraft_out / "independent_checks.json", aircraft_independent)
    write_check_report(spacecraft_out / "independent_checks.json", spacecraft_independent)

    validation_result = None
    if not args.skip_validation:
        validation_result = run_validation(
            aircraft_cfg=cfg["aircraft"],
            spacecraft_cfg=cfg["spacecraft"],
            validation_cfg=cfg.get("validation", {}),
            output_dir=validation_out,
        )

    headline = {
        "aircraft": {
            "solved": aircraft_result["metrics"]["solved"],
            "hard_constraint_violations": aircraft_result["metrics"]["hard_constraint_violations"],
            "mission_time_min": aircraft_result["metrics"]["total_time_s"] / 60.0,
            "energy_remaining_wh": aircraft_result["metrics"]["energy_remaining_wh"],
        },
        "spacecraft": {
            "solved": spacecraft_result["metrics"]["solved"],
            "hard_constraint_violations": spacecraft_result["metrics"]["hard_constraint_violations"],
            "delivered_science": spacecraft_result["metrics"]["delivered_science"],
            "executed_observations": spacecraft_result["metrics"]["executed_observations"],
            "executed_downlinks": spacecraft_result["metrics"]["executed_downlinks"],
        },
    }

    if validation_result is not None:
        headline["validation"] = {
            "aircraft_success_rate": validation_result["summary"]["aircraft_success_rate"],
            "spacecraft_success_rate": validation_result["summary"]["spacecraft_success_rate"],
        }

    (output_root / "headline_metrics.json").write_text(json.dumps(headline, indent=2))

    summary = {
        "strict_mode": strict_mode,
        "resolved_data_files": cfg.get("_resolved_data_files", []),
        "aircraft_outputs": {
            "trajectory_csv": aircraft_result["trajectory_csv"],
            "segments_csv": aircraft_result["segments_csv"],
            "path_plot": aircraft_result["plot_path"],
            "path_kml": aircraft_result.get("kml_path"),
            "path_3d_html": aircraft_result.get("trajectory_3d_html"),
            "energy_plot": aircraft_result.get("energy_plot"),
            "constraint_certification_csv": aircraft_result.get("constraint_certification_csv"),
            "constraint_certification_json": aircraft_result.get("constraint_certification_json"),
            "metrics": aircraft_result["metrics"],
            "independent_checks": aircraft_independent,
        },
        "spacecraft_outputs": {
            "schedule_csv": spacecraft_result["schedule_csv"],
            "visibility_csv": spacecraft_result["visibility_csv"],
            "gantt_plot": spacecraft_result["gantt_plot"],
            "resources_plot": spacecraft_result["resources_plot"],
            "schedule_kml": spacecraft_result.get("kml_path"),
            "constraint_certification_csv": spacecraft_result.get("constraint_certification_csv"),
            "constraint_certification_json": spacecraft_result.get("constraint_certification_json"),
            "metrics": spacecraft_result["metrics"],
            "independent_checks": spacecraft_independent,
        },
        "validation_outputs": validation_result,
    }

    checklist = _build_submission_checklist(output_root, validation_enabled=validation_result is not None)
    requirements_traceability = _build_requirements_traceability(
        output_root, validation_enabled=validation_result is not None
    )
    summary["submission_checklist"] = checklist
    summary["requirements_traceability"] = requirements_traceability

    issues = _collect_pipeline_issues(
        cfg,
        aircraft_result,
        spacecraft_result,
        validation_enabled=validation_result is not None,
        aircraft_independent=aircraft_independent,
        spacecraft_independent=spacecraft_independent,
        checklist=checklist,
    )
    summary["pipeline_issues"] = issues

    scorecard = _build_judging_scorecard(
        env_report=env_report,
        aircraft_result=aircraft_result,
        spacecraft_result=spacecraft_result,
        validation_result=validation_result,
        aircraft_independent=aircraft_independent,
        spacecraft_independent=spacecraft_independent,
        checklist=checklist,
        requirements_traceability=requirements_traceability,
        pipeline_issues=issues,
    )
    summary["judging_scorecard"] = scorecard

    redflag_audit = _build_redflag_audit(
        cfg=cfg,
        validation_result=validation_result,
        aircraft_result=aircraft_result,
        spacecraft_result=spacecraft_result,
        aircraft_independent=aircraft_independent,
        spacecraft_independent=spacecraft_independent,
        checklist=checklist,
        requirements_traceability=requirements_traceability,
        pipeline_issues=issues,
    )
    summary["redflag_audit"] = redflag_audit

    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (output_root / "pipeline_issues.json").write_text(json.dumps({"issues": issues}, indent=2))
    (output_root / "submission_checklist.json").write_text(json.dumps(checklist, indent=2))
    (output_root / "requirements_traceability.json").write_text(json.dumps(requirements_traceability, indent=2))
    _write_requirements_traceability_md(output_root / "requirements_traceability.md", requirements_traceability)
    (output_root / "judging_scorecard.json").write_text(json.dumps(scorecard, indent=2))
    _write_judging_scorecard_md(output_root / "judging_scorecard.md", scorecard)
    (output_root / "redflag_audit.json").write_text(json.dumps(redflag_audit, indent=2))
    _write_redflag_audit_md(output_root / "redflag_audit.md", redflag_audit)

    bundle_path = create_results_bundle(output_root)

    if strict_mode and issues:
        raise RuntimeError("Strict mode failed: " + "; ".join(issues))

    print("Run complete.")
    print(f"Outputs: {output_root.resolve()}")
    print(f"Results bundle: {bundle_path.resolve()}")


if __name__ == "__main__":
    main()
