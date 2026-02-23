# AeroCore: Unified Aerospace Mission Planning Framework

Single, unified planning + simulation framework for both:
- Aircraft mission planning under wind, energy, maneuver, and geofence constraints.
- Spacecraft 7-day observation/downlink scheduling under visibility, slew, battery, and duty-cycle constraints.

Both modules use the same underlying concept:
- Task graph with transition dynamics.
- Shared hard-constraint API.
- Shared weighted objective API.
- Shared solver stack:
  - `UnifiedPlanner` (`beam`, `greedy`, `multistart`) for discrete mission structure.
  - `UnifiedHybridOCPEngine` for hybrid discrete+continuous robust optimization.
    - Backend path supports `scipy` coordinate search now, with optional `casadi/ipopt` backend under the same interface when `casadi` is installed.

Certification-first positioning:
- Plan synthesis + independent feasibility certification + robustness certification + trade-study evidence.
- The framework is designed to produce auditable mission evidence, not only trajectories/schedules.
- We explicitly choose hybrid OCP over pure MILP/CP-SAT because core aircraft/orbital dynamics are nonlinear and continuous; this preserves physical fidelity under one shared architecture.

## Project Structure

```text
src/
  core/          # shared planner, task schema, constraints, objective
  aircraft/      # aircraft adapter: wind-aware transitions, geofence checks, outputs
  spacecraft/    # spacecraft adapter: opportunity generation, slew/power/data scheduling
  validation/    # Monte Carlo, baseline comparison, summary plots
configs/
  default.yaml
data/            # external mission inputs used by configs/*.yaml
outputs/         # generated artifacts
tests/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --config configs/default.yaml
# deeper validation profile:
# python run_all.py --config configs/full_validation.yaml
```

The default config runs in `strict: true` mode, which fails the run if either domain is unsolved, has hard violations, or the spacecraft schedule does not cover the full 7-day horizon with a terminal end action.
Default solver mode is `planner_strategy: auto_best` (beam baseline + hybrid OCP refinement with fallback to the better feasible result).
`run_all.py` resolves `aircraft.data_files` and `spacecraft.data_files` from config so mission datasets are shipped explicitly (not hidden in code).

For tests only:
```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Reproduce Results

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Run full pipeline (aircraft + spacecraft + validation):
```bash
python run_all.py --config configs/default.yaml
```
For a one-page quickstart, see `REPRODUCE.md`.

Optional strictness controls:
```bash
python run_all.py --config configs/default.yaml --strict
python run_all.py --config configs/default.yaml --no-strict
```

Optional solver mode:
```bash
python run_all.py --config configs/default.yaml --strict
# set planner_strategy in configs/default.yaml:
#   beam | greedy | multistart | hybrid_ocp | auto_best
```

3. Generated outputs:
- `outputs/aircraft/uav_flight_plan.csv`
- `outputs/aircraft/uav_route_segments.csv`
- `outputs/aircraft/uav_path.png`
- `outputs/aircraft/uav_path.kml`
- `outputs/aircraft/uav_trajectory_3d.html`
- `outputs/aircraft/uav_energy_profile.png`
- `outputs/aircraft/uav_metrics.json`
- `outputs/aircraft/uav_constraint_certification.csv`
- `outputs/aircraft/uav_constraint_certification.json`
- `outputs/spacecraft/spacecraft_7day_schedule.csv`
- `outputs/spacecraft/visibility_windows.csv`
- `outputs/spacecraft/spacecraft_gantt.png`
- `outputs/spacecraft/spacecraft_resources.png`
- `outputs/spacecraft/spacecraft_schedule.kml`
- `outputs/spacecraft/spacecraft_metrics.json`
- `outputs/spacecraft/spacecraft_constraint_certification.csv`
- `outputs/spacecraft/spacecraft_constraint_certification.json`
- `outputs/aircraft/independent_checks.json`
- `outputs/spacecraft/independent_checks.json`
- `outputs/validation/aircraft_monte_carlo.csv`
- `outputs/validation/spacecraft_monte_carlo.csv`
- `outputs/validation/baseline_comparison.csv`
- `outputs/validation/stress_tests.csv`
- `outputs/validation/stress_downlink_windows.csv`
- `outputs/validation/aircraft_pareto_trade_study.csv`
- `outputs/validation/spacecraft_pareto_trade_study.csv`
- `outputs/validation/deterministic_replay.json`
- `outputs/validation/spacecraft_monte_carlo_traces.csv`
- `outputs/validation/stress_spacecraft_low_solar_trace.csv`
- `outputs/validation/robustness_max_deviation.csv`
- `outputs/validation/validation_summary.json`
- `outputs/validation/aircraft_hybrid_vs_beam_variance.png`
- `outputs/validation/aircraft_scenario_sensitivity.png`
- `outputs/validation/spacecraft_hybrid_vs_beam_science.png`
- `outputs/validation/spacecraft_scenario_science.png`
- `outputs/validation/spacecraft_stress_resource_evolution.png`
- `outputs/validation/stress_margin_plot.png`
- `outputs/validation/stress_downlink_vs_constraints.png`
- `outputs/validation/aircraft_pareto_frontier.png`
- `outputs/validation/spacecraft_pareto_frontier.png`
- `outputs/submission_checklist.json`
- `outputs/requirements_traceability.json`
- `outputs/requirements_traceability.md`
- `outputs/judging_scorecard.json`
- `outputs/judging_scorecard.md`
- `outputs/redflag_audit.json`
- `outputs/redflag_audit.md`
- `outputs/results_bundle.zip`

TLE ingestion tooling:
- `tools/tle_to_spacecraft_config.py --tle data/sample_iss.tle --out outputs/tle_overrides.json`

4. Regenerate plots/metrics from scratch:
- Delete `outputs/`
- Re-run `python run_all.py --config configs/default.yaml`

## Runtime Assumptions

- Tested target: laptop CPU (Apple Silicon / x86 equivalent).
- Default runtime target: a few minutes depending on Monte Carlo counts in `configs/default.yaml`.
- To increase robustness depth (e.g., 100+ seeds), raise values under `validation`.

## Unified Formulation

Each domain maps to the same planning abstraction:

- **Task**: time window, duration, value, metadata, resource effects.
- **Transition**: feasibility, travel/slew time, energy implications.
- **State**: time + resources (energy/battery/data) + cumulative mission metrics.
- **Constraints**: shared `Constraint.evaluate(...)` interface with hard/soft support.
- **Objective**: shared `WeightedObjective` scoring incremental utility vs time/energy.
- **Solver**:
  - `UnifiedPlanner.solve(...)` for discrete action sequencing.
  - `UnifiedHybridOCPEngine.solve(...)` for hybrid optimal control refinement:
    - direct-shooting continuous control tuning per transition,
    - robust objective across uncertainty scenarios,
    - discrete sequence mutation and selection.
  - `planner_strategy: auto_best` runs both and keeps the better feasible plan.

### Aircraft Mapping

- Tasks: waypoint visits + return-to-base.
- Transitions: geofence-aware route generation (visibility graph + fallback routing), wind-aware travel, and bank/turn/climb considerations.
- Hard constraints: geofence, energy reserve, mission horizon, altitude, turn-rate.
- Objective emphasis: minimize time/energy with full required waypoint completion.

### Spacecraft Mapping

- Tasks: observation windows + downlink windows over 7 days.
- Transitions: slew-feasible retargeting between tasks with dynamic pointing vectors.
- Hard constraints: battery bounds, buffer bounds, max ops/orbit, task windows.
- Objective emphasis: maximize delivered science while respecting resources.

## Modeling Fidelity Upgrades

- Aircraft:
  - No-fly-aware routing between tasks (not straight-line shortcuts through restricted areas).
  - Wind-aware segment timing with time-varying spatial wind and random-phase harmonic gust field.
  - True aerodynamic power integration (Zero-lift & Induced Drag, Wing Area, Aspect Ratio, Oswald Efficiency).
  - Combined turn-rate + bank-angle feasibility.
  - Climb feasibility from altitude delta and climb-rate bound.
- Spacecraft:
  - Two-body circular orbit propagation in ECI with J2 secular RAAN drift.
  - ECI/ECEF conversion with Earth rotation.
  - Epoch-aware Sun vector for eclipse prediction.
  - Observation visibility from geometric LOS + off-nadir limits.
  - Downlink visibility from elevation-angle checks.
  - Eclipse-aware charging with solar-capture efficiency and active battery lower-bound enforcement.

## Current Headline Metrics

The exact metrics for your run are written to:
- `outputs/headline_metrics.json`
- `outputs/aircraft/uav_metrics.json`
- `outputs/spacecraft/spacecraft_metrics.json`
- `outputs/validation/validation_summary.json`

Latest default run snapshot in this repo:
- Aircraft hard violations: `0`
- Aircraft mission time: `7260.99 s`
- Aircraft energy remaining: `685.07 Wh`
- Spacecraft hard violations: `0`
- Spacecraft delivered science: `235.68`
- Validation success rate (default profile): `1.0 aircraft / 1.0 spacecraft`
- Deterministic replay certification: `pass`
- Auto judging scorecard: see `outputs/judging_scorecard.json`

## Validation

Validation runs from the same codebase and includes:
- Aircraft Monte Carlo (wind + battery perturbations).
- Spacecraft Monte Carlo (battery + solar perturbations, plus timing perturbations on a configured subset of runs).
- Baseline comparison (shared solver in `greedy` mode vs `beam` mode).
- Stress scenarios (tightened wind/energy and solar/battery cases).
- Pareto trade studies for both domains.
- Deterministic replay hash checks for reproducibility.
- Robust optimization summaries (p05/p50/p95 and worst-case margins).
- Constraint certification tables for both domains:
  - max violation,
  - slack margin,
  - active fraction,
  - pseudo-multiplier proxy.

Validation mode can be configured in `configs/default.yaml`:
- `mode: full` (default): stronger robustness sweep for submission artifacts.
- `mode: fast`: lighter profile for quick iteration.
- A ready-to-run deep profile is provided at `configs/full_validation.yaml`.
- Hybrid-evaluation depth and timing perturbation depth are configurable:
  - `aircraft_hybrid_eval_runs`
  - `spacecraft_hybrid_eval_runs`
  - `spacecraft_timing_perturbation_runs`

Validation implementation notes:
- Spacecraft opportunity generation and transitions are cached across repeated validation runs.
- Monte Carlo and stress tests reuse the same unified solver/problem structure and only perturb initial state and environment scale factors.

## Running Tests

```bash
python -m unittest discover -s tests -p "test_unittest_*.py"
```

Optional (`pytest`) test suite:
```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Setup Diagnostics

`run_all.py` performs runtime dependency preflight checks and writes:
- `outputs/environment_report.json`

This file captures Python executable/version and package availability so setup issues are visible immediately.

## Independent Constraint Audit

Beyond planner-reported constraints, `run_all.py` runs an independent post-simulation audit:
- Aircraft audit: geofence crossings, energy bounds, turn-rate bounds, altitude bounds, monotonic timeline.
- Spacecraft audit: schedule-window compliance, battery/data bounds, ops/orbit, terminal horizon closure, visibility-task consistency.

Artifacts:
- `outputs/aircraft/independent_checks.json`
- `outputs/spacecraft/independent_checks.json`

If strict mode is enabled, any independent-audit violation fails the run.

## Challenge Traceability

`run_all.py` also writes a requirement-to-evidence matrix aligned to AeroHack submission needs:
- `outputs/requirements_traceability.json`
- `outputs/requirements_traceability.md`

This maps core challenge requirements (aircraft outputs, spacecraft outputs, validation, bundle) to concrete generated files.

## Notes

- Orbit/visibility and aircraft dynamics are simplified but physically constrained and reproducible.
- The framework is intentionally structured for hackathon reliability: deterministic config, one-command execution, and consistent outputs.
- Submission-oriented docs:
  - Technical report draft: `docs/report.md`
  - Devpost summary draft: [devpost_summary.md](file:///Users/shauryabhushan/Downloads/AeroHacks/docs/devpost_summary.md)
