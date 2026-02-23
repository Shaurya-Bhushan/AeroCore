from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .models import MissionProblem, PlanResult, PlanStep, Transition
from .planner import UnifiedPlanner


NEG_INF = -1e18


@dataclass
class HybridOCPSettings:
    backend_preference: str = "auto"
    robust_mode: str = "worst_case"
    control_iterations: int = 2
    control_initial_step: float = 0.25
    max_sequence_candidates: int = 12
    max_mutations_per_seed: int = 24
    max_control_nodes_aircraft: int = 24
    max_control_nodes_spacecraft: int = 32
    aircraft_scenarios: Tuple[float, ...] = (1.0, 0.9, 1.1)
    spacecraft_scenarios: Tuple[float, ...] = (1.0, 0.9, 1.1)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _settings_from_cfg(cfg: Dict[str, Any] | None) -> HybridOCPSettings:
    cfg = cfg or {}
    return HybridOCPSettings(
        backend_preference=str(cfg.get("backend_preference", "auto")).lower(),
        robust_mode=str(cfg.get("robust_mode", "worst_case")).lower(),
        control_iterations=max(1, int(cfg.get("control_iterations", 2))),
        control_initial_step=max(0.01, float(cfg.get("control_initial_step", 0.25))),
        max_sequence_candidates=max(2, int(cfg.get("max_sequence_candidates", 12))),
        max_mutations_per_seed=max(1, int(cfg.get("max_mutations_per_seed", 24))),
        max_control_nodes_aircraft=max(1, int(cfg.get("max_control_nodes_aircraft", 24))),
        max_control_nodes_spacecraft=max(1, int(cfg.get("max_control_nodes_spacecraft", 32))),
        aircraft_scenarios=tuple(float(x) for x in cfg.get("aircraft_scenarios", [1.0, 0.9, 1.1])),
        spacecraft_scenarios=tuple(float(x) for x in cfg.get("spacecraft_scenarios", [1.0, 0.9, 1.1])),
    )


class UnifiedHybridOCPEngine:
    """
    Unified hybrid mission optimizer:
    - discrete sequence search (mission actions)
    - continuous control refinement (direct shooting over step controls)
    - robust objective selection across uncertainty scenarios
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.settings = _settings_from_cfg(cfg)
        self._casadi_available = importlib.util.find_spec("casadi") is not None

    def solve(self, problem: MissionProblem, planner: UnifiedPlanner | None = None) -> PlanResult:
        planner = planner or UnifiedPlanner()

        seed_sequences = self._seed_sequences(problem, planner)
        if not seed_sequences:
            return planner.solve(problem, strategy="beam")

        candidate_sequences = self._build_candidate_sequences(problem, seed_sequences)
        if not candidate_sequences:
            return planner.solve(problem, strategy="beam")

        best_result: PlanResult | None = None
        best_robust = NEG_INF

        for sequence in candidate_sequences:
            robust_score, nominal_result = self._optimize_controls(problem, sequence)
            if nominal_result is None:
                continue
            if best_result is None or robust_score > best_robust:
                best_robust = robust_score
                best_result = nominal_result

        if best_result is None:
            return planner.solve(problem, strategy="beam")

        best_result.final_state["hybrid_robust_objective"] = float(best_robust)
        best_result.final_state["hybrid_solver"] = "unified_hybrid_ocp"
        best_result.final_state["hybrid_candidate_count"] = len(candidate_sequences)
        best_result.final_state["hybrid_backend_preference"] = self.settings.backend_preference
        best_result.final_state["hybrid_casadi_available"] = bool(self._casadi_available)
        return best_result

    @staticmethod
    def is_better(lhs: PlanResult, rhs: PlanResult) -> bool:
        lhs_key = (
            1 if lhs.solved else 0,
            float(lhs.objective_score),
            -len(lhs.hard_violations),
            int(lhs.visited_required),
        )
        rhs_key = (
            1 if rhs.solved else 0,
            float(rhs.objective_score),
            -len(rhs.hard_violations),
            int(rhs.visited_required),
        )
        return lhs_key > rhs_key

    def _seed_sequences(self, problem: MissionProblem, planner: UnifiedPlanner) -> List[List[str]]:
        sequences: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for strategy in ("beam", "multistart", "greedy"):
            try:
                result = planner.solve(problem, strategy=strategy)
            except Exception:
                continue
            seq = list(result.sequence)
            if len(seq) < 2:
                continue
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            sequences.append(seq)
        return sequences

    def _build_candidate_sequences(
        self,
        problem: MissionProblem,
        seed_sequences: Sequence[Sequence[str]],
    ) -> List[List[str]]:
        scored: List[Tuple[float, List[str]]] = []
        seen: set[tuple[str, ...]] = set()

        def add_candidate(seq: Sequence[str]) -> None:
            key = tuple(seq)
            if key in seen:
                return
            seen.add(key)
            robust_score, nominal_result = self._evaluate_controls(problem, list(seq), controls=None)
            if nominal_result is None:
                return
            scored.append((robust_score, list(seq)))

        for seq in seed_sequences:
            add_candidate(seq)

        base_ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        for _, seq in base_ranked[: min(3, len(base_ranked))]:
            for mutated in self._mutate_sequence(problem, seq):
                add_candidate(mutated)
                if len(scored) >= self.settings.max_sequence_candidates * 2:
                    break
            if len(scored) >= self.settings.max_sequence_candidates * 2:
                break

        scored.sort(key=lambda x: x[0], reverse=True)
        return [seq for _, seq in scored[: self.settings.max_sequence_candidates]]

    def _mutate_sequence(self, problem: MissionProblem, sequence: Sequence[str]) -> Iterable[List[str]]:
        if len(sequence) <= 3:
            return []

        required = set(problem.required_set())
        end_id = problem.end_task_id
        max_emit = self.settings.max_mutations_per_seed
        emitted = 0

        # Local swap mutations
        for i in range(1, len(sequence) - 2):
            mutated = list(sequence)
            mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
            emitted += 1
            yield mutated
            if emitted >= max_emit:
                return

        # Drop optional actions
        for i in range(1, len(sequence) - 1):
            task_id = sequence[i]
            if task_id in required:
                continue
            if end_id is not None and task_id == end_id:
                continue
            mutated = list(sequence[:i]) + list(sequence[i + 1 :])
            emitted += 1
            yield mutated
            if emitted >= max_emit:
                return

    def _optimize_controls(self, problem: MissionProblem, sequence: List[str]) -> Tuple[float, PlanResult | None]:
        backend = self._select_backend(problem)
        if backend == "casadi":
            score, result = self._optimize_controls_casadi(problem, sequence)
            if result is not None:
                result.final_state["hybrid_backend_used"] = "casadi"
                return score, result
        score, result = self._optimize_controls_coordinate(problem, sequence)
        if result is not None:
            result.final_state["hybrid_backend_used"] = str(result.final_state.get("hybrid_backend_used", "scipy"))
        return score, result

    def _optimize_controls_coordinate(self, problem: MissionProblem, sequence: List[str]) -> Tuple[float, PlanResult | None]:
        score, result = self._optimize_controls_scipy(problem, sequence)
        if result is not None:
            return score, result
        return self._optimize_controls_coordinate_search(problem, sequence)

    def _optimize_controls_scipy(self, problem: MissionProblem, sequence: List[str]) -> Tuple[float, PlanResult | None]:
        try:
            from scipy import optimize as spo  # type: ignore
        except Exception:
            return NEG_INF, None

        n_steps = max(0, len(sequence) - 1)
        if n_steps == 0:
            return NEG_INF, None
        control_indices = self._control_indices(problem, sequence)
        if not control_indices:
            return self._evaluate_controls(problem, sequence, controls=None)

        n_vars = len(control_indices)
        x0 = np.full(n_vars, 0.5, dtype=float)

        def _evaluate_x(x_vec: np.ndarray) -> Tuple[float, PlanResult | None]:
            x_clip = np.clip(np.asarray(x_vec, dtype=float), 0.0, 1.0)
            controls = self._expand_controls(n_steps, control_indices, x_clip)
            return self._evaluate_controls(problem, sequence, controls=controls)

        best_score, best_result = _evaluate_x(x0)
        best_x = np.array(x0, dtype=float)

        def objective(x_vec: np.ndarray) -> float:
            score, result = _evaluate_x(x_vec)
            if result is None:
                reg = float(np.sum((np.asarray(x_vec, dtype=float) - 0.5) ** 2))
                return 1.0e9 + 1.0e4 * reg
            reg = 1.0e-4 * float(np.sum((np.asarray(x_vec, dtype=float) - 0.5) ** 2))
            return float(-score + reg)

        bounds = [(0.0, 1.0) for _ in range(n_vars)]
        rng = np.random.default_rng(17)
        starts = [x0]
        for _ in range(min(4, max(1, self.settings.control_iterations + 1))):
            starts.append(np.clip(x0 + rng.normal(0.0, 0.18, size=n_vars), 0.0, 1.0))

        # Primary continuous optimizer: SLSQP.
        for start in starts:
            try:
                res = spo.minimize(
                    objective,
                    np.asarray(start, dtype=float),
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": 80, "ftol": 1e-3, "disp": False},
                )
                cand_x = np.asarray(res.x, dtype=float) if hasattr(res, "x") else np.asarray(start, dtype=float)
            except Exception:
                cand_x = np.asarray(start, dtype=float)
            score, result = _evaluate_x(cand_x)
            if result is not None and (best_result is None or score > best_score + 1e-6):
                best_score = score
                best_result = result
                best_x = np.clip(cand_x, 0.0, 1.0)

        # Secondary fallback optimizer: COBYLA with explicit bound inequalities.
        cons = []
        for i in range(n_vars):
            cons.append({"type": "ineq", "fun": (lambda x, idx=i: x[idx])})
            cons.append({"type": "ineq", "fun": (lambda x, idx=i: 1.0 - x[idx])})
        for start in starts:
            try:
                res = spo.minimize(
                    objective,
                    np.asarray(start, dtype=float),
                    method="COBYLA",
                    constraints=cons,
                    options={"maxiter": 120, "tol": 1e-3},
                )
                cand_x = np.asarray(res.x, dtype=float) if hasattr(res, "x") else np.asarray(start, dtype=float)
            except Exception:
                continue
            score, result = _evaluate_x(np.clip(cand_x, 0.0, 1.0))
            if result is not None and (best_result is None or score > best_score + 1e-6):
                best_score = score
                best_result = result
                best_x = np.clip(cand_x, 0.0, 1.0)

        if best_result is None:
            return NEG_INF, None
        best_result.final_state["hybrid_control_vector"] = [float(v) for v in self._expand_controls(n_steps, control_indices, best_x)]
        best_result.final_state["hybrid_backend_used"] = "scipy_slsqp_cobyla"
        best_result.final_state["hybrid_scipy_status"] = "success"
        return best_score, best_result

    def _optimize_controls_coordinate_search(self, problem: MissionProblem, sequence: List[str]) -> Tuple[float, PlanResult | None]:
        n_steps = max(0, len(sequence) - 1)
        if n_steps == 0:
            return NEG_INF, None

        control_indices = self._control_indices(problem, sequence)
        if not control_indices:
            return self._evaluate_controls(problem, sequence, controls=None)

        x = np.full(len(control_indices), 0.5, dtype=float)
        step = self.settings.control_initial_step

        best_score, best_result = self._evaluate_controls(problem, sequence, controls=self._expand_controls(n_steps, control_indices, x))
        if best_result is None:
            return NEG_INF, None

        for _ in range(self.settings.control_iterations):
            improved = False
            for j in range(len(x)):
                current = float(x[j])
                candidates = sorted({_clamp(current - step, 0.0, 1.0), current, _clamp(current + step, 0.0, 1.0)})
                for value in candidates:
                    if abs(value - current) <= 1e-9:
                        continue
                    x_trial = np.array(x, dtype=float)
                    x_trial[j] = value
                    controls = self._expand_controls(n_steps, control_indices, x_trial)
                    score, result = self._evaluate_controls(problem, sequence, controls=controls)
                    if result is None:
                        continue
                    if score > best_score + 1e-6:
                        x = x_trial
                        best_score = score
                        best_result = result
                        improved = True
            if not improved:
                step *= 0.6
            if step < 0.03:
                break

        assert best_result is not None
        best_result.final_state["hybrid_control_vector"] = [float(v) for v in self._expand_controls(n_steps, control_indices, x)]
        best_result.final_state["hybrid_backend_used"] = "scipy_coordinate"
        best_result.final_state["hybrid_scipy_status"] = "coordinate_fallback"
        return best_score, best_result

    def _select_backend(self, problem: MissionProblem) -> str:
        pref = self.settings.backend_preference
        if pref == "scipy":
            return "scipy"
        if pref == "casadi":
            return "casadi" if self._casadi_available and problem.domain == "aircraft" else "scipy"
        # auto
        if self._casadi_available and problem.domain == "aircraft":
            return "casadi"
        return "scipy"

    def _optimize_controls_casadi(self, problem: MissionProblem, sequence: List[str]) -> Tuple[float, PlanResult | None]:
        if not self._casadi_available or problem.domain != "aircraft":
            return NEG_INF, None

        try:
            import casadi as ca  # type: ignore
        except Exception:
            return NEG_INF, None

        n_steps = max(0, len(sequence) - 1)
        if n_steps <= 0:
            return NEG_INF, None

        control_indices = self._control_indices(problem, sequence)
        if not control_indices:
            return NEG_INF, None

        objective = problem.objective
        value_w = float(getattr(objective, "value_weight", 1.0))
        time_w = float(getattr(objective, "time_weight", 0.01))
        energy_w = float(getattr(objective, "energy_weight", 0.01))
        completion_bonus = float(getattr(objective, "required_completion_bonus", 0.0))
        unsolved_penalty = float(getattr(objective, "unsolved_penalty", 0.0))

        required_set = problem.required_set()
        reserve_wh = float(problem.initial_state.get("energy_wh", 0.0)) * 0.0
        try:
            energy_c = next(c for c in problem.constraints if c.name == "energy_reserve_aircraft")
            reserve_wh = float(energy_c.min_value)
        except Exception:
            reserve_wh = 0.0
        initial_energy = float(problem.initial_state.get("energy_wh", 0.0))

        # Prepare constants from transition/task metadata.
        step_data: List[Dict[str, float]] = []
        value_sum = 0.0
        visited_required = set()
        for idx in range(n_steps):
            a = sequence[idx]
            b = sequence[idx + 1]
            trans = problem.transition(a, b)
            if trans is None or not trans.feasible:
                return NEG_INF, None
            task = problem.tasks[b]
            value_sum += value_w * float(task.value)
            if b in required_set:
                visited_required.add(b)
            step_data.append(
                {
                    "t_base": float(trans.travel_time_s),
                    "e_base": float(max(1e-6, trans.energy_cost_wh)),
                }
            )

        # If sequence does not satisfy mission structure, do not optimize.
        if visited_required != required_set:
            return NEG_INF, None
        if problem.end_task_id is not None and sequence[-1] != problem.end_task_id:
            return NEG_INF, None

        opti = ca.Opti()
        u = opti.variable(len(control_indices))
        opti.subject_to(opti.bounded(0.0, u, 1.0))

        controls = [ca.MX(0.5) for _ in range(n_steps)]
        for local_idx, global_idx in enumerate(control_indices):
            controls[global_idx] = u[local_idx]

        scenario_scores: List[ca.MX] = []
        for sc in self._scenario_overrides(problem):
            wind_mult = float(sc.get("wind_scale_multiplier", 1.0))
            time_scale = max(0.5, 1.0 + 0.35 * (wind_mult - 1.0))
            energy_scale = max(0.5, 1.0 + 0.20 * abs(wind_mult - 1.0))

            total_time = 0.0
            total_energy = 0.0
            for idx, d in enumerate(step_data):
                c = controls[idx]
                speed_scale = 0.75 + 0.50 * c
                t_i = d["t_base"] * time_scale / speed_scale
                e_i = d["e_base"] * energy_scale * (speed_scale ** 3)
                total_time += t_i
                total_energy += e_i

            # Energy reserve hard constraint surrogate.
            final_energy = initial_energy - total_energy
            opti.subject_to(final_energy >= reserve_wh)

            score = value_sum - time_w * total_time - energy_w * total_energy + completion_bonus - unsolved_penalty * 0.0
            scenario_scores.append(score)

        if self.settings.robust_mode == "mean":
            robust_score_expr = sum(scenario_scores) / max(1, len(scenario_scores))
        else:
            kappa = 8.0
            # Smooth approximation to min(scores)
            exp_terms = [ca.exp(-kappa * s) for s in scenario_scores]
            robust_score_expr = -(1.0 / kappa) * ca.log(ca.sum1(ca.vcat(exp_terms)))

        opti.minimize(-robust_score_expr)
        opts = {"print_time": False, "ipopt.print_level": 0}
        try:
            opti.solver("ipopt", opts)
            sol = opti.solve()
            u_star = np.array(sol.value(u), dtype=float).reshape(-1)
        except Exception:
            return NEG_INF, None

        full_controls = self._expand_controls(n_steps, control_indices, u_star)
        robust_score, nominal_result = self._evaluate_controls(problem, sequence, controls=full_controls)
        if nominal_result is None:
            return NEG_INF, None
        nominal_result.final_state["hybrid_control_vector"] = [float(v) for v in full_controls]
        nominal_result.final_state["hybrid_casadi_status"] = "success"
        return robust_score, nominal_result

    def _control_indices(self, problem: MissionProblem, sequence: Sequence[str]) -> List[int]:
        n = max(0, len(sequence) - 1)
        if n <= 0:
            return []

        max_nodes = (
            self.settings.max_control_nodes_aircraft
            if problem.domain == "aircraft"
            else self.settings.max_control_nodes_spacecraft
        )
        if n <= max_nodes:
            return list(range(n))

        ranked: List[Tuple[float, int]] = []
        for idx in range(n):
            to_task_id = sequence[idx + 1]
            task = problem.tasks[to_task_id]
            if problem.domain == "aircraft":
                priority = 1.0 + float(task.value)
            else:
                action_bonus = 2.0 if task.task_type in {"observation", "downlink"} else 0.0
                priority = action_bonus + float(task.value)
            ranked.append((priority, idx))
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = sorted(idx for _, idx in ranked[:max_nodes])
        return selected

    def _expand_controls(self, n_steps: int, control_indices: Sequence[int], values: np.ndarray) -> np.ndarray:
        full = np.full(n_steps, 0.5, dtype=float)
        for local_idx, global_idx in enumerate(control_indices):
            full[global_idx] = float(values[local_idx])
        return full

    def _scenario_overrides(self, problem: MissionProblem) -> List[Dict[str, float]]:
        if problem.domain == "aircraft":
            return [{"wind_scale_multiplier": s} for s in self.settings.aircraft_scenarios]
        return [{"solar_scale_multiplier": s} for s in self.settings.spacecraft_scenarios]

    def _evaluate_controls(
        self,
        problem: MissionProblem,
        sequence: List[str],
        controls: np.ndarray | None,
    ) -> Tuple[float, PlanResult | None]:
        overrides = self._scenario_overrides(problem)
        if not overrides:
            overrides = [{}]

        scenario_scores: List[float] = []
        scenario_results: List[PlanResult] = []

        for override in overrides:
            result = self._replay_sequence(problem, sequence, controls=controls, scenario_override=override)
            if not result.solved or result.hard_violations:
                return NEG_INF, None
            scenario_scores.append(float(result.objective_score))
            scenario_results.append(result)

        if self.settings.robust_mode == "mean":
            robust_score = float(np.mean(scenario_scores))
        else:
            robust_score = float(np.min(scenario_scores))

        nominal_result = scenario_results[0]
        nominal_result.final_state["robust_score_evaluated"] = robust_score
        nominal_result.final_state["scenario_scores"] = scenario_scores
        return robust_score, nominal_result

    def _replay_sequence(
        self,
        problem: MissionProblem,
        sequence: Sequence[str],
        controls: np.ndarray | None,
        scenario_override: Dict[str, float] | None = None,
    ) -> PlanResult:
        if not sequence or sequence[0] != problem.start_task_id:
            return PlanResult(
                problem_name=problem.name,
                solved=False,
                sequence=list(sequence),
                steps=[],
                final_state=dict(problem.initial_state),
                objective_score=NEG_INF,
                hard_violations=["sequence does not start with start_task_id"],
                soft_penalty=0.0,
                visited_required=0,
                required_total=len(problem.required_set()),
                status="invalid_sequence",
            )

        state = dict(problem.initial_state)
        if scenario_override:
            if problem.domain == "aircraft":
                mult = float(scenario_override.get("wind_scale_multiplier", 1.0))
                base = float(state.get("wind_scale", 1.0))
                state["wind_scale"] = base * mult
            elif problem.domain == "spacecraft":
                mult = float(scenario_override.get("solar_scale_multiplier", 1.0))
                base = float(state.get("solar_scale", 1.0))
                state["solar_scale"] = base * mult

        current_task_id = problem.start_task_id
        path: List[str] = [current_task_id]
        steps: List[PlanStep] = []
        visited = {current_task_id}
        visited_required: set[str] = set()
        score = 0.0
        hard_violations: List[str] = []
        soft_penalty = 0.0

        required = problem.required_set()

        for step_idx, next_task_id in enumerate(sequence[1:]):
            from_task = problem.tasks[current_task_id]
            to_task = problem.tasks.get(next_task_id)
            if to_task is None:
                hard_violations.append(f"unknown task_id in sequence: {next_task_id}")
                break

            base_transition = problem.transition(current_task_id, next_task_id)
            if base_transition is None or not base_transition.feasible:
                hard_violations.append(f"infeasible transition: {current_task_id}->{next_task_id}")
                break

            transition = base_transition
            if controls is not None and step_idx < len(controls):
                control_meta = self._control_metadata(problem, float(controls[step_idx]))
                transition = Transition(
                    from_task_id=base_transition.from_task_id,
                    to_task_id=base_transition.to_task_id,
                    travel_time_s=float(base_transition.travel_time_s),
                    energy_cost_wh=float(base_transition.energy_cost_wh),
                    feasible=bool(base_transition.feasible),
                    metadata={**base_transition.metadata, **control_meta},
                )

            sim_result = problem.simulate_step(dict(state), from_task, to_task, transition)
            if sim_result is None:
                hard_violations.append(f"simulate_step failed: {current_task_id}->{next_task_id}")
                break
            state_after, step_meta = sim_result

            local_soft = 0.0
            local_hard: List[str] = []
            for constraint in problem.constraints:
                c_result = constraint.evaluate(state, state_after, from_task, to_task, transition, step_meta)
                if not c_result.passed:
                    if c_result.hard:
                        msg = f"{c_result.constraint_name}: {c_result.message}".strip()
                        local_hard.append(msg)
                    else:
                        local_soft += float(c_result.violation)
            if local_hard:
                hard_violations.extend(local_hard)
                break

            score_delta = problem.objective.delta(state, state_after, from_task, to_task, transition, step_meta)
            # Keep replay scoring consistent with UnifiedPlanner: only penalize
            # the incremental soft violation from this step.
            penalty_delta = max(0.0, local_soft)
            soft_penalty += local_soft
            score = score + score_delta - (problem.settings.soft_penalty_weight * penalty_delta)

            step = PlanStep(
                from_task_id=current_task_id,
                to_task_id=next_task_id,
                transition_time_s=float(step_meta.get("transition_time_s", transition.travel_time_s)),
                task_duration_s=float(step_meta.get("task_duration_s", to_task.duration_s)),
                start_time_s=float(step_meta.get("task_start_s", state.get("time_s", 0.0))),
                end_time_s=float(step_meta.get("task_end_s", state_after.get("time_s", 0.0))),
                cumulative_score=float(score),
                state_after=dict(state_after),
                metadata={k: v for k, v in step_meta.items()},
            )
            steps.append(step)

            state = dict(state_after)
            current_task_id = next_task_id
            path.append(current_task_id)
            visited.add(current_task_id)
            if current_task_id in required:
                visited_required.add(current_task_id)

        missing = required - visited_required
        solved = not hard_violations and (len(missing) == 0)
        if solved and problem.end_task_id is not None:
            solved = current_task_id == problem.end_task_id

        terminal_adjustment = problem.objective.terminal_adjustment(len(missing), solved)
        final_score = float(score + terminal_adjustment)

        status = "solved" if solved else ("infeasible" if hard_violations else "partial")
        if hard_violations:
            final_score = NEG_INF

        return PlanResult(
            problem_name=problem.name,
            solved=solved,
            sequence=path,
            steps=steps,
            final_state=state,
            objective_score=final_score,
            hard_violations=hard_violations,
            soft_penalty=float(soft_penalty),
            visited_required=len(visited_required),
            required_total=len(required),
            status=status,
        )

    def _control_metadata(self, problem: MissionProblem, raw_u: float) -> Dict[str, float]:
        u = _clamp(raw_u, 0.0, 1.0)
        if problem.domain == "aircraft":
            speed_scale = 0.75 + 0.50 * u
            return {"control_speed_scale": float(speed_scale)}

        # Spacecraft: one scalar control drives a consistent aggressiveness profile.
        slew_scale = 0.9 + 0.3 * u
        power_scale = 0.85 + 0.3 * u
        downlink_rate_scale = 0.85 + 0.4 * u
        return {
            "control_slew_scale": float(slew_scale),
            "control_obs_power_scale": float(power_scale),
            "control_downlink_power_scale": float(power_scale),
            "control_downlink_rate_scale": float(downlink_rate_scale),
        }
