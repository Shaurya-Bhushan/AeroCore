from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .models import MissionProblem, PlanResult, PlanStep


@dataclass
class _Node:
    current_task_id: str
    state: Dict[str, Any]
    path: List[str]
    steps: List[PlanStep]
    visited: Set[str]
    visited_required: Set[str]
    score: float
    hard_violations: List[str]
    soft_penalty: float


class UnifiedPlanner:
    """
    Shared resource-constrained task graph planner used by both domains.

    The planner is intentionally domain-agnostic: domains provide transition
    simulation, constraints, and objective weights through MissionProblem.
    """

    def __init__(self) -> None:
        self._active_start_index = 0
        self._prune_time_bucket_s = 300.0
        self._prune_energy_bucket_wh = 25.0
        self._prune_battery_bucket_wh = 10.0
        self._prune_data_buffer_bucket_mb = 100.0
        self._prune_science_bucket = 10.0

    def solve(self, problem: MissionProblem, strategy: str = "beam") -> PlanResult:
        if strategy == "multistart":
            runs = max(2, int(problem.settings.multistart_runs))
            prev_idx = self._active_start_index
            best_result: PlanResult | None = None
            for start_idx in range(runs):
                self._active_start_index = start_idx
                candidate = self.solve(problem, strategy="beam")
                if best_result is None or self._is_better(candidate, best_result):
                    best_result = candidate
            self._active_start_index = prev_idx
            assert best_result is not None
            return best_result

        if strategy not in {"beam", "greedy"}:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self._prune_time_bucket_s = max(1e-6, float(problem.settings.prune_time_bucket_s))
        self._prune_energy_bucket_wh = max(1e-6, float(problem.settings.prune_energy_bucket_wh))
        self._prune_battery_bucket_wh = max(1e-6, float(problem.settings.prune_battery_bucket_wh))
        self._prune_data_buffer_bucket_mb = max(1e-6, float(problem.settings.prune_data_buffer_bucket_mb))
        self._prune_science_bucket = max(1e-6, float(problem.settings.prune_science_bucket))

        required = problem.required_set()
        initial_node = _Node(
            current_task_id=problem.start_task_id,
            state=dict(problem.initial_state),
            path=[problem.start_task_id],
            steps=[],
            visited={problem.start_task_id},
            visited_required=set(),
            score=0.0,
            hard_violations=[],
            soft_penalty=0.0,
        )

        frontier: List[_Node] = [initial_node]
        terminals: List[_Node] = []
        expansions = 0

        for _ in range(problem.settings.max_depth):
            next_frontier: List[_Node] = []
            for node in frontier:
                if self._is_terminal(problem, node, required):
                    terminals.append(node)
                    continue

                candidates = self._candidates(problem, node, required)
                if not candidates:
                    terminals.append(node)
                    continue

                child_nodes: List[_Node] = []
                for candidate_id in candidates:
                    if expansions >= problem.settings.max_expansions:
                        break
                    child = self._expand(problem, node, candidate_id)
                    expansions += 1
                    if child is not None:
                        child_nodes.append(child)

                if not child_nodes:
                    terminals.append(node)
                    continue

                if strategy == "greedy":
                    best_child = max(child_nodes, key=lambda n: n.score)
                    next_frontier.append(best_child)
                else:
                    next_frontier.extend(child_nodes)

            if expansions >= problem.settings.max_expansions:
                break
            if not next_frontier:
                break

            if strategy == "greedy":
                frontier = [max(next_frontier, key=lambda n: n.score)]
            else:
                next_frontier = self._prune_dominated(next_frontier)
                next_frontier.sort(key=lambda n: n.score, reverse=True)
                frontier = next_frontier[: problem.settings.beam_width]

        all_nodes = terminals + frontier
        if not all_nodes:
            return PlanResult(
                problem_name=problem.name,
                solved=False,
                sequence=[problem.start_task_id],
                steps=[],
                final_state=dict(problem.initial_state),
                objective_score=float("-inf"),
                hard_violations=["Planner produced no nodes"],
                soft_penalty=0.0,
                visited_required=0,
                required_total=len(required),
                status="no_solution",
            )

        scored_nodes: List[Tuple[float, _Node]] = []
        solved_nodes: List[Tuple[float, _Node]] = []
        for node in all_nodes:
            missing = len(required - node.visited_required)
            solved = self._is_terminal(problem, node, required)
            final_score = node.score + float(problem.objective.terminal_adjustment(missing, solved))
            scored_nodes.append((final_score, node))
            if solved:
                solved_nodes.append((final_score, node))

        pool = solved_nodes if solved_nodes else scored_nodes
        pool.sort(key=lambda x: x[0], reverse=True)
        final_score, best = pool[0]

        missing_required = required - best.visited_required
        solved = len(missing_required) == 0
        if solved and problem.end_task_id is not None:
            solved = best.current_task_id == problem.end_task_id

        status = "solved" if solved else "partial"
        if expansions >= problem.settings.max_expansions:
            status = "max_expansions"

        return PlanResult(
            problem_name=problem.name,
            solved=solved,
            sequence=list(best.path),
            steps=list(best.steps),
            final_state=dict(best.state),
            objective_score=final_score,
            hard_violations=list(best.hard_violations),
            soft_penalty=float(best.soft_penalty),
            visited_required=len(best.visited_required),
            required_total=len(required),
            status=status,
        )

    def _is_better(self, lhs: PlanResult, rhs: PlanResult) -> bool:
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

    def _is_terminal(self, problem: MissionProblem, node: _Node, required: Set[str]) -> bool:
        if required - node.visited_required:
            return False
        if problem.end_task_id is None:
            return True
        return node.current_task_id == problem.end_task_id

    def _candidates(self, problem: MissionProblem, node: _Node, required: Set[str]) -> List[str]:
        if problem.candidate_fn is not None:
            # Preserve deterministic ordering when passing visited tasks to callback.
            candidate_ids = problem.candidate_fn(problem, node.state, node.current_task_id, tuple(sorted(node.visited)))
            return self._apply_start_jitter(candidate_ids, node.current_task_id)

        now = float(node.state.get("time_s", 0.0))
        visited = node.visited
        missing_required = required - node.visited_required

        candidate_ids: List[str] = []
        for task_id, task in problem.tasks.items():
            if task_id == problem.start_task_id:
                continue
            if not problem.settings.allow_revisit and task_id in visited:
                continue
            if task.window_end_s < now:
                continue
            if (
                task.window_start_s > now + problem.settings.lookahead_s
                and task_id not in missing_required
                and task_id != problem.end_task_id
            ):
                continue
            if task_id == problem.end_task_id and missing_required:
                continue
            candidate_ids.append(task_id)

        candidate_ids.sort(
            key=lambda tid: (
                0 if tid in missing_required else 1,
                problem.tasks[tid].window_start_s,
                -problem.tasks[tid].value,
            )
        )
        candidate_ids = candidate_ids[: problem.settings.candidate_limit]
        return self._apply_start_jitter(candidate_ids, node.current_task_id)

    def _apply_start_jitter(self, candidate_ids: List[str], current_task_id: str) -> List[str]:
        if self._active_start_index <= 0 or len(candidate_ids) <= 1:
            return candidate_ids
        ranked = sorted(
            candidate_ids,
            key=lambda tid: self._stable_jitter_key(tid, current_task_id, self._active_start_index),
        )
        return ranked

    @staticmethod
    def _stable_jitter_key(task_id: str, current_task_id: str, start_idx: int) -> float:
        seed = f"{task_id}|{current_task_id}|{start_idx}".encode("utf-8")
        digest = hashlib.blake2b(seed, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return float(value) / float(2**64)

    def _expand(self, problem: MissionProblem, node: _Node, candidate_id: str) -> _Node | None:
        from_task = problem.tasks[node.current_task_id]
        to_task = problem.tasks[candidate_id]
        transition = problem.transition(node.current_task_id, candidate_id)

        if transition is None or not transition.feasible:
            return None

        sim_result = problem.simulate_step(dict(node.state), from_task, to_task, transition)
        if sim_result is None:
            return None

        state_after, step_meta = sim_result
        hard_violations = list(node.hard_violations)
        soft_penalty = float(node.soft_penalty)

        for constraint in problem.constraints:
            result = constraint.evaluate(
                node.state,
                state_after,
                from_task,
                to_task,
                transition,
                step_meta,
            )
            if not result.passed:
                if result.hard:
                    msg = f"{result.constraint_name}: {result.message}".strip()
                    hard_violations.append(msg)
                    return None
                soft_penalty += result.violation

        score_delta = problem.objective.delta(
            node.state,
            state_after,
            from_task,
            to_task,
            transition,
            step_meta,
        )
        # Node score already includes prior soft penalties; only apply the incremental change.
        penalty_delta = max(0.0, soft_penalty - float(node.soft_penalty))
        new_score = node.score + score_delta - (problem.settings.soft_penalty_weight * penalty_delta)

        step = PlanStep(
            from_task_id=node.current_task_id,
            to_task_id=candidate_id,
            transition_time_s=float(step_meta.get("transition_time_s", transition.travel_time_s)),
            task_duration_s=float(step_meta.get("task_duration_s", to_task.duration_s)),
            start_time_s=float(step_meta.get("task_start_s", node.state.get("time_s", 0.0))),
            end_time_s=float(step_meta.get("task_end_s", state_after.get("time_s", 0.0))),
            cumulative_score=new_score,
            state_after=dict(state_after),
            metadata={k: v for k, v in step_meta.items() if k not in {"state"}},
        )

        visited_required = set(node.visited_required)
        if candidate_id in problem.required_set():
            visited_required.add(candidate_id)

        visited = set(node.visited)
        visited.add(candidate_id)

        return _Node(
            current_task_id=candidate_id,
            state=dict(state_after),
            path=node.path + [candidate_id],
            steps=node.steps + [step],
            visited=visited,
            visited_required=visited_required,
            score=new_score,
            hard_violations=hard_violations,
            soft_penalty=soft_penalty,
        )

    def _dominance_key(self, node: _Node) -> tuple[Any, ...]:
        time_bucket = int(float(node.state.get("time_s", 0.0)) // self._prune_time_bucket_s)
        if "energy_wh" in node.state:
            resource_bucket = int(float(node.state["energy_wh"]) // self._prune_energy_bucket_wh)
        elif "battery_wh" in node.state:
            resource_bucket = int(float(node.state["battery_wh"]) // self._prune_battery_bucket_wh)
        else:
            resource_bucket = 0
        # Preserve mission-critical diversity for scheduling problems:
        # nodes with very different data/science backlog should not dominate each
        # other solely on time and battery.
        data_bucket: int | None = None
        if "data_buffer_mb" in node.state:
            data_bucket = int(float(node.state["data_buffer_mb"]) // self._prune_data_buffer_bucket_mb)
        science_bucket: int | None = None
        if "science_buffer_value" in node.state:
            science_bucket = int(float(node.state["science_buffer_value"]) // self._prune_science_bucket)
        req_key = tuple(sorted(node.visited_required))
        return (
            node.current_task_id,
            req_key,
            time_bucket,
            resource_bucket,
            data_bucket,
            science_bucket,
        )

    def _prune_dominated(self, nodes: List[_Node]) -> List[_Node]:
        best_by_key: Dict[tuple[Any, ...], _Node] = {}
        for node in nodes:
            key = self._dominance_key(node)
            prev = best_by_key.get(key)
            if prev is None:
                best_by_key[key] = node
                continue
            if node.score > prev.score:
                best_by_key[key] = node
                continue
            if node.score == prev.score and len(node.visited_required) > len(prev.visited_required):
                best_by_key[key] = node
        return list(best_by_key.values())
