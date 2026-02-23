from __future__ import annotations

import unittest
from typing import Dict, List, Sequence

from src.core import (
    CallableConstraint,
    MissionProblem,
    MissionTask,
    PlannerSettings,
    Transition,
    UnifiedHybridOCPEngine,
    UnifiedPlanner,
    WeightedObjective,
)


def _simple_simulate(
    state: Dict[str, float],
    from_task: MissionTask,
    to_task: MissionTask,
    transition: Transition,
):
    now = float(state.get("time_s", 0.0))
    transition_time = float(transition.travel_time_s)
    task_start = now + transition_time
    task_end = task_start + float(to_task.duration_s)
    next_state = dict(state)
    next_state["time_s"] = task_end
    return next_state, {
        "transition_time_s": transition_time,
        "task_duration_s": float(to_task.duration_s),
        "task_start_s": task_start,
        "task_end_s": task_end,
    }


def _build_fully_connected_transitions(tasks: Dict[str, MissionTask]) -> Dict[tuple[str, str], Transition]:
    transitions: Dict[tuple[str, str], Transition] = {}
    ids = list(tasks.keys())
    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            transitions[(src, dst)] = Transition(
                from_task_id=src,
                to_task_id=dst,
                travel_time_s=1.0,
                energy_cost_wh=0.0,
                feasible=True,
            )
    return transitions


class PlannerRegressionTests(unittest.TestCase):
    def test_soft_penalty_applies_incrementally(self) -> None:
        tasks = {
            "start": MissionTask("start", "toy", "start", 0.0, 100.0, 0.0, required=False),
            "wp": MissionTask("wp", "toy", "waypoint", 0.0, 100.0, 0.0, required=True),
            "end": MissionTask("end", "toy", "end", 0.0, 100.0, 0.0, required=False),
        }
        transitions = _build_fully_connected_transitions(tasks)

        def soft_once_constraint(
            state_before: Dict[str, float],
            state_after: Dict[str, float],
            from_task: MissionTask,
            to_task: MissionTask,
            transition: Transition,
            step_meta: Dict[str, float],
        ) -> tuple[bool, str, float]:
            if to_task.task_id == "wp":
                return False, "soft_cost_wp", 1.0
            return True, "", 0.0

        problem = MissionProblem(
            name="toy_soft_penalty",
            domain="toy",
            tasks=tasks,
            transitions=transitions,
            start_task_id="start",
            end_task_id="end",
            required_task_ids=["wp"],
            initial_state={"time_s": 0.0},
            simulate_step=_simple_simulate,
            constraints=[CallableConstraint(fn=soft_once_constraint, hard=False, name="soft_once")],
            objective=WeightedObjective(
                value_weight=0.0,
                delivered_weight=0.0,
                time_weight=0.0,
                energy_weight=0.0,
                required_completion_bonus=0.0,
                unfinished_required_penalty=0.0,
                unsolved_penalty=0.0,
            ),
            settings=PlannerSettings(
                beam_width=8,
                max_expansions=64,
                max_depth=8,
                candidate_limit=8,
                soft_penalty_weight=1.0,
                allow_revisit=False,
            ),
        )

        result = UnifiedPlanner().solve(problem, strategy="beam")

        self.assertTrue(result.solved)
        self.assertEqual(result.sequence, ["start", "wp", "end"])
        self.assertAlmostEqual(float(result.soft_penalty), 1.0, places=6)
        self.assertAlmostEqual(float(result.objective_score), -1.0, places=6)

    def test_candidate_callback_receives_deterministic_sorted_visited(self) -> None:
        tasks = {
            "start": MissionTask("start", "toy", "start", 0.0, 200.0, 0.0, required=False),
            "a": MissionTask("a", "toy", "waypoint", 0.0, 200.0, 0.0, required=True),
            "b": MissionTask("b", "toy", "waypoint", 0.0, 200.0, 0.0, required=True),
            "end": MissionTask("end", "toy", "end", 0.0, 200.0, 0.0, required=False),
        }
        transitions = _build_fully_connected_transitions(tasks)
        visited_inputs: List[tuple[str, ...]] = []

        def candidate_fn(
            problem: MissionProblem,
            state: Dict[str, float],
            current_task_id: str,
            visited: Sequence[str],
        ) -> List[str]:
            visited_tuple = tuple(visited)
            visited_inputs.append(visited_tuple)
            missing_required = problem.required_set() - set(visited_tuple)
            candidates = [tid for tid in tasks.keys() if tid != current_task_id and tid not in visited_tuple]
            if missing_required:
                candidates = [tid for tid in candidates if tid in missing_required]
            elif problem.end_task_id is not None:
                candidates = [tid for tid in candidates if tid == problem.end_task_id]
            return sorted(candidates)

        problem = MissionProblem(
            name="toy_candidate_sorted",
            domain="toy",
            tasks=tasks,
            transitions=transitions,
            start_task_id="start",
            end_task_id="end",
            required_task_ids=["a", "b"],
            initial_state={"time_s": 0.0},
            simulate_step=_simple_simulate,
            constraints=[],
            objective=WeightedObjective(
                value_weight=0.0,
                delivered_weight=0.0,
                time_weight=0.0,
                energy_weight=0.0,
                required_completion_bonus=0.0,
                unfinished_required_penalty=0.0,
                unsolved_penalty=0.0,
            ),
            settings=PlannerSettings(
                beam_width=8,
                max_expansions=128,
                max_depth=8,
                candidate_limit=8,
                allow_revisit=False,
            ),
            candidate_fn=candidate_fn,
        )

        result = UnifiedPlanner().solve(problem, strategy="beam")
        self.assertTrue(result.solved)
        self.assertEqual(result.sequence[-1], "end")
        self.assertGreaterEqual(len(visited_inputs), 1)
        for visited in visited_inputs:
            self.assertEqual(visited, tuple(sorted(visited)))

    def test_hybrid_replay_soft_penalty_matches_incremental_logic(self) -> None:
        tasks = {
            "start": MissionTask("start", "toy", "start", 0.0, 100.0, 0.0, required=False),
            "wp": MissionTask("wp", "toy", "waypoint", 0.0, 100.0, 0.0, required=True),
            "end": MissionTask("end", "toy", "end", 0.0, 100.0, 0.0, required=False),
        }
        transitions = _build_fully_connected_transitions(tasks)

        def soft_once_constraint(
            state_before: Dict[str, float],
            state_after: Dict[str, float],
            from_task: MissionTask,
            to_task: MissionTask,
            transition: Transition,
            step_meta: Dict[str, float],
        ) -> tuple[bool, str, float]:
            if to_task.task_id == "wp":
                return False, "soft_cost_wp", 1.0
            return True, "", 0.0

        problem = MissionProblem(
            name="toy_soft_penalty_replay",
            domain="toy",
            tasks=tasks,
            transitions=transitions,
            start_task_id="start",
            end_task_id="end",
            required_task_ids=["wp"],
            initial_state={"time_s": 0.0},
            simulate_step=_simple_simulate,
            constraints=[CallableConstraint(fn=soft_once_constraint, hard=False, name="soft_once")],
            objective=WeightedObjective(
                value_weight=0.0,
                delivered_weight=0.0,
                time_weight=0.0,
                energy_weight=0.0,
                required_completion_bonus=0.0,
                unfinished_required_penalty=0.0,
                unsolved_penalty=0.0,
            ),
            settings=PlannerSettings(
                beam_width=8,
                max_expansions=64,
                max_depth=8,
                candidate_limit=8,
                soft_penalty_weight=1.0,
                allow_revisit=False,
            ),
        )

        replay = UnifiedHybridOCPEngine()._replay_sequence(
            problem=problem,
            sequence=["start", "wp", "end"],
            controls=None,
            scenario_override=None,
        )
        self.assertTrue(replay.solved)
        self.assertAlmostEqual(float(replay.soft_penalty), 1.0, places=6)
        self.assertAlmostEqual(float(replay.objective_score), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
