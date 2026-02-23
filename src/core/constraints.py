from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .models import ConstraintResult, MissionTask, Transition


class Constraint:
    name: str
    hard: bool

    def evaluate(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> ConstraintResult:
        raise NotImplementedError


@dataclass
class TaskWindowConstraint(Constraint):
    name: str = "task_window"
    hard: bool = True

    def evaluate(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> ConstraintResult:
        task_start = float(step_meta.get("task_start_s", state_before.get("time_s", 0.0)))
        task_end = float(step_meta.get("task_end_s", task_start + to_task.duration_s))
        ok = task_start >= to_task.window_start_s and task_end <= to_task.window_end_s
        msg = ""
        if not ok:
            msg = (
                f"{to_task.task_id} outside window "
                f"[{to_task.window_start_s:.1f}, {to_task.window_end_s:.1f}]"
            )
        return ConstraintResult(self.name, ok, self.hard, msg, 0.0 if ok else 1.0)


@dataclass
class RangeConstraint(Constraint):
    key: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    name: str = "range"
    hard: bool = True

    def evaluate(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> ConstraintResult:
        value = float(state_after.get(self.key, 0.0))
        ok = True
        if self.min_value is not None and value < self.min_value:
            ok = False
        if self.max_value is not None and value > self.max_value:
            ok = False
        msg = ""
        if not ok:
            msg = (
                f"{self.key}={value:.3f} out of bounds "
                f"[{self.min_value if self.min_value is not None else '-inf'}, "
                f"{self.max_value if self.max_value is not None else 'inf'}]"
            )
        return ConstraintResult(self.name, ok, self.hard, msg, 0.0 if ok else 1.0)


@dataclass
class MaxOpsPerOrbitConstraint(Constraint):
    max_ops: int
    name: str = "ops_per_orbit"
    hard: bool = True

    def evaluate(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> ConstraintResult:
        ops = int(state_after.get("ops_this_orbit", 0))
        ok = ops <= self.max_ops
        msg = "" if ok else f"ops_this_orbit={ops} exceeds {self.max_ops}"
        return ConstraintResult(self.name, ok, self.hard, msg, 0.0 if ok else float(ops - self.max_ops))


@dataclass
class CallableConstraint(Constraint):
    fn: Callable[
        [
            Dict[str, Any],
            Dict[str, Any],
            MissionTask,
            MissionTask,
            Transition,
            Dict[str, Any],
        ],
        tuple[bool, str, float],
    ]
    name: str = "callable_constraint"
    hard: bool = True

    def evaluate(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> ConstraintResult:
        ok, msg, violation = self.fn(
            state_before,
            state_after,
            from_task,
            to_task,
            transition,
            step_meta,
        )
        return ConstraintResult(self.name, ok, self.hard, msg, violation)
