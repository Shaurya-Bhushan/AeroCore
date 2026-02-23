from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .models import MissionTask, Transition


class Objective:
    def delta(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> float:
        raise NotImplementedError

    def terminal_adjustment(self, missing_required: int, solved: bool) -> float:
        return 0.0


@dataclass
class WeightedObjective(Objective):
    value_weight: float = 1.0
    delivered_weight: float = 1.0
    time_weight: float = 0.01
    energy_weight: float = 0.01
    required_completion_bonus: float = 0.0
    unfinished_required_penalty: float = 1000.0
    unsolved_penalty: float = 0.0

    def delta(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        from_task: MissionTask,
        to_task: MissionTask,
        transition: Transition,
        step_meta: Dict[str, Any],
    ) -> float:
        delta_time = float(state_after.get("time_s", 0.0)) - float(state_before.get("time_s", 0.0))
        delta_energy_used = float(state_after.get("total_energy_used_wh", 0.0)) - float(
            state_before.get("total_energy_used_wh", 0.0)
        )
        delta_delivered = float(state_after.get("delivered_science", 0.0)) - float(
            state_before.get("delivered_science", 0.0)
        )
        return (
            self.value_weight * float(to_task.value)
            + self.delivered_weight * delta_delivered
            - self.time_weight * delta_time
            - self.energy_weight * delta_energy_used
        )

    def terminal_adjustment(self, missing_required: int, solved: bool) -> float:
        if solved:
            return self.required_completion_bonus
        return -self.unfinished_required_penalty * float(missing_required) - self.unsolved_penalty
