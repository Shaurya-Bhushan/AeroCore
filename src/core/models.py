from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


State = Dict[str, Any]


@dataclass(frozen=True)
class MissionTask:
    task_id: str
    domain: str
    task_type: str
    window_start_s: float
    window_end_s: float
    duration_s: float
    value: float = 0.0
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    from_task_id: str
    to_task_id: str
    travel_time_s: float
    energy_cost_wh: float
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintResult:
    constraint_name: str
    passed: bool
    hard: bool
    message: str = ""
    violation: float = 0.0


@dataclass
class PlanStep:
    from_task_id: str
    to_task_id: str
    transition_time_s: float
    task_duration_s: float
    start_time_s: float
    end_time_s: float
    cumulative_score: float
    state_after: State
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    problem_name: str
    solved: bool
    sequence: List[str]
    steps: List[PlanStep]
    final_state: State
    objective_score: float
    hard_violations: List[str]
    soft_penalty: float
    visited_required: int
    required_total: int
    status: str


@dataclass
class PlannerSettings:
    beam_width: int = 64
    max_expansions: int = 50000
    max_depth: int = 200
    candidate_limit: int = 32
    lookahead_s: float = 3600.0
    soft_penalty_weight: float = 10.0
    allow_revisit: bool = False
    multistart_runs: int = 1
    # Dominance-pruning buckets (domain-tunable).
    prune_time_bucket_s: float = 300.0
    prune_energy_bucket_wh: float = 25.0
    prune_battery_bucket_wh: float = 10.0
    prune_data_buffer_bucket_mb: float = 100.0
    prune_science_bucket: float = 10.0


SimulationStepFn = Callable[
    [State, MissionTask, MissionTask, Transition],
    Optional[Tuple[State, Dict[str, Any]]],
]
CandidateFn = Callable[["MissionProblem", State, str, Sequence[str]], List[str]]


@dataclass
class MissionProblem:
    name: str
    domain: str
    tasks: Dict[str, MissionTask]
    transitions: Dict[Tuple[str, str], Transition]
    start_task_id: str
    end_task_id: Optional[str]
    required_task_ids: Sequence[str]
    initial_state: State
    simulate_step: SimulationStepFn
    constraints: Sequence[Any]
    objective: Any
    settings: PlannerSettings = field(default_factory=PlannerSettings)
    candidate_fn: Optional[CandidateFn] = None

    def required_set(self) -> set[str]:
        return set(self.required_task_ids)

    def transition(self, from_task_id: str, to_task_id: str) -> Optional[Transition]:
        return self.transitions.get((from_task_id, to_task_id))
