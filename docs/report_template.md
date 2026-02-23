# AeroHack Technical Report Template

## 1. Problem Statement
- Aircraft mission requirements.
- Spacecraft 7-day mission requirements.
- Unified architecture requirement.

## 2. Unified Planning Formulation
### 2.1 Task-Transition-Resource Model
Define:
- Decision variables over task sequence and start times.
- State vector with time/resource components.

### 2.2 Shared Constraints
- Time window feasibility.
- Resource bounds.
- Domain-specific hard constraints via shared interface.

### 2.3 Shared Objective
\[
J = \sum_k (w_v V_k + w_d D_k - w_t \Delta t_k - w_e \Delta E_k)
\]
with terminal completion bonuses/penalties.

## 3. Aircraft Module
- Kinematic model with wind field.
- Maneuver/turn-rate and geofence formulation.
- Energy model and endurance constraints.

## 4. Spacecraft Module
- Orbit/visibility opportunity generation.
- Slew-rate, battery proxy, data-buffer/downlink constraints.
- Duty-cycle proxy (max ops per orbit).

## 5. Validation & Robustness
- Monte Carlo setup and parameter perturbations.
- Baseline (greedy) vs beam results.
- Stress scenarios and failure modes.

## 6. Results
- Aircraft metrics table and constraint summary.
- Spacecraft metrics table and constraint summary.
- Headline results and runtime.

## 7. Limitations and Next Steps
- Model simplifications.
- Planned fidelity upgrades.
