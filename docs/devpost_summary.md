# AeroCore: The Unified Aerospace Optimization Kernel

## 💡 Inspiration
The aerospace industry is fundamentally fractured. When designing a Mars rover entry sequence or coordinating a stratospheric balloon transitioning to deployment, engineers are forced to stitch together completely separate toolchains. UAV teams use flight-planning heuristics; satellite teams rely on heavyweight orbital mechanics software (STK/GMAT). 

But beneath the surface, both domains share identical mathematical DNA: they are constraint-satisfaction and optimal-control problems. We asked ourselves: *Why build two systems when one unified optimization kernel can rule them all?* 

We were inspired to build **AeroCore**—not just a pair of disjointed scripts, but a truly domain-agnostic Hybrid Optimal-Control engine. We aimed for absolute engineering discipline: strict feasibility margins, deterministic execution, and automated validation, ensuring that our flight paths and downlinks aren't just "good enough"—they're mathematically certified.

## 🚀 What it does
AeroCore is a production-grade, domain-agnostic mission planning framework that elegantly optimizes multi-objective trajectories for both atmospheric aircraft and LEO spacecraft.

🛩️ **Aircraft Module (Atmospheric Planning):**
- Plans geofence-aware routes using dynamic visibility-graph search with rigorous safety margins.
- Ingests harmonic-driven spatial wind models and computes nonlinear aerodynamic power (parasite + induced drag, load-factor effects, climb work, propulsive efficiency).
- Respects strict maneuvering limits (turn-rate, bank-angle, climb-rate feasibility).
- Generates ~121-minute optimal trajectories with zero hard-constraint violations and 100% geofence compliance.

🛰️ **Spacecraft Module (Orbital Operations):**
- Schedules 7-day observation and ground-station downlink sequences.
- Propagates real orbital mechanics: Two-body ECI formulas + true J2 Earth oblateness drift.
- Computes dynamic Sun vectors (accounting for mission epochs) for accurate eclipse/solar-charging cycles.
- Employs Diminishing Science Returns to penalize target farming and promote mission diversity.
- Achieved an optimal 235.68 delivered-science score across 53 executed actions with 0 hard battery-limit violations.

## 🛠️ How we built it
We aggressively separated the physics from the optimization logic to build a cohesive **three-layer hybrid architecture:**

1. **The Unified Formulation:**
   Every problem is distilled into a directed Task Graph constraint API. Whether it's an aircraft banking to a waypoint or a CubeSat slewing to a ground station, the engine simply evaluates `State + Transition = Next State` under `Constraint.evaluate()`.
   
2. **The Hybrid OCP Engine:**
   We didn't settle for pure discrete search. We coupled a Discrete Search (Beam/Greedy heuristic) with a **Hybrid Optimal-Control Engine** (`UnifiedHybridOCPEngine`). This pipeline utilizes direct-shooting continuous control refinement (via SciPy/CasADi) on top of mission sequences to optimize exact inter-task transition parameters.
   We explicitly did not force a pure MILP/CP-SAT stack because key aircraft/orbital dynamics here are nonlinear and continuous; hybrid OCP preserves physical fidelity while keeping one shared architecture across both domains.

3. **Strict Validation & Certification Pipeline:**
   We built a self-certifying pipeline that generates independent constraint audits:
   - **Constraint Slack Certification:** It doesn't just say a constraint passed; it calculates the exact slack, identifying the "binding" constraints dictating mission limits (e.g., UAV turn-rates maxed out at exactly `0.0698 rad/s`).
   - **Pareto Trade-Studies:** Automatically sweeps objectives to generate complex Pareto Frontiers (e.g., Mission Time vs. Energy Used).
   - **Deterministic Hash-Replay:** Ensures fixed-seed reproducibility vital for aerospace compliance.

## ⚔️ Challenges we ran into
- **The "Target Farming" Exploit:** Our early spacecraft scheduler found a loophole: staring at the highest-value target indefinitely. We fixed this by introducing a *Science Diminishing Returns* function with custom decay formulas to simulate real operational constraints.
- **Physics Fidelity Limits:** Hardcoding the Sun to the $+x$ vector led to inaccurate eclipse states. We completely re-engineered the celestial mechanics to dynamically propagate the Sun's position relative to the Julian epoch, synchronizing the power model with true orbital eclipses.
- **Geospatial Precision:** Managing complex, concave No-Fly Zones. Manual polygon-inflation algorithms caused self-intersecting geometries. We integrated sophisticated computational geometry (Shapely) buffers to guarantee absolute geofence clearance.

## 🏆 Accomplishments that we're proud of
- **Dual Domain Mastery:** Delivering a single architectural codebase that commands both a drone in a windstorm and a satellite in orbit.
- **100% Monte Carlo Robustness:** Built a full validation suite generating randomized wind/battery/solar perturbations, achieving a 1.0 success rate across all stress tests.
- **Zero Constraint Violations:** 100% hard constraint feasibility. No crashed drones, no dead batteries.
- **Data-Driven Transparency:** We emit industry-standard data formats (KMLs, CSVs, JSONs), complete visual timelines (Gantt, Resource evolution plots), and a fully automated `judging_scorecard.json`.

## 📚 What we learned
- **Slack is the Ultimate Metric:** Binary constraint checks (Pass/Fail) hide the true limits of your system. Generating constraint slack and pseudo-multipliers taught us exactly which subsystems act as bottlenecks.
- **Heuristics + OCP = Speed & Precision:** Combining a discrete Beam search with continuous parameter optimization offers the best of both worlds—rapid schedule generation coupled with precise continuous energy management.
- **Determinism is Non-Negotiable:** Engineering a mission planner that generates varying results across identical runs is unacceptable in aerospace. Isolating scopes and enforcing seed tracking became our gold standard.

## 🔮 What's next for AeroCore
- **Multi-Agent Constellations:** Upgrading the framework to orchestrate heterogeneous fleets—drone swarms and LEO constellations—simultaneously negotiating distributed tasks.
- **Hardware-in-the-Loop (HITL):** Exporting generated flight plans directly to ArduPilot/PX4 SITL instances for real-time physics validation.
- **Atmospheric Drag & Precision SGP4:** Injecting comprehensive drag profiles and transitioning from two-body J2 to full SGP4 TLE orbital tracking.

### 📊 Headline Metrics (Default Configuration)
| Metric | Value | 
|--------|-------|
| **Aircraft Mission Time** | 121.02 min |
| **Aircraft Energy Used** | 2514.93 Wh |
| **Aircraft Geofence Margin** | Verified Clear |
| **Spacecraft Delivered Science** | 235.68 |
| **Spacecraft Battery Min limit** | Respected (66.80 Wh lower bound) |
| **Robustness (Monte Carlo)** | 100% Success |
| **Constraint Violations** | 0 |
| **Pipeline Reproducibility** | Hash-Certified Pass |

**AeroCore proves that aerospace planning doesn't need to be siloed. With the right abstraction and relentless physics validation, we can command the atmosphere and the orbit from a single, unified kernel.**
