# Path Planning Scaffold

Python scaffold for benchmarking three Ackermann path planners under a consistent robot model (strict minimum turning radius and oriented rectangular footprint). Built to be hackable, lightweight, and fair for research experiments.

## Why use this scaffold

- One robot model for all planners: identical kinematics (Ackermann bicycle), turning limits, and oriented-box collision model for apples-to-apples comparisons.
- Minimal dependencies: NumPy-only core, no ROS; DQN guidance runs with or without PyTorch (deterministic NumPy fallback).
- Ready-to-run demo scenarios plus clear stats (path length, expansions, timing, node counts) for quick baselining.
- Modular library: small, readable modules for motion primitives, heuristics, collision checking, DQN guidance, and planners.
- Extensible hooks: swap heuristics, change motion primitives, plug in real DQN/RL models, or tighten safety via map inflation.

## Included planners

- **Hybrid A\***: Lattice-based search over SE(2) using fixed motion primitives (5 steering bins x forward/reverse). Reeds-Shepp-inspired admissible heuristic, cusp penalty for direction changes, and oriented box collision sampling along each primitive.
- **DQN-guided multi-heuristic Hybrid A\***: Two synchronized queues (anchor + DQN) maintain completeness while biasing expansions with a learned value estimate and action-ranking policy. Optional `dqn_top_k` reduces branching for speed.
- **Kinodynamic RRT\***: Samples poses (with goal bias), forward-simulates bicycle dynamics with bounded steering/velocity, rewires neighbors within a radius, and attempts straight-line goal connection when close.

## Robot and environment model

- **Ackermann params**: wheelbase `0.6 m`, minimum turning radius `1.6 m` -> `|delta| <= 0.359 rad` (`AckermannParams`).
- **Footprint**: oriented rectangle `0.924 m x 0.740 m` (`OrientedBoxFootprint`), shared by all planners.
- **Collision checking**: oriented bounding-box sampling plus interpolated poses along each motion (`geometry.motion_collides`).
- **Maps**: occupancy grid (`GridMap`) with world/grid conversions, inflation, random free-state sampling, and robot-aligned occupancy patches for DQN features.

## Project layout

- `pathplan/__init__.py` - library exports (planners, robot params/state, footprint, grid map).
- `pathplan/common.py` - utility math (angles, clamp, distance, lerp).
- `pathplan/geometry.py` - footprint geometry, OBB collision, pose interpolation.
- `pathplan/map_utils.py` - occupancy grid, inflation, sampling, local patches for DQN-guided features.
- `pathplan/robot.py` - Ackermann params/state, primitive propagation, forward simulation.
- `pathplan/primitives.py` - motion primitives and costs (10 defaults: 5 steering bins x {forward, reverse}).
- `pathplan/heuristics.py` - Euclidean + heading penalty and Reeds-Shepp-style admissible lower bound.
- `pathplan/hybrid_a_star.py` - classic Hybrid A* with cusp penalty and collision-checked primitives.
- `pathplan/dqn_hybrid_a_star.py` - multi-queue Hybrid A* with DQN guidance for heuristics and action ordering.
- `pathplan/dqn_models.py` - lightweight DQN-style value/policy using local occupancy patches (torch-optional stubs).
- `pathplan/rrt_star.py` - kinodynamic RRT* with forward simulation, rewiring, and soft goal connection.
- `examples/run_demo.py` - runnable scenarios exercising all three planners.
- `requirements.txt` - NumPy dependency pin.

## Quickstart

```bash
pip install -r requirements.txt
python -m examples.run_demo
```

The demo builds synthetic maps and runs all planners on each:
- Tight corridor with partial block (`make_corridor_map`)
- Open field (`make_open_map`)
Prints success, run time, path length, and search effort (expansions or node count). Add `make_parking_map` for a parking-bay stress test.

## Train a Torch DQN guidance (CUDA)

- Install training deps: `pip install -r requirements-train.txt` (installs PyTorch).
- Run training (uses CUDA when available): `python -m examples.train_dqn --device cuda`.
- Plug the trained model into the planner:
  ```python
  from pathplan import DQNHybridAStarPlanner, TorchDQNGuidance, AckermannParams
  planner = DQNHybridAStarPlanner(grid_map, footprint, AckermannParams())
  planner.dqn = TorchDQNGuidance(planner.params, "examples/outputs/dqn_guidance.pt", primitives=planner.primitives)
  ```

## How each planner works (key settings)

- **Hybrid A\*** (`HybridAStarPlanner`):
  - Discretization: XY resolution `0.1 m`, `72` heading bins.
  - Actions: 10 primitives (`0.3 m` step), reverse motions carry a slight cost multiplier; cusp penalty `0.2`.
  - Heuristic: admissible Reeds-Shepp lower bound; goal tolerance `0.3 m` and `15 deg`.
  - Collision: oriented-box sampling along each primitive (`collision_step=0.1 m`).

- **DQN-guided Hybrid A\*** (`DQNHybridAStarPlanner`):
  - Two queues: anchor uses admissible heuristic; DQN queue uses learned value + policy-ranked actions.
  - DQN features from a local occupancy patch (robot frame), goal distance/heading, and occupancy statistics.
  - Configurable `dqn_top_k` to limit branching; `dqn_weight` to scale the learned heuristic.

- **RRT\*** (`RRTStarPlanner`):
  - Forward simulate bicycle model for `0.6 s` at `0.8 m/s`; steering bounded by `max_steer`.
  - Goal bias `20%`, neighbor radius `1.5 m`, connect-to-goal threshold `1.0 m`.
  - Rewire neighbors when a cheaper kinodynamic connection is found; straight-line goal connection is collision-checked.

## Example: plan on your own map

```python
import numpy as np
from pathplan import (
    AckermannParams, AckermannState, GridMap,
    HybridAStarPlanner, OrientedBoxFootprint,
)

# Build a simple map (0 = free, 1 = occupied)
grid = np.zeros((80, 120), dtype=np.uint8)
grid[20:24, 30:90] = 1  # an obstacle band
grid_map = GridMap(grid, resolution=0.05)

params = AckermannParams()
footprint = OrientedBoxFootprint(length=0.924, width=0.740)
planner = HybridAStarPlanner(grid_map, footprint, params)

start = AckermannState(1.0, 1.0, 0.0)
goal = AckermannState(5.0, 2.5, 0.0)
path, stats = planner.plan(start, goal, timeout=4.0)
print("Success:", bool(path), "Expansions:", stats["expansions"], "Path length:", stats["path_length"])
```

Swap in `DQNHybridAStarPlanner` for faster searches, or `RRTStarPlanner` for kinodynamic sampling-based planning.

## Extending

- **Safety margin**: `GridMap.inflate(margin)` to pad obstacles before planning.
- **Actions**: customize steering bins/step length via `pathplan.primitives.default_primitives` or your own list.
- **Heuristics**: replace `admissible_heuristic` for domain-specific lower bounds.
- **DQN guidance**: drop in real models inside `dqn_models.DQNGuidance` (value/policy) or adjust patch size/features.
- **Dynamics**: tweak `AckermannParams` for different vehicles (wheelbase, turning radius, velocity bounds).

## Outputs and metrics

- **Hybrid A\***: `path` (list of poses), `stats` with `path_length`, `cusps` (placeholder), `expansions`, `time`, `timed_out`.
- **DQN-guided Hybrid**: adds `expansions_anchor`, `expansions_dqn`, `expansions_total`.
- **RRT\***: `path`, `nodes`, `iterations`, `time`, `success`.

Use these to benchmark planners or to tune action sets and heuristics.

## Reasons to choose this repo

- Fair, reproducible comparisons: identical collision model, vehicle limits, and map handling across planners.
- Clarity-first code: each module is short and ready to modify - ideal for teaching, prototyping, or papers.
- Learning-friendly: DQN guidance hooks are present but optional; runs identically with just NumPy.
- Lightweight and portable: no ROS; synthetic maps mean zero external assets.
- Research-ready: exposes stats, tolerances, and hyperparameters so you can sweep settings or plug into your experiment harness.
