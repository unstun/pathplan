# Path Planning Scaffold

Python scaffold for Ackermann planners (Hybrid A*, DQN-guided Hybrid A*, Artificial Potential Field, kinodynamic RRT*) with a shared robot model and grid-based collision checker. Everything is NumPy-first, ROS-free, and kept small so you can swap pieces without digging through a framework.

## What’s inside (code map)

- `pathplan/` core: collision geometry (`geometry.py`), occupancy grid (`map_utils.py`), robot model and propagation (`robot.py`), motion primitives (`primitives.py`), heuristics (`heuristics.py`), planners (`hybrid_a_star.py`, `dqn_hybrid_a_star.py`, `rrt_star.py`, `apf.py`), DQN guidance stubs/torch loader (`dqn_models.py`), and exports (`__init__.py`).
- `examples/run_demo.py` small scenarios (corridor, open field; parking helper included).
- `examples/forest_scene.py` dense-forest benchmarks (4 presets) with plots + CSVs in timestamped folders.
- `examples/train_dqn.py` minimal DQN training loop for guidance (PyTorch).
- `requirements.txt` (NumPy) and `requirements-train.txt` (NumPy + PyTorch) keep dependencies explicit; `matplotlib` is optional for plots.

## Robot + collision model (ground truth from code)

- Ackermann params: `wheelbase=0.6 m`, `min_turn_radius=1.6 m` → `|steer| ≤ 0.359 rad` (`AckermannParams`).
- Footprint: oriented rectangle `0.924 m x 0.740 m` shared by every planner (`OrientedBoxFootprint`).
- Collision: `GridFootprintChecker` precomputes, per heading bin, all grid-cell centers that lie inside the footprint (optional `padding`). `collides_pose` tests those centers; `collides_path`/`motion_collides` sample poses along a segment at `collision_step`. This center-based test can visually graze obstacles; add padding or inflate the map for stricter clearance.
- Maps: `GridMap` with world/grid transforms, random free-state sampling, occupancy patches for guidance, and simple inflation (`inflate`) for safety buffers.

## Planners (defaults in constructors)

- **Hybrid A\*** (`HybridAStarPlanner`): 10 primitives (`default_primitives`, step `0.3 m`, reverse weight `1.2`), cusp penalty `0.2`, heading bins `72`, `collision_step=0.1`, goal tolerances `0.3 m / 15°`, heuristic = Reeds-Shepp-style lower bound. `xy_resolution` defaults to the map resolution unless overridden (forests use coarser `0.60 m`, 48 bins).
- **DQN Hybrid A\*** (`DQNHybridAStarPlanner`): same lattice, dual queues (anchor + DQN). DQN branch uses lightweight hand-tuned `DQNGuidance` unless you swap in `TorchDQNGuidance`. Config: `dqn_top_k`, `dqn_weight`, `anchor_inflation`, `dqn_lead_threshold`, `max_dqn_streak`.
- **RRT\*** (`RRTStarPlanner`): forward-simulate bicycle for `step_time=0.6 s` at `velocity=0.8 m/s`, goal bias `0.2`, neighbor radius `1.5 m`, goal tolerances `0.2 m / 15°`, connect threshold `1.0 m`, optional rewiring, `collision_step=map.resolution*0.5`, `theta_bins=72`, `lazy_collision` switch for goal-only checks.
- **Artificial Potential Field** (`APFPlanner`): attractive + repulsive potentials over a precomputed obstacle distance map, curvature-limited heading rate, `collision_step` default `map.resolution*0.5` unless set. Can run coarse (point) collision via `coarse_collision=True` or use the footprint checker.

## Install & run

```bash
pip install -r requirements.txt         # NumPy core
pip install matplotlib                  # needed for plotting (optional)

# Quick demo (corridor + open field)
python -m examples.run_demo
```

Forest benchmarks (writes `examples/outputs/<timestamp>/`):
```bash
python -m examples.forest_scene --variant all
# or: --variant large_map | small_map | large_gap | small_gap
```

## Torch DQN guidance (optional)

- Install training deps: `pip install -r requirements-train.txt` (PyTorch).
- Train: `python -m examples.train_dqn --device cuda` (auto-falls back to CPU).
- Use in a planner:
  ```python
  from pathplan import AckermannParams, DQNHybridAStarPlanner, OrientedBoxFootprint, TorchDQNGuidance
  planner = DQNHybridAStarPlanner(grid_map, OrientedBoxFootprint(0.924, 0.740), AckermannParams())
  planner.dqn = TorchDQNGuidance(planner.params, "examples/outputs/dqn_guidance.pt", primitives=planner.primitives)
  ```

## Minimal API example

```python
import numpy as np
from pathplan import AckermannParams, AckermannState, GridMap, HybridAStarPlanner, OrientedBoxFootprint

grid = np.zeros((80, 120), dtype=np.uint8)
grid[20:24, 30:90] = 1  # obstacle band
grid_map = GridMap(grid, resolution=0.05)
params = AckermannParams()
footprint = OrientedBoxFootprint(length=0.924, width=0.740)
planner = HybridAStarPlanner(grid_map, footprint, params)

start = AckermannState(1.0, 1.0, 0.0)
goal = AckermannState(5.0, 2.5, 0.0)
path, stats = planner.plan(start, goal, timeout=4.0)
print("Success:", bool(path), "Expansions:", stats["expansions"], "Path length:", stats["path_length"])
```

Swap in `DQNHybridAStarPlanner` for guided search or `RRTStarPlanner` for sampling-based planning.

## Extending / safety knobs

- Inflate obstacles: `GridMap.inflate(margin)` or add `padding` to `GridFootprintChecker` to make center-based collision conservative.
- Change actions: `default_primitives(params, step_length, delta_scale)` or custom `MotionPrimitive` list.
- Tune heuristics: replace `admissible_heuristic` in planners for domain-specific bounds.
- Adjust fidelity: smaller `collision_step`, more `theta_bins` → stricter collision/heading resolution; larger values → faster/looser.
- Dynamics: edit `AckermannParams` for different vehicles (wheelbase, turning radius, velocity limits).

## Outputs and metrics

- Hybrid A*: `expansions`, `time`, `timed_out`, `path_length`, optional `trace_poses`/`trace_boxes` for plotting.
- DQN Hybrid A*: adds `expansions_anchor`, `expansions_dqn`, `expansions_total`.
- RRT*: `nodes`, `iterations`, `time`, `success`, `path`.
- APF: `time`, `reached`, `timed_out`, `path_length` (path empty on failure).

Quantitative CSVs and plots are written under `examples/outputs/` by the demos/benchmarks. Use these to benchmark, compare planners, or tune hyperparameters.
