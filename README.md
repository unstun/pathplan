# Path Planning Scaffold

Python scaffold for Ackermann planners (APF, Hybrid A*, D-Hybrid A* (DQN-guided Hybrid A*), Informed RRT* (kinodynamic)) with a shared robot model and grid-based collision checker. Everything is NumPy-first, ROS-free, and kept small so you can swap pieces without digging through a framework.

## What’s inside (code map)

- `pathplan/` core: collision geometry (`geometry.py`), occupancy grid (`map_utils.py`), robot model and propagation (`robot.py`), motion primitives (`primitives.py`), heuristics (`heuristics.py`), planners (`hybrid_a_star.py`, `dqn_hybrid_a_star.py`, `rrt_star.py`, `apf.py`), DQN guidance stubs/torch loader (`dqn_models.py`), and exports (`__init__.py`).
- `examples/run_demo.py` small scenarios (corridor, open field; parking helper included).
- `examples/forest_scene.py` dense-forest benchmarks (4 presets) with plots + CSVs in timestamped folders.
- `examples/train_dqn.py` minimal DQN training loop for guidance (PyTorch).
- `real_env1/` real SLAM forest point cloud segmentation -> occupancy grids/ROS maps; see `real_env1/README.md`.
- `requirements.txt` (NumPy) and `requirements-train.txt` (NumPy + PyTorch) keep dependencies explicit; `matplotlib` is optional for plots.

## Robot + collision model (ground truth from code)

- Ackermann params: `wheelbase=0.6 m`, `min_turn_radius=1.1284 m` → `|steer| ≤ 0.489 rad` (28°, `AckermannParams`).
- Footprint: oriented rectangle `0.924 m x 0.740 m` shared by every planner (`OrientedBoxFootprint`).
- Collision: `GridFootprintChecker` precomputes, per heading bin, all grid-cell centers that lie inside the footprint. It now pads the collision footprint by one map cell on every side by default (adds `2 * resolution` to length/width) while leaving the drawn footprint unchanged; set `padding=0.0` to match the visual box. `collides_pose` tests those centers; `collides_path`/`motion_collides` sample poses along a segment at `collision_step`. This center-based test can visually graze obstacles; increase padding or inflate the map for stricter clearance.
- Maps: `GridMap` with world/grid transforms, random free-state sampling, occupancy patches for guidance, and simple inflation (`inflate`) for safety buffers.

## Planners (defaults in constructors)

- **Hybrid A\*** (`HybridAStarPlanner`): 10 primitives (`default_primitives`, step `0.3 m`, reverse weight `1.2`), cusp penalty `0.2`, heading bins `72`, `collision_step≈0.1 m` (clamped), goal tolerances `0.1 m / 5°`, heuristic = Reeds-Shepp-style lower bound. `xy_resolution` defaults to the map resolution unless overridden (forests use coarser `0.60 m`, 48 bins).
- **D-Hybrid A\*** (`DQNHybridAStarPlanner`, DQN-guided Hybrid A*): same lattice, dual queues (anchor + DQN). DQN branch uses lightweight hand-tuned `DQNGuidance` unless you swap in `TorchDQNGuidance`. Config: `dqn_top_k`, `dqn_weight`, `anchor_inflation`, `dqn_lead_threshold`, `max_dqn_streak`.
- **Informed RRT\*** (`RRTStarPlanner`, kinodynamic): forward-simulate bicycle for `step_time=0.6 s` at `velocity=0.8 m/s`, goal bias `0.2`, neighbor radius `1.5 m`, goal tolerances `0.1 m / 5°`, optional rewiring, `collision_step` uses the shared default and collision is evaluated on-the-fly at every integration step (no lazy collision or heading-bin precomputation).
- **APF** (`APFPlanner`, Artificial Potential Field): attractive + repulsive potentials over a precomputed obstacle distance map, curvature-limited heading rate, `collision_step` uses the shared default and always relies on the footprint collision checker.

## Install & run

```bash
pip install -r requirements.txt         # NumPy core
pip install matplotlib                  # needed for plotting (optional)
pip install -r requirements-real-env.txt  # optional: real forest SLAM map pipeline (laspy, scipy, tqdm, matplotlib)

# Quick demo (corridor + open field)
python -m examples.run_demo
```

Forest benchmarks (writes `examples/outputs/<timestamp>/`):
```bash
python -m examples.forest_scene --variant all
# or: --variant large_map_large_gap | large_map_small_gap | small_map_large_gap | small_map_small_gap
# enable trajectory smoothing with the STOMP-style postprocessor:
python -m examples.forest_scene --variant small_map_small_gap --postprocess stomp
# or the constraint-aware Bezier optimizer:
python -m examples.forest_scene --variant small_map_small_gap --postprocess feng
```

## Trajectory postprocessing

- STOMP-style smoother: `stomp_optimize_path(path, grid_map, footprint, params, goal=None, **kwargs)` in `pathplan.postprocess`. Works on any returned path (Hybrid A*, D-Hybrid A*, Informed RRT*, APF). Prefers returning the latest optimized curve even when the cost does not beat the baseline; only falls back when controls cannot be inferred or no collision-free rollout exists. Info reports both `best_cost` (lowest seen) and `selected_cost` (returned path). Uses existing primitives for simulation (`propagate`) and collision checking (`GridFootprintChecker`), plus a chamfer distance map for clearance penalties. Key knobs: `step_size`, `rollouts`, `iters`, `lambda_`, `w_smooth`, `w_steer`, `w_clear`, `allow_reverse`, `laplacian_strength`, `laplacian_passes`. Defaults are tuned for fast postprocessing (sub-second on demo maps).
- Constraint-aware Bezier optimizer (Feng et al. 2025 inspired): `feng_optimize_path(...)` converts the path into multi-segment quintic Bezier curves, applies safety rectangles extracted from the grid, and minimizes jerk and terrain/clearance costs under continuity and dynamic limits. Use `samples_per_seg`, `iters`, `rect_size`, `w_safety`, `w_terrain`, `w_cont` to balance smoothness and safety; it falls back to the input path if constraints cannot be satisfied, and the forest CLI will auto-run a STOMP fallback when Feng returns an empty/unchanged path to keep an optimized trajectory in the results.
- Enable end-to-end from the CLI with `--postprocess stomp` or `--postprocess feng` (forest_scene) or call directly in your own scripts; optimizers return `(optimized_path, info_dict)`.

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

Swap in `DQNHybridAStarPlanner` for the D-Hybrid A* variant or `RRTStarPlanner` for Informed RRT* sampling.

## Extending / safety knobs

- Inflate obstacles: `GridMap.inflate(margin)` or bump `GridFootprintChecker.padding` above the map-resolution default to make center-based collision more conservative.
- Change actions: `default_primitives(params, step_length, delta_scale)` or custom `MotionPrimitive` list.
- Tune heuristics: replace `admissible_heuristic` in planners for domain-specific bounds.
- Adjust fidelity: smaller `collision_step`, more `theta_bins` → stricter collision/heading resolution; larger values → faster/looser.
- Dynamics: edit `AckermannParams` for different vehicles (wheelbase, turning radius, velocity limits).

## Outputs and metrics

- Hybrid A*: `expansions`, `time`, `timed_out`, `path_length`, optional `trace_poses`/`trace_boxes` for plotting.
- D-Hybrid A*: adds `expansions_anchor`, `expansions_dqn`, `expansions_total`.
- Informed RRT*: `nodes`, `iterations`, `time`, `success`, `path`.
- APF: `time`, `reached`, `timed_out`, `path_length` (path empty on failure).
- All planners may include `failure_reason` and `remediations` when self-checking is enabled.

Quantitative CSVs and plots are written under `examples/outputs/` by the demos/benchmarks. Use these to benchmark, compare planners, or tune hyperparameters.
