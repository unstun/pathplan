import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathplan import (
    APFPlanner,
    AckermannParams,
    AckermannState,
    DQNHybridAStarPlanner,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    RRTStarPlanner,
)
from pathplan.geometry import GridFootprintChecker
from pathplan.postprocess import controls_to_path, feng_optimize_path, path_to_controls, stomp_optimize_path


def make_empty_map(resolution: float = 0.1, size: tuple = (6.0, 4.0)) -> GridMap:
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    data = np.zeros((h_cells, w_cells), dtype=np.uint8)
    return GridMap(data, resolution, origin=(0.0, 0.0))


def test_empty_map_endpoints_and_collision_free():
    grid_map = make_empty_map()
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.9, width=0.6)
    path = [
        AckermannState(0.5, 0.5, 0.0),
        AckermannState(2.0, 0.5, 0.0),
        AckermannState(3.5, 1.0, 0.2),
    ]
    opt_path, info = stomp_optimize_path(
        path,
        grid_map,
        footprint,
        params,
        step_size=0.2,
        ds_check=0.05,
        rollouts=24,
        iters=15,
        w_clear=0.2,
        seed=4,
        goal=path[-1],
    )
    assert opt_path, "Optimizer returned empty path on a free map."
    end = opt_path[-1]
    assert math.hypot(end.x - path[-1].x, end.y - path[-1].y) <= 0.5
    assert abs(math.atan2(math.sin(end.theta - path[-1].theta), math.cos(end.theta - path[-1].theta))) <= 0.3
    checker = GridFootprintChecker(grid_map, footprint, theta_bins=32)
    assert not checker.collides_path(opt_path)
    assert info.get("iters_run", 0) > 0


def test_smoothness_improves_on_zigzag():
    grid_map = make_empty_map()
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.9, width=0.6)
    zigzag = [
        AckermannState(0.0, 0.0, 0.0),
        AckermannState(0.6, 0.0, 0.1),
        AckermannState(1.2, 0.3, 0.8),
        AckermannState(1.8, 0.3, -0.6),
        AckermannState(2.4, 0.0, 0.0),
    ]
    before_steers, _, _ = path_to_controls(zigzag, params, step_size=0.2, allow_reverse=True)
    opt_path, _ = stomp_optimize_path(
        zigzag,
        grid_map,
        footprint,
        params,
        step_size=0.2,
        rollouts=48,
        iters=25,
        w_clear=0.0,
        seed=7,
        goal=zigzag[-1],
    )
    after_steers, _, _ = path_to_controls(opt_path, params, step_size=0.2, allow_reverse=True)
    smooth_before = float(np.sum(np.diff(before_steers) ** 2)) if before_steers is not None else float("inf")
    smooth_after = float(np.sum(np.diff(after_steers) ** 2)) if after_steers is not None else float("inf")
    assert smooth_after <= smooth_before + 1e-9


def test_path_to_controls_preserves_segment_length():
    params = AckermannParams()
    path = [
        AckermannState(0.0, 0.0, 0.0),
        AckermannState(0.78, 0.0, 0.0),  # lattice-style stride that previously overshot when resampled
    ]
    steers, directions, step_sizes = path_to_controls(path, params, step_size=0.25, allow_reverse=True)
    recon = controls_to_path(path[0], steers, directions, params, step_size=0.25, step_sizes=step_sizes)
    assert math.isclose(recon[-1].x, path[-1].x, abs_tol=1e-3)
    assert math.isclose(recon[-1].y, path[-1].y, abs_tol=1e-3)


def test_planner_paths_accepted_by_stomp():
    grid_map = make_empty_map(resolution=0.1, size=(5.0, 4.0))
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.9, width=0.6)
    start = AckermannState(0.6, 0.6, 0.0)
    goal = AckermannState(2.8, 1.6, 0.0)

    planners = [
        ("APF", APFPlanner(grid_map, footprint, params, max_iters=1200)),
        ("Hybrid", HybridAStarPlanner(grid_map, footprint, params)),
        ("DQNHybrid", DQNHybridAStarPlanner(grid_map, footprint, params)),
        (
            "RRTStar",
            RRTStarPlanner(
                grid_map,
                footprint,
                params,
                goal_sample_rate=0.95,
                neighbor_radius=3.0,
                step_time=0.35,
                goal_xy_tol=0.25,
                goal_theta_tol=math.radians(35.0),
                seed_steps=4,
            ),
        ),
    ]

    for name, planner in planners:
        if name == "RRTStar":
            path, stats = planner.plan(start, goal, max_iter=800, timeout=1.5)
        else:
            path, stats = planner.plan(start, goal, timeout=1.5)
        if not path:
            path = [start, goal]  # fallback to validate postprocessor compatibility
        opt_path, info = stomp_optimize_path(
            path,
            grid_map,
            footprint,
            params,
            step_size=0.2,
            rollouts=12,
            iters=8,
            w_clear=0.0,
            seed=5,
            goal=goal,
        )
        assert opt_path, f"{name} optimizer output empty path"
        assert "best_cost" in info


def test_optimizer_falls_back_when_rollouts_collide():
    grid_map = make_empty_map(resolution=0.1, size=(6.0, 6.0))
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.9, width=0.6)
    # Add a solid block near the middle that reconstructed arcs may cut through.
    grid_map.data[25:35, 25:35] = 1  # 1 m square at (2.5, 2.5)
    path = [
        AckermannState(1.0, 1.0, 0.0),
        AckermannState(2.0, 2.0, math.pi / 2),
        AckermannState(4.0, 1.0, 0.0),
    ]
    checker = GridFootprintChecker(grid_map, footprint, theta_bins=64)
    assert not checker.collides_path(path)

    opt_path, info = stomp_optimize_path(
        path,
        grid_map,
        footprint,
        params,
        step_size=0.8,  # coarse reconstruction will collide with the block
        ds_check=0.15,
        rollouts=1,
        iters=3,
        w_clear=0.0,
        noise_std=0.0,
        goal=path[-1],
        seed=2,
    )
    assert opt_path == path  # falls back to the original, collision-free path
    assert info.get("improved") is False
    assert not checker.collides_path(opt_path)


def test_feng_optimizer_returns_collision_free_path():
    grid_map = make_empty_map(resolution=0.1, size=(6.0, 4.0))
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.9, width=0.6)
    path = [
        AckermannState(0.5, 0.5, 0.0),
        AckermannState(1.8, 0.6, 0.1),
        AckermannState(3.2, 1.3, 0.2),
    ]
    checker = GridFootprintChecker(grid_map, footprint, theta_bins=48)
    assert not checker.collides_path(path)

    opt_path, info = feng_optimize_path(
        path,
        grid_map,
        footprint,
        params,
        goal=path[-1],
        max_segments=6,
        samples_per_seg=10,
        iters=30,
        output_step=0.12,
        ds_check=0.08,
        seed=4,
    )
    assert opt_path
    assert "best_cost" in info
    assert not checker.collides_path(opt_path)
