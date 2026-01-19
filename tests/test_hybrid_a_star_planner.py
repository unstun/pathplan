import math

import numpy as np

from pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    TwoCircleFootprint,
)
from pathplan.geometry import GridFootprintChecker


def test_hybrid_a_star_analytic_expansion_open_map():
    grid = np.zeros((40, 60), dtype=np.uint8)
    grid_map = GridMap(grid, resolution=0.2, origin=(0.0, 0.0))
    params = AckermannParams()
    footprint = TwoCircleFootprint.from_box(length=0.8, width=0.5)

    planner = HybridAStarPlanner(
        grid_map,
        footprint,
        params,
        xy_resolution=0.4,
        theta_bins=48,
        analytic_expansion=True,
        analytic_expansion_interval=2,
        analytic_expansion_distance_scale=20.0,
        heuristic_weight=1.0,
    )

    start = AckermannState(1.0, 1.0, 0.0)
    goal = AckermannState(9.0, 5.0, 0.0)
    path, stats = planner.plan(start, goal, timeout=2.0, max_nodes=3000)
    assert path, f"Planner failed: {stats}"
    assert abs(path[-1].x - goal.x) < 1e-9
    assert abs(path[-1].y - goal.y) < 1e-9
    assert abs((path[-1].theta - goal.theta + math.pi) % (2 * math.pi) - math.pi) < 1e-9
    assert "trace_poses" in stats
    assert stats.get("expansions", 0) >= 1


def test_hybrid_a_star_plans_around_blocking_obstacle():
    grid = np.zeros((60, 60), dtype=np.uint8)
    # Central obstacle block forcing a detour.
    grid[25:35, 20:40] = 1
    grid_map = GridMap(grid, resolution=0.2, origin=(0.0, 0.0))
    params = AckermannParams()
    footprint = TwoCircleFootprint.from_box(length=0.8, width=0.5)

    planner = HybridAStarPlanner(
        grid_map,
        footprint,
        params,
        xy_resolution=0.4,
        theta_bins=48,
        analytic_expansion=True,
        analytic_expansion_interval=2,
        analytic_expansion_distance_scale=10.0,
        heuristic_weight=1.2,
    )

    start = AckermannState(1.0, 6.0, 0.0)
    goal = AckermannState(10.0, 6.0, 0.0)
    path, stats = planner.plan(start, goal, timeout=4.0, max_nodes=8000)
    assert path, f"Planner failed: {stats}"

    trace = stats.get("trace_poses", [])
    assert trace, "Expected arc trace poses for collision validation."
    checker = GridFootprintChecker(grid_map, footprint, theta_bins=planner.theta_bins)
    assert not checker.collides_path(trace)
