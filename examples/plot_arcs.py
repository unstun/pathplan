"""
Quick-start visualization for curved Hybrid A* trajectories.

What it does:
- Builds a tiny empty map.
- Runs Hybrid A* (and optionally RL-guided Hybrid A* if you flip a flag).
- Plots the arc samples and per-step vehicle rectangles using the stats
  fields exposed by the planners (`trace_poses`, `trace_boxes`).

Run:
    python -m examples.plot_arcs

If you don't have matplotlib installed, install it with:
    pip install matplotlib
"""

import math
from pathlib import Path

import numpy as np

from pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    RRTStarPlanner,
    RLGuidedHybridPlanner,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def make_open_map(resolution: float = 0.1, size=(6.0, 4.0)) -> GridMap:
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def plan_once(planner_cls):
    grid_map = make_open_map()
    start = AckermannState(0.6, 0.6, math.radians(0))
    goal = AckermannState(5.0, 2.8, math.radians(0))
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.924, width=0.740)

    planner = planner_cls(grid_map, footprint, params)
    if isinstance(planner, RRTStarPlanner):
        path, stats = planner.plan(start, goal, max_iter=4000, timeout=10.0)
    else:
        path, stats = planner.plan(start, goal, timeout=10.0)
    label = planner_cls.__name__
    expansions = stats.get("expansions_total", stats.get("expansions", stats.get("nodes")))
    print(f"Planner: {label}, success={bool(path)}, expansions/nodes={expansions}")
    return grid_map, start, goal, footprint, path, stats, label


def plot(grid_map: GridMap, start: AckermannState, goal: AckermannState, path, stats, title: str, footprint: OrientedBoxFootprint):
    if plt is None:
        print("matplotlib not available; install it with `pip install matplotlib` to see the plot.")
        return

    h, w = grid_map.data.shape
    extent = [
        grid_map.origin[0],
        grid_map.origin[0] + w * grid_map.resolution,
        grid_map.origin[1],
        grid_map.origin[1] + h * grid_map.resolution,
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(grid_map.data, cmap="gray_r", origin="lower", extent=extent, vmin=0, vmax=1)
    ax.scatter(start.x, start.y, c="green", marker="*", s=90, label="start")
    ax.scatter(goal.x, goal.y, c="red", marker="*", s=90, label="goal")

    # Arc samples
    trace = stats.get("trace_poses", [])
    if trace:
        xs, ys, _ = zip(*trace)
        ax.plot(xs, ys, c="blue", lw=2, label="arc trace")

    # Bounding boxes along the arc
    boxes = stats.get("trace_boxes", [])
    if not boxes and path:
        boxes = [footprint.corners(p.x, p.y, p.theta) for p in path]
    for box in boxes:
        bx, by = zip(*(box + [box[0]]))
        ax.plot(bx, by, c="orange", lw=0.8, alpha=0.6)

    # Coarse waypoints (search nodes)
    if path:
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(xs, ys, c="black", lw=1.0, ls="--", marker="o", markersize=3, label="node chain")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best")
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title.lower())
    safe_title = "_".join(part for part in safe_title.split("_") if part)
    out_path = out_dir / f"{safe_title}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    planners = [HybridAStarPlanner, RLGuidedHybridPlanner, RRTStarPlanner]
    for planner_cls in planners:
        grid_map, start, goal, footprint, path, stats, label = plan_once(planner_cls)
        title = f"{label} arc trace"
        plot(grid_map, start, goal, path, stats, title=title, footprint=footprint)


if __name__ == "__main__":
    main()
