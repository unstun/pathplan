import csv
import math
import time
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
except ImportError:  # matplotlib is optional
    plt = None


def make_corridor_map(resolution: float = 0.1, length: float = 10.0, width: float = 2.0):
    cells_x = int(length / resolution) + 1
    cells_y = int(3.0 / resolution)  # include walls above/below
    grid = np.zeros((cells_y, cells_x), dtype=np.uint8)
    corridor_cells = int(width / resolution)
    pad = (cells_y - corridor_cells) // 2
    grid[: pad, :] = 1
    grid[pad + corridor_cells :, :] = 1
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def make_parking_map(resolution: float = 0.05):
    w_cells = int(7.0 / resolution)
    h_cells = int(4.0 / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    # create two parked cars leaving a generous slot
    car_len = int(1.0 / resolution)
    car_wid = int(0.6 / resolution)
    offset = int(0.7 / resolution)
    gap = int(2.2 / resolution)
    grid[offset : offset + car_wid, offset : offset + car_len] = 1
    grid[offset : offset + car_wid, offset + car_len + gap : offset + 2 * car_len + gap] = 1
    # boundaries
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def make_open_map(resolution: float = 0.1, size: tuple = (5.0, 4.0)):
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    # empty map for a fast-success baseline scenario
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def path_length(path):
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i].x - path[i - 1].x
        dy = path[i].y - path[i - 1].y
        length += math.hypot(dx, dy)
    return length


def scenario_slug(name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())
    return "_".join([s for s in slug.split("_") if s])


def plot_paths(name: str, grid_map: GridMap, start: AckermannState, goal: AckermannState, paths, out_dir: Path):
    if plt is None or not paths:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    h, w = grid_map.data.shape
    extent = [
        grid_map.origin[0],
        grid_map.origin[0] + w * grid_map.resolution,
        grid_map.origin[1],
        grid_map.origin[1] + h * grid_map.resolution,
    ]
    ax.imshow(grid_map.data, cmap="gray_r", origin="lower", extent=extent, vmin=0, vmax=1)
    ax.scatter(start.x, start.y, c="green", marker="*", s=80, label="start")
    ax.scatter(goal.x, goal.y, c="red", marker="*", s=80, label="goal")
    for label, path in paths.items():
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(xs, ys, linewidth=2, label=label)
    ax.set_aspect("equal")
    ax.set_title(name)
    ax.legend(loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_slug(name)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_scenario(
    name: str,
    grid_map: GridMap,
    start: AckermannState,
    goal: AckermannState,
    results: list,
    plot_dir: Path,
):
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.924, width=0.740)

    print(f"\n=== Scenario: {name} ===")
    planners = [
        ("Hybrid A*", HybridAStarPlanner(grid_map, footprint, params)),
        ("RL-guided Hybrid A*", RLGuidedHybridPlanner(grid_map, footprint, params)),
        ("RRT*", RRTStarPlanner(grid_map, footprint, params)),
    ]

    found_paths = {}
    for label, planner in planners:
        t0 = time.time()
        if isinstance(planner, RRTStarPlanner):
            path, stats = planner.plan(start, goal, max_iter=2000, timeout=4.0)
        else:
            path, stats = planner.plan(start, goal, timeout=4.0)
        stats["time_wall"] = time.time() - t0
        length = path_length(path)
        expansions = stats.get("expansions", stats.get("expansions_total", stats.get("nodes", "-")))
        success = len(path) > 0
        print(
            f"{label}: success={success}, time={stats.get('time',0):.2f}s, "
            f"path_len={length:.2f}, expansions={expansions}"
        )
        results.append(
            {
                "scenario": name,
                "planner": label,
                "success": success,
                "path_length": length,
                "expansions_or_nodes": expansions,
                "time": stats.get("time", 0.0),
                "time_wall": stats.get("time_wall", 0.0),
            }
        )
        if success:
            found_paths[label] = path

    saved_plot = plot_paths(name, grid_map, start, goal, found_paths, plot_dir)
    if saved_plot:
        print(f"Saved plot: {saved_plot}")


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    results = []

    corridor = make_corridor_map(width=2.0)
    start = AckermannState(1.0, 1.5, 0.0)
    goal = AckermannState(8.5, 1.5, 0.0)
    run_scenario("Tight corridor with partial block", corridor, start, goal, results, output_dir)

    open_map = make_open_map()
    start2 = AckermannState(0.8, 0.8, 0.0)
    goal2 = AckermannState(4.2, 2.0, 0.0)
    run_scenario("Open field", open_map, start2, goal2, results, output_dir)

    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results.csv"
        fieldnames = ["scenario", "planner", "success", "path_length", "expansions_or_nodes", "time", "time_wall"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved quantitative results: {csv_path}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
