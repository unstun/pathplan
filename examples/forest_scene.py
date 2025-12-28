"""
Dense forest scenario with many tree obstacles (flat terrain, no slopes).
Run:
    python -m examples.forest_scene
The script will print planner stats and save a plot to examples/outputs/.
"""

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    RLGuidedHybridPlanner,
    RRTStarPlanner,
)
from pathplan.primitives import default_primitives

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def make_forest_map(
    resolution: float = 0.05,
    size: tuple = (20.0, 12.0),
    num_trees: int = 25,
    tree_radius: float = 0.35,
    clearance: float = 1.8,
    seed: int = 13,
    keep_clear: Optional[List[Tuple[float, float]]] = None,
    carve_points: Optional[List[Tuple[float, float]]] = None,
    carve_width: float = 6.0,
    wall_y: Optional[float] = None,
    wall_gap: Tuple[float, float] = (5.5, 1.2),
    wall_radius: Optional[float] = None,
    protected_path: Optional[List[Tuple[float, float]]] = None,
    protected_width: float = 6.0,
) -> GridMap:
    """
    Build a flat map with many circular tree trunks.
    `clearance` keeps trees away from start/goal and from each other.
    """
    keep_clear = keep_clear or []
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    centers = []
    min_spacing = tree_radius * 2.0 + clearance
    max_attempts = num_trees * 50
    margin = max(tree_radius + 0.3, tree_radius + clearance * 0.25)

    def dist_point_to_segment(p, a, b):
        ap = p - a
        ab = b - a
        denom = np.dot(ab, ab)
        if denom == 0:
            return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    attempts = 0
    while len(centers) < num_trees and attempts < max_attempts:
        candidate = np.array(
            [
                rng.uniform(margin, size[0] - margin),
                rng.uniform(margin, size[1] - margin),
            ]
        )
        too_close = any(np.linalg.norm(candidate - c) < min_spacing for c in centers)
        if not too_close:
            too_close = any(np.linalg.norm(candidate - np.asarray(p)) < clearance for p in keep_clear)
        if not too_close and protected_path:
            for i in range(len(protected_path) - 1):
                a = np.asarray(protected_path[i])
                b = np.asarray(protected_path[i + 1])
                if dist_point_to_segment(candidate, a, b) < protected_width / 2.0:
                    too_close = True
                    break
        if too_close:
            attempts += 1
            continue
        centers.append(candidate)
        attempts += 1

    def stamp_disk(cx: float, cy: float, radius: float):
        tree_cells = int(math.ceil(radius / resolution))
        radius_sq = radius * radius
        gx = int(round(cx / resolution))
        gy = int(round(cy / resolution))
        x_min = max(0, gx - tree_cells)
        x_max = min(w_cells - 1, gx + tree_cells)
        y_min = max(0, gy - tree_cells)
        y_max = min(h_cells - 1, gy + tree_cells)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                dx = (x - gx) * resolution
                dy = (y - gy) * resolution
                if dx * dx + dy * dy <= radius_sq:
                    grid[y, x] = 1

    # stamp the forest trees
    for cx, cy in centers:
        stamp_disk(cx, cy, tree_radius)

    # optional tree wall that blocks straight lines and forces a detour
    if wall_y is not None:
        wall_r = wall_radius if wall_radius is not None else tree_radius * 1.05
        gap_center, gap_width = wall_gap
        x_positions = np.arange(tree_radius + clearance, size[0] - tree_radius - clearance, wall_r * 2.0)
        for x in x_positions:
            if abs(x - gap_center) < gap_width / 2.0:
                continue
            stamp_disk(x, wall_y, wall_r)

    if carve_points:
        carve_radius_sq = carve_width * carve_width
        carve_cells = int(math.ceil(carve_width / resolution))
        for i in range(len(carve_points) - 1):
            x0, y0 = carve_points[i]
            x1, y1 = carve_points[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            steps = max(1, int(seg_len / (resolution * 0.5)))
            for s in range(steps + 1):
                t = s / steps
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                gx = int(round(x / resolution))
                gy = int(round(y / resolution))
                x_min = max(0, gx - carve_cells)
                x_max = min(w_cells - 1, gx + carve_cells)
                y_min = max(0, gy - carve_cells)
                y_max = min(h_cells - 1, gy + carve_cells)
                for xx in range(x_min, x_max + 1):
                    for yy in range(y_min, y_max + 1):
                        dx = (xx - gx) * resolution
                        dy = (yy - gy) * resolution
                        if dx * dx + dy * dy <= carve_radius_sq:
                            grid[yy, xx] = 0

    return GridMap(grid, resolution, origin=(0.0, 0.0))


def plot_with_boxes(
    name: str,
    grid_map: GridMap,
    start: AckermannState,
    goal: AckermannState,
    planner_results: List[Tuple[str, List[AckermannState], dict]],
    footprint: OrientedBoxFootprint,
    out_dir: Path,
):
    if plt is None or not planner_results:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    h, w = grid_map.data.shape
    extent = [
        grid_map.origin[0],
        grid_map.origin[0] + w * grid_map.resolution,
        grid_map.origin[1],
        grid_map.origin[1] + h * grid_map.resolution,
    ]
    ax.imshow(grid_map.data, cmap="gray_r", origin="lower", extent=extent, vmin=0, vmax=1)
    ax.scatter(start.x, start.y, c="green", marker="*", s=120, label="start")
    ax.scatter(goal.x, goal.y, c="red", marker="*", s=120, label="goal")

    for label, path, stats in planner_results:
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(xs, ys, linewidth=2.2, label=label)
        boxes = stats.get("trace_boxes", [])
        if not boxes and path:
            boxes = [footprint.corners(p.x, p.y, p.theta) for p in path]
        if boxes:
            # Downsample boxes for readability
            stride = max(1, len(boxes) // 40)
            for box in boxes[::stride]:
                bx, by = zip(*(box + [box[0]]))
                ax.plot(bx, by, linewidth=1.0, alpha=0.7)
        if path:
            heading = path[-1]
            ax.arrow(
                heading.x,
                heading.y,
                0.4 * math.cos(heading.theta),
                0.4 * math.sin(heading.theta),
                head_width=0.15,
                head_length=0.2,
                fc="k",
                ec="k",
                alpha=0.9,
                length_includes_head=True,
            )

    ax.set_aspect("equal")
    ax.set_title(name)
    ax.legend(loc="upper right")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dense_forest_no_slope_many_trees.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plan_forest_scene():
    output_dir = Path(__file__).resolve().parent / "outputs"
    start = AckermannState(2.0, 2.0, 0.0)
    goal = AckermannState(18.0, 10.0, 0.0)
    grid_map = make_forest_map(
        resolution=0.05,
        size=(20.0, 12.0),
        num_trees=25,
        tree_radius=0.35,
        clearance=1.8,
        seed=13,
        keep_clear=[(start.x, start.y), (goal.x, goal.y)],
        carve_points=[
            (start.x, start.y),
            (4.0, 4.0),
            (7.5, 6.0),
            (11.0, 7.5),
            (14.5, 8.8),
            (goal.x, goal.y),
        ],
        carve_width=6.0,
        wall_y=None,
        wall_gap=(7.0, 4.0),
        wall_radius=0.10,
        protected_path=[
            (start.x, start.y),
            (4.0, 4.0),
            (7.5, 6.0),
            (11.0, 7.5),
            (14.5, 8.8),
            (goal.x, goal.y),
        ],
        protected_width=6.0,
    )

    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.924, width=0.740)
    short_primitives = default_primitives(params, step_length=0.5)
    planner_kwargs = dict(
        xy_resolution=0.25,
        collision_step=0.15,
        goal_xy_tol=1.0,
        goal_theta_tol=math.pi,
    )
    planners = [
        ("Hybrid A*", HybridAStarPlanner(grid_map, footprint, params, primitives=short_primitives, **planner_kwargs)),
        ("RL-guided Hybrid A*", RLGuidedHybridPlanner(grid_map, footprint, params, primitives=short_primitives, **planner_kwargs)),
        ("RRT*", RRTStarPlanner(grid_map, footprint, params)),
    ]

    planner_results = []
    for label, planner in planners:
        t0 = time.time()
        if isinstance(planner, RRTStarPlanner):
            path, stats = planner.plan(start, goal, max_iter=12000, timeout=20.0)
        else:
            path, stats = planner.plan(start, goal, timeout=30.0, max_nodes=40000)
        stats["time_wall"] = time.time() - t0
        success = len(path) > 0
        expansions = stats.get("expansions", stats.get("expansions_total", stats.get("nodes", "-")))
        path_len = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            path_len += math.hypot(dx, dy)
        print(
            f"{label}: success={success}, time={stats.get('time',0):.2f}s, "
            f"path_len={path_len:.2f}, expansions/nodes={expansions}"
        )
        if success:
            planner_results.append((label, path, stats))

    saved_plot = plot_with_boxes(
        "Dense forest (no slope, many trees)",
        grid_map,
        start,
        goal,
        planner_results,
        footprint,
        output_dir,
    )
    if saved_plot:
        print(f"Saved plot: {saved_plot}")


if __name__ == "__main__":
    plan_forest_scene()
