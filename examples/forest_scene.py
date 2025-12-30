"""
Dense forest scenario with many tree obstacles (flat terrain, no slopes).
Four presets are available: large map, small map, large gap, small gap.
Run one command to generate all four maps and four planners (APF, Hybrid A*, DQN Hybrid A*, RRT*):
    python -m examples.forest_scene --variant all
Outputs (plots + CSV) are written to time-stamped folders in examples/outputs/.
"""

import argparse
import csv
import os
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Allow duplicated Intel OpenMP runtimes (NumPy + other libs) to coexist for this script.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    DQNHybridAStarPlanner,
    RRTStarPlanner,
    APFPlanner,
)
from pathplan.primitives import default_primitives

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Base defaults used by all presets and the interactive editor.
FOREST_MAP_KWARGS = dict(
    resolution=0.10,
    size=(40.0, 28.0),
    num_trees=400,  # dense but still navigable
    tree_radius=0.30,
    clearance=1.8,
    seed=13,
    keep_clear=None,  # filled dynamically based on the map size
    carve_points=None,
    wall_y=None,
    wall_gap=(7.0, 4.0),
    wall_radius=0.10,
    protected_path=None,
)

FOREST_VARIANTS = {
    "large_map": {
        "title": "Forest (large map)",
        "map_kwargs": {
            "size": (64.0, 40.0),
            "num_trees": 750,
            "clearance": 1.8,
            "resolution": 0.10,
        },
    },
    "small_map": {
        "title": "Forest (small map)",
        "map_kwargs": {
            "size": (24.0, 16.0),
            "num_trees": 180,
            "clearance": 1.5,
            "resolution": 0.08,
        },
    },
    "large_gap": {
        "title": "Forest (large gap)",
        "map_kwargs": {
            "num_trees": 240,
            "clearance": 2.3,
        },
    },
    "small_gap": {
        "title": "Forest (small gap)",
        "map_kwargs": {
            "num_trees": 500,
            "clearance": 1.3,
        },
    },
}

PATH_COLOR_MAP = {
    "Artificial Potential Field": "#d81b60",  # bold red/magenta
    "Hybrid A*": "#1f77b4",  # deep blue
    "DQN Hybrid A*": "#2ca02c",  # strong green
    "RRT*": "#ff7f0e",  # vivid orange
}
FALLBACK_COLORS = ["#6a3d9a", "#a6cee3", "#b2df8a"]  # purple plus high-contrast backups
PRIMARY_LINEWIDTH = 3.6


def compute_start_goal(map_kwargs):
    """
    Derive start/goal inside the map bounds from the map settings.
    Positions scale with map size and respect clearance/tree radius.
    """
    size_x, size_y = map_kwargs.get("size", (80.0, 48.0))
    tree_r = map_kwargs.get("tree_radius", 0.30)
    clearance = map_kwargs.get("clearance", 1.8)
    margin = max(tree_r * 2.0, tree_r + clearance + 0.6)
    start = AckermannState(
        max(margin, size_x * 0.08),
        max(margin, size_y * 0.20),
        0.0,
    )
    goal = AckermannState(
        min(size_x - margin, size_x * 0.92),
        min(size_y - margin, size_y * 0.80),
        0.0,
    )
    return start, goal


DEFAULT_START, DEFAULT_GOAL = compute_start_goal(FOREST_MAP_KWARGS)

# Planner budgets tuned for sub-second runs.
APF_TIMEOUT = 3.0
APF_MAX_ITERS = 8_000
HYBRID_TIMEOUT = 1.0
HYBRID_MAX_NODES = 8_000
DQN_TIMEOUT = 1.0
DQN_MAX_NODES = 8_000
RRT_TIMEOUT = 1.0
RRT_MAX_ITER = 20_000


def make_forest_map(
    resolution: float = 0.05,
    size: tuple = (20.0, 12.0),
    num_trees: int = 10,
    tree_radius: float = 1,
    clearance: float = 100,
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
    max_attempts = num_trees * 300  # larger scenes / larger clearance need more sampling attempts
    margin = max(tree_radius + 0.3, tree_radius + clearance * 0.25)
    if margin * 2.0 >= min(size):
        raise ValueError(
            f"Map too small for tree_radius={tree_radius} and clearance={clearance}. "
            f"Margin={margin:.2f} must be < half of map dimensions {size}. "
            "Reduce clearance/tree_radius or increase map size."
        )

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
        wall_spacing = max(wall_r * 2.0, tree_radius * 2.0 + clearance)
        x_positions = np.arange(tree_radius + clearance, size[0] - tree_radius - clearance, wall_spacing)
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
    variant_slug: str,
    grid_map: GridMap,
    start: AckermannState,
    goal: AckermannState,
    planner_results: List[Tuple[str, List[AckermannState], dict]],
    footprint: OrientedBoxFootprint,
    out_dir: Path,
):
    if plt is None or not planner_results:
        return None
    fig, ax = plt.subplots(figsize=(24, 14), facecolor="white")
    fig.subplots_adjust(right=0.98, left=0.06, top=0.98, bottom=0.08)
    ax.set_facecolor("white")
    h, w = grid_map.data.shape
    res = grid_map.resolution
    ox, oy = grid_map.origin
    extent = [
        ox - res * 0.5,
        ox + w * res - res * 0.5,
        oy - res * 0.5,
        oy + h * res - res * 0.5,
    ]
    ax.imshow(
        grid_map.data,
        cmap="gray_r",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=1.0,
        interpolation="nearest",
    )
    ax.scatter(start.x, start.y, c="#008000", marker="*", s=1760, label="start", edgecolors="black", linewidths=0)
    ax.scatter(goal.x, goal.y, c="#c8102e", marker="*", s=1760, label="goal", edgecolors="black", linewidths=0)

    color_cycle = list(PATH_COLOR_MAP.values()) + FALLBACK_COLORS
    for idx, (label, path, stats) in enumerate(planner_results):
        color = PATH_COLOR_MAP.get(label, color_cycle[idx % len(color_cycle)])
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(xs, ys, linewidth=PRIMARY_LINEWIDTH, color=color, solid_capstyle="round", label=label)
        boxes = stats.get("trace_boxes", [])
        if not boxes and path:
            boxes = [footprint.corners(p.x, p.y, p.theta) for p in path]
        if boxes:
            # Downsample boxes for readability
            stride = max(1, len(boxes) // 40)
            for box in boxes[::stride]:
                bx, by = zip(*(box + [box[0]]))
                ax.plot(bx, by, linewidth=1.0, alpha=0.45, color=color)
        if path:
            heading = path[-1]
            ax.arrow(
                heading.x,
                heading.y,
                0.4 * math.cos(heading.theta),
                0.4 * math.sin(heading.theta),
                head_width=0.15,
                head_length=0.2,
                fc=color,
                ec=color,
                alpha=0.85,
                length_includes_head=True,
            )

    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=16)
    ax.tick_params(length=0, labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("white")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_slug = variant_slug or "default"
    out_path = out_dir / f"dense_forest_{safe_slug}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def normalize_variant(name: str) -> str:
    slug = name.strip().lower().replace(" ", "_").replace("-", "_")
    if slug not in FOREST_VARIANTS:
        options = ", ".join(sorted(FOREST_VARIANTS))
        raise ValueError(f"Unknown variant '{name}'. Choose from: {options}.")
    return slug


def build_variant(variant: str):
    variant_slug = normalize_variant(variant)
    cfg = FOREST_VARIANTS[variant_slug]
    map_kwargs = dict(FOREST_MAP_KWARGS)
    map_kwargs.update(cfg.get("map_kwargs", {}))

    start, goal = compute_start_goal(map_kwargs)
    map_kwargs["keep_clear"] = [(start.x, start.y), (goal.x, goal.y)]
    title = cfg.get("title", f"Forest ({variant_slug})")
    return variant_slug, title, map_kwargs, start, goal


def plan_forest_scene(variant: str = "large_map", out_dir: Optional[Path] = None):
    output_dir = out_dir or (Path(__file__).resolve().parent / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_slug, title, map_kwargs, start, goal = build_variant(variant)
    grid_map = make_forest_map(**map_kwargs)

    print(
        f"Variant '{variant_slug}': size={map_kwargs['size']} m, "
        f"trees={map_kwargs['num_trees']}, resolution={map_kwargs['resolution']:.2f} m"
    )
    if map_kwargs.get("wall_y") is not None:
        gap_center, gap_width = map_kwargs["wall_gap"]
        print(f"  Wall at y={map_kwargs['wall_y']:.2f} with gap center={gap_center:.2f}, width={gap_width:.2f}")

    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.924, width=0.740)
    short_primitives = default_primitives(params, step_length=1.0)
    base_kwargs = dict(
        xy_resolution=0.60,
        collision_step=0.20,
        goal_xy_tol=1.5,
        goal_theta_tol=math.pi,
        theta_bins=48,
    )
    hybrid_kwargs = dict(base_kwargs, heuristic_weight=2.2)
    dqn_kwargs = dict(base_kwargs, dqn_top_k=3, anchor_inflation=2.0, dqn_weight=1.3)
    rrt_kwargs = dict(
        goal_sample_rate=0.55,  # lower bias to escape blocked goal rays faster
        neighbor_radius=3.5,
        step_time=1.40,  # longer rollout per sample = fewer iterations to reach the goal
        velocity=1.40,
        connect_threshold=5.0,
        goal_xy_tol=0.8,
        goal_theta_tol=math.pi,
        goal_check_freq=1,
        seed_steps=30,
        collision_step=0.20,
        lazy_collision=False,  # full motion collision checking (still sub-second with the longer step_time)
        rewire=False,
        theta_bins=48,
    )
    apf_kwargs = dict(
        step_size=0.3,
        goal_tol=1.0,
        repulse_radius=0.8,
        obstacle_gain=0.6,
        goal_gain=1.5,
        max_iters=APF_MAX_ITERS,
        collision_step=0.10,
        stall_steps=200,
        theta_bins=48,
        min_step=0.01,
        jitter_angle=1.0,
        heading_rate=1.0,
        coarse_collision=True,
    )
    planners = [
        ("Artificial Potential Field", APFPlanner(grid_map, footprint, params, **apf_kwargs)),
        ("Hybrid A*", HybridAStarPlanner(grid_map, footprint, params, primitives=short_primitives, **hybrid_kwargs)),
        ("DQN Hybrid A*", DQNHybridAStarPlanner(grid_map, footprint, params, primitives=short_primitives, **dqn_kwargs)),
        ("RRT*", RRTStarPlanner(grid_map, footprint, params, **rrt_kwargs)),
    ]

    records = []
    planner_results = []
    for label, planner in planners:
        t0 = time.time()
        if isinstance(planner, RRTStarPlanner):
            path, stats = planner.plan(start, goal, max_iter=RRT_MAX_ITER, timeout=RRT_TIMEOUT)
        elif isinstance(planner, DQNHybridAStarPlanner):
            path, stats = planner.plan(start, goal, timeout=DQN_TIMEOUT, max_nodes=DQN_MAX_NODES)
        elif isinstance(planner, HybridAStarPlanner):
            path, stats = planner.plan(start, goal, timeout=HYBRID_TIMEOUT, max_nodes=HYBRID_MAX_NODES)
        elif isinstance(planner, APFPlanner):
            path, stats = planner.plan(start, goal, timeout=APF_TIMEOUT)
        else:
            path, stats = planner.plan(start, goal)
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
        records.append(
            {
                "variant": variant_slug,
                "planner": label,
                "success": success,
                "path_length": path_len,
                "expansions_or_nodes": expansions,
                "time": stats.get("time", 0.0),
                "time_wall": stats.get("time_wall", 0.0),
            }
        )
        if success:
            planner_results.append((label, path, stats))

    saved_plot = plot_with_boxes(
        title,
        variant_slug,
        grid_map,
        start,
        goal,
        planner_results,
        footprint,
        output_dir,
    )
    if saved_plot:
        print(f"Saved plot: {saved_plot}")
        for rec in records:
            rec["plot_path"] = str(saved_plot)

    return records, saved_plot


def parse_args():
    parser = argparse.ArgumentParser(description="Forest scene planner presets.")
    parser.add_argument(
        "--variant",
        default="large_map",
        help="Choose from: large_map, small_map, large_gap, small_gap, or 'all' to run every preset.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Base folder for results. A time-stamped subfolder is created inside.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {out_dir}")

    all_records = []
    variants = list(FOREST_VARIANTS) if args.variant.strip().lower() in ("all", "any") else [args.variant]
    for variant_name in variants:
        print("\n" + "=" * 60)
        records, _ = plan_forest_scene(variant_name, out_dir=out_dir)
        all_records.extend(records)

    if all_records:
        csv_path = out_dir / "results.csv"
        fieldnames = ["variant", "planner", "success", "path_length", "expansions_or_nodes", "time", "time_wall", "plot_path"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_records:
                writer.writerow(row)
        print(f"\nSaved summary: {csv_path}")
    else:
        print("\nNo results to save.")
