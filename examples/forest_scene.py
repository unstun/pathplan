"""
Dense forest scenario with many tree obstacles (flat terrain, no slopes).
Five presets are available: the four synthetic variants (large/small maps with large/small gaps)
plus the real_env1 map built from the real SLAM occupancy grid.
Run one command to generate every preset and planner (APF, Hybrid A*, D-Hybrid A*, Informed RRT*):
    python -m examples.forest_scene --variant all
Outputs (plots + CSV) are written to time-stamped folders in examples/outputs/.
"""

import argparse
import ast
import csv
import os
import math
import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

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
    feng_optimize_path,
    stomp_optimize_path,
)
from pathplan.geometry import GridFootprintChecker
from pathplan.common import default_collision_step
from pathplan.primitives import default_primitives

from examples.planner_labels import (
    APF_NAME,
    DQNHYBRID_NAME,
    HYBRID_NAME,
    PLANNER_COLOR_MAP,
    RRT_NAME,
    formal_planner_name,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Base defaults used by all presets and the interactive editor.
FOREST_MAP_KWARGS = dict(
    resolution=0.1,
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

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_ENV1_DIR = REPO_ROOT / "real_env1"
REAL_ENV1_GRID = REAL_ENV1_DIR / "grid_out" / "occupancy.npy"
REAL_ENV1_MAP_YAML = REAL_ENV1_DIR / "grid_out" / "map_c.yaml"

FOREST_VARIANTS = {
    "large_map_large_gap": {
        "title": "Forest (large map, large gap)",
        "map_kwargs": {
            "size": (64.0, 40.0),
            "num_trees": 550,
            "clearance": 2.3,
        },
    },
    "large_map_small_gap": {
        "title": "Forest (large map, small gap)",
        "map_kwargs": {
            "size": (64.0, 40.0),
            "num_trees": 750,
            "clearance": 1.8,
        },
    },
    "small_map_large_gap": {
        "title": "Forest (small map, large gap)",
        "map_kwargs": {
            "size": (24.0, 16.0),
            "num_trees": 82,
            "clearance": 2.3,
        },
    },
    "small_map_small_gap": {
        "title": "Forest (small map, small gap)",
        "map_kwargs": {
            "size": (24.0, 16.0),
            "num_trees": 180,
            "clearance": 1.5,
        },
    },
    "real_env1": {
        "title": "Real forest (SLAM occupancy, real_env1)",
        "loader": "real_env1",
        "grid_path": REAL_ENV1_GRID,
        "yaml_path": REAL_ENV1_MAP_YAML,
        "preferred_start": (2.0, 7.0),
        "preferred_goal": (-28.0, 1.5),
    },
}

FALLBACK_COLORS = ["#6a3d9a", "#a6cee3", "#b2df8a"]  # purple plus high-contrast backups
AMBIGUOUS_LEGEND_NAMES = [
    "Path I",
    "Path l",
    "Path 1",
    "Path |",
    "Path !",
    "Plan I",
    "Plan l",
    "Plan 1",
    "Plan |",
    "Plan !",
]
PRIMARY_LINEWIDTH = 3.6


def strip_variant_suffix(label: str) -> str:
    """
    Remove variant suffixes like '(orig)' or '(stomp)' while keeping formal names
    that may themselves contain parentheses (e.g., '(APF)').
    """
    if " (" in label and label.endswith(")"):
        suffix = label[label.rfind("(") + 1 : -1].strip().lower()
        # Handle combined tags such as 'feng+stomp' by splitting on common separators.
        for sep in ("+", "/", "|", "&"):
            suffix = suffix.replace(sep, " ")
        suffix_tokens = {tok.strip() for tok in suffix.split() if tok.strip()}
        if suffix_tokens.intersection({"orig", "stomp", "feng"}):
            return label[: label.rfind(" (")].strip()
    return label


def normalize_planner_label(label: str) -> str:
    """
    Map user-facing labels/aliases to the formal planner label.
    Accepts shorthands like 'rrt' or 'hybrid'.
    """
    if not label:
        return ""
    text = str(label).strip()
    if not text:
        return ""
    formal = formal_planner_name(text)
    if formal and formal != text:
        return formal
    slug = text.lower().replace(" ", "")
    aliases = {
        "rrt": RRT_NAME,
        "rrt*": RRT_NAME,
        "informedrrt*": RRT_NAME,
        "apf": APF_NAME,
        "hybrid": HYBRID_NAME,
        "hybrida*": HYBRID_NAME,
        "dhybrid": DQNHYBRID_NAME,
        "d-hybrid": DQNHYBRID_NAME,
        "dhybrida*": DQNHYBRID_NAME,
        "d-hybrida*": DQNHYBRID_NAME,
    }
    return aliases.get(slug, text)


def parse_postprocess_allow(arg: Optional[str]) -> Optional[Set[str]]:
    """
    Parse a comma-separated allowlist of planners for postprocessing.
    Returns None when all planners are allowed.
    """
    if arg is None:
        return None
    text = str(arg).strip()
    if not text or text.lower() in ("all", "any", "*"):
        return None
    allowed: Set[str] = set()
    for part in text.split(","):
        name = normalize_planner_label(part)
        if name:
            allowed.add(strip_variant_suffix(name))
    return allowed or None


def path_length(path: List[AckermannState]) -> float:
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i].x - path[i - 1].x
        dy = path[i].y - path[i - 1].y
        length += math.hypot(dx, dy)
    return length


def max_xy_deviation(a: List[AckermannState], b: List[AckermannState], samples: int = 80) -> float:
    """
    Measure the maximum XY drift between two paths after resampling them
    to the same parametric positions. Useful for detecting when an optimizer
    effectively returned the input trajectory (even with different sampling).
    """
    if not a or not b:
        return float("inf")
    samples = max(2, int(samples))
    taus = np.linspace(0.0, 1.0, samples)

    def sample(path: List[AckermannState]) -> np.ndarray:
        if len(path) == 1:
            return np.tile([[path[0].x, path[0].y]], (samples, 1))
        pts = np.zeros((samples, 2), dtype=float)
        last_idx = len(path) - 1
        for i, tau in enumerate(taus):
            s = tau * last_idx
            i0 = int(math.floor(s))
            i1 = min(i0 + 1, last_idx)
            t = s - i0
            p0 = path[i0]
            p1 = path[i1]
            pts[i, 0] = p0.x + (p1.x - p0.x) * t
            pts[i, 1] = p0.y + (p1.y - p0.y) * t
        return pts

    pts_a = sample(a)
    pts_b = sample(b)
    return float(np.linalg.norm(pts_a - pts_b, axis=1).max())


def reorient_path_headings(path: List[AckermannState]) -> List[AckermannState]:
    """
    Align headings to the forward segment direction to avoid arc reconstruction
    cutting corners during postprocessing.
    """
    if not path:
        return []
    aligned: List[AckermannState] = []
    for i, pose in enumerate(path):
        if i + 1 < len(path):
            nxt = path[i + 1]
            heading = math.atan2(nxt.y - pose.y, nxt.x - pose.x)
        elif aligned:
            heading = aligned[-1].theta
        else:
            heading = pose.theta
        aligned.append(AckermannState(pose.x, pose.y, heading))
    return aligned


def laplacian_smooth_path(
    path: List[AckermannState],
    checker: GridFootprintChecker,
    ds_check: float,
    *,
    alpha: float = 0.18,
    passes: int = 4,
) -> Tuple[List[AckermannState], bool]:
    """
    Lightweight Laplacian smoothing on XY positions with collision checks.
    Keeps endpoints fixed and re-computes headings from neighboring points.
    """
    if len(path) < 3 or alpha <= 0.0 or passes <= 0:
        return list(path), False
    smoothed = list(path)
    for _ in range(passes):
        updated: List[AckermannState] = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            prev_p = smoothed[i - 1]
            curr = smoothed[i]
            nxt = smoothed[i + 1]
            x = curr.x + alpha * ((prev_p.x + nxt.x) - 2.0 * curr.x)
            y = curr.y + alpha * ((prev_p.y + nxt.y) - 2.0 * curr.y)
            heading = math.atan2(nxt.y - prev_p.y, nxt.x - prev_p.x)
            updated.append(AckermannState(x, y, heading))
        last = smoothed[-1]
        updated.append(AckermannState(last.x, last.y, last.theta))
        smoothed = updated
    if checker.collides_path(smoothed):
        return list(path), False
    for a, b in zip(smoothed[:-1], smoothed[1:]):
        if checker.motion_collides(a.as_tuple(), b.as_tuple(), step=ds_check):
            return list(path), False
    return smoothed, True


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


def parse_ros_map_yaml(yaml_path: Path) -> Tuple[float, Tuple[float, float]]:
    """Minimal parser for a ROS map YAML to extract resolution and origin."""
    txt = yaml_path.read_text(encoding="utf-8")
    data = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        data[key.strip()] = val.strip()
    try:
        resolution = float(data["resolution"])
        origin = ast.literal_eval(data["origin"])
        origin_xy = (float(origin[0]), float(origin[1]))
    except Exception as exc:  # pragma: no cover - defensive parse
        raise ValueError(f"Failed to parse ROS map YAML {yaml_path}: {exc}") from exc
    return resolution, origin_xy


def farthest_free_cells(free_mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int, int]:
    """
    Pick two free cells that are far apart within the largest connected component.
    Returns (start_gx, start_gy), (goal_gx, goal_gy), grid_distance, component_size.
    """
    h, w = free_mask.shape
    visited = np.zeros_like(free_mask, dtype=bool)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    largest_component = []

    for y in range(h):
        for x in range(w):
            if not free_mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and free_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(cells) > len(largest_component):
                largest_component = cells

    if not largest_component:
        raise ValueError("Occupancy grid has no free cells.")

    def bfs_far(start_yx: Tuple[int, int]) -> Tuple[Tuple[int, int], int]:
        sy, sx = start_yx
        dist = np.full(free_mask.shape, -1, dtype=int)
        dq = deque([(sy, sx)])
        dist[sy, sx] = 0
        far = (sy, sx, 0)
        while dq:
            cy, cx = dq.popleft()
            d = dist[cy, cx]
            if d > far[2]:
                far = (cy, cx, d)
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and free_mask[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    dq.append((ny, nx))
        # return (gx, gy) order for consistency with world conversion
        return (far[1], far[0]), far[2]

    first = largest_component[0]
    start_g, _ = bfs_far(first)
    goal_g, max_steps = bfs_far((start_g[1], start_g[0]))
    return start_g, goal_g, max_steps, len(largest_component)


def load_real_env1_map(
    cfg,
    start_override: Optional[Tuple[float, float, Optional[float]]] = None,
    goal_override: Optional[Tuple[float, float, Optional[float]]] = None,
) -> Tuple[GridMap, AckermannState, AckermannState, dict]:
    """
    Load the real_env1 occupancy grid + YAML, choose distant start/goal inside the
    largest free component, and build a GridMap for planning. Manual start/goal
    overrides (x, y[, theta]) are supported for this map.
    """
    grid_path = Path(cfg.get("grid_path", REAL_ENV1_GRID))
    yaml_path = Path(cfg.get("yaml_path", REAL_ENV1_MAP_YAML))
    if not grid_path.exists():
        raise FileNotFoundError(f"real_env1 grid not found: {grid_path}")
    occ = np.load(grid_path)
    if occ.ndim != 2:
        raise ValueError(f"Expected 2D occupancy grid at {grid_path}, got shape {occ.shape}")

    if yaml_path.exists():
        resolution, origin_xy = parse_ros_map_yaml(yaml_path)
    else:
        resolution = float(cfg.get("resolution", 0.1))
        origin_xy = tuple(cfg.get("origin", (0.0, 0.0)))

    grid_mask = (occ != 0).astype(np.uint8)  # treat unknown (-1) as obstacles
    free_mask = occ == 0
    grid_map = GridMap(grid_mask, resolution, origin_xy)

    footprint_length = float(cfg.get("footprint_length", 0.924))
    footprint_width = float(cfg.get("footprint_width", 0.740))
    clearance_margin = cfg.get("start_goal_clearance", None)
    if clearance_margin is None:
        # Use the circumscribed radius of the robot footprint plus half a cell to guarantee clearance.
        clearance_margin = math.hypot(footprint_length * 0.5, footprint_width * 0.5) + max(resolution * 0.5, 0.05)

    def apply_boundary_clearance(mask: np.ndarray, boundary_clearance: float) -> np.ndarray:
        """
        Zero out cells that are too close to the grid boundary to keep endpoints off walls.
        Falls back to the original mask if trimming would remove every free cell.
        """
        cells = int(math.ceil(boundary_clearance / resolution))
        if cells <= 0:
            return mask
        trimmed = mask.copy()
        trimmed[:cells, :] = False
        trimmed[-cells:, :] = False
        trimmed[:, :cells] = False
        trimmed[:, -cells:] = False
        return trimmed if int(np.count_nonzero(trimmed)) > 0 else mask

    safe_mask = free_mask
    if clearance_margin > 0.0:
        inflated = grid_map.inflate(clearance_margin)
        inflated_free_mask = inflated.data == 0
        if int(np.count_nonzero(inflated_free_mask)) > 0:
            safe_mask = inflated_free_mask
        else:
            clearance_margin = 0.0  # fall back if inflation removes every free cell

    # Use a slightly looser mask to pick endpoints so the goal can shift farther left while remaining reachable.
    endpoint_clearance = float(cfg.get("start_goal_clearance_factor", 0.55)) * clearance_margin
    endpoint_mask = safe_mask
    if clearance_margin > 0.0 and endpoint_clearance < clearance_margin:
        inflated = grid_map.inflate(endpoint_clearance)
        endpoint_free_mask = inflated.data == 0
        if int(np.count_nonzero(endpoint_free_mask)) > 0:
            endpoint_mask = endpoint_free_mask

    boundary_clearance = float(cfg.get("boundary_clearance", max(0.8, clearance_margin * 0.75)))
    safe_mask = apply_boundary_clearance(safe_mask, boundary_clearance)
    endpoint_mask = apply_boundary_clearance(endpoint_mask, boundary_clearance)
    clear_free_cells = int(np.count_nonzero(safe_mask))

    footprint = OrientedBoxFootprint(footprint_length, footprint_width)
    pose_checker = GridFootprintChecker(grid_map, footprint, theta_bins=72)
    pose_checker_coarse = GridFootprintChecker(grid_map, footprint, theta_bins=48)

    safe_start_g, safe_goal_g, _, _ = farthest_free_cells(safe_mask)
    start_g, goal_g, max_steps, comp_size = farthest_free_cells(endpoint_mask)

    def component_mask(mask: np.ndarray, seed_yx: Tuple[int, int]) -> np.ndarray:
        """Return the connected component within mask containing seed_yx."""
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        sy, sx = seed_yx
        if not mask[sy, sx]:
            return visited
        dq = deque([(sy, sx)])
        visited[sy, sx] = True
        while dq:
            cy, cx = dq.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dq.append((ny, nx))
        return visited

    def shift_left(cell: Tuple[int, int], mask: np.ndarray, max_cells: int = 24) -> Tuple[int, int]:
        """Slide a grid cell leftward to a nearby free spot to give the real map endpoints more margin."""
        gx, gy = cell
        for dx in range(1, max_cells + 1):
            cand = gx - dx
            if cand < 0:
                break
            if mask[gy, cand]:
                return (cand, gy)
        return cell

    def distance_from(start_yx: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
        """Grid-distance flood fill limited to the True cells of `mask`."""
        h, w = mask.shape
        dist = np.full(mask.shape, -1, dtype=int)
        sy, sx = start_yx
        if not mask[sy, sx]:
            return dist
        dq = deque([(sy, sx)])
        dist[sy, sx] = 0
        while dq:
            cy, cx = dq.popleft()
            cd = dist[cy, cx]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = cd + 1
                    dq.append((ny, nx))
        return dist

    comp_mask = component_mask(endpoint_mask, (start_g[1], start_g[0]))
    manual_start = start_override or cfg.get("start_override")
    manual_goal = goal_override or cfg.get("goal_override")

    def world_to_grid(x: float, y: float) -> Tuple[int, int]:
        gx = int(round((x - origin_xy[0]) / resolution))
        gy = int(round((y - origin_xy[1]) / resolution))
        return gx, gy

    def validate_manual_pose(x: float, y: float, heading: float, label: str) -> Tuple[int, int]:
        gx, gy = world_to_grid(x, y)
        if not (0 <= gx < grid_mask.shape[1] and 0 <= gy < grid_mask.shape[0]):
            raise ValueError(f"{label} pose ({x:.2f}, {y:.2f}) is outside the map bounds.")
        if grid_mask[gy, gx] != 0:
            raise ValueError(f"{label} pose ({x:.2f}, {y:.2f}) is not in free space (occupied/unknown cell).")
        if not endpoint_mask[gy, gx]:
            print(
                f"Warning: {label} pose ({x:.2f}, {y:.2f}) is outside the clearance-inflated free space; using it anyway."
            )
        if pose_checker.collides_pose(x, y, heading):
            raise ValueError(f"{label} pose ({x:.2f}, {y:.2f}) collides with obstacles for heading {heading:.2f} rad.")
        return gx, gy

    def nearest_reachable_cell(target_xy: Tuple[float, float], mask: np.ndarray) -> Optional[Tuple[int, int]]:
        tgx, tgy = world_to_grid(target_xy[0], target_xy[1])
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            return None
        deltas = candidates - np.array([tgy, tgx])
        d2 = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        cy, cx = candidates[int(np.argmin(d2))]
        return int(cx), int(cy)

    use_manual = manual_start is not None or manual_goal is not None
    preferred_applied = False
    if comp_mask.any() and not use_manual and cfg.get("use_preferred_endpoints", True):
        preferred_start = tuple(cfg.get("preferred_start", (2.0, 7.0)))
        preferred_goal = tuple(cfg.get("preferred_goal", (-28.0, 7.0)))
        candidate_mask = endpoint_mask & comp_mask
        pref_start_cell = nearest_reachable_cell(preferred_start, candidate_mask)
        pref_goal_cell = nearest_reachable_cell(preferred_goal, candidate_mask) if pref_start_cell else None
        if pref_start_cell and pref_goal_cell:
            start_g, goal_g = pref_start_cell, pref_goal_cell
            preferred_applied = True

    if use_manual:
        default_start_xy = (start_g[0] * resolution + origin_xy[0], start_g[1] * resolution + origin_xy[1])
        default_goal_xy = (goal_g[0] * resolution + origin_xy[0], goal_g[1] * resolution + origin_xy[1])
        start_x = manual_start[0] if manual_start else default_start_xy[0]
        start_y = manual_start[1] if manual_start else default_start_xy[1]
        goal_x = manual_goal[0] if manual_goal else default_goal_xy[0]
        goal_y = manual_goal[1] if manual_goal else default_goal_xy[1]
        heading_guess = math.atan2(goal_y - start_y, goal_x - start_x)
        start_heading = manual_start[2] if manual_start and len(manual_start) > 2 and manual_start[2] is not None else heading_guess
        goal_heading = manual_goal[2] if manual_goal and len(manual_goal) > 2 and manual_goal[2] is not None else heading_guess
        start_cell = validate_manual_pose(start_x, start_y, start_heading, "Start")
        goal_cell = validate_manual_pose(goal_x, goal_y, goal_heading, "Goal")
        mask_for_connectivity = endpoint_mask
        if not endpoint_mask[start_cell[1], start_cell[0]] or not endpoint_mask[goal_cell[1], goal_cell[0]]:
            mask_for_connectivity = free_mask
        comp_mask_override = component_mask(mask_for_connectivity, (start_cell[1], start_cell[0]))
        if comp_mask_override.any() and not comp_mask_override[goal_cell[1], goal_cell[0]]:
            raise ValueError("Manual start/goal are not in the same free-space component; adjust the coordinates.")
        start_g, goal_g = (start_cell[0], start_cell[1]), (goal_cell[0], goal_cell[1])
        heading = start_heading
        goal_heading = goal_heading
    else:
        goal_heading = None
    if comp_mask.any() and not use_manual and not preferred_applied:
        cols = np.where(comp_mask.any(axis=0))[0]
        center_row = comp_mask.shape[0] // 2
        # Bias the start upward a bit without hugging the boundary, and avoid parking the goal on the far-left wall.
        preferred_start_row = max(0, center_row - int(comp_mask.shape[0] * 0.10))
        goal_left_limit = min(cols[-1], cols[0] + int(max(2, round(grid_mask.shape[1] * 0.06))))
        goal_cols_scan = [c for c in cols if c >= goal_left_limit] + [c for c in cols if c < goal_left_limit]

        def rows_sorted(col: int, bias_row: int):
            rows_on_col = np.where(comp_mask[:, col])[0]
            return sorted(rows_on_col, key=lambda r: (abs(r - bias_row), abs(r - center_row)))

        best = None  # (start_col, start_row, goal_col, goal_row)
        for s_col in reversed(cols):  # prefer the rightmost viable start
            start_rows = rows_sorted(int(s_col), preferred_start_row)
            if not start_rows:
                continue
            for s_row in start_rows:
                dist_map = distance_from((s_row, s_col), comp_mask)
                start_x = s_col * resolution + origin_xy[0]
                start_y = s_row * resolution + origin_xy[1]
                if pose_checker_coarse.collides_pose(start_x, start_y, 0.0):
                    continue
                for g_col in goal_cols_scan:  # prefer goals that are left but not hugging the wall
                    goal_rows = rows_sorted(int(g_col), center_row)
                    if not goal_rows:
                        continue
                    for g_row in goal_rows:
                        if dist_map[g_row, g_col] < 0:
                            continue
                        gx = g_col * resolution + origin_xy[0]
                        gy = g_row * resolution + origin_xy[1]
                        heading = math.atan2(gy - start_y, gx - start_x)
                        if pose_checker_coarse.collides_pose(start_x, start_y, heading):
                            continue
                        if pose_checker_coarse.collides_pose(gx, gy, heading):
                            continue
                        if pose_checker.collides_pose(start_x, start_y, heading):
                            continue
                        if pose_checker.collides_pose(gx, gy, heading):
                            continue
                        cand = (s_col, s_row, g_col, g_row)
                        if (best is None) or (g_col < best[2]) or (g_col == best[2] and s_col > best[0]):
                            best = cand
                        break  # stop scanning rows on this goal column
                if best and best[0] == s_col and best[2] == goal_cols_scan[0]:
                    break  # already achieved the leftmost possible goal within the preferred band for this start
            if best and best[2] == goal_cols_scan[0]:
                break  # cannot do better than the leftmost preferred goal overall
        if best:
            start_g, goal_g = (best[0], best[1]), (best[2], best[3])
    elif not use_manual and not preferred_applied:
        goal_g = shift_left(goal_g, safe_mask, max_cells=28)
    if not use_manual:
        start_x = start_g[0] * resolution + origin_xy[0]
        start_y = start_g[1] * resolution + origin_xy[1]
        goal_x = goal_g[0] * resolution + origin_xy[0]
        goal_y = goal_g[1] * resolution + origin_xy[1]
        heading = math.atan2(goal_y - start_y, goal_x - start_x)
        if pose_checker.collides_pose(start_x, start_y, heading) or pose_checker.collides_pose(goal_x, goal_y, heading):
            # Fall back to the conservative endpoints if the relaxed picks still collide.
            start_g, goal_g = safe_start_g, safe_goal_g
            start_x = start_g[0] * resolution + origin_xy[0]
            start_y = start_g[1] * resolution + origin_xy[1]
            goal_x = goal_g[0] * resolution + origin_xy[0]
            goal_y = goal_g[1] * resolution + origin_xy[1]
            heading = math.atan2(goal_y - start_y, goal_x - start_x)
            goal_heading = heading
    start = AckermannState(start_x, start_y, heading)
    goal = AckermannState(goal_x, goal_y, goal_heading if goal_heading is not None else heading)

    size = (grid_mask.shape[1] * resolution, grid_mask.shape[0] * resolution)
    meta = dict(
        resolution=resolution,
        size=size,
        free_cells=int(np.count_nonzero(free_mask)),
        obstacle_cells=int(np.count_nonzero(grid_mask)),
        component_size=comp_size,
        max_grid_distance=max_steps,
        origin=origin_xy,
        clear_free_cells=clear_free_cells,
        start_goal_clearance=clearance_margin,
        boundary_clearance=boundary_clearance,
    )
    return grid_map, start, goal, meta


def adaptive_goal_tolerances(
    grid_map: GridMap, is_large_map: bool, collision_step: float, xy_resolution: float
) -> Tuple[float, float]:
    """
    Scale goal tolerances to the lattice fidelity so planners can actually
    dock at the goal without wasting time in sub-meter jitter.
    """
    base_xy = max(0.18, collision_step * 1.5, xy_resolution * 0.4)
    goal_xy_tol = max(base_xy, 0.28 if is_large_map else 0.22)
    goal_theta_tol = math.radians(10.0 if is_large_map else 7.0)
    return goal_xy_tol, goal_theta_tol


DEFAULT_START, DEFAULT_GOAL = compute_start_goal(FOREST_MAP_KWARGS)

# Planner budgets tuned for sub-second runs.
APF_TIMEOUT = 5.0
APF_MAX_ITERS = 12_000
HYBRID_TIMEOUT = 5.0
HYBRID_MAX_NODES = 8_000
DQN_TIMEOUT = 5.0
DQN_MAX_NODES = 8_000
RRT_TIMEOUT = 10.0
RRT_MAX_ITER = 30_000


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
    legend_labels: Optional[List[str]] = None,
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

    color_cycle = list(PLANNER_COLOR_MAP.values()) + FALLBACK_COLORS
    for idx, (label, path, stats) in enumerate(planner_results):
        base_label = strip_variant_suffix(label)
        color = PLANNER_COLOR_MAP.get(base_label, color_cycle[idx % len(color_cycle)])
        linestyle = "--" if "orig" in label.lower() else "-"
        linewidth = PRIMARY_LINEWIDTH * (0.85 if "orig" in label.lower() else 1.0)
        legend_label = legend_labels[idx] if legend_labels and idx < len(legend_labels) else label
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(
            xs,
            ys,
            linewidth=linewidth,
            color=color,
            solid_capstyle="round",
            label=legend_label,
            linestyle=linestyle,
        )
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
    slug = slug.replace("_with_", "_")
    legacy_aliases = {
        "large_map": "large_map_small_gap",
        "small_map": "small_map_small_gap",
    }
    slug = legacy_aliases.get(slug, slug)
    if slug not in FOREST_VARIANTS:
        options = ", ".join(sorted(FOREST_VARIANTS))
        raise ValueError(f"Unknown variant '{name}'. Choose from: {options}.")
    return slug


def build_variant(
    variant: str,
    start_override: Optional[Tuple[float, float, Optional[float]]] = None,
    goal_override: Optional[Tuple[float, float, Optional[float]]] = None,
):
    variant_slug = normalize_variant(variant)
    cfg = FOREST_VARIANTS[variant_slug]
    title = cfg.get("title", f"Forest ({variant_slug})")

    if cfg.get("loader") == "real_env1":
        grid_map, start, goal, meta = load_real_env1_map(cfg, start_override=start_override, goal_override=goal_override)
        map_kwargs = {
            "size": meta["size"],
            "resolution": meta["resolution"],
            "num_trees": meta.get("obstacle_cells", "obs_cells"),
            "free_cells": meta.get("free_cells", None),
            "obstacle_cells": meta.get("obstacle_cells", None),
            "clear_free_cells": meta.get("clear_free_cells", None),
            "start_goal_clearance": meta.get("start_goal_clearance", None),
        }
        return variant_slug, title, map_kwargs, start, goal, grid_map

    if start_override or goal_override:
        print("Manual start/goal overrides are only applied to the real_env1 map; ignoring for synthetic variants.")

    map_kwargs = dict(FOREST_MAP_KWARGS)
    map_kwargs.update(cfg.get("map_kwargs", {}))

    start, goal = compute_start_goal(map_kwargs)
    map_kwargs["keep_clear"] = [(start.x, start.y), (goal.x, goal.y)]
    return variant_slug, title, map_kwargs, start, goal, None


def plan_forest_scene(
    variant: str = "large_map_small_gap",
    out_dir: Optional[Path] = None,
    postprocess: str = "none",
    postprocess_allow: Optional[Set[str]] = None,
    start_override: Optional[Tuple[float, float, Optional[float]]] = None,
    goal_override: Optional[Tuple[float, float, Optional[float]]] = None,
    legend_style: str = "standard",
):
    output_dir = out_dir or (Path(__file__).resolve().parent / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    post_mode = (postprocess or "none").strip().lower()
    postprocess_allow = {strip_variant_suffix(p) for p in postprocess_allow} if postprocess_allow else None
    variant_slug, title, map_kwargs, start, goal, grid_map_override = build_variant(
        variant, start_override=start_override, goal_override=goal_override
    )
    grid_map = grid_map_override or make_forest_map(**map_kwargs)
    size_x, size_y = map_kwargs.get(
        "size", (grid_map.data.shape[1] * grid_map.resolution, grid_map.data.shape[0] * grid_map.resolution)
    )
    is_large_map = max(size_x, size_y) > 30.0
    params = AckermannParams()
    footprint = OrientedBoxFootprint(length=0.924, width=0.740)
    base_collision_step = default_collision_step(grid_map.resolution, preferred=0.15, max_step=0.25)
    apf_collision_step = default_collision_step(grid_map.resolution, preferred=0.10, max_step=0.18)
    lattice_xy_res = 0.60
    goal_xy_tol, goal_theta_tol = adaptive_goal_tolerances(
        grid_map, is_large_map, base_collision_step, lattice_xy_res
    )

    print(
        f"Variant '{variant_slug}': size=({size_x:.2f}, {size_y:.2f}) m, "
        f"trees/obstacles={map_kwargs.get('num_trees', 'n/a')}, resolution={map_kwargs.get('resolution', grid_map.resolution):.2f} m"
    )
    if map_kwargs.get("free_cells") is not None:
        print(
            f"  Cells: free={map_kwargs['free_cells']:,}, "
            f"obstacles/unknown={map_kwargs.get('obstacle_cells', 'n/a')}"
        )
    if map_kwargs.get("clear_free_cells") is not None:
        clearance = map_kwargs.get("start_goal_clearance")
        clearance_note = f" (>= {clearance:.2f} m from obstacles)" if clearance else ""
        print(f"  Start/goal clear cells: {map_kwargs['clear_free_cells']:,}{clearance_note}")
    if map_kwargs.get("wall_y") is not None:
        gap_center, gap_width = map_kwargs["wall_gap"]
        print(f"  Wall at y={map_kwargs['wall_y']:.2f} with gap center={gap_center:.2f}, width={gap_width:.2f}")
    print(
        f"  Goal tolerance: xy<={goal_xy_tol:.2f} m, theta<={math.degrees(goal_theta_tol):.1f} deg "
        f"(auto from resolution {grid_map.resolution:.2f} m)"
    )

    step_length = 0.8 if is_large_map else 0.7
    short_primitives = default_primitives(params, step_length=step_length)
    base_kwargs = dict(
        xy_resolution=lattice_xy_res,
        collision_step=base_collision_step,
        goal_xy_tol=goal_xy_tol,
        goal_theta_tol=goal_theta_tol,
        theta_bins=48,
    )
    # Stronger heuristic to offset the tighter turning radius (28° steer) speeding up search on large maps.
    hybrid_kwargs = dict(base_kwargs, heuristic_weight=2.5)
    dqn_kwargs = dict(base_kwargs, dqn_top_k=3, anchor_inflation=1.6, dqn_weight=1.1)
    rrt_kwargs = dict(
        goal_sample_rate=0.90,  # stronger bias helps when heading penalties tighten with higher steering limits
        neighbor_radius=5.0 if is_large_map else 2.8,
        step_time=1.20 if is_large_map else 0.50,  # small maps use tighter rollouts
        velocity=1.0,
        goal_xy_tol=max(goal_xy_tol, 0.40 if is_large_map else 0.30),
        goal_theta_tol=max(goal_theta_tol, math.radians(45.0 if is_large_map else 25.0)),
        goal_check_freq=1,
        seed_steps=4,
        collision_step=default_collision_step(grid_map.resolution, preferred=0.12 if is_large_map else 0.10, max_step=0.20),
        rewire=False,
    )
    rrt_timeout = RRT_TIMEOUT
    rrt_max_iter = RRT_MAX_ITER
    apf_kwargs = dict(
        step_size=0.22,
        goal_tol=max(goal_xy_tol, 0.25 if is_large_map else 0.18),
        repulse_radius=1.1,
        obstacle_gain=1.0,
        goal_gain=1.2,
        max_iters=APF_MAX_ITERS,
        collision_step=apf_collision_step,
        stall_steps=80,
        theta_bins=64,
        min_step=0.05,
        jitter_angle=0.6,
        heading_rate=0.65,
    )
    # Variants can override RRT tuning; the sparse small-map-with-gap benefits from longer strides + looser heading.
    if not is_large_map and variant_slug == "small_map_large_gap":
        rrt_kwargs.update(
            dict(
                goal_sample_rate=0.90,
                neighbor_radius=3.5,
                step_time=0.60,
                goal_theta_tol=max(goal_theta_tol, math.radians(35.0)),
                seed_steps=6,
                collision_step=default_collision_step(grid_map.resolution, preferred=0.10, max_step=0.20),
            )
        )
    if is_large_map and variant_slug == "large_map_small_gap":
        rrt_kwargs.update(
            dict(
                goal_sample_rate=0.20,
                neighbor_radius=3.0,
                step_time=0.40,
                velocity=1.0,
                goal_xy_tol=max(goal_xy_tol, 0.42),
                goal_theta_tol=max(goal_theta_tol, math.radians(75.0)),
                seed_steps=0,
                collision_step=default_collision_step(grid_map.resolution, preferred=0.10, max_step=0.20),
                rewire=False,
                steer_jitter=0.25,
            )
        )
        rrt_timeout = max(rrt_timeout, 18.0)
        rrt_max_iter = max(rrt_max_iter, 60_000)

    # Slightly loosen APF heading dynamics on the hardest scene to preserve success while smoothing later.
    apf_kwargs_variant = dict(apf_kwargs)
    if variant_slug == "large_map_small_gap":
        apf_kwargs_variant.update(dict(jitter_angle=0.8, heading_rate=0.72))
    if variant_slug == "real_env1":
        apf_kwargs_variant.update(
            dict(
                step_size=0.24,
                repulse_radius=0.95,
                obstacle_gain=0.85,
                goal_gain=1.35,
                heading_rate=0.70,
                jitter_angle=0.5,
                stall_steps=90,
                min_step=0.06,
            )
        )

    post_checker = GridFootprintChecker(grid_map, footprint, theta_bins=72)
    planners = [
        (APF_NAME, APFPlanner(grid_map, footprint, params, **apf_kwargs_variant)),
        (HYBRID_NAME, HybridAStarPlanner(grid_map, footprint, params, primitives=short_primitives, **hybrid_kwargs)),
        (DQNHYBRID_NAME, DQNHybridAStarPlanner(grid_map, footprint, params, primitives=short_primitives, **dqn_kwargs)),
        (RRT_NAME, RRTStarPlanner(grid_map, footprint, params, **rrt_kwargs)),
    ]

    records = []
    planner_results = []
    path_export_rows = []

    def paths_match(a: List[AckermannState], b: List[AckermannState], pos_tol: float = 1e-3, theta_tol: float = 1e-3) -> bool:
        """Check if two paths are effectively identical."""
        if len(a) != len(b):
            return False
        for pa, pb in zip(a, b):
            if math.hypot(pa.x - pb.x, pa.y - pb.y) > pos_tol:
                return False
            dtheta = abs((pa.theta - pb.theta + math.pi) % (2 * math.pi) - math.pi)
            if dtheta > theta_tol:
                return False
        return True

    def add_path_rows(label: str, path_seq: List[AckermannState]):
        if not path_seq:
            return
        base_label = strip_variant_suffix(label)
        lower_label = label.lower()
        kind = "stomp" if "stomp" in lower_label else ("orig" if "orig" in lower_label else "final")
        for idx, pose in enumerate(path_seq):
            path_export_rows.append(
                {
                    "variant": variant_slug,
                    "planner": base_label,
                    "path_kind": kind,
                    "point_idx": idx,
                    "x": pose.x,
                    "y": pose.y,
                    "theta": pose.theta,
                }
            )
    for label, planner in planners:
        base_label = strip_variant_suffix(label)
        allow_postprocess = postprocess_allow is None or base_label in postprocess_allow
        t0 = time.time()
        if isinstance(planner, RRTStarPlanner):
            path, stats = planner.plan(start, goal, max_iter=rrt_max_iter, timeout=rrt_timeout)
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
        raw_path = list(path)
        path_len = path_length(raw_path)
        plot_entries = []
        if success and post_mode in ("stomp", "feng") and allow_postprocess:
            plot_entries.append((f"{label} (orig)", raw_path, dict(stats)))
            opt_t0 = time.time()
            post_input_path = reorient_path_headings(raw_path)
            if label == APF_NAME:
                # Pre-smooth APF headings/positions to reduce zig-zags before optimization.
                pre_smooth_path, ok_pre = laplacian_smooth_path(
                    post_input_path,
                    post_checker,
                    ds_check=apf_collision_step,
                    alpha=0.14,
                    passes=3,
                )
                if ok_pre:
                    post_input_path = pre_smooth_path
            opt_path = []
            opt_info = {}
            suffix = post_mode
            if post_mode == "stomp":
                opt_path, opt_info = stomp_optimize_path(
                    post_input_path,
                    grid_map,
                    footprint,
                    params,
                    step_size=0.25,
                    ds_check=base_collision_step,
                    rollouts=96,
                    iters=50,
                    lambda_=0.6,
                    w_clear=0.6,
                    w_goal=2.5,
                    w_prior=0.1,
                    w_track=0.05,
                    allow_reverse=True,
                    seed=13,
                    goal=goal,
                    laplacian_strength=0.4,
                )
                suffix = "stomp"
                if not opt_path:
                    opt_path = post_input_path
                    opt_info = opt_info or {}
                    opt_info.setdefault("reason", "fallback_to_orig_path")
                    opt_info.setdefault("improved", False)
                    opt_info.setdefault("best_cost", opt_info.get("base_cost", float("inf")))
            else:
                opt_path, opt_info = feng_optimize_path(
                    post_input_path,
                    grid_map,
                    footprint,
                    params,
                    goal=goal,
                    degree=5,
                    max_segments=10,
                    samples_per_seg=14,
                    rect_size=2.8,
                    safety_margin=0.2,
                    output_step=0.22,
                    ds_check=base_collision_step,
                    iters=55,
                    lr=0.1,
                    w_terrain=1.4,
                    w_safety=28.0,
                    w_dyn=5.0,
                    w_cont=6.0,
                    w_track=0.12,
                    seed=7,
                )
                suffix = "feng"
                max_dev = max_xy_deviation(raw_path, opt_path, samples=96) if opt_path else float("inf")
                no_change_tol = max(grid_map.resolution * 1.2, 0.08)
                opt_info["max_deviation"] = max_dev
                opt_info["no_change_tol"] = no_change_tol
                no_change = math.isfinite(max_dev) and max_dev <= no_change_tol
                # If Feng returns empty/unchanged, retry Feng with a heavier second pass (same method, different seed).
                retry_needed = (
                    not opt_path
                    or opt_info.get("reason") == "fallback_to_input"
                    or paths_match(opt_path, post_input_path)
                    or no_change
                )
                if retry_needed:
                    second_path, second_info = feng_optimize_path(
                        post_input_path,
                        grid_map,
                        footprint,
                        params,
                        goal=goal,
                        degree=5,
                        max_segments=12,
                        samples_per_seg=16,
                        rect_size=3.0,
                        safety_margin=0.22,
                        output_step=0.22,
                        ds_check=base_collision_step,
                        iters=80,
                        lr=0.09,
                        w_terrain=1.4,
                        w_safety=28.0,
                        w_dyn=5.0,
                        w_cont=6.0,
                        w_track=0.10,
                        seed=19,
                    )
                    opt_info["second_pass"] = second_info
                    if second_path:
                        opt_path = second_path
                        opt_info["reason"] = second_info.get("reason", "feng_second_pass")
                        opt_info["best_cost"] = second_info.get("best_cost", opt_info.get("best_cost"))
                        opt_info["improved"] = second_info.get("improved", opt_info.get("improved"))
                        opt_info["iters_run"] = second_info.get("iters_run", opt_info.get("iters_run"))
                        opt_info["selected_pass"] = "second"
                        max_dev = max_xy_deviation(raw_path, opt_path, samples=96)
                        opt_info["max_deviation"] = max_dev
                    else:
                        opt_info.setdefault("reason", "feng_no_change")
            if not opt_path:
                opt_path = post_input_path
                opt_info = opt_info or {}
                opt_info.setdefault("reason", "fallback_to_orig_path")
                opt_info.setdefault("improved", False)
                opt_info.setdefault("best_cost", opt_info.get("base_cost", float("inf")))
            # Secondary smoothing when postprocessing could not change the geometry (APF often hugs obstacles).
            post_max_dev = max_xy_deviation(raw_path, opt_path, samples=120) if opt_path else float("inf")
            smooth_tol = max(grid_map.resolution * 0.8, 0.06)
            if not opt_path or post_max_dev <= smooth_tol:
                smooth_step = min(base_collision_step, apf_collision_step)
                smooth_alpha = 0.22 if label == APF_NAME else 0.14
                smoothed_path, smoothed = laplacian_smooth_path(
                    opt_path or post_input_path,
                    post_checker,
                    ds_check=smooth_step,
                    alpha=smooth_alpha,
                    passes=5 if label == APF_NAME else 3,
                )
                if smoothed:
                    smooth_dev = max_xy_deviation(raw_path, smoothed_path, samples=120)
                    if smooth_dev > post_max_dev + 1e-6:
                        opt_info.setdefault("fallback", {})["mode"] = "laplacian"
                        opt_info["reason"] = opt_info.get("reason", "postprocess_no_change")
                        opt_info["base_max_deviation"] = post_max_dev
                        opt_info["max_deviation"] = smooth_dev
                        opt_path = smoothed_path
                        suffix = f"{suffix}+smooth"
                        post_max_dev = smooth_dev
            stats["postprocess"] = opt_info
            stats["postprocess_time"] = time.time() - opt_t0
            path_len = path_length(opt_path)
            plot_entries.append((f"{label} ({suffix})", opt_path, dict(stats)))
            # Keep the optimized curve (including STOMP fallbacks) in exports/plots.
            path = opt_path
        print(
            f"{label}: success={success}, time={stats.get('time',0):.2f}s, "
            f"path_len={path_len:.2f}, expansions/nodes={expansions}"
        )
        failure_reason = stats.get("failure_reason", "")
        remediations = stats.get("remediations", [])
        post_info = stats.get("postprocess", {}) if isinstance(stats.get("postprocess"), dict) else {}
        if not success and failure_reason:
            remediation_str = ";".join(remediations) if remediations else "none"
            print(f"  failure_reason={failure_reason}, remediations={remediation_str}")
        records.append(
            {
                "variant": variant_slug,
                "planner": label,
                "success": success,
                "path_length": path_len,
                "expansions_or_nodes": expansions,
                "time": stats.get("time", 0.0),
                "time_wall": stats.get("time_wall", 0.0),
                "failure_reason": failure_reason,
                "remediations": ";".join(remediations) if remediations else "",
                "postprocess_mode": post_mode if post_mode in ("stomp", "feng") else "none",
                "postprocess_reason": post_info.get("reason", ""),
                "postprocess_improved": post_info.get("improved", ""),
                "postprocess_selected_cost": post_info.get("selected_cost", ""),
                "postprocess_time": stats.get("postprocess_time", 0.0),
            }
        )
        if success:
            if plot_entries:
                planner_results.extend(plot_entries)
                for lbl, pth, _ in plot_entries:
                    add_path_rows(lbl, pth)
            else:
                planner_results.append((label, path, stats))
                add_path_rows(label, path)

    legend_labels = None
    if legend_style == "ambiguous" and planner_results:
        legend_labels = [
            AMBIGUOUS_LEGEND_NAMES[i % len(AMBIGUOUS_LEGEND_NAMES)] for i in range(len(planner_results))
        ]

    saved_plot = plot_with_boxes(
        title,
        variant_slug,
        grid_map,
        start,
        goal,
        planner_results,
        footprint,
        output_dir,
        legend_labels=legend_labels,
    )
    if saved_plot:
        print(f"Saved plot: {saved_plot}")
        for rec in records:
            rec["plot_path"] = str(saved_plot)

    run_info = dict(
        variant=variant_slug,
        start_x=start.x,
        start_y=start.y,
        start_theta=start.theta,
        goal_x=goal.x,
        goal_y=goal.y,
        goal_theta=goal.theta,
        resolution=map_kwargs.get("resolution", grid_map.resolution),
        size_x=size_x,
        size_y=size_y,
        num_trees=map_kwargs.get("num_trees", ""),
        postprocess=post_mode if post_mode in ("stomp", "feng") else "none",
        postprocess_allow=",".join(sorted(postprocess_allow)) if postprocess_allow else "all",
        legend_style=legend_style or "standard",
    )

    return records, saved_plot, run_info, path_export_rows


def parse_pose_arg(arg: str, name: str) -> Tuple[float, float, Optional[float]]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"{name} must be 'x,y' or 'x,y,theta' (meters, radians).")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} values must be numeric (meters, radians).") from exc
    if len(values) == 2:
        values.append(None)
    return values[0], values[1], values[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Forest scene planner presets.")
    parser.add_argument(
        "--variant",
        default="large_map_small_gap",
        help="Choose from: large_map_large_gap, large_map_small_gap, small_map_large_gap, small_map_small_gap, real_env1, or 'all' to run every preset.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Base folder for results. A time-stamped subfolder is created inside.",
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "stomp", "feng"],
        default="none",
        help="Apply a trajectory postprocessor after planning (default: none).",
    )
    parser.add_argument(
        "--postprocess-allow",
        default="all",
        help="Comma-separated planners to postprocess (e.g., 'APF,RRT'). Use 'all' to keep the default behavior.",
    )
    parser.add_argument(
        "--legend-style",
        choices=["standard", "ambiguous"],
        default="standard",
        help="Legend labeling style. Choose 'ambiguous' for supplementary figures with similar-looking legend entries.",
    )
    parser.add_argument(
        "--start",
        type=lambda s: parse_pose_arg(s, "start"),
        help="Manual start pose 'x,y[,theta]' in meters/radians (real_env1 only).",
    )
    parser.add_argument(
        "--goal",
        type=lambda s: parse_pose_arg(s, "goal"),
        help="Manual goal pose 'x,y[,theta]' in meters/radians (real_env1 only).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {out_dir}")

    all_records = []
    all_run_info = []
    all_path_rows = []
    postprocess_allow = parse_postprocess_allow(args.postprocess_allow)
    variants = list(FOREST_VARIANTS) if args.variant.strip().lower() in ("all", "any") else [args.variant]
    for variant_name in variants:
        print("\n" + "=" * 60)
        records, _, run_info, path_rows = plan_forest_scene(
            variant_name,
            out_dir=out_dir,
            postprocess=args.postprocess,
            postprocess_allow=postprocess_allow,
            start_override=args.start,
            goal_override=args.goal,
            legend_style=args.legend_style,
        )
        all_records.extend(records)
        all_run_info.append(run_info)
        all_path_rows.extend(path_rows)

    if all_records:
        csv_path = out_dir / "results.csv"
        fieldnames = [
            "variant",
            "planner",
            "success",
            "path_length",
            "expansions_or_nodes",
            "time",
            "time_wall",
            "failure_reason",
            "remediations",
            "postprocess_mode",
            "postprocess_reason",
            "postprocess_improved",
            "postprocess_selected_cost",
            "postprocess_time",
            "plot_path",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_records:
                writer.writerow(row)
        print(f"\nSaved summary: {csv_path}")
    else:
        print("\nNo results to save.")

    def write_excel_summary(path, run_rows, path_rows):
        try:
            from openpyxl import Workbook
        except ImportError:
            print("openpyxl not installed; skipping Excel export. Install with 'pip install openpyxl'.", file=sys.stderr)
            return
        wb = Workbook()
        meta = wb.active
        meta.title = "runs"
        meta_headers = [
            "variant",
            "start_x",
            "start_y",
            "start_theta",
            "goal_x",
            "goal_y",
            "goal_theta",
            "resolution",
            "size_x",
            "size_y",
            "num_trees",
            "postprocess",
            "postprocess_allow",
            "legend_style",
        ]
        meta.append(meta_headers)
        for row in run_rows:
            meta.append([row[h] for h in meta_headers])

        path_sheet = wb.create_sheet("paths")
        path_headers = ["variant", "planner", "path_kind", "point_idx", "x", "y", "theta"]
        path_sheet.append(path_headers)
        for row in path_rows:
            path_sheet.append([row[h] for h in path_headers])

        wb.save(path)

    if all_run_info:
        excel_path = out_dir / "results.xlsx"
        write_excel_summary(excel_path, all_run_info, all_path_rows)
        print(f"Saved Excel paths: {excel_path}")
