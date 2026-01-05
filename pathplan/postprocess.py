import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .common import clamp, heading_diff, wrap_angle, default_collision_step
from .geometry import GridFootprintChecker
from .map_utils import GridMap
from .robot import AckermannParams, AckermannState, propagate

# Discovery notes:
# - forward simulation of controls: pathplan.robot.propagate
# - collision checking for trajectories: pathplan.geometry.GridFootprintChecker.motion_collides / collides_path
# - occupancy sampling: pathplan.map_utils.GridMap.is_occupied and GridMap.data

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def path_to_controls(
    path: Sequence[AckermannState],
    params: AckermannParams,
    step_size: float,
    allow_reverse: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Infer steering angles and directions from a sequence of poses.

    Uses curvature = dtheta/ds with a constant-steer approximation per segment.
    Returns (steers, directions, step_sizes) where step_sizes preserves the
    original segment lengths so reconstruction does not overshoot the source path.
    """
    if step_size <= 0 or len(path) < 2:
        return None, None, None
    steers: List[float] = []
    directions: List[int] = []
    step_sizes: List[float] = []
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        dtheta = heading_diff(b.theta, a.theta)
        # Approximate traveled arc length instead of straight-line chord length to
        # avoid overestimating curvature on tight turns.
        if abs(dtheta) > 1e-6 and dist > 1e-6:
            chord_ratio = 2.0 * abs(math.sin(dtheta / 2.0))
            arc_length = abs(dtheta) * dist / max(chord_ratio, 1e-9)
        else:
            arc_length = dist
        # Round rather than ceil so reconstructed arc length stays close to the
        # original segment length, avoiding overshoot that can introduce collisions.
        n_steps = max(1, int(round(arc_length / step_size)))
        forward_dot = dx * math.cos(a.theta) + dy * math.sin(a.theta)
        direction = -1 if (allow_reverse and forward_dot < -1e-3) else 1
        curvature = dtheta / (arc_length * direction + 1e-9)
        steering = math.atan(curvature * params.wheelbase)
        steering = clamp(steering, -params.max_steer, params.max_steer)
        seg_step = arc_length / float(n_steps)
        for _ in range(n_steps):
            steers.append(steering)
            directions.append(direction)
            step_sizes.append(seg_step)
    if not steers:
        return None, None, None
    return (
        np.asarray(steers, dtype=float),
        np.asarray(directions, dtype=int),
        np.asarray(step_sizes, dtype=float),
    )


def controls_to_path(
    start_state: AckermannState,
    steers: Sequence[float],
    directions: Sequence[int],
    params: AckermannParams,
    step_size: float,
    *,
    step_sizes: Optional[Sequence[float]] = None,
) -> List[AckermannState]:
    """Roll forward using propagate with constant-steer primitives."""
    if len(steers) != len(directions):
        raise ValueError("Steer and direction lengths differ")
    if step_sizes is not None and len(step_sizes) != len(steers):
        raise ValueError("Step size count does not match controls")
    path: List[AckermannState] = [start_state]
    state = start_state
    step_sizes_arr = np.asarray(step_sizes, dtype=float) if step_sizes is not None else None
    for idx, (steer, direction) in enumerate(zip(steers, directions)):
        ds = float(step_sizes_arr[idx]) if step_sizes_arr is not None else step_size
        state = propagate(state, float(steer), int(direction), ds, params)
        state = AckermannState(state.x, state.y, wrap_angle(state.theta))
        path.append(state)
    return path


def _trajectory_collides(
    path: Sequence[AckermannState],
    checker: GridFootprintChecker,
    ds_check: float,
) -> bool:
    if not path:
        return True
    for state in path:
        if checker.collides_pose(state.x, state.y, state.theta):
            return True
    for i in range(1, len(path)):
        a = path[i - 1]
        b = path[i]
        if checker.motion_collides(a.as_tuple(), b.as_tuple(), step=ds_check):
            return True
    return False


def _laplacian_smooth_1d(arr: np.ndarray, strength: float = 0.35, passes: int = 2) -> np.ndarray:
    """
    Lightweight Laplacian smoother for steering sequences to reduce zig-zags
    without drifting endpoints. Caller is responsible for clipping to bounds.
    """
    if arr.size < 3 or strength <= 0.0 or passes <= 0:
        return arr
    smoothed = arr.astype(float, copy=True)
    for _ in range(passes):
        middle = smoothed.copy()
        middle[1:-1] = smoothed[1:-1] + strength * (smoothed[:-2] - 2 * smoothed[1:-1] + smoothed[2:])
        smoothed = middle
    return smoothed


def _obstacle_distance_map(grid_map: GridMap) -> np.ndarray:
    """Chamfer distance to nearest occupied cell (meters)."""
    occ = np.asarray(grid_map.data, dtype=bool)
    h, w = occ.shape
    dist = np.full((h, w), np.inf, dtype=float)
    dist[occ] = 0.0
    sqrt2 = math.sqrt(2.0)

    for y in range(h):
        for x in range(w):
            if dist[y, x] == 0.0:
                continue
            best = dist[y, x]
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + sqrt2)
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + sqrt2)
            dist[y, x] = best

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            best = dist[y, x]
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 1.0)
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 1.0)
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + sqrt2)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + sqrt2)
            dist[y, x] = best

    dist *= grid_map.resolution
    if np.isinf(dist).any():
        finite = dist[np.isfinite(dist)]
        fallback = np.nanmax(finite) if finite.size else max(h, w) * grid_map.resolution
        dist[np.isinf(dist)] = fallback
    return dist


def _sample_distance(distance_map: np.ndarray, grid_map: GridMap, x: float, y: float) -> float:
    gx = (x - grid_map.origin[0]) / grid_map.resolution
    gy = (y - grid_map.origin[1]) / grid_map.resolution
    h, w = distance_map.shape
    if gx < 0 or gy < 0 or gx > (w - 1) or gy > (h - 1):
        return 0.0
    x0 = int(math.floor(gx))
    y0 = int(math.floor(gy))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = gx - x0
    wy = gy - y0
    d00 = distance_map[y0, x0]
    d10 = distance_map[y0, x1]
    d01 = distance_map[y1, x0]
    d11 = distance_map[y1, x1]
    d0 = d00 * (1 - wx) + d10 * wx
    d1 = d01 * (1 - wx) + d11 * wx
    return d0 * (1 - wy) + d1 * wy


def _sample_path_xy(path: Sequence[AckermannState], samples: int) -> np.ndarray:
    """Evenly resample XY points along a path for drift checks and priors."""
    if not path or samples <= 0:
        return np.zeros((0, 2), dtype=float)
    if len(path) == 1 or samples == 1:
        return np.array([[path[-1].x, path[-1].y]], dtype=float)
    idxs = np.linspace(0.0, len(path) - 1, samples)
    pts: List[Tuple[float, float]] = []
    for idx in idxs:
        i0 = int(math.floor(idx))
        i1 = min(i0 + 1, len(path) - 1)
        t = idx - i0
        x = path[i0].x + (path[i1].x - path[i0].x) * t
        y = path[i0].y + (path[i1].y - path[i0].y) * t
        pts.append((x, y))
    return np.asarray(pts, dtype=float)


def _goal_error(state: AckermannState, target: AckermannState, params: AckermannParams) -> float:
    pos_err = math.hypot(target.x - state.x, target.y - state.y)
    heading_err = abs(heading_diff(target.theta, state.theta))
    return pos_err + 0.5 * params.min_turn_radius * heading_err


def compute_cost(
    path: Sequence[AckermannState],
    steers: Sequence[float],
    checker: GridFootprintChecker,
    ds_check: float,
    w_smooth: float,
    w_steer: float,
    w_clear: float,
    clearance_map: Optional[np.ndarray],
    grid_map: GridMap,
    step_size: float,
    *,
    target_pose: Optional[AckermannState] = None,
    params: Optional[AckermannParams] = None,
    w_goal: float = 0.0,
    ref_steers: Optional[np.ndarray] = None,
    w_prior: float = 0.0,
    ref_path: Optional[Sequence[AckermannState]] = None,
    w_track: float = 0.0,
) -> float:
    """Composite cost: collision -> inf, otherwise smoothness + magnitude + clearance."""
    if not path:
        return float("inf")
    if _trajectory_collides(path, checker, ds_check):
        return float("inf")
    steer_arr = np.asarray(steers, dtype=float)
    smooth = np.sum(np.diff(steer_arr) ** 2) if steer_arr.size > 1 else 0.0
    magnitude = np.sum(steer_arr**2)
    clearance = 0.0
    if w_clear > 0.0 and clearance_map is not None:
        sample_stride = max(1, int(math.ceil(0.25 / max(step_size, 1e-6))))
        for idx in range(0, len(path), sample_stride):
            st = path[idx]
            d = _sample_distance(clearance_map, grid_map, st.x, st.y)
            clearance += 1.0 / max(d, 1e-2)
    goal_term = 0.0
    if w_goal > 0.0 and target_pose is not None and params is not None:
        end = path[-1]
        pos_err = math.hypot(target_pose.x - end.x, target_pose.y - end.y)
        heading_err = abs(heading_diff(target_pose.theta, end.theta))
        goal_term = pos_err + 0.5 * params.min_turn_radius * heading_err
    prior_term = 0.0
    if w_prior > 0.0 and ref_steers is not None and len(ref_steers) == len(steer_arr):
        prior_term = np.sum((steer_arr - ref_steers) ** 2)
    track_term = 0.0
    if w_track > 0.0 and ref_path is not None and len(ref_path) >= 2 and len(path) >= 2:
        samples = min(64, len(path), len(ref_path))
        ref_xy = _sample_path_xy(ref_path, samples)
        cand_xy = _sample_path_xy(path, samples)
        deltas = cand_xy - ref_xy
        track_term = float(np.sum(deltas**2))
    return (
        w_smooth * smooth
        + w_steer * magnitude
        + w_clear * clearance
        + w_goal * goal_term
        + w_prior * prior_term
        + w_track * track_term
    )


def stomp_optimize_path(
    path: Sequence[AckermannState],
    grid_map: GridMap,
    footprint,
    params: AckermannParams,
    *,
    step_size: float = 0.2,
    ds_check: float = 0.05,
    iters: int = 50,
    rollouts: int = 128,
    noise_std: Optional[float] = None,
    lambda_: float = 1.0,
    w_smooth: float = 1.0,
    w_steer: float = 0.1,
    w_clear: float = 0.5,
    seed: Optional[int] = None,
    allow_reverse: bool = False,
    theta_bins: int = 64,
    w_goal: float = 2.0,
    w_prior: float = 0.2,
    w_track: float = 0.2,
    goal: Optional[AckermannState] = None,
    laplacian_strength: float = 0.35,
    laplacian_passes: int = 2,
) -> Tuple[List[AckermannState], Dict[str, float]]:
    """
    STOMP-style stochastic trajectory smoothing over steering sequences.
    Returns optimized path and an info dict with metrics.
    """
    start_time = time.time()
    info: Dict[str, float] = {}
    if path is None or len(path) < 2:
        info["reason"] = "path_too_short"
        return list(path) if path else [], info
    if step_size <= 0:
        info["reason"] = "invalid_step"
        return list(path), info

    checker = GridFootprintChecker(grid_map, footprint, theta_bins)
    clearance_map = _obstacle_distance_map(grid_map) if w_clear > 0 else None
    rng = np.random.default_rng(seed)
    noise_scale = noise_std if noise_std is not None else 0.5 * params.max_steer
    noise_scale = max(noise_scale, params.max_steer * 0.05)
    ds_check = ds_check if ds_check is not None else default_collision_step(grid_map.resolution)
    ds_check = max(ds_check, 1e-4)

    steers, directions, step_sizes = path_to_controls(path, params, step_size, allow_reverse=allow_reverse)
    if steers is None or directions is None or step_sizes is None:
        info["reason"] = "controls_unavailable"
        return list(path), info
    step_sizes = np.asarray(step_sizes, dtype=float)
    step_sizes = np.clip(step_sizes, 1e-6, None)
    step_size_eval = float(step_sizes.min()) if step_sizes.size else step_size
    ref_steers = steers.copy()
    target_pose = goal if goal is not None else path[-1]
    ref_path = list(path)

    base_path = controls_to_path(path[0], steers, directions, params, step_size_eval, step_sizes=step_sizes)
    orig_cost = compute_cost(
        base_path,
        steers,
        checker,
        ds_check,
        w_smooth,
        w_steer,
        w_clear,
        clearance_map,
        grid_map,
        step_size_eval,
        target_pose=target_pose,
        params=params,
        w_goal=w_goal,
        ref_steers=ref_steers,
        w_prior=w_prior,
        ref_path=ref_path,
        w_track=w_track,
    )
    best_path = base_path if math.isfinite(orig_cost) else None
    best_steers = steers if math.isfinite(orig_cost) else None
    best_cost = orig_cost if math.isfinite(orig_cost) else float("inf")
    latest_path = best_path
    latest_steers = best_steers
    latest_cost: Optional[float] = float(best_cost) if math.isfinite(best_cost) else None
    base_goal_err = _goal_error(base_path[-1], target_pose, params)
    ref_goal_err = _goal_error(path[-1], target_pose, params)
    total_rollouts = 0
    rejected = 0
    patience = 6
    no_improve = 0
    max_time = 1.5
    iters_run = 0

    for it in range(iters):
        iters_run = it + 1
        if (time.time() - start_time) > max_time:
            break
        candidates: List[np.ndarray] = []
        cand_costs: List[float] = []
        for _ in range(rollouts):
            noise = rng.normal(0.0, noise_scale, size=steers.shape)
            steers_roll = np.clip(steers + noise, -params.max_steer, params.max_steer)
            path_roll = controls_to_path(path[0], steers_roll, directions, params, step_size_eval, step_sizes=step_sizes)
            cost = compute_cost(
                path_roll,
                steers_roll,
                checker,
                ds_check,
                w_smooth,
                w_steer,
                w_clear,
                clearance_map,
                grid_map,
                step_size_eval,
                target_pose=target_pose,
                params=params,
                w_goal=w_goal,
                ref_steers=ref_steers,
                w_prior=w_prior,
                ref_path=ref_path,
                w_track=w_track,
            )
            total_rollouts += 1
            if not math.isfinite(cost):
                rejected += 1
                continue
            candidates.append(steers_roll)
            cand_costs.append(cost)
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_path = path_roll
                best_steers = steers_roll
        if not candidates:
            noise_scale *= 0.5
            if noise_scale < 1e-4:
                info["reason"] = "no_feasible_rollouts"
                break
            continue

        costs_arr = np.asarray(cand_costs)
        shifted = costs_arr - costs_arr.min()
        weights = np.exp(-shifted / max(lambda_, 1e-6))
        weights_sum = weights.sum()
        if weights_sum <= 1e-12:
            info["reason"] = "weight_underflow"
            break
        weights /= weights_sum
        steers_stack = np.stack(candidates, axis=0)
        steers = np.clip(np.average(steers_stack, axis=0, weights=weights), -params.max_steer, params.max_steer)

        updated_path = controls_to_path(path[0], steers, directions, params, step_size_eval, step_sizes=step_sizes)
        updated_cost = compute_cost(
            updated_path,
            steers,
            checker,
            ds_check,
            w_smooth,
            w_steer,
            w_clear,
            clearance_map,
            grid_map,
            step_size_eval,
            target_pose=target_pose,
            params=params,
            w_goal=w_goal,
            ref_steers=ref_steers,
            w_prior=w_prior,
            ref_path=ref_path,
            w_track=w_track,
        )
        if math.isfinite(updated_cost):
            latest_path = updated_path
            latest_steers = steers
            latest_cost = updated_cost
        if updated_cost < best_cost - 1e-9:
            best_cost = updated_cost
            best_path = updated_path
            best_steers = steers
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            info["reason"] = "no_improvement"
            break
        noise_scale *= 0.92

    if best_steers is not None and laplacian_strength > 0.0 and laplacian_passes > 0:
        smoothed_steers = _laplacian_smooth_1d(best_steers, strength=laplacian_strength, passes=laplacian_passes)
        smoothed_steers = np.clip(smoothed_steers, -params.max_steer, params.max_steer)
        smoothed_path = controls_to_path(
            path[0], smoothed_steers, directions, params, step_size_eval, step_sizes=step_sizes
        )
        smoothed_cost = compute_cost(
            smoothed_path,
            smoothed_steers,
            checker,
            ds_check,
            w_smooth,
            w_steer,
            w_clear,
            clearance_map,
            grid_map,
            step_size_eval,
            target_pose=target_pose,
            params=params,
            w_goal=w_goal,
            ref_steers=ref_steers,
            w_prior=w_prior,
            ref_path=ref_path,
            w_track=w_track,
        )
        if math.isfinite(smoothed_cost):
            latest_path = smoothed_path
            latest_steers = smoothed_steers
            latest_cost = smoothed_cost
            if smoothed_cost < best_cost - 1e-9:
                best_cost = smoothed_cost
                best_path = smoothed_path
                best_steers = smoothed_steers

    if target_pose is not None and best_path:
        seg_start = best_path[-2].as_tuple() if len(best_path) >= 2 else best_path[-1].as_tuple()
        seg_end = (target_pose.x, target_pose.y, target_pose.theta)
        if not checker.motion_collides(seg_start, seg_end, step=ds_check):
            snapped_path = list(best_path[:-1]) + [target_pose]
            snap_cost = compute_cost(
                snapped_path,
                best_steers if best_steers is not None else steers,
                checker,
                ds_check,
                w_smooth,
                w_steer,
                w_clear,
                clearance_map,
                grid_map,
                step_size_eval,
                target_pose=target_pose,
                params=params,
                w_goal=w_goal,
                ref_steers=ref_steers,
                w_prior=w_prior,
                ref_path=ref_path,
                w_track=w_track,
            )
            if math.isfinite(snap_cost):
                latest_path = snapped_path
                latest_steers = best_steers if best_steers is not None else steers
                latest_cost = snap_cost
                if snap_cost <= best_cost + 1e-6:
                    best_cost = snap_cost
                    best_path = snapped_path

    improved_best = bool(math.isfinite(orig_cost) and math.isfinite(best_cost) and (best_cost + 1e-6 < orig_cost))
    selected_path = best_path
    selected_steers = best_steers
    selected_cost = best_cost
    selected_source = "best"
    if not improved_best and latest_path is not None:
        selected_path = latest_path
        selected_steers = latest_steers if latest_steers is not None else selected_steers
        selected_cost = latest_cost if latest_cost is not None else selected_cost
        selected_source = "latest"
    improved = bool(
        math.isfinite(orig_cost) and selected_cost is not None and math.isfinite(selected_cost) and (selected_cost + 1e-6 < orig_cost)
    )
    info.update(
        {
            "orig_cost": float(orig_cost),
            "best_cost": float(best_cost),
            "selected_cost": float(selected_cost) if selected_cost is not None else float("inf"),
            "improved": improved,
            "selected_source": selected_source,
            "iters_run": iters_run,
            "rollouts": rollouts,
            "reject_rate": rejected / float(total_rollouts) if total_rollouts else 1.0,
            "time_sec": time.time() - start_time,
        }
    )
    if selected_path is None or selected_cost is None or not math.isfinite(selected_cost) or _trajectory_collides(selected_path, checker, ds_check):
        if selected_path is None:
            info.setdefault("reason", "no_feasible_rollouts")
        elif selected_cost is None or not math.isfinite(selected_cost):
            info.setdefault("reason", "selected_cost_invalid")
        else:
            info.setdefault("reason", "selected_path_in_collision")
        info["improved"] = False
        return list(path), info

    best_goal_err = _goal_error(selected_path[-1], target_pose, params)
    max_dev = 0.0
    if ref_path and selected_path:
        samples = min(64, len(ref_path), len(selected_path))
        ref_xy = _sample_path_xy(ref_path, samples)
        best_xy = _sample_path_xy(selected_path, samples)
        if ref_xy.size and best_xy.size:
            max_dev = float(np.linalg.norm(ref_xy - best_xy, axis=1).max())
    info["goal_error"] = best_goal_err
    info["goal_error_base"] = base_goal_err
    info["max_deviation"] = max_dev
    track_tol = max(params.min_turn_radius * 1.0, 2.0)
    allowed_goal = max(ref_goal_err + 0.25, base_goal_err + 0.45)
    info["track_tol"] = track_tol
    info["allowed_goal_error"] = allowed_goal
    if best_goal_err > allowed_goal and max_dev > track_tol:
        info["reason"] = "goal_or_track_drift"
        info["best_cost"] = float(orig_cost)
        info["selected_cost"] = float(orig_cost)
        info["improved"] = False
        return list(path), info
    return selected_path if selected_path else list(path), info


def _bernstein_coeffs(n: int) -> np.ndarray:
    return np.asarray([math.comb(n, i) for i in range(n + 1)], dtype=float)


def _bernstein_basis(n: int, tau: float, coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    tau = float(np.clip(tau, 0.0, 1.0))
    coeff = _bernstein_coeffs(n) if coeffs is None else coeffs
    i = np.arange(n + 1, dtype=float)
    return coeff * (tau**i) * ((1.0 - tau) ** (n - i))


def _derivative_matrix(n: int, order: int) -> np.ndarray:
    """Matrix that maps control points to `order`-th derivative control points."""
    mat = np.eye(n + 1)
    curr_n = n
    for _ in range(order):
        diff = np.eye(curr_n, curr_n + 1, k=1) - np.eye(curr_n, curr_n + 1, k=0)
        mat = curr_n * diff @ mat
        curr_n -= 1
    return mat


def _downsample_path(path: Sequence[AckermannState], max_nodes: int) -> List[AckermannState]:
    if not path:
        return []
    max_nodes = max(2, int(max_nodes))
    if len(path) <= max_nodes:
        return list(path)
    idxs = np.linspace(0, len(path) - 1, max_nodes, dtype=int)
    idxs = np.unique(idxs)
    return [path[i] for i in idxs]


def _sample_distance_and_grad(distance_map: np.ndarray, grid_map: GridMap, x: float, y: float) -> Tuple[float, np.ndarray]:
    gx = (x - grid_map.origin[0]) / grid_map.resolution
    gy = (y - grid_map.origin[1]) / grid_map.resolution
    h, w = distance_map.shape
    if gx < 0 or gy < 0 or gx > (w - 1) or gy > (h - 1):
        return 0.0, np.zeros(2, dtype=float)
    x0 = int(math.floor(gx))
    y0 = int(math.floor(gy))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = gx - x0
    wy = gy - y0
    d00 = distance_map[y0, x0]
    d10 = distance_map[y0, x1]
    d01 = distance_map[y1, x0]
    d11 = distance_map[y1, x1]
    d0 = d00 * (1 - wx) + d10 * wx
    d1 = d01 * (1 - wx) + d11 * wx
    dist = d0 * (1 - wy) + d1 * wy
    ddx_grid = (d10 - d00) * (1 - wy) + (d11 - d01) * wy
    ddy_grid = (d01 - d00) * (1 - wx) + (d11 - d10) * wx
    grad = np.array([ddx_grid, ddy_grid], dtype=float) / max(grid_map.resolution, 1e-9)
    return float(dist), grad


def _build_safety_rectangles(
    path: Sequence[AckermannState],
    grid_map: GridMap,
    clearance_map: np.ndarray,
    *,
    max_size: float = 3.5,
    min_half: float = 0.35,
    margin: float = 0.2,
) -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    last_rect: Optional[Tuple[float, float, float, float]] = None
    for st in path:
        dist = _sample_distance(clearance_map, grid_map, st.x, st.y)
        half = max(min_half, min(max_size * 0.5, dist - margin))
        rect = (st.x - half, st.x + half, st.y - half, st.y + half)
        if last_rect is not None:
            x_in = last_rect[0] <= st.x <= last_rect[1]
            y_in = last_rect[2] <= st.y <= last_rect[3]
            if x_in and y_in:
                continue
        rects.append(rect)
        last_rect = rect
    if not rects and path:
        p = path[0]
        rects.append((p.x - min_half, p.x + min_half, p.y - min_half, p.y + min_half))
    return rects


def _init_bezier_from_path(
    path: Sequence[AckermannState],
    degree: int,
    params: AckermannParams,
    *,
    nominal_speed: float = 0.9,
    min_duration: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    degree = max(3, int(degree))
    num_segments = max(0, len(path) - 1)
    ctrl = np.zeros((num_segments, degree + 1, 2), dtype=float)
    durations = np.zeros(num_segments, dtype=float)
    for i in range(num_segments):
        a = path[i]
        b = path[i + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        dist = max(1e-4, math.hypot(dx, dy))
        dir_seg = np.array([dx, dy], dtype=float) / dist
        dir_start = np.array([math.cos(a.theta), math.sin(a.theta)], dtype=float)
        if np.linalg.norm(dir_start) < 1e-6:
            dir_start = dir_seg
        dir_end = np.array([math.cos(b.theta), math.sin(b.theta)], dtype=float)
        if np.linalg.norm(dir_end) < 1e-6:
            dir_end = dir_seg

        span = dist / float(max(degree, 1))
        ctrl[i, 0] = (a.x, a.y)
        ctrl[i, -1] = (b.x, b.y)
        ctrl[i, 1] = ctrl[i, 0] + dir_start * span * 1.2
        ctrl[i, -2] = ctrl[i, -1] - dir_end * span * 1.2
        for k in range(2, degree - 1):
            t = k / degree
            ctrl[i, k] = ctrl[i, 0] + np.array([dx, dy]) * t
        durations[i] = max(min_duration, dist / max(nominal_speed, 0.2))
    return ctrl, durations


def _bezier_states_from_ctrl(
    ctrl_points: np.ndarray,
    durations: np.ndarray,
    *,
    step: float = 0.25,
    goal: Optional[AckermannState] = None,
) -> List[AckermannState]:
    degree = ctrl_points.shape[1] - 1
    coeffs = _bernstein_coeffs(degree)
    diff1 = _derivative_matrix(degree, 1)
    states: List[AckermannState] = []
    for seg_idx, ctrl in enumerate(ctrl_points):
        T = float(durations[seg_idx])
        samples = max(3, int(math.ceil(max(T, step) / max(step, 1e-6))))
        taus = np.linspace(0.0, 1.0, samples, endpoint=False)
        for j, tau in enumerate(taus):
            basis = _bernstein_basis(degree, float(tau), coeffs)
            pos = basis @ ctrl
            vel_tau = _bernstein_basis(degree - 1, float(tau), _bernstein_coeffs(degree - 1)) @ (diff1 @ ctrl)
            vel = vel_tau / max(T, 1e-6)
            theta = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel) > 1e-6 else (states[-1].theta if states else 0.0)
            if not states and j == 0 and seg_idx == 0:
                states.append(AckermannState(pos[0], pos[1], theta))
            else:
                states.append(AckermannState(pos[0], pos[1], theta))
    if goal is not None:
        states.append(goal)
    else:
        # Add final endpoint if not already present
        end_ctrl = ctrl_points[-1]
        end_basis = _bernstein_basis(degree, 1.0, coeffs)
        end_pos = end_basis @ end_ctrl
        end_vel = _bernstein_basis(degree - 1, 1.0, _bernstein_coeffs(degree - 1)) @ (diff1 @ end_ctrl)
        theta_end = math.atan2(end_vel[1], end_vel[0]) if np.linalg.norm(end_vel) > 1e-6 else (states[-1].theta if states else 0.0)
        states.append(AckermannState(end_pos[0], end_pos[1], theta_end))
    return states


def _feng_cost_and_grad(
    ctrl_points: np.ndarray,
    durations: np.ndarray,
    *,
    basis_cache: Dict[str, np.ndarray],
    diff_mats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    clearance_map: np.ndarray,
    grid_map: GridMap,
    rects: List[Tuple[float, float, float, float]],
    rect_indices: np.ndarray,
    w_jerk: float,
    w_terrain: float,
    w_safety: float,
    w_dyn: float,
    w_cont: float,
    ref_ctrl: Optional[np.ndarray],
    w_track: float,
    v_max: float,
    a_max: float,
    collision_margin: float,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    num_segments, degree_plus, _ = ctrl_points.shape
    degree = degree_plus - 1
    diff1, diff2, diff3 = diff_mats
    weights = basis_cache["weights"]
    basis0 = basis_cache["basis0"]
    vel_coeff = basis_cache["vel_coeff"]
    acc_coeff = basis_cache["acc_coeff"]
    jerk_coeff = basis_cache["jerk_coeff"]

    grad = np.zeros_like(ctrl_points)
    total_cost = 0.0
    samples = weights.shape[0]
    for seg_idx in range(num_segments):
        ctrl = ctrl_points[seg_idx]
        T = max(1e-3, float(durations[seg_idx]))
        sample_w = weights * T

        pos = basis0 @ ctrl  # (samples, 2)
        vel = (vel_coeff @ ctrl) / T
        acc = (acc_coeff @ ctrl) / (T**2)
        jerk = (jerk_coeff @ ctrl) / (T**3)

        # Jerk cost
        jerk_cost = np.sum(sample_w * np.sum(jerk**2, axis=1))
        total_cost += w_jerk * jerk_cost
        jerk_grad = (
            (sample_w[:, None, None] * (2.0 * jerk[:, None, :]) * (jerk_coeff[:, :, None] / (T**3))).sum(axis=0)
        )
        grad[seg_idx] += w_jerk * jerk_grad

        # Terrain clearance (proxy for traversability)
        for j in range(samples):
            dist, dist_grad = _sample_distance_and_grad(clearance_map, grid_map, float(pos[j, 0]), float(pos[j, 1]))
            dist = max(dist, 1e-3)
            terrain_pen = 1.0 / dist
            total_cost += w_terrain * sample_w[j] * terrain_pen
            grad_pos = (-w_terrain * sample_w[j] / (dist**2)) * dist_grad
            grad[seg_idx] += basis0[j][:, None] * grad_pos

        # Dynamic limits
        speed = np.linalg.norm(vel, axis=1)
        accel = np.linalg.norm(acc, axis=1)
        speed_excess = np.maximum(0.0, speed - v_max)
        acc_excess = np.maximum(0.0, accel - a_max)
        if np.any(speed_excess > 0):
            factor = 2.0 * w_dyn * sample_w * speed_excess / np.maximum(speed, 1e-6)
            grad_speed = (
                (factor[:, None, None] * (vel_coeff[:, :, None] / T) * vel[:, None, :])[speed_excess > 0].sum(axis=0)
            )
            grad[seg_idx] += grad_speed
            total_cost += w_dyn * np.sum(sample_w * (speed_excess**2))
        if np.any(acc_excess > 0):
            factor = 2.0 * w_dyn * sample_w * acc_excess / np.maximum(accel, 1e-6)
            grad_acc = (
                (factor[:, None, None] * (acc_coeff[:, :, None] / (T**2)) * acc[:, None, :])[acc_excess > 0].sum(axis=0)
            )
            grad[seg_idx] += grad_acc
            total_cost += w_dyn * np.sum(sample_w * (acc_excess**2))

        # Soft collision guard (push away from low-clearance areas)
        for j in range(samples):
            dist, dist_grad = _sample_distance_and_grad(clearance_map, grid_map, float(pos[j, 0]), float(pos[j, 1]))
            if dist < collision_margin:
                shortfall = collision_margin - dist
                total_cost += w_safety * (sample_w[j] * shortfall * shortfall)
                grad_pos = (-2.0 * w_safety * sample_w[j] * shortfall) * dist_grad
                grad[seg_idx] += basis0[j][:, None] * grad_pos

        # Safety rectangles on control points
        for cp_idx in range(1, degree):  # keep endpoints fixed
            rect_idx = rect_indices[seg_idx, cp_idx]
            rect = rects[min(rect_idx, len(rects) - 1)]
            px, py = ctrl[cp_idx]
            gx = 0.0
            gy = 0.0
            if px < rect[0]:
                gx = rect[0] - px
            elif px > rect[1]:
                gx = rect[1] - px
            if py < rect[2]:
                gy = rect[2] - py
            elif py > rect[3]:
                gy = rect[3] - py
            if gx != 0.0 or gy != 0.0:
                total_cost += w_safety * (gx * gx + gy * gy)
                grad[seg_idx, cp_idx, 0] -= 2.0 * w_safety * gx
                grad[seg_idx, cp_idx, 1] -= 2.0 * w_safety * gy

        if ref_ctrl is not None and w_track > 0.0:
            diff = ctrl - ref_ctrl[seg_idx]
            total_cost += w_track * float(np.sum(diff**2))
            grad[seg_idx] += 2.0 * w_track * diff

    # Continuity penalties on velocity/acceleration across segments
    if num_segments > 1:
        b1_start = _bernstein_basis(degree - 1, 0.0, _bernstein_coeffs(degree - 1))
        b1_end = _bernstein_basis(degree - 1, 1.0, _bernstein_coeffs(degree - 1))
        b2_start = _bernstein_basis(degree - 2, 0.0, _bernstein_coeffs(degree - 2))
        b2_end = _bernstein_basis(degree - 2, 1.0, _bernstein_coeffs(degree - 2))
        for seg_idx in range(num_segments - 1):
            T0 = max(1e-3, float(durations[seg_idx]))
            T1 = max(1e-3, float(durations[seg_idx + 1]))
            vel_end = (b1_end @ (diff1 @ ctrl_points[seg_idx])) / T0
            vel_next = (b1_start @ (diff1 @ ctrl_points[seg_idx + 1])) / T1
            acc_end = (b2_end @ (diff2 @ ctrl_points[seg_idx])) / (T0**2)
            acc_next = (b2_start @ (diff2 @ ctrl_points[seg_idx + 1])) / (T1**2)

            v_gap = vel_end - vel_next
            a_gap = acc_end - acc_next
            total_cost += w_cont * (np.dot(v_gap, v_gap) + 0.5 * np.dot(a_gap, a_gap))

            v_coeff_end = (b1_end @ diff1) / T0
            v_coeff_next = (b1_start @ diff1) / T1
            a_coeff_end = (b2_end @ diff2) / (T0**2)
            a_coeff_next = (b2_start @ diff2) / (T1**2)

            grad[seg_idx] += 2.0 * w_cont * (v_coeff_end[:, None] * v_gap + 0.5 * a_coeff_end[:, None] * a_gap)
            grad[seg_idx + 1] -= 2.0 * w_cont * (
                v_coeff_next[:, None] * v_gap + 0.5 * a_coeff_next[:, None] * a_gap
            )

    grad[:, 0, :] = 0.0
    grad[:, -1, :] = 0.0
    metrics = {"cost": float(total_cost)}
    return total_cost, grad, metrics


def feng_optimize_path(
    path: Sequence[AckermannState],
    grid_map: GridMap,
    footprint,
    params: AckermannParams,
    *,
    goal: Optional[AckermannState] = None,
    degree: int = 5,
    max_segments: int = 12,
    samples_per_seg: int = 12,
    iters: int = 60,
    lr: float = 0.12,
    w_jerk: float = 1.0,
    w_terrain: float = 1.6,
    w_safety: float = 35.0,
    w_dyn: float = 4.0,
    w_cont: float = 6.0,
    w_track: float = 0.2,
    v_max: Optional[float] = None,
    a_max: float = 1.5,
    collision_margin: Optional[float] = None,
    rect_size: float = 3.0,
    safety_margin: float = 0.25,
    ds_check: Optional[float] = None,
    seed: Optional[int] = None,
    output_step: float = 0.25,
) -> Tuple[List[AckermannState], Dict[str, float]]:
    """
    Constraint-aware trajectory optimizer following Feng et al. (2025).

    The path is converted into multi-segment quintic Bézier curves with safety rectangles
    around the sampled waypoints. The objective minimizes jerk and traversability cost
    (using clearance as a proxy) under velocity/acceleration limits and continuity
    penalties. Safety rectangles are enforced via soft penalties and projection.
    """
    t0 = time.time()
    info: Dict[str, float] = {}
    degree = max(3, int(degree))
    if path is None or len(path) < 2:
        info["reason"] = "path_too_short"
        return list(path) if path else [], info

    ds_check = ds_check if ds_check is not None else default_collision_step(grid_map.resolution)
    checker = GridFootprintChecker(grid_map, footprint, theta_bins=64)
    clearance_map = _obstacle_distance_map(grid_map)
    rng = np.random.default_rng(seed)

    coarse_path = _downsample_path(path, max_segments + 1)
    rects = _build_safety_rectangles(coarse_path, grid_map, clearance_map, max_size=rect_size, margin=safety_margin)
    ctrl_points, durations = _init_bezier_from_path(
        coarse_path, degree, params, nominal_speed=params.v_max, min_duration=0.6
    )
    ref_ctrl = ctrl_points.copy()
    if collision_margin is None:
        diag = math.hypot(getattr(footprint, "length", 1.0), getattr(footprint, "width", 1.0))
        collision_margin = max(0.35, 0.5 * diag + 0.05)
    input_path = list(path)

    # Precompute basis terms for cost/grad
    samples_per_seg = max(4, samples_per_seg)
    taus = np.linspace(0.0, 1.0, samples_per_seg)
    coeff_base = _bernstein_coeffs(degree)
    basis0 = np.stack([_bernstein_basis(degree, float(t), coeff_base) for t in taus], axis=0)
    diff1 = _derivative_matrix(degree, 1)
    diff2 = _derivative_matrix(degree, 2)
    diff3 = _derivative_matrix(degree, 3)
    basis1 = np.stack([_bernstein_basis(degree - 1, float(t), _bernstein_coeffs(degree - 1)) for t in taus], axis=0)
    basis2 = np.stack([_bernstein_basis(degree - 2, float(t), _bernstein_coeffs(degree - 2)) for t in taus], axis=0)
    basis3 = np.stack([_bernstein_basis(degree - 3, float(t), _bernstein_coeffs(degree - 3)) for t in taus], axis=0)
    basis_cache = {
        "weights": np.ones_like(taus, dtype=float) / float(len(taus)),
        "basis0": basis0,
        "vel_coeff": basis1 @ diff1,
        "acc_coeff": basis2 @ diff2,
        "jerk_coeff": basis3 @ diff3,
    }
    diff_mats = (diff1, diff2, diff3)

    rect_indices = np.zeros((ctrl_points.shape[0], ctrl_points.shape[1]), dtype=int)
    for seg_idx in range(rect_indices.shape[0]):
        rect_indices[seg_idx, :] = min(seg_idx, len(rects) - 1)
        rect_indices[seg_idx, -2:] = min(seg_idx + 1, len(rects) - 1)

    v_max_val = float(v_max) if v_max is not None else float(params.v_max)
    base_cost, _, _ = _feng_cost_and_grad(
        ctrl_points,
        durations,
        basis_cache=basis_cache,
        diff_mats=diff_mats,
        clearance_map=clearance_map,
        grid_map=grid_map,
        rects=rects,
        rect_indices=rect_indices,
        w_jerk=w_jerk,
        w_terrain=w_terrain,
        w_safety=w_safety,
        w_dyn=w_dyn,
        w_cont=w_cont,
        ref_ctrl=ref_ctrl,
        w_track=w_track,
        v_max=v_max_val,
        a_max=a_max,
        collision_margin=collision_margin,
    )
    base_candidate = _bezier_states_from_ctrl(ctrl_points, durations, step=output_step, goal=goal)
    base_collision = _trajectory_collides(base_candidate, checker, ds_check)

    best_ctrl = ctrl_points.copy()
    best_cost = base_cost
    best_path: Optional[List[AckermannState]] = None if base_collision else base_candidate
    best_info: Dict[str, float] = {} if base_collision else {"cost": float(base_cost)}
    iters_run = 0

    for it in range(max(1, iters)):
        iters_run = it + 1
        cost, grad, metrics = _feng_cost_and_grad(
            ctrl_points,
            durations,
            basis_cache=basis_cache,
            diff_mats=diff_mats,
            clearance_map=clearance_map,
            grid_map=grid_map,
            rects=rects,
            rect_indices=rect_indices,
            w_jerk=w_jerk,
            w_terrain=w_terrain,
            w_safety=w_safety,
            w_dyn=w_dyn,
            w_cont=w_cont,
            ref_ctrl=ref_ctrl,
            w_track=w_track,
            v_max=v_max_val,
            a_max=a_max,
            collision_margin=collision_margin,
        )
        g_clip = np.clip(grad, -5.0, 5.0)
        step_scale = lr * (0.92 ** (it / 5.0))
        ctrl_points = ctrl_points - step_scale * g_clip

        # Project interior control points back into their rectangles
        for seg_idx in range(ctrl_points.shape[0]):
            for cp_idx in range(1, ctrl_points.shape[1] - 1):
                rect = rects[min(rect_indices[seg_idx, cp_idx], len(rects) - 1)]
                ctrl_points[seg_idx, cp_idx, 0] = clamp(ctrl_points[seg_idx, cp_idx, 0], rect[0], rect[1])
                ctrl_points[seg_idx, cp_idx, 1] = clamp(ctrl_points[seg_idx, cp_idx, 1], rect[2], rect[3])

        # Small random shake to escape flat basins
        if it and it % 12 == 0:
            noise = rng.normal(0.0, 0.02, size=ctrl_points.shape)
            noise[:, 0, :] = 0.0
            noise[:, -1, :] = 0.0
            ctrl_points += noise

        candidate_path = _bezier_states_from_ctrl(ctrl_points, durations, step=output_step, goal=goal)
        collision = _trajectory_collides(candidate_path, checker, ds_check)
        if not collision and (best_path is None or cost < best_cost - 1e-6):
            best_cost = cost
            best_ctrl = ctrl_points.copy()
            best_path = candidate_path
            best_info = metrics
        if cost < base_cost * 0.3:
            break

    if best_path is None:
        best_path = input_path
        best_info["reason"] = "fallback_to_input"
        best_cost = base_cost
    else:
        best_info.setdefault("cost", best_cost)

    best_info.update(
        {
            "base_cost": float(base_cost),
            "best_cost": float(best_cost),
            "improved": bool(best_cost + 1e-6 < base_cost),
            "iters_run": iters_run,
            "time_sec": time.time() - t0,
        }
    )
    return best_path, best_info
