import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..common import default_collision_step, euclidean, heading_diff, wrap_angle
from ..geometry import GridFootprintChecker
from ..robot import AckermannParams, AckermannState, propagate


@dataclass(frozen=True)
class CubicBezier:
    """Cubic Bezier curve in R^2."""

    p0: Tuple[float, float]
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    p3: Tuple[float, float]

    def point(self, t: float) -> Tuple[float, float]:
        t = float(t)
        u = 1.0 - t
        b0 = u * u * u
        b1 = 3.0 * u * u * t
        b2 = 3.0 * u * t * t
        b3 = t * t * t
        x = b0 * self.p0[0] + b1 * self.p1[0] + b2 * self.p2[0] + b3 * self.p3[0]
        y = b0 * self.p0[1] + b1 * self.p1[1] + b2 * self.p2[1] + b3 * self.p3[1]
        return x, y

    def deriv(self, t: float) -> Tuple[float, float]:
        t = float(t)
        u = 1.0 - t
        dx = (
            3.0 * u * u * (self.p1[0] - self.p0[0])
            + 6.0 * u * t * (self.p2[0] - self.p1[0])
            + 3.0 * t * t * (self.p3[0] - self.p2[0])
        )
        dy = (
            3.0 * u * u * (self.p1[1] - self.p0[1])
            + 6.0 * u * t * (self.p2[1] - self.p1[1])
            + 3.0 * t * t * (self.p3[1] - self.p2[1])
        )
        return dx, dy

    def heading(self, t: float, fallback: Optional[float] = None) -> float:
        dx, dy = self.deriv(t)
        if dx * dx + dy * dy <= 1e-18:
            return 0.0 if fallback is None else fallback
        return math.atan2(dy, dx)

    def control_polygon_length(self) -> float:
        return euclidean(self.p0, self.p1) + euclidean(self.p1, self.p2) + euclidean(self.p2, self.p3)


@dataclass(frozen=True)
class SplineEdge:
    """
    Two-segment motion primitive using concatenated cubic Bezier curves.

    The Yoon et al. (2017) SS-RRT* uses two concatenated cubic Bezier curves as a motion primitive.
    """

    seg1: CubicBezier
    seg2: CubicBezier


@dataclass
class TreeNode:
    state: AckermannState
    parent: int
    cost: float
    # Edge from parent -> this node. Root has edge=None.
    edge: Optional[SplineEdge] = None


class RRTStarPlanner:
    """
    Spline-based RRT* (SS-RRT*) planner for car-like vehicles.

    Paper: Yoon et al., "Spline-based RRT* Using Piecewise Continuous
    Collision-checking Algorithm for Car-like Vehicles", J Intell Robot Syst (2017).

    Repository adaptation:
    - The paper’s ObstacleFree() uses dominant trajectories + transition rectangles.
      In this repo we perform collision checks by sampling poses along the spline and
      querying `GridFootprintChecker` (occupancy grid + footprint).
    """

    def __init__(
        self,
        grid_map,
        footprint,
        params: AckermannParams,
        step_time: float = 0.6,
        velocity: float = 0.8,
        neighbor_radius: float = 1.5,
        goal_sample_rate: float = 0.2,
        goal_xy_tol: float = 0.1,
        goal_theta_tol: float = math.radians(5.0),
        goal_check_freq: int = 1,
        seed_steps: int = 0,
        collision_step: Optional[float] = None,
        rewire: bool = True,
        heading_penalty_weight: Optional[float] = None,
        steer_jitter: float = 0.25,
        rng_seed: Optional[int] = None,
        theta_bins: int = 72,
        collision_padding: Optional[float] = None,
        # Spline parameters (kept for compatibility with the paper nomenclature)
        gamma: float = 0.35,
        max_neighbors: int = 10,
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.step_time = float(step_time)
        self.velocity = float(velocity)
        self.neighbor_radius = float(neighbor_radius)
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_xy_tol = float(goal_xy_tol)
        self.goal_theta_tol = float(goal_theta_tol)
        self.goal_check_freq = max(1, int(goal_check_freq))
        self.seed_steps = max(0, int(seed_steps))
        self.collision_step = (
            float(collision_step) if collision_step is not None else default_collision_step(grid_map.resolution)
        )
        self.rewire = bool(rewire)
        self.steer_jitter = max(0.0, float(steer_jitter))

        baseline_turn_radius = 1.6
        heading_penalty_scale = max(1.0, baseline_turn_radius / max(self.params.min_turn_radius, 1e-6))
        default_heading_weight = 0.5 * self.params.min_turn_radius * heading_penalty_scale
        self.heading_penalty_weight = (
            float(heading_penalty_weight) if heading_penalty_weight is not None else default_heading_weight
        )

        self.gamma = max(0.0, min(1.0, float(gamma)))
        self.max_neighbors = max(1, int(max_neighbors))
        self.rng = np.random.default_rng(rng_seed)
        self.collision_checker = GridFootprintChecker(
            grid_map, footprint, theta_bins=theta_bins, padding=collision_padding
        )

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        max_iter: int = 3000,
        timeout: float = 5.0,
        self_check: bool = True,
        try_direct_connect: bool = True,
    ):
        start_time = time.time()
        remediations: List[str] = []

        if self._pose_collides(start.x, start.y, start.theta):
            return [], {
                "nodes": 1,
                "iterations": 0,
                "time": time.time() - start_time,
                "success": False,
                "failure_reason": "start_in_collision",
                "remediations": remediations,
            }
        if self._pose_collides(goal.x, goal.y, goal.theta):
            return [], {
                "nodes": 1,
                "iterations": 0,
                "time": time.time() - start_time,
                "success": False,
                "failure_reason": "goal_in_collision",
                "remediations": remediations,
            }

        if try_direct_connect:
            direct_edge = self._get_bezier_curve(start, goal)
            if direct_edge is not None:
                poses, _ = self._edge_poses_and_length(direct_edge, step=self.collision_step)
                if poses and not self.collision_checker.collides_path(poses):
                    return poses, {
                        "nodes": 2,
                        "iterations": 0,
                        "time": time.time() - start_time,
                        "success": True,
                        "failure_reason": None,
                        "remediations": remediations + ["direct_connect"],
                    }

        ref_path = self._grid_bfs_path(start, goal)

        tree: List[TreeNode] = [TreeNode(start, parent=-1, cost=0.0, edge=None)]
        children: List[set[int]] = [set()]
        best_goal_idx: Optional[int] = None
        iterations = 0

        if self.seed_steps > 0:
            seeded = self._seed_toward_goal(tree, children, goal, self.seed_steps)
            if seeded is not None:
                path = self._reconstruct(tree, seeded)
                return path, {
                    "nodes": len(tree),
                    "iterations": iterations,
                    "time": time.time() - start_time,
                    "success": True,
                    "failure_reason": None,
                    "remediations": remediations + ["seed_steps"],
                }

        for it in range(int(max_iter)):
            if (time.time() - start_time) > timeout:
                break
            iterations = it + 1

            x_rand = self._sample_state(goal)
            goal_sample = x_rand is goal
            if ref_path and not goal_sample and self.rng.random() < 0.45:
                gx, gy = ref_path[self.rng.integers(len(ref_path))]
                sx, sy = self.map.grid_to_world(int(gx), int(gy))
                heading = math.atan2(goal.y - sy, goal.x - sx)
                x_rand = AckermannState(sx, sy, heading)
            nearest_idx = self._nearest(tree, x_rand)
            x_nearest = tree[nearest_idx].state

            step_length = max(1e-6, self.velocity * self.step_time)
            if goal_sample:
                # Opportunistic RRT-Connect-style acceleration on goal samples.
                # This keeps the planner usable in small iteration budgets (demo settings),
                # while still using spline edges for feasibility/collision checks.
                current_idx = nearest_idx
                for _ in range(12):
                    current_state = tree[current_idx].state
                    steer_res = self._steer_motion(current_state, goal, step_length)
                    if steer_res is None:
                        break
                    x_step, steering, arc_len = steer_res
                    gx, gy = self.map.world_to_grid(x_step.x, x_step.y)
                    if not self.map.in_bounds(gx, gy):
                        break
                    edge = self._constant_curvature_edge(current_state, x_step, steering=steering, arc_len=arc_len)
                    poses, edge_len = self._edge_poses_and_length(edge, step=self.collision_step)
                    if self.collision_checker.collides_path(poses):
                        break
                    new_cost = tree[current_idx].cost + edge_len
                    tree.append(TreeNode(x_step, parent=current_idx, cost=new_cost, edge=edge))
                    children.append(set())
                    children[current_idx].add(len(tree) - 1)
                    current_idx = len(tree) - 1
                    if self._goal_reached(x_step, goal):
                        best_goal_idx = current_idx
                        break
                if best_goal_idx is not None:
                    break
                continue

            steer_res = self._steer_motion(x_nearest, x_rand, step_length)
            if steer_res is None:
                continue
            x_new, nearest_steering, nearest_arc_len = steer_res
            gx, gy = self.map.world_to_grid(x_new.x, x_new.y)
            if not self.map.in_bounds(gx, gy):
                continue

            edge_from_nearest = self._constant_curvature_edge(
                x_nearest, x_new, steering=nearest_steering, arc_len=nearest_arc_len
            )

            neighbor_indices = self._neighbors(tree, x_new)
            if nearest_idx not in neighbor_indices:
                neighbor_indices.append(nearest_idx)
            neighbor_indices.sort(key=lambda idx: euclidean((tree[idx].state.x, tree[idx].state.y), (x_new.x, x_new.y)))
            neighbor_indices = neighbor_indices[: self.max_neighbors]
            if nearest_idx not in neighbor_indices:
                neighbor_indices[-1] = nearest_idx

            # Algorithm 6 new wiring: choose xmin with min Cost(xnear)+CurveLength(B).
            best_parent: Optional[int] = None
            best_cost = float("inf")
            best_edge: Optional[SplineEdge] = None

            for cand_idx in neighbor_indices:
                if cand_idx == nearest_idx:
                    edge = edge_from_nearest
                else:
                    edge = self._get_bezier_curve(tree[cand_idx].state, x_new)
                    if edge is None:
                        continue
                poses, edge_len = self._edge_poses_and_length(edge, step=self.collision_step)
                if self.collision_checker.collides_path(poses):
                    continue
                cand_cost = tree[cand_idx].cost + edge_len
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_parent = cand_idx
                    best_edge = edge

            if best_parent is None or best_edge is None:
                continue

            tree.append(TreeNode(x_new, parent=best_parent, cost=best_cost, edge=best_edge))
            children.append(set())
            new_idx = len(tree) - 1
            children[best_parent].add(new_idx)

            # Algorithm 6 rewiring.
            if self.rewire:
                for near_idx in neighbor_indices:
                    if near_idx == best_parent or near_idx == new_idx:
                        continue
                    edge = self._get_bezier_curve(tree[new_idx].state, tree[near_idx].state)
                    if edge is None:
                        continue
                    poses, edge_len = self._edge_poses_and_length(edge, step=self.collision_step)
                    if self.collision_checker.collides_path(poses):
                        continue
                    proposed_cost = tree[new_idx].cost + edge_len
                    if proposed_cost + 1e-9 < tree[near_idx].cost:
                        self._rewire(
                            tree,
                            children,
                            near_idx=near_idx,
                            new_parent=new_idx,
                            new_edge=edge,
                            new_cost=proposed_cost,
                        )

            # Goal check (try to connect to exact goal when feasible).
            if self._goal_reached(tree[new_idx].state, goal):
                goal_edge = self._get_bezier_curve(tree[new_idx].state, goal)
                if goal_edge is not None:
                    poses, edge_len = self._edge_poses_and_length(goal_edge, step=self.collision_step)
                    if not self.collision_checker.collides_path(poses):
                        goal_cost = tree[new_idx].cost + edge_len
                        tree.append(TreeNode(goal, parent=new_idx, cost=goal_cost, edge=goal_edge))
                        children.append(set())
                        children[new_idx].add(len(tree) - 1)
                        best_goal_idx = len(tree) - 1
                        break
                best_goal_idx = new_idx
                break

            if it % self.goal_check_freq == 0:
                greedy_idx = self._nearest(tree, goal)
                if self._goal_reached(tree[greedy_idx].state, goal):
                    goal_edge = self._get_bezier_curve(tree[greedy_idx].state, goal)
                    if goal_edge is not None:
                        poses, edge_len = self._edge_poses_and_length(goal_edge, step=self.collision_step)
                        if not self.collision_checker.collides_path(poses):
                            goal_cost = tree[greedy_idx].cost + edge_len
                            tree.append(TreeNode(goal, parent=greedy_idx, cost=goal_cost, edge=goal_edge))
                            children.append(set())
                            children[greedy_idx].add(len(tree) - 1)
                            best_goal_idx = len(tree) - 1
                            break
                    best_goal_idx = greedy_idx
                    break

        if best_goal_idx is None:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                failure_reason = "timeout"
            elif iterations >= max_iter:
                failure_reason = "iteration_budget_exhausted"
            else:
                failure_reason = "search_failed"
            return [], {
                "nodes": len(tree),
                "iterations": iterations,
                "time": elapsed,
                "success": False,
                "failure_reason": failure_reason,
                "remediations": remediations,
            }

        path = self._reconstruct(tree, best_goal_idx)
        return path, {
            "nodes": len(tree),
            "iterations": iterations,
            "time": time.time() - start_time,
            "success": True,
            "failure_reason": None,
            "remediations": remediations,
        }

    def _sample_state(self, goal: AckermannState) -> AckermannState:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        x, y, theta = self.map.random_free_state(self.rng)
        return AckermannState(x, y, theta)

    def _nearest(self, tree: List[TreeNode], target: AckermannState) -> int:
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(tree):
            dist = euclidean((node.state.x, node.state.y), (target.x, target.y))
            dist += self.heading_penalty_weight * abs(heading_diff(target.theta, node.state.theta))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _neighbors(self, tree: List[TreeNode], state: AckermannState) -> List[int]:
        res: List[int] = []
        for i, node in enumerate(tree):
            d = euclidean((node.state.x, node.state.y), (state.x, state.y))
            if d <= self.neighbor_radius:
                res.append(i)
        return res

    def _goal_reached(self, state: AckermannState, goal: AckermannState) -> bool:
        dist = math.hypot(goal.x - state.x, goal.y - state.y)
        dtheta = abs(heading_diff(goal.theta, state.theta))
        return dist <= self.goal_xy_tol and dtheta <= self.goal_theta_tol

    def _pose_collides(self, x: float, y: float, theta: float) -> bool:
        return self.collision_checker.collides_pose(x, y, theta)

    def _grid_bfs_path(self, start: AckermannState, goal: AckermannState) -> Optional[List[Tuple[int, int]]]:
        """
        4-neighborhood BFS on the raw occupancy grid (ignores footprint/orientation).

        Used only as a sampling heuristic to bias exploration in cluttered scenes.
        """
        sx, sy = self.map.world_to_grid(start.x, start.y)
        gx, gy = self.map.world_to_grid(goal.x, goal.y)
        if self.map.is_occupied_index(sx, sy) or self.map.is_occupied_index(gx, gy):
            return None

        h, w = self.map.data.shape
        visited = np.full((h, w), -1, dtype=int)
        prev = np.full((h, w, 2), -1, dtype=int)
        q = deque()
        q.append((sx, sy))
        visited[sy, sx] = 0

        while q:
            cx, cy = q.popleft()
            if cx == gx and cy == gy:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = cx + dx
                ny = cy + dy
                if not self.map.in_bounds(nx, ny):
                    continue
                if visited[ny, nx] != -1:
                    continue
                if self.map.is_occupied_index(nx, ny):
                    continue
                visited[ny, nx] = visited[cy, cx] + 1
                prev[ny, nx, 0] = cx
                prev[ny, nx, 1] = cy
                q.append((nx, ny))

        if visited[gy, gx] == -1:
            return None
        path: List[Tuple[int, int]] = []
        cx, cy = gx, gy
        while cx >= 0 and cy >= 0:
            path.append((int(cx), int(cy)))
            if cx == sx and cy == sy:
                break
            pcx, pcy = prev[cy, cx, 0], prev[cy, cx, 1]
            cx, cy = int(pcx), int(pcy)
        path.reverse()
        return path

    def _steer(self, source: AckermannState, target: AckermannState, step_length: float) -> Optional[AckermannState]:
        res = self._steer_motion(source, target, step_length)
        if res is None:
            return None
        return res[0]

    def _steer_motion(
        self, source: AckermannState, target: AckermannState, step_length: float
    ) -> Optional[Tuple[AckermannState, float, float]]:
        """
        Steer(x_nearest, x_rand) -> (x_new, steering, arc_length).

        Uses a single constant-curvature bicycle-model primitive (forward) so x_new is
        kinematically reachable under the Ackermann min-turn-radius constraint.
        """
        dx = target.x - source.x
        dy = target.y - source.y
        dist_to_target = math.hypot(dx, dy)
        if dist_to_target <= 1e-9:
            return None

        motion_len = max(1e-3, min(float(step_length), dist_to_target))
        chord_heading = math.atan2(dy, dx)
        target_heading = target.theta if math.isfinite(target.theta) else chord_heading
        horizon = max(1e-6, 2.0 * float(step_length))
        heading_blend = self.gamma * max(0.0, min(1.0, (horizon - dist_to_target) / horizon))
        desired_heading = wrap_angle(chord_heading + heading_blend * heading_diff(target_heading, chord_heading))
        heading_error = heading_diff(desired_heading, source.theta)

        # Pick curvature to reduce heading error over this step: dtheta ~= k * s.
        desired_curvature = heading_error / max(motion_len, 1e-6)
        base_steer = math.atan(desired_curvature * self.params.wheelbase)
        base_steer = max(-self.params.max_steer, min(self.params.max_steer, base_steer))

        best_state: Optional[AckermannState] = None
        best_steer: Optional[float] = None
        best_metric = float("inf")
        for delta in (0.0, self.steer_jitter, -self.steer_jitter, 2.0 * self.steer_jitter, -2.0 * self.steer_jitter):
            steer = max(-self.params.max_steer, min(self.params.max_steer, base_steer + delta))
            nxt = propagate(source, steering=steer, direction=+1, step_length=motion_len, params=self.params)
            metric = euclidean((nxt.x, nxt.y), (target.x, target.y)) + self.heading_penalty_weight * abs(
                heading_diff(desired_heading, nxt.theta)
            )
            if metric < best_metric:
                best_metric = metric
                best_state = nxt
                best_steer = steer
        if best_state is None or best_steer is None:
            return None
        return best_state, best_steer, motion_len

    def _constant_curvature_edge(
        self,
        start: AckermannState,
        end: AckermannState,
        steering: float,
        arc_len: float,
    ) -> SplineEdge:
        """
        Build a two-segment cubic Bezier approximation of a constant-curvature arc.
        """
        if abs(arc_len) <= 1e-9:
            return self._line_edge((start.x, start.y), (end.x, end.y))

        k = math.tan(steering) / max(self.params.wheelbase, 1e-9)
        if abs(k) <= 1e-9:
            return self._line_edge((start.x, start.y), (end.x, end.y))

        r = 1.0 / k
        radius = abs(r)
        sweep = k * arc_len  # signed

        # Center is offset by radius along the left normal.
        cx = start.x + (-math.sin(start.theta)) * r
        cy = start.y + (math.cos(start.theta)) * r

        v0x = start.x - cx
        v0y = start.y - cy
        half = sweep * 0.5
        cos_h = math.cos(half)
        sin_h = math.sin(half)
        vmx = cos_h * v0x - sin_h * v0y
        vmy = sin_h * v0x + cos_h * v0y
        mid = (cx + vmx, cy + vmy)

        seg1 = self._arc_to_cubic_bezier((start.x, start.y), mid, (cx, cy), radius, half)
        seg2 = self._arc_to_cubic_bezier(mid, (end.x, end.y), (cx, cy), radius, half)
        return SplineEdge(seg1=seg1, seg2=seg2)

    def _line_edge(self, start_xy: Tuple[float, float], end_xy: Tuple[float, float]) -> SplineEdge:
        mx = (start_xy[0] + end_xy[0]) * 0.5
        my = (start_xy[1] + end_xy[1]) * 0.5
        mid = (mx, my)
        return SplineEdge(
            seg1=self._arc_to_cubic_bezier(start_xy, mid, (0.0, 0.0), radius=0.0, sweep=0.0),
            seg2=self._arc_to_cubic_bezier(mid, end_xy, (0.0, 0.0), radius=0.0, sweep=0.0),
        )

    def _seed_toward_goal(
        self, tree: List[TreeNode], children: List[set[int]], goal: AckermannState, steps: int
    ) -> Optional[int]:
        parent_idx = 0
        for _ in range(int(steps)):
            parent_state = tree[parent_idx].state
            step_length = max(1e-6, self.velocity * self.step_time)
            steer_res = self._steer_motion(parent_state, goal, step_length=step_length)
            if steer_res is None:
                break
            x_new, steering, arc_len = steer_res
            gx, gy = self.map.world_to_grid(x_new.x, x_new.y)
            if not self.map.in_bounds(gx, gy):
                break
            edge = self._constant_curvature_edge(parent_state, x_new, steering=steering, arc_len=arc_len)
            poses, edge_len = self._edge_poses_and_length(edge, step=self.collision_step)
            if self.collision_checker.collides_path(poses):
                break
            new_cost = tree[parent_idx].cost + edge_len
            tree.append(TreeNode(x_new, parent=parent_idx, cost=new_cost, edge=edge))
            children.append(set())
            children[parent_idx].add(len(tree) - 1)
            parent_idx = len(tree) - 1
            if self._goal_reached(x_new, goal):
                return parent_idx
        return None

    def _get_bezier_curve(self, start: AckermannState, end: AckermannState) -> Optional[SplineEdge]:
        """
        Build a two-segment motion primitive (biarc -> two cubic Beziers).

        The biarc construction ensures G1 continuity between the two segments. We enforce the
        Ackermann turning-radius constraint by rejecting biarcs whose arc radii fall below
        `params.min_turn_radius`.
        """
        eps = 1e-9
        p1x, p1y = start.x, start.y
        p2x, p2y = end.x, end.y
        t1x, t1y = math.cos(start.theta), math.sin(start.theta)
        t2x, t2y = math.cos(end.theta), math.sin(end.theta)

        vx = p2x - p1x
        vy = p2y - p1y
        v_dot_v = vx * vx + vy * vy
        if v_dot_v <= eps:
            return None

        tx = t1x + t2x
        ty = t1y + t2y
        v_dot_t = vx * tx + vy * ty
        t1_dot_t2 = t1x * t2x + t1y * t2y
        denom = 2.0 * (1.0 - t1_dot_t2)

        if abs(denom) <= 1e-6:
            v_dot_t2 = vx * t2x + vy * t2y
            if abs(v_dot_t2) <= 1e-6:
                return None
            d = v_dot_v / (4.0 * v_dot_t2)
        else:
            discriminant = v_dot_t * v_dot_t + denom * v_dot_v
            if discriminant < 0.0:
                return None
            d = (-v_dot_t + math.sqrt(discriminant)) / denom

        pmx = 0.5 * (p1x + p2x + d * (t1x - t2x))
        pmy = 0.5 * (p1y + p2y + d * (t1y - t2y))

        p1_to_pm = (pmx - p1x, pmy - p1y)
        p2_to_pm = (pmx - p2x, pmy - p2y)

        c1, r1, a1 = self._biarc_compute_arc((p1x, p1y), (t1x, t1y), p1_to_pm)
        c2, r2, a2 = self._biarc_compute_arc((p2x, p2y), (t2x, t2y), p2_to_pm)

        if d < 0.0:
            two_pi = 2.0 * math.pi
            a1 = (two_pi - abs(a1)) * (1.0 if a1 >= 0.0 else -1.0)
            a2 = (two_pi - abs(a2)) * (1.0 if a2 >= 0.0 else -1.0)

        r_min = float(self.params.min_turn_radius)
        if (r1 > 0.0 and r1 + eps < r_min) or (r2 > 0.0 and r2 + eps < r_min):
            return None

        seg1 = self._arc_to_cubic_bezier((p1x, p1y), (pmx, pmy), c1, r1, a1)
        seg2 = self._arc_to_cubic_bezier((pmx, pmy), (p2x, p2y), c2, r2, -a2)
        return SplineEdge(seg1=seg1, seg2=seg2)

    def _biarc_compute_arc(
        self,
        point: Tuple[float, float],
        tangent: Tuple[float, float],
        point_to_mid: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], float, float]:
        """
        Compute a circle arc (or line) from `point` toward `point+point_to_mid` with start tangent.

        Returns (center, radius, angle). radius==0 denotes a straight-line segment.
        """
        px, py = point
        tx, ty = tangent
        dx, dy = point_to_mid
        eps = 1e-9

        normal_z = dx * ty - dy * tx
        perp_x = ty * normal_z
        perp_y = -tx * normal_z

        denom = 2.0 * (perp_x * dx + perp_y * dy)
        if abs(denom) <= 1e-6:
            center = (px + 0.5 * dx, py + 0.5 * dy)
            return center, 0.0, 0.0

        center_dist = (dx * dx + dy * dy) / denom
        cx = px + perp_x * center_dist
        cy = py + perp_y * center_dist

        perp_mag = math.hypot(perp_x, perp_y)
        radius = abs(center_dist * perp_mag)
        if radius <= eps:
            return (cx, cy), 0.0, 0.0

        inv_r = 1.0 / radius
        center_to_end = (px - cx, py - cy)
        center_to_mid = (center_to_end[0] + dx, center_to_end[1] + dy)
        end_dir = (center_to_end[0] * inv_r, center_to_end[1] * inv_r)
        mid_dir = (center_to_mid[0] * inv_r, center_to_mid[1] * inv_r)

        dot = max(-1.0, min(1.0, end_dir[0] * mid_dir[0] + end_dir[1] * mid_dir[1]))
        twist = perp_x * dx + perp_y * dy
        angle = math.acos(dot) * (1.0 if twist >= 0.0 else -1.0)
        return (cx, cy), radius, angle

    def _arc_to_cubic_bezier(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        center_xy: Tuple[float, float],
        radius: float,
        sweep: float,
    ) -> CubicBezier:
        """
        Convert a circular arc (or line when radius==0) into a cubic Bezier approximation.
        """
        if radius <= 1e-9 or abs(sweep) <= 1e-9:
            x0, y0 = start_xy
            x3, y3 = end_xy
            p1 = (x0 + (x3 - x0) / 3.0, y0 + (y3 - y0) / 3.0)
            p2 = (x0 + 2.0 * (x3 - x0) / 3.0, y0 + 2.0 * (y3 - y0) / 3.0)
            return CubicBezier(p0=start_xy, p1=p1, p2=p2, p3=end_xy)

        cx, cy = center_xy
        r0x, r0y = start_xy[0] - cx, start_xy[1] - cy
        r1x, r1y = end_xy[0] - cx, end_xy[1] - cy

        if sweep >= 0.0:
            t0 = (-r0y / radius, r0x / radius)
            t1 = (-r1y / radius, r1x / radius)
        else:
            t0 = (r0y / radius, -r0x / radius)
            t1 = (r1y / radius, -r1x / radius)

        k = (4.0 / 3.0) * math.tan(abs(sweep) / 4.0) * radius
        p1 = (start_xy[0] + t0[0] * k, start_xy[1] + t0[1] * k)
        p2 = (end_xy[0] - t1[0] * k, end_xy[1] - t1[1] * k)
        return CubicBezier(p0=start_xy, p1=p1, p2=p2, p3=end_xy)

    def _edge_poses_and_length(self, edge: SplineEdge, step: float) -> Tuple[List[AckermannState], float]:
        step = max(1e-3, float(step))
        poses: List[AckermannState] = []
        length = 0.0

        prev_xy: Optional[Tuple[float, float]] = None
        prev_heading: Optional[float] = None

        for seg_idx, seg in enumerate((edge.seg1, edge.seg2)):
            approx = max(1e-6, seg.control_polygon_length())
            n = max(2, int(math.ceil(approx / step)))
            for i in range(n + 1):
                if seg_idx > 0 and i == 0:
                    continue
                t = i / n
                x, y = seg.point(t)
                heading = seg.heading(t, fallback=prev_heading)
                poses.append(AckermannState(x, y, heading))
                if prev_xy is not None:
                    length += euclidean(prev_xy, (x, y))
                prev_xy = (x, y)
                prev_heading = heading
        return poses, length

    def _rewire(
        self,
        tree: List[TreeNode],
        children: List[set[int]],
        near_idx: int,
        new_parent: int,
        new_edge: SplineEdge,
        new_cost: float,
    ) -> None:
        old_parent = tree[near_idx].parent
        old_cost = tree[near_idx].cost
        if old_parent >= 0:
            children[old_parent].discard(near_idx)

        tree[near_idx].parent = int(new_parent)
        tree[near_idx].edge = new_edge
        tree[near_idx].cost = float(new_cost)
        children[new_parent].add(near_idx)

        delta = new_cost - old_cost
        if abs(delta) <= 1e-12:
            return
        self._propagate_cost_delta(tree, children, root_idx=near_idx, delta=delta)

    def _propagate_cost_delta(self, tree: List[TreeNode], children: List[set[int]], root_idx: int, delta: float) -> None:
        q = deque(children[root_idx])
        while q:
            idx = q.popleft()
            tree[idx].cost += delta
            for child_idx in children[idx]:
                q.append(child_idx)

    def _reconstruct(self, tree: List[TreeNode], idx: int) -> List[AckermannState]:
        node_indices: List[int] = []
        cur = idx
        while cur >= 0:
            node_indices.append(cur)
            cur = tree[cur].parent
        node_indices.reverse()
        if not node_indices:
            return []

        path: List[AckermannState] = [tree[node_indices[0]].state]
        for node_idx in node_indices[1:]:
            edge = tree[node_idx].edge
            if edge is None:
                path.append(tree[node_idx].state)
                continue
            poses, _ = self._edge_poses_and_length(edge, step=self.collision_step)
            if poses:
                path.extend(poses[1:])
        return path
