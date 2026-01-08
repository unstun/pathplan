import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .common import euclidean, heading_diff, wrap_angle, default_collision_step
from .geometry import interpolate_poses
from .robot import AckermannParams, AckermannState


@dataclass
class TreeNode:
    state: AckermannState
    parent: int
    cost: float


class RRTStarPlanner:
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
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.step_time = step_time
        self.velocity = velocity
        self.neighbor_radius = neighbor_radius
        self.goal_sample_rate = goal_sample_rate
        self.goal_xy_tol = goal_xy_tol
        self.goal_theta_tol = goal_theta_tol
        self.goal_check_freq = max(1, int(goal_check_freq))
        self.seed_steps = max(0, int(seed_steps))
        self.collision_step = collision_step if collision_step is not None else default_collision_step(grid_map.resolution)
        self.rewire = rewire
        self.steer_jitter = max(0.0, float(steer_jitter))
        # Keep heading penalties comparable even if min_turn_radius changes (e.g., tighter steering).
        baseline_turn_radius = 1.6
        self.heading_penalty_scale = max(1.0, baseline_turn_radius / max(self.params.min_turn_radius, 1e-6))
        default_heading_weight = 0.5 * self.params.min_turn_radius * self.heading_penalty_scale
        self.heading_penalty_weight = heading_penalty_weight if heading_penalty_weight is not None else default_heading_weight
        self.rng = np.random.default_rng(rng_seed)
        self.integration_dt = 0.05

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        max_iter: int = 3000,
        timeout: float = 5.0,
        self_check: bool = True,
    ):
        start_time = time.time()
        remediations: List[str] = []
        failure_reason = None
        base_step_time = self.step_time
        base_goal_sample_rate = self.goal_sample_rate
        adapted = False
        blocked_attempts = 0
        ref_path = self._grid_bfs_path(start, goal)
        try:
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

            tree: List[TreeNode] = [TreeNode(start, parent=-1, cost=0.0)]
            best_goal_idx = None
            iterations = 0

            if self.seed_steps > 0:
                maybe_goal = self._seed_toward_goal(tree, goal, self.seed_steps)
                if maybe_goal is not None:
                    best_goal_idx = maybe_goal
                    path = self._reconstruct(tree, best_goal_idx)
                    return path, {
                        "nodes": len(tree),
                        "iterations": iterations,
                        "time": time.time() - start_time,
                        "success": True,
                        "remediations": remediations,
                    }

            if ref_path:
                maybe_goal = self._seed_grid_path(tree, ref_path, goal)
                if maybe_goal is not None:
                    best_goal_idx = maybe_goal
                    path = self._reconstruct(tree, best_goal_idx)
                    return path, {
                        "nodes": len(tree),
                        "iterations": iterations,
                        "time": time.time() - start_time,
                        "success": True,
                        "remediations": remediations + ["grid_seed"],
                    }

            for it in range(max_iter):
                if (time.time() - start_time) > timeout:
                    break
                iterations = it + 1
                sample_state = self._sample_state(goal)
                if ref_path and self.rng.random() < 0.55:
                    gx, gy = ref_path[self.rng.integers(len(ref_path))]
                    sx, sy = self.map.grid_to_world(gx, gy)
                    heading = math.atan2(goal.y - sy, goal.x - sx)
                    sample_state = AckermannState(sx, sy, heading)
                if self.rng.random() < 0.35 and len(tree) > 1:
                    anchor = tree[self.rng.integers(len(tree))].state
                    radius = self.neighbor_radius * 0.6
                    dx = self.rng.normal(0.0, radius * 0.5)
                    dy = self.rng.normal(0.0, radius * 0.5)
                    sample_state = AckermannState(anchor.x + dx, anchor.y + dy, self.rng.uniform(-math.pi, math.pi))
                sgx, sgy = self.map.world_to_grid(sample_state.x, sample_state.y)
                if not self.map.in_bounds(sgx, sgy):
                    blocked_attempts += 1
                    continue
                nearest_idx = self._nearest(tree, sample_state)
                dist_to_sample = euclidean(
                    (tree[nearest_idx].state.x, tree[nearest_idx].state.y), (sample_state.x, sample_state.y)
                )
                step_override = self.step_time
                if dist_to_sample < self.velocity * self.step_time:
                    step_override = max(self.collision_step * 1.2, dist_to_sample / max(self.velocity, 1e-6))
                new_state, rollout_path = self._steer(tree[nearest_idx].state, sample_state, step_time=step_override)
                if self._trajectory_collides(rollout_path):
                    retry_found = False
                    for scale in (0.65, 0.45):
                        new_state, rollout_path = self._steer(
                            tree[nearest_idx].state, sample_state, step_time=step_override * scale
                        )
                        if not self._trajectory_collides(rollout_path):
                            retry_found = True
                            if "short_step_retry" not in remediations:
                                remediations.append("short_step_retry")
                            break
                    if not retry_found:
                        rand_heading = tree[nearest_idx].state.theta + self.rng.uniform(-math.pi, math.pi)
                        rand_target = AckermannState(
                            tree[nearest_idx].state.x + math.cos(rand_heading) * self.velocity * self.step_time * 0.5,
                            tree[nearest_idx].state.y + math.sin(rand_heading) * self.velocity * self.step_time * 0.5,
                            rand_heading,
                        )
                        new_state, rollout_path = self._steer(
                            tree[nearest_idx].state, rand_target, step_time=step_override * 0.5
                        )
                        if not self._trajectory_collides(rollout_path):
                            retry_found = True
                            if "explore_step" not in remediations:
                                remediations.append("explore_step")
                    if not retry_found:
                        blocked_attempts += 1
                        if self_check and not adapted and blocked_attempts >= 200:
                            blocked_ratio = blocked_attempts / max(1, iterations)
                            if blocked_ratio > 0.7:
                                self.step_time = max(0.25, self.step_time * 0.7)
                                self.goal_sample_rate = max(0.05, self.goal_sample_rate * 0.7)
                                remediations.append("shrink_step_time")
                                remediations.append("reduce_goal_bias")
                                adapted = True
                        continue
                new_cost = tree[nearest_idx].cost + self._segment_cost(tree[nearest_idx].state, new_state)
                parent_idx = nearest_idx

                if self.rewire:
                    neighbors = self._neighbors(tree, new_state)
                    for n_idx in neighbors:
                        cand_cost = tree[n_idx].cost + self._segment_cost(tree[n_idx].state, new_state)
                        if cand_cost < new_cost and not self._segment_collides(tree[n_idx].state, new_state):
                            new_cost = cand_cost
                            parent_idx = n_idx

                tree.append(TreeNode(new_state, parent_idx, new_cost))
                new_idx = len(tree) - 1

                if self.rewire:
                    # rewire neighbors
                    for n_idx in neighbors:
                        cand_cost = new_cost + self._segment_cost(new_state, tree[n_idx].state)
                        if cand_cost + 1e-6 < tree[n_idx].cost and not self._segment_collides(new_state, tree[n_idx].state):
                            tree[n_idx].parent = new_idx
                            tree[n_idx].cost = cand_cost

                if self._goal_reached(new_state, goal):
                    goal_cost = new_cost + self._segment_cost(new_state, goal)
                    tree.append(TreeNode(goal, parent=new_idx, cost=goal_cost))
                    best_goal_idx = len(tree) - 1
                    break
                if it % self.goal_check_freq == 0:
                    greedy_idx = self._nearest(tree, goal)
                    greedy_state = tree[greedy_idx].state
                    if self._goal_reached(greedy_state, goal):
                        goal_cost = tree[greedy_idx].cost + self._segment_cost(greedy_state, goal)
                        tree.append(TreeNode(goal, parent=greedy_idx, cost=goal_cost))
                        best_goal_idx = len(tree) - 1
                        break

            path = self._reconstruct(tree, best_goal_idx) if best_goal_idx is not None else []
            if best_goal_idx is None and ref_path:
                fallback_path = self._grid_track_path(start, goal, ref_path)
                if fallback_path:
                    path = fallback_path
                    best_goal_idx = len(path) - 1
                    remediations.append("grid_track_fallback")
            if best_goal_idx is None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    failure_reason = "timeout"
                elif iterations >= max_iter:
                    failure_reason = "iteration_budget_exhausted"
                else:
                    failure_reason = "search_failed"
            return path, {
                "nodes": len(tree),
                "iterations": iterations,
                "time": time.time() - start_time,
                "success": best_goal_idx is not None,
                "failure_reason": failure_reason,
                "remediations": remediations,
            }
        finally:
            self.step_time = base_step_time
            self.goal_sample_rate = base_goal_sample_rate

    def _sample_state(self, goal: AckermannState) -> AckermannState:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        x, y, theta = self.map.random_free_state(self.rng)
        return AckermannState(x, y, theta)

    def _nearest(self, tree: List[TreeNode], target: AckermannState) -> int:
        best = 0
        best_dist = float("inf")
        for i, node in enumerate(tree):
            d = euclidean((node.state.x, node.state.y), (target.x, target.y))
            d += self.heading_penalty_weight * abs(heading_diff(target.theta, node.state.theta))
            if d < best_dist:
                best_dist = d
                best = i
        return best

    def _steer(
        self, source: AckermannState, target: AckermannState, step_time: Optional[float] = None
    ) -> Tuple[AckermannState, List[AckermannState]]:
        """Try both forward and reverse motions; pick the rollout that ends closest to the target pose."""
        def metric(state: AckermannState) -> float:
            pos_cost = euclidean((state.x, state.y), (target.x, target.y))
            heading_cost = self.heading_penalty_weight * abs(heading_diff(target.theta, state.theta))
            return pos_cost + heading_cost

        def rollout(direction_sign: float) -> Tuple[AckermannState, List[AckermannState]]:
            desired_heading = math.atan2(target.y - source.y, target.x - source.x)
            if direction_sign < 0:
                desired_heading = (desired_heading + math.pi) % (2 * math.pi)
            heading_error = heading_diff(desired_heading, source.theta)
            base_steer = max(-self.params.max_steer, min(self.params.max_steer, heading_error))
            best_state = None
            best_path = None
            for delta in (0.0, self.steer_jitter, -self.steer_jitter):
                steering = max(-self.params.max_steer, min(self.params.max_steer, base_steer + delta))
                path = self._simulate_motion(
                    source, steering=steering, velocity=direction_sign * self.velocity, step_time=step_time
                )
                state = path[-1]
                if best_state is None or metric(state) < metric(best_state):
                    best_state = state
                    best_path = path
            return best_state, best_path

        forward_state, forward_path = rollout(+1.0)
        reverse_state, reverse_path = rollout(-1.0)
        if metric(forward_state) <= metric(reverse_state):
            return forward_state, forward_path
        return reverse_state, reverse_path

    def _simulate_motion(
        self, state: AckermannState, steering: float, velocity: float, step_time: Optional[float] = None
    ) -> List[AckermannState]:
        """
        Forward simulate with fixed inputs and record every integration step for collision checking.
        Always includes the starting state as the first element.
        """
        steering = max(-self.params.max_steer, min(self.params.max_steer, steering))
        v = max(-self.params.v_max, min(self.params.v_max, velocity))
        dt = self.integration_dt
        if self.collision_step > 0 and abs(v) > 1e-6:
            dt = min(dt, self.collision_step / abs(v))
        dt = max(1e-3, dt)
        total_time = self.step_time if step_time is None else max(1e-3, step_time)
        steps = max(1, int(total_time / dt))
        x, y, theta = state.x, state.y, state.theta
        path: List[AckermannState] = [AckermannState(x, y, theta)]
        for _ in range(steps):
            x += v * dt * math.cos(theta)
            y += v * dt * math.sin(theta)
            theta = wrap_angle(theta + (v * dt / self.params.wheelbase) * math.tan(steering))
            path.append(AckermannState(x, y, theta))
        return path

    def _segment_cost(self, a: AckermannState, b: AckermannState) -> float:
        return euclidean((a.x, a.y), (b.x, b.y))

    def _neighbors(self, tree: List[TreeNode], state: AckermannState) -> List[int]:
        res: List[int] = []
        for i, node in enumerate(tree):
            d = euclidean((node.state.x, node.state.y), (state.x, state.y))
            if d <= self.neighbor_radius:
                res.append(i)
        return res

    def _goal_reached(self, state: AckermannState, goal: AckermannState) -> bool:
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal.theta, state.theta))
        return dist <= self.goal_xy_tol and dtheta <= self.goal_theta_tol

    def _seed_toward_goal(self, tree: List[TreeNode], goal: AckermannState, steps: int) -> Optional[int]:
        """Deterministically extend the tree toward the goal before random sampling."""
        parent_idx = 0
        for _ in range(steps):
            new_state, rollout_path = self._steer(tree[parent_idx].state, goal)
            if self._trajectory_collides(rollout_path):
                break
            new_cost = tree[parent_idx].cost + self._segment_cost(tree[parent_idx].state, new_state)
            tree.append(TreeNode(new_state, parent=parent_idx, cost=new_cost))
            parent_idx = len(tree) - 1
            if self._goal_reached(new_state, goal):
                return parent_idx
        return None

    def _seed_grid_path(
        self, tree: List[TreeNode], ref_path: List[Tuple[int, int]], goal: AckermannState
    ) -> Optional[int]:
        if not ref_path or len(tree) == 0:
            return None
        stride = max(1, len(ref_path) // 150)
        parent_idx = 0
        for i in range(1, len(ref_path), stride):
            gx, gy = ref_path[i]
            tx, ty = self.map.grid_to_world(gx, gy)
            heading = math.atan2(ty - tree[parent_idx].state.y, tx - tree[parent_idx].state.x)
            target = AckermannState(tx, ty, heading)
            dist = euclidean((tree[parent_idx].state.x, tree[parent_idx].state.y), (tx, ty))
            step_time = max(self.collision_step * 1.2, min(self.step_time, dist / max(self.velocity, 1e-6)))
            new_state, rollout_path = self._steer(tree[parent_idx].state, target, step_time=step_time)
            if self._trajectory_collides(rollout_path):
                continue
            new_cost = tree[parent_idx].cost + self._segment_cost(tree[parent_idx].state, new_state)
            tree.append(TreeNode(new_state, parent=parent_idx, cost=new_cost))
            parent_idx = len(tree) - 1
            if self._goal_reached(new_state, goal):
                return parent_idx
        return None

    def _grid_track_path(
        self, start: AckermannState, goal: AckermannState, ref_path: List[Tuple[int, int]]
    ) -> List[AckermannState]:
        if not ref_path:
            return []
        tracked: List[AckermannState] = [start]
        current = start
        for gx, gy in ref_path[1:]:
            tx, ty = self.map.grid_to_world(gx, gy)
            heading = math.atan2(ty - current.y, tx - current.x)
            target = AckermannState(tx, ty, heading)
            if self._segment_collides(current, target):
                continue
            tracked.append(target)
            current = target
        if self._segment_collides(current, goal):
            return []
        tracked.append(goal)
        return tracked

    def _trajectory_collides(self, path: List[AckermannState]) -> bool:
        for pose in path:
            if self._pose_collides(pose.x, pose.y, pose.theta):
                return True
        return False

    def _segment_collides(self, a: AckermannState, b: AckermannState) -> bool:
        if self._pose_collides(a.x, a.y, a.theta):
            return True
        for pose in interpolate_poses(a.as_tuple(), b.as_tuple(), step=self.collision_step):
            if self._pose_collides(pose[0], pose[1], pose[2]):
                return True
        return False

    def _pose_collides(self, x: float, y: float, theta: float) -> bool:
        corners = self.footprint.corners(x, y, theta)
        min_x = min(px for px, _ in corners)
        max_x = max(px for px, _ in corners)
        min_y = min(py for _, py in corners)
        max_y = max(py for _, py in corners)
        ox, oy = self.map.origin
        res = self.map.resolution
        gx_min = math.floor((min_x - ox) / res)
        gx_max = math.ceil((max_x - ox) / res)
        gy_min = math.floor((min_y - oy) / res)
        gy_max = math.ceil((max_y - oy) / res)
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                if not self.map.in_bounds(gx, gy):
                    return True
                cx, cy = self.map.grid_to_world(gx, gy)
                if self.footprint.point_inside(cx, cy, x, y, theta) and self.map.is_occupied_index(gx, gy):
                    return True
        return False

    def _grid_bfs_path(self, start: AckermannState, goal: AckermannState) -> Optional[List[Tuple[int, int]]]:
        sx, sy = self.map.world_to_grid(start.x, start.y)
        gx, gy = self.map.world_to_grid(goal.x, goal.y)
        if self.map.is_occupied_index(sx, sy) or self.map.is_occupied_index(gx, gy):
            return None
        h, w = self.map.data.shape
        visited = np.full((h, w), -1, dtype=int)
        prev = np.full((h, w, 2), -1, dtype=int)
        dq = deque()
        dq.append((sx, sy))
        visited[sy, sx] = 0
        while dq:
            cx, cy = dq.popleft()
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
                dq.append((nx, ny))
        if visited[gy, gx] == -1:
            return None
        path: List[Tuple[int, int]] = []
        cx, cy = gx, gy
        while cx >= 0 and cy >= 0:
            path.append((cx, cy))
            if cx == sx and cy == sy:
                break
            pcx, pcy = prev[cy, cx, 0], prev[cy, cx, 1]
            cx, cy = int(pcx), int(pcy)
        path.reverse()
        return path

    def _reconstruct(self, tree: List[TreeNode], idx: int) -> List[AckermannState]:
        path: List[AckermannState] = []
        while idx is not None and idx >= 0:
            node = tree[idx]
            path.append(node.state)
            idx = node.parent
        path.reverse()
        return path
