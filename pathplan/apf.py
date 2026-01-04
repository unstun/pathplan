import math
import time
from collections import deque
from typing import List, Tuple

import numpy as np

from .common import clamp, heading_diff, wrap_angle, default_collision_step, default_min_motion_step
from .geometry import GridFootprintChecker
from .robot import AckermannParams, AckermannState


class APFPlanner:
    def __init__(
        self,
        grid_map,
        footprint,
        params: AckermannParams,
        step_size: float = 0.2,
        goal_tol: float = 0.1,
        repulse_radius: float = 0.8,
        obstacle_gain: float = 0.8,
        goal_gain: float = 1.0,
        max_iters: int = 800,
        gradient_eps: float = None,
        collision_step: float = None,
        stall_steps: int = 25,
        theta_bins: int = 72,
        min_step: float = None,
        jitter_angle: float = 0.5,
        heading_rate: float = None,
        coarse_collision: bool = False,
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.step_size = step_size
        self.goal_tol = goal_tol
        self.repulse_radius = repulse_radius
        self.obstacle_gain = obstacle_gain
        self.goal_gain = goal_gain
        self.max_iters = max_iters
        self.gradient_eps = gradient_eps if gradient_eps is not None else grid_map.resolution
        base_step = default_collision_step(grid_map.resolution)
        self.collision_step = collision_step if collision_step is not None else base_step
        self.stall_steps = max(1, int(stall_steps))
        self.min_step = min_step if min_step is not None else default_min_motion_step(grid_map.resolution)
        self.jitter_angle = jitter_angle
        self._jitter_sign = 1.0
        self.heading_rate = heading_rate if heading_rate is not None else self.step_size / max(self.params.min_turn_radius, 1e-6)
        self.coarse_collision = coarse_collision
        self.collision_checker = GridFootprintChecker(grid_map, footprint, theta_bins) if not coarse_collision else None
        self.obstacle_dist_map = self._compute_obstacle_distance_map()
        self.goal_dist_map = None

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        timeout: float = 3.0,
    ) -> Tuple[List[AckermannState], dict]:
        start_time = time.time()
        if (self.coarse_collision and self.map.is_occupied(start.x, start.y)) or (
            not self.coarse_collision and self.collision_checker.collides_pose(start.x, start.y, start.theta)
        ):
            return [], self._stats([], 0, time.time() - start_time, timed_out=False, reached=False)

        self.goal_dist_map = self._compute_goal_distance_map(goal)
        if self.goal_dist_map is None:
            return [], self._stats([], 0, time.time() - start_time, timed_out=False, reached=False)

        path: List[AckermannState] = [start]
        current = start
        last_potential = self._potential(current.x, current.y, goal)
        stall_counter = 0
        expansions = 0
        reached = False
        timed_out = False

        for _ in range(self.max_iters):
            if (time.time() - start_time) > timeout:
                timed_out = True
                break

            dist_goal = math.hypot(goal.x - current.x, goal.y - current.y)
            if dist_goal <= self.goal_tol:
                reached = True
                if self._segment_free(current, goal):
                    path.append(goal)
                break

            grad_x, grad_y = self._gradient(current.x, current.y, goal)
            if not math.isfinite(grad_x) or not math.isfinite(grad_y):
                break
            grad_norm = math.hypot(grad_x, grad_y)
            if grad_norm < 1e-6:
                break

            desired_heading = math.atan2(-grad_y, -grad_x)
            if stall_counter >= self.stall_steps:
                desired_heading = wrap_angle(desired_heading + self._jitter_sign * self.jitter_angle)
                self._jitter_sign *= -1.0
                stall_counter = 0
                step_scale = 0.5
            else:
                step_scale = 1.0
            heading_error = heading_diff(desired_heading, current.theta)
            max_heading_change = max(self.heading_rate, 1e-3)
            delta_theta = clamp(heading_error, -max_heading_change, max_heading_change)
            new_theta = wrap_angle(current.theta + delta_theta)

            step = min(self.step_size * step_scale, dist_goal)
            new_state = self._advance(current, new_theta, step)
            if new_state is None:
                break

            new_potential = self._potential(new_state.x, new_state.y, goal)
            if not math.isfinite(new_potential):
                break
            if new_potential >= last_potential - 1e-4:
                stall_counter += 1
                if stall_counter >= self.stall_steps:
                    break
            else:
                stall_counter = 0

            path.append(new_state)
            current = new_state
            last_potential = new_potential
            expansions += 1

        if not reached:
            path = []

        return path, self._stats(path, expansions, time.time() - start_time, timed_out=timed_out, reached=reached)

    def _advance(self, state: AckermannState, theta: float, step: float) -> AckermannState:
        """Move a small step while obeying curvature and checking collision."""
        trial_step = step
        attempts = 0
        while attempts < 8 and trial_step >= self.min_step:
            nx = state.x + trial_step * math.cos(theta)
            ny = state.y + trial_step * math.sin(theta)
            candidate = AckermannState(nx, ny, theta)
            if self._segment_free(state, candidate):
                return candidate
            trial_step *= 0.5
            attempts += 1
        return None

    def _segment_free(self, a: AckermannState, b: AckermannState) -> bool:
        if self.coarse_collision:
            steps = max(1, int(math.ceil(math.hypot(b.x - a.x, b.y - a.y) / max(self.collision_step, 1e-6))))
            for i in range(steps + 1):
                s = i / steps
                x = a.x + (b.x - a.x) * s
                y = a.y + (b.y - a.y) * s
                if self.map.is_occupied(x, y):
                    return False
            return True
        else:
            return not self.collision_checker.motion_collides(a.as_tuple(), b.as_tuple(), step=self.collision_step)

    def _gradient(self, x: float, y: float, goal: AckermannState) -> Tuple[float, float]:
        eps = max(self.gradient_eps, 1e-4)
        px1 = self._potential(x + eps, y, goal)
        px0 = self._potential(x - eps, y, goal)
        py1 = self._potential(x, y + eps, goal)
        py0 = self._potential(x, y - eps, goal)
        gx = (px1 - px0) / (2.0 * eps)
        gy = (py1 - py0) / (2.0 * eps)
        return gx, gy

    def _potential(self, x: float, y: float, goal: AckermannState) -> float:
        d_goal = math.hypot(goal.x - x, goal.y - y)
        u_att = 0.5 * self.goal_gain * d_goal * d_goal

        d_goal = self._sample_goal_distance(x, y)
        if not math.isfinite(d_goal):
            return float("inf")
        u_att = self.goal_gain * d_goal

        d_obs = self._sample_obstacle_distance(x, y)
        if d_obs <= 1e-6:
            return float("inf")

        if d_obs < self.repulse_radius:
            inv = (1.0 / d_obs) - (1.0 / self.repulse_radius)
            u_rep = 0.5 * self.obstacle_gain * inv * inv
        else:
            u_rep = 0.0
        return u_att + u_rep

    def _sample_obstacle_distance(self, x: float, y: float) -> float:
        gx = (x - self.map.origin[0]) / self.map.resolution
        gy = (y - self.map.origin[1]) / self.map.resolution
        h, w = self.obstacle_dist_map.shape
        if gx < 0 or gy < 0 or gx > (w - 1) or gy > (h - 1):
            return 0.0
        x0 = int(math.floor(gx))
        y0 = int(math.floor(gy))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        wx = gx - x0
        wy = gy - y0
        d00 = self.obstacle_dist_map[y0, x0]
        d10 = self.obstacle_dist_map[y0, x1]
        d01 = self.obstacle_dist_map[y1, x0]
        d11 = self.obstacle_dist_map[y1, x1]
        d0 = d00 * (1 - wx) + d10 * wx
        d1 = d01 * (1 - wx) + d11 * wx
        return d0 * (1 - wy) + d1 * wy

    def _sample_goal_distance(self, x: float, y: float) -> float:
        gx = (x - self.map.origin[0]) / self.map.resolution
        gy = (y - self.map.origin[1]) / self.map.resolution
        if self.goal_dist_map is None:
            return float("inf")
        h, w = self.goal_dist_map.shape
        if gx < 0 or gy < 0 or gx > (w - 1) or gy > (h - 1):
            return float("inf")
        x0 = int(math.floor(gx))
        y0 = int(math.floor(gy))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        wx = gx - x0
        wy = gy - y0
        d00 = self.goal_dist_map[y0, x0]
        d10 = self.goal_dist_map[y0, x1]
        d01 = self.goal_dist_map[y1, x0]
        d11 = self.goal_dist_map[y1, x1]
        d0 = d00 * (1 - wx) + d10 * wx
        d1 = d01 * (1 - wx) + d11 * wx
        return d0 * (1 - wy) + d1 * wy

    def _compute_obstacle_distance_map(self) -> np.ndarray:
        """Approximate Euclidean distance-to-obstacle using a chamfer transform."""
        occ = np.asarray(self.map.data, dtype=bool)
        h, w = occ.shape
        dist = np.full((h, w), np.inf, dtype=float)
        dist[occ] = 0.0
        sqrt2 = math.sqrt(2.0)

        # forward pass
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

        # backward pass
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

        dist *= self.map.resolution
        if not np.isfinite(dist).any():
            fallback = max(h, w) * self.map.resolution * 2.0
            dist[:] = fallback
        else:
            if np.isinf(dist).any():
                fallback = np.nanmax(dist[np.isfinite(dist)])
                dist[np.isinf(dist)] = fallback
        return dist

    def _compute_goal_distance_map(self, goal: AckermannState) -> np.ndarray:
        """Grid shortest-path distance from every free cell to the goal (4-neighborhood)."""
        h, w = self.map.data.shape
        occ = np.asarray(self.map.data, dtype=bool)
        gxi, gyi = self.map.world_to_grid(goal.x, goal.y)
        if gxi < 0 or gxi >= w or gyi < 0 or gyi >= h or occ[gyi, gxi]:
            return None
        dist = np.full((h, w), np.inf, dtype=float)
        dist[gyi, gxi] = 0.0
        q = deque()
        q.append((gxi, gyi))
        while q:
            x, y = q.popleft()
            d = dist[y, x]
            nd = d + 1.0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if occ[ny, nx]:
                    continue
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    q.append((nx, ny))
        dist *= self.map.resolution
        if not np.isfinite(dist).any():
            return None
        return dist

    def _stats(
        self,
        path: List[AckermannState],
        expansions: int,
        elapsed: float,
        timed_out: bool,
        reached: bool,
    ) -> dict:
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            length += math.hypot(dx, dy)
        return {
            "path_length": length,
            "expansions": expansions,
            "time": elapsed,
            "timed_out": timed_out,
            "success": reached and bool(path),
        }
