import math
import time
from collections import deque
from typing import List, Optional, Tuple

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
        escape_samples: int = 12,
        escape_angle: float = 1.0,
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
        self.collision_checker = GridFootprintChecker(grid_map, footprint, theta_bins)
        # Keep the potential fields aligned with the inflated collision footprint by
        # dilating obstacles before building distance maps.
        padding_for_distance = max(self.collision_checker.padding, 0.0)
        if padding_for_distance > 0.0:
            self._distance_occ = grid_map.inflate(padding_for_distance).data
        else:
            self._distance_occ = np.asarray(grid_map.data)
        self.obstacle_dist_map = self._compute_obstacle_distance_map()
        self.goal_dist_map = None
        self.escape_samples = max(4, int(escape_samples))
        self.escape_angle = max(0.1, float(escape_angle))
        offsets = np.linspace(-self.escape_angle, self.escape_angle, self.escape_samples)
        self._escape_offsets = sorted(offsets.tolist(), key=lambda v: abs(v))

    def _goal_map_step(self, state: AckermannState) -> Optional[AckermannState]:
        """
        Take a small step that strictly decreases the grid shortest-path distance
        to the goal (ignores repulsion). This is a low-noise fallback when the
        potential gradient is invalid/flat, commonly encountered near ragged
        real-world map boundaries.
        """
        if self.goal_dist_map is None:
            return None
        gx = int(round((state.x - self.map.origin[0]) / self.map.resolution))
        gy = int(round((state.y - self.map.origin[1]) / self.map.resolution))
        h, w = self.goal_dist_map.shape
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return None
        current_d = self.goal_dist_map[gy, gx]
        if not math.isfinite(current_d):
            return None
        best = None
        best_d = current_d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = gx + dx
            ny = gy + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            nd = self.goal_dist_map[ny, nx]
            if not math.isfinite(nd):
                continue
            if nd + 1e-6 < best_d:
                best = (nx, ny, nd)
                best_d = nd
        if best is None:
            return None
        tx, ty = self.map.grid_to_world(best[0], best[1])
        heading = math.atan2(ty - state.y, tx - state.x)
        step = min(self.step_size, math.hypot(tx - state.x, ty - state.y) * 1.05)
        return self._advance(state, heading, step)

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        timeout: float = 3.0,
        self_check: bool = True,
    ) -> Tuple[List[AckermannState], dict]:
        start_time = time.time()
        remediations: List[str] = []
        failure_reason = None
        base_obstacle_gain = self.obstacle_gain
        base_goal_gain = self.goal_gain
        base_repulse_radius = self.repulse_radius
        try:
            if self.collision_checker.collides_pose(start.x, start.y, start.theta):
                return [], self._stats(
                    [],
                    0,
                    time.time() - start_time,
                    timed_out=False,
                    reached=False,
                    failure_reason="start_in_collision",
                    remediations=remediations,
                )

            self.goal_dist_map = self._compute_goal_distance_map(goal)
            if self.goal_dist_map is None:
                return [], self._stats(
                    [],
                    0,
                    time.time() - start_time,
                    timed_out=False,
                    reached=False,
                    failure_reason="goal_unreachable",
                    remediations=remediations,
                )

            path: List[AckermannState] = [start]
            current = start
            last_potential = self._potential(current.x, current.y, goal)
            stall_counter = 0
            expansions = 0
            reached = False
            timed_out = False
            recovery_attempts = 0

            for _ in range(self.max_iters):
                if (time.time() - start_time) > timeout:
                    timed_out = True
                    failure_reason = "timeout"
                    break

                dist_goal = math.hypot(goal.x - current.x, goal.y - current.y)
                if dist_goal <= self.goal_tol:
                    reached = True
                    if self._segment_free(current, goal):
                        path.append(goal)
                    break

                grad_x, grad_y = self._gradient(current.x, current.y, goal)
                if not math.isfinite(grad_x) or not math.isfinite(grad_y):
                    fallback = self._goal_map_step(current)
                    if self_check and fallback is not None:
                        path.append(fallback)
                        current = fallback
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(fallback.x, fallback.y, goal)
                        remediations.append("goal_grid_step")
                        continue
                    escape_state = self._attempt_escape(
                        current,
                        goal,
                        current.theta,
                        min(self.step_size, dist_goal),
                        last_potential,
                    )
                    if self_check and escape_state is not None:
                        path.append(escape_state)
                        current = escape_state
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(escape_state.x, escape_state.y, goal)
                        remediations.append("escape_step")
                        continue
                    failure_reason = "gradient_invalid"
                    break
                grad_norm = math.hypot(grad_x, grad_y)
                if grad_norm < 1e-6:
                    fallback = self._goal_map_step(current)
                    if self_check and fallback is not None:
                        path.append(fallback)
                        current = fallback
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(fallback.x, fallback.y, goal)
                        remediations.append("goal_grid_step")
                        continue
                    escape_state = self._attempt_escape(
                        current,
                        goal,
                        current.theta,
                        min(self.step_size, dist_goal),
                        last_potential,
                    )
                    if self_check and escape_state is not None:
                        path.append(escape_state)
                        current = escape_state
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(escape_state.x, escape_state.y, goal)
                        remediations.append("escape_step")
                        continue
                    failure_reason = "flat_gradient"
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
                    escape_state = self._attempt_escape(
                        current,
                        goal,
                        desired_heading,
                        step,
                        last_potential,
                    )
                    if self_check and escape_state is not None:
                        path.append(escape_state)
                        current = escape_state
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(escape_state.x, escape_state.y, goal)
                        remediations.append("escape_step")
                        continue
                    failure_reason = "collision_blocked"
                    break

                new_potential = self._potential(new_state.x, new_state.y, goal)
                if not math.isfinite(new_potential):
                    failure_reason = "invalid_potential"
                    break
                if new_potential >= last_potential - 1e-4:
                    stall_counter += 1
                    fallback = None
                    if self_check and stall_counter >= max(1, self.stall_steps // 2):
                        fallback = self._goal_map_step(current)
                    if self_check and fallback is not None:
                        path.append(fallback)
                        current = fallback
                        expansions += 1
                        stall_counter = 0
                        last_potential = self._potential(fallback.x, fallback.y, goal)
                        remediations.append("goal_grid_step")
                        continue
                    if stall_counter >= self.stall_steps:
                        escape_state = self._attempt_escape(
                            current,
                            goal,
                            desired_heading,
                            step,
                            last_potential,
                        )
                        if self_check and escape_state is not None:
                            path.append(escape_state)
                            current = escape_state
                            expansions += 1
                            stall_counter = 0
                            last_potential = self._potential(escape_state.x, escape_state.y, goal)
                            remediations.append("escape_step")
                            continue
                        if self_check and recovery_attempts < 2:
                            recovery_attempts += 1
                            self.obstacle_gain *= 0.85
                            self.goal_gain *= 1.08
                            self.repulse_radius *= 0.9
                            remediations.append("relax_repulsion")
                            stall_counter = 0
                            continue
                        failure_reason = "local_minima"
                        break
                else:
                    stall_counter = 0

                path.append(new_state)
                current = new_state
                last_potential = new_potential
                expansions += 1

            if not reached:
                path = []
                if failure_reason is None and not timed_out:
                    failure_reason = "search_failed"

            return path, self._stats(
                path,
                expansions,
                time.time() - start_time,
                timed_out=timed_out,
                reached=reached,
                failure_reason=failure_reason,
                remediations=remediations,
            )
        finally:
            self.obstacle_gain = base_obstacle_gain
            self.goal_gain = base_goal_gain
            self.repulse_radius = base_repulse_radius

    def _attempt_escape(
        self,
        current: AckermannState,
        goal: AckermannState,
        heading: float,
        step: float,
        last_potential: float,
    ) -> Optional[AckermannState]:
        best = None
        best_potential = last_potential
        fallback = None
        fallback_potential = float("inf")
        base_headings = [heading]
        if abs(wrap_angle(current.theta - heading)) > 1e-3:
            base_headings.append(current.theta)
        for base_heading in base_headings:
            for offset in self._escape_offsets + [-math.pi, math.pi]:
                theta = wrap_angle(base_heading + offset)
                candidate = self._advance(current, theta, step)
                if candidate is None:
                    continue
                candidate_potential = self._potential(candidate.x, candidate.y, goal)
                if candidate_potential < best_potential - 1e-4:
                    best_potential = candidate_potential
                    best = candidate
                if candidate_potential < fallback_potential:
                    fallback_potential = candidate_potential
                    fallback = candidate
        return best if best is not None else fallback

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

        repulse_gain = self.obstacle_gain
        if d_goal < self.repulse_radius:
            # Weaken repulsion close to the goal to avoid oscillations in tight clearances.
            scale = clamp(d_goal / max(self.repulse_radius, 1e-6), 0.35, 1.0)
            repulse_gain *= 0.5 + 0.5 * scale

        if d_obs < self.repulse_radius:
            inv = (1.0 / d_obs) - (1.0 / self.repulse_radius)
            u_rep = 0.5 * repulse_gain * inv * inv
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
        occ = np.asarray(self._distance_occ, dtype=bool)
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
        occ = np.asarray(self._distance_occ, dtype=bool)
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
        failure_reason: str = None,
        remediations: List[str] = None,
    ) -> dict:
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            length += math.hypot(dx, dy)
        stats = {
            "path_length": length,
            "expansions": expansions,
            "time": elapsed,
            "timed_out": timed_out,
            "success": reached and bool(path),
        }
        if failure_reason:
            stats["failure_reason"] = failure_reason
        if remediations:
            stats["remediations"] = remediations
        return stats
