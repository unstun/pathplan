import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .common import euclidean, heading_diff
from .geometry import GridFootprintChecker
from .robot import AckermannParams, AckermannState, simulate_forward


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
        goal_xy_tol: float = 0.2,
        goal_theta_tol: float = math.radians(15.0),
        connect_threshold: float = 1.0,
        goal_check_freq: int = 1,
        seed_steps: int = 0,
        collision_step: Optional[float] = None,
        rewire: bool = True,
        lazy_collision: bool = False,
        theta_bins: int = 72,
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
        self.connect_threshold = connect_threshold
        self.goal_check_freq = max(1, int(goal_check_freq))
        self.seed_steps = max(0, int(seed_steps))
        self.collision_step = collision_step if collision_step is not None else grid_map.resolution * 0.5
        self.rewire = rewire
        self.lazy_collision = lazy_collision
        self.rng = np.random.default_rng(3)
        self.collision_checker = GridFootprintChecker(grid_map, footprint, theta_bins)

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        max_iter: int = 3000,
        timeout: float = 5.0,
    ):
        start_time = time.time()
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
                }

        for it in range(max_iter):
            if (time.time() - start_time) > timeout:
                break
            iterations = it + 1
            sample_state = self._sample_state(goal)
            nearest_idx = self._nearest(tree, sample_state)
            new_state = self._steer(tree[nearest_idx].state, sample_state)
            if self._blocked(tree[nearest_idx].state, new_state):
                continue
            new_cost = tree[nearest_idx].cost + self._segment_cost(tree[nearest_idx].state, new_state)
            parent_idx = nearest_idx

            if self.rewire:
                neighbors = self._neighbors(tree, new_state)
                for n_idx in neighbors:
                    cand_cost = tree[n_idx].cost + self._segment_cost(tree[n_idx].state, new_state)
                    if cand_cost < new_cost and not self._blocked(tree[n_idx].state, new_state):
                        new_cost = cand_cost
                        parent_idx = n_idx

            tree.append(TreeNode(new_state, parent_idx, new_cost))
            new_idx = len(tree) - 1

            if self.rewire:
                # rewire neighbors
                for n_idx in neighbors:
                    cand_cost = new_cost + self._segment_cost(new_state, tree[n_idx].state)
                    if cand_cost + 1e-6 < tree[n_idx].cost and not self._blocked(new_state, tree[n_idx].state):
                        tree[n_idx].parent = new_idx
                        tree[n_idx].cost = cand_cost

            if self._goal_reached(new_state, goal):
                goal_cost = new_cost + self._segment_cost(new_state, goal)
                tree.append(TreeNode(goal, parent=new_idx, cost=goal_cost))
                best_goal_idx = len(tree) - 1
                break
            if self._try_connect_goal(new_state, goal):
                goal_cost = new_cost + self._segment_cost(new_state, goal)
                tree.append(TreeNode(goal, parent=new_idx, cost=goal_cost))
                best_goal_idx = len(tree) - 1
                break
            if it % self.goal_check_freq == 0:
                greedy_idx = self._nearest(tree, goal)
                greedy_state = tree[greedy_idx].state
                if self._try_connect_goal(greedy_state, goal):
                    goal_cost = tree[greedy_idx].cost + self._segment_cost(greedy_state, goal)
                    tree.append(TreeNode(goal, parent=greedy_idx, cost=goal_cost))
                    best_goal_idx = len(tree) - 1
                    break

        path = self._reconstruct(tree, best_goal_idx) if best_goal_idx is not None else []
        if not path:
            greedy_idx = self._nearest(tree, goal)
            greedy_state = tree[greedy_idx].state
            if self._try_connect_goal(greedy_state, goal):
                goal_cost = tree[greedy_idx].cost + self._segment_cost(greedy_state, goal)
                tree.append(TreeNode(goal, parent=greedy_idx, cost=goal_cost))
                best_goal_idx = len(tree) - 1
                path = self._reconstruct(tree, best_goal_idx)
        return path, {
            "nodes": len(tree),
            "iterations": iterations,
            "time": time.time() - start_time,
            "success": best_goal_idx is not None,
        }

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
            d += 0.5 * self.params.min_turn_radius * abs(heading_diff(target.theta, node.state.theta))
            if d < best_dist:
                best_dist = d
                best = i
        return best

    def _steer(self, source: AckermannState, target: AckermannState) -> AckermannState:
        """Try both forward and reverse motions; pick the one that ends closer to the target pose."""
        def rollout(direction_sign: float) -> AckermannState:
            desired_heading = math.atan2(target.y - source.y, target.x - source.x)
            if direction_sign < 0:
                desired_heading = (desired_heading + math.pi) % (2 * math.pi)
            heading_error = heading_diff(desired_heading, source.theta)
            steering = max(-self.params.max_steer, min(self.params.max_steer, heading_error))
            return simulate_forward(
                source,
                steering=steering,
                velocity=direction_sign * self.velocity,
                duration=self.step_time,
                params=self.params,
            )

        def metric(state: AckermannState) -> float:
            pos_cost = euclidean((state.x, state.y), (target.x, target.y))
            heading_cost = 0.5 * self.params.min_turn_radius * abs(heading_diff(target.theta, state.theta))
            return pos_cost + heading_cost

        forward_state = rollout(+1.0)
        reverse_state = rollout(-1.0)
        return forward_state if metric(forward_state) <= metric(reverse_state) else reverse_state

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

    def _try_connect_goal(self, state: AckermannState, goal: AckermannState) -> bool:
        """Soft connection: if within distance threshold and collision-free straight interpolation."""
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.hypot(dx, dy)
        if dist > self.connect_threshold:
            return False
        if self._blocked(state, goal):
            return False
        return True

    def _seed_toward_goal(self, tree: List[TreeNode], goal: AckermannState, steps: int) -> Optional[int]:
        """Deterministically extend the tree toward the goal before random sampling."""
        parent_idx = 0
        for _ in range(steps):
            new_state = self._steer(tree[parent_idx].state, goal)
            if self._blocked(tree[parent_idx].state, new_state):
                break
            new_cost = tree[parent_idx].cost + self._segment_cost(tree[parent_idx].state, new_state)
            tree.append(TreeNode(new_state, parent=parent_idx, cost=new_cost))
            parent_idx = len(tree) - 1
            if self._try_connect_goal(new_state, goal):
                goal_cost = new_cost + self._segment_cost(new_state, goal)
                tree.append(TreeNode(goal, parent=parent_idx, cost=goal_cost))
                return len(tree) - 1
            if self._goal_reached(new_state, goal):
                return parent_idx
        return None

    def _blocked(self, a: AckermannState, b: AckermannState) -> bool:
        if self.lazy_collision:
            return self.collision_checker.collides_pose(b.x, b.y, b.theta)
        return self.collision_checker.motion_collides(a.as_tuple(), b.as_tuple(), step=self.collision_step)

    def _reconstruct(self, tree: List[TreeNode], idx: int) -> List[AckermannState]:
        path: List[AckermannState] = []
        while idx is not None and idx >= 0:
            node = tree[idx]
            path.append(node.state)
            idx = node.parent
        path.reverse()
        return path
