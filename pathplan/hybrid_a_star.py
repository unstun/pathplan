import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import heading_diff, default_collision_step
from .geometry import GridFootprintChecker
from .heuristics import admissible_heuristic
from .primitives import MotionPrimitive, default_primitives, primitive_cost
from .robot import AckermannParams, AckermannState, sample_constant_steer_motion


@dataclass
class Node:
    state: AckermannState
    g: float
    h: float
    parent: Optional["Node"]
    action: Optional[MotionPrimitive]
    direction_changes: int = 0

    @property
    def f(self) -> float:
        return self.g + self.h


class HybridAStarPlanner:
    def __init__(
        self,
        grid_map,
        footprint,
        params: AckermannParams,
        primitives: Optional[List[MotionPrimitive]] = None,
        xy_resolution: Optional[float] = None,
        theta_bins: int = 72,
        collision_step: Optional[float] = None,
        goal_xy_tol: float = 0.1,
        goal_theta_tol: float = math.radians(5.0),
        heuristic_weight: float = 1.0,
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.primitives = primitives if primitives is not None else default_primitives(params)
        self.xy_resolution = xy_resolution if xy_resolution is not None else grid_map.resolution
        self.theta_bins = theta_bins
        self.collision_step = collision_step if collision_step is not None else default_collision_step(grid_map.resolution)
        self.goal_xy_tol = goal_xy_tol
        self.goal_theta_tol = goal_theta_tol
        self.heuristic_weight = max(heuristic_weight, 1e-6)
        self.collision_checker = GridFootprintChecker(grid_map, footprint, theta_bins)

    def _discretize(self, state: AckermannState) -> Tuple[int, int, int]:
        gx = int(round((state.x - self.map.origin[0]) / self.xy_resolution))
        gy = int(round((state.y - self.map.origin[1]) / self.xy_resolution))
        theta_id = int(round(((state.theta % (2 * math.pi)) / (2 * math.pi)) * self.theta_bins)) % self.theta_bins
        return gx, gy, theta_id

    def _goal_reached(self, state: AckermannState, goal: AckermannState) -> bool:
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal.theta, state.theta))
        return dist <= self.goal_xy_tol and dtheta <= self.goal_theta_tol

    def _priority(self, node: Node) -> float:
        return node.g + self.heuristic_weight * node.h

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        timeout: float = 5.0,
        max_nodes: int = 20000,
        self_check: bool = True,
    ) -> Tuple[List[AckermannState], Dict[str, float]]:
        """Return path (list of states) and stats dict. Empty path on failure."""
        start_time = time.time()
        remediations: List[str] = []
        failure_reason = None
        base_heuristic_weight = self.heuristic_weight
        node_budget = max_nodes
        max_node_budget = max(max_nodes, int(max_nodes * 1.5))
        stall_limit = max(100, int(max_nodes * 0.05))
        last_improve_expansion = 0
        open_heap: List[Tuple[float, int, Node]] = []
        h0 = admissible_heuristic(start.as_tuple(), goal.as_tuple(), self.params)
        start_node = Node(start, g=0.0, h=h0, parent=None, action=None)
        best_goal_dist = math.hypot(goal.x - start.x, goal.y - start.y)
        best_h = h0
        relaxed_weight = False
        added_fine_primitives = False
        extra_budget_used = False
        primitives = list(self.primitives)
        heapq.heappush(open_heap, (self._priority(start_node), 0, start_node))
        visited: Dict[Tuple[int, int, int], float] = {}
        expansions = 0
        insert_counter = 1
        try:
            if self.collision_checker.collides_pose(start.x, start.y, start.theta):
                return [], self._stats([], 0, time.time() - start_time, [], [], timed_out=False, failure_reason="start_in_collision", remediations=remediations)
            if self.collision_checker.collides_pose(goal.x, goal.y, goal.theta):
                return [], self._stats([], 0, time.time() - start_time, [], [], timed_out=False, failure_reason="goal_in_collision", remediations=remediations)

            while open_heap and (time.time() - start_time) < timeout and expansions < node_budget:
                _, _, current = heapq.heappop(open_heap)
                key = self._discretize(current.state)
                if key in visited and visited[key] <= current.g:
                    continue
                visited[key] = current.g

                goal_dist = math.hypot(goal.x - current.state.x, goal.y - current.state.y)
                if goal_dist < best_goal_dist - 1e-3 or current.h < best_h - 1e-3:
                    best_goal_dist = min(best_goal_dist, goal_dist)
                    best_h = min(best_h, current.h)
                    last_improve_expansion = expansions
                if self_check and expansions - last_improve_expansion >= stall_limit:
                    if not relaxed_weight and self.heuristic_weight > 1.0:
                        self.heuristic_weight = max(1.0, self.heuristic_weight * 0.5)
                        open_heap = [(self._priority(node), idx, node) for _, idx, node in open_heap]
                        heapq.heapify(open_heap)
                        remediations.append("relax_heuristic_weight")
                        relaxed_weight = True
                        last_improve_expansion = expansions
                    elif not added_fine_primitives:
                        min_step = min(abs(p.step) for p in primitives) if primitives else 0.0
                        if min_step > 0.4:
                            fine_step = max(0.1, min_step * 0.5)
                            primitives = primitives + default_primitives(self.params, step_length=fine_step)
                            remediations.append(f"add_fine_primitives:{fine_step:.2f}")
                            added_fine_primitives = True
                            last_improve_expansion = expansions

                if self._goal_reached(current.state, goal):
                    path, actions = self._reconstruct_with_actions(current)
                    trace_poses, trace_boxes = self._trace_path(path, actions)
                    # Ensure the goal pose is explicitly included for plotting/metrics.
                    if not (math.isclose(path[-1].x, goal.x) and math.isclose(path[-1].y, goal.y) and math.isclose(path[-1].theta, goal.theta)):
                        path.append(goal)
                    return path, self._stats(path, expansions, time.time() - start_time, trace_poses, trace_boxes, failure_reason=None, remediations=remediations)

                expansions += 1
                for prim in primitives:
                    arc_states, _ = sample_constant_steer_motion(
                        current.state,
                        prim.steering,
                        prim.direction,
                        prim.step,
                        self.params,
                        step=self.collision_step,
                        footprint=None,
                    )
                    nxt = arc_states[-1]
                    if self.collision_checker.collides_path(arc_states):
                        continue
                    g_new = current.g + primitive_cost(prim)
                    if current.action and prim.direction != current.action.direction:
                        g_new += 0.2  # cusp penalty
                    new_key = self._discretize(nxt)
                    if new_key in visited and visited[new_key] <= g_new:
                        continue
                    h_new = admissible_heuristic(nxt.as_tuple(), goal.as_tuple(), self.params)
                    node = Node(nxt, g=g_new, h=h_new, parent=current, action=prim)
                    heapq.heappush(open_heap, (self._priority(node), insert_counter, node))
                    insert_counter += 1

                if self_check and not extra_budget_used and expansions >= node_budget - 1:
                    elapsed = time.time() - start_time
                    if elapsed < timeout * 0.9 and expansions - last_improve_expansion < stall_limit:
                        new_budget = min(max_node_budget, int(node_budget * 1.25))
                        if new_budget > node_budget:
                            node_budget = new_budget
                            extra_budget_used = True
                            remediations.append("expanded_node_budget")

            elapsed = time.time() - start_time
            timed_out = elapsed >= timeout
            budget_exhausted = expansions >= node_budget
            if not open_heap:
                failure_reason = "open_set_exhausted"
            elif timed_out:
                failure_reason = "timeout"
            elif budget_exhausted:
                failure_reason = "node_budget_exhausted"
            else:
                failure_reason = "search_failed"
            return [], self._stats([], expansions, elapsed, [], [], timed_out=timed_out, failure_reason=failure_reason, remediations=remediations)
        finally:
            self.heuristic_weight = base_heuristic_weight

    def _reconstruct_with_actions(self, node: Node) -> Tuple[List[AckermannState], List[Optional[MotionPrimitive]]]:
        path: List[AckermannState] = []
        actions: List[Optional[MotionPrimitive]] = []
        while node is not None:
            path.append(node.state)
            actions.append(node.action)
            node = node.parent
        path.reverse()
        actions.reverse()
        return path, actions

    def _trace_path(
        self, path: List[AckermannState], actions: List[Optional[MotionPrimitive]]
    ) -> Tuple[List[Tuple[float, float, float]], List[List[Tuple[float, float]]]]:
        """Densify the path into arc samples and bounding boxes for visualization."""
        if not path:
            return [], []
        trace_poses: List[Tuple[float, float, float]] = []
        trace_boxes: List[List[Tuple[float, float]]] = []
        for i in range(1, len(path)):
            prim = actions[i]
            if prim is None:
                continue
            seg_states, seg_boxes = sample_constant_steer_motion(
                path[i - 1],
                prim.steering,
                prim.direction,
                prim.step,
                self.params,
                step=self.collision_step,
                footprint=self.footprint,
            )
            if i > 1 and seg_states:
                seg_states = seg_states[1:]
                if seg_boxes:
                    seg_boxes = seg_boxes[1:]
            trace_poses.extend([s.as_tuple() for s in seg_states])
            trace_boxes.extend(seg_boxes)
        return trace_poses, trace_boxes

    def _stats(
        self,
        path: List[AckermannState],
        expansions: int,
        elapsed: float,
        trace_poses: List[Tuple[float, float, float]],
        trace_boxes: List[List[Tuple[float, float]]],
        timed_out: bool = False,
        failure_reason: str = None,
        remediations: List[str] = None,
    ):
        length = 0.0
        cusps = 0
        prev_dir = None
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            length += math.hypot(dx, dy)
        # cusp counting uses stored actions; approximate using headings
        stats = {
            "path_length": length,
            "cusps": cusps,
            "expansions": expansions,
            "time": elapsed,
            "timed_out": timed_out,
        }
        if failure_reason:
            stats["failure_reason"] = failure_reason
        if remediations:
            stats["remediations"] = remediations
        if trace_poses:
            stats["trace_poses"] = trace_poses
        if trace_boxes:
            stats["trace_boxes"] = trace_boxes
        return stats
