import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import heading_diff
from .geometry import path_collides
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
        xy_resolution: float = 0.1,
        theta_bins: int = 72,
        collision_step: float = 0.1,
        goal_xy_tol: float = 0.3,
        goal_theta_tol: float = math.radians(15.0),
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.primitives = primitives if primitives is not None else default_primitives(params)
        self.xy_resolution = xy_resolution
        self.theta_bins = theta_bins
        self.collision_step = collision_step
        self.goal_xy_tol = goal_xy_tol
        self.goal_theta_tol = goal_theta_tol

    def _discretize(self, state: AckermannState) -> Tuple[int, int, int]:
        gx, gy = self.map.world_to_grid(state.x, state.y)
        theta_id = int(round(((state.theta % (2 * math.pi)) / (2 * math.pi)) * self.theta_bins)) % self.theta_bins
        return gx, gy, theta_id

    def _goal_reached(self, state: AckermannState, goal: AckermannState) -> bool:
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal.theta, state.theta))
        return dist <= self.goal_xy_tol and dtheta <= self.goal_theta_tol

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        timeout: float = 5.0,
        max_nodes: int = 20000,
    ) -> Tuple[List[AckermannState], Dict[str, float]]:
        """Return path (list of states) and stats dict. Empty path on failure."""
        start_time = time.time()
        open_heap: List[Tuple[float, int, Node]] = []
        h0 = admissible_heuristic(start.as_tuple(), goal.as_tuple(), self.params)
        start_node = Node(start, g=0.0, h=h0, parent=None, action=None)
        heapq.heappush(open_heap, (start_node.f, 0, start_node))
        visited: Dict[Tuple[int, int, int], float] = {}
        expansions = 0
        insert_counter = 1

        while open_heap and (time.time() - start_time) < timeout and expansions < max_nodes:
            _, _, current = heapq.heappop(open_heap)
            key = self._discretize(current.state)
            if key in visited and visited[key] <= current.g:
                continue
            visited[key] = current.g

            if self._goal_reached(current.state, goal):
                path, actions = self._reconstruct_with_actions(current)
                trace_poses, trace_boxes = self._trace_path(path, actions)
                # Ensure the goal pose is explicitly included for plotting/metrics.
                if not (math.isclose(path[-1].x, goal.x) and math.isclose(path[-1].y, goal.y) and math.isclose(path[-1].theta, goal.theta)):
                    path.append(goal)
                return path, self._stats(path, expansions, time.time() - start_time, trace_poses, trace_boxes)

            expansions += 1
            for prim in self.primitives:
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
                if path_collides(self.map, self.footprint, [s.as_tuple() for s in arc_states], sample_step=self.collision_step):
                    continue
                g_new = current.g + primitive_cost(prim)
                if current.action and prim.direction != current.action.direction:
                    g_new += 0.2  # cusp penalty
                h_new = admissible_heuristic(nxt.as_tuple(), goal.as_tuple(), self.params)
                node = Node(nxt, g=g_new, h=h_new, parent=current, action=prim)
                heapq.heappush(open_heap, (node.f, insert_counter, node))
                insert_counter += 1

        return [], self._stats([], expansions, time.time() - start_time, [], [], timed_out=True)

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
        if trace_poses:
            stats["trace_poses"] = trace_poses
        if trace_boxes:
            stats["trace_boxes"] = trace_boxes
        return stats
