import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import heading_diff
from .geometry import GridFootprintChecker
from .heuristics import admissible_heuristic
from .primitives import MotionPrimitive, default_primitives, primitive_cost
from .rl_models import RLGuidance
from .robot import AckermannParams, AckermannState, sample_constant_steer_motion


@dataclass
class Node:
    state: AckermannState
    g: float
    h_anchor: float
    h_rl: float
    parent: Optional["Node"]
    action: Optional[MotionPrimitive]

    @property
    def f_anchor(self) -> float:
        return self.g + self.h_anchor

    def f_rl(self, weight: float) -> float:
        return self.g + weight * self.h_rl


class RLGuidedHybridPlanner:
    """
    Multi-heuristic Hybrid A* with RL-guided queue + action ordering.
    Anchor queue preserves completeness; RL queue biases speed.
    """

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
        rl_top_k: Optional[int] = None,
        anchor_inflation: float = 1.0,
        rl_weight: float = 1.0,
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
        self.rl_top_k = rl_top_k
        self.anchor_inflation = anchor_inflation
        self.rl_weight = rl_weight
        self.rl = RLGuidance(params)
        self.collision_checker = GridFootprintChecker(grid_map, footprint, theta_bins)

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
        max_nodes: int = 25000,
    ) -> Tuple[List[AckermannState], Dict[str, float]]:
        start_time = time.time()
        anchor_heap: List[Tuple[float, int, Node]] = []
        rl_heap: List[Tuple[float, int, Node]] = []

        h_a = admissible_heuristic(start.as_tuple(), goal.as_tuple(), self.params)
        h_rl = self.rl.value(start.as_tuple(), goal.as_tuple(), self.map)
        start_node = Node(start, g=0.0, h_anchor=h_a, h_rl=h_rl, parent=None, action=None)
        heapq.heappush(anchor_heap, (start_node.f_anchor, 0, start_node))
        heapq.heappush(rl_heap, (start_node.f_rl(self.rl_weight), 0, start_node))

        visited: Dict[Tuple[int, int, int], float] = {}
        expansions_anchor = 0
        expansions_rl = 0
        iterations = 0
        counter = 1

        while (anchor_heap or rl_heap) and (time.time() - start_time) < timeout and (
            expansions_anchor + expansions_rl
        ) < max_nodes:
            # Alternate between anchor and RL to preserve coverage.
            iterations += 1
            use_anchor = True
            if iterations % 2 == 1 and rl_heap:
                use_anchor = False
            if not anchor_heap:
                use_anchor = False
            if not rl_heap:
                use_anchor = True
            heap = anchor_heap if use_anchor else rl_heap
            _, _, current = heapq.heappop(heap)
            key = self._discretize(current.state)
            if key in visited and visited[key] <= current.g:
                continue
            visited[key] = current.g

            if self._goal_reached(current.state, goal):
                path, actions = self._reconstruct_with_actions(current)
                trace_poses, trace_boxes = self._trace_path(path, actions)
                # Explicitly append the goal pose so outputs/plots hit the destination.
                if not (math.isclose(path[-1].x, goal.x) and math.isclose(path[-1].y, goal.y) and math.isclose(path[-1].theta, goal.theta)):
                    path.append(goal)
                return path, self._stats(
                    path,
                    expansions_anchor,
                    expansions_rl,
                    time.time() - start_time,
                    trace_poses,
                    trace_boxes,
                    timed_out=False,
                )

            if use_anchor:
                expansions_anchor += 1
                actions = list(self.primitives)
            else:
                expansions_rl += 1
                scores = self.rl.policy(current.state.as_tuple(), goal.as_tuple(), self.map, self.primitives)
                idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                k = self.rl_top_k if self.rl_top_k is not None else len(idx)
                chosen = idx[:k]
                actions = [self.primitives[i] for i in chosen]

            for prim in actions:
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
                    g_new += 0.2
                h_a_new = admissible_heuristic(nxt.as_tuple(), goal.as_tuple(), self.params)
                h_rl_new = self.rl.value(nxt.as_tuple(), goal.as_tuple(), self.map)
                new_key = self._discretize(nxt)
                if new_key in visited and visited[new_key] <= g_new:
                    continue
                node = Node(nxt, g=g_new, h_anchor=h_a_new, h_rl=h_rl_new, parent=current, action=prim)
                heapq.heappush(anchor_heap, (node.f_anchor, counter, node))
                heapq.heappush(rl_heap, (node.f_rl(self.rl_weight), counter, node))
                counter += 1

        return [], self._stats(
            [],
            expansions_anchor,
            expansions_rl,
            time.time() - start_time,
            [],
            [],
            timed_out=True,
        )

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
        expansions_anchor: int,
        expansions_rl: int,
        elapsed: float,
        trace_poses: List[Tuple[float, float, float]],
        trace_boxes: List[List[Tuple[float, float]]],
        timed_out: bool,
    ):
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            length += math.hypot(dx, dy)
        stats = {
            "path_length": length,
            "expansions_anchor": expansions_anchor,
            "expansions_rl": expansions_rl,
            "expansions_total": expansions_anchor + expansions_rl,
            "time": elapsed,
            "timed_out": timed_out,
        }
        if trace_poses:
            stats["trace_poses"] = trace_poses
        if trace_boxes:
            stats["trace_boxes"] = trace_boxes
        return stats
