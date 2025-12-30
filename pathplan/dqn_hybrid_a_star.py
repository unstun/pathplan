import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import heading_diff
from .geometry import GridFootprintChecker
from .heuristics import admissible_heuristic
from .primitives import MotionPrimitive, default_primitives, primitive_cost
from .dqn_models import DQNGuidance
from .robot import AckermannParams, AckermannState, sample_constant_steer_motion


@dataclass
class Node:
    state: AckermannState
    g: float
    h_anchor: float
    h_dqn: float
    parent: Optional["Node"]
    action: Optional[MotionPrimitive]
    dqn_features: Optional[Tuple[float, float, float, float]] = None
    dqn_heading_err: float = 0.0
    dqn_front_occ: float = 0.0

    @property
    def f_anchor(self) -> float:
        return self.g + self.h_anchor

    def f_dqn(self, weight: float) -> float:
        return self.g + weight * self.h_dqn


class DQNHybridAStarPlanner:
    """
    Multi-heuristic Hybrid A* with DQN-guided queue + action ordering.
    Anchor queue preserves completeness; DQN queue biases speed.
    Drop in a trained DQN inside DQNGuidance to replace the lightweight heuristic.
    """

    def __init__(
        self,
        grid_map,
        footprint,
        params: AckermannParams,
        primitives: Optional[List[MotionPrimitive]] = None,
        xy_resolution: Optional[float] = None,
        theta_bins: int = 72,
        collision_step: float = 0.1,
        goal_xy_tol: float = 0.3,
        goal_theta_tol: float = math.radians(15.0),
        dqn_top_k: Optional[int] = None,
        anchor_inflation: float = 1.0,
        dqn_weight: float = 1.0,
        dqn_lead_threshold: float = 1.0,
        max_dqn_streak: int = 3,
    ):
        self.map = grid_map
        self.footprint = footprint
        self.params = params
        self.primitives = primitives if primitives is not None else default_primitives(params)
        self.xy_resolution = xy_resolution if xy_resolution is not None else grid_map.resolution
        self.theta_bins = theta_bins
        self.collision_step = collision_step
        self.goal_xy_tol = goal_xy_tol
        self.goal_theta_tol = goal_theta_tol
        self.dqn_top_k = dqn_top_k
        self.anchor_inflation = anchor_inflation
        self.dqn_weight = dqn_weight
        self.dqn_lead_threshold = dqn_lead_threshold
        self.max_dqn_streak = max(1, max_dqn_streak)
        self.dqn = DQNGuidance(params)
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

    def _anchor_priority(self, node: Node) -> float:
        return node.g + self.anchor_inflation * node.h_anchor

    def plan(
        self,
        start: AckermannState,
        goal: AckermannState,
        timeout: float = 5.0,
        max_nodes: int = 25000,
    ) -> Tuple[List[AckermannState], Dict[str, float]]:
        start_time = time.time()
        anchor_heap: List[Tuple[float, int, Node]] = []
        dqn_heap: List[Tuple[float, int, Node]] = []

        h_a = admissible_heuristic(start.as_tuple(), goal.as_tuple(), self.params)
        dqn_feats, dqn_front_occ, dqn_heading_err = self.dqn.evaluate(start.as_tuple(), goal.as_tuple(), self.map)
        h_dqn = self.dqn.value_from_features(dqn_feats)
        start_node = Node(
            start,
            g=0.0,
            h_anchor=h_a,
            h_dqn=h_dqn,
            parent=None,
            action=None,
            dqn_features=dqn_feats,
            dqn_heading_err=dqn_heading_err,
            dqn_front_occ=dqn_front_occ,
        )
        heapq.heappush(anchor_heap, (self._anchor_priority(start_node), 0, start_node))
        heapq.heappush(dqn_heap, (start_node.f_dqn(self.dqn_weight), 0, start_node))

        visited: Dict[Tuple[int, int, int], float] = {}
        expansions_anchor = 0
        expansions_dqn = 0
        counter = 1
        dqn_streak = 0

        while (anchor_heap or dqn_heap) and (time.time() - start_time) < timeout and (
            expansions_anchor + expansions_dqn
        ) < max_nodes:
            # Prefer DQN only when its best candidate is better; cap consecutive DQN pops to avoid anchor starvation.
            use_anchor = True
            if anchor_heap and dqn_heap:
                anchor_key = anchor_heap[0][0]
                dqn_key = dqn_heap[0][0]
                dqn_leads = dqn_key <= anchor_key * self.dqn_lead_threshold
                if dqn_leads and dqn_streak < self.max_dqn_streak:
                    use_anchor = False
            elif not anchor_heap and dqn_heap:
                use_anchor = False
            elif not dqn_heap and anchor_heap:
                use_anchor = True

            heap = anchor_heap if use_anchor else dqn_heap
            _, _, current = heapq.heappop(heap)
            key = self._discretize(current.state)
            if key in visited and visited[key] <= current.g:
                continue
            visited[key] = current.g
            if current.dqn_features is None:
                feats, front_occ, heading_err = self.dqn.evaluate(current.state.as_tuple(), goal.as_tuple(), self.map)
                current.dqn_features = feats
                current.dqn_front_occ = front_occ
                current.dqn_heading_err = heading_err

            if self._goal_reached(current.state, goal):
                path, actions = self._reconstruct_with_actions(current)
                trace_poses, trace_boxes = self._trace_path(path, actions)
                # Explicitly append the goal pose so outputs/plots hit the destination.
                if not (math.isclose(path[-1].x, goal.x) and math.isclose(path[-1].y, goal.y) and math.isclose(path[-1].theta, goal.theta)):
                    path.append(goal)
                return path, self._stats(
                    path,
                    expansions_anchor,
                    expansions_dqn,
                    time.time() - start_time,
                    trace_poses,
                    trace_boxes,
                    timed_out=False,
                )

            if use_anchor:
                expansions_anchor += 1
                dqn_streak = 0
                actions = list(self.primitives)
            else:
                expansions_dqn += 1
                dqn_streak += 1
                scores = self.dqn.policy_from_eval(
                    current.dqn_features,
                    current.dqn_heading_err,
                    current.dqn_front_occ,
                    self.primitives,
                )
                idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                k = self.dqn_top_k if self.dqn_top_k is not None else len(idx)
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
                new_key = self._discretize(nxt)
                if new_key in visited and visited[new_key] <= g_new:
                    continue
                h_a_new = admissible_heuristic(nxt.as_tuple(), goal.as_tuple(), self.params)
                dqn_feats, dqn_front_occ, dqn_heading_err = self.dqn.evaluate(nxt.as_tuple(), goal.as_tuple(), self.map)
                h_dqn_new = self.dqn.value_from_features(dqn_feats)
                node = Node(
                    nxt,
                    g=g_new,
                    h_anchor=h_a_new,
                    h_dqn=h_dqn_new,
                    parent=current,
                    action=prim,
                    dqn_features=dqn_feats,
                    dqn_heading_err=dqn_heading_err,
                    dqn_front_occ=dqn_front_occ,
                )
                heapq.heappush(anchor_heap, (self._anchor_priority(node), counter, node))
                heapq.heappush(dqn_heap, (node.f_dqn(self.dqn_weight), counter, node))
                counter += 1

        return [], self._stats(
            [],
            expansions_anchor,
            expansions_dqn,
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
        expansions_dqn: int,
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
            "expansions_dqn": expansions_dqn,
            "expansions_total": expansions_anchor + expansions_dqn,
            "time": elapsed,
            "timed_out": timed_out,
        }
        if trace_poses:
            stats["trace_poses"] = trace_poses
        if trace_boxes:
            stats["trace_boxes"] = trace_boxes
        return stats
