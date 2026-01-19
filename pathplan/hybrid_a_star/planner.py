import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..common import default_collision_step, heading_diff
from ..geometry import GridFootprintChecker
from ..primitives import MotionPrimitive, default_primitives, primitive_cost
from ..robot import AckermannParams, AckermannState, sample_constant_steer_motion

from .holonomic_heuristic import dijkstra_2d_cost_to_go
from .reeds_shepp import reeds_shepp_shortest_path


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
        # Thesis-style cost model: turning/reversing are multiplicative, direction changes are additive.
        turning_penalty: float = 1.05,
        reverse_penalty: float = 1.0,
        direction_change_penalty: float = 2.0,
        steering_penalty: float = 0.0,
        steering_change_penalty: float = 0.0,
        # Same-cell expansion (thesis Algorithm 6).
        same_cell_tie_breaker: float = 1e-3,
        # Analytical expansion (thesis §6.1.2).
        analytic_expansion: bool = True,
        analytic_expansion_interval: int = 8,
        analytic_expansion_distance_scale: float = 10.0,
        # Heuristic configuration (thesis §6.2).
        use_holonomic_heuristic: bool = True,
        allow_diagonal: bool = True,
        use_reeds_shepp_heuristic: bool = True,
        reeds_shepp_heuristic_max_dist: float = 15.0,
        # Collision checker padding (meters), optional safety buffer.
        collision_padding: Optional[float] = None,
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
        self.turning_penalty = max(1.0, float(turning_penalty))
        self.reverse_penalty = max(0.0, float(reverse_penalty))
        self.direction_change_penalty = max(0.0, float(direction_change_penalty))
        self.steering_penalty = max(0.0, float(steering_penalty))
        self.steering_change_penalty = max(0.0, float(steering_change_penalty))
        self.same_cell_tie_breaker = max(0.0, float(same_cell_tie_breaker))

        self.analytic_expansion = bool(analytic_expansion)
        self.analytic_expansion_interval = max(1, int(analytic_expansion_interval))
        self.analytic_expansion_distance_scale = max(1e-6, float(analytic_expansion_distance_scale))

        self.use_holonomic_heuristic = bool(use_holonomic_heuristic)
        self.allow_diagonal = bool(allow_diagonal)
        self.use_reeds_shepp_heuristic = bool(use_reeds_shepp_heuristic)
        self.reeds_shepp_heuristic_max_dist = max(0.0, float(reeds_shepp_heuristic_max_dist))

        self.collision_checker = GridFootprintChecker(
            grid_map, footprint, theta_bins=theta_bins, padding=collision_padding
        )

        self._holonomic_cache_key = None
        self._holonomic_cache = None

    def _discretize(self, state: AckermannState) -> Tuple[int, int, int]:
        gx = int(math.floor((state.x - self.map.origin[0]) / self.xy_resolution))
        gy = int(math.floor((state.y - self.map.origin[1]) / self.xy_resolution))
        theta = state.theta % (2.0 * math.pi)
        theta_id = int(math.floor((theta / (2.0 * math.pi)) * self.theta_bins)) % self.theta_bins
        return gx, gy, theta_id

    def _goal_reached(self, state: AckermannState, goal: AckermannState) -> bool:
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal.theta, state.theta))
        return dist <= self.goal_xy_tol and dtheta <= self.goal_theta_tol

    def _priority(self, node: Node) -> float:
        return node.g + self.heuristic_weight * node.h

    def _transition_cost(self, prev: Optional[MotionPrimitive], prim: MotionPrimitive) -> float:
        cost = primitive_cost(prim)
        if prim.direction < 0:
            cost *= self.reverse_penalty
        if abs(prim.steering) > 1e-9:
            cost *= self.turning_penalty
        if prev is not None and prim.direction != prev.direction:
            cost += self.direction_change_penalty
        if self.steering_penalty > 0.0:
            cost += self.steering_penalty * abs(prim.steering)
        if self.steering_change_penalty > 0.0 and prev is not None:
            cost += self.steering_change_penalty * abs(prim.steering - prev.steering)
        return cost

    def _evaluate_primitive(
        self, current: Node, prim: MotionPrimitive, goal: AckermannState
    ) -> Optional[Tuple[AckermannState, float]]:
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
            return None
        g_new = current.g + self._transition_cost(current.action, prim)
        return nxt, g_new

    def _holonomic_cost_to_go(self, dist_map, goal_center: Tuple[float, float], goal_offset: float, state: AckermannState) -> float:
        gx, gy = self.map.world_to_grid(state.x, state.y)
        if not self.map.in_bounds(gx, gy):
            return float("inf")
        d = float(dist_map[gy, gx])
        if not math.isfinite(d):
            return float("inf")
        cx, cy = self.map.grid_to_world(gx, gy)
        offset = math.hypot(state.x - cx, state.y - cy)
        return max(0.0, d - (offset + goal_offset))

    def _heuristic(self, state: AckermannState, goal: AckermannState, dist_map, goal_center, goal_offset) -> float:
        # Unconstrained (holonomic) heuristic: Dijkstra cost-to-go on the occupancy grid.
        h_holo = 0.0
        if dist_map is not None:
            h_holo = self._holonomic_cost_to_go(dist_map, goal_center, goal_offset, state)
            if not math.isfinite(h_holo):
                h_holo = math.hypot(goal.x - state.x, goal.y - state.y)

        # Constrained heuristic: Reeds–Shepp distance (optionally gated for performance).
        h_rs = 0.0
        if self.use_reeds_shepp_heuristic and (dist_map is None or h_holo <= self.reeds_shepp_heuristic_max_dist):
            rs = reeds_shepp_shortest_path(state.as_tuple(), goal.as_tuple(), self.params.min_turn_radius)
            if rs is not None and math.isfinite(rs.total_length):
                h_rs = float(rs.total_length)
            else:
                # Robust fallback: straight-line plus heading change lower bound.
                dtheta = abs(heading_diff(goal.theta, state.theta))
                h_rs = math.hypot(goal.x - state.x, goal.y - state.y) + self.params.min_turn_radius * dtheta

        return max(h_holo, h_rs)

    def _analytic_interval(self, state: AckermannState, goal: AckermannState) -> int:
        if not self.analytic_expansion:
            return 0
        d = math.hypot(goal.x - state.x, goal.y - state.y)
        scale = max(0.2, min(1.0, d / self.analytic_expansion_distance_scale))
        return max(1, int(round(self.analytic_expansion_interval * scale)))

    def _try_analytic_expansion(self, state: AckermannState, goal: AckermannState) -> Optional[Tuple[List[AckermannState], List[MotionPrimitive]]]:
        rs = reeds_shepp_shortest_path(state.as_tuple(), goal.as_tuple(), self.params.min_turn_radius)
        if rs is None:
            return None

        cur = state
        extra_states: List[AckermannState] = []
        extra_actions: List[MotionPrimitive] = []
        for seg_type, seg_len in zip(rs.segment_types, rs.segment_lengths):
            seg_len = float(seg_len)
            if abs(seg_len) <= 1e-9:
                continue
            direction = 1 if seg_len >= 0.0 else -1
            step_len = abs(seg_len)
            if seg_type == "S":
                steering = 0.0
            elif seg_type == "L":
                steering = self.params.max_steer
            elif seg_type == "R":
                steering = -self.params.max_steer
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown Reeds–Shepp segment type: {seg_type!r}")

            seg_states, _ = sample_constant_steer_motion(
                cur,
                steering,
                direction,
                step_len,
                self.params,
                step=self.collision_step,
                footprint=None,
            )
            if self.collision_checker.collides_path(seg_states):
                return None
            cur = seg_states[-1]
            extra_states.append(cur)
            extra_actions.append(MotionPrimitive(steering=steering, direction=direction, step=step_len, weight=1.0))

        if not extra_states:
            return None
        if not self._goal_reached(extra_states[-1], goal):
            return None
        # Snap the final pose to the exact goal for clean plotting/metrics.
        extra_states[-1] = goal
        return extra_states, extra_actions

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

        if self.collision_checker.collides_pose(start.x, start.y, start.theta):
            return [], self._stats(
                [],
                [],
                expansions=0,
                elapsed=time.time() - start_time,
                trace_poses=[],
                trace_boxes=[],
                timed_out=False,
                failure_reason="start_in_collision",
                remediations=remediations,
            )
        if self.collision_checker.collides_pose(goal.x, goal.y, goal.theta):
            return [], self._stats(
                [],
                [],
                expansions=0,
                elapsed=time.time() - start_time,
                trace_poses=[],
                trace_boxes=[],
                timed_out=False,
                failure_reason="goal_in_collision",
                remediations=remediations,
            )

        dist_map = None
        goal_center = None
        goal_offset = 0.0
        if self.use_holonomic_heuristic:
            g_gx, g_gy = self.map.world_to_grid(goal.x, goal.y)
            cache_key = (id(self.map.data), self.map.data.shape, float(self.map.resolution), self.map.origin, g_gx, g_gy, self.allow_diagonal)
            if cache_key == self._holonomic_cache_key and self._holonomic_cache is not None:
                dist_map, goal_center, goal_offset = self._holonomic_cache
            else:
                dist_map = dijkstra_2d_cost_to_go(self.map.data, (g_gx, g_gy), self.map.resolution, allow_diagonal=self.allow_diagonal)
                goal_center = self.map.grid_to_world(g_gx, g_gy)
                goal_offset = math.hypot(goal.x - goal_center[0], goal.y - goal_center[1])
                self._holonomic_cache_key = cache_key
                self._holonomic_cache = (dist_map, goal_center, goal_offset)

        h0 = self._heuristic(start, goal, dist_map, goal_center, goal_offset)
        start_node = Node(start, g=0.0, h=h0, parent=None, action=None)

        open_nodes: Dict[Tuple[int, int, int], Node] = {}
        closed_nodes: Dict[Tuple[int, int, int], Node] = {}
        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        insert_counter = 0

        start_key = self._discretize(start)
        open_nodes[start_key] = start_node
        heapq.heappush(open_heap, (self._priority(start_node), insert_counter, start_key))
        insert_counter += 1

        expansions = 0

        while open_heap and (time.time() - start_time) < timeout and expansions < max_nodes:
            popped_priority, _, key = heapq.heappop(open_heap)
            current = open_nodes.get(key)
            if current is None:
                continue
            current_priority = self._priority(current)
            # stale heap entry?
            if abs(popped_priority - current_priority) > 1e-9:
                continue
            del open_nodes[key]
            closed_nodes[key] = current
            expansion_idx = expansions
            expansions += 1

            if self._goal_reached(current.state, goal):
                path, actions = self._reconstruct_with_actions(current)
                trace_poses, trace_boxes = self._trace_path(path, actions)
                return path, self._stats(
                    path,
                    actions,
                    expansions=expansions,
                    elapsed=time.time() - start_time,
                    trace_poses=trace_poses,
                    trace_boxes=trace_boxes,
                    failure_reason=None,
                    remediations=remediations,
                )

            if self_check and self.analytic_expansion:
                interval = self._analytic_interval(current.state, goal)
                if interval > 0 and expansion_idx % interval == 0:
                    analytic = self._try_analytic_expansion(current.state, goal)
                    if analytic is not None:
                        extra_states, extra_actions = analytic
                        path, actions = self._reconstruct_with_actions(current)
                        path.extend(extra_states)
                        actions.extend(extra_actions)
                        trace_poses, trace_boxes = self._trace_path(path, actions)
                        return path, self._stats(
                            path,
                            actions,
                            expansions=expansions,
                            elapsed=time.time() - start_time,
                            trace_poses=trace_poses,
                            trace_boxes=trace_boxes,
                            failure_reason=None,
                            remediations=remediations + ["analytic_expansion"],
                        )
            current_f = current_priority

            for prim in self.primitives:
                evaluated = self._evaluate_primitive(current, prim, goal)
                if evaluated is None:
                    continue
                nxt, g_new = evaluated
                nxt_key = self._discretize(nxt)

                h_new = self._heuristic(nxt, goal, dist_map, goal_center, goal_offset)
                node = Node(nxt, g=g_new, h=h_new, parent=current, action=prim)

                if nxt_key == key and self.same_cell_tie_breaker > 0.0:
                    if self._priority(node) > current_f + self.same_cell_tie_breaker:
                        continue
                else:
                    existing_closed = closed_nodes.get(nxt_key)
                    if existing_closed is not None and g_new >= existing_closed.g - 1e-9:
                        continue

                existing_open = open_nodes.get(nxt_key)
                if existing_open is not None and g_new >= existing_open.g - 1e-9:
                    continue

                open_nodes[nxt_key] = node
                heapq.heappush(open_heap, (self._priority(node), insert_counter, nxt_key))
                insert_counter += 1

        elapsed = time.time() - start_time
        timed_out = elapsed >= timeout
        if not open_heap:
            failure_reason = "open_set_exhausted"
        elif timed_out:
            failure_reason = "timeout"
        elif expansions >= max_nodes:
            failure_reason = "node_budget_exhausted"
        else:
            failure_reason = "search_failed"
        return [], self._stats(
            [],
            [],
            expansions=expansions,
            elapsed=elapsed,
            trace_poses=[],
            trace_boxes=[],
            timed_out=timed_out,
            failure_reason=failure_reason,
            remediations=remediations,
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
        actions: List[Optional[MotionPrimitive]],
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
        prev_dir: Optional[int] = None
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            length += math.hypot(dx, dy)
        for act in actions[1:] if len(actions) > 1 else []:
            if act is None:
                continue
            if prev_dir is None:
                prev_dir = act.direction
                continue
            if act.direction != prev_dir:
                cusps += 1
                prev_dir = act.direction
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
