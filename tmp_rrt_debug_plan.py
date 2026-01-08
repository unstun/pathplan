import math, time
from pathplan.rrt_star import RRTStarPlanner
from pathplan.geometry import OrientedBoxFootprint
from pathplan.robot import AckermannParams
from examples.forest_scene import build_variant, make_forest_map
from pathplan.common import default_collision_step

class DebugRRT(RRTStarPlanner):
    def plan(self, start, goal, max_iter=30000, timeout=3.0):
        start_time = time.time()
        tree = [self._make_node(start, parent=-1, cost=0.0)] if hasattr(self, '_make_node') else None
        # reuse base method but instrument by copying original body (simpler)
        tree = [type('TN',(object,),{'state':start,'parent':-1,'cost':0.0})()]
        best_goal_idx = None
        iterations = 0
        for it in range(max_iter):
            if (time.time() - start_time) > timeout:
                print('timeout at', it)
                break
            iterations = it + 1
            sample_state = goal if self.rng.random() < self.goal_sample_rate else AckermannState(*self.map.random_free_state(self.rng))
            nearest_idx = self._nearest(tree, sample_state)
            new_state, rollout_path = self._steer(tree[nearest_idx].state, sample_state)
            if self._trajectory_collides(rollout_path):
                continue
            new_cost = tree[nearest_idx].cost + self._segment_cost(tree[nearest_idx].state, new_state)
            parent_idx = nearest_idx
            tree.append(type(tree[0])( ))
            tree[-1].state = new_state; tree[-1].parent = parent_idx; tree[-1].cost = new_cost
            if self._goal_reached(new_state, goal):
                print('goal reached at', it+1, 'state', new_state)
                best_goal_idx = len(tree)-1
                break
        return best_goal_idx is not None

from pathplan.robot import AckermannState
variant_slug, title, map_kwargs, start, goal = build_variant('small_map_small_gap')
grid_map = make_forest_map(**map_kwargs)
params = AckermannParams()
footprint = OrientedBoxFootprint(length=0.924, width=0.740)
r = DebugRRT(grid_map, footprint, params,
    goal_sample_rate=0.70,
    neighbor_radius=4.0,
    step_time=0.90,
    velocity=1.0,
    goal_xy_tol=max(0.1, 0.25),
    goal_theta_tol=max(math.radians(5.0), math.radians(12.0)),
    goal_check_freq=1,
    seed_steps=40,
    collision_step=default_collision_step(grid_map.resolution, preferred=0.15, max_step=0.25),
    rewire=False,
)
print('result', r.plan(start, goal, max_iter=30000, timeout=3.0))
