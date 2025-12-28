import math
from typing import List, Sequence, Tuple

import numpy as np

from .common import heading_diff
from .robot import AckermannParams


class RLGuidance:
    """
    Lightweight RL-style guidance module (torch-optional).
    Provides:
    - value(): learned heuristic estimate of cost-to-go
    - policy(): action preference scores over discrete primitives
    """

    def __init__(self, params: AckermannParams, patch_size: float = 8.0, patch_cells: int = 32):
        self.params = params
        self.patch_size = patch_size
        self.patch_cells = patch_cells
        self.rng = np.random.default_rng(7)
        # deterministic pseudo-trained weights
        self.value_weights = np.array([1.0, 0.8, 2.0, 0.5], dtype=float)
        self.policy_weights = np.array([1.2, 0.6, 1.4, 0.25], dtype=float)

    def _extract_features(
        self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map
    ) -> np.ndarray:
        patch = grid_map.occupancy_patch(
            state[0], state[1], state[2], size_m=self.patch_size, cells=self.patch_cells
        )
        occupied = float(np.mean(patch))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal[2], state[2]))
        clearance = 1.0 - occupied
        return np.array([dist, dtheta, occupied, clearance], dtype=float)

    def value(self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map) -> float:
        """Approximate cost-to-go; positive scalar."""
        feats = self._extract_features(state, goal, grid_map)
        return float(np.dot(self.value_weights, feats))

    def policy(
        self,
        state: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        grid_map,
        actions: Sequence,
    ) -> List[float]:
        """
        Return preference scores for each action (higher = better).
        Uses goal heading alignment and local clearance in front of the robot.
        """
        feats = self._extract_features(state, goal, grid_map)
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        goal_heading = math.atan2(dy, dx)
        heading_err = heading_diff(goal_heading, state[2])
        # simple forward arc occupancy estimate
        patch = grid_map.occupancy_patch(
            state[0], state[1], state[2], size_m=self.patch_size, cells=self.patch_cells
        )
        front_band = patch[self.patch_cells // 2 :, self.patch_cells // 3 : 2 * self.patch_cells // 3]
        front_occ = float(np.mean(front_band))

        scores: List[float] = []
        for act in actions:
            steer_term = -abs(act.steering) / (self.params.max_steer + 1e-6)
            dir_term = 0.0 if act.direction > 0 else -0.3
            # encourage steering toward goal heading
            steer_sign = math.copysign(1.0, act.steering) if act.steering != 0 else 0.0
            goal_sign = math.copysign(1.0, heading_err) if heading_err != 0 else 0.0
            heading_term = -abs(heading_err) + 0.2 * (1.0 if steer_sign == goal_sign else 0.0)
            clearance_term = -front_occ
            score = (
                self.policy_weights[0] * steer_term
                + self.policy_weights[1] * heading_term
                + self.policy_weights[2] * dir_term
                + self.policy_weights[3] * clearance_term
            )
            scores.append(score)
        return scores
