import math
from typing import Tuple

from .common import heading_diff
from .robot import AckermannParams


def euclidean_heading(
    state: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    params: AckermannParams,
    heading_weight: float = 1.0,
) -> float:
    """Euclidean distance plus heading penalty scaled by turning radius."""
    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    dist = math.hypot(dx, dy)
    dtheta = abs(heading_diff(goal[2], state[2]))
    return dist + heading_weight * params.min_turn_radius * dtheta


def reeds_shepp_lower_bound(
    state: Tuple[float, float, float], goal: Tuple[float, float, float], params: AckermannParams
) -> float:
    """
    Lightweight lower bound of RS distance:
    straight-line plus min turning radius scaled heading change.
    """
    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    dist = math.hypot(dx, dy)
    dtheta = abs(heading_diff(goal[2], state[2]))
    return dist + params.min_turn_radius * dtheta


def admissible_heuristic(
    state: Tuple[float, float, float], goal: Tuple[float, float, float], params: AckermannParams
) -> float:
    """Use RS lower bound as anchor heuristic."""
    return reeds_shepp_lower_bound(state, goal, params)
