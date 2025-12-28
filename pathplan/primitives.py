from dataclasses import dataclass
from typing import List

from .robot import AckermannParams


@dataclass(frozen=True)
class MotionPrimitive:
    steering: float  # radians
    direction: int  # +1 forward, -1 reverse
    step: float  # meters
    weight: float = 1.0  # cost multiplier


def default_primitives(
    params: AckermannParams, step_length: float = 0.3, delta_scale: float = 0.5
) -> List[MotionPrimitive]:
    """Generate 10 actions: 5 steering bins x {forward, reverse}."""
    delta_max = params.max_steer
    delta_small = delta_scale * delta_max
    steering_bins = [-delta_max, -delta_small, 0.0, delta_small, delta_max]
    prims: List[MotionPrimitive] = []
    for d in steering_bins:
        prims.append(MotionPrimitive(d, +1, step_length, weight=1.0))
    for d in steering_bins:
        prims.append(MotionPrimitive(d, -1, step_length, weight=1.2))  # slight reverse penalty
    return prims


def primitive_cost(primitive: MotionPrimitive) -> float:
    """Base traversal cost for one primitive."""
    return abs(primitive.step) * primitive.weight
