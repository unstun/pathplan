import math
from dataclasses import dataclass
from typing import Tuple

from .common import wrap_angle


@dataclass
class AckermannParams:
    wheelbase: float = 0.6
    min_turn_radius: float = 1.1284
    v_max: float = 1.0

    @property
    def max_steer(self) -> float:
        return math.atan(self.wheelbase / self.min_turn_radius)


@dataclass
class AckermannState:
    x: float
    y: float
    theta: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta


def propagate(
    state: AckermannState,
    steering: float,
    direction: int,
    step_length: float,
    params: AckermannParams,
) -> AckermannState:
    """Integrate one motion primitive with constant steering and fixed arc length.

    Uses the exact circular-arc solution for the bicycle model so trajectories
    follow true curves instead of straight-line Euler steps.
    """
    steering = max(-params.max_steer, min(params.max_steer, steering))
    ds = step_length * float(direction)
    k = math.tan(steering) / params.wheelbase  # curvature

    if abs(k) < 1e-8:
        x = state.x + ds * math.cos(state.theta)
        y = state.y + ds * math.sin(state.theta)
        theta = wrap_angle(state.theta + ds * k)
    else:
        dtheta = ds * k
        x = state.x + (math.sin(state.theta + dtheta) - math.sin(state.theta)) / k
        y = state.y - (math.cos(state.theta + dtheta) - math.cos(state.theta)) / k
        theta = wrap_angle(state.theta + dtheta)
    return AckermannState(x, y, theta)


def simulate_forward(
    state: AckermannState,
    steering: float,
    velocity: float,
    duration: float,
    params: AckermannParams,
    dt: float = 0.05,
) -> AckermannState:
    """Forward simulate bicycle model with constant inputs."""
    steering = max(-params.max_steer, min(params.max_steer, steering))
    v = max(-params.v_max, min(params.v_max, velocity))
    steps = max(1, int(duration / dt))
    x, y, theta = state.x, state.y, state.theta
    for _ in range(steps):
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        theta = wrap_angle(theta + (v * dt / params.wheelbase) * math.tan(steering))
    return AckermannState(x, y, theta)


def sample_constant_steer_motion(
    state: AckermannState,
    steering: float,
    direction: int,
    step_length: float,
    params: AckermannParams,
    step: float = 0.05,
    footprint=None,
):
    """Sample poses (and optional bounding boxes) along a constant-steer arc.

    Returns (states, boxes) where boxes is a list of oriented-rectangle corner tuples
    if a footprint is provided, otherwise an empty list.
    """
    steering = max(-params.max_steer, min(params.max_steer, steering))
    ds_total = step_length * float(direction)
    k = math.tan(steering) / params.wheelbase
    n = max(1, int(math.ceil(abs(ds_total) / max(step, 1e-6))))

    states = []
    boxes = []
    for i in range(n + 1):
        s = ds_total * (i / n)
        if abs(k) < 1e-8:
            x = state.x + s * math.cos(state.theta)
            y = state.y + s * math.sin(state.theta)
            theta = wrap_angle(state.theta + s * k)
        else:
            dtheta = s * k
            x = state.x + (math.sin(state.theta + dtheta) - math.sin(state.theta)) / k
            y = state.y - (math.cos(state.theta + dtheta) - math.cos(state.theta)) / k
            theta = wrap_angle(state.theta + dtheta)
        pose = AckermannState(x, y, theta)
        states.append(pose)
        if footprint is not None:
            boxes.append(footprint.corners(x, y, theta))
    return states, boxes
