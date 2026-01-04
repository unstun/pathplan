import math
from typing import Tuple


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return a


def heading_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b."""
    return wrap_angle(a - b)


def euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def default_collision_step(resolution: float, preferred: float = 0.1, min_step: float = 0.05, max_step: float = 0.2) -> float:
    """
    Shared default for collision sampling step.
    Uses a preferred physical step (meters) with clamps so comparisons across
    map resolutions keep the same sampling distance.
    """
    step = preferred
    if step <= 0:
        step = min_step if min_step > 0 else max_step
    return max(min_step, min(max_step, step))


def default_min_motion_step(resolution: float, preferred: float = 0.1, min_step: float = 0.05, max_step: float = 0.2) -> float:
    """
    Shared default for the smallest forward progress step when halving motion.
    Kept equal to the collision sampling step for consistency.
    """
    return default_collision_step(resolution, preferred=preferred, min_step=min_step, max_step=max_step)
