import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .common import wrap_angle


@dataclass
class OrientedBoxFootprint:
    length: float
    width: float

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def corners(self, x: float, y: float, theta: float) -> List[Tuple[float, float]]:
        """Return world-frame rectangle corners starting from front-left and CCW."""
        hl, hw = self.half_length, self.half_width
        base = [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        pts = []
        for bx, by in base:
            wx = x + cos_t * bx - sin_t * by
            wy = y + sin_t * bx + cos_t * by
            pts.append((wx, wy))
        return pts

    def point_inside(self, px: float, py: float, x: float, y: float, theta: float) -> bool:
        """Check if world point lies inside oriented rectangle."""
        dx = px - x
        dy = py - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rx = cos_t * dx + sin_t * dy
        ry = -sin_t * dx + cos_t * dy
        return abs(rx) <= self.half_length and abs(ry) <= self.half_width

    def collides(
        self,
        grid_map,
        x: float,
        y: float,
        theta: float,
        sample_step: float = 0.05,
    ) -> bool:
        """Collision test by sampling grid cell centers within bounding square."""
        radius = math.hypot(self.half_length, self.half_width)
        step = max(sample_step, grid_map.resolution * 0.5)
        minx = x - radius
        maxx = x + radius
        miny = y - radius
        maxy = y + radius
        px = minx
        while px <= maxx + 1e-9:
            py = miny
            while py <= maxy + 1e-9:
                if self.point_inside(px, py, x, y, theta):
                    if grid_map.is_occupied(px, py):
                        return True
                py += step
            px += step
        return False


def interpolate_poses(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step: float,
) -> Iterable[Tuple[float, float, float]]:
    """Linear interpolation in SE(2) (heading via shortest wrap)."""
    x0, y0, t0 = start
    x1, y1, t1 = end
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(math.ceil(dist / step)))
    for i in range(1, steps + 1):
        s = i / steps
        x = x0 + (x1 - x0) * s
        y = y0 + (y1 - y0) * s
        dt = wrap_angle(t1 - t0)
        theta = wrap_angle(t0 + dt * s)
        yield x, y, theta


def motion_collides(
    grid_map,
    footprint: OrientedBoxFootprint,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step: float = 0.05,
) -> bool:
    """Check collision along motion by sampling intermediate poses."""
    if footprint.collides(grid_map, start[0], start[1], start[2], sample_step=step):
        return True
    for pose in interpolate_poses(start, end, step):
        if footprint.collides(grid_map, pose[0], pose[1], pose[2], sample_step=step):
            return True
    return False


def path_collides(
    grid_map,
    footprint: OrientedBoxFootprint,
    poses: Iterable[Tuple[float, float, float]],
    sample_step: float = 0.05,
) -> bool:
    """Check collision for a sequence of poses (e.g., an arc trace)."""
    for x, y, theta in poses:
        if footprint.collides(grid_map, x, y, theta, sample_step=sample_step):
            return True
    return False
