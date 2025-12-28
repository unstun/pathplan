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


def _footprint_offsets_for_heading(
    footprint: OrientedBoxFootprint, resolution: float, theta: float, padding: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Compute grid offsets whose cell centers fall inside the footprint at a given heading.
    Offsets are returned as integer (dx, dy) in grid cells relative to the robot center cell.
    """
    hl = footprint.half_length + padding
    hw = footprint.half_width + padding
    radius = math.hypot(hl, hw)
    cells = int(math.ceil(radius / resolution)) + 1  # include boundary cells
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    offsets: List[Tuple[int, int]] = []
    for gx in range(-cells, cells + 1):
        for gy in range(-cells, cells + 1):
            wx = gx * resolution
            wy = gy * resolution
            # rotate world offset into robot frame
            rx = cos_t * wx + sin_t * wy
            ry = -sin_t * wx + cos_t * wy
            if abs(rx) <= hl and abs(ry) <= hw:
                offsets.append((gx, gy))
    return offsets


class GridFootprintChecker:
    """
    Grid-based collision checker that precomputes footprint cell offsets per heading bin.
    """

    def __init__(self, grid_map, footprint: OrientedBoxFootprint, theta_bins: int, padding: float = 0.0):
        self.map = grid_map
        self.theta_bins = theta_bins
        self.offsets: List[List[Tuple[int, int]]] = []
        for i in range(theta_bins):
            theta = (2.0 * math.pi * i) / theta_bins
            self.offsets.append(_footprint_offsets_for_heading(footprint, grid_map.resolution, theta, padding))

    def _theta_index(self, theta: float) -> int:
        return int(round(((theta % (2 * math.pi)) / (2 * math.pi)) * self.theta_bins)) % self.theta_bins

    def _collides_grid(self, gx: int, gy: int, theta_idx: int) -> bool:
        h, w = self.map.data.shape
        for dx, dy in self.offsets[theta_idx]:
            cx = gx + dx
            cy = gy + dy
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                return True
            if self.map.data[cy, cx]:
                return True
        return False

    def collides_pose(self, x: float, y: float, theta: float) -> bool:
        gx, gy = self.map.world_to_grid(x, y)
        theta_idx = self._theta_index(theta)
        return self._collides_grid(gx, gy, theta_idx)

    def collides_path(self, poses: Iterable[Tuple[float, float, float]]) -> bool:
        for pose in poses:
            if hasattr(pose, "x"):
                x, y, theta = pose.x, pose.y, pose.theta
            else:
                x, y, theta = pose
            if self.collides_pose(x, y, theta):
                return True
        return False

    def motion_collides(
        self, start: Tuple[float, float, float], end: Tuple[float, float, float], step: float
    ) -> bool:
        if self.collides_pose(start[0], start[1], start[2]):
            return True
        for pose in interpolate_poses(start, end, step):
            if self.collides_pose(pose[0], pose[1], pose[2]):
                return True
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
