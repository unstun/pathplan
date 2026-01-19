import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

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
        theta_bins: int = 72,
    ) -> bool:
        """
        Collision test using the shared GridFootprintChecker.

        `sample_step` is retained for backward compatibility but the checker
        always relies on the grid-aligned footprint offsets.
        """
        checker = GridFootprintChecker(grid_map, self, theta_bins)
        return checker.collides_pose(x, y, theta)


@dataclass
class TwoCircleFootprint:
    """
    Conservative two-circle approximation of an oriented box footprint.

    The circles are centered on the robot x-axis at (+/-center_offset, 0)
    in the robot frame.
    """

    radius: float
    center_offset: float

    def __post_init__(self) -> None:
        self.radius = float(self.radius)
        self.center_offset = float(self.center_offset)
        if not math.isfinite(self.radius) or not math.isfinite(self.center_offset):
            raise ValueError("TwoCircleFootprint parameters must be finite.")
        if self.radius <= 0.0:
            raise ValueError("TwoCircleFootprint.radius must be > 0.")
        if self.center_offset < 0.0:
            raise ValueError("TwoCircleFootprint.center_offset must be >= 0.")

    @classmethod
    def from_box(cls, length: float, width: float) -> "TwoCircleFootprint":
        """
        Build a conservative two-circle cover for an oriented rectangle.

        The resulting union of circles contains the full rectangle in the robot frame.
        """
        length = float(length)
        width = float(width)
        if not math.isfinite(length) or not math.isfinite(width):
            raise ValueError("Box length/width must be finite.")
        if length <= 0.0 or width <= 0.0:
            raise ValueError("Box length/width must be > 0.")

        half_length = length / 2.0
        half_width = width / 2.0
        center_offset = half_length / 2.0
        radius = math.hypot(center_offset, half_width)
        return cls(radius=radius, center_offset=center_offset)

    @property
    def length(self) -> float:
        return 2.0 * (self.center_offset + self.radius)

    @property
    def width(self) -> float:
        return 2.0 * self.radius

    def corners(self, x: float, y: float, theta: float) -> List[Tuple[float, float]]:
        return OrientedBoxFootprint(self.length, self.width).corners(x, y, theta)

    def point_inside(self, px: float, py: float, x: float, y: float, theta: float) -> bool:
        """Check if world point lies inside the union of the two circles."""
        dx = px - x
        dy = py - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rx = cos_t * dx + sin_t * dy
        ry = -sin_t * dx + cos_t * dy
        r2 = self.radius * self.radius + 1e-12
        d = self.center_offset
        dx_front = rx - d
        if dx_front * dx_front + ry * ry <= r2:
            return True
        dx_rear = rx + d
        return dx_rear * dx_rear + ry * ry <= r2

    def circle_centers(self, x: float, y: float, theta: float) -> List[Tuple[float, float]]:
        """Return world-frame centers of the front and rear circles."""
        d = self.center_offset
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        front = (x + cos_t * d, y + sin_t * d)
        rear = (x - cos_t * d, y - sin_t * d)
        return [front, rear]

    def collides(
        self,
        grid_map,
        x: float,
        y: float,
        theta: float,
        sample_step: float = 0.05,
        theta_bins: int = 72,
    ) -> bool:
        """
        Collision test using the shared GridFootprintChecker.

        `sample_step` is retained for backward compatibility but the checker
        always relies on the grid-aligned footprint offsets.
        """
        checker = GridFootprintChecker(grid_map, self, theta_bins)
        return checker.collides_pose(x, y, theta)


Footprint = Union[OrientedBoxFootprint, TwoCircleFootprint]


def _disk_offsets(radius: float, resolution: float) -> List[Tuple[int, int]]:
    """
    Return integer grid offsets (dx, dy) whose cell squares intersect a disk.

    Used for fast grid-based circle collision checks.
    """
    radius = float(radius)
    resolution = float(resolution)
    if not math.isfinite(radius) or not math.isfinite(resolution):
        raise ValueError("Disk radius/resolution must be finite.")
    if resolution <= 0.0:
        raise ValueError("Grid resolution must be > 0.")
    if radius <= 0.0:
        return [(0, 0)]

    half = resolution * 0.5
    cells = int(math.ceil((radius + half) / resolution)) + 1
    r2 = radius * radius + 1e-12
    offsets: List[Tuple[int, int]] = []
    for dx in range(-cells, cells + 1):
        wx = dx * resolution
        for dy in range(-cells, cells + 1):
            wy = dy * resolution
            # Circle-square intersection: distance from circle center to the square
            # (axis-aligned, centered at (wx, wy), side=resolution) must be <= radius.
            dx_out = max(abs(wx) - half, 0.0)
            dy_out = max(abs(wy) - half, 0.0)
            if dx_out * dx_out + dy_out * dy_out <= r2:
                offsets.append((dx, dy))
    return offsets


def _footprint_offsets_for_heading(
    footprint: Footprint, resolution: float, theta: float, padding: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Compute grid offsets whose cell centers fall inside the footprint at a given heading.
    Offsets are returned as integer (dx, dy) in grid cells relative to the robot center cell.
    `padding` expands the footprint equally on all sides (meters).
    """
    if isinstance(footprint, OrientedBoxFootprint):
        hl = footprint.half_length + padding
        hw = footprint.half_width + padding
        radius = math.hypot(hl, hw)
    elif isinstance(footprint, TwoCircleFootprint):
        hl = None
        hw = None
        radius = footprint.center_offset + (footprint.radius + padding)
        circle_radius_sq = (footprint.radius + padding) ** 2 + 1e-12
        circle_center = footprint.center_offset
    else:
        raise TypeError(f"Unsupported footprint type: {type(footprint)!r}")

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
            if hl is not None and hw is not None:
                if abs(rx) <= hl and abs(ry) <= hw:
                    offsets.append((gx, gy))
                continue

            dx_front = rx - circle_center
            if dx_front * dx_front + ry * ry <= circle_radius_sq:
                offsets.append((gx, gy))
                continue
            dx_rear = rx + circle_center
            if dx_rear * dx_rear + ry * ry <= circle_radius_sq:
                offsets.append((gx, gy))
    return offsets


class GridFootprintChecker:
    """
    Grid-based collision checker that precomputes footprint cell offsets per heading bin.
    """

    def __init__(
        self, grid_map, footprint: Footprint, theta_bins: int, padding: Optional[float] = None
    ):
        self.map = grid_map
        self.footprint = footprint
        self.theta_bins = max(1, int(theta_bins))
        # Default padding is 0 so collision is determined by the provided grid map
        # (users can still pass a positive padding as a safety buffer).
        self.padding = 0.0 if padding is None else float(padding)
        # Cache a boolean view of the occupancy grid for fast numpy indexing.
        self._occ = np.asarray(grid_map.data, dtype=bool)
        self._h, self._w = self._occ.shape
        ox, oy = grid_map.origin
        res = grid_map.resolution
        self._x_centers = ox + np.arange(self._w, dtype=np.float64) * res
        self._y_centers = oy + np.arange(self._h, dtype=np.float64) * res
        self._two_circle = isinstance(footprint, TwoCircleFootprint)

        self.offsets: List[np.ndarray] = []
        self.offset_bounds: List[Tuple[int, int, int, int]] = []

        self._circle_center_offset = 0.0
        self._circle_radius = 0.0
        self._circle_radius_sq = 0.0
        self._circle_reach = 0.0

        if self._two_circle:
            self._circle_center_offset = float(footprint.center_offset)
            self._circle_radius = max(float(footprint.radius) + self.padding, 0.0)
            self._circle_radius_sq = self._circle_radius * self._circle_radius + 1e-12
            self._circle_reach = self._circle_radius + grid_map.resolution * 0.5
        else:
            for i in range(self.theta_bins):
                theta = (2.0 * math.pi * i) / self.theta_bins
                offsets = np.asarray(
                    _footprint_offsets_for_heading(footprint, grid_map.resolution, theta, self.padding),
                    dtype=np.int32,
                )
                self.offsets.append(offsets)
                if offsets.size:
                    dx_min = int(offsets[:, 0].min())
                    dx_max = int(offsets[:, 0].max())
                    dy_min = int(offsets[:, 1].min())
                    dy_max = int(offsets[:, 1].max())
                else:
                    dx_min = dx_max = dy_min = dy_max = 0
                self.offset_bounds.append((dx_min, dx_max, dy_min, dy_max))

    def _theta_index(self, theta: float) -> int:
        return int(round(((theta % (2 * math.pi)) / (2 * math.pi)) * self.theta_bins)) % self.theta_bins

    def _collides_grid(self, gx: int, gy: int, theta_idx: int) -> bool:
        dx_min, dx_max, dy_min, dy_max = self.offset_bounds[theta_idx]
        # Early reject if the footprint would extend outside the known map.
        if gx + dx_min < 0 or gx + dx_max >= self._w or gy + dy_min < 0 or gy + dy_max >= self._h:
            return True

        offsets = self.offsets[theta_idx]
        if offsets.size == 0:
            return False
        cx = gx + offsets[:, 0]
        cy = gy + offsets[:, 1]
        return bool(self._occ[cy, cx].any())

    def _collides_circle_world(self, x: float, y: float) -> bool:
        """
        Exact circle vs occupancy-grid collision test.

        A collision occurs when the circle intersects any occupied grid cell square.
        """
        reach = self._circle_reach
        if reach <= 0.0:
            return False

        res = self.map.resolution
        ox, oy = self.map.origin
        min_gx = int(math.ceil((x - reach - ox) / res))
        max_gx = int(math.floor((x + reach - ox) / res))
        min_gy = int(math.ceil((y - reach - oy) / res))
        max_gy = int(math.floor((y + reach - oy) / res))

        # Early reject if the circle would extend outside the known map.
        if min_gx < 0 or max_gx >= self._w or min_gy < 0 or max_gy >= self._h:
            return True

        sub_occ = self._occ[min_gy : max_gy + 1, min_gx : max_gx + 1]
        if not bool(sub_occ.any()):
            return False

        half = res * 0.5
        xs = self._x_centers[min_gx : max_gx + 1]
        ys = self._y_centers[min_gy : max_gy + 1]
        dx = np.maximum(np.abs(xs - x) - half, 0.0)
        dy = np.maximum(np.abs(ys - y) - half, 0.0)
        dist2 = dx[None, :] * dx[None, :] + dy[:, None] * dy[:, None]
        return bool((sub_occ & (dist2 <= self._circle_radius_sq)).any())

    def _collides_two_circle_pose(self, x: float, y: float, theta: float) -> bool:
        d = self._circle_center_offset
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        front_x = x + cos_t * d
        front_y = y + sin_t * d
        rear_x = x - cos_t * d
        rear_y = y - sin_t * d
        if self._collides_circle_world(front_x, front_y):
            return True
        return self._collides_circle_world(rear_x, rear_y)

    def collides_pose(self, x: float, y: float, theta: float) -> bool:
        if self._two_circle:
            return self._collides_two_circle_pose(x, y, theta)
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
    footprint: Footprint,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step: float = 0.05,
    theta_bins: int = 72,
) -> bool:
    """Check collision along motion using the unified GridFootprintChecker."""
    checker = GridFootprintChecker(grid_map, footprint, theta_bins)
    return checker.motion_collides(start, end, step=step)


def path_collides(
    grid_map,
    footprint: Footprint,
    poses: Iterable[Tuple[float, float, float]],
    sample_step: float = 0.05,
    theta_bins: int = 72,
) -> bool:
    """Check collision for a sequence of poses using the unified GridFootprintChecker."""
    checker = GridFootprintChecker(grid_map, footprint, theta_bins)
    return checker.collides_path(poses)
