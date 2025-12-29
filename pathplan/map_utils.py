import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .common import clamp


@dataclass
class GridMap:
    """
    Simple occupancy grid.
    data: numpy array (H, W), 1/True for occupied, 0/False for free.
    resolution: meters per cell.
    origin: world coordinates of grid index (0,0) cell center.
    """

    data: np.ndarray
    resolution: float
    origin: Tuple[float, float] = (0.0, 0.0)

    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round((x - self.origin[0]) / self.resolution))
        gy = int(round((y - self.origin[1]) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        h, w = self.data.shape
        return 0 <= gx < w and 0 <= gy < h

    def is_occupied_index(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return True
        return bool(self.data[gy, gx])

    def is_occupied(self, x: float, y: float) -> bool:
        gx, gy = self.world_to_grid(x, y)
        return self.is_occupied_index(gx, gy)

    def occupancy_patch(
        self,
        x: float,
        y: float,
        theta: float,
        size_m: float = 8.0,
        cells: int = 64,
    ) -> np.ndarray:
        """
        Extract a local occupancy patch centered at (x,y) with robot-orientation alignment.
        Returns a (cells, cells) array in robot frame (forward = +x).
        Uses nearest-neighbor sampling to avoid extra deps.
        """
        half = size_m / 2.0
        if not hasattr(self, "_patch_cache"):
            self._patch_cache = {}
        cache_key = (size_m, cells)
        if cache_key in self._patch_cache:
            xs, ys = self._patch_cache[cache_key]
        else:
            lin = np.linspace(-half, half, cells)
            xs, ys = np.meshgrid(lin, lin, indexing="xy")
            self._patch_cache[cache_key] = (xs, ys)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        world_x = x + cos_t * xs - sin_t * ys
        world_y = y + sin_t * xs + cos_t * ys
        gx = np.rint((world_x - self.origin[0]) / self.resolution).astype(int)
        gy = np.rint((world_y - self.origin[1]) / self.resolution).astype(int)

        h, w = self.data.shape
        gx = np.clip(gx, 0, w - 1)
        gy = np.clip(gy, 0, h - 1)
        return self.data[gy, gx]

    def copy(self) -> "GridMap":
        return GridMap(self.data.copy(), self.resolution, self.origin)

    def inflate(self, margin: float) -> "GridMap":
        """
        Inflate occupied cells by margin (meters) to add safety buffer.
        Simple convolution-free dilation using a Manhattan disk.
        """
        cells = int(math.ceil(margin / self.resolution))
        if cells <= 0:
            return self.copy()
        padded = np.pad(self.data, cells, constant_values=1)
        h, w = self.data.shape
        inflated = np.zeros_like(self.data)
        for y in range(h):
            for x in range(w):
                sub = padded[y : y + 2 * cells + 1, x : x + 2 * cells + 1]
                inflated[y, x] = 1 if np.any(sub) else 0
        return GridMap(inflated, self.resolution, self.origin)

    def random_free_state(
        self, rng: np.random.Generator, yaw_range: Tuple[float, float] = (-math.pi, math.pi)
    ) -> Tuple[float, float, float]:
        """Sample a collision-free pose uniformly from free cells and random yaw."""
        if not hasattr(self, "_free_indices_cache"):
            free_indices = np.argwhere(self.data == 0)
            if len(free_indices) == 0:
                raise ValueError("Map has no free cells")
            self._free_indices_cache = free_indices
        free_indices = self._free_indices_cache
        idx = rng.integers(0, len(free_indices))
        gy, gx = free_indices[idx]
        x, y = self.grid_to_world(int(gx), int(gy))
        theta = rng.uniform(yaw_range[0], yaw_range[1])
        return x, y, theta
