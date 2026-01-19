import heapq
import math
from typing import Tuple

import numpy as np


def dijkstra_2d_cost_to_go(
    occupancy: np.ndarray,
    goal: Tuple[int, int],
    resolution: float,
    allow_diagonal: bool = True,
) -> np.ndarray:
    """
    Compute a holonomic (x,y) cost-to-go grid using Dijkstra on an occupancy grid.

    - `occupancy` is an (H, W) array with nonzero meaning occupied.
    - `goal` is (gx, gy) in grid coordinates.
    - Returned array is (H, W) float64 distances in meters (inf for unreachable).

    This is used as the "unconstrained" heuristic in thesis-style Hybrid A*.
    """
    occ = np.asarray(occupancy, dtype=bool)
    h, w = occ.shape
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < w and 0 <= gy < h):
        raise ValueError("Goal is out of bounds for the occupancy grid.")

    dist = np.full((h, w), float("inf"), dtype=np.float64)
    if occ[gy, gx]:
        return dist

    dist[gy, gx] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, gx, gy)]

    res = float(resolution)
    if not math.isfinite(res) or res <= 0.0:
        raise ValueError("resolution must be finite and > 0")

    if allow_diagonal:
        diag = res * math.sqrt(2.0)
        neighbors = (
            (1, 0, res),
            (-1, 0, res),
            (0, 1, res),
            (0, -1, res),
            (1, 1, diag),
            (1, -1, diag),
            (-1, 1, diag),
            (-1, -1, diag),
        )
    else:
        neighbors = ((1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res))

    while heap:
        g, x, y = heapq.heappop(heap)
        if g != dist[y, x]:
            continue
        for dx, dy, step_cost in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if occ[ny, nx]:
                continue
            ng = g + step_cost
            if ng < dist[ny, nx]:
                dist[ny, nx] = ng
                heapq.heappush(heap, (ng, nx, ny))

    return dist

