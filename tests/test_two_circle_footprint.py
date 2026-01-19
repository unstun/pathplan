import math
import unittest

import numpy as np

from pathplan import AckermannParams, RRTStarPlanner
from pathplan.geometry import GridFootprintChecker, TwoCircleFootprint
from pathplan.map_utils import GridMap


class TestTwoCircleFootprint(unittest.TestCase):
    def test_from_box_covers_box_points(self):
        length = 0.924
        width = 0.740
        footprint = TwoCircleFootprint.from_box(length, width)

        corners = [
            (+length / 2.0, +width / 2.0),
            (+length / 2.0, -width / 2.0),
            (-length / 2.0, +width / 2.0),
            (-length / 2.0, -width / 2.0),
        ]
        for px, py in corners:
            self.assertTrue(footprint.point_inside(px, py, 0.0, 0.0, 0.0))

        rng = np.random.default_rng(0)
        for _ in range(2000):
            px = rng.uniform(-length / 2.0, length / 2.0)
            py = rng.uniform(-width / 2.0, width / 2.0)
            self.assertTrue(footprint.point_inside(px, py, 0.0, 0.0, 0.0))

    def test_grid_checker_supports_two_circle(self):
        grid = np.zeros((21, 21), dtype=np.uint8)
        grid[10, 11] = 1
        grid_map = GridMap(grid, resolution=1.0, origin=(0.0, 0.0))

        footprint = TwoCircleFootprint(radius=1.0, center_offset=1.0)
        checker = GridFootprintChecker(grid_map, footprint, theta_bins=16, padding=0.0)
        self.assertTrue(checker.collides_pose(10.0, 10.0, 0.0))

        clear_grid = np.zeros((21, 21), dtype=np.uint8)
        clear_map = GridMap(clear_grid, resolution=1.0, origin=(0.0, 0.0))
        clear_checker = GridFootprintChecker(clear_map, footprint, theta_bins=16, padding=0.0)
        self.assertFalse(clear_checker.collides_pose(10.0, 10.0, 0.0))

    def test_rrt_star_pose_collision_works_with_two_circle(self):
        grid = np.zeros((31, 31), dtype=np.uint8)
        grid[15, 16] = 1
        grid_map = GridMap(grid, resolution=1.0, origin=(0.0, 0.0))

        footprint = TwoCircleFootprint(radius=0.6, center_offset=1.0)
        planner = RRTStarPlanner(
            grid_map,
            footprint,
            AckermannParams(),
            collision_step=0.5,
            theta_bins=64,
            collision_padding=0.0,
        )

        self.assertTrue(planner._pose_collides(15.0, 15.0, 0.0))
        self.assertFalse(planner._pose_collides(15.0, 15.0, math.pi / 2.0))

    def test_grid_checker_circle_intersects_cell_square(self):
        grid = np.zeros((31, 31), dtype=np.uint8)
        grid[15, 16] = 1
        grid_map = GridMap(grid, resolution=1.0, origin=(0.0, 0.0))

        # Circle center near the cell boundary: it intersects the occupied cell square
        # but would be missed by a naive rounding-to-cell-center implementation.
        footprint = TwoCircleFootprint(radius=0.05, center_offset=0.0)
        checker = GridFootprintChecker(grid_map, footprint, theta_bins=16, padding=0.0)
        self.assertTrue(checker.collides_pose(15.49, 15.0, 0.0))


if __name__ == "__main__":
    unittest.main()
