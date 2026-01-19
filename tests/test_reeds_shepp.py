import math

from pathplan.hybrid_a_star.reeds_shepp import reeds_shepp_shortest_path


def test_reeds_shepp_straight_line():
    path = reeds_shepp_shortest_path((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), turning_radius=1.0)
    assert path is not None
    assert abs(path.total_length - 2.0) < 1e-6
    assert len(path.segment_types) == len(path.segment_lengths)


def test_reeds_shepp_in_place_half_turn_has_pi_length():
    path = reeds_shepp_shortest_path((0.0, 0.0, 0.0), (0.0, 0.0, math.pi), turning_radius=1.0)
    assert path is not None
    assert abs(path.total_length - math.pi) < 1e-6
