import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReedsSheppPath:
    """
    Reeds–Shepp path consisting of L/R/S segments.

    `segment_lengths` are signed arc lengths (meters). A negative value means the
    segment is traversed in reverse.
    """

    segment_types: Tuple[str, ...]
    segment_lengths: Tuple[float, ...]
    total_length: float


def _mod2pi(angle: float) -> float:
    """
    Wrap to [-pi, pi] (matching the convention used by many RS implementations).

    Unlike `wrap_angle()` this keeps +pi as +pi (instead of mapping it to -pi),
    which matters for path family feasibility checks.
    """
    v = math.fmod(angle, 2.0 * math.pi)
    if v < -math.pi:
        v += 2.0 * math.pi
    elif v > math.pi:
        v -= 2.0 * math.pi
    return v


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _lsl(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]
    return False, [], []


def _lsr(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, v], ["L", "S", "R"]
    return False, [], []


def _lrl(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]
    return False, [], []


def _lrl2(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]
    return False, [], []


def _lrl3(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 4.0:
        u = math.acos(1.0 - (u1 * u1) * 0.125)
        a = math.asin(2.0 * math.sin(u) / u1)
        t = _mod2pi(-a + theta + math.pi / 2.0)
        v = _mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]
    return False, [], []


def _lrlr(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    # Solutions for (2 < u1 <= 4) are considered sub-optimal in the source paper.
    if u1 <= 2.0:
        a = math.acos((u1 + 2.0) * 0.25)
        t = _mod2pi(theta + a + math.pi / 2.0)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2.0 * u)
        if t >= 0.0 and u >= 0.0 and v >= 0.0:
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]
    return False, [], []


def _lrlr2(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    u2 = (20.0 - u1 * u1) / 16.0
    if 0.0 <= u2 <= 1.0:
        u = math.acos(u2)
        a = math.asin(2.0 * math.sin(u) / u1)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]
    return False, [], []


def _lrs_l(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        u = math.sqrt(u1 * u1 - 4.0) - 2.0
        a = math.atan2(2.0, math.sqrt(u1 * u1 - 4.0))
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi + math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "L"]
    return False, [], []


def _ls_r_l(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        u = math.sqrt(u1 * u1 - 4.0) - 2.0
        a = math.atan2(math.sqrt(u1 * u1 - 4.0), 2.0)
        t = _mod2pi(theta - a + math.pi / 2.0)
        v = _mod2pi(t - phi - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "R", "L"]
    return False, [], []


def _lrs_r(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        t = _mod2pi(theta + math.pi / 2.0)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "R"]
    return False, [], []


def _lsl_r(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "L", "R"]
    return False, [], []


def _lrs_lr(x: float, y: float, phi: float) -> Tuple[bool, List[float], List[str]]:
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 4.0:
        u = math.sqrt(u1 * u1 - 4.0) - 4.0
        a = math.atan2(2.0, math.sqrt(u1 * u1 - 4.0))
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -math.pi / 2.0, v], ["L", "R", "S", "L", "R"]
    return False, [], []


def _timeflip(lengths: Sequence[float]) -> List[float]:
    return [-float(v) for v in lengths]


def _reflect(types: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in types:
        if t == "L":
            out.append("R")
        elif t == "R":
            out.append("L")
        else:
            out.append("S")
    return out


_PATH_FNS: Tuple[Callable[[float, float, float], Tuple[bool, List[float], List[str]]], ...] = (
    _lsl,
    _lsr,
    _lrl,
    _lrl2,
    _lrl3,
    _lrlr,
    _lrlr2,
    _lrs_l,
    _lrs_r,
    _ls_r_l,
    _lsl_r,
    _lrs_lr,
)


def reeds_shepp_shortest_path(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    turning_radius: float,
) -> Optional[ReedsSheppPath]:
    """
    Compute the shortest Reeds–Shepp path between `start` and `goal`.

    Returns None when no candidate families apply (should be rare).
    """
    turning_radius = float(turning_radius)
    if not math.isfinite(turning_radius) or turning_radius <= 0.0:
        raise ValueError("turning_radius must be finite and > 0")

    max_curvature = 1.0 / turning_radius
    sx, sy, syaw = start
    gx, gy, gyaw = goal

    dx = gx - sx
    dy = gy - sy
    c = math.cos(syaw)
    s = math.sin(syaw)

    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    phi = _mod2pi(gyaw - syaw)

    best: Optional[ReedsSheppPath] = None
    best_norm = float("inf")

    for fn in _PATH_FNS:
        # original
        ok, lengths, types = fn(x, y, phi)
        if ok:
            total_norm = sum(abs(v) for v in lengths)
            if total_norm < best_norm:
                best_norm = total_norm
                best = ReedsSheppPath(
                    segment_types=tuple(types),
                    segment_lengths=tuple(v / max_curvature for v in lengths),
                    total_length=total_norm / max_curvature,
                )

        # timeflip
        ok, lengths, types = fn(-x, y, -phi)
        if ok:
            lengths = _timeflip(lengths)
            total_norm = sum(abs(v) for v in lengths)
            if total_norm < best_norm:
                best_norm = total_norm
                best = ReedsSheppPath(
                    segment_types=tuple(types),
                    segment_lengths=tuple(v / max_curvature for v in lengths),
                    total_length=total_norm / max_curvature,
                )

        # reflect
        ok, lengths, types = fn(x, -y, -phi)
        if ok:
            types = _reflect(types)
            total_norm = sum(abs(v) for v in lengths)
            if total_norm < best_norm:
                best_norm = total_norm
                best = ReedsSheppPath(
                    segment_types=tuple(types),
                    segment_lengths=tuple(v / max_curvature for v in lengths),
                    total_length=total_norm / max_curvature,
                )

        # timeflip + reflect
        ok, lengths, types = fn(-x, -y, phi)
        if ok:
            lengths = _timeflip(lengths)
            types = _reflect(types)
            total_norm = sum(abs(v) for v in lengths)
            if total_norm < best_norm:
                best_norm = total_norm
                best = ReedsSheppPath(
                    segment_types=tuple(types),
                    segment_lengths=tuple(v / max_curvature for v in lengths),
                    total_length=total_norm / max_curvature,
                )

    return best


def path_segments(path: ReedsSheppPath) -> Iterable[Tuple[str, float]]:
    for t, length in zip(path.segment_types, path.segment_lengths):
        yield t, float(length)
