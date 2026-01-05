"""
Path planning scaffold for Ackermann robots with strict turning radius and oriented footprint.
Exports planners (abbreviated names in parentheses):
- HybridAStarPlanner (Hybrid A*)
- DQNHybridAStarPlanner (D-Hybrid A* / DQN-guided Hybrid A*)
- RRTStarPlanner (Informed RRT*)
- APFPlanner (APF)
"""

from .hybrid_a_star import HybridAStarPlanner
from .dqn_hybrid_a_star import DQNHybridAStarPlanner
from .rrt_star import RRTStarPlanner
from .apf import APFPlanner
from .robot import AckermannParams, AckermannState
from .geometry import OrientedBoxFootprint
from .map_utils import GridMap
from .dqn_models import TorchDQNGuidance
from .postprocess import feng_optimize_path, stomp_optimize_path

__all__ = [
    "HybridAStarPlanner",
    "DQNHybridAStarPlanner",
    "RRTStarPlanner",
    "APFPlanner",
    "AckermannParams",
    "AckermannState",
    "OrientedBoxFootprint",
    "GridMap",
    "TorchDQNGuidance",
    "feng_optimize_path",
    "stomp_optimize_path",
]
