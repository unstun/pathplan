"""
Centralized planner labels (abbreviated) used across examples.
Keep plot legends, CSVs, and prints aligned by importing these names
instead of hard-coding strings in each script.
"""

APF_NAME = "APF"
HYBRID_NAME = "Hybrid A*"
DQNHYBRID_NAME = "D-Hybrid A*"
RRT_NAME = "Informed RRT*"

# Stable ordering for legends/exports.
FORMAL_PLANNER_ORDER = [APF_NAME, HYBRID_NAME, DQNHYBRID_NAME, RRT_NAME]

PLANNER_COLOR_MAP = {
    APF_NAME: "#d81b60",  # bold red/magenta
    HYBRID_NAME: "#1f77b4",  # deep blue
    DQNHYBRID_NAME: "#2ca02c",  # strong green
    RRT_NAME: "#ff7f0e",  # vivid orange
}

# Allow mapping from class names or legacy labels to the formal label.
CLASS_NAME_TO_LABEL = {
    "APFPlanner": APF_NAME,
    "HybridAStarPlanner": HYBRID_NAME,
    "DQNHybridAStarPlanner": DQNHYBRID_NAME,
    "RRTStarPlanner": RRT_NAME,
}

LEGACY_LABEL_TO_FORMAL = {
    "Artificial Potential Field": APF_NAME,
    "APF": APF_NAME,
    "Hybrid A*": HYBRID_NAME,
    "Hybrid A* Lattice Search": HYBRID_NAME,
    "DQN Hybrid A*": DQNHYBRID_NAME,
    "DQN-Guided Hybrid A* Search": DQNHYBRID_NAME,
    "Informed RRT*": RRT_NAME,
    "RRT*": RRT_NAME,
    "Kinodynamic RRT*": RRT_NAME,
}


def formal_planner_name(obj_or_label) -> str:
    """
    Return the formal, human-friendly planner name for a given class, instance, or label string.
    Falls back to str(obj_or_label) when no mapping exists.
    """

    if obj_or_label is None:
        return ""

    if isinstance(obj_or_label, str):
        return LEGACY_LABEL_TO_FORMAL.get(obj_or_label, obj_or_label)

    if isinstance(obj_or_label, type):
        key = obj_or_label.__name__
    else:
        key = obj_or_label.__class__.__name__

    return CLASS_NAME_TO_LABEL.get(key, str(obj_or_label))
