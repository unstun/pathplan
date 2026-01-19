"""
Centralized planner labels (abbreviated) used across examples.
Keep plot legends, CSVs, and prints aligned by importing these names
instead of hard-coding strings in each script.
"""

HYBRID_NAME = "Hybrid A*"
RRT_NAME = "SS-RRT*"

# Stable ordering for legends/exports.
FORMAL_PLANNER_ORDER = [HYBRID_NAME, RRT_NAME]

PLANNER_COLOR_MAP = {
    HYBRID_NAME: "#1f77b4",  # deep blue
    RRT_NAME: "#ff7f0e",  # vivid orange
}

# Allow mapping from class names or legacy labels to the formal label.
CLASS_NAME_TO_LABEL = {
    "HybridAStarPlanner": HYBRID_NAME,
    "RRTStarPlanner": RRT_NAME,
}

LEGACY_LABEL_TO_FORMAL = {
    "Hybrid A*": HYBRID_NAME,
    "Hybrid A* Lattice Search": HYBRID_NAME,
    "SS-RRT*": RRT_NAME,
    "Spline-based RRT*": RRT_NAME,
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
