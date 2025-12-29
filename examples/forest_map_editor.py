"""
Interactive forest map viewer/editor.

Run:
    python -m examples.forest_map_editor

Controls:
- Left click: paint obstacles (set occupied to 1).
- Right click: erase obstacles (set free to 0).
- [ / ]: shrink / grow brush size (in grid cells).
- r: reset to original generated map.
- s: save current grid to disk (.npy) and a snapshot (.png).
- q or Esc: quit.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from pathplan import AckermannState, GridMap
from .forest_scene import (
    DEFAULT_GOAL,
    DEFAULT_START,
    FOREST_MAP_KWARGS,
    compute_start_goal,
    make_forest_map,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def build_default_map() -> Tuple[GridMap, AckermannState, AckermannState]:
    map_kwargs = dict(FOREST_MAP_KWARGS)
    start, goal = compute_start_goal(map_kwargs)
    map_kwargs["keep_clear"] = [(start.x, start.y), (goal.x, goal.y)]
    return make_forest_map(**map_kwargs), start, goal


def load_grid_map(input_path: Path, resolution: float) -> GridMap:
    data = np.load(input_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D occupancy grid in {input_path}, got shape {data.shape}")
    return GridMap(data.astype(np.uint8), resolution, origin=(0.0, 0.0))


class ForestMapEditor:
    def __init__(self, grid_map: GridMap, start: AckermannState, goal: AckermannState, save_path: Path, brush: int = 2):
        self.grid_map = grid_map
        self.original = grid_map.data.copy()
        self.start = start
        self.goal = goal
        self.save_path = save_path
        self.brush = max(0, int(brush))

        self.fig, self.ax = plt.subplots(figsize=(20, 12))
        self.im = None
        self.status_text = None
        self._init_plot()
        self._connect_events()

    def _in_bounds_world(self, x: float, y: float) -> bool:
        gx, gy = self.grid_map.world_to_grid(x, y)
        return self.grid_map.in_bounds(gx, gy)

    def _init_plot(self):
        h, w = self.grid_map.data.shape
        extent = [
            self.grid_map.origin[0],
            self.grid_map.origin[0] + w * self.grid_map.resolution,
            self.grid_map.origin[1],
            self.grid_map.origin[1] + h * self.grid_map.resolution,
        ]
        self.im = self.ax.imshow(
            self.grid_map.data,
            cmap="gray_r",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
        )
        if self._in_bounds_world(self.start.x, self.start.y):
            self.ax.scatter(self.start.x, self.start.y, c="green", marker="*", s=120, label="start")
        if self._in_bounds_world(self.goal.x, self.goal.y):
            self.ax.scatter(self.goal.x, self.goal.y, c="red", marker="*", s=120, label="goal")
        self.ax.set_aspect("equal")
        self.ax.set_title("Forest map editor")
        self.ax.legend(loc="upper right")
        self.status_text = self.fig.text(
            0.01,
            0.02,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        self._update_status("Ready")

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_status(self, message: str = ""):
        brush_m = self.brush * self.grid_map.resolution
        self.status_text.set_text(
            f"{message} | brush={self.brush} cells (~{brush_m:.2f} m) | "
            "Left add, Right erase, [ ] change brush, r reset, s save, q quit"
        )
        self.fig.canvas.draw_idle()

    def _apply_brush(self, gx: int, gy: int, value: int):
        half = self.brush
        x_min = max(0, gx - half)
        x_max = min(self.grid_map.data.shape[1] - 1, gx + half)
        y_min = max(0, gy - half)
        y_max = min(self.grid_map.data.shape[0] - 1, gy + half)
        self.grid_map.data[y_min : y_max + 1, x_min : x_max + 1] = value
        self.im.set_data(self.grid_map.data)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        gx, gy = self.grid_map.world_to_grid(event.xdata, event.ydata)
        if not self.grid_map.in_bounds(gx, gy):
            self._update_status("Click ignored (out of bounds)")
            return
        if event.button == 1:
            self._apply_brush(gx, gy, 1)
            self._update_status("Painted obstacle")
        elif event.button == 3:
            self._apply_brush(gx, gy, 0)
            self._update_status("Erased obstacle")

    def _on_key(self, event):
        if event.key in ("[", "{"):
            self.brush = max(0, self.brush - 1)
            self._update_status("Brush size changed")
        elif event.key in ("]", "}"):
            self.brush += 1
            self._update_status("Brush size changed")
        elif event.key in ("r", "R"):
            self.grid_map.data[:, :] = self.original
            self.im.set_data(self.grid_map.data)
            self._update_status("Reset to original map")
        elif event.key in ("s", "S"):
            self._save()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.save_path, self.grid_map.data.astype(np.uint8))
        png_path = self.save_path.with_suffix(".png")
        self.fig.savefig(png_path, dpi=300, bbox_inches="tight")
        self._update_status(f"Saved {self.save_path.name} and {png_path.name}")
        print(f"Saved map grid to {self.save_path}")
        print(f"Saved snapshot to {png_path}")

    def show(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="View and edit the forest occupancy map.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional .npy file to start from (2D occupancy grid). If omitted, generates the default forest map.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=FOREST_MAP_KWARGS["resolution"],
        help="Grid resolution in meters per cell when loading from --input.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "forest_map_manual.npy",
        help="Output path for the edited map (.npy). A .png with the same stem is also written.",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=2,
        help="Brush half-size in grid cells (0 = single cell).",
    )
    return parser.parse_args()


def main():
    if plt is None:
        print("matplotlib is required for interactive editing. Install it with `pip install matplotlib`.")
        sys.exit(1)

    args = parse_args()
    if args.input is not None:
        grid_map = load_grid_map(args.input, args.resolution)
        w_m = grid_map.data.shape[1] * grid_map.resolution
        h_m = grid_map.data.shape[0] * grid_map.resolution
        map_kwargs = {
            "size": (w_m, h_m),
            "tree_radius": FOREST_MAP_KWARGS.get("tree_radius", 0.30),
            "clearance": FOREST_MAP_KWARGS.get("clearance", 1.8),
        }
        start, goal = compute_start_goal(map_kwargs)
    else:
        grid_map, start, goal = build_default_map()

    editor = ForestMapEditor(grid_map, start, goal, Path(args.save), brush=args.brush)
    editor.show()


if __name__ == "__main__":
    main()
