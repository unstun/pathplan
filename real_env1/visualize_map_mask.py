"""
Extract and visualize points from a LAS file that fall inside the occupied (black)
cells of a ROS-style occupancy map (map_a). Honors the Z-rotation used when the map
was generated (parsed from grid_out/meta.txt). The output PNG now shows both the
2D occupancy grid with overlaid obstacle points and a 3D scatter view.

Inputs are hard-coded for the current project layout:
- grid_out/map_a.yaml + map_a.pgm
- grid_out/meta.txt (for rotation + bounds center)
- scans21.las

Outputs:
- grid_out/map_a_obstacles_scans21.png : occupancy grid + 3D scatter (down-sampled for speed)
- grid_out/map_a_obstacles_scans21.las : LAS containing only points in black cells
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


MAP_YAML = Path("grid_out/map_a.yaml")
LAS_PATH = Path("scans21.las")
OUT_PNG = Path("grid_out/map_a_obstacles_scans21.png")
OUT_LAS = Path("grid_out/map_a_obstacles_scans21.las")

# Limit points plotted for speed (LAS is fully saved regardless).
MAX_PLOT_POINTS = 300_000


def parse_simple_yaml(yaml_path: Path) -> Tuple[float, Tuple[float, float], Path]:
    """Parse resolution, origin, and image path from a simple ROS map YAML."""
    txt = yaml_path.read_text(encoding="utf-8")
    data = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        data[key.strip()] = val.strip()
    resolution = float(data["resolution"])
    origin = ast.literal_eval(data["origin"])
    image_path = yaml_path.parent / data["image"]
    return resolution, (float(origin[0]), float(origin[1])), image_path


def parse_meta(meta_path: Path) -> float:
    """Return rotation_deg from meta.txt; 0 if missing."""
    if not meta_path.exists():
        return 0.0

    rot = 0.0
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("rotation_z_deg:"):
            rot = float(line.split(":", 1)[1].strip())
            break
    return rot


def read_pgm(path: Path) -> np.ndarray:
    """Minimal P5 (binary) PGM reader returning a uint8 image array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"P5":
            raise ValueError("Only P5 PGM supported.")

        def _next_token() -> bytes:
            token = b""
            while True:
                ch = f.read(1)
                if ch == b"":
                    break
                if ch.isspace():
                    if token:
                        break
                    continue
                if ch == b"#":  # comment
                    f.readline()
                    continue
                token += ch
            return token

        width = int(_next_token())
        height = int(_next_token())
        maxval = int(_next_token())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM supported.")
        img = np.frombuffer(f.read(width * height), dtype=np.uint8)
        return img.reshape((height, width))


def main() -> None:
    if not MAP_YAML.exists():
        raise FileNotFoundError(f"{MAP_YAML} not found.")
    if not LAS_PATH.exists():
        raise FileNotFoundError(f"{LAS_PATH} not found.")
    try:
        import laspy  # noqa: F401
    except Exception:
        raise RuntimeError("laspy is required. Install with: pip install laspy") from None

    resolution, origin_xy, pgm_path = parse_simple_yaml(MAP_YAML)
    origin_x, origin_y = origin_xy
    pgm_img = read_pgm(pgm_path)

    rotation_deg = parse_meta(MAP_YAML.parent / "meta.txt")
    theta = np.deg2rad(rotation_deg)
    use_rotation = abs(theta) > 1e-9

    las = laspy.read(str(LAS_PATH))
    x = las.x
    y = las.y
    z = las.z

    # Center of rotation: use LAS header center (same convention as map build).
    cx = 0.5 * (float(las.header.mins[0]) + float(las.header.maxs[0]))
    cy = 0.5 * (float(las.header.mins[1]) + float(las.header.maxs[1]))

    if use_rotation:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0 = x - cx
        y0 = y - cy
        x = cos_t * x0 - sin_t * y0 + cx
        y = sin_t * x0 + cos_t * y0 + cy

    # Flip back to the original grid orientation used during export.
    occ_grid = np.flipud(pgm_img)
    obstacle_mask = occ_grid == 0  # black cells (occupied)

    ix = np.floor((x - origin_x) / resolution).astype(np.int64)
    iy = np.floor((y - origin_y) / resolution).astype(np.int64)

    h, w = occ_grid.shape
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ixv = ix[valid]
    iyv = iy[valid]

    in_obstacle = np.zeros_like(valid)
    in_obstacle[valid] = obstacle_mask[iyv, ixv]

    masked_points = np.stack([x[in_obstacle], y[in_obstacle], z[in_obstacle]], axis=1)
    n_total = len(x)
    n_masked = len(masked_points)

    # Save filtered LAS
    filtered = laspy.LasData(las.header)
    filtered.points = las.points[in_obstacle]
    filtered.write(str(OUT_LAS))

    # Down-sample for plotting if needed
    plot_points = masked_points
    if n_masked > MAX_PLOT_POINTS:
        idx = np.random.choice(n_masked, size=MAX_PLOT_POINTS, replace=False)
        plot_points = masked_points[idx]

    # Figure with occupancy grid + 3D scatter
    fig = plt.figure(figsize=(12, 6))

    # Left: 2D occupancy grid with overlaid points
    ax_map = fig.add_subplot(1, 2, 1)
    h, w = occ_grid.shape
    extent = (
        origin_x,
        origin_x + w * resolution,
        origin_y,
        origin_y + h * resolution,
    )
    ax_map.imshow(
        occ_grid,
        origin="lower",
        cmap="gray",
        extent=extent,
        vmin=0,
        vmax=255,
    )
    if n_masked > 0:
        ax_map.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            s=0.3,
            c="red",
            alpha=0.6,
            linewidths=0,
        )
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("map_a occupancy (black) with in-mask points")
    ax_map.set_aspect("equal", adjustable="box")

    # Right: 3D scatter of the filtered points
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    if n_masked > 0:
        sc = ax3d.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            plot_points[:, 2],
            s=0.5,
            c=plot_points[:, 2],
            cmap="viridis",
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1, label="Z (m)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(
        f"Points inside map_a occupied cells (black)\n{n_masked:,}/{n_total:,} points\n"
        f"rotation_z_deg={rotation_deg:.3f}"
    )
    ax3d.view_init(elev=30, azim=60)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Total points: {n_total:,}")
    print(f"In black cells: {n_masked:,}")
    print(f"Saved filtered LAS: {OUT_LAS}")
    print(f"Saved 3D scatter PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
