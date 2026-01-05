from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import laspy
from scipy import ndimage as ndi

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def write_pgm(pgm_path: Path, occ_grid: np.ndarray) -> None:
    """
    Write ROS-compatible PGM (P5).
    occ_grid values: {-1, 0, 100}
      -1 unknown -> 205
       0 free    -> 254
     100 occ     -> 0
    """
    unk, free, occ = 205, 254, 0
    img = np.full(occ_grid.shape, unk, dtype=np.uint8)
    img[occ_grid == 0] = free
    img[occ_grid == 100] = occ

    img = np.flipud(img)  # ROS map image convention

    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with open(pgm_path, "wb") as f:
        f.write(header)
        f.write(img.tobytes())


def write_ros_yaml(yaml_path: Path, pgm_filename: str, resolution: float, origin_xy: Tuple[float, float]) -> None:
    origin_x, origin_y = origin_xy
    text = (
        f"image: {pgm_filename}\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_x}, {origin_y}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def export_ros_map(
    out_dir: Path,
    base_name: str,
    occ_grid: np.ndarray,
    resolution: float,
    origin_xy: Tuple[float, float],
) -> None:
    pgm_name = f"{base_name}.pgm"
    write_pgm(out_dir / pgm_name, occ_grid)
    write_ros_yaml(out_dir / f"{base_name}.yaml", pgm_name, resolution, origin_xy)


def occupancy_from_stats(
    count: np.ndarray,
    roughness: np.ndarray,
    min_points: int,
    roughness_thresh: float,
) -> np.ndarray:
    occ = np.full(count.shape, -1, dtype=np.int16)
    enough = count >= min_points
    occ[enough & (roughness > roughness_thresh)] = 100
    occ[enough & (roughness <= roughness_thresh)] = 0
    return occ


def occupancy_points_as_free(
    count: np.ndarray,
    min_points: int,
    occupied_value: int = 100,
) -> np.ndarray:
    """
    Mark any cell with at least min_points as free (0); others as occupied.
    This forces white for all observed cells and black elsewhere.
    """
    occ = np.full(count.shape, occupied_value, dtype=np.int16)
    occ[count >= min_points] = 0
    return occ


def save_occupancy_png(png_path: Path, occ_grid: np.ndarray) -> None:
    vis = np.full(occ_grid.shape, 128, dtype=np.uint8)
    vis[occ_grid == 0] = 255
    vis[occ_grid == 100] = 0
    plt.imsave(png_path, np.flipud(vis), cmap="gray")


def remove_small_obstacles(
    occ_grid: np.ndarray,
    min_size: int = 3,
    fill_value: int = -1,
) -> np.ndarray:
    """
    Remove obstacle blobs (value 100) smaller than min_size cells.
    Converts those cells to fill_value to avoid pepper noise.
    """
    if min_size <= 1:
        return occ_grid
    mask = occ_grid == 100
    labeled, nlab = ndi.label(mask)
    if nlab == 0:
        return occ_grid
    counts = np.bincount(labeled.ravel())
    small_ids = np.where(counts < min_size)[0]
    if len(small_ids) == 0:
        return occ_grid
    cleaned = occ_grid.copy()
    cleaned[np.isin(labeled, small_ids)] = fill_value
    return cleaned


def clean_occupancy(
    occ_grid: np.ndarray,
    min_size: int = 5,
    morph_open: int = 1,
    morph_close: int = 1,
    reset_value: int = -1,
) -> np.ndarray:
    """
    Binary-morphology + small-component removal on obstacle cells (value 100).
    - morph_open: removes isolated dots (erosion+ dilation)
    - morph_close: fills tiny gaps (dilation+ erosion)
    - min_size: remove connected components smaller than this.
    - reset_value: value assigned to removed obstacle cells (-1 unknown or 0 free).
    Unknown (-1) and free (0) are preserved unless overwritten by cleaned obstacles.
    """
    if min_size <= 1 and morph_open <= 0 and morph_close <= 0:
        return occ_grid

    obs_mask = occ_grid == 100
    if morph_open > 0:
        obs_mask = ndi.binary_opening(obs_mask, iterations=morph_open)
    if morph_close > 0:
        obs_mask = ndi.binary_closing(obs_mask, iterations=morph_close)

    if min_size > 1:
        labeled, nlab = ndi.label(obs_mask)
        if nlab > 0:
            counts = np.bincount(labeled.ravel())
            small_ids = np.where(counts < min_size)[0]
            obs_mask[np.isin(labeled, small_ids)] = False

    cleaned = occ_grid.copy()
    cleaned[occ_grid == 100] = reset_value  # reset obstacles
    cleaned[obs_mask] = 100        # apply cleaned obstacle mask
    return cleaned


def roughness_median_filter(roughness: np.ndarray, size: int = 3) -> np.ndarray:
    if size <= 1:
        return roughness
    valid = np.isfinite(roughness)
    if not np.any(valid):
        return roughness
    filled = roughness.copy()
    fill_value = float(np.nanmin(roughness[valid]))
    filled[~valid] = fill_value
    filtered = ndi.median_filter(filled, size=size)
    filtered[~valid] = np.nan
    return filtered


def build_grids_from_las(
    las_path: Path,
    resolution: float,
    min_points: int,
    roughness_thresh: float,
    z_min: Optional[float],
    z_max: Optional[float],
    chunk_size: int,
    rotate_deg: float = 0.0,
) -> dict:
    with laspy.open(str(las_path)) as reader:
        hdr = reader.header

        min_x_hdr, min_y_hdr, min_z_hdr = hdr.mins
        max_x_hdr, max_y_hdr, max_z_hdr = hdr.maxs

        if z_min is None:
            z_min = float(min_z_hdr)
        if z_max is None:
            z_max = float(max_z_hdr)

        theta = np.deg2rad(rotate_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        use_rotation = abs(theta) > 1e-9
        cx = 0.5 * (min_x_hdr + max_x_hdr)
        cy = 0.5 * (min_y_hdr + max_y_hdr)

        def rotate_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if not use_rotation:
                return x, y
            x0 = x - cx
            y0 = y - cy
            return (
                cos_t * x0 - sin_t * y0 + cx,
                sin_t * x0 + cos_t * y0 + cy,
            )

        corners = np.array(
            [
                (min_x_hdr, min_y_hdr),
                (min_x_hdr, max_y_hdr),
                (max_x_hdr, min_y_hdr),
                (max_x_hdr, max_y_hdr),
            ],
            dtype=np.float64,
        )
        rot_x, rot_y = rotate_xy(corners[:, 0], corners[:, 1])
        min_x = float(np.min(rot_x))
        max_x = float(np.max(rot_x))
        min_y = float(np.min(rot_y))
        max_y = float(np.max(rot_y))

        nx = int(np.ceil((max_x - min_x) / resolution))
        ny = int(np.ceil((max_y - min_y) / resolution))
        if nx <= 0 or ny <= 0:
            raise ValueError("Invalid grid size; check LAS bounds/resolution.")

        ncell = nx * ny
        count = np.zeros(ncell, dtype=np.int32)
        z_sum = np.zeros(ncell, dtype=np.float64)
        z_min_grid = np.full(ncell, np.inf, dtype=np.float64)
        z_max_grid = np.full(ncell, -np.inf, dtype=np.float64)

        def update_with_points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
            m = (z >= z_min) & (z <= z_max)
            if not np.any(m):
                return
            x, y, z = x[m], y[m], z[m]

            x, y = rotate_xy(x, y)

            ix = np.floor((x - min_x) / resolution).astype(np.int64)
            iy = np.floor((y - min_y) / resolution).astype(np.int64)

            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            if not np.any(valid):
                return
            ix, iy, z = ix[valid], iy[valid], z[valid]

            flat = iy * nx + ix
            np.add.at(count, flat, 1)
            np.add.at(z_sum, flat, z)
            np.minimum.at(z_min_grid, flat, z)
            np.maximum.at(z_max_grid, flat, z)

        for pts in reader.chunk_iterator(chunk_size):
            update_with_points(pts.x, pts.y, pts.z)

        count2 = count.reshape(ny, nx)
        z_sum2 = z_sum.reshape(ny, nx)
        zmin2 = z_min_grid.reshape(ny, nx)
        zmax2 = z_max_grid.reshape(ny, nx)

        mean_z = np.full((ny, nx), np.nan, dtype=np.float32)
        valid_cells = count2 > 0
        mean_z[valid_cells] = (z_sum2[valid_cells] / count2[valid_cells]).astype(np.float32)

        roughness = np.full((ny, nx), np.nan, dtype=np.float32)
        roughness[valid_cells] = (zmax2[valid_cells] - zmin2[valid_cells]).astype(np.float32)

        occ = occupancy_from_stats(count2, roughness, min_points, roughness_thresh)

        return {
            "bounds": (min_x, min_y, max_x, max_y),
            "shape": (ny, nx),
            "count": count2,
            "z_min": zmin2,
            "z_max": zmax2,
            "mean_z": mean_z,
            "roughness": roughness,
            "occupancy": occ,
        }


def main():
    # ==========================
    # 1) 直接在这里写你的参数
    # ==========================
    LAS_PATH = Path(r"E:\tongbu\BaiduSyncdisk\study\phdprojec\博士第一篇sci\点云分割2\test\test1.1 - 副本\traversable_scans21_r0p5_a20p0_rf0p35_z1p0.las")
    OUT_DIR = Path(r"E:\tongbu\BaiduSyncdisk\study\phdprojec\博士第一篇sci\点云分割2\test\test1.1 - 副本\grid_out")

    RESOLUTION = 0.1     # 栅格分辨率(m)
    MIN_POINTS_BASE = 1          # 每格至少点数，否则 unknown
    MIN_POINTS_STRICT = 3        # 更稳健的点数阈值
    ROUGHNESS_THRESH = 0.20      # max_z - min_z > 阈值 => obstacle
    Z_MIN = None                 # 可选：高度裁剪下限，例如 -1.0
    Z_MAX = None                 # 可选：高度裁剪上限，例如  1.0
    CHUNK_SIZE = 2_000_000       # 分块读取点数（内存不够就调小）
    ROTATE_Z_DEG = 13.8           # 先绕 Z 轴旋转点云再建图（度，逆时针为正）

    MORPH_OPEN = 1               # 1~2 通常够去孤立点
    MORPH_CLOSE = 1              # 1~2 用于补小缝
    MIN_OBS_CELLS = 16           # 删除面积小于该值的孤立黑点（障碍）
    ROUGHNESS_MEDIAN_SIZE = 3    # map_d: 空间中值滤波窗口大小(<=1 关闭)

    MAP_A_POINTS_AS_FREE = True   # map_a: 只要该格有点就标记为可通行（白格），其余标为障碍（黑）
    DEFAULT_MAP_NAME = "map_c"   # occupancy.npy uses this variant
    EXPORT_ROS_MAP = True        # 输出 map_*.pgm + map_*.yaml
    SAVE_DEBUG_PNG = True        # 输出 occupancy_*.png / mean_z.png / roughness.png（需要 matplotlib）

    # ==========================
    # 2) 开始处理
    # ==========================
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grids = build_grids_from_las(
        las_path=LAS_PATH,
        resolution=RESOLUTION,
        min_points=MIN_POINTS_BASE,
        roughness_thresh=ROUGHNESS_THRESH,
        z_min=Z_MIN,
        z_max=Z_MAX,
        chunk_size=CHUNK_SIZE,
        rotate_deg=ROTATE_Z_DEG,
    )

    if SAVE_DEBUG_PNG and not HAS_MPL:
        raise RuntimeError("matplotlib not installed. pip install matplotlib")

    count = grids["count"]
    roughness = grids["roughness"]
    roughness_d = roughness
    if ROUGHNESS_MEDIAN_SIZE > 1:
        roughness_d = roughness_median_filter(roughness, size=ROUGHNESS_MEDIAN_SIZE)

    (min_x, min_y, max_x, max_y) = grids["bounds"]
    variants = [
        ("map_a", MIN_POINTS_BASE, roughness, 100),  # keep obstacles as 100 to avoid gray
        ("map_b", MIN_POINTS_BASE, roughness, 0),
        ("map_c", MIN_POINTS_STRICT, roughness, 0),
        ("map_d", MIN_POINTS_STRICT, roughness_d, 0),
    ]
    default_occupancy = None

    for name, min_points, roughness_src, reset_value in variants:
        if name == "map_a" and MAP_A_POINTS_AS_FREE:
            occ = occupancy_points_as_free(count, min_points, occupied_value=100)
        else:
            occ = occupancy_from_stats(count, roughness_src, min_points, ROUGHNESS_THRESH)
        occ = clean_occupancy(
            occ,
            min_size=MIN_OBS_CELLS,
            morph_open=MORPH_OPEN,
            morph_close=MORPH_CLOSE,
            reset_value=reset_value,
        )
        if name == DEFAULT_MAP_NAME:
            default_occupancy = occ
        if EXPORT_ROS_MAP:
            export_ros_map(OUT_DIR, name, occ, RESOLUTION, (min_x, min_y))
        if SAVE_DEBUG_PNG:
            save_occupancy_png(OUT_DIR / f"occupancy_{name}.png", occ)

    if default_occupancy is None:
        raise ValueError(f"DEFAULT_MAP_NAME '{DEFAULT_MAP_NAME}' not in variants.")

    grids["occupancy"] = default_occupancy

    # 保存数组
    np.save(OUT_DIR / "occupancy.npy", grids["occupancy"])
    np.save(OUT_DIR / "mean_z.npy", grids["mean_z"])
    np.save(OUT_DIR / "roughness.npy", grids["roughness"])
    np.save(OUT_DIR / "count.npy", grids["count"])
    np.save(OUT_DIR / "z_min.npy", grids["z_min"])
    np.save(OUT_DIR / "z_max.npy", grids["z_max"])

    meta = (
        f"bounds: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}\n"
        f"shape: ny={grids['shape'][0]}, nx={grids['shape'][1]}\n"
        f"resolution: {RESOLUTION}\n"
        f"rotation_z_deg: {ROTATE_Z_DEG}\n"
        f"min_points_base: {MIN_POINTS_BASE}\n"
        f"min_points_strict: {MIN_POINTS_STRICT}\n"
        f"roughness_thresh: {ROUGHNESS_THRESH}\n"
        f"map_a_points_as_free: {MAP_A_POINTS_AS_FREE}\n"
        f"morph_open: {MORPH_OPEN}\n"
        f"morph_close: {MORPH_CLOSE}\n"
        f"min_obs_cells: {MIN_OBS_CELLS}\n"
        f"roughness_median_size: {ROUGHNESS_MEDIAN_SIZE}\n"
        f"default_map: {DEFAULT_MAP_NAME}\n"
        f"variants: {', '.join(name for name, *_ in variants)}\n"
        f"z_filter: [{Z_MIN}, {Z_MAX}]\n"
    )
    (OUT_DIR / "meta.txt").write_text(meta, encoding="utf-8")

    # Debug PNG
    if SAVE_DEBUG_PNG:
        mz = grids["mean_z"]
        if np.any(np.isfinite(mz)):
            mz2 = np.nan_to_num(mz, nan=float(np.nanmin(mz[np.isfinite(mz)])))
        else:
            mz2 = np.zeros_like(mz, dtype=np.float32)
        plt.imsave(OUT_DIR / "mean_z.png", np.flipud(mz2))

        rf = grids["roughness"]
        rf2 = np.nan_to_num(rf, nan=0.0)
        plt.imsave(OUT_DIR / "roughness.png", np.flipud(rf2))

    print("Done. Outputs in:", str(OUT_DIR.resolve()))
    print("Maps:", ", ".join(name for name, *_ in variants))


if __name__ == "__main__":
    main()
