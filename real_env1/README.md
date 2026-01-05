# real_env1 (real SLAM forest map)

Point-cloud pipeline for a real forest SLAM scene: segment traversable points and turn them into occupancy grids/ROS maps you can feed to the planners.

Contents:
- Raw SLAM cloud: `scans21.las` (plus intermediate traversable/untraversable LAS and HPC dumps).
- Segmentation: `fengev2v2_1h.py` (KDTree-based traversability filter with slope and height gates).
- Grid build + ROS export: `map.py` (LAS -> occupancy -> `.pgm/.yaml` + `.npy` tensors, with cleaning and rotation support).
- Validation: `visualize_map_mask.py` (overlay occupancy masks on the original cloud and save a filtered LAS).
- Sample outputs live in `real_env1/grid_out/` (generated at 0.1 m, 13.8 deg rotation, `map_c` as default occupancy).

Install the extra tools once:
```bash
pip install -r requirements-real-env.txt  # laspy, scipy, tqdm, matplotlib (plus numpy)
```

## 1) (Optional) Segment traversable vs. untraversable points
- Edit the parameters near the top of `real_env1/fengev2v2_1h.py`:
  - `las_file` (input), `pre_z_max_threshold` (pre-filter), `radius`, `max_climbing_angle`, `relaxation_factor`.
  - Optional `filter_high_z_points` step is available after segmentation.
- Run it: `python -m real_env1.fengev2v2_1h` (or `python real_env1/fengev2v2_1h.py`).
- Outputs: `traversable_<tags>.las` and `untraversable_<tags>.las` in the same folder; tags encode the parameters used.

## 2) Build occupancy grids and ROS maps
- Set `LAS_PATH` (raw or traversable LAS) and `OUT_DIR` at the top of `real_env1/map.py` (defaults target `grid_out/`).
- Key knobs:
  - `RESOLUTION` (m per cell) and optional `ROTATE_Z_DEG` to align the cloud before gridding.
  - `MIN_POINTS_BASE/STRICT`, `ROUGHNESS_THRESH` (z_max - z_min) decide obstacle vs. free.
  - Optional Z clipping (`Z_MIN/Z_MAX`) and morphology cleanup (`MORPH_OPEN/CLOSE`, `MIN_OBS_CELLS`).
  - Map variants: `map_a` (every observed cell marked free by default), `map_b/c` (roughness-based, with stricter min-points), `map_d` (roughness with median filter). `DEFAULT_MAP_NAME` selects which variant is saved as `occupancy.npy`.
- Run it: `python -m real_env1.map` (or `python real_env1/map.py`). Requires matplotlib if `SAVE_DEBUG_PNG=True`.
- Outputs in `grid_out/`:
  - `map_*.pgm/.yaml` (ROS-ready), `occupancy.npy` (int16, values {-1,0,100}), `mean_z.npy`, `roughness.npy`, `count.npy`, `z_min.npy`, `z_max.npy`, plus debug PNGs.
  - `meta.txt` records bounds, rotation, resolution, and thresholds.

## 3) Validate alignment
- Adjust the paths at the top of `real_env1/visualize_map_mask.py` if needed (defaults to `grid_out/map_a.yaml` + `scans21.las`).
- Run: `python -m real_env1.visualize_map_mask`.
- Outputs: `grid_out/map_a_obstacles_scans21.png` (occupancy with overlaid obstacle points + 3D scatter) and `grid_out/map_a_obstacles_scans21.las` (points that fall inside black cells).

## 4) Use the map in the planners
`pathplan.GridMap` expects a binary obstacle mask (1 = occupied, 0 = free). Convert the saved occupancy grid and reuse the origin/resolution from the ROS YAML:
```python
import numpy as np
from pathplan import GridMap

occ = np.load("real_env1/grid_out/occupancy.npy")      # -1 unknown, 0 free, 100 occupied
grid_mask = (occ == 100).astype(np.uint8)              # 1 = obstacle
resolution = 0.1                                       # from map_c.yaml/meta.txt
origin = (-35.57937792469317, -8.541080032918341)      # update from your YAML/meta
grid_map = GridMap(grid_mask, resolution, origin)
```
Swap in your own origin/resolution from the generated `map_*.yaml` or `meta.txt`, then call any planner as usual.
