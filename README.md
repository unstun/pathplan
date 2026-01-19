# Path Planning Scaffold

Python scaffold for 2D occupancy-grid path planning with an Ackermann robot model.

## Planners

- Artificial Potential Field (APF): `pathplan.APFPlanner`
- Hybrid A*: `pathplan.HybridAStarPlanner`
- Spline-based RRT* (SS-RRT*): `pathplan.RRTStarPlanner`

## Install

```bash
pip install -r requirements.txt
# optional (plots)
pip install matplotlib
```

## Run examples

```bash
python -m examples.run_demo
python -m examples.forest_scene --variant all
```

Outputs are written under `examples/outputs/`.
