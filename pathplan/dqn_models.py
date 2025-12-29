import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .common import heading_diff
from .robot import AckermannParams

try:
    import torch
    import torch.nn as nn
except ImportError:  # torch is optional; keep lightweight defaults working without it.
    torch = None
    nn = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


@dataclass
class TorchGuidanceFeatures:
    """
    Feature container for Torch-based guidance that includes the occupancy patch.
    """

    patch: np.ndarray  # shape (patch_cells, patch_cells), float32 in [0, 1]
    dist: float
    dtheta: float
    occupied: float
    clearance: float
    front_occ: float
    heading_err: float


class DQNGuidance:
    """
    Lightweight DQN-style guidance module (torch-optional).
    Provides:
    - value(): learned heuristic estimate of cost-to-go
    - policy(): action preference scores over discrete primitives
    Uses hand-tuned weights by default; swap in a trained network for true DQN inference.
    """

    def __init__(self, params: AckermannParams, patch_size: float = 6.0, patch_cells: int = 24):
        self.params = params
        self.patch_size = patch_size
        self.patch_cells = patch_cells
        self.rng = np.random.default_rng(7)
        # deterministic pseudo-trained weights (kept small for speed)
        self.value_weights = (1.0, 0.8, 2.0, 0.5)
        self.policy_weights = (1.2, 0.6, 1.4, 0.25)

    def evaluate(
        self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map
    ) -> Tuple[Tuple[float, float, float, float], float, float]:
        """
        Compute lightweight DQN features once (distance, heading error, occupancy stats)
        plus a front-band occupancy used for action ranking.
        """
        patch = grid_map.occupancy_patch(
            state[0], state[1], state[2], size_m=self.patch_size, cells=self.patch_cells
        )
        occupied = float(np.mean(patch))
        front_band = patch[self.patch_cells // 2 :, self.patch_cells // 3 : 2 * self.patch_cells // 3]
        front_occ = float(np.mean(front_band))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal[2], state[2]))
        heading_err = heading_diff(math.atan2(dy, dx), state[2])
        clearance = 1.0 - occupied
        features = (dist, dtheta, occupied, clearance)
        return features, front_occ, heading_err

    def value_from_features(self, features: Tuple[float, float, float, float]) -> float:
        return (
            features[0] * self.value_weights[0]
            + features[1] * self.value_weights[1]
            + features[2] * self.value_weights[2]
            + features[3] * self.value_weights[3]
        )

    def policy_from_eval(
        self,
        features: Tuple[float, float, float, float],
        heading_err: float,
        front_occ: float,
        actions: Sequence,
    ) -> List[float]:
        """
        Return preference scores for each action (higher = better) using precomputed stats.
        """
        scores: List[float] = []
        for act in actions:
            steer_term = -abs(act.steering) / (self.params.max_steer + 1e-6)
            dir_term = 0.0 if act.direction > 0 else -0.3
            steer_sign = math.copysign(1.0, act.steering) if act.steering != 0 else 0.0
            goal_sign = math.copysign(1.0, heading_err) if heading_err != 0 else 0.0
            heading_term = -abs(heading_err) + 0.2 * (1.0 if steer_sign == goal_sign else 0.0)
            clearance_term = -front_occ
            score = (
                self.policy_weights[0] * steer_term
                + self.policy_weights[1] * heading_term
                + self.policy_weights[2] * dir_term
                + self.policy_weights[3] * clearance_term
            )
            scores.append(score)
        return scores

    def value(self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map) -> float:
        """Approximate cost-to-go; positive scalar."""
        feats, _, _ = self.evaluate(state, goal, grid_map)
        return self.value_from_features(feats)

    def policy(
        self,
        state: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        grid_map,
        actions: Sequence,
    ) -> List[float]:
        """
        Return preference scores for each action (higher = better).
        Uses goal heading alignment and local clearance in front of the robot.
        """
        feats, front_occ, heading_err = self.evaluate(state, goal, grid_map)
        return self.policy_from_eval(feats, heading_err, front_occ, actions)


if TORCH_AVAILABLE:

    class ConvGuidanceNet(nn.Module):
        """
        Small CNN + MLP that maps an occupancy patch and scalar features to Q-values.
        """

        def __init__(self, num_actions: int, patch_cells: int):
            super().__init__()
            self.patch_cells = patch_cells
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # patch_cells -> patch_cells/2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # patch_cells/2 -> patch_cells/4
            )
            conv_out = 64 * (patch_cells // 4) * (patch_cells // 4)
            self.head = nn.Sequential(
                nn.Linear(conv_out + 5, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.q = nn.Linear(128, num_actions)

        def forward(self, patch: "torch.Tensor", feats: "torch.Tensor") -> "torch.Tensor":
            x = self.conv(patch)
            x = torch.flatten(x, 1)
            x = torch.cat([x, feats], dim=1)
            x = self.head(x)
            return self.q(x)


class TorchDQNGuidance:
    """
    Torch-based guidance that loads a trained ConvGuidanceNet and serves the same API
    as DQNGuidance for the planner (value() + policy()).
    """

    def __init__(
        self,
        params: AckermannParams,
        model_path: str,
        primitives: Optional[Sequence] = None,
        patch_size: float = 6.0,
        patch_cells: int = 24,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchDQNGuidance but is not installed.")
        self.params = params
        self.patch_size = patch_size
        self.patch_cells = patch_cells
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            patch_cells = state_dict.get("patch_cells", patch_cells)
            self.patch_cells = patch_cells
            state_dict = state_dict["model_state_dict"]
        num_actions = len(primitives) if primitives is not None else 10
        self.model = ConvGuidanceNet(num_actions, self.patch_cells).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate(
        self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map
    ) -> Tuple[TorchGuidanceFeatures, float, float]:
        patch = grid_map.occupancy_patch(
            state[0], state[1], state[2], size_m=self.patch_size, cells=self.patch_cells
        ).astype(np.float32)
        occupied = float(np.mean(patch))
        front_band = patch[self.patch_cells // 2 :, self.patch_cells // 3 : 2 * self.patch_cells // 3]
        front_occ = float(np.mean(front_band))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = math.hypot(dx, dy)
        dtheta = abs(heading_diff(goal[2], state[2]))
        heading_err = heading_diff(math.atan2(dy, dx), state[2])
        clearance = 1.0 - occupied
        feats = TorchGuidanceFeatures(
            patch=patch,
            dist=dist,
            dtheta=dtheta,
            occupied=occupied,
            clearance=clearance,
            front_occ=front_occ,
            heading_err=heading_err,
        )
        return feats, front_occ, heading_err

    def _q_values(self, features: TorchGuidanceFeatures) -> "torch.Tensor":
        patch = torch.from_numpy(features.patch).float().unsqueeze(0).unsqueeze(0).to(self.device)
        feat_vec = torch.tensor(
            [[features.dist, features.dtheta, features.occupied, features.clearance, features.front_occ]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return self.model(patch, feat_vec).squeeze(0)

    def value_from_features(self, features: TorchGuidanceFeatures) -> float:
        # Use the negative of the best Q as a heuristic cost (>= 0).
        best_q = float(self._q_values(features).max().item())
        return max(0.0, -best_q)

    def policy_from_eval(
        self,
        features: TorchGuidanceFeatures,
        heading_err: float,
        front_occ: float,
        actions: Sequence,
    ) -> List[float]:
        return self._q_values(features).detach().cpu().tolist()

    def value(self, state: Tuple[float, float, float], goal: Tuple[float, float, float], grid_map) -> float:
        feats, _, _ = self.evaluate(state, goal, grid_map)
        return self.value_from_features(feats)

    def policy(
        self,
        state: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        grid_map,
        actions: Sequence,
    ) -> List[float]:
        feats, front_occ, heading_err = self.evaluate(state, goal, grid_map)
        return self.policy_from_eval(feats, heading_err, front_occ, actions)
