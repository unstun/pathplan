"""
Minimal DQN training loop for the D-Hybrid A* (DQN-guided Hybrid A*) guidance head.
Uses a lightweight CNN over local occupancy patches plus scalar goal features.
CUDA is used automatically when available.
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathplan import AckermannParams, AckermannState, GridMap, OrientedBoxFootprint
from pathplan.common import heading_diff, default_collision_step
from pathplan.dqn_models import ConvGuidanceNet, TorchGuidanceFeatures
from pathplan.geometry import GridFootprintChecker
from pathplan.primitives import default_primitives, primitive_cost
from pathplan.robot import sample_constant_steer_motion


def make_corridor_map(resolution: float = 0.1, length: float = 10.0, width: float = 2.0):
    cells_x = int(length / resolution) + 1
    cells_y = int(3.0 / resolution)
    grid = np.zeros((cells_y, cells_x), dtype=np.uint8)
    corridor_cells = int(width / resolution)
    pad = (cells_y - corridor_cells) // 2
    grid[: pad, :] = 1
    grid[pad + corridor_cells :, :] = 1
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def make_open_map(resolution: float = 0.1, size: Tuple[float, float] = (6.0, 5.0)):
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def make_random_blocks_map(rng: np.random.Generator, resolution: float = 0.1, size: Tuple[float, float] = (8.0, 6.0)):
    w_cells = int(size[0] / resolution)
    h_cells = int(size[1] / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    # scatter a few rectangular obstacles
    for _ in range(10):
        w = rng.integers(3, max(4, w_cells // 4))
        h = rng.integers(3, max(4, h_cells // 4))
        x0 = rng.integers(1, max(2, w_cells - w - 1))
        y0 = rng.integers(1, max(2, h_cells - h - 1))
        grid[y0 : y0 + h, x0 : x0 + w] = 1
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return GridMap(grid, resolution, origin=(0.0, 0.0))


def extract_features(
    grid_map: GridMap,
    state: AckermannState,
    goal: AckermannState,
    patch_size: float,
    patch_cells: int,
) -> TorchGuidanceFeatures:
    patch = grid_map.occupancy_patch(state.x, state.y, state.theta, size_m=patch_size, cells=patch_cells).astype(
        np.float32
    )
    occupied = float(np.mean(patch))
    front_band = patch[patch_cells // 2 :, patch_cells // 3 : 2 * patch_cells // 3]
    front_occ = float(np.mean(front_band))
    dx = goal.x - state.x
    dy = goal.y - state.y
    dist = math.hypot(dx, dy)
    dtheta = abs(heading_diff(goal.theta, state.theta))
    heading_err = heading_diff(math.atan2(dy, dx), state.theta)
    clearance = 1.0 - occupied
    return TorchGuidanceFeatures(
        patch=patch,
        dist=dist,
        dtheta=dtheta,
        occupied=occupied,
        clearance=clearance,
        front_occ=front_occ,
        heading_err=heading_err,
    )


@dataclass
class Transition:
    obs: TorchGuidanceFeatures
    action: int
    reward: float
    next_obs: TorchGuidanceFeatures
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


class DQNEnv:
    """
    Simple kinodynamic environment: states are continuous SE(2) poses, actions are motion primitives.
    Reward shaping: progress toward goal, penalties for collision, cusps, and time.
    """

    def __init__(
        self,
        seed: int = 0,
        patch_size: float = 6.0,
        patch_cells: int = 24,
        max_steps: int = 120,
        collision_step: Optional[float] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.params = AckermannParams()
        self.footprint = OrientedBoxFootprint(length=0.924, width=0.740)
        self.primitives = default_primitives(self.params)
        self.patch_size = patch_size
        self.patch_cells = patch_cells
        self.max_steps = max_steps
        self._collision_step_override = collision_step
        self.collision_step = collision_step if collision_step is not None else 0.0
        self.map = None
        self.goal = None
        self.state = None
        self.steps = 0
        self.prev_action = None
        self.prev_goal_dist = 0.0
        self.collision_checker = None

    def _sample_map(self) -> GridMap:
        pick = self.rng.random()
        if pick < 0.4:
            return make_random_blocks_map(self.rng)
        if pick < 0.7:
            return make_corridor_map(width=2.0)
        return make_open_map()

    def _goal_dist(self, state: AckermannState) -> float:
        dx = self.goal.x - state.x
        dy = self.goal.y - state.y
        return math.hypot(dx, dy)

    def _goal_reached(self, state: AckermannState) -> bool:
        return self._goal_dist(state) <= 0.1 and abs(heading_diff(self.goal.theta, state.theta)) <= math.radians(5.0)

    def reset(self) -> TorchGuidanceFeatures:
        self.map = self._sample_map()
        self.collision_checker = GridFootprintChecker(self.map, self.footprint, theta_bins=72)
        base_step = default_collision_step(self.map.resolution)
        self.collision_step = self._collision_step_override if self._collision_step_override is not None else base_step
        self.state = AckermannState(*self.map.random_free_state(self.rng))
        for _ in range(100):
            gx, gy, gtheta = self.map.random_free_state(self.rng)
            if math.hypot(gx - self.state.x, gy - self.state.y) > 1.0:
                self.goal = AckermannState(gx, gy, gtheta)
                break
        else:
            self.goal = AckermannState(*self.map.random_free_state(self.rng))
        self.steps = 0
        self.prev_action = None
        self.prev_goal_dist = self._goal_dist(self.state)
        return extract_features(self.map, self.state, self.goal, self.patch_size, self.patch_cells)

    def step(self, action_idx: int) -> Tuple[TorchGuidanceFeatures, float, bool, dict]:
        self.steps += 1
        prim = self.primitives[action_idx]
        arc_states, _ = sample_constant_steer_motion(
            self.state,
            prim.steering,
            prim.direction,
            prim.step,
            self.params,
            step=self.collision_step,
            footprint=None,
        )
        collided = self.collision_checker.collides_path(arc_states)
        reward = -0.05 - 0.01 * primitive_cost(prim)
        done = False
        success = False

        if collided:
            reward -= 1.0
            done = True
        else:
            nxt = arc_states[-1]
            dist_after = self._goal_dist(nxt)
            reward += 0.3 * (self.prev_goal_dist - dist_after)
            if self.prev_action and prim.direction != self.prev_action.direction:
                reward -= 0.05
            self.state = nxt
            self.prev_action = prim
            self.prev_goal_dist = dist_after
            if self._goal_reached(nxt):
                reward += 5.0
                done = True
                success = True
            elif self.steps >= self.max_steps:
                reward -= 0.5
                done = True

        obs = extract_features(self.map, self.state, self.goal, self.patch_size, self.patch_cells)
        return obs, reward, done, {"success": success}


def select_action(policy_net: ConvGuidanceNet, obs: TorchGuidanceFeatures, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(policy_net.q.out_features)
    patch = torch.from_numpy(obs.patch).float().unsqueeze(0).unsqueeze(0).to(device)
    feats = torch.tensor([[obs.dist, obs.dtheta, obs.occupied, obs.clearance, obs.front_occ]], device=device)
    with torch.no_grad():
        q_values = policy_net(patch, feats)
    return int(torch.argmax(q_values).item())


def optimize_model(
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    policy_net: ConvGuidanceNet,
    target_net: ConvGuidanceNet,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    if len(buffer) < batch_size:
        return 0.0

    batch = buffer.sample(batch_size)
    patches = torch.stack(
        [torch.from_numpy(t.obs.patch).float().unsqueeze(0) for t in batch],
        dim=0,
    ).to(device)
    feats = torch.tensor(
        [[t.obs.dist, t.obs.dtheta, t.obs.occupied, t.obs.clearance, t.obs.front_occ] for t in batch],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_patches = torch.stack(
        [torch.from_numpy(t.next_obs.patch).float().unsqueeze(0) for t in batch],
        dim=0,
    ).to(device)
    next_feats = torch.tensor(
        [[t.next_obs.dist, t.next_obs.dtheta, t.next_obs.occupied, t.next_obs.clearance, t.next_obs.front_occ] for t in batch],
        dtype=torch.float32,
        device=device,
    )
    done = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=device)

    q_values = policy_net(patches, feats).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_patches, next_feats).max(1)[0]
        target = rewards + gamma * (1.0 - done) * next_q
    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser(
        description="Train a DQN guidance model for the D-Hybrid A* (DQN-guided Hybrid A*) planner (uses CUDA when available)."
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--target-update", type=int, default=10, help="Episodes between target net updates.")
    parser.add_argument("--eps-start", type=float, default=0.9, help="Initial epsilon for exploration.")
    parser.add_argument("--eps-final", type=float, default=0.05, help="Minimum epsilon.")
    parser.add_argument("--eps-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode.")
    parser.add_argument("--max-steps", type=int, default=120, help="Steps per episode.")
    parser.add_argument("--save-path", type=Path, default=Path(__file__).resolve().parent / "outputs" / "dqn_guidance.pt")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. 'cuda:0' or 'cpu'.")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("CUDA not available, falling back to CPU.")

    env = DQNEnv(max_steps=args.max_steps)
    policy_net = ConvGuidanceNet(num_actions=len(env.primitives), patch_cells=env.patch_cells).to(device)
    target_net = ConvGuidanceNet(num_actions=len(env.primitives), patch_cells=env.patch_cells).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    epsilon = args.eps_start
    last_loss = 0.0
    success_counter = 0

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        for _ in range(env.max_steps):
            action = select_action(policy_net, obs, epsilon, device)
            next_obs, reward, done, info = env.step(action)
            buffer.push(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
            obs = next_obs
            episode_reward += reward
            loss_val = optimize_model(buffer, args.batch_size, args.gamma, policy_net, target_net, optimizer, device)
            if loss_val:
                last_loss = loss_val
            if done:
                success_counter += int(info.get("success", False))
                break

        epsilon = max(args.eps_final, epsilon * args.eps_decay)
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            avg_success = success_counter / 10.0
            success_counter = 0
            print(
                f"Episode {episode:04d}: reward={episode_reward:6.2f}, loss={last_loss:6.4f}, "
                f"epsilon={epsilon:.3f}, success_rate(last 10)={avg_success:.2f}"
            )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "patch_cells": env.patch_cells,
        },
        args.save_path,
    )
    print(f"Saved trained model to {args.save_path}")


if __name__ == "__main__":
    main()
