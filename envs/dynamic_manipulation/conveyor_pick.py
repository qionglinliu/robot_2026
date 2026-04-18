# Copyright 2026 The Robot2026 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Conveyor Belt Pick Environment.

A 2D environment where objects move on a conveyor belt and the robot must
pick them up within a limited time window. This task emphasizes:
- Time-critical interaction window
- Need to predict object position at grasp time
- Adaptive execution horizon for efficiency vs. robustness tradeoff
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConveyorPickEnv(gym.Env):
    """Conveyor Belt Pick Environment.

    Objects appear on the left side of the workspace and move rightward on a
    conveyor belt at configurable speed. The robot must intercept and grasp
    objects before they exit the reachable workspace on the right.

    The workspace is 2D: x ∈ [-1, 1] (horizontal, conveyor direction), y ∈ [-0.5, 0.5].
    The conveyor moves objects in the +x direction. The robot EE can move freely
    in the 2D workspace.

    Observation (state mode): [ee_x, ee_y, obj_x, obj_y, obj_vx, obj_vy, grasp, time_remaining_norm]
    Action: [dx, dy, grasp_signal]

    Args:
        max_episode_steps: Maximum steps per episode.
        fps: Frames per second.
        conveyor_speed: Speed of the conveyor in workspace units/s.
        n_objects_per_episode: Number of objects to pick per episode.
        graspable_region: (x_min, x_max) defining where the robot can reach.
        success_threshold: Distance for successful grasp.
        execution_delay: Simulated execution delay in seconds.
        obs_type: "state", "pixels", or "pixels_agent_pos".
        image_size: Rendered image size.
        reward_type: "sparse" or "shaped".
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "ConveyorPick-v0",
    }

    def __init__(
        self,
        max_episode_steps: int = 400,
        fps: int = 30,
        conveyor_speed: float = 0.2,
        n_objects_per_episode: int = 3,
        graspable_region: tuple[float, float] = (-0.3, 0.7),
        success_threshold: float = 0.06,
        execution_delay: float = 0.0,
        obs_type: str = "state",
        image_size: int = 128,
        reward_type: str = "shaped",
        render_mode: str | None = None,
        disable_env_checker: bool = True,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.fps = fps
        self.dt = 1.0 / fps
        self.conveyor_speed = conveyor_speed
        self.n_objects_per_episode = n_objects_per_episode
        self.graspable_region = graspable_region
        self.success_threshold = success_threshold
        self.execution_delay = execution_delay
        self.obs_type = obs_type
        self.image_size = image_size
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Robot properties
        self.max_ee_speed = 0.5
        self.workspace_x = (-1.0, 1.0)
        self.workspace_y = (-0.5, 0.5)

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space
        if obs_type == "state":
            # [ee_x, ee_y, obj_x, obj_y, obj_vx, obj_vy, grasp, time_to_reach_norm]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )
        elif obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(
                    low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
            })

        # State
        self.ee_pos = np.zeros(2)
        self.obj_pos = np.zeros(2)
        self.obj_vel = np.array([conveyor_speed, 0.0])
        self.grasp_active = False
        self.step_count = 0
        self.current_object_idx = 0
        self.objects_picked = 0
        self.object_active = False

        # Timing for object spawning
        self._spawn_interval = max_episode_steps // (n_objects_per_episode + 1)
        self._next_spawn_step = 10
        self._grasp_contact_steps = 0

    def _spawn_object(self):
        """Spawn a new object on the left side."""
        self.obj_pos = np.array([
            self.workspace_x[0] + 0.1,
            self.np_random.uniform(self.workspace_y[0] + 0.1, self.workspace_y[1] - 0.1),
        ])
        self.obj_vel = np.array([self.conveyor_speed, 0.0])
        self.object_active = True

    def _get_state_obs(self) -> np.ndarray:
        """Get state observation."""
        if self.object_active:
            obj_state = np.array([self.obj_pos[0], self.obj_pos[1], self.obj_vel[0], self.obj_vel[1]])
        else:
            obj_state = np.zeros(4)

        time_norm = self.step_count / self.max_episode_steps
        return np.array([
            self.ee_pos[0], self.ee_pos[1],
            obj_state[0], obj_state[1], obj_state[2], obj_state[3],
            float(self.grasp_active),
            time_norm,
        ], dtype=np.float32)

    def _get_pixel_obs(self) -> np.ndarray:
        """Render workspace."""
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 240

        def to_pixel(pos):
            px = int((pos[0] - self.workspace_x[0]) / (self.workspace_x[1] - self.workspace_x[0]) * self.image_size)
            py = int((pos[1] - self.workspace_y[0]) / (self.workspace_y[1] - self.workspace_y[0]) * self.image_size)
            return np.clip(px, 0, self.image_size - 1), np.clip(py, 0, self.image_size - 1)

        # Draw graspable region (green band)
        x_min_px, _ = to_pixel(np.array([self.graspable_region[0], 0]))
        x_max_px, _ = to_pixel(np.array([self.graspable_region[1], 0]))
        img[:, x_min_px:x_max_px] = img[:, x_min_px:x_max_px] * 0.8 + np.array([230, 255, 230]) * 0.2

        if self.object_active:
            obj_px, obj_py = to_pixel(self.obj_pos)
            radius = max(3, int(0.04 * self.image_size))
            y_min = max(0, obj_py - radius)
            y_max = min(self.image_size, obj_py + radius + 1)
            x_min = max(0, obj_px - radius)
            x_max = min(self.image_size, obj_px + radius + 1)
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if (x - obj_px)**2 + (y - obj_py)**2 <= radius**2:
                        img[y, x] = [30, 30, 200]

        ee_px, ee_py = to_pixel(self.ee_pos)
        ee_radius = max(4, int(0.05 * self.image_size))
        ee_color = [200, 30, 30] if self.grasp_active else [200, 100, 100]
        y_min = max(0, ee_py - ee_radius)
        y_max = min(self.image_size, ee_py + ee_radius + 1)
        x_min = max(0, ee_px - ee_radius)
        x_max = min(self.image_size, ee_px + ee_radius + 1)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - ee_px)**2 + (y - ee_py)**2 <= ee_radius**2:
                    img[y, x] = ee_color

        return img

    def _get_obs(self):
        if self.obs_type == "state":
            return self._get_state_obs()
        elif self.obs_type == "pixels":
            return self._get_pixel_obs()
        elif self.obs_type == "pixels_agent_pos":
            return {"pixels": self._get_pixel_obs(), "agent_pos": self._get_state_obs()}

    def _estimate_interaction_delay(self) -> float:
        fixed_delay = 0.02
        dist = np.linalg.norm(self.ee_pos - self.obj_pos) if self.object_active else 0.5
        return fixed_delay + dist / self.max_ee_speed + self.execution_delay

    def get_interaction_state(self) -> np.ndarray:
        if not self.object_active:
            return self.obj_pos.copy()
        delay = self._estimate_interaction_delay()
        future_pos = self.obj_pos + self.obj_vel * delay
        return np.clip(future_pos, self.workspace_x[0] + 0.1, self.workspace_x[1] - 0.1)

    def get_full_state(self) -> dict:
        return {
            "ee_pos": self.ee_pos.copy(),
            "obj_pos": self.obj_pos.copy() if self.object_active else None,
            "obj_vel": self.obj_vel.copy() if self.object_active else None,
            "grasp_active": self.grasp_active,
            "interaction_state": self.get_interaction_state(),
            "estimated_delay": self._estimate_interaction_delay(),
            "objects_picked": self.objects_picked,
            "object_active": self.object_active,
            "step": self.step_count,
        }

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Execution delay simulation
        if self.execution_delay > 0 and self.object_active:
            self.obj_pos = self.obj_pos + self.obj_vel * self.execution_delay
            self.obj_pos[0] = np.clip(self.obj_pos[0], self.workspace_x[0] + 0.1, self.workspace_x[1] - 0.1)

        # Move EE
        ee_velocity = action[:2] * self.max_ee_speed
        self.ee_pos = self.ee_pos + ee_velocity * self.dt
        self.ee_pos[0] = np.clip(self.ee_pos[0], self.workspace_x[0] + 0.05, self.workspace_x[1] - 0.05)
        self.ee_pos[1] = np.clip(self.ee_pos[1], self.workspace_y[0] + 0.05, self.workspace_y[1] - 0.05)

        self.grasp_active = action[2] > 0.5

        # Update object position
        if self.object_active:
            self.obj_pos = self.obj_pos + self.obj_vel * self.dt
            # Check if object exits workspace
            if self.obj_pos[0] > self.workspace_x[1] - 0.1:
                self.object_active = False

        # Compute reward
        reward = 0.0
        terminated = False

        if self.object_active:
            dist = np.linalg.norm(self.ee_pos - self.obj_pos)
            if self.grasp_active and dist < self.success_threshold:
                self._grasp_contact_steps += 1
                if self._grasp_contact_steps >= 2:
                    reward = 5.0
                    self.objects_picked += 1
                    self.object_active = False
                    if self.objects_picked >= self.n_objects_per_episode:
                        terminated = True
            else:
                self._grasp_contact_steps = 0
                if self.reward_type == "shaped" and self.object_active:
                    reward = max(0, 1.0 - dist) * 0.05 - 0.01

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        # Spawn new objects
        if not self.object_active and self.step_count >= self._next_spawn_step and self.current_object_idx < self.n_objects_per_episode:
            self._spawn_object()
            self.current_object_idx += 1
            self._next_spawn_step = self.step_count + self._spawn_interval

        info = {"full_state": self.get_full_state(), "action": action.copy()}
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ee_pos = self.np_random.uniform(-0.3, 0.3, size=2)
        self.grasp_active = False
        self.step_count = 0
        self.current_object_idx = 0
        self.objects_picked = 0
        self.object_active = False
        self._grasp_contact_steps = 0
        self._next_spawn_step = 10

        # Spawn first object
        self._spawn_object()
        self.current_object_idx = 1
        self._next_spawn_step = self._spawn_interval

        info = {"full_state": self.get_full_state()}
        return self._get_obs(), info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_pixel_obs()
        return self._get_pixel_obs()

    def close(self):
        pass


class ConveyorPickSlow(ConveyorPickEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("conveyor_speed", 0.1)
        super().__init__(**kwargs)


class ConveyorPickFast(ConveyorPickEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("conveyor_speed", 0.35)
        super().__init__(**kwargs)