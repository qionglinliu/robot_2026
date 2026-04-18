# Copyright 2026 The Robot2026 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Moving Target Grasp Environment.

A 2D environment where a robot end-effector must grasp a moving target.
This environment is designed to demonstrate the supervision temporal misalignment
problem in few-shot dynamic manipulation.

Key properties:
- Target moves with configurable speed and motion pattern
- Robot has a fixed perception-inference-execution delay
- Observation at time t sees target at s_t, but action executes when target is at s_{t+Δt}
- Success requires the robot to predict where the target will be, not where it is

Observation modes:
- "state": Low-dimensional state vector [ee_x, ee_y, obj_x, obj_y, obj_vx, obj_vy, grasp]
- "pixels": Top-down RGB image of the workspace
- "pixels_agent_pos": Both image and state

Action space:
- Continuous: [dx, dy, grasp] where dx, dy are EE velocity commands, grasp is binary
"""

from __future__ import annotations

import enum
import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MotionPattern(enum.Enum):
    """Motion patterns for the target object."""
    LINEAR = "linear"           # Constant velocity in one direction
    SINUSOIDAL = "sinusoidal"   # Sinusoidal trajectory
    CIRCULAR = "circular"       # Circular trajectory
    RANDOM_WALK = "random_walk" # Random walk with momentum
    BOUNCE = "bounce"           # Bouncing off walls


class MovingGraspEnv(gym.Env):
    """Moving Target Grasp Environment.

    A 2D workspace where a point robot must intercept and grasp a moving target.

    The workspace is a 2D box [-1, 1] x [-1, 1]. The robot end-effector (EE) starts
    at a random position, and a target object moves according to a specified motion
    pattern. The robot must move its EE to intercept the target and activate the
    gripper at the right time.

    The supervision misalignment arises because:
    - At time t, the agent observes target at position s_t with velocity v_t
    - The action a_t is selected based on this observation
    - Due to inherent delay Δt, the action only takes effect at time t + Δt
    - At that point, the target has moved to s_{t + Δt} ≈ s_t + v_t * Δt
    - Standard BC trains π(s_t) ≈ a_t, but a_t was actually aimed at s_{t+Δt}

    Args:
        max_episode_steps: Maximum number of steps per episode.
        fps: Frames per second (controls dt = 1/fps).
        target_speed: Speed of the target object in workspace units per second.
        motion_pattern: How the target moves (linear, sinusoidal, circular, etc.).
        execution_delay: Simulated delay in seconds between observation and action execution.
        obs_type: Observation mode ("state", "pixels", or "pixels_agent_pos").
        image_size: Size of rendered images (only used with pixel observations).
        success_threshold: Distance threshold for successful grasp.
        reward_type: "sparse" (0/1) or "shaped" (distance-based shaping).
        seed: Random seed.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "MovingGrasp-v0",
    }

    def __init__(
        self,
        max_episode_steps: int = 300,
        fps: int = 30,
        target_speed: float = 0.15,
        motion_pattern: str = "linear",
        execution_delay: float = 0.0,  # seconds of simulated delay
        obs_type: str = "state",
        image_size: int = 128,
        success_threshold: float = 0.05,
        reward_type: str = "shaped",
        render_mode: str | None = None,
        disable_env_checker: bool = True,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.fps = fps
        self.dt = 1.0 / fps
        self.target_speed = target_speed
        self.motion_pattern = MotionPattern(motion_pattern)
        self.execution_delay = execution_delay
        self.obs_type = obs_type
        self.image_size = image_size
        self.success_threshold = success_threshold
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Workspace bounds
        self.workspace_min = -1.0
        self.workspace_max = 1.0

        # Robot properties
        self.max_ee_speed = 0.5  # Max EE speed in workspace units/s
        self.grasp_threshold = 0.05  # Distance to activate grasp

        # Action: [dx, dy, grasp_signal]
        # dx, dy: normalized velocity commands in [-1, 1], scaled by max_ee_speed
        # grasp_signal: continuous in [0, 1], thresholded at 0.5
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space
        if obs_type == "state":
            # [ee_x, ee_y, obj_x, obj_y, obj_vx, obj_vy, grasp_active]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        elif obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        elif obs_type == "pixels_agent_pos":
            # Dict with image and state
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(
                    low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            })
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        # State variables
        self.ee_pos = np.zeros(2, dtype=np.float64)  # End-effector position
        self.obj_pos = np.zeros(2, dtype=np.float64)  # Object position
        self.obj_vel = np.zeros(2, dtype=np.float64)  # Object velocity
        self.grasp_active = False
        self.step_count = 0
        self.episode_reward = 0.0

        # Motion pattern state
        self._motion_time = 0.0
        self._motion_phase = np.random.uniform(0, 2 * np.pi, size=2)
        self._motion_center = np.zeros(2)
        self._motion_amplitude = np.zeros(2)
        self._initial_obj_pos = np.zeros(2)
        self._initial_obj_vel = np.zeros(2)
        self._random_walk_accel = np.zeros(2)

        # History buffer for potential temporal observations
        self._obj_pos_history = []
        self._obj_vel_history = []
        self._ee_pos_history = []

        # Grasp success tracking
        self._grasp_contact_steps = 0
        self._grasp_required_steps = 3  # Need to hold for N steps

    def _init_motion_pattern(self):
        """Initialize motion pattern parameters."""
        margin = 0.3
        if self.motion_pattern == MotionPattern.LINEAR:
            # Random direction, constant speed
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.obj_vel = self.target_speed * np.array([np.cos(angle), np.sin(angle)])
            self.obj_pos = self.np_random.uniform(-0.5, 0.5, size=2)

        elif self.motion_pattern == MotionPattern.SINUSOIDAL:
            # Sinusoidal trajectory along a primary axis
            self.obj_pos = self.np_random.uniform(-0.3, 0.3, size=2)
            primary_axis = self.np_random.integers(0, 2)
            self._motion_amplitude = np.zeros(2)
            self._motion_amplitude[primary_axis] = self.np_random.uniform(0.2, 0.5)
            self._motion_center = self.obj_pos.copy()
            self.obj_vel = np.zeros(2)

        elif self.motion_pattern == MotionPattern.CIRCULAR:
            # Circular trajectory
            center = self.np_random.uniform(-0.3, 0.3, size=2)
            radius = self.np_random.uniform(0.2, 0.4)
            self._motion_center = center
            self._motion_amplitude = np.array([radius, radius])
            self.obj_pos = center + radius * np.array([
                np.cos(self._motion_phase[0]), np.sin(self._motion_phase[0])
            ])
            self.obj_vel = np.zeros(2)

        elif self.motion_pattern == MotionPattern.RANDOM_WALK:
            # Random walk with momentum
            self.obj_pos = self.np_random.uniform(-0.3, 0.3, size=2)
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.obj_vel = self.target_speed * np.array([np.cos(angle), np.sin(angle)])
            self._random_walk_accel = np.zeros(2)

        elif self.motion_pattern == MotionPattern.BOUNCE:
            # Bouncing off walls
            self.obj_pos = self.np_random.uniform(-0.5, 0.5, size=2)
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.obj_vel = self.target_speed * np.array([np.cos(angle), np.sin(angle)])

        self._initial_obj_pos = self.obj_pos.copy()
        self._initial_obj_vel = self.obj_vel.copy()

    def _update_motion(self):
        """Update target object position based on motion pattern."""
        self._motion_time += self.dt

        if self.motion_pattern == MotionPattern.LINEAR:
            self.obj_pos = self.obj_pos + self.obj_vel * self.dt
            # Bounce off walls to keep in workspace
            for i in range(2):
                if self.obj_pos[i] < self.workspace_min + 0.1:
                    self.obj_pos[i] = self.workspace_min + 0.1
                    self.obj_vel[i] = abs(self.obj_vel[i])
                elif self.obj_pos[i] > self.workspace_max - 0.1:
                    self.obj_pos[i] = self.workspace_max - 0.1
                    self.obj_vel[i] = -abs(self.obj_vel[i])

        elif self.motion_pattern == MotionPattern.SINUSOIDAL:
            freq = self.target_speed / (2 * np.pi * max(self._motion_amplitude.max(), 0.01))
            phase = self._motion_phase + 2 * np.pi * freq * self._motion_time
            old_pos = self.obj_pos.copy()
            self.obj_pos = self._motion_center + self._motion_amplitude * np.sin(phase)
            self.obj_vel = (self.obj_pos - old_pos) / self.dt

        elif self.motion_pattern == MotionPattern.CIRCULAR:
            angular_speed = self.target_speed / max(self._motion_amplitude[0], 0.01)
            angle = self._motion_phase[0] + angular_speed * self._motion_time
            old_pos = self.obj_pos.copy()
            self.obj_pos = self._motion_center + self._motion_amplitude[0] * np.array([
                np.cos(angle), np.sin(angle)
            ])
            self.obj_vel = (self.obj_pos - old_pos) / self.dt

        elif self.motion_pattern == MotionPattern.RANDOM_WALK:
            # Random acceleration with momentum
            noise = self.np_random.normal(0, 1.0, size=2) * self.target_speed * 2.0
            self._random_walk_accel = 0.9 * self._random_walk_accel + 0.1 * noise
            self.obj_vel = self.obj_vel + self._random_walk_accel * self.dt
            # Limit speed
            speed = np.linalg.norm(self.obj_vel)
            if speed > self.target_speed * 1.5:
                self.obj_vel = self.obj_vel / speed * self.target_speed * 1.5
            self.obj_pos = self.obj_pos + self.obj_vel * self.dt
            # Bounce off walls
            for i in range(2):
                if self.obj_pos[i] < self.workspace_min + 0.1:
                    self.obj_pos[i] = self.workspace_min + 0.1
                    self.obj_vel[i] = abs(self.obj_vel[i])
                elif self.obj_pos[i] > self.workspace_max - 0.1:
                    self.obj_pos[i] = self.workspace_max - 0.1
                    self.obj_vel[i] = -abs(self.obj_vel[i])

        elif self.motion_pattern == MotionPattern.BOUNCE:
            self.obj_pos = self.obj_pos + self.obj_vel * self.dt
            # Bounce off walls
            for i in range(2):
                if self.obj_pos[i] < self.workspace_min + 0.1:
                    self.obj_pos[i] = self.workspace_min + 0.1
                    self.obj_vel[i] = abs(self.obj_vel[i])
                elif self.obj_pos[i] > self.workspace_max - 0.1:
                    self.obj_pos[i] = self.workspace_max - 0.1
                    self.obj_vel[i] = -abs(self.obj_vel[i])

        # Store history
        self._obj_pos_history.append(self.obj_pos.copy())
        self._obj_vel_history.append(self.obj_vel.copy())

    def _get_state_obs(self) -> np.ndarray:
        """Get low-dimensional state observation."""
        return np.array([
            self.ee_pos[0], self.ee_pos[1],
            self.obj_pos[0], self.obj_pos[1],
            self.obj_vel[0], self.obj_vel[1],
            float(self.grasp_active),
        ], dtype=np.float32)

    def _get_pixel_obs(self) -> np.ndarray:
        """Render the workspace as a top-down image."""
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 240  # Light gray bg

        # Draw workspace boundary
        img[0:2, :] = [200, 200, 200]
        img[-2:, :] = [200, 200, 200]
        img[:, 0:2] = [200, 200, 200]
        img[:, -2:] = [200, 200, 200]

        # Convert workspace coords to pixel coords
        def to_pixel(pos):
            px = int((pos[0] - self.workspace_min) / (self.workspace_max - self.workspace_min) * self.image_size)
            py = int((pos[1] - self.workspace_min) / (self.workspace_max - self.workspace_min) * self.image_size)
            px = np.clip(px, 0, self.image_size - 1)
            py = np.clip(py, 0, self.image_size - 1)
            return px, py

        # Draw target object (blue circle)
        obj_px, obj_py = to_pixel(self.obj_pos)
        radius = max(3, int(0.04 * self.image_size))
        y_min = max(0, obj_py - radius)
        y_max = min(self.image_size, obj_py + radius + 1)
        x_min = max(0, obj_px - radius)
        x_max = min(self.image_size, obj_px + radius + 1)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - obj_px)**2 + (y - obj_py)**2 <= radius**2:
                    img[y, x] = [30, 30, 200]  # Blue

        # Draw velocity arrow (green)
        vel_scale = 30  # Scale factor for velocity visualization
        end_px = int(obj_px + self.obj_vel[0] * vel_scale)
        end_py = int(obj_py + self.obj_vel[1] * vel_scale)
        end_px = np.clip(end_px, 0, self.image_size - 1)
        end_py = np.clip(end_py, 0, self.image_size - 1)
        # Simple line drawing
        length = max(abs(end_px - obj_px), abs(end_py - obj_py), 1)
        for i in range(int(length)):
            t = i / length
            ix = int(obj_px + t * (end_px - obj_px))
            iy = int(obj_py + t * (end_py - obj_py))
            if 0 <= iy < self.image_size and 0 <= ix < self.image_size:
                img[iy, ix] = [0, 200, 0]  # Green

        # Draw EE (red circle, larger if grasping)
        ee_px, ee_py = to_pixel(self.ee_pos)
        ee_radius = max(4, int(0.05 * self.image_size))
        if self.grasp_active:
            ee_color = [200, 30, 30]  # Red when grasping
        else:
            ee_color = [200, 100, 100]  # Light red when not grasping
        y_min = max(0, ee_py - ee_radius)
        y_max = min(self.image_size, ee_py + ee_radius + 1)
        x_min = max(0, ee_px - ee_radius)
        x_max = min(self.image_size, ee_px + ee_radius + 1)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - ee_px)**2 + (y - ee_py)**2 <= ee_radius**2:
                    img[y, x] = ee_color

        return img

    def _get_obs(self) -> np.ndarray | dict:
        """Get current observation."""
        if self.obs_type == "state":
            return self._get_state_obs()
        elif self.obs_type == "pixels":
            return self._get_pixel_obs()
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": self._get_pixel_obs(),
                "agent_pos": self._get_state_obs(),
            }

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Compute reward and termination condition.

        Returns:
            (reward, terminated)
        """
        dist = np.linalg.norm(self.ee_pos - self.obj_pos)
        grasp_signal = action[2] > 0.5

        if self.reward_type == "sparse":
            # Sparse reward: only reward on successful grasp
            if dist < self.success_threshold and grasp_signal:
                self._grasp_contact_steps += 1
                if self._grasp_contact_steps >= self._grasp_required_steps:
                    return 1.0, True
            else:
                self._grasp_contact_steps = 0
            return 0.0, False

        elif self.reward_type == "shaped":
            # Shaped reward
            reward = 0.0

            # Distance reward: closer is better
            reward += max(0, 1.0 - dist) * 0.1

            # Grasp reward: bonus for being close AND grasping
            if dist < self.success_threshold and grasp_signal:
                self._grasp_contact_steps += 1
                reward += 0.3
                if self._grasp_contact_steps >= self._grasp_required_steps:
                    reward += 10.0  # Big success bonus
                    return reward, True
            else:
                self._grasp_contact_steps = max(0, self._grasp_contact_steps - 1)

            # Penalty for grasping when far away
            if grasp_signal and dist > self.success_threshold * 3:
                reward -= 0.1

            # Small time penalty
            reward -= 0.01

            return reward, False

    def _estimate_interaction_delay(self) -> float:
        """Estimate the interaction delay based on current state.

        The delay depends on:
        1. Fixed perception + inference delay
        2. Variable execution delay based on EE-to-target distance

        Returns:
            Estimated delay in seconds
        """
        fixed_delay = 0.02  # 20ms perception + inference
        dist = np.linalg.norm(self.ee_pos - self.obj_pos)
        execution_time = dist / self.max_ee_speed
        return fixed_delay + execution_time + self.execution_delay

    def get_interaction_state(self) -> np.ndarray:
        """Get the future interaction state (where the target will be when action executes).

        This is the 'ground truth' interaction state that expert actions are conditioned on.

        Returns:
            Predicted target position at interaction time: [x, y]
        """
        delay = self._estimate_interaction_delay()
        # Simple linear extrapolation (ground truth uses actual velocity)
        future_pos = self.obj_pos + self.obj_vel * delay
        # Clamp to workspace
        future_pos = np.clip(
            future_pos, self.workspace_min + 0.1, self.workspace_max - 0.1
        )
        return future_pos

    def get_full_state(self) -> dict:
        """Get the full environment state for data collection and analysis.

        Returns a dictionary with all state information including the interaction state,
        which is crucial for studying the supervision misalignment problem.
        """
        interaction_state = self.get_interaction_state()
        estimated_delay = self._estimate_interaction_delay()
        return {
            "ee_pos": self.ee_pos.copy(),
            "obj_pos": self.obj_pos.copy(),
            "obj_vel": self.obj_vel.copy(),
            "grasp_active": self.grasp_active,
            "interaction_state": interaction_state,
            "estimated_delay": estimated_delay,
            "ee_to_obj_dist": np.linalg.norm(self.ee_pos - self.obj_pos),
            "ee_to_interaction_dist": np.linalg.norm(self.ee_pos - interaction_state),
            "step": self.step_count,
            "time": self.step_count * self.dt,
        }

    def step(self, action: np.ndarray):
        """Execute one step in the environment.

        Args:
            action: [dx, dy, grasp_signal] where dx, dy ∈ [-1, 1] and grasp_signal ∈ [0, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply execution delay: simulate that the action was decided 'execution_delay' ago
        # The target has moved during this delay
        if self.execution_delay > 0:
            # Advance target by execution_delay to simulate the actual interaction state
            old_pos = self.obj_pos.copy()
            # Move target forward by the delay
            self.obj_pos = self.obj_pos + self.obj_vel * self.execution_delay
            self.obj_pos = np.clip(
                self.obj_pos, self.workspace_min + 0.1, self.workspace_max - 0.1
            )

        # Move EE according to action
        ee_velocity = action[:2] * self.max_ee_speed
        self.ee_pos = self.ee_pos + ee_velocity * self.dt
        self.ee_pos = np.clip(
            self.ee_pos, self.workspace_min + 0.05, self.workspace_max - 0.05
        )

        # Update grasp state
        self.grasp_active = action[2] > 0.5

        # Store EE history
        self._ee_pos_history.append(self.ee_pos.copy())

        # Update target motion (after execution delay simulation)
        self._update_motion()

        # Compute reward
        reward, terminated = self._compute_reward(action)

        # Check truncation
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        # Info dict with rich state information for analysis
        info = {
            "full_state": self.get_full_state(),
            "action": action.copy(),
        }

        self.episode_reward += reward

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray | dict, dict]:
        """Reset the environment.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Initialize EE at a random position (away from center)
        self.ee_pos = self.np_random.uniform(-0.8, -0.3, size=2)
        if self.np_random.random() > 0.5:
            self.ee_pos[0] = self.np_random.uniform(0.3, 0.8)

        # Initialize target motion
        self._motion_time = 0.0
        self._motion_phase = self.np_random.uniform(0, 2 * np.pi, size=2)
        self._init_motion_pattern()

        # Reset state
        self.grasp_active = False
        self.step_count = 0
        self.episode_reward = 0.0
        self._grasp_contact_steps = 0
        self._obj_pos_history = [self.obj_pos.copy()]
        self._obj_vel_history = [self.obj_vel.copy()]
        self._ee_pos_history = [self.ee_pos.copy()]

        info = {
            "full_state": self.get_full_state(),
        }

        obs = self._get_obs()
        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._get_pixel_obs()
        elif self.render_mode == "human":
            # For human rendering, we just return the pixel array
            # (actual display would require pygame/matplotlib)
            return self._get_pixel_obs()

    def close(self):
        """Clean up resources."""
        pass


class MovingGraspEasy(MovingGraspEnv):
    """Easy variant: slow target speed."""
    def __init__(self, **kwargs):
        kwargs.setdefault("target_speed", 0.05)
        kwargs.setdefault("motion_pattern", "linear")
        super().__init__(**kwargs)


class MovingGraspMedium(MovingGraspEnv):
    """Medium variant: moderate target speed with sinusoidal motion."""
    def __init__(self, **kwargs):
        kwargs.setdefault("target_speed", 0.15)
        kwargs.setdefault("motion_pattern", "sinusoidal")
        super().__init__(**kwargs)


class MovingGraspHard(MovingGraspEnv):
    """Hard variant: fast target with random walk."""
    def __init__(self, **kwargs):
        kwargs.setdefault("target_speed", 0.30)
        kwargs.setdefault("motion_pattern", "random_walk")
        super().__init__(**kwargs)


class MovingGraspWithDelay(MovingGraspEnv):
    """Variant with explicit execution delay to amplify the misalignment."""
    def __init__(self, **kwargs):
        kwargs.setdefault("target_speed", 0.15)
        kwargs.setdefault("execution_delay", 0.1)  # 100ms delay
        super().__init__(**kwargs)