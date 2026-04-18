# Copyright 2026 The Robot2026 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Scripted Expert Policies for Dynamic Manipulation Environments.

These experts use ground-truth state information to generate high-quality
demonstrations. They implement "oracle" strategies that account for the
interaction delay, making them ideal for studying the supervision misalignment.

Key insight: The expert's actions are conditioned on the FUTURE interaction state
(where the target will be when the action executes), NOT on the current observation.
This creates the supervision misalignment in standard BC.
"""

from __future__ import annotations

import numpy as np


class MovingGraspExpert:
    """Expert policy for MovingGraspEnv.

    Uses a predictive intercept strategy:
    1. Estimate where the target will be at interaction time (s_{t+Δt})
    2. Move EE toward that predicted position
    3. Activate grasp when close enough

    The expert naturally demonstrates the "interaction-aware" decision rule.
    When standard BC is applied to these demonstrations, it tries to learn
    π(s_t) ≈ a_t, but a_t was actually generated as π*(s_{t+Δt}).
    """

    def __init__(
        self,
        env,
        prediction_mode: str = "oracle",
        noise_scale: float = 0.0,
        delay_offset: float = 0.0,
    ):
        """
        Args:
            env: The MovingGraspEnv instance.
            prediction_mode:
                - "oracle": Use ground-truth velocity for prediction (perfect expert)
                - "linear": Use linear extrapolation from recent positions
                - "none": No prediction, just chase current position (baseline expert)
            noise_scale: Add Gaussian noise to actions (for diversity).
            delay_offset: Additional delay to account for (simulates conservative prediction).
        """
        self.env = env
        self.prediction_mode = prediction_mode
        self.noise_scale = noise_scale
        self.delay_offset = delay_offset

    def predict(
        self,
        obs: np.ndarray,
        full_state: dict | None = None,
    ) -> np.ndarray:
        """Select action based on the expert strategy.

        Args:
            obs: Current observation.
            full_state: Full environment state (for oracle prediction).

        Returns:
            action: [dx, dy, grasp_signal]
        """
        if full_state is None:
            # Fallback: extract from observation
            ee_pos = obs[:2]
            obj_pos = obs[2:4]
            obj_vel = obs[4:6]
        else:
            ee_pos = full_state["ee_pos"]
            obj_pos = full_state["obj_pos"]
            obj_vel = full_state["obj_vel"]

        # Predict target position at interaction time
        if self.prediction_mode == "oracle":
            # Use ground-truth delay and velocity
            delay = self.env._estimate_interaction_delay() + self.delay_offset
            target_pos = obj_pos + obj_vel * delay
        elif self.prediction_mode == "linear":
            # Linear extrapolation using velocity estimate
            delay = self.env._estimate_interaction_delay() + self.delay_offset
            target_pos = obj_pos + obj_vel * delay
        elif self.prediction_mode == "none":
            # No prediction: aim at current position
            target_pos = obj_pos
        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")

        # Clamp to workspace
        target_pos = np.clip(target_pos, self.env.workspace_min + 0.1, self.env.workspace_max - 0.1)

        # Compute direction to target
        diff = target_pos - ee_pos
        dist = np.linalg.norm(diff)

        # Compute velocity command
        if dist > self.env.success_threshold:
            # Move toward predicted position
            direction = diff / (dist + 1e-6)
            speed = min(1.0, dist / 0.1)  # Ramp down when close
            dx = direction[0] * speed
            dy = direction[1] * speed
            grasp = 0.0  # Don't grasp while moving
        else:
            # Close enough: slow down and grasp
            dx = diff[0] * 2.0  # Fine positioning
            dy = diff[1] * 2.0
            dx = np.clip(dx, -0.3, 0.3)
            dy = np.clip(dy, -0.3, 0.3)
            grasp = 1.0  # Activate grasp

        action = np.array([dx, dy, grasp], dtype=np.float32)

        # Add noise for diversity
        if self.noise_scale > 0:
            noise = np.random.normal(0, self.noise_scale, size=3)
            noise[2] = 0  # Don't add noise to grasp signal
            action[:2] += noise[:2]

        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


class ConveyorPickExpert:
    """Expert policy for ConveyorPickEnv.

    Strategy:
    1. Wait for object to enter the optimal grasp zone
    2. Move to predicted intercept position
    3. Grasp at the right moment
    """

    def __init__(
        self,
        env,
        prediction_mode: str = "oracle",
        noise_scale: float = 0.0,
        intercept_x: float | None = None,
    ):
        self.env = env
        self.prediction_mode = prediction_mode
        self.noise_scale = noise_scale
        self.intercept_x = intercept_x  # Optimal x position to intercept

    def predict(self, obs: np.ndarray, full_state: dict | None = None) -> np.ndarray:
        if full_state is None:
            ee_pos = obs[:2]
            obj_pos = obs[2:4] if np.any(obs[2:4] != 0) else None
            obj_vel = obs[4:6] if np.any(obs[4:6] != 0) else np.array([self.env.conveyor_speed, 0.0])
        else:
            ee_pos = full_state["ee_pos"]
            obj_pos = full_state["obj_pos"]
            obj_vel = full_state["obj_vel"]

        if obj_pos is None:
            # No active object, stay at ready position
            ready_pos = np.array([0.0, 0.0])
            diff = ready_pos - ee_pos
            dist = np.linalg.norm(diff)
            if dist > 0.02:
                direction = diff / (dist + 1e-6)
                return np.array([direction[0] * 0.5, direction[1] * 0.5, 0.0], dtype=np.float32)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Predict intercept position
        if self.prediction_mode == "oracle":
            delay = self.env._estimate_interaction_delay()
            target_pos = obj_pos + obj_vel * delay
        elif self.prediction_mode == "none":
            target_pos = obj_pos
        else:
            target_pos = obj_pos

        # If intercept_x is set, aim for optimal intercept point
        if self.intercept_x is not None and obj_pos[0] < self.intercept_x:
            time_to_reach = (self.intercept_x - obj_pos[0]) / max(self.env.conveyor_speed, 0.01)
            target_pos = np.array([self.intercept_x, obj_pos[1]])

        diff = target_pos - ee_pos
        dist = np.linalg.norm(diff)

        if dist > self.env.success_threshold:
            direction = diff / (dist + 1e-6)
            speed = min(1.0, dist / 0.1)
            action = np.array([direction[0] * speed, direction[1] * speed, 0.0], dtype=np.float32)
        else:
            action = np.array([diff[0] * 2.0, diff[1] * 2.0, 1.0], dtype=np.float32)
            action[:2] = np.clip(action[:2], -0.3, 0.3)

        if self.noise_scale > 0:
            noise = np.random.normal(0, self.noise_scale, size=3)
            noise[2] = 0
            action[:2] += noise[:2]

        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


def collect_demonstration(
    env,
    expert,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict:
    """Collect a single demonstration episode.

    Returns:
        Dictionary containing:
        - observations: (T, obs_dim) array
        - actions: (T, action_dim) array
        - rewards: (T,) array
        - full_states: list of full state dicts at each step
        - interaction_states: (T, 2) array of future interaction states
        - delays: (T,) array of estimated delays
        - success: whether the episode was successful
        - total_reward: cumulative reward
    """
    if max_steps is None:
        max_steps = env.max_episode_steps

    observations = []
    actions = []
    rewards = []
    full_states = []
    interaction_states = []
    delays = []

    obs, info = env.reset(seed=seed)
    full_state = info.get("full_state", {})

    for step in range(max_steps):
        # Get expert action (with access to full state for oracle)
        action = expert.predict(obs, full_state=full_state)

        # Record pre-step state
        observations.append(obs if isinstance(obs, np.ndarray) else obs)
        actions.append(action)
        if full_state:
            interaction_states.append(full_state.get("interaction_state", np.zeros(2)))
            delays.append(full_state.get("estimated_delay", 0.0))
        full_states.append(full_state)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        full_state = info.get("full_state", {})
        rewards.append(reward)

        if terminated or truncated:
            # Record final observation
            observations.append(obs if isinstance(obs, np.ndarray) else obs)
            full_states.append(full_state)
            break

    # Convert to arrays where possible
    result = {
        "observations": observations,
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "full_states": full_states,
        "interaction_states": np.array(interaction_states, dtype=np.float32) if interaction_states else None,
        "delays": np.array(delays, dtype=np.float32) if delays else None,
        "success": sum(rewards) > 5.0,  # Threshold for success
        "total_reward": sum(rewards),
        "episode_length": len(actions),
    }

    return result


def collect_dataset(
    env,
    expert,
    n_episodes: int = 10,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Collect multiple demonstration episodes.

    Args:
        env: Environment instance.
        expert: Expert policy.
        n_episodes: Number of episodes to collect.
        seeds: Optional list of seeds (if None, uses 0, 1, 2, ...).
        verbose: Print progress.

    Returns:
        List of demonstration dicts.
    """
    if seeds is None:
        seeds = list(range(n_episodes))

    demos = []
    successes = 0
    for i, seed in enumerate(seeds):
        demo = collect_demonstration(env, expert, seed=seed)
        demos.append(demo)
        if demo["success"]:
            successes += 1
        if verbose:
            print(f"Episode {i+1}/{n_episodes}: "
                  f"reward={demo['total_reward']:.2f}, "
                  f"steps={demo['episode_length']}, "
                  f"success={demo['success']}")

    if verbose:
        print(f"\nSuccess rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")

    return demos