#!/usr/bin/env python3
# Copyright 2026 The Robot2026 Authors. All rights reserved.
"""Test script for dynamic manipulation environments.

Usage:
    cd /home/lql/Code
    conda activate lerobot
    python scripts/test_env.py
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dynamic_manipulation.moving_grasp import (
    MovingGraspEnv, MovingGraspEasy, MovingGraspMedium, MovingGraspHard, MovingGraspWithDelay,
)
from envs.dynamic_manipulation.conveyor_pick import ConveyorPickEnv, ConveyorPickSlow, ConveyorPickFast
from envs.dynamic_manipulation.expert_policy import (
    MovingGraspExpert, ConveyorPickExpert, collect_dataset,
)


def test_moving_grasp_env():
    """Test MovingGraspEnv basic functionality."""
    print("=" * 60)
    print("Testing MovingGraspEnv")
    print("=" * 60)

    for pattern in ["linear", "sinusoidal", "circular", "bounce", "random_walk"]:
        env = MovingGraspEnv(
            target_speed=0.15,
            motion_pattern=pattern,
            obs_type="state",
            reward_type="shaped",
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,), f"Expected obs shape (7,), got {obs.shape}"
        assert "full_state" in info

        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            assert obs.shape == (7,)
            if terminated or truncated:
                break

        print(f"  Pattern '{pattern}': obs_shape={obs.shape}, reward={total_reward:.2f}, "
              f"delay={info['full_state']['estimated_delay']:.3f}s")
        env.close()

    print("  ✓ All motion patterns work correctly\n")


def test_conveyor_pick_env():
    """Test ConveyorPickEnv basic functionality."""
    print("=" * 60)
    print("Testing ConveyorPickEnv")
    print("=" * 60)

    env = ConveyorPickEnv(obs_type="state", conveyor_speed=0.2)
    obs, info = env.reset(seed=42)
    print(f"  Obs shape: {obs.shape}")
    print(f"  Full state keys: {list(info['full_state'].keys())}")

    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  Random policy reward: {total_reward:.2f}")
    print("  ✓ ConveyorPickEnv works correctly\n")


def test_expert_policy():
    """Test expert policy with oracle prediction."""
    print("=" * 60)
    print("Testing Expert Policy")
    print("=" * 60)

    # Test with different motion patterns
    configs = [
        ("Easy (slow linear)", MovingGraspEasy()),
        ("Medium (sinusoidal)", MovingGraspMedium()),
        ("Hard (fast random)", MovingGraspHard()),
        ("With delay", MovingGraspWithDelay()),
    ]

    for name, env in configs:
        expert = MovingGraspExpert(env, prediction_mode="oracle")
        demos = collect_dataset(env, expert, n_episodes=5, verbose=False)

        rewards = [d["total_reward"] for d in demos]
        successes = sum(1 for d in demos if d["success"])
        lengths = [d["episode_length"] for d in demos]

        print(f"  {name}:")
        print(f"    Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"    Success: {successes}/5")
        print(f"    Avg steps: {np.mean(lengths):.1f}")

        # Verify data shapes
        for d in demos:
            assert d["actions"].shape[1] == 3, f"Action dim should be 3, got {d['actions'].shape[1]}"
            if d["interaction_states"] is not None:
                assert d["interaction_states"].shape[1] == 2, "Interaction state should be 2D"
        env.close()

    print("  ✓ Expert policy works correctly\n")


def test_supervision_misalignment():
    """Demonstrate the supervision misalignment phenomenon."""
    print("=" * 60)
    print("Demonstrating Supervision Temporal Misalignment")
    print("=" * 60)

    env = MovingGraspWithDelay()  # With delay to amplify the effect
    expert = MovingGraspExpert(env, prediction_mode="oracle")

    obs, info = env.reset(seed=42)
    full_state = info["full_state"]

    print(f"\n  Execution delay: {env.execution_delay*1000:.0f}ms")
    print(f"  Target speed: {env.target_speed:.2f} units/s")

    misalignment_dists = []
    for step in range(50):
        action = expert.predict(obs, full_state=full_state)

        # Current observation: target at obj_pos
        obj_pos_current = full_state["obj_pos"].copy()

        # Interaction state: where target will be
        interaction_state = full_state["interaction_state"].copy()

        # The misalignment distance
        misalignment = np.linalg.norm(interaction_state - obj_pos_current)
        misalignment_dists.append(misalignment)

        obs, reward, terminated, truncated, info = env.step(action)
        full_state = info["full_state"]
        if terminated or truncated:
            break

    avg_misalignment = np.mean(misalignment_dists)
    max_misalignment = np.max(misalignment_dists)
    print(f"\n  Average supervision misalignment: {avg_misalignment:.4f} workspace units")
    print(f"  Maximum supervision misalignment: {max_misalignment:.4f} workspace units")
    print(f"  Success threshold: {env.success_threshold:.4f} workspace units")
    print(f"\n  → Misalignment is {avg_misalignment/env.success_threshold:.1f}x the success threshold!")
    print(f"  → Standard BC would train π(s_t) ≈ a_t, but a_t was aimed at s_{{t+Δt}}")
    print(f"  → This is the core supervision temporal misalignment problem.\n")

    env.close()


def test_pixel_obs():
    """Test pixel observation mode."""
    print("=" * 60)
    print("Testing Pixel Observations")
    print("=" * 60)

    env = MovingGraspEnv(obs_type="pixels", image_size=64)
    obs, info = env.reset(seed=42)
    assert obs.shape == (64, 64, 3), f"Expected (64,64,3), got {obs.shape}"
    print(f"  Pixel obs shape: {obs.shape}, dtype: {obs.dtype}")

    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs.shape == (64, 64, 3)
    print("  ✓ Pixel observations work correctly\n")
    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Dynamic Manipulation Environment Test Suite")
    print("=" * 60 + "\n")

    test_moving_grasp_env()
    test_conveyor_pick_env()
    test_expert_policy()
    test_supervision_misalignment()
    test_pixel_obs()

    print("=" * 60)
    print("  All tests passed! ✓")
    print("=" * 60)