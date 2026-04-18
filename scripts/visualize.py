#!/usr/bin/env python3
# Copyright 2026 The Robot2026 Authors. All rights reserved.
"""Visualization script for dynamic manipulation environments.

Generates publication-quality figures demonstrating:
1. Environment trajectories with different motion patterns
2. Expert vs naive (no prediction) policy comparison
3. Supervision temporal misalignment visualization
4. Conveyor pick task visualization

Usage:
    cd /home/lql/Code
    conda activate lerobot
    python scripts/visualize.py
    # Output: experiments/results/figures/
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dynamic_manipulation.moving_grasp import (
    MovingGraspEnv, MovingGraspEasy, MovingGraspMedium, MovingGraspHard, MovingGraspWithDelay,
)
from envs.dynamic_manipulation.conveyor_pick import ConveyorPickEnv
from envs.dynamic_manipulation.expert_policy import (
    MovingGraspExpert, ConveyorPickExpert, collect_demonstration,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Color palette
# ============================================================
COLORS = {
    "ee": "#E63946",          # Red
    "obj": "#457B9D",         # Blue
    "interaction": "#2A9D8F", # Teal
    "expert": "#264653",      # Dark
    "naive": "#E9C46A",       # Yellow
    "success": "#2A9D8F",     # Teal
    "fail": "#E63946",        # Red
    "bg": "#F1FAEE",          # Light green
    "grid": "#A8DADC",        # Light blue
    "conveyor": "#6D6875",    # Gray
}


def collect_trajectory(env, expert, seed=42, max_steps=300):
    """Collect full trajectory data for visualization."""
    obs, info = env.reset(seed=seed)
    full_state = info["full_state"]

    data = {
        "ee_pos": [full_state["ee_pos"].copy()],
        "obj_pos": [full_state["obj_pos"].copy()],
        "obj_vel": [full_state["obj_vel"].copy()],
        "interaction_states": [full_state["interaction_state"].copy()],
        "delays": [full_state["estimated_delay"]],
        "actions": [],
        "rewards": [],
        "success": False,
    }

    for step in range(max_steps):
        action = expert.predict(obs, full_state=full_state)
        data["actions"].append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        full_state = info["full_state"]
        data["rewards"].append(reward)

        data["ee_pos"].append(full_state["ee_pos"].copy())
        data["obj_pos"].append(full_state["obj_pos"].copy())
        data["obj_vel"].append(full_state["obj_vel"].copy())
        data["interaction_states"].append(full_state["interaction_state"].copy())
        data["delays"].append(full_state["estimated_delay"])

        if terminated:
            data["success"] = True
            break
        if truncated:
            break

    for key in ["ee_pos", "obj_pos", "obj_vel", "interaction_states"]:
        data[key] = np.array(data[key])
    data["actions"] = np.array(data["actions"])
    data["rewards"] = np.array(data["rewards"])
    data["delays"] = np.array(data["delays"])
    return data


def draw_workspace(ax, env, title=""):
    """Draw workspace boundary and grid."""
    ax.set_xlim(env.workspace_min - 0.05, env.workspace_max + 0.05)
    ax.set_ylim(env.workspace_min - 0.05, env.workspace_max + 0.05)
    ax.set_aspect("equal")
    ax.set_facecolor(COLORS["bg"])

    # Workspace boundary
    rect = patches.Rectangle(
        (env.workspace_min, env.workspace_min),
        env.workspace_max - env.workspace_min,
        env.workspace_max - env.workspace_min,
        linewidth=2, edgecolor=COLORS["grid"], facecolor="none"
    )
    ax.add_patch(rect)

    # Grid
    for v in np.arange(-1.0, 1.1, 0.5):
        ax.axhline(y=v, color=COLORS["grid"], alpha=0.3, linewidth=0.5)
        ax.axvline(x=v, color=COLORS["grid"], alpha=0.3, linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


# ============================================================
# Figure 1: Motion Patterns + Expert Trajectories
# ============================================================
def fig1_motion_patterns():
    """Visualize all motion patterns with expert trajectories."""
    print("Generating Figure 1: Motion Patterns...")

    patterns = ["linear", "sinusoidal", "circular", "bounce", "random_walk"]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for ax, pattern in zip(axes, patterns):
        env = MovingGraspEnv(
            target_speed=0.15, motion_pattern=pattern,
            execution_delay=0.0, reward_type="shaped",
        )
        expert = MovingGraspExpert(env, prediction_mode="oracle")
        data = collect_trajectory(env, expert, seed=42, max_steps=200)

        draw_workspace(ax, env, title=pattern.replace("_", " ").title())

        # Object trajectory (colored by time)
        obj_traj = data["obj_pos"]
        n = len(obj_traj)
        for i in range(n - 1):
            alpha = 0.3 + 0.7 * i / max(n - 1, 1)
            ax.plot(obj_traj[i:i+2, 0], obj_traj[i:i+2, 1],
                    color=COLORS["obj"], alpha=alpha, linewidth=1.5)

        # EE trajectory
        ee_traj = data["ee_pos"]
        for i in range(len(ee_traj) - 1):
            alpha = 0.3 + 0.7 * i / max(len(ee_traj) - 1, 1)
            ax.plot(ee_traj[i:i+2, 0], ee_traj[i:i+2, 1],
                    color=COLORS["ee"], alpha=alpha, linewidth=1.5)

        # Start positions
        ax.plot(*obj_traj[0], "o", color=COLORS["obj"], markersize=8, label="Target start")
        ax.plot(*ee_traj[0], "s", color=COLORS["ee"], markersize=8, label="EE start")

        # End positions (grasp point)
        if data["success"]:
            ax.plot(*obj_traj[-1], "*", color=COLORS["success"], markersize=15,
                    zorder=5, label="Grasp ✓")
        else:
            ax.plot(*ee_traj[-1], "x", color=COLORS["fail"], markersize=10, label="Miss ✗")

        # Draw interaction state at final step
        if len(data["interaction_states"]) > 0:
            ax.plot(*data["interaction_states"][-1], "D", color=COLORS["interaction"],
                    markersize=6, alpha=0.7, label="Interaction state")

        env.close()

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Motion Patterns with Expert Trajectories", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_motion_patterns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 2: Expert vs Naive Comparison
# ============================================================
def fig2_expert_vs_naive():
    """Compare oracle expert (with prediction) vs naive expert (chasing current pos)."""
    print("Generating Figure 2: Expert vs Naive...")

    configs = [
        ("Easy", MovingGraspEasy(), 42),
        ("Medium", MovingGraspMedium(), 42),
        ("Hard", MovingGraspHard(), 42),
        ("With Delay (100ms)", MovingGraspWithDelay(), 42),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("Oracle Expert (top) vs Naive Expert (bottom)", fontsize=14, fontweight="bold")

    for col, (name, env, seed) in enumerate(configs):
        # Oracle expert
        expert = MovingGraspExpert(env, prediction_mode="oracle")
        data_oracle = collect_trajectory(env, expert, seed=seed, max_steps=300)

        # Naive expert (no prediction)
        naive = MovingGraspExpert(env, prediction_mode="none")
        data_naive = collect_trajectory(env, naive, seed=seed, max_steps=300)

        for row, (data, label) in enumerate([(data_oracle, "Oracle"), (data_naive, "Naive")]):
            ax = axes[row][col]
            draw_workspace(ax, env, title=f"{name}\n({label})")

            # Object trajectory
            obj_traj = data["obj_pos"]
            ax.plot(obj_traj[:, 0], obj_traj[:, 1],
                    color=COLORS["obj"], alpha=0.5, linewidth=1.5, label="Target")

            # EE trajectory
            ee_traj = data["ee_pos"]
            ax.plot(ee_traj[:, 0], ee_traj[:, 1],
                    color=COLORS["ee"], alpha=0.7, linewidth=1.5, label="EE")

            # Start and end
            ax.plot(*obj_traj[0], "o", color=COLORS["obj"], markersize=8)
            ax.plot(*ee_traj[0], "s", color=COLORS["ee"], markersize=8)

            if data["success"]:
                ax.plot(*ee_traj[-1], "*", color=COLORS["success"], markersize=15, zorder=5)
                ax.set_title(f"{name}\n({label}: ✓)", fontsize=11, color=COLORS["success"])
            else:
                ax.plot(*ee_traj[-1], "x", color=COLORS["fail"], markersize=10)
                ax.set_title(f"{name}\n({label}: ✗)", fontsize=11, color=COLORS["fail"])

        env.close()

    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_expert_vs_naive.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 3: Supervision Temporal Misalignment
# ============================================================
def fig3_supervision_misalignment():
    """Visualize the supervision misalignment phenomenon."""
    print("Generating Figure 3: Supervision Misalignment...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Supervision Temporal Misalignment", fontsize=14, fontweight="bold")

    # --- Panel A: Trajectory with misalignment vectors ---
    ax = axes[0]
    env = MovingGraspWithDelay()
    expert = MovingGraspExpert(env, prediction_mode="oracle")
    data = collect_trajectory(env, expert, seed=42, max_steps=100)

    draw_workspace(ax, env, title="(a) Observation vs Interaction State")

    # Plot trajectory
    ax.plot(data["obj_pos"][:, 0], data["obj_pos"][:, 1],
            color=COLORS["obj"], alpha=0.6, linewidth=2, label="Target trajectory")
    ax.plot(data["ee_pos"][:, 0], data["ee_pos"][:, 1],
            color=COLORS["ee"], alpha=0.6, linewidth=2, label="EE trajectory")

    # Draw misalignment arrows every N steps
    step = 5
    for i in range(0, min(len(data["obj_pos"]) - 1, 80), step):
        obj_p = data["obj_pos"][i]
        int_p = data["interaction_states"][i]
        dist = np.linalg.norm(int_p - obj_p)
        if dist > 0.005:
            ax.annotate("", xy=int_p, xytext=obj_p,
                        arrowprops=dict(arrowstyle="->", color=COLORS["interaction"],
                                       lw=1.5, alpha=0.8))
            ax.plot(*obj_p, "o", color=COLORS["obj"], markersize=4)
            ax.plot(*int_p, "D", color=COLORS["interaction"], markersize=4)

    # Legend entries
    ax.plot([], [], "-", color=COLORS["obj"], linewidth=2, label="Target (observed)")
    ax.plot([], [], "D", color=COLORS["interaction"], markersize=6, label="Interaction state (actual target)")
    ax.plot([], [], "->", color=COLORS["interaction"], linewidth=1.5, label="Misalignment")
    ax.legend(fontsize=8, loc="lower left")

    env.close()

    # --- Panel B: Misalignment distance over time ---
    ax = axes[1]
    speeds = [0.05, 0.15, 0.30]
    delays = [0.0, 0.05, 0.1]
    labels_speed = ["slow (0.05)", "medium (0.15)", "fast (0.30)"]
    labels_delay = ["no delay", "50ms delay", "100ms delay"]

    for i, (speed, label) in enumerate(zip(speeds, labels_speed)):
        env = MovingGraspEnv(target_speed=speed, motion_pattern="linear", execution_delay=0.0)
        expert = MovingGraspExpert(env, prediction_mode="oracle")
        data = collect_trajectory(env, expert, seed=42, max_steps=150)
        misalignments = np.linalg.norm(
            data["interaction_states"][:len(data["obj_vel"])] - data["obj_pos"][:len(data["obj_vel"])],
            axis=1
        )
        ax.plot(misalignments, label=f"Speed: {label}", alpha=0.8, linewidth=1.5)
        env.close()

    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Success threshold")
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Misalignment Distance", fontsize=11)
    ax.set_title("(b) Misalignment vs Target Speed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Misalignment with different delays ---
    ax = axes[2]
    for i, (delay, label) in enumerate(zip(delays, labels_delay)):
        env = MovingGraspEnv(target_speed=0.15, motion_pattern="linear", execution_delay=delay)
        expert = MovingGraspExpert(env, prediction_mode="oracle")
        data = collect_trajectory(env, expert, seed=42, max_steps=150)
        misalignments = np.linalg.norm(
            data["interaction_states"][:len(data["obj_vel"])] - data["obj_pos"][:len(data["obj_vel"])],
            axis=1
        )
        ax.plot(misalignments, label=f"{label}", alpha=0.8, linewidth=1.5)
        env.close()

    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Success threshold")
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Misalignment Distance", fontsize=11)
    ax.set_title("(c) Misalignment vs Execution Delay", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_supervision_misalignment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 4: Success Rate Heatmap (Speed × Delay)
# ============================================================
def fig4_success_heatmap():
    """Heatmap of success rate across target speed and execution delay."""
    print("Generating Figure 4: Success Rate Heatmap...")

    speeds = np.round(np.linspace(0.03, 0.40, 8), 2)
    delays = np.round(np.linspace(0.0, 0.15, 6), 2)
    n_trials = 10

    # Test both oracle and naive experts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (mode, title) in zip(axes, [
        ("oracle", "Oracle Expert (with prediction)"),
        ("none", "Naive Expert (chase current position)"),
    ]):
        success_rates = np.zeros((len(delays), len(speeds)))

        for i, delay in enumerate(delays):
            for j, speed in enumerate(speeds):
                successes = 0
                for trial in range(n_trials):
                    env = MovingGraspEnv(
                        target_speed=speed, motion_pattern="linear",
                        execution_delay=delay, reward_type="shaped",
                    )
                    expert = MovingGraspExpert(env, prediction_mode=mode)
                    data = collect_trajectory(env, expert, seed=trial, max_steps=300)
                    if data["success"]:
                        successes += 1
                    env.close()
                success_rates[i, j] = successes / n_trials

        im = ax.imshow(success_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{s:.2f}" for s in speeds], fontsize=9)
        ax.set_yticks(range(len(delays)))
        ax.set_yticklabels([f"{d*1000:.0f}ms" for d in delays], fontsize=9)
        ax.set_xlabel("Target Speed", fontsize=11)
        ax.set_ylabel("Execution Delay", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(len(delays)):
            for j in range(len(speeds)):
                color = "white" if success_rates[i, j] < 0.5 else "black"
                ax.text(j, i, f"{success_rates[i, j]:.0%}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Success Rate")

    fig.suptitle("Success Rate: Target Speed × Execution Delay", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_success_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 5: Conveyor Pick Task
# ============================================================
def fig5_conveyor_pick():
    """Visualize the conveyor pick task."""
    print("Generating Figure 5: Conveyor Pick...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Conveyor Pick Task", fontsize=14, fontweight="bold")

    speeds = [("Slow (0.1)", 0.1), ("Medium (0.2)", 0.2), ("Fast (0.35)", 0.35)]

    for ax, (label, speed) in zip(axes, speeds):
        env = ConveyorPickEnv(conveyor_speed=speed, n_objects_per_episode=2)
        expert = ConveyorPickExpert(env, prediction_mode="oracle")

        obs, info = env.reset(seed=42)

        ee_traj = [info["full_state"]["ee_pos"].copy()]
        obj_trajs = [[]]

        for step in range(400):
            action = expert.predict(obs, full_state=info["full_state"])
            obs, reward, terminated, truncated, info = env.step(action)

            ee_traj.append(info["full_state"]["ee_pos"].copy())

            fs = info["full_state"]
            if fs["object_active"] and fs["obj_pos"] is not None:
                obj_trajs[-1].append(fs["obj_pos"].copy())
            elif len(obj_trajs[-1]) > 0 and fs["objects_picked"] < len(obj_trajs):
                obj_trajs.append([])

            if terminated or truncated:
                break

        ee_traj = np.array(ee_traj)

        # Draw workspace
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.set_facecolor("#F8F8F8")

        # Conveyor direction arrow
        ax.annotate("", xy=(0.9, -0.45), xytext=(-0.9, -0.45),
                    arrowprops=dict(arrowstyle="->", color=COLORS["conveyor"], lw=2))
        ax.text(0, -0.52, "Conveyor →", ha="center", fontsize=9, color=COLORS["conveyor"])

        # EE trajectory
        ax.plot(ee_traj[:, 0], ee_traj[:, 1],
                color=COLORS["ee"], alpha=0.5, linewidth=1.5, label="EE")
        ax.plot(*ee_traj[0], "s", color=COLORS["ee"], markersize=8)

        # Object trajectories
        for k, traj in enumerate(obj_trajs):
            if len(traj) > 1:
                traj = np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1],
                        color=COLORS["obj"], alpha=0.6, linewidth=2,
                        label=f"Object {k+1}")
                ax.plot(*traj[0], "o", color=COLORS["obj"], markersize=8)

        ax.set_title(f"{label}\nPicked: {info['full_state']['objects_picked']} objects",
                    fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

        env.close()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_conveyor_pick.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 6: Temporal State Analysis
# ============================================================
def fig6_temporal_analysis():
    """Visualize temporal state evolution for the paper."""
    print("Generating Figure 6: Temporal State Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Temporal State Analysis: Why Current Observation is Insufficient",
                fontsize=14, fontweight="bold")

    env = MovingGraspWithDelay()
    expert = MovingGraspExpert(env, prediction_mode="oracle")
    data = collect_trajectory(env, expert, seed=42, max_steps=120)

    # Panel A: EE-Object distance over time (current vs interaction)
    ax = axes[0][0]
    dist_current = np.linalg.norm(data["ee_pos"][:-1] - data["obj_pos"][:-1], axis=1)
    dist_interaction = np.linalg.norm(data["ee_pos"][:-1] - data["interaction_states"][:-1], axis=1)
    steps = range(len(dist_current))
    ax.fill_between(steps, dist_current, dist_interaction, alpha=0.2, color=COLORS["interaction"],
                    label="Difference (misalignment)")
    ax.plot(steps, dist_current, color=COLORS["obj"], linewidth=1.5, label="Dist to current pos")
    ax.plot(steps, dist_interaction, color=COLORS["interaction"], linewidth=1.5, label="Dist to interaction state")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Success threshold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Distance")
    ax.set_title("(a) EE Distance: Current vs Interaction State", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Object velocity over time
    ax = axes[0][1]
    vel_mag = np.linalg.norm(data["obj_vel"][:-1], axis=1)
    ax.plot(data["obj_vel"][:-1, 0], color=COLORS["obj"], linewidth=1.5, label="vx")
    ax.plot(data["obj_vel"][:-1, 1], color=COLORS["interaction"], linewidth=1.5, label="vy")
    ax.plot(vel_mag, color="black", linewidth=1.5, linestyle="--", label="|v|", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Velocity")
    ax.set_title("(b) Target Velocity Components", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Delay estimate over time
    ax = axes[1][0]
    ax.plot(data["delays"][:-1] * 1000, color=COLORS["ee"], linewidth=1.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Estimated Delay (ms)")
    ax.set_title("(c) Interaction Delay Estimate", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel D: Misalignment distribution
    ax = axes[1][1]
    misalignments = np.linalg.norm(
        data["interaction_states"][:len(data["obj_vel"])] - data["obj_pos"][:len(data["obj_vel"])],
        axis=1
    )
    ax.hist(misalignments, bins=25, color=COLORS["interaction"], alpha=0.7, edgecolor="black")
    ax.axvline(x=0.05, color="red", linestyle="--", linewidth=2, label="Success threshold")
    ax.axvline(x=np.mean(misalignments), color="black", linestyle="-", linewidth=2,
               label=f"Mean: {np.mean(misalignments):.3f}")
    ax.set_xlabel("Misalignment Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("(d) Misalignment Distribution", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    env.close()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_temporal_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Dynamic Manipulation Visualization")
    print("=" * 60 + "\n")

    fig1_motion_patterns()
    fig2_expert_vs_naive()
    fig3_supervision_misalignment()
    fig5_conveyor_pick()
    fig6_temporal_analysis()

    # Heatmap takes longer, run last
    fig4_success_heatmap()

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)