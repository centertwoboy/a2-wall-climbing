"""
Plotting utilities — torque curves, contact forces, attitude, comparison charts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Chinese font setup — try common CJK fonts
for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass


def plot_joint_torques(logger, output_path: str, title: str = "Joint Torques"):
    """Plot all 12 joint torques vs time."""
    time = logger.get_column("time")
    if len(time) == 0:
        print("[Plot] No data to plot.")
        return

    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    leg_labels = ["FL", "FR", "RL", "RR"]
    joint_labels = ["hip", "thigh", "calf"]

    for leg_idx, leg in enumerate(leg_labels):
        for jnt_idx, jnt in enumerate(joint_labels):
            ax = axes[leg_idx][jnt_idx]
            col_name = f"tau_{leg}_{jnt}"
            tau = logger.get_column(col_name)
            ax.plot(time, tau, linewidth=0.8)
            ax.set_title(f"{leg} {jnt}")
            ax.set_ylabel("Torque [N.m]")
            ax.axhline(y=180, color="r", linestyle="--", alpha=0.5, label="180 N.m")
            ax.axhline(y=-180, color="r", linestyle="--", alpha=0.5)
            ax.axhline(y=126, color="orange", linestyle=":", alpha=0.5)
            ax.axhline(y=-126, color="orange", linestyle=":", alpha=0.5)
            ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time [s]")
    axes[-1][1].set_xlabel("Time [s]")
    axes[-1][2].set_xlabel("Time [s]")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved torque plot to {output_path}")


def plot_base_attitude(logger, output_path: str, title: str = "Base Attitude"):
    """Plot base position and orientation vs time."""
    time = logger.get_column("time")
    if len(time) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=14)

    # Position
    for idx, axis in enumerate(["x", "y", "z"]):
        ax = axes[0][idx]
        col = f"base_{axis}"
        pos = logger.get_column(col)
        ax.plot(time, pos, linewidth=0.8)
        ax.set_title(f"Base {axis} [m]")
        ax.grid(True, alpha=0.3)

    # Orientation
    for idx, axis in enumerate(["roll", "pitch", "yaw"]):
        ax = axes[1][idx]
        col = f"base_{axis}"
        rpy = logger.get_column(col)
        if len(rpy) > 0:
            ax.plot(time, np.degrees(rpy), linewidth=0.8)
        ax.set_title(f"Base {axis} [deg]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved attitude plot to {output_path}")


def plot_contact_forces(logger, output_path: str, title: str = "Foot Contact Forces"):
    """Plot normal contact forces for each foot."""
    time = logger.get_column("time")
    if len(time) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    for idx, leg in enumerate(["FL", "FR", "RL", "RR"]):
        ax = axes[idx // 2][idx % 2]
        col = f"{leg}_contact_fn"
        fn = logger.get_column(col)
        ax.plot(time, fn, linewidth=0.8)
        ax.set_title(f"{leg} Normal Force [N]")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved contact force plot to {output_path}")


def plot_torque_comparison(cases: dict, output_path: str,
                           title: str = "Torque Comparison"):
    """
    Bar chart comparing peak torques across test cases.

    Args:
        cases: {case_name: {joint_name: peak_torque}}
    """
    if not cases:
        return

    joint_names = list(list(cases.values())[0].keys())
    n_joints = len(joint_names)
    n_cases = len(cases)

    x = np.arange(n_joints)
    width = 0.8 / n_cases

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (case_name, metrics) in enumerate(cases.items()):
        values = [metrics.get(jn, 0) for jn in joint_names]
        ax.bar(x + idx * width, values, width, label=case_name)

    ax.set_xticks(x + width * (n_cases - 1) / 2)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=180, color="r", linestyle="--", alpha=0.5, label="180 N.m limit")
    ax.axhline(y=126, color="orange", linestyle=":", alpha=0.5)
    ax.set_ylabel("Peak Torque [N.m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved comparison plot to {output_path}")
