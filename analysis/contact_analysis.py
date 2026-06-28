"""
Contact force extraction and slip analysis.
"""

import numpy as np
import mujoco


def extract_contact_forces(model: mujoco.MjModel, data: mujoco.MjData,
                           foot_geom_names: list = None) -> dict:
    """
    Extract contact forces for specified foot geometries.

    Returns:
        dict: {foot_name: {"force": (3,), "torque": (3,), "normal": float}}
    """
    if foot_geom_names is None:
        foot_geom_names = [
            "FL_magnet_geom", "FR_magnet_geom",
            "RL_magnet_geom", "RR_magnet_geom",
        ]

    result = {name: {"force": np.zeros(3), "torque": np.zeros(3),
                     "normal": 0.0, "n_contacts": 0}
              for name in foot_geom_names}

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        g1 = model.geom(contact.geom1)
        g2 = model.geom(contact.geom2)

        # Identify which foot is involved
        foot_name = None
        for fn in foot_geom_names:
            if g1.name == fn or g2.name == fn:
                foot_name = fn
                break

        if foot_name is None:
            continue

        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, wrench)

        result[foot_name]["force"] += wrench[:3]
        result[foot_name]["torque"] += wrench[3:]
        result[foot_name]["n_contacts"] += 1

        # Normal force (along contact frame x-axis)
        frame = contact.frame.reshape(3, 3)
        result[foot_name]["normal"] += abs(np.dot(wrench[:3], frame[:, 0]))

    return result


def compute_slip_velocity(foot_vel: np.ndarray, wall_normal: np.ndarray) -> float:
    """
    Compute tangential (slip) velocity of foot relative to wall.

    Args:
        foot_vel: (3,) world velocity of foot
        wall_normal: (3,) unit wall normal vector

    Returns:
        float: slip speed [m/s]
    """
    n = np.asarray(wall_normal) / np.linalg.norm(wall_normal)
    # Tangential component: v - (v·n)n
    v_normal = np.dot(foot_vel, n) * n
    v_tangential = foot_vel - v_normal
    return float(np.linalg.norm(v_tangential))


def compute_cumulative_slip(v_slip: np.ndarray, dt: float) -> float:
    """Integrate slip velocity over time."""
    return float(np.sum(v_slip) * dt)
