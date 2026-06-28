"""
Single-leg forward and inverse kinematics for A2 3-DOF leg.

A2 leg kinematic chain:
  Hip:    rotation about X (abduction/adduction), axis="1 0 0"
  Thigh:  rotation about Y (pitch),                axis="0 1 0"
  Calf:   rotation about Y (pitch),                axis="0 1 0"

Body-relative offsets (from MJCF):
  Hip from base:   (±0.25944, ±0.075113, 0)
  Thigh from hip:  (0, ±0.12779, 0)
  Calf from thigh: (0, 0, -0.275)
  Foot from calf:  (0, 0, -0.275) [fixed in MJCF]
"""

import numpy as np


def rx(angle: float) -> np.ndarray:
    """Rotation matrix about X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(angle: float) -> np.ndarray:
    """Rotation matrix about Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class LegKinematics:
    """FK/IK for one A2 leg with hip(X), thigh(Y), calf(Y)."""

    def __init__(self, hip_offset: np.ndarray,
                 thigh_offset: np.ndarray = None,
                 calf_offset: np.ndarray = None,
                 foot_offset: np.ndarray = None):
        """
        Args:
            hip_offset: (3,) from base to hip joint
            thigh_offset: (3,) from hip to thigh joint (default: [0, 0.12779, 0])
            calf_offset: (3,) from thigh to calf joint (default: [0, 0, -0.275])
            foot_offset: (3,) from calf to foot site (default: [0, 0, -0.275])
        """
        self.hip_offset = np.asarray(hip_offset)
        self.thigh_offset = (np.array([0, 0.12779, 0]) if thigh_offset is None
                             else np.asarray(thigh_offset))
        self.calf_offset = (np.array([0, 0, -0.275]) if calf_offset is None
                            else np.asarray(calf_offset))
        self.foot_offset = (np.array([0, 0, -0.275]) if foot_offset is None
                            else np.asarray(foot_offset))

    def forward(self, q: np.ndarray) -> np.ndarray:
        """
        Compute foot position in hip frame.

        Args:
            q: [hip_x, thigh_y, calf_y] joint angles [rad]

        Returns:
            (3,) foot position in hip frame
        """
        hip, thigh, calf = q

        # Hip rotation about X
        R_hip = rx(hip)
        # Thigh rotation about Y
        R_thigh = ry(thigh)
        # Calf rotation about Y
        R_calf = ry(calf)

        # Chain: hip_origin -> thigh_origin -> calf_origin -> foot
        # Knee position in hip frame
        knee_pos = self.hip_offset + R_hip @ self.thigh_offset

        # Calf position in hip frame
        calf_pos = knee_pos + R_hip @ R_thigh @ self.calf_offset

        # Foot position in hip frame
        foot_pos = calf_pos + R_hip @ R_thigh @ R_calf @ self.foot_offset

        return foot_pos

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute (3,3) positional Jacobian by finite differences."""
        eps = 1e-6
        J = np.zeros((3, 3))
        p0 = self.forward(q)
        for i in range(3):
            q_pert = q.copy()
            q_pert[i] += eps
            J[:, i] = (self.forward(q_pert) - p0) / eps
        return J

    def inverse(self, p_target: np.ndarray, q0: np.ndarray = None,
                max_iter: int = 100, tol: float = 1e-4,
                damping: float = 1e-3,
                q_min: np.ndarray = None, q_max: np.ndarray = None) -> np.ndarray:
        """
        Damped least-squares inverse kinematics.

        Args:
            p_target: desired foot position in hip frame
            q0: initial guess (default: mid-stance configuration)
            max_iter: maximum iterations
            tol: convergence tolerance [m]
            damping: damping factor lambda^2
            q_min, q_max: joint limits

        Returns:
            (3,) joint angles [hip, thigh, calf]
        """
        if q0 is None:
            # Nominal standing config
            q0 = np.array([0.0, 0.8, -1.6])
        q = q0.copy().astype(float)

        for _ in range(max_iter):
            p_current = self.forward(q)
            error = p_target - p_current
            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian(q)
            JJT = J @ J.T
            damped = JJT + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(damped, error)

            # Line search to stay within limits
            alpha = 1.0
            for _ in range(10):
                q_new = q + alpha * dq
                if q_min is not None and np.any(q_new < q_min):
                    alpha *= 0.5
                    continue
                if q_max is not None and np.any(q_new > q_max):
                    alpha *= 0.5
                    continue
                break

            q = q + alpha * dq

        return q
