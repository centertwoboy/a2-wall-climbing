"""
Quintic-polynomial foot trajectory generator for swing phase.

Produces smooth position (0 accel at endpoints) along the wall plane
and sinusoidal clearance normal to the wall.
"""

import numpy as np


def quintic_smooth(s: float) -> float:
    """
    Quintic (5th-order) smooth step: h(0)=0, h'(0)=0, h''(0)=0,
    h(1)=1, h'(1)=0, h''(1)=0.

    h(s) = 10*s^3 - 15*s^4 + 6*s^5
    """
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5


def hermite_smooth(s: float) -> float:
    """
    Cubic Hermite smooth step: h(s) = 3*s^2 - 2*s^3.
    Used for adhesion switching transitions.
    """
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s ** 2 - 2.0 * s ** 3


class FootTrajectory:
    """Generates swing-foot trajectories in world frame."""

    def __init__(self, wall_normal: np.ndarray = None):
        """
        Args:
            wall_normal: unit normal vector pointing OUT of wall.
                         Default: +Y (wall in x-z plane).
        """
        if wall_normal is None:
            self.n_w = np.array([0.0, 1.0, 0.0])
        else:
            self.n_w = np.asarray(wall_normal) / np.linalg.norm(wall_normal)

    def generate_swing(self, p_start: np.ndarray, p_end: np.ndarray,
                       clearance: float, s: float) -> np.ndarray:
        """
        Compute foot position at normalized time s in [0, 1].

        Args:
            p_start: start foot position (on wall)
            p_end: target foot position (on wall)
            clearance: max normal displacement from wall [m]
            s: normalized swing time [0, 1]

        Returns:
            (3,) foot position
        """
        h = quintic_smooth(s)

        # In-plane motion
        p_parallel = p_start + h * (p_end - p_start)

        # Normal clearance: d_n(s) = clearance * sin(pi * s)
        d_n = clearance * np.sin(np.pi * s)

        # Move away from wall (subtract normal * clearance)
        p_foot = p_parallel - d_n * self.n_w

        return p_foot

    def generate_straight_line(self, p_start: np.ndarray, p_end: np.ndarray,
                               s: float) -> np.ndarray:
        """Linear interpolation in world frame."""
        s = np.clip(s, 0.0, 1.0)
        h = quintic_smooth(s)
        return p_start + h * (p_end - p_start)
