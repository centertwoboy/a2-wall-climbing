"""
Joint-level controller: PD + feedforward + torque rate limiting + clamping.

Control law:
  tau_demand = Kp*(q_d - q) + Kd*(dq_d - dq) + tau_ff
  tau_rate_limited = clip(tau_demand, previous - rate*dt, previous + rate*dt)
  tau_actual = clip(tau_rate_limited, -limit, +limit)
"""

import numpy as np


class JointController:
    """PD controller with gravity/Coriolis feedforward and rate limiting."""

    def __init__(self, kp: np.ndarray, kd: np.ndarray,
                 torque_limit: float = 300.0,
                 max_torque_rate: float = 500.0):
        """
        Args:
            kp: (12,) proportional gains
            kd: (12,) derivative gains
            torque_limit: symmetric torque clamp [N.m]
            max_torque_rate: max rate of torque change [N.m/s]
        """
        self.kp = np.asarray(kp)
        self.kd = np.asarray(kd)
        self.torque_limit = torque_limit
        self.max_torque_rate = max_torque_rate
        self.previous_tau = np.zeros(len(kp))
        self._tau_demand = np.zeros(len(kp))

    def compute(self, q: np.ndarray, dq: np.ndarray,
                q_des: np.ndarray, dq_des: np.ndarray,
                tau_ff: np.ndarray, dt: float) -> tuple:
        """
        Compute torque command.

        Returns:
            (tau_actual, tau_demand) — both (12,) arrays
        """
        # PD + feedforward
        tau_demand = (
            self.kp * (q_des - q)
            + self.kd * (dq_des - dq)
            + tau_ff
        )
        self._tau_demand = tau_demand.copy()

        # Rate limiting
        max_step = self.max_torque_rate * dt
        tau_rate_limited = np.clip(
            tau_demand,
            self.previous_tau - max_step,
            self.previous_tau + max_step,
        )

        # Torque clamping
        tau_actual = np.clip(tau_rate_limited, -self.torque_limit, self.torque_limit)

        self.previous_tau = tau_actual.copy()
        return tau_actual, tau_demand

    @property
    def tau_demand(self) -> np.ndarray:
        return self._tau_demand.copy()

    def reset(self):
        self.previous_tau = np.zeros_like(self.previous_tau)
        self._tau_demand = np.zeros_like(self._tau_demand)
