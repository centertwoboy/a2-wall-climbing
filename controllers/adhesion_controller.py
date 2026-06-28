"""
Adhesion controller with smooth on/off switching using Hermite interpolation.

Switching profile: h(s) = 3*s^2 - 2*s^3  (cubic Hermite)
"""

import numpy as np
from .trajectory_generator import hermite_smooth


class AdhesionController:
    """Manages smooth adhesion force transitions per foot."""

    def __init__(self, switch_time: float = 0.3, max_force: float = 1000.0):
        self.switch_time = switch_time    # transition duration [s]
        self.max_force = max_force        # max adhesion force [N]

        # Per-leg timers
        self._timers = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
        self._states = {"FL": "on", "FR": "on", "RL": "on", "RR": "on"}
        self._values = {"FL": 1.0, "FR": 1.0, "RL": 1.0, "RR": 1.0}

    def set_state(self, leg: str, target: str):
        """Request adhesion 'on' or 'off' for a leg."""
        if self._states[leg] != target:
            self._states[leg] = target
            self._timers[leg] = 0.0

    def update(self, dt: float) -> dict:
        """
        Advance timers and return current adhesion values.

        Returns:
            dict {leg: control_value in [0, 1]}
        """
        for leg in ["FL", "FR", "RL", "RR"]:
            t = self._timers[leg]
            if t < self.switch_time:
                t += dt
                self._timers[leg] = min(t, self.switch_time)
                s = t / self.switch_time
            else:
                s = 1.0

            h = hermite_smooth(s)

            if self._states[leg] == "on":
                # Ramping up: u_on = h(s)
                self._values[leg] = h
            else:
                # Ramping down: u_off = 1 - h(s)
                self._values[leg] = 1.0 - h

        return self._values

    def get_value(self, leg: str) -> float:
        return self._values.get(leg, 0.0)

    def is_transitioning(self, leg: str) -> bool:
        return self._timers.get(leg, 0.0) < self.switch_time
