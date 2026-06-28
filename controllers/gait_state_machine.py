"""
Crawl gait state machine for wall climbing.

Gait order: FL -> RR -> FR -> RL  (diagonal crawl)
Each leg goes through 9 states: SUPPORT -> LOAD_TRANSFER -> ADHESION_RELEASE
  -> FOOT_LIFT -> FOOT_SWING -> FOOT_APPROACH -> CONTACT_CONFIRM
  -> ADHESION_BUILD -> SUPPORT_CONFIRM
"""

import enum
import numpy as np


class LegState(enum.IntEnum):
    SUPPORT = 0
    LOAD_TRANSFER = 1
    ADHESION_RELEASE = 2
    FOOT_LIFT = 3
    FOOT_SWING = 4
    FOOT_APPROACH = 5
    CONTACT_CONFIRM = 6
    ADHESION_BUILD = 7
    SUPPORT_CONFIRM = 8


class GaitStateMachine:
    """Static crawl gait: one leg in swing, three in support."""

    GAIT_ORDER = ["FL", "RR", "FR", "RL"]  # diagonal crawl sequence

    def __init__(self, swing_time: float = 1.5, support_time: float = 0.5,
                 step_length: float = 0.05, step_height: float = 0.02,
                 climb_direction: np.ndarray = None):
        """
        Args:
            swing_time: duration of each leg's swing phase [s]
            support_time: extra support stabilization time [s]
            step_length: distance per step along wall [m]
            step_height: normal clearance during swing [m]
            climb_direction: unit vector of climbing direction (default: +Z)
        """
        self.swing_time = swing_time
        self.support_time = support_time
        self.step_length = step_length
        self.step_height = step_height
        self.climb_dir = (np.array([0, 0, 1.0]) if climb_direction is None
                          else np.asarray(climb_direction) / np.linalg.norm(climb_direction))

        # Per-leg state
        self.leg_states = {leg: LegState.SUPPORT for leg in self.GAIT_ORDER}
        self.leg_timers = {leg: 0.0 for leg in self.GAIT_ORDER}

        # Current swing leg index in GAIT_ORDER
        self._swing_idx = 0
        self._step_count = 0

        # Foot target positions (in world frame)
        self.foot_targets = {leg: None for leg in self.GAIT_ORDER}
        self.foot_start_positions = {leg: None for leg in self.GAIT_ORDER}

        # Sub-phase durations (rough allocation of swing_time)
        self._phase_durations = {
            LegState.LOAD_TRANSFER: 0.1,
            LegState.ADHESION_RELEASE: 0.05,
            LegState.FOOT_LIFT: 0.1,
            LegState.FOOT_SWING: 0.0,      # remainder of swing_time
            LegState.FOOT_APPROACH: 0.15,
            LegState.CONTACT_CONFIRM: 0.05,
            LegState.ADHESION_BUILD: 0.05,
            LegState.SUPPORT_CONFIRM: 0.1,
        }

        self.switch_condition_met = False

    @property
    def swing_leg(self) -> str:
        return self.GAIT_ORDER[self._swing_idx]

    @property
    def step_count(self) -> int:
        return self._step_count

    def get_swing_phase(self, leg: str) -> float:
        """
        Return normalized swing progress [0, 1] for the swinging leg.
        Only valid when leg is in FOOT_SWING state.
        """
        if self.leg_states[leg] != LegState.FOOT_SWING:
            return 0.0

        # Compute swing fraction within the swing phase
        t_swing = self.leg_timers[leg]
        # Subtract time spent in states before FOOT_SWING
        pre_time = sum(
            d for s, d in self._phase_durations.items()
            if s < LegState.FOOT_SWING
        )
        swing_duration = self.swing_time - pre_time - sum(
            d for s, d in self._phase_durations.items()
            if s > LegState.FOOT_SWING
        )
        if swing_duration <= 0:
            return 1.0
        return np.clip((t_swing - pre_time) / swing_duration, 0.0, 1.0)

    def update(self, dt: float, foot_positions: dict,
               contact_forces: dict, foot_velocities: dict):
        """
        Advance state machine by dt.

        Args:
            dt: time step [s]
            foot_positions: {leg: (3,) world pos}
            contact_forces: {leg: normal_force}
            foot_velocities: {leg: (3,) world vel}

        Returns:
            dict with keys: adhesion_commands, swing_progress, new_targets
        """
        adhesion_commands = {}
        swing_progress = {}
        new_targets = {}

        sw_leg = self.swing_leg

        for leg in self.GAIT_ORDER:
            state = self.leg_states[leg]
            self.leg_timers[leg] += dt

            if leg == sw_leg:
                # ---- Swing leg: advance through sub-states ----
                t = self.leg_timers[leg]

                if state == LegState.SUPPORT:
                    adhesion_commands[leg] = "on"
                    if t >= self.support_time:
                        self._transition(leg, LegState.LOAD_TRANSFER)
                        self.foot_start_positions[leg] = foot_positions[leg].copy()
                        # Compute target: current + step in climb direction
                        self.foot_targets[leg] = (
                            foot_positions[leg] + self.step_length * self.climb_dir
                        )

                elif state == LegState.LOAD_TRANSFER:
                    adhesion_commands[leg] = "on"
                    if t >= self._phase_durations[LegState.LOAD_TRANSFER]:
                        self._transition(leg, LegState.ADHESION_RELEASE)

                elif state == LegState.ADHESION_RELEASE:
                    adhesion_commands[leg] = "off"
                    if t >= self._phase_durations[LegState.ADHESION_RELEASE]:
                        self._transition(leg, LegState.FOOT_LIFT)

                elif state == LegState.FOOT_LIFT:
                    adhesion_commands[leg] = "off"
                    if t >= self._phase_durations[LegState.FOOT_LIFT]:
                        self._transition(leg, LegState.FOOT_SWING)

                elif state == LegState.FOOT_SWING:
                    adhesion_commands[leg] = "off"
                    swing_progress[leg] = self.get_swing_phase(leg)
                    # Swing complete when progress reaches 1.0
                    if swing_progress[leg] >= 1.0:
                        self._transition(leg, LegState.FOOT_APPROACH)

                elif state == LegState.FOOT_APPROACH:
                    adhesion_commands[leg] = "off"
                    if t >= self._phase_durations[LegState.FOOT_APPROACH]:
                        self._transition(leg, LegState.CONTACT_CONFIRM)

                elif state == LegState.CONTACT_CONFIRM:
                    adhesion_commands[leg] = "off"
                    # Check contact: normal force > threshold
                    fn = contact_forces.get(leg, 0.0)
                    v = np.linalg.norm(foot_velocities.get(leg, np.ones(3)))
                    if fn > 20.0 and v < 0.02:
                        self._transition(leg, LegState.ADHESION_BUILD)

                elif state == LegState.ADHESION_BUILD:
                    adhesion_commands[leg] = "on"
                    if t >= self._phase_durations[LegState.ADHESION_BUILD]:
                        self._transition(leg, LegState.SUPPORT_CONFIRM)

                elif state == LegState.SUPPORT_CONFIRM:
                    adhesion_commands[leg] = "on"
                    if t >= self._phase_durations[LegState.SUPPORT_CONFIRM]:
                        # Done! Advance to next swing leg
                        self._transition(leg, LegState.SUPPORT)
                        self._advance_swing_leg()
                        new_targets["step_complete"] = True

            else:
                # ---- Support leg: keep adhesion on ----
                adhesion_commands[leg] = "on"

        return {
            "adhesion_commands": adhesion_commands,
            "swing_progress": swing_progress,
            "new_targets": new_targets,
            "swing_leg": sw_leg,
            "step_count": self._step_count,
        }

    def _transition(self, leg: str, new_state: LegState):
        self.leg_states[leg] = new_state
        self.leg_timers[leg] = 0.0

    def _advance_swing_leg(self):
        self._swing_idx = (self._swing_idx + 1) % len(self.GAIT_ORDER)
        self._step_count += 1

    def reset(self):
        self.leg_states = {leg: LegState.SUPPORT for leg in self.GAIT_ORDER}
        self.leg_timers = {leg: 0.0 for leg in self.GAIT_ORDER}
        self._swing_idx = 0
        self._step_count = 0
