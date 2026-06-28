"""
Unified robot joint interface — provides address lookup and read/write
access to all 12 leg joints through consistent naming conventions.

Motor naming: {LEG}_{joint}  (e.g., FL_hip, FL_thigh, FL_calf)
"""

import numpy as np
import mujoco


class RobotInterface:
    """Maps joint names to MuJoCo qpos/dof addresses."""

    # Leg joint naming convention (3 joints per leg, 4 legs)
    LEG_JOINTS = {
        "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "RL": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "RR": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    }

    # Motor names match the A2 MJCF: {leg}_{joint_type} (no _motor suffix)
    LEG_MOTORS = {
        "FL": ["FL_hip", "FL_thigh", "FL_calf"],
        "FR": ["FR_hip", "FR_thigh", "FR_calf"],
        "RL": ["RL_hip", "RL_thigh", "RL_calf"],
        "RR": ["RR_hip", "RR_thigh", "RR_calf"],
    }

    FOOT_BODIES = {
        "FL": "FL_magnet_body",
        "FR": "FR_magnet_body",
        "RL": "RL_magnet_body",
        "RR": "RR_magnet_body",
    }

    FOOT_SITES = {
        "FL": "FL_magnet_site",
        "FR": "FR_magnet_site",
        "RL": "RL_magnet_site",
        "RR": "RR_magnet_site",
    }

    LEGS = ["FL", "FR", "RL", "RR"]

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Build joint address maps
        self._joint_qposadr = {}
        self._joint_dofadr = {}
        self._joint_ids = {}
        self._motor_ids = {}

        for leg in self.LEGS:
            for joint_name in self.LEG_JOINTS[leg]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid < 0:
                    raise ValueError(f"Joint not found: {joint_name}")
                self._joint_ids[joint_name] = jid
                self._joint_qposadr[joint_name] = model.jnt_qposadr[jid]
                self._joint_dofadr[joint_name] = model.jnt_dofadr[jid]

            for motor_name in self.LEG_MOTORS[leg]:
                mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
                if mid < 0:
                    raise ValueError(f"Motor not found: {motor_name}")
                self._motor_ids[motor_name] = mid

        # Build foot body/site ID maps
        self._foot_body_ids = {}
        self._foot_site_ids = {}
        for leg in self.LEGS:
            self._foot_body_ids[leg] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.FOOT_BODIES[leg]
            )
            self._foot_site_ids[leg] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, self.FOOT_SITES[leg]
            )

        # Adhesion actuator IDs
        self._adhesion_ids = {}
        for leg in self.LEGS:
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_adhesion"
            )
            if aid >= 0:
                self._adhesion_ids[leg] = aid

        # Build ordered arrays for bulk read/write
        # Order: FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf
        self._dof_ids = np.array(
            [self._joint_dofadr[jn] for leg in self.LEGS for jn in self.LEG_JOINTS[leg]],
            dtype=np.int32,
        )
        self._motor_ctrl_ids = np.array(
            [self._motor_ids[mn] for leg in self.LEGS for mn in self.LEG_MOTORS[leg]],
            dtype=np.int32,
        )
        self._qpos_ids = np.array(
            [self._joint_qposadr[jn] for leg in self.LEGS for jn in self.LEG_JOINTS[leg]],
            dtype=np.int32,
        )

    # ---- Joint-level access ----

    def get_joint_positions(self) -> np.ndarray:
        """Return (12,) array of joint positions [rad]."""
        return self.data.qpos[self._qpos_ids].copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Return (12,) array of joint velocities [rad/s]."""
        return self.data.qvel[self._dof_ids].copy()

    def get_joint_torques(self) -> np.ndarray:
        """Return (12,) array of actual actuator torques [N.m]."""
        return self.data.qfrc_actuator[self._dof_ids].copy()

    def get_bias_torques(self) -> np.ndarray:
        """Return (12,) array of bias (gravity+Coriolis) generalized forces."""
        return self.data.qfrc_bias[self._dof_ids].copy()

    def set_joint_torques(self, torques: np.ndarray):
        """Send torque commands to all 12 motors."""
        self.data.ctrl[self._motor_ctrl_ids] = torques

    # ---- Per-leg joint access ----

    def get_leg_positions(self, leg: str) -> np.ndarray:
        return np.array([self.data.qpos[self._joint_qposadr[jn]]
                         for jn in self.LEG_JOINTS[leg]])

    def get_leg_velocities(self, leg: str) -> np.ndarray:
        return np.array([self.data.qvel[self._joint_dofadr[jn]]
                         for jn in self.LEG_JOINTS[leg]])

    # ---- Foot site access ----

    def get_foot_pos(self, leg: str) -> np.ndarray:
        """Return (3,) world position of foot site."""
        return self.data.site_xpos[self._foot_site_ids[leg]].copy()

    def get_foot_vel(self, leg: str) -> np.ndarray:
        """Return (3,) world linear velocity of foot body."""
        bid = self._foot_body_ids[leg]
        return self.data.cvel[bid * 6 + 3 : bid * 6 + 6].copy()

    def get_foot_body_id(self, leg: str) -> int:
        return self._foot_body_ids[leg]

    # ---- Base state ----

    def get_base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def get_base_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def get_base_vel(self) -> np.ndarray:
        return self.data.qvel[0:6].copy()

    def get_base_rpy(self) -> np.ndarray:
        """Convert base quaternion to roll-pitch-yaw (fixed XYZ)."""
        R = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R.reshape(-1), self.data.qpos[3:7])
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw])

    # ---- Adhesion ----

    def set_adhesion(self, leg: str, value: float):
        if leg in self._adhesion_ids:
            self.data.ctrl[self._adhesion_ids[leg]] = np.clip(value, 0.0, 1.0)

    def get_adhesion_ids(self) -> dict:
        return self._adhesion_ids.copy()

    # ---- Utility ----

    def get_dof_ids(self) -> np.ndarray:
        return self._dof_ids.copy()

    def get_motor_ctrl_ids(self) -> np.ndarray:
        return self._motor_ctrl_ids.copy()
