"""
T1: Ground standing test — verify gravity compensation, PD tuning, joint directions.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestGroundStand(SimulationBase):
    """Horizontal ground standing — basis for controller verification."""

    def __init__(self, scene_path: str):
        super().__init__(scene_path)

        # Run control at full sim rate (1000 Hz) for T1 stability test
        self.control_decimation = 1
        self.control_dt = self.dt

        # Standing joint configuration — all 4 legs same
        standing_q = np.array([
            0.0, 0.7, -1.4,   # FL
            0.0, 0.7, -1.4,   # FR
            0.0, 0.7, -1.4,   # RL
            0.0, 0.7, -1.4,   # RR
        ])
        qpos_ids = self.robot._qpos_ids
        for i, qid in enumerate(qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        # Compute base height so feet touch the ground
        mujoco.mj_forward(self.model, self.data)
        base_z0 = self.data.qpos[2]
        foot_z_min = min(self.robot.get_foot_pos(leg)[2] for leg in self.robot.LEGS)
        base_z_new = base_z0 - foot_z_min  # shift base so lowest foot lands at z=0
        self.data.qpos[0:3] = [0.0, 0.0, base_z_new]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        # Save initial standing pose as fixed control target
        self._q_des = self.robot.get_joint_positions().copy()

        # High-gain PD parameters (override config for T1 stability)
        self._kp = np.tile([200, 300, 250], 4)  # hip, thigh, calf × 4 legs
        self._kd = np.tile([5, 8, 6], 4)

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()

        tau = self._kp * (self._q_des - q) + self._kd * (0 - dq)
        tau = np.clip(tau, -self._joint_limits, self._joint_limits)
        self.robot.set_joint_torques(tau)

    def log_data(self):
        row = {"time": self.sim_time}
        base_pos = self.robot.get_base_pos()
        row.update({"base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2]})
        tau = self.robot.get_joint_torques()
        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"tau_{name}"] = tau[i]
        self.logger.record(row)


def main():
    # Build a ground scene XML inline
    import tempfile
    ground_scene = """<mujoco model="scene_ground">
    <include file="robots/a2.xml"/>
    <statistic center="0 0 0.4"/>
    <visual>
        <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="flat" rgb1="0.15 0.15 0.2" rgb2="0.15 0.15 0.2"
                 width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="0.55 0.55 0.6" rgb2="0.45 0.45 0.5"
                 markrgb="0.9 0.9 0.9" width="300" height="300"/>
        <material name="groundplane" texture="groundplane"
                  texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light pos="2 2 2" dir="-1 -1 -1" directional="true"
               diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <camera name="fixed_cam" pos="2 2 0.8" xyaxes="-1 0 0 0 -0.5 1"/>
        <geom name="ground" type="plane" size="10 10 0.05"
              friction="0.8 0.02 0.001" material="groundplane"
              solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </worldbody>
    </mujoco>"""

    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    scene_path = os.path.join(models_dir, "scene_ground_temp.xml")
    with open(scene_path, "w") as f:
        f.write(ground_scene)

    try:
        test = TestGroundStand(scene_path)
        test.run(duration=5.0)
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )
        test.logger.save(os.path.join(results_dir, "csv", "T1_ground_stand.csv"))
        print("Data saved to results/csv/T1_ground_stand.csv")
    finally:
        os.remove(scene_path)


if __name__ == "__main__":
    main()
