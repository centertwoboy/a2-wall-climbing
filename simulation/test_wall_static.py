"""
T2: Static four-foot wall adhesion on vertical wall.
Two orientations:
  A - body +Y points up along wall (side-standing, gravity in body -Y)
  B - body +X points up along wall (head-up, gravity in body -X)
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestWallStatic(SimulationBase):
    """Four-foot wall adhesion — static equilibrium (xfrc + adhesion)."""

    ORIENTATIONS = {
        "A": {  # side-standing: body +Y -> world +Z (up)
            "quat": [0.0, 0.0, 0.707107, 0.707107],       # 180 about YZ: body+Y=Z, body+Z=Y
            "gravity_body": "body -Y (lateral, right side)",
            "base_pos": [0.0, None, 1.5],
            "wall_normal_force": 800.0,
            "standing_q": [0.0, 0.7, -1.4],
            "foot_site_offset": 0.0,
        },
        "B": {  # head-up: body +X -> world +Z (up)
            "quat": [0.5, -0.5, -0.5, -0.5],                   # X->Z, Z->Y, Y->X
            "gravity_body": "body -X",
            "base_pos": [0.0, None, 1.5],
            "wall_normal_force": 800.0,
            "standing_q": [0.0, 1.0, -1.0],
            "foot_site_offset": -0.012,
        },
    }

    def __init__(self, scene_path: str, orientation: str = "A"):
        super().__init__(scene_path)

        if orientation not in self.ORIENTATIONS:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'A' or 'B'.")
        cfg = self.ORIENTATIONS[orientation]
        self._orientation = orientation
        self._wall_force = cfg["wall_normal_force"]

        # Run control at full sim rate (1000 Hz) for stability
        self.control_decimation = 1
        self.control_dt = self.dt

        # Standing joint configuration (orientation-specific)
        sq = cfg["standing_q"]
        standing_q = np.array(sq + sq + sq + sq)  # FL, FR, RL, RR
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        self.data.qpos[3:7] = cfg["quat"]

        # Compute base position so magnet faces touch the wall surface
        mujoco.mj_forward(self.model, self.data)
        base_y0 = self.data.qpos[1]
        foot_y_min = min(self.robot.get_foot_pos(leg)[1] for leg in self.robot.LEGS)
        target_y = 0.025 + cfg.get("foot_site_offset", 0.0)
        base_y_new = base_y0 - (foot_y_min - target_y)
        self.data.qpos[0:3] = [cfg["base_pos"][0], base_y_new, cfg["base_pos"][2]]

        mujoco.mj_forward(self.model, self.data)

        # Fixed target pose
        self._q_des = self.robot.get_joint_positions().copy()

        # High-gain PD
        self._kp = np.tile([200, 300, 250], 4)
        self._kd = np.tile([5, 8, 6], 4)

        # Disable foot collision spheres that shadow magnet geoms,
        # and increase magnet friction for better wall grip.
        for i in range(self.model.ngeom):
            body_name = self.model.body(self.model.geom_bodyid[i]).name or ""
            if "calf" in body_name:
                gtype = self.model.geom_type[i]
                gpos = self.model.geom_pos[i]
                if gtype == mujoco.mjtGeom.mjGEOM_SPHERE and abs(gpos[2] + 0.275) < 0.001:
                    self.model.geom_contype[i] = 0
                    self.model.geom_conaffinity[i] = 0
            if "magnet_geom" in (self.model.geom(i).name or ""):
                self.model.geom_friction[i] = [0.5, 0.02, 0.001]

        self._trunk_body_id = 2  # body 2 = base_link (1 = wall)
        self._foot_xfrc_ids = {
            leg: self.robot.get_foot_body_id(leg) for leg in self.robot.LEGS
        }

        print(f"[T2{orientation}] gravity in {cfg['gravity_body']}"
              f", wall_force={self._wall_force}N/foot")

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()

        # xfrc pushes feet toward wall -> contact -> friction resists gravity
        if self._wall_force > 0:
            for leg in self.robot.LEGS:
                bid = self._foot_xfrc_ids[leg]
                self.data.xfrc_applied[bid, 1] = -self._wall_force

        # PD + live gravity compensation
        tau_ff = self.robot.get_bias_torques()
        tau = self._kp * (self._q_des - q) + self._kd * (0 - dq) + tau_ff
        tau = np.clip(tau, -self._joint_limits, self._joint_limits)
        self.robot.set_joint_torques(tau)

        for leg in self.robot.LEGS:
            self.robot.set_adhesion(leg, 0.3)

    def log_data(self):
        row = {"time": self.sim_time}
        base_pos = self.robot.get_base_pos()
        base_rpy = self.robot.get_base_rpy()
        row.update({
            "base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2],
            "base_roll": base_rpy[0], "base_pitch": base_rpy[1], "base_yaw": base_rpy[2],
        })
        base_vel = self.robot.get_base_vel()
        row.update({
            "base_vx": base_vel[0], "base_vy": base_vel[1], "base_vz": base_vel[2],
        })

        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        tau = self.robot.get_joint_torques()
        tau_d = self.joint_ctrl.tau_demand

        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"q_{name}"] = q[i]
            row[f"dq_{name}"] = dq[i]
            row[f"tau_{name}"] = tau[i]
            row[f"tau_demand_{name}"] = tau_d[i]

        for leg in self.robot.LEGS:
            row[f"{leg}_adhesion"] = self.adhesion_ctrl.get_value(leg)
        contact = self.get_contact_forces()
        for leg in self.robot.LEGS:
            row[f"{leg}_contact_fn"] = contact.get(leg, 0.0)

        self.logger.record(row)


def main():
    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "scene_a2.xml",
    )
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    for orientation in ["A", "B"]:
        print(f"\n{'='*50}")
        print(f"Running T2{orientation}")
        print(f"{'='*50}")
        test = TestWallStatic(scene_path, orientation=orientation)
        test.run(duration=10.0)
        test.logger.save(os.path.join(results_dir, "csv", f"T2{orientation}_wall_static.csv"))
        print(f"Data saved to results/csv/T2{orientation}_wall_static.csv")


if __name__ == "__main__":
    main()
