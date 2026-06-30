"""
T3-T6: Three-leg support tests on vertical wall.
Based on T2 orientation — one leg lifted while three maintain support.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestThreeLegSupport(SimulationBase):
    """Three-leg support — one leg lifted, xfrc only."""

    ORIENTATIONS = {
        "A": {
            "quat": [0.0, 0.0, 0.707107, 0.707107],
            "gravity_body": "body -Y",
            "base_pos": [0.0, None, 1.5],
            "wall_normal_force": 800.0,
            "standing_q": [0.0, 0.7, -1.4],
            "foot_site_offset": 0.0,
        },
        "B": {
            "quat": [0.5, -0.5, -0.5, -0.5],
            "gravity_body": "body -X",
            "base_pos": [0.0, None, 1.5],
            "wall_normal_force": 800.0,
            "standing_q": [0.0, 1.0, -1.0],
            "foot_site_offset": -0.012,     # gap so xfrc pre-loads PD before contact
        },
    }

    def __init__(self, scene_path: str, lifted_leg: str = "FL",
                 orientation: str = "B"):
        super().__init__(scene_path)

        if orientation not in self.ORIENTATIONS:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'A' or 'B'.")
        cfg = self.ORIENTATIONS[orientation]
        self._orientation = orientation
        self._wall_force = cfg["wall_normal_force"]
        self.lifted_leg = lifted_leg

        # Run control at full sim rate for stability
        self.control_decimation = 1
        self.control_dt = self.dt

        # Standing joint configuration
        sq = cfg["standing_q"]
        standing_q = np.array(sq + sq + sq + sq)
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        self.data.qpos[3:7] = cfg["quat"]

        # Position base so magnet faces touch wall
        mujoco.mj_forward(self.model, self.data)
        base_y0 = self.data.qpos[1]
        foot_y_min = min(self.robot.get_foot_pos(leg)[1] for leg in self.robot.LEGS)
        target_y = 0.025 + cfg.get("foot_site_offset", 0.0)
        base_y_new = base_y0 - (foot_y_min - target_y)
        self.data.qpos[0:3] = [cfg["base_pos"][0], base_y_new, cfg["base_pos"][2]]
        mujoco.mj_forward(self.model, self.data)

        # Fixed target pose (will be modified when lifting)
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

        self._trunk_body_id = 2
        self._foot_xfrc_ids = {
            leg: self.robot.get_foot_body_id(leg) for leg in self.robot.LEGS
        }

        # Lift state machine
        self._lift_start = None
        self._lift_done = False
        self._leg_idx = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}[lifted_leg]

        print(f"[T3{self._orientation}] Lifting {lifted_leg}"
              f", wall_force={self._wall_force}N/foot")

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()

        # Track lift timing
        if self.sim_time > 2.0 and not self._lift_done:
            if self._lift_start is None:
                self._lift_start = self.sim_time
                print(f"[T3] Releasing {self.lifted_leg} at t={self.sim_time:.2f}s")
            if self.sim_time - self._lift_start >= 0.3:
                self._lift_done = True

        lifted = self._lift_done or self.sim_time > 2.0

        # xfrc only on support legs
        if self._wall_force > 0:
            for leg in self.robot.LEGS:
                if not (leg == self.lifted_leg and lifted):
                    bid = self._foot_xfrc_ids[leg]
                    self.data.xfrc_applied[bid, 1] = -self._wall_force

        # PD for all legs
        tau_ff = self.robot.get_bias_torques()
        tau = self._kp * (self._q_des - q) + self._kd * (0 - dq) + tau_ff

        # Zero torque on lifted leg — let it hang free
        if lifted:
            for j in range(3):
                tau[self._leg_idx + j] = 0.0

        tau = np.clip(tau, -self._joint_limits, self._joint_limits)
        self.robot.set_joint_torques(tau)

        # Adhesion: off for lifted leg
        for leg in self.robot.LEGS:
            if leg == self.lifted_leg and lifted:
                self.robot.set_adhesion(leg, 0.0)
            else:
                self.robot.set_adhesion(leg, 0.3)

    def log_data(self):
        row = {"time": self.sim_time, "lifted_leg": self.lifted_leg}
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

        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"q_{name}"] = q[i]
            row[f"dq_{name}"] = dq[i]
            row[f"tau_{name}"] = tau[i]

        for leg in self.robot.LEGS:
            row[f"{leg}_adhesion"] = self.adhesion_ctrl.get_value(leg)
        contact = self.get_contact_forces()
        for leg in self.robot.LEGS:
            row[f"{leg}_contact_fn"] = contact.get(leg, 0.0)

        self.logger.record(row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", type=str, default="FL",
                        choices=["FL", "FR", "RL", "RR"])
    parser.add_argument("--orientation", type=str, default="B",
                        choices=["A", "B"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", type=str, default="")
    args = parser.parse_args()

    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "scene_a2.xml",
    )
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    test = TestThreeLegSupport(scene_path, lifted_leg=args.leg,
                               orientation=args.orientation)
    duration = 8.0
    render = True if args.render else None  # None = check env A2_RENDER
    test.run(duration=duration, render=render, record_path=args.record or None)
    tag = f"T3_lift_{args.leg}_{args.orientation}"
    test.logger.save(os.path.join(results_dir, "csv", f"{tag}.csv"))
    print(f"Data saved to results/csv/{tag}.csv")


if __name__ == "__main__":
    main()
