"""
T3-T6: Three-leg support tests.
Lift one leg while three maintain adhesion on vertical wall.
Verifies static stability with real A2 model.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestThreeLegSupport(SimulationBase):
    """Three-leg support — one leg lifted."""

    def __init__(self, scene_path: str, lifted_leg: str = "FL"):
        super().__init__(scene_path)
        self.lifted_leg = lifted_leg

        # Initial base pose: facing wall
        self.data.qpos[0:3] = [0.0, 0.8, 1.5]
        self.data.qpos[3:7] = [0.707107, 0.707107, 0.0, 0.0]

        # Standing pose: hip=0, thigh=0.8, calf=-1.6
        standing_q = [0.0, 0.8, -1.6] * 4
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        mujoco.mj_forward(self.model, self.data)

        self._lift_start = None
        self._lift_done = False

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        tau_ff = self.robot.get_bias_torques()
        q_des = q.copy()
        dq_des = np.zeros(12)
        leg_idx = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}[self.lifted_leg]

        # Settle for 2s, then lift the designated leg
        if self.sim_time > 2.0 and not self._lift_done:
            if self._lift_start is None:
                self._lift_start = self.sim_time
                print(f"[T3-T6] Lifting {self.lifted_leg} at t={self.sim_time:.2f}s")

            progress = min((self.sim_time - self._lift_start) / 1.5, 1.0)
            # Flex thigh more, contract calf (less negative = straighter leg)
            q_des[leg_idx + 1] += 0.4 * progress
            q_des[leg_idx + 2] += 0.8 * progress  # toward 0 from -1.6

            if progress >= 1.0:
                self._lift_done = True

            self.robot.set_adhesion(self.lifted_leg, 0.0)
        elif self.sim_time <= 2.0:
            self.robot.set_adhesion(self.lifted_leg, 1.0)
        else:
            self.robot.set_adhesion(self.lifted_leg, 0.0)

        # Support legs: keep adhesion on
        for leg in self.robot.LEGS:
            if leg != self.lifted_leg:
                self.robot.set_adhesion(leg, 1.0)

        tau_actual, tau_demand = self.joint_ctrl.compute(
            q, dq, q_des, dq_des, tau_ff, self.control_dt
        )
        self.robot.set_joint_torques(tau_actual)

    def log_data(self):
        row = {"time": self.sim_time, "lifted_leg": self.lifted_leg}
        base_pos = self.robot.get_base_pos()
        row.update({"base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2]})

        tau = self.robot.get_joint_torques()
        tau_d = self.joint_ctrl.tau_demand
        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"tau_{name}"] = tau[i]
            row[f"tau_demand_{name}"] = tau_d[i]

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
    args = parser.parse_args()

    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "scene_a2.xml",
    )
    test = TestThreeLegSupport(scene_path, lifted_leg=args.leg)
    test.run(duration=8.0)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    test.logger.save(os.path.join(results_dir, "csv", f"T3-T6_lift_{args.leg}.csv"))


if __name__ == "__main__":
    main()
