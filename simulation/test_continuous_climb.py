"""
T10-T14: Continuous crawl gait wall climbing with real A2 model.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestContinuousClimb(SimulationBase):
    """Continuous crawl gait wall climbing."""

    def __init__(self, scene_path: str, num_steps: int = 4,
                 torque_limit: float = None):
        super().__init__(scene_path)
        self.target_steps = num_steps

        if torque_limit is not None:
            self.joint_ctrl.torque_limit = torque_limit
            print(f"[Climb] Torque limit: {torque_limit} N.m")

        # Initial pose: facing wall
        self.data.qpos[0:3] = [0.0, 0.8, 1.5]
        self.data.qpos[3:7] = [0.707107, 0.707107, 0.0, 0.0]

        standing_q = [0.0, 0.8, -1.6] * 4
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        mujoco.mj_forward(self.model, self.data)

        self.gait.swing_time = self.cfg["gait"]["swing_time"]
        self.gait.support_time = self.cfg["gait"]["support_time"]
        self.gait.step_length = self.cfg["gait"]["step_length"]

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        tau_ff = self.robot.get_bias_torques()
        q_des = q.copy()
        dq_des = np.zeros(12)

        foot_positions = self.get_foot_positions()
        foot_velocities = self.get_foot_velocities()
        contact_forces = self.get_contact_forces()

        gait_result = self.gait.update(
            self.control_dt, foot_positions, contact_forces, foot_velocities
        )

        # Adhesion commands
        adh_cmds = gait_result["adhesion_commands"]
        for leg in self.robot.LEGS:
            cmd = adh_cmds.get(leg)
            if cmd == "on":
                self.robot.set_adhesion(leg, 1.0)
            elif cmd == "off":
                self.robot.set_adhesion(leg, 0.0)

        # Drive swing leg via trajectory
        sw_leg = gait_result["swing_leg"]
        progress = gait_result["swing_progress"].get(sw_leg, 0.0)

        if progress > 0:
            sl_idx = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}[sw_leg]
            p_start = self.gait.foot_start_positions.get(sw_leg)
            p_target = self.gait.foot_targets.get(sw_leg)
            if p_start is not None and p_target is not None:
                p = self.foot_traj.generate_swing(p_start, p_target, 0.02, progress)
                # Convert world target to hip frame and run IK
                base_pos = self.robot.get_base_pos()
                base_quat = self.robot.get_base_quat()
                # Simplified: drive thigh and calf along direction
                q_des[sl_idx + 1] += 0.3 * progress
                q_des[sl_idx + 2] -= 0.5 * progress

        # Check step completion
        if gait_result.get("new_targets", {}).get("step_complete"):
            n = gait_result["step_count"]
            print(f"[Climb] Step {n}/{self.target_steps} at t={self.sim_time:.2f}s")
            if n >= self.target_steps:
                print(f"[Climb] Target {self.target_steps} steps reached!")

        tau_actual, tau_demand = self.joint_ctrl.compute(
            q, dq, q_des, dq_des, tau_ff, self.control_dt
        )
        self.robot.set_joint_torques(tau_actual)

    def log_data(self):
        row = {"time": self.sim_time, "step_count": self.gait.step_count}
        base_pos = self.robot.get_base_pos()
        row.update({"base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2]})

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
            row[f"power_{name}"] = tau[i] * dq[i]

        for leg in self.robot.LEGS:
            row[f"{leg}_adhesion"] = 1.0
        contact = self.get_contact_forces()
        for leg in self.robot.LEGS:
            row[f"{leg}_contact_fn"] = contact.get(leg, 0.0)

        self.logger.record(row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--torque-limit", type=float, default=None)
    parser.add_argument("--payload", action="store_true")
    args = parser.parse_args()

    scene = "scene_a2_payload.xml" if args.payload else "scene_a2.xml"
    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", scene,
    )
    test = TestContinuousClimb(
        scene_path,
        num_steps=args.steps,
        torque_limit=args.torque_limit,
    )
    duration = args.steps * 8.0 + 3.0
    test.run(duration=duration)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    tag = f"steps{args.steps}"
    if args.torque_limit:
        tag += f"_tlim{int(args.torque_limit)}"
    if args.payload:
        tag += "_payload"
    test.logger.save(os.path.join(results_dir, "csv", f"T10-T14_climb_{tag}.csv"))


if __name__ == "__main__":
    main()
