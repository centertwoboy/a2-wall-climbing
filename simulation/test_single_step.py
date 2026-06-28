"""
T7-T9: Single leg full motion — detach, swing, reattach, verify contact impact.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class TestSingleStep(SimulationBase):
    """Complete single-leg cycle: detach -> swing -> reattach."""

    def __init__(self, scene_path: str, swing_leg: str = "FL"):
        super().__init__(scene_path)
        self.swing_leg = swing_leg

        # Initial pose: on wall
        self.data.qpos[0:3] = [0.0, 0.8, 1.5]
        self.data.qpos[3:7] = [0.707, 0.707, 0, 0]

        standing_q = [0.0, 0.8, -1.6] * 4  # all legs
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]

        mujoco.mj_forward(self.model, self.data)

        self._phase = "settle"       # settle -> detach -> lift -> swing -> approach -> attach
        self._phase_start = 0.0
        self._foot_start = None
        self._foot_target = None

        self.settle_time = 2.0
        self.detach_time = 0.3
        self.lift_time = 0.2
        self.swing_time = 1.5
        self.approach_time = 0.3
        self.attach_time = 0.5

        self.climb_dir = np.array(self.cfg["environment"]["climb_direction"])
        self.wall_normal = np.array(self.cfg["environment"]["wall_normal"])

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        tau_ff = self.robot.get_bias_torques()
        q_des = q.copy()
        dq_des = np.zeros(12)

        sl = self.swing_leg
        sl_idx = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}[sl]

        # Phase transitions
        elapsed = self.sim_time - self._phase_start
        contact_forces = self.get_contact_forces()
        foot_vel = self.robot.get_foot_vel(sl)

        if self._phase == "settle" and elapsed > self.settle_time:
            self._phase = "detach"
            self._phase_start = self.sim_time
            self._foot_start = self.robot.get_foot_pos(sl).copy()
            self._foot_target = self._foot_start + self.gait.step_length * self.climb_dir
            print(f"[T7-T9] Detaching {sl} at t={self.sim_time:.2f}s")

        elif self._phase == "detach" and elapsed > self.detach_time:
            self._phase = "lift"
            self._phase_start = self.sim_time
            print(f"[T7-T9] Lifting {sl}")

        elif self._phase == "lift" and elapsed > self.lift_time:
            self._phase = "swing"
            self._phase_start = self.sim_time
            print(f"[T7-T9] Swinging {sl}")

        elif self._phase == "swing" and elapsed > self.swing_time:
            self._phase = "approach"
            self._phase_start = self.sim_time
            print(f"[T7-T9] Approaching wall with {sl}")

        elif self._phase == "approach" and elapsed > self.approach_time:
            self._phase = "attach"
            self._phase_start = self.sim_time
            print(f"[T7-T9] Attaching {sl}")

        elif self._phase == "attach" and elapsed > self.attach_time:
            self._phase = "done"
            print(f"[T7-T9] Single step complete for {sl}")

        # Apply control per phase
        if self._phase in ["settle"]:
            # Hold all legs, full adhesion
            for leg in self.robot.LEGS:
                self.robot.set_adhesion(leg, 1.0)

        elif self._phase in ["detach", "lift", "swing", "approach"]:
            # Release adhesion on swing leg
            self.robot.set_adhesion(sl, 0.0)
            for leg in self.robot.LEGS:
                if leg != sl:
                    self.robot.set_adhesion(leg, 1.0)

            # Compute swing leg desired position
            if self._phase == "detach":
                # Start reducing normal force only
                pass
            elif self._phase == "lift":
                progress = elapsed / self.lift_time
                clearance = 0.02 * progress
                p_current = self.robot.get_foot_pos(sl)
                p_lifted = p_current - clearance * self.wall_normal
                # IK to find joint angles
                foot_local = p_lifted - self.robot.get_base_pos()
                # Simplified: just offset thigh/calf for lift
                q_des[sl_idx + 1] += 0.15 * progress  # thigh
                q_des[sl_idx + 2] -= 0.3 * progress    # calf
            elif self._phase == "swing":
                progress = np.clip(elapsed / self.swing_time, 0, 1)
                # Generate trajectory
                p = self.foot_traj.generate_swing(
                    self._foot_start, self._foot_target,
                    clearance=0.02, s=progress,
                )
                # Simplify: drive joints along direction
                q_des[sl_idx + 1] += 0.3 * progress   # thigh forward
                q_des[sl_idx + 2] -= 0.6 * progress   # calf
            elif self._phase == "approach":
                progress = elapsed / self.approach_time
                q_des[sl_idx + 1] = q[sl_idx + 1]
                q_des[sl_idx + 2] = q[sl_idx + 2] + 0.3 * progress

        elif self._phase in ["attach", "done"]:
            # Restore adhesion
            for leg in self.robot.LEGS:
                self.robot.set_adhesion(leg, 1.0)

        tau_actual, tau_demand = self.joint_ctrl.compute(
            q, dq, q_des, dq_des, tau_ff, self.control_dt
        )
        self.robot.set_joint_torques(tau_actual)

    def log_data(self):
        row = {"time": self.sim_time, "phase": self._phase, "swing_leg": self.swing_leg}
        base_pos = self.robot.get_base_pos()
        row.update({"base_x": base_pos[0], "base_y": base_pos[1], "base_z": base_pos[2]})

        q = self.robot.get_joint_positions()
        tau = self.robot.get_joint_torques()
        tau_d = self.joint_ctrl.tau_demand
        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"q_{name}"] = q[i]
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
    test = TestSingleStep(scene_path, swing_leg=args.leg)
    test.run(duration=6.0)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    test.logger.save(os.path.join(results_dir, "csv", f"T7-T9_single_step_{args.leg}.csv"))


if __name__ == "__main__":
    main()
