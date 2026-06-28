"""
Parameter sweep automation — batch run adhesion force, friction, speed,
step length, and payload variations.
"""

import os
import sys
import itertools
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation_base import SimulationBase


class ParameterSweepRunner(SimulationBase):
    """Minimal simulation runner for batch parameter sweeps."""

    def __init__(self, scene_path: str, params: dict, duration: float = 30.0):
        super().__init__(scene_path)
        self.test_duration = duration
        self.override_params = params

        # Apply parameter overrides
        if "adhesion_force" in params:
            gain = params["adhesion_force"]
            self.cfg["adhesion"]["max_force"] = gain
            self.cfg["adhesion"]["gain"] = gain
        if "step_length" in params:
            self.gait.step_length = params["step_length"]
        if "swing_time" in params:
            self.gait.swing_time = params["swing_time"]

        # Initial pose on wall
        self.data.qpos[0:3] = [0.0, 0.8, 1.5]
        self.data.qpos[3:7] = [0.707, 0.707, 0, 0]
        standing_q = [0.0, 0.8, -1.6] * 4
        for i, qid in enumerate(self.robot._qpos_ids):
            self.data.qpos[qid] = standing_q[i]
        mujoco.mj_forward(self.model, self.data)

    def control_update(self):
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        tau_ff = self.robot.get_bias_torques()

        foot_positions = self.get_foot_positions()
        foot_velocities = self.get_foot_velocities()
        contact_forces = self.get_contact_forces()

        gait_result = self.gait.update(
            self.control_dt, foot_positions, contact_forces, foot_velocities
        )

        adh_cmds = gait_result["adhesion_commands"]
        for leg in self.robot.LEGS:
            if adh_cmds.get(leg) == "on":
                self.robot.set_adhesion(leg, 1.0)
            elif adh_cmds.get(leg) == "off":
                self.robot.set_adhesion(leg, 0.0)

        q_des = q.copy()
        dq_des = np.zeros(12)
        tau_actual, tau_demand = self.joint_ctrl.compute(
            q, dq, q_des, dq_des, tau_ff, self.control_dt
        )
        self.robot.set_joint_torques(tau_actual)

    def log_data(self):
        row = {"time": self.sim_time}
        tau = self.robot.get_joint_torques()
        leg_names = ["FL_hip", "FL_thigh", "FL_calf",
                     "FR_hip", "FR_thigh", "FR_calf",
                     "RL_hip", "RL_thigh", "RL_calf",
                     "RR_hip", "RR_thigh", "RR_calf"]
        for i, name in enumerate(leg_names):
            row[f"tau_{name}"] = tau[i]
        self.logger.record(row)

    def run_sweep(self):
        """Run sweep and return max absolute torque across all joints."""
        n_steps = int(self.test_duration / self.dt)
        for _ in range(n_steps):
            self.step_simulation()
        all_tau = []
        for name in self.logger.columns:
            if name.startswith("tau_"):
                all_tau.extend(np.abs(self.logger.data[name]))
        return float(np.max(all_tau)) if all_tau else 0.0


def main():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
    )
    with open(os.path.join(config_dir, "test_cases.yaml"), "r") as f:
        test_cfg = yaml.safe_load(f)

    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "scene_a2.xml",
    )
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    sweeps = test_cfg["parameter_sweeps"]

    # Adhesion force sweep
    print("\n" + "=" * 50)
    print("Sweep: Adhesion Force")
    print("=" * 50)
    for F in sweeps["adhesion"]["values"]:
        runner = ParameterSweepRunner(scene_path, {"adhesion_force": F}, duration=30.0)
        max_tau = runner.run_sweep()
        print(f"  F_adh = {F:5.0f} N  ->  max |tau| = {max_tau:.1f} N.m")

    # Friction sweep
    print("\n" + "=" * 50)
    print("Sweep: Friction Coefficient")
    print("=" * 50)
    for mu in sweeps["friction"]["values"]:
        runner = ParameterSweepRunner(scene_path, {"friction": mu}, duration=30.0)
        max_tau = runner.run_sweep()
        print(f"  mu = {mu:.1f}  ->  max |tau| = {max_tau:.1f} N.m")

    # Speed sweep (via swing_time)
    print("\n" + "=" * 50)
    print("Sweep: Speed")
    print("=" * 50)
    step_len = 0.05
    for v in sweeps["speed"]["values"]:
        swing_t = step_len / v if v > 0 else 1.5
        runner = ParameterSweepRunner(scene_path, {"swing_time": swing_t}, duration=30.0)
        max_tau = runner.run_sweep()
        print(f"  v = {v:.2f} m/s (T_swing={swing_t:.1f}s)  ->  max |tau| = {max_tau:.1f} N.m")

    print("\nParameter sweep complete.")


if __name__ == "__main__":
    main()
