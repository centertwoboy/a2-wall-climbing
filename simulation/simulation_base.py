"""
Simulation base class — provides common setup for all test cases.
Updated for real A2 model (26.5 kg, hip-X, thigh-Y, calf-Y).
"""

import os
import sys
import yaml
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.robot_interface import RobotInterface
from controllers.joint_controller import JointController
from controllers.adhesion_controller import AdhesionController
from controllers.gait_state_machine import GaitStateMachine
from controllers.trajectory_generator import FootTrajectory
from controllers.leg_kinematics import LegKinematics
from analysis.data_logger import DataLogger


class SimulationBase:
    """Base class for all simulation test cases."""

    def __init__(self, scene_path: str, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
            )

        # Load configurations
        with open(os.path.join(config_dir, "controller.yaml"), "r") as f:
            self.cfg = yaml.safe_load(f)

        with open(os.path.join(config_dir, "robot_a2.yaml"), "r") as f:
            self.robot_cfg = yaml.safe_load(f)

        # Load MuJoCo model
        print(f"[Sim] Loading scene: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.control_decimation = self.cfg["simulation"]["control_decimation"]
        self.control_dt = self.dt * self.control_decimation

        # Robot interface
        self.robot = RobotInterface(self.model, self.data)

        # Joint controller — per-joint limits from A2 specs
        cfg_gains = self.cfg["gains"]
        kp_vals = []
        kd_vals = []
        joint_limits = []
        for leg in self.robot.LEGS:
            kp_vals.extend([cfg_gains["kp_hip"], cfg_gains["kp_thigh"], cfg_gains["kp_calf"]])
            kd_vals.extend([cfg_gains["kd_hip"], cfg_gains["kd_thigh"], cfg_gains["kd_calf"]])
            t_lim = self.cfg["torque"]["motor_limits"]
            joint_limits.extend([t_lim["hip"], t_lim["thigh"], t_lim["calf"]])

        self.joint_ctrl = JointController(
            kp=np.array(kp_vals),
            kd=np.array(kd_vals),
            torque_limit=self.cfg["torque"]["demand_limit"],
            max_torque_rate=self.cfg["torque"]["max_rate"],
        )
        self._joint_limits = np.array(joint_limits)

        # Adhesion controller
        self.adhesion_ctrl = AdhesionController(
            switch_time=self.cfg["adhesion"]["switch_time"],
            max_force=self.cfg["adhesion"]["max_force"],
        )

        # Gait state machine
        self.gait = GaitStateMachine(
            swing_time=self.cfg["gait"]["swing_time"],
            support_time=self.cfg["gait"]["support_time"],
            step_length=self.cfg["gait"]["step_length"],
            step_height=self.cfg["gait"]["step_height"],
        )

        # Foot trajectory generator
        wall_normal = np.array(self.cfg["environment"]["wall_normal"])
        self.foot_traj = FootTrajectory(wall_normal)

        # Leg kinematics — one per leg with real A2 offsets
        offsets = self.robot_cfg["robot"]["offsets"]
        self.leg_ik = {}
        hip_offsets = {
            "FL": np.array([offsets["hip_x"], offsets["hip_y"], 0]),
            "FR": np.array([offsets["hip_x"], -offsets["hip_y"], 0]),
            "RL": np.array([-offsets["hip_x"], offsets["hip_y"], 0]),
            "RR": np.array([-offsets["hip_x"], -offsets["hip_y"], 0]),
        }
        for leg in self.robot.LEGS:
            self.leg_ik[leg] = LegKinematics(
                hip_offset=hip_offsets[leg],
                thigh_offset=np.array(offsets["thigh"]),
                calf_offset=np.array(offsets["calf"]),
                foot_offset=np.array(offsets["foot"]),
            )

        # Data logger
        self.logger = DataLogger()

        # Simulation time tracking
        self.sim_time = 0.0
        self.step_count = 0

    def get_foot_positions(self) -> dict:
        return {leg: self.robot.get_foot_pos(leg) for leg in self.robot.LEGS}

    def get_foot_velocities(self) -> dict:
        return {leg: self.robot.get_foot_vel(leg) for leg in self.robot.LEGS}

    def get_contact_forces(self) -> dict:
        """Get normal contact forces for each foot via foot sensors."""
        forces = {leg: 0.0 for leg in self.robot.LEGS}
        # Use force sensors if available
        for leg in self.robot.LEGS:
            sensor_name = f"{leg}_foot_force"
            try:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                if sid >= 0:
                    adr = self.model.sensor_adr[sid]
                    f = self.data.sensordata[adr:adr+3]
                    # Normal component is the wall-facing component (Y direction)
                    forces[leg] = abs(f[1])  # Y is wall normal
            except Exception:
                pass

        # Fallback: check contact pairs as well
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1_name = ""
            g2_name = ""
            try:
                g1_name = self.model.geom(contact.geom1).name or ""
                g2_name = self.model.geom(contact.geom2).name or ""
            except Exception:
                continue

            for leg in self.robot.LEGS:
                magnet_name = f"{leg}_magnet_geom"
                if (g1_name == magnet_name and "wall" in g2_name) or \
                   (g2_name == magnet_name and "wall" in g1_name):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    frame = contact.frame.reshape(3, 3)
                    forces[leg] += abs(np.dot(wrench[:3], frame[:, 0]))

        return forces

    def step_simulation(self):
        """Advance simulation by one timestep with control update."""
        self.step_count += 1
        self.sim_time = self.data.time

        if self.step_count % self.control_decimation == 0:
            self.control_update()

        mujoco.mj_step(self.model, self.data)
        self.log_data()

    def control_update(self):
        """Override in subclasses."""
        pass

    def log_data(self):
        """Override in subclasses."""
        pass

    def run(self, duration: float, render: bool = None, record_path: str = None):
        """Run simulation for specified duration."""
        if render is None:
            render = os.environ.get("A2_RENDER", "0") == "1"
        if record_path is None:
            record_path = os.environ.get("A2_RECORD", "")
        if record_path and not record_path.startswith("/"):
            record_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                record_path)
        n_steps = int(duration / self.dt)
        print(f"[Sim] Running {duration:.1f}s ({n_steps} steps)...")

        viewer = None
        if render:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                   show_left_ui=False,
                                                   show_right_ui=False)
            viewer.cam.lookat = [0, 0, 0.4]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = 25
            print("[Sim] Viewer launched (close window or press Esc to exit)")

        recorder = None
        frames = []
        video_fps = 30
        capture_every = max(1, int(1.0 / (video_fps * self.dt)))
        if record_path:
            import imageio
            import mujoco as _mj
            os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
            renderer = _mj.Renderer(self.model, 480, 640)
            print(f"[Sim] Recording to {record_path} ({video_fps} fps)")

        try:
            for i in range(n_steps):
                self.step_simulation()
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                elif viewer is not None:
                    break

                if record_path and i % capture_every == 0:
                    renderer.update_scene(self.data)
                    frames.append(renderer.render())

        finally:
            if viewer is not None:
                viewer.close()
            if record_path and frames:
                print(f"[Sim] Writing video ({len(frames)} frames)...")
                imageio.mimsave(record_path, frames, fps=video_fps)
                print(f"[Sim] Video saved to {record_path}")

        print(f"[Sim] Done. Sim time: {self.data.time:.3f}s")
