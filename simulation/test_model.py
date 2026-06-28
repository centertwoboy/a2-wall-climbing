"""
T1: Model loading and verification with real A2 MJCF/URDF.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "scene_a2.xml",
    )

    print("=" * 60)
    print("T1: A2 Model Loading and Verification")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print(f"\nModel dimensions:")
    print(f"  nq     = {model.nq}   (expected: 19 = 7 base + 12 joints)")
    print(f"  nv     = {model.nv}   (expected: 18 = 6 base + 12 joints)")
    print(f"  nu     = {model.nu}   (expected: 16 = 12 motors + 4 adhesion)")
    print(f"  nbody  = {model.nbody}")
    print(f"  njnt   = {model.njnt}")
    print(f"  ngeom  = {model.ngeom}")

    # Mass audit
    print(f"\nBody mass audit:")
    total_mass = 0.0
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mass = model.body_mass[body_id]
        if mass > 0:
            total_mass += mass
            print(f"  [{body_id:2d}] {name:30s}  mass = {mass:7.3f} kg")
    print(f"\n  Total mass: {total_mass:.3f} kg  (target: ~27.5 kg with magnets)")

    # Joint audit
    print(f"\nJoint configuration:")
    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if name:
            qpos_adr = model.jnt_qposadr[jnt_id]
            dof_adr = model.jnt_dofadr[jnt_id]
            jnt_type = model.jnt_type[jnt_id]
            jnt_range = model.jnt_range[jnt_id] if model.jnt_limited[jnt_id] else None
            type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            axis_str = ""
            if jnt_type == 3:
                axis_str = f" axis={model.jnt_axis[jnt_id]}"
            print(f"  {name:25s} type={type_names.get(jnt_type, '?')}, "
                  f"qpos_adr={qpos_adr}, dof_adr={dof_adr}, "
                  f"range={jnt_range}{axis_str}")

    # Actuator audit
    print(f"\nActuator list:")
    for act_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        ctrl_range = model.actuator_ctrlrange[act_id]
        print(f"  [{act_id:2d}] {name:25s}  ctrlrange={ctrl_range}")

    print("\n[T1] Model verification complete.")


if __name__ == "__main__":
    main()
