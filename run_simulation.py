"""
A2 Wall-Climbing MuJoCo Simulation — Command-line entry point.

Usage:
    python run_simulation.py --test T2                # wall static test
    python run_simulation.py --test T10 --steps 4      # 4-step climb
    python run_simulation.py --test T12 --tlim 180     # torque-limited climb
    python run_simulation.py --test T3 --leg FL        # lift FL leg
    python run_simulation.py --sweep adhesion           # adhesion force sweep
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="A2 Wall-Climbing MuJoCo Simulation"
    )
    parser.add_argument("--test", type=str, default="T2",
                        help="Test case ID (T1-T14)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of crawl steps for continuous tests")
    parser.add_argument("--tlim", type=float, default=None,
                        help="Torque limit override [N.m]")
    parser.add_argument("--leg", type=str, default="FL",
                        choices=["FL", "FR", "RL", "RR"],
                        help="Target leg for single-leg tests")
    parser.add_argument("--payload", type=float, default=None,
                        help="Payload mass [kg]")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["adhesion", "friction", "speed", "step", "payload", "all"],
                        help="Run parameter sweep")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override simulation duration [s]")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots after simulation")
    parser.add_argument("--render", action="store_true",
                        help="Launch interactive MuJoCo viewer")
    parser.add_argument("--record", type=str, default=None,
                        help="Record video to path (e.g. results/videos/T1.mp4)")

    args = parser.parse_args()

    if args.render:
        os.environ["A2_RENDER"] = "1"
    if args.record:
        os.environ["A2_RECORD"] = args.record

    if args.sweep:
        from simulation.run_parameter_sweep import main as sweep_main
        sweep_main()
        return

    test_id = args.test.upper()
    print(f"[Main] Running test {test_id}")

    if test_id == "T1":
        from simulation.test_ground_stand import main as t_main
        t_main()

    elif test_id == "T2":
        from simulation.test_wall_static import main as t_main
        t_main()

    elif test_id in ("T2A", "T2B"):
        from simulation.test_wall_static import TestWallStatic
        base = os.path.dirname(os.path.abspath(__file__))
        scene_path = os.path.join(base, "models", "scene_a2.xml")
        orientation = test_id[-1]  # "A" or "B"
        test = TestWallStatic(scene_path, orientation=orientation)
        test.run(duration=10.0)
        results_dir = os.path.join(base, "results")
        test.logger.save(os.path.join(results_dir, "csv", f"T2{orientation}_wall_static.csv"))
        print(f"Data saved to results/csv/T2{orientation}_wall_static.csv")

    elif test_id in ("T3", "T4", "T5", "T6"):
        from simulation.test_three_leg_support import main as t_main
        sys.argv = [sys.argv[0], "--leg", args.leg]
        t_main()

    elif test_id in ("T7", "T8", "T9"):
        from simulation.test_single_step import main as t_main
        sys.argv = [sys.argv[0], "--leg", args.leg]
        t_main()

    elif test_id in ("T10", "T11", "T12", "T13", "T14"):
        from simulation.test_continuous_climb import main as t_main
        extra = ["--steps", str(args.steps)]
        if args.tlim is not None:
            extra += ["--torque-limit", str(args.tlim)]
        if args.payload is not None:
            extra += ["--payload", str(args.payload)]
        sys.argv = [sys.argv[0]] + extra
        t_main()

    else:
        print(f"Unknown test: {test_id}")
        print("Available: T1, T2, T2A, T2B, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14")

    # Generate plots if requested
    if args.plot:
        print("[Main] Generating plots...")
        from analysis.plot_results import (
            plot_joint_torques, plot_base_attitude, plot_contact_forces
        )
        # Find latest CSV in results/
        results_dir = os.path.join(os.path.dirname(__file__), "results", "csv")
        csv_files = sorted(
            [f for f in os.listdir(results_dir) if f.endswith(".csv")],
            key=lambda x: os.path.getmtime(os.path.join(results_dir, x)),
            reverse=True,
        )
        if csv_files:
            from analysis.data_logger import DataLogger
            import pandas as pd

            latest = os.path.join(results_dir, csv_files[0])
            df = pd.read_csv(latest)
            temp_logger = DataLogger()
            for col in df.columns:
                temp_logger.data[col] = df[col].tolist()
            temp_logger.columns = list(df.columns)
            temp_logger._n_rows = len(df)

            figs_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
            base = os.path.splitext(csv_files[0])[0]
            plot_joint_torques(temp_logger, os.path.join(figs_dir, f"{base}_torque.png"))
            plot_base_attitude(temp_logger, os.path.join(figs_dir, f"{base}_attitude.png"))
            plot_contact_forces(temp_logger, os.path.join(figs_dir, f"{base}_contact.png"))


if __name__ == "__main__":
    main()
