"""Rebuild the trajectories of stored CEM candidates (CEM data set, Collect_Data_For_ML with CEM_CPP).

The data set stores per planning cycle the start state (contexts.parquet) and per scenario the points
of the planner's reference line (reference_line.parquet). A candidate's trajectory is fully determined
by these and its sampling parameters (d, v, T), so it is rebuilt with the planner's own generator and
Frenet -> Cartesian transform (FrenetPlanner.generate_trajectory) instead of being stored.

    from scripts.cem_trajectories import CEMTrajectories
    rebuild = CEMTrajectories("<OUTPUT_DIR>", "<scenario>")
    traj = rebuild.trajectory(time_step, d, v, T)     # frenet_planner_cpp.FrenetTrajectory (global frame)
    xy = rebuild.to_ego(traj, time_step)               # points in the ego frame of that cycle

python scripts/cem_trajectories.py --data <OUTPUT_DIR> --verify [--scenarios 20]
checks the rebuild: every cycle's executed plan against best_trajectory.parquet, and for all candidates
of some cycles the speed / acceleration violations against the stored labels.
Needs the C++ module (C_Planner/build); ENABLE_DATA_COLLECTION is not required.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fiss_plus_planner.planners.FOP_cpp_wrapper import frenet_planner_cpp as fp  # noqa: E402
from fiss_plus_planner.planners.common.utils import transform_points_to_ego  # noqa: E402

# Settings of the trajectory generator (the rest only matters for checks and costs)
GENERATOR_SETTINGS = ("tick_t", "max_t", "low_speed_threshold", "low_speed_min_lateral_length")
START_STATE = ("s", "s_d", "s_dd", "d", "d_d", "d_dd", "d_s", "d_ss")


class CEMTrajectories:
    """Trajectories of the candidates of one scenario."""

    def __init__(self, data_dir: str, scenario: str):
        self.dir = os.path.join(data_dir, "cem_data", scenario)
        self.info = json.load(open(os.path.join(data_dir, "cem_data", "dataset_info.json")))
        settings = fp.SettingParameters()
        for name in GENERATOR_SETTINGS:
            setattr(settings, name, self.info["planner_settings"][name])
        no_obstacles = np.zeros((1, 1, 4, 2)), np.zeros((1, 1), dtype=np.int32)
        self.planner = fp.FrenetPlanner(settings, fp.VehicleParams(), *no_obstacles, 1, 1, 4)
        reference = pd.read_parquet(os.path.join(self.dir, "reference_line.parquet")).to_numpy(np.float64)
        self.planner.generate_frenet_frame(np.ascontiguousarray(reference))
        self.contexts = pd.read_parquet(os.path.join(self.dir, "contexts.parquet")).set_index("time_step")

    def start_state(self, time_step: int):
        c = self.contexts.loc[time_step]
        state = fp.FrenetState()
        for name in START_STATE:
            setattr(state, name, float(c[name]))  # NaN d_s / d_ss: derived as in planning
        return state

    def trajectory(self, time_step: int, d: float, v: float, T: float):
        """Trajectory of sample (d, v, T) of the cycle, exactly as the planner evaluated it."""
        return self.planner.generate_trajectory(self.start_state(time_step), float(d), float(v), float(T), int(time_step))

    def to_ego(self, traj, time_step: int) -> np.ndarray:
        """Trajectory points (x, y) in the ego frame of the cycle (as the BEV image)."""
        c = self.contexts.loc[time_step]
        return transform_points_to_ego(np.column_stack([traj.x, traj.y]), np.array([c.x, c.y]), float(c.yaw))


def first_violation_share(exceeded: np.ndarray, n: int) -> float:
    """(N - k) / N for the first index k where exceeded is true, else 0 (TrajectoryEvaluator)."""
    k = np.flatnonzero(exceeded)
    return 0.0 if len(k) == 0 else (n - k[0]) / n


def verify(data_dir: str, n_scenarios: int, cycles_per_scenario: int = 3, seed: int = 0) -> bool:
    info = json.load(open(os.path.join(data_dir, "cem_data", "dataset_info.json")))
    v_max, a_max = info["vehicle"]["max_speed"], info["vehicle"]["max_accel"]
    scenarios = sorted(s for s in os.listdir(os.path.join(data_dir, "cem_data"))
                       if os.path.isdir(os.path.join(data_dir, "cem_data", s)))
    rng = np.random.default_rng(seed)
    best_err, label_err, n_best, n_cand = 0.0, 0.0, 0, 0
    for scenario in rng.permutation(scenarios)[:n_scenarios]:
        rebuild = CEMTrajectories(data_dir, scenario)
        best = pd.read_parquet(os.path.join(rebuild.dir, "best_trajectory.parquet")).set_index("time_step")
        for t, c in rebuild.contexts.iterrows():
            if np.isnan(c.best_d):
                continue
            xy = rebuild.to_ego(rebuild.trajectory(t, c.best_d, c.best_v, c.best_T), t)
            stored = np.column_stack([best.loc[t, "x"], best.loc[t, "y"]])
            assert xy.shape == stored.shape, (scenario, t, xy.shape, stored.shape)
            best_err = max(best_err, float(np.abs(xy - stored).max()))
            n_best += 1
        cand = pd.read_parquet(os.path.join(rebuild.dir, "candidates.parquet"))
        for t in rng.choice(rebuild.contexts.index, size=min(cycles_per_scenario, len(rebuild.contexts)), replace=False):
            for row in cand[(cand.time_step == t) & (cand.V_transform == 0)].itertuples():
                traj = rebuild.trajectory(t, row.d, row.v, row.T)
                s_d, s_dd, n = np.array(traj.s_d), np.array(traj.s_dd), len(traj.s)
                v_speed = first_violation_share((s_d > v_max) | (s_d < -1e-2), n)  # too fast or reversing
                v_acc = first_violation_share(np.abs(s_dd) > a_max, n)
                label_err = max(label_err, abs(v_speed - row.V_speed), abs(v_acc - row.V_acceleration))
                n_cand += 1
    ok = best_err < 1e-4 and label_err < 1e-6
    print(f"executed plans: {n_best} rebuilt, max |rebuilt - stored| = {best_err:.2e} m (float32 storage)")
    print(f"candidates: {n_cand} rebuilt, max |V_speed / V_acceleration - stored| = {label_err:.2e}")
    print("VERIFIED" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="OUTPUT_DIR of the CEM collection")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--scenarios", type=int, default=20, help="scenarios checked by --verify")
    args = p.parse_args()
    if args.verify:
        sys.exit(0 if verify(args.data, args.scenarios) else 1)
