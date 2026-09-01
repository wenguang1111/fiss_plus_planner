import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from commonroad.scenario.state import InitialState

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.utils import transform_points_from_ego, transform_points_to_ego
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer


def _load_world_model_inference():
    """Load trajectory_planner_world_model/inference.py so its generate_samples can be called directly.

    That project is a separate, unpackaged repo checked out next to this one (no __init__.py / install
    hooks), so it is loaded by file path rather than by a normal import. Its own top-level
    `from models.CVAE import CVAE` only resolves if the repo directory is on sys.path, so we append it
    (append, not insert, to minimise shadowing of any other package named `models`).

    Both the CVAE class and generate_samples are taken from that module, so this planner always runs the
    exact inference code that ships with the trained model.
    """
    world_model_dir = Path(__file__).parent.parent.parent / 'trajectory_planner_world_model'
    if str(world_model_dir) not in sys.path:
        sys.path.append(str(world_model_dir))

    inference_path = world_model_dir / 'inference.py'
    spec = importlib.util.spec_from_file_location('fiss_plus_planner._world_model_inference', inference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SparsePlannerWorldModelSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5,
                 scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.num_samples: int = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Must match the checkpoint below exactly (verified against its state_dict tensor shapes).
        self.hidden_dim: int = 32
        self.input_dim: int = 2
        self.traj_len: int = 20
        self.img_size: int = 256
        self.latent_dim: int = 64

        # generate_samples hardcodes traj_len=20 and _resample_path's ds=2.0; mirrored here only so we can
        # compute how long the reference path must be for it to yield a full traj_len steps.
        self.reference_resample_ds: float = 2.0
        self.reference_path_lookahead_m: float = ScenarioDrawer.VIEW_SIZE_DEFAULT / 2.0

        project_root = Path(__file__).parent.parent.parent
        self.weights_path = project_root / 'trajectory_planner_world_model' / 'CVAE_baseline_weights.pt'


class SparsePlannerWorldModel(FrenetOptimalPlanner):
    """Sparse planner backed by the world-model-conditioned CVAE trajectory generator.

    Unlike SparsePlanner/SparsePlannerFOP (whose CVAE predicts Frenet sampling parameters that still go
    through calc_frenet_paths/calc_global_paths), this model was trained in an ego-centered frame (ego at
    the origin, yaw=0) and predicts full candidate trajectories directly as (x, y) points, one every
    tick_t=0.1s. Candidates are transformed back into the global frame, re-projected into Frenet (s, d) so
    the existing check_constraints / check_collision_multithread / cost_function pipeline can be reused
    unchanged, and the minimum-cost feasible candidate is returned.
    """

    def __init__(self, planner_settings: SparsePlannerWorldModelSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)

        self.scenario_drawer = ScenarioDrawer(
            self.settings.scenario_file,
            self.settings.scenario_dir,
            obstacles_array=self.obstacles_array,
            obstacles_num_vertices=self.obstacles_num_vertices,
        )

        self._world_model = _load_world_model_inference()
        self.model = self._world_model.CVAE(
            hidden_dim=self.settings.hidden_dim,
            input_dim=self.settings.input_dim,
            input_len=self.settings.traj_len,
            img_size=self.settings.img_size,
            latent_dim=self.settings.latent_dim,
        )
        self.model.load_state_dict(torch.load(str(self.settings.weights_path), map_location=self.settings.device))
        self.model = self.model.to(self.settings.device)
        self.model.eval()

        self.image_history: List[Tuple[int, Image.Image]] = []
        self.ref_lane_pts_global: Optional[np.ndarray] = None
        self._ref_cum_s: Optional[np.ndarray] = None
        self.all_trajs = []
        self.time_image_generation = 0.0
        self.num_FOP_intervention = 0  # no dense fallback for this planner; kept for stats-collection compatibility

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        cubic_spline, ref_lane_pts = super().generate_frenet_frame(centerline_pts)
        self.ref_lane_pts_global = ref_lane_pts
        self._ref_cum_s = None
        return cubic_spline, ref_lane_pts

    def _update_image_history(self, time_step_now: int, current_state: InitialState) -> List[Image.Image]:
        img = self.scenario_drawer.create_scenario_img_at_time_step(time_step_now, current_state).convert("L")
        self.image_history.append((time_step_now, img))

        if time_step_now >= 2:
            return [self.image_history[-3][1], self.image_history[-2][1], self.image_history[-1][1]]
        elif time_step_now == 1:
            return [self.image_history[-2][1], self.image_history[-1][1], self.image_history[-1][1]]
        else:
            return [self.image_history[-1][1], self.image_history[-1][1], self.image_history[-1][1]]

    def _local_reference_path(self, ego_pos: np.ndarray, ego_yaw: float) -> Optional[np.ndarray]:
        """Reference centerline ahead of ego, transformed into the ego-centered frame the model expects."""
        ref_local_xy = transform_points_to_ego(self.ref_lane_pts_global[:, :2], ego_pos, ego_yaw)
        start_idx = int(np.argmin(np.linalg.norm(ref_local_xy, axis=1)))
        ref_path_ahead = ref_local_xy[start_idx:]

        if len(ref_path_ahead) < 2:
            return None

        seg_dists = np.linalg.norm(np.diff(ref_path_ahead, axis=0), axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_dists)])
        ref_path_ahead = ref_path_ahead[cum_dist <= self.settings.reference_path_lookahead_m]

        if len(ref_path_ahead) < 2:
            return None

        return self._extend_reference_path(ref_path_ahead)

    def _extend_reference_path(self, ref_path: np.ndarray) -> np.ndarray:
        """Extrapolate the reference path so it is long enough for generate_samples to resample traj_len steps.

        generate_samples resamples at ds and then truncates with [:traj_len] without padding, so a path
        shorter than that yields fewer than traj_len conditioning steps and the model's positional
        embedding (fixed at traj_len) fails to broadcast. The remaining reference path does get this short
        as ego approaches the goal, so extend it straight along its final heading instead.
        """
        ds = self.settings.reference_resample_ds
        required_length = (self.settings.traj_len + 1) * ds

        seg_dists = np.linalg.norm(np.diff(ref_path, axis=0), axis=1)
        total_length = float(seg_dists.sum())
        if total_length >= required_length:
            return ref_path

        direction = ref_path[-1] - ref_path[-2]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return ref_path
        direction = direction / norm

        num_extra = int(np.ceil((required_length - total_length) / ds))
        extra = ref_path[-1] + np.outer(np.arange(1, num_extra + 1) * ds, direction)
        return np.vstack([ref_path, extra])

    def _generate_local_trajectories(self, images_last_3_frames: List[Image.Image],
                                      ref_path_local: np.ndarray) -> List[List[List[float]]]:
        """Candidate trajectories in the ego-centered frame, via the world model's own generate_samples."""
        return self._world_model.generate_samples(
            self.model,
            images_last_3_frames,
            ref_path_local[:, 0],
            ref_path_local[:, 1],
            self.settings.num_samples,
            device=self.settings.device,
        )

    def _project_to_frenet_batch(self, x_all: np.ndarray, y_all: np.ndarray,
                                  yaw_all: np.ndarray, v_all: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Vectorized re-implementation of FrenetState.from_state's math for a whole batch of points at
        once. from_state itself sums polyline segment lengths in a plain Python loop per call, which is
        fine for the one initial-state conversion per planning step done elsewhere, but far too slow here
        where every point of every candidate trajectory (num_samples * traj_len per step) needs it.
        """
        poly = self.ref_lane_pts_global
        poly_x, poly_y, poly_yaw = poly[:, 0], poly[:, 1], poly[:, 2]

        if self._ref_cum_s is None:
            seg = np.hypot(np.diff(poly_x), np.diff(poly_y))
            self._ref_cum_s = np.concatenate([[0.0], np.cumsum(seg)])

        dists = np.hypot(poly_x[None, :] - x_all[:, None], poly_y[None, :] - y_all[:, None])
        nearest_idx = np.argmin(dists, axis=1)

        heading = np.arctan2(poly_y[nearest_idx] - y_all, poly_x[nearest_idx] - x_all)
        angle = np.abs(np.arctan2(np.sin(yaw_all - heading), np.cos(yaw_all - heading)))
        angle = np.minimum(2 * np.pi - angle, angle)
        next_wp_id = np.where(angle > np.pi / 2, nearest_idx + 1, nearest_idx)
        next_wp_id = np.clip(next_wp_id, 1, len(poly_x) - 1)
        prev_wp_id = next_wp_id - 1

        n_x = poly_x[next_wp_id] - poly_x[prev_wp_id]
        n_y = poly_y[next_wp_id] - poly_y[prev_wp_id]
        x_x = x_all - poly_x[prev_wp_id]
        x_y = y_all - poly_y[prev_wp_id]

        denom = n_x * n_x + n_y * n_y
        proj_norm = np.divide(x_x * n_x + x_y * n_y, denom, out=np.zeros_like(denom), where=denom != 0)
        proj_x, proj_y = proj_norm * n_x, proj_norm * n_y

        d = np.hypot(x_x - proj_x, x_y - proj_y)
        wp_yaw = poly_yaw[prev_wp_id]
        # d > 0 is left of the reference line; same cross-product test as FrenetState.from_state.
        d = np.where((n_x * x_y - n_y * x_x) < 0, -d, d)

        delta_yaw = np.arctan2(np.sin(yaw_all - wp_yaw), np.cos(yaw_all - wp_yaw))

        s = self._ref_cum_s[prev_wp_id]
        s_d = v_all * np.cos(delta_yaw)
        d_d = v_all * np.sin(delta_yaw)

        return s, d, s_d, d_d

    def _build_frenet_trajectory(self, global_xy: np.ndarray, yaw: np.ndarray,
                                  s_arr: np.ndarray, d_arr: np.ndarray,
                                  s_d_arr: np.ndarray, d_d_arr: np.ndarray) -> FrenetTrajectory:
        dt = self.settings.tick_t
        n = global_xy.shape[0]
        x, y = global_xy[:, 0], global_xy[:, 1]
        ds = np.hypot(np.diff(x), np.diff(y))

        fp = FrenetTrajectory()
        fp.t = [i * dt for i in range(n)]
        fp.x, fp.y, fp.yaw, fp.ds = x, y, yaw, ds
        fp.c = np.divide(np.diff(yaw), ds, out=np.zeros_like(ds), where=ds != 0)
        fp.c_d = np.gradient(fp.c, dt) if len(fp.c) > 1 else np.zeros(0)
        fp.c_dd = np.gradient(fp.c_d, dt) if len(fp.c_d) > 1 else np.zeros(0)

        fp.s, fp.d = s_arr.tolist(), d_arr.tolist()
        fp.s_d, fp.d_d = s_d_arr.tolist(), d_d_arr.tolist()
        s_dd, d_dd = np.gradient(s_d_arr, dt), np.gradient(d_d_arr, dt)
        fp.s_dd, fp.d_dd = s_dd.tolist(), d_dd.tolist()
        fp.s_ddd = np.gradient(s_dd, dt).tolist()
        fp.d_ddd = np.gradient(d_dd, dt).tolist()

        return fp

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list,
             time_step_now: int = 0, current_state: InitialState = None) -> Optional[FrenetTrajectory]:
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed

        time_start = time.time()
        images_last_3_frames = self._update_image_history(time_step_now, current_state)

        ego_pos = np.asarray(current_state.position, dtype=float)
        ego_yaw = float(current_state.orientation)
        ref_path_local = self._local_reference_path(ego_pos, ego_yaw)
        self.time_image_generation = time.time() - time_start

        if ref_path_local is None:
            self.all_trajs.append([])
            return None

        local_trajs = self._generate_local_trajectories(images_last_3_frames, ref_path_local)
        global_trajs = [transform_points_from_ego(np.asarray(traj, dtype=float), ego_pos, ego_yaw)
                        for traj in local_trajs]
        global_trajs = [traj for traj in global_trajs if traj.shape[0] >= 2]

        if len(global_trajs) == 0:
            self.all_trajs.append([])
            return None

        dt = self.settings.tick_t
        yaws, speeds = [], []
        for global_xy in global_trajs:
            x_d, y_d = np.diff(global_xy[:, 0]), np.diff(global_xy[:, 1])
            seg_yaw = np.arctan2(y_d, x_d)
            yaw = np.append(seg_yaw, seg_yaw[-1])
            ds = np.hypot(x_d, y_d)
            speed = np.append(ds / dt, (ds / dt)[-1])
            yaws.append(yaw)
            speeds.append(speed)

        # Project every point of every candidate to Frenet in one vectorized batch call (see
        # _project_to_frenet_batch's docstring for why this must not be done point-by-point).
        x_all = np.concatenate([g[:, 0] for g in global_trajs])
        y_all = np.concatenate([g[:, 1] for g in global_trajs])
        yaw_all = np.concatenate(yaws)
        v_all = np.concatenate(speeds)
        s_all, d_all, s_d_all, d_d_all = self._project_to_frenet_batch(x_all, y_all, yaw_all, v_all)

        candidates = []
        offset = 0
        for global_xy, yaw in zip(global_trajs, yaws):
            m = global_xy.shape[0]
            sl = slice(offset, offset + m)
            offset += m
            candidates.append(self._build_frenet_trajectory(
                global_xy, yaw, s_all[sl], d_all[sl], s_d_all[sl], d_d_all[sl]))

        self.all_trajs.append(candidates)
        self.stats.num_trajs_generated = len(candidates)

        fplist = self.check_constraints(candidates)
        self.stats.num_trajs_validated = len(fplist)

        fplist = self.check_collision_multithread(fplist, time_step_now)
        self.stats.num_collison_checks = len(fplist)

        fplist = self.cost_function.calc_cost(
            fplist, max_target_speed, self.obstacles_array, self.obstacles_num_vertices, time_step_now)

        if len(fplist) == 0:
            return None

        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp
        return self.best_traj
