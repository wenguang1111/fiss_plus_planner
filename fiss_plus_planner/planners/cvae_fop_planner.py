"""CVAE-FOP: FOP with learned sampling parameters and a dense FOP fallback.

Per planning cycle the cost-aware CVAE (CVAE_efficient_sampling/CVAE.py: CostAwareCVAE) proposes K
samples z = [d, v, T] from the last three BEV frames; they are clipped to the cycle's search space and
evaluated by the shared C++ backend (same generator, checks, cost and ranking as FOP), with the
clearance requirement. If none is feasible, the dense FOP grid (num_width x num_speed x num_t) is planned
instead (num_FOP_intervention), with the clearance requirement and, if allowed (SAFETY:
clearance_fallback) and nothing keeps it, without it, as every other planner does.

Config section CVAE_FOP:
    run          run directory of train_cost_cvae.py (config.json, normalizer/, model_best.pth)
    num_samples  K
    sampler      cvae | uniform | marginal (ablation: the same K from the search space, or from the
                 training targets' normal distribution, without the scene)
    seed         samples of cycle t are drawn with seed + t
    warm_start   add the previous cycle's executed z as sample K + 1 (as CEM starts from its previous best)
    log_cycles   keep per-cycle diagnostics in cycle_log (written to OUTPUT_DIR/cvae_cycles)
    regret       also plan the dense grid in cycles solved by the samples, for the regret (untimed)

time_untimed is the part of the last plan() call that is not planning: rendering the BEV frame (in a
vehicle the raster comes from perception) and the regret diagnostics. cycle_log rows: stage (samples /
grid / failed: what produced the plan), counts and rejections of the K samples, J of the samples' best,
of the grid and of the plan, the chosen z and the stage times. Stages: samples, grid, grid_no_clearance
(the grid without the clearance requirement), failed.
"""
import time

import numpy as np
import torch

from CVAE_efficient_sampling.CVAE import CostAwareCVAE
from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper, to_cpp_state, to_py_trajectory
from fiss_plus_planner.planners.frenet_optimal_planner import Stats
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer


class CVAE_FOP_Wrapper(FOP_CPP_Wrapper):

    def __init__(self, planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices, number_threads=1,
                 runtime_measurement=False, cvae_cfg: dict = None, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices, number_threads,
                         runtime_measurement)
        cfg = cvae_cfg or {}
        self.k = int(cfg.get("num_samples", 16))
        self.sampler = cfg.get("sampler", "cvae")
        self.seed = int(cfg.get("seed", 0))
        self.log_cycles = bool(cfg.get("log_cycles", False))
        self.regret = bool(cfg.get("regret", False))
        self.warm_start = bool(cfg.get("warm_start", False))
        self.prev_z = None   # z of the previous cycle's plan (warm start)
        self.scenario_dir, self.scenario_name = scenario_dir, scenario_file.removesuffix(".xml")
        # The CPU work of the sampler is small; torch's default pool (one thread per core) only
        # oversubscribes the CPU when several planner processes run in parallel
        torch.set_num_threads(max(1, number_threads))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cvae = CostAwareCVAE(cfg["run"], self.device)
        self.drawer = None   # built with the reference line (generate_frenet_frame)
        self.frames = []     # BEV history, oldest first
        self.time_untimed = 0.0
        self.num_FOP_intervention = 0
        self.cycle_stats = Stats()
        self.cycle_log = []
        self.cpp_settings = self.cpp_planner.settings  # clearance_fallback is switched off for the samples
        self.cvae.generate_samples([np.zeros((self.cvae.img_size,) * 2, np.uint8)], self.k,  # CUDA warm-up
                                   ego=np.zeros(8) if self.cvae.uses_ego else None)

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        csp, ref_pts = super().generate_frenet_frame(centerline_pts)
        # As the drawer that rendered the training images (planning.py, Collect_Data_For_ML)
        self.drawer = ScenarioDrawer(self.scenario_name, self.scenario_dir, ref_ego_lane_pts=ref_pts,
                                     vehicle_params=self.vehicle, obstacles_array=self.obstacles_array,
                                     obstacles_num_vertices=self.obstacles_num_vertices)
        return csp, ref_pts

    def get_stats(self) -> Stats:
        return self.cycle_stats

    def draw_samples(self, space, time_step_now: int, ego=None) -> np.ndarray:
        """K samples [d, v, T] of the configured sampler (before clipping); ego: the raw ego state for a CVAE
        trained with --ego."""
        if self.sampler == "cvae":
            torch.manual_seed(self.seed + time_step_now)
            return self.cvae.generate_samples(self.frames, self.k, ego=ego)
        rng = np.random.default_rng(self.seed + time_step_now)
        if self.sampler == "uniform":
            lo, hi = np.array(space[0::2]), np.array(space[1::2])
            return lo + (hi - lo) * rng.random((self.k, 3))
        scaler = self.cvae.normalizer.target_scaler  # "marginal"
        return scaler.mean_ + scaler.scale_ * rng.standard_normal((self.k, 3))

    def plan(self, frenet_state, max_target_speed: float, obstacles: list, time_step_now: int = 0,
             initial_state=None, desired_speed: float = None):
        t0 = time.perf_counter()
        if self.sampler == "cvae":
            self.frames = (self.frames + [self.drawer.create_scenario_img_at_time_step(time_step_now, initial_state)])[-3:]
        t1 = time.perf_counter()
        self.num_FOP_intervention = 0
        self.cycle_stats = Stats()
        v_des = -1.0 if desired_speed is None else desired_speed
        cpp_state = to_cpp_state(frenet_state)

        # 1. K samples, clipped to the search space, evaluated with the clearance requirement
        best, sample_stats, n_outside = None, None, 0
        space = self.cpp_planner.search_space(frenet_state.s, max_target_speed)
        if space is not None:
            lo, hi = np.array(space[0::2]), np.array(space[1::2])
            # Ego state and task as in the training data (planning.py cem_cycle_record, cem_dataset.EGO_COLUMNS)
            ego = None if not self.cvae.uses_ego else [
                initial_state.velocity, initial_state.acceleration, frenet_state.s_d, frenet_state.s_dd,
                frenet_state.d, frenet_state.d_d, frenet_state.d_dd,
                max_target_speed if desired_speed is None else desired_speed]
            z = self.draw_samples(space, time_step_now, ego)
            if self.warm_start and self.prev_z is not None:
                z = np.vstack([z, self.prev_z])
            n_outside = int(np.any((z < lo) | (z > hi), axis=1).sum())
            z = np.clip(z, lo, hi)
        t2 = time.perf_counter()
        if space is not None:
            self.cpp_settings.clearance_fallback = False
            best = self.cpp_planner.best_traj_generation(cpp_state, [tuple(map(float, zi)) for zi in z],
                                                         max_target_speed, time_step_now, self.number_threads, v_des)
            self.cpp_settings.clearance_fallback = self.settings.clearance_fallback
            sample_stats = self.cpp_planner.get_stats()
            self.cycle_stats += Stats.from_cpp(sample_stats)
        t3 = time.perf_counter()

        # 2. Fallback: the dense FOP grid with the clearance requirement, then without it (if allowed)
        stage = "samples"
        if best is None or not best.is_generated:
            stage, self.num_FOP_intervention = "grid", 1
            best = self.cpp_planner.plan(cpp_state, max_target_speed, time_step_now, self.number_threads, v_des)
            grid_stats = self.cpp_planner.get_stats()
            self.cycle_stats += Stats.from_cpp(grid_stats)
            if not best.is_generated:
                stage = "failed"
            elif grid_stats.num_clearance_fallbacks:
                stage = "grid_no_clearance"
        t4 = time.perf_counter()

        j_grid = best.cost_final if stage.startswith("grid") else np.nan
        if self.regret and stage == "samples":  # untimed: the grid on the same state
            grid = self.cpp_planner.plan(cpp_state, max_target_speed, time_step_now, self.number_threads, v_des)
            j_grid = grid.cost_final if grid.is_generated else np.inf
        self.time_untimed = (t1 - t0) + (time.perf_counter() - t4)

        warm_chosen = (stage == "samples" and self.warm_start and self.prev_z is not None
                       and np.allclose([best.sampling_param.d, best.sampling_param.s_d, best.sampling_param.t],
                                       np.clip(self.prev_z, lo, hi), rtol=0, atol=1e-12))
        self.prev_z = (np.array([best.sampling_param.d, best.sampling_param.s_d, best.sampling_param.t])
                       if best.is_generated else None)

        if self.log_cycles:
            st = sample_stats
            self.cycle_log.append({
                "time_step": time_step_now, "stage": stage, "v": frenet_state.s_d, "v_max": max_target_speed,
                "low_speed": frenet_state.s_d < self.settings.low_speed_threshold,
                "num_samples": len(z) if st else 0, "num_outside": n_outside, "warm_chosen": warm_chosen,
                "num_feasible": st.num_collision_free if st else 0,
                "rejected_dynamic": st.num_rejected_dynamic if st else 0,
                "rejected_offroad": st.num_rejected_offroad if st else 0,
                "rejected_clearance": st.num_rejected_clearance if st else 0,
                "rejected_collision": st.num_rejected_collision if st else 0,
                "J_samples": best.cost_final if stage == "samples" else np.nan,
                "J_grid": j_grid,
                "J_plan": best.cost_final if best.is_generated else np.nan,
                "d": best.sampling_param.d if best.is_generated else np.nan,
                "v_end": best.sampling_param.s_d if best.is_generated else np.nan,
                "T": best.sampling_param.t if best.is_generated else np.nan,
                "t_render_ms": 1e3 * (t1 - t0), "t_sample_ms": 1e3 * (t2 - t1), "t_eval_ms": 1e3 * (t3 - t2),
                "t_grid_ms": 1e3 * (t4 - t3), "t_plan_ms": 1e3 * (t4 - t1),
            })
        if not best.is_generated:
            return None
        self.best_traj = to_py_trajectory(best)
        self.best_traj.stage = stage  # shown in the GIF frames
        if not self.doing_runtime_measurement:  # trajectories for SAVE_GIF (planning.py)
            self.all_trajs.append([self.best_traj])
        return self.best_traj
