import time
from pathlib import Path
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import torch
from commonroad.scenario.state import InitialState

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.frenet_optimal_planner import Stats
from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper
from fiss_plus_planner.planners.sparse_planner_fop import SparsePlannerFOPSettings, SparsePlannerFOP
from fiss_plus_planner.planners.sparse_planning.cvae_paper import CVAEGraphSampler, CVAETRTSampler


class SparsePlannerSettings_CPP(SparsePlannerFOPSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t, scenario_dir, scenario_file)
        # CVAE inference: "graph_fp16" / "graph_fp32" (CUDA graph, CVAEGraphSampler), "trt_fp16" (TensorRT engine of the
        # fp16 model, CVAETRTSampler, engines cached in trt_cache_dir) or "eager" (same draws and samples as the Python
        # Sparse_FOP, for checking the C++ part)
        self.cvae_backend = "graph_fp16"
        self.trt_cache_dir = Path(__file__).parent.parent / "data/models/trt_cache"
        # Sampler of the N parameters per cycle (ablation of the learned distribution, all evaluated as the CVAE samples):
        # "cvae"; "marginal": N rows of the CVAE training targets (the distribution without the image condition);
        # "uniform": uniform in FOP's sampling box (d, v, T ranges of get_samples); "mean": N copies of the target mean
        self.sampler = "cvae"
        self.sampler_seed = 0
        self.marginal_path = Path(__file__).parent.parent / "data/sp_training_2026-02-15/sampled_vars.15.parquet"
        # Same-state evaluation: at every cycle of the closed loop also evaluate these samplers with num_samples
        # parameters (and "grid": the shadow_grid^3 FOP grid) without acting on them; one row per cycle and sampler
        # (scenario, time_step, sampler, n, n_feasible, best_cost) in shadow_log
        self.shadow_samplers: List[str] = []
        self.shadow_grid = 10
        self.shadow_log = ""


class SparsePlanner_CPP(SparsePlannerFOP):
    """Sparse Planner with the C++ FOP: the CVAE samples (d, s_d, t) are evaluated by the C++ FOP pipeline
    (best_traj_generation, the same checks and cost as the Python Sparse_FOP), the fallback is the C++ FOP grid
    N_W x N_S x N_T. The BEV images and their preprocessing are inputs (time_image_generation, not planning time);
    the CVAE inference from the H2D copy of the condition to the samples on the host is planning time."""

    def __init__(self, planner_settings: SparsePlannerSettings_CPP, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None, number_threads: int = 1, runtime_measurement: bool = False):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        # torch on the CPU only does the image preprocessing: with its OpenMP pool the idle workers spin after it and
        # take the cores of the C++ threads (16 cores, N = 16: 2.85 ms instead of 1.16 ms per cycle)
        torch.set_num_threads(1)
        self.cpp_wrapper =FOP_CPP_Wrapper(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices,
                                           number_threads, runtime_measurement)
        self.rng = np.random.default_rng(self.settings.sampler_seed)
        self.shadow_rng = np.random.default_rng(self.settings.sampler_seed + 1)
        targets = pd.read_parquet(self.settings.marginal_path, columns=["d", "v", "t"]).to_numpy(np.float64)
        self.marginal_targets = targets  # (d, s_d, t)
        self.target_mean = targets.mean(axis=0)
        self.shadow_wrapper = None
        if self.settings.shadow_samplers:
            self.shadow_wrapper = FOP_CPP_Wrapper(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices,
                                                  number_threads, True)
        self.cvae_graph = None if self.settings.sampler != "cvae" or self.settings.cvae_backend == "eager" \
            else self.build_graph_sampler()

    def build_graph_sampler(self):
        """CUDA graph / TensorRT sampler of the learned model for cvae_backend graph_* / trt_fp16 (overridden by
        GaussianPlanner_CPP)."""
        if self.settings.cvae_version != "paper" or self.settings.device.type != "cuda":
            raise ValueError("cvae_backend graph_* / trt_fp16 needs SP_CVAE paper and a CUDA device")
        if self.settings.cvae_backend == "trt_fp16":
            return CVAETRTSampler(self.cvae_efficient_model, self.settings.num_samples, str(self.settings.trt_cache_dir))
        return CVAEGraphSampler(self.cvae_efficient_model, self.settings.num_samples,
                                half=self.settings.cvae_backend == "graph_fp16")

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        cubic_spline, ref = super().generate_frenet_frame(centerline_pts)
        self.cpp_wrapper.generate_frenet_frame(centerline_pts)
        if self.shadow_wrapper is not None:
            self.shadow_wrapper.generate_frenet_frame(centerline_pts)
        return cubic_spline, ref

    def sample_parameters(self, sampler: str, n: int, max_target_speed: float, rng: np.random.Generator) -> np.ndarray:
        """(n, 3) array of (d, s_d, t) of the samplers without the CVAE."""
        st = self.settings
        if sampler == "marginal":
            return self.marginal_targets[rng.integers(0, len(self.marginal_targets), n)]
        if sampler == "mean":
            return np.repeat(self.target_mean[None, :], n, axis=0)
        half_width = (st.max_road_width - self.vehicle.w) / 2.0
        if sampler == "uniform":
            return rng.uniform([-half_width, st.lowest_speed, st.min_t], [half_width, max_target_speed, st.max_t], (n, 3))
        if sampler == "grid":
            k = st.shadow_grid
            return np.array(list(product(np.linspace(-half_width, half_width, k), np.linspace(st.lowest_speed, max_target_speed, k),
                                         np.linspace(st.min_t, st.max_t, k))))
        raise ValueError(f"unknown sampler {sampler}")

    def cvae_parameters(self, time_step_now: int, current_state: InitialState) -> np.ndarray:
        images_last_3_frame = self.condition_images(time_step_now, current_state)
        time_start = time.time()  # image preprocessing: input, not planning time
        imgs_tensor = self.cvae_efficient_model.condition_tensor(images_last_3_frame)
        if self.cvae_graph is not None:
            self.cvae_graph.set_condition(imgs_tensor)
        self.time_image_generation += time.time() - time_start

        if self.cvae_graph is not None:
            cvae_samples = self.cvae_graph.sample()
        else:
            cvae_samples = self.cvae_efficient_model.generate_from_tensor(imgs_tensor, self.settings.num_samples)
        return np.ascontiguousarray(cvae_samples[:, [1, 2, 0]])  # [t, d, v] -> (d, s_d, t)

    def shadow_evaluation(self, frenet_state: FrenetState, max_target_speed: float, time_step_now: int, main_evals):
        rows = []
        for sampler in self.settings.shadow_samplers:
            if sampler == self.settings.sampler:
                evals = main_evals  # the samples the planner acted on
            else:
                if sampler == "cvae":
                    raise ValueError("shadow sampler cvae needs SP_SAMPLER cvae")
                samples = self.sample_parameters(sampler, self.settings.num_samples, max_target_speed, self.shadow_rng)
                self.shadow_wrapper.best_traj_generation(frenet_state, samples, max_target_speed, time_step_now)
                evals = self.shadow_wrapper.cpp_planner.get_last_evals()
            costs = [c for feasible, _, c in evals if feasible]
            rows.append(f"{self.settings.scenario_file},{time_step_now},{sampler},{len(evals)},{len(costs)},"
                        f"{min(costs) if costs else float('inf')!r}\n")
        with open(self.settings.shadow_log, "a") as f:
            f.writelines(rows)

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, current_state: InitialState = None) -> FrenetTrajectory:
        """Plan using CVAE sampled trajectories."""
        self.num_FOP_intervention = 0
        self.settings.highest_speed = max_target_speed
        self.time_image_generation = 0.0
        if self.settings.sampler == "cvae":
            samples = self.cvae_parameters(time_step_now, current_state)
        else:
            samples = self.sample_parameters(self.settings.sampler, self.settings.num_samples, max_target_speed, self.rng)

        self.best_traj = self.cpp_wrapper.best_traj_generation(frenet_state, samples, max_target_speed, time_step_now)
        self.stats = self.cpp_wrapper.get_stats()
        if self.settings.shadow_samplers:
            self.shadow_evaluation(frenet_state, max_target_speed, time_step_now, self.cpp_wrapper.cpp_planner.get_last_evals())
        if self.best_traj is None and self.settings.fallback:
            self.num_FOP_intervention = 1
            self.best_traj = self.cpp_wrapper.plan(frenet_state, max_target_speed, obstacles, time_step_now, current_state)
            self.stats += self.cpp_wrapper.get_stats()  # the CVAE samples and the fallback grid
        return self.best_traj

    def get_stats(self) -> Stats:
        return self.stats
