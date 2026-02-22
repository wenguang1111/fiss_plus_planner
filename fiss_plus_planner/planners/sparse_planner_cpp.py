import copy
from queue import PriorityQueue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.state import InitialState
from typing import List, Tuple
from PIL import Image
import torch
torch.manual_seed(0)
import time

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer
from CVAE_efficient_sampling.CVAE import CVAE_Efficient
class SparsePlannerSettings_CPP(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        # self.w_heuristic = 10.0
        self.vis_all_candidates = False
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.num_samples: int = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_dir = Path(__file__).parent.parent.parent
        self.cvae_model_path = current_dir / Path("CVAE_efficient_sampling/weights/attn_cvae_gen_zdim_64_sigmoid_1.0_stall_end.pth")
        
class SparsePlanner_CPP(FrenetOptimalPlanner):
    # -------may check the code from FissPlanner--------- #
    def __init__(self, planner_settings: SparsePlannerSettings_CPP, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        
        self.scenario_drawer = ScenarioDrawer(
            self.settings.scenario_file,
            self.settings.scenario_dir,
            obstacles_array=self.obstacles_array,
            obstacles_num_vertices=self.obstacles_num_vertices,
        )

        self.image_history: List[Tuple[int, Image.Image]] = []
        self.cvae_efficient_model = CVAE_Efficient(device=self.settings.device, model_path=str(self.settings.cvae_model_path))
        self.all_trajs = []
        self.cpp_wrapper = FOP_CPP_Wrapper(
            planner_settings,
            ego_vehicle,
            obstacles_array,
            obstacles_num_vertices
        )

    def record_generated_sampling_parameters(self, samples: List[List[float]], time_step_now: int):
        """Record generated sampling parameters to a file."""
        output_dir = Path("output/sampling_parameters")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.settings.scenario_file}.csv"
        with open(output_file, "a") as f:
            for sample in samples:
                t, d, s_d = sample
                f.write(f"{time_step_now}, {d}, {s_d}, {t}\n")

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        cubic_spline, ref = super().generate_frenet_frame(centerline_pts)
        self.cpp_wrapper.generate_frenet_frame(centerline_pts)
        return cubic_spline, ref
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, current_state: InitialState = None) -> FrenetTrajectory:
        """Plan using CVAE sampled trajectories."""
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        images_last_3_frame: List[Image.Image] = []

        img = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now,
            current_state
        )
        self.image_history.append((time_step_now, img))

        # #Useful for debugging, make sure if you want to delete it
        # output_dir = Path("output/generated_images") / Path(self.settings.scenario_file)
        # output_dir.mkdir(parents=True, exist_ok=True)
        # img.save(output_dir / f"{time_step_now}.png")

        if time_step_now >= 2:
            images_last_3_frame = [
                self.image_history[-3][1],
                self.image_history[-2][1],
                self.image_history[-1][1],
            ]
        elif time_step_now == 1:
            images_last_3_frame = [
                self.image_history[-2][1],
                self.image_history[-1][1],
                self.image_history[-1][1],
            ]
        else:
            images_last_3_frame = [
                self.image_history[-1][1],
                self.image_history[-1][1],
                self.image_history[-1][1],
            ]
        
        # Output is t, d, s_d -> reorder to  d, s_d, t.
        cvae_samples = self.cvae_efficient_model.generate_samples(images_last_3_frame, self.settings.num_samples)
        cvae_samples = [[sample[1],sample[2],sample[0]] for sample in cvae_samples]

        # self.record_generated_sampling_parameters(cvae_samples, time_step_now)
        self.best_traj = self.cpp_wrapper.best_traj_generation(
            frenet_state,
            cvae_samples,
            max_target_speed,
            time_step_now
        )
        self.stats = self.cpp_wrapper.get_stats()
        return self.best_traj

    def get_stats(self) -> Stats:
        """Expose C++ planner stats for benchmark pipeline compatibility."""
        try:
            return self.cpp_wrapper.get_stats()
        except Exception:
            return self.stats
