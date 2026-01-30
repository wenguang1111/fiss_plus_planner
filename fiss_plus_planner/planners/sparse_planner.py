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

from fiss_plus_planner.planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer
# from fiss_plus_planner.planners.sparse_planning.model import CVAE

from CVAE_efficient_sampling.CVAE import CVAE_Efficient

# @dataclass
# class ParameterSample:
#     """Represents a trajectory sample with lateral, longitudinal, and temporal parameters."""
#     d: float       # lateral position
#     s_d: float     # longitudinal velocity
#     t: float       # time horizon


class SparsePlannerSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        self.w_heuristic = 10.0
        self.vis_all_candidates = False
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.z_dim: int = 32 
        self.num_samples: int = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c_dim: int = 6 + 64
        current_dir = Path(__file__).parent.parent.parent
        self.cvae_model_path = current_dir / Path("CVAE_efficient_sampling/weights/hcvae_opt_batch_64_epochs_20_zdim_32_sigmoid_0.1_stall_end.pth")
        
class SparsePlanner(FrenetOptimalPlanner):
    # -------may check the code from FissPlanner--------- #
    def __init__(self, planner_settings: SparsePlannerSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        # self.cvae_model = CVAE(X_dim=3, 
        #                            c_dim=planner_settings.c_dim, 
        #                            z_dim=planner_settings.z_dim,
        #                            h_Q_dim=512,
        #                            h_P_dim=512)
        # self.cvae_model.load_state_dict(torch.load(
        #     planner_settings.cvae_model_path, map_location=torch.device(self.settings.device)))
        # self.cvae_model = self.cvae_model.to(self.settings.device)
        # self.cvae_model.eval()
        
        self.scenario_drawer = ScenarioDrawer(
            self.settings.scenario_file,
            self.settings.scenario_dir,
            obstacles_array=self.obstacles_array,
            obstacles_num_vertices=self.obstacles_num_vertices,
        )

        self.image_history: List[Tuple[int, Image.Image]] = []
        self.cvae_efficient_model = CVAE_Efficient(device=self.settings.device, model_path=str(self.settings.cvae_model_path))
        
    
    # def get_samples(self, current_state: InitialState = None, current_time_step: int = 0):
    #     """Get CVAE samples conditioned on the current state and scenario image."""
        
    #     condition = np.array([
    #         current_state.position[0],
    #         current_state.position[1],
    #         current_state.orientation,
    #         current_state.velocity,
    #         current_state.acceleration,
    #         current_state.yaw_rate
    #     ], dtype=np.float32)

    #     img = self.scenario_drawer.generate_image_at_time_step(
    #         current_time_step,
    #         current_state,
    #         self.settings.highest_speed,
    #     )

    #     with torch.inference_mode():
    #         # time_s = time.time()
    #         z = torch.randn(self.settings.num_samples, self.settings.z_dim)
    #         z = z.to(torch.float32).to(self.settings.device)
    #         c = torch.tensor(condition, dtype=torch.float32).repeat(self.settings.num_samples, 1).to(self.settings.device)
    #         img_features = self.cvae_model.cnn_extractor(img)
    #         img_features = img_features.repeat(self.settings.num_samples, 1)
    #         samples = self.cvae_model.decode(z, c, img_features).cpu().numpy()
            
    #     return samples

    def record_generated_sampling_parameters(self, samples: List[List[float]], time_step_now: int):
        """Record generated sampling parameters to a file."""
        output_dir = Path("output/sampling_parameters")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.settings.scenario_file}.csv"
        with open(output_file, "a") as f:
            for sample in samples:
                t, d, s_d = sample
                f.write(f"{time_step_now}, {d}, {s_d}, {t}\n")
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, current_state: InitialState = None) -> FrenetTrajectory:
        """Plan using CVAE sampled trajectories."""
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed

        # img = self.scenario_drawer.generate_image_at_time_step(
        #     time_step_now,
        #     current_state,
        #     self.settings.highest_speed,
        # )
        img = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now,
            current_state
        )
        self.image_history.append((time_step_now, img))


        output_dir = Path("output/generated_images") / Path(self.settings.scenario_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / f"{time_step_now}.png")

        images_last_3_frame: List[Image.Image] = []
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
        
        # cvae_samples = self.get_samples(current_state=current_state, current_time_step=time_step_now)

        # Output is t, d, s_d -> reorder to  d, s_d, t.
        cvae_samples = self.cvae_efficient_model.generate_samples(images_last_3_frame, self.settings.num_samples)
        cvae_samples = [[sample[1],sample[2],sample[0]] for sample in cvae_samples]
        

        self.record_generated_sampling_parameters(cvae_samples, time_step_now)

        fplist = self.calc_frenet_paths(frenet_state, cvae_samples)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        self.stats.num_trajs_validated = len(fplist)
        self.stats.num_collison_checks = len(fplist)
        fplist = self.check_constraints(fplist)
        # print(len(fplist), "trajectories passed constraint check")
        # fplist = self.check_collisions(fplist, obstacles, time_step_now)
        fplist = self.check_collision_multithread(fplist, time_step_now)
        # print(len(fplist), "trajectories passed collision check")

        # find minimum cost path
        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp

        return self.best_traj
