import copy
from queue import PriorityQueue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.state import InitialState
from typing import List
import torch
torch.manual_seed(0)
import time

from planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from planners.common.vehicle.vehicle import Vehicle
from planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from planners.sparse_planning.enviroment_drawer import EnviromentDrawer
from planners.sparse_planning.model import CVAE

@dataclass
class ParameterSample:
    """Represents a trajectory sample with lateral, longitudinal, and temporal parameters."""
    d: float       # lateral position
    s_d: float     # longitudinal velocity
    t: float       # time horizon


class SparsePlannerSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_file: str="", scenario_dir: str|Path=""):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        self.w_heuristic = 10.0
        self.vis_all_candidates = False
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.z_dim: int = 32  #latent dimension
        self.num_samples: int = 1
        self.device = "cpu"
        self.c_dim: int = 6 + 512
        current_dir = Path(__file__).parent
        self.cvae_model_path = current_dir / Path("sparse_planning/cvae_weights/cvae_model_lr_0.0001_batch_1024_epochs_5_zdim_32_cos_0.05.pth")
        
class SparsePlanner(FrenetOptimalPlanner):
    # -------may check the code from FissPlanner--------- #
    def __init__(self, planner_settings: SparsePlannerSettings, ego_vehicle: Vehicle):
        super().__init__(planner_settings, ego_vehicle)
        self.cvae_model = CVAE(X_dim=3, 
                                   c_dim=planner_settings.c_dim, 
                                   z_dim=planner_settings.z_dim,
                                   h_Q_dim=512,
                                   h_P_dim=512)
        self.cvae_model.load_state_dict(torch.load(
            planner_settings.cvae_model_path, map_location=torch.device(self.config_sampling.device)))
        self.cvae_model = self.cvae_model.to(self.config_sampling.device)
        self.cvae_model.eval()
        
    
        
    #------------------------ Main Planning Function -----------------------#
    def sampling_with_cvae(self,vehicle_state: InitialState, time_step: int) \
            -> List[ParameterSample]:
        list_samples = list()
        with torch.inference_mode():
                time_s = time.time()
                drawer = EnviromentDrawer(self.settings.scenario_file, self.settings.scenario_dir)
                img_bytes = drawer.generate_image_at_time_step(time_step, vehicle_state, None)
                #"x":-5.844833856993835,"y":-2.8847846236418326,"theta":-2.626104083888073,"velocity":8.179240172624612,"acceleration":0.433544797284917,"yaw_rate":0.11156740789998398}
                #TODO: convert InitialState to the data type of cvae_condition
                cvae_condition = [vehicle_state.position.x, vehicle_state.position.y,
                                  vehicle_state.orientation, vehicle_state.velocity,
                                  vehicle_state.acceleration, vehicle_state.yaw_rate]
                cvae_condition = np.array(cvae_condition, dtype=np.float32)
                z = torch.randn(self.settings.num_samples, self.settings.z_dim)
                z = z.to(torch.float32).to(self.settings.device)
                c = torch.tensor(cvae_condition, dtype=torch.float32).repeat(self.settings.num_samples, 1).to(self.settings.device)
                img_features = self.cvae_model.cnn_extractor(img_bytes)
                img_features = img_features.repeat(self.cvae_num_samples, 1)
                x_sampled = self.cvae_model.decode(z, c, img_features).cpu().numpy()

                self.cvae_inference_time_list.append(time.time() - time_s)
        return list_samples
         
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed

        # cvae sampling d, s_d, t
        sampling_param_list = List[ParameterSample]
        sampling_param_list = self.sampling_with_cvae(initial_state, time_step_now)
        # generate trajectorys in frenet coordination 
        fplist = List[FrenetTrajectory]
        fplist = self.generate_sampling_with_cvae(frenet_state, sampling_param_list)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        self.stats.num_trajs_validated = len(fplist)
        self.stats.num_collison_checks = len(fplist)
        fplist = self.check_constraints(fplist)
        fplist = self.check_collisions(fplist, obstacles, time_step_now)
        
        # find minimum cost path
        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp

        return self.best_traj