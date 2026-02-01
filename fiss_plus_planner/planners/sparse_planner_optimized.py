import copy
from collections import deque
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
from fiss_plus_planner.planners.sparse_planning.model import CVAE
from CVAE_efficient_sampling.CVAE import CVAE_Efficient

class SparsePlannerOptimizedSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        # self.w_heuristic = 10.0
        self.vis_all_candidates = False
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.num_samples: int = 5
        self.max_refine_iters = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_dir = Path(__file__).parent.parent.parent
        self.cvae_model_path = current_dir / Path("CVAE_efficient_sampling/weights/hcvae_opt_batch_64_epochs_20_zdim_32_sigmoid_0.1_stall_end.pth")
        self.decaying_factor = 0.5

class SparsePlannerOptimized(FrenetOptimalPlanner):
    # -------may check the code from FissPlanner--------- #
    def __init__(self, planner_settings: SparsePlannerOptimizedSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        
        self.scenario_drawer = ScenarioDrawer(
            self.settings.scenario_file,
            self.settings.scenario_dir,
            obstacles_array=self.obstacles_array,
            obstacles_num_vertices=self.obstacles_num_vertices,
        )
        self.sampling_res = np.empty(3)
        self.sampling_min = np.empty(3)
        self.sampling_max = np.empty(3)
        sampling_width = self.settings.max_road_width - self.vehicle.w + 0.3
        _, self.sampling_res[0] = np.linspace(-sampling_width/2, sampling_width/2, self.settings.num_width, retstep=True)
        _, self.sampling_res[1] = np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed, retstep=True)
        _, self.sampling_res[2] = np.linspace(self.settings.min_t, self.settings.max_t, self.settings.num_t, retstep=True)
        self.sampling_min[0] = -sampling_width/2
        self.sampling_max[0] = sampling_width/2
        self.sampling_min[1] = self.settings.lowest_speed
        self.sampling_max[1] = self.settings.highest_speed
        self.sampling_min[2] = self.settings.min_t
        self.sampling_max[2] = self.settings.max_t

        self.refined_trajs = PriorityQueue()
        self.image_history = deque(maxlen=3)
        self.cvae_efficient_model = CVAE_Efficient(device=self.settings.device, model_path=str(self.settings.cvae_model_path))

        self.start_state = None
        self.time_inference = 0.0
        self.time_image_generation = 0.0
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, current_state: InitialState = None) -> FrenetTrajectory:
        """Plan using CVAE sampled trajectories."""
        # reset stats
        self.stats = Stats()
        self.refined_trajs = PriorityQueue()
        self.settings.highest_speed = max_target_speed
        images_last_3_frame: List[Image.Image] = []
        self.start_state = frenet_state
        
        time_image_start = time.time()
        img = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now,
            current_state
        )
        time_image_end = time.time()
        self.time_image_generation += (time_image_end - time_image_start)
        self.image_history.append((time_step_now, img))

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
        time_inference_start = time.time()
        cvae_samples = self.cvae_efficient_model.generate_samples(images_last_3_frame, self.settings.num_samples)
        time_inference_end = time.time()
        self.time_inference += (time_inference_end - time_inference_start)
        cvae_samples = [[sample[1],sample[2],sample[0]] for sample in cvae_samples]


        # cvae_sampled_end_state = FrenetState(t=cvae_samples[2], s=0.0, s_d=cvae_samples[1], s_dd=0.0, s_ddd=0.0, d=cvae_samples[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
        # cvae_traj = self.generate_trajectory_by_end_state(cvae_sampled_end_state)
        sampled_best = self.getBestFromCVAE(frenet_state, cvae_samples, time_step_now)
        
        refined_traj = self.refine_solution(sampled_best, obstacles, time_step_now)
        if refined_traj is not None:
            self.best_traj = refined_traj

        return self.best_traj
    
    def getBestFromCVAE(self, frenet_state: FrenetState, cvae_samples: list, time_step_now:int) -> FrenetTrajectory:
        fplist = self.calc_frenet_paths(frenet_state, cvae_samples)
        fplist = self.calc_global_paths(fplist)
        for fp, sample in zip(fplist, cvae_samples):
            fp.end_state = FrenetState(t=sample[2], s=0.0, s_d=sample[1], s_dd=0.0, s_ddd=0.0, d=sample[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
        
        fplist = self.check_constraints(fplist)
        # fplist = self.check_collisions(fplist, obstacles, time_step_now)
        fplist = self.check_collision_multithread(fplist, time_step_now)

        # find minimum cost path
        min_cost = float("inf")
        best_trajectory = FrenetTrajectory()
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                best_trajectory = fp
        return best_trajectory

    def refine_solution(self, traj: FrenetTrajectory, obstacles: list, time_step_now: int) -> FrenetTrajectory:
        #FIXME: The FISS_PLUS does not have the copy()
        resolutions = self.sampling_res.copy()
        alpha = self.settings.decaying_factor
        
        J_new = traj.cost_final
        x = np.array([traj.end_state.d, traj.end_state.s_d, traj.end_state.t])
        
        for i in range(self.settings.max_refine_iters):
            _, J_new, x, resolutions = self.gradient_decent(J_new, x, resolutions, alpha)

        
        while not self.refined_trajs.empty():
            candidate = self.refined_trajs.get()
            if candidate.cost_final > traj.cost_final:
                break

            self.stats.num_trajs_validated += 1
            candidate = self.calc_global_paths([candidate])

            passed_candidate = self.check_constraints(candidate)
            
            if passed_candidate:
                # safe_candidate = self.check_collisions(passed_candidate, obstacles, time_step_now)
                safe_candidate = self.check_collision_multithread(passed_candidate, time_step_now)
                self.stats.num_collison_checks += 1
                if safe_candidate:
                    return safe_candidate[0]
                else:
                    continue
            else:
                continue
        return None
    
    def generate_trajectory_by_end_state(self, end_state: FrenetState) -> FrenetTrajectory:
        # Create the trajectory
        traj = FrenetTrajectory()
        
        # Generate the end state by given specs
        traj.end_state = end_state
        
        self.stats.num_trajs_generated += 1
        traj.is_generated = True
        traj.t = [t for t in np.arange(0.0, end_state.t, self.settings.tick_t)]
        
        # Generate lateral quintic polynomial
        lat_qp = QuinticPolynomial(self.start_state.d, self.start_state.d_d, self.start_state.d_dd, end_state.d, end_state.d_d, end_state.d_dd, end_state.t)
        traj.d = [lat_qp.calc_point(t) for t in traj.t]
        traj.d_d = [lat_qp.calc_first_derivative(t) for t in traj.t]
        traj.d_dd = [lat_qp.calc_second_derivative(t) for t in traj.t]
        traj.d_ddd = [lat_qp.calc_third_derivative(t) for t in traj.t]
        
        # Generate longitudinal quartic polynomial
        lon_qp = QuarticPolynomial(self.start_state.s, self.start_state.s_d, self.start_state.s_dd, end_state.s_d, end_state.s_dd, end_state.t)
        traj.s = [lon_qp.calc_point(t) for t in traj.t]
        traj.s_d = [lon_qp.calc_first_derivative(t) for t in traj.t]
        traj.s_dd = [lon_qp.calc_second_derivative(t) for t in traj.t]
        traj.s_ddd = [lon_qp.calc_third_derivative(t) for t in traj.t]

        # Compute the final cost
        traj.cost_final = self.cost_function.cost_total(traj, self.settings.highest_speed)
        
        # Add this trajectory to the candidate queue
        self.refined_trajs.put(traj)
            
        return traj
    
    def gradient_decent(self, J: float, x: np.ndarray, resolutions: np.ndarray, decaying_factor: float) -> tuple:
        # Compute the initial gradient at the best trajectory
        d_J = np.empty(3)
        d_x = np.empty(3)
        
        for dim in range(3):
            
            # left neighbor
            x_l = copy.deepcopy(x)
            x_l[dim] -= resolutions[dim]
            x_l = np.clip(x_l, self.sampling_min, self.sampling_max)
            end_state_l = FrenetState(t=x_l[2], s=0.0, s_d=x_l[1], s_dd=0.0, s_ddd=0.0, d=x_l[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            J_l = self.get_cost_from_traj(self.generate_trajectory_by_end_state(end_state_l))
            
            # right neighbor
            x_r = copy.deepcopy(x)
            x_r[dim] += resolutions[dim]
            x_r = np.clip(x_r, self.sampling_min, self.sampling_max)
            end_state_r = FrenetState(t=x_r[2], s=0.0, s_d=x_r[1], s_dd=0.0, s_ddd=0.0, d=x_r[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            J_r = self.get_cost_from_traj(self.generate_trajectory_by_end_state(end_state_r))
            
            # gradient
            d_J[dim] = J_r - J_l
            d_x[dim] = x_r[dim] - x_l[dim]
            
        grad = d_J/d_x
        resolutions *= decaying_factor
        x_new = x - resolutions * grad/np.linalg.norm(grad)
        
        # Validate the location of the next candidate is within the sampling region
        x_new_clipped = np.clip(x_new, self.sampling_min, self.sampling_max)

        # Generate the next candiate
        end_state_new = FrenetState(t=x_new_clipped[2], s=0.0, s_d=x_new_clipped[1], s_dd=0.0, s_ddd=0.0, d=x_new_clipped[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
        J_new = self.get_cost_from_traj(self.generate_trajectory_by_end_state(end_state_new))

        return True, J_new, x_new_clipped, resolutions
    
    def get_cost_from_traj(self, traj: FrenetTrajectory) -> float:
        return traj.cost_final
