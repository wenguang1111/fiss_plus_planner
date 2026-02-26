import copy
import csv
import os
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
from CVAE_efficient_sampling.CVAE import CVAE_Efficient
class SparsePlannerFOPSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        # self.w_heuristic = 10.0
        self.vis_all_candidates = False
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.num_samples: int = 1
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        current_dir = Path(__file__).parent.parent.parent
        self.cvae_model_path = current_dir / Path("CVAE_efficient_sampling/weights/attn_cvae_zoom_out_zdim_64_sigmoid_1.0_stall_end.pth")
        
class SparsePlannerFOP(FrenetOptimalPlanner):
    # -------may check the code from FissPlanner--------- #
    def __init__(self, planner_settings: SparsePlannerFOPSettings, ego_vehicle: Vehicle,
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
        self.time_image_generation = 0.0
        self.num_FOP_intervention = 0

    # def record_generated_sampling_parameters(self, samples: List[List[float]], time_step_now: int):
    #     """Record generated sampling parameters to a file."""
    #     output_dir = Path("output/sampling_parameters")
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     output_file = output_dir / f"{self.settings.scenario_file}.csv"
    #     with open(output_file, "a") as f:
    #         for sample in samples:
    #             t, d, s_d = sample
    #             f.write(f"{time_step_now}, {d}, {s_d}, {t}\n")
    
    def record_generated_sampling_parameters(self, fp_list: List[FrenetTrajectory], time_step_now: int):
        """Record generated sampling parameters to a file."""
        
        output_dir = Path("data/output/sampling_parameters/sparse")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.settings.scenario_file[:-4]}.csv"
        
        if output_file.exists() and time_step_now == 0:
            os.remove(output_file)
            
        with open(output_file, "a") as f:
            writer = csv.writer(f)
            # file_exists = output_file.exists()
            if time_step_now == 0:
                writer.writerow(["time_step", "t", "d", "s_d", "cost_final"])
            for fp in fp_list:
                params = fp.sampling_param
                t, d, s_d = params.t, params.d, params.s_d
                writer.writerow([time_step_now, t, d, s_d, fp.cost_final])
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, current_state: InitialState = None) -> FrenetTrajectory:
        """Plan using CVAE sampled trajectories."""
        # reset stats
        print(f"Planning at time step {time_step_now}...")
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        images_last_3_frame: List[Image.Image] = []

        view_size_1, view_size_2 = 60, 105
        time_start = time.time()
        img = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now,
            current_state,
            view_size=view_size_1
        )
        self.image_history.append((time_step_now, img))

        # #Useful for debugging, make sure if you want to delete it
        output_dir = Path("data/output/bw_imgs") / Path(self.settings.scenario_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / f"{time_step_now}_{view_size_1}.pdf", dpi=(300, 300))
        
        img = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now,
            current_state,
            view_size=view_size_2
        )

        # #Useful for debugging, make sure if you want to delete it
        output_dir = Path("data/output/bw_imgs") / Path(self.settings.scenario_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / f"{time_step_now}_{view_size_2}.pdf", dpi=(300, 300))

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

        self.time_image_generation = time.time() - time_start
        # Output is t, d, s_d -> reorder to  d, s_d, t.
        cvae_samples = self.cvae_efficient_model.generate_samples(images_last_3_frame, self.settings.num_samples)
        cvae_samples = [[sample[1],sample[2],sample[0]] for sample in cvae_samples]

        # self.record_generated_sampling_parameters(cvae_samples, time_step_now)

        fplist = self.calc_frenet_paths(frenet_state, cvae_samples)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        self.stats.num_trajs_validated = len(fplist)
        self.stats.num_collison_checks = len(fplist)
        fplist = self.check_constraints(fplist)
        # fplist = self.check_collisions(fplist, obstacles, time_step_now)
        fplist = self.check_collision_multithread(fplist, time_step_now)
        
        self.record_generated_sampling_parameters(fplist, time_step_now)

        if(len(fplist) == 0):
            print(f"No valid trajectory found at time step {time_step_now}, invoking FOP as fallback.")
            self.num_FOP_intervention  = 1
            return super().plan(
                frenet_state,
                max_target_speed,
                obstacles,
                time_step_now,
                current_state,
            )
        else:
            print(f"Found {len(fplist)} valid trajectories at time step {time_step_now}, selecting the one with minimum cost.")
        
        # find minimum cost path
        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp

        return self.best_traj
