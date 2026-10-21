import numpy as np

from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import InitialState

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.fop_plus_planner import FopPlusPlanner
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlannerSettings, Stats


class FopPlusRandomPlanner(FopPlusPlanner):
    """FOP+ candidate-generation baseline with uniform random trajectory selection."""

    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        planner_settings.min_t = 1.0
        planner_settings.max_t = 1.0
        planner_settings.tick_t = 0.1
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list[Obstacle],
             time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed

        candidates = self.calc_frenet_paths(frenet_state)
        candidates = self.calc_global_paths(candidates)
        self.stats.num_trajs_generated = len(candidates)
        if not candidates:
            return None

        selected_idx = int(np.random.randint(len(candidates)))
        selected = candidates[selected_idx]

        passed = self.check_constraints([selected])
        self.stats.num_trajs_validated += 1
        if not passed:
            return None

        safe = self.check_collision_multithread(passed, time_step_now)
        self.stats.num_collison_checks += 1
        if not safe:
            return None

        # Keep the chosen trajectory's cost populated for downstream measurement/logging,
        # without costing the entire candidate grid as standard FOP+ does.
        self.cost_function.calc_cost(
            safe, max_target_speed, self.obstacles_array,
            self.obstacles_num_vertices, time_step_now,
        )
        self.best_traj = safe[0]
        return self.best_traj
