import numpy as np

from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.iterative_sampling_planner import IterativeSamplingPlanner, IterativeSamplingSettings


class CEMPlannerSettings(IterativeSamplingSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, num_iterations: int = 8):
        super().__init__(num_width, num_speed, num_t, num_iterations)
        self.elite_fraction = 0.1  # share of the population refitted as elite set
        self.smoothing = 0.7       # weight of the elite fit vs. the previous proposal


class CEMPlanner(IterativeSamplingPlanner):
    """Cross-entropy method over the FOP sampling parameters (Rubinstein & Kroese; elite selection
    by feasibility first, then cost, as in MMD-OPT, Sharma & Singh 2025): the elite set are the best
    samples ranked by (feasible, V, J); the proposal is refitted to their mean and standard deviation
    and smoothed with the previous proposal."""

    def __init__(self, planner_settings: CEMPlannerSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)

    def update_proposal(self, mean, std, units, feasible, violation, cost):
        n = len(units)
        n_elite = int(np.clip(round(self.settings.elite_fraction * n), 1, n))
        order = np.lexsort((np.where(feasible, cost, violation), ~feasible))
        elite_mean, elite_std = self.fit(units[order[:n_elite]], np.ones(n_elite))

        alpha = self.settings.smoothing
        return alpha * elite_mean + (1.0 - alpha) * mean, alpha * elite_std + (1.0 - alpha) * std
