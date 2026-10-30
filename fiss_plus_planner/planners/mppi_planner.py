import numpy as np

from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.iterative_sampling_planner import IterativeSamplingPlanner, IterativeSamplingSettings


class MPPIPlannerSettings(IterativeSamplingSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, num_iterations: int = 8):
        super().__init__(num_width, num_speed, num_t, num_iterations)
        self.temperature = 0.1    # gamma, applied to the per-iteration normalized score
        self.learning_rate = 0.7  # eta of the mean / covariance update


class MPPIPlanner(IterativeSamplingPlanner):
    """MPPI-style update over the FOP sampling parameters: the whole population is weighted by
    exp(-score / gamma) (MPPI, Williams et al. 2017) and the proposal moves towards the weighted
    mean and covariance with learning rate eta, the update MMD-OPT (Sharma & Singh 2025, Eq. 30)
    uses for Frenet behavioural inputs (there over its elite set only). The score is J over the
    feasible samples (infeasible ones get weight 0); if no sample is feasible it is V over all
    samples. Scores are normalized to [0, 1] per iteration because J differs by orders of
    magnitude between scenarios."""

    def __init__(self, planner_settings: MPPIPlannerSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)

    def update_proposal(self, mean, std, units, feasible, violation, cost):
        used = feasible if feasible.any() else np.ones(len(units), dtype=bool)
        score = cost[used] if feasible.any() else violation
        span = score.max() - score.min()
        normalized = (score - score.min()) / span if span > 1e-12 else np.zeros_like(score)
        weights = np.zeros(len(units))
        weights[used] = np.exp(-normalized / self.settings.temperature)
        weighted_mean, weighted_std = self.fit(units, weights)

        eta = self.settings.learning_rate
        return ((1.0 - eta) * mean + eta * weighted_mean,
                np.sqrt((1.0 - eta) * std ** 2 + eta * weighted_std ** 2))
