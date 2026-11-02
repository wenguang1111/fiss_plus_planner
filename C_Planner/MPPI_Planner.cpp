#include "MPPI_Planner.h"
#include <algorithm>
#include <cmath>

MPPI_Planner::MPPI_Planner(const SettingParameters& settings_param,
                           const MPPISettings& mppi_param,
                           const VehicleParams& vehicle_param,
                           const double* obs_array,
                           const int* num_verts,
                           int n_time_steps,
                           int n_obstacles,
                           int max_verts)
    : Iterative_Sampling_Planner(settings_param, vehicle_param, obs_array, num_verts,
                                 n_time_steps, n_obstacles, max_verts),
      mppi_settings(mppi_param) {}

void MPPI_Planner::update_proposal(GaussianProposal& proposal,
                                   const std::vector<Unit3>& feasible,
                                   const std::vector<double>& costs) const {
    const auto [lo, hi] = std::minmax_element(costs.begin(), costs.end());
    const double range = *hi - *lo;
    std::vector<double> weights(costs.size());
    for (size_t i = 0; i < costs.size(); ++i) {
        const double normalized = range > 1e-12 ? (costs[i] - *lo) / range : 0.0;
        weights[i] = std::exp(-normalized / mppi_settings.temperature);
    }

    Unit3 weighted_mean, weighted_std;
    GaussianProposal::fit(feasible, weights, weighted_mean, weighted_std);

    const double eta = mppi_settings.learning_rate;
    for (int dim = 0; dim < 3; ++dim) {
        proposal.mean[dim] = (1.0 - eta) * proposal.mean[dim] + eta * weighted_mean[dim];
        const double var = (1.0 - eta) * proposal.stddev[dim] * proposal.stddev[dim] +
                           eta * weighted_std[dim] * weighted_std[dim];
        proposal.stddev[dim] = std::sqrt(var);
    }
}
