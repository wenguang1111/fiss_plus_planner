#include "MPPI_Planner.h"
#include <algorithm>
#include <cmath>
#include <limits>

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
                                   const std::vector<Unit3>& samples,
                                   const std::vector<EvaluationResult>& results) const {
    const size_t n = samples.size();
    const bool any_feasible = std::any_of(results.begin(), results.end(),
                                          [](const EvaluationResult& r) { return r.feasible; });

    // Score of every sample that takes part in the update (NaN: weight 0)
    std::vector<double> score(n, std::numeric_limits<double>::quiet_NaN());
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (size_t i = 0; i < n; ++i) {
        if (any_feasible && !results[i].feasible) continue;
        score[i] = any_feasible ? results[i].cost.total() : results[i].violation.total();
        lo = std::min(lo, score[i]);
        hi = std::max(hi, score[i]);
    }

    const double range = hi - lo;
    std::vector<double> weights(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(score[i])) continue;
        const double normalized = range > 1e-12 ? (score[i] - lo) / range : 0.0;
        weights[i] = std::exp(-normalized / mppi_settings.temperature);
    }

    Unit3 weighted_mean, weighted_std;
    GaussianProposal::fit(samples, weights, weighted_mean, weighted_std);

    const double eta = mppi_settings.learning_rate;
    for (int dim = 0; dim < 3; ++dim) {
        proposal.mean[dim] = (1.0 - eta) * proposal.mean[dim] + eta * weighted_mean[dim];
        const double var = (1.0 - eta) * proposal.stddev[dim] * proposal.stddev[dim] +
                           eta * weighted_std[dim] * weighted_std[dim];
        proposal.stddev[dim] = std::sqrt(var);
    }
}
