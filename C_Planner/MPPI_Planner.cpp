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
                                   const std::vector<Unit3>& units,
                                   const std::vector<SampleEval>& evals) const {
    // Score: J over the feasible samples (infeasible ones get weight 0); V over all samples if none is
    // feasible. Normalized to [0, 1] per iteration, weights exp(-score / temperature).
    const bool any_feasible = std::any_of(evals.begin(), evals.end(), [](const SampleEval& e) { return e.feasible; });
    auto used = [&](size_t i) { return !any_feasible || evals[i].feasible; };
    auto score = [&](size_t i) { return any_feasible ? evals[i].cost : evals[i].violation; };
    double lo = std::numeric_limits<double>::infinity(), hi = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < evals.size(); ++i) {
        if (!used(i)) continue;
        lo = std::min(lo, score(i));
        hi = std::max(hi, score(i));
    }
    const double range = hi - lo;
    std::vector<double> weights(evals.size(), 0.0);
    for (size_t i = 0; i < evals.size(); ++i) {
        if (!used(i)) continue;
        const double normalized = range > 1e-12 ? (score(i) - lo) / range : 0.0;
        weights[i] = std::exp(-normalized / mppi_settings.temperature);
    }
    Unit3 weighted_mean, weighted_std;
    GaussianProposal::fit(units, weights, weighted_mean, weighted_std);

    const double eta = mppi_settings.learning_rate;
    for (int dim = 0; dim < 3; ++dim) {
        proposal.mean[dim] = (1.0 - eta) * proposal.mean[dim] + eta * weighted_mean[dim];
        const double var = (1.0 - eta) * proposal.stddev[dim] * proposal.stddev[dim] +
                           eta * weighted_std[dim] * weighted_std[dim];
        proposal.stddev[dim] = std::sqrt(var);
    }
}
