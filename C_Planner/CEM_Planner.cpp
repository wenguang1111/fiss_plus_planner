#include "CEM_Planner.h"
#include <algorithm>
#include <cmath>
#include <numeric>

CEM_Planner::CEM_Planner(const SettingParameters& settings_param,
                         const CEMSettings& cem_param,
                         const VehicleParams& vehicle_param,
                         const double* obs_array,
                         const int* num_verts,
                         int n_time_steps,
                         int n_obstacles,
                         int max_verts)
    : Iterative_Sampling_Planner(settings_param, vehicle_param, obs_array, num_verts,
                                 n_time_steps, n_obstacles, max_verts),
      cem_settings(cem_param) {}

void CEM_Planner::update_proposal(GaussianProposal& proposal,
                                  const std::vector<Unit3>& units,
                                  const std::vector<SampleEval>& evals) const {
    // Elite set: the best round(elite_fraction * n) samples ranked by (feasible first, then cost for
    // feasible / V for infeasible), stable as np.lexsort; std::nearbyint rounds half to even as Python's round
    const int n = static_cast<int>(units.size());
    const int n_elite = std::clamp(static_cast<int>(std::nearbyint(cem_settings.elite_fraction * n)), 1, n);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    auto key = [&evals](int i) { return evals[i].feasible ? evals[i].cost : evals[i].violation; };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (evals[a].feasible != evals[b].feasible) return evals[a].feasible;
        return key(a) < key(b);
    });

    std::vector<Unit3> elites;
    elites.reserve(n_elite);
    for (int i = 0; i < n_elite; ++i) {
        elites.push_back(units[order[i]]);
    }
    Unit3 elite_mean, elite_std;
    GaussianProposal::fit(elites, std::vector<double>(n_elite, 1.0), elite_mean, elite_std);

    const double alpha = cem_settings.smoothing;
    for (int dim = 0; dim < 3; ++dim) {
        proposal.mean[dim] = alpha * elite_mean[dim] + (1.0 - alpha) * proposal.mean[dim];
        proposal.stddev[dim] = alpha * elite_std[dim] + (1.0 - alpha) * proposal.stddev[dim];
    }
}
