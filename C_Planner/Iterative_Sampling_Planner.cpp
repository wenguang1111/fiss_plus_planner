#include "Iterative_Sampling_Planner.h"
#include <algorithm>
#include <limits>

FrenetTrajectory Iterative_Sampling_Planner::plan(const FrenetState& frenet_state,
                                                  double max_target_speed,
                                                  int time_step_now,
                                                  int num_threads) {
    settings.highest_speed = max_target_speed;
    const IterativeSamplingSettings& cfg = iteration_settings();

    // Sampling ranges of FOP's grid (get_samples)
    const double sampling_width = settings.max_road_width - vehicle_params.w;
    const Unit3 lo{-sampling_width / 2.0, settings.lowest_speed, settings.min_t};
    const Unit3 hi{sampling_width / 2.0, settings.highest_speed, settings.max_t};
    auto to_unit = [&](const SamplingParam& p) {
        const Unit3 v{p.d, p.s_d, p.t};
        Unit3 u;
        for (int dim = 0; dim < 3; ++dim) {
            const double range = hi[dim] - lo[dim];
            u[dim] = range > 0.0 ? std::clamp((v[dim] - lo[dim]) / range, 0.0, 1.0) : 0.5;
        }
        return u;
    };

    GaussianProposal proposal;
    proposal.stddev.fill(cfg.init_std);
    if (has_prev_best_) {
        proposal.mean = to_unit(prev_best_);
    }
    std::mt19937_64 rng(cfg.seed + static_cast<unsigned>(time_step_now));

    FrenetTrajectory best;
    best.cost_final = std::numeric_limits<double>::infinity();
    std::vector<FrenetTrajectory> all_feasible;
    PlanStats total;
    for (int iter = 0; iter < cfg.num_iterations; ++iter) {
        // The first sample of the first iteration is the proposal mean (the warm start)
        std::vector<Unit3> units = proposal.sample(cfg.population, rng, iter == 0);
        const size_t needed = static_cast<size_t>(cfg.num_iterations) * cfg.population * 3;
        if (external_normals_.size() == needed) {
            for (int i = (iter == 0 ? 1 : 0); i < cfg.population; ++i) {
                for (int dim = 0; dim < 3; ++dim) {
                    const double z = external_normals_[(static_cast<size_t>(iter) * cfg.population + i) * 3 + dim];
                    units[i][dim] = std::clamp(proposal.mean[dim] + proposal.stddev[dim] * z, 0.0, 1.0);
                }
            }
        }
        std::vector<std::tuple<double, double, double>> samples;
        samples.reserve(units.size());
        for (const Unit3& u : units) {
            samples.emplace_back(lo[0] + u[0] * (hi[0] - lo[0]),
                                 lo[1] + u[1] * (hi[1] - lo[1]),
                                 lo[2] + u[2] * (hi[2] - lo[2]));
        }

        best_traj_generation(frenet_state, samples, max_target_speed, time_step_now, num_threads);
        total += last_stats;

        // The first of equal costs, as the Python evaluate (fp.cost_final < best.cost_final)
        for (const FrenetTrajectory& traj : last_fplist) {
            if (traj.cost_final < best.cost_final) {
                best = traj;
            }
        }
        all_feasible.insert(all_feasible.end(), last_fplist.begin(), last_fplist.end());
        if (iter + 1 < cfg.num_iterations) {
            update_proposal(proposal, units, last_evals);
        }
    }

    best_traj = best;
    last_fplist = std::move(all_feasible);
    last_stats = total;
    if (best_traj.is_generated) {  // kept on failure, as the Python planner
        prev_best_ = best_traj.sampling_param;
        has_prev_best_ = true;
    }
    return best_traj;
}
