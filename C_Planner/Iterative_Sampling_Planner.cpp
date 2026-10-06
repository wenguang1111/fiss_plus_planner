#include "Iterative_Sampling_Planner.h"
#include <chrono>
#include <limits>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

FrenetTrajectory Iterative_Sampling_Planner::plan(const FrenetState& frenet_state,
                                                  double max_target_speed,
                                                  int time_step_now,
                                                  int num_threads,
                                                  double desired_speed) {
    const auto t_start = Clock::now();
    settings.highest_speed = max_target_speed;
    last_stats = PlanStats();
    last_fplist.clear();
    best_traj = FrenetTrajectory();
    best_traj.cost_final = std::numeric_limits<double>::infinity();

    const IterativeSamplingSettings& cfg = iteration_settings();
    SearchSpace space;
    if (cfg.num_iterations > 0 && cfg.population > 0 && search_space(frenet_state.s, space)) {
        // Iterative samplers evaluate complete trajectories, so J_D is part of the objective
        PlanningContext context = make_context(
            frenet_state, resolve_desired_speed(max_target_speed, desired_speed), time_step_now, true);
        run_iterations(frenet_state, space, context, num_threads);
        if (!best_traj.is_generated && fall_back_without_clearance(context)) {
            run_iterations(frenet_state, space, context, num_threads);
        }
    }

    has_prev_best_ = best_traj.is_generated;
    if (has_prev_best_) {
        prev_best_ = best_traj.sampling_param;
    }
    last_stats.timing.total_ms = ms_since(t_start);
    return best_traj;
}

void Iterative_Sampling_Planner::run_iterations(const FrenetState& frenet_state, const SearchSpace& space,
                                                const PlanningContext& context, int num_threads) {
    const IterativeSamplingSettings& cfg = iteration_settings();
    std::mt19937_64 rng(cfg.seed + static_cast<unsigned>(context.time_step_now));

    GaussianProposal proposal;
    proposal.stddev.fill(cfg.init_std);
    if (has_prev_best_) {
        proposal.mean = space.to_unit(prev_best_);
    }

    for (int iter = 0; iter < cfg.num_iterations; ++iter) {
        ++last_stats.num_search_iterations;
        auto t0 = Clock::now();
        const std::vector<Unit3> units = proposal.sample(cfg.population, rng, iter == 0);
        std::vector<SamplingParam> samples;
        samples.reserve(units.size());
        for (const Unit3& u : units) {
            samples.push_back(space.from_unit(u));
        }
        last_stats.timing.sampling_ms += ms_since(t0);

        BatchResult batch = evaluate_batch(frenet_state, samples, context, num_threads);
        for (const FrenetTrajectory& traj : batch.feasible) {
            if (traj.cost_final < best_traj.cost_final) {
                best_traj = traj;
            }
        }
        last_fplist.insert(last_fplist.end(), std::make_move_iterator(batch.feasible.begin()),
                           std::make_move_iterator(batch.feasible.end()));

        if (iter + 1 < cfg.num_iterations) {
            t0 = Clock::now();
            update_proposal(proposal, units, batch.results);
            last_stats.timing.sampling_ms += ms_since(t0);
        }
    }
}
