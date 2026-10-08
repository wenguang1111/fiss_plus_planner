#include "Iterative_Sampling_Planner.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>

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
#ifdef ENABLE_DATA_COLLECTION
    last_record = CycleRecord();
#endif
    if (cfg.num_iterations > 0 && cfg.population > 0 && search_space(frenet_state.s, space)) {
#ifdef ENABLE_DATA_COLLECTION
        last_record.space = space;
#endif
        // Iterative samplers evaluate complete trajectories, so J_D is part of the objective
        PlanningContext context = make_context(
            frenet_state, resolve_desired_speed(max_target_speed, desired_speed), time_step_now, true);
        run_iterations(frenet_state, space, context, num_threads, 0);
        if (!best_traj.is_generated && fall_back_without_clearance(context)) {
            run_iterations(frenet_state, space, context, num_threads, 1);
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
                                                const PlanningContext& context, int num_threads, int pass) {
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

#ifdef ENABLE_DATA_COLLECTION
        BatchResult batch = evaluate_batch(frenet_state, samples, context, num_threads,
                                           cfg.record_candidates ? EvalMode::kFullViolation : EvalMode::kEarlyExit);
        if (cfg.record_candidates) {
            record_iteration(pass, iter, proposal, samples, batch.results);
        }
#else
        (void)pass;
        BatchResult batch = evaluate_batch(frenet_state, samples, context, num_threads);
#endif
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

#ifdef ENABLE_DATA_COLLECTION
void Iterative_Sampling_Planner::record_iteration(int pass, int iteration, const GaussianProposal& proposal,
                                                  const std::vector<SamplingParam>& samples,
                                                  std::vector<EvaluationResult>& results) {
    const int n = static_cast<int>(samples.size());
    const std::vector<EvaluationResult> full = results;
    for (EvaluationResult& r : results) {
        r = early_exit_view(r);
    }
    // Rank as in the proposal update (the CEM elites are the first ranks)
    std::vector<int> order(n), rank(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&results](int a, int b) { return ranks_before(results[a], results[b]); });
    for (int i = 0; i < n; ++i) {
        rank[order[i]] = i;
    }
    for (int i = 0; i < n; ++i) {
        last_record.candidates.push_back({pass, iteration, i, rank[i], samples[i], full[i]});
    }
    last_record.proposals.push_back({pass, iteration, proposal});
}
#endif
