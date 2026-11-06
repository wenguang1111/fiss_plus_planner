#ifndef ITERATIVE_SAMPLING_PLANNER_H
#define ITERATIVE_SAMPLING_PLANNER_H

#include "Frenet_Planner.h"
#include "common/sampling/gaussian_proposal.h"

// Settings shared by the iterative sampling planners. Trajectories evaluated per cycle:
// num_iterations * population.
struct IterativeSamplingSettings {
    int num_iterations = 8;   // proposal updates per planning cycle
    int population = 15;      // samples per iteration (wrappers: N_W * N_S * N_T / num_iterations)
    double init_std = 0.3;    // initial standard deviation in unit coordinates
    unsigned seed = 0;        // the generator of a cycle is seeded with seed + time_step_now
};

// FOP with an adaptive sampler: instead of the fixed grid of get_samples(), z = (d, s_d, t) is
// drawn from a Gaussian proposal over the same sampling ranges, refined over a few iterations.
// Every population goes through FOP's best_traj_generation (same trajectory generation,
// constraint check, collision check and cost). Samples are ranked feasible before infeasible,
// feasible ones by cost, infeasible ones by the violation V of their first failed check
// (SampleEval), as the Python IterativeSamplingPlanner. The proposal starts at the previous
// cycle's solution (warm start; kept when a cycle fails). Subclasses define the proposal update.
class Iterative_Sampling_Planner : public Frenet_Planner {
public:
    using Frenet_Planner::Frenet_Planner;
    ~Iterative_Sampling_Planner() override = default;

    // Same interface as Frenet_Planner::plan; returns the cheapest feasible trajectory of all
    // iterations (is_generated == false if there is none). last_fplist / last_stats cover all
    // iterations.
    FrenetTrajectory plan(const FrenetState& frenet_state,
                          double max_target_speed,
                          int time_step_now = 0,
                          int num_threads = 1);

    // Test hook: standard normal draws (num_iterations x population x 3, row-major) used instead of the
    // generator for the following plan() calls, e.g. numpy's draws to compare with the Python planner.
    // An empty vector restores the generator.
    void set_external_normals(std::vector<double> normals) { external_normals_ = std::move(normals); }

protected:
    virtual const IterativeSamplingSettings& iteration_settings() const = 0;

    // Refines the proposal from all samples of one iteration (unit coordinates and their
    // evaluation); called after every iteration but the last.
    virtual void update_proposal(GaussianProposal& proposal,
                                 const std::vector<Unit3>& units,
                                 const std::vector<SampleEval>& evals) const = 0;

private:
    SamplingParam prev_best_;
    bool has_prev_best_ = false;
    std::vector<double> external_normals_;
};

#endif // ITERATIVE_SAMPLING_PLANNER_H
