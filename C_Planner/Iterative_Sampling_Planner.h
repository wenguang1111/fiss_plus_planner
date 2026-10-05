#ifndef ITERATIVE_SAMPLING_PLANNER_H
#define ITERATIVE_SAMPLING_PLANNER_H

#include "Frenet_Planner.h"
#include "common/sampling/gaussian_proposal.h"

// Settings shared by the iterative sampling planners. Trajectories evaluated per cycle:
// num_iterations * population.
struct IterativeSamplingSettings {
    int num_iterations = 4;   // proposal updates per planning cycle
    int population = 250;     // samples per iteration
    double init_std = 0.3;    // initial standard deviation in unit coordinates
    unsigned seed = 0;        // the generator of a cycle is seeded with seed + time_step_now
};

// FOP with an adaptive sampler: instead of a fixed grid, z = (d, s_d, t) is drawn from a
// Gaussian proposal that is refined over a few iterations. Every sample goes through the
// shared FOP backend (same generator, checks and objective as FOP). The proposal starts
// at the previous cycle's solution (warm start). Subclasses define the proposal update.
class Iterative_Sampling_Planner : public Frenet_Planner {
public:
    using Frenet_Planner::Frenet_Planner;
    ~Iterative_Sampling_Planner() override = default;

    // Same interface as Frenet_Planner::plan; returns the best feasible trajectory found
    // in all iterations (is_generated == false if there is none).
    FrenetTrajectory plan(const FrenetState& frenet_state,
                          double max_target_speed,
                          int time_step_now = 0,
                          int num_threads = 1,
                          double desired_speed = -1.0);

protected:
    virtual const IterativeSamplingSettings& iteration_settings() const = 0;

    // Refines the proposal from the evaluated population (unit samples and their results).
    virtual void update_proposal(GaussianProposal& proposal,
                                 const std::vector<Unit3>& samples,
                                 const std::vector<EvaluationResult>& results) const = 0;

private:
    SamplingParam prev_best_;
    bool has_prev_best_ = false;
};

#endif // ITERATIVE_SAMPLING_PLANNER_H
