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
#ifdef ENABLE_DATA_COLLECTION
    bool record_candidates = false;  // keep every candidate of a cycle in last_record
#endif
};

#ifdef ENABLE_DATA_COLLECTION
// Training data of one planning cycle (Collect_Data_For_ML): every candidate evaluated with
// all checks (kFullViolation), while the planner itself ranks the early-exit view of the
// same results, so its decisions are those of a normal run.
struct CandidateRecord {
    int pass;        // 0, or 1 for the re-plan without the clearance requirement
    int iteration;
    int index;       // position in the iteration's population (index 0 of iteration 0 is the mean)
    int rank;        // rank in the iteration by (F, V, J) of the early-exit view, 0 = best
    SamplingParam z;
    EvaluationResult result;  // every violation component, J for every trajectory
};

// Proposal an iteration's population was drawn from (unit coordinates of the search space,
// samples clipped to [0, 1]^3).
struct ProposalRecord {
    int pass;
    int iteration;
    GaussianProposal proposal;
};

struct CycleRecord {
    SearchSpace space;
    std::vector<CandidateRecord> candidates;
    std::vector<ProposalRecord> proposals;
};
#endif

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

#ifdef ENABLE_DATA_COLLECTION
    CycleRecord last_record;  // filled when iteration_settings().record_candidates
#endif

protected:
    virtual const IterativeSamplingSettings& iteration_settings() const = 0;

    // Refines the proposal from the evaluated population (unit samples and their results).
    virtual void update_proposal(GaussianProposal& proposal,
                                 const std::vector<Unit3>& samples,
                                 const std::vector<EvaluationResult>& results) const = 0;

private:
    // num_iterations rounds of sampling and proposal updates; keeps the best feasible
    // trajectory in best_traj and all feasible ones in last_fplist. pass is 1 for the
    // re-plan without the clearance requirement.
    void run_iterations(const FrenetState& frenet_state, const SearchSpace& space,
                        const PlanningContext& context, int num_threads, int pass);

#ifdef ENABLE_DATA_COLLECTION
    // Stores the full results of one iteration in last_record and replaces them by their
    // early-exit view for the planner.
    void record_iteration(int pass, int iteration, const GaussianProposal& proposal,
                          const std::vector<SamplingParam>& samples,
                          std::vector<EvaluationResult>& results);
#endif

    SamplingParam prev_best_;
    bool has_prev_best_ = false;
};

#endif // ITERATIVE_SAMPLING_PLANNER_H
