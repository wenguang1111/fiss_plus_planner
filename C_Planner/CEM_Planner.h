#ifndef CEM_PLANNER_H
#define CEM_PLANNER_H

#include "Iterative_Sampling_Planner.h"

struct CEMSettings : IterativeSamplingSettings {
    double elite_fraction = 0.1;  // share of the population refitted as elite set
    double smoothing = 0.7;       // weight of the elite fit vs. the previous proposal
};

// Cross-entropy method over the FOP sampling parameters (Rubinstein & Kroese; elite
// selection by feasibility first, then cost, as in MMD-OPT, Sharma & Singh 2025): the
// elite set are the best samples ranked by (feasible, V, J); the proposal is refitted to
// their mean and standard deviation and smoothed with the previous proposal.
class CEM_Planner : public Iterative_Sampling_Planner {
public:
    CEMSettings cem_settings;

    CEM_Planner(const SettingParameters& settings_param,
                const CEMSettings& cem_param,
                const VehicleParams& vehicle_param,
                const double* obs_array,
                const int* num_verts,
                int n_time_steps,
                int n_obstacles,
                int max_verts);

protected:
    const IterativeSamplingSettings& iteration_settings() const override { return cem_settings; }
    void update_proposal(GaussianProposal& proposal,
                         const std::vector<Unit3>& samples,
                         const std::vector<EvaluationResult>& results) const override;
};

#endif // CEM_PLANNER_H
