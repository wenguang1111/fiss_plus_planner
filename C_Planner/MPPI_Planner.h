#ifndef MPPI_PLANNER_H
#define MPPI_PLANNER_H

#include "Iterative_Sampling_Planner.h"

struct MPPISettings : IterativeSamplingSettings {
    double temperature = 0.1;    // gamma, applied to the per-iteration normalized score
    double learning_rate = 0.7;  // eta of the mean / covariance update
};

// MPPI-style update over the FOP sampling parameters: the whole population is weighted
// by exp(-score / gamma) (MPPI, Williams et al. 2017) and the proposal moves towards the
// weighted mean and covariance with learning rate eta, the update MMD-OPT (Sharma & Singh
// 2025, Eq. 30) uses for Frenet behavioural inputs (there over its elite set only). The
// score is J over the feasible samples (infeasible ones get weight 0); if no sample is
// feasible it is V over all samples. Scores are normalized to [0, 1] per iteration
// because J differs by orders of magnitude between scenarios.
class MPPI_Planner : public Iterative_Sampling_Planner {
public:
    MPPISettings mppi_settings;

    MPPI_Planner(const SettingParameters& settings_param,
                 const MPPISettings& mppi_param,
                 const VehicleParams& vehicle_param,
                 const double* obs_array,
                 const int* num_verts,
                 int n_time_steps,
                 int n_obstacles,
                 int max_verts);

protected:
    const IterativeSamplingSettings& iteration_settings() const override { return mppi_settings; }
    void update_proposal(GaussianProposal& proposal,
                         const std::vector<Unit3>& samples,
                         const std::vector<EvaluationResult>& results) const override;
};

#endif // MPPI_PLANNER_H
